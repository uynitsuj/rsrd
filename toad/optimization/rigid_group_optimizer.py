import torch
import matplotlib.pyplot as plt
from nerfstudio.utils.eval_utils import eval_setup
from pathlib import Path
import numpy as np
from nerfstudio.viewer.viewer import Viewer
from nerfstudio.configs.base_config import ViewerConfig
import cv2
from torchvision.transforms import ToTensor
from PIL import Image
from typing import List,Optional,Literal
from nerfstudio.utils import writer
import time
from threading import Lock
import kornia
from lerf.dig import DiGModel
from lerf.data.utils.dino_dataloader import DinoDataloader
from nerfstudio.cameras.cameras import Cameras
from copy import deepcopy
from torchvision.transforms.functional import resize
from contextlib import nullcontext
from nerfstudio.engine.schedulers import ExponentialDecayScheduler,ExponentialDecaySchedulerConfig
import warp as wp
from toad.optimization.atap_loss import ATAPLoss
from toad.utils import *
import viser.transforms as vtf

def quatmul(q0:torch.Tensor,q1:torch.Tensor):
    w0, x0, y0, z0 = torch.unbind(q0, dim=-1)
    w1, x1, y1, z1 = torch.unbind(q1, dim=-1)
    return torch.stack(
            [
                -x0 * x1 - y0 * y1 - z0 * z1 + w0 * w1,
                x0 * w1 + y0 * z1 - z0 * y1 + w0 * x1,
                -x0 * z1 + y0 * w1 + z0 * x1 + w0 * y1,
                x0 * y1 - y0 * x1 + z0 * w1 + w0 * z1,
            ],
            dim = -1
        )

def depth_ranking_loss(rendered_depth, gt_depth):
    """
    Depth ranking loss as described in the SparseNeRF paper
    Assumes that the layout of the batch comes from a PairPixelSampler, so that adjacent samples in the gt_depth
    and rendered_depth are from pixels with a radius of each other
    """
    m = 1e-4
    if rendered_depth.shape[0] % 2 != 0:
        # chop off one index
        rendered_depth = rendered_depth[:-1, :]
        gt_depth = gt_depth[:-1, :]
    dpt_diff = gt_depth[::2, :] - gt_depth[1::2, :]
    out_diff = rendered_depth[::2, :] - rendered_depth[1::2, :] + m
    differing_signs = torch.sign(dpt_diff) != torch.sign(out_diff)
    loss = (out_diff[differing_signs] * torch.sign(out_diff[differing_signs]))
    med = loss.quantile(.8)
    return loss[loss < med].mean()

@wp.kernel
def apply_to_model(pose_deltas: wp.array(dtype = float, ndim = 2), means: wp.array(dtype = wp.vec3), quats: wp.array(dtype = float,ndim=2),
                    group_labels: wp.array(dtype = int), centroids: wp.array(dtype = wp.vec3),
                    means_out: wp.array(dtype = wp.vec3), quats_out: wp.array(dtype = float,ndim=2)):
    """
    Takes the current pose_deltas and applies them to each of the group masks
    """
    tid = wp.tid()
    group_id = group_labels[tid]
    position = wp.vector(pose_deltas[group_id,0],pose_deltas[group_id,1],pose_deltas[group_id,2])
    #pose_deltas are in w x y z, we need to flip
    quaternion = wp.quaternion(pose_deltas[group_id,4],pose_deltas[group_id,5],pose_deltas[group_id,6],pose_deltas[group_id,3])
    transform = wp.transformation(position,quaternion)
    means_out[tid] = wp.transform_point(transform,means[tid] - centroids[tid]) + centroids[tid]
    gauss_quaternion = wp.quaternion(quats[tid,1],quats[tid,2],quats[tid,3],quats[tid,0])
    newquat = quaternion*gauss_quaternion
    quats_out[tid,0] = newquat[3]
    quats_out[tid,1] = newquat[0]
    quats_out[tid,2] = newquat[1]
    quats_out[tid,3] = newquat[2]

def mnn_matcher(feat_a, feat_b):
    """
    feat_a: NxD
    feat_b: MxD
    return: K, K (indices in feat_a and feat_b)
    """
    device = feat_a.device
    sim = feat_a.mm(feat_b.t())
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = (ids1 == nn21[nn12])
    return ids1[mask], nn12[mask]

class RigidGroupOptimizer:
    use_depth: bool = True
    depth_ignore_threshold: float = 0.02 # in meters
    use_atap: bool = True
    pose_lr: float = .005
    pose_lr_final: float = .0005
    mask_hands: bool = False
    
    def __init__(self, dig_model: DiGModel, dino_loader: DinoDataloader, init_c2o: Cameras,
                  group_masks: List[torch.Tensor], group_labels: torch.Tensor, dataset_scale: float,
                    render_lock = nullcontext()):
        """
        This one takes in a list of gaussian ID masks to optimize local poses for
        Each rigid group can be optimized independently, with no skeletal constraints
        """
        self.dataset_scale = dataset_scale
        self.tape = None
        self.dig_model = dig_model
        #detach all the params to avoid retain_graph issue
        self.dig_model.gauss_params['means'] = self.dig_model.gauss_params['means'].detach()
        self.dig_model.gauss_params['quats'] = self.dig_model.gauss_params['quats'].detach()
        self.dino_loader = dino_loader
        self.group_labels = group_labels
        self.group_masks = group_masks
        #store a 7-vec of trans, rotation for each group
        self.pose_deltas = torch.zeros(len(group_masks),7,dtype=torch.float32,device='cuda')
        self.pose_deltas[:,3:] = torch.tensor([1,0,0,0],dtype=torch.float32,device='cuda')
        self.pose_deltas = torch.nn.Parameter(self.pose_deltas)
        k = 3
        s = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
        self.blur = kornia.filters.GaussianBlur2d((k, k), (s, s))
        #NOT USED RN
        self.connectivity_weights = torch.nn.Parameter(-torch.ones(len(group_masks),len(group_masks),dtype=torch.float32,device='cuda'))
        self.optimizer = torch.optim.Adam([self.pose_deltas],lr=self.pose_lr)
        # self.weights_optimizer = torch.optim.Adam([self.connectivity_weights],lr=.001)
        self.keyframes = []
        # lock to prevent blocking the render thread if provided
        self.render_lock = render_lock
        if self.use_atap:
            self.atap = ATAPLoss(dig_model,group_masks,group_labels)
        self.init_c2o = deepcopy(init_c2o).to('cuda')
        self._init_centroids()

    def _init_centroids(self):
        self.init_means = self.dig_model.gauss_params['means'].detach().clone()
        self.init_quats = self.dig_model.gauss_params['quats'].detach().clone()
        self.centroids = torch.empty((self.dig_model.num_points,3),dtype=torch.float32,device='cuda',requires_grad=False)
        for i,mask in enumerate(self.group_masks):
            with torch.no_grad():
                self.centroids[mask] = self.dig_model.gauss_params['means'][mask].mean(dim=0)

    def initialize_obj_pose(self, niter = 200, n_seeds = 6, render = False):
        renders = []
        def try_opt(start_pose_adj):
            "tries to optimize the pose, returns None if failed, otherwise returns outputs and loss"
            self.reset_transforms()
            whole_obj_gp_labels = torch.zeros(self.dig_model.num_points).int().cuda()
            whole_obj_centroids = self.dig_model.means.mean(dim=0,keepdim=True).repeat(self.dig_model.num_points,1)
            whole_pose_adj = start_pose_adj.clone()
            whole_pose_adj = torch.nn.Parameter(whole_pose_adj)
            optimizer = torch.optim.Adam([whole_pose_adj],lr=.01)
            for i in range(niter):
                with torch.no_grad():
                    whole_pose_adj[:,3:] = whole_pose_adj[:,3:]/whole_pose_adj[:,3:].norm(dim=1,keepdim=True)
                tape = wp.Tape()
                optimizer.zero_grad()
                with self.render_lock:
                    self.dig_model.eval()
                    with tape:
                        self.apply_to_model(whole_pose_adj,whole_obj_centroids,whole_obj_gp_labels)
                    dig_outputs = self.dig_model.get_outputs(self.init_c2o)
                    if 'dino' not in dig_outputs:
                        return None,None
                dino_feats = self.blur(dig_outputs["dino"].permute(2,0,1)[None]).squeeze().permute(1,2,0)
                pix_loss = (self.frame_pca_feats - dino_feats)
                loss = pix_loss.norm(dim=-1).mean()
                loss.backward()
                tape.backward()
                optimizer.step()
                if whole_pose_adj.grad.norm() < 2.5e-2:
                    break
                if render: 
                    renders.append(dig_outputs['rgb'].detach())
            return dig_outputs, loss, whole_pose_adj.data.clone()
        best_loss = float('inf')
        
        def find_pixel(n_gauss = 10000):
            """
            returns the y,x coord and box size of the object in the video frame, based on the dino features
            and mutual nearest neighbors
            """
            samps = torch.randint(0,self.dig_model.num_points,(n_gauss,),device='cuda')
            nn_inputs = self.dig_model.gauss_params['dino_feats'][samps]
            dino_feats = self.dig_model.nn(nn_inputs.half()).float()# NxC
            downsamp_factor = 4
            downsamp_frame_feats = self.frame_pca_feats[::downsamp_factor,::downsamp_factor,:]
            frame_feats = downsamp_frame_feats.reshape(-1,downsamp_frame_feats.shape[-1]) # (H*W) x C
            _,match_ids = mnn_matcher(dino_feats,frame_feats)
            x,y = (match_ids % (self.init_c2o.width/downsamp_factor)).float(), (match_ids // (self.init_c2o.width/downsamp_factor)).float()
            x,y = x*downsamp_factor,y*downsamp_factor
            return y,x,torch.tensor([y.mean().item(),x.mean().item()],device='cuda')
            
        ys,xs,best_pix = find_pixel()
        obj_centroid = self.dig_model.means.mean(dim=0,keepdim=True) # 1x3
        ray = self.init_c2o.generate_rays(0,best_pix)
        dist = 1.0
        point = ray.origins + ray.directions*dist
        for z_rot in np.linspace(0,np.pi*2,n_seeds):
            whole_pose_adj = torch.zeros(1,7,dtype=torch.float32,device='cuda')
            # x y z qw qx qy qz
            # (x,y,z) = something along ray - centroid
            quat = torch.from_numpy(vtf.SO3.from_z_radians(z_rot).wxyz).cuda()
            whole_pose_adj[:,:3] = point - obj_centroid
            whole_pose_adj[:,3:] = quat
            dig_outputs, loss, final_pose = try_opt(whole_pose_adj)
            if loss is not None and loss < best_loss:
                best_loss = loss
                best_outputs = dig_outputs
                best_pose = final_pose
        self.reset_transforms()
        with torch.no_grad():
            self.pose_deltas[:,3:]  = best_pose[:,3:]
            for i in range(len(self.group_masks)):
                self.pose_deltas[i,:3] = best_pose[:,:3]
        return xs, ys, best_outputs, renders
    
    def get_poses_relative_to_camera(self, c2w: torch.Tensor):
        """
        Returns the current group2cam transform as defined by the specified camera pose in world coords
        c2w: 4x4 tensor of camera to world transform
        
        Coordinate origin of the object aligns with world axes and centered at centroid

        returns:
        Nx4x4 tensor of obj2camera transform for each of the N groups, in the same ordering as the cluster labels
        """
        with torch.no_grad():
            w2c = c2w.inverse().cpu().numpy()
            obj2cam = torch.zeros(len(self.group_masks),4,4,dtype=torch.float32,device='cuda')
            for i in range(len(self.group_masks)):
                gp_centroid = self.init_means[self.group_masks[i]].mean(dim=0)
                new_centroid = gp_centroid + self.pose_deltas[i,:3]
                new_quat = self.pose_deltas[i,3:]
                world2obj = vtf.SE3.from_rotation_and_translation(vtf.SO3(new_quat.cpu().numpy()),new_centroid.cpu().numpy())
                obj2cam[i,:,:] = torch.tensor(w2c @ world2obj.inverse().as_matrix(),dtype=torch.float32,device='cuda')
        return obj2cam

    def step(self, niter = 1, use_depth = True, use_rgb = False, metric_depth = False):
        scheduler = ExponentialDecayScheduler(ExponentialDecaySchedulerConfig(lr_final = self.pose_lr_final, max_steps=niter)).get_scheduler(self.optimizer, self.pose_lr)
        for i in range(niter):
            # renormalize rotation representation
            with torch.no_grad():
                self.pose_deltas[:,3:] = self.pose_deltas[:,3:]/self.pose_deltas[:,3:].norm(dim=1,keepdim=True)
            tape = wp.Tape()
            self.optimizer.zero_grad()
            # self.weights_optimizer.zero_grad()
            with self.render_lock:
                self.dig_model.eval()
                with tape:
                    self.apply_to_model(self.pose_deltas,self.centroids,self.group_labels)
                dig_outputs = self.dig_model.get_outputs(self.init_c2o)
            if 'dino' not in dig_outputs:
                self.reset_transforms()
                raise RuntimeError("Lost tracking")
            with torch.no_grad():
                object_mask = dig_outputs['accumulation']>.9
            dino_feats = self.blur(dig_outputs["dino"].permute(2,0,1)[None]).squeeze().permute(1,2,0)
            if self.mask_hands:
                pix_loss = (self.frame_pca_feats - dino_feats)[self.hand_mask]
            else:
                pix_loss = (self.frame_pca_feats - dino_feats)
            # THIS IS BAD WE NEED TO FIX THIS (because resizing makes the image very slightly misaligned)
            loss = pix_loss.norm(dim=-1).mean()
            if use_depth and self.use_depth:
                if metric_depth:
                    physical_depth = dig_outputs['depth']/self.dataset_scale
                    valids = object_mask & (~self.frame_depth.isnan())
                    if self.mask_hands:
                        valids = valids & self.hand_mask.unsqueeze(-1)
                    pix_loss = (physical_depth - self.frame_depth)**2
                    pix_loss = pix_loss[valids & (pix_loss<self.depth_ignore_threshold**2)]
                    loss = loss + 0.1*pix_loss.mean()
                else:
                    # This is ranking loss for monodepth (which is disparity)
                    disparity = 1.0 / dig_outputs['depth']
                    N = 20000
                    if self.mask_hands:
                        object_mask = object_mask & self.hand_mask.unsqueeze(-1)
                    valid_ids = torch.where(object_mask)
                    rand_samples = torch.randint(0,valid_ids[0].shape[0],(N,),device='cuda')
                    rand_samples = (valid_ids[0][rand_samples],valid_ids[1][rand_samples])
                    rend_samples = disparity[rand_samples]
                    mono_samples = self.frame_depth[rand_samples]
                    rank_loss = depth_ranking_loss(rend_samples,mono_samples)
                    loss = loss + 0.5*rank_loss
            if use_rgb:
                loss = loss + .05*(dig_outputs['rgb']-self.rgb_frame).abs().mean()
            if self.use_atap:
                null_weights = torch.ones_like(self.connectivity_weights)
                # null_weights = self.connectivity_weights.exp()
                weights = torch.clip(null_weights,0,1)
                with tape:
                    atap_loss = self.atap(weights)
                rigidity_loss = .02*(1-weights).mean()
                symmetric_loss = (weights - weights.T).abs().mean()
                #maximize the connectivity weights, as well as similarity
                loss = loss + atap_loss + symmetric_loss + rigidity_loss
            loss.backward()
            tape.backward()
            self.optimizer.step()
            # self.weights_optimizer.step()
            scheduler.step()
        #reset lr
        self.optimizer.param_groups[0]['lr'] = self.pose_lr
        return dig_outputs
    
    def apply_to_model(self, pose_deltas, centroids, group_labels):
        """
        Takes the current pose_deltas and applies them to each of the group masks
        """
        self.reset_transforms()
        new_quats = torch.empty_like(self.dig_model.gauss_params['quats'],requires_grad=False)
        new_means = torch.empty_like(self.dig_model.gauss_params['means'],requires_grad=True)
        wp.launch(
            kernel = apply_to_model,
            dim = self.dig_model.num_points,
            inputs = [wp.from_torch(pose_deltas),wp.from_torch(self.dig_model.gauss_params['means'],dtype=wp.vec3),
                    wp.from_torch(self.dig_model.gauss_params['quats']),wp.from_torch(group_labels),
                    wp.from_torch(centroids,dtype=wp.vec3)],
            outputs = [wp.from_torch(new_means,dtype=wp.vec3),wp.from_torch(new_quats)]
        )
        self.dig_model.gauss_params['quats'] = new_quats
        self.dig_model.gauss_params['means'] = new_means

    def register_keyframe(self):
        """
        Saves the current pose_deltas as a keyframe
        """
        self.keyframes.append(self.pose_deltas.detach().clone())

    def apply_keyframe(self,i):
        """
        Applies the ith keyframe to the pose_deltas
        """
        with torch.no_grad():
            self.apply_to_model(self.keyframes[i])

    def reset_transforms(self):
        with torch.no_grad():
            self.dig_model.gauss_params['means'] = self.init_means.clone()
            self.dig_model.gauss_params['quats'] = self.init_quats.clone()

    def set_frame(self, rgb_frame: torch.Tensor, depth: torch.Tensor = None):
        """
        Sets the rgb_frame to optimize the pose for
        rgb_frame: HxWxC tensor image
        init_c2o: initial camera to object transform (given whatever coordinates the self.dig_model is in)
        """
        with torch.no_grad():
            self.rgb_frame = resize(rgb_frame.permute(2,0,1), (self.init_c2o.height,self.init_c2o.width),antialias = True).permute(1,2,0)
            self.frame_pca_feats = self.dino_loader.get_pca_feats(rgb_frame.permute(2,0,1).unsqueeze(0),keep_cuda=True).squeeze()
            self.frame_pca_feats = resize(self.frame_pca_feats.permute(2,0,1), (self.init_c2o.height,self.init_c2o.width),antialias = True).permute(1,2,0)
            # HxWxC
            if self.use_depth:
                if depth is None:
                    depth = get_depth((self.rgb_frame*255).to(torch.uint8))
                self.frame_depth = resize(depth.unsqueeze(0), (self.init_c2o.height,self.init_c2o.width),antialias = True).squeeze().unsqueeze(-1)
            if self.mask_hands:
                self.hand_mask = get_hand_mask((self.rgb_frame*255).to(torch.uint8))
                self.hand_mask = torch.nn.functional.max_pool2d(self.hand_mask[None,None],3,padding=1,stride=1).squeeze() == 0.0
