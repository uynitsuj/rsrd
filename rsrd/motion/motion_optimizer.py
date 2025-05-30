from copy import deepcopy
import json
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, List, Optional, Tuple, cast, Union, TYPE_CHECKING

import kornia
import numpy as np
import torch
from torchvision.transforms import ToTensor
from PIL import Image
from loguru import logger
from tqdm import tqdm
from jaxtyping import Float
import warp as wp

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.engine.schedulers import (
    ExponentialDecayScheduler,
    ExponentialDecaySchedulerConfig,
)
from nerfstudio.model_components.losses import depth_ranking_loss
from nerfstudio.pipelines.base_pipeline import Pipeline

from dig.dig import DiGModel
from dig.dig_pipeline import ObjectMode
from dig.data.utils.dino_dataloader import DinoDataloader

import rsrd.transforms as tf
from rsrd.motion.atap_loss import ATAPLoss, ATAPConfig
from rsrd.motion.observation import PosedObservation, VideoSequence, Frame
from rsrd.util.warp_kernels import apply_to_model_warp, traj_smoothness_loss_warp, apply_to_model_warp_multi_object
from rsrd.util.common import identity_7vec, extrapolate_poses, mnn_matcher

try:
    from hamer_helper import HandOutputsWrtCamera
except ModuleNotFoundError:
    HandOutputsWrtCamera = None

@dataclass
class RigidGroupOptimizerConfig:
    use_depth: bool = True
    use_rgb: bool = False
    rank_loss_mult: float = 0.2
    rank_loss_erode: int = 3
    depth_ignore_threshold: float = 0.1  # in meters
    atap_config: ATAPConfig = field(default_factory=ATAPConfig)
    use_roi: bool = True
    roi_inflate: float = 0.25
    pose_lr: float = 0.001
    pose_lr_final: float = 0.0005
    mask_hands: bool = False
    blur_kernel_size: int = 5
    mask_threshold: float = 0.7
    rgb_loss_weight: float = 0.05
    part_still_weight: float = 0.01

    approx_dist_to_obj: float = 0.45  # in meters
    altitude_down: float = 0.1 #np.pi / 6  # in radians

class RigidGroupOptimizer:
    dig_model: DiGModel
    dino_loader: DinoDataloader
    dataset_scale: float

    num_groups: int
    group_labels: torch.Tensor
    group_masks: List[torch.Tensor]

    # Poses, as scanned.
    init_means: torch.Tensor
    init_quats: torch.Tensor
    T_world_objinit: torch.Tensor
    init_p2o: Float[torch.Tensor, "group 7"]  # noqa: F722

    part_deltas: Float[torch.nn.Parameter, "time group 7"]  # noqa: F722
    T_objreg_objinit: Optional[Float[torch.Tensor, "1 7"]]  # noqa: F722
    """
    Transform:
    - from: `objreg`, in camera frame from which it was registered (e.g., robot camera).
    - from: `objinit`, in original frame from which it was scanned
    """

    hands_info: dict[int, tuple[Optional[HandOutputsWrtCamera], Optional[HandOutputsWrtCamera]]]

    def __init__(
        self,
        config: RigidGroupOptimizerConfig,
        pipeline: Pipeline,
        render_lock: Union[Lock, nullcontext] = nullcontext(),
    ):
        """
        This one takes in a list of gaussian ID masks to optimize local poses for
        Each rigid group can be optimized independently, with no skeletal constraints
        """
        self.config = config
        self.dig_model = cast(DiGModel, pipeline.model)
        self.dino_loader = pipeline.datamanager.dino_dataloader

        assert pipeline.datamanager.train_dataset is not None
        self.dataset_scale = pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_scale
        self.render_lock = render_lock

        k = self.config.blur_kernel_size
        s = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
        self.blur = kornia.filters.GaussianBlur2d((k, k), (s, s))

        # Detach all the params to avoid retain_graph issue.
        for k, v in self.dig_model.gauss_params.items():
            self.dig_model.gauss_params[k] = v.detach().clone()

        # Store the initial means and quats, for state restoration later on.
        self.init_means = self.dig_model.gauss_params["means"].detach().clone()
        self.init_quats = self.dig_model.gauss_params["quats"].detach().clone()

        # Initialize parts + optimizers.
        if pipeline.cluster_labels is None:
            labels = torch.zeros(pipeline.model.num_points).int().cuda()
        else:
            labels = pipeline.cluster_labels.int().cuda()
            
        assert hasattr(pipeline, "object_mode")
        self.object_mode = pipeline.object_mode
        self.fixed_object_ids = pipeline.fixed_obj_ids
        self.configure_from_clusters(labels)

        self.sequence = VideoSequence()
        self.T_objreg_objinit = None
        self.hands_info = {}
        self._track_path = None

    def set_track_path(self, track_path: Path):
        self._track_path = track_path.parent
        
    def configure_from_clusters(self, group_labels: torch.Tensor):
        """
        Given `group_labels`, set the group masks and labels.

        Affects all attributes affected by # of parts:
        - `self.num_groups
        - `self.group_labels`
        - `self.group_masks`
        - `self.part_deltas`
        - `self.init_p2o`
        - `self.atap`
        , as well as `self.init_o2w`.

        NOTE(cmk) why do you need to store both `self.group_labels` and `self.group_masks`?
        """
        # Get group / cluster label info.
        self.group_labels = group_labels.cuda()
        self.num_groups = int(self.group_labels.max().item() + 1)
        self.group_masks = [(self.group_labels == cid).cuda() for cid in range(self.group_labels.max() + 1)]

        # Store pose of each part, as wxyz_xyz.
        part_deltas = torch.zeros(0, self.num_groups, 7, dtype=torch.float32, device="cuda")
        self.part_deltas = torch.nn.Parameter(part_deltas)

        if self.object_mode == ObjectMode.ARTICULATED:
            # Initialize the object pose. Centered at object centroid, and identity rotation.
            self.T_world_objinit = identity_7vec()
            self.T_world_objinit[0, 4:] = self.init_means.mean(dim=0).squeeze()

            # Initialize the part poses to identity. Again, wxyz_xyz.
            # Parts are initialized at the centroid of the part cluster.
            self.init_p2o = identity_7vec().repeat(self.num_groups, 1)
            for i, g in enumerate(self.group_masks):
                gp_centroid = self.init_means[g].mean(dim=0)
                self.init_p2o[i, 4:] = gp_centroid - self.init_means.mean(dim=0)
                
        elif self.object_mode == ObjectMode.RIGID_OBJECTS:
            self.T_world_objmeans = identity_7vec()
            self.T_world_objmeans[0, 4:] = self.init_means.mean(dim=0).squeeze()
            
            self.config.atap_config.use_atap = False # Disable atap for multi-rigid
            self.T_world_objinit = identity_7vec().repeat(self.num_groups, 1)
            for i, g in enumerate(self.group_masks):
                gp_centroid = self.init_means[g].mean(dim=0)
                self.T_world_objinit[i, 4:] = gp_centroid
            self.init_p2o = identity_7vec().repeat(self.num_groups, 1)

        self.atap = ATAPLoss(
            self.config.atap_config,
            self.dig_model,
            self.group_masks,
            self.group_labels,
            self.dataset_scale,
        )

    def initialize_obj_pose(
        self,
        first_obs: PosedObservation,
        niter=180,
        n_seeds=8,
        use_depth=False,
        render=False,
    ):
        """
        Initializes object pose w/ observation. Also sets:
        - `self.T_objreg_objinit`
        """
        renders = []
        frame_rgb = (first_obs.frame.rgb.cpu().numpy()*255).astype(np.uint8)
        if self._track_path is not None:
            # if track_init exists, load the best pose
            if (self._track_path/f"track_init.json").exists():
                with open(self._track_path / f"track_init.json", "r") as f:
                    track_init_json = json.load(f)
                    track_init = torch.tensor(track_init_json["best_pose"]).to(self.T_world_objinit.device)
                    if self.object_mode == ObjectMode.ARTICULATED:
                        assert track_init.shape == (1, 7), track_init.shape
                    elif self.object_mode == ObjectMode.RIGID_OBJECTS:
                        assert track_init.shape == (self.num_groups, 7), track_init.shape
                    self.T_objreg_objinit = track_init 
                    # Apply transforms to model
                    self.apply_to_model(
                        self.T_objreg_objinit,
                        identity_7vec().repeat(len(self.group_masks), 1)
                    )
                    
                    first_obs.compute_and_set_roi(self)
                    _, best_pose, rend = self._try_opt(
                        track_init, first_obs.roi_frame, 1, use_depth = True, lr=0.005, render=render, camera=first_obs.frame.camera
                    )

                    logger.info("Initialized object pose")
                    self.T_objreg_objinit = best_pose
                    
                    rend_final_opt_frame = 0.6*rend[-1] + 0.4*frame_rgb
                    rend = [0.6*r + 0.4*frame_rgb for r in rend]
                    renders.extend(rend)
                    return renders, rend_final_opt_frame
                    

        # Initial guess for 3D object location.
        est_dist_to_obj = self.config.approx_dist_to_obj * self.dataset_scale  # scale to nerfstudio world.

        xs, ys, est_loc_2d = self._find_object_pixel_location(first_obs)
        
        # TODO: Remove debug plot when done devel
        import matplotlib.pyplot as plt
        plt.imshow(first_obs.frame.rgb.cpu().numpy())
        for i in range(est_loc_2d.shape[0]):
            plt.scatter(est_loc_2d[i][1].cpu().numpy(), est_loc_2d[i][0].cpu().numpy())
        plt.savefig("sample_dino2d.png") 
            
        ray = first_obs.frame.camera.generate_rays(0, est_loc_2d)
        est_loc_3d = ray.origins + ray.directions * est_dist_to_obj

        # Take `n_seed` rotations around the object centroid, optimize, then pick the best one.
        # Don't use ROI for this step.
        best_pose, best_loss = identity_7vec(), float("inf")
        if self.object_mode == ObjectMode.ARTICULATED:
            obj_centroid = self.dig_model.means.mean(dim=0, keepdim=True)  # 1x3
        elif self.object_mode == ObjectMode.RIGID_OBJECTS:
            obj_centroid = self.T_world_objinit.clone()[:, 4:]
            
        
        for z_rot in tqdm(np.linspace(0, np.pi * 2, n_seeds), "Trying seeds..."):
            if self.object_mode == ObjectMode.ARTICULATED:
                candidate_pose = torch.zeros(1, 7, dtype=torch.float32, device="cuda")
            elif self.object_mode == ObjectMode.RIGID_OBJECTS:
                # Generate a uniform random z_rot per rigid object
                # z_rot = torch.rand((self.num_groups,)) * 2 * np.pi
                
                candidate_pose = torch.zeros(self.num_groups, 7, dtype=torch.float32, device="cuda")
            candidate_pose[:, :4] = (
                (
                    tf.SO3.from_x_radians(
                        torch.tensor(-np.pi / 2)
                    )  # Camera in opengl, while object is in world coord.
                    @ tf.SO3.from_x_radians(
                        torch.tensor(self.config.altitude_down)
                    )  # Look slightly down at the object.
                    @ tf.SO3.from_z_radians(torch.tensor(z_rot))
                )
                .wxyz.float()
                .cuda()
            )
            candidate_pose[:, 4:] = est_loc_3d - obj_centroid
                            
            loss, final_pose, rend = self._try_opt(
                candidate_pose, first_obs.frame, niter, use_depth, render=render
            )
            # composite the render on top of the frame
            rend = [0.6*r + 0.4*frame_rgb for r in rend]
            renders.extend(rend)

            # TODO: see if per-object loss is possible to find best poses (otherwise all objects need to converge to optimal init pose on the same seed)
            if loss is not None and loss < best_loss:
                best_loss = loss
                best_pose = final_pose

        # Extra optimization steps, with the best pose.
        # Use ROI for this step, since we're close to the GT object pose.
        
        first_obs.compute_and_set_roi(self)
        
        
        # # Note the lower LR, for this fine-tuning step.
        
        _, best_pose, rend = self._try_opt(
            best_pose, first_obs.roi_frame, niter, use_depth = True, lr=0.005, render=render, camera=first_obs.frame.camera
        )
        rend_final_opt_frame = 0.6*rend[-1] + 0.4*frame_rgb
        
        if self.object_mode == ObjectMode.ARTICULATED:
            assert best_pose.shape == (1, 7), best_pose.shape
        elif self.object_mode == ObjectMode.RIGID_OBJECTS:
            assert best_pose.shape == (self.num_groups, 7), best_pose.shape
        self.T_objreg_objinit = best_pose
        logger.info("Initialized object pose")
        
        if self._track_path is not None:
            if not self._track_path.exists():
                self._track_path.mkdir()
            with open(self._track_path / f"track_init.json", "w") as f:
                json.dump(
                    {
                        "best_pose": self.T_objreg_objinit.cpu().numpy().tolist(),
                    },
                    f,
                )
        
        return renders, rend_final_opt_frame

    def fit(self, frame_idxs: List[int], niter=1):
        # TODO(cmk) temporarily removed all_frames)
        assert self.T_objreg_objinit is not None, "Must initialize first with the first frame"
        lr_init = self.config.pose_lr

        # optimizer = torch.optim.Adam([self.part_deltas], lr=lr_init)
        
        optimizable_params = []
        if len(self.fixed_object_ids) > 0:
            mask = torch.ones(self.num_groups, dtype=torch.bool, device="cuda")
            for idx in self.fixed_object_ids:
                mask[idx] = False
                
            self.part_deltas.register_hook(lambda grad: grad * mask.view(1, -1, 1))
            optimizable_params.append(self.part_deltas)
        else:
            optimizable_params.append(self.part_deltas)

        optimizer = torch.optim.Adam(optimizable_params, lr=lr_init)
    
        scheduler = ExponentialDecayScheduler(
            ExponentialDecaySchedulerConfig(
                lr_final=self.config.pose_lr_final,
                max_steps=niter,
                ramp="linear",
                lr_pre_warmup=1e-5,
                warmup_steps=10 if len(frame_idxs) > 1 else 0,
            )
        ).get_scheduler(optimizer, lr_init)

        for _ in range(niter):
            # renormalize rotation representation
            with torch.no_grad():
                self.part_deltas[..., :4] = self.part_deltas[..., :4] / self.part_deltas[..., :4].norm(dim=-1, keepdim=True)

            optimizer.zero_grad()

            # Compute loss
            if len(frame_idxs) > 1:
                # temporal smoothness loss 
                tape = wp.Tape()
                with tape:
                    part_smoothness= .2 * self.get_smoothness_loss(self.part_deltas, 1.0, 1.0, frame_idxs)
                part_smoothness.backward()
                tape.backward()
                # zero the grad of the non-active ones
                i = torch.ones(self.part_deltas.shape[0], dtype=torch.bool, device="cuda")
                i[frame_idxs] = 0
                self.part_deltas.grad[i] = 0.0
                
                if len(self.fixed_object_ids) > 0:
                    with torch.no_grad():
                        for idx in self.fixed_object_ids:
                            self.part_deltas.grad[:, idx] = 0.0
            for frame_idx in frame_idxs:
                tape = wp.Tape()
                with tape:
                    if self.object_mode == ObjectMode.ARTICULATED:
                        this_obj_delta = self.T_objreg_objinit.view(1, 7)
                    elif self.object_mode == ObjectMode.RIGID_OBJECTS:
                        this_obj_delta = self.T_objreg_objinit.view(self.num_groups, 7)
                    this_part_deltas = self.part_deltas[frame_idx]
                    observation = self.sequence[frame_idx]
                    frame = (
                        observation.frame
                        if not self.config.use_roi
                        else observation.roi_frame
                    )
                    loss = self._get_loss(
                        frame,
                        this_obj_delta,
                        this_part_deltas,
                        self.config.use_depth,
                        self.config.use_rgb,
                        self.config.atap_config.use_atap,
                    )

                assert loss is not None
                loss.backward()
                tape.backward()  # torch, then tape backward, to propagate gradients to warp kernels.

                # tape backward only propagates up to the slice, so we need to call autograd again to continue.
                assert this_part_deltas.grad is not None
                
                torch.autograd.backward(
                    [this_part_deltas],
                    grad_tensors=[this_part_deltas.grad],
                )
                if len(self.fixed_object_ids) > 0:
                    with torch.no_grad():
                        for idx in self.fixed_object_ids:
                            this_part_deltas.grad[idx] = 0.0
                            
            optimizer.step()
            scheduler.step()

    def get_smoothness_loss(self, deltas: torch.Tensor, position_lambda: float, rotation_lambda: float, active_timesteps = slice(None)):
        """
        Returns the smoothness loss for the given deltas
        """
        loss = torch.empty(deltas.shape[0], deltas.shape[1], dtype=torch.float32, device="cuda", requires_grad=True)
        wp.launch(
            kernel=traj_smoothness_loss_warp,
            dim=(deltas.shape[0], deltas.shape[1]),
            inputs=[wp.from_torch(deltas), position_lambda, rotation_lambda],
            outputs = [wp.from_torch(loss)],
        )
        return loss[active_timesteps].sum()

    def _try_opt(
        self,
        pose: torch.Tensor,
        frame: Frame | List[Frame],
        niter: int,
        use_depth: bool,
        lr: float = 0.01,
        render: bool = False,
        camera: Optional[Cameras] = None,
    ) -> Tuple[float, torch.Tensor, List[torch.Tensor]]:
        "tries to optimize the pose, returns None if failed, otherwise returns outputs and loss"
        pose = torch.nn.Parameter(pose.detach().clone())
        
        optimizer = torch.optim.Adam([pose], lr=lr)
        scheduler = ExponentialDecayScheduler(
            ExponentialDecaySchedulerConfig(
                lr_final=0.005,
                max_steps=niter,
            )
        ).get_scheduler(optimizer, lr)

        loss = torch.inf
        renders = []
        for _ in tqdm(range(niter), "Optimizing pose...", leave=False):
            with torch.no_grad():
                pose[..., :4] = pose[..., :4] / pose[..., :4].norm(dim=-1, keepdim=True)

            tape = wp.Tape()
            optimizer.zero_grad()
            with tape:
                loss = self._get_loss(
                    frame,
                    pose,
                    identity_7vec().repeat(len(self.group_masks), 1),
                    use_depth,
                    False,
                    False,
                )

            if loss is None:
                return torch.inf, pose.data.detach(), renders

            loss.backward()
            tape.backward()  # torch, then tape backward, to propagate gradients to warp kernels.
            optimizer.step()
            scheduler.step()

            loss = loss.item()
            if render:
                with torch.no_grad():
                    if isinstance(frame, list) or camera is not None:
                        assert camera is not None, "For multi-ROI frames please explicitly provide original camera for full render"
                        dig_outputs = self.dig_model.get_outputs(camera)
                    else:
                        dig_outputs = self.dig_model.get_outputs(frame.camera)
                assert isinstance(dig_outputs["rgb"], torch.Tensor)
                renders.append((dig_outputs["rgb"].detach() * 255).int().cpu().numpy())

        return loss, pose.data.detach(), renders

    def _find_object_pixel_location(self, obs: PosedObservation, n_gauss=20000):
        """
        returns the y,x coord and box size of the object in the video frame, based on the dino features
        and mutual nearest neighbors
        """
        if self.object_mode == ObjectMode.ARTICULATED:
            n_gauss = min(n_gauss, self.dig_model.num_points)
            samps = torch.randint(0, self.dig_model.num_points, (n_gauss,), device="cuda")
            nn_inputs = self.dig_model.gauss_params["dino_feats"][samps]
            dino_feats = self.dig_model.nn(nn_inputs)  # NxC
            downsamp_frame_feats = obs.frame.dino_feats
            frame_feats = downsamp_frame_feats.reshape(
                -1, downsamp_frame_feats.shape[-1]
            )  # (H*W) x C
            downsamp = 4
            frame_feats = frame_feats[::downsamp]
            _, match_ids = mnn_matcher(dino_feats, frame_feats)
            x, y = (match_ids*downsamp % (obs.frame.camera.width)).float(), (
                match_ids*downsamp // (obs.frame.camera.width)
            ).float()
            return x, y, torch.tensor([[y.median().item(), x.median().item()]], device="cuda")
        
        elif self.object_mode == ObjectMode.RIGID_OBJECTS: # return a list of xs and ys per rigid object
            xs, ys = [], []
            for g in self.group_masks:
                n_gauss = min(n_gauss, g.sum())
                samps = torch.randint(0, g.sum(), (n_gauss,), device="cuda")
                nn_inputs = self.dig_model.gauss_params["dino_feats"][g][samps]
                dino_feats = self.dig_model.nn(nn_inputs)  # NxC
                downsamp_frame_feats = obs.frame.dino_feats
                frame_feats = downsamp_frame_feats.reshape(
                    -1, downsamp_frame_feats.shape[-1]
                )
                downsamp = 4
                frame_feats = frame_feats[::downsamp]
                _, match_ids = mnn_matcher(dino_feats, frame_feats)
                x, y = (match_ids*downsamp % (obs.frame.camera.width)).float(), (
                    match_ids*downsamp // (obs.frame.camera.width)
                ).float()
                xs.append(x)
                ys.append(y)

            return None, None, torch.cat([torch.stack([y.median() for y in ys]).unsqueeze(0), torch.stack([x.median() for x in xs]).unsqueeze(0)]).T
        
    def _loss_impl(self, frame: Frame, use_depth: bool, use_rgb: bool, use_atap: bool, obj_id: int = None):
        loss = torch.Tensor([0.0]).cuda()
        outputs = (cast(
            dict[str, torch.Tensor],
            self.dig_model.get_outputs(frame.camera)#, obj_id=obj_id)
        ))
        assert "accumulation" in outputs, outputs.keys()
        with torch.no_grad():
            object_mask = outputs["accumulation"] > self.config.mask_threshold
        if not object_mask.any():
            logger.warning(f"Object {obj_id} not detected in frame")
            return None
        dino_loss = self._get_dino_loss(outputs, frame, object_mask)
        loss = loss + dino_loss
        if use_depth:
            depth_loss = self._get_depth_loss(outputs, frame, object_mask)
            loss = loss + depth_loss
        if use_rgb:
            rgb_loss = 0.05 * (outputs["rgb"] - frame.rgb).abs().mean()
            loss = loss + rgb_loss
        if use_atap:
            weights = torch.full(
                (self.num_groups, self.num_groups),
                1,
                dtype=torch.float32,
                device="cuda",
            )
            atap_loss = self.atap(weights)
            loss = loss + atap_loss
        return loss
        
    def _get_loss(
        self,
        frame: Frame | List[Frame],
        obj_delta: torch.Tensor,
        part_deltas: torch.Tensor,
        use_depth: bool,
        use_rgb: bool,
        use_atap: bool,
    ) -> Optional[torch.Tensor]:
        """
        Returns a backpropable loss for the given frame.
        """

        with self.render_lock:
            self.dig_model.eval()
            self.apply_to_model(obj_delta, part_deltas)
            if isinstance(frame, list):
                loss = torch.Tensor([0.0]).cuda()
                for obj_id, fr in enumerate(frame):
                    loss += self._loss_impl(fr, use_depth, use_rgb, use_atap, obj_id=obj_id)
            else:
                loss = self._loss_impl(frame, use_depth, use_rgb, use_atap, obj_id=frame.obj_id)
        return loss

    def _get_dino_loss(
        self,
        outputs: dict[str, torch.Tensor],
        frame: Frame,
        object_mask: torch.Tensor,
    ) -> torch.Tensor:
        assert "dino" in outputs and isinstance(outputs["dino"], torch.Tensor)

        blurred_dino_feats = (
            self.blur(outputs["dino"].permute(2, 0, 1)[None]).squeeze().permute(1, 2, 0)
        )
        dino_feats = torch.where(object_mask, outputs["dino"], blurred_dino_feats)
        # from rsrd.util.dev_helpers import plot_pca
        # plot_pca(dino_feats, name="rendered_dino_feats")
        # plot_pca(frame.dino_feats, name="real_dino_feats")
        # import pdb; pdb.set_trace()
        if frame.hand_mask is not None:
            loss = (frame.dino_feats[frame.hand_mask] - dino_feats[frame.hand_mask]).norm(dim=-1).mean()
        else:
            loss = (frame.dino_feats - dino_feats).norm(dim=-1).mean()
        del dino_feats  # Explicitly delete
        torch.cuda.empty_cache()
        return loss

    def _get_depth_loss(
        self,
        outputs: dict[str, torch.Tensor],
        frame: Frame,
        object_mask: torch.Tensor,
        n_samples_for_ranking: int = 20000,
    ) -> torch.Tensor:
        if frame.has_metric_depth:
            physical_depth = outputs["depth"] / self.dataset_scale

            valids = object_mask & (~frame.monodepth.isnan())
            if frame.hand_mask is not None:
                valids = valids & frame.hand_mask.unsqueeze(-1)

            pix_loss = (physical_depth - frame.monodepth) ** 2
            pix_loss = pix_loss[
                valids & (pix_loss < self.config.depth_ignore_threshold**2)
            ]
            return pix_loss.mean()

        # Otherwise, we're using disparity.
        frame_depth = 1 / frame.monodepth # convert disparity to depth
        # erode the mask by like 10 pixels
        object_mask = object_mask & (~frame_depth.isnan())
        object_mask = kornia.morphology.erosion(
            object_mask.squeeze().unsqueeze(0).unsqueeze(0).float(),
            torch.ones((self.config.rank_loss_erode, self.config.rank_loss_erode), device='cuda')
        ).squeeze().bool()
        if frame.hand_mask is not None:
            object_mask = object_mask & frame.hand_mask
        valid_ids = torch.where(object_mask)

        if len(valid_ids[0]) > 0:
            rand_samples = torch.randint(
                0, valid_ids[0].shape[0], (n_samples_for_ranking,), device="cuda"
            )
            rand_samples = (
                valid_ids[0][rand_samples],
                valid_ids[1][rand_samples],
            )
            rend_samples = outputs["depth"][rand_samples]
            mono_samples = frame_depth[rand_samples]
            rank_loss = depth_ranking_loss(rend_samples, mono_samples)
            return self.config.rank_loss_mult*rank_loss

        return torch.Tensor([0.0])

    def apply_to_model(self, obj_delta, part_deltas):
        """
        Takes the current part_deltas and applies them to each of the group masks
        """
        self.reset_transforms()
        new_quats = torch.empty_like(
            self.dig_model.gauss_params["quats"], requires_grad=False
        )
        new_means = torch.empty_like(
            self.dig_model.gauss_params["means"], requires_grad=True
        )
        if self.object_mode == ObjectMode.ARTICULATED:
            assert obj_delta.shape == (1, 7), obj_delta.shape
            wp.launch(
            kernel=apply_to_model_warp,
            dim=self.dig_model.num_points,
            inputs = [
                wp.from_torch(self.T_world_objinit),
                wp.from_torch(self.init_p2o),
                wp.from_torch(obj_delta),
                wp.from_torch(part_deltas),
                wp.from_torch(self.group_labels),
                wp.from_torch(self.dig_model.gauss_params["means"], dtype=wp.vec3),
                wp.from_torch(self.dig_model.gauss_params["quats"]),
            ],
            outputs=[wp.from_torch(new_means, dtype=wp.vec3), wp.from_torch(new_quats)],
            )
        elif self.object_mode == ObjectMode.RIGID_OBJECTS:
            assert obj_delta.shape == (self.num_groups,7), obj_delta.shape
            wp.launch(
                kernel=apply_to_model_warp_multi_object,
                dim=self.dig_model.num_points,
                inputs = [
                    wp.from_torch(self.T_world_objinit),
                    wp.from_torch(obj_delta),
                    wp.from_torch(part_deltas),
                    wp.from_torch(self.group_labels),
                    wp.from_torch(self.dig_model.gauss_params["means"], dtype=wp.vec3),
                    wp.from_torch(self.dig_model.gauss_params["quats"]),
                ],
                outputs=[wp.from_torch(new_means, dtype=wp.vec3), wp.from_torch(new_quats)],
            )
        self.dig_model.gauss_params["quats"] = new_quats
        self.dig_model.gauss_params["means"] = new_means

    @torch.no_grad()
    def apply_keyframe(self, i):
        """
        Applies the ith keyframe to the pose_deltas
        """
        assert self.T_objreg_objinit is not None, "Must initialize first with the first frame"
        self.apply_to_model(
            self.T_objreg_objinit,
            self.part_deltas[i]
        )

    def load_tracks(self, path: Path):
        """
        Loads the trajectory from a file. Sets keyframes and hand_frames.
        """
        data = json.loads(path.read_text())
        self.part_deltas = torch.nn.Parameter(torch.tensor(data["part_deltas"]).cuda())
        self.T_objreg_objinit = torch.tensor(data["T_objreg_objinit"]).cuda()
        # Load hand info, if available.
        if "hands" in data:
            self.hands_info = {}
            for tstep, (hand_l_dict, hand_r_dict) in data["hands"].items():
                # Load hands from json files.
                # Left hand:
                if hand_l_dict is not None:
                    # Turn all values into np.ndarray.
                    for k, v in hand_l_dict.items():
                        hand_l_dict[k] = np.array(v)
                    hand_l = HandOutputsWrtCamera(**hand_l_dict)
                else:
                    hand_l = None

                # Right hand:
                if hand_r_dict is not None:
                    for k, v in hand_r_dict.items():
                        hand_r_dict[k] = np.array(v)
                    hand_r = HandOutputsWrtCamera(**hand_r_dict)
                else:
                    hand_r = None

                self.hands_info[int(tstep)] = (hand_l, hand_r)
        else:
            self.hands_info = {}

    def save_tracks(
        self,
        path: Path,
        hands: Optional[
            dict[int, tuple[Optional[HandOutputsWrtCamera], Optional[HandOutputsWrtCamera]]]
        ] = None,
    ):
        """
        Saves the trajectory to a file
        """
        assert self.T_objreg_objinit is not None, "Must initialize first with the first frame"
        save_dict: dict[str, Any] = {
            "part_deltas": self.part_deltas.detach().cpu().tolist(),
            "T_objreg_objinit": self.T_objreg_objinit.detach().cpu().tolist(),
            "T_world_objinit": self.T_world_objinit.detach().cpu().tolist(),
        }

        # Save hand info, if available.
        if self.hands_info is not None and hands is None:
            hands = self.hands_info
        if hands is not None:
            save_dict["hands"] = {}
            for tstep, (hand_l, hand_r) in hands.items():
                # Make hands into dicts that can be written as json files.
                # Left hand:
                if hand_l is not None:
                    hand_l_as_dict = deepcopy(hand_l)
                    for k, v in hand_l_as_dict.items():
                        assert isinstance(v, np.ndarray)
                        hand_l_as_dict[k] = v.tolist()
                else:
                    hand_l_as_dict = None

                # Right hand:
                if hand_r is not None:
                    hand_r_as_dict = deepcopy(hand_r)
                    for k, v in hand_r_as_dict.items():
                        assert isinstance(v, np.ndarray)
                        hand_r_as_dict[k] = v.tolist()
                else:
                    hand_r_as_dict = None
                
                save_dict["hands"][tstep] = (hand_l_as_dict, hand_r_as_dict)

        path.write_text(json.dumps(save_dict))


    def reset_transforms(self):
        with torch.no_grad():
            self.dig_model.gauss_params["means"] = self.init_means.detach().clone()
            self.dig_model.gauss_params["quats"] = self.init_quats.detach().clone()
    # @profile
    def add_observation(self, obs: PosedObservation, extrapolate_velocity = True):
        """
        Sets the rgb_frame to optimize the pose for
        rgb_frame: HxWxC tensor image
        """
        assert self.T_objreg_objinit is not None, "Must initialize first with the first frame"

        if self.config.use_roi:
            obs.compute_and_set_roi(self)
        self.sequence.append(obs)

        if extrapolate_velocity and self.part_deltas.shape[0] > 1:
            with torch.no_grad():
                next_part_delta = extrapolate_poses(
                    self.part_deltas[-2], self.part_deltas[-1], 0.2
                )
        elif self.part_deltas.shape[0] == 0:
            next_part_delta = torch.zeros(self.num_groups, 7, device="cuda")
            next_part_delta[..., 0] = 1.0  # wxyz_xyz.
        else:
            next_part_delta = self.part_deltas[-1]
        self.part_deltas = torch.nn.Parameter(
            torch.cat([self.part_deltas, next_part_delta.unsqueeze(0)], dim=0)
        )

    def create_observation_from_rgb_and_camera(
        self, rgb: np.ndarray, camera: Cameras, metric_depth: Optional[np.ndarray] = None
    ) -> PosedObservation:
        """
        Expects [H, W, C], and int.
        """
        target_frame_rgb = ToTensor()(Image.fromarray(rgb)).permute(1, 2, 0).cuda()
        def dino_fn(x):
            return self.dino_loader.get_pca_feats(x, keep_cuda=True)

        frame = PosedObservation(
            target_frame_rgb,
            camera,
            dino_fn,
            metric_depth_img=(
                None if metric_depth is None else torch.from_numpy(metric_depth)
            ),
            precompute_2Dhand_masks=True
        )
        return frame

    def detect_hands(self, frame_id: int):
        """
        Detects hands in the frame, and saves the hand info.
        """
        assert frame_id >= 0 and frame_id < len(self.sequence)
        assert self.T_objreg_objinit is not None, "Must initialize first with the first frame"

        with torch.no_grad():
            self.apply_keyframe(frame_id)
            curr_obs = self.sequence[frame_id]

            if self.object_mode == ObjectMode.RIGID_OBJECTS:
                obj_id = [i for i in range(self.num_groups) if i not in self.fixed_object_ids]
            else:
                obj_id = None
            
            outputs = cast(
                dict[str, torch.Tensor],
                self.dig_model.get_outputs(curr_obs.frame.camera, rgb_only=True, obj_id=obj_id),
            )
            object_mask = outputs["accumulation"] > self.config.mask_threshold

            # Hands in camera frame.
            left_hand, right_hand = curr_obs.frame.get_hand_3d(
                object_mask, outputs["depth"], self.dataset_scale
            )

            # Save hands in _object_ frame.
            for hand in [left_hand, right_hand]:
                if hand is None:
                    continue
                
                transform = (
                    self.T_objreg_world
                    .inverse()
                    .as_matrix()
                    .cpu()
                    .numpy()
                    .squeeze()
                )
                rotmat = transform[:3, :3]
                translation = transform[:3, 3]
                for key in ["mano_hand_global_orient", "mano_hand_pose", "verts", "keypoints_3d"]:
                    hand[key] = np.einsum("ij,...j->...i", rotmat, hand[key])
                
                hand["verts"] += translation
                hand["keypoints_3d"] += translation
            self.hands_info[frame_id] = (left_hand, right_hand)
    
    @property
    def T_objreg_world(self):
        assert self.T_objreg_objinit is not None, "Must initialize first with the first frame"
        if self.object_mode == ObjectMode.ARTICULATED:
            # Original single object case 
            return tf.SE3(self.T_world_objinit) @ tf.SE3(self.T_objreg_objinit)
        elif self.object_mode == ObjectMode.RIGID_OBJECTS:
            # Use first object's transform as base frame
            return tf.SE3(self.T_world_objinit[0:1]) @ tf.SE3(self.T_objreg_objinit[0:1])
    
    def detect_motion_phases(self, radius_pos=0.1):
        """
        Detects start and end of motion by comparing against initial pose.
        Assumes three-phase motion: stationary -> moving -> stationary
        
        Args:
            part_deltas: Tensor of shape [num_timesteps, num_parts, 7] (wxyz_xyz format)
            radius_quat: Quaternion distance threshold from initial pose
            radius_pos: Position distance threshold from initial pose (in same units as positions)
        
        Returns:
            starts: Tensor of shape [num_parts] containing start frame indices
            ends: Tensor of shape [num_parts] containing end frame indices
        """
        num_timesteps, num_parts, _ = self.part_deltas.shape
        part_deltas = self.part_deltas.detach().clone()
        
        initial_pos = part_deltas[0, :, 4:]   # [num_parts, 3]
        final_pos = part_deltas[-1, :, 4:]    # [num_parts, 3]
        
        pos_dist_to_init = torch.norm(part_deltas[:, :, 4:] - initial_pos, dim=2)                    # [num_timesteps, num_parts]
        pos_dist_to_final = torch.flip(torch.norm(part_deltas[:, :, 4:] - final_pos, dim=2), [0])    # [num_timesteps, num_parts]
        
        starts = torch.zeros(num_parts, dtype=torch.long)
        ends = torch.zeros(num_parts, dtype=torch.long)

        for part in range(num_parts):
            move_start = torch.where(pos_dist_to_init[:, part] > radius_pos)[0]
            move_end = torch.where(pos_dist_to_final[:, part] > radius_pos)[0]
            
            if len(move_start) > 0:
                starts[part] = move_start[0].item()
                ends[part] = num_timesteps - move_end[0].item()
            else:
                starts[part] = 0
                ends[part] = num_timesteps
                
        self.motion_phases = torch.zeros(num_timesteps, num_parts, dtype=torch.bool)
        for part in range(num_parts):
            self.motion_phases[starts[part]:ends[part], part] = True