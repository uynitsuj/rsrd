import warp as wp
import torch
from toad.transforms import SE3,SO3

def extrapolate_poses(p1_7v, p2_7v):    
    r1 = SO3(p1_7v[...,3:])
    t1 = SE3.from_rotation_and_translation(r1, p1_7v[...,:3])
    r2 = SO3(p2_7v[...,3:])
    t2 = SE3.from_rotation_and_translation(r2, p2_7v[...,:3])
    t_2_1 = t1.inverse() @ t2
    new_t = (t2 @ t_2_1)
    return new_t.wxyz_xyz.roll(3,dims=-1)

def zero_optim_state(optimizer:torch.optim.Adam, timestamps):
    param = optimizer.param_groups[0]["params"][0]
    param_state = optimizer.state[param]
    if "max_exp_avg_sq" in param_state:
        # for amsgrad
        param_state["max_exp_avg_sq"][timestamps] = 0.0
    if "exp_avg" in param_state:
        param_state["exp_avg"][timestamps] = 0.0
        param_state["exp_avg_sq"][timestamps] = 0.0

def append_in_optim(optimizer:torch.optim.Adam, new_params):
    """adds the parameters to the optimizer"""
    param = optimizer.param_groups[0]["params"][0]
    param_state = optimizer.state[param]
    if "max_exp_avg_sq" in param_state:
        # for amsgrad
        param_state["max_exp_avg_sq"] = torch.cat(
            [
                param_state["max_exp_avg_sq"],
                param_state["max_exp_avg_sq"][-1].unsqueeze(0)
            ],
            dim=0,
        )
    if "exp_avg" in param_state:
        param_state["exp_avg"] = torch.cat(
            [
                param_state["exp_avg"],
                param_state["exp_avg"][-1].unsqueeze(0)
            ],
            dim=0,
        )
        param_state["exp_avg_sq"] = torch.cat(
            [
                param_state["exp_avg_sq"],
                param_state["exp_avg_sq"][-1].unsqueeze(0)
            ],
            dim=0,
        )

    del optimizer.state[param]
    optimizer.state[new_params[0]] = param_state
    optimizer.param_groups[0]["params"] = new_params
    del param

@wp.func
def poses_7vec_to_transform(poses: wp.array(dtype=float, ndim=2), i: int):
    """
    Kernel helper for converting x y z qw qx qy qz to a wp.Transformation
    """
    position = wp.vector(poses[i,0], poses[i,1], poses[i,2])
    quaternion = wp.quaternion(poses[i,4], poses[i,5], poses[i,6], poses[i,3])
    return wp.transformation(position, quaternion)

@wp.func
def poses_7vec_to_transform(poses: wp.array(dtype=float, ndim=3), i: int, j: int):
    """
    Kernel helper for converting x y z qw qx qy qz to a wp.Transformation
    """
    position = wp.vector(poses[i, j, 0], poses[i, j, 1], poses[i, j, 2])
    quaternion = wp.quaternion(poses[i, j, 4], poses[i, j, 5], poses[i, j, 6], poses[i, j, 3])
    return wp.transformation(position, quaternion)

@wp.kernel
def apply_to_model(
    init_o2w: wp.array(dtype=float, ndim=2),
    init_p2os: wp.array(dtype=float, ndim=2),
    o_delta: wp.array(dtype=float, ndim=2),
    p_deltas: wp.array(dtype=float, ndim=2),
    group_labels: wp.array(dtype=int),
    means: wp.array(dtype=wp.vec3),
    quats: wp.array(dtype=float, ndim=2),
    #outputs
    means_out: wp.array(dtype=wp.vec3),
    quats_out: wp.array(dtype=float, ndim=2),
):
    """
    Kernel for applying the transforms to a gaussian splat

    init_o2w: 1x7 tensor of initial object to world poses
    init_p2os: Nx7 tensor of initial pose to object poses
    o_delta: Nx7 tensor of object pose deltas represented as objnew_to_objoriginal
    p_deltas: Nx7 tensor of pose deltas represented as partnew_to_partoriginal
    group_labels: N, tensor of group labels (0->K-1) for K groups
    means: Nx3 tensor of means
    quats: Nx4 tensor of quaternions (wxyz)
    means_out: Nx3 tensor of output means
    quats_out: Nx4 tensor of output quaternions (wxyz)
    """
    tid = wp.tid()
    group_id = group_labels[tid]
    o2w_T = poses_7vec_to_transform(init_o2w,0)
    p2o_T = poses_7vec_to_transform(init_p2os,group_id)
    odelta_T = poses_7vec_to_transform(o_delta,0)
    pdelta_T = poses_7vec_to_transform(p_deltas,group_id)
    g2w_T = wp.transformation(means[tid], wp.quaternion(quats[tid, 1], quats[tid, 2], quats[tid, 3], quats[tid, 0]))
    g2p_T = wp.transform_inverse(p2o_T) * wp.transform_inverse(o2w_T) * g2w_T
    new_g2w_T = o2w_T * odelta_T * p2o_T * pdelta_T * g2p_T
    means_out[tid] = wp.transform_get_translation(new_g2w_T)
    new_quat = wp.transform_get_rotation(new_g2w_T)
    quats_out[tid, 0] = new_quat[3] #w
    quats_out[tid, 1] = new_quat[0] #x
    quats_out[tid, 2] = new_quat[1] #y
    quats_out[tid, 3] = new_quat[2] #z

@wp.kernel
def traj_smoothness_loss(
    deltas: wp.array(dtype=float, ndim=3),
    position_lambda: float,
    rotation_lambda: float,
    loss: wp.array(dtype=float, ndim=2),
):
    """
    Kernel for computing the smoothness loss of a trajectory

    deltas: TxNx7 tensor of deltas, where T is the number of frames, N is the number of parts, and 7 is the pose vector
    loss: TxN tensor of loss per-pose, computed by the weighted sum of position and rotation distance from neighbors
    """
    t, n = wp.tid()
    pose = poses_7vec_to_transform(deltas,t,n)
    local_loss = float(0.0)
    dummy_axis = wp.vector(0.,0.,0.)
    if t>0:
        prev_pose = poses_7vec_to_transform(deltas,t-1,n)
        local_loss += wp.length_sq(wp.transform_get_translation(pose) - wp.transform_get_translation(prev_pose)) * position_lambda
        q_delta_1 = wp.transform_get_rotation(pose) * wp.quat_inverse(wp.transform_get_rotation(prev_pose))
        q_dist_1 = float(0.0)
        wp.quat_to_axis_angle(q_delta_1, dummy_axis, q_dist_1)
        local_loss += q_dist_1 * rotation_lambda
    if t < deltas.shape[0] - 1:
        next_pose = poses_7vec_to_transform(deltas,t+1,n)
        local_loss += wp.length_sq(wp.transform_get_translation(pose) - wp.transform_get_translation(next_pose)) * position_lambda
        q_delta_2 = wp.transform_get_rotation(pose) * wp.quat_inverse(wp.transform_get_rotation(next_pose))
        q_dist_2 = float(0.0)
        wp.quat_to_axis_angle(q_delta_2, dummy_axis, q_dist_2)
        local_loss += q_dist_2 * rotation_lambda

    loss[t,n] = local_loss
    

def identity_7vec(device='cuda'):
    """
    Returns a 7-tensor of identity pose
    """
    return torch.tensor([[0, 0, 0, 1, 0, 0, 0]], dtype=torch.float32, device=device)

def normalized_quat_to_rotmat(quat):
    """
    Converts a quaternion to a 3x3 rotation matrix
    """
    assert quat.shape[-1] == 4, quat.shape
    w, x, y, z = torch.unbind(quat, dim=-1)
    mat = torch.stack(
        [
            1 - 2 * (y**2 + z**2),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x**2 + z**2),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x**2 + y**2),
        ],
        dim=-1,
    )
    return mat.reshape(quat.shape[:-1] + (3, 3))

def torch_posevec_to_mat(posevecs):
    """
    Converts a Nx7-tensor to Nx4x4 matrix

    posevecs: Nx7 tensor of pose vectors
    returns: Nx4x4 tensor of transformation matrices
    """
    assert posevecs.shape[-1] == 7, posevecs.shape
    assert len(posevecs.shape) == 2, posevecs.shape
    out = torch.eye(4, device=posevecs.device).unsqueeze(0).expand(posevecs.shape[0], -1, -1)
    out[:, :3, 3] = posevecs[:, :3]
    out[:, :3, :3] = normalized_quat_to_rotmat(posevecs[:, 3:])
    return out

def mnn_matcher(feat_a, feat_b):
    """
    Returns mutual nearest neighbors between two sets of features

    feat_a: NxD
    feat_b: MxD
    return: K, K (indices in feat_a and feat_b)
    """
    device = feat_a.device
    sim = feat_a.mm(feat_b.t())
    nn12 = torch.max(sim, dim=1)[1]
    nn21 = torch.max(sim, dim=0)[1]
    ids1 = torch.arange(0, sim.shape[0], device=device)
    mask = ids1 == nn21[nn12]
    return ids1[mask], nn12[mask]