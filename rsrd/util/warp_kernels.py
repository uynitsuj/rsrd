import warp as wp


@wp.func
def poses_7vec_to_transform(poses: wp.array(dtype=float, ndim=2), i: int):
    """
    Kernel helper for converting wxyz_xyz to a wp.Transformation (xyzw, xyz).
    """
    quaternion = wp.quaternion(poses[i, 1], poses[i, 2], poses[i, 3], poses[i, 0])
    position = wp.vector(poses[i, 4], poses[i, 5], poses[i, 6])
    return wp.transformation(position, quaternion)


@wp.func
def poses_7vec_to_transform(poses: wp.array(dtype=float, ndim=3), i: int, j: int):
    """
    Kernel helper for converting wxyz_xyz to a wp.Transformation
    """
    quaternion = wp.quaternion(
        poses[i, j, 1], poses[i, j, 2], poses[i, j, 3], poses[i, j, 0]
    )
    position = wp.vector(poses[i, j, 4], poses[i, j, 5], poses[i, j, 6])
    return wp.transformation(position, quaternion)


@wp.kernel
def apply_to_model_warp(
    init_o2w: wp.array(dtype=float, ndim=2),
    init_p2os: wp.array(dtype=float, ndim=2),
    o_delta: wp.array(dtype=float, ndim=2),
    p_deltas: wp.array(dtype=float, ndim=2),
    group_labels: wp.array(dtype=int),
    means: wp.array(dtype=wp.vec3),
    quats: wp.array(dtype=float, ndim=2),
    # outputs
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
    o2w_T = poses_7vec_to_transform(init_o2w, 0)
    p2o_T = poses_7vec_to_transform(init_p2os, group_id)
    odelta_T = poses_7vec_to_transform(o_delta, 0)
    pdelta_T = poses_7vec_to_transform(p_deltas, group_id)
    g2w_T = wp.transformation(
        means[tid],
        wp.quaternion(quats[tid, 1], quats[tid, 2], quats[tid, 3], quats[tid, 0]),
    )
    g2p_T = wp.transform_inverse(p2o_T) * wp.transform_inverse(o2w_T) * g2w_T
    new_g2w_T = o2w_T * odelta_T * p2o_T * pdelta_T * g2p_T
    means_out[tid] = wp.transform_get_translation(new_g2w_T)
    new_quat = wp.transform_get_rotation(new_g2w_T)
    quats_out[tid, 0] = new_quat[3]  # w
    quats_out[tid, 1] = new_quat[0]  # x
    quats_out[tid, 2] = new_quat[1]  # y
    quats_out[tid, 3] = new_quat[2]  # z

@wp.kernel
def apply_to_model_warp_multi_object(
    init_o2w: wp.array(dtype=float, ndim=2),
    obj_deltas: wp.array(dtype=float, ndim=2),
    part_deltas: wp.array(dtype=float, ndim=2),
    # part_labels: wp.array(dtype=int),
    obj_labels: wp.array(dtype=int),
    means: wp.array(dtype=wp.vec3),
    quats: wp.array(dtype=float, ndim=2),
    # outputs
    means_out: wp.array(dtype=wp.vec3),
    quats_out: wp.array(dtype=float, ndim=2),
):
    """
    Kernel for applying hierarchical transforms to gaussian splats in a multi-object scene
    with articulated objects (objects with multiple movable parts).

    Transform hierarchy: Gaussian -> Part -> Object -> World

    Parameters:
    -----------
    init_o2w: Mx7 tensor
        Initial object-to-world poses for M objects. Each pose is a 7D vector [x,y,z, qx,qy,qz,qw]
    init_p2o: Px7 tensor
        Initial part-to-object poses for P parts across all objects. Each pose is a 7D vector
    obj_deltas: Mx7 tensor
        Object pose deltas (objnew_to_objoriginal) for each of the M objects
    part_deltas: Px7 tensor
        Part pose deltas (partnew_to_partoriginal) for each of the P parts
    part_labels: N tensor
        Labels mapping each Gaussian to its corresponding part (0->P-1)
    obj_labels: N tensor
        Labels mapping each Gaussian to its corresponding object (0->M-1)
    means: Nx3 tensor
        Gaussian means in world space
    quats: Nx4 tensor
        Gaussian orientations as quaternions (wxyz)
    
    Outputs:
    --------
    means_out: Nx3 tensor
        Transformed Gaussian means
    quats_out: Nx4 tensor
        Transformed Gaussian orientations (wxyz)

    Transform Chain:
    --------------
    1. Gaussian to Part: g2p_T = inv(p2o_T) * inv(o2w_T) * g2w_T
    2. Apply Deltas: new_g2w_T = o2w_T * obj_delta_T * p2o_T * part_delta_T * g2p_T
    """
    tid = wp.tid()
    # part_id = part_labels[tid]  # Which part this Gaussian belongs to
    obj_id = obj_labels[tid]   # Which object this Gaussian belongs to
    
    # Get the initial transforms
    o2w_T = poses_7vec_to_transform(init_o2w, obj_id)      # Object -> World
    # p2o_T = poses_7vec_to_transform(init_p2o, part_id)     # Part -> Object
    
    # Get the delta transforms
    obj_delta_T = poses_7vec_to_transform(obj_deltas, obj_id)    # Object registration
    part_delta_T = poses_7vec_to_transform(part_deltas, obj_id) # Sub-Part motion
    
    # Current Gaussian pose in world space
    g2w_T = wp.transformation(
        means[tid],
        wp.quaternion(quats[tid, 1], quats[tid, 2], quats[tid, 3], quats[tid, 0]),
    )
    
    # g2p_T = wp.transform_inverse(p2o_T) * wp.transform_inverse(o2w_T) * g2w_T
    
    g2p_T = wp.transform_inverse(o2w_T) * g2w_T

    # new_g2w_T = o2w_T * obj_delta_T * p2o_T * part_delta_T * g2p_T
    
    new_g2w_T = o2w_T * obj_delta_T * part_delta_T * g2p_T
    
    means_out[tid] = wp.transform_get_translation(new_g2w_T)
    new_quat = wp.transform_get_rotation(new_g2w_T)
    quats_out[tid, 0] = new_quat[3]  # w
    quats_out[tid, 1] = new_quat[0]  # x
    quats_out[tid, 2] = new_quat[1]  # y
    quats_out[tid, 3] = new_quat[2]  # z

@wp.kernel
def traj_smoothness_loss_warp(
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
    pose = poses_7vec_to_transform(deltas, t, n)
    local_loss = float(0.0)
    dummy_axis = wp.vector(0.0, 0.0, 0.0)
    if t > 0:
        prev_pose = poses_7vec_to_transform(deltas, t - 1, n)
        local_loss += (
            wp.length_sq(
                wp.transform_get_translation(pose)
                - wp.transform_get_translation(prev_pose)
            )
            * position_lambda
        )
        q_delta_1 = wp.transform_get_rotation(pose) * wp.quat_inverse(
            wp.transform_get_rotation(prev_pose)
        )
        q_dist_1 = float(0.0)
        wp.quat_to_axis_angle(q_delta_1, dummy_axis, q_dist_1)
        local_loss += q_dist_1 * rotation_lambda
    if t < deltas.shape[0] - 1:
        next_pose = poses_7vec_to_transform(deltas, t + 1, n)
        local_loss += (
            wp.length_sq(
                wp.transform_get_translation(pose)
                - wp.transform_get_translation(next_pose)
            )
            * position_lambda
        )
        q_delta_2 = wp.transform_get_rotation(pose) * wp.quat_inverse(
            wp.transform_get_rotation(next_pose)
        )
        q_dist_2 = float(0.0)
        wp.quat_to_axis_angle(q_delta_2, dummy_axis, q_dist_2)
        local_loss += q_dist_2 * rotation_lambda

    loss[t, n] = local_loss


# https://openaccess.thecvf.com/content_CVPR_2019/papers/Barron_A_General_and_Adaptive_Robust_Loss_Function_CVPR_2019_paper.pdf
@wp.func
def jon_loss(x: float, alpha: float, c: float):
    pow_part = ((x / c) ** 2.0) / wp.abs(alpha - 2.0) + 1.0
    return (wp.abs(alpha - 2.0) / alpha) * (wp.pow(pow_part, alpha / 2.0) - 1.0)


@wp.kernel
def atap_loss_warp(
    cur_means: wp.array(dtype=wp.vec3),
    dists: wp.array(dtype=float),
    ids: wp.array(dtype=int),
    match_ids: wp.array(dtype=int),
    group_ids1: wp.array(dtype=int),
    group_ids2: wp.array(dtype=int),
    connectivity_weights: wp.array(dtype=float, ndim=2),
    loss: wp.array(dtype=float),
    alpha: float,
):
    tid = wp.tid()
    id1 = ids[tid]
    id2 = match_ids[tid]
    gid1 = group_ids1[tid]
    gid2 = group_ids2[tid]
    con_weight = connectivity_weights[gid1, gid2]
    curdist = wp.length(cur_means[id1] - cur_means[id2])
    loss[tid] = jon_loss(curdist - dists[tid], alpha, 0.001) * con_weight * 0.001
