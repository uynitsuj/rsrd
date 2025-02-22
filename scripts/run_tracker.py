"""
Script for running the tracker.
"""

import json
from typing import Optional, cast, Literal
import time
from pathlib import Path
from threading import Lock
# import moviepy.editor as mpy
import moviepy as mpy
import plotly.express as px

import cv2
import numpy as np
import torch
import tqdm
import warp as wp
import trimesh

import viser
import tyro
from loguru import logger

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.utils import writer
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.viewer.viewer import Viewer

from rsrd.motion.motion_optimizer import (
    RigidGroupOptimizer,
    RigidGroupOptimizerConfig,
)
import rsrd.transforms as tf
from rsrd.motion.atap_loss import ATAPConfig
from rsrd.extras.cam_helpers import (
    CameraIntr,
    IPhoneIntr,
    get_ns_camera_at_origin,
    get_vid_frame,
)
from rsrd.robot.graspable_obj import GraspableObject
from rsrd.extras.viser_rsrd import ViserRSRD
from rsrd.util.frame_detectors import Hand2DDetector, Hand3DDetector, MonoDepthEstimator

torch.set_float32_matmul_precision("high")

#Temp
import jax.numpy as jnp
from jaxtyping import Float
import jax_dataclasses as jdc
import jaxlie
from torchvision.transforms.functional import resize

from PIL import Image


def main(
    output_dir: Path,
    is_obj_jointed: Optional[bool] = None,
    dig_config_path: Optional[Path] = None, 
    # TODO: Might be able to tell if setting is multi-rigid obj or articulated from the dig state.pt(s); i.e. in the case of multi-rigid objects, will have multiple state.pts
    # Can't think of a scenario where we'd be manipulating a single rigid object w.r.t. no other object/reference, but could be wrong
    video_path: Optional[Path] = None,
    hand_mode: Literal["single", "bimanual"] = "bimanual",
    camera_intr_type: CameraIntr = IPhoneIntr(),
    save_hand: bool = True,
    plot_hands: bool = False, # Currently buggy for multi-rigid obj? Or maybe hand detection is suboptimal
    ):
    """Track objects in video using RSRD.

    If a `cache_info.json` file is found in the output directory,
    the tracker will load using the cached data + paths and skip tracking.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the paths to the cache file.
    if (output_dir / "cache_info.json").exists():
        cache_data = json.loads((output_dir / "cache_info.json").read_text())
    
        if is_obj_jointed is None:
            is_obj_jointed = bool(cache_data["is_obj_jointed"])
        if video_path is None:
            video_path = Path(cache_data["video_path"])
        if dig_config_path is None:
            dig_config_path = Path(cache_data["dig_config_path"])
    
    assert is_obj_jointed is not None, "Must provide whether the object is jointed."
    assert dig_config_path is not None, "Must provide a dig config path."
    assert video_path is not None, "Must provide a video path."

    cache_data = {
        "is_obj_jointed": is_obj_jointed,
        "video_path": str(video_path),
        "dig_config_path": str(dig_config_path),
    }
    (output_dir / "cache_info.json").write_text(json.dumps(cache_data))

    # Load video.
    assert video_path.exists(), f"Video path {video_path} does not exist."
    video = cv2.VideoCapture(str(video_path.absolute()))

    # Load DIG model, create viewer.
    train_config, pipeline, _, _ = eval_setup(dig_config_path)
    del pipeline.garfield_pipeline
    pipeline.eval()
    viewer_lock = Lock()
    Viewer(
        ViewerConfig(default_composite_depth=False, num_rays_per_chunk=-1),
        dig_config_path.parent,
        pipeline.datamanager.get_datapath(),
        pipeline,
        train_lock=viewer_lock,
    )
    # Need to set up the writer to track number of rays, otherwise the viewer will not calculate the resolution correctly.
    writer.setup_local_writer(
        train_config.logging, max_iter=train_config.max_num_iterations
    )
        
    try:
        pipeline.load_state()
        pipeline.reset_colors()
    except FileNotFoundError:
        print("No state found, starting from scratch")

    # Initialize tracker.
    wp.init()  # Must be called before any other warp API call.
    optimizer_config = RigidGroupOptimizerConfig(
        atap_config=ATAPConfig(
            loss_alpha=(1.0 if is_obj_jointed else 0.1),
        )
    )
    optimizer = RigidGroupOptimizer(
        optimizer_config,
        pipeline,
        render_lock=viewer_lock,
    )
    logger.info("Initialized tracker.")

    # Generate + load keyframes.
    camopt_render_path = output_dir / "camopt_render.mp4"
    frame_opt_path = output_dir / "frame_opt.mp4"
    track_data_path = output_dir / "keyframes.txt"
    if not track_data_path.exists():
        track_and_save_motion(
            optimizer,
            video,
            camera_intr_type,
            camopt_render_path,
            frame_opt_path,
            track_data_path,
            save_hand,
        )

    optimizer.load_tracks(track_data_path)
    
    optimizer.detect_motion_phases()
    
    # Load camera and hands info, in the object frame.
    assert optimizer.T_objreg_objinit is not None
    
    T_cam_obj = optimizer.T_objreg_world.inverse()
    
    T_cam_obj = (
        T_cam_obj @
        tf.SE3.from_rotation(tf.SO3.from_x_radians(torch.Tensor([torch.pi]).cuda()))
        @ tf.SE3.from_rotation(tf.SO3.from_z_radians(torch.Tensor([torch.pi]).cuda()))
    )
    hands = optimizer.hands_info
    assert hands is not None

    # overlay_video = cv2.VideoCapture(str(frame_opt_path))

    # Before visualizing, reset colors...
    optimizer.reset_transforms()

    server = viser.ViserServer()
    viser_rsrd = ViserRSRD(
        server, optimizer, root_node_name="/object", show_finger_keypoints=False
    )

    height, width = camera_intr_type.height, camera_intr_type.width
    # aspect = height / width

    camera_handle = server.scene.add_camera_frustum(
        "camera",
        fov=80,
        aspect=width / height,
        scale=0.1,
        position=T_cam_obj.translation().detach().cpu().numpy().squeeze(),
        wxyz=T_cam_obj.rotation().wxyz.detach().cpu().numpy().squeeze(),
    )
    @camera_handle.on_click
    def _(event: viser.GuiEvent):
        client = event.client
        if client is None:
            return
        client.camera.position = T_cam_obj.translation().detach().cpu().numpy().squeeze()
        client.camera.wxyz = T_cam_obj.rotation().wxyz.detach().cpu().numpy().squeeze()

    timesteps = len(optimizer.part_deltas)
    track_slider = server.gui.add_slider("timestep", 0, timesteps - 1, 1, 0)
    play_checkbox = server.gui.add_checkbox("play", True)
    regen_grasps = server.gui.add_button("Regenerate Grasps", disabled=True)
    
    logger.info("Performing analytical grasp sampling & scoring via finger proximity...")
    
    meshes = []
    
    def sample_grasps(optimizer):
        nonlocal meshes
        if len(meshes) > 0:
            for mesh in meshes:
                mesh.remove()
                
        meshes = []
        obj = GraspableObject(optimizer)
        
        # TODO: This is probably not going to work for multi-rigid objects where grasps happen asynchronously. Will need to modify based on detected multi-rigid mode or articulated mode
        if hand_mode == "bimanual":
            parts_moved_by_hand = obj.rank_parts_to_move_bimanual()[0] # Returns tuple with (left_idx, right_idx)
        elif hand_mode == "single":
            parts_moved_by_hand = [obj.rank_parts_to_move_single()[0]]
        
        _, new_grasps = obj.rank_grasps_from_hands()
        
        for i, part in enumerate(obj.parts):
            
            meshes.append(
                server.scene.add_mesh_trimesh(
                    f"/object/group_{i}/delta/mesh_{i}",
                    part.mesh,
                    scale = optimizer.dataset_scale,
                    visible=False,
                )
            )
            
            if i in parts_moved_by_hand:
                meshes.append(
                    server.scene.add_mesh_trimesh(
                    f"/object/group_{i}/delta/grasps/mesh",
                    new_grasps[i].to_trimesh(axes_radius=0.001, axes_height=0.05),
                    scale = optimizer.dataset_scale,
                    visible=False,
                    )
                )

                top_grasp = new_grasps[i].finger_prox_scores.argmax().item() # Find and plot the top grasp from finger proximity scores
                transform = np.eye(4)
                rotation = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
                transform[:3, :3] = rotation[:3, :3]
                mesh = trimesh.creation.cylinder(
                    radius=0.001, height=0.05, transform=transform
                )
                grasp_tf = new_grasps[i].to_se3(along_axis='x').as_matrix()[top_grasp]
                mesh.apply_transform(grasp_tf)
                mesh.visual.vertex_colors = np.array([150, 150, 255, 255])
                meshes.append(
                    server.scene.add_mesh_trimesh(
                        f"/object/group_{i}/delta/top_grasp/mesh",
                        mesh,
                        scale = optimizer.dataset_scale,
                    )
                )
    
    sample_grasps(optimizer)
    regen_grasps.disabled = False
    
    @regen_grasps.on_click
    def _(event: viser.GuiEvent):
        logger.info("Generating grasp samples, please be patient...")
        sample_grasps(optimizer)
    
    
    while True:
        if play_checkbox.value:
            track_slider.value = (track_slider.value + 1) % timesteps
        tstep = track_slider.value
        vid_frame = get_vid_frame(video, frame_idx=tstep)
        part_deltas = optimizer.part_deltas[tstep]
        viser_rsrd.update_cfg(part_deltas)
        if plot_hands:
            viser_rsrd.update_hands(tstep)
        camera_handle = server.scene.add_camera_frustum(
            "camera",
            fov=80,
            aspect=width / height,
            scale=0.05,
            position=T_cam_obj.translation().detach().cpu().numpy().squeeze(),
            wxyz=T_cam_obj.rotation().wxyz.detach().cpu().numpy().squeeze(),
            image = vid_frame
        )
        

def render_video(
    optimizer: RigidGroupOptimizer,
    motion_clip: cv2.VideoCapture,
    num_frames: int
):
    renders = []
    # Create a render for the video.
    for frame_id in tqdm.trange(0, num_frames):
        rgb = get_vid_frame(motion_clip, frame_idx=frame_id)
        optimizer.apply_keyframe(frame_id)
        with torch.no_grad():
            outputs = cast(
                dict[str, torch.Tensor],
                optimizer.dig_model.get_outputs(optimizer.sequence[frame_id].frame.camera)
            )
        
        # hand_mask = None
        # if hasattr(optimizer.sequence[frame_id], '_hand_mask'):
        #     hand_mask = optimizer.sequence[frame_id]._hand_mask
        #     hand_mask = resize(hand_mask[None, None], (optimizer.sequence[frame_id].frame.camera.height, optimizer.sequence[frame_id].frame.camera.width), antialias=True).squeeze()
        
        render = (outputs["rgb"].cpu() * 255).numpy().astype(np.uint8)
        rgb = cv2.resize(rgb, (render.shape[1], render.shape[0]))
        render = (render * 0.8 + rgb * 0.2).astype(np.uint8)
        
        renders.append(render)
    return renders
    

# But this should _really_ be in the rigid optimizer.

def track_and_save_motion(
    optimizer: RigidGroupOptimizer,
    motion_clip: cv2.VideoCapture,
    camera_type: CameraIntr,
    camopt_render_path: Path,
    frame_opt_path: Path,
    track_data_path: Path,
    save_hand: bool = False,
):
    """Get part poses for each frame in the video, ad save the keyframes to a file."""
    camera = get_ns_camera_at_origin(camera_type)
    num_frames = int(motion_clip.get(cv2.CAP_PROP_FRAME_COUNT))

    rgb = get_vid_frame(motion_clip, frame_idx=0)
    obs = optimizer.create_observation_from_rgb_and_camera(rgb, camera)

    # Initialize.
    optimizer.set_track_path(track_data_path)
    renders, final_frame  = optimizer.initialize_obj_pose(obs, render=True, niter=150, n_seeds=8)
    
    if final_frame is not None:
        final_frame_pil = Image.fromarray(final_frame.astype(np.uint8))
        final_frame_pil.save(str(camopt_render_path).replace('.mp4','_final_opt.png'))
    
    if renders is not None:
        # Save the frames.
        out_clip = mpy.ImageSequenceClip(renders, fps=30)
        out_clip.write_videofile(str(camopt_render_path), codec="libx264",bitrate='5000k')
        out_clip.write_videofile(str(camopt_render_path).replace('.mp4','_mac_compat.mp4'),codec='mpeg4',bitrate='5000k')
    
    for frame_id in tqdm.trange(0, num_frames):
        try:
            rgb = get_vid_frame(motion_clip, frame_idx=frame_id)
        except ValueError:
            num_frames = frame_id
            break

        obs = optimizer.create_observation_from_rgb_and_camera(rgb, camera)
        optimizer.add_observation(obs)
        optimizer.fit([frame_id], 50)
        # if num_frames > 100:
        #     obs.clear_cache() # Clear the cache to save memory (can overflow on very long videos)
        if frame_id % 20 == 0:
            import gc
            gc.collect()
            
        if save_hand:
            optimizer.detect_hands(frame_id)
    
    # Save a pre-smoothing video
    renders = render_video(optimizer, motion_clip, num_frames)
    out_clip = mpy.ImageSequenceClip(renders, fps=30)
    out_clip.write_videofile(str(frame_opt_path).replace(".mp4","_pre_smooth.mp4"), codec="libx264",bitrate='5000k')
    out_clip.write_videofile(str(frame_opt_path).replace('.mp4','_pre_smooth_mac_compat.mp4'),codec='mpeg4',bitrate='5000k')
    
    # Save part trajectories pre-smooth in case it OOMs.
    # optimizer.save_tracks(track_data_path)
    
    # Smooth all frames, together.
    logger.info("Performing temporal smoothing...")
    
    # Generate overlapping segments of 10 frames to smooth.
    # segment_len = 10
    # for start in tqdm.trange(0, num_frames, segment_len):
    #     end = min(start + segment_len, num_frames)
    #     optimizer.fit(list(range(start, end)), 50)
    optimizer.fit(list(range(num_frames)), 50)
    logger.info("Finished temporal smoothing.")

    # Save part trajectories.
    optimizer.save_tracks(track_data_path)

    renders = render_video(optimizer, motion_clip, num_frames)
    # Save the final video
    out_clip = mpy.ImageSequenceClip(renders, fps=30)
    out_clip.write_videofile(str(frame_opt_path), codec="libx264",bitrate='5000k')
    out_clip.write_videofile(str(frame_opt_path).replace('.mp4','_mac_compat.mp4'),codec='mpeg4',bitrate='5000k')

def save_frame_to_disk(frame: np.ndarray, output_dir: Path, frame_idx: int) -> Path:
    """Save a single frame to disk.
    
    Args:
        frame: numpy array of frame data
        output_dir: directory to save frames in
        frame_idx: index of the frame
        
    Returns:
        Path to saved frame
    """
    frame_dir = output_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_path = frame_dir / f"frame_{frame_idx:06d}.png"
    cv2.imwrite(str(frame_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    return frame_path

def create_video_from_frames(frame_dir: Path, output_path: Path, fps: int = 30):
    """Create a video from frames saved on disk.
    
    Args:
        frame_dir: directory containing the frames
        output_path: path to save the video
        fps: frames per second for the video
    """
    frame_paths = sorted(frame_dir.glob("frame_*.png"))
    if not frame_paths:
        raise ValueError(f"No frames found in {frame_dir}")
        
    # Read first frame to get dimensions
    first_frame = cv2.imread(str(frame_paths[0]))
    height, width = first_frame.shape[:2]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Write frames
    for frame_path in tqdm.tqdm(frame_paths, desc="Creating video"):
        frame = cv2.imread(str(frame_path))
        writer.write(frame)
    
    writer.release()
    
if __name__ == "__main__":
    tyro.cli(main)
