"""
Articulated gaussians, visible in the 3D viser viewer.
"""

import numpy as onp
import numpy.typing as onpt
import torch
import viser
import viser.transforms as vtf
import trimesh
from rsrd.motion.motion_optimizer import RigidGroupOptimizer
from typing import Literal, Type, Mapping, Any, Optional, List, Dict, TypedDict
from jaxmp.extras.urdf_loader import load_urdf
import viser.extras
from pathlib import Path
from plyfile import PlyData
import time
from rsrd.robot.graspable_obj import GraspablePart
from rsrd.util.common import identity_7vec


class GraspDevMRO:
    """Add grasps to RSRD's rigid objects in Viser."""
    
    class SplatFile(TypedDict):
        """Data loaded from an antimatter15-style splat file."""

        centers: onpt.NDArray[onp.floating]
        """(N, 3)."""
        rgbs: onpt.NDArray[onp.floating]
        """(N, 3). Range [0, 1]."""
        opacities: onpt.NDArray[onp.floating]
        """(N, 1). Range [0, 1]."""
        covariances: onpt.NDArray[onp.floating]
        """(N, 3, 3)."""
        
    _server: viser.ViserServer
    optimizer: RigidGroupOptimizer
    _scale: float

    def __init__(
        self,
        server: viser.ViserServer,
        pipeline,
        out_path,
        scale: float = 1.0,
        tcp_offset: float = 0.12,
    ):
        self._server = server

        self._scale = scale
        self.tcp_offset = tcp_offset
        self.pipeline = pipeline
        
        while self.pipeline.cluster_labels is None:
            time.sleep(0.5) # lazy wait for cluster labels to be available
            
        self.out_path = out_path
        self.grasp_axis_mesh_handles = []
        self.mesh_handles = []
        self.group_idx = 0
        self.rigid_state_files = sorted(self.pipeline.state_dir.glob("state_rigid_*.pt"), key=lambda x: x.stem.split('_')[3])  # Extract timestamp from filename
        self.rigid_state_handle = self._server.gui.add_dropdown('View Rigid State', [str(f.stem) for f in self.rigid_state_files])
        self._load_rigid_state()

        @self.rigid_state_handle.on_update
        def _(_):
            self.clear_scene()
            self._load_rigid_state()
            
        self.save_grasp_btn = self._server.gui.add_button("Save Grasp")
        @self.save_grasp_btn.on_click
        def _(_):
            self._save_grasp()
        
    def clear_scene(self):
        self.mesh_handle.remove()
        self.grasp_dev_slider_handle.remove()
        for mesh in self.mesh_handles:
            mesh.remove()
        for grasp_axis in self.grasp_axis_mesh_handles:
            grasp_axis.remove()
            
        self.grasp_axis_mesh_handles = []
        self.mesh_handles = []
        
    def _load_rigid_state(self):
        self.group_idx = [str(f.stem) for f in self.rigid_state_files].index(self.rigid_state_handle.value)
        state_path = self.pipeline.state_dir / self.rigid_state_handle.value
        gaussian_ply_path =  state_path / 'point_cloud.ply'
        self._load_gaussian_ply(gaussian_ply_path)
        
        # First try to find fixed_normals_centered obj file
        mesh_obj_path = list(state_path.glob("*_fixed_normals_centered.obj"))
        if not mesh_obj_path:
            # If not found, look for any obj file
            mesh_obj_path = list(state_path.glob("*.obj"))
            if not mesh_obj_path:
                print(f"Warning: No .obj files found in {state_path}")
                return
        self._load_mesh(mesh_obj_path[0])
    
    def _load_gaussian_ply(self, ply_path: Path):
        splat_data = self.load_ply_file(ply_path, center=True)
            
        self.graspable_part = GraspablePart.from_points(splat_data["centers"], max_width=0.045)
        
        self.mesh_handles.append(
            self._server.scene.add_mesh_trimesh(
                f"/graspable_mesh/mesh",
                self.graspable_part.mesh,
                # scale = 1,
                visible=False,
            )
        )
        # Save the graspable part mesh to file
        mesh_save_path = self.pipeline.state_dir / self.rigid_state_handle.value / 'graspable_part_mesh.obj'
        self.graspable_part.mesh.export(mesh_save_path)
        print(f"Saved graspable part mesh to {mesh_save_path}")
        
        self.grasp_axis_mesh_handles.append(
                self._server.scene.add_mesh_trimesh(
                f"/graspable_mesh/grasps/mesh",
                self.graspable_part._grasps.to_trimesh(axes_radius=0.001, axes_height=0.05),
                visible=True,
                )
            )
        
        self.grasp_dev_slider_handle = self._server.gui.add_slider('Grasp Index', 0, len(self.graspable_part._grasps.centers) - 1, 1, 0)
        
        if hasattr(self, "gripper_width_slider"):
            self.gripper_width_slider.remove()
        self._update_grasp_dev()
        
        @self.grasp_dev_slider_handle.on_update
        def _(_):
            self.grip_tcp_frame.position = self.graspable_part._grasps.centers[int(self.grasp_dev_slider_handle.value)]
            self.grip_tcp_frame.wxyz = self.graspable_part._grasps.to_se3(along_axis='x').rotation().wxyz[int(self.grasp_dev_slider_handle.value)]
            
        # self.gs_handle = self._server.scene._add_gaussian_splats(
        #     f"/gaussian_splat",
        #     centers=splat_data["centers"],
        #     rgbs=splat_data["rgbs"],
        #     opacities=splat_data["opacities"],
        #     covariances=splat_data["covariances"],
        # )
    
    def _update_grasp_dev(self):
        # Load URDF.
        self.gripper_urdf = load_urdf(
            robot_urdf_path=Path(__file__).parent
            / "../../data/yumi_description/urdf/yumi_servo_gripper.urdf"
        )

        self.grip_tcp_frame = self._server.scene.add_transform_controls("/grip_tcp_frame", position=onp.array([0,0,self.tcp_offset]), scale=0.05, line_width= 4.5, disable_sliders=True, disable_axes=False, depth_test=False) #, rotation_limits=((-1000.0, 1000.0), (0.0, 0.0), (0.0, 0.0)))
        self.grip_base_frame = self._server.scene.add_frame("/grip_tcp_frame/grip_base_frame", position=onp.array([0,0,-self.tcp_offset]), visible=True, show_axes=False)

        viser_urdf = viser.extras.ViserUrdf(self._server, self.gripper_urdf, root_node_name="/grip_tcp_frame/grip_base_frame/yumi_gripper")
        viser_urdf.update_cfg([0])
        
        self.gripper_width_slider = self._server.gui.add_slider('Gripper Width', 0.0, 0.025, 0.001, 0.0)

        @self.gripper_width_slider.on_update
        def _(_):
            viser_urdf.update_cfg([float(self.gripper_width_slider.value)])
            
        self.grip_tcp_frame.position = self.graspable_part._grasps.centers[int(self.grasp_dev_slider_handle.value)]
        self.grip_tcp_frame.wxyz = self.graspable_part._grasps.to_se3(along_axis='x').rotation().wxyz[int(self.grasp_dev_slider_handle.value)]
        
    def _load_mesh(self, mesh_path: Path):
        mesh = trimesh.load_mesh(mesh_path)
        # Save fixed mesh to file with name _fixed.obj
        if len(sorted(mesh_path.parent.glob("*_fixed_normals_centered.obj"))) == 0:
            trimesh.repair.fix_normals(mesh)
            translation = -self.obj2w.squeeze(0)
            mesh.apply_translation(translation)
            mesh.export(mesh_path.parent / f"{mesh_path.stem}_fixed_normals_centered.obj")
        
        self.mesh_handle = self._server.scene.add_mesh_trimesh("/mesh", mesh, visible=True)
    
    def _save_grasp(self):
        # assert self.grip_tcp_frame in list(self.graspable_part._grasps.centers), "Grasp not on a graspable part center"
        grasp_pos = self.grip_tcp_frame.position
        grasp_wxyz = self.grip_tcp_frame.wxyz
        grasp_width = self.gripper_width_slider.value
        grasp_width = float(grasp_width)
        grasp_data = {
            'position': grasp_pos.tolist(),
            'orientation': grasp_wxyz.tolist(),
            'width': grasp_width
        }
        grasp_file = self.rigid_state_files[self.group_idx].parent / self.rigid_state_files[self.group_idx].stem  / 'grasps.txt'

        import json

        if grasp_file.exists():
            with open(grasp_file, 'r') as f:
                try:
                    existing_grasps = json.load(f)
                except json.JSONDecodeError:
                    existing_grasps = []
                    
            # Check for duplicates
            is_duplicate = False
            for existing_grasp in existing_grasps:
                if (onp.allclose(existing_grasp['position'], grasp_data['position']) and
                    onp.allclose(existing_grasp['orientation'], grasp_data['orientation']) and
                    onp.allclose(existing_grasp['width'], grasp_data['width'])):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                existing_grasps.append(grasp_data)
                with open(grasp_file, 'w') as f:
                    json.dump(existing_grasps, f, indent=2)
                print(f"Grasp saved to {grasp_file}")
            else:
                print("Grasp already exists, skipping save")
        else:
            with open(grasp_file, 'w') as f:
                json.dump([grasp_data], f, indent=2)
            print(f"Created new grasp file at {grasp_file}")
    
    def load_ply_file(self, ply_file_path: Path, center: bool = False) -> SplatFile:
        """Load Gaussians stored in a PLY file."""
        start_time = time.time()

        SH_C0 = 0.28209479177387814

        plydata = PlyData.read(ply_file_path)
        v = plydata["vertex"]
        positions = onp.stack([v["x"], v["y"], v["z"]], axis=-1)
        scales = onp.exp(onp.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1))
        wxyzs = onp.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1)
        colors = 0.5 + SH_C0 * onp.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
        opacities = 1.0 / (1.0 + onp.exp(-v["opacity"][:, None]))

        Rs = vtf.SO3(wxyzs).as_matrix()
        covariances = onp.einsum(
            "nij,njk,nlk->nil", Rs, onp.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs
        )
        if center:
            self.obj2w = onp.mean(positions, axis=0, keepdims=True)
            positions -= self.obj2w

        num_gaussians = len(v)
        print(
            f"PLY file {ply_file_path} with {num_gaussians=} loaded in {time.time() - start_time} seconds"
        )
        return {
            "centers": positions,
            "rgbs": colors,
            "opacities": opacities,
            "covariances": covariances,
        }


class GraspDevArticulated:
    """Add grasps to RSRD's articulated objects in Viser."""
    
    class SplatFile(TypedDict):
        """Data loaded from an antimatter15-style splat file."""

        centers: onpt.NDArray[onp.floating]
        """(N, 3)."""
        rgbs: onpt.NDArray[onp.floating]
        """(N, 3). Range [0, 1]."""
        opacities: onpt.NDArray[onp.floating]
        """(N, 1). Range [0, 1]."""
        covariances: onpt.NDArray[onp.floating]
        """(N, 3, 3)."""
        
    _server: viser.ViserServer
    optimizer: RigidGroupOptimizer
    _scale: float

    def __init__(
        self,
        server: viser.ViserServer,
        pipeline,
        out_path,
        scale: float = 1.0,
        tcp_offset: float = 0.12,
    ):
        self._server = server

        self._scale = scale
        self.tcp_offset = tcp_offset
        self.pipeline = pipeline
        
        while self.pipeline.cluster_labels is None:
            time.sleep(0.5) # lazy wait for cluster labels to be available
            
        self.out_path = out_path
        self.grasp_axis_mesh_handles = []
        self.mesh_handles = []
        self.group_idx = 0
        self.state_file = sorted(self.pipeline.state_dir.glob("state.pt"), key=lambda x: x.stem)
        self.sub_part_files = sorted(self.pipeline.state_dir.glob("*_sub_part_*"), key=lambda x: x.stem)
        self.subpart_handle = self._server.gui.add_dropdown('View Subpart State', [str(f.stem) for f in self.sub_part_files])

        assert len(self.state_file) == 1, f"Expected 1 state file, got {len(self.state_file)}"
        
        # save pipeline's init_p2o to txt file for isaaclab
        self._save_init_p2o()
        
        self._load_subpart()
            
        @self.subpart_handle.on_update
        def _(_):
            self.clear_scene()
            self._load_subpart()
            
        self.save_grasp_btn = self._server.gui.add_button("Save Grasp")
        @self.save_grasp_btn.on_click
        def _(_):
            self._save_grasp()
        
    def clear_scene(self):
        self.mesh_handle.remove()
        self.grasp_dev_slider_handle.remove()
        for mesh in self.mesh_handles:
            mesh.remove()
        for grasp_axis in self.grasp_axis_mesh_handles:
            grasp_axis.remove()
            
        self.grasp_axis_mesh_handles = []
        self.mesh_handles = []
        
    def _load_subpart(self):
        self.group_idx = [str(f.stem) for f in self.sub_part_files].index(self.subpart_handle.value)
        sub_part_file = self.pipeline.state_dir / self.subpart_handle.value
        gaussian_ply_path =  sub_part_file / 'point_cloud.ply'
        self._load_gaussian_ply(gaussian_ply_path)
            
        # First try to find fixed_normals_centered obj file
        mesh_obj_path = list(sub_part_file.glob("*_fixed_normals_centered.obj"))
        if not mesh_obj_path:
            # If not found, look for any obj file
            mesh_obj_path = list(sub_part_file.glob("*.obj"))
            if not mesh_obj_path:
                print(f"Warning: No .obj files found in {sub_part_file}")
                return
        self._load_mesh(mesh_obj_path[0])
    
    def _load_gaussian_ply(self, ply_path: Path):
        splat_data = self.load_ply_file(ply_path, center=True)
            
        self.graspable_part = GraspablePart.from_points(splat_data["centers"], max_width=0.045)
        
        self.mesh_handles.append(
            self._server.scene.add_mesh_trimesh(
                f"/graspable_mesh/mesh",
                self.graspable_part.mesh,
                # scale = 1,
                visible=False,
            )
        )
        
        self.grasp_axis_mesh_handles.append(
                self._server.scene.add_mesh_trimesh(
                f"/graspable_mesh/grasps/mesh",
                self.graspable_part._grasps.to_trimesh(axes_radius=0.001, axes_height=0.05),
                visible=True,
                )
            )
        
        self.grasp_dev_slider_handle = self._server.gui.add_slider('Grasp Index', 0, len(self.graspable_part._grasps.centers) - 1, 1, 0)
        
        if hasattr(self, "gripper_width_slider"):
            self.gripper_width_slider.remove()
        self._update_grasp_dev()
        
        @self.grasp_dev_slider_handle.on_update
        def _(_):
            self.grip_tcp_frame.position = self.graspable_part._grasps.centers[int(self.grasp_dev_slider_handle.value)]
            self.grip_tcp_frame.wxyz = self.graspable_part._grasps.to_se3(along_axis='x').rotation().wxyz[int(self.grasp_dev_slider_handle.value)]
            
        # self.gs_handle = self._server.scene._add_gaussian_splats(
        #     f"/gaussian_splat",
        #     centers=splat_data["centers"],
        #     rgbs=splat_data["rgbs"],
        #     opacities=splat_data["opacities"],
        #     covariances=splat_data["covariances"],
        # )
    
    def _update_grasp_dev(self):
        # Load URDF.
        self.gripper_urdf = load_urdf(
            robot_urdf_path=Path(__file__).parent
            / "../../data/yumi_description/urdf/yumi_servo_gripper.urdf"
        )

        self.grip_tcp_frame = self._server.scene.add_transform_controls("/grip_tcp_frame", position=onp.array([0,0,self.tcp_offset]), scale=0.05, line_width= 4.5, disable_sliders=True, disable_axes=True, depth_test=False) #, rotation_limits=((-1000.0, 1000.0), (0.0, 0.0), (0.0, 0.0)))
        self.grip_base_frame = self._server.scene.add_frame("/grip_tcp_frame/grip_base_frame", position=onp.array([0,0,-self.tcp_offset]), visible=True, show_axes=False)

        viser_urdf = viser.extras.ViserUrdf(self._server, self.gripper_urdf, root_node_name="/grip_tcp_frame/grip_base_frame/yumi_gripper")
        viser_urdf.update_cfg([0])
        
        self.gripper_width_slider = self._server.gui.add_slider('Gripper Width', 0.0, 0.025, 0.001, 0.0)

        @self.gripper_width_slider.on_update
        def _(_):
            viser_urdf.update_cfg([float(self.gripper_width_slider.value)])
            
        self.grip_tcp_frame.position = self.graspable_part._grasps.centers[int(self.grasp_dev_slider_handle.value)]
        self.grip_tcp_frame.wxyz = self.graspable_part._grasps.to_se3(along_axis='x').rotation().wxyz[int(self.grasp_dev_slider_handle.value)]
        
    def _load_mesh(self, mesh_path: Path): # TODO: automatically fix all meshes upon init and load those here instead
        mesh = trimesh.load_mesh(mesh_path)
        # Save fixed mesh to file with name _fixed.obj
        
        if len(sorted(mesh_path.parent.glob("*_fixed_normals_centered.obj"))) == 0:
            trimesh.repair.fix_normals(mesh)
            translation = -self.obj2w.squeeze(0)
            mesh.apply_translation(translation)
            mesh.export(mesh_path.parent / f"{mesh_path.stem}_fixed_normals_centered.obj")
        
        self.mesh_handle = self._server.scene.add_mesh_trimesh("/mesh", mesh, visible=True)
    
    def _save_grasp(self):
        # assert self.grip_tcp_frame in list(self.graspable_part._grasps.centers), "Grasp not on a graspable part center"
        grasp_pos = self.grip_tcp_frame.position
        grasp_wxyz = self.grip_tcp_frame.wxyz
        grasp_width = self.gripper_width_slider.value
        grasp_width = float(grasp_width)
        grasp_data = {
            'position': grasp_pos.tolist(),
            'orientation': grasp_wxyz.tolist(),
            'width': grasp_width
        }
        grasp_file = self.sub_part_files[self.group_idx].parent / self.sub_part_files[self.group_idx].stem  / 'grasps.txt'

        import json

        if grasp_file.exists():
            with open(grasp_file, 'r') as f:
                try:
                    existing_grasps = json.load(f)
                except json.JSONDecodeError:
                    existing_grasps = []
                    
            # Check for duplicates
            is_duplicate = False
            for existing_grasp in existing_grasps:
                if (onp.allclose(existing_grasp['position'], grasp_data['position']) and
                    onp.allclose(existing_grasp['orientation'], grasp_data['orientation']) and
                    onp.allclose(existing_grasp['width'], grasp_data['width'])):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                existing_grasps.append(grasp_data)
                with open(grasp_file, 'w') as f:
                    json.dump(existing_grasps, f, indent=2)
                print(f"Grasp saved to {grasp_file}")
            else:
                print("Grasp already exists, skipping save")
        else:
            with open(grasp_file, 'w') as f:
                json.dump([grasp_data], f, indent=2)
            print(f"Created new grasp file at {grasp_file}")
    
    def load_ply_file(self, ply_file_path: Path, center: bool = False) -> SplatFile:
        """Load Gaussians stored in a PLY file."""
        start_time = time.time()

        SH_C0 = 0.28209479177387814

        plydata = PlyData.read(ply_file_path)
        v = plydata["vertex"]
        positions = onp.stack([v["x"], v["y"], v["z"]], axis=-1)
        scales = onp.exp(onp.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=-1))
        wxyzs = onp.stack([v["rot_0"], v["rot_1"], v["rot_2"], v["rot_3"]], axis=1)
        colors = 0.5 + SH_C0 * onp.stack([v["f_dc_0"], v["f_dc_1"], v["f_dc_2"]], axis=1)
        opacities = 1.0 / (1.0 + onp.exp(-v["opacity"][:, None]))

        Rs = vtf.SO3(wxyzs).as_matrix()
        covariances = onp.einsum(
            "nij,njk,nlk->nil", Rs, onp.eye(3)[None, :, :] * scales[:, None, :] ** 2, Rs
        )
        if center:
            self.obj2w = onp.mean(positions, axis=0, keepdims=True)
            positions -= self.obj2w

        num_gaussians = len(v)
        print(
            f"PLY file {ply_file_path} with {num_gaussians=} loaded in {time.time() - start_time} seconds"
        )
        return {
            "centers": positions,
            "rgbs": colors,
            "opacities": opacities,
            "covariances": covariances,
        }
    
    def _save_init_p2o(self):
        assert self.pipeline.cluster_labels is not None, "Cluster labels not found"
            
        labels = self.pipeline.cluster_labels.int().cuda()
            
        self.configure_from_clusters(labels)
        
        with open(self.pipeline.state_dir / "init_p2o.txt", "w") as f:
            f.truncate(0)
            for i in range(len(labels.unique())):
                f.write(f"{self.init_p2o[i].tolist()}\n")
        
    def configure_from_clusters(self, group_labels: torch.Tensor):

        # Get group / cluster label info.
        self.group_labels = group_labels.cuda()
        self.num_groups = int(self.group_labels.max().item() + 1)
        self.group_masks = [(self.group_labels == cid).cuda() for cid in range(self.group_labels.max() + 1)]

        # Store pose of each part, as wxyz_xyz.
        part_deltas = torch.zeros(0, self.num_groups, 7, dtype=torch.float32, device="cuda")
        self.part_deltas = torch.nn.Parameter(part_deltas)

        # Initialize the object pose. Centered at object centroid, and identity rotation.
        self.T_world_objinit = identity_7vec()
        self.T_world_objinit[0, 4:] = self.pipeline.model.gauss_params["means"].detach().clone().mean(dim=0).squeeze()

        # Initialize the part poses to identity. Again, wxyz_xyz.
        # Parts are initialized at the centroid of the part cluster.
        self.init_p2o = identity_7vec().repeat(self.num_groups, 1)
        for i, g in enumerate(self.group_masks):
            gp_centroid = self.pipeline.model.gauss_params["means"].detach().clone()[g].mean(dim=0)
            self.init_p2o[i, 4:] = gp_centroid - self.pipeline.model.gauss_params["means"].detach().clone().mean(dim=0)
        