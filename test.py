from __future__ import annotations

import time
from pathlib import Path
from typing import List, Literal, Dict, Any, Union, Optional

import torch
import numpy as np
import tyro
import viser
from viser.extras import ViserUrdf

# cuRobo
from curobo.types.math import Pose
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig, CudaRobotModelState, CudaRobotGeneratorConfig
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.types.state import JointState
from curobo.util_file import get_assets_path, get_robot_path, join_path, load_yaml
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig, IKResult
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

class YumiCurobo:
    robot_type: Literal["yumi"]
    _robot_config: CudaRobotModelConfig
    _robot_kin_model: CudaRobotModel
    _ik_solver: IKSolver
    _device: torch.device
    _curr_cfg: np.ndarray
    _viser_urdf: ViserUrdf
    _world_config: Dict[str, Any]

    _base_dir: Path
    """All paths are provided relative to the root of the repository."""

    def __init__(
        self,
        target: Union[viser.ViserServer, viser.ClientHandle],
        world_config: Dict[str, Any],
    ):
        # check if cuda is available
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available -- curobo requires CUDA.")

        self._world_config = world_config
        self._base_dir = Path(__file__).parent
        self._setup(target)

    def _setup(self, target):
        # convenience function to store tensor type and device
        tensor_args = TensorDeviceType()
        self._device = tensor_args.device

        # this example loads urdf from a configuration file, you can also load from path directly
        # load a urdf, the base frame and the end-effector frame:
        config_file = load_yaml(join_path(self._base_dir, f"data/yumi.yml"))

        urdf_path = config_file["robot_cfg"]["kinematics"]["urdf_path"]
        base_link = config_file["robot_cfg"]["kinematics"]["base_link"]
        ee_link = config_file["robot_cfg"]["kinematics"]["ee_link"]
        link_names = config_file["robot_cfg"]["kinematics"]["link_names"]

        config = CudaRobotGeneratorConfig(base_link, ee_link, tensor_args, link_names=link_names, urdf_path=urdf_path, external_robot_configs_path=self._base_dir, external_asset_path=self._base_dir)
        cuda_robot_model_config = CudaRobotModelConfig.from_config(config)

        robot_cfg = RobotConfig(
            cuda_robot_model_config,
            tensor_args=tensor_args,
        )

        kin_model = CudaRobotModel(robot_cfg.kinematics)
        self._robot_config = robot_cfg  # type: ignore
        self._robot_kin_model = kin_model

        ik_config = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            None,
            rotation_threshold=0.05,
            position_threshold=0.002,
            num_seeds=1000,
            self_collision_check=False,
            self_collision_opt=False,
            tensor_args=tensor_args,
            use_cuda_graph=True,
        )
        self._ik_solver = IKSolver(ik_config)

        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            self._world_config,
            interpolation_dt=0.01,
        )
        self._motion_gen = MotionGen(motion_gen_config)
        self._motion_gen.warmup()

        self._viser_urdf = ViserUrdf(target, Path(urdf_path))

    @property
    def joint_pos(self):
        return self._curr_cfg

    @joint_pos.setter
    def joint_pos(self, joint_pos: np.ndarray):
        assert joint_pos.shape == (self._robot_kin_model.get_dof(),)  # gives you 14 things.
        _joint_pos = np.concatenate(
            (joint_pos[7:], joint_pos[:7], [0.01, 0.01])
        )
        self._viser_urdf.update_cfg(_joint_pos)
        self._curr_cfg = joint_pos

    def fk(self, q: torch.Tensor) -> CudaRobotModelState:
        if len(q.shape) == 1:
            q = q.unsqueeze(0)

        assert len(q.shape) == 2 and q.shape[-1] == self._robot_kin_model.get_dof()
        q = q.to(self._device)
        state = self._robot_kin_model.get_state(q)
        return state

    def ik(
            self,
            goal_l_pos: torch.Tensor,
            goal_l_wxyz: torch.Tensor,
            goal_r_pos: torch.Tensor,
            goal_r_wxyz: torch.Tensor,
        ) -> IKResult:
        # Takes around ~0.01 seconds
        assert goal_l_pos.shape == (3,) and goal_l_wxyz.shape == (4,) and goal_r_pos.shape == (3,) and goal_r_wxyz.shape == (4,)

        goal_l_pos = goal_l_pos.to(self._device).float()
        goal_l_wxyz = goal_l_wxyz.to(self._device).float()
        goal_r_pos = goal_r_pos.to(self._device).float()
        goal_r_wxyz = goal_r_wxyz.to(self._device).float()
        goal_l = Pose(goal_l_pos, goal_l_wxyz)
        goal_r = Pose(goal_r_pos, goal_r_wxyz)

        result = self._ik_solver.solve_batch(goal_l, link_poses={"gripper_l_base": goal_l, "gripper_r_base": goal_r})
        return result

    def motiongen(
        self,
        start_l_pos: torch.Tensor,
        start_l_wxyz: torch.Tensor,
        goal_l_pos: torch.Tensor,
        goal_l_wxyz: torch.Tensor,
        start_r_pos: torch.Tensor,
        start_r_wxyz: torch.Tensor,
        goal_r_pos: torch.Tensor,
        goal_r_wxyz: torch.Tensor,
        start_state: Optional[JointState] = None,
    ) -> JointState:
        # Takes around ~0.1 seconds
        assert start_l_pos.shape == (3,) and start_l_wxyz.shape == (4,)
        assert goal_l_pos.shape == (3,) and goal_l_wxyz.shape == (4,)
        assert start_r_pos.shape == (3,) and start_r_wxyz.shape == (4,)
        assert goal_r_pos.shape == (3,) and goal_r_wxyz.shape == (4,)

        goal_l_pos = goal_l_pos.to(self._device).float()
        goal_l_wxyz = goal_l_wxyz.to(self._device).float()
        goal_l_pose = Pose(goal_l_pos, goal_l_wxyz)

        goal_r_pos = goal_r_pos.to(self._device).float()
        goal_r_wxyz = goal_r_wxyz.to(self._device).float()
        goal_r_pose = Pose(goal_r_pos, goal_r_wxyz)

        # get the current joint locations
        if start_state is None:
            start_result = self.ik(start_l_pos, start_l_wxyz, start_r_pos, start_r_wxyz)
            start_joints = start_result.js_solution.position[0]
            assert type(start_joints) == torch.Tensor and len(start_joints.shape) == 2
            start_state = JointState.from_position(start_joints.cuda())

        result = self._motion_gen.plan_single(
            start_state,
            goal_l_pose,
            MotionGenPlanConfig(max_attempts=1),
            link_poses={"gripper_l_base": goal_l_pose, "gripper_r_base": goal_r_pose},
        )
        return result.get_interpolated_plan()


server = viser.ViserServer()
urdf = YumiCurobo(
    server,
    world_config = {
        "cuboid": {
            "table": {
                "dims": [1.0, 1.0, 0.2],  # x, y, z
                "pose": [0.0, 0.0, -0.1, 1, 0, 0, 0.0],  # x, y, z, qw, qx, qy, qz
            },
        }
    }
)  # ... can take a while to load...

drag_l_handle = server.add_transform_controls(
    name="drag_l_handle",
    scale=0.1,
    position=(0.4, 0.2, 0.5),
    wxyz=(0, 1, 0, 0)
)
drag_r_handle = server.add_transform_controls(
    name="drag_r_handle",
    scale=0.1,
    position=(0.4, -0.2, 0.5),
    wxyz=(0, 1, 0, 0)
)

pos_l, quat_l = drag_l_handle.position, drag_l_handle.wxyz
pos_r, quat_r = drag_r_handle.position, drag_r_handle.wxyz
ik_result = urdf.ik(
    torch.tensor(pos_l), torch.tensor(quat_l),
    torch.tensor(pos_r), torch.tensor(quat_r),
) # .position.cpu().numpy()  # type: ignore
urdf.joint_pos = ik_result.js_solution.position.squeeze().cpu().numpy()

waypoint_button = server.add_gui_button("Add waypoint")
drag_button = server.add_gui_button("Match drag")
drag_slider = server.add_gui_slider("Time", 0, 1, 0.01, 0, disabled=True)

traj = None
waypoint_queue = [[], []]
@waypoint_button.on_click
def _(_):
    waypoint_queue[0].append([torch.Tensor(drag_l_handle.position).flatten(), torch.Tensor(drag_l_handle.wxyz).flatten()])
    waypoint_queue[1].append([torch.Tensor(drag_r_handle.position).flatten(), torch.Tensor(drag_r_handle.wxyz).flatten()])

@drag_button.on_click
def _(_):
    global traj
    drag_slider.disabled = True
    drag_button.disabled = True

    if len(waypoint_queue[0]) == 0:
        waypoint_queue[0].append([torch.Tensor(drag_l_handle.position).flatten(), torch.Tensor(drag_l_handle.wxyz).flatten()])
        waypoint_queue[1].append([torch.Tensor(drag_r_handle.position).flatten(), torch.Tensor(drag_r_handle.wxyz).flatten()])

    pose_r = urdf.fk(torch.tensor(urdf.joint_pos)).link_poses["gripper_r_base"]
    pose_l = urdf.fk(torch.tensor(urdf.joint_pos)).link_poses["gripper_l_base"]

    waypoint_queue[0].insert(0, [torch.Tensor(pose_l.position).flatten(), torch.Tensor(pose_l.quaternion.flatten())])
    waypoint_queue[1].insert(0, [torch.Tensor(pose_r.position).flatten(), torch.Tensor(pose_r.quaternion.flatten())])

    start = time.time()
    traj_pieces = []
    prev_start_state = None
    for i in range(len(waypoint_queue[0])-1):
        prev_start_state = urdf.motiongen(
            waypoint_queue[0][i][0], waypoint_queue[0][i][1],
            waypoint_queue[0][i+1][0], waypoint_queue[0][i+1][1],
            waypoint_queue[1][i][0], waypoint_queue[1][i][1],
            waypoint_queue[1][i+1][0], waypoint_queue[1][i+1][1],
            start_state=None if prev_start_state is None else prev_start_state[-1:],
        )
        try:
            traj_pieces.append(prev_start_state.position.cpu().numpy())
        except:
            import pdb; pdb.set_trace()
    traj = np.concatenate(traj_pieces)
    # traj = urdf.motiongen(
    #     pose_l.position.flatten(), pose_l.quaternion.flatten(),
    #     torch.Tensor(drag_l_handle.position), torch.Tensor(drag_l_handle.wxyz),
    #     pose_r.position.flatten(), pose_r.quaternion.flatten(),
    #     torch.Tensor(drag_r_handle.position), torch.Tensor(drag_r_handle.wxyz)
    # ).position.cpu().numpy()  # type: ignore
    print("MotionGen took", time.time() - start, "seconds")
    waypoint_queue[0] = []
    waypoint_queue[1] = []

    urdf.joint_pos = traj[0]
    drag_slider.value = 0
    drag_button.disabled = False
    drag_slider.disabled = False

@drag_slider.on_update
def _(_):
    assert traj is not None
    idx = int(drag_slider.value * (len(traj)-1))
    urdf.joint_pos = traj[idx]

# pos_l, quat_l = drag_l_handle.position, drag_l_handle.wxyz
# pos_r, quat_r = drag_r_handle.position, drag_r_handle.wxyz
# urdf.motiongen(
#     torch.tensor(pos_l), torch.tensor(quat_l),
#     torch.tensor(pos_l) + torch.tensor([0, 0.1, 0]), torch.tensor(quat_l),
#     torch.tensor(pos_r), torch.tensor(quat_r),
#     torch.tensor(pos_r) + torch.tensor([0, 0.1, 0]), torch.tensor(quat_r),
# )

# def update_joints():
#     start = time.time()
#     pos_l, quat_l = drag_l_handle.position, drag_l_handle.wxyz
#     pos_r, quat_r = drag_r_handle.position, drag_r_handle.wxyz
#     ik_result = urdf.ik(
#         torch.tensor(pos_l), torch.tensor(quat_l),
#         torch.tensor(pos_r), torch.tensor(quat_r),
#     ) # .position.cpu().numpy()  # type: ignore
#     urdf.joint_pos = ik_result.js_solution.position.squeeze().cpu().numpy()
#     print("ik took", time.time() - start, "seconds")
# @drag_r_handle.on_update
# def _(_):
#     update_joints()
# @drag_l_handle.on_update
# def _(_):
#     update_joints()
# update_joints()

# drag_button = server.add_gui_button("Match drag")
# @drag_button.on_click
# def _(_):
#     global traj
#     update_joints()

while True:
    time.sleep(1)
