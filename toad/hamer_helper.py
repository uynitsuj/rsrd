import contextlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator

import imageio.v3 as iio
import numpy as np
import torch
import tyro
from jaxtyping import Float
from torch import Tensor
from hamer.datasets.vitdet_dataset import (
    DEFAULT_MEAN,
    DEFAULT_STD,
    ViTDetDataset,
)
from hamer.utils.renderer import Renderer, cam_crop_to_full

@contextlib.contextmanager
def stopwatch(message: str):
    print("[STOPWATCH]", message)
    start = time.time()
    yield
    print("[STOPWATCH]", message, f"finished in {time.time() - start} seconds!")


@dataclass(frozen=True)
class HamerOutputs:
    """A typed wrapper for outputs from HaMeR."""

    # Comments here are what I got when printing out the shapes of different
    # HaMeR outputs.

    # pred_cam torch.Size([1, 3])
    pred_cam: Float[Tensor, "num_hands 3"]
    # pred_mano_params global_orient torch.Size([1, 1, 3, 3])
    pred_mano_global_orient: Float[Tensor, "num_hands 1 3 3"]
    # pred_mano_params hand_pose torch.Size([1, 15, 3, 3])
    pred_mano_hand_pose: Float[Tensor, "num_hands 15 3 3"]
    # pred_mano_params betas torch.Size([1, 10])
    pred_mano_hand_betas: Float[Tensor, "num_hands 10"]
    # pred_cam_t torch.Size([1, 3])
    pred_cam_t: Float[Tensor, "num_hands 3"]

    # focal length from model is ignored
    # focal_length torch.Size([1, 2])
    # focal_length: Float[Tensor, "num_hands 2"]

    # pred_keypoints_3d torch.Size([1, 21, 3])
    pred_keypoints_3d: Float[Tensor, "num_hands 21 3"]
    # pred_vertices torch.Size([1, 778, 3])
    pred_vertices: Float[Tensor, "num_hands 778 3"]
    # pred_keypoints_2d torch.Size([1, 21, 2])
    pred_keypoints_2d: Float[Tensor, "num_hands 21 3"]

    pred_right: Float[Tensor, "num_hands"]
    """A given hand is a right hand if this value is >0.5."""

    # These aren't technically HaMeR outputs, but putting them here for your convenience.
    mano_faces_right: Tensor
    mano_faces_left: Tensor


@contextlib.contextmanager
def temporary_cwd_context(x: Path) -> Generator[None, None, None]:
    """Temporarily change our working directory."""
    d = os.getcwd()
    os.chdir(x)
    try:
        yield
    finally:
        os.chdir(d)


class HamerHelper:
    """Helper class for running HaMeR. Adapted from HaMeR demo script."""

    def __init__(self) -> None:
        import hamer
        from hamer.models import DEFAULT_CHECKPOINT, load_hamer
        from vitpose_model import ViTPoseModel

        # HaMeR hardcodes a bunch of relative paths...
        # Instead of modifying HaMeR we're going to hack this by creating some symlinks... :)
        hamer_directory = Path(hamer.__file__).parent.parent

        with temporary_cwd_context(hamer_directory):
            # Download and load checkpoints
            # download_models(Path(hamer.__file__).parent.parent /CACHE_DIR_HAMER)
            with stopwatch("Loading HaMeR model..."):
                model, model_cfg = load_hamer(
                    str(Path(hamer.__file__).parent.parent / DEFAULT_CHECKPOINT)
                )

            # Setup HaMeR model
            with stopwatch("Configuring HaMeR model..."):
                device = torch.device("cuda")
                model = model.to(device)
                model.eval()

            # Load detector
            import hamer
            from detectron2.config import LazyConfig
            from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy

            with stopwatch("Creating Detectron2 predictor..."):
                cfg_path = (
                    Path(hamer.__file__).parent
                    / "configs"
                    / "cascade_mask_rcnn_vitdet_h_75ep.py"
                )
                detectron2_cfg = LazyConfig.load(str(cfg_path))
                detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"  # type: ignore
                for i in range(3):
                    detectron2_cfg.model.roi_heads.box_predictors[  # type: ignore
                        i
                    ].test_score_thresh = 0.25
                detector = DefaultPredictor_Lazy(detectron2_cfg)

            # keypoint detector
            with stopwatch("Creating ViT pose model..."):
                cpm = ViTPoseModel(device)

            self.model = model
            self.model_cfg = model_cfg
            self.detector = detector
            self.cpm = cpm
            self.device = device

    def look_for_hands(
        self,
        image: Float[np.ndarray, "height width 3"],
        focal_length: float | None,
        rescale_factor: float = 2.0,
    ):
        """Look for hands.

        Arguments:
            image: Image to look for hands in.
            focal_length: Focal length of camera, used for 3D predictions.
            rescale_factor: Rescale factor for running ViT detector. I think 2 is fine, probably.
        """
        assert image.shape[-1] == 3

        # Detectron expects BGR image.
        det_out = self.detector(image[:, :, ::-1])
        det_instances = det_out["instances"]
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()

        # Detect human keypoints for each person
        vitposes_out = self.cpm.predict_pose(
            image,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        bboxes = []
        is_right = []

        # Use hands based on hand keypoint detections
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes["keypoints"][-42:-21]
            right_hand_keyp = vitposes["keypoints"][-21:]

            lbbox = None
            rbbox = None

            # Rejecting not confident detections
            ldetect = rdetect = False
            keyp = left_hand_keyp
            valid = keyp[:, 2] > 0.5
            if sum(valid) > 3:
                lbbox = [
                    keyp[valid, 0].min(),
                    keyp[valid, 1].min(),
                    keyp[valid, 0].max(),
                    keyp[valid, 1].max(),
                ]
                ldetect = True
            keyp = right_hand_keyp
            valid = keyp[:, 2] > 0.5
            if sum(valid) > 3:
                rbbox = [
                    keyp[valid, 0].min(),
                    keyp[valid, 1].min(),
                    keyp[valid, 0].max(),
                    keyp[valid, 1].max(),
                ]
                rdetect = True

            # suppressing
            if ldetect == True and rdetect == True:
                bboxes_dims = [
                    left_hand_keyp[:, 0].max() - left_hand_keyp[:, 0].min(),
                    left_hand_keyp[:, 1].max() - left_hand_keyp[:, 1].min(),
                    right_hand_keyp[:, 0].max() - right_hand_keyp[:, 0].min(),
                    right_hand_keyp[:, 1].max() - right_hand_keyp[:, 1].min(),
                ]
                norm_side = max(bboxes_dims)
                keyp_dist = (
                    np.sqrt(
                        np.sum(
                            (right_hand_keyp[:, :2] - left_hand_keyp[:, :2]) ** 2,
                            axis=1,
                        )
                    )
                    / norm_side
                )
                if np.mean(keyp_dist) < 0.5:
                    if left_hand_keyp[0, 2] - right_hand_keyp[0, 2] > 0:
                        assert lbbox is not None
                        bboxes.append(lbbox)
                        is_right.append(0)
                    else:
                        assert rbbox is not None
                        bboxes.append(rbbox)
                        is_right.append(1)
                else:
                    assert lbbox is not None
                    assert rbbox is not None
                    bboxes.append(lbbox)
                    is_right.append(0)
                    bboxes.append(rbbox)
                    is_right.append(1)
            elif ldetect == True:
                assert lbbox is not None
                bboxes.append(lbbox)
                is_right.append(0)
            elif rdetect == True:
                assert rbbox is not None
                bboxes.append(rbbox)
                is_right.append(1)

        if len(bboxes) == 0:
            return None,None

        boxes = np.stack(bboxes)
        right = np.stack(is_right)

        

        dataset = ViTDetDataset(
            self.model_cfg,
            # HaMeR expects BGR.
            image[:, :, ::-1],
            boxes,
            right,
            rescale_factor=rescale_factor,
        )

        # ViT detector will give us multiple detections. We want to run HaMeR
        # on each.
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=8, shuffle=False, num_workers=0
        )
        outputs: list[HamerOutputs] = []
        from hamer.utils import recursive_to

        for batch in dataloader:
            batch: Any = recursive_to(batch, self.device)
            with torch.no_grad():
                out = self.model.forward(batch)

            multiplier = 2 * batch["right"] - 1
            pred_cam = out["pred_cam"]
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            multiplier = 2 * batch["right"] - 1

            if focal_length is None:
                scaled_focal_length = (
                    self.model_cfg.EXTRA.FOCAL_LENGTH
                    / self.model_cfg.MODEL.IMAGE_SIZE
                    * img_size.max()
                )
            else:
                scaled_focal_length = focal_length


            pred_cam_t_full = cam_crop_to_full(
                pred_cam, box_center, box_size, img_size, scaled_focal_length
            )
            hamer_out = HamerOutputs(
                    mano_faces_left=torch.from_numpy(
                        self.model.mano.faces[:, [0, 2, 1]].astype(np.int64)
                    ).to(device=self.device),
                    mano_faces_right=torch.from_numpy(
                        self.model.mano.faces.astype(np.int64)
                    ).to(device=self.device),
                    pred_cam=out["pred_cam"],
                    pred_mano_global_orient=out["pred_mano_params"]["global_orient"],
                    pred_mano_hand_pose=out["pred_mano_params"]["hand_pose"],
                    pred_mano_hand_betas=out["pred_mano_params"]["betas"],
                    pred_cam_t=pred_cam_t_full,
                    pred_keypoints_3d=out["pred_keypoints_3d"],
                    pred_vertices=out["pred_vertices"],
                    pred_keypoints_2d=out["pred_keypoints_2d"],
                    pred_right=batch["right"],
                )
            
            outputs.append(
                hamer_out
            )

        assert len(outputs) > 0
        stacked_outputs = HamerOutputs(
            **{
                field_name: torch.cat([getattr(x, field_name) for x in outputs], dim=0)
                for field_name in vars(outputs[0]).keys()
            },
        )
        # begin new brent stuff
        verts = stacked_outputs.pred_vertices.numpy(force=True)
        keypoints_3d = stacked_outputs.pred_keypoints_3d.numpy(force=True)
        pred_cam_t = stacked_outputs.pred_cam_t.numpy(force=True)
        mano_hand_pose = stacked_outputs.pred_mano_hand_pose.numpy(force=True)
        mano_hand_betas = stacked_outputs.pred_mano_hand_betas.numpy(force=True)
        R_camera_hand = stacked_outputs.pred_mano_global_orient.squeeze(dim=1).numpy(
            force=True
        )

        is_right = (stacked_outputs.pred_right > 0.5).numpy(force=True)
        is_left = ~is_right

        if np.sum(is_right) == 0:
            detections_right_wrt_cam = None
        else:
            detections_right_wrt_cam = {
                "verts": verts[is_right] + pred_cam_t[is_right, None, :],
                "keypoints_3d": keypoints_3d[is_right] + pred_cam_t[is_right, None, :],
                "mano_hand_pose": mano_hand_pose[is_right],
                "mano_hand_betas": mano_hand_betas[is_right],
                "mano_hand_global_orient": R_camera_hand[is_right],
                "faces": self.model.mano.faces.copy(),
            }

        if np.sum(is_left) == 0:
            detections_left_wrt_cam = None
        else:

            def flip_rotmats(rotmats: np.ndarray) -> np.ndarray:
                assert rotmats.shape[-2:] == (3, 3)
                from viser import transforms
                logspace = transforms.SO3.from_matrix(rotmats).log()
                logspace[..., 1] *= -1
                logspace[..., 2] *= -1
                return transforms.SO3.exp(logspace).as_matrix()

            detections_left_wrt_cam = {
                "verts": verts[is_left] * np.array([-1, 1, 1])
                + pred_cam_t[is_left, None, :],
                "keypoints_3d": keypoints_3d[is_left] * np.array([-1, 1, 1])
                + pred_cam_t[is_left, None, :],
                "mano_hand_pose": flip_rotmats(mano_hand_pose[is_left]),
                "mano_hand_betas": mano_hand_betas[is_left],
                "mano_hand_global_orient": flip_rotmats(R_camera_hand[is_left]),
                "faces": self.model.mano.faces[:, [0, 2, 1]].copy(),
            }
        # end new brent stuff
        return detections_left_wrt_cam,detections_right_wrt_cam
    
    def render_detection(self, output_dict, hand_id, w, h, focal_length):
        import pyrender
        import trimesh
        render_res = (h,w)
        renderer = pyrender.OffscreenRenderer(viewport_width=render_res[1],
                                              viewport_height=render_res[0],
                                              point_size=1.0)

        vertices = output_dict['verts'][hand_id]
        faces = output_dict['faces']

        mesh = trimesh.Trimesh(vertices.copy(), faces.copy())
        mesh = pyrender.Mesh.from_trimesh(mesh)

        scene = pyrender.Scene(bg_color=[*(1.0,1.0,1.0), 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, 'mesh')

        camera_center = [render_res[1] / 2., render_res[0] / 2.]
        camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length,
                                           cx=camera_center[0], cy=camera_center[1], zfar=1e12,znear=.001)

        # Create camera node and add it to pyRender scene
        camera_pose = np.eye(4)
        camera_pose[1:3,:] *= -1 #flip the y and z axes to match opengl
        camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        scene.add_node(camera_node)

        color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        mask = color[...,-1] > 0
        return color[...,:3],rend_depth, mask

def main(
    input_images: list[Path],
    render_output_dir: Path = Path("./hamer_test_script_outputs"),
) -> None:
    # Set up HaMeR.
    # This should magically work as long as the HaMeR repo is set up.
    hamer_helper = HamerHelper()

    print("#" * 80)
    print("#" * 80)
    print("#" * 80)
    print(
        "Done setting up HaMeR! There were probably lots of errors, including a scary gigantic one about state dict stuff, but it's probably fine!"
    )
    print("#" * 80)
    print("#" * 80)
    print("#" * 80)

    for input_image in input_images:
        # Read an image.
        image = iio.imread(input_image)

        # RGB => RGBA.
        if image.shape[-1] == 4:
            image = image / 255.0
            image = image[:, :, :3] * image[:, :, 3:4] + 1.0 * (1.0 - image[:, :, 3:4])
            image = (image * 255).astype(np.uint8)

        # Run HaMeR.
        det_left,det_right = hamer_helper.look_for_hands(
            image=image,
            focal_length=(1137.0/1280) * image.shape[1]
        )

        if det_left is not None:
            rgb, depth,mask = hamer_helper.render_detection(det_left,0,1280,720,1137.0)
        if det_right is not None:
            rgb, depth,mask = hamer_helper.render_detection(det_right,0,1280,720,1137.0)
        import matplotlib.pyplot as plt
        fig,axs = plt.subplots(1,4)
        axs[0].imshow(image)
        axs[1].imshow(rgb)
        axs[2].imshow(depth)
        axs[3].imshow(mask)
        plt.show()


if __name__ == "__main__":
    tyro.cli(main)
