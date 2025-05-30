from copy import deepcopy
import torch
from typing import Union, cast, Optional
import numpy as np
from PIL import Image
from transformers import (
    AutoImageProcessor,
    Mask2FormerForUniversalSegmentation,
    AutoModelForDepthEstimation,
)
from torchvision.transforms.functional import resize

try:
    from hamer_helper import HamerHelper, HandOutputsWrtCamera
    hamer_not_installed = False
except ImportError:
    HamerHelper, HandOutputsWrtCamera = None, None
    hamer_not_installed = True

from rsrd.util.common import Future


class Hand2DDetector:
    hand_processor = Future(lambda: AutoImageProcessor.from_pretrained(
        "facebook/mask2former-swin-base-coco-panoptic")
    )
    hand_model = Future(lambda: Mask2FormerForUniversalSegmentation.from_pretrained(
        "facebook/mask2former-swin-base-coco-panoptic"
    ).to("cuda"))

    @classmethod
    def get_hand_mask(cls, img: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        assert img.shape[2] == 3
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        image = Image.fromarray(img)

        # prepare image for the model
        inputs = cls.hand_processor.retrieve()(images=image, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].cuda()

        with torch.no_grad():
            outputs = cls.hand_model.retrieve()(**inputs)

        # Perform post-processing to get panoptic segmentation map
        seg_ids = cls.hand_processor.retrieve().post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]
        hand_mask = (seg_ids == cls.hand_model.retrieve().config.label2id["person"]).float()
        return hand_mask


class Hand3DDetector:
    hamer_helper = Future(lambda: HamerHelper())

    @classmethod
    def detect_hands(
        cls, image: torch.Tensor, focal: float
    ) -> tuple[Optional[HandOutputsWrtCamera], Optional[HandOutputsWrtCamera]]:
        """
        Detects hands in the image and aligns them with the ground truth depth image.
        """
        if hamer_not_installed:
            raise ImportError("Hamer is not installed. Please install it or set `--no-save-hand` in the command line.")
        left, right = cast(
            HamerHelper,
            cls.hamer_helper.retrieve()
        ).look_for_hands(
            (image * 255).cpu().numpy().astype(np.uint8),
            focal_length=focal
        )
        return left, right

    @classmethod
    def get_aligned_hands_3d(
        cls, 
        hand_outputs: Optional[HandOutputsWrtCamera],  
        monodepth: torch.Tensor, 
        object_mask: torch.Tensor, 
        rendered_scaled_depth: torch.Tensor, 
        focal_length: float
    ) -> Optional[HandOutputsWrtCamera]:
        if hand_outputs is None:
            return None

        num_hands = hand_outputs["verts"].shape[0]
        if num_hands == 0:
            return hand_outputs
        
        for i in range(num_hands):
            rgb, hand_depth, hand_mask = cast(
                        HamerHelper, cls.hamer_helper.retrieve()
                    ).render_detection(
                        hand_outputs,
                        i,
                        monodepth.shape[0],
                        monodepth.shape[1],
                        focal_length,
                    )
            hand_depth = torch.from_numpy(hand_depth).cuda().float()
            hand_mask = torch.from_numpy(hand_mask).cuda()
            masked_hand_depth = hand_depth[hand_mask]#this line needs to be before the resize
            hand_mask = resize(
                    hand_mask.unsqueeze(0),
                    (object_mask.shape[0], object_mask.shape[1]),
                    antialias = True,
                ).permute(1, 2, 0).bool()
            object_mask &= ~hand_mask
            obj_gauss_depth = rendered_scaled_depth[object_mask]
            obj_global_depth = monodepth[object_mask]
            
            # calculate affine that aligns the two
            scale = obj_gauss_depth.std() / obj_global_depth.std()
            scaled_global = (monodepth - obj_global_depth.mean()) * scale + obj_gauss_depth.mean()

            scaled_hand = scaled_global[hand_mask]
            hand_offset = (masked_hand_depth.mean() - scaled_hand.mean()).item()
            hand_outputs['verts'][i][:, 2] -= hand_offset
            hand_outputs['keypoints_3d'][i][:, 2] -= hand_offset
        return hand_outputs

class MonoDepthEstimator:
    image_processor = Future(lambda: AutoImageProcessor.from_pretrained(
        "depth-anything/Depth-Anything-V2-Base-hf"
    ))
    model = Future(lambda: AutoModelForDepthEstimation.from_pretrained(
        "depth-anything/Depth-Anything-V2-Base-hf"
    ).to("cuda"))

    @classmethod
    def get_depth(cls, img: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        assert img.shape[2] == 3
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        image = Image.fromarray(img)

        # prepare image for the model
        inputs = cls.image_processor.retrieve()(images=image, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].cuda()

        with torch.no_grad():
            outputs = cls.model.retrieve()(**inputs)
            predicted_depth = outputs.predicted_depth

        # interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        return prediction.squeeze()
