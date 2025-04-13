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
from rsrd.extras.grasp_helper import GraspDevMRO, GraspDevArticulated

from dig.dig_pipeline import ObjectMode
torch.set_float32_matmul_precision("high")

#Temp
import jax.numpy as jnp
from jaxtyping import Float
import jax_dataclasses as jdc
import jaxlie
from torchvision.transforms.functional import resize

from PIL import Image


def main(
    dig_config: Path, 
    ):
    
    assert dig_config is not None, "Must provide a dig config path."
    server = viser.ViserServer()
    
    # Load DIG model, create viewer.
    train_config, pipeline, _, _ = eval_setup(dig_config)
    del pipeline.garfield_pipeline
    pipeline.eval()
    pipeline.load_state()
    dataset_scale = pipeline.datamanager.train_dataset._dataparser_outputs.dataparser_scale
    ns_output_path = dig_config.parent.parent.parent

    if pipeline.object_mode == ObjectMode.RIGID_OBJECTS:
        GraspDevMRO(
            server,
            pipeline,
            ns_output_path,
            1/dataset_scale,
            )
    elif pipeline.object_mode == ObjectMode.ARTICULATED:
        GraspDevArticulated(
            server,
            pipeline,
            ns_output_path,
            1/dataset_scale,
            )
    while True:
        time.sleep(0.2)
    
if __name__ == "__main__":
    tyro.cli(main)
