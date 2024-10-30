from typing import Tuple, Dict

import cv2
import torch
import numpy as np

from pyscenekit.utils.common import log
from pyscenekit.scenekit2d.utils import ImageInput
from pyscenekit.scenekit2d.common import SceneKitImage
from pyscenekit.scenekit2d.camera.modules.vp_estimation_prior_gravity import (
    VPEstimationPriorGravity,
)


class VPEstimationPriorGravityModel:
    """
    Vanishing Point Estimation in Uncalibrated Images with Prior Gravity Direction

    Authors: Rémi Pautrat, Shaohui Liu, Petr Hruby, Marc Pollefeys, Daniel Barath

    https://github.com/cvg/VP-Estimation-with-Prior-Gravity

    @InProceedings{Pautrat_2023_UncalibratedVP,
        author = {Pautrat, Rémi and Liu, Shaohui and Hruby, Petr and Pollefeys, Marc and Barath, Daniel},
        title = {Vanishing Point Estimation in Uncalibrated Images with Prior Gravity Direction},
        booktitle = {International Conference on Computer Vision (ICCV)},
        year = {2023},
    }
    """

    def __init__(self, model_path: str = None):
        self.model = None
        self.load_model()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input = None
        self.resize_mode = cv2.INTER_LANCZOS4
        self.gravity = np.array([0.0, 1.0, 0.0])

    def load_model(self):
        self.model = VPEstimationPriorGravity()

    def __call__(
        self,
        image: ImageInput,
        resolution: Tuple[int, int] = None,
        gravity: np.ndarray = np.array([0.0, 1.0, 0.0]),
    ):
        input_image = SceneKitImage(image).to_gray()
        if resolution is not None:
            input_image = cv2.resize(
                input_image, resolution, interpolation=self.resize_mode
            )
        self.model.set_gravity(gravity)
        f, vp = self.model.estimate_vps(input_image)
        return f, vp

    def to(self, device: str):
        pass


class VP2Rotation:
    def __init__(self, vps: np.ndarray, image_size: Tuple[int, int]):
        self.vps = vps
        self.image_width = image_size[0]
        self.image_height = image_size[1]

    def set_vps(self, vps: np.ndarray):
        self.vps = vps

    def set_image_size(self, image_size: Tuple[int, int]):
        self.image_width = image_size[0]
        self.image_height = image_size[1]

    def __call__(self, focal_length: float):
        K = np.array(
            [
                [focal_length, 0, self.image_width / 2],
                [0, focal_length, self.image_height / 2],
                [0, 0, 1],
            ]
        )
        # TODO: Implement
        pass
