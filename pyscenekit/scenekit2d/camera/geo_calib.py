from typing import Tuple, Dict

import cv2
import torch
import huggingface_hub

from pyscenekit.utils.common import log
from pyscenekit.scenekit2d.utils import ImageInput
from pyscenekit.scenekit2d.common import SceneKitImage
from pyscenekit.scenekit2d.camera.modules.geo_calib import GeoCalib
from pyscenekit.scenekit2d.camera.modules.geo_calib.utils import (
    rad2deg,
    deg2rad,
    print_calibration,
)


class GeoCalibModel:
    """
    GeoCalib: Single-image Calibration with Geometric Optimization

    Authors: Alexander Veicht, Paul-Edouard Sarlin, Philipp Lindenberger, Marc Pollefeys

    https://github.com/cvg/GeoCalib

    @inproceedings{veicht2024geocalib,
        author    = {Alexander Veicht and
                    Paul-Edouard Sarlin and
                    Philipp Lindenberger and
                    Marc Pollefeys},
        title     = {{GeoCalib: Single-image Calibration with Geometric Optimization}},
        booktitle = {ECCV},
        year      = {2024}
    }
    """

    def __init__(self, model_path: str = None):
        self.model_path = model_path
        if self.model_path is None:
            self.model_path = huggingface_hub.hf_hub_download(
                "ysmao/pyscenekit",
                subfolder="geocalib",
                filename="geocalib-pinhole.tar",
            )
        self.model = None
        self.load_model()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input = None
        self.resize_mode = cv2.INTER_LANCZOS4
        self.focal_length = None

    def load_model(self):
        self.model = GeoCalib(self.model_path)

    def __call__(
        self,
        image: ImageInput,
        resolution: Tuple[int, int] = None,
        focal_length: float = None,
    ):
        def img_to_tensor(img: ImageInput, resolution: Tuple[int, int]):
            img_t = SceneKitImage(img)

            if resolution is not None:
                img_t.resize(resolution, self.resize_mode)

            return img_t.to_tensor()

        if isinstance(image, list):
            self.input = []
            for img in image:
                input = img_to_tensor(img, resolution)
                self.input.append(input)
            self.input = torch.stack(self.input).to(self.device)

            if focal_length is not None:
                log.warning(
                    "focal_length is not supported for multiple images, "
                    "it will be ignored"
                )

            result = self.model.calibrate(self.input, shared_intrinsics=True)
        else:
            priors = None
            if focal_length is not None:
                priors = {"focal": torch.tensor(focal_length)}
            self.input = img_to_tensor(image, resolution).to(self.device)
            result = self.model.calibrate(self.input, priors=priors)
        return result

    def to(self, device: str):
        self.model.to(device)

    @staticmethod
    def print_calibration(results: Dict[str, torch.Tensor]):
        print_calibration(results)
