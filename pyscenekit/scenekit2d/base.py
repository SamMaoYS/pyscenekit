import abc
from typing import Tuple

import cv2
import torch
import numpy as np

from pyscenekit.scenekit2d.utils import ImageInput
from pyscenekit.scenekit2d.common import SceneKitImage


class BaseImageModel(abc.ABC):
    @abc.abstractmethod
    def __init__(self, model_name: str = None):
        self.model_path = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

        # input and output are SceneKitImage objects
        self.input = None
        self.output = None

        # resolution is a tuple of (height, width)
        self.resolution_input = None
        self.resolution_pred = None
        self.resolution_output = None

        self.resize_mode = cv2.INTER_LINEAR

    @abc.abstractmethod
    def load_model(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _predict(self, image: np.ndarray, **kwargs) -> np.ndarray:
        raise NotImplementedError

    @torch.no_grad()
    def __call__(
        self,
        image: ImageInput,
        resolution: Tuple[int, int] = None,
        resize_to_input: bool = True,
        **kwargs,
    ):
        self.input = SceneKitImage(image)

        input_image = self.input.image
        self.resolution_input = input_image.shape[:2]
        if resolution is not None:
            self.resolution_pred = resolution

        if self.resolution_pred is not None:
            input_image = self.resize(
                input_image, self.resolution_pred, cv2.INTER_LANCZOS4
            )
            self.resolution_output = self.resolution_pred

        output = self._predict(input_image, **kwargs)
        output_depth = output["depth"]
        if resize_to_input:
            self.resolution_output = self.resolution_input

        if self.resolution_output is not None:
            output_depth = self.resize(
                output_depth, self.resolution_output, self.resize_mode
            )

        return output_depth, output

    # resize image to the given resolution
    def resize(
        self,
        image: np.ndarray,
        resolution: Tuple[int, int],
        resize_mode: int = cv2.INTER_LANCZOS4,
    ) -> np.ndarray:
        h, w = resolution
        image = cv2.resize(image, (w, h), interpolation=resize_mode)
        return image

    @abc.abstractmethod
    def to(self, device: str):
        raise NotImplementedError
