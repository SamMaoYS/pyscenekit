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

    @abc.abstractmethod
    def load_model(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _predict(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @torch.no_grad()
    def __call__(
        self,
        image: ImageInput,
        resolution: Tuple[int, int] = None,
        resize_to_input: bool = True,
    ):
        self.input = SceneKitImage(image)

        input_image = self.input.image
        self.resolution_input = input_image.shape[:2]
        if resolution is not None:
            self.resolution_pred = resolution

        if self.resolution_pred is not None:
            input_image = self.resize(input_image, self.resolution_pred)
            self.resolution_output = self.resolution_pred

        output_image = self._predict(input_image)
        if resize_to_input:
            self.resolution_output = self.resolution_input

        if self.resolution_output is not None:
            output_image = self.resize(output_image, self.resolution_output)

        return output_image

    # resize image to the given resolution
    def resize(self, image: np.ndarray, resolution: Tuple[int, int]) -> np.ndarray:
        h, w = resolution
        image = cv2.resize(image, (w, h))
        return image

    @abc.abstractmethod
    def to(self, device: str):
        raise NotImplementedError
