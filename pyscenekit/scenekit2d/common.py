from typing import Tuple

import cv2
import torch
import PIL.Image
import numpy as np

from pyscenekit.scenekit2d.utils import ImageInput


class SceneKitImage:
    def __init__(self, image: ImageInput, keep_original=False):
        self.image = None
        self._input_image = image

        self.to_numpy()  # image will stored as numpy array
        if not keep_original:
            self._input_image = None

    def to_numpy(self, mean_shift=None):
        if isinstance(self._input_image, str):
            self.image = cv2.imread(self._input_image)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        elif isinstance(self._input_image, PIL.Image.Image):
            self.image = np.array(self._input_image)
        elif isinstance(self._input_image, np.ndarray):
            self.image = self._input_image
        elif isinstance(self._input_image, torch.Tensor):
            # tensor shape: [C, H, W]
            self.image = self._input_image.cpu().permute(1, 2, 0).numpy()
            if self.image.shape[2] == 1:
                self.image = self.image[:, :, 0]
            if mean_shift is not None:
                self.image = self.image + mean_shift
        else:
            raise ValueError(f"Unsupported image type: {type(self._input_image)}")

        return self.image

    @staticmethod
    def numpy_to_tensor(image):
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        return torch.from_numpy(image).permute(2, 0, 1)

    def to_tensor(self, mean_shift=None):
        image = self.image  # image in numpy array format
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        if mean_shift is not None:
            assert isinstance(mean_shift, float), "mean_shift must be a float"
            self._mean_shift = mean_shift
            image = image + mean_shift
        return torch.from_numpy(image).permute(2, 0, 1)

    def to_pil(self):
        return PIL.Image.fromarray(self.image)

    def resize(self, resolution: Tuple[int, int], mode: int):
        return cv2.resize(self.image, resolution, interpolation=mode)
