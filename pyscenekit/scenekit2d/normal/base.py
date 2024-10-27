import cv2
import numpy as np

from pyscenekit.scenekit2d.base import BaseImageModel


class BaseNormalEstimation(BaseImageModel):
    def __init__(self, model_name: str = None):
        super().__init__(model_name)

    def to_rgb(self, normal: np.ndarray) -> np.ndarray:
        if normal.dtype != np.uint8:
            normal = (normal + 1.0) / 2.0 * 255
        return normal.astype(np.uint8)
