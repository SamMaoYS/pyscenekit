import cv2
import numpy as np

from pyscenekit.scenekit2d.base import BaseImageModel


class BaseDepthEstimation(BaseImageModel):
    def __init__(self, model_name: str = None):
        super().__init__(model_name)

    # min-max normalization of depth map
    @staticmethod
    def normalize(depth: np.ndarray) -> np.ndarray:
        # depth input shape: [H, W]
        depth = depth.astype(np.float32)
        depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
        return depth

    @staticmethod
    def colormap(depth: np.ndarray, cmap: str = "viridis") -> np.ndarray:
        if depth.dtype != np.uint8:
            depth = (depth * 255).astype(np.uint8)

        return cv2.applyColorMap(depth, getattr(cv2, f"COLORMAP_{cmap.upper()}"))
