import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from pyscenekit.scenekit2d.depth.base import BaseDepthEstimation


class DepthAnythingV2DepthEstimation(BaseDepthEstimation):
    def __init__(self, model_path: str = None):
        super().__init__(model_path)
        if self.model_path is None:
            self.model_path = "depth-anything/Depth-Anything-V2-Large"

        self.image_processor = None
        self.load_model()

    def load_model(self):
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_path)
        self.model = AutoModelForDepthEstimation.from_pretrained(self.model_path)

    @torch.no_grad()
    def _predict(self, image: np.ndarray) -> np.ndarray:
        self.to(self.device)
        inputs = self.image_processor(images=image, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].to(self.device)
        outputs = self.model(**inputs)
        post_processed_output = self.image_processor.post_process_depth_estimation(
            outputs
        )
        depth = post_processed_output[0]["predicted_depth"]
        depth = depth.detach().cpu().numpy()
        return depth

    def to(self, device: str):
        self.device = torch.device(device)
        self.model.to(self.device)