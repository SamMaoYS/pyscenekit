import torch
import numpy as np
from transformers import DPTImageProcessor
from transformers import DPTForDepthEstimation

from pyscenekit.scenekit2d.depth.base import BaseDepthEstimation


class MidasDepthEstimation(BaseDepthEstimation):
    def __init__(self, model_path: str = None):
        super().__init__(model_path)
        if self.model_path is None:
            self.model_path = "Intel/dpt-hybrid-midas"

        self.image_processor = None
        self.load_model()

    def load_model(self):
        self.image_processor = DPTImageProcessor.from_pretrained(self.model_path)
        self.model = DPTForDepthEstimation.from_pretrained(self.model_path)

    @torch.no_grad()
    def _predict(self, image: np.ndarray) -> np.ndarray:
        self.model.to(self.device)
        inputs = self.image_processor(images=image, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].to(self.device)
        outputs = self.model(**inputs)
        depth = outputs.predicted_depth
        depth = depth.squeeze().cpu().numpy()
        return depth

    def to(self, device: str):
        self.device = torch.device(device)
        self.model.to(self.device)
