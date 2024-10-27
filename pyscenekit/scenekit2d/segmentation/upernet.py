import torch
import numpy as np
from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

from pyscenekit.scenekit2d.segmentation.base import BaseImageSegmentation


class UperNetSemanticSegmentation(BaseImageSegmentation):
    def __init__(self, model_path: str = None):
        super().__init__(model_path)
        if self.model_path is None:
            self.model_path = "openmmlab/upernet-convnext-large"

        self.image_processor = None
        self.load_model()

    def load_model(self):
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_path)
        self.model = UperNetForSemanticSegmentation.from_pretrained(self.model_path)

    @torch.no_grad()
    def _predict(self, image: np.ndarray) -> np.ndarray:
        self.to(self.device)
        pixel_values = self.image_processor(
            images=image, return_tensors="pt"
        ).pixel_values
        pixel_values = pixel_values.to(self.device)
        outputs = self.model(pixel_values)
        semantic_image = self.image_processor.post_process_semantic_segmentation(
            outputs
        )[0]
        semantic_image = semantic_image.detach().cpu().numpy().astype(np.uint16)
        return semantic_image

    def to(self, device: str):
        self.device = torch.device(device)
        self.model.to(self.device)
