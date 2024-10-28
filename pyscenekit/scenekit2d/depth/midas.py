import torch
import numpy as np
from transformers import DPTImageProcessor
from transformers import DPTForDepthEstimation

from pyscenekit.scenekit2d.depth.base import BaseDepthEstimation


class MidasDepthEstimation(BaseDepthEstimation):
    """
    Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-shot Cross-dataset Transfer

    Authors: René Ranftl, Katrin Lasinger, David Hafner, Konrad Schindler, Vladlen Koltun.

    https://github.com/isl-org/MiDaS

    @ARTICLE {Ranftl2022,
        author  = "Ren\'{e} Ranftl and Katrin Lasinger and David Hafner and Konrad Schindler and Vladlen Koltun",
        title   = "Towards Robust Monocular Depth Estimation: Mixing Datasets for Zero-Shot Cross-Dataset Transfer",
        journal = "IEEE Transactions on Pattern Analysis and Machine Intelligence",
        year    = "2022",
        volume  = "44",
        number  = "3"
    }
    @article{Ranftl2021,
        author    = {Ren\'{e} Ranftl and Alexey Bochkovskiy and Vladlen Koltun},
        title     = {Vision Transformers for Dense Prediction},
        journal   = {ICCV},
        year      = {2021},
    }
    """

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
        self.to(self.device)
        inputs = self.image_processor(images=image, return_tensors="pt")
        inputs["pixel_values"] = inputs["pixel_values"].to(self.device)
        outputs = self.model(**inputs)
        depth = outputs.predicted_depth
        depth = depth.squeeze().cpu().numpy()
        return depth

    def to(self, device: str):
        self.device = torch.device(device)
        self.model.to(self.device)
