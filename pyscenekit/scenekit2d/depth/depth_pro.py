import torch
import numpy as np
import huggingface_hub

from pyscenekit.scenekit2d.depth.base import BaseDepthEstimation
from pyscenekit.scenekit2d.depth.modules.depth_pro import create_model_and_transforms
from pyscenekit.scenekit2d.depth.modules.depth_pro.depth_pro import (
    DEFAULT_MONODEPTH_CONFIG_DICT,
)


class DepthProDepthEstimation(BaseDepthEstimation):
    """
    Depth Pro: Sharp Monocular Metric Depth in Less Than a Second

    Authors: Aleksei Bochkovskii, Amaël Delaunoy, Hugo Germain, Marcel Santos, Yichao Zhou, Stephan R. Richter, and Vladlen Koltun.

    https://github.com/apple/ml-depth-pro



    @article{Bochkovskii2024:arxiv,
        author     = {Aleksei Bochkovskii and Ama\"{e}l Delaunoy and Hugo Germain and Marcel Santos and
                    Yichao Zhou and Stephan R. Richter and Vladlen Koltun}
        title      = {Depth Pro: Sharp Monocular Metric Depth in Less Than a Second},
        journal    = {arXiv},
        year       = {2024},
        url        = {https://arxiv.org/abs/2410.02073},
    }
    """

    def __init__(self, model_path: str = None):
        super().__init__(model_path)
        if self.model_path is None:
            self.model_path = huggingface_hub.hf_hub_download(
                "apple/DepthPro", filename="depth_pro.pt"
            )

        self.default_config = DEFAULT_MONODEPTH_CONFIG_DICT
        self.default_config.checkpoint_uri = self.model_path

        self.image_processor = None
        self.load_model()

    def load_model(self):
        self.model, self.image_processor = create_model_and_transforms(
            self.default_config, device=self.device
        )

    @torch.no_grad()
    def _predict(self, image: np.ndarray) -> np.ndarray:
        self.to(self.device)
        image = self.image_processor(image)
        # TODO: support f_px input
        prediction = self.model.infer(image, f_px=None)
        depth = prediction["depth"]
        focallength_px = prediction["focallength_px"]
        depth = depth.detach().cpu().numpy()
        return depth

    def to(self, device: str):
        self.device = torch.device(device)
        self.model.to(self.device)
