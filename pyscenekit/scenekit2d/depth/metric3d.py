import torch
import numpy as np
import huggingface_hub

from pyscenekit.scenekit2d.depth.base import BaseDepthEstimation
from pyscenekit.scenekit2d.depth.modules.metric3d import Metric3D


class Metric3DDepthEstimation(BaseDepthEstimation):
    """
    Metric3D: Towards Zero-shot Metric 3D Prediction from A Single Image

    Authors: Wei Yin, Chi Zhang, Hao Chen, Zhipeng Cai, Gang Yu, Kaixuan Wang, Xiaozhi Chen, Chunhua Shen

    https://github.com/YvanYin/Metric3D


    @article{yin2023metric,
        title={Metric3D: Towards Zero-shot Metric 3D Prediction from A Single Image},
        author={Wei Yin, Chi Zhang, Hao Chen, Zhipeng Cai, Gang Yu, Kaixuan Wang, Xiaozhi Chen, Chunhua Shen},
        booktitle={ICCV},
        year={2023}
    }

    Metric3D v2: A Versatile Monocular Geometric Foundation Model for Zero-shot Metric Depth and Surface Normal Estimation

    Authors: Hu, Mu and Yin, Wei and Zhang, Chi and Cai, Zhipeng and Long, Xiaoxiao and Chen, Hao and Wang, Kaixuan and Yu, Gang and Shen, Chunhua and Shen, Shaojie

    https://github.com/YvanYin/Metric3D

    @article{hu2024metric3d,
        title={Metric3D v2: A Versatile Monocular Geometric Foundation Model for Zero-shot Metric Depth and Surface Normal Estimation},
        author={Hu, Mu and Yin, Wei and Zhang, Chi and Cai, Zhipeng and Long, Xiaoxiao and Chen, Hao and Wang, Kaixuan and Yu, Gang and Shen, Chunhua and Shen, Shaojie},
        journal={arXiv preprint arXiv:2404.15506},
        year={2024}
    }
    """

    def __init__(self, model_path: str = None, model_type: str = "large"):
        super().__init__(model_path)
        if self.model_path is None:
            if model_type == "large":
                self.model_path = huggingface_hub.hf_hub_download(
                    "JUGGHM/Metric3D", filename="metric_depth_vit_large_800k.pth"
                )
            elif model_type == "giant":
                self.model_path = huggingface_hub.hf_hub_download(
                    "JUGGHM/Metric3D", filename="metric_depth_vit_giant2_800k.pth"
                )
            else:
                raise ValueError(f"Invalid model type: {model_type}")

        self.load_model(model_type)

    def load_model(self, model_type: str):
        if model_type == "large":
            config_path = "configs/decoder/vit.raft5.large.py"
        elif model_type == "giant":
            config_path = "configs/decoder/vit.raft5.giant2.py"
        else:
            raise ValueError(f"Invalid model type: {model_type}")
        self.model = Metric3D(config_path)
        self.model.load_ckpt(self.model_path)

    @torch.no_grad()
    def _predict(
        self,
        image: np.ndarray,
        fx: float = None,
        fy: float = None,
        cx: float = None,
        cy: float = None,
    ):
        self.to(self.device)
        if fx is None or fy is None or cx is None or cy is None:
            fov = np.pi / 3
            focal_length = image.shape[1] / (2 * np.tan(fov / 2))
            intrinsic = [
                focal_length,
                focal_length,
                image.shape[1] / 2,
                image.shape[0] / 2,
            ]
        else:
            intrinsic = [fx, fy, cx, cy]
        depth, normal = self.model.inference(image, intrinsic=intrinsic)
        return {"depth": depth, "normal": normal}

    def to(self, device: str):
        self.device = torch.device(device)
        self.model.to(self.device)
