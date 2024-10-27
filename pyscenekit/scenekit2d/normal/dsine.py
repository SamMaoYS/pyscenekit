import torch
import numpy as np
import huggingface_hub
from torchvision import transforms
from torch.nn import functional as F

from pyscenekit.scenekit2d.normal.base import BaseNormalEstimation
from pyscenekit.scenekit2d.normal.modules.dsine import DSINE, dsine_utils


class DsineNormalEstimation(BaseNormalEstimation):
    def __init__(self, model_path: str = None, efficientnet_path: str = None):
        super().__init__(model_path)
        if self.model_path is None:
            self.model_path = huggingface_hub.hf_hub_download(
                "ysmao/pyscenekit", subfolder="dsine", filename="dsine.pt"
            )
        self.efficientnet_path = efficientnet_path
        if self.efficientnet_path is None:
            self.efficientnet_path = huggingface_hub.hf_hub_download(
                "ysmao/pyscenekit",
                subfolder="dsine",
                filename="tf_efficientnet_b5_ap-9e82fae8.pth",
            )

        self.load_model()

        self.t_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        self.intrinsics = None

    def load_model(self):
        self.model = DSINE(self.efficientnet_path)
        self.model = dsine_utils.load_checkpoint(self.model_path, self.model)

    def set_fov_intrinsics(self, height: int, width: int, fov: float = 60.0):
        self.fov = fov
        self.intrinsics = dsine_utils.get_intrins_from_fov(
            self.fov, height, width, self.device
        ).unsqueeze(0)
        return self.intrinsics

    @torch.no_grad()
    def _predict(self, image: np.ndarray) -> np.ndarray:
        self.to(self.device)
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(self.device)
        _, _, orig_H, orig_W = image.shape
        # zero-pad the input image so that both the width and height are multiples of 32
        l, r, t, b = dsine_utils.pad_input(orig_H, orig_W)
        image = F.pad(image, (l, r, t, b), mode="constant", value=0.0)
        image = self.t_normalize(image)

        if self.intrinsics is None:
            self.set_fov_intrinsics(orig_H, orig_W)

        self.intrinsics[:, 0, 2] += l
        self.intrinsics[:, 1, 2] += t

        pred_normal = self.model(image, self.intrinsics)[-1]
        pred_normal = pred_normal[:, :, t : t + orig_H, l : l + orig_W]
        normal = pred_normal.cpu().detach().numpy().squeeze(0).transpose(1, 2, 0)

        return normal

    def to(self, device: str):
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.pixel_coords = self.model.pixel_coords.to(self.device)
