import pathlib
import numpy as np
from PIL import Image
from mmengine import Config
from pyscenekit.scenekit2d.depth.modules.metric3d.models.monodepth_model import (
    get_configured_monodepth_model,
)
from pyscenekit.scenekit2d.depth.modules.metric3d.utils.running import load_ckpt
from pyscenekit.scenekit2d.depth.modules.metric3d.utils.do_test import (
    do_scalecano_test_with_custom_data,
)


class Metric3D:
    def __init__(self, config_path="configs/decoder/vit.raft5.large.py"):
        self.config_path = pathlib.Path(__file__).parent.resolve() / config_path
        print(self.config_path)
        self.cfg = Config.fromfile(self.config_path)
        self.model = get_configured_monodepth_model(
            self.cfg,
        )

    def load_ckpt(self, ckpt_path):
        self.model = load_ckpt(ckpt_path, self.model, strict_match=False)
        self.model.eval()

    def to(self, device):
        self.model.to(device)

    def inference(self, rgb, intrinsic=None):
        depth, normal = do_scalecano_test_with_custom_data(
            self.model, self.cfg, rgb, intrinsic
        )

        n_img_l2 = np.sqrt(np.sum(normal**2, axis=2, keepdims=True))
        n_img_norm = -normal / (n_img_l2 + 1e-8)
        return depth, n_img_norm
