from enum import Enum

from pyscenekit.scenekit2d.depth.depth_anything_v2 import DepthAnythingV2DepthEstimation
from pyscenekit.scenekit2d.depth.midas import MidasDepthEstimation


class DepthEstimationMethod(Enum):
    MIDAS = "midas"
    DEPTH_ANYTHING_V2 = "depth_anything_v2"
    METRIC3D = "metric3d"
    DEPTH_PRO = "depth_pro"


class DepthEstimationModel:
    def __new__(cls, method: DepthEstimationMethod, model_path: str = None):
        if isinstance(method, str):
            method = DepthEstimationMethod[method.upper()]

        if method == DepthEstimationMethod.MIDAS:
            return MidasDepthEstimation(model_path)
        elif method == DepthEstimationMethod.DEPTH_ANYTHING_V2:
            return DepthAnythingV2DepthEstimation(model_path)
        else:
            raise NotImplementedError(
                f"Depth estimation method {method} not implemented"
            )
