from enum import Enum

from pyscenekit.scenekit2d.depth.midas import MidasDepthEstimation
from pyscenekit.scenekit2d.depth.depth_pro import DepthProDepthEstimation
from pyscenekit.scenekit2d.depth.depth_anything_v2 import DepthAnythingV2DepthEstimation
from pyscenekit.scenekit2d.depth.lotus import LotusDepthEstimation
from pyscenekit.scenekit2d.depth.metric3d import Metric3DDepthEstimation


class DepthEstimationMethod(Enum):
    MIDAS = "midas"
    DEPTH_ANYTHING_V2 = "depth_anything_v2"
    METRIC3D = "metric3d"
    DEPTH_PRO = "depth_pro"
    LOTUS_DEPTH = "lotus_depth"


class DepthEstimationModel:
    def __new__(cls, method: DepthEstimationMethod, model_path: str = None, **kwargs):
        if isinstance(method, str):
            method = DepthEstimationMethod[method.upper()]

        if method == DepthEstimationMethod.MIDAS:
            return MidasDepthEstimation(model_path)
        elif method == DepthEstimationMethod.DEPTH_ANYTHING_V2:
            return DepthAnythingV2DepthEstimation(model_path)
        elif method == DepthEstimationMethod.DEPTH_PRO:
            return DepthProDepthEstimation(model_path)
        elif method == DepthEstimationMethod.LOTUS_DEPTH:
            return LotusDepthEstimation(model_path)
        elif method == DepthEstimationMethod.METRIC3D:
            model_type = kwargs.get("model_type", "large")
            return Metric3DDepthEstimation(model_path, model_type)
        else:
            raise NotImplementedError(
                f"Depth estimation method {method} not implemented"
            )
