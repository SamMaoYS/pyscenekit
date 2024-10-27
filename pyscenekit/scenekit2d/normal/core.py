from enum import Enum

from pyscenekit.scenekit2d.normal.dsine import DsineNormalEstimation


class NormalEstimationMethod(Enum):
    DSINE = "dsine"


class NormalEstimationModel:
    def __new__(cls, method: NormalEstimationMethod, model_path: str = None, **kwargs):
        if isinstance(method, str):
            method = NormalEstimationMethod[method.upper()]

        if method == NormalEstimationMethod.DSINE:
            efficientnet_path = kwargs.get("efficientnet_path", None)
            return DsineNormalEstimation(model_path, efficientnet_path)
        else:
            raise NotImplementedError(
                f"Normal estimation method {method} not implemented"
            )
