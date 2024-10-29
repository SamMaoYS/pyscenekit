from enum import Enum

from pyscenekit.scenekit3d.reconstruction.moge import MoGeReconstruction
from pyscenekit.scenekit3d.reconstruction.dust3r import Dust3rReconstruction


class SingleViewReconstructionMethod(Enum):
    MOGE = "moge"


class SingleViewReconstructionModel:
    def __new__(cls, method: SingleViewReconstructionMethod, model_path: str = None):
        if isinstance(method, str):
            method = SingleViewReconstructionMethod[method.upper()]

        if method == SingleViewReconstructionMethod.MOGE:
            return MoGeReconstruction(model_path)
        else:
            raise NotImplementedError(
                f"Single-view reconstruction method {method} not implemented"
            )


class MultiViewReconstructionMethod(Enum):
    DUST3R = "dust3r"


class MultiViewReconstructionModel:
    def __new__(cls, method: MultiViewReconstructionMethod, model_path: str = None):
        if isinstance(method, str):
            method = MultiViewReconstructionMethod[method.upper()]

        if method == MultiViewReconstructionMethod.DUST3R:
            return Dust3rReconstruction(model_path)
        else:
            raise NotImplementedError(
                f"Multi-view reconstruction method {method} not implemented"
            )
