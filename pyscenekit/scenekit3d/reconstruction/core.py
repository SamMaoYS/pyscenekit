from enum import Enum

from pyscenekit.scenekit3d.reconstruction.dust3r import Dust3rReconstruction


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
