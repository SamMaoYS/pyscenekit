from enum import Enum

from pyscenekit.scenekit2d.segmentation.upernet import UperNetSemanticSegmentation


class SemanticSegmentationMethod(Enum):
    UPERNET = "upernet"


class SemanticSegmentationModel:
    def __new__(
        cls, method: SemanticSegmentationMethod, model_path: str = None, **kwargs
    ):
        if isinstance(method, str):
            method = SemanticSegmentationMethod[method.upper()]

        if method == SemanticSegmentationMethod.UPERNET:
            return UperNetSemanticSegmentation(model_path)
        else:
            raise NotImplementedError(
                f"Normal estimation method {method} not implemented"
            )
