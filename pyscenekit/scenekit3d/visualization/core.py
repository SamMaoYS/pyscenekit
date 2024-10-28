from enum import Enum

from pyscenekit.scenekit3d.visualization.pyrender_render import PyRenderRender


class VisualizationMethod(Enum):
    PYRENDER = "pyrender"


class SceneKitRenderer:
    def __new__(cls, method: VisualizationMethod, **kwargs):
        if isinstance(method, str):
            method = VisualizationMethod[method.upper()]

        if method == VisualizationMethod.PYRENDER:
            return PyRenderRender(**kwargs)
        else:
            raise NotImplementedError(f"Visualization method {method} not implemented")
