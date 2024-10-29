import numpy as np
import pyrender
from pyrender.constants import RenderFlags

from pyscenekit.scenekit3d.common import (
    SceneKitMesh,
    SceneKitPointCloud,
    SceneKitStructuredPointCloud,
)
from pyscenekit.scenekit3d.visualization.base import SceneKitRender


class PyRenderRender(SceneKitRender):
    def __init__(
        self,
        resolution=np.array([1024, 1024]),
        background_color=np.array([1.0, 1.0, 1.0, 0.0]),
    ):
        super().__init__(resolution, background_color)

        self.scene = pyrender.Scene(bg_color=self.background_color)
        self.lights = []
        self.render_flags = RenderFlags.SHADOWS_DIRECTIONAL | RenderFlags.VERTEX_NORMALS
        self.znear = 1e-5
        self.zfar = 1e3

        self.camera_node = None
        self.renderer = None

    def add_node(self, node):
        self.scene.add_node(node)

    def remove_node(self, node):
        self.scene.remove_node(node)

    def from_trimesh_scene(self, scene, transform=None):
        trimesh_scene = scene.copy()
        if transform is not None:
            trimesh_scene.apply_transform(transform)

        self.scene = pyrender.Scene.from_trimesh_scene(
            trimesh_scene, bg_color=self.background_color
        )

    def get_camera_node(self):
        icam = pyrender.camera.IntrinsicsCamera(
            fx=self.camera.fx,
            fy=self.camera.fy,
            cx=self.camera.cx,
            cy=self.camera.cy,
            znear=self.znear,
            zfar=self.zfar,
            name=self.camera.name,
        )
        return pyrender.Node(camera=icam, matrix=self.camera.camera_pose)

    def add_directional_light(self, color=[1.0, 1.0, 1.0], intensity=2.0):
        dl = pyrender.DirectionalLight(color=color, intensity=intensity)
        self.lights.append(dl)

    def double_faces(self):
        self.render_flags |= pyrender.constants.RenderFlags.SKIP_CULL_FACES

    def reset(self):
        self.geometries = []
        self.lights = []
        self.scene.clear()
        self.background_color = np.array([0, 0, 0, 0])
        self.ambient_color = np.array([1, 1, 1])
        self.render_flags = RenderFlags.SHADOWS_DIRECTIONAL | RenderFlags.VERTEX_NORMALS
        self.znear = 1e-5
        self.zfar = 1e3
        self.camera_node = None
        self.renderer.delete()
        self.renderer = None
        self.reset_camera()

    def set_renderer(self):
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.resolution[0],
            viewport_height=self.resolution[1],
            point_size=self.point_size,
        )

    def render(self, window_title: str = "PyRenderWindow", interactive: bool = False):
        self.update_scene()
        self.scene.ambient_light = self.ambient_color

        for geometry in self.geometries:
            if isinstance(geometry, SceneKitMesh):
                mesh = geometry.get_trimesh_mesh()
                self.scene.add(pyrender.Mesh.from_trimesh(mesh))
            elif isinstance(geometry, SceneKitPointCloud) or isinstance(
                geometry, SceneKitStructuredPointCloud
            ):
                vertices = geometry.get_vertices()
                colors = geometry.get_colors()
                self.scene.add(
                    pyrender.Mesh.from_points(vertices, colors.astype(float))
                )
            else:
                raise ValueError(f"Unsupported geometry type: {type(geometry)}")

        if interactive:
            pyrender.Viewer(
                self.scene,
                viewport_size=self.resolution,
                use_raymond_lighting=True,
                window_title=window_title,
            )
            return

        for light in self.lights:
            self.add_node(light)

        if self.renderer is None:
            self.set_renderer()

        if self.camera_node is None:
            self.camera_node = self.get_camera_node()
            self.add_node(self.camera_node)

        self.scene.set_pose(self.camera_node, pose=self.camera.camera_pose)

        color, depth = self.renderer.render(self.scene, self.render_flags)
        return color, depth
