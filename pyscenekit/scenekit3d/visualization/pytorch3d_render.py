import torch
import torch.nn as nn
import numpy as np

from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
    HardFlatShader,
    MeshRenderer,
)
from pytorch3d.utils import cameras_from_opencv_projection

from pyscenekit.scenekit3d.common import SceneKitCamera

class MeshRendererWithDepth(MeshRenderer):
    def __init__(self, rasterizer, shader):
        super().__init__(rasterizer, shader)

    def forward(self, meshes_world, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(meshes_world, **kwargs)
        images = self.shader(fragments, meshes_world, **kwargs)
        return images, fragments.zbuf

class PyTorch3DRenderer:
    def __init__(self, mesh=None, device="cuda", width=640, height=480):
        self.device = torch.device(device)
        self.mesh = None
        self.cameras = []

        self.width = width
        self.height = height

        if mesh is not None:
            self.set_mesh(mesh)

        self.set_device(device)
        self.raster_settings = RasterizationSettings(
            image_size=[self.height, self.width],
            blur_radius=0.0,
            faces_per_pixel=1,
            cull_backfaces=True,
        )

    def load_obj(self, obj_path):
        self.mesh = load_objs_as_meshes([obj_path], device=self.device)
        return self.mesh

    def from_scenekit_camera(self, camera: SceneKitCamera):
        camera_extrinsics = camera.extrinsics
        R = (
            torch.from_numpy(camera_extrinsics[:3, :3])
            .float()
            .unsqueeze(0)
            .to(self.device)
        )
        tvec = (
            torch.from_numpy(camera_extrinsics[:3, 3])
            .float()
            .unsqueeze(0)
            .to(self.device)
        )
        camera_matrix = (
            torch.from_numpy(camera.intrinsics).float().unsqueeze(0).to(self.device)
        )
        image_size = (
            torch.tensor([self.height, self.width]).float().unsqueeze(0).to(self.device)
        )
        camera = cameras_from_opencv_projection(
            R, tvec, camera_matrix, image_size=image_size
        )
        return camera

    def add_camera(self, camera: SceneKitCamera):
        self.cameras.append(self.from_scenekit_camera(camera))

    def reset_cameras(self):
        self.cameras = []

    def set_mesh(self, mesh):
        if isinstance(mesh, str):
            self.mesh = self.load_obj(mesh)
        else:
            self.mesh = mesh

    def set_device(self, device="cuda:0"):
        self.device = torch.device(device)

    def rasterize(self):
        resolutoin = [self.height, self.width]
        self.raster_settings.image_size = resolutoin

        rasterizer = MeshRasterizer(
            cameras=self.cameras, raster_settings=self.raster_settings
        )

        fragments = rasterizer(self.mesh.to(self.device))
        return fragments

    def render(self):
        resolutoin = [self.height, self.width]
        self.raster_settings.image_size = resolutoin

        rasterizer = MeshRasterizer(
            cameras=self.cameras, raster_settings=self.raster_settings
        )
        shader = HardFlatShader(
            device=self.device,
            cameras=self.cameras,
        )

        renderer = MeshRendererWithDepth(rasterizer=rasterizer, shader=shader)
        images, depth = renderer(self.mesh.to(self.device))
        return images, depth
