import torch
import torch.nn as nn
import numpy as np
from typing import List
from pytorch3d.io import load_objs_as_meshes, load_ply
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

    def load_ply(self, ply_path):
        verts, faces = load_ply(ply_path)
        self.mesh = Meshes(verts=[verts], faces=[faces])
        return self.mesh.to(self.device)

    def convert_scenekit_cameras(self, target_resolution: int = 640):
        R_list = []
        tvec_list = []
        camera_matrix_list = []
        image_size_list = []
        for camera in self.cameras:
            height = camera.height
            width = camera.width

            scale = target_resolution / self.width
            width = int(width * scale)
            height = int(height * scale)
            camera_extrinsics = camera.extrinsics
            R = torch.from_numpy(camera_extrinsics[:3, :3]).float()
            tvec = torch.from_numpy(camera_extrinsics[:3, 3]).float()
            camera_matrix = torch.from_numpy(camera.intrinsics).float()
            camera_matrix[:2, :] = camera_matrix[:2, :] * scale
            image_size = torch.tensor([height, width]).float()
            R_list.append(R)
            tvec_list.append(tvec)
            camera_matrix_list.append(camera_matrix)
            image_size_list.append(image_size)

        R = torch.stack(R_list).to(self.device)
        tvec = torch.stack(tvec_list).to(self.device)
        camera_matrix = torch.stack(camera_matrix_list).to(self.device)
        image_size = torch.stack(image_size_list).to(self.device)
        return cameras_from_opencv_projection(
            R, tvec, camera_matrix, image_size=image_size
        )

    def add_camera(self, camera: SceneKitCamera):
        self.cameras.append(camera)

    def set_cameras(self, cameras: List[SceneKitCamera]):
        self.cameras = cameras

    def reset_cameras(self):
        self.cameras = []

    def set_mesh(self, mesh):
        if isinstance(mesh, str):
            self.mesh = self.load_obj(mesh)
        else:
            self.mesh = mesh

    def set_device(self, device="cuda:0"):
        self.device = torch.device(device)

    def rasterize(self, target_resolution: int = 640):
        cameras = self.convert_scenekit_cameras(target_resolution)
        image_size = cameras[0].image_size[0].cpu().numpy()
        resolutoin = [int(image_size[0]), int(image_size[1])]
        self.raster_settings.image_size = resolutoin

        rasterizer = MeshRasterizer(
            cameras=cameras, raster_settings=self.raster_settings
        )

        batch_size = len(cameras)
        if len(self.mesh.verts_list()) != batch_size:
            self.mesh = self.mesh.extend(batch_size).to(self.device)
        fragments = rasterizer(self.mesh)
        return fragments

    def render(self, target_resolution: int = 640):
        cameras = self.convert_scenekit_cameras(target_resolution)
        image_size = cameras[0].image_size[0].cpu().numpy()
        resolutoin = [int(image_size[0]), int(image_size[1])]
        self.raster_settings.image_size = resolutoin

        rasterizer = MeshRasterizer(
            cameras=cameras, raster_settings=self.raster_settings
        )
        shader = HardFlatShader(
            device=self.device,
            cameras=cameras,
        )

        renderer = MeshRendererWithDepth(rasterizer=rasterizer, shader=shader)
        batch_size = len(cameras)
        if len(self.mesh.verts_list()) != batch_size:
            self.mesh = self.mesh.extend(batch_size).to(self.device)
        images, depth = renderer(self.mesh)
        return images, depth
