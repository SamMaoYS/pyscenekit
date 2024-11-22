import os
import cv2
from typing import List
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field

from pyscenekit.utils.common import log, read_json
from pyscenekit.scenekit3d.visualization.pytorch3d_render import PyTorch3DRenderer
from pyscenekit.scenekit3d.common import SceneKitCamera


class ScanNetPPMeshDataset:
    def __init__(self, data_dir: str, output_dir: str = None):
        self.data_dir = data_dir
        self.output_dir = output_dir if output_dir is not None else data_dir
        self.renderer = PyTorch3DRenderer()
        self.batch_size = 8
        self.cameras = []

        self.start_idx = 0
        self.end_idx = -1

    @property
    def mesh_path(self):
        return os.path.join(self.data_dir, "mesh_aligned_0.05.ply")

    def set_cameras(self, cameras: dict):
        self.cameras = cameras

    def export_all_depth(self, output_dir: str, target_resolution: int = 0):
        num_cameras = len(self.cameras)
        start_idx = self.start_idx
        batch_size = self.batch_size
        end_idx = self.end_idx if self.end_idx != -1 else num_cameras

        if target_resolution <= 0:
            target_resolution = self.cameras[0].width

        os.makedirs(output_dir, exist_ok=True)
        self.load_mesh()
        for i in range(start_idx, end_idx, batch_size):
            batch_cameras = self.cameras[i : i + batch_size]
            from time import time

            start = time()
            depth = self.render_depth(batch_cameras, target_resolution)
            end = time()
            print(f"Time taken: {end - start} seconds")

            for i in range(len(depth)):
                camera = batch_cameras[i]
                depth_i = depth[i, :, :, 0]
                # min max normalize
                mask = depth_i < 0
                depth_i[mask] = 0

                depth_i = (depth_i * 1000).astype(np.uint16)
                file_name = camera.name.split(".")[0] + ".png"
                cv2.imwrite(os.path.join(output_dir, file_name), depth_i)

    def export_all_triangles(self, output_dir: str, target_resolution: int = 0):
        num_cameras = len(self.cameras)
        start_idx = self.start_idx
        batch_size = self.batch_size
        end_idx = self.end_idx if self.end_idx != -1 else num_cameras

        if target_resolution <= 0:
            target_resolution = self.cameras[0].width

        os.makedirs(output_dir, exist_ok=True)
        self.load_mesh()
        for i in range(start_idx, end_idx, batch_size):
            batch_cameras = self.cameras[i : i + batch_size]
            from time import time

            start = time()
            triangles = self.rasterize_triangles(batch_cameras, target_resolution)
            end = time()
            print(f"Time taken: {end - start} seconds")

            for i in range(len(triangles)):
                camera = batch_cameras[i]
                triangles_i = triangles[i, :, :, 0]
                file_name = camera.name.split(".")[0] + ".npy"
                np.save(os.path.join(output_dir, file_name), triangles_i)

    def load_mesh(self):
        self.renderer.load_ply(self.mesh_path)

    def render_depth(self, cameras: List[SceneKitCamera], target_resolution: int = 640):
        self.renderer.set_cameras(cameras)
        fragments = self.renderer.rasterize(target_resolution)
        depth = fragments.zbuf.cpu().numpy()
        return depth

    def rasterize_triangles(
        self, cameras: List[SceneKitCamera], target_resolution: int = 640
    ):
        self.renderer.set_cameras(cameras)
        fragments = self.renderer.rasterize(target_resolution)
        pix_to_face = fragments.pix_to_face.cpu().numpy()
        return pix_to_face
