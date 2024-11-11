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

    def export_all_depth(self, output_dir: str):
        num_cameras = len(self.cameras)
        start_idx = self.start_idx
        batch_size = self.batch_size
        end_idx = self.end_idx if self.end_idx != -1 else num_cameras

        for i in range(start_idx, end_idx, batch_size):
            batch_cameras = self.cameras[i : i + batch_size]
            depth = self.render_depth(batch_cameras)

            import pdb

            pdb.set_trace()

    def render_depth(self, cameras: List[SceneKitCamera]):
        self.renderer.load_ply(self.mesh_path)
        self.renderer.set_cameras(cameras)
        fragments = self.renderer.rasterize()
        depth = fragments.zbuf.cpu().numpy()
        return depth
