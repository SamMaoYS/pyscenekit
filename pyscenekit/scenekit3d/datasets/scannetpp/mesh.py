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
        self.batch_size = 1
        self.cameras = []

    @property
    def mesh_path(self):
        return os.path.join(self.data_dir, "mesh_aligned_0.05.ply")
    
    def set_cameras(self, cameras: dict):
        self.cameras = cameras

    def render_depth(self, cameras: List[SceneKitCamera]):
        self.renderer.load_ply(self.mesh_path)
        self.renderer.set_cameras(cameras)
        fragments = self.renderer.rasterize()
        depth = fragments.zbuf.cpu().numpy()
        return depth
