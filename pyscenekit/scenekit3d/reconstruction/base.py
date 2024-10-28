import abc
from typing import Tuple, List
from dataclasses import dataclass

import cv2
import torch
import numpy as np
import open3d as o3d
from pyscenekit.scenekit2d.utils import ImageInput
from pyscenekit.scenekit2d.common import SceneKitImage
from pyscenekit.scenekit3d.common import (
    SceneKitCamera,
    SceneKitMesh,
    SceneKitPointCloud,
)


@dataclass
class MultiViewReconstructionInput:
    image_list: List[np.ndarray] = None
    camera_list: List[SceneKitCamera] = None


@dataclass
class MultiViewReconstructionOutput:
    color_list: List[np.ndarray] = None
    depth_list: List[np.ndarray] = None
    confidence_list: List[np.ndarray] = None
    mask_list: List[np.ndarray] = None
    cameras: List[SceneKitCamera] = None
    point_cloud_list: List[SceneKitPointCloud] = None
    mesh_list: List[SceneKitMesh] = None

    def export_pcd(self, output_path: str):
        # merge point clouds
        vertices = np.concatenate([pc.get_vertices() for pc in self.point_cloud_list])
        colors = np.concatenate([pc.get_colors() for pc in self.point_cloud_list])
        normals = np.concatenate([pc.get_normals() for pc in self.point_cloud_list])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        o3d.io.write_point_cloud(output_path, pcd)

    def to_dict(self):
        point_cloud_vertices_list = [
            point_cloud.get_vertices() for point_cloud in self.point_cloud_list
        ]
        point_cloud_colors_list = [
            point_cloud.get_colors() for point_cloud in self.point_cloud_list
        ]

        mesh_vertices_list = [mesh.get_vertices() for mesh in self.mesh_list]
        mesh_face_colors_list = [mesh.get_face_colors() for mesh in self.mesh_list]
        mesh_faces_list = [mesh.get_faces() for mesh in self.mesh_list]
        return {
            "color_list": self.color_list,
            "depth_list": self.depth_list,
            "confidence_list": self.confidence_list,
            "mask_list": self.mask_list,
            "cameras": self.cameras,
            "point_cloud_vertices_list": point_cloud_vertices_list,
            "point_cloud_colors_list": point_cloud_colors_list,
            "mesh_vertices_list": mesh_vertices_list,
            "mesh_face_colors_list": mesh_face_colors_list,
            "mesh_faces_list": mesh_faces_list,
        }


class MultiViewReconstructionModel(abc.ABC):
    @abc.abstractmethod
    def __init__(self, model_name: str = None):
        self.model_path = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

        # input and output are SceneKitImage objects
        self.input = MultiViewReconstructionInput()
        self.output = MultiViewReconstructionOutput()

    @abc.abstractmethod
    def load_model(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _predict(self) -> MultiViewReconstructionOutput:
        raise NotImplementedError

    def __call__(
        self, image_list: ImageInput, camera_list: List[SceneKitCamera] = None
    ) -> MultiViewReconstructionOutput:
        input_image_list = [SceneKitImage(image).image for image in image_list]
        self.input = MultiViewReconstructionInput(input_image_list, camera_list)
        output = self._predict()
        return output

    @abc.abstractmethod
    def to(self, device: str):
        raise NotImplementedError
