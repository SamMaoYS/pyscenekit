import abc
from typing import Tuple, List
from dataclasses import dataclass

import cv2
import torch
import trimesh
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
class SingleViewReconstructionInput:
    image: SceneKitImage = None
    camera: SceneKitCamera = None


@dataclass
class SingleViewReconstructionOutput:
    color: np.ndarray = None
    depth: np.ndarray = None
    confidence: np.ndarray = None
    mask: np.ndarray = None
    camera: SceneKitCamera = None
    point_cloud: SceneKitPointCloud = None
    mesh: SceneKitMesh = None

    def export_pcd(self, output_path: str):
        self.point_cloud.export(output_path)

    def export_mesh(self, output_path: str):
        self.mesh.export(output_path)

    def to_dict(self):
        point_cloud_vertices = self.point_cloud.get_vertices()
        point_cloud_colors = self.point_cloud.get_colors()

        mesh_vertices = self.mesh.get_vertices()
        mesh_face_colors = self.mesh.get_face_colors()
        mesh_faces = self.mesh.get_faces()
        return {
            "color": self.color,
            "depth": self.depth,
            "confidence": self.confidence,
            "mask": self.mask,
            "camera": self.camera,
            "point_cloud_vertices": point_cloud_vertices,
            "point_cloud_colors": point_cloud_colors,
            "mesh_vertices": mesh_vertices,
            "mesh_face_colors": mesh_face_colors,
            "mesh_faces": mesh_faces,
        }

    @classmethod
    def from_dict(cls, data: dict):
        color = data["color"]
        depth = data["depth"]
        confidence = data["confidence"]
        mask = data["mask"]
        camera = data["camera"]

        point_cloud_vertices = data["point_cloud_vertices"]
        point_cloud_colors = data["point_cloud_colors"]
        mesh_vertices = data["mesh_vertices"]
        mesh_face_colors = data["mesh_face_colors"]
        mesh_faces = data["mesh_faces"]

        point_cloud = SceneKitPointCloud(
            trimesh.PointCloud(vertices=point_cloud_vertices, colors=point_cloud_colors)
        )

        mesh = SceneKitMesh(
            trimesh.Trimesh(
                vertices=mesh_vertices, faces=mesh_faces, face_colors=mesh_face_colors
            )
        )

        return cls(
            color,
            depth,
            confidence,
            mask,
            camera,
            point_cloud,
            mesh,
        )


class SingleViewReconstructionModel(abc.ABC):
    @abc.abstractmethod
    def __init__(self, model_name: str = None):
        self.model_path = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

        self.input: SceneKitImage = None
        self.output: SingleViewReconstructionOutput = None

    @abc.abstractmethod
    def load_model(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _predict(self) -> SingleViewReconstructionOutput:
        raise NotImplementedError

    def __call__(
        self, image: ImageInput, camera: SceneKitCamera = None
    ) -> SingleViewReconstructionOutput:
        input_image = SceneKitImage(image)
        self.input = SingleViewReconstructionInput(input_image, camera)
        output = self._predict()
        return output

    @abc.abstractmethod
    def to(self, device: str):
        raise NotImplementedError


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

    def export_pcd(self, output_path: str = None):
        # merge point clouds
        vertices = np.concatenate(
            [pc.get_vertices() for pc in self.point_cloud_list], axis=0
        )
        colors = np.concatenate(
            [pc.get_colors() for pc in self.point_cloud_list], axis=0
        )
        normals = np.concatenate(
            [pc.get_normals() for pc in self.point_cloud_list], axis=0
        )

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        if output_path is not None:
            o3d.io.write_point_cloud(output_path, pcd)

        return SceneKitPointCloud(pcd)

    def export_mesh(self, output_path: str = None):
        vertices = np.concatenate(
            [mesh.get_vertices() for mesh in self.mesh_list], axis=0
        )
        faces = np.concatenate([mesh.get_faces() for mesh in self.mesh_list], axis=0)
        colors = np.concatenate(
            [mesh.get_face_colors() for mesh in self.mesh_list], axis=0
        )

        vertices, faces, colors = zip(
            *[
                (m.get_vertices(), m.get_faces(), m.get_face_colors())
                for m in self.mesh_list
            ]
        )
        n_vertices = np.cumsum([0] + [len(v) for v in vertices])
        for i in range(len(faces)):
            faces[i][:] += n_vertices[i]

        vertices = np.concatenate(vertices)
        colors = np.concatenate(colors)
        faces = np.concatenate(faces)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, face_colors=colors)
        if output_path is not None:
            mesh.export(output_path)
        return SceneKitMesh(mesh)

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

    @classmethod
    def from_dict(cls, data: dict):
        color_list = data["color_list"]
        depth_list = data["depth_list"]
        confidence_list = data["confidence_list"]
        mask_list = data["mask_list"]
        cameras = data["cameras"]

        point_cloud_vertices_list = data["point_cloud_vertices_list"]
        point_cloud_colors_list = data["point_cloud_colors_list"]
        mesh_vertices_list = data["mesh_vertices_list"]
        mesh_face_colors_list = data["mesh_face_colors_list"]
        mesh_faces_list = data["mesh_faces_list"]

        point_cloud_list = [
            SceneKitPointCloud(trimesh.PointCloud(vertices=vertices, colors=colors))
            for vertices, colors in zip(
                point_cloud_vertices_list, point_cloud_colors_list
            )
        ]

        mesh_list = [
            SceneKitMesh(
                trimesh.Trimesh(vertices=vertices, faces=faces, face_colors=colors)
            )
            for vertices, faces, colors in zip(
                mesh_vertices_list, mesh_faces_list, mesh_face_colors_list
            )
        ]

        return cls(
            color_list,
            depth_list,
            confidence_list,
            mask_list,
            cameras,
            point_cloud_list,
            mesh_list,
        )


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
