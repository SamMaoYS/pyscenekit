import abc
from copy import deepcopy
from typing import Union

import trimesh
import numpy as np
import open3d as o3d

from pyscenekit.scenekit3d.common import SceneKitCamera, SceneKitGeometry


class SceneKitRender(abc.ABC):
    def __init__(
        self,
        resolution=np.array([1024, 1024]),
        background_color=np.array([1.0, 1.0, 1.0, 0.0]),
    ):
        self.background_color = background_color
        self.ambient_color = np.array([1.0, 1.0, 1.0])
        self.resolution = resolution
        self.scene_centroid = np.array([0, 0, 0])
        self.scene_diameter = 1.0
        self.camera = SceneKitCamera.from_fov(
            hfov=np.pi / 3, vfov=np.pi / 3, width=resolution[0], height=resolution[1]
        )
        self.geometries = []

        self.point_size = 1.0
        self.backface_culling = True

        self._world_up = np.array([0, 1, 0])

    def add_geometry(
        self,
        geometry: SceneKitGeometry,
        transform: np.ndarray = None,
    ):
        tmp_geometry = deepcopy(geometry)
        if transform is not None:
            tmp_geometry.transform(transform)
        self.geometries.append(tmp_geometry)

    def update_scene(self):
        num_geometries = len(self.geometries)
        if num_geometries == 0:
            return 0.0

        vertices = np.concatenate(
            [geometry.get_vertices() for geometry in self.geometries]
        )
        min_bound = np.min(vertices, axis=0)
        max_bound = np.max(vertices, axis=0)
        self.scene_centroid = (min_bound + max_bound) / 2.0
        self.scene_diameter = np.linalg.norm(max_bound - min_bound)

    def update_camera(self, camera: SceneKitCamera):
        self.camera = camera

    def update_camera_pose(self, camera_pose: np.ndarray):
        self.camera.set_camera_pose(camera_pose)

    def reset_camera(self):
        self.camera = SceneKitCamera()
        self.set_camera_pose_by_angle()

    def set_camera_pose_by_angle(self, angle: float = np.pi * 0.75):
        self.update_scene()
        look_at_pos = self.scene_centroid
        distance = self.scene_diameter

        offset_tmp = distance * np.array([-np.cos(angle), -np.sin(angle), 1.0])
        offset_tmp = np.append(offset_tmp, [1.0])
        tmp_world_up = np.array([0, 0, 1])
        if not np.all(tmp_world_up == self._world_up):
            transform_tmp = trimesh.geometry.align_vectors(tmp_world_up, self._world_up)
            offset_tmp = transform_tmp.dot(offset_tmp)
        camera_pos = look_at_pos + offset_tmp[:3]

        forward = camera_pos - look_at_pos
        forward /= np.linalg.norm(forward)
        right = np.cross(self._world_up, forward)
        up = np.cross(forward, right)
        look_at = np.vstack((right, up, forward, camera_pos))
        cp = np.eye(4)
        cp[:3, :4] = look_at.T
        self.update_camera_pose(cp)

    def set_world_up(self, world_up: np.ndarray):
        self._world_up = world_up

    @abc.abstractmethod
    def render(self):
        raise NotImplementedError

    def set_point_size(self, point_size: float):
        self.point_size = point_size

    def set_backface_culling(self, backface_culling: bool):
        self.backface_culling = backface_culling
