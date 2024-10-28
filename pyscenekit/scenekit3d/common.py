import abc
from typing import Union, Literal

import cv2
import trimesh
import numpy as np
import open3d as o3d


class SceneKitCamera:
    def __init__(
        self,
        intrinsics: np.ndarray = None,
        extrinsics: np.ndarray = np.eye(4),
        name: str = "camera",
    ):
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.name = name
        self.camera_pose = np.linalg.inv(extrinsics)

    def __post_init__(self):
        if self.intrinsics is None:
            self.intrinsics = o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
            ).intrinsic_matrix

    def set_intrinsics(self, intrinsics: np.ndarray):
        self.intrinsics = intrinsics

    def set_extrinsics(self, extrinsics: np.ndarray):
        self.extrinsics = extrinsics
        self.camera_pose = np.linalg.inv(extrinsics)

    def set_camera_pose(self, camera_pose: np.ndarray):
        self.camera_pose = camera_pose
        self.extrinsics = np.linalg.inv(camera_pose)

    def set_name(self, name: str):
        self.name = name

    @property
    def fx(self):
        return self.intrinsics[0, 0]

    @property
    def fy(self):
        return self.intrinsics[1, 1]

    @property
    def cx(self):
        return self.intrinsics[0, 2]

    @property
    def cy(self):
        return self.intrinsics[1, 2]

    # convert to perspective camera with fov convention
    @property
    def fovh(self, width: int):
        return 2 * np.arctan2(float(width), 2 * self.fx)

    @property
    def fovv(self, height: int):
        return 2 * np.arctan2(float(height), 2 * self.fy)

    # convert from perspective camera with fov convention to camera intrinsics
    def from_fov(self, fovh: float, fovv: float, width: int, height: int):
        width = float(width)
        height = float(height)
        fx = width / (2 * np.tan(fovh / 2))
        fy = height / (2 * np.tan(fovv / 2))
        cx = width / 2
        cy = height / 2
        self.intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        return self.intrinsics


class SceneKitGeometry(abc.ABC):
    @abc.abstractmethod
    def get_vertices(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_colors(self):
        raise NotImplementedError

    @property
    def centroid(self):
        return np.mean(self.get_vertices(), axis=0)

    @property
    def min_bound(self):
        return np.min(self.get_vertices(), axis=0)

    @property
    def max_bound(self):
        return np.max(self.get_vertices(), axis=0)


class SceneKitMesh(SceneKitGeometry):
    def __init__(
        self, mesh: Union[str, o3d.geometry.TriangleMesh, trimesh.Trimesh] = None
    ):
        if isinstance(mesh, str):
            self.mesh = self.load_o3d_mesh(mesh)
        elif isinstance(mesh, o3d.geometry.TriangleMesh):
            self.mesh = mesh
        elif isinstance(mesh, trimesh.Trimesh):
            self.mesh = mesh
        elif mesh is not None:
            raise ValueError(f"Unsupported mesh type: {type(mesh)}")

    def load_o3d_mesh(self):
        self.mesh = o3d.io.read_triangle_mesh(self.mesh_path)

    def load_trimesh_mesh(self):
        self.mesh = trimesh.load(self.mesh_path)

    def get_vertices(self):
        if isinstance(self.mesh, o3d.geometry.TriangleMesh):
            return np.asarray(self.mesh.vertices)
        elif isinstance(self.mesh, trimesh.Trimesh):
            return self.mesh.vertices
        else:
            raise ValueError("Unsupported mesh type")

    def get_faces(self):
        if isinstance(self.mesh, o3d.geometry.TriangleMesh):
            return np.asarray(self.mesh.triangles)
        elif isinstance(self.mesh, trimesh.Trimesh):
            return self.mesh.faces
        else:
            raise ValueError("Unsupported mesh type")

    def get_colors(self, color_type: Literal["vertex", "face"] = "vertex"):
        if color_type == "vertex":
            return self.get_vertex_colors()
        elif color_type == "face":
            return self.get_face_colors()
        else:
            raise ValueError(f"Unsupported color type: {color_type}")

    def get_vertex_colors(self):
        if isinstance(self.mesh, o3d.geometry.TriangleMesh):
            return np.asarray(self.mesh.vertex_colors)
        elif isinstance(self.mesh, trimesh.Trimesh):
            return self.mesh.visual.vertex_colors
        else:
            raise ValueError("Unsupported mesh type")

    def get_face_colors(self):
        if isinstance(self.mesh, o3d.geometry.TriangleMesh):
            raise ValueError("o3d.geometry.TriangleMesh does not support face colors")
        elif isinstance(self.mesh, trimesh.Trimesh):
            return self.mesh.visual.face_colors
        else:
            raise ValueError("Unsupported mesh type")

    def transform(self, transform: np.ndarray):
        if isinstance(self.mesh, o3d.geometry.TriangleMesh):
            self.mesh.transform(transform)
        elif isinstance(self.mesh, trimesh.Trimesh):
            self.mesh.apply_transform(transform)


class SceneKitPointCloud(SceneKitGeometry):
    def __init__(
        self,
        point_cloud: Union[str, o3d.geometry.PointCloud, trimesh.PointCloud] = None,
    ):
        self.point_cloud = None
        if isinstance(point_cloud, str):
            self.point_cloud = self.load_point_cloud(point_cloud)
        elif isinstance(point_cloud, o3d.geometry.PointCloud):
            self.set_point_cloud(point_cloud)
        elif isinstance(point_cloud, trimesh.PointCloud):
            self.from_trimesh_point_cloud(point_cloud)
        elif point_cloud is not None:
            raise ValueError(f"Unsupported point_cloud type: {type(point_cloud)}")

    def set_point_cloud(self, point_cloud: o3d.geometry.PointCloud):
        self.point_cloud = point_cloud

    def load_point_cloud(self, point_cloud_path: str):
        self.point_cloud = o3d.io.read_point_cloud(point_cloud_path)

    def get_vertices(self):
        return np.asarray(self.point_cloud.points)

    def get_colors(self):
        return np.asarray(self.point_cloud.colors)

    def get_normals(self):
        # if point_cloud has normals
        if not self.point_cloud.has_normals():
            self.estimate_normals()
        return np.asarray(self.point_cloud.normals)

    def estimate_normals(self, knn: int = 30):
        self.point_cloud.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn)
        )

    def to_trimesh_point_cloud(self):
        return trimesh.PointCloud(self.get_vertices(), self.get_colors())

    def from_trimesh_point_cloud(self, point_cloud: trimesh.PointCloud):
        self.point_cloud = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(point_cloud.vertices),
            o3d.utility.Vector3dVector(point_cloud.colors),
        )

    from diffusers import ControlNetModel

    @classmethod
    def from_vertices(cls, vertices: np.ndarray, colors: np.ndarray = None):
        point_cloud = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(vertices),
        )
        if colors is not None:
            if colors.dtype == np.uint8:
                colors = colors.astype(np.float32) / 255.0
            point_cloud.colors = o3d.utility.Vector3dVector(colors)
        return cls(point_cloud)

    def transform(self, transform: np.ndarray):
        self.point_cloud.transform(transform)


class SceneKitStructuredPointCloud:
    def __init__(
        self,
        depth_image: Union[str, np.ndarray, o3d.geometry.Image],
        camera: SceneKitCamera,
        rgb_image: Union[str, np.ndarray, o3d.geometry.Image] = None,
    ):
        if isinstance(depth_image, str):
            self.depth_image = cv2.imread(depth_image, cv2.IMREAD_UNCHANGED)

        if isinstance(depth_image, np.ndarray):
            # convert to meters
            if depth_image.dtype == np.uint16:
                depth_image = depth_image.astype(np.float32)
                depth_image = depth_image / 1000.0
            self.depth_image = o3d.geometry.Image(depth_image)
        elif isinstance(depth_image, o3d.geometry.Image):
            self.depth_image = depth_image
        else:
            raise ValueError(f"Unsupported depth_image type: {type(depth_image)}")

        self.rgb_image = None
        if rgb_image is not None:
            if isinstance(rgb_image, str):
                rgb_image = cv2.imread(rgb_image)
                rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
            if isinstance(rgb_image, np.ndarray):
                if rgb_image.dtype != np.uint8:
                    rgb_image *= 255
                    rgb_image = rgb_image.astype(np.uint8)
                self.rgb_image = o3d.geometry.Image(rgb_image)
            elif isinstance(rgb_image, o3d.geometry.Image):
                self.rgb_image = rgb_image
            else:
                raise ValueError(f"Unsupported rgb_image type: {type(rgb_image)}")

        self.camera = camera

        self.point_cloud = self.unproject_point_cloud()

    def unproject_point_cloud(self, colormap_depth: bool = False):
        image_width, image_height = self.rgb_image.width, self.rgb_image.height
        o3d_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=image_width,
            height=image_height,
            fx=self.camera.intrinsics[0, 0],
            fy=self.camera.intrinsics[1, 1],
            cx=self.camera.intrinsics[0, 2],
            cy=self.camera.intrinsics[1, 2],
        )

        if colormap_depth:
            self.rgb_image = self.colormap_depth()

        if self.rgb_image is not None:
            return self._unproject_rgbd(o3d_intrinsics)
        else:
            return self._unproject_depth(o3d_intrinsics)

    def _unproject_rgbd(self, o3d_intrinsics: o3d.camera.PinholeCameraIntrinsic):
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            self.rgb_image, self.depth_image
        )
        return o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd, o3d_intrinsics, self.camera.extrinsics
        )

    def _unproject_depth(self, o3d_intrinsics: o3d.camera.PinholeCameraIntrinsic):
        return o3d.geometry.PointCloud.create_from_depth_image(
            self.depth_image, o3d_intrinsics, self.camera.extrinsics
        )

    def colormap_depth(self):
        from pyscenekit.scenekit2d.depth.base import BaseDepthEstimation

        rgb_image = BaseDepthEstimation.colormap(np.asarray(self.depth_image))
        return o3d.geometry.Image(rgb_image)

    def transform(self, transform: np.ndarray):
        self.point_cloud.transform(transform)

    def get_vertices(self):
        return np.asarray(self.point_cloud.points)

    def get_colors(self):
        return np.asarray(self.point_cloud.colors)
