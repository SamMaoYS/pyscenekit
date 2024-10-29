import abc
from typing import Union, Literal

import cv2
import trimesh
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

from pyscenekit.utils.common import log
from pyscenekit.scenekit3d.utils import intersect_lines, rotation_from2vectors


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
    def hfov(self, width: int):
        return 2 * np.arctan2(float(width), 2 * self.fx)

    def vfov(self, height: int):
        return 2 * np.arctan2(float(height), 2 * self.fy)

    # convert from perspective camera with fov convention to camera intrinsics
    @classmethod
    def from_fov(cls, hfov: float, vfov: float, width: int, height: int):
        width = float(width)
        height = float(height)
        fx = width / (2 * np.tan(hfov / 2))
        fy = height / (2 * np.tan(vfov / 2))
        cx = width / 2
        cy = height / 2
        intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        return cls(intrinsics)


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

    def gravity_aligned_obb(
        self,
        gravity=np.array([0.0, 1.0, 0.0]),
        align_axis=np.array([0.0, 0.0, 1.0]),
        nb_neighbors=20,
        std_ratio=3.0,
        visualize=False,
    ) -> o3d.geometry.OrientedBoundingBox:
        vertices = self.get_vertices()
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(vertices))
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors, std_ratio)
        points = np.asarray(pcd.points)

        def mobb_area(
            left_start,
            left_dir,
            right_start,
            right_dir,
            top_start,
            top_dir,
            bottom_start,
            bottom_dir,
        ):
            upper_left = intersect_lines(left_start, left_dir, top_start, top_dir)
            upper_right = intersect_lines(right_start, right_dir, top_start, top_dir)
            bottom_left = intersect_lines(
                bottom_start, bottom_dir, left_start, left_dir
            )

            return np.linalg.norm(upper_left - upper_right) * np.linalg.norm(
                upper_left - bottom_left
            )

        align_gravity = rotation_from2vectors(gravity, align_axis)

        tmp_points = np.matmul(align_gravity, points.transpose()).transpose()
        points_2d = tmp_points[:, 0:2]
        hull = ConvexHull(points_2d)

        # plot conver hull
        if visualize:
            log.debug(len(hull.vertices))
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111)
            plt.plot(points_2d[:, 0], points_2d[:, 1], ".")
            plt.plot(
                points_2d[hull.vertices, 0], points_2d[hull.vertices, 1], "r--", lw=4
            )
            plt.plot(
                points_2d[(hull.vertices[-1], hull.vertices[0]), 0],
                points_2d[(hull.vertices[-1], hull.vertices[0]), 1],
                "r--",
                lw=4,
            )
            plt.plot(
                points_2d[hull.vertices[:], 0],
                points_2d[hull.vertices[:], 1],
                marker="o",
                markersize=7,
                color="red",
            )
            ax.set_aspect("equal", adjustable="box")
        assert len(hull.vertices) > 0, "convex hull vertices number must be positive"

        # the vertices are in counterclockwise order
        hull_points = points_2d[hull.vertices]

        edge_dirs = np.roll(hull_points, -1, axis=0) - hull_points
        edge_norm = np.linalg.norm(edge_dirs, axis=1)
        edge_dirs /= edge_norm[:, None]

        min_idx = np.argmin(hull_points, axis=0)
        max_idx = np.argmax(hull_points, axis=0)
        min_pt = np.array((hull_points[min_idx[0]][0], hull_points[min_idx[1]][1]))
        max_pt = np.array((hull_points[max_idx[0]][0], hull_points[max_idx[1]][1]))

        left_idx = min_idx[0]
        right_idx = max_idx[0]
        top_idx = max_idx[1]
        bottom_idx = min_idx[1]

        left_dir = np.array((0, -1))
        right_dir = np.array((0, 1))
        top_dir = np.array((-1, 0))
        bottom_dir = np.array((1, 0))

        if visualize:
            plt.plot(
                hull_points[bottom_idx][0],
                hull_points[bottom_idx][1],
                marker="o",
                markersize=14,
                color="r",
            )
            plt.axline(
                (hull_points[bottom_idx][0], hull_points[bottom_idx][1]),
                (
                    hull_points[bottom_idx][0] + bottom_dir[0],
                    hull_points[bottom_idx][1] + bottom_dir[1],
                ),
            )
            plt.plot(
                hull_points[left_idx][0],
                hull_points[left_idx][1],
                marker="o",
                markersize=14,
                color="r",
            )
            plt.axline(
                (hull_points[left_idx][0], hull_points[left_idx][1]),
                (
                    hull_points[left_idx][0] + left_dir[0],
                    hull_points[left_idx][1] + left_dir[1],
                ),
            )
            plt.plot(
                hull_points[right_idx][0],
                hull_points[right_idx][1],
                marker="o",
                markersize=14,
                color="r",
            )
            plt.axline(
                (hull_points[right_idx][0], hull_points[right_idx][1]),
                (
                    hull_points[right_idx][0] + right_dir[0],
                    hull_points[right_idx][1] + right_dir[1],
                ),
            )
            plt.plot(
                hull_points[top_idx][0],
                hull_points[top_idx][1],
                marker="o",
                markersize=14,
                color="r",
            )
            plt.axline(
                (hull_points[top_idx][0], hull_points[top_idx][1]),
                (
                    hull_points[top_idx][0] + top_dir[0],
                    hull_points[top_idx][1] + top_dir[1],
                ),
            )

        min_area = np.finfo(np.float32).max
        best_bottom_dir = np.array((np.nan, np.nan))
        best_bottom_idx = -1
        best_left_dir = np.array((np.nan, np.nan))
        best_left_idx = -1
        best_top_dir = np.array((np.nan, np.nan))
        best_top_idx = -1
        best_right_dir = np.array((np.nan, np.nan))
        best_right_idx = -1

        def ortho(v):
            return np.array([v[1], -v[0]])

        for i in range((len(hull.vertices))):
            angles = [
                np.arccos(np.clip(np.dot(left_dir, edge_dirs[left_idx]), -1.0, 1.0)),
                np.arccos(np.clip(np.dot(right_dir, edge_dirs[right_idx]), -1.0, 1.0)),
                np.arccos(np.clip(np.dot(top_dir, edge_dirs[top_idx]), -1.0, 1.0)),
                np.arccos(
                    np.clip(np.dot(bottom_dir, edge_dirs[bottom_idx]), -1.0, 1.0)
                ),
            ]
            angles = np.asarray(angles)

            best_line = np.argmin(angles)
            min_angle = angles[best_line]

            if best_line == 0:
                left_dir = edge_dirs[left_idx]
                right_dir = -left_dir
                top_dir = ortho(left_dir)
                bottom_dir = -top_dir
                left_idx = (left_idx + 1) % len(hull.vertices)
            elif best_line == 1:
                right_dir = edge_dirs[right_idx]
                left_dir = -right_dir
                top_dir = ortho(left_dir)
                bottom_dir = -top_dir
                right_idx = (right_idx + 1) % len(hull.vertices)
            elif best_line == 2:
                top_dir = edge_dirs[top_idx]
                bottom_dir = -top_dir
                left_dir = ortho(bottom_dir)
                right_dir = -left_dir
                top_idx = (top_idx + 1) % len(hull.vertices)
            elif best_line == 3:
                bottom_dir = edge_dirs[bottom_idx]
                top_dir = -bottom_dir
                left_dir = ortho(bottom_dir)
                right_dir = -left_dir
                bottom_idx = (bottom_idx + 1) % len(hull.vertices)
            else:
                assert False

            area = mobb_area(
                hull_points[left_idx],
                left_dir,
                hull_points[right_idx],
                right_dir,
                hull_points[top_idx],
                top_dir,
                hull_points[bottom_idx],
                bottom_dir,
            )

            if area < min_area:
                min_area = area
                best_bottom_dir = bottom_dir
                best_bottom_idx = bottom_idx
                best_left_dir = left_dir
                best_left_idx = left_idx
                best_right_dir = right_dir
                best_right_idx = right_idx
                best_top_dir = top_dir
                best_top_idx = top_idx

        p_bl = intersect_lines(
            hull_points[best_bottom_idx],
            best_bottom_dir,
            hull_points[best_left_idx],
            best_left_dir,
        )
        p_br = intersect_lines(
            hull_points[best_bottom_idx],
            best_bottom_dir,
            hull_points[best_right_idx],
            best_right_dir,
        )
        p_tl = intersect_lines(
            hull_points[best_left_idx],
            best_left_dir,
            hull_points[best_top_idx],
            best_top_dir,
        )

        len_b = np.linalg.norm(p_bl - p_br)
        len_l = np.linalg.norm(p_bl - p_tl)

        if len_b < len_l:
            vec = best_bottom_dir / np.linalg.norm(best_bottom_dir)
        else:
            vec = best_left_dir / np.linalg.norm(best_left_dir)
            log.debug(vec)
        vec = np.concatenate([vec, [0]])
        if visualize:
            plt.axline(
                (hull_points[best_bottom_idx][0], hull_points[best_bottom_idx][1]),
                (
                    hull_points[best_bottom_idx][0] + best_bottom_dir[0],
                    hull_points[best_bottom_idx][1] + best_bottom_dir[1],
                ),
                color="m",
                lw=6,
            )
            plt.axline(
                (hull_points[best_left_idx][0], hull_points[best_left_idx][1]),
                (
                    hull_points[best_left_idx][0] + best_left_dir[0],
                    hull_points[best_left_idx][1] + best_left_dir[1],
                ),
                color="m",
                lw=6,
            )
            plt.axline(
                (hull_points[best_right_idx][0], hull_points[best_right_idx][1]),
                (
                    hull_points[best_right_idx][0] + best_right_dir[0],
                    hull_points[best_right_idx][1] + best_right_dir[1],
                ),
                color="m",
                lw=6,
            )
            plt.axline(
                (hull_points[best_top_idx][0], hull_points[best_top_idx][1]),
                (
                    hull_points[best_top_idx][0] + best_top_dir[0],
                    hull_points[best_top_idx][1] + best_top_dir[1],
                ),
                color="m",
                lw=6,
            )

            plt.plot(
                hull_points[best_bottom_idx][0],
                hull_points[best_bottom_idx][1],
                marker="o",
                markersize=21,
                color="g",
            )
            plt.plot(
                hull_points[best_left_idx][0],
                hull_points[best_left_idx][1],
                marker="o",
                markersize=21,
                color="g",
            )
            plt.plot(
                hull_points[best_right_idx][0],
                hull_points[best_right_idx][1],
                marker="o",
                markersize=21,
                color="g",
            )
            plt.plot(
                hull_points[best_top_idx][0],
                hull_points[best_top_idx][1],
                marker="o",
                markersize=21,
                color="g",
            )
            plt.tight_layout()
            plt.savefig("mobb.png")
            log.debug("mobb.png saved")
            plt.cla()

        third_t = np.array([np.cross(-vec, align_axis), -vec, align_axis])
        trans_w2b = np.matmul(third_t, align_gravity)
        aligned_points = np.matmul(trans_w2b, points.transpose()).transpose()

        min_pt = np.amin(aligned_points, axis=0)
        max_pt = np.amax(aligned_points, axis=0)

        center = (min_pt + max_pt) / 2.0

        trans_inv = np.linalg.inv(trans_w2b)
        obb_center = np.matmul(trans_inv, center)
        obb_size = max_pt - min_pt

        return o3d.geometry.OrientedBoundingBox(obb_center, trans_inv, obb_size)


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

    def load_o3d_mesh(self, mesh_path: str):
        self.mesh = o3d.io.read_triangle_mesh(mesh_path)
        return self.mesh

    def load_trimesh_mesh(self, mesh_path: str):
        self.mesh = trimesh.load(mesh_path)
        return self.mesh

    def get_vertices(self):
        if isinstance(self.mesh, o3d.geometry.TriangleMesh):
            return np.asarray(self.mesh.vertices)
        elif isinstance(self.mesh, trimesh.Trimesh):
            return self.mesh.vertices
        else:
            raise ValueError(f"Unsupported mesh type: {type(self.mesh)}")

    def get_faces(self):
        if isinstance(self.mesh, o3d.geometry.TriangleMesh):
            return np.asarray(self.mesh.triangles)
        elif isinstance(self.mesh, trimesh.Trimesh):
            return self.mesh.faces
        else:
            raise ValueError(f"Unsupported mesh type: {type(self.mesh)}")

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

    def get_trimesh_mesh(self):
        if isinstance(self.mesh, o3d.geometry.TriangleMesh):
            vertices = np.asarray(self.mesh.vertices)
            faces = np.asarray(self.mesh.triangles)
            if self.mesh.has_vertex_colors():
                colors = np.asarray(self.mesh.vertex_colors)
            else:
                colors = None
            return trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors)
        elif isinstance(self.mesh, trimesh.Trimesh):
            return self.mesh
        else:
            raise ValueError(f"Unsupported mesh type: {type(self.mesh)}")

    def export(self, output_path: str):
        if isinstance(self.mesh, o3d.geometry.TriangleMesh):
            o3d.io.write_triangle_mesh(output_path, self.mesh)
        elif isinstance(self.mesh, trimesh.Trimesh):
            self.mesh.export(output_path)
        else:
            raise ValueError(f"Unsupported mesh type: {type(self.mesh)}")


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
            o3d.utility.Vector3dVector(point_cloud.vertices)
        )
        self.point_cloud.colors = o3d.utility.Vector3dVector(
            point_cloud.colors.astype(np.float32)[:, :3] / 255.0
        )

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

    def export(self, output_path: str):
        o3d.io.write_point_cloud(output_path, self.point_cloud)


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
