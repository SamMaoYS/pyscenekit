import os
import cv2
import numpy as np
from glob import glob
from natsort import natsorted
from dataclasses import dataclass, field

from pyscenekit.utils.common import read_json
from pyscenekit.scenekit3d.common import SceneKitCamera
from pyscenekit.scenekit3d.utils import qvec2rotmat


@dataclass
class DistortionParams:
    height: int
    width: int
    fx: float
    fy: float
    cx: float
    cy: float
    k1: float
    k2: float
    k3: float
    k4: float
    K: np.ndarray = field(init=False)
    distortion_params: np.ndarray = field(init=False)
    new_K: np.ndarray = field(init=False)
    map1: np.ndarray = field(init=False)
    map2: np.ndarray = field(init=False)

    def compute_undistort_intrinsic(self):
        assert len(self.distortion_params.shape) == 1
        assert self.distortion_params.shape[0] == 4  # OPENCV_FISHEYE has k1, k2, k3, k4

        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            self.K,
            self.distortion_params,
            (self.width, self.height),
            np.eye(3),
            balance=0.0,
        )
        # Make the cx and cy to be the center of the image
        new_K[0, 2] = self.width / 2.0
        new_K[1, 2] = self.height / 2.0
        return new_K

    def __post_init__(self):
        self.K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])
        self.distortion_params = np.array([self.k1, self.k2, self.k3, self.k4])
        self.new_K = self.compute_undistort_intrinsic()
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K,
            self.distortion_params,
            np.eye(3),
            self.new_K,
            (self.width, self.height),
            cv2.CV_32FC1,
        )


# reference: https://github.com/scannetpp/scannetpp/blob/main/dslr/undistort.py
class ScanNetPPDLSRDataset:
    def __init__(self, data_dir: str, output_dir: str = None, undistort=True):
        self.data_dir = data_dir
        self.output_dir = output_dir if output_dir is not None else data_dir
        self.undistort = undistort
        self.image_paths = self.get_image_paths()
        self.mask_paths = self.get_mask_paths()
        self.num_images = len(self.image_paths)
        self.transforms = self.get_transforms() if self.undistort else None
        self.distortion_params = (
            self.get_distortion_params() if self.undistort else None
        )

        self._extrinsics = {}
        self._read_cameras()

    @property
    def image_dir(self):
        return os.path.join(self.data_dir, "resized_images")

    @property
    def mask_dir(self):
        return os.path.join(self.data_dir, "resized_anon_masks")

    @property
    def colmap_path(self):
        return os.path.join(self.data_dir, "colmap")

    @property
    def transforms_path(self):
        return os.path.join(self.data_dir, "nerfstudio", "transforms.json")

    def get_transforms(self):
        return read_json(self.transforms_path)

    def get_image_paths(self):
        if not os.path.exists(self.image_dir):
            return []
        return natsorted(glob(os.path.join(self.image_dir, "*.JPG")))

    def get_mask_paths(self):
        if not os.path.exists(self.mask_dir):
            return []
        return natsorted(glob(os.path.join(self.mask_dir, "*.png")))

    def get_image_path_by_index(self, index: int):
        assert index < len(self.image_paths), "Index out of images range"
        return self.image_paths[index]

    def get_mask_path_by_index(self, index: int):
        assert index < len(self.mask_paths), "Index out of masks range"
        return self.mask_paths[index]

    def get_image_by_index(self, index: int):
        image_path = self.get_image_path_by_index(index)
        return self.get_image_by_path(image_path)

    def get_image_by_path(self, image_path: str):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.undistort:
            image = self.undistort_image(image)
        return image

    def get_mask_by_index(self, index: int):
        mask_path = self.get_mask_path_by_index(index)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if self.undistort:
            mask = self.undistort_mask(mask)
        return mask

    def undistort_image(self, image: np.ndarray):
        map1, map2 = self.distortion_params.map1, self.distortion_params.map2
        return cv2.remap(
            image,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    def undistort_mask(self, mask: np.ndarray):
        height, width = self.distortion_params.height, self.distortion_params.width
        if np.all(mask > 0):
            # No invalid pixels
            return np.zeros((height, width), dtype=np.uint8) + 255

        map1, map2 = self.distortion_params.map1, self.distortion_params.map2
        undistorted_mask = cv2.remap(
            mask,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )
        undistorted_mask[undistorted_mask < 255] = 0
        return undistorted_mask

    def get_distortion_params(self):
        height = int(self.transforms["h"])
        width = int(self.transforms["w"])
        k1 = float(self.transforms["k1"])
        k2 = float(self.transforms["k2"])
        k3 = float(self.transforms["k3"])
        k4 = float(self.transforms["k4"])
        fx = float(self.transforms["fl_x"])
        fy = float(self.transforms["fl_y"])
        cx = float(self.transforms["cx"])
        cy = float(self.transforms["cy"])
        return DistortionParams(height, width, fx, fy, cx, cy, k1, k2, k3, k4)

    def convert_to_video(self, output_path: str):
        # convert images to video with cv2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        first_image = self.get_image_by_index(0)
        height, width, _ = first_image.shape
        video_writer = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
        video_writer.write(first_image)
        for i in range(1, self.num_images):
            video_writer.write(self.get_image_by_index(i))
        video_writer.release()

    def read_cameras(self):
        # check if self._intrinsics and self._extrinsics are empty
        if len(self._extrinsics) == 0:
            self._read_cameras()

        cameras = []
        for image_name, extrinsics_data in self._extrinsics.items():
            extrinsics = extrinsics_data["extrinsics"]
            if self.undistort:
                intrinsics = self.distortion_params.new_K
            else:
                intrinsics = self.distortion_params.K
            camera = SceneKitCamera(
                name=image_name,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                width=self.distortion_params.width,
                height=self.distortion_params.height,
            )
            cameras.append(camera)
        return cameras

    def _read_cameras(self):
        extrinsics_file = os.path.join(self.colmap_path, "images.txt")
        with open(extrinsics_file, "r") as fid:
            while True:
                line = fid.readline()
                if not line:
                    break
                line = line.strip()
                if len(line) > 0 and line[0] != "#":
                    elems = line.split()
                    image_id = int(elems[0])
                    qvec = np.array(tuple(map(float, elems[1:5])))
                    tvec = np.array(tuple(map(float, elems[5:8])))
                    camera_id = int(elems[8])
                    image_name = elems[9]
                    elems = fid.readline().split()
                    extrinsics_matrix = np.eye(4)
                    extrinsics_matrix[:3, :3] = qvec2rotmat(qvec)
                    extrinsics_matrix[:3, 3] = tvec

                    self._extrinsics[image_name] = {
                        "image_id": image_id,
                        "camera_id": camera_id,
                        "extrinsics": extrinsics_matrix,
                    }
