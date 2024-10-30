import os
import cv2
import numpy as np
from glob import glob
from natsort import natsorted
from dataclasses import dataclass, field

from pyscenekit.utils.common import read_json


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
    def __init__(self, data_dir: str, undistort=True):
        self.data_dir = data_dir
        self.undistort = undistort
        self.image_paths = self.get_image_paths()
        self.num_images = len(self.image_paths)
        self.transforms = self.get_transforms() if self.undistort else None
        self.distortion_params = (
            self.get_distortion_params() if self.undistort else None
        )

    @property
    def image_dir(self):
        return os.path.join(self.data_dir, "resized_images")

    @property
    def transforms_path(self):
        return os.path.join(self.data_dir, "nerfstudio", "transforms.json")

    def get_transforms(self):
        return read_json(self.transforms_path)

    def get_image_paths(self):
        return natsorted(glob(os.path.join(self.image_dir, "*.JPG")))

    def get_image_path_by_index(self, index: int):
        assert index < len(self.image_paths), "Index out of images range"
        return self.image_paths[index]

    def get_image_by_index(self, index: int):
        image_path = self.get_image_path_by_index(index)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.undistort:
            image = self.undistort_image(image)
        return image

    def undistort_image(self, image: np.ndarray):
        map1, map2 = self.distortion_params.map1, self.distortion_params.map2
        return cv2.remap(
            image,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

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
