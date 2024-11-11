import os
import cv2
import sys
import zlib
import subprocess
import lz4.block

import numpy as np
import imageio as iio
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from dataclasses import dataclass, field

from pyscenekit.utils.common import log, read_json
from pyscenekit.scenekit3d.common import SceneKitCamera
from pyscenekit.scenekit3d.utils import qvec2rotmat


def run_command(cmd: str, verbose=False, exit_on_error=True):
    """Runs a command and returns the output.

    Args:
        cmd: Command to run.
        verbose: If True, logs the output of the command.
    Returns:
        The output of the command if return_output is True, otherwise None.
    """
    out = subprocess.run(cmd, capture_output=not verbose, shell=True, check=False)
    if out.returncode != 0:
        if out.stderr is not None:
            print(out.stderr.decode("utf-8"))
        if exit_on_error:
            sys.exit(1)
    if out.stdout is not None:
        return out.stdout.decode("utf-8")
    return out


# reference: https://github.com/scannetpp/scannetpp/blob/main/iphone/prepare_iphone_data.py
class ScanNetPPiPhoneDataset:
    def __init__(self, data_dir: str, output_dir: str = None, undistort: bool = True):
        self.data_dir = data_dir
        self.output_dir = output_dir if output_dir is not None else data_dir
        self.image_paths = self.get_image_paths()
        self.mask_paths = self.get_mask_paths()
        self.depth_paths = self.get_depth_paths()
        self.num_images = len(self.image_paths)
        self.undistort = undistort

        self._intrinsics = {}
        self._extrinsics = {}
        self._read_cameras()

    @property
    def rgb_path(self):
        return os.path.join(self.data_dir, "rgb.mp4")

    @property
    def depth_path(self):
        return os.path.join(self.data_dir, "depth.bin")

    @property
    def mask_path(self):
        return os.path.join(self.data_dir, "rgb_mask.mkv")

    @property
    def colmap_path(self):
        return os.path.join(self.data_dir, "colmap")

    @property
    def rgb_folder(self):
        return os.path.join(self.output_dir, "rgb")

    @property
    def depth_folder(self):
        return os.path.join(self.output_dir, "depth")

    @property
    def mask_folder(self):
        return os.path.join(self.output_dir, "mask")

    def extract_rgb(self, num_workers: int = 4):
        from pyscenekit.scenekit3d.datasets.multiscan.decoder import DecoderRGB

        frame_indices = list(self._extrinsics.keys())
        frame_indices = [
            int(os.path.basename(frame_index).split("_")[1].split(".")[0])
            for frame_index in frame_indices
        ]

        decoder = DecoderRGB(self.rgb_path)
        decoder.frame_indices = frame_indices
        os.makedirs(self.rgb_folder, exist_ok=True)
        log.info(f"Extracting RGB images to {self.rgb_folder}")
        decoder.export(
            self.rgb_folder,
            format="jpg",
            frame_param={"width": 0, "height": 0},
            num_workers=num_workers,
        )

    def extract_masks(self, num_workers: int = 4):
        from pyscenekit.scenekit3d.datasets.multiscan.decoder import DecoderRGB

        frame_indices = list(self._extrinsics.keys())
        frame_indices = [
            int(os.path.basename(frame_index).split("_")[1].split(".")[0])
            for frame_index in frame_indices
        ]
        decoder = DecoderRGB(self.mask_path)
        decoder.frame_indices = frame_indices
        os.makedirs(self.mask_folder, exist_ok=True)
        log.info(f"Extracting masks to {self.mask_folder}")
        decoder.export(
            self.mask_folder,
            format="png",
            frame_param={"width": 0, "height": 0, "grayscale": True},
            num_workers=num_workers,
        )

    def extract_depth(self):
        # global compression with zlib
        height, width = 192, 256
        frame_indices = list(self._extrinsics.keys())
        frame_indices = [
            int(os.path.basename(frame_index).split("_")[1].split(".")[0])
            for frame_index in frame_indices
        ]
        log.info(f"Extracting depth images to {self.depth_folder}")
        os.makedirs(self.depth_folder, exist_ok=True)

        try:
            with open(self.depth_path, "rb") as infile:
                data = infile.read()
                data = zlib.decompress(data, wbits=-zlib.MAX_WBITS)
                depth = np.frombuffer(data, dtype=np.float32).reshape(-1, height, width)

            for frame_id in tqdm(frame_indices, desc="decode_depth"):
                iio.imwrite(
                    os.path.join(self.depth_folder, f"frame_{frame_id:06}.png"),
                    (depth * 1000).astype(np.uint16),
                )
        # per frame compression with lz4/zlib
        except:
            frame_id = 0
            with open(self.depth_path, "rb") as infile:
                while True:
                    size = infile.read(4)  # 32-bit integer
                    if len(size) == 0:
                        break
                    size = int.from_bytes(size, byteorder="little")
                    if frame_id not in frame_indices:
                        infile.seek(size, 1)
                        frame_id += 1
                        continue

                    # read the whole file
                    data = infile.read(size)
                    try:
                        # try using lz4
                        data = lz4.block.decompress(
                            data, uncompressed_size=height * width * 2
                        )  # UInt16 = 2bytes
                        depth = np.frombuffer(data, dtype=np.uint16).reshape(
                            height, width
                        )
                    except:
                        # try using zlib
                        data = zlib.decompress(data, wbits=-zlib.MAX_WBITS)
                        depth = np.frombuffer(data, dtype=np.float32).reshape(
                            height, width
                        )
                        depth = (depth * 1000).astype(np.uint16)

                    # 6 digit frame id = 277 minute video at 60 fps
                    iio.imwrite(
                        os.path.join(self.depth_folder, f"frame_{frame_id:06}.png"),
                        depth,
                    )
                    frame_id += 1

    def get_image_paths(self):
        if not os.path.exists(self.rgb_folder):
            return []
        return natsorted(glob(os.path.join(self.rgb_folder, "*.jpg")))

    def get_mask_paths(self):
        if not os.path.exists(self.mask_folder):
            return []
        return natsorted(glob(os.path.join(self.mask_folder, "*.png")))

    def get_depth_paths(self):
        if not os.path.exists(self.depth_folder):
            return []
        return natsorted(glob(os.path.join(self.depth_folder, "*.png")))

    def get_image_path_by_index(self, index: int):
        return self.image_paths[index]

    def get_mask_path_by_index(self, index: int):
        return self.mask_paths[index]

    def get_depth_path_by_index(self, index: int):
        return self.depth_paths[index]

    def get_image_by_index(self, index: int):
        image_path = self.get_image_path_by_index(index)
        image = cv2.imread(image_path)
        if self.undistort:
            image_name = os.path.basename(image_path)
            camera_id = self._extrinsics.get(image_name, {}).get("camera_id", None)
            if camera_id is not None:
                image = cv2.remap(
                    image,
                    self._intrinsics[camera_id]["map1"],
                    self._intrinsics[camera_id]["map2"],
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def get_mask_by_index(self, index: int):
        mask_path = self.get_mask_path_by_index(index)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if self.undistort:
            image_name = os.path.basename(mask_path).replace(".png", ".jpg")
            camera_id = self._extrinsics.get(image_name, {}).get("camera_id", None)
            if camera_id is not None:
                mask = cv2.remap(
                    mask,
                    self._intrinsics[camera_id]["map1"],
                    self._intrinsics[camera_id]["map2"],
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=255,
                )
        return mask

    def get_depth_by_index(self, index: int):
        depth_path = self.get_depth_path_by_index(index)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = depth.astype(np.float32) / 1000.0
        return depth

    def read_cameras(self):
        # check if self._intrinsics and self._extrinsics are empty
        if len(self._intrinsics) == 0 or len(self._extrinsics) == 0:
            self._read_cameras()

        cameras = []

        for image_name, extrinsics_data in self._extrinsics.items():
            intrinsics_data = self._intrinsics[extrinsics_data["camera_id"]]
            extrinsics = extrinsics_data["extrinsics"]
            if self.undistort:
                intrinsics = intrinsics_data["new_K"]
            else:
                intrinsics = intrinsics_data["K"]
            camera = SceneKitCamera(
                name=image_name,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                width=intrinsics_data["width"],
                height=intrinsics_data["height"],
            )
            cameras.append(camera)
        return cameras

    def _read_cameras(self):
        intrinsics_file = os.path.join(self.colmap_path, "cameras.txt")
        with open(intrinsics_file, "r") as fid:
            while True:
                line = fid.readline()
                if not line:
                    break
                line = line.strip()
                if len(line) > 0 and line[0] != "#":
                    elems = line.split()
                    camera_id = int(elems[0])
                    model = elems[1]
                    width = int(elems[2])
                    height = int(elems[3])
                    params = np.array(tuple(map(float, elems[4:])))
                    intrinsics_matrix = np.eye(3)
                    intrinsics_matrix[0, 0] = params[0]
                    intrinsics_matrix[1, 1] = params[1]
                    intrinsics_matrix[0, 2] = params[2]
                    intrinsics_matrix[1, 2] = params[3]
                    distortion_params = np.array(tuple(map(float, params[4:])))

                    new_K, _ = cv2.getOptimalNewCameraMatrix(
                        intrinsics_matrix,
                        distortion_params,
                        (width, height),
                        1,
                        (width, height),
                        True,
                    )
                    map1, map2 = cv2.initUndistortRectifyMap(
                        intrinsics_matrix,
                        distortion_params,
                        np.eye(3),
                        new_K,
                        (width, height),
                        cv2.CV_32FC1,
                    )

                    self._intrinsics[camera_id] = {
                        "model": model,
                        "width": width,
                        "height": height,
                        "K": intrinsics_matrix,
                        "distortion_params": distortion_params,
                        "new_K": new_K,
                        "map1": map1,
                        "map2": map2,
                    }

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
                    intrinsics_matrix = self._intrinsics[camera_id]["K"]
                    extrinsics_matrix = np.eye(4)
                    extrinsics_matrix[:3, :3] = qvec2rotmat(qvec)
                    extrinsics_matrix[:3, 3] = tvec

                    self._extrinsics[image_name] = {
                        "image_id": image_id,
                        "camera_id": camera_id,
                        "extrinsics": extrinsics_matrix,
                    }
