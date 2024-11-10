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
    def __init__(self, data_dir: str, output_dir: str=None):
        self.data_dir = data_dir
        self.output_dir = output_dir if output_dir is not None else data_dir
        self.image_paths = self.get_image_paths()
        self.mask_paths = self.get_mask_paths()
        self.depth_paths = self.get_depth_paths()
        self.num_images = len(self.image_paths)

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
    def rgb_folder(self):
        return os.path.join(self.output_dir, "rgb")

    @property
    def depth_folder(self):
        return os.path.join(self.output_dir, "depth")

    @property
    def mask_folder(self):
        return os.path.join(self.output_dir, "mask")

    def extract_rgb(self):
        os.makedirs(self.rgb_folder, exist_ok=True)
        log.info(f"Extracting RGB images to {self.rgb_folder}")
        output_path = os.path.join(self.rgb_folder, "frame_%06d.jpg")
        cmd = f"ffmpeg -y -i {self.rgb_path} -start_number 0 -q:v 1 {output_path}"
        run_command(cmd, verbose=False)

    def extract_masks(self):
        os.makedirs(self.mask_folder, exist_ok=True)
        log.info(f"Extracting masks to {self.mask_folder}")
        output_path = os.path.join(self.mask_folder, "frame_%06d.png")
        cmd = f"ffmpeg -y -i {self.mask_path} -pix_fmt gray -start_number 0 {output_path}"
        run_command(cmd, verbose=False)

    def extract_depth(self):
        # global compression with zlib
        height, width = 192, 256
        sample_rate = 1
        log.info(f"Extracting depth images to {self.depth_folder}")
        os.makedirs(self.depth_folder, exist_ok=True)

        try:
            with open(self.depth_path, "rb") as infile:
                data = infile.read()
                data = zlib.decompress(data, wbits=-zlib.MAX_WBITS)
                depth = np.frombuffer(data, dtype=np.float32).reshape(-1, height, width)

            for frame_id in tqdm(
                range(0, depth.shape[0], sample_rate), desc="decode_depth"
            ):
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
                    if frame_id % sample_rate != 0:
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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def get_mask_by_index(self, index: int):
        mask_path = self.get_mask_path_by_index(index)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        return mask

    def get_depth_by_index(self, index: int):
        depth_path = self.get_depth_path_by_index(index)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        depth = depth.astype(np.float32) / 1000.0
        return depth
