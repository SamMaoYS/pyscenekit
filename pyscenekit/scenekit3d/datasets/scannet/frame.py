import os
import cv2
import struct
import numpy as np
import zlib
import imageio
import png
from tqdm import tqdm

from pyscenekit.utils.common import log, read_json

COMPRESSION_TYPE_COLOR = {-1: "unknown", 0: "raw", 1: "png", 2: "jpeg"}
COMPRESSION_TYPE_DEPTH = {
    -1: "unknown",
    0: "raw_ushort",
    1: "zlib_ushort",
    2: "occi_ushort",
}


# reference: https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/SensorData.py
class RGBDFrame:
    def load(self, file_handle):
        self.camera_to_world = np.asarray(
            struct.unpack("f" * 16, file_handle.read(16 * 4)), dtype=np.float32
        ).reshape(4, 4)
        self.timestamp_color = struct.unpack("Q", file_handle.read(8))[0]
        self.timestamp_depth = struct.unpack("Q", file_handle.read(8))[0]
        self.color_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.depth_size_bytes = struct.unpack("Q", file_handle.read(8))[0]
        self.color_data = b"".join(
            struct.unpack(
                "c" * self.color_size_bytes, file_handle.read(self.color_size_bytes)
            )
        )
        self.depth_data = b"".join(
            struct.unpack(
                "c" * self.depth_size_bytes, file_handle.read(self.depth_size_bytes)
            )
        )

    def decompress_depth(self, compression_type):
        if compression_type == "zlib_ushort":
            return self.decompress_depth_zlib()
        else:
            raise

    def decompress_depth_zlib(self):
        return zlib.decompress(self.depth_data)

    def decompress_color(self, compression_type):
        if compression_type == "jpeg":
            return self.decompress_color_jpeg()
        else:
            raise

    def decompress_color_jpeg(self):
        return imageio.imread(self.color_data)


# reference: https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/SensorData.py
class SensorData:
    def __init__(self, filename):
        self.version = 4
        self.scene_id = os.path.basename(filename).split(".")[0]
        self.load(filename)

    def load(self, filename):
        with open(filename, "rb") as f:
            version = struct.unpack("I", f.read(4))[0]
            assert self.version == version
            strlen = struct.unpack("Q", f.read(8))[0]
            self.sensor_name = b"".join(struct.unpack("c" * strlen, f.read(strlen)))
            self.intrinsic_color = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.extrinsic_color = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.intrinsic_depth = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.extrinsic_depth = np.asarray(
                struct.unpack("f" * 16, f.read(16 * 4)), dtype=np.float32
            ).reshape(4, 4)
            self.color_compression_type = COMPRESSION_TYPE_COLOR[
                struct.unpack("i", f.read(4))[0]
            ]
            self.depth_compression_type = COMPRESSION_TYPE_DEPTH[
                struct.unpack("i", f.read(4))[0]
            ]
            self.color_width = struct.unpack("I", f.read(4))[0]
            self.color_height = struct.unpack("I", f.read(4))[0]
            self.depth_width = struct.unpack("I", f.read(4))[0]
            self.depth_height = struct.unpack("I", f.read(4))[0]
            self.depth_shift = struct.unpack("f", f.read(4))[0]
            num_frames = struct.unpack("Q", f.read(8))[0]
            self.frames = []
            for i in range(num_frames):
                frame = RGBDFrame()
                frame.load(f)
                self.frames.append(frame)

    def export_depth_images(self, output_path, image_size=None, frame_skip=1):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print(
            "exporting", len(self.frames) // frame_skip, " depth frames to", output_path
        )
        for f in range(0, len(self.frames), frame_skip):
            depth_data = self.frames[f].decompress_depth(self.depth_compression_type)
            depth = np.fromstring(depth_data, dtype=np.uint16).reshape(
                self.depth_height, self.depth_width
            )
            if image_size is not None:
                depth = cv2.resize(
                    depth,
                    (image_size[1], image_size[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            # imageio.imwrite(os.path.join(output_path, str(f) + '.png'), depth)
            with open(
                os.path.join(output_path, str(f) + ".png"), "wb"
            ) as f:  # write 16-bit
                writer = png.Writer(
                    width=depth.shape[1], height=depth.shape[0], bitdepth=16
                )
                depth = depth.reshape(-1, depth.shape[1]).tolist()
                writer.write(f, depth)

    def export_color_images(self, output_path, image_size=None, frame_skip=1):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print(
            "exporting", len(self.frames) // frame_skip, "color frames to", output_path
        )
        for f in range(0, len(self.frames), frame_skip):
            color = self.frames[f].decompress_color(self.color_compression_type)
            if image_size is not None:
                color = cv2.resize(
                    color,
                    (image_size[1], image_size[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            imageio.imwrite(os.path.join(output_path, str(f) + ".jpg"), color)

    def save_mat_to_file(self, matrix, filename):
        with open(filename, "w") as f:
            for line in matrix:
                np.savetxt(f, line[np.newaxis], fmt="%f")

    def export_poses(self, output_path, frame_skip=1):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print(
            "exporting", len(self.frames) // frame_skip, "camera poses to", output_path
        )
        for f in range(0, len(self.frames), frame_skip):
            self.save_mat_to_file(
                self.frames[f].camera_to_world,
                os.path.join(output_path, str(f) + ".txt"),
            )

    def export_intrinsics(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        print("exporting camera intrinsics to", output_path)
        self.save_mat_to_file(
            self.intrinsic_color, os.path.join(output_path, "intrinsic_color.txt")
        )
        self.save_mat_to_file(
            self.extrinsic_color, os.path.join(output_path, "extrinsic_color.txt")
        )
        self.save_mat_to_file(
            self.intrinsic_depth, os.path.join(output_path, "intrinsic_depth.txt")
        )
        self.save_mat_to_file(
            self.extrinsic_depth, os.path.join(output_path, "extrinsic_depth.txt")
        )

    def export_as_hdf5(self, output_path, skip_step=10):
        import h5py

        num_frames = len(self.frames)
        with h5py.File(output_path, "w") as f:
            f.attrs["scene_id"] = self.scene_id
            f.attrs["total_frames"] = num_frames
            f.attrs["color_width"] = self.color_width
            f.attrs["color_height"] = self.color_height
            f.attrs["depth_width"] = self.depth_width
            f.attrs["depth_height"] = self.depth_height
            f.attrs["skip_step"] = skip_step
            f.create_dataset(
                "color_intrinsic",
                data=self.intrinsic_color,
                shape=self.intrinsic_color.shape,
            )
            f.create_dataset(
                "color_extrinsic",
                data=self.extrinsic_color,
                shape=self.extrinsic_color.shape,
            )
            f.create_dataset(
                "depth_intrinsic",
                data=self.intrinsic_depth,
                shape=self.intrinsic_depth.shape,
            )
            f.create_dataset(
                "depth_extrinsic",
                data=self.extrinsic_depth,
                shape=self.extrinsic_depth.shape,
            )

            f_indices = np.arange(0, num_frames, skip_step).astype(np.int32)
            f.create_dataset(
                "frame_indices",
                data=f_indices,
                shape=f_indices.shape,
                compression="gzip",
            )
            f.attrs["num_frames"] = len(f_indices)

            log.info(f"Exporting {len(f_indices)} frames to {output_path}")
            for i in tqdm(f_indices):
                frame = self.frames[i]
                h5group = f.create_group(f"frame_{i}")

                color = frame.decompress_color(self.color_compression_type)
                depth_data = frame.decompress_depth(self.depth_compression_type)
                depth = np.fromstring(depth_data, dtype=np.uint16).reshape(
                    self.depth_height, self.depth_width
                )

                pose = frame.camera_to_world

                h5group.create_dataset(
                    "color", data=color, shape=color.shape, compression="gzip"
                )
                h5group.create_dataset(
                    "depth", data=depth, shape=depth.shape, compression="gzip"
                )
                h5group.create_dataset(
                    "pose", data=pose, shape=pose.shape, compression="gzip"
                )


# reference: https://github.com/scannetpp/scannetpp/blob/main/dslr/undistort.py
class ScanNetFrameDataset:
    def __init__(self, scene_id: str, data_dir: str, output_dir: str = None):
        self.scene_id = scene_id
        self.data_dir = data_dir
        self.output_dir = output_dir if output_dir is not None else data_dir
        self.image_paths = []
        self.num_images = 0
        assert os.path.isfile(
            self.sensfile_path
        ), f"File {self.sensfile_path} not found"
        self.sensor_data = SensorData(self.sensfile_path)

    @property
    def sensfile_path(self):
        return os.path.join(self.data_dir, f"{self.scene_id}.sens")

    @property
    def hdf5_path(self):
        return os.path.join(self.output_dir, f"{self.scene_id}.hdf5")

    @property
    def rgb_folder(self):
        return os.path.join(self.output_dir, "color")

    @property
    def depth_folder(self):
        return os.path.join(self.output_dir, "depth")

    @property
    def pose_folder(self):
        return os.path.join(self.output_dir, "pose")

    @property
    def intrinsics_folder(self):
        return os.path.join(self.output_dir, "intrinsic")

    def extract_rgb(self, output_dir: str = None):
        if output_dir is None:
            output_dir = self.rgb_folder
        os.makedirs(output_dir, exist_ok=True)
        log.info(f"Extracting RGB images to {output_dir}")
        self.sensor_data.export_color_images(output_dir)

    def extract_depth(self, output_dir: str = None):
        if output_dir is None:
            output_dir = self.depth_folder
        os.makedirs(output_dir, exist_ok=True)
        log.info(f"Extracting depth images to {output_dir}")
        self.sensor_data.export_depth_images(output_dir)

    def extract_poses(self, output_dir: str = None):
        if output_dir is None:
            output_dir = self.pose_folder
        os.makedirs(output_dir, exist_ok=True)
        log.info(f"Extracting poses to {output_dir}")
        self.sensor_data.export_poses(output_dir)

    def extract_intrinsics(self, output_dir: str = None):
        if output_dir is None:
            output_dir = self.intrinsics_folder
        os.makedirs(output_dir, exist_ok=True)
        log.info(f"Extracting intrinsics to {output_dir}")
        self.sensor_data.export_intrinsics(output_dir)

    def export_as_hdf5(self):
        log.info(f"Exporting {self.scene_id} to {self.hdf5_path}")
        self.sensor_data.export_as_hdf5(self.hdf5_path)
