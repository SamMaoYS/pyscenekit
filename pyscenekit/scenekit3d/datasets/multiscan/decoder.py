import os
import zlib
import json
import numpy as np
import multiprocessing
from enum import Enum

from pyscenekit.scenekit3d.datasets.multiscan.utils import Camera, get_suffix
from pyscenekit.scenekit3d.datasets.multiscan.export import export_frames, export_cameras
from pyscenekit.utils.common import log


class DecoderType(Enum):
    RGB = "rgb"
    DEPTH = "depth"
    CONFIDENCE = "confidence"
    CAMERA = "camera"


_file_types = {
    "mp4": DecoderType.RGB,
    "depth.zlib": DecoderType.DEPTH,
    "confidence.zlib": DecoderType.CONFIDENCE,
    "jsonl": DecoderType.CAMERA,
}


class DecoderBase:
    def __init__(self, input: str, *args, **kwargs):
        self.input = input
        self.debug = kwargs.get("debug", False)
        self.frame_indices = []
        self.num_workers = 0
        self.chunk_size = 50
        self.input_dtype = None
        self.output_dtype = None
        self.width = 0
        self.height = 0
        self.pixel_size = self.get_pixel_size()
        self.frame_size = self.get_frame_size()

    def get_pixel_size(self):
        self.pixel_size = np.dtype(self.input_dtype).itemsize
        return self.pixel_size

    def get_frame_size(self):
        self.get_pixel_size()
        self.frame_size = self.width * self.height * self.pixel_size
        return self.frame_size

    def total_frames(self):
        return -1

    def set_frame_indices(self, start: int, stop: int, step=1):
        total_frames = self.total_frames()
        start_frame = min(start, total_frames - 1)
        assert 0 <= start_frame < total_frames, "Start frame index out of range"
        stop_frame = stop if 0 < stop <= total_frames else total_frames
        self.frame_indices = np.arange(start_frame, stop_frame, step)
        return self.frame_indices

    def export(self, output: str, format: str, frame_param: dict, num_workers: int = 0):
        log.info(f"Decoding stream {self.input} to {output}")
        if len(self.frame_indices) == 0:
            raise "Please set up the export frame indices first"
        os.makedirs(output, exist_ok=True)
        total_workers = multiprocessing.cpu_count()
        self.num_workers = (
            num_workers if 0 < num_workers < total_workers else total_workers
        )

        frame_param["dtype"] = self.output_dtype
        if not frame_param["width"]:
            frame_param["width"] = self.width
        if not frame_param["height"]:
            frame_param["height"] = self.height
        chunk_size = min(self.chunk_size, len(self.frame_indices))
        num_chunks = len(self.frame_indices) // chunk_size
        frame_indices_chunks = np.array_split(self.frame_indices, num_chunks)

        for i, frame_indices in enumerate(frame_indices_chunks):
            log.info(f"Processing chunk {i} / {num_chunks}")
            self._export_impl(output, format, frame_param, frame_indices)

    def _export_impl(
        self, output: str, format: str, frame_param: dict, frame_indices: list
    ):
        pass


class DecoderRGB(DecoderBase):
    def __init__(self, input: str, *args, **kwargs):
        super().__init__(input, *args, **kwargs)
        from decord import VideoReader

        self.video_reader = VideoReader(input)
        tmp_frame = self.video_reader[0].asnumpy()
        self.input_dtype = tmp_frame.dtype
        self.output_dtype = tmp_frame.dtype
        self.width = tmp_frame[1]
        self.height = tmp_frame[0]
        self.pixel_size = self.get_pixel_size()
        self.frame_size = self.get_frame_size()

    def total_frames(self):
        return len(self.video_reader)

    def _export_impl(
        self, output: str, format: str, frame_param: dict, frame_indices: list
    ):
        frames = self.video_reader.get_batch(frame_indices).asnumpy()
        export_frames(
            frames, output, frame_param, frame_indices, format, self.num_workers
        )


class DecoderDepth(DecoderBase):
    def __init__(self, input: str, *args, **kwargs):
        super().__init__(input, *args, **kwargs)
        self._uncompressed_input = self._uncompress(kwargs.get("tmp_dir", "tmp"))
        self.input_dtype = np.float16
        self.output_dtype = np.uint16
        self.pixel_scale = 1000.0
        self.width = 256
        self.height = 192
        self.pixel_size = self.get_pixel_size()
        self.frame_size = self.get_frame_size()

    def _uncompress(self, outupt_dir):
        d = zlib.decompressobj(-zlib.MAX_WBITS)
        chuck_size = 4096 * 4
        tempfile = os.path.basename(self.input) + ".tmp"
        os.makedirs(outupt_dir, exist_ok=True)
        output_filepath = os.path.join(outupt_dir, tempfile)

        tmp = open(output_filepath, "wb+")
        with open(self.input, "rb") as f:
            buffer = f.read(chuck_size)
            while buffer:
                outstr = d.decompress(buffer)
                tmp.write(outstr)
                buffer = f.read(chuck_size)

            tmp.write(d.flush())
        tmp.close()
        return output_filepath

    def total_frames(self):
        file_size = os.stat(self._uncompressed_input).st_size
        return int(file_size / self.frame_size)

    def get_batch_frames(self, frame_indices: list):
        frames = []
        for i in frame_indices:
            fp = np.memmap(
                self._uncompressed_input,
                dtype=self.input_dtype,
                mode="r",
                shape=(self.height, self.width),
                offset=self.frame_size * i,
            ).copy()
            if self.pixel_scale != 1:
                fp *= self.pixel_scale
            if self.output_dtype == self.input_dtype:
                fp = fp.astype(self.output_dtype)
            frames.append(fp)
        return np.asarray(frames)

    def _export_impl(
        self, output: str, format: str, frame_param: dict, frame_indices: list
    ):
        frames = self.get_batch_frames(frame_indices)
        export_frames(
            frames, output, frame_param, frame_indices, format, self.num_workers
        )


class DecoderConfidence(DecoderDepth):
    def __init__(self, input: str, *args, **kwargs):
        super().__init__(input, *args, **kwargs)
        self.input_dtype = np.uint8
        self.output_dtype = np.uint8
        self.pixel_scale = 1
        self.pixel_size = self.get_pixel_size()
        self.frame_size = self.get_frame_size()


class DecoderCamera(DecoderBase):
    def __init__(self, input: str, *args, **kwargs):
        super().__init__(input, *args, **kwargs)
        self._lines = []
        self.width = 1920
        self.height = 1440

    def total_frames(self):
        with open(self.input, "r") as fp:
            self._lines = [line for line in fp.readlines() if line.strip()]
        return len(self._lines)

    def get_cameras(self, frame_indices: list):
        cameras = []
        for i in frame_indices:
            camera = Camera(json.loads(self._lines[i]), name=str(i + 1))
            cameras.append(camera)
        cameras = np.asarray(cameras)
        return cameras

    def _export_impl(
        self, output: str, format: str, frame_param: dict, frame_indices: list
    ):
        cameras = self.get_cameras(frame_indices)
        export_cameras(
            cameras, output, frame_param, frame_indices, format, self.num_workers
        )


class Decoder(object):
    def __new__(cls, input: str, *args, **kwargs):
        suffix = get_suffix(input)
        try:
            decoder_type = _file_types[suffix]
        except:
            raise NotImplementedError(f"file with suffix {suffix} is not supported")

        if decoder_type == DecoderType.RGB:
            return DecoderRGB(input, *args, **kwargs)
        elif decoder_type == DecoderType.DEPTH:
            return DecoderDepth(input, *args, **kwargs)
        elif decoder_type == DecoderType.CONFIDENCE:
            return DecoderConfidence(input, *args, **kwargs)
        elif decoder_type == DecoderType.CAMERA:
            return DecoderCamera(input, *args, **kwargs)
        else:
            raise NotImplementedError(
                f"Decoder with type {decoder_type.value} is not supported"
            )
