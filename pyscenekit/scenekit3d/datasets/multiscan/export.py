from __future__ import annotations

import os
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from functools import partial
from tqdm.contrib.concurrent import process_map


@dataclass
class ExportFrameParam:
    width: int
    height: int
    dtype: np.dtype
    scale_str: str = ""
    crop_str: str = ""
    scale: list[int] = field(init=False)
    crop: list[int] = field(init=False)

    def __post_init__(self):
        if self.scale_str:
            self.scale = [int(el) for el in self.scale_str.split("x")]
        if self.crop_str:
            self.crop = [int(el) for el in self.crop_str.split(",")]


def get_export_frame_param(frame_param: dict) -> ExportFrameParam:
    return ExportFrameParam(
        width=frame_param["width"],
        height=frame_param["height"],
        dtype=frame_param["dtype"],
        scale_str=frame_param["scale"],
        crop_str=frame_param["crop"],
    )


def export_frame(
    frame: tuple, output_dir: str = None, frame_param: dict = None, format: str = None
):
    efp = get_export_frame_param(frame_param)
    frame_data, frame_index = frame
    img = Image.fromarray(frame_data.astype(efp.dtype))
    if hasattr(efp, "scale"):
        img = img.resize(efp.scale)
    if hasattr(efp, "crop"):
        img = img.crop((efp.crop[0], efp.crop[1], efp.crop[2], efp.crop[3]))
    os.makedirs(output_dir, exist_ok=True)
    img.save(os.path.join(output_dir, f"{frame_index:08d}.{format}"))


def export_camera(
    camera: tuple, output_dir: str = None, frame_param: dict = None, format: str = None
):
    efp = get_export_frame_param(frame_param)
    camera_data, camera_index = camera
    if hasattr(efp, "scale"):
        camera_data.scale(efp.scale[0] / efp.width, efp.scale[1] / efp.height)
    if hasattr(efp, "crop"):
        camera_data.crop(efp.crop[0], efp.crop[1], efp.crop[2], efp.crop[3])
    output_path = os.path.join(output_dir, f"{camera_index:08d}.{format}")
    camera_data.export(output_path)


def export_frames(
    frames: list,
    output_dir: str,
    frame_param: dict,
    frame_indices: list,
    format: str,
    num_workers: int,
):
    export_data(
        export_frame,
        frames,
        output_dir,
        frame_param,
        frame_indices,
        format,
        num_workers,
    )


def export_cameras(
    cameras: list,
    output_dir: str,
    frame_param: dict,
    frame_indices: list,
    format: str,
    num_workers: int,
):
    export_data(
        export_camera,
        cameras,
        output_dir,
        frame_param,
        frame_indices,
        format,
        num_workers,
    )


def export_data(
    func,
    data: list,
    output_dir: str,
    frame_param: dict,
    frame_indices: list,
    format: str,
    num_workers: int,
):
    if num_workers != 1:
        process_map(
            partial(
                func, output_dir=output_dir, frame_param=frame_param, format=format
            ),
            zip(data, frame_indices + 1),
            disable=True,
            max_workers=num_workers,
        )
    else:
        for data_i in zip(data, frame_indices + 1):
            func(data_i, output_dir, frame_param, format)
