import os
import json
import numpy as np
from dataclasses import dataclass, field


def write_json(data, filepath, indent=None):
    folder_path = os.path.dirname(filepath)
    if os.path.isdir(folder_path):
        os.makedirs(folder_path, exist_ok=True)

    with open(filepath, "w+") as fp:
        json.dump(data, fp, indent=indent)


def get_suffix(filepath, delimiter=".", last=False):
    filename = os.path.basename(filepath)
    tmp_el = filename.split(delimiter)
    suffix = tmp_el[-1] if last else delimiter.join(tmp_el[1:])
    return suffix


@dataclass
class Camera:
    param_dict: dict
    name: str = "camera"
    intrinsic: np.ndarray = field(init=False)
    extrinsic: np.ndarray = field(init=False)

    def __post_init__(self):
        intrinsic = self.param_dict.get("intrinsics", None)
        self.intrinsic = np.reshape(intrinsic, (3, 3), order="F")

        transform = self.param_dict.get("transform", None)
        transform = np.reshape(transform, (4, 4), order="F")
        transform = np.matmul(transform, np.diag([1, -1, -1, 1]))
        transform = transform / transform[3, 3]
        self.extrinsic = np.linalg.inv(transform)

    def scale(self, x: float, y: float):
        scale = np.diag([x, y, 1.0])
        self.intrinsic = np.dot(scale, self.intrinsic)

    def crop(self, left: int, top: int, right: int, bottom: int):
        self.intrinsic[0, 2] -= (left + right) / 2
        self.intrinsic[1, 2] -= (top + bottom) / 2

    def export(self, filepath: str):
        intrinsic = self.intrinsic.flatten(order="F")
        extrinsic = self.extrinsic.flatten(order="F")
        camera_dict = {
            "name": self.name,
            "intrinsic": intrinsic.tolist(),
            "extrinsic": extrinsic.tolist(),
        }
        write_json(camera_dict, filepath)
