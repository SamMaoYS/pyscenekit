import torch
import numpy as np

import trimesh
from mini_dust3r.viz import pts3d_to_trimesh

from pyscenekit.scenekit3d.common import (
    SceneKitCamera,
    SceneKitPointCloud,
    SceneKitMesh,
)
from pyscenekit.scenekit3d.reconstruction.base import (
    SingleViewReconstructionModel,
    SingleViewReconstructionOutput,
)
from pyscenekit.scenekit3d.reconstruction.modules.moge import MoGeModel


class MoGeReconstruction(SingleViewReconstructionModel):
    """
    MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision

    Authors: Ruicheng Wang, Sicheng Xu, Cassie Dai, Jianfeng Xiang, Yu Deng, Xin Tong, Jiaolong Yang

    https://github.com/microsoft/moge

    @misc{wang2024moge,
        title={MoGe: Unlocking Accurate Monocular Geometry Estimation for Open-Domain Images with Optimal Training Supervision},
        author={Wang, Ruicheng and Xu, Sicheng and Dai, Cassie and Xiang, Jianfeng and Deng, Yu and Tong, Xin and Yang, Jiaolong},
        year={2024},
        eprint={2410.19115},
        archivePrefix={arXiv},
        primaryClass={cs.CV},
        url={https://arxiv.org/abs/2410.19115},
    }
    """

    def __init__(self, model_path: str = None):
        super().__init__(model_path)
        if self.model_path is None:
            self.model_path = "Ruicheng/moge-vitl"

        self.load_model()

    def load_model(self):
        self.model = MoGeModel.from_pretrained(self.model_path)

    def _predict(self) -> np.ndarray:
        self.to(self.device)
        input_image = self.input.image
        image = input_image.to_tensor().to(self.device)
        output = self.model.infer(image)

        points = output["points"].cpu().numpy()
        mask = output["mask"].cpu().numpy()
        depth = output["depth"].cpu().numpy()
        intrinsics = output["intrinsics"].cpu().numpy()

        output = SingleViewReconstructionOutput(
            color=input_image.image,
            depth=depth,
            mask=mask,
            camera=SceneKitCamera(intrinsics=intrinsics),
            point_cloud=SceneKitPointCloud.from_vertices(
                vertices=points.reshape(-1, 3), colors=input_image.image.reshape(-1, 3)
            ),
            mesh=SceneKitMesh(
                trimesh.Trimesh(**pts3d_to_trimesh(input_image.image, points, mask))
            ),
        )
        return output

    def to(self, device: str):
        self.device = torch.device(device)
        self.model.to(self.device)
