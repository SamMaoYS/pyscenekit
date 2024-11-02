import sys
import math
import random
import torch
import torch.nn as nn
import numpy as np
import numpy.linalg as LA
import torch.nn.functional as F
from pyscenekit.scenekit2d.camera.modules.vp_houghtransform_gaussiansphere.dgcn import (
    DGCN,
)
from pyscenekit.scenekit2d.camera.modules.vp_houghtransform_gaussiansphere.ht.ht_cuda import (
    HT_CUDA,
)
from pyscenekit.scenekit2d.camera.modules.vp_houghtransform_gaussiansphere.iht.iht_cuda import (
    IHT_CUDA,
)
from pyscenekit.scenekit2d.camera.modules.vp_houghtransform_gaussiansphere.sphere.sphere_cuda import (
    SPHERE_CUDA,
)
from pyscenekit.scenekit2d.camera.modules.vp_houghtransform_gaussiansphere.convs import (
    HT_CONV,
    SPHERE_CONV,
)


class VanishingNet(nn.Module):
    def __init__(self, backbone, vote_ht_dict, vote_sphere_dict):
        super().__init__()
        self.backbone = backbone
        self.bn = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)

        self.ht = HT_CUDA(vote_mapping_dict=vote_ht_dict)
        self.iht = IHT_CUDA(vote_mapping_dict=vote_ht_dict)
        self.sphere = SPHERE_CUDA(vote_mapping_dict=vote_sphere_dict)

        self.ht_conv = HT_CONV(inplanes=128, outplanes=128)
        self.sphere_conv = SPHERE_CONV(inplanes=128, outplanes=M.num_channels)

        self.hsn = DGCN(
            nf=[
                M.num_channels,
                M.num_channels,
                M.num_channels,
                M.num_channels,
                M.num_channels,
            ],
            num_nodes=C.io.num_nodes,
            num_neighbors=C.io.num_neighbors,
        )

    def forward(self, input_dict):
        image = input_dict["image"]
        x = self.backbone(image)[0]
        x = self.bn(x)
        x = self.relu(x)

        x = self.ht(x)
        x = self.ht_conv(x)
        x = self.sphere(x)
        x = self.sphere_conv(x)
        x = self.hsn(x)

        return {
            "prediction": x.sigmoid().reshape(-1, C.io.num_nodes),
        }
