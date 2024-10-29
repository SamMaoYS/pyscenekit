import os
import glob

import cv2
import hydra
import torch
import numpy as np
from omegaconf import DictConfig

from pyscenekit import attach_to_log
from pyscenekit.utils.common import log
from pyscenekit.scenekit3d.reconstruction import (
    SingleViewReconstructionModel,
    SingleViewReconstructionOutput,
)


def multiview_reconstruction_pipeline(cfg: DictConfig):
    # select model path based on method
    model_path = cfg.get(cfg.singleview_reconstruction.method).model_path

    singleview_reconstructor = SingleViewReconstructionModel(
        cfg.singleview_reconstruction.method,
        model_path,
    )
    singleview_reconstructor.to(cfg.device)

    log.info(f"Running single-view reconstruction with image: {cfg.input}")

    result = singleview_reconstructor(cfg.input)
    torch.save(result.to_dict(), cfg.output)
    log.info(f"Single-view reconstruction saved to {cfg.output}")
    return result


@hydra.main(config_path="../configs", config_name="scenekit3d", version_base="1.3")
def main(cfg: DictConfig):
    if cfg.verbose:
        attach_to_log()

    if not os.path.exists(cfg.output):
        result = multiview_reconstruction_pipeline(cfg)
    else:
        result = SingleViewReconstructionOutput.from_dict(torch.load(cfg.output))

    # save point cloud and mesh outputs
    if cfg.singleview_reconstruction.export_point_cloud:
        result.export_pcd(os.path.splitext(cfg.output)[0] + ".ply")

    if cfg.singleview_reconstruction.export_mesh:
        result.export_mesh(os.path.splitext(cfg.output)[0] + ".obj")


if __name__ == "__main__":
    main()
