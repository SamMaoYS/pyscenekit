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
    MultiViewReconstructionModel,
    MultiViewReconstructionOutput,
)
from pyscenekit.scenekit3d.visualization import SceneKitRenderer


def multiview_reconstruction_pipeline(cfg: DictConfig):
    # select model path based on method
    model_path = cfg.get(cfg.multiview_reconstruction.method).model_path

    multiview_reconstructor = MultiViewReconstructionModel(
        cfg.multiview_reconstruction.method,
        model_path,
    )
    multiview_reconstructor.to(cfg.device)

    image_list = glob.glob(cfg.multiview_reconstruction.image_list)

    log.info(
        f"Running multi-view reconstruction with {len(image_list)} images: {image_list}"
    )

    result = multiview_reconstructor(image_list)
    torch.save(result.to_dict(), cfg.output)
    log.info(f"Multi-view reconstruction saved to {cfg.output}")
    return result


def visualize_point_cloud(cfg: DictConfig, result: MultiViewReconstructionOutput):
    # visualize point clouds
    renderer = SceneKitRenderer(cfg.visualization.method)
    renderer.set_world_up(np.array([0, -1, 0]))

    point_cloud_list = result.point_cloud_list
    for point_cloud in point_cloud_list:
        renderer.add_geometry(point_cloud)
    renderer.set_camera_pose_by_angle()
    if cfg.visualization.interactive:
        renderer.render(interactive=True)
    else:
        rgb, depth = renderer.render(interactive=False)
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255.0).astype(np.uint8)

        output_image_path = os.path.splitext(cfg.output)[0] + ".jpg"
        cv2.imwrite(output_image_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        log.info(f"Visualization saved to {output_image_path}")


@hydra.main(config_path="../configs", config_name="scenekit3d", version_base="1.3")
def main(cfg: DictConfig):
    if cfg.verbose:
        attach_to_log()

    if not os.path.exists(cfg.output):
        result = multiview_reconstruction_pipeline(cfg)
    else:
        result = MultiViewReconstructionOutput.from_dict(torch.load(cfg.output))

    # save point cloud and mesh outputs
    if cfg.multiview_reconstruction.export_point_cloud:
        result.export_pcd(os.path.splitext(cfg.output)[0] + ".ply")

    if cfg.multiview_reconstruction.export_mesh:
        result.export_mesh(os.path.splitext(cfg.output)[0] + ".obj")

    # visualize point clouds
    visualize_point_cloud(cfg, result)


if __name__ == "__main__":
    main()
