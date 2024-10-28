import os
import glob

import cv2
import hydra
import torch
import numpy as np
from omegaconf import DictConfig

from pyscenekit import attach_to_log
from pyscenekit.utils.common import log
from pyscenekit.scenekit3d.reconstruction import MultiViewReconstructionModel
from pyscenekit.scenekit3d.visualization import SceneKitRenderer


@hydra.main(config_path="../configs", config_name="scenekit3d", version_base="1.3")
def main(cfg: DictConfig):
    if cfg.verbose:
        attach_to_log()

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

    if cfg.multiview_reconstruction.export_point_cloud:
        result.export_pcd(os.path.splitext(cfg.output)[0] + ".ply")

    # visualize point clouds
    renderer = SceneKitRenderer(cfg.visualization.method)

    point_cloud_list = result.point_cloud_list
    for point_cloud in point_cloud_list:
        renderer.add_geometry(point_cloud)
    if cfg.visualization.interactive:
        renderer.render(interactive=True)
    else:
        rgb, depth = renderer.render(interactive=False)
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255.0).astype(np.uint8)

        output_image_path = os.path.splitext(cfg.output)[0] + ".jpg"
        cv2.imwrite(output_image_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        log.info(f"Visualization saved to {output_image_path}")


if __name__ == "__main__":
    main()
