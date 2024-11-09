import cv2
import hydra
from omegaconf import DictConfig

from pyscenekit import attach_to_log
from pyscenekit.utils.common import log
from pyscenekit.scenekit2d.depth import DepthEstimationModel


@hydra.main(config_path="../configs", config_name="scenekit2d", version_base="1.3")
def main(cfg: DictConfig):
    if cfg.verbose:
        attach_to_log()

    # select model path based on method
    model_path = cfg.get(cfg.depth_estimation.method, None).model_path

    additional_kwargs = {}
    if cfg.depth_estimation.method == "metric3d":
        additional_kwargs["model_type"] = cfg.depth_estimation.model_type

    depth_estimator = DepthEstimationModel(
        cfg.depth_estimation.method, model_path, **additional_kwargs
    )
    depth_estimator.to(cfg.device)

    if cfg.depth_estimation.method == "metric3d":
        fx = cfg.get("fx")
        fy = cfg.get("fy")
        cx = cfg.get("cx")
        cy = cfg.get("cy")
        depth, _ = depth_estimator(cfg.input, fx=fx, fy=fy, cx=cx, cy=cy)
    else:
        depth, _ = depth_estimator(cfg.input)

    depth = depth_estimator.normalize(depth)
    depth = depth_estimator.colormap(depth, cmap="viridis")

    cv2.imwrite(cfg.output, depth)
    log.info(f"Depth map saved to {cfg.output}")


if __name__ == "__main__":
    main()
