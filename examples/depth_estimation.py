import cv2
import hydra
from omegaconf import DictConfig

from pyscenekit import attach_to_log
from pyscenekit.utils.common import log
from pyscenekit.scenekit2d.depth import MidasDepthEstimation


@hydra.main(config_path="../configs", config_name="scenekit2d", version_base="1.3")
def main(cfg: DictConfig):
    if cfg.verbose:
        attach_to_log()

    depth_estimator = MidasDepthEstimation(cfg.midas.model_path)
    depth_estimator.to(cfg.device)

    image = cv2.imread(cfg.input)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    depth = depth_estimator(image)
    depth = depth_estimator.normalize(depth)
    depth = depth_estimator.colormap(depth, cmap="jet")

    cv2.imwrite(cfg.output, depth)
    log.info(f"Depth map saved to {cfg.output}")


if __name__ == "__main__":
    main()
