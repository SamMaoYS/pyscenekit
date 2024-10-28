import cv2
import hydra
from omegaconf import DictConfig

from pyscenekit import attach_to_log
from pyscenekit.utils.common import log
from pyscenekit.scenekit2d.normal import NormalEstimationModel


@hydra.main(config_path="../configs", config_name="scenekit2d", version_base="1.3")
def main(cfg: DictConfig):
    if cfg.verbose:
        attach_to_log()

    # select model path based on method
    model_path = cfg.get(cfg.normal_estimation.method).model_path
    additional_kwargs = {}
    if cfg.normal_estimation.method == "dsine":
        efficientnet_path = cfg.get(cfg.normal_estimation.method).efficientnet_path
        additional_kwargs["efficientnet_path"] = efficientnet_path

    normal_estimator = NormalEstimationModel(
        cfg.normal_estimation.method,
        model_path,
        **additional_kwargs,
    )
    normal_estimator.to(cfg.device)

    normal = normal_estimator(cfg.input)
    normal = normal_estimator.to_rgb(normal)
    cv2.imwrite(cfg.output, cv2.cvtColor(normal, cv2.COLOR_RGB2BGR))
    log.info(f"Normal map saved to {cfg.output}")


if __name__ == "__main__":
    main()
