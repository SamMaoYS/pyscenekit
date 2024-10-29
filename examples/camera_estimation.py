import torch
import hydra
from omegaconf import DictConfig

from pyscenekit import attach_to_log
from pyscenekit.utils.common import log
from pyscenekit.scenekit2d.camera import GeoCalibModel


@hydra.main(config_path="../configs", config_name="scenekit2d", version_base="1.3")
def main(cfg: DictConfig):
    if cfg.verbose:
        attach_to_log()

    # select model path based on method
    model_path = cfg.get(cfg.camera_estimation.method, None).model_path
    geo_calibrator = GeoCalibModel(model_path)
    geo_calibrator.to(cfg.device)

    result = geo_calibrator(cfg.input)
    geo_calibrator.print_calibration(result)

    gravity = result["gravity"].squeeze().cpu().numpy()
    log.info(f"Gravity: {gravity}")
    torch.save(result, cfg.output)

    log.info(f"Camera parameters saved to {cfg.output}")


if __name__ == "__main__":
    main()
