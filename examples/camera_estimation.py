import torch
import hydra
from omegaconf import DictConfig

from pyscenekit import attach_to_log
from pyscenekit.utils.common import log
from pyscenekit.scenekit2d.camera import GeoCalibModel
from pyscenekit.scenekit2d.camera import VPEstimationPriorGravityModel


def geo_calib_model(cfg: DictConfig):
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


def vp_estimation_model(cfg: DictConfig):
    vp_estimator = VPEstimationPriorGravityModel()
    f, vp = vp_estimator(cfg.input)
    log.info(f"Focal length: {f}")
    log.info(f"Vanishing point: {vp}")


@hydra.main(config_path="../configs", config_name="scenekit2d", version_base="1.3")
def main(cfg: DictConfig):
    if cfg.verbose:
        attach_to_log()

    if cfg.camera_estimation.method == "geo_calib":
        geo_calib_model(cfg)
    elif cfg.camera_estimation.method == "vp_estimation_prior_gravity":
        vp_estimation_model(cfg)


if __name__ == "__main__":
    main()
