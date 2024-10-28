import cv2
import hydra
from omegaconf import DictConfig

from pyscenekit import attach_to_log
from pyscenekit.utils.common import log
from pyscenekit.scenekit2d.segmentation import SemanticSegmentationModel


@hydra.main(config_path="../configs", config_name="scenekit2d", version_base="1.3")
def main(cfg: DictConfig):
    if cfg.verbose:
        attach_to_log()

    # select model path based on method
    model_path = cfg.get(cfg.image_segmentation.method).model_path
    semantic_model = SemanticSegmentationModel(
        cfg.image_segmentation.method, model_path
    )
    semantic_model.to(cfg.device)

    semantic_image = semantic_model(cfg.input)
    semantic_image = semantic_model.semantic_colorize(semantic_image)

    cv2.imwrite(cfg.output, cv2.cvtColor(semantic_image, cv2.COLOR_RGB2BGR))
    log.info(f"Semantic image saved to {cfg.output}")


if __name__ == "__main__":
    main()
