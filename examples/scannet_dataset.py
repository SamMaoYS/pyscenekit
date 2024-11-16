import os
import cv2
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from pyscenekit import attach_to_log
from pyscenekit.utils.common import log
from pyscenekit.scenekit3d.datasets import ScanNetDataset


def export_frame_dataset(cfg: DictConfig):
    dataset = ScanNetDataset(cfg.scannet.data_dir)
    scenes_ids = dataset.scenes_ids
    scene_idx = int(cfg.scene_idx)
    scene_id = scenes_ids[scene_idx]
    log.info(f"Exporting frame dataset for scene {scene_id}")
    dataset.set_scene_id(scene_id)
    dataset.frame_dataset.export_as_hdf5()


@hydra.main(config_path="../configs", config_name="scenekit3d", version_base="1.3")
def main(cfg: DictConfig):
    if cfg.verbose:
        attach_to_log()

    export_frame_dataset(cfg)


if __name__ == "__main__":
    main()
