import os
import cv2
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from pyscenekit import attach_to_log
from pyscenekit.utils.common import log
from pyscenekit.scenekit3d.datasets import ScanNetPPDataset


def test_dlsr_dataset(dataset: ScanNetPPDataset, output_dir: str):
    scenes_ids = dataset.scenes_ids
    log.info(f"Saving undistorted images for {len(scenes_ids)} scenes")
    for scene_id in tqdm(scenes_ids):
        # check if output image already exists
        output_path = os.path.join(output_dir, f"{scene_id}.jpg")
        if os.path.isfile(output_path):
            continue
        dataset.set_scene_id(scene_id)
        undistorted_image = dataset.dlsr_dataset.get_image_by_index(0)
        # save image by scene_id
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(output_path, undistorted_image)

    # save all images of the scene 6b40d1a939
    scene_id = "6b40d1a939"
    dataset.set_scene_id(scene_id)
    image_paths = dataset.dlsr_dataset.image_paths
    output_image_dir = os.path.join(output_dir, scene_id)
    os.makedirs(output_image_dir, exist_ok=True)
    log.info(f"Saving undistorted images for scene {scene_id}")
    for i in tqdm(range(len(image_paths))):
        image_filename = os.path.basename(image_paths[i])
        output_path = os.path.join(output_image_dir, image_filename)
        if os.path.isfile(output_path):
            continue
        undistorted_image = dataset.dlsr_dataset.get_image_by_index(i)
        cv2.imwrite(output_path, undistorted_image)


@hydra.main(config_path="../configs", config_name="scenekit3d", version_base="1.3")
def main(cfg: DictConfig):
    if cfg.verbose:
        attach_to_log()

    dataset = ScanNetPPDataset(cfg.scannetpp.data_dir)
    output_dir = (
        os.path.dirname(cfg.output) if os.path.isfile(cfg.output) else cfg.output
    )
    test_dlsr_dataset(dataset, output_dir)


if __name__ == "__main__":
    main()
