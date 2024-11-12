import os
import cv2
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from functools import partial
from tqdm.contrib.concurrent import process_map

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
        undistorted_image = cv2.cvtColor(undistorted_image, cv2.COLOR_RGB2BGR)
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
        undistorted_image = cv2.cvtColor(undistorted_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, undistorted_image)


def export_iphone_dataset(cfg: DictConfig):
    dataset = ScanNetPPDataset(cfg.scannetpp.data_dir)
    scenes_ids = dataset.scenes_ids
    scene_idx = int(cfg.scene_idx)
    scene_id = scenes_ids[scene_idx]
    log.info(f"Exporting iPhone dataset for scene {scene_id}")
    dataset.set_scene_id(scene_id)
    dataset.iphone_dataset.extract_rgb(num_workers=2)
    dataset.iphone_dataset.extract_masks(num_workers=2)
    # dataset.iphone_dataset.extract_depth()


def render_iphone_depth(cfg: DictConfig):
    dataset = ScanNetPPDataset(cfg.scannetpp.data_dir)
    dataset.set_scene_id(cfg.scene_id)
    cameras = dataset.iphone_dataset.read_cameras()

    mesh_dataset = dataset.mesh_dataset
    mesh_dataset.set_cameras(cameras)
    mesh_dataset.export_all_depth(
        output_dir=os.path.join(dataset.iphone_dataset.output_dir, "render_depth")
    )


def unproject_render_depth(cfg: DictConfig):
    import open3d as o3d
    from pyscenekit.scenekit3d.common import SceneKitStructuredPointCloud

    dataset = ScanNetPPDataset(cfg.scannetpp.data_dir)
    dataset.set_scene_id(cfg.scene_id)
    cameras = dataset.iphone_dataset.read_cameras()

    rgb_dir = os.path.join(dataset.iphone_dataset.output_dir, "rgb")
    depth_dir = os.path.join(dataset.iphone_dataset.output_dir, "render_depth")
    output_dir = os.path.join(dataset.iphone_dataset.output_dir, "point_cloud")
    os.makedirs(output_dir, exist_ok=True)
    for i in tqdm(range(len(cameras))):
        camera = cameras[i]
        image_name = camera.name.split(".")[0]
        rgb_path = os.path.join(rgb_dir, f"{image_name}.jpg")
        depth_path = os.path.join(depth_dir, f"{image_name}.png")

        point_cloud = SceneKitStructuredPointCloud(
            depth_path, camera, rgb_path
        ).point_cloud
        o3d.io.write_point_cloud(
            os.path.join(output_dir, f"{image_name}.ply"), point_cloud
        )


def depth_to_normal(cfg: DictConfig):
    import cv2
    import numpy as np
    from pyscenekit.scenekit3d.common import SceneKitStructuredPointCloud

    dataset = ScanNetPPDataset(cfg.scannetpp.data_dir)
    dataset.set_scene_id(cfg.scene_id)
    cameras = dataset.iphone_dataset.read_cameras()

    rgb_dir = os.path.join(dataset.iphone_dataset.output_dir, "rgb")
    depth_dir = os.path.join(dataset.iphone_dataset.output_dir, "render_depth")
    output_dir = os.path.join(dataset.iphone_dataset.output_dir, "normal")
    os.makedirs(output_dir, exist_ok=True)
    for i in tqdm(range(len(cameras))):
        camera = cameras[i]
        camera.set_camera_pose(np.eye(4))
        image_name = camera.name.split(".")[0]
        rgb_path = os.path.join(rgb_dir, f"{image_name}.jpg")
        depth_path = os.path.join(depth_dir, f"{image_name}.png")

        skt_point_cloud = SceneKitStructuredPointCloud(depth_path, camera, rgb_path)
        point_cloud = skt_point_cloud.point_cloud
        point_cloud.estimate_normals()
        point_cloud.orient_normals_towards_camera_location(camera.camera_pose[:3, 3])

        depth_image = np.asarray(skt_point_cloud.depth_image)
        normals = np.asarray(point_cloud.normals)
        depth_mask = depth_image > 0
        normal_image = np.zeros((depth_image.shape[0], depth_image.shape[1], 3))
        normal_image[depth_mask] = normals

        normal_image = (-normal_image + 1) * 127.5
        normal_image[~depth_mask] = 0
        normal_image = normal_image.astype(np.uint8)
        cv2.imwrite(
            os.path.join(output_dir, f"{image_name}.png"),
            cv2.cvtColor(normal_image, cv2.COLOR_RGB2BGR),
        )


@hydra.main(config_path="../configs", config_name="scenekit3d", version_base="1.3")
def main(cfg: DictConfig):
    if cfg.verbose:
        attach_to_log()

    # output_dir = (
    #     os.path.dirname(cfg.output) if os.path.isfile(cfg.output) else cfg.output
    # )
    # test_dlsr_dataset(dataset, output_dir)

    export_iphone_dataset(cfg)

    render_iphone_depth(cfg)

    unproject_render_depth(cfg)

    depth_to_normal(cfg)


if __name__ == "__main__":
    main()
