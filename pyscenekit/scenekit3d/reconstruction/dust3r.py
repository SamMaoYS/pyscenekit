import torch
import numpy as np
from copy import deepcopy
from typing import Literal, List

import trimesh
import PIL.Image
from tqdm import tqdm
from jaxtyping import Float32, Bool
from torchvision import transforms
from mini_dust3r.utils.image import ImageDict
from mini_dust3r.inference import inference, Dust3rResult
from mini_dust3r.model import AsymmetricCroCo3DStereo
from mini_dust3r.image_pairs import make_pairs
from mini_dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from mini_dust3r.cloud_opt.base_opt import BasePCOptimizer
from mini_dust3r.viz import pts3d_to_trimesh
from mini_dust3r.model import AsymmetricCroCo3DStereo

from pyscenekit.scenekit3d.common import (
    SceneKitCamera,
    SceneKitPointCloud,
    SceneKitMesh,
)
from pyscenekit.scenekit3d.reconstruction.base import (
    MultiViewReconstructionModel,
    MultiViewReconstructionOutput,
)


class Dust3rReconstruction(MultiViewReconstructionModel):
    def __init__(self, model_path: str = None):
        super().__init__(model_path)
        if self.model_path is None:
            self.model_path = "nielsr/DUSt3R_ViTLarge_BaseDecoder_512_dpt"

        self.load_model()
        self.image_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def load_model(self):
        self.model = AsymmetricCroCo3DStereo.from_pretrained(self.model_path)

    # TODO: implement common way to set method specific parameters
    def _predict(self) -> np.ndarray:
        self.to(self.device)
        image_list = self.input.image_list

        return self.inferece_dust3r(
            image_list=image_list,
            device=self.device.type,
        )

    def to(self, device: str):
        self.device = torch.device(device)
        self.model.to(self.device)

    # ref: https://github.com/pablovela5620/mini-dust3r
    def inferece_dust3r(
        self,
        image_list: List[np.ndarray],
        device: Literal["cpu", "cuda", "mps"],
        batch_size: int = 1,
        image_size: Literal[224, 512] = 512,
        niter: int = 100,
        schedule: Literal["linear", "cosine"] = "linear",
        min_conf_thr: float = 10,
    ) -> MultiViewReconstructionOutput:
        """
        Perform inference using the Dust3r algorithm.

        Args:
            image_list (List[np.ndarray]): The list of input images.
            model (AsymmetricCroCo3DStereo): The Dust3r model to use for inference.
            device (Literal["cpu", "cuda", "mps"]): The device to use for inference ("cpu", "cuda", or "mps").
            batch_size (int, optional): The batch size for inference. Defaults to 1.
            image_size (Literal[224, 512], optional): The size of the input images. Defaults to 512.
            niter (int, optional): The number of iterations for the global alignment optimization. Defaults to 100.
            schedule (Literal["linear", "cosine"], optional): The learning rate schedule for the global alignment optimization. Defaults to "linear".
            min_conf_thr (float, optional): The minimum confidence threshold for the optimized result. Defaults to 10.

        Returns:
            MultiViewReconstructionOutput: The optimized result containing the RGB, depth, and confidence images.

        Raises:
            ValueError: If `image_dir_or_list` is neither a list of paths nor a path.
        """
        imgs = self.load_images(image_list, image_size)

        # if only one image was loaded, duplicate it to feed into stereo network
        if len(imgs) == 1:
            imgs = [imgs[0], deepcopy(imgs[0])]
            imgs[1]["idx"] = 1

        pairs: list[tuple[ImageDict, ImageDict]] = make_pairs(
            imgs, scene_graph="complete", prefilter=None, symmetrize=True
        )
        output: Dust3rResult = inference(
            pairs, self.model, device, batch_size=batch_size
        )

        mode = (
            GlobalAlignerMode.PointCloudOptimizer
            if len(imgs) > 2
            else GlobalAlignerMode.PairViewer
        )
        scene: BasePCOptimizer = global_aligner(
            dust3r_output=output, device=device, mode=mode
        )

        lr = 0.01

        if mode == GlobalAlignerMode.PointCloudOptimizer:
            loss = scene.compute_global_alignment(
                init="mst", niter=niter, schedule=schedule, lr=lr
            )

        # get the optimized result from the scene
        optimized_result: MultiViewReconstructionOutput = self.scene_to_results(
            scene, min_conf_thr
        )
        return optimized_result

    # ref: https://github.com/pablovela5620/mini-dust3r
    @staticmethod
    def scene_to_results(
        scene: BasePCOptimizer, min_conf_thr: int
    ) -> MultiViewReconstructionOutput:
        ### get camera parameters K and T
        K_b33: Float32[np.ndarray, "b 3 3"] = scene.get_intrinsics().numpy(force=True)
        world_T_cam_b44: Float32[np.ndarray, "b 4 4"] = scene.get_im_poses().numpy(
            force=True
        )

        cameras = []
        for i in range(len(K_b33)):
            cam = SceneKitCamera(
                intrinsics=K_b33[i],
                name=f"{i}",
            )
            cam.set_camera_pose(world_T_cam_b44[i])
            cameras.append(cam)

        ### image, confidence, depths
        rgb_hw3_list: list[Float32[np.ndarray, "h w 3"]] = scene.imgs
        depth_hw_list: list[Float32[np.ndarray, "h w"]] = [
            depth.numpy(force=True) for depth in scene.get_depthmaps()
        ]
        # normalized depth
        # depth_hw_list = [depth_hw / depth_hw.max() for depth_hw in depth_hw_list]

        conf_hw_list: list[Float32[np.ndarray, "h w"]] = [
            c.numpy(force=True) for c in scene.im_conf
        ]
        # normalize confidence
        # conf_hw_list = [conf_hw / conf_hw.max() for conf_hw in conf_hw_list]

        # point cloud, mesh
        pts3d_list: list[Float32[np.ndarray, "h w 3"]] = [
            pt3d.numpy(force=True) for pt3d in scene.get_pts3d()
        ]
        # get log confidence
        log_conf_trf: Float32[torch.Tensor, ""] = scene.conf_trf(
            torch.tensor(min_conf_thr)
        )
        # set the minimum confidence threshold
        scene.min_conf_thr = float(log_conf_trf)
        masks_list: list[Bool[np.ndarray, "h w"]] = [
            mask.numpy(force=True) for mask in scene.get_masks()
        ]

        vertices_list: list[Float32[np.ndarray, "num_points 3"]] = [
            p[m] for p, m in zip(pts3d_list, masks_list)
        ]
        vertex_colors_list: list[Float32[np.ndarray, "num_points 3"]] = [
            p[m] for p, m in zip(rgb_hw3_list, masks_list)
        ]

        point_cloud_list = [
            SceneKitPointCloud.from_vertices(vertices, vertex_colors)
            for vertices, vertex_colors in zip(vertices_list, vertex_colors_list)
        ]

        meshes = []
        pbar = tqdm(zip(rgb_hw3_list, pts3d_list, masks_list), total=len(rgb_hw3_list))
        for rgb_hw3, pts3d, mask in pbar:
            meshes.append(
                SceneKitMesh(trimesh.Trimesh(**pts3d_to_trimesh(rgb_hw3, pts3d, mask)))
            )

        optimised_result = MultiViewReconstructionOutput(
            color_list=rgb_hw3_list,
            depth_list=depth_hw_list,
            confidence_list=conf_hw_list,
            mask_list=masks_list,
            cameras=cameras,
            point_cloud_list=point_cloud_list,
            mesh_list=meshes,
        )
        return optimised_result

    @staticmethod
    def _resize_pil_image(img, long_edge_size):
        S = max(img.size)
        if S > long_edge_size:
            interp = PIL.Image.LANCZOS
        elif S <= long_edge_size:
            interp = PIL.Image.BICUBIC
        new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
        return img.resize(new_size, interp)

    # TODO: move image resize to base class
    def load_images(
        self,
        image_list: List[np.ndarray],
        size: Literal[224, 512],
        square_ok: bool = False,
    ) -> list[ImageDict]:
        imgs = []
        for img in image_list:
            # convert to PIL image
            img = PIL.Image.fromarray(img)
            W1, H1 = img.size
            if size == 224:
                # resize short side to 224 (then crop)
                img = self._resize_pil_image(img, round(size * max(W1 / H1, H1 / W1)))
            else:
                # resize long side to 512
                img = self._resize_pil_image(img, size)
            W, H = img.size
            cx, cy = W // 2, H // 2
            if size == 224:
                half = min(cx, cy)
                img = img.crop((cx - half, cy - half, cx + half, cy + half))
            else:
                halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
                if not (square_ok) and W == H:
                    halfh = 3 * halfw / 4
                img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

            imgs.append(
                dict(
                    img=self.image_transform(img)[None],
                    true_shape=np.int32([img.size[::-1]]),
                    idx=len(imgs),
                    instance=str(len(imgs)),
                )
            )

        return imgs
