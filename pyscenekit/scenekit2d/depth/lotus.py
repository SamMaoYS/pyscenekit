import torch
import numpy as np
from contextlib import nullcontext

from pyscenekit.scenekit2d.depth.base import BaseDepthEstimation

# TODO: support LotusDPipeline in the future
from pyscenekit.scenekit2d.depth.modules.lotus.pipeline import LotusGPipeline


class LotusDepthEstimation(BaseDepthEstimation):
    def __init__(self, model_path: str = None):
        super().__init__(model_path)
        if self.model_path is None:
            self.model_path = "jingheya/lotus-depth-g-v1-0"

        self.weight_dtype = torch.float16

        self.image_processor = None
        self.load_model()

        self.seed = -1
        self.generator = None
        self.timestep = 999

    def load_model(self):
        self.model = LotusGPipeline.from_pretrained(
            self.model_path, torch_dtype=self.weight_dtype
        )

    def set_seed(self, seed: int):
        if seed >= 0:
            self.seed = seed
            self.generator = torch.Generator(device=self.device).manual_seed(self.seed)

    def set_timestep(self, timestep: int):
        self.timestep = timestep

    @torch.no_grad()
    def _predict(self, image: np.ndarray) -> np.ndarray:
        self.to(self.device)
        if torch.backends.mps.is_available():
            autocast_ctx = nullcontext()
        else:
            autocast_ctx = torch.autocast(self.model.device.type)
        with autocast_ctx:
            image = np.asarray(image).astype(np.float32)
            image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)
            image = image / 127.5 - 1.0
            image = image.to(self.device)

            task_emb = (
                torch.tensor([1, 0]).float().unsqueeze(0).repeat(1, 1).to(self.device)
            )
            task_emb = torch.cat(
                [torch.sin(task_emb), torch.cos(task_emb)], dim=-1
            ).repeat(1, 1)

            pred = self.model(
                rgb_in=image,
                prompt="",
                num_inference_steps=1,
                generator=self.generator,
                output_type="np",
                timesteps=[self.timestep],
                task_emb=task_emb,
            ).images[0]

            depth = pred.mean(axis=-1)
        return depth

    def to(self, device: str):
        self.device = torch.device(device)
        self.model.to(self.device)
