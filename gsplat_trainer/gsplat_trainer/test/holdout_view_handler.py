from typing import List
from gsplat_trainer.model.gaussian_model import GaussianModel
import torch
from PIL import Image
import os
from pathlib import Path
import numpy as np
from imageio_ffmpeg import write_frames
import imageio


class HoldoutViewHandler:
    def __init__(
        self,
        holdout_view_matrix: torch.Tensor,
        K: torch.Tensor,
        W: int,
        H: int,
        out_dir: str,
        bg_color: torch.Tensor | None,
        device="cuda",
        thumbnail_size=(600, 600),
    ):
        self.holdout_view_matrix = holdout_view_matrix.to(torch.float32)
        self.K = K.to(torch.float32)
        self.W = W
        self.H = H
        self.frames: List[Image.Image] = []
        self.bg_color = bg_color
        self.device = device
        self.out_dir = Path(out_dir)
        self.thumbnail_size = thumbnail_size

    def to(self, device: str) -> None:
        self.device = device
        self.holdout_view_matrix = self.holdout_view_matrix.to(device)
        self.K = self.K.to(device)
        return self

    def compute_holdout_view(self, model: GaussianModel, sh_degrees_to_use: int):
        renders, _, _ = model(
            view_matrix=self.holdout_view_matrix.unsqueeze(0).to(self.device),
            K=self.K.unsqueeze(0).to(self.device),
            W=self.W,
            H=self.H,
            sh_degree_to_use=sh_degrees_to_use,
            bg_color=self.bg_color.unsqueeze(0).to(self.device),
        )
        out_img = renders[0]
        out_img = torch.clamp(out_img, 0.0, 1.0)
        image = Image.fromarray((out_img.detach().cpu().numpy() * 255).astype(np.uint8))
        image.thumbnail(self.thumbnail_size)
        self.frames.append(image)

    def export_gif(self) -> None:
        export_dir = Path(os.getcwd()) / self.out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.frames[0].save(
            f"{export_dir}/training.gif",
            save_all=True,
            append_images=self.frames[1:],
            optimize=False,
            duration=5,
            loop=0,
        )

    def export_mp4(self) -> None:
        export_dir = Path(os.getcwd()) / self.out_dir
        output_path = f"{export_dir}/training.mp4"
        os.makedirs(self.out_dir, exist_ok=True)

        writer = write_frames(
            path=output_path,
            size=self.frames[0].size,
            fps=25,
            codec="libx264",
        )
        writer.send(None)
        try:
            for image in self.frames:
                frame = np.array(image.convert("RGB"))
                writer.send(frame)
        finally:
            writer.close()
