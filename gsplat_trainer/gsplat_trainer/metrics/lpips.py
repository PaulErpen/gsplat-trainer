from typing import Literal

import torch


class LPIPS:
    def __init__(self, device: Literal["cpu", "cuda"]):
        self.device = device
        if device == "cuda":
            import lpips

            self.lpips_alex = lpips.LPIPS(net="alex").to("cuda")

    def compute_lpips(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        if self.device == "cuda":
            return self.lpips_alex(img1, img2)
        else:
            return torch.rand(1)
