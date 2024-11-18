from logging import Logger
from typing import List, Optional
from gsplat_trainer.data.nvs_dataset import NVSDataset
from gsplat_trainer.image.image_util import add_backround
from gsplat_trainer.metrics.psnr import psnr
from gsplat_trainer.metrics.ssim import ssim
from gsplat_trainer.model.gaussian_model import GaussianModel
import torch
import numpy as np

class ValidationHandler:
    def __init__(
        self,
        train_dataset: NVSDataset,
        test_dataset: Optional[NVSDataset],
        test_iterations: List[int],
        device: str,
        bg_color: torch.Tensor,
        sh_degree_interval: int,
        logger: Optional[Logger] = None,
    ):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.test_iterations = test_iterations
        self.logger = logger
        self.device = device
        self.sh_degree_interval = sh_degree_interval
        self.bg_color = bg_color

        self.validation_configs = [
            {
                "name": "train_every_5th",
                "range": range(0, len(self.train_dataset), 5),
                "data": self.train_dataset,
            }
        ]
        if self.test_dataset is not None:
            self.validation_configs.append(
                {
                    "name": "test_full",
                    "range": range(len(self.test_dataset)),
                    "data": self.test_dataset,
                }
            )

    def handle_validation(self, gaussian_model: GaussianModel, step: int):
        if step in self.test_iterations and self.logger is not None:
            ssims = []
            psnrs = []

            for config in self.validation_configs:
                for test_iter in config["range"]:
                    dataset = config["data"]
                    K = dataset.intrinsics.to(self.device).to(torch.float32)
                    observation = dataset[test_iter]
                    viewmat = observation[0].to(self.device)
                    gt_image = observation[1].to(self.device)
                    gt_alpha = observation[2].to(self.device)

                    bg_image = torch.ones_like(gt_image).to(self.device)

                    gt_image = add_backround(gt_image, bg_image, gt_alpha)

                    sh_degree_to_use = min(
                        step // self.sh_degree_interval, gaussian_model.sh_degree
                    )

                    H, W, _ = gt_image.shape
                    renders, alphas, info = gaussian_model(
                        view_matrix=viewmat.unsqueeze(0).to(self.device),
                        K=K.unsqueeze(0).to(self.device),
                        W=W,
                        H=H,
                        sh_degree_to_use=sh_degree_to_use,
                        bg_color=self.bg_color.unsqueeze(0).to(self.device),
                    )
                    out_img = renders[0]

                    ssims.append(
                        ssim(out_img.permute(2, 0, 1), gt_image.permute(2, 0, 1))
                        .detach()
                        .cpu()
                    )
                    psnrs.append(
                        psnr(out_img.permute(2, 0, 1), gt_image.permute(2, 0, 1))
                        .detach()
                        .cpu()
                    )

                self.logger.log(
                    {
                        f"{config['name']}/ssim": np.mean(ssims),
                        f"{config['name']}/psnr": np.mean(psnrs),
                    },
                    step,
                )
