from typing import Optional
from gsplat_trainer.data.nvs_dataset import NVSDataset
from gsplat_trainer.image.image_util import add_backround
from gsplat_trainer.loggers.logger import Logger
from gsplat_trainer.metrics.ssim import ssim
from gsplat_trainer.model.gaussian_model import GaussianModel
import torch
import numpy as np


class EarlyStoppingHandler:
    def __init__(
        self,
        test_dataset: NVSDataset,
        n_patience_epochs: int,
        sh_degree_interval: int,
        bg_color: torch.Tensor,
        device: str,
        logger: Optional[Logger] = None,
    ) -> None:
        self.test_dataset = test_dataset
        self.device = device
        self.best_ssim = -1.0
        self.n_epochs_without_improvement = 0
        self.n_patience_epochs = n_patience_epochs
        self.sh_degree_interval = sh_degree_interval
        self.bg_color = bg_color
        self.logger = logger

    @torch.no_grad()
    def check_continue_at_current_epoch(
        self, gaussian_model: GaussianModel, step: int
    ) -> bool:
        ssims = []

        for i in range(len(self.test_dataset)):
            pose, gt_image, gt_alpha, K = self.test_dataset[i]
            viewmat = pose.to(self.device)
            K = K.to(self.device)
            gt_image = gt_image.to(self.device)
            gt_alpha = gt_alpha.to(self.device)

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
                ssim(out_img.permute(2, 0, 1), gt_image.permute(2, 0, 1)).detach().cpu()
            )

        new_ssim = np.mean(ssims)

        if self.logger != None:
            self.logger.log({"early_stopping_test/ssim": new_ssim}, iteration=step)

        if new_ssim > self.best_ssim:
            self.best_ssim = new_ssim
            self.n_epochs_without_improvement = 0
        else:
            self.n_epochs_without_improvement = self.n_epochs_without_improvement + 1

        if self.n_epochs_without_improvement > self.n_patience_epochs:
            print(
                f"No improvement in SSIM for {self.n_epochs_without_improvement}, stopping training at step {step}"
            )
            return False

        return True
