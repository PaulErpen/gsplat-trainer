import time
from typing import Dict, Optional, List
from gsplat_trainer.config.config import Config
from gsplat_trainer.data.nvs_dataset import NVSDataset
from gsplat_trainer.image.image_util import add_backround
from gsplat_trainer.loggers.logger import Logger
from gsplat_trainer.metrics.psnr import psnr
from gsplat_trainer.metrics.ssim import ssim
from gsplat_trainer.model.gaussian_model import GaussianModel
from gsplat_trainer.test.early_stopping_handler import EarlyStoppingHandler
from gsplat_trainer.test.holdout_view_handler import HoldoutViewHandler
from gsplat_trainer.test.validation_handler import ValidationHandler
from gsplat_trainer.train.strategy.strategy_wrapper import Strategy
import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm


class SimpleTrainer:
    """Trains random gaussians to fit an image."""

    def __init__(
        self,
        train_dataset: NVSDataset,
        test_dataset: Optional[NVSDataset],
        gaussian_model: GaussianModel,
        strategy: Strategy,
        optimizers: Dict,
        schedulers: List,
        config: Config = Config(),
        holdout_view_handler: Optional[HoldoutViewHandler] = None,
        logger: Optional[Logger] = None,
        validation_handler: Optional[ValidationHandler] = None,
        device="cuda",
        early_stopping_handler: Optional[EarlyStoppingHandler] = None,
    ):
        self.device = device
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.holdout_view_handler = holdout_view_handler
        self.config = config
        self.holdout_view_frequency = config.holdout_view_frequency

        self.gaussian_model = gaussian_model.to(self.device)

        self.optimizers = optimizers
        self.schedulers = schedulers

        self.strategy = strategy
        self.logger = logger

        self.strategy.check_sanity(self.gaussian_model.params, self.optimizers)

        self.cum_created = 0
        self.cum_deleted = 0

        self.bg_color = self.config.bg_color.to(self.device)

        self.validation_handler = validation_handler
        self.early_stopping_handler = early_stopping_handler

    def train(
        self,
    ) -> None:
        times = [0] * 2  # rasterization, backward

        indeces = list(range(len(self.train_dataset)))

        # shuffle indeces
        np.random.shuffle(indeces)

        tqdm_progress = tqdm(range(1, self.config.max_steps + 1), desc="Training")
        for iter in tqdm_progress:
            start = time.time()
            pose, gt_image, gt_alpha, K = self.train_dataset[
                indeces[(iter - 1) % len(indeces)]
            ]
            viewmat = pose.to(self.device)
            K = K.to(self.device)
            gt_image = gt_image.to(self.device)
            gt_alpha = gt_alpha.to(self.device)

            bg_image = torch.ones_like(gt_image).to(self.device) * self.bg_color

            gt_image = add_backround(gt_image, bg_image, gt_alpha)

            sh_degree_to_use = min(
                iter // self.config.sh_degree_interval, self.gaussian_model.sh_degree
            )

            H, W, _ = gt_image.shape
            renders, alphas, info = self.gaussian_model(
                view_matrix=viewmat.unsqueeze(0),
                K=K.unsqueeze(0),
                W=W,
                H=H,
                sh_degree_to_use=sh_degree_to_use,
                bg_color=self.bg_color.unsqueeze(0),
            )
            info["means2d"].retain_grad()  # used for running stats

            out_img = renders[0]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times[0] += time.time() - start

            self.strategy.step_pre_backward(
                params=self.gaussian_model.params,
                optimizers=self.optimizers,
                step=iter,
                info=info,
            )

            # loss
            l1loss = F.l1_loss(out_img, gt_image)
            ssimloss = 1.0 - ssim(
                out_img.permute(2, 0, 1).unsqueeze(0),
                gt_image.permute(2, 0, 1).unsqueeze(0),
            )
            ssim_lambda = self.config.ssim_lambda
            loss = l1loss * (1.0 - ssim_lambda) + ssimloss * ssim_lambda

            # regularizations
            if self.config.opacity_reg > 0.0:
                loss = (
                    loss
                    + self.config.opacity_reg
                    * torch.abs(
                        torch.sigmoid(self.gaussian_model.params["opacities"])
                    ).mean()
                )
            if self.config.scale_reg > 0.0:
                loss = (
                    loss
                    + self.config.scale_reg
                    * torch.abs(torch.exp(self.gaussian_model.params["scales"])).mean()
                )

            if self.logger is not None:
                self.logger.log(
                    {
                        "train/psnr": psnr(
                            out_img.permute(2, 0, 1), gt_image.permute(2, 0, 1)
                        ),
                        "train/ssim": ssim(
                            out_img.permute(2, 0, 1), gt_image.permute(2, 0, 1)
                        ),
                    },
                    iter,
                )

            for key, optimizer in self.optimizers.items():
                optimizer.zero_grad()

            start = time.time()

            loss.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            n_created, n_deleted = self.strategy.step_post_backward(
                params=self.gaussian_model.params,
                optimizers=self.optimizers,
                step=iter,
                info=info,
                packed=self.config.packed,
                schedulers=self.schedulers,
            )

            for key, optimizer in self.optimizers.items():
                optimizer.step()
            for scheduler in self.schedulers:
                scheduler.step()

            times[1] += time.time() - start

            tqdm_progress.set_postfix({"Loss": loss.item()})

            with torch.no_grad():
                if (
                    self.holdout_view_handler is not None
                    and iter % self.config.holdout_view_frequency == 0
                ):
                    self.holdout_view_handler.compute_holdout_view(
                        self.gaussian_model, sh_degree_to_use
                    )

                self.cum_created = self.cum_created + n_created
                self.cum_deleted = self.cum_deleted + n_deleted
                if self.logger is not None:
                    self.logger.log(
                        {
                            "n_gaussians": self.gaussian_model.params["means"].shape[0],
                            "cum_created": self.cum_created,
                            "cum_deleted": self.cum_deleted,
                        },
                        iter,
                    )

                self.validation_handler.handle_validation(self.gaussian_model, iter)

            if iter % len(self.train_dataset) == 0:
                # epoch end
                # shuffle indeces
                np.random.shuffle(indeces)

                grace_period_after_opa_reset = (
                    iter > self.config.refine_start_iter
                    and iter < self.config.refine_stop_iter
                    and iter % self.config.refine_every
                    < self.config.early_stopping_opa_grace_period
                )

                if self.early_stopping_handler is not None and not grace_period_after_opa_reset:
                    if not self.early_stopping_handler.check_continue_at_current_epoch(
                        self.gaussian_model, step=iter
                    ):
                        break

        if self.holdout_view_handler is not None:
            self.holdout_view_handler.export_gif()
        print(f"Total(s):\nRasterization: {times[0]:.3f}, Backward: {times[1]:.3f}")
        print(
            f"Per step(s):\nRasterization: {times[0]/self.config.max_steps:.5f}, Backward: {times[1]/self.config.max_steps:.5f}"
        )

        return self.gaussian_model
