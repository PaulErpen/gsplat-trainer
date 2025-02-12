from gsplat_trainer.eval.eval_dataloader import EvalDataLoader
from gsplat_trainer.eval.eval_model_loader import EvalModelLoader
from gsplat_trainer.image.image_util import add_backround
from gsplat_trainer.metrics.lpips import LPIPS
from gsplat_trainer.metrics.psnr import psnr
from gsplat_trainer.metrics.ssim import ssim
import pandas as pd
import torch


class EvalHandler:
    def __init__(self, data_dir: str, device: str) -> None:
        self.data_dir = data_dir
        self.device = device
        self.lpips = LPIPS(self.device)

    def compute_metrics_dataframe(self) -> pd.DataFrame:
        records_single = []

        with torch.no_grad():
            for dataset in ["truck", "room", "stump"]:

                test_split = EvalDataLoader(self.data_dir, dataset).get_eval_split()

                for method in [
                    "default",
                    "eagles",
                    "gaussian-pro",
                    "geo-gaussian",
                    "mcmc",
                    "mini-splatting",
                    "mip-splatting",
                ]:
                    for size in ["low", "medium", "high"]:
                        model = EvalModelLoader(
                            self.data_dir, method, size, dataset, self.device
                        ).get_model()
                        for idx in range(len(test_split)):
                            pose, gt_image, gt_alpha, K = test_split[idx]
                            viewmat = pose.to(self.device)
                            K = K.to(self.device)
                            gt_image = gt_image.to(self.device)
                            gt_alpha = gt_alpha.to(self.device)

                            bg_color = torch.ones((3,)).to(self.device)
                            bg_image = (
                                torch.ones_like(gt_image).to(self.device) * bg_color
                            )

                            gt_image = add_backround(gt_image, bg_image, gt_alpha)

                            sh_degree_to_use = 0

                            H, W, _ = gt_image.shape
                            renders, alphas, info = model(
                                view_matrix=viewmat.unsqueeze(0),
                                K=K.unsqueeze(0),
                                W=W,
                                H=H,
                                sh_degree_to_use=sh_degree_to_use,
                                bg_color=bg_color.unsqueeze(0),
                            )

                            out_img = renders[0]

                            curr_psnr = (
                                psnr(
                                    out_img.permute(2, 0, 1), gt_image.permute(2, 0, 1)
                                )
                                .detach()
                                .cpu()
                                .item()
                            )
                            curr_ssim = (
                                ssim(
                                    out_img.permute(2, 0, 1), gt_image.permute(2, 0, 1)
                                )
                                .detach()
                                .cpu()
                                .item()
                            )
                            curr_lpips = (
                                self.lpips.compute_lpips(
                                    out_img.permute(2, 0, 1), gt_image.permute(2, 0, 1)
                                )
                                .detach()
                                .cpu()
                                .item()
                            )

                            records_single.append(
                                {
                                    "model": method,
                                    "dataset": dataset,
                                    "size": size,
                                    "view_idx": idx,
                                    "psnr": curr_psnr,
                                    "ssim": curr_ssim,
                                    "lpips": curr_lpips,
                                }
                            )
        return pd.DataFrame.from_records(records_single)
