from typing import Tuple
from gsplat_trainer.eval.eval_dataloader import EvalDataLoader
from gsplat_trainer.eval.eval_model_loader import EvalModelLoader
from gsplat_trainer.eval.perturbed_dataset import PerturbedDataset
from gsplat_trainer.image.image_util import add_backround
from gsplat_trainer.metrics.lpips import LPIPS
from gsplat_trainer.metrics.psnr import psnr
from gsplat_trainer.metrics.ssim import ssim
from gsplat_trainer.model.gaussian_model import GaussianModel
import pandas as pd
import torch
from brisque import BRISQUE


class EvalHandler:
    def __init__(self, data_dir: str, device: str, n_pertubations: int) -> None:
        self.data_dir = data_dir
        self.device = device
        self.lpips = LPIPS(self.device)
        self.bg_color = torch.ones((3,)).to(self.device)
        self.brisque = BRISQUE()
        self.n_pertubations = n_pertubations

    def compute_metrics_dataframe(self) -> pd.DataFrame:
        records_single = []

        with torch.no_grad():
            for dataset in ["truck", "room", "stump"]:

                test_split = EvalDataLoader(self.data_dir, dataset).get_eval_split()
                perturbed_dataset = PerturbedDataset(
                    test_split, self.n_pertubations, trans_std=0.5
                )

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

                            bg_image = (
                                torch.ones_like(gt_image).to(self.device)
                                * self.bg_color
                            )

                            gt_image = add_backround(gt_image, bg_image, gt_alpha)
                            H, W, _ = gt_image.shape

                            out_img = self.render(model, K, viewmat, H, W)

                            curr_psnr, curr_ssim, curr_lpips = (
                                self.compute_reference_metrics(gt_image, out_img)
                            )
                            brisque_score = self.brisque.get_score(
                                out_img.detach().cpu().numpy()
                            )

                            record = {
                                "model": method,
                                "dataset": dataset,
                                "size": size,
                                "view_idx": idx,
                                "psnr": curr_psnr,
                                "ssim": curr_ssim,
                                "lpips": curr_lpips,
                                "brisque": brisque_score,
                            }

                            for idx_perturbed in range(self.n_pertubations):
                                perturbed_pose = perturbed_dataset.get_perturbed_pose(
                                    idx, idx_perturbed
                                )

                                out_img = self.render(
                                    model, K, perturbed_pose.to(self.device), H, W
                                )
                                brisque_score = self.brisque.get_score(
                                    out_img.detach().cpu().numpy()
                                )
                                record[f"brisque_{idx_perturbed}"] = brisque_score

                            records_single.append(record)
        return pd.DataFrame.from_records(records_single)

    def compute_reference_metrics(
        self, gt_image: torch.Tensor, out_img: torch.Tensor
    ) -> Tuple[float, float, float]:
        curr_psnr = (
            psnr(out_img.permute(2, 0, 1), gt_image.permute(2, 0, 1))
            .detach()
            .cpu()
            .item()
        )
        curr_ssim = (
            ssim(out_img.permute(2, 0, 1), gt_image.permute(2, 0, 1))
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

        return curr_psnr, curr_ssim, curr_lpips

    def render(
        self,
        model: GaussianModel,
        K: torch.Tensor,
        viewmat: torch.Tensor,
        H: int,
        W: int,
    ):
        sh_degree_to_use = 0

        renders, _alphas, _info = model(
            view_matrix=viewmat.unsqueeze(0),
            K=K.unsqueeze(0),
            W=W,
            H=H,
            sh_degree_to_use=sh_degree_to_use,
            bg_color=self.bg_color.unsqueeze(0),
        )

        out_img = renders[0]
        return out_img
