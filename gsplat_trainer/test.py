import argparse
from pathlib import Path

from gsplat_trainer.config.config import Config
from gsplat_trainer.data.data_service import DataManager
from gsplat_trainer.image.image_util import add_backround
from gsplat_trainer.model_io.ply_handling import load_ply
import torch
from gsplat_trainer.metrics.psnr import psnr
from gsplat_trainer.metrics.ssim import ssim
import lpips
import pandas as pd


def get_dataset_path(dataDir: str, dataset_name: str) -> str:
    if dataset_name == "truck":
        return f"{dataDir}/data/TNT_GOF/TrainingSet/Truck/"
    if dataset_name == "room":
        return f"{dataDir}/data/room/"
    if dataset_name == "stump":
        return f"{dataDir}/data/stump/"
    raise Exception(f'Dataset name "{dataset_name}" does not exist')


def get_model_path(data_dir: str, method: str, size: str, dataset: str) -> str:
    if method in ["default", "mcmc"]:
        return f"{data_dir}/models/{method}/{method}-{dataset}-{size}-1/{method}-{dataset}-{size}-1_model.ply"
    elif method == "mini-splatting":
        return f"{data_dir}/models/mini-splatting/mini_splatting-{dataset}-{size}-1/point_cloud/iteration_30000/point_cloud.ply"
    elif method == "eagles":
        return f"{data_dir}/models/eagles/eagles-{dataset}-{size}-1/point_cloud/iteration_30000/point_cloud.ply"
    elif method == "mip-splatting":
        return f"{data_dir}/models/mip-splatting/mip_splatting-{dataset}-{size}-1/point_cloud/iteration_30000/point_cloud.ply"
    elif method == "gaussian-pro":
        return f"{data_dir}/models/gaussian-pro/gaussian_pro-{dataset}-{size}-1/point_cloud/iteration_30000/point_cloud.ply"
    elif method == "geo-gaussian":
        return f"{data_dir}/models/geogaussian/geo_gaussian-{dataset}-{size}-1/point_cloud/iteration_30000/point_cloud.ply"
    else:
        raise Exception(f"technique {method} unknown!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataDir",
        "-d",
        type=str,
        help="The path where the data is stored",
        required=True,
    )

    parser.add_argument(
        "--singleDfPathXl",
        "-s",
        type=str,
        help="The path where the data for the single results dataframe will be stored in excel format",
        required=True,
    )

    parsed_args = parser.parse_args()

    lpips_alex = lpips.LPIPS(net="alex").to("cuda")

    records_single = []

    with torch.no_grad():
        for dataset in ["truck", "room", "stump"]:

            config: Config = Config()
            config.dataset_path = get_dataset_path(parsed_args.dataDir, dataset)

            data_manager = DataManager(config=config)

            test_split = data_manager.get_split("test")

            for method in [
                "default",
                "eagles",
                "gaussian-pro",
                "geo-gaussian",
                "mcmc",
                "mini-splatting",
                "mip-splatting",
            ]:
                for size in ["low", "medium", "high", "extended"]:
                    model_path = get_model_path(
                        parsed_args.dataDir,
                        method=method,
                        size=size,
                        dataset=dataset,
                    )
                    if not Path(model_path).exists():
                        print(f'Error! model "{model_path}" does not exist!')
                    else:
                        model = load_ply(
                            model_path,
                            scene_scale=test_split.norm.radius,
                        )

                        for idx in range(len(test_split)):
                            pose, gt_image, gt_alpha, K = test_split[idx]
                            viewmat = pose.to("cuda")
                            K = K.to("cuda")
                            gt_image = gt_image.to("cuda")
                            gt_alpha = gt_alpha.to("cuda")

                            bg_color = torch.ones((3,)).to("cuda")
                            bg_image = torch.ones_like(gt_image).to("cuda") * bg_color

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

                            curr_psnr = psnr(
                                out_img.permute(2, 0, 1), gt_image.permute(2, 0, 1)
                            ).detach().cpu()
                            curr_ssim = ssim(
                                out_img.permute(2, 0, 1), gt_image.permute(2, 0, 1)
                            ).detach().cpu()
                            curr_lpips = lpips_alex(
                                out_img.permute(2, 0, 1), gt_image.permute(2, 0, 1)
                            ).detach().cpu()

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

        df_single = pd.DataFrame.from_records(records_single)
        df_single.to_excel(parsed_args.singleDfPathXl)
