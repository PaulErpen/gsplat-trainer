import argparse
import os
from pathlib import Path
import time


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
        help="The path where the data is tored",
        required=True,
    )

    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        help="The timeout in seconds between the calls to slurm.",
        required=True,
    )

    parsed_args = parser.parse_args()

    i = 0
    for dataset in ["truck", "room", "stump"]:
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
                exec_path = f"repos/slurm-repo/{method}/runs/{dataset}/{size}/run-{method}.slurm"
                model_path = get_model_path(parsed_args.dataDir, method, size, dataset)
                if Path(model_path).exists():
                    print(f'Warning: Model path already exists! "{model_path}"')
                    print(f'Not executing "{exec_path}"')
                else:
                    if Path(exec_path).exists():
                        i = i + 1
                        os.system(f"sbatch {exec_path}")
                        print(f'Submitted: "{exec_path}"')
                        time.sleep(parsed_args.timeout)
                    else:
                        print('Warning: "{exec_path}" does not exist!')
    print(f"Submitted {i} jobs to the queue")
