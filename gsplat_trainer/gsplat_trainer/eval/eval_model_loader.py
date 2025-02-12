from gsplat_trainer.model.gaussian_model import GaussianModel
from gsplat_trainer.model_io.ply_handling import load_ply


class EvalModelLoader:
    def __init__(
        self, data_dir: str, method: str, size: str, dataset: str, device: str
    ):
        self.data_dir = data_dir
        self.method = method
        self.size = size
        self.dataset = dataset
        self.device = device

    def _get_model_path(self) -> str:
        if self.method in ["default", "mcmc"]:
            return f"{self.data_dir}/models/{self.method}/{self.method}-{self.dataset}-{self.size}-1/{self.method}-{self.dataset}-{self.size}-1_model.ply"
        elif self.method == "mini-splatting":
            return f"{self.data_dir}/models/mini-splatting/mini_splatting-{self.dataset}-{self.size}-1/point_cloud/iteration_30000/point_cloud.ply"
        elif self.method == "eagles":
            return f"{self.data_dir}/models/eagles/eagles-{self.dataset}-{self.size}-1/point_cloud/iteration_30000/point_cloud.ply"
        elif self.method == "mip-splatting":
            return f"{self.data_dir}/models/mip-splatting/mip_splatting-{self.dataset}-{self.size}-1/point_cloud/iteration_30000/point_cloud.ply"
        elif self.method == "gaussian-pro":
            return f"{self.data_dir}/models/gaussian-pro/gaussian_pro-{self.dataset}-{self.size}-1/point_cloud/iteration_30000/point_cloud.ply"
        elif self.method == "geo-gaussian":
            return f"{self.data_dir}/models/geogaussian/geo_gaussian-{self.dataset}-{self.size}-1/point_cloud/iteration_30000/point_cloud.ply"
        else:
            raise Exception(f"technique {self.method} unknown!")

    def get_model(self) -> GaussianModel:
        return load_ply(self._get_model_path(), 1.0, self.device)
