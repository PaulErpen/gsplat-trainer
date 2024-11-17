from gsplat_trainer.data.basicpointcloud import BasicPointCloud
from gsplat_trainer.data.nerfnorm import NerfNorm
from gsplat_trainer.data.nvs_dataset import NVSDataset
from gsplat_trainer.model.gaussian_model import GaussianModel
import torch
import numpy as np


class MockFactory:
    @staticmethod
    def create_mocked_gaussian_model(n_points: int = 20) -> GaussianModel:
        dataset = MockFactory.create_mocked_nvs_dataset(n_points=n_points)
        return GaussianModel.from_point_cloud(dataset.pcd, dataset.norm.radius)

    @staticmethod
    def create_mocked_nvs_dataset(
        n_entries: int = 40, H: int = 128, W: int = 128, n_points: int = 50
    ) -> NVSDataset:
        return NVSDataset(
            poses=torch.rand((n_entries, 4, 4)),
            images=torch.rand((n_entries, H, W, 3)),
            alphas=torch.rand((n_entries, H, W, 1)),
            intrinsics=torch.rand((n_entries, 3, 3)),
            pcd=BasicPointCloud(
                np.random.rand(n_points, 3),
                np.random.rand(n_points, 3),
                np.random.rand(n_points, 3),
            ),
            norm=NerfNorm(
                torch.rand(
                    3,
                ),
                3.1,
            ),
        )
