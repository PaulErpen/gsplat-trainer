from typing import Tuple
import unittest
from unittest.mock import patch
from torch import nn
from gsplat_trainer.data.nvs_dataset import NVSDataset
from gsplat_trainer.eval.eval_handler import EvalHandler
import torch
import pandas as pd

W, H = 32, 32
N = 2


class MockedNvsDataset(NVSDataset):
    def __init__(self) -> None:
        pass

    def __len__(self) -> int:
        return N

    def __getitem__(self, _idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.rand(4, 4),
            torch.rand(W, H, 3),
            torch.rand(W, H, 1),
            torch.rand(4, 4),
        )


class MockedGaussianModel(nn.Module):
    def __init__(self) -> None:
        super(MockedGaussianModel, self).__init__()

    def forward(self, **kwargs) -> Tuple:
        return torch.rand(1, W, H, 3), torch.rand(1, W, H, 1), {}


class EvalHandlerTest(unittest.TestCase):
    @patch("gsplat_trainer.eval.eval_handler.EvalDataLoader")
    @patch("gsplat_trainer.eval.eval_handler.EvalModelLoader")
    def test_given_mocked_members__when_computing_the_metrics_dataframe__then_the_dataframe_must_be_correct(
        self, MockEvalModelLoader, MockEvalDataLoader
    ) -> None:
        self.setup_mocks(MockEvalDataLoader, MockEvalModelLoader)

        df = EvalHandler(
            "./nonexisten_unittest_dat_dir", "cpu", 5
        ).compute_metrics_dataframe()

        self.assertEqual(df.shape, (63 * N, 13))
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df["view_idx"].dtype, "int64")
        self.assertEqual(df["psnr"].dtype, "float64")
        self.assertEqual(df["ssim"].dtype, "float64")
        self.assertEqual(df["lpips"].dtype, "float64")

    def setup_mocks(self, MockEvalDataLoader, MockEvalModelLoader) -> None:
        # mock_model_loader_instance = MockEvalModelLoader.return_value
        # mock_model_loader_instance.get_model.return_value = MockedGaussianModel()
        # def mock_model_loader_init(*args, **kwargs):
        model_loader = unittest.mock.Mock()
        model_loader.get_model.return_value = MockedGaussianModel()
        #    return instance

        MockEvalModelLoader.return_value = model_loader

        # def mock_data_loader_init(*args, **kwargs):
        data_loader = unittest.mock.Mock()
        data_loader.get_eval_split.return_value = MockedNvsDataset()
        #     return instance

        # mock_data_loader_instance = MockEvalDataLoader.return_value
        # mock_data_loader_instance.get_eval_split.return_value = MockedNvsDataset()
        MockEvalDataLoader.return_value = data_loader
