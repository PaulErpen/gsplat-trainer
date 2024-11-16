import shutil
import unittest
from gsplat_trainer.test.holdout_view_handler import HoldoutViewHandler
import torch
from gsplat_trainer.model.gaussian_model import GaussianModel
from torch import nn
import os


class HoldoutviewHandlerTest(unittest.TestCase):
    def setUp(self):
        N = 100
        self.gaussian_model = GaussianModel(
            params=nn.ParameterDict(
                {
                    "means": nn.Parameter(torch.rand((N, 3))),
                    "scales": nn.Parameter(torch.rand((N, 3))),
                    "quats": nn.Parameter(torch.rand((N, 4)).to(torch.float32)),
                    "opacities": torch.rand((N,)),
                    "sh0": torch.nn.Parameter(torch.rand((N, 1, 3))),
                    "shN": torch.nn.Parameter(torch.rand((N, 15, 3))),
                }
            ),
            scene_scale=123.4,
        )
        self.out_path = "unittests/test-renders"
        os.makedirs(self.out_path, exist_ok=True)

    def test_given_a_valid_holdout_view_handler__when_computing_a_set_of_views_and_exporting__then_the_gif_must_exist(
        self,
    ):
        holdout_view_handler = HoldoutViewHandler(
            torch.eye(4),
            torch.eye(3),
            800,
            800,
            bg_color=torch.rand(3),
            out_dir="unittests/test-renders",
            device="cpu",
        )

        holdout_view_handler.compute_holdout_view(self.gaussian_model, 3)
        holdout_view_handler.export_gif()

        assert os.path.exists(f"{self.out_path}/training.gif")

    def tearDown(self):
        shutil.rmtree("unittests/test-renders", ignore_errors=True)