import shutil
import unittest
from gsplat_trainer.test.holdout_view_handler import HoldoutViewHandler
import torch
from gsplat_trainer.model.gaussian_model import GaussianModel
from torch import nn
import os
from PIL import Image


class HoldoutviewHandlerTest(unittest.TestCase):
    def setUp(self) -> None:
        N = 100
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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
        ).to(self.device)
        self.out_path = "unittests/test-renders"
        os.makedirs(self.out_path, exist_ok=True)

    def test_given_a_valid_holdout_view_handler__when_computing_a_set_of_views_and_exporting__then_the_gif_must_exist(
        self,
    ) -> None:
        holdout_view_handler = HoldoutViewHandler(
            torch.eye(4),
            torch.eye(3),
            128,
            128,
            bg_color=torch.rand(3),
            out_dir="unittests/test-renders",
            device=self.device,
        )

        holdout_view_handler.compute_holdout_view(self.gaussian_model, 3)
        holdout_view_handler.export_gif()

        assert os.path.exists(f"{self.out_path}/training.gif")

    def test_given_a_valid_holdout_view_handler__when_computing_a_set_of_views_and_exporting_to_mp4__then_the_mp4_must_exist(
        self,
    ) -> None:
        holdout_view_handler = HoldoutViewHandler(
            torch.eye(4),
            torch.eye(3),
            128,
            128,
            bg_color=torch.rand(3),
            out_dir="unittests/test-renders",
            device=self.device,
        )

        holdout_view_handler.compute_holdout_view(self.gaussian_model, 3)
        holdout_view_handler.export_mp4()

        assert os.path.exists(f"{self.out_path}/training.mp4")

    def test_given_a_valid_holdout_view_handler_with_a_large_resolution__when_computing_a_set_of_views_and_exporting_to_mp4__then_the_mp4_must_exist(
        self,
    ) -> None:
        holdout_view_handler = HoldoutViewHandler(
            torch.eye(4),
            torch.eye(3),
            600,
            335,
            bg_color=torch.rand(3),
            out_dir="unittests/test-renders",
            device=self.device,
            thumbnail_size=(600, 335)
        )

        holdout_view_handler.compute_holdout_view(self.gaussian_model, 3)
        holdout_view_handler.export_mp4()

        assert os.path.exists(f"{self.out_path}/training.mp4")

    def test_given_a_valid_holdout_view_handler__when_computing_a_set_of_views_and_exporting__then_the_gif_must_have_the_correct_thumbnail_size(
        self,
    ) -> None:
        holdout_view_handler = HoldoutViewHandler(
            torch.eye(4),
            torch.eye(3),
            128,
            128,
            bg_color=torch.rand(3),
            out_dir="unittests/test-renders",
            device=self.device,
            thumbnail_size=(120, 120),
        )

        holdout_view_handler.compute_holdout_view(self.gaussian_model, 3)
        holdout_view_handler.export_gif()

        image = Image.open(f"{self.out_path}/training.gif")

        self.assertEqual(image.size, (120, 120))

    def tearDown(self) -> None:
        shutil.rmtree("unittests/test-renders", ignore_errors=True)

        if self.device == "cuda":
            torch.cuda.empty_cache()
