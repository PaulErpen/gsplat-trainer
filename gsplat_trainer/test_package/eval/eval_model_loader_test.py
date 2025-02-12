import os
import shutil
import unittest
from gsplat_trainer.model_io.ply_handling import save_ply
from torch import nn
import torch
from gsplat_trainer.eval.eval_model_loader import EvalModelLoader
from gsplat_trainer.model.gaussian_model import GaussianModel


class EvalModelLoaderTest(unittest.TestCase):
    def test_given_a_nonextistent_model__when_loading_the_model__then_throw_an_exception(
        self,
    ) -> None:
        with self.assertRaises(Exception):
            EvalModelLoader("mocked_data", "mcmc", "medium", "stump", "cpu").get_model()

    def test_given_an_extistent_model__when_loading_the_model__then_return_the_model(
        self,
    ) -> None:
        DATA_DIR = "./mocked_test_data"
        METHOD = "mcmc"
        DATASET = "stump"
        SIZE = "medium"
        GS_MODEL_DIR = f"{DATA_DIR}/models/{METHOD}/{METHOD}-{DATASET}-{SIZE}-1"
        os.makedirs(GS_MODEL_DIR, exist_ok=True)
        model = self.create_gaussian_model()
        GS_MODEL_PATH = f"{GS_MODEL_DIR}/{METHOD}-{DATASET}-{SIZE}-1_model.ply"
        save_ply(
            model,
            GS_MODEL_PATH,
        )
        loaded_model = EvalModelLoader(
            DATA_DIR, METHOD, SIZE, DATASET, "cpu"
        ).get_model()
        self.assertIsInstance(loaded_model, GaussianModel)

        os.unlink(GS_MODEL_PATH)
        shutil.rmtree(DATA_DIR)

    def create_gaussian_model(self, N=100, device="cpu") -> GaussianModel:
        return GaussianModel(
            {
                "means": nn.Parameter(
                    torch.tensor(
                        torch.rand(N, 3), dtype=torch.float, device=device
                    ).requires_grad_(True)
                ),
                "sh0": nn.Parameter(
                    torch.tensor(torch.rand(N, 3, 1), dtype=torch.float, device=device)
                    .transpose(1, 2)
                    .contiguous()
                    .requires_grad_(True)
                ),
                "shN": nn.Parameter(
                    torch.tensor(torch.rand(N, 3, 15), dtype=torch.float, device=device)
                    .transpose(1, 2)
                    .contiguous()
                    .requires_grad_(True)
                ),
                "opacities": nn.Parameter(
                    torch.tensor(
                        torch.rand(
                            N,
                        ),
                        dtype=torch.float,
                        device=device,
                    ).requires_grad_(True)
                ),
                "scales": nn.Parameter(
                    torch.tensor(
                        torch.rand(N, 3), dtype=torch.float, device=device
                    ).requires_grad_(True)
                ),
                "quats": nn.Parameter(
                    torch.tensor(
                        torch.rand(N, 4), dtype=torch.float, device=device
                    ).requires_grad_(True)
                ),
            },
            scene_scale=1.0,
        )
