import unittest

from gsplat_trainer.model.gaussian_model import GaussianModel
from gsplat_trainer.train.optimizer_factory import OptimizerFactory
from torch import nn
import torch


class OptimizerFactoryTest(unittest.TestCase):
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

    def test_given_a_gaussian_model__when_creating_the_optimizers__then_return_a_dict_with_the_correct_amount_of_keys(
        self,
    ):
        optimizers = OptimizerFactory.create_optimizers(self.gaussian_model)

        self.assertEqual(len(optimizers), len(self.gaussian_model.params))

    def test_given_a_gaussian_model__when_creating_the_optimizers__then_return_a_dict_with_the_correct_sequence_of_keys(
        self,
    ):
        optimizers = OptimizerFactory.create_optimizers(self.gaussian_model)

        self.assertSequenceEqual(
            list(optimizers.keys()),
            [
                "means",
                "opacities",
                "quats",
                "scales",
                "sh0",
                "shN",
            ],
        )
