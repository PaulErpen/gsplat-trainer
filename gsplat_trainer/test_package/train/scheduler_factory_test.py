from typing import NamedTuple
import unittest
from gsplat_trainer.train.scheduler_factory import SchedulerFactory
from torch import optim
import torch.nn
import torch

class MockedConfig(NamedTuple):
    max_steps: int

class SchedulerFactoryTest(unittest.TestCase):
    def setUp(self):
        self.optimizers = {
            "means": optim.Adam(
                [
                    {
                        "params": torch.nn.Parameter(torch.rand((20,))),
                        "lr": 1.0,
                        "name": "means",
                    }
                ]
            )
        }
        self.config = MockedConfig(10000)

    def test_given_optimizers_with_a_means_optimizer__when_creating_the_schedulers__then_return_a_means_scheduler(
        self,
    ):
        schedulers = SchedulerFactory.create_schedulers(
            self.optimizers, self.config
        )

        self.assertEqual(len(schedulers), 1)
        self.assertEqual(schedulers[0].optimizer.param_groups[0]["name"], "means")
