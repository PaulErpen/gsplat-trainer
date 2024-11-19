import unittest

from gsplat_trainer.config.config import Config
from gsplat_trainer.train.optimizer_factory import OptimizerFactory
from gsplat_trainer.train.scheduler_factory import SchedulerFactory
from gsplat_trainer.train.simple_trainer import SimpleTrainer
from gsplat_trainer.train.strategy.strategy_wrapper import Strategy
from test_package.mocks.mock_factory import MockFactory
import torch


class SimpleTrainerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.train_dataset = MockFactory.create_mocked_nvs_dataset()
        self.test_dataset = MockFactory.create_mocked_nvs_dataset()
        self.gaussian_model = MockFactory.create_mocked_gaussian_model()
        self.config = Config()
        self.strategy = Strategy(self.config)
        self.optimizers = OptimizerFactory.create_optimizers(self.gaussian_model)
        self.lr_schedulers = SchedulerFactory.create_schedulers(
            self.optimizers, self.config
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config.max_steps = 10

    @unittest.skip(reason="debug")
    def test_given_a_simple_trainer_with_all_needed_parameters__when_initializing__then_do_not_throw_an_error(
        self,
    ) -> None:
        SimpleTrainer(
            train_dataset=self.train_dataset,
            test_dataset=None,
            gaussian_model=self.gaussian_model,
            strategy=self.strategy,
            optimizers=self.optimizers,
            schedulers=self.lr_schedulers,
            config=self.config,
            holdout_view_handler=None,
            logger=None,
            validation_handler=None,
            device=self.device,
        )
