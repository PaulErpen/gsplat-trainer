import unittest

from gsplat_trainer.logging.logger_factory import LoggerFactory
from gsplat_trainer.test.validation_handler import ValidationHandler
import torch

from ..mocks.mock_factory import MockFactory


class ValidationHandlerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.N = 50
        self.H, self.W = 128, 128
        self.n_pdc_points = 20
        self.train_dataset = MockFactory.create_mocked_nvs_dataset(
            n_entries=self.N, H=self.H, W=self.W, n_points=self.n_pdc_points
        )
        self.test_dataset = MockFactory.create_mocked_nvs_dataset(
            n_entries=self.N, H=self.H, W=self.W, n_points=self.n_pdc_points
        )
        self.gaussian_model = MockFactory.create_mocked_gaussian_model(
            self.n_pdc_points
        )
        self.bg = torch.ones(3)

    def test_given_only_a_train_dataset__when_intializing_the_validation_handler__then_it_must_only_contain_a_single_validation_config(
        self,
    ) -> None:
        validation_handler = ValidationHandler(
            train_dataset=self.train_dataset,
            test_dataset=None,
            test_iterations=[1000],
            device="cpu",
            W=self.W,
            H=self.H,
            bg_color=self.bg,
            sh_degree_interval=1000,
            logger=None,
        )

        self.assertEqual(len(validation_handler.validation_configs), 1)

    def test_given_a_train_and_test_dataset__when_initializing_the_validation_handler__then_it_must_contain_two_validation_configs(
        self,
    ) -> None:
        validation_handler = ValidationHandler(
            train_dataset=self.train_dataset,
            test_dataset=self.test_dataset,
            test_iterations=[1000],
            device="cpu",
            W=self.W,
            H=self.H,
            bg_color=self.bg,
            sh_degree_interval=1000,
            logger=None,
        )

        self.assertEqual(len(validation_handler.validation_configs), 2)

    def test_given_a_logger_and_two_datasets__when_calling_the_validation_method_with_the_correct_step__then_the_logger_must_have_been_called_twice(
        self,
    ) -> None:
        logger = LoggerFactory.create_mocked_logger()
        validation_handler = ValidationHandler(
            train_dataset=self.train_dataset,
            test_dataset=self.test_dataset,
            test_iterations=[1000],
            device="cpu",
            W=self.W,
            H=self.H,
            bg_color=self.bg,
            sh_degree_interval=1000,
            logger=logger,
        )

        validation_handler.handle_validation(self.gaussian_model, 1000)

        self.assertEqual(len(logger.logger_backend.calls), 2)
    
    def test_given_a_logger_and_one_dataset__when_calling_the_validation_method_with_the_correct_step__then_the_logger_must_have_been_called_once(
        self,
    ) -> None:
        logger = LoggerFactory.create_mocked_logger()
        validation_handler = ValidationHandler(
            train_dataset=self.train_dataset,
            test_dataset=None,
            test_iterations=[1000],
            device="cpu",
            W=self.W,
            H=self.H,
            bg_color=self.bg,
            sh_degree_interval=1000,
            logger=logger,
        )

        validation_handler.handle_validation(self.gaussian_model, 1000)

        self.assertEqual(len(logger.logger_backend.calls), 1)
