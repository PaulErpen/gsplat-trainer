import unittest
from gsplat_trainer.test.early_stopping_handler import EarlyStoppingHandler
from test_package.mocks.mock_factory import MockFactory
import torch


class EarlyStoppingHandlerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.test_dataset = MockFactory.create_mocked_nvs_dataset(n_entries=3)
        self.gaussian_model = MockFactory.create_mocked_gaussian_model(
            self.test_dataset.pcd.points.shape[0]
        ).to(self.device)
        self.bg_color = torch.zeros((3))

    def test_given_valid_parameters__when_initializing_the_early_stopping_handler__then_do_not_raise_an_error(
        self,
    ) -> None:
        self.handler = EarlyStoppingHandler(
            test_dataset=self.test_dataset,
            n_patience_epochs=10,
            sh_degree_interval=1000,
            bg_color=self.bg_color,
            device=self.device,
        )

    def test_given_a_patience_of_10___when__calling_the_epoch_continuation__then_do_not_throw_an_error(
        self,
    ) -> None:
        self.handler = EarlyStoppingHandler(
            test_dataset=self.test_dataset,
            n_patience_epochs=10,
            sh_degree_interval=1000,
            bg_color=self.bg_color,
            device=self.device,
        )
        self.handler.check_continue_at_current_epoch(self.gaussian_model, 50)

    def test_given_a_patience_of_10___when__calling_the_epoch_continuation__then_return_true(
        self,
    ) -> None:
        self.handler = EarlyStoppingHandler(
            test_dataset=self.test_dataset,
            n_patience_epochs=10,
            sh_degree_interval=1000,
            bg_color=self.bg_color,
            device=self.device,
        )
        self.assertTrue(
            self.handler.check_continue_at_current_epoch(self.gaussian_model, 50)
        )

    def test_given_a_patience_of_1___when__calling_the_epoch_continuation_many_times__then_eventually_return_false(
        self,
    ) -> None:
        self.handler = EarlyStoppingHandler(
            test_dataset=self.test_dataset,
            n_patience_epochs=1,
            sh_degree_interval=1000,
            bg_color=self.bg_color,
            device=self.device,
        )

        returned_only_true = True

        for i in range(20):
            returned_value = self.handler.check_continue_at_current_epoch(
                self.gaussian_model, i
            )
            if not returned_value:
                returned_only_true = False

        self.assertFalse(returned_only_true)
