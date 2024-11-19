import unittest

from gsplat_trainer.config.config import Config
import torch


class ConfigTest(unittest.TestCase):
    def test_given_no_dataset_path__when_parsing_the_config_from_args__then_exit_with_an_error(
        self,
    ) -> None:
        with self.assertRaises(SystemExit):
            Config.from_cli_args([])

    def test_given_no_output_path__when_parsing_the_config_from_args__then_exit_with_an_error(
        self,
    ) -> None:
        with self.assertRaises(SystemExit):
            Config.from_cli_args(["--dataset_path", "hello"])

    def test_given_all_required_arguments__when_parsing_the_config_from_args__then_raise_no_error(
        self,
    ) -> None:
        Config.from_cli_args(
            [
                "--dataset_path",
                "hello",
                "--output_path",
                "hello",
                "--run_name",
                "run_name_1",
            ]
        )

    def test_given_the_white_bg_arg__when_parsing_the_config_from_args__the_config_bg_must_be_white(
        self,
    ) -> None:
        config = Config.from_cli_args(
            [
                "--dataset_path",
                "hello",
                "--output_path",
                "hello",
                "--run_name",
                "run_name_1",
                "--white_background",
            ]
        )

        self.assertTrue(torch.equal(config.bg_color, torch.ones((3,))))

    def test_given_args_without_the_white_bg__when_parsing_the_config_from_args__the_config_bg_must_be_black(
        self,
    ) -> None:
        config = Config.from_cli_args(
            [
                "--dataset_path",
                "hello",
                "--output_path",
                "hello",
                "--run_name",
                "run_name_1",
            ]
        )

        self.assertTrue(torch.equal(config.bg_color, torch.zeros((3,))))
