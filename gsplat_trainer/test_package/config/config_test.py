from dataclasses import dataclass
from typing import Any, Dict, List
import unittest

from gsplat_trainer.config.config import Config
import torch


@dataclass
class ClassValueParamsAndResult:
    class_member: str
    params: List[str]
    result: Any


class ConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.class_member_to_params_mapping: Dict[str, ClassValueParamsAndResult] = {
            "dataset_path": ClassValueParamsAndResult(
                "dataset_path",
                ["--dataset_path", "test_dataset_path"],
                "test_dataset_path",
            ),
            "output_path": ClassValueParamsAndResult(
                "output_path", ["--output_path", "test_output_path"], "test_output_path"
            ),
            "init_num_gaussians": ClassValueParamsAndResult(
                "init_num_gaussians", ["--init_num_gaussians", "123456"], 123456
            ),
            "sh_degree": ClassValueParamsAndResult(
                "sh_degree", ["--sh_degree", "2"], 2
            ),
            "packed": ClassValueParamsAndResult("packed", ["--packed"], True),
            "bg_color": ClassValueParamsAndResult(
                "bg_color", ["--white_background"], torch.ones((3,))
            ),
            "image_downscale": ClassValueParamsAndResult(
                "image_downscale", ["--image_downscale", "8"], 8
            ),
            "holdout_view_index": ClassValueParamsAndResult(
                "holdout_view_index", ["--holdout_view_index", "10"], 10
            ),
            "strategy_type": ClassValueParamsAndResult(
                "strategy_type", ["--strategy_type", "mcmc"], "mcmc"
            ),
            "max_steps": ClassValueParamsAndResult(
                "max_steps", ["--max_steps", "128"], 128
            ),
            "opacity_reg": ClassValueParamsAndResult(
                "opacity_reg", ["--opacity_reg", "0.123"], 0.123
            ),
            "scale_reg": ClassValueParamsAndResult(
                "scale_reg", ["--scale_reg", "0.456"], 0.456
            ),
            "holdout_view_frequency": ClassValueParamsAndResult(
                "holdout_view_frequency", ["--holdout_view_frequency", "2918"], 2918
            ),
            "ssim_lambda": ClassValueParamsAndResult(
                "ssim_lambda", ["--ssim_lambda", "0.23142"], 0.23142
            ),
            "test_iterations": ClassValueParamsAndResult(
                "test_iterations",
                ["--test_iterations", "100", "300", "500"],
                [100, 300, 500],
            ),
            "verbose": ClassValueParamsAndResult("verbose", ["--verbose"], True),
            "cap_max": ClassValueParamsAndResult(
                "cap_max", ["--cap_max", "12301"], 12301
            ),
            "noise_lr": ClassValueParamsAndResult(
                "noise_lr", ["--noise_lr", "0.9287"], 0.9287
            ),
            "refine_start_iter": ClassValueParamsAndResult(
                "refine_start_iter", ["--refine_start_iter", "192931"], 192931
            ),
            "refine_stop_iter": ClassValueParamsAndResult(
                "refine_stop_iter", ["--refine_stop_iter", "89218"], 89218
            ),
            "refine_every": ClassValueParamsAndResult(
                "refine_every", ["--refine_every", "9238"], 9238
            ),
            "min_opacity": ClassValueParamsAndResult(
                "min_opacity", ["--min_opacity", "0.12376"], 0.12376
            ),
            "sh_degree_interval": ClassValueParamsAndResult(
                "sh_degree_interval", ["--sh_degree_interval", "889231"], 889231
            ),
            "reset_every": ClassValueParamsAndResult(
                "reset_every", ["--reset_every", "43028"], 43028
            ),
            "wandb_project_name": ClassValueParamsAndResult(
                "wandb_project_name",
                ["--wandb_project_name", "test_wandb_project_name"],
                "test_wandb_project_name",
            ),
            "run_name": ClassValueParamsAndResult(
                "run_name", ["--run_name", "test_run_name"], "test_run_name"
            ),
        }

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

    def test_given_a_list_of_class_members__when_comparing_to_the_config_classes_anotations__then_they_must_match(
        self,
    ) -> None:
        class_members = list(self.class_member_to_params_mapping.keys())

        not_found: List[str] = []

        for annotation in Config.__annotations__:
            if not annotation in class_members:
                not_found.append(annotation)

        self.assertSequenceEqual(not_found, [])

    def test_given_a_list_of_parameters_class_names_and_result_values__when__feeding_the_parameters_to_the_factory_method__the_resulting_config_must_contain_the_result_values(
        self,
    ) -> None:
        collected_params = []

        for class_member in self.class_member_to_params_mapping:
            collected_params.extend(
                self.class_member_to_params_mapping[class_member].params
            )

        config = Config.from_cli_args(collected_params)

        for class_member in self.class_member_to_params_mapping:
            attr = getattr(config, class_member)

            if isinstance(attr, torch.Tensor):
                self.assertTrue(
                    torch.equal(
                        attr, self.class_member_to_params_mapping[class_member].result
                    )
                )
            else:
                self.assertEqual(
                    attr,
                    self.class_member_to_params_mapping[class_member].result,
                    f'Failed for parameter "{class_member}"',
                )
