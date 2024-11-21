import argparse
from dataclasses import dataclass, field
from typing import List, Literal, Sequence
import torch


@dataclass
class Config:
    # PATHS
    # the path where the dataset is located
    dataset_path: str = ""
    # the path where the model will be output
    output_path: str = ""

    # INITIALIZATION
    # Initial number of Gaussians
    init_num_gaussians: int = 2000
    # The number of spherical harmonics degrees to compute the color
    sh_degree: int = 3

    # RENDERING
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # The background color to use to render the splats (is also pasted "under" the train and test images)
    bg_color: torch.tensor = field(default_factory=lambda: torch.ones((3,)))
    # the downscale factor for the input image resolution
    image_downscale: Literal[1, 2, 4, 8] = 1
    # the test set index for the holdout view
    holdout_view_index: int = 0

    # TRAINING
    # Strategy to use for densifying the gaussians
    strategy_type: Literal["mcmc", "default"] = "mcmc"
    # Number of training steps
    max_steps: int = 10_000
    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0
    # the frequency with which holdout views are created
    holdout_view_frequency: int = 100
    # the amount of ssim included in the loss
    ssim_lambda: float = 0.2
    # the exact iterations when the testing loop is supposed to be executed
    test_iterations: List[int] = field(
        default_factory=lambda: [1, 500, 1000, 5000, 7500, 9000, 10000]
    )
    # Wether to print verbose info for the strategy
    verbose: bool = False
    # maximum cap for the number of gaussians
    cap_max: int = 18_000
    # MCMC samping noise learning rate. Default to 5e5.
    noise_lr: float = 5e5
    # Start refining GSs after this iteration. Default to 500.
    refine_start_iter: int = 500
    # Stop refining GSs after this iteration. Default to 25_000.
    refine_stop_iter: int = 25_000
    # Refine GSs every this steps. Default to 100.
    refine_every: int = 100
    # GSs with opacity below this value will be pruned. Default to 0.005.
    min_opacity: float = 0.005
    # Interval with which the spherical harmonics degree is increased to reach the maximum
    sh_degree_interval: int = 1000

    # DEFAULT STRATEGY
    # Reset opacities every this steps. Default is 3001
    reset_every: int = 3001

    # LOGGING
    # the project in wandb the run is gonna be logged under
    wandb_project_name: str = "gs-on-a-budget"
    # the run name for this praticular traning instance (should ideally be unique)
    run_name: str = ""

    @classmethod
    def from_cli_args(cls, args: Sequence[str]) -> "Config":
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--dataset_path",
            type=str,
            required=True,
            help="the path where the dataset is located",
        )
        parser.add_argument(
            "--output_path",
            type=str,
            required=True,
            help="the path where the model will be output",
        )
        parser.add_argument(
            "--init_num_gaussians",
            type=int,
            default=2000,
            help="Initial number of Gaussians",
        )
        parser.add_argument(
            "--sh_degree",
            type=int,
            default=3,
            help="The number of spherical harmonics degrees to compute the color",
        )
        parser.add_argument(
            "--packed_rasterization",
            default=False,
            action="store_true",
            help="Use packed mode for rasterization, this leads to less memory usage but slightly slower",
        )
        parser.add_argument(
            "--white_background",
            default=False,
            action="store_true",
            help='The background color to use to render the splats (is also pasted "under" the train and test images)',
        )
        parser.add_argument(
            "--image_downscale",
            default=1,
            type=int,
            choices=[1, 2, 4, 8],
            help="The downscale factor for the input image resolution",
        )
        parser.add_argument(
            "--holdout_view_index",
            default=0,
            type=int,
            help="the test set index for the holdout view",
        )
        parser.add_argument(
            "--strategy_type",
            type=str,
            default="mcmc",
            help="Strategy to use for densifying the gaussians",
            choices=["mcmc", "default"],
        )
        parser.add_argument(
            "--max_steps",
            type=int,
            default=10_000,
            help="Number of training steps",
        )
        parser.add_argument(
            "--opacity_reg",
            type=float,
            default=0.0,
            help="Opacity regularization",
        )
        parser.add_argument(
            "--scale_reg",
            type=float,
            default=0.0,
            help="Scale regularization",
        )
        parser.add_argument(
            "--holdout_view_frequency",
            type=int,
            default=100,
            help="The frequency with which holdout views are created",
        )
        parser.add_argument(
            "--ssim_lambda",
            type=float,
            default=0.2,
            help="the amount of ssim included in the loss",
        )
        parser.add_argument(
            "--test_iterations",
            type=List,
            default=[1, 500, 1000, 5000, 7500, 9000, 10000],
            nargs="+",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            default=False,
            help="Wether to print verbose info for the strategy",
        )
        parser.add_argument(
            "--cap_max",
            type=int,
            default=18_000,
            help="maximum cap for the number of gaussians",
        )
        parser.add_argument(
            "--noise_lr",
            type=float,
            default=5e5,
            help="MCMC samping noise learning rate. Default to 5e5.",
        )
        parser.add_argument(
            "--refine_start_iter",
            type=int,
            default=500,
            help="Start refining GSs after this iteration. Default to 500.",
        )
        parser.add_argument(
            "--refine_stop_iter",
            type=int,
            default=25_000,
            help="Stop refining GSs after this iteration. Default to 25_000.",
        )
        parser.add_argument(
            "--refine_every",
            type=int,
            default=100,
            help="Refine GSs every this steps. Default to 100.",
        )
        parser.add_argument(
            "--min_opacity",
            type=float,
            default=0.005,
            help="GSs with opacity below this value will be pruned. Default to 0.005.",
        )
        parser.add_argument(
            "--sh_degree_interval",
            type=int,
            default=1000,
            help="Interval with which the spherical harmonics degree is increased to reach the maximum",
        )
        parser.add_argument(
            "--reset_every",
            type=int,
            default=3001,
            help="Reset opacities every this steps. Default is 3001",
        )
        parser.add_argument(
            "--wandb_project_name",
            type=str,
            default="gs-on-a-budget",
            help="the project in wandb the run is gonna be logged under",
        )
        parser.add_argument(
            "--run_name",
            type=str,
            default="",
            help="the run name for this praticular traning instance (should ideally be unique)",
            required=True,
        )

        parsed_args = parser.parse_args(args)
        config: "Config" = cls()

        config.dataset_path = parsed_args.dataset_path
        config.output_path = parsed_args.output_path
        config.init_num_gaussians = parsed_args.init_num_gaussians
        config.sh_degree = parsed_args.sh_degree
        config.packed = parsed_args.packed_rasterization
        config.bg_color = (
            torch.ones((3,)) if parsed_args.white_background else torch.zeros((3,))
        )
        config.strategy_type = parsed_args.strategy_type
        config.max_steps = parsed_args.max_steps
        config.opacity_reg = parsed_args.opacity_reg
        config.scale_reg = parsed_args.scale_reg
        config.holdout_view_frequency = parsed_args.holdout_view_frequency
        config.ssim_lambda = parsed_args.ssim_lambda
        config.test_iterations = parsed_args.test_iterations
        config.verbose = parsed_args.verbose
        config.cap_max = parsed_args.cap_max
        config.noise_lr = parsed_args.noise_lr
        config.refine_start_iter = parsed_args.refine_start_iter
        config.refine_stop_iter = parsed_args.refine_stop_iter
        config.refine_every = parsed_args.refine_every
        config.min_opacity = parsed_args.min_opacity
        config.sh_degree_interval = parsed_args.sh_degree_interval
        config.wandb_project_name = parsed_args.wandb_project_name
        config.run_name = parsed_args.run_name
        config.image_downscale = parsed_args.image_downscale
        parser.holdout_view_index = parsed_args.holdout_view_index

        return config
