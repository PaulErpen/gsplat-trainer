import sys

from gsplat_trainer.config.config import Config
from gsplat_trainer.config.export_config import export_configs
from gsplat_trainer.data.data_service import DataManager
from gsplat_trainer.data.nvs_dataset import NVSDataset
from gsplat_trainer.model_io.ply_handling import save_ply
from gsplat_trainer.loggers.logger import Logger
from gsplat_trainer.loggers.logger_factory import LoggerFactory
from gsplat_trainer.model.gaussian_model import GaussianModel
from gsplat_trainer.test.holdout_view_handler import HoldoutViewHandler
from gsplat_trainer.test.validation_handler import ValidationHandler
from gsplat_trainer.train.optimizer_factory import OptimizerFactory
from gsplat_trainer.train.scheduler_factory import SchedulerFactory
from gsplat_trainer.train.simple_trainer import SimpleTrainer
from gsplat_trainer.train.strategy.strategy_wrapper import Strategy
import torch


def create_holdout_view_handler(
    test_dataset: NVSDataset, device: str, config: Config
) -> HoldoutViewHandler:
    pose, image, alpha, intrinsics = test_dataset[0]
    H, W, C = image.shape
    return HoldoutViewHandler(
        holdout_view_matrix=pose,
        K=intrinsics,
        W=W,
        H=H,
        out_dir=config.output_path,
        bg_color=config.bg_color,
        device=device,
    )


if __name__ == "__main__":
    args = sys.argv

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device {device}")

    config: Config = Config.from_cli_args(args[1:])

    data_manager = DataManager(config=config)

    train_split = data_manager.get_split("train")
    test_split = data_manager.get_split("test")

    holdout_view_handler = create_holdout_view_handler(test_split, device, config)

    initial_model = GaussianModel.from_point_cloud(
        pcd=train_split.pcd,
        scene_scale=train_split.norm.radius,
        sh_degree=config.sh_degree,
    )

    strategy = Strategy(config=config)

    optimizers = OptimizerFactory.create_optimizers(initial_model)
    schedulers = SchedulerFactory.create_schedulers(
        optimizers=optimizers, config=config
    )

    logger: Logger = LoggerFactory.create_wandb_logger(
        config.wandb_project_name, config.run_name
    )

    try:
        validation_handler = ValidationHandler(
            train_dataset=train_split,
            test_dataset=test_split,
            test_iterations=config.test_iterations,
            device=device,
            bg_color=config.bg_color,
            sh_degree_interval=config.sh_degree_interval,
            logger=logger,
        )

        simple_trainer = SimpleTrainer(
            train_dataset=train_split,
            test_dataset=test_split,
            gaussian_model=initial_model,
            strategy=strategy,
            optimizers=optimizers,
            schedulers=schedulers,
            config=config,
            holdout_view_handler=holdout_view_handler,
            logger=logger,
            validation_handler=validation_handler,
            device=device,
        )

        export_configs(config.output_path, config=config, strategy=strategy)

        final_model = simple_trainer.train()

        save_ply(final_model, f"{config.output_path}/final_model.ply")

    finally:
        logger.finish()
