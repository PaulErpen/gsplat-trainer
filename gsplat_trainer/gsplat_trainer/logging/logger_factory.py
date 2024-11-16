from gsplat_trainer.logging.backends.mocked_logger_backend import MockedLoggerBackend
from gsplat_trainer.logging.logger import Logger

from gsplat_trainer.logging.backends.wandb_logger_backend import WandBLoggerBackend


class LoggerFactory:
    @staticmethod
    def create_wandb_logger(
        wandb_project: str,
        wandb_run_name: str,
        wandb_key_file_path: str,
    ) -> Logger:
        wandb_backend = WandBLoggerBackend(
            wandb_project, wandb_run_name, wandb_key_file_path
        )
        return Logger(logger_backend=wandb_backend)

    @staticmethod
    def create_mocked_logger() -> Logger:
        return Logger(MockedLoggerBackend())