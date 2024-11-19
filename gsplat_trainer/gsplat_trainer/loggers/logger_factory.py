from gsplat_trainer.loggers.backends.mocked_logger_backend import MockedLoggerBackend
from gsplat_trainer.loggers.logger import Logger

from gsplat_trainer.loggers.backends.wandb_logger_backend import WandBLoggerBackend


class LoggerFactory:
    @staticmethod
    def create_wandb_logger(
        wandb_project: str,
        wandb_run_name: str,
    ) -> Logger:
        wandb_backend = WandBLoggerBackend(wandb_project, wandb_run_name)
        return Logger(logger_backend=wandb_backend)

    @staticmethod
    def create_mocked_logger() -> Logger:
        return Logger(MockedLoggerBackend())
