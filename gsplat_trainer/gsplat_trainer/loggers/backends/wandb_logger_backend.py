import os
from pathlib import Path
from typing import Dict
from gsplat_trainer.loggers.backends.logger_backend import LoggerBackend

try:
    import wandb

    WANDB_AVAILABLE = True
except:
    WANDB_AVAILABLE = False

WANDB_API_KEY_ENV_VAR = "WANDB_API_KEY"


class WandBLoggerBackend(LoggerBackend):
    def __init__(
        self,
        wandb_project: str,
        wandb_run_name: str,
    ) -> None:
        if not WANDB_AVAILABLE:
            raise Exception("WandB backend isn't available")
        super().__init__()

        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name

        if os.environ.get(WANDB_API_KEY_ENV_VAR) is None:
            raise Exception(
                f'no wandb key found in environment "{WANDB_API_KEY_ENV_VAR}"'
            )

        self.wandb_key = os.environ.get(WANDB_API_KEY_ENV_VAR)

        wandb.login(key=self.wandb_key)
        self.run = wandb.init(
            project=self.wandb_project, name=self.wandb_run_name, force=True
        )

    def log(self, metrics: Dict, step: int) -> None:
        self.run.log(data=metrics, step=step)

    def finish(self) -> None:
        self.run.finish()
