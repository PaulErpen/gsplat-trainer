from pathlib import Path
from typing import Dict
from gsplat_trainer.logging.backends.logger_backend import LoggerBackend

try:
    import wandb

    WANDB_AVAILABLE = True
except:
    WANDB_AVAILABLE = False


class WandBLoggerBackend(LoggerBackend):
    def __init__(
        self,
        wandb_project: str,
        wandb_run_name: str,
        wandb_key_file_path: str,
    ) -> None:
        if not WANDB_AVAILABLE:
            raise Exception("WandB backend isn't available")
        super().__init__()

        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name

        if not Path(wandb_key_file_path).exists():
            raise Exception(f'no wandb key file available at "{wandb_key_file_path}"')

        with open(wandb_key_file_path, "r") as f:
            self.wandb_key = f.read().strip()

        wandb.login(key=self.wandb_key)
        self.run = wandb.init(
            project=self.wandb_project, name=self.wandb_run_name, force=True
        )

    def log(self, metrics: Dict, step: int) -> None:
        self.run.log(data=metrics, step=step)

    def finish(self) -> None:
        self.run.finish()
