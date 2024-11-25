from typing import Dict, List
from gsplat_trainer.config.config import Config
import torch
from torch.optim.lr_scheduler import LRScheduler


class SchedulerFactory:
    @staticmethod
    def create_schedulers(optimizers: Dict, config: Config) -> List[LRScheduler]:
        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                optimizers["means"], gamma=0.01 ** (1.0 / config.max_steps)
            )
        ]
        return schedulers
