from typing import Dict, List
import torch

class SchedulerFactory:
  @staticmethod
  def create_schedulers(optimizers: Dict, config: "Config") -> List:
    schedulers = [
        torch.optim.lr_scheduler.ExponentialLR(
          optimizers["means"], gamma=0.01 ** (1.0 / config.max_steps)
        )
    ]
    return schedulers