from typing import Dict

from gsplat_trainer.loggers.backends.logger_backend import LoggerBackend


class Logger:
    def __init__(self,
                 logger_backend: LoggerBackend):
        self.logger_backend = logger_backend

    def log(self, dict: Dict, iteration: int):
        self.logger_backend.log(dict, iteration)

    def finish(self) -> None:
        self.logger_backend.finish()