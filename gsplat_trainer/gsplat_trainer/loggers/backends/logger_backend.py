from abc import ABC
from typing import Dict


class LoggerBackend(ABC):
    def log(self, metrics: Dict, step: int) -> None:
        pass

    def finish(self) -> None:
        pass