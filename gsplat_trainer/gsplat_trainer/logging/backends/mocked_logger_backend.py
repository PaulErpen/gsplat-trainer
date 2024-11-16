from dataclasses import dataclass
from typing import Dict, List
from gsplat_trainer.logging.backends.logger_backend import LoggerBackend


@dataclass
class LoggingCall:
    step: int
    metrics: Dict


class MockedLoggerBackend(LoggerBackend):
    def __init__(self) -> None:
        super().__init__()

        self.calls: List[LoggingCall] = []

    def log(self, metrics: Dict, step: int) -> None:
        self.calls.append(LoggingCall(step, metrics))

    def finish(self) -> None:
        return super().finish()
