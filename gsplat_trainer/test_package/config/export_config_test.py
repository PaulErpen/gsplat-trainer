from pathlib import Path
import shutil
import unittest

from gsplat_trainer.config.config import Config
from gsplat_trainer.config.export_config import export_configs
from gsplat_trainer.train.strategy.strategy_wrapper import Strategy


class ExportConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config()
        self.strategy = Strategy(self.config, 123.4)
        self.export_path = "out_dir"

    def test_given_a_config_and_a_strategy__when_exporting__then_the_export_file_must_exist(
        self,
    ) -> None:
        export_configs(
            out_dir=self.export_path, config=self.config, strategy=self.strategy
        )

        self.assertTrue(Path(f"{self.export_path}/config.json").exists())

    def tearDown(self) -> None:
        shutil.rmtree(self.export_path)
