from pathlib import Path
from typing import Optional

from gsplat_trainer.config.config import Config
from gsplat_trainer.data.blender.blender_synthetic_dataset_factory import (
    BlenderSyntheticDatasetFactory,
)
from gsplat_trainer.data.dataset_factory import DatasetFactory
from gsplat_trainer.data.nvs_dataset import NVSDataset


class DataManager:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.invalid_reason = ""
        self.factory: Optional[DatasetFactory] = None

        if not Path(self.config.dataset_path).exists():
            self.invalid_reason = "The dataset path does not exist"
            return

        if ((Path(self.config.dataset_path) / Path(f"transforms_train.json"))).exists():
            self.factory = BlenderSyntheticDatasetFactory(
                data_root=self.config.dataset_path,
                splits=["train", "test"],
                max_num_init_points=config.init_num_gaussians,
            )
            return

        if self.factory is None:
            self.invalid_reason = "Dataset not recognized"
            return

    def get_split(self, split: str) -> NVSDataset:
        if self.factory is None:
            raise Exception(
                f'Trying to access dataset that could not be initialized. Reason: "{self.invalid_reason}"'
            )
        split = self.factory.get_split(split)
        if split is None:
            raise Exception(
                f'Error, dataset at path "{self.config.dataset_path}" does not have a split "{split}"'
            )
        return split
