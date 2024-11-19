from abc import ABC

from gsplat_trainer.data.nvs_dataset import NVSDataset


class DatasetFactory(ABC):
    def get_split(split: str) -> NVSDataset:
        raise NotImplementedError()
