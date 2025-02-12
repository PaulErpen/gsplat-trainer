from gsplat_trainer.config.config import Config
from gsplat_trainer.data.data_service import DataManager
from gsplat_trainer.data.nvs_dataset import NVSDataset


def get_dataset_path(dataDir: str, dataset_name: str) -> str:
    if dataset_name == "truck":
        return f"{dataDir}/data/TNT_GOF/TrainingSet/Truck/"
    if dataset_name == "room":
        return f"{dataDir}/data/room/"
    if dataset_name == "stump":
        return f"{dataDir}/data/stump/"
    raise Exception(f'Dataset name "{dataset_name}" does not exist')


class EvalDataLoader:
    def __init__(self, data_dir: str, dataset: str) -> None:
        self.config: Config = Config()
        self.config.dataset_path = get_dataset_path(data_dir, dataset)
        self.split = None

    def get_eval_split(self) -> NVSDataset:
        if self.split is None:
            self.split = DataManager(config=self.config).get_split("test")
        return self.split
