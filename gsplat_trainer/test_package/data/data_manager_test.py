import os
import unittest

from gsplat_trainer.config.config import Config
from gsplat_trainer.data.data_service import DataManager


class DataManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config()

    def test_given_a_valid_blender_dataset_path__when_retrieving_the_train_split__then_do_not_return_none(
        self,
    ) -> None:
        self.config.dataset_path = f"{os.getcwd()}/mocked_datasets/lego"
        data_manager = DataManager(self.config)

        self.assertIsNotNone(data_manager.get_split("train"))

    def test_given_a_valid_blender_dataset_path__when_retrieving_the_test_split__then_do_not_return_none(
        self,
    ) -> None:
        self.config.dataset_path = f"{os.getcwd()}/mocked_datasets/lego"
        data_manager = DataManager(self.config)

        self.assertIsNotNone(data_manager.get_split("test"))

    def test_given_a_valid_blender_dataset_path__when_retrieving_the_val_split__then_raise_an_exception(
        self,
    ) -> None:
        self.config.dataset_path = f"{os.getcwd()}/mocked_datasets/lego"
        data_manager = DataManager(self.config)

        with self.assertRaises(Exception):
            data_manager.get_split("val")
    
    def test_given_an_invalid_dataset_path__when_retrieving_the_train_split__then_raise_an_exception(
        self,
    ) -> None:
        self.config.dataset_path = f"{os.getcwd()}/mocked_datasets/nonexistent"
        data_manager = DataManager(self.config)

        with self.assertRaises(Exception):
            data_manager.get_split("train")
