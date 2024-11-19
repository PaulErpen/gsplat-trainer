import unittest

from gsplat_trainer.data.blender.blender_synthetic_dataset_factory import (
    BlenderSyntheticDatasetFactory,
)
from gsplat_trainer.data.nvs_dataset import NVSDataset
import os
import numpy as np


class BlenderSyntheticDatasetFactoryTest(unittest.TestCase):
    def setUp(self):
        self.dataset_factory = BlenderSyntheticDatasetFactory(
            data_root=f"{os.getcwd()}/mocked_datasets/lego",
            splits=["train"],
            max_num_init_points=100,
        )

    def test_given_a_valid_dataset_path__when_retrieving_the_first_observation__the_poses_must_have_the_correct_shape(
        self,
    ) -> None:
        dataset: NVSDataset = self.dataset_factory.get_split("train")

        self.assertEqual(dataset[0][0].shape, (4, 4))

    def test_given_a_valid_dataset_path__when_retrieving_the_first_observation__the_image_must_have_the_correct_shape(
        self,
    ) -> None:
        dataset: NVSDataset = self.dataset_factory.get_split("train")

        self.assertEqual(dataset[0][1].shape, (800, 800, 3))

    def test_given_a_valid_dataset_path__when_retrieving_the_first_observation__the_alphas_must_have_the_correct_shape(
        self,
    ) -> None:
        dataset: NVSDataset = self.dataset_factory.get_split("train")

        self.assertEqual(dataset[0][2].shape, (800, 800, 1))

    def test_given_a_valid_dataset_path__when_retrieving_the_first_observation__the_intrinsics_must_have_the_correct_shape(
        self,
    ) -> None:
        dataset: NVSDataset = self.dataset_factory.get_split("train")

        self.assertEqual(dataset[0][3].shape, (3, 3))

    def test_given_a_dataset_that_subsamples_the_initial_point_cloud__then_the_point_cloud_should_only_contain_the_number_of_specified_points(
        self,
    ) -> None:
        dataset: NVSDataset = self.dataset_factory.get_split("train")

        self.assertEqual(dataset.pcd.points.shape[0], 100)
        self.assertEqual(dataset.pcd.colors.shape[0], 100)
        self.assertEqual(dataset.pcd.normals.shape[0], 100)

    def test_given_a_dataset__when_getting_the_split__then_the_norm_should_have_legitimate_values(
        self,
    ) -> None:
        dataset: NVSDataset = self.dataset_factory.get_split("train")

        self.assertIsNotNone(dataset.norm)
        self.assertEqual(dataset.norm.translation.shape, (3,))
        self.assertIsInstance(dataset.norm.radius, np.float32)
