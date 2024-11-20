import os
from pathlib import Path
import unittest

from gsplat_trainer.data.colmap.comap_dataset_factory import ColmapDatasetFactory


class ColmapDatasetFactoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset_path = f"{os.getcwd()}/mocked_datasets/backyard"

    def test_given_a_path_to_a_valid_colmap_dataset__when_initializing__do_not_throw_any_errors(
        self,
    ) -> None:
        ColmapDatasetFactory(
            data_root=self.dataset_path,
            splits=["train", "test"],
            max_num_init_points=1000,
        )

    def test_given_a_path_to_a_valid_colmap_dataset__when_initializing__the_points3d_ply_file_must_exist(
        self,
    ) -> None:
        ColmapDatasetFactory(
            data_root=self.dataset_path,
            splits=["train", "test"],
            max_num_init_points=1000,
        )

        self.assertTrue(
            Path(f"{os.getcwd()}/mocked_datasets/backyard/points3d.ply").exists()
        )

    def test_given_a_path_to_a_valid_colmap_dataset__when_initializing__then_the_dataset_must_have_a_train_split(
        self,
    ) -> None:
        colmap_datatset_factory = ColmapDatasetFactory(
            data_root=self.dataset_path,
            splits=["train", "test"],
            max_num_init_points=1000,
        )

        self.assertIsNotNone(colmap_datatset_factory.get_split("train"))

    def test_given_a_path_to_a_valid_colmap_dataset__when_initializing__then_the_dataset_must_have_a_test_split(
        self,
    ) -> None:
        colmap_datatset_factory = ColmapDatasetFactory(
            data_root=self.dataset_path,
            splits=["train", "test"],
            max_num_init_points=1000,
        )

        self.assertIsNotNone(colmap_datatset_factory.get_split("test"))

    def test_given_a_path_to_a_valid_colmap_dataset__when_initializing__then_the_train_split_must_have_the_correct_length(
        self,
    ) -> None:
        colmap_datatset_factory = ColmapDatasetFactory(
            data_root=self.dataset_path,
            splits=["train", "test"],
            max_num_init_points=1000,
        )

        self.assertEqual(len(colmap_datatset_factory.get_split("train")), 13)

    def test_given_a_path_to_a_valid_colmap_dataset__when_initializing__then_the_test_split_must_have_the_correct_length(
        self,
    ) -> None:
        colmap_datatset_factory = ColmapDatasetFactory(
            data_root=self.dataset_path,
            splits=["train", "test"],
            max_num_init_points=1000,
        )

        self.assertEqual(len(colmap_datatset_factory.get_split("test")), 2)
    
    def test_given_a_valid_colmap_dataset_factory__when_retrieving_the_first_train_datapoint__then_the_image_must_have_the_correct_dimensions(self) -> None:
        train_split = ColmapDatasetFactory(
            data_root=self.dataset_path,
            splits=["train", "test"],
            max_num_init_points=1000,
        ).get_split("train")
        pose, image, alpha, intrinsics = train_split[0]

        H, W = 2142, 2856
        self.assertEqual(image.shape, (H, W, 3))
    
    def test_given_a_valid_colmap_dataset_factory__when_retrieving_the_first_test_datapoint__then_the_image_must_have_the_correct_dimensions(self) -> None:
        test_split = ColmapDatasetFactory(
            data_root=self.dataset_path,
            splits=["train", "test"],
            max_num_init_points=1000,
        ).get_split("test")
        pose, image, alpha, intrinsics = test_split[0]

        H, W = 2142, 2856
        self.assertEqual(image.shape, (H, W, 3))

    def test_given_a_valid_colmap_dataset_factory__when_retrieving_the_first_train_datapoint__then_the_pose_must_have_the_correct_dimensions(self) -> None:
        train_split = ColmapDatasetFactory(
            data_root=self.dataset_path,
            splits=["train", "test"],
            max_num_init_points=1000,
        ).get_split("train")
        pose, image, alpha, intrinsics = train_split[0]

        self.assertEqual(pose.shape, (4, 4))
    
    def test_given_a_valid_colmap_dataset_factory__when_retrieving_the_first_test_datapoint__then_the_pose_must_have_the_correct_dimensions(self) -> None:
        test_split = ColmapDatasetFactory(
            data_root=self.dataset_path,
            splits=["train", "test"],
            max_num_init_points=1000,
        ).get_split("test")
        pose, image, alpha, intrinsics = test_split[0]

        self.assertEqual(pose.shape, (4, 4))

    def test_given_a_valid_colmap_dataset_factory__when_retrieving_the_first_train_datapoint__then_the_alpha_must_have_the_correct_dimensions(self) -> None:
        train_split = ColmapDatasetFactory(
            data_root=self.dataset_path,
            splits=["train", "test"],
            max_num_init_points=1000,
        ).get_split("train")
        pose, image, alpha, intrinsics = train_split[0]

        H, W = 2142, 2856
        self.assertEqual(alpha.shape, (H, W, 3))
    
    def test_given_a_valid_colmap_dataset_factory__when_retrieving_the_first_test_datapoint__then_the_alpha_must_have_the_correct_dimensions(self) -> None:
        test_split = ColmapDatasetFactory(
            data_root=self.dataset_path,
            splits=["train", "test"],
            max_num_init_points=1000,
        ).get_split("test")
        pose, image, alpha, intrinsics = test_split[0]

        H, W = 2142, 2856
        self.assertEqual(alpha.shape, (H, W, 3))
    
    #intrinsics
    
    def test_given_a_valid_colmap_dataset_factory__when_retrieving_the_first_train_datapoint__then_the_intrinsics_must_have_the_correct_dimensions(self) -> None:
        train_split = ColmapDatasetFactory(
            data_root=self.dataset_path,
            splits=["train", "test"],
            max_num_init_points=1000,
        ).get_split("train")
        pose, image, alpha, intrinsics = train_split[0]

        self.assertEqual(intrinsics.shape, (3, 3))
    
    def test_given_a_valid_colmap_dataset_factory__when_retrieving_the_first_test_datapoint__then_the_intrinsics_must_have_the_correct_dimensions(self) -> None:
        test_split = ColmapDatasetFactory(
            data_root=self.dataset_path,
            splits=["train", "test"],
            max_num_init_points=1000,
        ).get_split("test")
        pose, image, alpha, intrinsics = test_split[0]

        self.assertEqual(intrinsics.shape, (3, 3))

    def tearDown(self) -> None:
        os.unlink(f"{os.getcwd()}/mocked_datasets/backyard/points3d.ply")
