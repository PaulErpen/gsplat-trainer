import os
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

    def tearDown(self) -> None:
        os.unlink(f"{os.getcwd()}/mocked_datasets/backyard/points3d.ply")
