import unittest
import os
from pathlib import Path

from gsplat_trainer.io.ply_handling import load_ply, save_ply
from test_package.mocks.mock_factory import MockFactory


class PlyHandlingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.gaussian_model = MockFactory.create_mocked_gaussian_model()
        self.temp_file_path = "./unittests/temporary_unittest_file.ply"

    def test_given_an_initial_gaussian_model__when_saving_it_as_a_ply_file__the_ply_file_should_exist(
        self,
    ) -> None:
        save_ply(self.gaussian_model, self.temp_file_path)

        self.assertTrue(Path(self.temp_file_path).exists())

    def test_given_a_saved_gaussian_model__when_loading_it_from_a_file__the_resulting_model_should_be_valid(
        self,
    ) -> None:
        save_ply(self.gaussian_model, self.temp_file_path)
        gaussian_model = load_ply(self.temp_file_path, scene_scale=0.123, device="cpu")

        self.assertEqual(gaussian_model.params["means"].shape, (20, 3))
        self.assertEqual(gaussian_model.params["quats"].shape, (20, 4))
        self.assertEqual(gaussian_model.params["scales"].shape, (20, 3))
        self.assertEqual(gaussian_model.params["sh0"].shape, (20, 1, 3))
        self.assertEqual(gaussian_model.params["opacities"].shape, (20,))
        self.assertEqual(gaussian_model.params["shN"].shape, (20, 15, 3))
        self.assertEqual(gaussian_model.sh_degree, 3)

    def tearDown(self) -> None:
        os.unlink(self.temp_file_path)
