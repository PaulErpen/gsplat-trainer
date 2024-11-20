import unittest
from test_package.mocks.mock_factory import MockFactory
import torch


class GaussianModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.N = 10
        self.pdc_points = 40
        self.H, self.W = 128, 128
        self.dataset = MockFactory.create_mocked_nvs_dataset(
            n_entries=self.N, H=self.H, W=self.W, n_points=self.pdc_points
        )

    def test_given_a_valid_point_cloud__when_initializing_the_model__then_do_not_raise_an_error(
        self,
    ) -> None:
        self.gaussian_model = MockFactory.create_mocked_gaussian_model(
            n_points=self.pdc_points
        ).to(self.device)

    def test_given_a_valid_gaussian_model__when_moving_to_a_device__the_model_parameters_must_be_on_that_device(
        self,
    ) -> None:
        self.gaussian_model = MockFactory.create_mocked_gaussian_model(
            n_points=self.pdc_points
        ).to(self.device)
        self.assertTrue(str(self.gaussian_model.params["means"].device) == self.device)
        self.assertTrue(str(self.gaussian_model.params["quats"].device) == self.device)
        self.assertTrue(str(self.gaussian_model.params["scales"].device) == self.device)
        self.assertTrue(
            str(self.gaussian_model.params["opacities"].device) == self.device
        )
        self.assertTrue(str(self.gaussian_model.params["sh0"].device) == self.device)
        self.assertTrue(str(self.gaussian_model.params["shN"].device) == self.device)

    def test_given_a_valid_gaussian_model__when_forwarding_with_camera_parameters__then_do_not_throw_an_error(
        self,
    ) -> None:
        self.gaussian_model = MockFactory.create_mocked_gaussian_model(
            n_points=self.pdc_points
        ).to(self.device)
        pose, image, alphas, intrinsics = self.dataset[0]
        H, W, C = image.shape
        self.gaussian_model(
            view_matrix=pose.unsqueeze(0),
            K=intrinsics.unsqueeze(0),
            H=H,
            W=W,
            sh_degree_to_use=3,
            bg_color=torch.ones((3,)).unsqueeze(0),
        )

    def test_given_a_valid_gaussian_model__when_forwarding_with_camera_parameters__then_return_a_meta_dictionary_with_the_correct_keys(
        self,
    ) -> None:
        self.gaussian_model = MockFactory.create_mocked_gaussian_model(
            n_points=self.pdc_points
        ).to(self.device)
        pose, image, alphas, intrinsics = self.dataset[0]
        H, W, C = image.shape
        self.assertEqual(
            self.gaussian_model(
                view_matrix=pose.unsqueeze(0),
                K=intrinsics.unsqueeze(0),
                H=H,
                W=W,
                sh_degree_to_use=3,
                bg_color=torch.ones((3,)).unsqueeze(0),
            )[0].shape,
            (1, 128, 128, 3),
        )

    def test_given_a_valid_gaussian_model__when_forwarding_with_camera_parameters__then_return_a_meta_dictionary_with_the_right_keys(
        self,
    ) -> None:
        self.gaussian_model = MockFactory.create_mocked_gaussian_model(
            n_points=self.pdc_points
        ).to(self.device)
        self.assertCountEqual(
            list(
                self.gaussian_model(
                    torch.eye(4).unsqueeze(0),
                    torch.eye(3).unsqueeze(0),
                    128,
                    128,
                    3,
                    bg_color=torch.ones((3,)).unsqueeze(0),
                )[2].keys()
            ),
            [
                "camera_ids",
                "gaussian_ids",
                "radii",
                "means2d",
                "depths",
                "conics",
                "opacities",
                "tile_width",
                "tile_height",
                "tiles_per_gauss",
                "isect_ids",
                "flatten_ids",
                "isect_offsets",
                "width",
                "height",
                "tile_size",
                "n_cameras",
            ],
        )

    def test_given_correct_paramaters__when_initializing__the_parameters_should_have_the_correct_dimensions(
        self,
    ) -> None:
        self.gaussian_model = MockFactory.create_mocked_gaussian_model(
            n_points=self.pdc_points
        ).to(self.device)
        self.assertEqual(
            self.gaussian_model.params["means"].shape, (self.pdc_points, 3)
        )
        self.assertEqual(
            self.gaussian_model.params["scales"].shape, (self.pdc_points, 3)
        )
        self.assertEqual(
            self.gaussian_model.params["opacities"].shape, (self.pdc_points,)
        )
        self.assertEqual(
            self.gaussian_model.params["quats"].shape, (self.pdc_points, 4)
        )
        self.assertEqual(
            self.gaussian_model.params["sh0"].shape, (self.pdc_points, 1, 3)
        )
        self.assertEqual(
            self.gaussian_model.params["shN"].shape, (self.pdc_points, 15, 3)
        )

    def tearDown(self) -> None:
        if self.device == "cuda":
            torch.cuda.empty_cache()
