import unittest
from gsplat_trainer.data.basicpointcloud import BasicPointCloud
from gsplat_trainer.data.nerfnorm import NerfNorm
from gsplat_trainer.model.gaussian_model import GaussianModel
import torch
import numpy as np
from gsplat_trainer.data.nvs_dataset import NVSDataset


class GaussianModelTest(unittest.TestCase):
    def setUp(self):
        self.N = 100
        self.H, self.W = 800, 800
        self.dataset = NVSDataset(
            poses=torch.rand((self.N, 4, 4)),
            images=torch.rand((self.N, self.H, self.W, 3)),
            intrinsics=torch.rand((self.N, 3, 3)),
            pcd=BasicPointCloud(
                np.random.rand(100, 3), np.random.rand(100, 3), np.random.rand(100, 3)
            ),
            norm=NerfNorm(
                torch.rand(
                    3,
                ),
                3.1,
            ),
        )
        self.gaussian_model = GaussianModel.from_point_cloud(
            self.dataset.pcd, self.dataset.norm.radius
        )

    def test_given_a_valid_gaussian_model__when_forwarding_with_camera_parameters__then_return_a_meta_dictionary_with_the_correct_keys(
        self,
    ):

        pose, image, intrinsics = self.dataset[0]
        H, W, C = image.shape
        self.assertEqual(
            self.gaussian_model(
                view_matrix=pose.unsqueeze(0),
                K=intrinsics.unsqueeze(0),
                H=H,
                W=W,
                sh_degree_to_use=3,
                bg_color=torch.ones(1, 3),
            )[0].shape,
            (1, 800, 800, 3),
        )

    def test_given_a_valid_gaussian_model__when_forwarding_with_camera_parameters__then_return_an_image_batch_of_the_correct_dimensions(
        self,
    ):
        self.assertSequenceEqual(
            self.gaussian_model(
                torch.eye(4).unsqueeze(0).to("cuda"),
                torch.eye(3).unsqueeze(0).to("cuda"),
                800,
                800,
                3,
                torch.ones(1, 3).to("cuda"),
            )[2].keys(),
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
    ):
        self.assertEqual(self.gaussian_model.params["means"].shape, (100, 3))
        self.assertEqual(self.gaussian_model.params["scales"].shape, (100, 3))
        self.assertEqual(self.gaussian_model.params["opacities"].shape, (100,))
        self.assertEqual(self.gaussian_model.params["quats"].shape, (100, 4))
        self.assertEqual(self.gaussian_model.params["sh0"].shape, (100, 1, 3))
        self.assertEqual(self.gaussian_model.params["shN"].shape, (100, 15, 3))
