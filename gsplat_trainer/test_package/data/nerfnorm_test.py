import unittest
from gsplat_trainer.data.blender.blender_util import transform_matrix_to_w2c
from test_package.data.nerf_lego_train_transforms import nerf_lego_train_transforms
import torch
import numpy as np
from gsplat_trainer.data.nerfnorm import NerfNorm


class NerfNormTest(unittest.TestCase):
    def setUp(self):
        self.c2w_offset_3 = torch.eye(4).repeat(1000, 1, 1)
        self.c2w_offset_3[:, :1, 3] = (
            torch.rand_like(self.c2w_offset_3[:, :1, 3]) - 0.5
        ) * 6

    def test_given_a_world_to_cam_stack__when_computing_the_norm__then_the_shape_of_the_center_must_be_correct(
        self,
    ) -> None:
        norm = NerfNorm.from_w2c_stack(self.c2w_offset_3)

        self.assertEqual(norm.translation.shape, (3,))

    def test_given_a_world_to_cam_stack__when_computing_the_norm__then_the_radius_must_be_positive(
        self,
    ) -> None:
        norm = NerfNorm.from_w2c_stack(self.c2w_offset_3)

        self.assertTrue(norm.radius > 0)

    def test_given_a_random_world_to_cam_stack__when_offsetting_the_center_by_3__then_the_radius_must_be_roughly_3(
        self,
    ) -> None:
        norm = NerfNorm.from_w2c_stack(self.c2w_offset_3)

        self.assertAlmostEqual(norm.radius, 3.0, delta=0.1)

    def test_given_the_cam_2_worlds_of_the_lego_nerf_dataset__when_creating_the_norm__it_must_equal(
        self,
    ) -> None:
        w2c = []
        for i in range(nerf_lego_train_transforms.shape[0]):
            w2c.append(transform_matrix_to_w2c(nerf_lego_train_transforms[i]))
        w2c = np.array(w2c)
        w2c = torch.from_numpy(w2c)

        norm = NerfNorm.from_w2c_stack(w2c)

        self.assertAlmostEqual(norm.radius, 4.789909410476685, delta=0.1)


if __name__ == "__main__":
    unittest.main()
