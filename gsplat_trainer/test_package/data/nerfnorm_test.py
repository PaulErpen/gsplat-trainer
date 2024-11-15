import unittest
from test_package.data.nerf_lego_train_transforms import nerf_lego_train_transforms
import torch

from gsplat_trainer.data.nerfnorm import NerfNorm

class NerfNormTest(unittest.TestCase):
    def setUp(self):
        self.c2w_offset_3 = torch.eye(4).repeat(1_000_000, 1, 1)
        self.c2w_offset_3[:, :1, 3] = (torch.rand_like(self.c2w_offset_3[:, :1, 3]) - 0.5) * 6

        self.c2w_offset_9 = torch.eye(4).repeat(1_000_000, 1, 1)
        self.c2w_offset_9[:, :3, 3] = (torch.rand_like(self.c2w_offset_9[:, :3, 3])) * 9

    def test_given_a_world_to_cam_stack__when_computing_the_norm__then_the_shape_of_the_center_must_be_correct(self):
        norm = NerfNorm.from_c2w_stack(self.c2w_offset_3)

        self.assertEqual(norm.translation.shape, (3, ))
    
    def test_given_a_world_to_cam_stack__when_computing_the_norm__then_the_radius_must_be_positive(self):
        norm = NerfNorm.from_c2w_stack(self.c2w_offset_3)

        self.assertTrue(norm.radius > 0)
    
    def test_given_a_random_world_to_cam_stack__when_offsetting_the_center_by_3__then_the_radius_must_be_roughly_3(self):
        norm = NerfNorm.from_c2w_stack(self.c2w_offset_3)

        self.assertAlmostEqual(norm.radius, 3.3, places=1)
    
    def test_given_the_cam_2_worlds_of_the_ego_nerf_dataset__when_creating_the_norm__it_must_equal(self):
        norm = NerfNorm.from_c2w_stack(nerf_lego_train_transforms)

        self.assertAlmostEqual(norm.radius, 5.202148, places=2)
    

if __name__ == '__main__':
    unittest.main()