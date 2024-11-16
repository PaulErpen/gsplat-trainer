import unittest
from gsplat_trainer.metrics.ssim import ssim
import torch
from numpy.testing import assert_almost_equal

class SSIMTest(unittest.TestCase):
    def test_given_two_images_that_are_the_same__when_computing_the_ssim__then_it_must_be_one(self) -> None:
        im1 = torch.ones(1, 800, 800, 3)
        im2 = torch.ones(1, 800, 800, 3)

        self.assertEqual(ssim(im1, im2), 1.0)

    def test_given_two_images_that_are_the_opposite__when_computing_the_ssim__then_it_must_be_almost_zero(self) -> None:
        im1 = torch.ones(1, 800, 800, 3)
        im2 = torch.zeros(1, 800, 800, 3)

        assert_almost_equal(ssim(im1, im2), 0.0, 6)