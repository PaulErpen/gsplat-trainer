import unittest
from gsplat_trainer.metrics.psnr import psnr
import torch
import numpy as np


class PSNRTest(unittest.TestCase):
    def test_given_two_images_that_are_the_opposite__when_computing_the_psnr_then__then_it_must_be_zero(
        self,
    ) -> None:
        im1 = torch.ones(1, 800, 800, 3)
        im2 = torch.zeros(1, 800, 800, 3)

        self.assertEqual(psnr(im1, im2), 0.0)

    def test_given_two_images_that_are_the_same__when_computing_the_psnr__then_it_must_be_infinite(
        self,
    ) -> None:
        im1 = torch.ones(1, 800, 800, 3)
        im2 = torch.ones(1, 800, 800, 3)

        self.assertTrue(psnr(im1, im2) == np.inf)

    def test_given_two_images_that_are_random__when_computing_the_psnr__then_the_value_must_be_finite(
        self,
    ) -> None:
        im1 = torch.rand(1, 800, 800, 3)
        im2 = torch.rand(1, 800, 800, 3)

        self.assertTrue(np.isfinite(psnr(im1, im2)))
