import os
import unittest

import PIL
import PIL.Image
from gsplat_trainer.graphics.graphics_helpers import image_downscale
import torch
from torchvision import transforms


class GraphicsHelperTest(unittest.TestCase):
    def setUp(self) -> None:
        self.to_pil_image = transforms.ToPILImage()

    def test_given_a_image_with_w_800_and_h_900__when_downscaling_by_2__then_the_resolution_must_by_correct(
        self,
    ) -> None:
        W = 800
        H = 900
        image = torch.rand((3, H, W))
        image = self.to_pil_image(image)

        self.assertEqual(image_downscale(image, 2).shape, (450, 400, 3))

    def test_given_a_lego_image__when_downscaling_by_2__then_the_resolution_must_by_correct(
        self,
    ) -> None:
        image = PIL.Image.open(f"{os.getcwd()}/mocked_datasets/lego/train/r_0.png")

        self.assertEqual(image_downscale(image, 2).shape, (400, 400, 4))
    
    def test_given_a_backyard_image__when_downscaling_by_2__then_the_resolution_must_by_correct(
        self,
    ) -> None:
        image = PIL.Image.open(f"{os.getcwd()}/mocked_datasets/backyard/images/IMG_1130.JPG")

        self.assertEqual(image_downscale(image, 2).shape, (1071, 1428, 3))
