from math import floor
from typing import Tuple
import torch
from torchvision import transforms
from PIL import Image

to_tensor = transforms.ToTensor()


def compute_resolution(image: Image) -> Tuple[float, float]:
    W, H = image.size
    downscale = W / 1600
    resolution = floor(W / downscale), floor(H / downscale)
    return resolution

def image_downscale(image: Image, downscale_factor: int) -> torch.Tensor:
    W, H = image.size
    if downscale_factor != -1:
        resolution = floor(W / downscale_factor), floor(H / downscale_factor)
    else:
        resolution = compute_resolution(image)
        print(
            f'Image width exceeds 1600 pix. Downscaling to {resolution}. If this is unwanted use "--resolution 1".'
        )
    return to_tensor(image.resize(resolution)).permute(1, 2, 0)
