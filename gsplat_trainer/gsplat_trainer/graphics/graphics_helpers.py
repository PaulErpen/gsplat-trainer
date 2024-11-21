from math import floor
import torch
from torchvision import transforms
from PIL import Image

to_tensor = transforms.ToTensor()

def image_downscale(image: Image, downscale_factor: int) -> torch.Tensor:
    W, H = image.size
    resolution = floor(W / downscale_factor), floor(H / downscale_factor)
    return to_tensor(image.resize(resolution)).permute(1, 2, 0)
