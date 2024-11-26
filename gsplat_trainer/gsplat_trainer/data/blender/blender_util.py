from pathlib import Path
from PIL import Image
from typing import Tuple
import torch
from torchvision import transforms
import numpy as np


def image_path_to_tensors(image_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    img = Image.open(image_path)
    to_tensor = transforms.ToTensor()
    img_tensor = to_tensor(img).permute(1, 2, 0)[..., :3]
    alpha_tensor = to_tensor(img).permute(1, 2, 0)[..., 3:]
    return img_tensor, alpha_tensor


def compute_intrinsics_matrix(focal_length, width, height) -> torch.Tensor:
    return torch.tensor(
        [
            [focal_length, 0, width / 2],
            [0, focal_length, height / 2],
            [0, 0, 1],
        ],
        dtype=torch.float,
    )


def transform_matrix_to_w2c(transform_matrix):
    c2w = transform_matrix
    c2w[:3, 1:3] *= -1  # change axis convention from Blender to colmap
    return np.linalg.inv(c2w)
