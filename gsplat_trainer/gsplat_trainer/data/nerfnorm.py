# Nerf Plus Plus Norm
# Deals with the center and the extent of the dataset
from dataclasses import dataclass
import torch
import numpy as np

@dataclass
class NerfNorm:
  translation: torch.Tensor
  radius: float

  @classmethod
  def from_c2w_stack(cls, c2w: torch.Tensor):
    
    centers = c2w[:, :3, 3]
    center = torch.mean(centers, dim=0)

    assert center.shape == (3,), center.shape

    centered = (centers - center)
    dist = np.linalg.norm(centered, axis=1, keepdims=True)
    diagonal = np.max(dist)

    translate = -center

    return cls(translate, diagonal * 1.1)