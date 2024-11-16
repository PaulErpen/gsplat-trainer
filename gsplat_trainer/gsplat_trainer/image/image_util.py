import torch

def add_backround(rgbs: torch.Tensor, background: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
  blended_image = rgbs * alpha + background * (1 - alpha)

  return blended_image