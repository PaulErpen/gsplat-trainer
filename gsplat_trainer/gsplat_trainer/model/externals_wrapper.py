from typing import Callable, Dict, Tuple
import numpy as np
import torch

rasterization: Callable[[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    int,
    bool,
    int,
    torch.Tensor,
], Tuple[torch.Tensor, torch.Tensor, Dict]] = None
distCUDA2: Callable[[np.array], torch.Tensor] = None

def mocked_rasterization(
    means=torch.Tensor,
    quats=torch.Tensor,
    scales=torch.Tensor,
    opacities=torch.Tensor,
    colors=torch.Tensor,
    viewmats=torch.Tensor,
    Ks=torch.Tensor,
    width=int,
    height=int,
    packed=bool,
    sh_degree=int,
    backgrounds=torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    print(Ks)
    B, d1, d2 = Ks.shape
    return (
        torch.rand(B, height, width, 3),
        torch.rand(B, height, width),
        {
            'camera_ids', 
            'gaussian_ids', 
            'radii', 
            'means2d', 
            'depths', 
            'conics',
            'opacities', 
            'tile_width', 
            'tile_height', 
            'tiles_per_gauss', 
            'isect_ids',
            'flatten_ids', 
            'isect_offsets', 
            'width', 
            'height', 
            'tile_size'
        }
    )

def mocked_dist_cuda_2(points: np.array) -> torch.Tensor:
    N, _ = points.shape

    return torch.rand((N,))

try:
    from gsplat import rasterization
    from simple_knn._C import distCUDA2
    rasterization = rasterization
    distCUDA2 = distCUDA2
except (ImportError, RuntimeError):
    rasterization = mocked_rasterization
    distCUDA2 = mocked_dist_cuda_2

