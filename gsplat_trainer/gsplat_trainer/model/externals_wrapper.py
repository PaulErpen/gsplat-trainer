from typing import Callable, Dict, Tuple
import numpy as np
import torch

rasterization: Callable[
    [
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
    ],
    Tuple[torch.Tensor, torch.Tensor, Dict],
] = None
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
    B = Ks.shape[0]
    N = means.shape[0]
    return (
        torch.rand(B, height, width, 3),
        torch.rand(B, height, width),
        {
            "camera_ids": torch.tensor([], requires_grad=True),
            "gaussian_ids": torch.tensor([], requires_grad=True),
            "radii": torch.tensor([], requires_grad=True),
            "means2d": torch.tensor([], requires_grad=True),
            "depths": torch.tensor([], requires_grad=True),
            "conics": torch.tensor([], requires_grad=True),
            "opacities": torch.tensor([], requires_grad=True),
            "tile_width": torch.tensor([], requires_grad=True),
            "tile_height": torch.tensor([], requires_grad=True),
            "tiles_per_gauss": torch.tensor([], requires_grad=True),
            "isect_ids": torch.tensor([], requires_grad=True),
            "flatten_ids": torch.tensor([], requires_grad=True),
            "isect_offsets": torch.tensor([], requires_grad=True),
            "width": torch.tensor([], requires_grad=True),
            "height": torch.tensor([], requires_grad=True),
            "tile_size": torch.tensor([], requires_grad=True),
        },
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
    print(
        'Could not get cuda specific libraries "gsplat" and "simple_knn" reverting to mocked backend!'
    )
    assert (
        torch.cuda.is_available() == False
    ), 'Warning! Cuda is available, but "gsplat" and "simple_knn" libraries are not available! This is an illegal state!'
    rasterization = mocked_rasterization
    distCUDA2 = mocked_dist_cuda_2
