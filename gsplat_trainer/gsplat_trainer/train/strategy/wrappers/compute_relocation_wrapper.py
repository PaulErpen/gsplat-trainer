from typing import Callable, Tuple
from torch import Tensor
import torch


def mocked_compute_relocation(
    opacities: Tensor, scales: Tensor, ratios: Tensor, binoms: Tensor, n_max: int
) -> Tuple[Tensor, Tensor]:
    N = opacities.shape[0]
    return (torch.rand((N,)), torch.rand((N, 3)))


def mocked_make_lazy_cuda_func(func_name: str) -> Callable:
    if func_name == "compute_relocation":
        return mocked_compute_relocation
    raise Exception(
        f'Cannot make mocked lazy cuda function with function name "{func_name}"'
    )


try:
    from gsplat.cuda._wrapper import _make_lazy_cuda_func
except ImportError:
    _make_lazy_cuda_func = mocked_make_lazy_cuda_func


def compute_relocation(
    opacities: Tensor,  # [N]
    scales: Tensor,  # [N, 3]
    ratios: Tensor,  # [N]
    binoms: Tensor,  # [n_max, n_max]
) -> Tuple[Tensor, Tensor]:
    """Compute new Gaussians from a set of old Gaussians.

    This function interprets the Gaussians as samples from a likelihood distribution.
    It uses the old opacities and scales to compute the new opacities and scales.
    This is an implementation of the paper
    `3D Gaussian Splatting as Markov Chain Monte Carlo <https://arxiv.org/pdf/2404.09591>`_,

    Args:
        opacities: The opacities of the Gaussians. [N]
        scales: The scales of the Gaussians. [N, 3]
        ratios: The relative frequencies for each of the Gaussians. [N]
        binoms: Precomputed lookup table for binomial coefficients used in
          Equation 9 in the paper. [n_max, n_max]

    Returns:
        A tuple:

        **new_opacities**: The opacities of the new Gaussians. [N]
        **new_scales**: The scales of the Gaussians. [N, 3]
    """

    N = opacities.shape[0]
    n_max, _ = binoms.shape
    assert scales.shape == (N, 3), scales.shape
    assert ratios.shape == (N,), ratios.shape
    opacities = opacities.contiguous()
    scales = scales.contiguous()
    ratios.clamp_(min=1, max=n_max)
    ratios = ratios.int().contiguous()

    new_opacities, new_scales = _make_lazy_cuda_func("compute_relocation")(
        opacities, scales, ratios, binoms, n_max
    )
    return new_opacities, new_scales
