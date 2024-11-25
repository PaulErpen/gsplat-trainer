from typing import Any, Tuple
import torch

quat_scale_to_covar_preci = None


def mocked_quat_scale_to_covar_preci(
    quats: torch.Tensor,
    scales: torch.Tensor,
    compute_covar: bool,
    compute_preci: bool,
    triu: bool,
) -> Tuple[torch.Tensor, Any]:
    return torch.rand((quats.shap[0], 3, 3))


try:
    from gsplat import quat_scale_to_covar_preci as actual_quat_scale_to_covar_preci

    assert torch.cuda.is_available(), "Cuda is not available, yet the \"quat_scale_to_covar_preci\" is not being mocked!" 

    quat_scale_to_covar_preci = actual_quat_scale_to_covar_preci
except ImportError:
    quat_scale_to_covar_preci = mocked_quat_scale_to_covar_preci

    assert torch.cuda.is_available() == False, "Cuda is available, yet the \"quat_scale_to_covar_preci\" is being mocked!" 
