import torch


def get_top_k_indices(tensor: torch.Tensor, k: int) -> torch.Tensor:
    if k > tensor.shape[-1]:
        return torch.arange(tensor.shape[-1], device=tensor.device)
    v, i = torch.topk(tensor, k, largest=True)
    return i
