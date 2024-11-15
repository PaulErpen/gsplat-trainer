from typing import Tuple
import torch.utils.data
import torch

from gsplat_trainer.data.basicpointcloud import BasicPointCloud
from gsplat_trainer.data.nerfnorm import NerfNorm

class NVSDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 poses: torch.Tensor,
                 images: torch.Tensor,
                 intrinsics: torch.Tensor,
                 pcd: BasicPointCloud,
                 norm: NerfNorm):
        self.poses = poses
        self.images = images
        self.intrinsics = intrinsics
        self.pcd = pcd
        self.norm = norm
    
    def __len__(self):
        return self.poses.shape[0]

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.poses[idx], self.images[idx], self.intrinsics[idx]