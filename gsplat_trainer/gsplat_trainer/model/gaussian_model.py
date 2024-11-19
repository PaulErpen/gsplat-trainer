from gsplat_trainer.colors.colors import rgb_to_sh
from gsplat_trainer.data.basicpointcloud import BasicPointCloud
from gsplat_trainer.model.externals_wrapper import distCUDA2, rasterization
from gsplat_trainer.model.model_utils import inverse_sigmoid
from torch import nn
import torch
import numpy as np
from torch import nn


class GaussianModel(nn.Module):
    def __init__(
        self, params: nn.ParameterDict, scene_scale: float, sh_degree: int = 3
    ):
        super(GaussianModel, self).__init__()

        self.sh_degree = sh_degree
        self.scene_scale = scene_scale
        self.params = params

    @classmethod
    def from_point_cloud(
        cls, pcd: BasicPointCloud, scene_scale: float, sh_degree: int = 3
    ):
        points = torch.tensor(np.asarray(pcd.points)).float()
        num_points = points.shape[0]

        # color is SH coefficients.
        colors = torch.zeros((num_points, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(torch.tensor(np.asarray(pcd.colors)).float())

        dist2 = torch.clamp_min(
            distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float()), 0.0000001
        )
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)

        opacities = inverse_sigmoid(0.1 * torch.ones((num_points), dtype=torch.float))

        params = nn.ParameterDict(
            {
                "means": nn.Parameter(points.to(torch.float32)),
                "scales": nn.Parameter(scales.to(torch.float32)),
                "quats": nn.Parameter(torch.rand((num_points, 4)).to(torch.float32)),
                "opacities": opacities,
                "sh0": torch.nn.Parameter(colors[:, :1, :]),
                "shN": torch.nn.Parameter(colors[:, 1:, :]),
            }
        )

        return cls(params, scene_scale, sh_degree)

    def get_params(self) -> nn.ParameterDict:
        return self.params

    def set_params(self, params: torch.nn.ParameterDict) -> None:
        self.params = params

    def forward(
        self,
        view_matrix: torch.Tensor,
        K: torch.Tensor,
        W: int,
        H: int,
        sh_degree_to_use: int,
        bg_color: torch.Tensor | None,
    ):
        colors = torch.cat([self.params["sh0"], self.params["shN"]], 1)

        device = self.params["means"].device

        renders, alphas, info = rasterization(
            means=self.params["means"].to(device),
            quats=self.params["quats"].to(device),
            scales=torch.exp(self.params["scales"].to(device)),
            opacities=torch.sigmoid(self.params["opacities"].to(device)),
            colors=colors.to(device),
            viewmats=view_matrix.to(device),
            Ks=K,
            width=W,
            height=H,
            packed=False,
            sh_degree=sh_degree_to_use,
            backgrounds=bg_color.to(device),
        )
        return renders, alphas, info
