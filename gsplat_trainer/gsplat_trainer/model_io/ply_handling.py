from gsplat_trainer.model.gaussian_model import GaussianModel
import numpy as np
from plyfile import PlyData, PlyElement
import torch
from torch import nn
import os


def construct_list_of_attributes(model: GaussianModel):
    l = ["x", "y", "z", "nx", "ny", "nz"]
    # All channels except the 3 DC
    for i in range(model.params["sh0"].shape[1] * model.params["sh0"].shape[2]):
        l.append("f_dc_{}".format(i))
    for i in range(model.params["shN"].shape[1] * model.params["shN"].shape[2]):
        l.append("f_rest_{}".format(i))
    l.append("opacity")
    for i in range(model.params["scales"].shape[1]):
        l.append("scale_{}".format(i))
    for i in range(model.params["quats"].shape[1]):
        l.append("rot_{}".format(i))
    return l


@torch.no_grad()
def save_ply(model: GaussianModel, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    xyz = model.params["means"].detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = (
        model.params["sh0"]
        .detach()
        .transpose(1, 2)
        .flatten(start_dim=1)
        .contiguous()
        .cpu()
        .numpy()
    )
    f_rest = (
        model.params["shN"]
        .detach()
        .transpose(1, 2)
        .flatten(start_dim=1)
        .contiguous()
        .cpu()
        .numpy()
    )
    opacities = model.params["opacities"].detach().unsqueeze(-1).cpu().numpy()
    scale = model.params["scales"].detach().cpu().numpy()
    rotation = model.params["quats"].detach().cpu().numpy()

    dtype_full = [
        (attribute, "f4") for attribute in construct_list_of_attributes(model)
    ]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate(
        (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
    )
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(path)


@torch.no_grad()
def load_ply(path: str, scene_scale: float, device: str = "cuda") -> GaussianModel:
    plydata = PlyData.read(path)

    xyz = np.stack(
        (
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ),
        axis=1,
    )
    opacities = np.asarray(plydata.elements[0]["opacity"])

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")
    ]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))

    n_extra_sh = int(len(extra_f_names) / 3)
    sh_degree = int((n_extra_sh + 1) ** 0.5) - 1

    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, n_extra_sh))

    scale_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")
    ]
    scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
    ]
    rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    params = torch.nn.ParameterDict(
        {
            "means": nn.Parameter(
                torch.tensor(xyz, dtype=torch.float, device=device).requires_grad_(True)
            ),
            "sh0": nn.Parameter(
                torch.tensor(features_dc, dtype=torch.float, device=device)
                .transpose(1, 2)
                .contiguous()
                .requires_grad_(True)
            ),
            "shN": nn.Parameter(
                torch.tensor(features_extra, dtype=torch.float, device=device)
                .transpose(1, 2)
                .contiguous()
                .requires_grad_(True)
            ),
            "opacities": nn.Parameter(
                torch.tensor(
                    opacities, dtype=torch.float, device=device
                ).requires_grad_(True)
            ),
            "scales": nn.Parameter(
                torch.tensor(scales, dtype=torch.float, device=device).requires_grad_(
                    True
                )
            ),
            "quats": nn.Parameter(
                torch.tensor(rots, dtype=torch.float, device=device).requires_grad_(
                    True
                )
            ),
        }
    )

    gaussian_model = GaussianModel(params, sh_degree=sh_degree, scene_scale=scene_scale)

    return gaussian_model
