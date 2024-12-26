from pathlib import Path
from typing import Dict, List, Literal

from gsplat_trainer.data.basicpointcloud import BasicPointCloud
from gsplat_trainer.data.blender.blender_util import compute_intrinsics_matrix
from gsplat_trainer.data.colmap.helpers.camera_helpers import (
    compute_intrinsics_matrix_pinhole,
)
from gsplat_trainer.data.colmap.helpers.read_cameras import readColmapCameras
from gsplat_trainer.data.colmap.helpers.read_extrinsics import read_extrinsics_binary
from gsplat_trainer.data.colmap.helpers.read_intrinsics import read_intrinsics_binary
from gsplat_trainer.data.colmap.helpers.read_points import read_points3D_binary
from gsplat_trainer.data.dataset_factory import DatasetFactory
from gsplat_trainer.data.nerfnorm import NerfNorm
from gsplat_trainer.data.nvs_dataset import NVSDataset
from gsplat_trainer.geometry.geometry_utils import getWorld2View2
from gsplat_trainer.graphics.graphics_helpers import compute_resolution, image_downscale
import numpy as np
import torch


class ColmapDatasetFactory(DatasetFactory):
    def __init__(
        self,
        data_root: str,
        splits: List[Literal["train", "test"]],
        max_num_init_points: int,
        image_downscale_factor: Literal[-1, 1, 2, 4, 8] = 1,
        holdout_interval=8,
    ):
        data_root = Path(data_root)
        if (data_root / Path("sparse/0/images.bin")).exists() and (
            data_root / Path("sparse/0/cameras.bin")
        ).exists():
            cam_extrinsics = read_extrinsics_binary(f"{data_root}/sparse/0/images.bin")
            cam_intrinsics = read_intrinsics_binary(f"{data_root}/sparse/0/cameras.bin")
        else:
            raise FileNotFoundError(
                f'Missing binary files in data root: "{data_root}/sparse/0/". txt files not supported!'
            )

        cam_infos_unsorted = readColmapCameras(
            cam_extrinsics=cam_extrinsics,
            cam_intrinsics=cam_intrinsics,
            images_folder=(data_root / "images"),
        )
        cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

        ply_path = data_root / "sparse/0/points3D.ply"
        bin_path = data_root / "sparse/0/points3D.bin"
        if not ply_path.exists():
            print(
                "Converting point3d.bin to .ply, will happen only the first time you open the scene."
            )
            xyz, rgb, _ = read_points3D_binary(bin_path)
            normals = np.zeros_like(xyz)
            BasicPointCloud(xyz, rgb, normals).storePly(ply_path)

        pcd = BasicPointCloud.load_initial_points(ply_path, max_num_init_points)

        assert pcd is not None, "The point-cloud cannot be None after initialization!"

        self.splits: Dict[str, NVSDataset] = {}

        if "train" in splits:
            train_cam_infos = [
                c for idx, c in enumerate(cam_infos) if idx % holdout_interval != 0
            ]
            nerf_norm = NerfNorm.from_cam_infos(train_cam_infos)

            poses = torch.stack(
                [torch.from_numpy(getWorld2View2(c.R, c.T)) for c in train_cam_infos]
            ).float()
            intrinsics = self.compute_intrinsics_stack(
                image_downscale_factor, train_cam_infos
            )
            images = torch.stack(
                [
                    image_downscale(c.image, image_downscale_factor)[..., :3]
                    for c in train_cam_infos
                ]
            ).float()
            alphas = torch.ones_like(images)

            self.splits["train"] = NVSDataset(
                poses=poses,
                images=images,
                alphas=alphas,
                intrinsics=intrinsics,
                pcd=pcd,
                norm=nerf_norm,
            )

        if "test" in splits:
            test_cam_infos = [
                c for idx, c in enumerate(cam_infos) if idx % holdout_interval == 0
            ]
            nerf_norm = NerfNorm.from_cam_infos(test_cam_infos)

            poses = torch.stack(
                [torch.from_numpy(getWorld2View2(c.R, c.T)) for c in test_cam_infos]
            ).float()
            intrinsics = self.compute_intrinsics_stack(
                image_downscale_factor, test_cam_infos
            )
            images = torch.stack(
                [
                    image_downscale(c.image, image_downscale_factor)[..., :3]
                    for c in test_cam_infos
                ]
            ).float()
            alphas = torch.ones_like(images)

            self.splits["test"] = NVSDataset(
                poses=poses,
                images=images,
                alphas=alphas,
                intrinsics=intrinsics,
                pcd=pcd,
                norm=nerf_norm,
            )

    def compute_intrinsics_stack(
        self, image_downscale_factor, train_cam_infos
    ) -> torch.Tensor:
        intrinsics = []
        for c in train_cam_infos:
            res_x, res_y = 1.0, 1.0
            if image_downscale_factor == -1:
                res_x, res_y = compute_resolution(c.image)
            intrinsics.append(
                compute_intrinsics_matrix_pinhole(
                    c.focal_length_x / image_downscale_factor * res_x,
                    c.focal_length_y / image_downscale_factor * res_y,
                    c.width / image_downscale_factor * res_x,
                    c.height / image_downscale_factor * res_y,
                )
            )
        return torch.stack(intrinsics).float()

    def get_split(self, split: str) -> NVSDataset:
        return self.splits[split]
