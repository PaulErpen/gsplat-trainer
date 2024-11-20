from pathlib import Path
from typing import Dict, List, Literal

from gsplat_trainer.data.basicpointcloud import BasicPointCloud
from gsplat_trainer.data.blender.blender_util import compute_intrinsics_matrix
from gsplat_trainer.data.colmap.helpers.read_cameras import readColmapCameras
from gsplat_trainer.data.colmap.helpers.read_extrinsics import read_extrinsics_binary
from gsplat_trainer.data.colmap.helpers.read_intrinsics import read_intrinsics_binary
from gsplat_trainer.data.colmap.helpers.read_points import read_points3D_binary
from gsplat_trainer.data.dataset_factory import DatasetFactory
from gsplat_trainer.data.nerfnorm import NerfNorm
from gsplat_trainer.data.nvs_dataset import NVSDataset
from gsplat_trainer.geometry.geometry_utils import getWorld2View2
import numpy as np
import torch
from torchvision import transforms


class ColmapDatasetFactory(DatasetFactory):
    def __init__(
        self,
        data_root: str,
        splits: List[Literal["train", "test"]],
        max_num_init_points: int,
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
            BasicPointCloud(xyz, rgb, normals).storePly(f"{data_root}/points3d.ply")

        pcd = BasicPointCloud.load_initial_points(data_root, max_num_init_points)

        assert pcd is not None, "The point-cloud cannot be None after initialization!"

        self.splits: Dict[str, NVSDataset] = {}
        to_tensor_tf = transforms.ToTensor()

        if "train" in splits:
            train_cam_infos = [
                c for idx, c in enumerate(cam_infos) if idx % holdout_interval != 0
            ]
            nerf_norm = NerfNorm.from_cam_infos(train_cam_infos)

            poses = torch.stack(
                [torch.from_numpy(getWorld2View2(c.R, c.T)) for c in train_cam_infos]
            ).float()
            intrinsics = torch.stack(
                [
                    compute_intrinsics_matrix(c.focal_length, c.width, c.height)
                    for c in train_cam_infos
                ]
            ).float()
            images = torch.stack(
                [to_tensor_tf(c.image).permute(1, 2, 0)[..., :3] for c in train_cam_infos]
            ).float()
            alphas = torch.zeros_like(images)

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
            intrinsics = torch.stack(
                [
                    compute_intrinsics_matrix(c.focal_length, c.width, c.height)
                    for c in test_cam_infos
                ]
            ).float()
            images = torch.stack(
                [to_tensor_tf(c.image).permute(1, 2, 0)[..., :3] for c in test_cam_infos]
            ).float()
            alphas = torch.zeros_like(images)

            self.splits["test"] = NVSDataset(
                poses=poses,
                images=images,
                alphas=alphas,
                intrinsics=intrinsics,
                pcd=pcd,
                norm=nerf_norm,
            )

    def get_split(self, split: str) -> NVSDataset:
        return self.splits[split]
