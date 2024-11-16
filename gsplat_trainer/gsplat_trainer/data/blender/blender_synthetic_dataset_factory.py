import json
from pathlib import Path
from typing import Dict, List, Literal
from gsplat_trainer.data.basicpointcloud import BasicPointCloud
from gsplat_trainer.data.blender.blender_util import compute_intrinsics_matrix, image_path_to_tensors, transform_matrix_to_w2c
from gsplat_trainer.data.nerfnorm import NerfNorm
import torch
from gsplat_trainer.data.nvs_dataset import NVSDataset
import numpy as np


class BlenderSyntheticDatasetFactory:
    def __init__(self, 
                 data_root: str,
                 splits: List[Literal["train", "val", "test"]],
                 max_num_init_points: int,
                 limit: int | None = None):
        self.datasets: Dict[str, NVSDataset] = {}
        data_root = Path(data_root)

        for split in splits:
            meta = {}
            meta_path = Path(data_root) / Path(f"transforms_{split}.json")

            if not data_root.exists() or not meta_path.exists():
                raise FileNotFoundError(
                    f"The specified split file does not exist: {data_root}"
                )

            with open(meta_path, "r") as fp:
                meta = json.load(fp)

                tmp_img = []
                tmp_alpha = []
                poses = []
                for idx, frame in enumerate(meta["frames"]):
                    if limit is not None and idx >= limit:
                        break

                    img_path = data_root / Path(frame["file_path"] + ".png")
                    # Load the image
                    img, _ = image_path_to_tensors(img_path)
                    tmp_img.append(img)
                    poses.append(transform_matrix_to_w2c(np.array(frame["transform_matrix"])))
                
                poses = np.array(poses).astype(np.float32)
                poses = torch.from_numpy(poses)
                images = torch.stack(tmp_img, dim=0)
                camera_angle_x = float(meta["camera_angle_x"])

                focal_length = (
                    0.5 * 800 / np.tan(0.5 * camera_angle_x)
                )
                intrinsics = []
                for image_index in range(images.shape[0]):
                    H, W, _ = images[image_index].shape
                    intrinsics.append(compute_intrinsics_matrix(
                        focal_length, H, W
                    ))
                intrinsics = torch.stack(intrinsics, dim=0)

                pcd = BasicPointCloud.load_initial_points(data_root, max_num_init_points)

                norm = NerfNorm.from_c2w_stack(poses)

                self.datasets[split] = NVSDataset(
                    poses=poses,
                    images=images,
                    intrinsics=intrinsics,
                    pcd=pcd,
                    norm=norm
                )

    def get_split(self, split: Literal["train", "test", "val"]) -> NVSDataset:
        return self.datasets[split]