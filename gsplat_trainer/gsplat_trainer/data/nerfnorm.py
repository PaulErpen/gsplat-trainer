# Nerf Plus Plus Norm
# Deals with the center and the extent of the dataset
from dataclasses import dataclass
from typing import List
from gsplat_trainer.data.colmap.helpers.types import CameraInfo
from gsplat_trainer.geometry.geometry_utils import getWorld2View2
import torch
import numpy as np


@dataclass
class NerfNorm:
    translation: torch.Tensor
    radius: float

    @classmethod
    def from_w2c_stack(cls, w2c: torch.Tensor) -> "NerfNorm":

        def get_center_and_diag(cam_centers):
            cam_centers = np.hstack(cam_centers)
            avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
            center = avg_cam_center
            dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
            diagonal = np.max(dist)
            return center.flatten(), diagonal

        cam_centers = []

        for i in range(w2c.shape[0]):
            C2W = np.linalg.inv(w2c[i])
            cam_centers.append(C2W[:3, 3:4])

        center, diagonal = get_center_and_diag(cam_centers)
        radius = diagonal * 1.1

        translate = -center

        return cls(translate, diagonal)

    @classmethod
    def from_cam_infos(cls, cam_infos: List[CameraInfo]) -> "NerfNorm":
        def get_center_and_diag(cam_centers):
            cam_centers = np.hstack(cam_centers)
            avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
            center = avg_cam_center
            dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
            diagonal = np.max(dist)
            return center.flatten(), diagonal

        cam_centers = []

        for cam in cam_infos:
            W2C = getWorld2View2(cam.R, cam.T)
            C2W = np.linalg.inv(W2C)
            cam_centers.append(C2W[:3, 3:4])

        center, diagonal = get_center_and_diag(cam_centers)
        radius = diagonal * 1.1

        translate = -center

        return cls(translate, radius)
