from typing import NamedTuple
import numpy as np


class CameraModel(NamedTuple):
    model_id: int
    model_name: str
    num_params: int


class Camera(NamedTuple):
    id: int
    model: str
    width: int
    height: int
    params: np.array


class Image(NamedTuple):
    id: int
    qvec: np.array
    tvec: np.array
    camera_id: int
    name: str
    xys: np.array
    point3D_ids: np.array


class Point3D(NamedTuple):
    id: int
    xyz: np.array
    rgb: np.array
    error: float
    image_ids: np.array
    point2D_idxs: np.array


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    focal_length: float
    cam_center: np.array = None
