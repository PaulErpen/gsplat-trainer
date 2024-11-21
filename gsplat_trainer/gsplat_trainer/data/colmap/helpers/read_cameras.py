import os
import sys
from typing import Dict, List
from gsplat_trainer.data.blender.blender_util import (
    compute_intrinsics_matrix,
    compute_intrinsics_matrix_pinhole,
)
from gsplat_trainer.data.colmap.helpers.types import (
    Camera,
    CameraInfo,
    Image as ImageInfo,
)
from gsplat_trainer.geometry.geometry_utils import (
    focal2fov,
    getWorld2View2,
    qvec2rotmat,
)
import numpy as np
from PIL import Image


def readColmapCameras(
    cam_extrinsics: Dict[int, ImageInfo],
    cam_intrinsics: Dict[int, Camera],
    images_folder: str,
) -> List[CameraInfo]:
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write("\r")
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            intrinsics = compute_intrinsics_matrix(
                focal_length=focal_length_x, width=width, height=height
            )
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            intrinsics = compute_intrinsics_matrix_pinhole(
                focal_length_x=focal_length_x,
                focal_length_y=focal_length_y,
                width=width,
                height=height,
            )
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert (
                False
            ), "Colmap camera model not handled: only undistorted datasets (SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        W2C = getWorld2View2(R, T)
        C2W = np.linalg.inv(W2C)
        cam_center = C2W[:3, 3:4]

        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=image,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
            intrinsics=intrinsics,
            cam_center=cam_center,
        )
        cam_infos.append(cam_info)
    sys.stdout.write("\n")
    return cam_infos
