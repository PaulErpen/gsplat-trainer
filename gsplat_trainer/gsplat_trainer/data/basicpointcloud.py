from pathlib import Path
from typing import NamedTuple
from gsplat_trainer.colors.colors import sh_to_rgb
import numpy as np
from plyfile import PlyData, PlyElement


class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array

    @classmethod
    def fetchPly(cls, path) -> "BasicPointCloud":
        plydata = PlyData.read(path)
        vertices = plydata["vertex"]
        positions = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
        colors = (
            np.vstack([vertices["red"], vertices["green"], vertices["blue"]]).T / 255.0
        )
        normals = np.vstack([vertices["nx"], vertices["ny"], vertices["nz"]]).T
        return cls(points=positions, colors=colors, normals=normals)

    @classmethod
    def load_initial_points(cls, ply_path: Path, num_points: int):
        if not (ply_path).exists():
            print(f"Generating random point cloud ({num_points})...")

            # We create random points inside the bounds of the synthetic Blender scenes
            xyz = np.random.random((num_points, 3)) * 2.6 - 1.3
            shs = np.random.random((num_points, 3)) / 255.0
            pcd = cls(
                points=xyz, colors=sh_to_rgb(shs), normals=np.zeros((num_points, 3))
            )

            BasicPointCloud(xyz, sh_to_rgb(shs) * 255, np.zeros_like(xyz)).storePly(
                ply_path,
            )
        else:
            pcd = BasicPointCloud.fetchPly(ply_path)

            if pcd.points.shape[0] > num_points:
                print(f"Subsampling point cloud ({num_points})...")
                np.random.seed(123)
                chosen_points = np.random.choice(
                    pcd.points.shape[0], num_points, replace=False
                )
                pcd = BasicPointCloud(
                    pcd.points[chosen_points],
                    pcd.colors[chosen_points],
                    pcd.normals[chosen_points],
                )

        return pcd

    def storePly(self, path: str) -> None:
        # Define the dtype for the structured array
        dtype = [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]

        elements = np.empty(self.points.shape[0], dtype=dtype)
        attributes = np.concatenate((self.points, self.normals, self.colors), axis=1)
        elements[:] = list(map(tuple, attributes))

        # Create the PlyData object and write to file
        vertex_element = PlyElement.describe(elements, "vertex")
        ply_data = PlyData([vertex_element])
        ply_data.write(path)
