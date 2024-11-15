
from pathlib import Path
from typing import NamedTuple
from gsplat_trainer.colors.colors import sh_to_rgb
import numpy as np
from plyfile import PlyData, PlyElement

class BasicPointCloud(NamedTuple):
  points : np.array
  colors : np.array
  normals : np.array

  @classmethod
  def fetchPly(cls, path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return cls(points=positions, colors=colors, normals=normals)

  @classmethod
  def load_initial_points(cls, dataset_path: str, num_points: int):
    ply_path = Path(dataset_path) / Path("points3d.ply")

    if not (ply_path).exists():
      print(f"Generating random point cloud ({num_points})...")

      # We create random points inside the bounds of the synthetic Blender scenes
      xyz = np.random.random((num_points, 3)) * 2.6 - 1.3
      shs = np.random.random((num_points, 3)) / 255.0
      pcd = cls(points=xyz, colors=sh_to_rgb(shs), normals=np.zeros((num_points, 3)))

      BasicPointCloud.storePly(ply_path, xyz, sh_to_rgb(shs) * 255)
    else:
      pcd = BasicPointCloud.fetchPly(ply_path)

      if pcd.points.shape[0] > num_points:
        print(f"Subsampling point cloud ({num_points})...")
        np.random.seed(123)
        chosen_points = np.random.choice(pcd.points.shape[0], num_points, replace=False)
        pcd = BasicPointCloud(pcd.points[chosen_points], pcd.colors[chosen_points], pcd.normals[chosen_points])

    return pcd
  
  @staticmethod
  def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)