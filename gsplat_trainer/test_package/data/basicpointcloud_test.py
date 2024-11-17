from pathlib import Path
import unittest
import os

from gsplat_trainer.data.basicpointcloud import BasicPointCloud

class BasicPointcloudTest(unittest.TestCase):
    def setUp(self):
        self.dataset_path = "./tmp/dataset"
        self.num_points = 10

        os.makedirs(self.dataset_path)


    def test_given_no_initial_point_cloud__when_loading__the_initial_point_cloud_must_exist_afterward(self):
        BasicPointCloud.load_initial_points(self.dataset_path, self.num_points)

        self.assertTrue(Path(f"{self.dataset_path}/points3d.ply").exists())
    
    def tearDown(self):
        os.system(f"rm -rf {self.dataset_path}")