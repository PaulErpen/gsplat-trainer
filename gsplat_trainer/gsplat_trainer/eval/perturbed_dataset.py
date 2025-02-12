from gsplat_trainer.data.nvs_dataset import NVSDataset
import torch
import numpy as np


class PerturbedDataset:
    def __init__(
        self,
        original: NVSDataset,
        n_pertubations: int,
        trans_std: float = 0.01,
        max_rot_degrees: float = 180.0,
    ) -> None:
        new_poses = []
        for idx_orig in range(len(original)):
            pose, gt_image, gt_alpha, K = original[idx_orig]
            for _idx_pert in range(n_pertubations):
                new_poses.append(
                    self.perturb_camera_pose(pose, trans_std, max_rot_degrees)
                )
        self.perturbed_poses = torch.stack(new_poses)
        self.n_pertubations = n_pertubations

    def get_perturbed_pose(self, idx_dataset: int, idx_pose: int) -> torch.Tensor:
        return self.perturbed_poses[idx_dataset * self.n_pertubations + idx_pose]

    def perturb_camera_pose(
        self,
        pose: torch.Tensor,
        trans_std: float = 0.01,
        max_rot_degrees: float = 180.0,
    ):
        # Extract translation (last column of the pose matrix)
        translation = pose[:3, 3]

        # Add Gaussian noise to the translation
        translation_noise = torch.randn_like(translation) * trans_std
        new_translation = translation + translation_noise

        # Create a small rotation matrix from axis-angle representation
        # Apply a strong random rotation
        rotation_matrix = pose[:3, :3]
        rotation_perturbation = random_rotation_matrix(max_degrees=max_rot_degrees)
        new_rotation = rotation_matrix @ rotation_perturbation

        # Construct new pose matrix
        new_pose = pose.clone()
        new_pose[:3, :3] = new_rotation
        new_pose[:3, 3] = new_translation

        return new_pose


def random_rotation_matrix(max_degrees: float = 180):
    """
    Generates a random 3x3 rotation matrix with an angle up to `max_degrees`.

    Args:
        max_degrees (float): Maximum rotation in degrees.

    Returns:
        torch.Tensor: A (3,3) rotation matrix.
    """
    # Choose a random rotation axis
    axis = torch.randn(3)
    axis = axis / torch.norm(axis)  # Normalize to unit length

    # Sample an angle uniformly from [-max_degrees, max_degrees]
    angle = torch.empty(1).uniform_(-max_degrees, max_degrees)  # Degrees
    angle = np.radians(angle.item())  # Convert to radians

    # Construct skew-symmetric matrix for Rodrigues' rotation formula
    skew_sym_matrix = torch.tensor(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )

    # Compute the rotation matrix using Rodrigues' formula
    rotation_matrix = (
        torch.eye(3)
        + np.sin(angle) * skew_sym_matrix
        + (1 - np.cos(angle)) * (skew_sym_matrix @ skew_sym_matrix)
    )

    return rotation_matrix
