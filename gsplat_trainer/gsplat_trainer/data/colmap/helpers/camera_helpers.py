import torch

def compute_intrinsics_matrix_pinhole(
    focal_length_x, focal_length_y, width, height
) -> torch.Tensor:
    """
    Compute the camera intrinsics matrix with separate focal lengths for x and y axes.

    Args:
        focal_length_x (float): Focal length along the x-axis.
        focal_length_y (float): Focal length along the y-axis.
        width (float): Width of the image sensor.
        height (float): Height of the image sensor.

    Returns:
        torch.Tensor: 3x3 camera intrinsics matrix.
    """
    return torch.tensor(
        [
            [focal_length_x, 0, width / 2],  # fx and cx
            [0, focal_length_y, height / 2],  # fy and cy
            [0, 0, 1],  # homogeneous coordinates
        ],
        dtype=torch.float,
    )