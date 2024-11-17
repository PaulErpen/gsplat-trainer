from dataclasses import dataclass, field
from typing import List, Literal, Optional
import torch

@dataclass
class Config:
  # INITIALIZATION
  # Radius of the scene (Especially important for NeRF data i think)
  scene_radius: Optional[float] = None
  # Initial number of Gaussians
  init_num_gaussians: int = 2000
  # The number of spherical harmonics degrees to compute the color
  sh_degree: int = 3

  # RENDERING
  # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
  packed: bool = False
  # The background color to use to render the splats (is also pasted "under" the train and test images)
  bg_color: torch.Tensor = torch.ones(3)

  # TRAINING
  # Strategy to use for densifying the gaussians
  strategy_type: Literal["mcmc", "default"] = "mcmc"
  # Number of training steps
  max_steps: int = 10_000
  # Opacity regularization
  opacity_reg: float = 0.0
  # Scale regularization
  scale_reg: float = 0.0
  # the frequency with which holdout views are created
  holdout_view_frequency: int = 100
  # the amount of ssim included in the loss
  ssim_lambda: float = 0.2
  # the exact iterations when the testing loop is supposed to be executed
  test_iterations: List[int] = field(default_factory=lambda: [1, 500, 1000, 5000, 7500, 9000, 10000])
  # Wether to print verbose info
  verbose: bool = False
  # maximum cap for the number of gaussians
  cap_max: int = 18_000
  # MCMC samping noise learning rate. Default to 5e5.
  noise_lr: float = 5e5
  # Start refining GSs after this iteration. Default to 500.
  refine_start_iter: int = 500
  # Stop refining GSs after this iteration. Default to 25_000.
  refine_stop_iter: int = 25_000
  # Refine GSs every this steps. Default to 100.
  refine_every: int = 100
  # GSs with opacity below this value will be pruned. Default to 0.005.
  min_opacity: float = 0.005
  # Interval with which the spherical harmonics degree is increased to reach the maximum
  sh_degree_interval: int = 1000

  # DEFAULT STRATEGY
  # Reset opacities every this steps. Default is 3001
  reset_every: int = 3001