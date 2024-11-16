from typing import Dict, Union
from gsplat_trainer.model.gaussian_model import GaussianModel
from torch import optim

class OptimizerFactory:
  @staticmethod
  def create_optimizers(model: GaussianModel, lr_by_name: Union[None, Dict[str, float]] = None) -> Dict:
    if lr_by_name is None:
      lr_by_name = OptimizerFactory.get_default_learning_rates(model.scene_scale)

    optimizers = {}

    for name, parameters in model.params.items():
      lr = lr_by_name[name]

      if lr is None:
        raise ValueError(f"No learning rate for {name}")

      optimizers[name] = optim.Adam(
          [{"params": parameters, "lr": lr, "name": name}]
      )

    return optimizers

  @staticmethod
  def get_default_learning_rates(spatial_lr_scale: float) -> Dict[str, float]:
    return {
      "means": 1.6e-4 * spatial_lr_scale,
      "scales": 5e-3,
      "quats": 1e-3,
      "opacities": 5e-2,
      "sh0": 2.5e-3,
      "shN": 2.5e-3 / 20
    }