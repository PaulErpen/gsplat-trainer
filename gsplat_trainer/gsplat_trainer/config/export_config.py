import json
import os
from pathlib import Path
from gsplat_trainer.config.config import Config
from gsplat_trainer.train.strategy.strategy_wrapper import Strategy


def export_configs(out_dir: str, config: Config, strategy: Strategy):
    config_string = json.dumps({k: str(v) for k, v in vars(config).items()})
    strategy_string = json.dumps(
        {k: str(v) for k, v in vars(strategy.strategy).items()}
    )

    if not Path(out_dir).exists():
        os.makedirs(out_dir)

    with open(f"{out_dir}/config.json", "w") as f:
        f.write("# config \n")
        f.write(config_string)
        f.write("\n# strategy \n")
        f.write(strategy_string)
