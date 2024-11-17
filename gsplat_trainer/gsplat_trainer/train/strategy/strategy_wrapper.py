from typing import Tuple

from gsplat_trainer.config.config import Config
from gsplat_trainer.train.strategy.default_strategy import DefaultStrategy
from gsplat_trainer.train.strategy.mcmc_strategy import MCMCStrategy


class Strategy:
    def __init__(self, config: Config):
        if config.strategy_type == "mcmc":
            self.strategy = MCMCStrategy(
                cap_max=config.cap_max,
                noise_lr=config.noise_lr,
                refine_start_iter=config.refine_start_iter,
                refine_stop_iter=config.refine_stop_iter,
                refine_every=config.refine_every,
                min_opacity=config.min_opacity,
                verbose=config.verbose,
            )
            self.strategy_state = self.strategy.initialize_state()
        elif config.strategy_type == "default":
            self.strategy = DefaultStrategy(
                cap_max=config.cap_max, reset_every=config.reset_every
            )
            assert config.scene_radius is not None
            self.strategy_state = self.strategy.initialize_state(
                scene_scale=config.scene_radius
            )
        else:
            AssertionError(f"Unknown strategy {self.strategy}")

    def check_sanity(self, params, optimizers) -> None:
        self.strategy.check_sanity(params, optimizers)

    def step_pre_backward(self, params, optimizers, step: int, info):
        self.strategy.step_pre_backward(
            params=params,
            optimizers=optimizers,
            state=self.strategy_state,
            step=step,
            info=info,
        )

    def step_post_backward(
        self, params, optimizers, step: int, info, packed, schedulers
    ) -> Tuple[int, int]:
        if isinstance(self.strategy, DefaultStrategy):
            return self.strategy.step_post_backward(
                params=params,
                optimizers=optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
                packed=packed,
            )
        if isinstance(self.strategy, MCMCStrategy):
            return self.strategy.step_post_backward(
                params=params,
                optimizers=optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
                lr=schedulers[0].get_last_lr()[0],
            )
        else:
            AssertionError(f"Unknown strategy {self.strategy}")
