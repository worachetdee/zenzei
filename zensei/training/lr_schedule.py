"""Learning rate schedulers for Zensei pretraining.

Provides:
- ``WarmupCosineScheduler``: linear warmup followed by cosine decay to a
  configurable minimum learning rate.
- ``TwoStageScheduler``: orchestrates two ``WarmupCosineScheduler`` instances
  for the two-stage continued-pretraining recipe (embedding warmup, then full
  model training).

All schedulers implement the interface expected by
``torch.optim.lr_scheduler._LRScheduler`` and are compatible with DeepSpeed.
"""

import math
from typing import Optional

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """Linear warmup followed by cosine decay to ``min_lr``.

    Args:
        optimizer: Wrapped optimizer.
        peak_lr: Maximum learning rate after warmup.
        min_lr: Minimum learning rate at end of cosine decay.
        warmup_steps: Number of linear warmup steps.
        max_steps: Total number of training steps (warmup + decay).
        last_epoch: Index of last epoch (for resuming). Default ``-1``.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        peak_lr: float,
        min_lr: float,
        warmup_steps: int,
        max_steps: int,
        last_epoch: int = -1,
    ) -> None:
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        # Must call super().__init__ AFTER setting attributes
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> list[float]:  # noqa: D401
        """Compute learning rate for the current step."""
        step = self.last_epoch
        if step < self.warmup_steps:
            # Linear warmup
            lr = self.peak_lr * step / max(1, self.warmup_steps)
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / max(
                1, self.max_steps - self.warmup_steps
            )
            progress = min(progress, 1.0)
            lr = self.min_lr + 0.5 * (self.peak_lr - self.min_lr) * (
                1.0 + math.cos(math.pi * progress)
            )
        return [lr for _ in self.base_lrs]

    @classmethod
    def from_config(cls, optimizer: Optimizer, cfg: dict) -> "WarmupCosineScheduler":
        """Construct from a configuration dictionary.

        Expected keys: ``peak_lr``, ``min_lr``, ``warmup_steps``, ``max_steps``.
        """
        return cls(
            optimizer=optimizer,
            peak_lr=cfg["peak_lr"],
            min_lr=cfg["min_lr"],
            warmup_steps=cfg["warmup_steps"],
            max_steps=cfg["max_steps"],
        )


class TwoStageScheduler:
    """Orchestrates learning rate scheduling for two-stage pretraining.

    Stage 1 (embedding warmup):
        - ``peak_lr = 1e-3``, warmup 500 steps, cosine to ``1e-4``.
        - Only embedding and LM-head parameters are trained.

    Stage 2 (full continued pretraining):
        - ``peak_lr = 1e-5``, warmup 1000 steps, cosine to ``1e-6``.
        - All model parameters are unfrozen.

    The scheduler exposes :meth:`step`, :meth:`get_last_lr`, and
    :meth:`state_dict` / :meth:`load_state_dict` for compatibility with
    typical training loops and DeepSpeed.

    Args:
        optimizer: Wrapped optimizer.
        stage1_cfg: Config dict for stage 1 (see :class:`WarmupCosineScheduler`).
        stage2_cfg: Config dict for stage 2.
    """

    # Sensible defaults matching the Zensei recipe
    DEFAULT_STAGE1 = {
        "peak_lr": 1e-3,
        "min_lr": 1e-4,
        "warmup_steps": 500,
        "max_steps": 5000,
    }
    DEFAULT_STAGE2 = {
        "peak_lr": 1e-5,
        "min_lr": 1e-6,
        "warmup_steps": 1000,
        "max_steps": 200000,
    }

    def __init__(
        self,
        optimizer: Optimizer,
        stage1_cfg: Optional[dict] = None,
        stage2_cfg: Optional[dict] = None,
    ) -> None:
        self.optimizer = optimizer
        self.stage1_cfg = stage1_cfg or self.DEFAULT_STAGE1
        self.stage2_cfg = stage2_cfg or self.DEFAULT_STAGE2

        self._current_stage = 1
        self._global_step = 0
        self._stage_step = 0

        self._scheduler: _LRScheduler = WarmupCosineScheduler.from_config(
            optimizer, self.stage1_cfg
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def current_stage(self) -> int:
        return self._current_stage

    def switch_to_stage2(self) -> None:
        """Transition from Stage 1 to Stage 2.

        Should be called after unfreezing all model parameters and
        (optionally) rebuilding the optimizer with the new parameter groups.
        """
        self._current_stage = 2
        self._stage_step = 0
        self._scheduler = WarmupCosineScheduler.from_config(
            self.optimizer, self.stage2_cfg
        )

    def step(self) -> None:
        """Advance one training step."""
        self._scheduler.step()
        self._global_step += 1
        self._stage_step += 1

    def get_last_lr(self) -> list[float]:
        """Return the last computed learning rate."""
        return self._scheduler.get_last_lr()

    # ------------------------------------------------------------------ #
    # Serialization (DeepSpeed compatible)
    # ------------------------------------------------------------------ #

    def state_dict(self) -> dict:
        return {
            "current_stage": self._current_stage,
            "global_step": self._global_step,
            "stage_step": self._stage_step,
            "scheduler_state": self._scheduler.state_dict(),
            "stage1_cfg": self.stage1_cfg,
            "stage2_cfg": self.stage2_cfg,
        }

    def load_state_dict(self, state: dict) -> None:
        self._current_stage = state["current_stage"]
        self._global_step = state["global_step"]
        self._stage_step = state["stage_step"]
        self.stage1_cfg = state["stage1_cfg"]
        self.stage2_cfg = state["stage2_cfg"]

        cfg = self.stage1_cfg if self._current_stage == 1 else self.stage2_cfg
        self._scheduler = WarmupCosineScheduler.from_config(self.optimizer, cfg)
        self._scheduler.load_state_dict(state["scheduler_state"])
