"""Training callbacks for monitoring and control.

Provides pluggable callbacks for early stopping, checkpointing,
and learning rate logging during training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Args:
        patience: Number of epochs to wait before stopping.
        min_delta: Minimum change to qualify as improvement.
        mode: 'min' for loss, 'max' for accuracy.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.counter = 0
        self.should_stop = False

    def __call__(self, trainer, logs: Dict[str, Any]):
        value = logs.get("loss", logs.get("val_loss", 0))

        if self.mode == "min":
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"  Early stopping triggered after {self.patience} epochs without improvement")

    def reset(self):
        """Reset the callback state."""
        self.best_value = float("inf") if self.mode == "min" else float("-inf")
        self.counter = 0
        self.should_stop = False


class ModelCheckpoint:
    """Save model checkpoints during training.

    Args:
        save_dir: Directory to save checkpoints.
        save_every: Save every N steps.
        save_best: Whether to save the best model.
        mode: 'min' for loss, 'max' for accuracy.
    """

    def __init__(
        self,
        save_dir: str = "workspace/checkpoints",
        save_every: int = 500,
        save_best: bool = True,
        mode: str = "min",
    ):
        self.save_dir = Path(save_dir)
        self.save_every = save_every
        self.save_best = save_best
        self.mode = mode
        self.best_value = float("inf") if mode == "min" else float("-inf")

    def __call__(self, trainer, logs: Dict[str, Any]):
        step = logs.get("step", 0)
        value = logs.get("loss", 0)

        if step % self.save_every == 0 and step > 0:
            self._save(trainer, f"step_{step}")

        if self.save_best:
            if self.mode == "min":
                improved = value < self.best_value
            else:
                improved = value > self.best_value

            if improved:
                self.best_value = value
                self._save(trainer, "best")

    def _save(self, trainer, name: str):
        """Save a checkpoint."""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        path = self.save_dir / f"checkpoint_{name}.pt"

        torch.save({
            "model_state_dict": trainer.model.state_dict(),
            "global_step": trainer.global_step,
            "best_loss": trainer.best_loss,
        }, path)


class LearningRateLogger:
    """Log learning rate during training.

    Args:
        log_every: Log every N steps.
    """

    def __init__(self, log_every: int = 100):
        self.log_every = log_every
        self.lr_history = []

    def __call__(self, trainer, logs: Dict[str, Any]):
        step = logs.get("step", 0)

        if step % self.log_every == 0 and hasattr(trainer, "_optimizer") and trainer._optimizer:
            lr = trainer._optimizer.param_groups[0]["lr"]
            self.lr_history.append({"step": step, "lr": lr})
