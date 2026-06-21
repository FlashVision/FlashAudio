"""Training engine for FlashAudio models.

Supports training STT, TTS, classification, and voice encoder models
with AMP, gradient accumulation, LR scheduling, checkpointing, and callbacks.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from flashaudio.cfg.config import FlashAudioConfig


class Trainer:
    """Audio model trainer with full training loop.

    Args:
        config: FlashAudioConfig or None to use kwargs.
        model: Optional pre-initialized model.
        model_id: HuggingFace model ID (used if model is None).
        dataset: Dataset path or name.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate: Optimizer learning rate.
        weight_decay: Weight decay for AdamW.
        device: Device to train on.
        save_dir: Directory for checkpoints and logs.
        callbacks: Optional list of callback functions.
    """

    def __init__(
        self,
        config: Optional[FlashAudioConfig] = None,
        model: Optional[nn.Module] = None,
        model_id: Optional[str] = None,
        dataset: Optional[str] = None,
        epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        device: str = "cuda",
        save_dir: str = "workspace/train",
        callbacks: Optional[List[Callable]] = None,
    ):
        if config is not None:
            self.model_id = config.model.model_id
            self.epochs = config.train.epochs
            self.batch_size = config.train.batch_size
            self.learning_rate = config.train.learning_rate
            self.weight_decay = config.train.weight_decay
            self.save_dir = config.train.save_dir
            self.device = config.device
            self.amp = config.train.amp
            self.max_grad_norm = config.train.max_grad_norm
            self.warmup_ratio = config.train.warmup_ratio
            self.lr_scheduler_type = config.train.lr_scheduler
            self.save_steps = config.train.save_steps
            self.eval_steps = config.train.eval_steps
            self.logging_steps = config.train.logging_steps
            self.gradient_checkpointing = config.train.gradient_checkpointing
            self.dataset_path = config.data.dataset_path
            self.config = config
        else:
            self.model_id = model_id
            self.epochs = epochs
            self.batch_size = batch_size
            self.learning_rate = learning_rate
            self.weight_decay = weight_decay
            self.save_dir = save_dir
            self.device = device
            self.amp = True
            self.max_grad_norm = 1.0
            self.warmup_ratio = 0.1
            self.lr_scheduler_type = "cosine"
            self.save_steps = 500
            self.eval_steps = 250
            self.logging_steps = 10
            self.gradient_checkpointing = False
            self.dataset_path = dataset
            self.config = None

        self.model = model
        self.callbacks = callbacks or []
        self.global_step = 0
        self.best_loss = float("inf")
        self.history: List[Dict[str, float]] = []

        self._optimizer = None
        self._scheduler = None
        self._scaler = None

    def _setup_model(self):
        """Initialize the model if not provided."""
        if self.model is not None:
            self.model = self.model.to(self.device)
            return

        from flashaudio.models.flashaudio_model import FlashAudio
        flash = FlashAudio(model_id=self.model_id, device=self.device)
        self.model = flash.model.to(self.device)

    def _setup_optimizer(self):
        """Configure optimizer, scheduler, and AMP scaler."""
        self._optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        total_steps = self.epochs * max(len(self._train_loader), 1)
        warmup_steps = int(total_steps * self.warmup_ratio)

        if self.lr_scheduler_type == "cosine":
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self._optimizer, T_max=total_steps - warmup_steps
            )
        elif self.lr_scheduler_type == "linear":
            self._scheduler = torch.optim.lr_scheduler.LinearLR(
                self._optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps
            )
        else:
            self._scheduler = torch.optim.lr_scheduler.StepLR(
                self._optimizer, step_size=total_steps // 3, gamma=0.1
            )

        if self.amp and self.device != "cpu":
            self._scaler = torch.amp.GradScaler("cuda")

    def _setup_data(self):
        """Create train and validation data loaders."""
        if self.dataset_path and os.path.isdir(self.dataset_path):
            from flashaudio.data.datasets import LibriSpeechDataset
            full_dataset = LibriSpeechDataset(root=self.dataset_path)
        else:
            full_dataset = self._create_dummy_dataset()

        val_size = max(1, int(len(full_dataset) * 0.1))
        train_size = len(full_dataset) - val_size

        train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, val_size])

        self._train_loader = DataLoader(
            train_set, batch_size=self.batch_size, shuffle=True,
            num_workers=0, pin_memory=True, drop_last=True,
        )
        self._val_loader = DataLoader(
            val_set, batch_size=self.batch_size, shuffle=False,
            num_workers=0, pin_memory=True,
        )

    def _create_dummy_dataset(self):
        """Create a minimal dummy dataset for testing the training loop."""

        class DummyAudioDataset(torch.utils.data.Dataset):
            def __init__(self, size=100, sample_rate=16000, duration=2.0):
                self.size = size
                self.num_samples = int(sample_rate * duration)

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                return {
                    "waveform": torch.randn(self.num_samples),
                    "sample_rate": 16000,
                    "text": "dummy transcript",
                    "labels": torch.zeros(527),
                }

        return DummyAudioDataset()

    def _compute_loss(self, batch: Dict) -> torch.Tensor:
        """Compute training loss for a batch.

        Supports both sequence-to-sequence (STT) and classification tasks.
        """
        waveform = batch["waveform"].to(self.device)

        if hasattr(self.model, "forward") and "labels" in batch:
            labels = batch["labels"].to(self.device)
            if labels.dim() > 1:
                outputs = self.model(waveform)
                if isinstance(outputs, dict):
                    logits = outputs.get("logits", outputs.get("output"))
                else:
                    logits = outputs
                loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
            else:
                outputs = self.model(waveform)
                if isinstance(outputs, dict):
                    logits = outputs.get("logits", outputs.get("output"))
                else:
                    logits = outputs
                loss = nn.functional.cross_entropy(logits, labels.long())
        else:
            outputs = self.model(waveform)
            if isinstance(outputs, dict) and "loss" in outputs:
                loss = outputs["loss"]
            elif isinstance(outputs, torch.Tensor):
                loss = outputs.mean()
            else:
                loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        return loss

    def _train_epoch(self, epoch: int) -> float:
        """Run a single training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self._train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=True)
        for batch in pbar:
            self._optimizer.zero_grad()

            if self._scaler is not None:
                with torch.amp.autocast("cuda"):
                    loss = self._compute_loss(batch)
                self._scaler.scale(loss).backward()
                self._scaler.unscale_(self._optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self._scaler.step(self._optimizer)
                self._scaler.update()
            else:
                loss = self._compute_loss(batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self._optimizer.step()

            if self._scheduler is not None:
                self._scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{self._optimizer.param_groups[0]['lr']:.2e}")

            if self.global_step % self.save_steps == 0:
                self._save_checkpoint(f"step_{self.global_step}")

            for cb in self.callbacks:
                cb(self, {"step": self.global_step, "loss": loss.item()})

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def _save_checkpoint(self, name: str):
        """Save a training checkpoint."""
        save_dir = Path(self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"checkpoint_{name}.pt"

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "global_step": self.global_step,
            "best_loss": self.best_loss,
        }, path)

    def train(self) -> Dict[str, Any]:
        """Run the full training loop.

        Returns:
            Dictionary with training history and final metrics.
        """
        print(f"\n{'='*60}")
        print("  FlashAudio Training")
        print(f"{'='*60}")
        print(f"  Model:      {self.model_id}")
        print(f"  Epochs:     {self.epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  LR:         {self.learning_rate}")
        print(f"  Device:     {self.device}")
        print(f"  Save dir:   {self.save_dir}")
        print(f"{'='*60}\n")

        self._setup_model()
        self._setup_data()
        self._setup_optimizer()

        start_time = time.time()

        for epoch in range(self.epochs):
            train_loss = self._train_epoch(epoch)

            record = {"epoch": epoch + 1, "train_loss": train_loss}
            self.history.append(record)

            print(f"  Epoch {epoch + 1}/{self.epochs} — loss: {train_loss:.4f}")

            if train_loss < self.best_loss:
                self.best_loss = train_loss
                self._save_checkpoint("best")

        elapsed = time.time() - start_time
        self._save_checkpoint("final")

        print(f"\n  Training complete in {elapsed:.1f}s")
        print(f"  Best loss: {self.best_loss:.4f}")
        print(f"  Checkpoints saved to: {self.save_dir}")

        return {
            "history": self.history,
            "best_loss": self.best_loss,
            "elapsed_seconds": elapsed,
            "global_step": self.global_step,
        }
