"""Validation engine for FlashAudio models.

Evaluates model performance on a held-out dataset with configurable
metrics (WER, CER, accuracy, loss).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class Validator:
    """Model validator for evaluation on a dataset.

    Args:
        model: The model to evaluate.
        device: Device for inference.
        metrics: List of metric names to compute.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        metrics: Optional[List[str]] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.metrics = metrics or ["loss"]

    def validate(
        self,
        dataloader: DataLoader,
        criterion: Optional[nn.Module] = None,
    ) -> Dict[str, float]:
        """Run validation on a dataset.

        Args:
            dataloader: Validation DataLoader.
            criterion: Loss function. Defaults to CrossEntropyLoss.

        Returns:
            Dictionary of metric_name -> value.
        """
        self.model.eval()
        criterion = criterion or nn.CrossEntropyLoss()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating", leave=False):
                waveform = batch["waveform"].to(self.device)

                outputs = self.model(waveform)
                if isinstance(outputs, dict):
                    logits = outputs.get("logits", outputs.get("output", None))
                    if logits is None and "loss" in outputs:
                        total_loss += outputs["loss"].item()
                        total_samples += waveform.shape[0]
                        continue
                else:
                    logits = outputs

                if "labels" in batch:
                    labels = batch["labels"].to(self.device)

                    if labels.dim() > 1:
                        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
                        preds = (torch.sigmoid(logits) > 0.5).float()
                        total_correct += (preds == labels).all(dim=-1).sum().item()
                    else:
                        loss = criterion(logits, labels.long())
                        preds = logits.argmax(dim=-1)
                        total_correct += (preds == labels).sum().item()

                    total_loss += loss.item() * waveform.shape[0]
                    total_samples += waveform.shape[0]

                    all_predictions.extend(preds.cpu().tolist())
                    all_targets.extend(labels.cpu().tolist())
                else:
                    total_samples += waveform.shape[0]

        results = {}
        if total_samples > 0:
            results["loss"] = total_loss / total_samples
            results["accuracy"] = total_correct / total_samples
            results["num_samples"] = total_samples

        if "wer" in self.metrics:
            from flashaudio.analytics.metrics import compute_wer
            if all_predictions and all_targets:
                results["wer"] = compute_wer(all_targets, all_predictions)

        return results
