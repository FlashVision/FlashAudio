"""Audio classification model.

CNN-based architecture for audio event classification operating
on mel spectrogram inputs. Supports AudioSet-style multi-label
classification.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashaudio.registry import MODELS


class ConvBlock(nn.Module):
    """Double convolution block with batch norm and optional pooling.

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Convolution kernel size.
        pool_size: Max pooling size (None to skip).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool_size: Optional[int] = 2,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(pool_size) if pool_size else nn.Identity()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(F.relu(self.bn2(self.conv2(x))))
        return self.pool(x)


@MODELS.register("AudioClassifier")
class AudioClassifierModel(nn.Module):
    """CNN audio classifier operating on mel spectrograms.

    Takes raw waveforms (auto-converted to mel spectrograms) or
    pre-computed mel spectrogram inputs and outputs class logits.

    Args:
        num_classes: Number of output classes.
        n_mels: Number of mel frequency bins.
        sample_rate: Expected audio sample rate.
        base_channels: Base number of channels (doubled at each block).
    """

    def __init__(
        self,
        num_classes: int = 527,
        n_mels: int = 80,
        sample_rate: int = 16000,
        base_channels: int = 32,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.n_mels = n_mels
        self.sample_rate = sample_rate

        self.features = nn.Sequential(
            ConvBlock(1, base_channels),
            ConvBlock(base_channels, base_channels * 2),
            ConvBlock(base_channels * 2, base_channels * 4),
            ConvBlock(base_channels * 4, base_channels * 8),
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(base_channels * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor. Accepts:
               - Raw waveform [batch, samples]
               - Mel spectrogram [batch, n_mels, time]
               - 4D tensor [batch, 1, n_mels, time]

        Returns:
            Logits tensor [batch, num_classes].
        """
        if x.dim() == 2 and x.shape[1] > self.n_mels * 4:
            x = self._waveform_to_mel(x)

        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.features(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        return self.classifier(x)

    def _waveform_to_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert raw waveform to log mel spectrogram."""
        import torchaudio
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=self.n_mels,
        ).to(waveform.device)
        mel = mel_transform(waveform)
        mel = torch.log(mel.clamp(min=1e-5))
        return mel

    def predict(self, x: torch.Tensor, top_k: int = 5) -> Dict:
        """Predict class probabilities.

        Args:
            x: Input tensor (waveform or mel).
            top_k: Number of top predictions to return.

        Returns:
            Dictionary with 'probabilities' and 'indices'.
        """
        self.eval()
        with torch.no_grad():
            logits = self(x)
            probs = torch.sigmoid(logits)
            top_probs, top_indices = probs.topk(top_k, dim=-1)

        return {
            "probabilities": top_probs,
            "indices": top_indices,
        }
