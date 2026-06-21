"""Speech Emotion Recognition (SER) from prosodic and spectral features.

Implements a speech emotion recognition system using:
- Multi-scale feature extraction from mel spectrograms
- Temporal attention pooling for utterance-level representation
- Classification into standard emotion categories

Reference: General SER approaches using deep learning on speech features.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashaudio.registry import MODELS


EMOTION_LABELS = ["neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"]


class MultiScaleConvBlock(nn.Module):
    """Multi-scale 1D convolution for extracting features at different temporal resolutions."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_small = nn.Conv1d(in_channels, out_channels // 3, 3, 1, 1)
        self.conv_med = nn.Conv1d(in_channels, out_channels // 3, 7, 1, 3)
        self.conv_large = nn.Conv1d(in_channels, out_channels - 2 * (out_channels // 3), 15, 1, 7)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.conv_small(x)
        m = self.conv_med(x)
        lg = self.conv_large(x)
        return F.relu(self.bn(torch.cat([s, m, lg], dim=1)), inplace=True)


class TemporalAttentionPooling(nn.Module):
    """Attention-based temporal pooling for utterance-level features."""

    def __init__(self, dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.Tanh(),
            nn.Linear(dim // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) sequential features.

        Returns:
            (B, D) attention-weighted pooled features.
        """
        weights = self.attention(x).softmax(dim=1)
        return (x * weights).sum(dim=1)


class ProsodyExtractor(nn.Module):
    """Extract prosodic features (pitch, energy, rate) from mel spectrogram."""

    def __init__(self, n_mels: int = 80, out_dim: int = 64):
        super().__init__()
        self.pitch_net = nn.Sequential(
            nn.Conv1d(n_mels, 64, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, out_dim // 2, 3, 1, 1),
        )
        self.energy_net = nn.Sequential(
            nn.Conv1d(1, 32, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, out_dim // 2, 3, 1, 1),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (B, n_mels, T) mel spectrogram.

        Returns:
            (B, out_dim, T) prosodic features.
        """
        pitch_feat = self.pitch_net(mel)
        energy = mel.mean(dim=1, keepdim=True)
        energy_feat = self.energy_net(energy)
        return torch.cat([pitch_feat, energy_feat], dim=1)


class EmotionEncoder(nn.Module):
    """Deep encoder for spectral emotion features."""

    def __init__(self, in_channels: int = 80, hidden_dim: int = 256):
        super().__init__()
        self.layers = nn.Sequential(
            MultiScaleConvBlock(in_channels, hidden_dim),
            nn.MaxPool1d(2),
            MultiScaleConvBlock(hidden_dim, hidden_dim),
            nn.MaxPool1d(2),
            MultiScaleConvBlock(hidden_dim, hidden_dim),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = hidden_dim

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        return self.layers(mel)


@MODELS.register("EmotionRecognition")
class EmotionRecognizer(nn.Module):
    """Speech Emotion Recognition model.

    Combines spectral feature encoding with prosodic analysis
    and temporal attention for utterance-level emotion classification.

    Args:
        n_mels: Input mel spectrogram channels.
        hidden_dim: Encoder hidden dimension.
        num_emotions: Number of emotion categories.
        prosody_dim: Prosodic feature dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        n_mels: int = 80,
        hidden_dim: int = 256,
        num_emotions: int = 7,
        prosody_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.num_emotions = num_emotions
        self.emotion_labels = EMOTION_LABELS[:num_emotions]

        self.encoder = EmotionEncoder(n_mels, hidden_dim)
        self.prosody = ProsodyExtractor(n_mels, prosody_dim)

        combined_dim = hidden_dim + prosody_dim
        self.fusion = nn.Sequential(
            nn.Conv1d(combined_dim, hidden_dim, 1),
            nn.ReLU(inplace=True),
        )

        self.temporal_pool = TemporalAttentionPooling(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_emotions),
        )

        self.valence_arousal = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
            nn.Tanh(),
        )

    def forward(self, mel: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict emotion from mel spectrogram.

        Args:
            mel: (B, n_mels, T) mel spectrogram.

        Returns:
            Dict with 'logits', 'probabilities', 'valence_arousal', 'features'.
        """
        spectral = self.encoder(mel)
        prosodic = self.prosody(mel)

        if prosodic.shape[-1] != spectral.shape[-1]:
            prosodic = F.interpolate(prosodic, size=spectral.shape[-1], mode="linear", align_corners=False)

        combined = self.fusion(torch.cat([spectral, prosodic], dim=1))
        features = self.temporal_pool(combined.transpose(1, 2))

        logits = self.classifier(features)
        probs = F.softmax(logits, dim=-1)
        va = self.valence_arousal(features)

        return {
            "logits": logits,
            "probabilities": probs,
            "valence_arousal": va,
            "features": features,
        }

    def predict(self, mel: torch.Tensor) -> Dict[str, any]:
        """Inference with label names.

        Args:
            mel: (B, n_mels, T) mel spectrogram.

        Returns:
            Dict with emotion predictions.
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(mel)
        pred_idx = output["logits"].argmax(dim=-1)
        labels = [self.emotion_labels[i] for i in pred_idx.tolist()]
        return {
            "emotion": labels,
            "confidence": output["probabilities"].max(dim=-1).values.tolist(),
            "valence": output["valence_arousal"][:, 0].tolist(),
            "arousal": output["valence_arousal"][:, 1].tolist(),
        }

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        va_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute training loss.

        Args:
            predictions: Model output.
            targets: (B,) emotion class labels.
            va_targets: Optional (B, 2) valence-arousal targets.

        Returns:
            Loss dictionary.
        """
        ce_loss = F.cross_entropy(predictions["logits"], targets)
        losses = {"classification": ce_loss}

        if va_targets is not None:
            va_loss = F.mse_loss(predictions["valence_arousal"], va_targets)
            losses["valence_arousal"] = va_loss

        losses["total"] = sum(losses.values())
        return losses
