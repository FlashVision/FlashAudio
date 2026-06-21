"""Voice encoder for speaker embeddings.

ECAPA-TDNN inspired architecture that produces fixed-size speaker
embeddings from variable-length audio, used for voice cloning
and speaker verification.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashaudio.registry import MODELS


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention.

    Args:
        channels: Number of input channels.
        reduction: Channel reduction ratio.
    """

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.shape
        w = self.pool(x).squeeze(-1)
        w = self.fc(w).unsqueeze(-1)
        return x * w


class Res2Block(nn.Module):
    """Res2Net-style block with multi-scale processing.

    Args:
        channels: Number of channels.
        kernel_size: Convolution kernel size.
        scale: Number of parallel scales.
        dilation: Dilation rate.
    """

    def __init__(self, channels: int, kernel_size: int = 3, scale: int = 4, dilation: int = 1):
        super().__init__()
        self.scale = scale
        width = channels // scale

        self.convs = nn.ModuleList([
            nn.Conv1d(width, width, kernel_size, padding=(kernel_size // 2) * dilation, dilation=dilation)
            for _ in range(scale - 1)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm1d(width) for _ in range(scale - 1)])
        self.se = SEBlock(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        chunks = torch.chunk(x, self.scale, dim=1)
        outputs = [chunks[0]]
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            if i == 0:
                out = F.relu(bn(conv(chunks[i + 1])))
            else:
                out = F.relu(bn(conv(chunks[i + 1] + outputs[-1])))
            outputs.append(out)

        x = torch.cat(outputs, dim=1)
        x = self.se(x)
        return x + residual


class AttentiveStatisticsPooling(nn.Module):
    """Attentive statistics pooling layer.

    Computes weighted mean and standard deviation over the time axis
    using a learned attention mechanism.

    Args:
        channels: Input channel dimension.
        attention_dim: Attention hidden dimension.
    """

    def __init__(self, channels: int, attention_dim: int = 128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels, attention_dim, 1),
            nn.Tanh(),
            nn.Conv1d(attention_dim, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = F.softmax(self.attention(x), dim=-1)
        mean = (x * weights).sum(dim=-1)
        var = ((x ** 2) * weights).sum(dim=-1) - mean ** 2
        std = torch.clamp(var, min=1e-9).sqrt()
        return torch.cat([mean, std], dim=-1)


@MODELS.register("VoiceEncoder")
class VoiceEncoderModel(nn.Module):
    """ECAPA-TDNN inspired speaker embedding network.

    Produces fixed-size speaker embeddings from variable-length audio.

    Args:
        input_dim: Number of input features (e.g. 80 for mel spectrogram).
        channels: Base channel width.
        embedding_dim: Output embedding dimension.
        num_blocks: Number of Res2Blocks.
    """

    def __init__(
        self,
        input_dim: int = 80,
        channels: int = 512,
        embedding_dim: int = 192,
        num_blocks: int = 3,
    ):
        super().__init__()

        self.input_conv = nn.Sequential(
            nn.Conv1d(input_dim, channels, 5, padding=2),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )

        self.blocks = nn.ModuleList([
            Res2Block(channels, dilation=2 ** i) for i in range(num_blocks)
        ])

        self.mfa = nn.Conv1d(channels * num_blocks, channels, 1)

        self.asp = AttentiveStatisticsPooling(channels)

        self.bn = nn.BatchNorm1d(channels * 2)
        self.fc = nn.Linear(channels * 2, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract speaker embedding.

        Args:
            x: Input tensor. Accepts:
               - Raw waveform [batch, samples] (auto-converts to mel)
               - Mel spectrogram [batch, n_mels, time]

        Returns:
            Speaker embedding [batch, embedding_dim].
        """
        if x.dim() == 2 and x.shape[1] > 1000:
            x = self._waveform_to_mel(x)

        if x.dim() == 3 and x.shape[1] > x.shape[2]:
            x = x.transpose(1, 2)

        x = self.input_conv(x)

        block_outputs = []
        for block in self.blocks:
            x = block(x)
            block_outputs.append(x)

        x = torch.cat(block_outputs, dim=1)
        x = self.mfa(x)

        x = self.asp(x)
        x = self.bn(x)
        x = self.fc(x)

        return F.normalize(x, p=2, dim=-1)

    def _waveform_to_mel(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert raw waveform to mel spectrogram."""
        import torchaudio
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=512, hop_length=160, n_mels=80
        ).to(waveform.device)
        mel = mel_transform(waveform)
        mel = torch.log(mel.clamp(min=1e-5))
        return mel

    def extract_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract a speaker embedding from a waveform.

        Args:
            waveform: [samples] or [1, samples] waveform.

        Returns:
            [embedding_dim] speaker embedding vector.
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        with torch.no_grad():
            return self(waveform).squeeze(0)
