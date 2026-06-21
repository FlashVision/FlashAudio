"""HiFi-GAN Vocoder — High-fidelity neural vocoder for speech synthesis.

Implements the HiFi-GAN architecture with:
- Generator with transposed convolutions and multi-receptive-field fusion
- Multi-Period Discriminator (MPD) for periodic pattern modeling
- Multi-Scale Discriminator (MSD) for multi-resolution assessment

Reference: "HiFi-GAN: Generative Adversarial Networks for Efficient and
High Fidelity Speech Synthesis" (Kong et al., NeurIPS 2020)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashaudio.registry import MODELS


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


class ResBlock1(nn.Module):
    """Residual block with dilated convolutions (Type 1 from HiFi-GAN)."""

    def __init__(self, channels: int, kernel_size: int = 3, dilations: Tuple[int, ...] = (1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.1),
                nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=get_padding(kernel_size, d)),
            )
            for d in dilations
        ])
        self.convs2 = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.1),
                nn.Conv1d(channels, channels, kernel_size, dilation=1, padding=get_padding(kernel_size, 1)),
            )
            for _ in dilations
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = c1(x)
            xt = c2(xt)
            x = x + xt
        return x


class ResBlock2(nn.Module):
    """Residual block (Type 2 — smaller, for lightweight variants)."""

    def __init__(self, channels: int, kernel_size: int = 3, dilations: Tuple[int, ...] = (1, 3)):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.LeakyReLU(0.1),
                nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=get_padding(kernel_size, d)),
            )
            for d in dilations
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c in self.convs:
            x = x + c(x)
        return x


class MultiReceptiveFieldFusion(nn.Module):
    """Multi-Receptive Field Fusion (MRF) module."""

    def __init__(self, channels: int, kernel_sizes: Tuple[int, ...] = (3, 7, 11), dilations_list: Optional[List] = None):
        super().__init__()
        if dilations_list is None:
            dilations_list = [(1, 3, 5), (1, 3, 5), (1, 3, 5)]
        self.resblocks = nn.ModuleList([
            ResBlock1(channels, k, d) for k, d in zip(kernel_sizes, dilations_list)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        for block in self.resblocks:
            out = out + block(x)
        return out / len(self.resblocks)


@MODELS.register("HiFiGAN")
class HiFiGANGenerator(nn.Module):
    """HiFi-GAN Generator for mel-spectrogram to waveform synthesis.

    Uses transposed convolutions for upsampling with multi-receptive-field
    fusion modules for capturing patterns at different temporal scales.

    Args:
        in_channels: Input mel spectrogram channels.
        upsample_rates: Upsampling rates per stage.
        upsample_kernel_sizes: Kernel sizes for transposed convolutions.
        upsample_initial_channel: Initial hidden channels.
        resblock_kernel_sizes: Kernel sizes for MRF residual blocks.
        resblock_dilations: Dilation patterns per MRF block.
    """

    def __init__(
        self,
        in_channels: int = 80,
        upsample_rates: Tuple[int, ...] = (8, 8, 2, 2),
        upsample_kernel_sizes: Tuple[int, ...] = (16, 16, 4, 4),
        upsample_initial_channel: int = 512,
        resblock_kernel_sizes: Tuple[int, ...] = (3, 7, 11),
        resblock_dilations: Optional[List] = None,
    ):
        super().__init__()
        if resblock_dilations is None:
            resblock_dilations = [(1, 3, 5), (1, 3, 5), (1, 3, 5)]

        self.conv_pre = nn.Conv1d(in_channels, upsample_initial_channel, 7, 1, 3)

        self.ups = nn.ModuleList()
        self.mrfs = nn.ModuleList()

        ch = upsample_initial_channel
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                nn.ConvTranspose1d(ch, ch // 2, k, u, padding=(k - u) // 2)
            )
            self.mrfs.append(MultiReceptiveFieldFusion(ch // 2, resblock_kernel_sizes, resblock_dilations))
            ch = ch // 2

        self.conv_post = nn.Conv1d(ch, 1, 7, 1, 3)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Generate waveform from mel spectrogram.

        Args:
            mel: Mel spectrogram (B, n_mels, T).

        Returns:
            Waveform (B, 1, T * prod(upsample_rates)).
        """
        x = self.conv_pre(mel)
        for up, mrf in zip(self.ups, self.mrfs):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            x = mrf(x)
        x = F.leaky_relu(x, 0.1)
        x = torch.tanh(self.conv_post(x))
        return x.unsqueeze(1) if x.dim() == 2 else x

    @torch.no_grad()
    def infer(self, mel: torch.Tensor) -> torch.Tensor:
        """Inference mode waveform generation."""
        self.eval()
        return self.forward(mel)


class PeriodDiscriminator(nn.Module):
    """Sub-discriminator operating on a specific periodic pattern."""

    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3):
        super().__init__()
        self.period = period

        self.convs = nn.ModuleList([
            nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), (get_padding(kernel_size, 1), 0)),
            nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), (get_padding(kernel_size, 1), 0)),
            nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), (get_padding(kernel_size, 1), 0)),
            nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), (get_padding(kernel_size, 1), 0)),
            nn.Conv2d(1024, 1024, (kernel_size, 1), 1, (get_padding(kernel_size, 1), 0)),
        ])
        self.conv_post = nn.Conv2d(1024, 1, (3, 1), 1, (1, 0))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []
        B, C, T = x.shape
        pad_len = (self.period - (T % self.period)) % self.period
        if pad_len > 0:
            x = F.pad(x, (0, pad_len), "reflect")
        T_padded = x.shape[-1]
        x = x.reshape(B, C, T_padded // self.period, self.period)

        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return x.flatten(1, -1), fmap


class MultiPeriodDiscriminator(nn.Module):
    """Multi-Period Discriminator from HiFi-GAN."""

    def __init__(self, periods: Tuple[int, ...] = (2, 3, 5, 7, 11)):
        super().__init__()
        self.discriminators = nn.ModuleList([PeriodDiscriminator(p) for p in periods])

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        outputs = []
        fmaps = []
        for d in self.discriminators:
            out, fmap = d(x)
            outputs.append(out)
            fmaps.append(fmap)
        return outputs, fmaps


class ScaleDiscriminator(nn.Module):
    """Sub-discriminator operating at a single scale."""

    def __init__(self, norm: str = "spectral"):
        super().__init__()
        norm_fn = nn.utils.spectral_norm if norm == "spectral" else nn.utils.weight_norm

        self.convs = nn.ModuleList([
            norm_fn(nn.Conv1d(1, 128, 15, 1, 7)),
            norm_fn(nn.Conv1d(128, 128, 41, 2, 20, groups=4)),
            norm_fn(nn.Conv1d(128, 256, 41, 2, 20, groups=16)),
            norm_fn(nn.Conv1d(256, 512, 41, 4, 20, groups=16)),
            norm_fn(nn.Conv1d(512, 1024, 41, 4, 20, groups=16)),
            norm_fn(nn.Conv1d(1024, 1024, 41, 1, 20, groups=16)),
            norm_fn(nn.Conv1d(1024, 1024, 5, 1, 2)),
        ])
        self.conv_post = norm_fn(nn.Conv1d(1024, 1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        fmap = []
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return x.flatten(1, -1), fmap


class MultiScaleDiscriminator(nn.Module):
    """Multi-Scale Discriminator from HiFi-GAN."""

    def __init__(self, num_scales: int = 3):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator(norm="spectral" if i == 0 else "weight")
            for i in range(num_scales)
        ])
        self.pooling = nn.ModuleList([
            nn.AvgPool1d(4, 2, 2) for _ in range(num_scales - 1)
        ])

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], List[List[torch.Tensor]]]:
        outputs = []
        fmaps = []
        for i, d in enumerate(self.discriminators):
            if i > 0:
                x = self.pooling[i - 1](x)
            out, fmap = d(x)
            outputs.append(out)
            fmaps.append(fmap)
        return outputs, fmaps


class HiFiGANLoss:
    """Loss functions for HiFi-GAN training."""

    @staticmethod
    def generator_loss(disc_outputs: List[torch.Tensor]) -> torch.Tensor:
        loss = torch.tensor(0.0)
        for dg in disc_outputs:
            loss = loss + torch.mean((1 - dg) ** 2)
        return loss

    @staticmethod
    def discriminator_loss(
        disc_real_outputs: List[torch.Tensor],
        disc_gen_outputs: List[torch.Tensor],
    ) -> torch.Tensor:
        loss = torch.tensor(0.0)
        for dr, dg in zip(disc_real_outputs, disc_gen_outputs):
            loss = loss + torch.mean((1 - dr) ** 2) + torch.mean(dg ** 2)
        return loss

    @staticmethod
    def feature_matching_loss(
        fmap_real: List[List[torch.Tensor]],
        fmap_gen: List[List[torch.Tensor]],
    ) -> torch.Tensor:
        loss = torch.tensor(0.0)
        for fr, fg in zip(fmap_real, fmap_gen):
            for r, g in zip(fr, fg):
                loss = loss + F.l1_loss(r, g)
        return loss * 2

    @staticmethod
    def mel_spectrogram_loss(y: torch.Tensor, y_hat: torch.Tensor, n_mels: int = 80) -> torch.Tensor:
        return F.l1_loss(y, y_hat) * 45
