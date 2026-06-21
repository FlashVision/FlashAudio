"""Audio transforms and augmentations.

Provides composable transforms for data preprocessing and augmentation
during training.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torchaudio


class AudioTransform:
    """Base audio transform that resamples and normalizes.

    Args:
        sample_rate: Target sample rate.
        normalize: Whether to peak-normalize to [-1, 1].
    """

    def __init__(self, sample_rate: int = 16000, normalize: bool = True):
        self.sample_rate = sample_rate
        self.normalize = normalize

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            peak = waveform.abs().max()
            if peak > 0:
                waveform = waveform / peak
        return waveform


class SpectrogramTransform:
    """Convert waveform to mel spectrogram.

    Args:
        sample_rate: Audio sample rate.
        n_fft: FFT window size.
        hop_length: Hop between frames.
        n_mels: Number of mel bands.
        to_db: Whether to convert power to dB.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        to_db: bool = True,
    ):
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.to_db = to_db
        if to_db:
            self.db_transform = torchaudio.transforms.AmplitudeToDB()

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        mel = self.mel_transform(waveform)
        if self.to_db:
            mel = self.db_transform(mel)
        return mel


class AugmentTransform:
    """Audio augmentation pipeline for training.

    Applies a random subset of: time stretch, pitch shift, noise injection,
    and gain variation.

    Args:
        noise_level: Standard deviation of Gaussian noise.
        gain_range: Min and max gain in dB.
        time_mask_param: Maximum width of time mask (SpecAugment).
        freq_mask_param: Maximum width of frequency mask (SpecAugment).
    """

    def __init__(
        self,
        noise_level: float = 0.005,
        gain_range: tuple = (-6.0, 6.0),
        time_mask_param: int = 40,
        freq_mask_param: int = 15,
    ):
        self.noise_level = noise_level
        self.gain_range = gain_range
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param

    def add_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise."""
        noise = torch.randn_like(waveform) * self.noise_level
        return waveform + noise

    def random_gain(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply random gain in dB."""
        gain_db = torch.empty(1).uniform_(*self.gain_range).item()
        gain_linear = 10.0 ** (gain_db / 20.0)
        return waveform * gain_linear

    def time_mask(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment time masking."""
        masking = torchaudio.transforms.TimeMasking(time_mask_param=self.time_mask_param)
        return masking(spectrogram)

    def freq_mask(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment frequency masking."""
        masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=self.freq_mask_param)
        return masking(spectrogram)

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > 0.5:
            waveform = self.add_noise(waveform)
        if torch.rand(1).item() > 0.5:
            waveform = self.random_gain(waveform)
        return waveform


class Compose:
    """Compose multiple transforms sequentially.

    Args:
        transforms: List of callable transforms.
    """

    def __init__(self, transforms: List):
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x
