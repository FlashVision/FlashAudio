"""Core audio I/O and processing utilities.

Provides load/save, resampling, and mel spectrogram computation
using torchaudio and librosa as backends.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torchaudio


def load_audio(
    path: Union[str, Path],
    sample_rate: Optional[int] = 16000,
    mono: bool = True,
    normalize: bool = True,
) -> Tuple[torch.Tensor, int]:
    """Load an audio file and optionally resample.

    Args:
        path: Path to the audio file.
        sample_rate: Target sample rate. If None, returns native rate.
        mono: If True, mix down to mono.
        normalize: If True, normalize waveform to [-1, 1].

    Returns:
        Tuple of (waveform tensor [channels, samples], sample_rate).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    waveform, sr = torchaudio.load(str(path))

    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate is not None and sr != sample_rate:
        waveform = resample_audio(waveform, sr, sample_rate)
        sr = sample_rate

    if normalize:
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak

    return waveform, sr


def save_audio(
    path: Union[str, Path],
    waveform: torch.Tensor,
    sample_rate: int,
    encoding: str = "PCM_S",
    bits_per_sample: int = 16,
) -> Path:
    """Save a waveform tensor to an audio file.

    Args:
        path: Output file path (.wav, .flac, etc.).
        waveform: Audio tensor of shape [channels, samples] or [samples].
        sample_rate: Sample rate in Hz.
        encoding: Audio encoding (default PCM_S for WAV).
        bits_per_sample: Bit depth (default 16).

    Returns:
        Path to the saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    waveform = waveform.cpu().float()

    peak = waveform.abs().max()
    if peak > 1.0:
        waveform = waveform / peak

    torchaudio.save(
        str(path),
        waveform,
        sample_rate,
        encoding=encoding,
        bits_per_sample=bits_per_sample,
    )
    return path


def resample_audio(
    waveform: torch.Tensor,
    orig_sr: int,
    target_sr: int,
) -> torch.Tensor:
    """Resample audio to a target sample rate.

    Args:
        waveform: Input tensor [channels, samples].
        orig_sr: Original sample rate.
        target_sr: Target sample rate.

    Returns:
        Resampled waveform tensor.
    """
    if orig_sr == target_sr:
        return waveform
    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
    return resampler(waveform)


def compute_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 80,
    win_length: Optional[int] = None,
    fmin: float = 0.0,
    fmax: Optional[float] = 8000.0,
    power: float = 2.0,
    normalized: bool = True,
    to_db: bool = True,
) -> torch.Tensor:
    """Compute a mel spectrogram from a waveform.

    Args:
        waveform: Input tensor [channels, samples] or [samples].
        sample_rate: Sample rate of the waveform.
        n_fft: FFT window size.
        hop_length: Hop length between frames.
        n_mels: Number of mel frequency bands.
        win_length: Window length (defaults to n_fft).
        fmin: Minimum frequency for mel filterbank.
        fmax: Maximum frequency for mel filterbank.
        power: Exponent for magnitude spectrogram (2.0 = power, 1.0 = amplitude).
        normalized: Whether to normalize the mel spectrogram.
        to_db: Whether to convert to decibel scale.

    Returns:
        Mel spectrogram tensor of shape [channels, n_mels, time].
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    win_length = win_length or n_fft

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        win_length=win_length,
        f_min=fmin,
        f_max=fmax,
        power=power,
        normalized=normalized,
    )

    mel_spec = mel_transform(waveform)

    if to_db:
        amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power" if power == 2.0 else "magnitude")
        mel_spec = amplitude_to_db(mel_spec)

    return mel_spec


def get_audio_info(path: Union[str, Path]) -> dict:
    """Get metadata about an audio file.

    Returns:
        Dictionary with keys: sample_rate, num_channels, num_frames, duration, encoding.
    """
    info = torchaudio.info(str(path))
    return {
        "sample_rate": info.sample_rate,
        "num_channels": info.num_channels,
        "num_frames": info.num_frames,
        "duration": info.num_frames / info.sample_rate,
        "encoding": info.encoding,
    }


def pad_or_trim(waveform: torch.Tensor, target_length: int) -> torch.Tensor:
    """Pad or trim a waveform to a target length.

    Args:
        waveform: Input tensor [..., samples].
        target_length: Target number of samples.

    Returns:
        Padded or trimmed waveform.
    """
    current_length = waveform.shape[-1]

    if current_length == target_length:
        return waveform
    elif current_length > target_length:
        return waveform[..., :target_length]
    else:
        pad_size = target_length - current_length
        return torch.nn.functional.pad(waveform, (0, pad_size))


def to_numpy(waveform: torch.Tensor) -> np.ndarray:
    """Convert a torch waveform to a numpy array."""
    return waveform.detach().cpu().numpy()
