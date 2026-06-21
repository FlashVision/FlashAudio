"""Visualization utilities for waveforms, spectrograms, and features.

Requires matplotlib (optional dependency: pip install flashaudio[analytics]).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch


def plot_waveform(
    waveform: Union[torch.Tensor, np.ndarray],
    sample_rate: int = 16000,
    title: str = "Waveform",
    output_path: Optional[str] = None,
    figsize: tuple = (12, 3),
):
    """Plot an audio waveform.

    Args:
        waveform: Audio tensor or array.
        sample_rate: Sample rate for time axis.
        title: Plot title.
        output_path: If provided, save the figure to this path.
        figsize: Figure size (width, height).
    """
    import matplotlib.pyplot as plt

    if isinstance(waveform, torch.Tensor):
        waveform = waveform.cpu().numpy()

    if waveform.ndim == 2:
        waveform = waveform[0]

    time_axis = np.arange(len(waveform)) / sample_rate

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(time_axis, waveform, linewidth=0.5, color="#2196F3")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)
    ax.set_xlim(0, time_axis[-1])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close()


def plot_spectrogram(
    spectrogram: Union[torch.Tensor, np.ndarray],
    sample_rate: int = 16000,
    hop_length: int = 256,
    title: str = "Spectrogram",
    output_path: Optional[str] = None,
    figsize: tuple = (12, 4),
    cmap: str = "magma",
):
    """Plot a spectrogram.

    Args:
        spectrogram: 2D spectrogram [freq, time].
        sample_rate: Audio sample rate.
        hop_length: Hop length used to compute the spectrogram.
        title: Plot title.
        output_path: Save path.
        figsize: Figure size.
        cmap: Colormap name.
    """
    import matplotlib.pyplot as plt

    if isinstance(spectrogram, torch.Tensor):
        spectrogram = spectrogram.cpu().numpy()

    if spectrogram.ndim == 3:
        spectrogram = spectrogram[0]

    fig, ax = plt.subplots(figsize=figsize)
    num_frames = spectrogram.shape[1]
    duration = num_frames * hop_length / sample_rate

    im = ax.imshow(
        spectrogram,
        aspect="auto",
        origin="lower",
        extent=[0, duration, 0, sample_rate / 2],
        cmap=cmap,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Magnitude (dB)")
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close()


def plot_mel_spectrogram(
    mel: Union[torch.Tensor, np.ndarray],
    sample_rate: int = 16000,
    hop_length: int = 256,
    title: str = "Mel Spectrogram",
    output_path: Optional[str] = None,
    figsize: tuple = (12, 4),
    cmap: str = "inferno",
):
    """Plot a mel spectrogram.

    Args:
        mel: Mel spectrogram [n_mels, time].
        sample_rate: Audio sample rate.
        hop_length: Hop length.
        title: Plot title.
        output_path: Save path.
        figsize: Figure size.
        cmap: Colormap.
    """
    import matplotlib.pyplot as plt

    if isinstance(mel, torch.Tensor):
        mel = mel.cpu().numpy()

    if mel.ndim == 3:
        mel = mel[0]

    fig, ax = plt.subplots(figsize=figsize)
    num_frames = mel.shape[1]
    duration = num_frames * hop_length / sample_rate

    im = ax.imshow(
        mel,
        aspect="auto",
        origin="lower",
        extent=[0, duration, 0, mel.shape[0]],
        cmap=cmap,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel bin")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Power (dB)")
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close()


def plot_features(
    features: dict,
    sample_rate: int = 16000,
    hop_length: int = 256,
    output_path: Optional[str] = None,
):
    """Plot multiple audio features in a grid.

    Args:
        features: Dictionary of feature_name -> array.
        sample_rate: Audio sample rate.
        hop_length: Hop length.
        output_path: Save path.
    """
    import matplotlib.pyplot as plt

    plot_features_list = ["mel_spectrogram", "mfcc", "chromagram"]
    available = [f for f in plot_features_list if f in features]

    if not available:
        return

    fig, axes = plt.subplots(len(available), 1, figsize=(12, 3 * len(available)))
    if len(available) == 1:
        axes = [axes]

    for ax, name in zip(axes, available):
        data = features[name]
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        if data.ndim == 3:
            data = data[0]

        ax.imshow(data, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(name.replace("_", " ").title())
        ax.set_ylabel("Bin")

    axes[-1].set_xlabel("Frame")
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")

    plt.show()
    plt.close()
