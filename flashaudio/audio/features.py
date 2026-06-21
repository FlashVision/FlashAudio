"""Audio feature extraction.

Computes mel spectrograms, MFCCs, chromagrams, spectral features,
and other audio representations used across FlashAudio pipelines.
"""

from __future__ import annotations

from typing import Dict, Optional, Union

import numpy as np
import torch
import torchaudio


class AudioFeatureExtractor:
    """Comprehensive audio feature extractor.

    Extracts a variety of spectral and temporal features from
    audio waveforms using torchaudio and librosa.

    Args:
        sample_rate: Expected audio sample rate.
        n_fft: FFT window size.
        hop_length: Hop length between frames.
        n_mels: Number of mel frequency bins.
        n_mfcc: Number of MFCC coefficients.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
        n_mfcc: int = 13,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc

    def mel_spectrogram(
        self,
        waveform: torch.Tensor,
        sample_rate: Optional[int] = None,
        to_db: bool = True,
    ) -> torch.Tensor:
        """Compute mel spectrogram.

        Args:
            waveform: [channels, samples] or [samples].
            sample_rate: Override sample rate.
            to_db: Convert to decibel scale.

        Returns:
            Mel spectrogram [channels, n_mels, time].
        """
        sr = sample_rate or self.sample_rate
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        )
        mel = mel_transform(waveform)

        if to_db:
            mel = torchaudio.transforms.AmplitudeToDB()(mel)

        return mel

    def mfcc(
        self,
        waveform: torch.Tensor,
        sample_rate: Optional[int] = None,
    ) -> torch.Tensor:
        """Compute Mel-frequency cepstral coefficients.

        Args:
            waveform: [channels, samples] or [samples].
            sample_rate: Override sample rate.

        Returns:
            MFCC features [channels, n_mfcc, time].
        """
        sr = sample_rate or self.sample_rate
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sr,
            n_mfcc=self.n_mfcc,
            melkwargs={
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "n_mels": self.n_mels,
            },
        )
        return mfcc_transform(waveform)

    def chromagram(
        self,
        waveform: torch.Tensor,
        sample_rate: Optional[int] = None,
        n_chroma: int = 12,
    ) -> np.ndarray:
        """Compute chromagram (pitch class distribution over time).

        Args:
            waveform: [channels, samples] or [samples].
            sample_rate: Override sample rate.
            n_chroma: Number of chroma bins (default 12 = semitones).

        Returns:
            Chromagram numpy array [n_chroma, time].
        """
        import librosa

        sr = sample_rate or self.sample_rate

        if isinstance(waveform, torch.Tensor):
            y = waveform.cpu().numpy()
        else:
            y = waveform

        if y.ndim == 2:
            y = y.mean(axis=0)

        chroma = librosa.feature.chroma_stft(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length, n_chroma=n_chroma
        )
        return chroma

    def spectral_centroid(
        self,
        waveform: torch.Tensor,
        sample_rate: Optional[int] = None,
    ) -> np.ndarray:
        """Compute spectral centroid (brightness measure).

        Args:
            waveform: Audio waveform.
            sample_rate: Override sample rate.

        Returns:
            Spectral centroid [1, time].
        """
        import librosa

        sr = sample_rate or self.sample_rate
        y = self._to_numpy(waveform)

        return librosa.feature.spectral_centroid(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
        )

    def spectral_rolloff(
        self,
        waveform: torch.Tensor,
        sample_rate: Optional[int] = None,
        roll_percent: float = 0.85,
    ) -> np.ndarray:
        """Compute spectral rolloff frequency.

        Args:
            waveform: Audio waveform.
            sample_rate: Override sample rate.
            roll_percent: Rolloff percentage.

        Returns:
            Spectral rolloff [1, time].
        """
        import librosa

        sr = sample_rate or self.sample_rate
        y = self._to_numpy(waveform)

        return librosa.feature.spectral_rolloff(
            y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length, roll_percent=roll_percent
        )

    def zero_crossing_rate(self, waveform: torch.Tensor) -> np.ndarray:
        """Compute zero-crossing rate.

        Args:
            waveform: Audio waveform.

        Returns:
            Zero-crossing rate [1, time].
        """
        import librosa

        y = self._to_numpy(waveform)
        return librosa.feature.zero_crossing_rate(y=y, hop_length=self.hop_length)

    def rms_energy(self, waveform: torch.Tensor) -> np.ndarray:
        """Compute RMS energy over time.

        Args:
            waveform: Audio waveform.

        Returns:
            RMS energy [1, time].
        """
        import librosa

        y = self._to_numpy(waveform)
        return librosa.feature.rms(y=y, hop_length=self.hop_length)

    def extract_all(
        self,
        waveform: torch.Tensor,
        sample_rate: Optional[int] = None,
    ) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        """Extract all available features from a waveform.

        Args:
            waveform: Audio waveform.
            sample_rate: Override sample rate.

        Returns:
            Dictionary of feature_name -> feature_tensor/array.
        """
        sr = sample_rate or self.sample_rate

        features = {
            "mel_spectrogram": self.mel_spectrogram(waveform, sr),
            "mfcc": self.mfcc(waveform, sr),
            "chromagram": self.chromagram(waveform, sr),
            "spectral_centroid": self.spectral_centroid(waveform, sr),
            "spectral_rolloff": self.spectral_rolloff(waveform, sr),
            "zero_crossing_rate": self.zero_crossing_rate(waveform),
            "rms_energy": self.rms_energy(waveform),
        }

        return features

    def _to_numpy(self, waveform: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Convert waveform to 1D numpy array."""
        if isinstance(waveform, torch.Tensor):
            y = waveform.cpu().numpy()
        else:
            y = waveform
        if y.ndim == 2:
            y = y.mean(axis=0)
        return y
