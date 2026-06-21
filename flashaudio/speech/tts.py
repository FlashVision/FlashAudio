"""Text-to-Speech pipeline.

Mel spectrogram synthesis with Griffin-Lim vocoder for waveform
reconstruction. Supports configurable sample rates and speaking speed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from flashaudio.registry import PIPELINES


@PIPELINES.register("TextToSpeech")
class TextToSpeech:
    """Text-to-speech synthesis pipeline.

    Uses a Tacotron2-style model to generate mel spectrograms from text,
    then applies Griffin-Lim to reconstruct waveforms.

    Args:
        model: Optional pre-loaded TTS model.
        device: Device for inference.
        sample_rate: Output audio sample rate.
        n_fft: FFT window size for Griffin-Lim.
        hop_length: Hop length for spectrogram.
        n_mels: Number of mel frequency bins.
    """

    def __init__(
        self,
        model=None,
        device: str = "cuda",
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 80,
    ):
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self._model = model

    @property
    def model(self):
        if self._model is None:
            from flashaudio.models.architectures.tts_model import TTSModel
            self._model = TTSModel(n_mels=self.n_mels).to(self.device)
            self._model.eval()
        return self._model

    def synthesize(
        self,
        text: str,
        output_path: str = "output.wav",
        sample_rate: Optional[int] = None,
        speed: float = 1.0,
    ) -> str:
        """Synthesize speech and save to file.

        Args:
            text: Input text to synthesize.
            output_path: Path to save the output WAV file.
            sample_rate: Output sample rate (overrides default).
            speed: Playback speed multiplier.

        Returns:
            Path to the saved audio file.
        """
        waveform = self.synthesize_to_tensor(text, sample_rate=sample_rate, speed=speed)

        sr = sample_rate or self.sample_rate
        from flashaudio.data.audio_utils import save_audio
        save_audio(output_path, waveform, sr)

        return output_path

    def synthesize_to_tensor(
        self,
        text: str,
        sample_rate: Optional[int] = None,
        speed: float = 1.0,
    ) -> torch.Tensor:
        """Synthesize speech and return as a tensor.

        Args:
            text: Input text.
            sample_rate: Target sample rate.
            speed: Speed multiplier.

        Returns:
            Waveform tensor [1, samples].
        """
        text_ids = self._text_to_ids(text).to(self.device)

        max_steps = max(50, int(len(text) * 10 / speed))

        with torch.no_grad():
            output = self.model(text_ids, max_steps=max_steps)

        mel = output["mel_postnet"]

        waveform = self._griffin_lim(mel.squeeze(0))

        sr = sample_rate or self.sample_rate
        if sr != self.sample_rate:
            from flashaudio.data.audio_utils import resample_audio
            waveform = resample_audio(waveform.unsqueeze(0), self.sample_rate, sr).squeeze(0)

        return waveform.unsqueeze(0)

    def _text_to_ids(self, text: str) -> torch.Tensor:
        """Convert text to character IDs."""
        ids = [min(ord(c), 255) for c in text.lower()]
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)

    def _griffin_lim(
        self,
        mel_spec: torch.Tensor,
        n_iter: int = 32,
    ) -> torch.Tensor:
        """Reconstruct waveform from mel spectrogram using Griffin-Lim.

        This is a classical phase-reconstruction algorithm that iteratively
        estimates the phase of the STFT from a magnitude spectrogram.

        Args:
            mel_spec: Mel spectrogram [n_mels, time].
            n_iter: Number of Griffin-Lim iterations.

        Returns:
            Reconstructed waveform tensor [samples].
        """
        import torchaudio

        mel_spec = mel_spec.cpu()

        if mel_spec.dim() == 3:
            mel_spec = mel_spec.squeeze(0)

        mel_spec_linear = torch.pow(10.0, mel_spec / 20.0)

        inverse_mel = torchaudio.transforms.InverseMelScale(
            n_stft=self.n_fft // 2 + 1,
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
        )
        spectrogram = inverse_mel(mel_spec_linear)

        griffin_lim_transform = torchaudio.transforms.GriffinLim(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_iter=n_iter,
            power=1.0,
        )
        waveform = griffin_lim_transform(spectrogram)

        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak * 0.95

        return waveform

    def get_mel_spectrogram(self, text: str) -> torch.Tensor:
        """Generate mel spectrogram from text without vocoder.

        Useful for visualization or custom vocoder pipelines.

        Args:
            text: Input text.

        Returns:
            Mel spectrogram tensor [1, n_mels, time].
        """
        text_ids = self._text_to_ids(text).to(self.device)
        max_steps = max(50, len(text) * 10)

        with torch.no_grad():
            output = self.model(text_ids, max_steps=max_steps)

        return output["mel_postnet"]
