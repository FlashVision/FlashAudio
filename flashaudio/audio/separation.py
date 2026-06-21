"""Audio source separation pipeline.

U-Net based model that separates a mixed audio signal into
individual source components (e.g. vocals, drums, bass, other).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashaudio.registry import PIPELINES


class SeparationUNet(nn.Module):
    """U-Net for audio source separation in the spectrogram domain.

    Operates on magnitude spectrograms and produces a mask for each
    source that is applied to the original spectrogram.

    Args:
        num_sources: Number of sources to separate.
        n_fft: FFT window size (determines input frequency bins).
        channels: Base channel width.
    """

    def __init__(self, num_sources: int = 2, n_fft: int = 1024, channels: int = 32):
        super().__init__()
        self.num_sources = num_sources
        n_fft // 2 + 1

        self.encoder = nn.ModuleList([
            self._conv_block(1, channels),
            self._conv_block(channels, channels * 2),
            self._conv_block(channels * 2, channels * 4),
        ])

        self.bottleneck = self._conv_block(channels * 4, channels * 4)

        self.decoder = nn.ModuleList([
            self._upconv_block(channels * 8, channels * 2),
            self._upconv_block(channels * 4, channels),
            self._upconv_block(channels * 2, channels),
        ])

        self.mask_layer = nn.Sequential(
            nn.Conv2d(channels, num_sources, 1),
            nn.Sigmoid(),
        )

    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def _upconv_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Produce separation masks.

        Args:
            x: Magnitude spectrogram [batch, 1, freq, time].

        Returns:
            Masks [batch, num_sources, freq, time].
        """
        skips = []

        h = x
        for enc in self.encoder:
            h = enc(h)
            skips.append(h)
            h = F.max_pool2d(h, 2)

        h = self.bottleneck(h)

        for dec, skip in zip(self.decoder, reversed(skips)):
            h = F.interpolate(h, size=skip.shape[2:], mode="bilinear", align_corners=False)
            h = torch.cat([h, skip], dim=1)
            h = dec(h)

        h = F.interpolate(h, size=x.shape[2:], mode="bilinear", align_corners=False)
        masks = self.mask_layer(h)
        return masks


@PIPELINES.register("SourceSeparator")
class SourceSeparator:
    """Audio source separation pipeline.

    Separates a mixed audio signal into individual sources
    using a U-Net model operating on STFT spectrograms.

    Args:
        model: Optional pre-loaded separation model.
        device: Device for inference.
        num_sources: Number of sources to separate into.
        source_names: Names for each source channel.
        n_fft: FFT window size.
        hop_length: STFT hop length.
    """

    DEFAULT_SOURCES = ["vocals", "accompaniment"]

    def __init__(
        self,
        model=None,
        device: str = "cuda",
        num_sources: int = 2,
        source_names: Optional[List[str]] = None,
        n_fft: int = 1024,
        hop_length: int = 256,
    ):
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.num_sources = num_sources
        self.source_names = source_names or self.DEFAULT_SOURCES[:num_sources]
        self.n_fft = n_fft
        self.hop_length = hop_length
        self._model = model

    @property
    def model(self):
        if self._model is None:
            self._model = SeparationUNet(
                num_sources=self.num_sources, n_fft=self.n_fft
            ).to(self.device)
            self._model.eval()
        return self._model

    def separate(
        self,
        audio: Union[str, torch.Tensor],
        sample_rate: int = 16000,
        output_dir: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """Separate audio into individual sources.

        Args:
            audio: Path to audio file or waveform tensor.
            sample_rate: Sample rate of the input.
            output_dir: If provided, save separated sources to this directory.

        Returns:
            Dictionary mapping source_name -> waveform tensor.
        """
        waveform = self._load_audio(audio, sample_rate)

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            return_complex=True,
            window=torch.hann_window(self.n_fft, device=waveform.device),
        )

        magnitude = stft.abs().unsqueeze(0)

        with torch.no_grad():
            masks = self.model(magnitude.to(self.device))

        masks = masks.cpu()
        sources = {}

        for i, name in enumerate(self.source_names):
            mask = masks[:, i:i+1]
            mask = F.interpolate(mask, size=magnitude.shape[2:], mode="bilinear", align_corners=False)

            source_stft = stft * mask.squeeze(0)

            source_waveform = torch.istft(
                source_stft,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=torch.hann_window(self.n_fft),
            )
            sources[name] = source_waveform

        if output_dir:
            from flashaudio.data.audio_utils import save_audio
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            for name, wav in sources.items():
                save_audio(output_path / f"{name}.wav", wav, sample_rate)

        return sources

    def _load_audio(self, audio: Union[str, torch.Tensor], sample_rate: int) -> torch.Tensor:
        """Load audio from path or validate tensor."""
        if isinstance(audio, (str, Path)):
            from flashaudio.data.audio_utils import load_audio
            waveform, sr = load_audio(str(audio), sample_rate=sample_rate)
            return waveform.squeeze(0)
        elif isinstance(audio, torch.Tensor):
            if audio.dim() == 2:
                audio = audio.squeeze(0)
            return audio
        else:
            raise TypeError(f"Expected str or torch.Tensor, got {type(audio)}")
