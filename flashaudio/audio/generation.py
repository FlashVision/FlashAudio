"""Audio generation pipeline.

Diffusion-based music and sound effect generation from text prompts
or unconditional sampling.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashaudio.registry import PIPELINES


class DiffusionUNet(nn.Module):
    """Simple U-Net for audio diffusion.

    Operates on mel spectrograms, predicting the noise to subtract
    at each denoising step.

    Args:
        n_mels: Number of mel frequency bins.
        channels: Base channel width.
        time_embed_dim: Timestep embedding dimension.
    """

    def __init__(self, n_mels: int = 80, channels: int = 128, time_embed_dim: int = 128):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, channels * 2),
            nn.SiLU(),
            nn.Conv2d(channels * 2, channels * 2, 3, padding=1),
            nn.GroupNorm(8, channels * 2),
            nn.SiLU(),
        )

        self.mid = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, 3, padding=1),
            nn.GroupNorm(8, channels * 2),
            nn.SiLU(),
        )

        self.time_proj = nn.Linear(time_embed_dim, channels * 2)

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(channels * 2, channels, 4, stride=2, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, 1, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict noise.

        Args:
            x: Noisy mel spectrogram [batch, 1, n_mels, time].
            t: Timestep [batch, 1] normalized to [0, 1].

        Returns:
            Predicted noise [batch, 1, n_mels, time].
        """
        t_emb = self.time_embed(t)

        d1 = self.down1(x)
        d2 = self.down2(d1)

        mid = self.mid(d2)
        t_proj = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        mid = mid + t_proj

        u1 = self.up1(mid)

        if u1.shape != d1.shape:
            u1 = F.interpolate(u1, size=d1.shape[2:], mode="bilinear", align_corners=False)

        u1 = torch.cat([u1, d1], dim=1)
        return self.up2(u1)


@PIPELINES.register("AudioGenerator")
class AudioGenerator:
    """Audio generation via diffusion on mel spectrograms.

    Generates music or sound effects by iteratively denoising
    a random mel spectrogram, then converting to audio with Griffin-Lim.

    Args:
        model: Optional pre-loaded diffusion model.
        device: Device for inference.
        sample_rate: Output audio sample rate.
        n_mels: Number of mel bins.
        n_fft: FFT window size.
        hop_length: Hop length for spectrogram.
        num_steps: Number of diffusion steps.
    """

    def __init__(
        self,
        model=None,
        device: str = "cuda",
        sample_rate: int = 22050,
        n_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 256,
        num_steps: int = 50,
    ):
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_steps = num_steps
        self._model = model

    @property
    def model(self):
        if self._model is None:
            self._model = DiffusionUNet(n_mels=self.n_mels).to(self.device)
            self._model.eval()
        return self._model

    def generate(
        self,
        duration: float = 5.0,
        output_path: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> Union[str, torch.Tensor]:
        """Generate audio.

        Args:
            duration: Duration of generated audio in seconds.
            output_path: If provided, save to this path and return it.
            seed: Random seed for reproducibility.

        Returns:
            Path to saved file (if output_path) or waveform tensor.
        """
        if seed is not None:
            torch.manual_seed(seed)

        num_frames = int(duration * self.sample_rate / self.hop_length)

        x = torch.randn(1, 1, self.n_mels, num_frames, device=self.device)

        betas = torch.linspace(1e-4, 0.02, self.num_steps, device=self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        for step in reversed(range(self.num_steps)):
            t = torch.tensor([[step / self.num_steps]], device=self.device)

            with torch.no_grad():
                predicted_noise = self.model(x, t)

            alpha_t = alphas_cumprod[step]
            alphas_cumprod[step - 1] if step > 0 else torch.tensor(1.0)

            x = (1 / torch.sqrt(alphas[step])) * (
                x - (betas[step] / torch.sqrt(1 - alpha_t)) * predicted_noise
            )

            if step > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(betas[step])
                x = x + sigma * noise

        mel = x.squeeze(0).squeeze(0)
        waveform = self._mel_to_audio(mel)

        if output_path:
            from flashaudio.data.audio_utils import save_audio
            save_audio(output_path, waveform, self.sample_rate)
            return output_path

        return waveform

    def _mel_to_audio(self, mel: torch.Tensor) -> torch.Tensor:
        """Convert mel spectrogram to audio using Griffin-Lim."""
        import torchaudio

        mel = mel.cpu()
        mel_linear = torch.pow(10.0, mel / 20.0).clamp(min=1e-5)

        inverse_mel = torchaudio.transforms.InverseMelScale(
            n_stft=self.n_fft // 2 + 1,
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
        )
        spectrogram = inverse_mel(mel_linear)

        griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_iter=32,
            power=1.0,
        )
        waveform = griffin_lim(spectrogram)

        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak * 0.95

        return waveform.unsqueeze(0)
