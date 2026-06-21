"""VITS — Variational Inference with adversarial learning for TTS.

Implements the VITS architecture combining:
- Posterior encoder (from waveform/mel to latent z)
- Prior network (text encoder with normalizing flow)
- HiFi-GAN decoder for waveform generation
- Monotonic Alignment Search (MAS) for duration modeling

Reference: "Conditional Variational Autoencoder with Adversarial Learning
for End-to-End Text-to-Speech" (Kim et al., ICML 2021)
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashaudio.registry import MODELS


class TextEncoder(nn.Module):
    """Text encoder with transformer layers and projection to prior statistics."""

    def __init__(self, vocab_size: int = 256, hidden_dim: int = 192, num_layers: int = 6, num_heads: int = 2, filter_size: int = 768):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(1024, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=filter_size,
            dropout=0.1, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.proj_mu = nn.Linear(hidden_dim, hidden_dim)
        self.proj_log_sigma = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, text_ids: torch.Tensor, text_lengths: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B, L = text_ids.shape
        positions = torch.arange(L, device=text_ids.device).unsqueeze(0).expand(B, -1)

        x = self.embedding(text_ids) + self.pos_embedding(positions)

        mask = None
        if text_lengths is not None:
            mask = torch.arange(L, device=text_ids.device).unsqueeze(0) >= text_lengths.unsqueeze(1)

        x = self.encoder(x, src_key_padding_mask=mask)

        mu = self.proj_mu(x)
        log_sigma = self.proj_log_sigma(x)

        return {"mu": mu, "log_sigma": log_sigma, "hidden": x}


class PosteriorEncoder(nn.Module):
    """Posterior encoder: waveform/mel -> latent z."""

    def __init__(self, in_channels: int = 80, hidden_dim: int = 192, kernel_size: int = 5, num_layers: int = 16):
        super().__init__()
        self.pre = nn.Conv1d(in_channels, hidden_dim, 1)

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** (i % 4)
            padding = (kernel_size * dilation - dilation) // 2
            self.convs.append(nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, dilation=dilation, padding=padding),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
            ))

        self.proj_mu = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.proj_log_sigma = nn.Conv1d(hidden_dim, hidden_dim, 1)

    def forward(self, mel: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.pre(mel)
        for conv in self.convs:
            x = x + conv(x)

        mu = self.proj_mu(x)
        log_sigma = self.proj_log_sigma(x)

        z = mu + torch.randn_like(mu) * torch.exp(log_sigma) if self.training else mu
        return {"z": z, "mu": mu, "log_sigma": log_sigma}


class ResidualCouplingLayer(nn.Module):
    """Affine coupling layer for normalizing flow."""

    def __init__(self, channels: int, hidden_dim: int = 192, kernel_size: int = 5, num_layers: int = 4):
        super().__init__()
        self.half_channels = channels // 2

        self.pre = nn.Conv1d(self.half_channels, hidden_dim, 1)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2),
                nn.GELU(),
            )
            for _ in range(num_layers)
        ])
        self.post = nn.Conv1d(hidden_dim, self.half_channels, 1)
        nn.init.zeros_(self.post.weight)
        nn.init.zeros_(self.post.bias)

    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        x0, x1 = x.split(self.half_channels, dim=1)
        h = self.pre(x0)
        for conv in self.convs:
            h = h + conv(h)
        shift = self.post(h)

        if not reverse:
            x1 = x1 + shift
            log_det = torch.zeros(x.shape[0], device=x.device)
        else:
            x1 = x1 - shift
            log_det = torch.zeros(x.shape[0], device=x.device)

        return torch.cat([x0, x1], dim=1), log_det


class FlowModule(nn.Module):
    """Normalizing flow with residual coupling layers."""

    def __init__(self, channels: int, num_flows: int = 4, hidden_dim: int = 192):
        super().__init__()
        self.flows = nn.ModuleList([
            ResidualCouplingLayer(channels, hidden_dim) for _ in range(num_flows)
        ])

    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        total_log_det = torch.zeros(x.shape[0], device=x.device)
        flows = reversed(self.flows) if reverse else self.flows
        for flow in flows:
            x, log_det = flow(x, reverse=reverse)
            total_log_det = total_log_det + log_det
            x = x.flip(1)
        return x, total_log_det


class DurationPredictor(nn.Module):
    """Stochastic duration predictor."""

    def __init__(self, in_channels: int = 192, hidden_dim: int = 256, kernel_size: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) input features.

        Returns:
            (B, T) log durations.
        """
        return self.net(x).squeeze(1)


class HiFiGANDecoder(nn.Module):
    """Lightweight HiFi-GAN decoder for VITS."""

    def __init__(self, in_channels: int = 192, upsample_rates: Tuple[int, ...] = (8, 8, 2, 2)):
        super().__init__()
        ch = 512
        self.pre = nn.Conv1d(in_channels, ch, 7, 1, 3)
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()
        for rate in upsample_rates:
            k = rate * 2
            self.ups.append(nn.ConvTranspose1d(ch, ch // 2, k, rate, padding=rate // 2))
            ch //= 2
            self.resblocks.append(nn.Sequential(
                nn.LeakyReLU(0.1),
                nn.Conv1d(ch, ch, 3, 1, 1),
                nn.LeakyReLU(0.1),
                nn.Conv1d(ch, ch, 3, 1, 1),
            ))
        self.post = nn.Conv1d(ch, 1, 7, 1, 3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.pre(z)
        for up, res in zip(self.ups, self.resblocks):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            x = x + res(x)
        x = F.leaky_relu(x, 0.1)
        return torch.tanh(self.post(x))


def monotonic_alignment_search(log_p: torch.Tensor, text_len: int, mel_len: int) -> torch.Tensor:
    """Monotonic Alignment Search (MAS) — Viterbi for duration alignment.

    Args:
        log_p: (text_len, mel_len) log probability matrix.
        text_len: Length of text sequence.
        mel_len: Length of mel sequence.

    Returns:
        (text_len,) duration for each text token.
    """
    Q = torch.full((text_len, mel_len), -float("inf"), device=log_p.device)
    Q[0, 0] = log_p[0, 0]
    for j in range(1, mel_len):
        for i in range(min(j + 1, text_len)):
            if i == 0:
                Q[i, j] = Q[i, j - 1] + log_p[i, j]
            else:
                Q[i, j] = torch.max(Q[i, j - 1], Q[i - 1, j - 1]) + log_p[i, j]

    durations = torch.zeros(text_len, dtype=torch.long, device=log_p.device)
    i, j = text_len - 1, mel_len - 1
    while i >= 0 and j >= 0:
        durations[i] += 1
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        elif Q[i, j - 1] >= Q[i - 1, j - 1]:
            j -= 1
        else:
            i -= 1
            j -= 1
    return durations


@MODELS.register("VITS")
class VITS(nn.Module):
    """VITS: Variational Inference Text-to-Speech.

    End-to-end TTS model combining VAE, normalizing flow, and
    adversarial training for high-quality speech synthesis.

    Args:
        vocab_size: Text vocabulary size.
        hidden_dim: Model hidden dimension.
        num_layers: Text encoder layers.
        n_mels: Mel spectrogram channels (for posterior encoder).
        num_flows: Normalizing flow layers.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        hidden_dim: int = 192,
        num_layers: int = 6,
        n_mels: int = 80,
        num_flows: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.text_encoder = TextEncoder(vocab_size, hidden_dim, num_layers)
        self.posterior_encoder = PosteriorEncoder(n_mels, hidden_dim)
        self.flow = FlowModule(hidden_dim, num_flows)
        self.duration_predictor = DurationPredictor(hidden_dim)
        self.decoder = HiFiGANDecoder(hidden_dim)

    def forward(
        self,
        text_ids: torch.Tensor,
        text_lengths: Optional[torch.Tensor] = None,
        mel: Optional[torch.Tensor] = None,
        mel_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass.

        Args:
            text_ids: (B, L) text token IDs.
            text_lengths: (B,) text lengths.
            mel: (B, n_mels, T) mel spectrogram (for training).
            mel_lengths: (B,) mel lengths.

        Returns:
            Dict with generated audio and loss components.
        """
        text_out = self.text_encoder(text_ids, text_lengths)
        prior_mu = text_out["mu"]
        prior_log_sigma = text_out["log_sigma"]

        if mel is not None:
            post_out = self.posterior_encoder(mel)
            z = post_out["z"]

            z_flow, flow_log_det = self.flow(z, reverse=False)

            log_dur = self.duration_predictor(text_out["hidden"].transpose(1, 2))
            audio = self.decoder(z)

            kl_loss = self._kl_divergence(
                post_out["mu"], post_out["log_sigma"],
                prior_mu.transpose(1, 2), prior_log_sigma.transpose(1, 2),
            )

            return {
                "audio": audio,
                "z": z,
                "prior_mu": prior_mu,
                "prior_log_sigma": prior_log_sigma,
                "posterior_mu": post_out["mu"],
                "posterior_log_sigma": post_out["log_sigma"],
                "log_duration": log_dur,
                "kl_loss": kl_loss,
            }
        else:
            return self.infer(text_ids, text_lengths)

    @torch.no_grad()
    def infer(
        self,
        text_ids: torch.Tensor,
        text_lengths: Optional[torch.Tensor] = None,
        noise_scale: float = 0.667,
        length_scale: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Inference: generate audio from text.

        Args:
            text_ids: (B, L) text token IDs.
            text_lengths: (B,) text lengths.
            noise_scale: Prior noise scale.
            length_scale: Duration scaling factor.

        Returns:
            Dict with 'audio' waveform.
        """
        text_out = self.text_encoder(text_ids, text_lengths)
        mu = text_out["mu"]
        log_sigma = text_out["log_sigma"]

        log_dur = self.duration_predictor(text_out["hidden"].transpose(1, 2))
        durations = torch.clamp(torch.round(torch.exp(log_dur) * length_scale), min=1).long()

        total_len = durations.sum(dim=-1).max().item()
        B, L, D = mu.shape

        z_prior = torch.zeros(B, D, int(total_len), device=mu.device)
        for b in range(B):
            pos = 0
            for i in range(min(L, durations.shape[-1])):
                dur = durations[b, i].item()
                z_prior[b, :, pos:pos+dur] = mu[b, i].unsqueeze(-1).expand(-1, dur)
                pos += dur

        z_prior = z_prior + torch.randn_like(z_prior) * noise_scale

        z, _ = self.flow(z_prior, reverse=True)
        audio = self.decoder(z)

        return {"audio": audio}

    @staticmethod
    def _kl_divergence(mu_q: torch.Tensor, log_sigma_q: torch.Tensor,
                       mu_p: torch.Tensor, log_sigma_p: torch.Tensor) -> torch.Tensor:
        min_len = min(mu_q.shape[-1], mu_p.shape[-1])
        mu_q, log_sigma_q = mu_q[..., :min_len], log_sigma_q[..., :min_len]
        mu_p, log_sigma_p = mu_p[..., :min_len], log_sigma_p[..., :min_len]

        kl = log_sigma_p - log_sigma_q + (torch.exp(2 * log_sigma_q) + (mu_q - mu_p) ** 2) / (2 * torch.exp(2 * log_sigma_p)) - 0.5
        return kl.mean()

    def text_to_ids(self, text: str) -> torch.Tensor:
        ids = [min(ord(c), 255) for c in text]
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)
