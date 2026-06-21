"""wav2vec 2.0 — Self-supervised speech representation learning.

Wraps HuggingFace wav2vec2 architecture and provides CTC/attention
decoder for automatic speech recognition.

Reference: "wav2vec 2.0: A Framework for Self-Supervised Learning
of Speech Representations" (Baevski et al., NeurIPS 2020)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashaudio.registry import MODELS


class ConvFeatureExtractor(nn.Module):
    """Multi-layer 1D CNN for raw waveform feature extraction."""

    def __init__(self, out_dim: int = 512):
        super().__init__()
        channels = [1, 512, 512, 512, 512, 512, 512, out_dim]
        kernels = [10, 3, 3, 3, 3, 2, 2]
        strides = [5, 2, 2, 2, 2, 2, 2]

        layers = []
        for i in range(len(kernels)):
            layers.extend([
                nn.Conv1d(channels[i], channels[i + 1], kernels[i], strides[i]),
                nn.GroupNorm(1, channels[i + 1]) if i == 0 else nn.Identity(),
                nn.GELU(),
            ])
        self.layers = nn.Sequential(*layers)
        self.out_dim = out_dim

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (B, T) raw audio waveform.

        Returns:
            (B, C, T') feature map.
        """
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        return self.layers(waveform)


class TransformerEncoder(nn.Module):
    """Transformer encoder for contextualized representations."""

    def __init__(self, dim: int = 768, depth: int = 12, num_heads: int = 12, ff_dim: int = 3072, drop: float = 0.1):
        super().__init__()
        self.pos_conv = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=128, padding=64, groups=16),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, dim_feedforward=ff_dim,
            dropout=drop, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        pos = self.pos_conv(x.transpose(1, 2)).transpose(1, 2)
        x = x + pos[:, :x.shape[1]]
        x = self.norm(x)
        return self.transformer(x, src_key_padding_mask=mask)


class CTCDecoder(nn.Module):
    """CTC decoder head for speech recognition."""

    def __init__(self, in_dim: int = 768, vocab_size: int = 32):
        super().__init__()
        self.proj = nn.Linear(in_dim, vocab_size)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features)

    def decode_greedy(self, logits: torch.Tensor) -> List[List[int]]:
        """Greedy CTC decoding (collapse repeated tokens, remove blanks).

        Args:
            logits: (B, T, vocab_size)

        Returns:
            List of decoded token ID sequences.
        """
        predictions = logits.argmax(dim=-1)
        results = []
        for seq in predictions:
            decoded = []
            prev = -1
            for token in seq.tolist():
                if token != 0 and token != prev:
                    decoded.append(token)
                prev = token
            results.append(decoded)
        return results


class FeatureProjection(nn.Module):
    """Project CNN features to transformer dimension."""

    def __init__(self, in_dim: int = 512, out_dim: int = 768, drop: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, out_dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.drop(self.proj(x))


@MODELS.register("Wav2Vec2")
class Wav2Vec2(nn.Module):
    """wav2vec 2.0 for self-supervised speech recognition.

    Combines a multi-layer CNN feature extractor with a transformer
    encoder and CTC decoder for end-to-end ASR.

    Args:
        vocab_size: Output vocabulary size (includes CTC blank).
        feature_dim: CNN output dimension.
        hidden_dim: Transformer hidden dimension.
        num_layers: Transformer layers.
        num_heads: Attention heads.
        ff_dim: Feed-forward dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        vocab_size: int = 32,
        feature_dim: int = 512,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        ff_dim: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_extractor = ConvFeatureExtractor(feature_dim)
        self.projection = FeatureProjection(feature_dim, hidden_dim, dropout)
        self.encoder = TransformerEncoder(hidden_dim, num_layers, num_heads, ff_dim, dropout)
        self.ctc_head = CTCDecoder(hidden_dim, vocab_size)
        self.vocab_size = vocab_size

    def extract_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract contextualized features from raw audio.

        Args:
            waveform: (B, T) raw audio.

        Returns:
            (B, T', hidden_dim) contextualized features.
        """
        conv_features = self.feature_extractor(waveform)
        conv_features = conv_features.transpose(1, 2)
        projected = self.projection(conv_features)
        return self.encoder(projected)

    def forward(
        self,
        waveform: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        input_lengths: Optional[torch.Tensor] = None,
        label_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with optional CTC loss.

        Args:
            waveform: (B, T) raw audio.
            labels: (B, L) target token IDs for CTC loss.
            input_lengths: (B,) lengths of input sequences.
            label_lengths: (B,) lengths of label sequences.

        Returns:
            Dict with 'logits', optionally 'loss'.
        """
        features = self.extract_features(waveform)
        logits = self.ctc_head(features)

        output = {"logits": logits, "features": features}

        if labels is not None:
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
            T = log_probs.shape[0]
            B = log_probs.shape[1]

            if input_lengths is None:
                input_lengths = torch.full((B,), T, dtype=torch.long, device=waveform.device)
            if label_lengths is None:
                label_lengths = (labels != 0).sum(dim=-1)

            loss = F.ctc_loss(log_probs, labels, input_lengths, label_lengths, blank=0, zero_infinity=True)
            output["loss"] = loss

        return output

    @torch.no_grad()
    def transcribe(self, waveform: torch.Tensor) -> List[List[int]]:
        """Transcribe audio to token sequences.

        Args:
            waveform: (B, T) raw audio.

        Returns:
            List of decoded token ID sequences.
        """
        self.eval()
        output = self.forward(waveform)
        return self.ctc_head.decode_greedy(output["logits"])
