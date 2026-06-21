"""Conformer-CTC — Conformer encoder with CTC decoder for ASR.

Implements the Conformer architecture combining convolution and
self-attention for speech recognition with CTC decoding.

Reference: "Conformer: Convolution-augmented Transformer for Speech
Recognition" (Gulati et al., Interspeech 2020)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashaudio.registry import MODELS


class RelativePositionalEncoding(nn.Module):
    """Relative positional encoding for Conformer attention."""

    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.shape[1]]


class FeedForwardModule(nn.Module):
    """Feed-forward module with layer norm and residual scaling."""

    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.5 * self.net(x)


class ConvolutionModule(nn.Module):
    """Convolution module for local pattern extraction."""

    def __init__(self, dim: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.pointwise1 = nn.Conv1d(dim, 2 * dim, 1)
        self.glu = nn.GLU(dim=1)
        self.depthwise = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.batch_norm = nn.BatchNorm1d(dim)
        self.pointwise2 = nn.Conv1d(dim, dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.glu(self.pointwise1(x))
        x = self.act(self.batch_norm(self.depthwise(x)))
        x = self.dropout(self.pointwise2(x))
        return residual + x.transpose(1, 2)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with relative positional encoding."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.attn(x, x, x, key_padding_mask=mask, need_weights=False)[0]
        return residual + self.dropout(x)


class ConformerBlock(nn.Module):
    """Single Conformer block: FFN -> MHSA -> Conv -> FFN."""

    def __init__(self, dim: int, num_heads: int = 4, conv_kernel: int = 31,
                 ff_expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ff1 = FeedForwardModule(dim, ff_expansion, dropout)
        self.mhsa = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.conv = ConvolutionModule(dim, conv_kernel, dropout)
        self.ff2 = FeedForwardModule(dim, ff_expansion, dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.ff1(x)
        x = self.mhsa(x, mask)
        x = self.conv(x)
        x = self.ff2(x)
        return self.norm(x)


class ConformerEncoder(nn.Module):
    """Stack of Conformer blocks."""

    def __init__(self, dim: int = 256, depth: int = 12, num_heads: int = 4,
                 conv_kernel: int = 31, dropout: float = 0.1):
        super().__init__()
        self.pos_enc = RelativePositionalEncoding(dim)
        self.blocks = nn.ModuleList([
            ConformerBlock(dim, num_heads, conv_kernel, dropout=dropout) for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.pos_enc(x)
        for block in self.blocks:
            x = block(x, mask)
        return x


class AudioFrontend(nn.Module):
    """Frontend for converting mel/fbank features to conformer dimension."""

    def __init__(self, in_dim: int = 80, out_dim: int = 256, subsampling: int = 4):
        super().__init__()
        self.subsampling = subsampling
        if subsampling == 4:
            self.conv = nn.Sequential(
                nn.Conv2d(1, out_dim, 3, 2, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_dim, out_dim, 3, 2, 1),
                nn.ReLU(inplace=True),
            )
            self.linear = nn.Linear(out_dim * (in_dim // 4), out_dim)
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(1, out_dim, 3, 2, 1),
                nn.ReLU(inplace=True),
            )
            self.linear = nn.Linear(out_dim * (in_dim // 2), out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_mels, T) mel spectrogram.

        Returns:
            (B, T', dim) subsampled features.
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv(x)
        B, C, F, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * F)
        return self.linear(x)


@MODELS.register("ConformerCTC")
class ConformerCTC(nn.Module):
    """Conformer with CTC decoder for automatic speech recognition.

    Combines convolutional subsampling frontend, Conformer encoder,
    and CTC output head.

    Args:
        vocab_size: Output vocabulary size (includes CTC blank at index 0).
        input_dim: Input feature dimension (e.g., 80 for mel).
        encoder_dim: Conformer hidden dimension.
        num_layers: Number of Conformer blocks.
        num_heads: Attention heads.
        conv_kernel: Convolution kernel size in Conformer.
        dropout: Dropout rate.
        subsampling: Frontend subsampling factor (2 or 4).
    """

    def __init__(
        self,
        vocab_size: int = 32,
        input_dim: int = 80,
        encoder_dim: int = 256,
        num_layers: int = 12,
        num_heads: int = 4,
        conv_kernel: int = 31,
        dropout: float = 0.1,
        subsampling: int = 4,
    ):
        super().__init__()
        self.frontend = AudioFrontend(input_dim, encoder_dim, subsampling)
        self.encoder = ConformerEncoder(encoder_dim, num_layers, num_heads, conv_kernel, dropout)
        self.ctc_head = nn.Linear(encoder_dim, vocab_size)
        self.vocab_size = vocab_size

    def forward(
        self,
        features: torch.Tensor,
        feature_lengths: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        label_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            features: (B, n_mels, T) or (B, T, n_mels) input features.
            feature_lengths: (B,) feature lengths before subsampling.
            labels: (B, L) target labels for CTC loss.
            label_lengths: (B,) label lengths.

        Returns:
            Dict with 'logits' and optional 'loss'.
        """
        x = self.frontend(features)
        mask = None
        if feature_lengths is not None:
            sub_lengths = feature_lengths // self.frontend.subsampling
            max_len = x.shape[1]
            mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= sub_lengths.unsqueeze(1)

        encoded = self.encoder(x, mask)
        logits = self.ctc_head(encoded)

        output = {"logits": logits, "encoded": encoded}

        if labels is not None:
            log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)
            T = log_probs.shape[0]
            B = log_probs.shape[1]
            if feature_lengths is None:
                input_lengths = torch.full((B,), T, dtype=torch.long, device=features.device)
            else:
                input_lengths = feature_lengths // self.frontend.subsampling
                input_lengths = input_lengths.clamp(max=T)
            if label_lengths is None:
                label_lengths = (labels != 0).sum(dim=-1)
            loss = F.ctc_loss(log_probs, labels, input_lengths, label_lengths, blank=0, zero_infinity=True)
            output["loss"] = loss

        return output

    @torch.no_grad()
    def decode_greedy(self, logits: torch.Tensor) -> List[List[int]]:
        """Greedy CTC decoding.

        Args:
            logits: (B, T, vocab_size) model output.

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

    @torch.no_grad()
    def recognize(self, features: torch.Tensor) -> List[List[int]]:
        """End-to-end speech recognition.

        Args:
            features: (B, n_mels, T) input features.

        Returns:
            Decoded token sequences.
        """
        self.eval()
        output = self.forward(features)
        return self.decode_greedy(output["logits"])
