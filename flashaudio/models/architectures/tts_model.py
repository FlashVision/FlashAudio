"""Text-to-Speech model architecture.

Implements a Tacotron2-style encoder-decoder that converts text to mel
spectrograms, plus a lightweight vocoder (Griffin-Lim based) for waveform
reconstruction.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashaudio.registry import MODELS


class TextEncoder(nn.Module):
    """Character-level text encoder using convolutional layers and LSTM.

    Converts character embeddings into a hidden representation suitable
    for the attention-based decoder.

    Args:
        vocab_size: Size of the character vocabulary.
        embed_dim: Embedding dimension.
        encoder_dim: Hidden dimension of the encoder.
        num_conv_layers: Number of 1D convolution layers.
        kernel_size: Convolution kernel size.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        embed_dim: int = 256,
        encoder_dim: int = 256,
        num_conv_layers: int = 3,
        kernel_size: int = 5,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        conv_layers = []
        for i in range(num_conv_layers):
            in_ch = embed_dim if i == 0 else encoder_dim
            conv_layers.extend([
                nn.Conv1d(in_ch, encoder_dim, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(encoder_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
        self.convolutions = nn.Sequential(*conv_layers)

        self.lstm = nn.LSTM(encoder_dim, encoder_dim // 2, batch_first=True, bidirectional=True)

    def forward(self, text_ids: torch.Tensor) -> torch.Tensor:
        """Encode text to hidden representation.

        Args:
            text_ids: [batch, seq_len] integer character IDs.

        Returns:
            [batch, seq_len, encoder_dim] encoder output.
        """
        x = self.embedding(text_ids)
        x = x.transpose(1, 2)
        x = self.convolutions(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        return x


class MelDecoder(nn.Module):
    """Autoregressive mel spectrogram decoder.

    Attends to the encoder output and produces mel spectrogram frames
    one step at a time (with teacher forcing during training).

    Args:
        encoder_dim: Dimension of encoder output.
        n_mels: Number of mel frequency bins.
        decoder_dim: Hidden dimension of the decoder LSTM.
        attention_dim: Dimension of the attention mechanism.
        prenet_dim: Dimension of the prenet layers.
    """

    def __init__(
        self,
        encoder_dim: int = 256,
        n_mels: int = 80,
        decoder_dim: int = 512,
        attention_dim: int = 128,
        prenet_dim: int = 128,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.decoder_dim = decoder_dim

        self.prenet = nn.Sequential(
            nn.Linear(n_mels, prenet_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(prenet_dim, prenet_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

        self.attention_rnn = nn.LSTMCell(prenet_dim + encoder_dim, decoder_dim)

        self.attention_query = nn.Linear(decoder_dim, attention_dim)
        self.attention_key = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.attention_v = nn.Linear(attention_dim, 1, bias=False)

        self.decoder_rnn = nn.LSTMCell(decoder_dim + encoder_dim, decoder_dim)

        self.mel_proj = nn.Linear(decoder_dim + encoder_dim, n_mels)
        self.stop_proj = nn.Linear(decoder_dim + encoder_dim, 1)

    def _attention(
        self, query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Location-sensitive attention."""
        query_proj = self.attention_query(query).unsqueeze(1)
        key_proj = self.attention_key(keys)

        energy = self.attention_v(torch.tanh(query_proj + key_proj)).squeeze(-1)
        weights = F.softmax(energy, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), values).squeeze(1)
        return context, weights

    def forward(
        self,
        encoder_output: torch.Tensor,
        max_steps: int = 200,
        target_mel: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Decode mel spectrogram from encoder output.

        Args:
            encoder_output: [batch, seq_len, encoder_dim].
            max_steps: Maximum decoding steps.
            target_mel: Optional target mel for teacher forcing [batch, n_mels, time].

        Returns:
            Dictionary with 'mel' [batch, n_mels, time] and 'stop_logits'.
        """
        batch_size = encoder_output.shape[0]
        device = encoder_output.device

        if target_mel is not None:
            max_steps = target_mel.shape[2]

        go_frame = torch.zeros(batch_size, self.n_mels, device=device)
        h_att = torch.zeros(batch_size, self.decoder_dim, device=device)
        c_att = torch.zeros(batch_size, self.decoder_dim, device=device)
        h_dec = torch.zeros(batch_size, self.decoder_dim, device=device)
        c_dec = torch.zeros(batch_size, self.decoder_dim, device=device)
        context = torch.zeros(batch_size, encoder_output.shape[2], device=device)

        mel_outputs = []
        stop_outputs = []
        prev_frame = go_frame

        for t in range(max_steps):
            prenet_out = self.prenet(prev_frame)
            att_input = torch.cat([prenet_out, context], dim=-1)
            h_att, c_att = self.attention_rnn(att_input, (h_att, c_att))

            context, _ = self._attention(h_att, encoder_output, encoder_output)

            dec_input = torch.cat([h_att, context], dim=-1)
            h_dec, c_dec = self.decoder_rnn(dec_input, (h_dec, c_dec))

            decoder_context = torch.cat([h_dec, context], dim=-1)
            mel_frame = self.mel_proj(decoder_context)
            stop_logit = self.stop_proj(decoder_context)

            mel_outputs.append(mel_frame)
            stop_outputs.append(stop_logit)

            if target_mel is not None and t < max_steps - 1:
                prev_frame = target_mel[:, :, t]
            else:
                prev_frame = mel_frame

        mel = torch.stack(mel_outputs, dim=2)
        stop_logits = torch.cat(stop_outputs, dim=-1)

        return {"mel": mel, "stop_logits": stop_logits}


class PostNet(nn.Module):
    """Post-processing network to refine mel spectrograms.

    Five convolutional layers that learn a residual correction
    to the decoder's mel output.

    Args:
        n_mels: Number of mel bins.
        postnet_dim: Hidden dimension.
        num_layers: Number of convolution layers.
        kernel_size: Convolution kernel size.
    """

    def __init__(
        self,
        n_mels: int = 80,
        postnet_dim: int = 256,
        num_layers: int = 5,
        kernel_size: int = 5,
    ):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_ch = n_mels if i == 0 else postnet_dim
            out_ch = n_mels if i == num_layers - 1 else postnet_dim
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(out_ch),
            ])
            if i < num_layers - 1:
                layers.append(nn.Tanh())
                layers.append(nn.Dropout(0.1))

        self.network = nn.Sequential(*layers)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """Apply post-net residual correction.

        Args:
            mel: [batch, n_mels, time].

        Returns:
            Refined mel spectrogram.
        """
        return mel + self.network(mel)


@MODELS.register("TTSModel")
class TTSModel(nn.Module):
    """Complete TTS model: TextEncoder -> MelDecoder -> PostNet.

    Converts character ID sequences into mel spectrograms.

    Args:
        vocab_size: Character vocabulary size.
        encoder_dim: Encoder hidden dimension.
        n_mels: Number of mel bins.
        decoder_dim: Decoder hidden dimension.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        encoder_dim: int = 256,
        n_mels: int = 80,
        decoder_dim: int = 512,
    ):
        super().__init__()
        self.encoder = TextEncoder(vocab_size=vocab_size, encoder_dim=encoder_dim)
        self.decoder = MelDecoder(encoder_dim=encoder_dim, n_mels=n_mels, decoder_dim=decoder_dim)
        self.postnet = PostNet(n_mels=n_mels)
        self.n_mels = n_mels

    def forward(
        self,
        text_ids: torch.Tensor,
        target_mel: Optional[torch.Tensor] = None,
        max_steps: int = 200,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass: text -> mel spectrogram.

        Args:
            text_ids: [batch, seq_len] character IDs.
            target_mel: Optional target mel for teacher forcing.
            max_steps: Maximum decoding steps.

        Returns:
            Dictionary with 'mel', 'mel_postnet', 'stop_logits'.
        """
        encoder_output = self.encoder(text_ids)
        decoder_output = self.decoder(encoder_output, max_steps=max_steps, target_mel=target_mel)

        mel = decoder_output["mel"]
        mel_postnet = self.postnet(mel)

        return {
            "mel": mel,
            "mel_postnet": mel_postnet,
            "stop_logits": decoder_output["stop_logits"],
        }

    def text_to_ids(self, text: str) -> torch.Tensor:
        """Convert text string to character IDs.

        Uses a simple byte-level encoding (ord of each character).
        """
        ids = [min(ord(c), 255) for c in text]
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)
