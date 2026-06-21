"""Sound Event Detection — Audio tagging and temporal event detection.

Implements sound event detection using AudioSet-pretrained features
with frame-level predictions for temporal localization.

Reference: "PANNs: Large-Scale Pretrained Audio Neural Networks for
Audio Pattern Recognition" (Kong et al., IEEE TASLP 2020)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashaudio.registry import MODELS


class ConvBlock2D(nn.Module):
    """Double convolution block for spectrogram processing."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        return x


class CNN14Backbone(nn.Module):
    """CNN14-style backbone (from PANNs) for audio feature extraction.

    Processes mel spectrograms through convolutional blocks with
    pooling to produce frame-level and clip-level features.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 64):
        super().__init__()
        self.conv_block1 = ConvBlock2D(in_channels, base_channels)
        self.conv_block2 = ConvBlock2D(base_channels, base_channels * 2)
        self.conv_block3 = ConvBlock2D(base_channels * 2, base_channels * 4)
        self.conv_block4 = ConvBlock2D(base_channels * 4, base_channels * 8)
        self.conv_block5 = ConvBlock2D(base_channels * 8, base_channels * 16)
        self.conv_block6 = ConvBlock2D(base_channels * 16, base_channels * 32)

        self.pool = nn.AvgPool2d(2)
        self.out_channels = base_channels * 32

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 1, n_mels, T) mel spectrogram.

        Returns:
            frame_features: (B, C, T') frame-level features.
            clip_features: (B, C) clip-level features.
        """
        x = self.pool(self.conv_block1(x))
        x = self.pool(self.conv_block2(x))
        x = self.pool(self.conv_block3(x))
        x = self.pool(self.conv_block4(x))
        x = self.pool(self.conv_block5(x))
        x = self.conv_block6(x)

        frame_features = x.mean(dim=2)
        clip_features = frame_features.mean(dim=-1)
        return frame_features, clip_features


class AttentionPooling(nn.Module):
    """Attention pooling for multi-instance learning in audio tagging."""

    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(dim // 4, num_classes),
        )
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, frame_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            frame_features: (B, T, D)

        Returns:
            clip_logits: (B, num_classes) aggregated clip predictions.
            frame_logits: (B, T, num_classes) per-frame predictions.
        """
        attn_weights = torch.sigmoid(self.attention(frame_features))
        frame_logits = self.classifier(frame_features)

        clip_logits = (attn_weights * frame_logits).sum(dim=1) / (attn_weights.sum(dim=1) + 1e-8)
        return clip_logits, frame_logits


class TemporalConvModule(nn.Module):
    """Temporal convolution for frame-level event boundary refinement."""

    def __init__(self, dim: int, num_classes: int, kernel_size: int = 9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, num_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, D, T) frame features.

        Returns:
            (B, num_classes, T) frame-level logits.
        """
        return self.net(x)


@MODELS.register("SoundEventDetector")
class SoundEventDetector(nn.Module):
    """Sound Event Detection model with audio tagging and temporal detection.

    Uses a CNN14-style backbone with attention pooling for clip-level
    tagging and temporal convolutions for frame-level event detection.

    Args:
        num_classes: Number of event categories (527 for full AudioSet).
        n_mels: Input mel spectrogram bins.
        base_channels: CNN backbone base channel width.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        num_classes: int = 527,
        n_mels: int = 64,
        base_channels: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.n_mels = n_mels

        self.backbone = CNN14Backbone(in_channels=1, base_channels=base_channels)
        feat_dim = self.backbone.out_channels

        self.fc = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.attention_pool = AttentionPooling(512, num_classes)
        self.temporal_det = TemporalConvModule(512, num_classes)

    def forward(self, mel: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for audio tagging and event detection.

        Args:
            mel: (B, n_mels, T) mel spectrogram.

        Returns:
            Dict with:
                'clip_logits': (B, num_classes) clip-level predictions.
                'frame_logits': (B, num_classes, T') frame-level predictions.
                'clip_probs': (B, num_classes) clip probabilities.
        """
        if mel.dim() == 3:
            mel = mel.unsqueeze(1)

        frame_feat, clip_feat = self.backbone(mel)
        frame_feat_t = frame_feat.transpose(1, 2)
        frame_feat_t = self.fc(frame_feat_t)

        clip_logits, frame_attn_logits = self.attention_pool(frame_feat_t)

        frame_det_logits = self.temporal_det(frame_feat_t.transpose(1, 2))

        return {
            "clip_logits": clip_logits,
            "frame_logits": frame_det_logits,
            "frame_attn_logits": frame_attn_logits,
            "clip_probs": torch.sigmoid(clip_logits),
        }

    @torch.no_grad()
    def detect_events(
        self,
        mel: torch.Tensor,
        threshold: float = 0.5,
        min_duration_frames: int = 3,
    ) -> List[List[Dict]]:
        """Detect sound events with temporal boundaries.

        Args:
            mel: (B, n_mels, T) mel spectrogram.
            threshold: Detection probability threshold.
            min_duration_frames: Minimum event duration in frames.

        Returns:
            List of event lists per batch item, each event is a dict
            with 'class_idx', 'start_frame', 'end_frame', 'confidence'.
        """
        self.eval()
        output = self.forward(mel)
        frame_probs = torch.sigmoid(output["frame_logits"])
        B, C, T = frame_probs.shape

        results = []
        for b in range(B):
            events = []
            for c in range(C):
                active = frame_probs[b, c] > threshold
                segments = self._find_segments(active, min_duration_frames)
                for start, end in segments:
                    conf = frame_probs[b, c, start:end].mean().item()
                    events.append({
                        "class_idx": c,
                        "start_frame": start,
                        "end_frame": end,
                        "confidence": conf,
                    })
            results.append(events)
        return results

    @staticmethod
    def _find_segments(active: torch.Tensor, min_duration: int) -> List[Tuple[int, int]]:
        """Find contiguous active segments."""
        segments = []
        in_segment = False
        start = 0
        for i in range(len(active)):
            if active[i] and not in_segment:
                start = i
                in_segment = True
            elif not active[i] and in_segment:
                if i - start >= min_duration:
                    segments.append((start, i))
                in_segment = False
        if in_segment and len(active) - start >= min_duration:
            segments.append((start, len(active)))
        return segments

    @torch.no_grad()
    def tag_audio(self, mel: torch.Tensor, top_k: int = 5) -> Dict[str, torch.Tensor]:
        """Audio tagging (clip-level classification).

        Args:
            mel: (B, n_mels, T) mel spectrogram.
            top_k: Number of top predictions to return.

        Returns:
            Dict with 'indices' and 'probabilities'.
        """
        self.eval()
        output = self.forward(mel)
        probs = output["clip_probs"]
        top_probs, top_indices = probs.topk(top_k, dim=-1)
        return {"indices": top_indices, "probabilities": top_probs}

    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        frame_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute training loss.

        Args:
            predictions: Model output.
            targets: (B, num_classes) multi-label clip targets.
            frame_targets: Optional (B, num_classes, T) frame-level targets.

        Returns:
            Loss dictionary.
        """
        clip_loss = F.binary_cross_entropy_with_logits(predictions["clip_logits"], targets.float())
        losses = {"clip_loss": clip_loss}

        if frame_targets is not None:
            frame_logits = predictions["frame_logits"]
            if frame_logits.shape[-1] != frame_targets.shape[-1]:
                frame_targets = F.interpolate(
                    frame_targets.float().unsqueeze(1),
                    size=(frame_targets.shape[1], frame_logits.shape[-1]),
                    mode="bilinear", align_corners=False,
                ).squeeze(1)
            frame_loss = F.binary_cross_entropy_with_logits(frame_logits, frame_targets.float())
            losses["frame_loss"] = frame_loss

        losses["total"] = sum(losses.values())
        return losses
