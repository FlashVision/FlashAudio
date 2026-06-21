"""Speaker diarization pipeline.

Segments audio by speaker identity using speaker embeddings
and agglomerative clustering.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from flashaudio.registry import PIPELINES


@PIPELINES.register("SpeakerDiarizer")
class SpeakerDiarizer:
    """Speaker diarization: who spoke when.

    Splits audio into segments, extracts speaker embeddings,
    and clusters them to assign speaker labels.

    Args:
        encoder_model: Optional pre-loaded voice encoder.
        device: Device for inference.
        num_speakers: Expected number of speakers (None = auto-detect).
        segment_duration: Duration of each analysis segment in seconds.
        hop_duration: Hop between segments in seconds.
        similarity_threshold: Cosine similarity threshold for clustering.
    """

    def __init__(
        self,
        encoder_model=None,
        device: str = "cuda",
        num_speakers: Optional[int] = None,
        segment_duration: float = 1.5,
        hop_duration: float = 0.75,
        similarity_threshold: float = 0.6,
    ):
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.num_speakers = num_speakers
        self.segment_duration = segment_duration
        self.hop_duration = hop_duration
        self.similarity_threshold = similarity_threshold
        self._encoder = encoder_model

    @property
    def encoder(self):
        if self._encoder is None:
            from flashaudio.models.architectures.voice_encoder import VoiceEncoderModel
            self._encoder = VoiceEncoderModel().to(self.device)
            self._encoder.eval()
        return self._encoder

    def diarize(
        self,
        audio: Union[str, torch.Tensor],
        sample_rate: int = 16000,
    ) -> List[Dict]:
        """Perform speaker diarization on audio.

        Args:
            audio: Path to audio file or waveform tensor.
            sample_rate: Sample rate.

        Returns:
            List of segments with 'start', 'end', 'speaker' keys.
        """
        waveform = self._load_audio(audio, sample_rate)
        segments = self._segment_audio(waveform, sample_rate)
        embeddings = self._extract_embeddings(segments)

        if len(embeddings) == 0:
            return []

        labels = self._cluster_embeddings(embeddings)

        results = []
        for i, (start, end, _) in enumerate(segments):
            results.append({
                "start": round(start, 2),
                "end": round(end, 2),
                "speaker": f"SPEAKER_{labels[i]:02d}",
            })

        return self._merge_adjacent(results)

    def _segment_audio(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> List[Tuple[float, float, torch.Tensor]]:
        """Split audio into overlapping segments."""
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)

        total_samples = waveform.shape[0]
        segment_samples = int(self.segment_duration * sample_rate)
        hop_samples = int(self.hop_duration * sample_rate)

        segments = []
        start = 0

        while start + segment_samples <= total_samples:
            chunk = waveform[start:start + segment_samples]
            start_time = start / sample_rate
            end_time = (start + segment_samples) / sample_rate
            segments.append((start_time, end_time, chunk))
            start += hop_samples

        if start < total_samples and total_samples - start > sample_rate * 0.5:
            chunk = waveform[start:]
            padded = F.pad(chunk, (0, segment_samples - chunk.shape[0]))
            segments.append((start / sample_rate, total_samples / sample_rate, padded))

        return segments

    def _extract_embeddings(
        self,
        segments: List[Tuple[float, float, torch.Tensor]],
    ) -> torch.Tensor:
        """Extract speaker embeddings for each segment."""
        embeddings = []
        for _, _, chunk in segments:
            chunk = chunk.unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self.encoder(chunk)
            embeddings.append(emb.squeeze(0))

        if not embeddings:
            return torch.tensor([])

        return torch.stack(embeddings)

    def _cluster_embeddings(self, embeddings: torch.Tensor) -> List[int]:
        """Cluster embeddings using agglomerative clustering.

        Uses cosine similarity with a threshold-based merge strategy.
        """
        n = embeddings.shape[0]
        if n == 0:
            return []

        embeddings_norm = F.normalize(embeddings, p=2, dim=-1)
        similarity = torch.mm(embeddings_norm, embeddings_norm.t()).cpu().numpy()

        labels = list(range(n))
        next_label = n

        if self.num_speakers:
            target_clusters = self.num_speakers
        else:
            target_clusters = max(1, n // 10)

        current_clusters = n

        while current_clusters > target_clusters:
            max_sim = -1
            merge_i, merge_j = -1, -1

            unique_labels = list(set(labels))
            for i_idx, li in enumerate(unique_labels):
                for j_idx in range(i_idx + 1, len(unique_labels)):
                    lj = unique_labels[j_idx]
                    members_i = [k for k, l in enumerate(labels) if l == li]
                    members_j = [k for k, l in enumerate(labels) if l == lj]

                    avg_sim = np.mean([similarity[a][b] for a in members_i for b in members_j])
                    if avg_sim > max_sim:
                        max_sim = avg_sim
                        merge_i, merge_j = li, lj

            if max_sim < self.similarity_threshold:
                break

            for k in range(n):
                if labels[k] == merge_j:
                    labels[k] = merge_i

            current_clusters -= 1

        label_map = {}
        counter = 0
        result = []
        for l in labels:
            if l not in label_map:
                label_map[l] = counter
                counter += 1
            result.append(label_map[l])

        return result

    def _merge_adjacent(self, segments: List[Dict]) -> List[Dict]:
        """Merge adjacent segments with the same speaker."""
        if not segments:
            return []

        merged = [segments[0].copy()]
        for seg in segments[1:]:
            if seg["speaker"] == merged[-1]["speaker"]:
                merged[-1]["end"] = seg["end"]
            else:
                merged.append(seg.copy())

        return merged

    def _load_audio(self, audio: Union[str, torch.Tensor], sample_rate: int) -> torch.Tensor:
        """Load audio from path or validate tensor."""
        if isinstance(audio, str):
            from flashaudio.data.audio_utils import load_audio
            waveform, sr = load_audio(audio, sample_rate=sample_rate)
            return waveform.squeeze(0)
        elif isinstance(audio, torch.Tensor):
            if audio.dim() == 2:
                audio = audio.squeeze(0)
            return audio
        else:
            raise TypeError(f"Expected str or torch.Tensor, got {type(audio)}")
