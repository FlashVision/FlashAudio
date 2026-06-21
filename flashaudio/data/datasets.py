"""Audio dataset loaders for LibriSpeech, CommonVoice, and AudioSet.

Each dataset class wraps torchaudio or HuggingFace datasets and provides
a uniform interface returning (waveform, sample_rate, metadata) tuples.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
from torch.utils.data import Dataset

from flashaudio.data.audio_utils import load_audio, pad_or_trim
from flashaudio.registry import DATASETS


@DATASETS.register("LibriSpeech")
class LibriSpeechDataset(Dataset):
    """LibriSpeech speech recognition dataset.

    Expects the standard LibriSpeech directory layout:
        root/speaker_id/chapter_id/speaker_id-chapter_id-utterance_id.flac

    Each sample returns (waveform, sample_rate, transcript, speaker_id).
    """

    def __init__(
        self,
        root: str,
        split: str = "train-clean-100",
        sample_rate: int = 16000,
        max_duration: float = 30.0,
        transform: Optional[Callable] = None,
    ):
        self.root = Path(root) / split
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.max_samples = int(max_duration * sample_rate)
        self.transform = transform
        self.samples: List[Dict] = []

        self._scan_directory()

    def _scan_directory(self):
        """Scan directory for FLAC files and their transcripts."""
        if not self.root.exists():
            return

        for trans_file in sorted(self.root.rglob("*.trans.txt")):
            transcript_dir = trans_file.parent
            with open(trans_file) as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) == 2:
                        utterance_id, text = parts
                        audio_path = transcript_dir / f"{utterance_id}.flac"
                        if audio_path.exists():
                            speaker_id = utterance_id.split("-")[0]
                            self.samples.append({
                                "audio_path": str(audio_path),
                                "text": text.strip(),
                                "speaker_id": speaker_id,
                                "utterance_id": utterance_id,
                            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        waveform, sr = load_audio(sample["audio_path"], sample_rate=self.sample_rate)
        waveform = pad_or_trim(waveform, self.max_samples)

        if self.transform is not None:
            waveform = self.transform(waveform)

        return {
            "waveform": waveform.squeeze(0),
            "sample_rate": self.sample_rate,
            "text": sample["text"],
            "speaker_id": sample["speaker_id"],
        }


@DATASETS.register("CommonVoice")
class CommonVoiceDataset(Dataset):
    """Mozilla Common Voice dataset.

    Loads from a TSV manifest (validated.tsv, train.tsv, etc.) pointing
    to MP3 clips in a clips/ subdirectory.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        language: str = "en",
        sample_rate: int = 16000,
        max_duration: float = 15.0,
        transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.language = language
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.max_samples = int(max_duration * sample_rate)
        self.transform = transform
        self.samples: List[Dict] = []

        tsv_name = f"{split}.tsv"
        self._load_manifest(self.root / tsv_name)

    def _load_manifest(self, tsv_path: Path):
        """Load samples from a TSV manifest file."""
        if not tsv_path.exists():
            return

        with open(tsv_path, encoding="utf-8") as f:
            header = f.readline().strip().split("\t")
            path_idx = header.index("path") if "path" in header else 1
            sentence_idx = header.index("sentence") if "sentence" in header else 2

            for line in f:
                parts = line.strip().split("\t")
                if len(parts) > max(path_idx, sentence_idx):
                    clip_path = self.root / "clips" / parts[path_idx]
                    if clip_path.exists():
                        self.samples.append({
                            "audio_path": str(clip_path),
                            "text": parts[sentence_idx],
                        })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        waveform, sr = load_audio(sample["audio_path"], sample_rate=self.sample_rate)
        waveform = pad_or_trim(waveform, self.max_samples)

        if self.transform is not None:
            waveform = self.transform(waveform)

        return {
            "waveform": waveform.squeeze(0),
            "sample_rate": self.sample_rate,
            "text": sample["text"],
        }


@DATASETS.register("AudioSet")
class AudioSetDataset(Dataset):
    """AudioSet audio classification dataset.

    Expects a directory with audio files and a CSV label file where each
    row maps a filename to one or more comma-separated labels.
    """

    AUDIOSET_LABELS: List[str] = [
        "Speech", "Music", "Environmental sound", "Animal", "Vehicle",
        "Water", "Wind", "Silence", "Noise", "Alarm",
        "Laughter", "Crying", "Singing", "Clapping", "Footsteps",
        "Door", "Keyboard", "Telephone", "Engine", "Siren",
    ]

    def __init__(
        self,
        root: str,
        label_file: str = "labels.csv",
        sample_rate: int = 16000,
        max_duration: float = 10.0,
        num_labels: int = 527,
        transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.max_samples = int(max_duration * sample_rate)
        self.num_labels = num_labels
        self.transform = transform
        self.samples: List[Dict] = []

        self._load_labels(self.root / label_file)

    def _load_labels(self, csv_path: Path):
        """Load labels from a CSV file (filename, labels)."""
        if not csv_path.exists():
            return

        with open(csv_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",", 1)
                if len(parts) == 2:
                    filename, labels_str = parts
                    audio_path = self.root / filename.strip()
                    if audio_path.exists():
                        labels = [lbl.strip() for lbl in labels_str.split(",")]
                        self.samples.append({
                            "audio_path": str(audio_path),
                            "labels": labels,
                        })

    def _encode_labels(self, labels: List[str]) -> torch.Tensor:
        """Encode string labels to a multi-hot vector."""
        label_to_idx = {lbl: i for i, lbl in enumerate(self.AUDIOSET_LABELS)}
        vector = torch.zeros(self.num_labels)
        for label in labels:
            if label in label_to_idx:
                vector[label_to_idx[label]] = 1.0
        return vector

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        waveform, sr = load_audio(sample["audio_path"], sample_rate=self.sample_rate)
        waveform = pad_or_trim(waveform, self.max_samples)

        if self.transform is not None:
            waveform = self.transform(waveform)

        return {
            "waveform": waveform.squeeze(0),
            "sample_rate": self.sample_rate,
            "labels": self._encode_labels(sample["labels"]),
            "label_names": sample["labels"],
        }
