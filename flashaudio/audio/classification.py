"""Audio event classification pipeline.

Classifies audio into event categories (e.g. AudioSet ontology)
using a CNN-based classifier operating on mel spectrograms.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch

from flashaudio.registry import PIPELINES


AUDIOSET_LABELS = [
    "Speech", "Music", "Singing", "Musical instrument", "Drum",
    "Guitar", "Piano", "Keyboard (musical)", "Bass guitar", "Violin",
    "Flute", "Trumpet", "Saxophone", "Percussion", "Clapping",
    "Laughter", "Crying", "Cough", "Sneeze", "Breathing",
    "Footsteps", "Door", "Knock", "Bell", "Alarm",
    "Siren", "Buzzer", "Telephone", "Ring", "Ringtone",
    "Engine", "Vehicle", "Car", "Train", "Aircraft",
    "Motorcycle", "Truck", "Bus", "Bicycle", "Boat",
    "Water", "Rain", "Thunder", "Wind", "Stream",
    "Ocean", "Waterfall", "Fire", "Crackling", "Explosion",
    "Gunshot", "Fireworks", "Glass", "Breaking", "Crash",
    "Dog", "Cat", "Bird", "Chicken", "Rooster",
    "Cow", "Pig", "Horse", "Frog", "Cricket",
    "Insect", "Bee", "Mosquito", "Fly", "Snake",
    "Crowd", "Applause", "Cheering", "Booing", "Screaming",
    "Whispering", "Humming", "Whistling", "Yelling", "Chanting",
    "Typing", "Click", "Tap", "Scraping", "Scratching",
    "Rubbing", "Sawing", "Filing", "Hammering", "Drilling",
    "Chopping", "Stirring", "Pouring", "Boiling", "Sizzling",
    "Frying", "Microwave oven", "Blender", "Vacuum cleaner", "Hair dryer",
]


@PIPELINES.register("AudioClassifier")
class AudioClassifier:
    """Audio event classifier pipeline.

    Args:
        model: Optional pre-loaded classification model.
        device: Device for inference.
        labels: List of label names. Defaults to AudioSet-style labels.
        sample_rate: Expected input sample rate.
    """

    def __init__(
        self,
        model=None,
        device: str = "cuda",
        labels: Optional[List[str]] = None,
        sample_rate: int = 16000,
    ):
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.labels = labels or AUDIOSET_LABELS
        self.sample_rate = sample_rate
        self._model = model

    @property
    def model(self):
        if self._model is None:
            from flashaudio.models.architectures.audio_classifier import AudioClassifierModel
            self._model = AudioClassifierModel(num_classes=len(self.labels)).to(self.device)
            self._model.eval()
        return self._model

    def classify(
        self,
        audio: Union[str, torch.Tensor],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[Tuple[str, float]]:
        """Classify audio events.

        Args:
            audio: Path to audio file or waveform tensor.
            top_k: Number of top predictions to return.
            threshold: Minimum probability threshold.

        Returns:
            List of (label, probability) tuples sorted by probability.
        """
        waveform = self._load_audio(audio)

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        waveform = waveform.to(self.device)

        with torch.no_grad():
            logits = self.model(waveform)
            probs = torch.sigmoid(logits).squeeze(0)

        top_probs, top_indices = probs.topk(min(top_k, len(self.labels)))

        results = []
        for prob, idx in zip(top_probs.cpu().tolist(), top_indices.cpu().tolist()):
            if prob >= threshold:
                label = self.labels[idx] if idx < len(self.labels) else f"class_{idx}"
                results.append((label, prob))

        return results

    def classify_batch(
        self,
        audio_paths: List[str],
        top_k: int = 5,
    ) -> List[List[Tuple[str, float]]]:
        """Classify multiple audio files.

        Args:
            audio_paths: List of audio file paths.
            top_k: Number of top predictions per file.

        Returns:
            List of classification results.
        """
        return [self.classify(path, top_k=top_k) for path in audio_paths]

    def classify_segments(
        self,
        audio: Union[str, torch.Tensor],
        segment_duration: float = 2.0,
        hop_duration: float = 1.0,
        top_k: int = 3,
    ) -> List[Dict]:
        """Classify audio in sliding windows.

        Args:
            audio: Path to audio file or waveform tensor.
            segment_duration: Duration of each segment in seconds.
            hop_duration: Hop between segments in seconds.
            top_k: Top predictions per segment.

        Returns:
            List of {start, end, predictions} dictionaries.
        """
        waveform = self._load_audio(audio)
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)

        segment_samples = int(segment_duration * self.sample_rate)
        hop_samples = int(hop_duration * self.sample_rate)
        total_samples = waveform.shape[0]

        results = []
        start = 0

        while start + segment_samples <= total_samples:
            chunk = waveform[start:start + segment_samples]
            predictions = self.classify(chunk, top_k=top_k)
            results.append({
                "start": round(start / self.sample_rate, 2),
                "end": round((start + segment_samples) / self.sample_rate, 2),
                "predictions": predictions,
            })
            start += hop_samples

        return results

    def _load_audio(self, audio: Union[str, torch.Tensor]) -> torch.Tensor:
        """Load audio from path or validate tensor."""
        if isinstance(audio, (str, Path)):
            from flashaudio.data.audio_utils import load_audio
            waveform, sr = load_audio(str(audio), sample_rate=self.sample_rate)
            return waveform.squeeze(0)
        elif isinstance(audio, torch.Tensor):
            return audio
        else:
            raise TypeError(f"Expected str or torch.Tensor, got {type(audio)}")
