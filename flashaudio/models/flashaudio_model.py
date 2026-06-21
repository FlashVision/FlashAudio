"""FlashAudio — unified entry point for audio & speech models.

Wraps HuggingFace models and custom architectures behind a single
interface that routes by task (stt, tts, classification, features).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from flashaudio.registry import MODELS


@MODELS.register("FlashAudio")
class FlashAudio:
    """Unified audio model interface.

    Args:
        task: One of 'stt', 'tts', 'classification', 'voice_clone', 'generation'.
        model_id: HuggingFace model ID or local path.
        device: Device for inference.
        config: Optional FlashAudioConfig.
    """

    SUPPORTED_TASKS = {"stt", "tts", "classification", "voice_clone", "generation", "features"}

    def __init__(
        self,
        task: str = "stt",
        model_id: str = "openai/whisper-base",
        device: str = "cuda",
        config: Optional[Any] = None,
    ):
        self.task = task
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.config = config
        self._model = None
        self._processor = None
        self._pipeline = None

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self):
        """Load the appropriate model based on task."""
        if self.task == "stt":
            self._load_stt_model()
        elif self.task == "tts":
            self._load_tts_model()
        elif self.task == "classification":
            self._load_classification_model()
        elif self.task == "voice_clone":
            self._load_voice_encoder()
        else:
            self._load_stt_model()

    def _load_stt_model(self):
        """Load a Whisper model for speech-to-text."""
        try:
            from transformers import WhisperForConditionalGeneration, WhisperProcessor
            self._model = WhisperForConditionalGeneration.from_pretrained(
                self.model_id, torch_dtype=torch.float32
            ).to(self.device)
            self._processor = WhisperProcessor.from_pretrained(self.model_id)
        except Exception:
            from flashaudio.models.architectures.whisper import WhisperWrapper
            self._model = WhisperWrapper(model_id=self.model_id).to(self.device)

    def _load_tts_model(self):
        """Load a TTS model."""
        from flashaudio.models.architectures.tts_model import TTSModel
        self._model = TTSModel().to(self.device)

    def _load_classification_model(self):
        """Load an audio classification model."""
        from flashaudio.models.architectures.audio_classifier import AudioClassifierModel
        self._model = AudioClassifierModel().to(self.device)

    def _load_voice_encoder(self):
        """Load a voice encoder for speaker embeddings."""
        from flashaudio.models.architectures.voice_encoder import VoiceEncoderModel
        self._model = VoiceEncoderModel().to(self.device)

    def transcribe(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """Transcribe an audio file (STT task).

        Args:
            audio_path: Path to audio file.
            **kwargs: Additional arguments (language, word_timestamps).

        Returns:
            Dictionary with 'text' and optional 'segments'.
        """
        from flashaudio.speech.stt import SpeechToText
        stt = SpeechToText(model_id=self.model_id, device=self.device)
        return stt.transcribe(audio_path, **kwargs)

    def synthesize(self, text: str, output_path: str = "output.wav", **kwargs) -> str:
        """Synthesize speech from text (TTS task).

        Args:
            text: Input text.
            output_path: Output audio file path.
            **kwargs: Additional arguments (sample_rate, speed).

        Returns:
            Path to the output audio file.
        """
        from flashaudio.speech.tts import TextToSpeech
        tts = TextToSpeech(device=self.device)
        return tts.synthesize(text, output_path=output_path, **kwargs)

    def classify(self, audio_path: str, **kwargs) -> List:
        """Classify an audio file.

        Args:
            audio_path: Path to audio file.
            **kwargs: Additional arguments (top_k).

        Returns:
            List of (label, score) tuples.
        """
        from flashaudio.audio.classification import AudioClassifier
        classifier = AudioClassifier(device=self.device)
        return classifier.classify(audio_path, **kwargs)

    def extract_features(self, audio_path: str, **kwargs) -> Dict[str, torch.Tensor]:
        """Extract audio features.

        Args:
            audio_path: Path to audio file.

        Returns:
            Dictionary of feature_name -> tensor.
        """
        from flashaudio.audio.features import AudioFeatureExtractor
        from flashaudio.data.audio_utils import load_audio

        extractor = AudioFeatureExtractor(**kwargs)
        waveform, sr = load_audio(audio_path)
        return extractor.extract_all(waveform, sr)

    def __repr__(self) -> str:
        return f"FlashAudio(task={self.task!r}, model_id={self.model_id!r}, device={self.device!r})"
