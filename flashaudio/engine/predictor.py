"""Prediction engine for FlashAudio models.

Provides a unified interface for running inference across all audio tasks:
STT, TTS, classification, and feature extraction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn


class Predictor:
    """Unified prediction engine for audio models.

    Args:
        model_id: HuggingFace model ID or local path.
        model: Optional pre-loaded model.
        task: Task type (stt, tts, classification, features).
        device: Device for inference.
    """

    def __init__(
        self,
        model_id: str = "openai/whisper-base",
        model: Optional[nn.Module] = None,
        task: str = "stt",
        device: str = "cuda",
    ):
        self.model_id = model_id
        self.task = task
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self._model = model
        self._processor = None

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self):
        """Lazy-load model from HuggingFace or local path."""
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        try:
            self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id, torch_dtype=torch.float32
            ).to(self.device)
            self._processor = AutoProcessor.from_pretrained(self.model_id)
        except Exception:
            from flashaudio.models.flashaudio_model import FlashAudio
            flash = FlashAudio(model_id=self.model_id, device=self.device)
            self._model = flash.model

    def predict(
        self,
        audio_path: Optional[str] = None,
        waveform: Optional[torch.Tensor] = None,
        text: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run prediction based on the configured task.

        Args:
            audio_path: Path to an audio file.
            waveform: Raw waveform tensor.
            text: Text input (for TTS).
            **kwargs: Additional task-specific arguments.

        Returns:
            Dictionary with task-specific results.
        """
        if self.task == "stt":
            return self._predict_stt(audio_path=audio_path, waveform=waveform, **kwargs)
        elif self.task == "tts":
            return self._predict_tts(text=text, **kwargs)
        elif self.task == "classification":
            return self._predict_classification(audio_path=audio_path, waveform=waveform, **kwargs)
        else:
            return self._predict_features(audio_path=audio_path, waveform=waveform, **kwargs)

    def _predict_stt(
        self,
        audio_path: Optional[str] = None,
        waveform: Optional[torch.Tensor] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Speech-to-text prediction."""
        from flashaudio.data.audio_utils import load_audio

        if waveform is None and audio_path is not None:
            waveform, sr = load_audio(audio_path, sample_rate=16000)
            waveform = waveform.squeeze(0)

        if self._processor is not None:
            inputs = self._processor(
                waveform.numpy(), sampling_rate=16000, return_tensors="pt"
            )
            input_features = inputs.input_features.to(self.device)

            generate_kwargs = {}
            if language:
                generate_kwargs["language"] = language

            with torch.no_grad():
                predicted_ids = self.model.generate(input_features, **generate_kwargs)

            transcription = self._processor.batch_decode(predicted_ids, skip_special_tokens=True)
            return {"text": transcription[0] if transcription else "", "language": language}

        with torch.no_grad():
            output = self.model(waveform.unsqueeze(0).to(self.device))

        if isinstance(output, dict):
            return output
        return {"text": str(output), "language": language}

    def _predict_tts(self, text: str, **kwargs) -> Dict[str, Any]:
        """Text-to-speech prediction."""
        from flashaudio.speech.tts import TextToSpeech

        tts = TextToSpeech(device=self.device)
        waveform = tts.synthesize_to_tensor(text, **kwargs)
        return {"waveform": waveform, "sample_rate": kwargs.get("sample_rate", 22050)}

    def _predict_classification(
        self,
        audio_path: Optional[str] = None,
        waveform: Optional[torch.Tensor] = None,
        top_k: int = 5,
        **kwargs,
    ) -> Dict[str, Any]:
        """Audio classification prediction."""
        from flashaudio.data.audio_utils import load_audio

        if waveform is None and audio_path is not None:
            waveform, sr = load_audio(audio_path, sample_rate=16000)

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        with torch.no_grad():
            output = self.model(waveform.to(self.device))
            if isinstance(output, dict):
                logits = output.get("logits", output.get("output"))
            else:
                logits = output

        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = probs.topk(top_k, dim=-1)

        return {
            "probabilities": top_probs.cpu().tolist(),
            "indices": top_indices.cpu().tolist(),
        }

    def _predict_features(
        self,
        audio_path: Optional[str] = None,
        waveform: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Extract audio features."""
        from flashaudio.audio.features import AudioFeatureExtractor
        from flashaudio.data.audio_utils import load_audio

        if waveform is None and audio_path is not None:
            waveform, sr = load_audio(audio_path, sample_rate=16000)

        extractor = AudioFeatureExtractor()
        features = extractor.extract_all(waveform)
        return features
