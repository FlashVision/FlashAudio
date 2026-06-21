"""Whisper wrapper for speech-to-text.

Provides a PyTorch nn.Module interface around OpenAI Whisper or
HuggingFace Whisper models for STT inference and fine-tuning.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from flashaudio.registry import MODELS


@MODELS.register("WhisperSTT")
class WhisperWrapper(nn.Module):
    """Wrapper around HuggingFace Whisper for speech-to-text.

    Provides a simplified interface for transcription with optional
    language detection and word-level timestamps.

    Args:
        model_id: HuggingFace Whisper model ID.
        language: Default language for transcription.
    """

    def __init__(
        self,
        model_id: str = "openai/whisper-base",
        language: Optional[str] = None,
    ):
        super().__init__()
        self.model_id = model_id
        self.language = language
        self._model = None
        self._processor = None
        self._feature_extractor = None

    def _ensure_loaded(self):
        """Lazy-load the Whisper model and processor."""
        if self._model is not None:
            return

        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self._processor = WhisperProcessor.from_pretrained(self.model_id)
        self._model = WhisperForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=torch.float32
        )

    def forward(self, input_features: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass through the Whisper encoder.

        Args:
            input_features: Mel spectrogram features [batch, n_mels, time].

        Returns:
            Dictionary with encoder outputs.
        """
        self._ensure_loaded()
        device = input_features.device
        self._model = self._model.to(device)

        encoder_outputs = self._model.get_encoder()(input_features)
        return {"encoder_output": encoder_outputs.last_hidden_state}

    def transcribe(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        word_timestamps: bool = False,
        return_segments: bool = True,
    ) -> Dict[str, Any]:
        """Transcribe a waveform to text.

        Args:
            waveform: Audio tensor [samples] or [1, samples].
            sample_rate: Sample rate of the waveform.
            language: Language code (None = auto-detect).
            word_timestamps: Whether to return word-level timestamps.
            return_segments: Whether to return segment-level output.

        Returns:
            Dictionary with 'text', optional 'language', 'segments'.
        """
        self._ensure_loaded()
        device = next(self._model.parameters()).device

        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)

        waveform_np = waveform.cpu().numpy()

        inputs = self._processor(
            waveform_np,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        input_features = inputs.input_features.to(device)

        generate_kwargs = {}
        lang = language or self.language
        if lang:
            generate_kwargs["language"] = lang
        if word_timestamps:
            generate_kwargs["return_timestamps"] = True

        with torch.no_grad():
            predicted_ids = self._model.generate(input_features, **generate_kwargs)

        transcription = self._processor.batch_decode(predicted_ids, skip_special_tokens=True)
        text = transcription[0] if transcription else ""

        result: Dict[str, Any] = {"text": text.strip()}

        if lang:
            result["language"] = lang

        if return_segments:
            duration = len(waveform_np) / sample_rate
            result["segments"] = [{
                "start": 0.0,
                "end": duration,
                "text": text.strip(),
            }]

        return result

    def get_encoder_output(self, input_features: torch.Tensor) -> torch.Tensor:
        """Get the raw encoder hidden states.

        Useful for downstream tasks like speaker verification or
        audio embeddings.
        """
        self._ensure_loaded()
        device = input_features.device
        self._model = self._model.to(device)

        with torch.no_grad():
            encoder_outputs = self._model.get_encoder()(input_features)
        return encoder_outputs.last_hidden_state
