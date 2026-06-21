"""Speech-to-Text pipeline.

Whisper-based transcription with language detection, word-level timestamps,
and batch processing support.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from flashaudio.registry import PIPELINES


@PIPELINES.register("SpeechToText")
class SpeechToText:
    """Speech-to-text pipeline using OpenAI Whisper.

    Args:
        model_id: HuggingFace Whisper model ID.
        device: Device for inference.
        language: Default language (None = auto-detect).
        compute_type: Computation dtype ('float32' or 'float16').
    """

    SUPPORTED_MODELS = [
        "openai/whisper-tiny",
        "openai/whisper-base",
        "openai/whisper-small",
        "openai/whisper-medium",
        "openai/whisper-large-v3",
    ]

    def __init__(
        self,
        model_id: str = "openai/whisper-base",
        device: str = "cuda",
        language: Optional[str] = None,
        compute_type: str = "float32",
    ):
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.language = language
        self.compute_type = compute_type
        self._model = None
        self._processor = None

    def _ensure_loaded(self):
        """Lazy-load model and processor."""
        if self._model is not None:
            return

        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        dtype = torch.float16 if self.compute_type == "float16" and self.device != "cpu" else torch.float32

        self._processor = WhisperProcessor.from_pretrained(self.model_id)
        self._model = WhisperForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=dtype
        ).to(self.device)
        self._model.eval()

    def transcribe(
        self,
        audio: Union[str, torch.Tensor],
        language: Optional[str] = None,
        word_timestamps: bool = False,
        return_segments: bool = True,
        sample_rate: int = 16000,
    ) -> Dict[str, Any]:
        """Transcribe audio to text.

        Args:
            audio: Path to audio file or waveform tensor.
            language: Language code (None = auto-detect).
            word_timestamps: Whether to return word-level timestamps.
            return_segments: Whether to return segment-level output.
            sample_rate: Sample rate (used when audio is a tensor).

        Returns:
            Dictionary with 'text', optional 'language', 'segments', 'words'.
        """
        self._ensure_loaded()

        waveform = self._load_audio(audio, sample_rate)
        waveform_np = waveform.cpu().numpy()

        inputs = self._processor(
            waveform_np, sampling_rate=sample_rate, return_tensors="pt"
        )
        input_features = inputs.input_features.to(self.device)

        generate_kwargs = {}
        lang = language or self.language
        if lang:
            generate_kwargs["language"] = lang
        if word_timestamps:
            generate_kwargs["return_timestamps"] = True

        with torch.no_grad():
            predicted_ids = self._model.generate(input_features, **generate_kwargs)

        transcription = self._processor.batch_decode(predicted_ids, skip_special_tokens=True)
        text = transcription[0].strip() if transcription else ""

        result: Dict[str, Any] = {"text": text}

        if lang:
            result["language"] = lang

        if return_segments:
            duration = len(waveform_np) / sample_rate
            result["segments"] = self._create_segments(text, duration)

        return result

    def transcribe_batch(
        self,
        audio_paths: List[str],
        language: Optional[str] = None,
        sample_rate: int = 16000,
    ) -> List[Dict[str, Any]]:
        """Transcribe multiple audio files.

        Args:
            audio_paths: List of paths to audio files.
            language: Language code for all files.
            sample_rate: Expected sample rate.

        Returns:
            List of transcription result dictionaries.
        """
        results = []
        for path in audio_paths:
            result = self.transcribe(path, language=language, sample_rate=sample_rate)
            result["file"] = path
            results.append(result)
        return results

    def detect_language(self, audio: Union[str, torch.Tensor], sample_rate: int = 16000) -> str:
        """Detect the language of the audio.

        Args:
            audio: Path to audio file or waveform tensor.
            sample_rate: Sample rate.

        Returns:
            Detected language code.
        """
        self._ensure_loaded()

        waveform = self._load_audio(audio, sample_rate)
        waveform_np = waveform.cpu().numpy()

        inputs = self._processor(waveform_np, sampling_rate=sample_rate, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)

        with torch.no_grad():
            encoder_output = self._model.get_encoder()(input_features)
            decoder_input_ids = torch.tensor([[self._model.config.decoder_start_token_id]]).to(self.device)
            logits = self._model(
                encoder_outputs=(encoder_output,),
                decoder_input_ids=decoder_input_ids,
            ).logits

        lang_token_ids = {
            v: k for k, v in self._processor.tokenizer.get_vocab().items()
            if k.startswith("<|") and k.endswith("|>") and len(k) == 6
        }

        if lang_token_ids:
            lang_logits = {lang: logits[0, 0, idx].item() for idx, lang in lang_token_ids.items()}
            detected = max(lang_logits, key=lang_logits.get)
            return detected.strip("<|>")

        return "en"

    def _load_audio(self, audio: Union[str, torch.Tensor], sample_rate: int) -> torch.Tensor:
        """Load audio from path or validate tensor."""
        if isinstance(audio, (str, Path)):
            from flashaudio.data.audio_utils import load_audio
            waveform, sr = load_audio(str(audio), sample_rate=sample_rate)
            return waveform.squeeze(0)
        elif isinstance(audio, torch.Tensor):
            if audio.dim() == 2:
                audio = audio.squeeze(0)
            return audio
        else:
            raise TypeError(f"Expected str or torch.Tensor, got {type(audio)}")

    def _create_segments(self, text: str, duration: float) -> List[Dict]:
        """Create simple segments from full text."""
        if not text:
            return []

        sentences = []
        current = ""
        for char in text:
            current += char
            if char in ".!?" and current.strip():
                sentences.append(current.strip())
                current = ""
        if current.strip():
            sentences.append(current.strip())

        if not sentences:
            return [{"start": 0.0, "end": duration, "text": text}]

        time_per_sentence = duration / len(sentences)
        segments = []
        for i, sentence in enumerate(sentences):
            segments.append({
                "start": round(i * time_per_sentence, 2),
                "end": round((i + 1) * time_per_sentence, 2),
                "text": sentence,
            })
        return segments
