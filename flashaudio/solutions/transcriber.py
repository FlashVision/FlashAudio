"""Transcriber solution — high-level batch transcription with formatting.

Wraps the STT pipeline with output formatting, file handling,
subtitle generation, and speaker-labeled transcription.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


class Transcriber:
    """High-level transcription solution.

    Provides batch transcription, subtitle generation, and formatted
    output on top of the SpeechToText pipeline.

    Args:
        model_id: Whisper model ID.
        device: Device for inference.
        language: Default language (None = auto-detect).
        output_format: Default output format ('text', 'json', 'srt', 'vtt').
    """

    def __init__(
        self,
        model_id: str = "openai/whisper-base",
        device: str = "cuda",
        language: Optional[str] = None,
        output_format: str = "text",
    ):
        self.model_id = model_id
        self.device = device
        self.language = language
        self.output_format = output_format
        self._stt = None

    @property
    def stt(self):
        if self._stt is None:
            from flashaudio.speech.stt import SpeechToText
            self._stt = SpeechToText(
                model_id=self.model_id,
                device=self.device,
                language=self.language,
            )
        return self._stt

    def transcribe(
        self,
        audio: Union[str, List[str]],
        output_format: Optional[str] = None,
        output_path: Optional[str] = None,
        word_timestamps: bool = False,
    ) -> Union[str, List[Dict]]:
        """Transcribe one or more audio files.

        Args:
            audio: Path to audio file or list of paths.
            output_format: Override output format.
            output_path: If provided, save output to this file.
            word_timestamps: Enable word-level timestamps.

        Returns:
            Transcription text or list of result dictionaries.
        """
        fmt = output_format or self.output_format

        if isinstance(audio, str):
            result = self.stt.transcribe(audio, word_timestamps=word_timestamps)
            formatted = self._format_result(result, fmt)
        else:
            results = self.stt.transcribe_batch(audio)
            formatted = self._format_batch(results, fmt)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(formatted if isinstance(formatted, str) else json.dumps(formatted, indent=2))

        return formatted

    def transcribe_to_srt(
        self,
        audio: str,
        output_path: str = "subtitles.srt",
    ) -> str:
        """Transcribe and generate SRT subtitles.

        Args:
            audio: Path to audio file.
            output_path: Path to save SRT file.

        Returns:
            SRT formatted string.
        """
        result = self.stt.transcribe(audio, return_segments=True)
        srt = self._to_srt(result.get("segments", []))

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(srt)

        return srt

    def transcribe_to_vtt(
        self,
        audio: str,
        output_path: str = "subtitles.vtt",
    ) -> str:
        """Transcribe and generate WebVTT subtitles.

        Args:
            audio: Path to audio file.
            output_path: Path to save VTT file.

        Returns:
            VTT formatted string.
        """
        result = self.stt.transcribe(audio, return_segments=True)
        vtt = self._to_vtt(result.get("segments", []))

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(vtt)

        return vtt

    def _format_result(self, result: Dict, fmt: str) -> Union[str, Dict]:
        """Format a single transcription result."""
        if fmt == "text":
            return result.get("text", "")
        elif fmt == "json":
            return result
        elif fmt == "srt":
            return self._to_srt(result.get("segments", []))
        elif fmt == "vtt":
            return self._to_vtt(result.get("segments", []))
        return result.get("text", "")

    def _format_batch(self, results: List[Dict], fmt: str) -> Union[str, List]:
        """Format batch transcription results."""
        if fmt == "text":
            return "\n\n".join(r.get("text", "") for r in results)
        elif fmt == "json":
            return results
        return results

    def _to_srt(self, segments: List[Dict]) -> str:
        """Convert segments to SRT format."""
        lines = []
        for i, seg in enumerate(segments, 1):
            start = self._format_timestamp(seg.get("start", 0), ",")
            end = self._format_timestamp(seg.get("end", 0), ",")
            text = seg.get("text", "").strip()
            lines.append(f"{i}\n{start} --> {end}\n{text}\n")
        return "\n".join(lines)

    def _to_vtt(self, segments: List[Dict]) -> str:
        """Convert segments to WebVTT format."""
        lines = ["WEBVTT\n"]
        for seg in segments:
            start = self._format_timestamp(seg.get("start", 0), ".")
            end = self._format_timestamp(seg.get("end", 0), ".")
            text = seg.get("text", "").strip()
            lines.append(f"{start} --> {end}\n{text}\n")
        return "\n".join(lines)

    @staticmethod
    def _format_timestamp(seconds: float, sep: str = ",") -> str:
        """Format seconds as HH:MM:SS,mmm timestamp."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millis:03d}"
