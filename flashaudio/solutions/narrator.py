"""Narrator solution — high-level text-to-speech with document processing.

Converts documents, articles, or structured text into natural-sounding
audio narration with configurable voice and pacing.
"""

from __future__ import annotations

import re
from typing import List

import torch


class Narrator:
    """High-level text-to-speech narration solution.

    Processes long-form text, splitting into paragraphs and sentences,
    synthesizing each with natural pauses, and concatenating into
    a single audio file.

    Args:
        device: Device for inference.
        sample_rate: Output audio sample rate.
        pause_duration: Duration of pause between sentences (seconds).
        paragraph_pause: Duration of pause between paragraphs (seconds).
    """

    def __init__(
        self,
        device: str = "cuda",
        sample_rate: int = 22050,
        pause_duration: float = 0.3,
        paragraph_pause: float = 0.8,
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.pause_duration = pause_duration
        self.paragraph_pause = paragraph_pause
        self._tts = None

    @property
    def tts(self):
        if self._tts is None:
            from flashaudio.speech.tts import TextToSpeech
            self._tts = TextToSpeech(device=self.device, sample_rate=self.sample_rate)
        return self._tts

    def narrate(
        self,
        text: str,
        output_path: str = "narration.wav",
        speed: float = 1.0,
    ) -> str:
        """Narrate a text document.

        Splits text into paragraphs and sentences, synthesizes each,
        and concatenates with natural pauses.

        Args:
            text: Input text to narrate.
            output_path: Output audio file path.
            speed: Speaking speed multiplier.

        Returns:
            Path to the output audio file.
        """
        paragraphs = self._split_paragraphs(text)
        audio_chunks = []

        sentence_pause = torch.zeros(1, int(self.pause_duration * self.sample_rate))
        paragraph_pause = torch.zeros(1, int(self.paragraph_pause * self.sample_rate))

        for i, paragraph in enumerate(paragraphs):
            sentences = self._split_sentences(paragraph)

            for j, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if not sentence:
                    continue

                waveform = self.tts.synthesize_to_tensor(sentence, speed=speed)
                audio_chunks.append(waveform.cpu())

                if j < len(sentences) - 1:
                    audio_chunks.append(sentence_pause)

            if i < len(paragraphs) - 1:
                audio_chunks.append(paragraph_pause)

        if not audio_chunks:
            audio_chunks.append(torch.zeros(1, self.sample_rate))

        full_audio = torch.cat(audio_chunks, dim=-1)

        from flashaudio.data.audio_utils import save_audio
        save_audio(output_path, full_audio, self.sample_rate)

        return output_path

    def narrate_file(
        self,
        file_path: str,
        output_path: str = "narration.wav",
        speed: float = 1.0,
    ) -> str:
        """Narrate a text file.

        Args:
            file_path: Path to a text file.
            output_path: Output audio file path.
            speed: Speaking speed.

        Returns:
            Path to the output audio.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return self.narrate(text, output_path=output_path, speed=speed)

    def narrate_sections(
        self,
        sections: List[dict],
        output_path: str = "narration.wav",
    ) -> str:
        """Narrate structured content with sections.

        Args:
            sections: List of dicts with 'title' and 'content' keys.
            output_path: Output audio path.

        Returns:
            Path to the output audio.
        """
        full_text = ""
        for section in sections:
            title = section.get("title", "")
            content = section.get("content", "")
            if title:
                full_text += f"{title}.\n\n"
            full_text += f"{content}\n\n"

        return self.narrate(full_text.strip(), output_path=output_path)

    def estimate_duration(self, text: str, speed: float = 1.0) -> float:
        """Estimate narration duration in seconds.

        Uses a rough words-per-minute model.

        Args:
            text: Input text.
            speed: Speed multiplier.

        Returns:
            Estimated duration in seconds.
        """
        words = len(text.split())
        wpm = 150 * speed
        speaking_time = words / wpm * 60

        paragraphs = self._split_paragraphs(text)
        total_sentences = sum(len(self._split_sentences(p)) for p in paragraphs)
        pause_time = (total_sentences * self.pause_duration +
                      len(paragraphs) * self.paragraph_pause)

        return speaking_time + pause_time

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = re.split(r"\n\s*\n", text.strip())
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s for s in sentences if s.strip()]
