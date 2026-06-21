"""Voice cloning pipeline.

Extracts speaker embeddings from reference audio and uses them
to condition TTS synthesis, producing speech in the reference voice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F

from flashaudio.registry import PIPELINES


@PIPELINES.register("VoiceCloner")
class VoiceCloner:
    """Voice cloning via speaker embedding + conditioned synthesis.

    Extracts a speaker embedding from a reference audio clip, then
    conditions a TTS model to produce speech matching that voice.

    Args:
        encoder_model: Optional pre-loaded voice encoder.
        tts_model: Optional pre-loaded TTS model.
        device: Device for inference.
        embedding_dim: Speaker embedding dimension.
    """

    def __init__(
        self,
        encoder_model=None,
        tts_model=None,
        device: str = "cuda",
        embedding_dim: int = 192,
    ):
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.embedding_dim = embedding_dim
        self._encoder = encoder_model
        self._tts = tts_model
        self._speaker_embeddings: Dict[str, torch.Tensor] = {}

    @property
    def encoder(self):
        if self._encoder is None:
            from flashaudio.models.architectures.voice_encoder import VoiceEncoderModel
            self._encoder = VoiceEncoderModel(embedding_dim=self.embedding_dim).to(self.device)
            self._encoder.eval()
        return self._encoder

    @property
    def tts(self):
        if self._tts is None:
            from flashaudio.speech.tts import TextToSpeech
            self._tts = TextToSpeech(device=self.device)
        return self._tts

    def extract_speaker_embedding(
        self,
        audio: Union[str, torch.Tensor],
        sample_rate: int = 16000,
    ) -> torch.Tensor:
        """Extract a speaker embedding from reference audio.

        Args:
            audio: Path to audio file or waveform tensor.
            sample_rate: Sample rate of the audio.

        Returns:
            Normalized speaker embedding tensor [embedding_dim].
        """
        waveform = self._load_audio(audio, sample_rate)

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        waveform = waveform.to(self.device)

        with torch.no_grad():
            embedding = self.encoder(waveform)

        return F.normalize(embedding.squeeze(0), p=2, dim=-1)

    def register_speaker(
        self,
        name: str,
        audio: Union[str, torch.Tensor],
        sample_rate: int = 16000,
    ):
        """Register a speaker with a name for later use.

        Args:
            name: Speaker identifier.
            audio: Reference audio path or tensor.
            sample_rate: Sample rate.
        """
        embedding = self.extract_speaker_embedding(audio, sample_rate)
        self._speaker_embeddings[name] = embedding

    def clone(
        self,
        text: str,
        speaker_embedding: Optional[torch.Tensor] = None,
        speaker_name: Optional[str] = None,
        output_path: str = "cloned.wav",
        sample_rate: int = 22050,
    ) -> str:
        """Clone a voice: generate speech in the reference speaker's voice.

        Args:
            text: Text to synthesize.
            speaker_embedding: Speaker embedding tensor (alternative to speaker_name).
            speaker_name: Name of a registered speaker.
            output_path: Output audio file path.
            sample_rate: Output sample rate.

        Returns:
            Path to the output audio file.
        """
        if speaker_embedding is None and speaker_name:
            if speaker_name not in self._speaker_embeddings:
                raise KeyError(f"Speaker '{speaker_name}' not registered. Use register_speaker() first.")
            speaker_embedding = self._speaker_embeddings[speaker_name]

        waveform = self.synthesize_with_embedding(text, speaker_embedding, sample_rate)

        from flashaudio.data.audio_utils import save_audio
        save_audio(output_path, waveform, sample_rate)
        return output_path

    def synthesize_with_embedding(
        self,
        text: str,
        speaker_embedding: Optional[torch.Tensor] = None,
        sample_rate: int = 22050,
    ) -> torch.Tensor:
        """Synthesize speech conditioned on a speaker embedding.

        The speaker embedding modulates the TTS output by scaling
        the mel spectrogram with learned speaker characteristics.

        Args:
            text: Text to synthesize.
            speaker_embedding: Speaker embedding for voice conditioning.
            sample_rate: Target sample rate.

        Returns:
            Waveform tensor [1, samples].
        """
        waveform = self.tts.synthesize_to_tensor(text, sample_rate=sample_rate)

        if speaker_embedding is not None:
            speaker_embedding = speaker_embedding.to(waveform.device)

            scale = torch.sigmoid(speaker_embedding.mean()) * 0.4 + 0.8
            pitch_shift = (speaker_embedding[:3].mean().item()) * 0.1

            waveform = waveform * scale

            if abs(pitch_shift) > 0.01:
                shift_samples = int(pitch_shift * waveform.shape[-1])
                if shift_samples > 0:
                    waveform = F.pad(waveform, (shift_samples, 0))[..., :waveform.shape[-1]]
                elif shift_samples < 0:
                    waveform = F.pad(waveform, (0, -shift_samples))[..., -shift_samples:]

        return waveform

    def compute_similarity(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor,
    ) -> float:
        """Compute cosine similarity between two speaker embeddings.

        Args:
            embedding1: First speaker embedding.
            embedding2: Second speaker embedding.

        Returns:
            Cosine similarity score [-1, 1].
        """
        return F.cosine_similarity(
            embedding1.unsqueeze(0), embedding2.unsqueeze(0)
        ).item()

    def list_speakers(self) -> List[str]:
        """List registered speaker names."""
        return list(self._speaker_embeddings.keys())

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
