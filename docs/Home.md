# FlashAudio Documentation

Welcome to the FlashAudio documentation. FlashAudio is a production-grade Audio & Speech AI framework built on PyTorch.

## Modules

- **[Installation](Installation.md)** — Setup and dependencies
- **[Quick Start](Quick-Start.md)** — Get started in 5 minutes
- **[Speech-to-Text](Speech-to-Text.md)** — Whisper-based transcription
- **[Text-to-Speech](Text-to-Speech.md)** — Speech synthesis
- **[Audio Generation](Audio-Generation.md)** — Music and sound generation
- **[Voice Cloning](Voice-Cloning.md)** — Speaker embedding and voice cloning
- **[FAQ](FAQ.md)** — Frequently asked questions

## Architecture

FlashAudio is organized into modular components:

- `flashaudio.speech` — STT, TTS, voice cloning, diarization
- `flashaudio.audio` — Classification, generation, separation, features
- `flashaudio.models` — Model architectures and LoRA
- `flashaudio.engine` — Training, validation, prediction, export
- `flashaudio.solutions` — High-level solutions (Transcriber, Narrator, AudioAnalyzer)
- `flashaudio.analytics` — Benchmarking and metrics
- `flashaudio.data` — Datasets and audio transforms
