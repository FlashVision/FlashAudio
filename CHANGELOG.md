# Changelog

All notable changes to FlashAudio will be documented in this file.

## [1.0.0] — 2026-06-21

### Added
- **Package structure** — `pip install` from GitHub or PyPI
- **CLI** — `flashaudio transcribe`, `speak`, `train`, `classify`, `export`, `benchmark`, `check`, `settings`, `version`
- **Python API** — `FlashAudio`, `Trainer`, `Predictor`, `Exporter`, `Validator`
- **Speech-to-Text** — Whisper-based transcription with language detection and word timestamps
- **Text-to-Speech** — Mel spectrogram + vocoder synthesis pipeline
- **Voice Cloning** — Speaker embedding extraction and conditioned synthesis
- **Speaker Diarization** — ECAPA-TDNN speaker embedding and clustering
- **Audio Classification** — CNN-based audio event classification (AudioSet labels)
- **Audio Generation** — Diffusion-based music and sound effect generation
- **Source Separation** — U-Net based audio source separation
- **Audio Features** — Mel spectrograms, MFCCs, chromagrams, spectral features
- **Datasets** — LibriSpeech, CommonVoice, AudioSet loaders
- **Solutions** — Transcriber, Narrator, AudioAnalyzer
- **Analytics** — Benchmark, WER, CER, MOS, PESQ metrics
- **LoRA fine-tuning** — Parameter-efficient fine-tuning for audio models
- **Export** — ONNX format
- **CI/CD** — GitHub Actions (lint + test on Python 3.9-3.12)
- **Examples** — 5 runnable example scripts

### Architecture
- HuggingFace Transformers backend (Whisper, wav2vec2)
- torchaudio for audio I/O and transforms
- librosa for feature extraction
- Registry-based component system
- Config-driven training with YAML files
