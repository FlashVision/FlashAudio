# Quick Start

## Transcribe audio (STT)

```python
from flashaudio import FlashAudio

model = FlashAudio(task="stt", model_id="openai/whisper-base")
result = model.transcribe("recording.wav")
print(result["text"])
```

## Synthesize speech (TTS)

```python
from flashaudio import FlashAudio

model = FlashAudio(task="tts")
model.synthesize("Hello, world!", output_path="hello.wav")
```

## Classify audio

```python
from flashaudio import FlashAudio

model = FlashAudio(task="classification")
labels = model.classify("sound.wav")
for label, score in labels:
    print(f"  {label}: {score:.3f}")
```

## CLI usage

```bash
flashaudio transcribe --audio recording.wav
flashaudio speak --text "Hello world" --output hello.wav
flashaudio classify --audio sound.wav
flashaudio benchmark --model openai/whisper-base
```
