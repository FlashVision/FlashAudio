# Speech-to-Text

FlashAudio provides Whisper-based speech recognition with language detection and word-level timestamps.

## Basic usage

```python
from flashaudio.speech import SpeechToText

stt = SpeechToText(model_id="openai/whisper-base", device="cuda")
result = stt.transcribe("audio.wav")
print(result["text"])
```

## Language detection

```python
language = stt.detect_language("audio.wav")
print(f"Detected: {language}")
```

## Word timestamps

```python
result = stt.transcribe("audio.wav", word_timestamps=True)
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}")
```

## Batch transcription

```python
results = stt.transcribe_batch(["file1.wav", "file2.wav", "file3.wav"])
for r in results:
    print(f"{r['file']}: {r['text']}")
```

## Supported models

- `openai/whisper-tiny` — 39M params, fastest
- `openai/whisper-base` — 74M params, good balance
- `openai/whisper-small` — 244M params
- `openai/whisper-medium` — 769M params
- `openai/whisper-large-v3` — 1.55B params, best accuracy
