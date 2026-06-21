# FAQ

## What models are supported?

FlashAudio supports any HuggingFace Whisper model for STT. For TTS, it includes a built-in Tacotron2-style model. Audio classification uses a CNN architecture trained on AudioSet labels.

## Do I need a GPU?

No, FlashAudio works on CPU, but GPU inference is significantly faster. Use `--device cpu` for CPU-only mode.

## How do I fine-tune Whisper?

```bash
flashaudio train --config configs/flashaudio_stt.yaml
```

Or in Python:
```python
from flashaudio import Trainer
trainer = Trainer(model_id="openai/whisper-base", dataset="data/librispeech")
trainer.train()
```

## What audio formats are supported?

WAV, FLAC, MP3, OGG, and any format supported by torchaudio/soundfile.

## How do I export a model?

```bash
flashaudio export --model path/to/checkpoint.pt --output model.onnx
```

## How do I compute WER?

```python
from flashaudio.analytics.metrics import compute_wer
wer = compute_wer("the quick brown fox", "the quik brown fox")
print(f"WER: {wer:.4f}")
```

## Can I use LoRA fine-tuning?

```python
from flashaudio import FlashAudio, apply_lora
model = FlashAudio(task="stt", model_id="openai/whisper-base")
apply_lora(model.model, rank=8, alpha=16)
```
