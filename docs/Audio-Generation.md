# Audio Generation

FlashAudio provides diffusion-based audio generation for music and sound effects.

## Basic usage

```python
from flashaudio.audio import AudioGenerator

generator = AudioGenerator(device="cuda", sample_rate=22050)
generator.generate(duration=5.0, output_path="generated.wav")
```

## With seed for reproducibility

```python
generator.generate(duration=10.0, output_path="music.wav", seed=42)
```

## Get tensor output

```python
waveform = generator.generate(duration=3.0)
print(f"Shape: {waveform.shape}")
```
