# Text-to-Speech

FlashAudio provides mel spectrogram + vocoder speech synthesis.

## Basic usage

```python
from flashaudio.speech import TextToSpeech

tts = TextToSpeech(device="cuda")
tts.synthesize("Hello from FlashAudio!", output_path="output.wav")
```

## Get waveform tensor

```python
waveform = tts.synthesize_to_tensor("Hello world", sample_rate=22050)
print(f"Shape: {waveform.shape}")
```

## Get mel spectrogram

```python
mel = tts.get_mel_spectrogram("Visualize this")
from flashaudio.utils.visualize import plot_mel_spectrogram
plot_mel_spectrogram(mel.squeeze(0))
```

## Narration (long-form)

```python
from flashaudio.solutions import Narrator

narrator = Narrator(device="cuda")
narrator.narrate("Long text goes here...", output_path="narration.wav")
narrator.narrate_file("article.txt", output_path="article.wav")
```
