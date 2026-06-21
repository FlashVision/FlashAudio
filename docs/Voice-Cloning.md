# Voice Cloning

FlashAudio provides speaker embedding extraction and voice cloning.

## Basic usage

```python
from flashaudio.speech import VoiceCloner

cloner = VoiceCloner(device="cuda")

# Extract speaker embedding
embedding = cloner.extract_speaker_embedding("reference.wav")

# Clone the voice
cloner.clone("Say this in the reference voice", embedding, output_path="cloned.wav")
```

## Register speakers

```python
cloner.register_speaker("alice", "alice_reference.wav")
cloner.register_speaker("bob", "bob_reference.wav")

cloner.clone("Hello!", speaker_name="alice", output_path="alice_hello.wav")
cloner.clone("Hello!", speaker_name="bob", output_path="bob_hello.wav")
```

## Speaker similarity

```python
emb1 = cloner.extract_speaker_embedding("speaker1.wav")
emb2 = cloner.extract_speaker_embedding("speaker2.wav")
similarity = cloner.compute_similarity(emb1, emb2)
print(f"Similarity: {similarity:.4f}")
```

## Speaker diarization

```python
from flashaudio.speech import SpeakerDiarizer

diarizer = SpeakerDiarizer(device="cuda", num_speakers=2)
segments = diarizer.diarize("meeting.wav")
for seg in segments:
    print(f"[{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['speaker']}")
```
