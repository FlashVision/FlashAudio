"""FlashAudio — Speech-to-Text Example

Demonstrates Whisper-based transcription with language detection,
word timestamps, and batch processing.

Usage:
    python examples/speech_to_text.py --audio recording.wav
    python examples/speech_to_text.py --audio recording.wav --model openai/whisper-small
"""

import argparse

import torch


def main():
    parser = argparse.ArgumentParser(description="FlashAudio STT Example")
    parser.add_argument("--audio", default=None, help="Path to audio file")
    parser.add_argument("--model", default="openai/whisper-base", help="Whisper model ID")
    parser.add_argument("--language", default=None, help="Language code (auto-detect if omitted)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from flashaudio.speech.stt import SpeechToText

    stt = SpeechToText(model_id=args.model, device=args.device)

    if args.audio:
        print(f"\nTranscribing: {args.audio}")
        print(f"Model: {args.model}")
        print(f"Device: {args.device}\n")

        result = stt.transcribe(args.audio, language=args.language, word_timestamps=True)

        print(f"Text: {result['text']}")
        if result.get("language"):
            print(f"Language: {result['language']}")

        if result.get("segments"):
            print("\nSegments:")
            for seg in result["segments"]:
                print(f"  [{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['text']}")
    else:
        print("FlashAudio Speech-to-Text Example")
        print("=" * 40)
        print("\nUsing dummy audio for demonstration...\n")

        dummy_audio = torch.randn(16000 * 3)
        print(f"Dummy audio shape: {dummy_audio.shape}")
        print(f"Duration: {dummy_audio.shape[0] / 16000:.1f}s")

        from flashaudio.analytics.metrics import compute_wer, compute_cer

        ref = "the quick brown fox jumps over the lazy dog"
        hyp = "the quik brown fox jumped over a lazy dog"
        print(f"\nWER Example:")
        print(f"  Reference:  {ref}")
        print(f"  Hypothesis: {hyp}")
        print(f"  WER: {compute_wer(ref, hyp):.4f}")
        print(f"  CER: {compute_cer(ref, hyp):.4f}")


if __name__ == "__main__":
    main()
