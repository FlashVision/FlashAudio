"""FlashAudio — Audio Classification Example

Demonstrates audio event classification and sliding-window analysis.

Usage:
    python examples/audio_classify.py --audio sound.wav
"""

import argparse

import torch


def main():
    parser = argparse.ArgumentParser(description="FlashAudio Classification Example")
    parser.add_argument("--audio", default=None, help="Path to audio file")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K predictions")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from flashaudio.audio.classification import AudioClassifier

    print("FlashAudio Audio Classification Example")
    print("=" * 40)
    print(f"Device: {args.device}\n")

    classifier = AudioClassifier(device=args.device)

    if args.audio:
        print(f"Classifying: {args.audio}\n")
        predictions = classifier.classify(args.audio, top_k=args.top_k)

        print("Top predictions:")
        for label, score in predictions:
            bar = "█" * int(score * 40)
            print(f"  {label:<25s} {score:.4f} {bar}")

        print("\nSliding-window analysis:")
        segments = classifier.classify_segments(args.audio, segment_duration=2.0, top_k=3)
        for seg in segments:
            print(f"\n  [{seg['start']:.1f}s - {seg['end']:.1f}s]")
            for label, score in seg["predictions"]:
                print(f"    {label}: {score:.4f}")
    else:
        print("Using dummy audio for demonstration...\n")
        dummy_audio = torch.randn(16000 * 5)
        print(f"Dummy audio: {dummy_audio.shape[0] / 16000:.1f}s\n")

        predictions = classifier.classify(dummy_audio, top_k=args.top_k)
        print("Top predictions (random audio):")
        for label, score in predictions:
            bar = "█" * int(score * 40)
            print(f"  {label:<25s} {score:.4f} {bar}")


if __name__ == "__main__":
    main()
