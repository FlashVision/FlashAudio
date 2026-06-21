"""FlashAudio — Voice Cloning Example

Demonstrates speaker embedding extraction and voice cloning.

Usage:
    python examples/voice_cloning.py --reference speaker.wav --text "Clone this voice"
"""

import argparse

import torch


def main():
    parser = argparse.ArgumentParser(description="FlashAudio Voice Cloning Example")
    parser.add_argument("--reference", default=None, help="Reference audio for voice cloning")
    parser.add_argument("--text", default="Hello, this is a voice cloning demonstration.",
                        help="Text to synthesize in cloned voice")
    parser.add_argument("--output", default="cloned_output.wav", help="Output path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from flashaudio.speech.voice_clone import VoiceCloner

    print("FlashAudio Voice Cloning Example")
    print("=" * 40)
    print(f"Device: {args.device}\n")

    cloner = VoiceCloner(device=args.device)

    if args.reference:
        print(f"Extracting speaker embedding from: {args.reference}")
        embedding = cloner.extract_speaker_embedding(args.reference)
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding norm: {embedding.norm():.4f}")

        print(f"\nCloning voice with text: {args.text}")
        output = cloner.clone(args.text, speaker_embedding=embedding, output_path=args.output)
        print(f"✓ Cloned audio saved to: {output}")
    else:
        print("Using dummy speaker embedding for demonstration...\n")

        dummy_embedding = torch.randn(192)
        dummy_embedding = torch.nn.functional.normalize(dummy_embedding, p=2, dim=0)
        print(f"Embedding shape: {dummy_embedding.shape}")

        print(f"\nSynthesizing with speaker conditioning...")
        waveform = cloner.synthesize_with_embedding(args.text, dummy_embedding)
        print(f"Output waveform shape: {waveform.shape}")
        print(f"Duration: {waveform.shape[-1] / 22050:.2f}s")

        emb1 = torch.randn(192)
        emb2 = torch.randn(192)
        similarity = cloner.compute_similarity(emb1, emb2)
        print(f"\nSpeaker similarity (random): {similarity:.4f}")


if __name__ == "__main__":
    main()
