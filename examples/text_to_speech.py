"""FlashAudio — Text-to-Speech Example

Demonstrates speech synthesis from text using the TTS pipeline.

Usage:
    python examples/text_to_speech.py --text "Hello world" --output hello.wav
"""

import argparse

import torch


def main():
    parser = argparse.ArgumentParser(description="FlashAudio TTS Example")
    parser.add_argument("--text", default="Hello from FlashAudio! This is a text to speech demo.",
                        help="Text to synthesize")
    parser.add_argument("--output", default="tts_output.wav", help="Output audio path")
    parser.add_argument("--sample-rate", type=int, default=22050, help="Output sample rate")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from flashaudio.speech.tts import TextToSpeech

    print("FlashAudio Text-to-Speech Example")
    print("=" * 40)
    print(f"Text:        {args.text}")
    print(f"Output:      {args.output}")
    print(f"Sample rate: {args.sample_rate}")
    print(f"Device:      {args.device}\n")

    tts = TextToSpeech(device=args.device, sample_rate=args.sample_rate)

    print("Generating mel spectrogram...")
    mel = tts.get_mel_spectrogram(args.text)
    print(f"Mel shape: {mel.shape}")

    print("Synthesizing waveform...")
    output_path = tts.synthesize(args.text, output_path=args.output, sample_rate=args.sample_rate)
    print(f"\n✓ Audio saved to: {output_path}")

    waveform = tts.synthesize_to_tensor(args.text, sample_rate=args.sample_rate)
    duration = waveform.shape[-1] / args.sample_rate
    print(f"Waveform shape: {waveform.shape}")
    print(f"Duration: {duration:.2f}s")


if __name__ == "__main__":
    main()
