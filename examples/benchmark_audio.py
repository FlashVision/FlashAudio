"""FlashAudio — Benchmark Example

Benchmarks model latency, throughput, and memory usage for
different audio tasks.

Usage:
    python examples/benchmark_audio.py --task stt
    python examples/benchmark_audio.py --task classification --num-runs 20
"""

import argparse

import torch


def main():
    parser = argparse.ArgumentParser(description="FlashAudio Benchmark Example")
    parser.add_argument("--task", default="classification", choices=["stt", "tts", "classification"],
                        help="Task to benchmark")
    parser.add_argument("--num-runs", type=int, default=10, help="Number of benchmark runs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("FlashAudio Benchmark Example")
    print("=" * 40)
    print(f"Task:    {args.task}")
    print(f"Device:  {args.device}")
    print(f"Runs:    {args.num_runs}\n")

    if args.task == "classification":
        from flashaudio.models.architectures.audio_classifier import AudioClassifierModel
        model = AudioClassifierModel().to(args.device)
        model.eval()

        dummy = torch.randn(1, 16000 * 5, device=args.device)

        import time
        latencies = []
        for _ in range(2):
            with torch.no_grad():
                model(dummy)

        for _ in range(args.num_runs):
            start = time.perf_counter()
            with torch.no_grad():
                model(dummy)
            if args.device == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

        avg = sum(latencies) / len(latencies)
        print(f"Results:")
        print(f"  Avg latency:  {avg:.2f} ms")
        print(f"  Min latency:  {min(latencies):.2f} ms")
        print(f"  Max latency:  {max(latencies):.2f} ms")
        print(f"  Throughput:   {1000/avg:.2f} samples/sec")

    elif args.task == "tts":
        from flashaudio.models.architectures.tts_model import TTSModel
        model = TTSModel().to(args.device)
        model.eval()

        text_ids = torch.randint(0, 256, (1, 30), device=args.device)

        import time
        latencies = []
        for _ in range(2):
            with torch.no_grad():
                model(text_ids, max_steps=50)

        for _ in range(args.num_runs):
            start = time.perf_counter()
            with torch.no_grad():
                model(text_ids, max_steps=50)
            if args.device == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)

        avg = sum(latencies) / len(latencies)
        print(f"Results:")
        print(f"  Avg latency:  {avg:.2f} ms")
        print(f"  Min latency:  {min(latencies):.2f} ms")
        print(f"  Throughput:   {1000/avg:.2f} samples/sec")

    else:
        print(f"STT benchmark requires Whisper model download.")
        print(f"Run: flashaudio benchmark --model openai/whisper-base --task stt")

    from flashaudio.analytics.metrics import compute_wer, compute_cer
    print("\nMetrics Example:")
    ref = "the quick brown fox jumps over the lazy dog"
    hyp = "the quik brown fox jumped over a lazy dog"
    print(f"  WER: {compute_wer(ref, hyp):.4f}")
    print(f"  CER: {compute_cer(ref, hyp):.4f}")


if __name__ == "__main__":
    main()
