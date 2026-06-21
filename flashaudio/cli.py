"""FlashAudio CLI — command-line interface for audio & speech AI tasks."""

import argparse
import sys


def _colored(text, color):
    """Simple ANSI color helper."""
    colors = {"green": "\033[92m", "blue": "\033[94m", "yellow": "\033[93m", "red": "\033[91m", "bold": "\033[1m"}
    return f"{colors.get(color, '')}{text}\033[0m"


def _print_banner():
    print(_colored("FlashAudio", "bold") + f" v{_get_version()}")
    print(_colored("Production-grade Audio & Speech AI", "blue"))
    print()


def _get_version():
    from flashaudio import __version__
    return __version__


def cmd_version(args):
    """Print version info."""
    _print_banner()


def cmd_settings(args):
    """Print system settings and environment info."""
    import torch
    import platform
    import numpy as np

    _print_banner()
    print(_colored("System", "bold"))
    print(f"  Python:         {platform.python_version()}")
    print(f"  OS:             {platform.system()} {platform.release()}")
    print(f"  Machine:        {platform.machine()}")
    print()
    print(_colored("Dependencies", "bold"))
    print(f"  PyTorch:        {torch.__version__}")
    print(f"  NumPy:          {np.__version__}")
    try:
        import torchaudio
        print(f"  torchaudio:     {torchaudio.__version__}")
    except ImportError:
        print("  torchaudio:     Not installed")
    try:
        import transformers
        print(f"  Transformers:   {transformers.__version__}")
    except ImportError:
        print("  Transformers:   Not installed")
    try:
        import librosa
        print(f"  librosa:        {librosa.__version__}")
    except ImportError:
        print("  librosa:        Not installed")
    try:
        import soundfile
        print(f"  soundfile:      {soundfile.__version__}")
    except ImportError:
        print("  soundfile:      Not installed")
    print(f"  CUDA:           {torch.version.cuda or 'Not available'}")
    print(f"  cuDNN:          {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
    print()
    print(_colored("Hardware", "bold"))
    if torch.cuda.is_available():
        print(f"  GPU:            {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"  VRAM:           {mem:.1f} GB")
    else:
        print("  GPU:            None (CPU only)")
    print(f"  CPU cores:      {__import__('os').cpu_count()}")


def cmd_check(args):
    """Verify installation — imports, GPU, and basic pipeline loading."""
    _print_banner()
    errors = []

    print(_colored("Checking installation...", "bold"))
    print()

    try:
        import flashaudio  # noqa: F401
        print(f"  {_colored('✓', 'green')} flashaudio package")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} flashaudio package: {e}")
        errors.append(str(e))

    try:
        from flashaudio.engine import Trainer, Predictor, Exporter, Validator  # noqa: F401
        print(f"  {_colored('✓', 'green')} engine (Trainer, Predictor, Exporter, Validator)")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} engine: {e}")
        errors.append(str(e))

    try:
        from flashaudio.speech import SpeechToText, TextToSpeech, VoiceCloner, SpeakerDiarizer  # noqa: F401
        print(f"  {_colored('✓', 'green')} speech (STT, TTS, VoiceCloner, Diarizer)")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} speech: {e}")
        errors.append(str(e))

    try:
        from flashaudio.audio import AudioClassifier, AudioGenerator, SourceSeparator  # noqa: F401
        print(f"  {_colored('✓', 'green')} audio (Classifier, Generator, Separator)")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} audio: {e}")
        errors.append(str(e))

    try:
        from flashaudio.solutions import Transcriber, Narrator, AudioAnalyzer  # noqa: F401
        print(f"  {_colored('✓', 'green')} solutions (Transcriber, Narrator, AudioAnalyzer)")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} solutions: {e}")
        errors.append(str(e))

    try:
        from flashaudio.analytics import Benchmark  # noqa: F401
        print(f"  {_colored('✓', 'green')} analytics (Benchmark)")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} analytics: {e}")
        errors.append(str(e))

    try:
        import torchaudio  # noqa: F401
        print(f"  {_colored('✓', 'green')} torchaudio ({torchaudio.__version__})")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} torchaudio: {e}")
        errors.append(str(e))

    try:
        import transformers  # noqa: F401
        print(f"  {_colored('✓', 'green')} transformers ({transformers.__version__})")
    except ImportError as e:
        print(f"  {_colored('✗', 'red')} transformers: {e}")
        errors.append(str(e))

    import torch
    if torch.cuda.is_available():
        print(f"  {_colored('✓', 'green')} CUDA ({torch.cuda.get_device_name(0)})")
    else:
        print(f"  {_colored('⚠', 'yellow')} No CUDA GPU (inference will be slow)")

    print()
    if errors:
        print(_colored(f"✗ {len(errors)} check(s) failed", "red"))
        sys.exit(1)
    else:
        print(_colored("✓ All checks passed! FlashAudio is ready.", "green"))


def cmd_transcribe(args):
    """Transcribe audio to text."""
    from flashaudio.speech.stt import SpeechToText

    stt = SpeechToText(model_id=args.model, device=args.device)
    result = stt.transcribe(
        args.audio,
        language=args.language,
        word_timestamps=args.word_timestamps,
    )

    print(f"\n{_colored('Transcription', 'bold')}")
    print(f"  Audio:    {args.audio}")
    print(f"  Model:    {args.model}")
    if result.get("language"):
        print(f"  Language: {result['language']}")
    print()
    print(result["text"])

    if args.word_timestamps and result.get("segments"):
        print(f"\n{_colored('Segments', 'bold')}")
        for seg in result["segments"]:
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            text = seg.get("text", "")
            print(f"  [{start:.2f}s - {end:.2f}s] {text}")


def cmd_speak(args):
    """Synthesize speech from text."""
    from flashaudio.speech.tts import TextToSpeech

    tts = TextToSpeech(device=args.device)
    output_path = tts.synthesize(
        args.text,
        output_path=args.output,
        sample_rate=args.sample_rate,
    )
    print(f"\n{_colored('✓', 'green')} Audio saved to: {output_path}")


def cmd_train(args):
    """Train / fine-tune an audio model."""
    from flashaudio.engine.trainer import Trainer

    if args.config:
        from flashaudio.cfg import load_yaml_config
        cfg = load_yaml_config(args.config)
        print(f"{_colored('Config:', 'bold')} {args.config}")
        trainer = Trainer(config=cfg, device=args.device)
    else:
        if not args.model:
            print(_colored("Error:", "red") + " --model or --config is required")
            sys.exit(1)
        trainer = Trainer(
            model_id=args.model,
            dataset=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
            save_dir=args.save_dir,
            learning_rate=args.lr,
        )

    trainer.train()


def cmd_classify(args):
    """Classify audio events."""
    from flashaudio.audio.classification import AudioClassifier

    classifier = AudioClassifier(device=args.device)
    predictions = classifier.classify(args.audio, top_k=args.top_k)

    print(f"\n{_colored('Classification Results', 'bold')}")
    print(f"  Audio: {args.audio}")
    print()
    for label, score in predictions:
        bar = "█" * int(score * 40)
        print(f"  {label:<30s} {score:.4f} {bar}")


def cmd_export(args):
    """Export model to ONNX."""
    from flashaudio.engine.exporter import Exporter

    exporter = Exporter(model_path=args.model)
    path = exporter.export(output=args.output, fmt=args.format)
    print(f"\n{_colored('✓', 'green')} Exported: {path}")


def cmd_benchmark(args):
    """Benchmark model throughput and latency."""
    from flashaudio.analytics.benchmark import Benchmark

    bench = Benchmark(model_id=args.model, device=args.device, task=args.task)
    results = bench.run(num_runs=args.num_runs)

    print(f"\n{_colored('Benchmark Results', 'bold')}")
    print(f"  Model:          {args.model}")
    print(f"  Task:           {args.task}")
    print(f"  Device:         {args.device}")
    print(f"  Avg latency:    {results['avg_latency_ms']:.1f} ms")
    print(f"  Throughput:     {results['throughput']:.2f} samples/sec")
    print(f"  Memory (MB):    {results['memory_mb']:.0f}")


def main():
    parser = argparse.ArgumentParser(
        prog="flashaudio",
        description="FlashAudio: Production-grade Audio & Speech AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  flashaudio check                                         Verify installation
  flashaudio transcribe --audio recording.wav              Transcribe audio
  flashaudio speak --text "Hello world" --output hello.wav Generate speech
  flashaudio classify --audio sound.wav                    Classify audio
  flashaudio train --config configs/flashaudio_stt.yaml    Train a model
  flashaudio benchmark --model openai/whisper-base         Benchmark speed

Documentation: https://github.com/FlashVision/FlashAudio
""",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # version
    subparsers.add_parser("version", help="Show version info")

    # settings
    subparsers.add_parser("settings", help="Show system settings (Python, PyTorch, CUDA, GPU)")

    # check
    subparsers.add_parser("check", help="Verify installation and run health check")

    # transcribe
    tr_p = subparsers.add_parser("transcribe", help="Transcribe audio to text (STT)")
    tr_p.add_argument("--audio", required=True, help="Path to audio file")
    tr_p.add_argument("--model", default="openai/whisper-base", help="Whisper model ID (default: openai/whisper-base)")
    tr_p.add_argument("--language", default=None, help="Language code (auto-detect if omitted)")
    tr_p.add_argument("--word-timestamps", action="store_true", help="Enable word-level timestamps")
    tr_p.add_argument("--device", default="cuda", help="Device: cuda or cpu (default: cuda)")

    # speak
    sp_p = subparsers.add_parser("speak", help="Synthesize speech from text (TTS)")
    sp_p.add_argument("--text", required=True, help="Text to synthesize")
    sp_p.add_argument("--output", default="output.wav", help="Output audio path (default: output.wav)")
    sp_p.add_argument("--sample-rate", type=int, default=22050, help="Output sample rate (default: 22050)")
    sp_p.add_argument("--device", default="cuda", help="Device (default: cuda)")

    # train
    train_p = subparsers.add_parser("train", help="Train / fine-tune an audio model")
    train_p.add_argument("--config", default=None, help="Path to YAML config")
    train_p.add_argument("--model", default=None, help="HuggingFace model ID")
    train_p.add_argument("--dataset", default=None, help="Path to training dataset")
    train_p.add_argument("--epochs", type=int, default=10, help="Training epochs (default: 10)")
    train_p.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    train_p.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    train_p.add_argument("--device", default="cuda", help="Device (default: cuda)")
    train_p.add_argument("--save-dir", default="workspace/train", help="Output directory")

    # classify
    cls_p = subparsers.add_parser("classify", help="Classify audio events")
    cls_p.add_argument("--audio", required=True, help="Path to audio file")
    cls_p.add_argument("--top-k", type=int, default=5, help="Top-K predictions (default: 5)")
    cls_p.add_argument("--device", default="cuda", help="Device (default: cuda)")

    # export
    exp_p = subparsers.add_parser("export", help="Export model to ONNX")
    exp_p.add_argument("--model", required=True, help="Model path or HuggingFace ID")
    exp_p.add_argument("--output", default="model.onnx", help="Output path")
    exp_p.add_argument("--format", default="onnx", choices=["onnx"], help="Export format (default: onnx)")

    # benchmark
    bench_p = subparsers.add_parser("benchmark", help="Benchmark model speed")
    bench_p.add_argument("--model", default="openai/whisper-base", help="Model ID (default: openai/whisper-base)")
    bench_p.add_argument("--task", default="stt", choices=["stt", "tts", "classification"],
                         help="Task to benchmark (default: stt)")
    bench_p.add_argument("--device", default="cuda", help="Device (default: cuda)")
    bench_p.add_argument("--num-runs", type=int, default=10, help="Number of runs (default: 10)")

    args = parser.parse_args()

    if args.command is None:
        _print_banner()
        parser.print_help()
        sys.exit(0)

    commands = {
        "version": cmd_version,
        "settings": cmd_settings,
        "check": cmd_check,
        "transcribe": cmd_transcribe,
        "speak": cmd_speak,
        "train": cmd_train,
        "classify": cmd_classify,
        "export": cmd_export,
        "benchmark": cmd_benchmark,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
