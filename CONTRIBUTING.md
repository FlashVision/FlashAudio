# Contributing to FlashAudio

Thanks for your interest in contributing! Here's how to get started.

## Setup

```bash
git clone https://github.com/FlashVision/FlashAudio.git
cd FlashAudio
pip install -e ".[dev,all]"
```

## Development Workflow

1. Create a branch: `git checkout -b feature/your-feature`
2. Make changes
3. Run lint: `ruff check flashaudio/`
4. Run tests: `pytest tests/ -v`
5. Commit and push
6. Open a Pull Request

## Code Style

- We use [ruff](https://docs.astral.sh/ruff/) for linting (line length: 120)
- Type hints are encouraged
- Docstrings for all public functions (Google style)
- No hardcoded file paths — use relative or configurable paths

## Adding a New Pipeline

1. Create `flashaudio/speech/your_pipeline.py` or `flashaudio/audio/your_pipeline.py`
2. Follow the existing pattern: accept `model_id` or config kwargs
3. Implement the main method (e.g., `transcribe()`, `synthesize()`, `classify()`)
4. Register in `flashaudio/registry.py`
5. Add to the appropriate `__init__.py`

## Adding a New Solution

1. Create `flashaudio/solutions/your_solution.py`
2. Follow the existing pattern: accept `model_id` or `predictor`
3. Implement the main method
4. Add to `flashaudio/solutions/__init__.py`

## Commit Messages

Use clear, descriptive messages:
- `Add speaker diarization pipeline`
- `Fix mel spectrogram normalization`
- `Update README with TTS examples`

## Reporting Issues

- Use GitHub Issues
- Include: Python version, PyTorch version, GPU info, error traceback
- Run `flashaudio settings` and paste the output

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
