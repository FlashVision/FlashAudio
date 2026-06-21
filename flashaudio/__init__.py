"""FlashAudio — Production-grade Audio & Speech AI."""

__version__ = "1.0.0"

from flashaudio.models.flashaudio_model import FlashAudio
from flashaudio.models.lora import apply_lora, merge_lora_weights
from flashaudio.engine.trainer import Trainer
from flashaudio.engine.validator import Validator
from flashaudio.engine.predictor import Predictor
from flashaudio.engine.exporter import Exporter
from flashaudio.cfg import get_config
from flashaudio.solutions import Transcriber, Narrator, AudioAnalyzer
from flashaudio.analytics import Benchmark

__all__ = [
    "FlashAudio", "Trainer", "Predictor", "Validator", "Exporter",
    "apply_lora", "merge_lora_weights", "get_config",
    "Transcriber", "Narrator", "AudioAnalyzer",
    "Benchmark",
    "__version__",
]
