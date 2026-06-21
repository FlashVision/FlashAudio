"""Configuration management for FlashAudio.

Supports loading from YAML files and programmatic construction.
Provides typed access to model, data, training, and audio parameters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class ModelConfig:
    """Model configuration."""
    model_id: str = "openai/whisper-base"
    task: str = "stt"
    torch_dtype: str = "float32"
    max_length: int = 448
    num_labels: int = 527


@dataclass
class DataConfig:
    """Data configuration."""
    dataset_path: str = ""
    dataset_name: str = "librispeech"
    sample_rate: int = 16000
    max_duration: float = 30.0
    val_split: float = 0.1
    num_workers: int = 4


@dataclass
class TrainConfig:
    """Training configuration."""
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler: str = "cosine"
    max_grad_norm: float = 1.0
    save_dir: str = "workspace/train"
    save_steps: int = 500
    eval_steps: int = 250
    logging_steps: int = 10
    amp: bool = True
    gradient_checkpointing: bool = False


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    sample_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 80
    win_length: int = 1024
    fmin: float = 0.0
    fmax: Optional[float] = 8000.0


@dataclass
class FlashAudioConfig:
    """Top-level FlashAudio configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    device: str = "cuda"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to flat dictionary."""
        result = {}
        for section_name in ["model", "data", "train", "audio"]:
            section = getattr(self, section_name)
            for k, v in section.__dict__.items():
                result[f"{section_name}.{k}"] = v
        result["device"] = self.device
        return result


def load_yaml_config(path: str) -> FlashAudioConfig:
    """Load configuration from a YAML file.

    Args:
        path: Path to the YAML config file.

    Returns:
        FlashAudioConfig populated from the YAML.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    cfg = FlashAudioConfig()

    if "model" in raw:
        for k, v in raw["model"].items():
            if hasattr(cfg.model, k):
                setattr(cfg.model, k, v)

    if "data" in raw:
        for k, v in raw["data"].items():
            if hasattr(cfg.data, k):
                setattr(cfg.data, k, v)

    if "train" in raw:
        for k, v in raw["train"].items():
            if hasattr(cfg.train, k):
                setattr(cfg.train, k, v)

    if "audio" in raw:
        for k, v in raw["audio"].items():
            if hasattr(cfg.audio, k):
                setattr(cfg.audio, k, v)

    if "device" in raw:
        cfg.device = raw["device"]

    return cfg


def get_config(path: Optional[str] = None, **overrides) -> FlashAudioConfig:
    """Get configuration, optionally loading from YAML with overrides.

    Args:
        path: Optional YAML config path.
        **overrides: Key-value overrides in "section.key" format.

    Returns:
        FlashAudioConfig instance.
    """
    if path and os.path.isfile(path):
        cfg = load_yaml_config(path)
    else:
        cfg = FlashAudioConfig()

    for key, value in overrides.items():
        if "." in key:
            section, attr = key.split(".", 1)
            section_obj = getattr(cfg, section, None)
            if section_obj and hasattr(section_obj, attr):
                setattr(section_obj, attr, value)
        elif hasattr(cfg, key):
            setattr(cfg, key, value)

    return cfg
