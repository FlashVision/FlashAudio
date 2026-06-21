"""I/O utilities for file handling, model downloading, and caching."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional


def ensure_dir(path: str) -> Path:
    """Create a directory if it doesn't exist.

    Args:
        path: Directory path.

    Returns:
        Path object for the directory.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def hash_file(path: str, algorithm: str = "sha256") -> str:
    """Compute the hash of a file.

    Args:
        path: File path.
        algorithm: Hash algorithm ('md5', 'sha256').

    Returns:
        Hex digest string.
    """
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_model(
    model_id: str,
    cache_dir: Optional[str] = None,
    force: bool = False,
) -> str:
    """Download a model from HuggingFace Hub.

    Args:
        model_id: HuggingFace model identifier.
        cache_dir: Local cache directory.
        force: Force re-download even if cached.

    Returns:
        Path to the downloaded model directory.
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "flashaudio", "models")

    cache_path = Path(cache_dir) / model_id.replace("/", "--")

    if cache_path.exists() and not force:
        return str(cache_path)

    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(
            repo_id=model_id,
            local_dir=str(cache_path),
            local_dir_use_symlinks=False,
        )
        return path
    except ImportError:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_id, cache_dir=cache_dir)
        return str(cache_path)


def get_cache_dir() -> Path:
    """Get the FlashAudio cache directory."""
    cache_dir = os.environ.get(
        "FLASHAUDIO_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "flashaudio"),
    )
    return ensure_dir(cache_dir)


def format_size(num_bytes: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"
