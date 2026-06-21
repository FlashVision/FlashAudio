"""Benchmarking engine for FlashAudio models.

Measures latency, throughput, memory usage, and optionally accuracy
metrics for STT, TTS, and classification tasks.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import torch


class Benchmark:
    """Audio model benchmark.

    Args:
        model_id: HuggingFace model ID.
        device: Device for inference.
        task: Task to benchmark ('stt', 'tts', 'classification').
    """

    def __init__(
        self,
        model_id: str = "openai/whisper-base",
        device: str = "cuda",
        task: str = "stt",
    ):
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.task = task

    def run(
        self,
        num_runs: int = 10,
        warmup_runs: int = 2,
        audio_duration: float = 5.0,
        sample_rate: int = 16000,
    ) -> Dict[str, Any]:
        """Run the benchmark.

        Args:
            num_runs: Number of benchmark iterations.
            warmup_runs: Number of warmup iterations.
            audio_duration: Duration of test audio in seconds.
            sample_rate: Sample rate for test audio.

        Returns:
            Dictionary with timing and memory results.
        """
        dummy_audio = torch.randn(1, int(audio_duration * sample_rate))

        pipeline = self._get_pipeline()

        for _ in range(warmup_runs):
            self._run_once(pipeline, dummy_audio)

        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        latencies = []
        for _ in range(num_runs):
            start = time.perf_counter()
            self._run_once(pipeline, dummy_audio)
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        memory_mb = 0.0
        if self.device == "cuda" and torch.cuda.is_available():
            memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        throughput = 1000.0 / avg_latency if avg_latency > 0 else 0.0

        return {
            "model": self.model_id,
            "task": self.task,
            "device": self.device,
            "num_runs": num_runs,
            "audio_duration_sec": audio_duration,
            "avg_latency_ms": round(avg_latency, 2),
            "min_latency_ms": round(min_latency, 2),
            "max_latency_ms": round(max_latency, 2),
            "throughput": round(throughput, 4),
            "memory_mb": round(memory_mb, 2),
            "latencies": [round(lat, 2) for lat in latencies],
            "real_time_factor": round(avg_latency / (audio_duration * 1000), 4),
        }

    def _get_pipeline(self):
        """Get the appropriate pipeline for the task."""
        if self.task == "stt":
            from flashaudio.models.architectures.whisper import WhisperWrapper
            return WhisperWrapper(model_id=self.model_id)
        elif self.task == "tts":
            from flashaudio.models.architectures.tts_model import TTSModel
            model = TTSModel().to(self.device)
            model.eval()
            return model
        elif self.task == "classification":
            from flashaudio.models.architectures.audio_classifier import AudioClassifierModel
            model = AudioClassifierModel().to(self.device)
            model.eval()
            return model
        else:
            raise ValueError(f"Unknown task: {self.task}")

    def _run_once(self, pipeline, audio: torch.Tensor):
        """Run a single benchmark iteration."""
        audio = audio.to(self.device)

        with torch.no_grad():
            if self.task == "stt":
                if hasattr(pipeline, "transcribe"):
                    pipeline.transcribe(audio.squeeze(0), sample_rate=16000)
                else:
                    pipeline(audio)
            elif self.task == "tts":
                text_ids = torch.randint(0, 256, (1, 20), device=self.device)
                pipeline(text_ids, max_steps=50)
            elif self.task == "classification":
                pipeline(audio)

    def compare(
        self,
        model_ids: List[str],
        num_runs: int = 10,
    ) -> List[Dict[str, Any]]:
        """Compare multiple models on the same benchmark.

        Args:
            model_ids: List of model IDs to benchmark.
            num_runs: Number of runs per model.

        Returns:
            List of benchmark results.
        """
        results = []
        for model_id in model_ids:
            self.model_id = model_id
            result = self.run(num_runs=num_runs)
            results.append(result)
        return results
