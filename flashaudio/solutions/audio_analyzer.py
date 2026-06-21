"""AudioAnalyzer solution — comprehensive audio analysis and reporting.

Combines classification, feature extraction, diarization, and
transcription into a single analysis report.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch


class AudioAnalyzer:
    """Comprehensive audio analysis solution.

    Analyzes audio files by combining multiple pipelines:
    transcription, classification, feature extraction, and
    speaker diarization into a unified report.

    Args:
        device: Device for inference.
        sample_rate: Expected sample rate.
    """

    def __init__(
        self,
        device: str = "cuda",
        sample_rate: int = 16000,
    ):
        self.device = device
        self.sample_rate = sample_rate

    def analyze(
        self,
        audio_path: str,
        transcribe: bool = True,
        classify: bool = True,
        extract_features: bool = True,
        diarize: bool = False,
    ) -> Dict[str, Any]:
        """Run comprehensive analysis on an audio file.

        Args:
            audio_path: Path to audio file.
            transcribe: Whether to include transcription.
            classify: Whether to include classification.
            extract_features: Whether to include feature extraction.
            diarize: Whether to include speaker diarization.

        Returns:
            Analysis report dictionary.
        """
        from flashaudio.data.audio_utils import load_audio, get_audio_info

        info = get_audio_info(audio_path)
        waveform, sr = load_audio(audio_path, sample_rate=self.sample_rate)

        report: Dict[str, Any] = {
            "file": str(audio_path),
            "info": info,
            "statistics": self._compute_statistics(waveform, sr),
        }

        if transcribe:
            try:
                from flashaudio.speech.stt import SpeechToText
                stt = SpeechToText(device=self.device)
                report["transcription"] = stt.transcribe(audio_path)
            except Exception as e:
                report["transcription"] = {"error": str(e)}

        if classify:
            try:
                from flashaudio.audio.classification import AudioClassifier
                classifier = AudioClassifier(device=self.device)
                report["classification"] = classifier.classify(waveform.squeeze(0), top_k=10)
            except Exception as e:
                report["classification"] = {"error": str(e)}

        if extract_features:
            try:
                from flashaudio.audio.features import AudioFeatureExtractor
                extractor = AudioFeatureExtractor(sample_rate=sr)
                features = extractor.extract_all(waveform)
                report["features"] = {
                    k: {"shape": list(v.shape) if hasattr(v, "shape") else "scalar"}
                    for k, v in features.items()
                }
            except Exception as e:
                report["features"] = {"error": str(e)}

        if diarize:
            try:
                from flashaudio.speech.diarization import SpeakerDiarizer
                diarizer = SpeakerDiarizer(device=self.device)
                report["diarization"] = diarizer.diarize(audio_path)
            except Exception as e:
                report["diarization"] = {"error": str(e)}

        return report

    def _compute_statistics(self, waveform: torch.Tensor, sample_rate: int) -> Dict[str, float]:
        """Compute basic audio statistics."""
        if waveform.dim() == 2:
            w = waveform.squeeze(0)
        else:
            w = waveform

        return {
            "duration_seconds": round(w.shape[-1] / sample_rate, 3),
            "sample_rate": sample_rate,
            "num_samples": w.shape[-1],
            "peak_amplitude": round(w.abs().max().item(), 6),
            "rms_amplitude": round(w.pow(2).mean().sqrt().item(), 6),
            "dynamic_range_db": round(
                20 * torch.log10(w.abs().max() / w.abs()[w.abs() > 0].min() + 1e-10).item(), 2
            ) if (w.abs() > 0).any() else 0.0,
            "zero_crossings": int((w[:-1] * w[1:] < 0).sum().item()),
            "silence_ratio": round((w.abs() < 0.01).float().mean().item(), 4),
        }

    def analyze_batch(
        self,
        audio_paths: List[str],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Analyze multiple audio files.

        Args:
            audio_paths: List of audio file paths.
            **kwargs: Arguments passed to analyze().

        Returns:
            List of analysis reports.
        """
        return [self.analyze(path, **kwargs) for path in audio_paths]

    def compare(
        self,
        audio_path_1: str,
        audio_path_2: str,
    ) -> Dict[str, Any]:
        """Compare two audio files.

        Args:
            audio_path_1: First audio file.
            audio_path_2: Second audio file.

        Returns:
            Comparison report.
        """
        report_1 = self.analyze(audio_path_1, diarize=False)
        report_2 = self.analyze(audio_path_2, diarize=False)

        return {
            "file_1": report_1,
            "file_2": report_2,
            "duration_diff": (
                report_1["statistics"]["duration_seconds"] -
                report_2["statistics"]["duration_seconds"]
            ),
            "rms_diff": (
                report_1["statistics"]["rms_amplitude"] -
                report_2["statistics"]["rms_amplitude"]
            ),
        }

    def summary(self, report: Dict[str, Any]) -> str:
        """Generate a human-readable summary from an analysis report.

        Args:
            report: Analysis report from analyze().

        Returns:
            Formatted summary string.
        """
        lines = [
            f"Audio Analysis: {report.get('file', 'unknown')}",
            f"{'=' * 50}",
            f"  Duration:      {report['statistics']['duration_seconds']:.2f}s",
            f"  Sample rate:   {report['statistics']['sample_rate']} Hz",
            f"  Peak:          {report['statistics']['peak_amplitude']:.4f}",
            f"  RMS:           {report['statistics']['rms_amplitude']:.4f}",
            f"  Silence:       {report['statistics']['silence_ratio']*100:.1f}%",
        ]

        if "transcription" in report and isinstance(report["transcription"], dict):
            text = report["transcription"].get("text", "")
            if text:
                preview = text[:100] + "..." if len(text) > 100 else text
                lines.append(f"\n  Transcription: {preview}")

        if "classification" in report and isinstance(report["classification"], list):
            lines.append("\n  Top classifications:")
            for label, score in report["classification"][:5]:
                lines.append(f"    {label}: {score:.4f}")

        return "\n".join(lines)
