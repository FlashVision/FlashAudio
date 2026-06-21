"""Model export engine for FlashAudio.

Exports trained models to ONNX format for deployment.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn


class Exporter:
    """Export audio models to deployment formats.

    Args:
        model_path: Path to a saved model checkpoint or HuggingFace model ID.
        model: Optional pre-loaded model.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[nn.Module] = None,
    ):
        self.model_path = model_path
        self._model = model

    @property
    def model(self) -> nn.Module:
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self):
        """Load model from checkpoint or HuggingFace."""
        path = Path(self.model_path)

        if path.exists() and path.suffix == ".pt":
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
            if "model_state_dict" in checkpoint:
                from flashaudio.models.architectures.audio_classifier import AudioClassifierModel
                self._model = AudioClassifierModel()
                self._model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self._model = checkpoint
        else:
            from flashaudio.models.flashaudio_model import FlashAudio
            flash = FlashAudio(model_id=self.model_path, device="cpu")
            self._model = flash.model

    def export(
        self,
        output: str = "model.onnx",
        fmt: str = "onnx",
        input_shape: Optional[Tuple[int, ...]] = None,
        opset_version: int = 14,
        dynamic_axes: bool = True,
    ) -> str:
        """Export the model.

        Args:
            output: Output file path.
            fmt: Export format ('onnx').
            input_shape: Shape of dummy input tensor.
            opset_version: ONNX opset version.
            dynamic_axes: Whether to use dynamic axes for variable-length inputs.

        Returns:
            Path to the exported model file.
        """
        if fmt == "onnx":
            return self._export_onnx(output, input_shape, opset_version, dynamic_axes)
        else:
            raise ValueError(f"Unsupported export format: {fmt}. Supported: onnx")

    def _export_onnx(
        self,
        output: str,
        input_shape: Optional[Tuple[int, ...]] = None,
        opset_version: int = 14,
        dynamic_axes: bool = True,
    ) -> str:
        """Export model to ONNX format."""
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model = self.model
        model.eval()

        if input_shape is None:
            input_shape = (1, 16000 * 3)

        dummy_input = torch.randn(*input_shape)

        axes = None
        if dynamic_axes:
            axes = {"input": {0: "batch", 1: "time"}, "output": {0: "batch"}}

        torch.onnx.export(
            model,
            dummy_input,
            str(output_path),
            opset_version=opset_version,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes=axes,
        )

        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  Model exported to ONNX: {output_path} ({size_mb:.1f} MB)")
        return str(output_path)

    def validate_export(self, output: str, input_shape: Optional[Tuple[int, ...]] = None) -> bool:
        """Validate an exported ONNX model.

        Args:
            output: Path to the ONNX model.
            input_shape: Shape to test with.

        Returns:
            True if validation passes.
        """
        try:
            import onnx
            import onnxruntime as ort
            import numpy as np

            onnx_model = onnx.load(output)
            onnx.checker.check_model(onnx_model)

            if input_shape is None:
                input_shape = (1, 16000 * 3)

            session = ort.InferenceSession(output)
            dummy = np.random.randn(*input_shape).astype(np.float32)
            result = session.run(None, {"input": dummy})

            print(f"  ONNX validation passed. Output shape: {result[0].shape}")
            return True

        except ImportError:
            print("  ONNX validation skipped (install onnx and onnxruntime)")
            return False
        except Exception as e:
            print(f"  ONNX validation failed: {e}")
            return False
