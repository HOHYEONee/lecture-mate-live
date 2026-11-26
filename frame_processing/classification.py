"""Local GPU classification for the frame service."""

from __future__ import annotations

import os
import sys
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from typing import Protocol

import numpy as np

from interface_spec.contracts import FrameType

from .classification_policy import LayoutBox, LayoutRules, layout_prediction

SERVICE_LAYOUT_MODEL = "PP-DocLayout_plus-L"
MODEL_DIR_ENV = "CLUE_FRAME_MODEL_DIR"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / f"{SERVICE_LAYOUT_MODEL}_infer"
MODEL_FILES = ("inference.json", "inference.yml", "inference.pdiparams")


class FrameClassifier(Protocol):
    def classify(self, bgr: np.ndarray) -> FrameType: ...


class UnrecoverableClassificationError(RuntimeError):
    """The current GPU session can no longer classify frames."""


def preload_torch_on_windows() -> None:
    if sys.platform == "win32" and find_spec("torch") is not None:
        try:
            import_module("torch")
        except Exception as exc:
            raise RuntimeError(f"PyTorch could not load before Paddle: {exc}") from exc


class PaddleLayoutDetector:
    """Local Paddle layout detector shared with the offline experiments."""

    def __init__(self, name: str, path: Path, device: str, threads: int = 4) -> None:
        path = path.resolve()
        missing = [filename for filename in MODEL_FILES if not (path / filename).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {name} model files in {path}: {', '.join(missing)}")

        self._device = "gpu:0" if device == "gpu" else device
        self._gpu = self._device.startswith("gpu:")
        if self._gpu:
            preload_torch_on_windows()

        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"
        import paddle

        if self._gpu:
            if not paddle.is_compiled_with_cuda():
                raise RuntimeError("PaddlePaddle CUDA device unavailable (CPU build installed; requires gpu:0)")
            index = int(self._device.split(":", 1)[1])
            if not 0 <= index < paddle.device.cuda.device_count():
                raise RuntimeError(f"PaddlePaddle CUDA device unavailable (requires {self._device})")
            paddle.device.set_device(self._device)
        from paddleocr import LayoutDetection

        self._paddle = paddle
        self._detector = LayoutDetection(model_name=name, model_dir=str(path),
                                         device=self._device if self._gpu else "cpu",
                                         enable_mkldnn=False, cpu_threads=threads)
        if self._gpu:
            paddle.device.synchronize(self._device)

    def predict(self, bgr: np.ndarray) -> tuple[LayoutBox, ...]:
        result, = self._detector.predict(bgr, batch_size=1)
        if self._gpu:
            self._paddle.device.synchronize(self._device)
        return tuple(LayoutBox(box["label"], float(box["score"]), tuple(map(float, box["coordinate"])))
                     for box in result["boxes"])

    def peak_reserved_mib(self) -> float:
        return self._paddle.device.cuda.max_memory_reserved() / 1024**2

    def release_gpu(self) -> None:
        self._detector = None
        self._paddle.device.cuda.empty_cache()


class PlusLClassifier:
    def __init__(self, device: str = "gpu") -> None:
        model_dir = Path(os.environ.get(MODEL_DIR_ENV) or DEFAULT_MODEL_DIR)
        self._detector = PaddleLayoutDetector(SERVICE_LAYOUT_MODEL, model_dir, device)
        self._rules = LayoutRules()

    def warm_up(self, width: int, height: int) -> None:
        self.classify(np.zeros((height, width, 3), dtype=np.uint8))

    def classify(self, bgr: np.ndarray) -> FrameType:
        try:
            boxes = self._detector.predict(bgr)
        except Exception as exc:
            if any(message in str(exc).lower() for message in (
                "illegal memory access", "device-side assert", "device is lost", "device lost",
                "cudaerrorillegaladdress", "cudaerrorassert", "cuda error(700)", "cuda error(719)",
            )):
                raise UnrecoverableClassificationError(str(exc)) from exc
            raise
        return layout_prediction(boxes, bgr.shape[1], bgr.shape[0], self._rules)[0]
