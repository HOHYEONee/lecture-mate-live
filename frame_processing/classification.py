"""Local GPU classification for the frame service."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

import numpy as np

from interface_spec.contracts import FrameType

from .classification_policy import LayoutBox, LayoutRules, layout_prediction

SERVICE_LAYOUT_MODEL = "PP-DocLayout_plus-L"
MODEL_DIR_ENV = "CLUE_FRAME_MODEL_DIR"
DEFAULT_MODEL_DIR = Path("models") / f"{SERVICE_LAYOUT_MODEL}_infer"
MODEL_FILES = ("inference.json", "inference.yml", "inference.pdiparams")


class FrameClassifier(Protocol):
    def classify(self, bgr: np.ndarray) -> FrameType: ...


class UnrecoverableClassificationError(RuntimeError):
    """The current GPU session can no longer classify frames."""


class PaddleLayoutDetector:
    """Local Paddle layout detector shared with the offline experiments."""

    def __init__(self, name: str, path: Path, device: str, threads: int = 4) -> None:
        path = path.resolve()
        missing = [filename for filename in MODEL_FILES if not (path / filename).is_file()]
        if missing:
            raise FileNotFoundError(f"Missing {name} model files in {path}: {', '.join(missing)}")

        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"
        import paddle

        self._gpu = device == "gpu"
        if self._gpu:
            if not (paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count()):
                raise RuntimeError("PaddlePaddle CUDA device unavailable (requires gpu:0)")
            paddle.device.set_device("gpu:0")
        from paddleocr import LayoutDetection

        self._paddle = paddle
        self._detector = LayoutDetection(model_name=name, model_dir=str(path),
                                         device="gpu:0" if self._gpu else "cpu",
                                         enable_mkldnn=False, cpu_threads=threads)
        if self._gpu:
            paddle.device.synchronize("gpu:0")

    def predict(self, bgr: np.ndarray) -> tuple[LayoutBox, ...]:
        result, = self._detector.predict(bgr, batch_size=1)
        if self._gpu:
            self._paddle.device.synchronize("gpu:0")
        return tuple(LayoutBox(box["label"], float(box["score"]), tuple(map(float, box["coordinate"])))
                     for box in result["boxes"])

    def peak_reserved_mib(self) -> float:
        return self._paddle.device.cuda.max_memory_reserved() / 1024**2

    def release_gpu(self) -> None:
        self._detector = None
        self._paddle.device.cuda.empty_cache()


class PlusLClassifier:
    def __init__(self) -> None:
        model_dir = Path(os.environ.get(MODEL_DIR_ENV) or DEFAULT_MODEL_DIR)
        self._detector = PaddleLayoutDetector(SERVICE_LAYOUT_MODEL, model_dir, "gpu")
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
