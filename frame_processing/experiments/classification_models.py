"""Optional local classifiers with the same classify(bgr) contract as the baseline."""

from __future__ import annotations

from dataclasses import dataclass, field
import gc
import logging
import os
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np

from interface_spec.contracts import FrameType

from ..classification import PaddleLayoutDetector, SERVICE_LAYOUT_MODEL
from ..classification_policy import LayoutBox, LayoutRules, layout_prediction
from .classification_policy import (
    SiglipRules, TEXT_PROMPTS, VISUAL_PROMPTS, YoloLayoutRules, siglip_prediction, yolo_layout_prediction,
)
from .heuristic import LayoutHeuristicClassifier

log = logging.getLogger(__name__)
HEURISTIC = "heuristic"
DOCLAYOUT_MODELS = ("PP-DocLayout-S", SERVICE_LAYOUT_MODEL)
SIGLIP = "siglip2-large"
DOCLAYOUT_YOLO = "doclayout-yolo"
BASELINE_CLASSIFICATION_MODELS = (HEURISTIC, *DOCLAYOUT_MODELS, SIGLIP)
CLASSIFICATION_MODELS = (*BASELINE_CLASSIFICATION_MODELS, DOCLAYOUT_YOLO)
LAYOUT_DETECTORS = (*DOCLAYOUT_MODELS, DOCLAYOUT_YOLO)


@dataclass(frozen=True)
class ClassifierSettings:
    model: str = HEURISTIC
    model_dir: Path | None = None
    device: Literal["gpu", "cpu"] = "gpu"
    cpu_fallback: bool = True
    cpu_threads: int = 4
    layout: LayoutRules = field(default_factory=LayoutRules)
    siglip: SiglipRules = field(default_factory=SiglipRules)
    yolo_layout: YoloLayoutRules = field(default_factory=YoloLayoutRules)

    def __post_init__(self) -> None:
        if self.model not in CLASSIFICATION_MODELS or self.device not in ("cpu", "gpu"):
            raise ValueError("unsupported classifier or device")
        if self.cpu_threads < 1:
            raise ValueError("cpu_threads must be positive")


@dataclass(frozen=True)
class ClassificationResult:
    frame_type: FrameType
    reason: str
    latency_ms: float
    device: str
    scores: dict = field(default_factory=dict)
    boxes: tuple[LayoutBox, ...] = ()
    error: str = ""


def _error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


class LocalClassifier:
    """Load once, keep models local, and preserve the simple FrameType interface."""

    def __init__(self, settings: ClassifierSettings = ClassifierSettings()) -> None:
        self.settings = settings
        self.active_device = "cpu" if settings.model == HEURISTIC else settings.device
        self.errors: list[str] = []
        self._engine = None
        self._allocator_peak_mib: float | None = None
        started = perf_counter()
        if not self._load(self.active_device) and self.active_device == "gpu" and settings.cpu_fallback:
            self._switch_to_cpu()
        self.load_ms = (perf_counter() - started) * 1000

    def _load(self, device: str) -> bool:
        self.active_device = device
        try:
            self._engine = self._create(device)
        except Exception as exc:
            self.errors.append(f"load/{device}: {_error(exc)}")
            log.warning("Could not load frame classifier %s on %s", self.settings.model, device, exc_info=True)
            return False
        return True

    def _create(self, device: str):
        settings = self.settings
        if settings.model == HEURISTIC:
            return LayoutHeuristicClassifier()
        path = settings.model_dir
        if path is None or not path.is_dir():
            raise FileNotFoundError("a local model directory is required")
        if settings.model == SIGLIP:
            return _SiglipEngine(path, device, settings.cpu_threads)
        if settings.model == DOCLAYOUT_YOLO:
            from .yolo_layout import YoloLayoutEngine

            return YoloLayoutEngine(path, device, settings.cpu_threads)
        return PaddleLayoutDetector(settings.model, path, device, settings.cpu_threads)

    def _switch_to_cpu(self) -> None:
        if self._engine is not None:
            engine, self._engine = self._engine, None
            try:
                self._allocator_peak_mib = engine.peak_reserved_mib()
            except Exception:
                log.debug("GPU statistics unavailable during CPU fallback", exc_info=True)
            try:
                engine.release_gpu()
            except Exception as exc:
                self.errors.append(f"release/gpu: {_error(exc)}")
                log.warning("GPU cleanup failed during CPU fallback", exc_info=True)
            finally:
                del engine
                gc.collect()
        self._load("cpu")

    def classify(self, bgr: np.ndarray) -> FrameType:
        return self.classify_detailed(bgr).frame_type

    def classify_detailed(self, bgr: np.ndarray) -> ClassificationResult:
        started = perf_counter()
        if self._engine is None:
            return self._failure(started, "model_unavailable", " | ".join(self.errors))
        try:
            return self._predict(bgr, started)
        except Exception as exc:
            error = _error(exc)
        # Retry only GPU/resource failures, keeping malformed inputs on the same backend.
        log.warning("Frame classifier %s failed on %s: %s",
                    self.settings.model, self.active_device, error)
        if self.active_device == "gpu" and self.settings.cpu_fallback and any(
            token in error.lower() for token in ("cuda", "cudnn", "out of memory", "resourceexhausted")
        ):
            self.errors.append(f"inference/gpu: {error}")
            try:
                self._switch_to_cpu()
                if self._engine is not None:
                    return self._predict(bgr, started, error=error)
            except Exception as exc:
                self.errors.append(f"inference/cpu: {_error(exc)}")
                log.warning("CPU fallback for frame classification failed", exc_info=True)
            error = " | ".join(self.errors)
        return self._failure(started, "inference_failed", error)

    def _failure(self, started: float, reason: str, error: str) -> ClassificationResult:
        return ClassificationResult(FrameType.UNKNOWN, reason, (perf_counter() - started) * 1000,
                                    self.active_device, error=error)

    def _predict(self, bgr: np.ndarray, started: float, error: str = "") -> ClassificationResult:
        boxes = ()
        if self.settings.model == HEURISTIC:
            frame_type, reason, scores = self._engine.classify(bgr), "connected_components", {}
        elif self.settings.model == SIGLIP:
            frame_type, reason, scores = siglip_prediction(self._engine.predict(bgr), self.settings.siglip)
        elif self.settings.model == DOCLAYOUT_YOLO:
            boxes = self._engine.predict(bgr)
            frame_type, reason, scores = yolo_layout_prediction(
                boxes, bgr.shape[1], bgr.shape[0], self.settings.yolo_layout,
            )
        else:
            boxes = self._engine.predict(bgr)
            frame_type, reason, scores = layout_prediction(boxes, bgr.shape[1], bgr.shape[0], self.settings.layout)
        return ClassificationResult(frame_type, reason, (perf_counter() - started) * 1000,
                                    self.active_device, scores, boxes, error)

    def peak_gpu_reserved_mib(self) -> float | None:
        if self.active_device == "gpu" and self._engine is not None:
            return self._engine.peak_reserved_mib()
        return self._allocator_peak_mib


class _SiglipEngine:
    def __init__(self, path: Path, device: str, threads: int) -> None:
        os.environ["HF_HUB_OFFLINE"] = "1"
        import torch
        from transformers import AutoModel, AutoProcessor

        if device == "gpu" and not torch.cuda.is_available():
            raise RuntimeError("PyTorch CUDA device unavailable")
        if device == "cpu":
            torch.set_num_threads(threads)
        self._torch = torch
        self._device = "cuda:0" if device == "gpu" else "cpu"
        self._dtype = torch.float16 if device == "gpu" else torch.float32
        options = {"local_files_only": True, "trust_remote_code": False}
        self._processor = AutoProcessor.from_pretrained(path, use_fast=False, **options)
        self._model = AutoModel.from_pretrained(path, dtype=self._dtype, **options).eval().to(self._device)
        # The prompts never change, so tokenize them once including the 64-token training padding.
        self._text = self._processor(text=list(TEXT_PROMPTS + VISUAL_PROMPTS), padding="max_length",
                                     max_length=64, truncation=True, return_tensors="pt").to(self._device)
        if device == "gpu":
            torch.cuda.synchronize()

    def predict(self, bgr: np.ndarray) -> list[float]:
        pixels = self._processor(images=np.ascontiguousarray(bgr[:, :, ::-1]),
                                 return_tensors="pt").to(self._device, dtype=self._dtype)
        with self._torch.inference_mode():
            return self._model(**pixels, **self._text).logits_per_image[0].float().cpu().tolist()

    def peak_reserved_mib(self) -> float:
        return self._torch.cuda.max_memory_reserved() / 1024**2

    def release_gpu(self) -> None:
        self._model = None
        self._text = None
        self._torch.cuda.empty_cache()
