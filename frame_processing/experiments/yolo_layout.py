"""Optional DocStructBench SDK adapter for the verified official local checkpoint."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from hashlib import file_digest
import os
from pathlib import Path
import sys

import numpy as np

from ..classification_policy import LayoutBox

MODEL_FILENAME = "doclayout_yolo_docstructbench_imgsz1024.pt"
MODEL_SHA256 = "9a2ee0220fe3d9ad31b47e1d9f1282f46959a54e4618fce9cffcc9715b8286e2"
MODEL_CLASSES = (
    "title", "plain text", "abandon", "figure", "figure_caption", "table",
    "table_caption", "table_footnote", "isolate_formula", "formula_caption",
)
INPUT_SIZE = 1024
DETECTION_FLOOR = .20

_offline = ContextVar("doclayout_yolo_offline", default=False)
_socket_events = {"socket.connect", "socket.getaddrinfo", "socket.gethostbyname",
                  "socket.gethostbyaddr", "socket.sendto", "socket.sendmsg"}


def _deny_network(event, _args) -> None:
    if event in _socket_events and _offline.get():
        raise OSError(f"network disabled for DocLayout-YOLO: {event}")


# The upstream SDK probes connectivity at import. The context confines the
# guard to this adapter on the calling thread, leaving other modules alone.
sys.addaudithook(_deny_network)


@contextmanager
def _local_only():
    token = _offline.set(True)
    try:
        yield
    finally:
        _offline.reset(token)


class YoloLayoutEngine:
    def __init__(self, path: Path, device: str, threads: int) -> None:
        checkpoint = (path / MODEL_FILENAME).resolve(strict=True)
        # The upstream .pt loader deserializes Python objects. Only accept the
        # bytes published for this specific checkpoint before invoking it.
        with checkpoint.open("rb") as stream:
            if file_digest(stream, "sha256").hexdigest() != MODEL_SHA256:
                raise ValueError("DocLayout-YOLO checkpoint SHA-256 does not match the official release")
        os.environ["YOLO_AUTOINSTALL"] = "false"
        with _local_only():
            import torch
            from doclayout_yolo import YOLOv10

            self._gpu = device == "gpu"
            if self._gpu and not torch.cuda.is_available():
                raise RuntimeError("PyTorch CUDA device unavailable")
            if not self._gpu:
                torch.set_num_threads(threads)
            self._torch = torch
            self._device = "cuda:0" if self._gpu else "cpu"
            self._model = YOLOv10(str(checkpoint)).to(self._device)
            if tuple(self._model.names[index] for index in range(len(self._model.names))) != MODEL_CLASSES:
                raise ValueError("unexpected DocStructBench class map")
            if self._gpu:
                torch.cuda.synchronize()

    def predict(self, bgr: np.ndarray) -> tuple[LayoutBox, ...]:
        with _local_only():
            result, = self._model.predict(
                source=bgr, imgsz=INPUT_SIZE, conf=DETECTION_FLOOR, device=self._device,
                half=self._gpu, batch=1, verbose=False, save=False,
            )
        # This checkpoint returns axis-aligned detections with xyxy, score and class ID.
        return tuple(LayoutBox(MODEL_CLASSES[int(category)], score, (left, top, right, bottom))
                     for left, top, right, bottom, score, category in result.boxes.data.cpu().tolist())

    def peak_reserved_mib(self) -> float:
        return self._torch.cuda.max_memory_reserved() / 1024**2

    def release_gpu(self) -> None:
        self._model = None
        self._torch.cuda.empty_cache()
