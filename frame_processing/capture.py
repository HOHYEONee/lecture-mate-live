"""Capture adapters. The pipeline normalizes raw pixels to owned uint8 BGR."""

from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

from interface_spec.contracts import CaptureRegion


class CaptureBackend(Protocol):
    def grab(self, region: CaptureRegion) -> np.ndarray: ...


def normalize_bgr(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.size == 0:
        raise ValueError("capture must be a nonempty uint8 array")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("capture must have gray, BGR, or BGRA channels")
    return np.array(image, copy=True, order="C")


class MSSCapture:
    """Open/close native resources in the calling thread, including snapshots.

    No cross-thread MSS handle is retained. Cursor capture remains disabled.
    Region coordinates are physical pixels; negative monitor origins are valid.
    """

    def grab(self, region: CaptureRegion) -> np.ndarray:
        import mss

        with mss.mss() as screen:
            shot = screen.grab({
                "left": region.left, "top": region.top,
                "width": region.width, "height": region.height,
            })
            # Keep conversion/copy outside grab so the pipeline timestamps first.
            # numpy retains the ScreenShot's backing byte buffer after close().
            return np.asarray(shot)
