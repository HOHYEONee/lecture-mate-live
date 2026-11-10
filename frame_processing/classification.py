"""Replaceable lightweight layout baseline; not an OCR or validated slide model."""

from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

from interface_spec.contracts import FrameType


class FrameClassifier(Protocol):
    def classify(self, bgr: np.ndarray) -> FrameType: ...


class LayoutHeuristicClassifier:
    """Provisional evidence from connected components, with Visual priority.

    Large connected diagrams/tables count as Visual; small repeated components
    aligned on multiple rows count as Text. Blank/unrecognized layouts abstain.
    Small isolated logos do not establish Visual. This deliberately cannot
    establish semantic meaning or reliably handle photos/formulas/all layouts.
    """

    def classify(self, bgr: np.ndarray) -> FrameType:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        if min(h, w) < 16 or float(gray.std()) < 2:
            return FrameType.UNKNOWN
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        if np.count_nonzero(mask) > mask.size / 2:
            mask = cv2.bitwise_not(mask)
        _, _, stats, centers = cv2.connectedComponentsWithStats(mask, connectivity=8)
        text_rows: set[int] = set()
        text_count = 0
        for (_, _, cw, ch, area), (_, cy) in zip(stats[1:], centers[1:]):
            box_fraction = cw * ch / (h * w)
            if cw >= .18 * w and ch >= .16 * h and box_fraction >= .04 and area >= 20:
                return FrameType.VISUAL
            if 2 <= cw <= .10 * w and 3 <= ch <= .08 * h and area >= 3:
                text_count += 1
                text_rows.add(int(cy / max(4, .04 * h)))
        if text_count >= 12 and len(text_rows) >= 2:
            return FrameType.TEXT
        return FrameType.UNKNOWN
