"""SSIM preprocessing and the four P/C/R decisions, independent of capture/I/O."""

from __future__ import annotations

from math import isfinite

import cv2
import numpy as np

from interface_spec.contracts import FrameDecision


def decide_periodic(
    s_pc: float | None, s_rc: float | None, threshold: float,
) -> FrameDecision:
    if not isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError("SSIM threshold must be finite and in [-1, 1]")
    if s_pc is None or s_rc is None:
        return FrameDecision.NEW_GROUP
    if not isfinite(s_pc) or not isfinite(s_rc):
        raise ValueError("SSIM scores must be finite")
    # P/C takes priority, including Case 4 (P differs while R is similar).
    if s_pc < threshold:
        return FrameDecision.NEW_GROUP
    if s_rc < threshold:
        return FrameDecision.NEW_REPRESENTATIVE
    return FrameDecision.REUSED


class SSIMComparator:
    """Color SSIM, data_range=255, 7x7 uniform window and sample covariance.

    Scaling is optional (default longest side 960). Color channels are retained
    so annotations that only change color can affect similarity. Tiny regions
    are edge-padded to 7 pixels. Threshold must be explicitly chosen by caller.
    """

    def __init__(self, threshold: float, max_side: int | None = 960) -> None:
        decide_periodic(None, None, threshold)
        if max_side is not None and max_side < 7:
            raise ValueError("max_side must be at least 7 or None")
        self.threshold = threshold
        self.max_side = max_side

    def prepare(self, bgr: np.ndarray) -> np.ndarray:
        image = bgr
        h, w = image.shape[:2]
        if self.max_side is not None and max(h, w) > self.max_side:
            scale = self.max_side / max(h, w)
            image = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))),
                               interpolation=cv2.INTER_AREA)
        h, w = image.shape[:2]
        if min(h, w) < 7:
            image = cv2.copyMakeBorder(image, 0, max(0, 7 - h), 0, max(0, 7 - w),
                                       cv2.BORDER_REPLICATE)
        return image.copy()

    def score(self, a: np.ndarray, b: np.ndarray) -> float:
        if a.shape != b.shape:
            raise ValueError("SSIM inputs must have identical shapes")
        # Same uniform-window SSIM formula as skimage's explicit reference
        # settings. boxFilter handles all three channels in one native pass.
        # Float64 avoids cancellation for almost-uniform lecture backgrounds.
        x, y = a.astype(np.float64), b.astype(np.float64)
        def mean(image: np.ndarray) -> np.ndarray:
            return cv2.boxFilter(image, -1, (7, 7), normalize=True,
                                 borderType=cv2.BORDER_REFLECT)
        ux, uy = mean(x), mean(y)
        ux2, uy2, uxy = ux*ux, uy*uy, ux*uy
        covariance_normalization = 49 / 48
        vx = covariance_normalization * (mean(x*x) - ux2)
        vy = covariance_normalization * (mean(y*y) - uy2)
        vxy = covariance_normalization * (mean(x*y) - uxy)
        c1, c2 = (.01*255)**2, (.03*255)**2
        score_map = ((2*uxy+c1) * (2*vxy+c2)) / ((ux2+uy2+c1) * (vx+vy+c2))
        # Match the reference's exclusion of a three-pixel filter border.
        return float(score_map[3:-3, 3:-3].mean(dtype=np.float64))
