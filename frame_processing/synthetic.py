"""Original generated slides; no lecture media, desktop, models, or network."""

from __future__ import annotations

import cv2
import numpy as np

from interface_spec.contracts import CaptureRegion


def text_slide(width: int = 1280, height: int = 720) -> np.ndarray:
    image = np.full((height, width, 3), 250, np.uint8)
    scale = min(width / 640, height / 360)
    cv2.putText(image, "CLUE: synthetic linear models", (round(20*scale), round(45*scale)),
                cv2.FONT_HERSHEY_SIMPLEX, .8*scale, (30, 40, 50), max(1, round(scale)))
    for i, line in enumerate(["y = ax + b", "Fit parameters using observations",
                              "Compare predictions and measured values", "Update model with small steps"]):
        cv2.putText(image, line, (round(20*scale), round((100+i*45)*scale)),
                    cv2.FONT_HERSHEY_SIMPLEX, .6*scale, (30, 40, 50), max(1, round(scale)))
    return image


def graph_slide(width: int = 1280, height: int = 720) -> np.ndarray:
    image = text_slide(width, height)
    cv2.rectangle(image, (width//2, height//2), (width-30, height-30), (20, 20, 20), 3)
    cv2.line(image, (width//2, height-30), (width-40, height//2+20), (180, 60, 30), 5)
    return image


class SyntheticCapture:
    def grab(self, region: CaptureRegion) -> np.ndarray:
        return text_slide(region.width, region.height)
