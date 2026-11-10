"""JPEG storage with Unicode paths and exclusive creation of immutable images."""

from __future__ import annotations

from pathlib import Path
import re

import cv2
import numpy as np


class JPEGFrameStore:
    def __init__(self, session_dir: str | Path, quality: int = 90) -> None:
        self.session_dir = Path(session_dir)
        if not self.session_dir.is_absolute() or not self.session_dir.is_dir():
            raise ValueError("session_dir must be an existing absolute directory from the app")
        if not 1 <= quality <= 100:
            raise ValueError("JPEG quality must be in [1, 100]")
        self.frames_dir = self.session_dir / "frames"
        self.frames_dir.mkdir(exist_ok=True)
        if self.frames_dir.is_symlink() or self.frames_dir.resolve().parent != self.session_dir.resolve():
            raise ValueError("frames directory must stay inside session_dir")
        self.quality = quality

    def save(self, frame_id: str, bgr: np.ndarray) -> str:
        if re.fullmatch(r"[A-Za-z0-9_-]+", frame_id) is None:
            raise ValueError("invalid frame_id")
        ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            raise OSError("JPEG encoding failed")
        path = self.frames_dir / f"{frame_id}.jpg"
        # Path.open avoids cv2.imwrite's platform-dependent Unicode path handling.
        # Never overwrite a past representative, even on an ID collision.
        created = False
        try:
            with path.open("xb") as stream:
                created = True
                payload = encoded.tobytes()
                if stream.write(payload) != len(payload):
                    raise OSError("incomplete JPEG write")
        except BaseException:
            if created:
                path.unlink(missing_ok=True)
            raise
        return f"frames/{frame_id}.jpg"
