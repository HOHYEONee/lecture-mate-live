"""Collect every scheduled observation as a lossless PNG and record its timing."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from hashlib import sha256
import json
from math import ceil, isfinite
from pathlib import Path
from time import monotonic_ns, sleep

import cv2
import numpy as np

from interface_spec.contracts import CaptureRegion

from ..capture import CaptureBackend, normalize_bgr

NS_PER_SECOND = 1_000_000_000
DEFAULT_ROOT = Path("experiments")


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


@dataclass(frozen=True, slots=True)
class RawFrame:
    frame_index: int
    timestamp_ms: int
    scheduled_slot: int
    sha256: str

    @property
    def filename(self) -> str:
        return f"frame_{self.frame_index:06d}.png"


@dataclass(frozen=True, slots=True)
class Dataset:
    path: Path
    region: CaptureRegion
    frames: tuple[RawFrame, ...]
    manifest_sha256: str

    def select_frames(self, start: int | None = None, end: int | None = None) -> tuple[RawFrame, ...]:
        first = 1 if start is None else start
        last = len(self.frames) if end is None else end
        if not 1 <= first <= last <= len(self.frames):
            raise ValueError(f"expected 1 <= start frame <= end frame <= {len(self.frames)}")
        return self.frames[first - 1:last]

    def read_frame(self, frame: RawFrame) -> np.ndarray:
        path = self.path / "raw_frames" / frame.filename
        payload = path.read_bytes()
        if sha256(payload).hexdigest() != frame.sha256:
            raise ValueError(f"PNG checksum mismatch: {path}")
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.shape != (self.region.height, self.region.width, 3):
            raise ValueError(f"invalid PNG dimensions or pixels: {path}")
        return image


def load_dataset(path: Path) -> Dataset:
    path = path.resolve(strict=True)
    payload = (path / "capture_metadata.json").read_bytes()
    data = json.loads(payload)
    if data["schema_version"] != 1 or data["fps"] != 1:
        raise ValueError("unsupported capture format")
    if data["status"] != "completed":
        raise ValueError(f"capture is incomplete ({data['status']}): {path}")
    region = CaptureRegion(**data["region"])
    frames = tuple(RawFrame(**entry) for entry in data["frames"])
    if not frames or data["frame_count"] != len(frames):
        raise ValueError("capture manifest has no frames or an inconsistent frame count")
    previous_ms, previous_slot = -1, -1
    for index, frame in enumerate(frames, 1):
        if (frame.frame_index != index or frame.timestamp_ms < max(0, previous_ms)
                or frame.scheduled_slot <= previous_slot
                or frame.scheduled_slot >= data["scheduled_slots"]):
            raise ValueError(f"invalid capture sequence at frame {index}")
        previous_ms, previous_slot = frame.timestamp_ms, frame.scheduled_slot
    return Dataset(path, region, frames, sha256(payload).hexdigest())


def capture_dataset(root: Path, name: str, region: CaptureRegion, seconds: float,
                    backend: CaptureBackend, *,
                    now_ns: Callable[[], int] = monotonic_ns,
                    wait: Callable[[float], None] = sleep) -> Path:
    if (not name or len(name) > 80 or name != name.strip() or name.endswith(".")
            or any(char in '<>:"/\\|?*' or ord(char) < 32 for char in name)):
        raise ValueError("choose a nonempty experiment name valid in a Windows folder")
    if not isfinite(seconds) or seconds <= 0:
        raise ValueError("capture duration must be finite and positive")
    planned = ceil(seconds)
    duration_ns = seconds * NS_PER_SECOND
    started_at = datetime.now().astimezone()
    folder = root / f"{started_at:%Y%m%d_%H%M%S_%f}_{name}"
    folder.mkdir(parents=True, exist_ok=False)
    raw_dir = folder / "raw_frames"
    raw_dir.mkdir()
    frames: list[RawFrame] = []
    status = "failed"
    slot = 0
    start = now_ns()
    try:
        while slot < planned:
            elapsed = now_ns() - start
            delay = slot * NS_PER_SECOND - elapsed
            if delay > 0:
                wait(delay / NS_PER_SECOND)
                continue
            if elapsed >= duration_ns:
                break
            # If the scheduler itself woke up late, use only the most recent slot.
            slot = max(slot, elapsed // NS_PER_SECOND)
            pixels = backend.grab(region)
            timestamp_ms = (now_ns() - start) // 1_000_000
            image = normalize_bgr(pixels)
            if image.shape[:2] != (region.height, region.width):
                raise ValueError("captured image dimensions do not match the region")
            ok, encoded = cv2.imencode(".png", image)
            if not ok:
                raise OSError("PNG encoding failed")
            payload = encoded.tobytes()
            frame = RawFrame(len(frames) + 1, timestamp_ms, slot, sha256(payload).hexdigest())
            (raw_dir / frame.filename).write_bytes(payload)
            frames.append(frame)
            slot += 1
            # Skip slots missed during capture/storage instead of burst-capturing.
            elapsed = now_ns() - start
            slot = max(slot, (elapsed + NS_PER_SECOND - 1) // NS_PER_SECOND)
        if not frames:
            raise ValueError("capture duration ended before the first observation")
        status = "completed"
    except KeyboardInterrupt:
        status = "interrupted"
        raise
    finally:
        write_json(folder / "capture_metadata.json", {
            "schema_version": 1, "name": name, "started_at": started_at.isoformat(),
            "region": asdict(region), "resolution": [region.width, region.height],
            "fps": 1, "requested_seconds": seconds, "status": status,
            "backend": type(backend).__name__, "elapsed_ms": (now_ns() - start) // 1_000_000,
            "scheduled_slots": planned, "missed_slots": planned - len(frames),
            "frame_count": len(frames), "frames": [asdict(frame) for frame in frames],
        })
    return folder
