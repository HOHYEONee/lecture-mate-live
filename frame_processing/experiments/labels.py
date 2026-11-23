"""Prepare editable labels and inspect the captured frames side by side."""

from __future__ import annotations

import csv
from io import StringIO
from pathlib import Path

import cv2
import numpy as np

from interface_spec.contracts import FrameDecision

from .capture import Dataset
from .replay import write_csv

CLASSES = (FrameDecision.NEW_GROUP, FrameDecision.NEW_REPRESENTATIVE, FrameDecision.REUSED)
AMBIGUOUS = "ambiguous"
LABEL_COLUMNS = ["frame_index", "timestamp_ms", "label", "notes"]


def ensure_labels(dataset: Dataset) -> Path:
    path = dataset.path / "labels.csv"
    try:
        write_csv(path, LABEL_COLUMNS, (
            {"frame_index": frame.frame_index, "timestamp_ms": frame.timestamp_ms,
             "label": "new_group" if frame.frame_index == 1 else "", "notes": ""}
            for frame in dataset.frames
        ), exclusive=True)
    except FileExistsError:
        pass
    return path


def parse_labels(dataset: Dataset, payload: bytes) -> dict[int, str]:
    reader = csv.DictReader(StringIO(payload.decode("utf-8-sig")))
    if not {"frame_index", "timestamp_ms", "label"}.issubset(reader.fieldnames or []):
        raise ValueError("labels.csv needs frame_index, timestamp_ms and label columns")
    labels = {}
    for line, row in enumerate(reader, 2):
        try:
            index, timestamp_ms = int(row["frame_index"]), int(row["timestamp_ms"])
            label = row["label"].strip()
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"invalid label row at line {line}") from exc
        if not 1 <= index <= len(dataset.frames) or index in labels:
            raise ValueError(f"duplicate or unknown label frame: {index}")
        if timestamp_ms != dataset.frames[index - 1].timestamp_ms:
            raise ValueError(f"label timestamp mismatch at frame {index}")
        if label and label != AMBIGUOUS and label not in CLASSES:
            raise ValueError(f"unknown label {label!r} at frame {index}")
        labels[index] = label
    return labels


def review_frame(dataset: Dataset, index: int, reference: int | None = None, *,
                 start_frame: int = 1, end_frame: int | None = None) -> Path:
    end_frame = len(dataset.frames) if end_frame is None else end_frame
    if not start_frame <= index <= end_frame or (
        reference is not None and not start_frame <= reference <= end_frame
    ):
        raise ValueError(f"review frames must be within {start_frame}–{end_frame}")
    panels = []
    if index > start_frame:
        panels.append(("Previous", index - 1))
    panels.append(("Current", index))
    if reference is not None:
        panels.append(("Reference", reference))
    suffix = f"_from_{start_frame:06d}" if start_frame != 1 else ""
    return write_panels(dataset, panels, dataset.path / "reviews" /
                        f"frame_{index:06d}_ref_{reference or 0:06d}{suffix}.png")


def write_panels(dataset: Dataset, panels: list[tuple[str, int]], output: Path) -> Path:
    images = []
    for title, index in panels:
        if not 1 <= index <= len(dataset.frames):
            raise ValueError(f"frame index out of range: {index}")
        frame = dataset.frames[index - 1]
        image = dataset.read_frame(frame)
        height, width = image.shape[:2]
        scale = min(1, 720 / width, 540 / height)
        if scale < 1:
            image = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))),
                               interpolation=cv2.INTER_AREA)
        panel = np.full((image.shape[0] + 32, max(320, image.shape[1]), 3), 245, np.uint8)
        panel[32:, :image.shape[1]] = image
        cv2.putText(panel, f"{title}: {index} ({frame.timestamp_ms} ms)", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
        images.append(panel)
    ok, encoded = cv2.imencode(".png", np.hstack(images))
    if not ok:
        raise OSError("review image encoding failed")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(encoded.tobytes())
    return output
