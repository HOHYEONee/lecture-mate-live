"""Replay an immutable capture through a fresh production pipeline per threshold."""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from decimal import Decimal, InvalidOperation
from itertools import product
import json
from math import isfinite
from pathlib import Path
import platform

import cv2
import numpy as np

from interface_spec.contracts import CaptureRegion, FrameDecision, FrameType

from ..comparison import SSIMComparator
from ..pipeline import FramePipeline
from ..storage import JPEGFrameStore
from .capture import Dataset, RawFrame, write_json
from .progress import SweepProgress

DEFAULT_THRESHOLDS = tuple(Decimal(index) / 100 for index in range(90, 100))
MAX_CANDIDATES = 2001


def parse_decimal(text: str) -> Decimal:
    try:
        value = Decimal(text.strip())
    except InvalidOperation as exc:
        raise ValueError(f"invalid threshold number: {text!r}") from exc
    if not value.is_finite() or not -1 <= value <= 1:
        raise ValueError("threshold numbers must be finite and in [-1, 1]")
    if value != value.quantize(Decimal("0.000001")):
        raise ValueError("threshold numbers support up to six decimal places")
    return value


def threshold_text(value: Decimal) -> str:
    if not value:
        return "0"
    return format(value.normalize(), "f")


def parse_thresholds(explicit: str | None = None,
                     limits: tuple[str, str, str] | None = None) -> tuple[Decimal, ...]:
    if explicit is not None and limits is not None:
        raise ValueError("use either a threshold list or a range")
    if limits is not None:
        start, stop, step = map(parse_decimal, limits)
        if step <= 0 or start > stop:
            raise ValueError("range requires START <= STOP and a positive STEP")
        count = int((stop - start) // step) + 1
        if count > MAX_CANDIDATES:
            raise ValueError(f"at most {MAX_CANDIDATES} thresholds are supported")
        return tuple(start + index * step for index in range(count))
    if explicit is None:
        return DEFAULT_THRESHOLDS
    values = tuple(map(parse_decimal, explicit.split(",")))
    if len(values) > MAX_CANDIDATES or len(set(values)) != len(values):
        raise ValueError(f"provide at most {MAX_CANDIDATES} distinct thresholds")
    return values


@dataclass(frozen=True, slots=True)
class ThresholdPair:
    group: Decimal
    representative: Decimal | None = None

    @property
    def name(self) -> str:
        if self.representative is None:
            return threshold_text(self.group)
        return f"g{threshold_text(self.group)}_r{threshold_text(self.representative)}"

    def columns(self) -> dict[str, str]:
        representative = self.group if self.representative is None else self.representative
        return {"group_threshold": threshold_text(self.group),
                "representative_threshold": threshold_text(representative)}


def parse_threshold_pairs(groups: str, representatives: str) -> tuple[ThresholdPair, ...]:
    group_values, representative_values = parse_thresholds(groups), parse_thresholds(representatives)
    if len(group_values) * len(representative_values) > MAX_CANDIDATES:
        raise ValueError(f"at most {MAX_CANDIDATES} threshold pairs are supported")
    return tuple(ThresholdPair(group, representative) for group, representative
                 in product(group_values, representative_values))


def write_csv(path: Path, columns: list[str], rows, *, exclusive: bool = False) -> None:
    with path.open("x" if exclusive else "w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


@dataclass(frozen=True, slots=True)
class Prediction:
    frame_index: int
    timestamp_ms: int
    threshold: str
    ssim_pc: float | None
    ssim_rc: float | None
    decision: FrameDecision
    group_id: str
    representative_id: str
    representative_index: int
    stored: bool


class ReplayInput:
    def __init__(self) -> None:
        self.pixels: np.ndarray
        self.timestamp_ms = 0

    def grab(self, region: CaptureRegion) -> np.ndarray:
        return self.pixels

    def now_ms(self) -> int:
        return self.timestamp_ms


class ComparisonCache:
    """Share comparison work for the same pixels, never P/C/R state or decisions."""

    def __init__(self, max_side: int | None) -> None:
        self.scorer = SSIMComparator(0, max_side)
        self.current: np.ndarray | None = None
        self.scores: dict[int, float] = {}

    def reset(self) -> None:
        self.current = None
        self.scores.clear()

    def prepare(self, pixels: np.ndarray) -> np.ndarray:
        if self.current is None:
            self.current = self.scorer.prepare(pixels)
            self.current.flags.writeable = False
        return self.current

    def score(self, a: np.ndarray, b: np.ndarray) -> float:
        # All requests share this observation's C. P/R arrays remain alive in the pipelines.
        key = id(a)
        if key not in self.scores:
            self.scores[key] = self.scorer.score(a, b)
        return self.scores[key]


class RecordingComparator(SSIMComparator):
    def __init__(self, thresholds: ThresholdPair, comparison: ComparisonCache) -> None:
        representative = thresholds.representative
        super().__init__(float(thresholds.group), comparison.scorer.max_side,
                         representative_threshold=None if representative is None else float(representative))
        self.comparison = comparison
        self.scores: list[float] = []

    def prepare(self, bgr: np.ndarray) -> np.ndarray:
        return self.comparison.prepare(bgr)

    def score(self, a: np.ndarray, b: np.ndarray) -> float:
        score = self.comparison.score(a, b)
        self.scores.append(score)
        return score


class ReferenceStore:
    def save(self, frame_id: str, bgr: np.ndarray) -> str:
        # Satisfy the pipeline's path contract without writing another image.
        return f"frames/{frame_id}.jpg"


class Unclassified:
    def classify(self, bgr: np.ndarray) -> FrameType:
        return FrameType.UNKNOWN


class ReplaySession:
    def __init__(self, dataset: Dataset, thresholds: ThresholdPair, source: ReplayInput,
                 comparison: ComparisonCache, jpeg_dir: Path | None) -> None:
        self.threshold = thresholds.name
        self.comparator = RecordingComparator(thresholds, comparison)
        store = ReferenceStore() if jpeg_dir is None else JPEGFrameStore(jpeg_dir.resolve())
        self.pipeline = FramePipeline(dataset.path.name, dataset.region, source, source, store,
                                      self.comparator, Unclassified())
        self.predictions: list[Prediction] = []
        self.representative_index = 0

    def observe(self, frame: RawFrame) -> None:
        self.comparator.scores.clear()
        record = self.pipeline.periodic()
        # Production scores P/C first and calls R/C only when P/C is similar.
        pc, rc = (self.comparator.scores + [None, None])[:2]
        stored = record.decision != FrameDecision.REUSED
        if stored:
            self.representative_index = frame.frame_index
        self.predictions.append(Prediction(frame.frame_index, record.timestamp_ms, self.threshold,
                                           pc, rc, record.decision, record.group_id, record.frame_id,
                                           self.representative_index, stored))


def _replay_batch(dataset: Dataset, frames: tuple[RawFrame, ...], jobs: list[tuple[ThresholdPair, Path | None]],
                  max_side: int | None, *, progress: SweepProgress | None = None,
                  threshold_offset: int = 0) -> list[ReplaySession]:
    source = ReplayInput()
    comparison = ComparisonCache(max_side)
    sessions = [ReplaySession(dataset, threshold, source, comparison, directory) for threshold, directory in jobs]
    for frame_number, frame in enumerate(frames, 1):
        source.pixels = dataset.read_frame(frame)
        source.timestamp_ms = frame.timestamp_ms
        comparison.reset()
        for threshold_index, session in enumerate(sessions, threshold_offset + 1):
            session.observe(frame)
            if progress is not None:
                progress.advance(session.threshold, threshold_index, frame_number)
    return sessions


def replay(dataset: Dataset, threshold: Decimal | ThresholdPair, *, max_side: int | None = 960,
           jpeg_dir: Path | None = None, start_frame: int | None = None,
           end_frame: int | None = None) -> list[Prediction]:
    frames = dataset.select_frames(start_frame, end_frame)
    pair = threshold if isinstance(threshold, ThresholdPair) else ThresholdPair(threshold)
    return _replay_batch(dataset, frames, [(pair, jpeg_dir)], max_side)[0].predictions


def summarize(predictions: list[Prediction], jpeg_bytes: int | None = None) -> dict:
    counts = Counter(row.decision for row in predictions)
    total = len(predictions)
    groups = counts[FrameDecision.NEW_GROUP]
    added = counts[FrameDecision.NEW_REPRESENTATIVE]
    reused = counts[FrameDecision.REUSED]
    stored = groups + added
    return {"threshold": predictions[0].threshold, "observed": total, "new_groups": groups,
            "new_representatives": added, "reused": reused, "stored": stored,
            "storage_reduction": reused / total, "representatives_per_group": stored / groups,
            "representative_jpeg_bytes": jpeg_bytes}


def run_sweep(dataset: Dataset, thresholds: tuple[Decimal | ThresholdPair, ...], *,
              max_side: int | None = 960, write_representatives: bool = False,
              start_frame: int | None = None, end_frame: int | None = None,
              progress: SweepProgress | None = None) -> Path:
    candidates = tuple(t if isinstance(t, ThresholdPair) else ThresholdPair(t) for t in thresholds)
    if not candidates or len(set(candidates)) != len(candidates) or len(candidates) > MAX_CANDIDATES:
        raise ValueError("sweep requires distinct thresholds")
    dual = candidates[0].representative is not None
    if any((t.representative is not None) != dual for t in candidates):
        raise ValueError("use either single thresholds or threshold pairs in one sweep")
    for candidate in candidates:
        for value in candidate.columns().values():
            parse_decimal(value)
    SSIMComparator(float(candidates[0].group), max_side)  # Validate before creating output.
    frames = dataset.select_frames(start_frame, end_frame)
    run = dataset.path / "results" / f"sweep_{datetime.now():%Y%m%d_%H%M%S_%f}"
    run.mkdir(parents=True, exist_ok=False)
    if progress is not None:
        progress.start(len(candidates), len(frames))
    summaries = []
    # Bound the number of simultaneously live representatives for very large grids.
    batch_size = 32
    for offset in range(0, len(candidates), batch_size):
        batch = candidates[offset:offset + batch_size]
        directories = [run / f"threshold_{threshold.name}" for threshold in batch]
        for directory in directories:
            directory.mkdir()
        jobs = [(threshold, directory if write_representatives else None)
                for threshold, directory in zip(batch, directories)]
        sessions = _replay_batch(dataset, frames, jobs, max_side, progress=progress, threshold_offset=offset)
        for directory, session, candidate in zip(directories, sessions, batch):
            predictions = session.predictions
            columns = candidate.columns() if dual else {}
            write_csv(directory / "decisions.csv", [field.name for field in fields(Prediction)] + list(columns),
                      ({**asdict(row), **columns} for row in predictions))
            jpeg_bytes = None
            if write_representatives:
                jpeg_bytes = sum(path.stat().st_size for path in (directory / "frames").glob("*.jpg"))
            summaries.append({**summarize(predictions, jpeg_bytes), **columns})
    write_csv(run / "summary.csv", list(summaries[0]), summaries)
    metadata = {
        "schema_version": 3 if dual else 2, "capture_manifest_sha256": dataset.manifest_sha256,
        "start_frame": frames[0].frame_index, "end_frame": frames[-1].frame_index,
        "frame_count": len(frames), "thresholds": [t.name for t in candidates],
        "comparison_max_side": max_side, "write_representatives": write_representatives,
        "representative_jpeg_quality": 90 if write_representatives else None,
        "versions": {"python": platform.python_version(), "opencv": cv2.__version__,
                     "numpy": np.__version__},
    }
    if dual:
        metadata["threshold_pairs"] = [t.columns() for t in candidates]
    write_json(run / "sweep_metadata.json", metadata)
    if progress is not None:
        progress.finish()
    return run


@dataclass(frozen=True, slots=True)
class SweepSelection:
    candidates: tuple[ThresholdPair, ...]
    frames: tuple[RawFrame, ...]

    @property
    def thresholds(self) -> tuple[str, ...]:
        return tuple(candidate.name for candidate in self.candidates)

    @property
    def start(self) -> int:
        return self.frames[0].frame_index

    @property
    def end(self) -> int:
        return self.frames[-1].frame_index


def load_sweep(dataset: Dataset, run: Path) -> SweepSelection:
    metadata = json.loads((run / "sweep_metadata.json").read_bytes())
    version = metadata["schema_version"]
    if version not in (1, 2, 3) or metadata["capture_manifest_sha256"] != dataset.manifest_sha256:
        raise ValueError("sweep was not produced from this capture manifest")
    frames = (dataset.frames if version == 1 else
              dataset.select_frames(metadata["start_frame"], metadata["end_frame"]))
    if metadata["frame_count"] != len(frames):
        raise ValueError("inconsistent sweep frame count")
    names = tuple(metadata["thresholds"])
    if version == 3:
        pairs = tuple(ThresholdPair(parse_decimal(row["group_threshold"]), parse_decimal(row["representative_threshold"]))
                      for row in metadata["threshold_pairs"])
    else:
        pairs = tuple(ThresholdPair(parse_decimal(name)) for name in names)
    if not names or len(set(names)) != len(names) or tuple(t.name for t in pairs) != names:
        raise ValueError("invalid sweep threshold list")
    return SweepSelection(pairs, frames)


def load_predictions(selection: SweepSelection, run: Path, threshold: str) -> list[Prediction]:
    candidate = next((t for t in selection.candidates if t.name == threshold), None)
    if candidate is None:
        raise ValueError(f"unknown sweep threshold: {threshold}")
    path = run / f"threshold_{threshold}" / "decisions.csv"
    if not path.is_file() and candidate.representative is None:
        # Some older results display two decimal places in their folder names.
        alias = f"{candidate.group:.2f}"
        if Decimal(alias) == candidate.group and alias != threshold:
            path = run / f"threshold_{alias}" / "decisions.csv"
    columns = candidate.columns() if candidate.representative is not None else {}
    rows = read_csv(path)
    if len(rows) != len(selection.frames):
        raise ValueError(f"inconsistent prediction count: {path}")
    predictions = []
    for frame, row in zip(selection.frames, rows):
        index, timestamp = int(row["frame_index"]), int(row["timestamp_ms"])
        decision = FrameDecision(row["decision"])
        stored = decision != FrameDecision.REUSED
        representative_index = int(row["representative_index"])
        pc, rc = (float(row[key]) if row[key] else None for key in ("ssim_pc", "ssim_rc"))
        if (index != frame.frame_index or timestamp != frame.timestamp_ms or row["threshold"] != threshold
                or decision == FrameDecision.QUESTION_SNAPSHOT or row["stored"] != str(stored)
                or not row["group_id"] or not row["representative_id"]
                or any(row.get(key) != value for key, value in columns.items())
                or any(score is not None and not isfinite(score) for score in (pc, rc))):
            raise ValueError(f"inconsistent prediction at frame {frame.frame_index}: {path}")
        if not predictions and (decision != FrameDecision.NEW_GROUP or pc is not None or rc is not None):
            raise ValueError(f"first prediction must create a group: {path}")
        expected_index = index if stored else predictions[-1].representative_index
        if representative_index != expected_index:
            raise ValueError(f"inconsistent active representative at frame {index}: {path}")
        predictions.append(Prediction(index, timestamp, threshold, pc, rc, decision, row["group_id"],
                                      row["representative_id"], representative_index, stored))
    return predictions
