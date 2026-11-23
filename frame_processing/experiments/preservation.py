"""Review whether an annotated piece of information survives in any saved frame."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import re

from .capture import Dataset
from .replay import Prediction, write_csv

EVENT_COLUMNS = ["event_id", "first_frame", "last_frame", "description"]
CHECK_IDENTIFIERS = ["dataset_sha256", "event_sha256"]
CHECK_COLUMNS = ["event_id", "frame_index", "verdict", "notes", "raw_image", "representative_jpeg",
                 *CHECK_IDENTIFIERS]
EVENT_RESULT_COLUMNS = ["threshold", "event_id", "description", "first_frame", "last_frame",
                        "onset_stored", "candidate_frames", "unreviewed_candidates", "status",
                        "reason", "first_confirmed_frame", "confirmed_delay_ms"]


@dataclass(frozen=True, slots=True)
class Event:
    event_id: str
    first_frame: int
    last_frame: int
    description: str

    @property
    def fingerprint(self) -> str:
        definition = json.dumps(asdict(self), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return sha256(definition.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class LabelRequirements:
    """Operational review defaults, configurable for each experiment."""

    min_labeled_frames: int = 20
    min_labeling_ratio: float = 0.5
    min_labels_per_binary_class: int = 5

    def __post_init__(self) -> None:
        if self.min_labeled_frames < 1 or self.min_labels_per_binary_class < 1:
            raise ValueError("minimum label counts must be positive")
        if not 0 <= self.min_labeling_ratio <= 1:
            raise ValueError("minimum labeling ratio must be in [0, 1]")


def ensure_events(dataset: Dataset) -> Path:
    path = dataset.path / "events.csv"
    try:
        write_csv(path, EVENT_COLUMNS, (), exclusive=True)
    except FileExistsError:
        pass
    return path


def _rows(payload: bytes, required: list[str], name: str) -> csv.DictReader:
    reader = csv.DictReader(StringIO(payload.decode("utf-8-sig")))
    if not set(required).issubset(reader.fieldnames or []):
        raise ValueError(f"missing required columns in {name}: {', '.join(required)}")
    return reader


def parse_events(dataset: Dataset, payload: bytes) -> list[Event]:
    events = []
    seen = set()
    for line, row in enumerate(_rows(payload, EVENT_COLUMNS, "events.csv"), 2):
        try:
            first, last = int(row["first_frame"]), int(row["last_frame"])
            event_id = row["event_id"].strip()
            description = row["description"].strip()
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"invalid event row at line {line}") from exc
        if (not re.fullmatch(r"[A-Za-z0-9_-]+", event_id) or event_id in seen or not description
                or not 1 <= first <= last <= len(dataset.frames)):
            raise ValueError(f"invalid or duplicate event at line {line}: {event_id!r}")
        events.append(Event(event_id, first, last, description))
        seen.add(event_id)
    return events


def _legacy_check_identity(dataset: Dataset, folder: Path) -> tuple[str | None, dict[str, str]]:
    """Recover row identity for a legacy sheet beside its original evaluation snapshots."""
    try:
        metadata = json.loads((folder / "evaluation_metadata.json").read_bytes())
        old_dataset = metadata["capture_manifest_sha256"]
        if old_dataset != dataset.manifest_sha256:
            return old_dataset, {}
        snapshot = (folder / "events_snapshot.csv").read_bytes()
        if sha256(snapshot).hexdigest() == metadata["events_sha256"]:
            return old_dataset, {event.event_id: event.fingerprint for event in parse_events(dataset, snapshot)}
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return None, {}


def parse_checks(dataset: Dataset, events: list[Event], payload: bytes, source_path: Path
                 ) -> tuple[dict[tuple[str, int], tuple[str, str]], list[dict]]:
    event_map = {event.event_id: event for event in events}
    fingerprints = {event.event_id: event.fingerprint for event in events}
    reader = _rows(payload, CHECK_COLUMNS[:3], "preservation checks")
    legacy_dataset, legacy_events = None, {}
    if not set(CHECK_IDENTIFIERS).intersection(reader.fieldnames or []):
        legacy_dataset, legacy_events = _legacy_check_identity(dataset, source_path.parent)
    checks = {}
    rejected = []
    seen = set()
    for line, row in enumerate(reader, 2):
        try:
            event_id, index = row["event_id"].strip(), int(row["frame_index"])
            verdict = row["verdict"].strip()
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError(f"invalid preservation check at line {line}") from exc
        key = (event_id, index)
        if key in seen or verdict not in ("", "visible", "absent", "uncertain"):
            raise ValueError(f"invalid or duplicate preservation check at line {line}: {key}")
        seen.add(key)
        old_dataset = row.get("dataset_sha256", legacy_dataset)
        old_event = row.get("event_sha256", legacy_events.get(event_id))
        event = event_map.get(event_id)
        if old_dataset and old_dataset != dataset.manifest_sha256:
            reason = "dataset_changed"
        elif event is None:
            reason = "event_removed"
        elif not old_dataset or not old_event:
            reason = "missing_identity"
        elif old_event != fingerprints[event_id]:
            reason = "event_changed"
        elif not event.first_frame <= index <= event.last_frame:
            reason = "frame_outside_event"
        else:
            checks[key] = verdict, row.get("notes") or ""
            continue
        rejected.append({**{column: row.get(column, "") for column in CHECK_COLUMNS},
                         "dataset_sha256": old_dataset, "event_sha256": old_event, "reason": reason})
    return checks, rejected


def event_results(events: list[Event], predictions: list[Prediction],
                  checks: dict[tuple[str, int], tuple[str, str]]) -> tuple[list[dict], dict]:
    result = []
    evidence = {}
    threshold = predictions[0].threshold
    start = predictions[0].frame_index
    for event in events:
        if not start <= event.first_frame <= event.last_frame <= predictions[-1].frame_index:
            raise ValueError(f"event window is outside the sweep: {event.event_id}")
        window = predictions[event.first_frame - start:event.last_frame - start + 1]
        candidates = [row for row in window if row.stored]
        confirmed = []
        pending = 0
        for candidate in candidates:
            key = (event.event_id, candidate.frame_index)
            evidence[key] = candidate
            verdict, _ = checks.get(key, ("", ""))
            if verdict == "visible":
                confirmed.append(candidate)
            elif verdict != "absent":
                pending += 1
        first = confirmed[0] if confirmed else None
        if first is not None:
            status, reason = "preserved", "confirmed_visible"
        elif pending:
            status, reason = "needs_review", "unresolved_candidates"
        elif candidates:
            status, reason = "not_preserved", "all_candidates_absent"
        else:
            status, reason = "not_preserved", "no_representative_in_window"
        result.append({"threshold": threshold, "event_id": event.event_id,
                       "description": event.description, "first_frame": event.first_frame,
                       "last_frame": event.last_frame, "onset_stored": window[0].stored,
                       "candidate_frames": " ".join(str(row.frame_index) for row in candidates),
                       "unreviewed_candidates": pending, "status": status, "reason": reason,
                       "first_confirmed_frame": first.frame_index if first is not None else None,
                       "confirmed_delay_ms": first.timestamp_ms - window[0].timestamp_ms
                       if first is not None else None})
    return result, evidence


def candidate_report(metrics: list[dict], requirements: LabelRequirements) -> dict:
    """Prioritize preservation, then expose trade-offs without arbitrary weights."""
    first = metrics[0]  # All thresholds use the same reference labels.
    positives, negatives = first["tp"] + first["fn"], first["tn"] + first["fp"]
    labeling_ratio = first["labeling_ratio"]
    observed = {"labeled_frames": first["labeled_frames"], "labeling_ratio": labeling_ratio,
                "meaningful_change_labels": positives, "reused_labels": negatives}
    minimums = {"labeled_frames": requirements.min_labeled_frames,
                "labeling_ratio": requirements.min_labeling_ratio,
                "meaningful_change_labels": requirements.min_labels_per_binary_class,
                "reused_labels": requirements.min_labels_per_binary_class}
    unmet = [name for name, minimum in minimums.items()
             if observed[name] is None or observed[name] < minimum]
    report = {"labeling": {"eligible_frames": first["eligible_frames"], **observed,
                           "minimums": minimums, "unmet": unmet}, "candidates": []}
    if unmet:
        return {**report, "status": "insufficient_frame_labels"}
    if not first["events"]:
        return {**report, "status": "no_annotated_events"}
    if any(row["events_needing_review"] for row in metrics):
        return {**report, "status": "review_pending"}
    minimum_loss = min(row["events_not_preserved"] for row in metrics)
    best = [row for row in metrics if row["events_not_preserved"] == minimum_loss]
    details = {"minimum_events_not_preserved": minimum_loss,
               "best_preservation_thresholds": [row["threshold"] for row in best]}
    axes = ["fn", "fp", "group_mismatches", "stored"]
    vectors = {tuple(row[axis] for axis in axes) for row in best}
    frontier = {point for point in vectors if not any(
        other != point and all(a <= b for a, b in zip(other, point)) for other in vectors
    )}
    candidates = [row["threshold"] for row in best if tuple(row[axis] for axis in axes) in frontier]
    return {**report, "status": "evaluated_capture_candidates", "candidates": candidates,
            "tradeoff_axes_minimized": axes, **details}
