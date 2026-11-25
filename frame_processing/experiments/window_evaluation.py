"""Score stored-frame decisions against manually annotated inclusive windows."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from datetime import datetime
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path

from interface_spec.contracts import FrameDecision

from .capture import Dataset, write_json
from .labels import parse_labels
from .preservation import Event, parse_events
from .replay import Prediction, load_predictions, load_sweep, parse_decimal, threshold_text, write_csv

UPDATE = "new_representative"
TARGET = "target_new_representatives"
STORED = "stored_representative"
EXCLUDED = ("unnecessary", "scroll")
DETAIL_COLUMNS = ["threshold", "kind", "item_id", "event_id", "first_frame", "last_frame",
                  "candidate_frames", "status", "stored_count", "stored_frames", "stored_decisions",
                  "new_representative_frames", "new_group_frames", "target_count"]


@dataclass(frozen=True, slots=True)
class VerificationWindow:
    check_id: str
    kind: str
    first_frame: int
    last_frame: int
    event_id: str
    candidates: tuple[int, ...] = ()
    target_count: int | None = None


def parse_verification_windows(payload: bytes, events: list[Event], first: int,
                               last: int) -> list[VerificationWindow]:
    reader = csv.DictReader(StringIO(payload.decode("utf-8-sig")))
    columns = {"check_id", "kind", "first_frame", "last_frame", "event_id"}
    if not columns.issubset(reader.fieldnames or []):
        raise ValueError("missing required columns in verification_windows.csv")
    parents = {event.event_id: event for event in events}
    used = set()
    checks = []
    for number, row in enumerate(reader, 2):
        try:
            check_id, kind, event_id = (row[key].strip() for key in ("check_id", "kind", "event_id"))
            start, end = int(row["first_frame"]), int(row["last_frame"])
            candidates = tuple(map(int, (row.get("candidate_frames") or "").split()))
            target_value = (row.get("target_count") or "").strip()
            target = int(target_value) if target_value else None
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid verification row {number}") from exc
        if (not check_id or check_id in used or kind not in (UPDATE, TARGET, STORED, *EXCLUDED)
                or not first <= start <= end <= last):
            raise ValueError(f"invalid verification window at row {number}: {check_id}")
        if candidates and (tuple(sorted(set(candidates))) != candidates
                           or candidates[0] < start or candidates[-1] > end):
            raise ValueError(f"invalid candidate frame list for {check_id}")
        if kind == TARGET:
            if target is None or not 1 <= target <= (len(candidates) or end - start + 1):
                raise ValueError(f"invalid reference target for {check_id}")
        elif target is not None:
            raise ValueError(f"target_count is only supported for {TARGET}: {check_id}")
        if kind not in EXCLUDED:
            parent = parents.get(event_id)
            if parent is None or not parent.first_frame <= start <= end <= parent.last_frame:
                raise ValueError(f"verification window outside its parent event: {check_id}")
        used.add(check_id)
        checks.append(VerificationWindow(check_id, kind, start, end, event_id, candidates, target))
    return checks


def _indices(start: int, end: int, candidates: tuple[int, ...] = ()) -> tuple[int, ...] | range:
    return candidates or range(start, end + 1)


def score_windows(threshold: str, predictions: list[Prediction], events: list[Event],
                  checks: list[VerificationWindow]) -> tuple[dict, list[dict]]:
    saved = {row.frame_index: row.decision for row in predictions if row.stored}
    details = []
    missed = []
    preserved = 0
    update_hits = strict_checks = stored_hits = stored_checks = targets_met = target_checks = 0
    group_only = []
    unwanted = {kind: set() for kind in EXCLUDED}
    annotated = set()

    def detail(kind: str, item_id: str, event_id: str, start: int, end: int,
               candidates: tuple[int, ...] = (), target: int | None = None) -> tuple[dict, list[int], list[int]]:
        stored = [index for index in _indices(start, end, candidates) if index in saved]
        updates = [index for index in stored if saved[index] == FrameDecision.NEW_REPRESENTATIVE]
        groups = [index for index in stored if saved[index] == FrameDecision.NEW_GROUP]
        row = {"threshold": threshold, "kind": kind, "item_id": item_id, "event_id": event_id,
               "first_frame": start, "last_frame": end, "target_count": target,
               "candidate_frames": " ".join(map(str, candidates)), "stored_count": len(stored),
               "stored_frames": " ".join(map(str, stored)),
               "stored_decisions": " ".join(f"{i}={saved[i]}" for i in stored),
               "new_representative_frames": " ".join(map(str, updates)),
               "new_group_frames": " ".join(map(str, groups))}
        details.append(row)
        return row, updates, groups

    for event in events:
        row, _, _ = detail("required", event.event_id, event.event_id, event.first_frame, event.last_frame)
        annotated.update(range(event.first_frame, event.last_frame + 1))
        if row["stored_count"]:
            row["status"] = "hit"
            preserved += 1
        else:
            row["status"] = "missed"
            missed.append(f"{event.first_frame}-{event.last_frame}")

    for check in checks:
        row, updates, groups = detail(check.kind, check.check_id, check.event_id, check.first_frame,
                                      check.last_frame, check.candidates, check.target_count)
        if check.kind == UPDATE:
            strict_checks += 1
            update_hits += bool(updates)
            row["status"] = "hit" if updates else "new_group_only" if groups else "missed"
            if not updates and groups:
                group_only.extend(groups)
        elif check.kind == TARGET:
            target_checks += 1
            met = len(updates) >= check.target_count
            targets_met += met
            row["status"] = "reference_target_met" if met else "below_reference_target"
        elif check.kind == STORED:
            stored_checks += 1
            stored_hits += bool(row["stored_count"])
            row["status"] = "hit" if row["stored_count"] else "missed"
        else:
            indices = _indices(check.first_frame, check.last_frame, check.candidates)
            unwanted[check.kind].update(index for index in indices if index in saved)
            annotated.update(indices)
            row["status"] = "unnecessary_saved" if row["stored_count"] else "none_saved"

    summary = {
        "threshold": threshold, "observed": len(predictions), "required_windows": len(events),
        "preserved_windows": preserved, "preservation_rate": preserved / len(events) if events else None,
        "stored_frames": len(saved), "deduplication_rate": 1 - len(saved) / len(predictions),
        "new_groups": sum(decision == FrameDecision.NEW_GROUP for decision in saved.values()),
        "new_representatives": sum(decision == FrameDecision.NEW_REPRESENTATIVE for decision in saved.values()),
        "unnecessary_stored_frames": len(set.union(*unwanted.values())),
        "explicit_unnecessary_stored_frames": len(unwanted["unnecessary"]),
        "scroll_stored_frames": len(unwanted["scroll"]), "unjudged_stored_frames": len(saved.keys() - annotated),
        "update_checks": strict_checks, "update_hits": update_hits,
        "update_rate": update_hits / strict_checks if strict_checks else None,
        "new_group_instead_checks": sum(row["status"] == "new_group_only" for row in details),
        "new_group_instead_frames": " ".join(map(str, sorted(set(group_only)))),
        "stored_representative_checks": stored_checks, "stored_representative_hits": stored_hits,
        "reference_targets": target_checks, "reference_targets_met": targets_met,
        "missed_required_windows": "; ".join(missed),
    }
    return summary, details


def _parse_required_frames(dataset: Dataset, payload: bytes, first: int, last: int) -> list[int]:
    reader = csv.DictReader(StringIO(payload.decode("utf-8-sig")))
    if not {"frame_index", "timestamp_ms", "expected_store"}.issubset(reader.fieldnames or []):
        raise ValueError("required_frames.csv needs frame_index, timestamp_ms and expected_store")
    indices = set()
    for row in reader:
        index = int(row["frame_index"])
        if (not first <= index <= last or index in indices or row["expected_store"].lower() != "true"
                or int(row["timestamp_ms"]) != dataset.frames[index - 1].timestamp_ms):
            raise ValueError(f"invalid required frame {index}")
        indices.add(index)
    if not indices:
        raise ValueError("required_frames.csv contains no labels")
    return sorted(indices)


def score_required_frames(threshold: str, predictions: list[Prediction], indices: list[int]) -> tuple[dict, list[dict]]:
    """Match each required point to its frame or predecessor, with no reuse of matches."""
    summary, _ = score_windows(threshold, predictions, [], [])
    stored = {row.frame_index: row.decision for row in predictions if row.stored}
    unused = set(stored)
    missed, details = [], []
    first = predictions[0].frame_index
    for index in indices:
        candidates = tuple(i for i in (index - 1, index) if i >= first)
        match = next((i for i in candidates if i in unused), None)
        if match is None:
            missed.append(index)
        else:
            unused.remove(match)
        details.append({"threshold": threshold, "kind": "required_frame", "item_id": f"frame_{index}",
                        "first_frame": index, "last_frame": index, "candidate_frames": " ".join(map(str, candidates)),
                        "status": "missed" if match is None else "hit", "stored_count": int(match is not None),
                        "stored_frames": match, "stored_decisions": f"{match}={stored[match]}" if match is not None else ""})
    hits = len(indices) - len(missed)
    summary.update(required_windows=len(indices), preserved_windows=hits, preservation_rate=hits / len(indices),
                   missed_required_windows="; ".join(map(str, missed)), unjudged_stored_frames=len(unused),
                   unnecessary_stored_frames=None, explicit_unnecessary_stored_frames=None, scroll_stored_frames=None)
    return summary, details


TRANSITION_COLUMNS = ["threshold", "group_threshold", "representative_threshold", "frame_index", "timestamp_ms",
                      "manual_new_group", "baseline_threshold", "baseline_decision", "required_window_start",
                      "ssim_pc", "ssim_rc", "decision", "group_id", "representative_before", "representative_after"]


def transition_cases(predictions: list[Prediction], prior: Prediction | None, baseline_groups: set[int],
                     manual_groups: set[int], event_starts: set[int], baseline_threshold: str | None):
    for row in predictions:
        index = row.frame_index
        if row.decision == FrameDecision.NEW_REPRESENTATIVE and (index in baseline_groups or index in manual_groups):
            yield {"threshold": row.threshold, "frame_index": index, "timestamp_ms": row.timestamp_ms,
                   "manual_new_group": index in manual_groups, "baseline_threshold": baseline_threshold,
                   "baseline_decision": FrameDecision.NEW_GROUP if index in baseline_groups else "",
                   "required_window_start": index in event_starts, "ssim_pc": row.ssim_pc, "ssim_rc": row.ssim_rc,
                   "decision": row.decision, "group_id": row.group_id,
                   "representative_before": prior.representative_index if prior else None,
                   "representative_after": row.representative_index}
        prior = row


def evaluate_windows(dataset: Dataset, run: Path, *, start_frame: int | None = None,
                     end_frame: int | None = None, required_frames: bool = False,
                     baseline: Path | None = None, transition_threshold: str = "0.95") -> Path:
    selection = load_sweep(dataset, run)
    first = selection.start if start_frame is None else start_frame
    last = selection.end if end_frame is None else end_frame
    if not selection.start <= first <= last <= selection.end:
        raise ValueError(f"evaluation scope must be within source sweep {selection.start}–{selection.end}")
    labels_payloads = {}
    if required_frames:
        labels_payloads["required_frames"] = (dataset.path / "required_frames.csv").read_bytes()
        indices = _parse_required_frames(dataset, labels_payloads["required_frames"], first, last)
        event_starts = set(indices)

        def score(threshold, predictions):
            return score_required_frames(threshold, predictions, indices)
    else:
        for name in ("events", "verification_windows"):
            labels_payloads[name] = (dataset.path / f"{name}.csv").read_bytes()
        events = parse_events(dataset, labels_payloads["events"])
        if any(not first <= event.first_frame <= event.last_frame <= last for event in events):
            raise ValueError("all required windows must be fully inside the evaluation scope")
        checks = parse_verification_windows(labels_payloads["verification_windows"], events, first, last)
        event_starts = {event.first_frame for event in events}

        def score(threshold, predictions):
            return score_windows(threshold, predictions, events, checks)

    labels_path = dataset.path / "labels.csv"
    manual_groups = set()
    if labels_path.is_file():
        payload = labels_path.read_bytes()
        manual_groups = {i for i, label in parse_labels(dataset, payload).items()
                         if label == FrameDecision.NEW_GROUP and i != 1 and i != selection.start and first <= i <= last}
        if manual_groups:
            labels_payloads["labels"] = payload

    start_offset, end_offset = first - selection.start, last - selection.start + 1
    baseline_metrics = []
    baseline_groups = set()
    reference_name = None
    diagonal = {}
    if baseline is not None:
        reference = load_sweep(dataset, baseline)
        if (reference.start != selection.start or reference.end < last
                or any(candidate.representative is not None for candidate in reference.candidates)):
            raise ValueError("baseline must be a single-threshold sweep with the same replay start covering the evaluation range")
        primary_meta = json.loads((run / "sweep_metadata.json").read_bytes())
        baseline_meta = json.loads((baseline / "sweep_metadata.json").read_bytes())
        if primary_meta.get("comparison_max_side") != baseline_meta.get("comparison_max_side"):
            raise ValueError("baseline and current sweep must use the same comparison size")
        reference_name = threshold_text(parse_decimal(transition_threshold))
        if reference_name not in reference.thresholds:
            raise ValueError(f"transition threshold {reference_name} is absent from the baseline")
        equal_values = {candidate.group for candidate in selection.candidates
                        if candidate.group == candidate.representative}
        for candidate in reference.candidates:
            predictions = load_predictions(reference, baseline, candidate.name)
            scoped = predictions[start_offset:end_offset]
            summary, _ = score(candidate.name, scoped)
            baseline_metrics.append({**candidate.columns(), **summary})
            if candidate.name == reference_name:
                baseline_groups = {row.frame_index for row in scoped if row.decision == FrameDecision.NEW_GROUP
                                   and row.frame_index != reference.start}
            if candidate.group in equal_values:
                diagonal[candidate.group] = predictions

    summaries, details, transitions, regression = [], [], [], []
    for candidate in selection.candidates:
        predictions = load_predictions(selection, run, candidate.name)
        scoped = predictions[start_offset:end_offset]
        summary, rows = score(candidate.name, scoped)
        columns = candidate.columns()
        summaries.append({**columns, **summary})
        details.extend({**columns, **row} for row in rows)
        prior = predictions[start_offset - 1] if start_offset else None
        transitions.extend({**columns, **row} for row in transition_cases(scoped, prior, baseline_groups, manual_groups,
                                                                        event_starts, reference_name))
        if candidate.group == candidate.representative and candidate.group in diagonal:
            baseline_rows = diagonal[candidate.group]
            baseline_name = threshold_text(candidate.group)
            common_frames = min(len(predictions), len(baseline_rows))
            mismatches = [row.frame_index for row, old in zip(predictions, baseline_rows)
                          if replace(row, threshold=baseline_name) != old]
            regression.append({**columns, "compared_frames": common_frames, "identical": not mismatches,
                               "mismatched_frames": mismatches})

    output = dataset.path / "evaluations" / f"required_windows_{first:04d}_{last:04d}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "summary.csv", list(summaries[0]), summaries)
    write_csv(output / "window_results.csv", ["group_threshold", "representative_threshold", *DETAIL_COLUMNS], details)
    write_csv(output / "transition_review.csv", TRANSITION_COLUMNS, transitions)
    if baseline_metrics:
        write_csv(output / "baseline_summary.csv", list(baseline_metrics[0]), baseline_metrics)
    for name, payload in labels_payloads.items():
        (output / f"{name}_snapshot.csv").write_bytes(payload)
    write_json(output / "evaluation_metadata.json", {
        "schema_version": 1, "source_sweep": str(run.resolve()),
        "source_start_frame": selection.start, "source_end_frame": selection.end,
        "first_frame": first, "last_frame": last, "selected_frames": last - first + 1,
        "capture_manifest_sha256": dataset.manifest_sha256,
        **{f"{name}_sha256": sha256(payload).hexdigest() for name, payload in labels_payloads.items()},
        "preservation_rule": ("One-to-one required frame or its immediate predecessor; selection only." if required_frames
                              else "Any stored decision in the inclusive required window; selection only."),
        "update_rule": "At least one new_representative in the specified candidates; new_group is reported separately.",
        "reference_target_rule": "Optional; counted as met when the number of new_representative decisions reaches the target.",
        "baseline_sweep": str(baseline.resolve()) if baseline is not None else None,
        "transition_baseline_threshold": reference_name,
        "manual_new_group_frames": len(manual_groups), "equal_threshold_regression": regression,
    })
    return output
