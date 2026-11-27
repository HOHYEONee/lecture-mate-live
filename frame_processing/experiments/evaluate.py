"""Compute frame detection and three-class scores from editable human labels."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import datetime
from hashlib import sha256
from pathlib import Path

from interface_spec.contracts import FrameDecision

from .capture import Dataset, write_json
from .charts import write_charts
from .labels import AMBIGUOUS, CLASSES, ensure_labels, parse_labels
from .preservation import (
    CHECK_COLUMNS, EVENT_COLUMNS, EVENT_RESULT_COLUMNS, LabelRequirements, candidate_report, ensure_events, event_results,
    parse_checks, parse_events,
)
from .replay import Prediction, load_predictions, load_sweep, summarize, write_csv


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def scores(tp: int, fp: int, fn: int) -> dict:
    return {"precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
            "f1": ratio(2 * tp, 2 * tp + fp + fn)}


def frame_metrics(predictions: list[Prediction], labels: dict[int, str]) -> tuple[dict, list, list, list]:
    threshold = predictions[0].threshold
    matrix: Counter = Counter()
    unlabeled = ambiguous = 0
    errors = []
    for prediction in predictions[1:]:  # Initial state always stores a new group.
        truth = labels.get(prediction.frame_index, "")
        if not truth:
            unlabeled += 1
            continue
        if truth == AMBIGUOUS:
            ambiguous += 1
            continue
        predicted = prediction.decision
        matrix[truth, predicted] += 1
        if truth != predicted:
            if predicted == FrameDecision.REUSED:
                kind = "false_negative"
            elif truth == FrameDecision.REUSED:
                kind = "false_positive"
            else:
                kind = "group_mismatch"
            errors.append({"threshold": threshold, "frame_index": prediction.frame_index,
                           "timestamp_ms": prediction.timestamp_ms, "actual": truth,
                           "predicted": predicted, "error": kind})

    reused = FrameDecision.REUSED
    changed = CLASSES[:2]
    tp = sum(matrix[a, p] for a in changed for p in changed)
    fp = sum(matrix[reused, p] for p in changed)
    fn = sum(matrix[a, reused] for a in changed)
    tn = matrix[reused, reused]
    eligible, labeled = len(predictions) - 1, tp + fp + fn + tn
    binary = {"threshold": threshold, "eligible_frames": eligible, "labeled_frames": labeled,
              "labeling_ratio": ratio(labeled, eligible), "unlabeled_frames": unlabeled,
              "ambiguous_frames": ambiguous, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
              **scores(tp, fp, fn), "false_positive_rate": ratio(fp, fp + tn),
              "false_negative_rate": ratio(fn, tp + fn), "unnecessary_stores": fp,
              "group_mismatches": sum(matrix[a, p] for a in changed for p in changed if a != p)}

    classes = []
    for label in CLASSES:
        correct = matrix[label, label]
        support = sum(matrix[label, p] for p in CLASSES)
        predicted_count = sum(matrix[a, label] for a in CLASSES)
        classes.append({"threshold": threshold, "class": label, "support": support,
                        "predicted_count": predicted_count,
                        **scores(correct, predicted_count - correct, support - correct)})
    confusion = [{"threshold": threshold, "actual": a, "predicted": p, "count": matrix[a, p]}
                 for a in CLASSES for p in CLASSES]
    return binary, classes, confusion, errors


def evaluate_run(dataset: Dataset, run: Path, checks_path: Path | None = None, *,
                 label_requirements: LabelRequirements = LabelRequirements()) -> Path:
    selection = load_sweep(dataset, run)
    thresholds = selection.thresholds
    labels_payload = ensure_labels(dataset).read_bytes()
    labels = parse_labels(dataset, labels_payload)
    event_payload = ensure_events(dataset).read_bytes()
    events = parse_events(dataset, event_payload)
    event_fingerprints = {event.event_id: event.fingerprint for event in events}
    checks_payload = None
    checks, rejected = {}, []
    if checks_path is not None:
        checks_payload = checks_path.read_bytes()
        checks, rejected = parse_checks(dataset, events, checks_payload, checks_path)
    included, excluded = [], []
    for event in events:
        if selection.start <= event.first_frame <= event.last_frame <= selection.end:
            included.append(event)
        else:
            outside = event.last_frame < selection.start or event.first_frame > selection.end
            excluded.append({**asdict(event), "reason": "outside_selection" if outside else "partial_overlap"})
    output = run / f"evaluation_{datetime.now():%Y%m%d_%H%M%S_%f}"
    output.mkdir(exist_ok=False)
    metrics, classes, confusion, errors = [], [], [], []
    preservation = []
    review_rows = {}
    for candidate in selection.candidates:
        threshold = candidate.name
        predictions = load_predictions(selection, run, threshold)
        binary, class_rows, confusion_rows, error_rows = frame_metrics(predictions, labels)
        storage = summarize(predictions)
        information, evidence = event_results(included, predictions, checks)
        statuses = Counter(row["status"] for row in information)
        columns = candidate.columns() if candidate.representative is not None else {}
        metrics.append({**columns, **binary, "stored": storage["stored"], "new_groups": storage["new_groups"],
                        "storage_reduction": storage["storage_reduction"], "events": len(included),
                        "events_preserved": statuses["preserved"],
                        "events_not_preserved": statuses["not_preserved"],
                        "events_needing_review": statuses["needs_review"],
                        "preservation_rate": ratio(statuses["preserved"], len(included))
                        if not statuses["needs_review"] else None})
        classes.extend(class_rows)
        confusion.extend(confusion_rows)
        errors.extend(error_rows)
        preservation.extend(information)
        for (event_id, index), candidate in evidence.items():
            verdict, notes = checks.get((event_id, index), ("", ""))
            row = review_rows.setdefault((event_id, index), {
                "event_id": event_id, "frame_index": index, "verdict": verdict, "notes": notes,
                "raw_image": f"raw_frames/{dataset.frames[index - 1].filename}",
                "representative_jpeg": "",
                "dataset_sha256": dataset.manifest_sha256, "event_sha256": event_fingerprints[event_id],
            })
            jpeg_path = Path(f"threshold_{threshold}") / "frames" / f"{candidate.representative_id}.jpg"
            if not row["representative_jpeg"] and (run / jpeg_path).is_file():
                row["representative_jpeg"] = jpeg_path.as_posix()

    write_csv(output / "metrics.csv", list(metrics[0]), metrics)
    write_csv(output / "class_metrics.csv", list(classes[0]), classes)
    write_csv(output / "confusion_matrix.csv", list(confusion[0]), confusion)
    write_csv(output / "errors.csv", ["threshold", "frame_index", "timestamp_ms", "actual",
                                      "predicted", "error"], errors)
    write_csv(output / "preservation_events.csv", EVENT_RESULT_COLUMNS, preservation)
    write_csv(output / "events_excluded.csv", [*EVENT_COLUMNS, "reason"], excluded)
    write_csv(output / "evidence_to_review.csv", CHECK_COLUMNS,
              (review_rows[key] for key in sorted(review_rows)))
    write_csv(output / "checks_requiring_review.csv", [*CHECK_COLUMNS, "reason"], rejected)
    write_json(output / "candidates.json", candidate_report(metrics, label_requirements))
    chart_rows = [{**row, "fp_plot": row["fp"] if row["labeled_frames"] else None,
                   "fn_plot": row["fn"] if row["labeled_frames"] else None} for row in metrics]
    write_charts(output / "charts", chart_rows, confusion)
    (output / "labels_snapshot.csv").write_bytes(labels_payload)
    (output / "events_snapshot.csv").write_bytes(event_payload)
    if checks_payload is not None:
        (output / "checks_snapshot.csv").write_bytes(checks_payload)
    write_json(output / "evaluation_metadata.json", {
        "schema_version": 3, "capture_manifest_sha256": dataset.manifest_sha256,
        "labels_sha256": sha256(labels_payload).hexdigest(), "initial_frame_excluded": True,
        "start_frame": selection.start, "end_frame": selection.end, "initial_frame_index": selection.start,
        "events_sha256": sha256(event_payload).hexdigest(),
        "checks_sha256": sha256(checks_payload).hexdigest() if checks_payload is not None else None,
        "thresholds": list(thresholds), "event_count": len(included), "events_excluded": len(excluded),
        "label_requirements": asdict(label_requirements),
        "checks_matched": len(checks), "checks_requiring_review": len(rejected),
    })
    return output


def compare_frame(dataset: Dataset, run: Path, index: int) -> Path:
    selection = load_sweep(dataset, run)
    if not selection.start <= index <= selection.end:
        raise ValueError(f"frame index outside sweep range {selection.start}–{selection.end}: {index}")
    output = run / f"compare_frame_{index:06d}.csv"
    rows = [{**asdict(load_predictions(selection, run, candidate.name)[index - selection.start]),
             **(candidate.columns() if candidate.representative is not None else {})}
            for candidate in selection.candidates]
    write_csv(output, list(rows[0]), rows)
    return output
