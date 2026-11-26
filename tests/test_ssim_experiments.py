from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from decimal import Decimal
from io import StringIO
from itertools import chain, repeat
import json
from pathlib import Path
from queue import Queue
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from xml.etree import ElementTree

import cv2
import numpy as np

from interface_spec.contracts import CaptureRegion, FrameDecision

from frame_processing import FrameProcessor
from frame_processing.comparison import SSIMComparator
from frame_processing.experiments import __main__ as cli
from frame_processing.experiments.capture import capture_dataset, load_dataset
from frame_processing.experiments.evaluate import compare_frame, evaluate_run
from frame_processing.experiments.labels import LABEL_COLUMNS, ensure_labels, parse_labels, review_frame
from frame_processing.experiments.preservation import (
    CHECK_COLUMNS, CHECK_IDENTIFIERS, EVENT_COLUMNS, LabelRequirements, ensure_events,
)
from frame_processing.experiments.progress import SweepProgress
from frame_processing.experiments.replay import (
    ThresholdPair, Unclassified, load_predictions, load_sweep, parse_threshold_pairs, parse_thresholds, read_csv,
    replay, run_sweep, write_csv,
)
from frame_processing.experiments.window_evaluation import evaluate_windows
from frame_processing.pipeline import FramePipeline
from frame_processing.storage import JPEGFrameStore


class FakeTime:
    def __init__(self):
        self.ns = 0

    def now_ns(self):
        return self.ns

    def sleep(self, seconds):
        self.ns += round(seconds * 1_000_000_000)


class SequenceBackend:
    def __init__(self, images, after_grab=None):
        self.images = iter(images)
        self.after_grab = after_grab

    def grab(self, region):
        if self.after_grab is not None:
            self.after_grab()
        return next(self.images)


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.region = CaptureRegion(-40, 0, 40, 32)
        self.images = [np.full((32, 40, 3), value, np.uint8)
                       for value in (100, 110, 120, 130, 140)]
        rng = np.random.default_rng(2026)
        self.images.extend([rng.integers(0, 256, (32, 40, 3), dtype=np.uint8)] * 2)

    def dataset(self):
        clock = FakeTime()
        folder = capture_dataset(self.root, "강의 필기", self.region, len(self.images),
                                 SequenceBackend(self.images), now_ns=clock.now_ns, wait=clock.sleep)
        return load_dataset(folder)

    def write_events(self, dataset, *definitions):
        path = ensure_events(dataset)
        write_csv(path, EVENT_COLUMNS, (dict(zip(EVENT_COLUMNS, event)) for event in definitions))

    def mark_evidence(self, output, *, legacy=False):
        path = output / "evidence_to_review.csv"
        rows = read_csv(path)
        columns = [column for column in CHECK_COLUMNS if not legacy or column not in CHECK_IDENTIFIERS]
        for row in rows:
            row.update(verdict="visible", notes="기존 메모")
        write_csv(path, columns, ({column: row[column] for column in columns} for row in rows))
        return path

    def test_png_capture_preserves_every_pixel_order_and_time(self):
        dataset = self.dataset()
        self.assertEqual(len(dataset.frames), len(self.images))
        for index, (frame, image) in enumerate(zip(dataset.frames, self.images)):
            self.assertEqual((frame.frame_index, frame.timestamp_ms), (index + 1, index * 1000))
            np.testing.assert_array_equal(dataset.read_frame(frame), image)
        self.assertNotEqual(dataset.frames[-1].filename, dataset.frames[-2].filename)

    def test_capture_reports_missed_slots_instead_of_bursting(self):
        clock = FakeTime()
        backend = SequenceBackend(self.images, lambda: clock.sleep(1.25))
        folder = capture_dataset(self.root, "slow", self.region, 5, backend,
                                 now_ns=clock.now_ns, wait=clock.sleep)
        dataset = load_dataset(folder)
        self.assertEqual([frame.scheduled_slot for frame in dataset.frames], [0, 2, 4])
        self.assertEqual([frame.timestamp_ms for frame in dataset.frames], [1250, 3250, 5250])
        self.assertEqual(json.loads((folder / "capture_metadata.json").read_bytes())["missed_slots"], 2)

    def test_replay_matches_service_with_and_without_representative_jpegs(self):
        dataset = self.dataset()
        for threshold in parse_thresholds("0.95,0.98,0.99,1"):
            with self.subTest(threshold=threshold):
                class RawClock:
                    def now_ms(self):
                        return next(timestamps)

                timestamps = iter(frame.timestamp_ms for frame in dataset.frames)
                service_dir = self.root / f"service_{threshold}"
                jpeg_dir = self.root / f"export_{threshold}"
                service_dir.mkdir()
                jpeg_dir.mkdir()
                pipeline = FramePipeline("service", self.region, RawClock(), SequenceBackend(self.images),
                                         JPEGFrameStore(service_dir), SSIMComparator(float(threshold)), Unclassified())
                records = [pipeline.periodic() for _ in self.images]
                predictions = replay(dataset, threshold)
                self.assertEqual(predictions, replay(dataset, threshold, jpeg_dir=jpeg_dir))
                self.assertEqual([(row.frame_index, row.timestamp_ms, row.decision, row.group_id,
                                   row.representative_id) for row in predictions],
                                 [(row.capture_seq, row.timestamp_ms, row.decision, row.group_id,
                                   row.frame_id) for row in records])
                saved = [row for row in predictions if row.stored]
                self.assertEqual(len(list((jpeg_dir / "frames").glob("*.jpg"))), len(saved))
                for row in saved:
                    image = f"frames/{row.representative_id}.jpg"
                    self.assertEqual((service_dir / image).read_bytes(), (jpeg_dir / image).read_bytes())
                self.assertEqual(saved[-1].frame_index, len(self.images) - 1)
                jpeg = cv2.imdecode(np.frombuffer((jpeg_dir / image).read_bytes(), np.uint8),
                                    cv2.IMREAD_COLOR)
                self.assertFalse(np.array_equal(jpeg, self.images[-1]))
                self.assertEqual(predictions[-1].decision, FrameDecision.REUSED)
                self.assertEqual(predictions[-1].ssim_rc, 1)

    def test_service_worker_matches_replay_for_default_pair_and_legacy_mode(self):
        dataset = self.dataset()
        times = [frame.timestamp_ms for frame in dataset.frames]
        group, representative = Decimal("0.90"), Decimal("0.98")
        results = {}
        for mode, options, threshold in (
            ("default", {}, ThresholdPair(group, representative)),
            ("legacy", {"ssim_threshold": 0.90}, group),
            ("equal_pair", {"group_threshold": 0.90, "representative_threshold": 0.90}, ThresholdPair(group, group)),
        ):
            with self.subTest(mode=mode):
                timestamps = iter(times)

                class RawClock:
                    def now_ms(self):
                        return next(timestamps, times[-1])

                # The worker can acquire another tick before stop() without exhausting the source.
                backend = SequenceBackend(chain(self.images, repeat(self.images[-1])))
                service_dir = self.root / f"worker_{mode}"
                service_dir.mkdir()
                processor = FrameProcessor(backend=backend, interval_s=0.01, **options)
                processor.set_region(self.region)
                output, errors = Queue(), Queue()
                with patch.object(processor, "_initialize_classifier", return_value=Unclassified()):
                    processor.start(mode, str(service_dir), RawClock(), output, errors.put)
                try:
                    records = [output.get(timeout=2) for _ in self.images]
                finally:
                    processor.stop()
                self.assertTrue(errors.empty())
                predictions = replay(dataset, threshold)
                actual = [(row.capture_seq, row.timestamp_ms, row.decision, row.group_id, row.frame_id)
                          for row in records]
                expected = [(row.frame_index, row.timestamp_ms, row.decision, row.group_id, row.representative_id)
                            for row in predictions]
                self.assertEqual(actual, expected)
                saved_names = {f"{row.representative_id}.jpg" for row in predictions if row.stored}
                self.assertEqual({path.name for path in (service_dir / "frames").glob("*.jpg")}, saved_names)
                results[mode] = actual
        self.assertEqual(results["legacy"], results["equal_pair"])
        self.assertEqual(results["default"][3][2], FrameDecision.NEW_REPRESENTATIVE)
        self.assertEqual(results["legacy"][3][2], FrameDecision.REUSED)

    def test_sweep_has_independent_representatives_and_stable_order(self):
        dataset = self.dataset()
        low, high = parse_thresholds("0.98,0.99")
        run = run_sweep(dataset, (low, high))
        reverse = run_sweep(dataset, (high, low))
        low_rows = read_csv(run / "threshold_0.98" / "decisions.csv")
        high_rows = read_csv(run / "threshold_0.99" / "decisions.csv")
        self.assertEqual(low_rows, read_csv(reverse / "threshold_0.98" / "decisions.csv"))
        self.assertEqual(high_rows, read_csv(reverse / "threshold_0.99" / "decisions.csv"))
        self.assertEqual(low_rows[2]["decision"], "reused")
        self.assertEqual(high_rows[2]["decision"], "new_representative")
        self.assertEqual(low_rows[3]["ssim_pc"], high_rows[3]["ssim_pc"])
        self.assertNotEqual(low_rows[3]["ssim_rc"], high_rows[3]["ssim_rc"])
        self.assertEqual(low_rows[0]["ssim_pc"], "")
        self.assertEqual(low_rows[0]["ssim_rc"], "")
        self.assertEqual(low_rows[5]["ssim_rc"], "")
        summary = read_csv(run / "summary.csv")
        self.assertEqual([row["threshold"] for row in summary], ["0.98", "0.99"])
        for rows, aggregate in zip((low_rows, high_rows), summary):
            self.assertEqual(int(aggregate["stored"]), sum(row["stored"] == "True" for row in rows))

    def test_range_restarts_pcr_and_shared_comparisons_match_isolated_replays(self):
        dataset = self.dataset()
        manifest = dataset.path / "capture_metadata.json"
        original = manifest.read_bytes()
        thresholds = parse_thresholds("0.98,0.99")
        run = run_sweep(dataset, thresholds, start_frame=2, end_frame=6, write_representatives=True)
        selection = load_sweep(dataset, run)
        self.assertEqual((selection.start, selection.end), (2, 6))
        self.assertEqual([row["observed"] for row in read_csv(run / "summary.csv")], ["5", "5"])
        for name, threshold in zip(selection.thresholds, thresholds):
            rows = load_predictions(selection, run, name)
            self.assertEqual(rows, replay(dataset, threshold, start_frame=2, end_frame=6))
            self.assertEqual([row.frame_index for row in rows], list(range(2, 7)))
            self.assertEqual([row.timestamp_ms for row in rows], [1000, 2000, 3000, 4000, 5000])
            self.assertEqual((rows[0].decision, rows[0].representative_index, rows[0].ssim_pc, rows[0].ssim_rc),
                             (FrameDecision.NEW_GROUP, 2, None, None))
            self.assertTrue((run / f"threshold_{name}" / "frames" / f"{rows[0].representative_id}.jpg").is_file())
        low, high = [load_predictions(selection, run, name) for name in selection.thresholds]
        self.assertNotEqual(low[2].representative_index, high[2].representative_index)
        self.assertNotEqual(low[3].ssim_rc, high[3].ssim_rc)
        self.assertEqual(manifest.read_bytes(), original)
        self.assertEqual(len(list((dataset.path / "raw_frames").glob("*.png"))), len(dataset.frames))

    def test_dual_sweep_preserves_diagonal_and_independent_state(self):
        dataset = self.dataset()
        candidates = parse_threshold_pairs("0.98,0.99", "0.98,0.99")
        self.assertEqual(len(candidates), 4)
        run = run_sweep(dataset, candidates)
        selection = load_sweep(dataset, run)
        self.assertEqual(selection.candidates, candidates)
        for candidate in candidates:
            rows = load_predictions(selection, run, candidate.name)
            self.assertEqual(rows, replay(dataset, candidate))
            if candidate.group == candidate.representative:
                self.assertEqual([replace(row, threshold=str(candidate.group)) for row in rows],
                                 replay(dataset, candidate.group))
        slow = load_predictions(selection, run, "g0.98_r0.98")
        fast = load_predictions(selection, run, "g0.98_r0.99")
        self.assertEqual((slow[2].decision, fast[2].decision), (FrameDecision.REUSED, FrameDecision.NEW_REPRESENTATIVE))
        self.assertNotEqual(slow[3].ssim_rc, fast[3].ssim_rc)
        self.assertEqual(fast[5].decision, FrameDecision.NEW_GROUP)
        self.assertIsNone(fast[5].ssim_rc)

    def test_range_evaluation_excludes_bootstrap_and_events_with_incomplete_windows(self):
        dataset = self.dataset()
        run = run_sweep(dataset, parse_thresholds("0.98"), start_frame=3, end_frame=6)
        path = ensure_labels(dataset)
        labels = read_csv(path)
        for row in labels:
            row["label"] = "reused"
        write_csv(path, LABEL_COLUMNS, labels)
        self.write_events(dataset, ("in", 3, 3, "선택 구간의 첫 화면"),
                          ("overlap", 5, 7, "선택 구간에 잘린 이벤트"), ("out", 1, 2, "앞부분"))
        output = evaluate_run(dataset, run)
        row, = read_csv(output / "metrics.csv")
        self.assertEqual((row["eligible_frames"], row["labeled_frames"], row["events"]), ("3", "3", "1"))
        self.assertNotIn("3", {error["frame_index"] for error in read_csv(output / "errors.csv")})
        evidence, = read_csv(output / "evidence_to_review.csv")
        self.assertEqual((evidence["event_id"], evidence["frame_index"]), ("in", "3"))
        self.assertEqual({row["event_id"]: row["reason"] for row in read_csv(output / "events_excluded.csv")},
                         {"overlap": "partial_overlap", "out": "outside_selection"})
        with self.assertRaisesRegex(ValueError, "outside sweep range"):
            compare_frame(dataset, run, 2)
        self.assertEqual(read_csv(compare_frame(dataset, run, 6))[0]["frame_index"], "6")
        with self.assertRaisesRegex(ValueError, "within 3–6"):
            review_frame(dataset, 3, reference=2, start_frame=3, end_frame=6)
        preview = review_frame(dataset, 3, start_frame=3, end_frame=6)
        image = cv2.imdecode(np.frombuffer(preview.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(image.shape, (64, 320, 3))  # No previous frame from outside the range.

    def test_range_validation_and_legacy_full_sweep_metadata(self):
        dataset = self.dataset()
        for start, end in ((0, 5), (4, 3), (1, 8)):
            with self.subTest(start=start, end=end), self.assertRaisesRegex(ValueError, "start frame"):
                run_sweep(dataset, parse_thresholds("0.98"), start_frame=start, end_frame=end)
        self.assertEqual(len(dataset.select_frames(end=2)), 2)
        self.assertEqual(len(dataset.select_frames(start=6)), 2)
        run = run_sweep(dataset, parse_thresholds("0.98"))
        path = run / "sweep_metadata.json"
        metadata = json.loads(path.read_bytes())
        metadata["schema_version"] = 1
        del metadata["start_frame"], metadata["end_frame"]
        path.write_text(json.dumps(metadata), encoding="utf-8")
        self.assertEqual(load_sweep(dataset, run).frames, dataset.frames)
        self.assertEqual(read_csv(evaluate_run(dataset, run) / "metrics.csv")[0]["eligible_frames"], "6")

    def test_window_evaluation_filters_existing_predictions_without_restarting(self):
        dataset = self.dataset()
        run = run_sweep(dataset, parse_thresholds("0.98"))
        decision_file = run / "threshold_0.98" / "decisions.csv"
        original = decision_file.read_bytes()
        self.write_events(dataset, ("window", 2, 3, "adjacent frames reused"))
        check_file = dataset.path / "verification_windows.csv"
        check_file.write_text("check_id,kind,first_frame,last_frame,event_id\n", encoding="utf-8")
        output = evaluate_windows(dataset, run, start_frame=2, end_frame=3)
        row, = read_csv(output / "summary.csv")
        self.assertEqual((row["observed"], row["preserved_windows"], row["stored_frames"]), ("2", "0", "0"))
        self.assertEqual((output / "verification_windows_snapshot.csv").read_bytes(), check_file.read_bytes())
        self.assertEqual(decision_file.read_bytes(), original)
        self.assertRaises(ValueError, evaluate_windows, dataset, run, start_frame=3, end_frame=4)

    def test_dual_evaluation_reads_single_baseline_and_reports_equal_threshold_regression(self):
        dataset = self.dataset()
        baseline = run_sweep(dataset, parse_thresholds("0.9,0.99"))
        (baseline / "threshold_0.9").rename(baseline / "threshold_0.90")
        run = run_sweep(dataset, parse_threshold_pairs("0.98,0.99", "0.99"))
        frames_path = dataset.path / "required_frames.csv"
        frames_path.write_text("frame_index,timestamp_ms,expected_store\n1,0,true\n3,2000,true\n", encoding="utf-8")
        output = evaluate_windows(dataset, run, required_frames=True, baseline=baseline, transition_threshold="0.99")
        summary = read_csv(output / "summary.csv")
        self.assertEqual([row["group_threshold"] for row in summary], ["0.98", "0.99"])
        self.assertEqual([row["threshold"] for row in read_csv(output / "baseline_summary.csv")], ["0.9", "0.99"])
        metadata = json.loads((output / "evaluation_metadata.json").read_bytes())
        checked, = metadata["equal_threshold_regression"]
        self.assertEqual((checked["compared_frames"], checked["identical"]), (len(dataset.frames), True))
        self.assertEqual((output / "required_frames_snapshot.csv").read_bytes(), frames_path.read_bytes())
        standard = evaluate_run(dataset, run)
        self.assertEqual(read_csv(standard / "metrics.csv")[0]["representative_threshold"], "0.99")

    def test_progress_counts_across_batches_and_keeps_csv_identical(self):
        dataset = self.dataset()
        thresholds = parse_thresholds(limits=("0.60", "0.92", "0.01"))
        stream = StringIO()
        clock = iter(range(100))
        options = {"start_frame": 2, "end_frame": 3}
        with_progress = run_sweep(dataset, thresholds, **options,
                                  progress=SweepProgress(stream, clock=lambda: next(clock)))
        without_progress = run_sweep(dataset, thresholds, **options)
        self.assertIn("Current 2/2 | Total 64/66", stream.getvalue())
        self.assertIn("Threshold 0.92 (33/33) | Current 1/2 | Total 65/66", stream.getvalue())
        self.assertIn("Threshold 0.92 (33/33) | Current 2/2 | Total 66/66 | 100.0%", stream.getvalue())
        for report in with_progress.rglob("*.csv"):
            relative = report.relative_to(with_progress)
            self.assertEqual(report.read_bytes(), (without_progress / relative).read_bytes(), str(relative))

    def test_exact_threshold_range_and_invalid_candidates(self):
        values = parse_thresholds(limits=("0.970", "0.990", "0.001"))
        self.assertEqual((len(values), values[1], values[-1]), (21, Decimal("0.971"), Decimal("0.990")))
        self.assertEqual(len(parse_thresholds()), 10)
        self.assertEqual(parse_thresholds(limits=("0.970", "0.972", "0.003")), (Decimal("0.970"),))
        for invalid in ("NaN", "0.95,0.950", "", "1.001"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_thresholds(invalid)

    def test_replay_rejects_modified_input(self):
        dataset = self.dataset()
        (dataset.path / "raw_frames" / dataset.frames[0].filename).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "PNG checksum mismatch"):
            replay(dataset, Decimal("0.95"))

    def test_label_template_and_review_preserve_human_edits(self):
        dataset = self.dataset()
        path = ensure_labels(dataset)
        rows = read_csv(path)
        self.assertEqual([row["label"] for row in rows], ["new_group"] + [""] * 6)
        rows[1].update(label="ambiguous", notes="스크롤")
        write_csv(path, LABEL_COLUMNS, rows)
        before = path.read_bytes()
        self.assertEqual(ensure_labels(dataset).read_bytes(), before)
        preview = review_frame(dataset, 2, reference=1)
        image = cv2.imdecode(np.frombuffer(preview.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(image.shape, (64, 960, 3))
        self.assertEqual(parse_labels(dataset, before)[2], "ambiguous")

    def test_label_validation_rejects_wrong_time_and_duplicate(self):
        dataset = self.dataset()
        header = b"frame_index,timestamp_ms,label\n"
        with self.assertRaisesRegex(ValueError, "timestamp mismatch"):
            parse_labels(dataset, header + b"2,0,reused\n")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            parse_labels(dataset, header + b"2,1000,reused\n2,1000,new_group\n")

    def test_metrics_keep_group_errors_distinct_and_relabel_without_replay(self):
        dataset = self.dataset()
        run = run_sweep(dataset, parse_thresholds("0.98"))
        decision_path = run / "threshold_0.98" / "decisions.csv"
        decisions = decision_path.read_bytes()
        path = ensure_labels(dataset)
        rows = read_csv(path)
        for row, label in zip(rows, ("new_group", "new_representative", "ambiguous",
                                     "new_group", "", "reused", "reused")):
            row["label"] = label
        write_csv(path, LABEL_COLUMNS, rows)
        first = evaluate_run(dataset, run)
        metrics, = read_csv(first / "metrics.csv")
        self.assertEqual([metrics[field] for field in ("tp", "fp", "fn", "tn", "group_mismatches",
                                                        "unlabeled_frames", "ambiguous_frames")], ["1"] * 7)
        self.assertEqual([metrics[field] for field in ("precision", "recall", "f1")], ["0.5"] * 3)
        confusion = read_csv(first / "confusion_matrix.csv")
        cell = next(row for row in confusion if row["actual"] == "new_group"
                    and row["predicted"] == "new_representative")
        self.assertEqual(cell["count"], "1")
        self.assertEqual([row["error"] for row in read_csv(first / "errors.csv")],
                         ["false_negative", "group_mismatch", "false_positive"])
        rows[1]["label"] = "reused"
        write_csv(path, LABEL_COLUMNS, rows)
        second = evaluate_run(dataset, run)
        self.assertNotEqual(first, second)
        self.assertEqual(read_csv(second / "metrics.csv")[0]["recall"], "1.0")
        self.assertEqual(decision_path.read_bytes(), decisions)

    def test_unlabeled_metrics_remain_empty(self):
        dataset = self.dataset()
        run = run_sweep(dataset, parse_thresholds("0.95"))
        output = evaluate_run(dataset, run)
        row, = read_csv(output / "metrics.csv")
        self.assertEqual(row["eligible_frames"], "6")
        self.assertEqual(row["labeled_frames"], "0")
        self.assertEqual(row["labeling_ratio"], "0.0")
        self.assertEqual([row[field] for field in ("precision", "recall", "f1")], [""] * 3)

    def test_candidate_selection_requires_configurable_label_coverage_and_support(self):
        dataset = self.dataset()
        run = run_sweep(dataset, parse_thresholds("0.98"))
        path = ensure_labels(dataset)
        rows = read_csv(path)
        rows[1]["label"], rows[2]["label"], rows[3]["label"] = "new_representative", "reused", "ambiguous"
        write_csv(path, LABEL_COLUMNS, rows)
        self.write_events(dataset, ("transient", 2, 2, "잠깐 보이는 정보"))

        default = evaluate_run(dataset, run)
        report = json.loads((default / "candidates.json").read_bytes())
        self.assertEqual(report["status"], "insufficient_frame_labels")
        self.assertEqual(report["candidates"], [])
        labeling = report["labeling"]
        self.assertEqual([labeling[key] for key in ("eligible_frames", "labeled_frames", "labeling_ratio",
                                                    "meaningful_change_labels", "reused_labels")],
                         [6, 2, 2 / 6, 1, 1])
        self.assertEqual(float(read_csv(default / "metrics.csv")[0]["labeling_ratio"]), 2 / 6)

        for requirements, unmet in (
            (LabelRequirements(3, 0, 1), ["labeled_frames"]),
            (LabelRequirements(2, 0.5, 1), ["labeling_ratio"]),
            (LabelRequirements(2, 0, 2), ["meaningful_change_labels", "reused_labels"]),
            (LabelRequirements(2, 2 / 6, 1), []),
        ):
            with self.subTest(requirements=requirements):
                output = evaluate_run(dataset, run, label_requirements=requirements)
                report = json.loads((output / "candidates.json").read_bytes())
                self.assertEqual(report["labeling"]["unmet"], unmet)
                self.assertEqual(report["candidates"], [] if unmet else ["0.98"])
                self.assertEqual(report["status"], "insufficient_frame_labels" if unmet
                                 else "evaluated_capture_candidates")

    def test_important_information_is_reviewed_separately_from_onset_detection(self):
        dataset = self.dataset()
        run = run_sweep(dataset, parse_thresholds("0.95,0.98,0.99"), write_representatives=True)
        label_file = ensure_labels(dataset)
        labels = read_csv(label_file)
        for row, label in zip(labels, ("new_group", "reused", "new_representative", "reused",
                                      "reused", "reused", "reused")):
            row["label"] = label
        write_csv(label_file, LABEL_COLUMNS, labels)
        events = ensure_events(dataset)
        write_csv(events, EVENT_COLUMNS, [
            {"event_id": "ink", "first_frame": 3, "last_frame": 4, "description": "새 필기"},
            {"event_id": "transient", "first_frame": 2, "last_frame": 2, "description": "잠깐 등장"},
            {"event_id": "detail", "first_frame": 6, "last_frame": 6, "description": "작은 수식"},
        ])
        smoke_requirements = LabelRequirements(2, 0.5, 1)
        first = evaluate_run(dataset, run, label_requirements=smoke_requirements)
        self.assertEqual(json.loads((first / "candidates.json").read_bytes())["status"], "review_pending")
        first_events = {(row["threshold"], row["event_id"]): row
                        for row in read_csv(first / "preservation_events.csv")}
        self.assertEqual(first_events["0.98", "ink"]["status"], "needs_review")
        self.assertEqual(first_events["0.98", "transient"]["reason"], "no_representative_in_window")
        evidence_path = first / "evidence_to_review.csv"
        checks = read_csv(evidence_path)
        self.assertEqual({(row["event_id"], row["frame_index"]) for row in checks},
                         {("ink", "3"), ("ink", "4"), ("detail", "6")})
        for row in checks:
            self.assertTrue((dataset.path / row["raw_image"]).is_file())
            self.assertTrue((run / row["representative_jpeg"]).is_file())
            row["verdict"] = "visible" if row["event_id"] == "ink" else "absent"
        write_csv(evidence_path, CHECK_COLUMNS, checks)
        second = evaluate_run(dataset, run, evidence_path, label_requirements=smoke_requirements)
        final = {(row["threshold"], row["event_id"]): row
                 for row in read_csv(second / "preservation_events.csv")}
        self.assertEqual(final["0.98", "ink"]["onset_stored"], "False")
        self.assertEqual(final["0.98", "ink"]["status"], "preserved")
        self.assertEqual(final["0.98", "ink"]["confirmed_delay_ms"], "1000")
        self.assertEqual(final["0.99", "ink"]["confirmed_delay_ms"], "0")
        self.assertEqual(final["0.99", "detail"]["reason"], "all_candidates_absent")
        self.assertEqual(final["0.95", "ink"]["status"], "not_preserved")
        report = json.loads((second / "candidates.json").read_bytes())
        self.assertEqual(report["status"], "evaluated_capture_candidates")
        self.assertEqual(report["best_preservation_thresholds"], ["0.98", "0.99"])
        self.assertNotIn("0.95", report["candidates"])

    def test_changed_events_are_invalidated_without_discarding_unchanged_reviews(self):
        dataset = self.dataset()
        run = run_sweep(dataset, parse_thresholds("0.98"))
        same = ("same", 3, 4, "유지할 필기")
        self.write_events(dataset, same, ("description", 6, 6, "이전 설명"),
                          ("range", 3, 4, "범위를 조정할 필기"), ("removed", 6, 6, "삭제할 이벤트"))
        original = self.mark_evidence(evaluate_run(dataset, run))
        original_bytes = original.read_bytes()
        self.write_events(dataset, ("range", 4, 5, "범위를 조정할 필기"),
                          ("description", 6, 6, "새 설명"), same)
        output = evaluate_run(dataset, run, original)
        rows = {row["event_id"]: row for row in read_csv(output / "evidence_to_review.csv")}
        self.assertEqual(rows["same"]["verdict"], "visible")
        self.assertEqual(rows["same"]["notes"], "기존 메모")
        self.assertEqual(rows["description"]["verdict"], "")
        self.assertEqual(rows["range"]["verdict"], "")  # The old representative is still in range.
        results = {row["event_id"]: row["status"] for row in read_csv(output / "preservation_events.csv")}
        self.assertEqual(results, {"same": "preserved", "description": "needs_review", "range": "needs_review"})
        rejected = {row["event_id"]: row for row in read_csv(output / "checks_requiring_review.csv")}
        self.assertEqual({name: row["reason"] for name, row in rejected.items()},
                         {"description": "event_changed", "range": "event_changed", "removed": "event_removed"})
        self.assertEqual(rejected["range"]["verdict"], "visible")
        self.assertEqual(rejected["range"]["notes"], "기존 메모")
        self.assertEqual((output / "checks_snapshot.csv").read_bytes(), original_bytes)
        self.assertEqual(original.read_bytes(), original_bytes)

    def test_checks_from_a_different_capture_are_not_reused(self):
        first, second = self.dataset(), self.dataset()
        event = ("ink", 3, 4, "같은 이벤트 정의")
        self.write_events(first, event)
        self.write_events(second, event)
        run = run_sweep(first, parse_thresholds("0.98"))
        original = self.mark_evidence(evaluate_run(first, run))
        output = evaluate_run(second, run_sweep(second, parse_thresholds("0.98")), original)
        self.assertEqual(read_csv(output / "preservation_events.csv")[0]["status"], "needs_review")
        self.assertEqual(read_csv(output / "checks_requiring_review.csv")[0]["reason"], "dataset_changed")

    def test_legacy_reviews_recover_identity_only_from_original_snapshots(self):
        dataset = self.dataset()
        run = run_sweep(dataset, parse_thresholds("0.98"))
        same = ("same", 3, 4, "유지할 이벤트")
        self.write_events(dataset, same, ("changed", 6, 6, "이전 설명"))
        old = self.mark_evidence(evaluate_run(dataset, run), legacy=True)
        self.write_events(dataset, ("changed", 6, 6, "새 설명"), same)
        output = evaluate_run(dataset, run, old)
        rows = {row["event_id"]: row for row in read_csv(output / "evidence_to_review.csv")}
        self.assertEqual(rows["same"]["verdict"], "visible")
        self.assertEqual(rows["same"]["dataset_sha256"], dataset.manifest_sha256)
        self.assertEqual(rows["changed"]["verdict"], "")
        self.assertEqual(read_csv(output / "checks_requiring_review.csv")[0]["reason"], "event_changed")
        detached = self.root / "legacy.csv"
        detached.write_bytes(old.read_bytes())
        output = evaluate_run(dataset, run, detached)
        self.assertEqual({row["reason"] for row in read_csv(output / "checks_requiring_review.csv")},
                         {"missing_identity"})

    def test_visual_reports_and_comparison_contain_threshold_data(self):
        dataset = self.dataset()
        run = run_sweep(dataset, parse_thresholds("0.99,0.98"))
        output = evaluate_run(dataset, run)
        chart_dir = output / "charts"
        ns = {"svg": "http://www.w3.org/2000/svg"}
        self.assertEqual(len(list(chart_dir.rglob("*.svg"))), 5)
        for file in chart_dir.rglob("*.svg"):
            root = ElementTree.parse(file).getroot()
            self.assertIsNotNone(root.find("svg:title", ns))
        no_labels = (chart_dir / "detection_scores.svg").read_text(encoding="utf-8")
        self.assertIn("No defined values", no_labels)
        rows = read_csv(compare_frame(dataset, run, 3))
        self.assertEqual([(row["threshold"], row["decision"]) for row in rows],
                         [("0.99", "new_representative"), ("0.98", "reused")])

    def test_cli_round_trip_without_desktop(self):
        stderr = StringIO()

        def command(*args):
            output = StringIO()
            with (patch.object(sys, "argv", ["experiments", *map(str, args)]),
                  redirect_stdout(output), redirect_stderr(stderr)):
                cli.main()
            return [Path(line) for line in output.getvalue().splitlines()]

        with patch.object(cli, "DEFAULT_ROOT", self.root):
            dataset, = command("capture", "--synthetic", "--name", "강의", "--seconds", 0.25,
                               "--region", -40, 0, 40, 32)
        run, = command("sweep", dataset, "--start-frame", 1, "--end-frame", 1,
                       "--range", "0.95", "0.96", "0.01", "--write-representatives")
        self.assertIn("Total 2/2 | 100.0%", stderr.getvalue())
        stderr.seek(0)
        stderr.truncate()
        quiet, = command("sweep", dataset, "--thresholds", "0.95", "--no-progress")
        self.assertTrue((quiet / "summary.csv").is_file())
        self.assertEqual(stderr.getvalue(), "")
        labels, events, preview = command("review", dataset, "--run", run, "--frame", 1)
        self.assertTrue(all(path.is_file() for path in (labels, events, preview)))
        comparison, = command("compare", dataset, run, "--frame", 1)
        self.assertEqual([row["decision"] for row in read_csv(comparison)], ["new_group"] * 2)
        report, = command("evaluate", dataset, run, "--min-labeled-frames", 3,
                          "--min-labeling-ratio", 0.25, "--min-labels-per-binary-class", 2)
        self.assertTrue((report / "charts" / "storage.svg").is_file())
        metadata = json.loads((report / "evaluation_metadata.json").read_bytes())
        self.assertEqual(metadata["label_requirements"], {"min_labeled_frames": 3, "min_labeling_ratio": 0.25,
                                                          "min_labels_per_binary_class": 2})
