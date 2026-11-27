from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np

from frame_processing.experiments.capture import capture_dataset, load_dataset
from frame_processing.experiments.classification_data import LABEL_COLUMNS, LectureSource, Sample, prepare, read_image
from frame_processing.experiments.labeling_session import LabelingSession, editing_lock
from frame_processing.experiments.replay import ThresholdPair, read_csv, run_sweep, write_csv
from interface_spec.contracts import CaptureRegion, FrameDecision, FrameType


class RepresentativeProjectTests(unittest.TestCase):
    def test_loads_all_saved_candidates_with_original_identity_and_png(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            region = CaptureRegion(0, 0, 24, 16)
            backend = Mock()
            values = (255, 100, 100, 125, 125, 0)
            backend.grab.side_effect = [np.full((16, 24, 3), value, np.uint8) for value in values]
            elapsed = 0

            def wait(seconds):
                nonlocal elapsed
                elapsed += round(seconds * 1_000_000_000)

            source = capture_dataset(root, "lecture", region, len(values), backend,
                                     now_ns=lambda: elapsed, wait=wait)
            dataset = load_dataset(source)
            single = run_sweep(dataset, (Decimal(".95"),), start_frame=2)
            dual = run_sweep(dataset, (ThresholdPair(Decimal(".90"), Decimal(".98")),), start_frame=2)
            original_files = {path: path.read_bytes() for path in source.rglob("*") if path.is_file()}
            output = root / "labeling"
            rows = prepare(output, [LectureSource("static", source, single, "0.95"),
                                    LectureSource("animation", source, dual, "g0.9_r0.98")], count=None)
            self.assertEqual([(row.lecture, row.frame_index) for row in rows],
                             [("static", 2), ("static", 6), ("animation", 2),
                              ("animation", 4), ("animation", 6)])
            self.assertEqual(rows[3].decision, FrameDecision.NEW_REPRESENTATIVE)
            self.assertTrue(all(row.selection == "all_saved" for row in rows))
            self.assertEqual(rows[3].timestamp_ms, 3000)
            self.assertTrue(rows[3].image_path.endswith("/raw_frames/frame_000004.png"))
            self.assertTrue(np.all(read_image(output, rows[3]) == 125))
            self.assertTrue(all(path.read_bytes() == data for path, data in original_files.items()))
            with self.assertRaises(FileExistsError):
                prepare(output, [LectureSource("static", source, single, "0.95")], count=None)


class LabelingSessionTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output = Path(directory.name)
        samples = [Sample(f"{i:03d}", lecture, index, 0, "new_group", "all_saved",
                          "frame.png", "hash", "sweep", "0.95")
                   for i, (lecture, index) in enumerate(
                       (("static", 2), ("static", 8), ("animation", 2), ("animation", 6)), 1)]
        write_csv(self.output / "manifest.csv", list(asdict(samples[0])), map(asdict, samples))
        self.labels = self.output / "labels.csv"
        write_csv(self.labels, LABEL_COLUMNS, [
            {"sample_id": sample.sample_id, "lecture": sample.lecture, "frame_index": sample.frame_index,
             "notes": "검토 메모" if sample.sample_id == "001" else ""}
            for sample in samples
        ])

    def test_assign_edit_cross_lecture_resume_and_selection(self):
        session = LabelingSession(self.output)
        self.assertEqual((session.current.lecture, session.current.frame_index), ("static", 2))
        self.assertEqual(session.progress, (0, 4))
        session.assign(FrameType.TEXT)
        self.assertEqual(read_csv(self.labels)[0]["label"], "Text")
        self.assertEqual(session.current.frame_index, 8)
        self.assertTrue(session.move(-1))
        self.assertEqual(session.current_label, "Text")
        session.assign(FrameType.VISUAL)
        session.assign(FrameType.UNKNOWN)
        self.assertEqual((session.current.lecture, session.current.frame_index), ("animation", 2))
        session.save_cursor()
        reopened = LabelingSession(self.output)
        self.assertEqual(reopened.current, session.current)
        edited = read_csv(self.labels)[0]
        self.assertEqual((edited["label"], edited["notes"]), ("Visual", "검토 메모"))
        reopened.select("animation")
        reopened.assign(FrameType.TEXT)
        reopened.save_cursor()
        reopened = LabelingSession(self.output)
        self.assertEqual((reopened.scope, reopened.current.frame_index), ("animation", 6))
        self.assertEqual(reopened.progress, (1, 2))
        self.assertEqual(reopened.total_progress, (3, 4))
        reopened.assign(FrameType.UNKNOWN)
        self.assertEqual(reopened.progress, (2, 2))
        self.assertFalse(reopened.move(1))

    def test_last_frame_jumps_to_pending_and_corrupt_cursor_can_resume(self):
        session = LabelingSession(self.output)
        session.select("animation")
        session.move(1)
        session.assign(FrameType.VISUAL)
        self.assertEqual(session.current.frame_index, 2)
        (self.output / "labeling_state.json").write_text("invalid JSON", encoding="utf-8")
        resumed = LabelingSession(self.output)
        self.assertEqual(resumed.total_progress, (1, 4))
        self.assertEqual(resumed.current.sample_id, "001")

    def test_failed_write_or_changed_label_sheet_preserves_position(self):
        session = LabelingSession(self.output)
        previous = self.labels.read_bytes()
        with patch("frame_processing.experiments.labeling_session._replace", side_effect=OSError("locked")):
            with self.assertRaises(OSError):
                session.assign(FrameType.TEXT)
        self.assertEqual(self.labels.read_bytes(), previous)
        self.assertEqual(session.current.sample_id, "001")
        self.assertEqual(session.current_label, "")
        replacement = previous.replace(b"001,static,2,", b"001,static,2,Unknown")
        self.labels.write_bytes(replacement)
        with self.assertRaises(OSError):
            session.assign(FrameType.TEXT)
        self.assertEqual(self.labels.read_bytes(), replacement)
        self.assertEqual(session.current.sample_id, "001")
        self.assertEqual(LabelingSession(self.output).progress, (1, 4))

    def test_two_labeling_windows_cannot_write_to_the_same_project(self):
        with editing_lock(self.output):
            with self.assertRaises(OSError), editing_lock(self.output):
                self.fail("second editor obtained the lock")
        with editing_lock(self.output):
            pass
