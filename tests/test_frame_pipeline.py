from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from frame_processing.comparison import SSIMComparator
from frame_processing.pipeline import FramePipeline, FrameProcessingError
from frame_processing.storage import JPEGFrameStore
from interface_spec.contracts import CaptureRegion, FrameDecision as D, FrameSource as S, FrameType


class Clock:
    def __init__(self):
        self.value = 123

    def now_ms(self):
        return self.value


class Images:
    def __init__(self, *images):
        self.images = iter(images)
        self.regions = []

    def grab(self, region):
        self.regions.append(region)
        image = next(self.images)
        if isinstance(image, Exception):
            raise image
        return image


class FixedClassifier:
    def __init__(self, result=FrameType.TEXT):
        self.calls = 0
        self.result = result

    def classify(self, image):
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.clock = Clock()
        self.classifier = FixedClassifier()
        self.white = np.full((96, 128, 3), 255, np.uint8)

    def pipeline(self, *images, comparator=None, store=None):
        return FramePipeline("session", CaptureRegion(-128, -96, 128, 96), self.clock,
                             Images(*images), store or JPEGFrameStore(self.root),
                             comparator or SSIMComparator(.9), self.classifier)

    def test_reused_observations_keep_saved_metadata_and_new_time(self):
        p = self.pipeline(self.white, self.white, self.white)
        first = p.periodic()
        self.clock.value = 456
        second = p.periodic()
        third = p.periodic()
        self.assertEqual((first.decision, second.decision), (D.NEW_GROUP, D.REUSED))
        self.assertEqual((second.frame_id, second.image_path, second.frame_type),
                         (first.frame_id, first.image_path, first.frame_type))
        self.assertEqual([first.capture_seq, second.capture_seq, third.capture_seq], [1, 2, 3])
        self.assertEqual(second.timestamp_ms, 456)
        self.assertEqual(len(list((self.root / "frames").glob("*.jpg"))), 1)
        self.assertEqual(self.classifier.calls, 1)

    def test_snapshots_are_independent_and_do_not_change_periodic_state(self):
        black = np.zeros_like(self.white)
        p = self.pipeline(self.white, black, black, self.white)
        first = p.periodic()
        start = p.snapshot(S.QUESTION_START)
        submit = p.snapshot(S.QUESTION_SUBMIT)
        last = p.periodic()
        self.assertEqual(last.decision, D.REUSED)
        self.assertEqual(last.frame_id, first.frame_id)
        self.assertEqual(len({first.frame_id, start.frame_id, submit.frame_id}), 3)
        self.assertEqual([first.capture_seq, start.capture_seq, submit.capture_seq, last.capture_seq], [1, 2, 3, 4])
        for snapshot in [start, submit]:
            self.assertIsNone(snapshot.group_id)
            self.assertEqual(snapshot.decision, D.QUESTION_SNAPSHOT)
            self.assertEqual(snapshot.frame_type, FrameType.UNKNOWN)

    def test_real_ssim_accumulation_and_previous_slide_return(self):
        # Small cumulative changes: adjacent pairs stay similar, R eventually differs.
        slides = []
        for count in range(7):
            image = self.white.copy()
            for i in range(count):
                cv2.rectangle(image, (8+i*17, 15), (17+i*17, 80), (0, 0, 0), -1)
            slides.append(image)
        comp = SSIMComparator(.8)
        p = self.pipeline(*slides, self.white, comparator=comp)
        records = [p.periodic() for _ in slides]
        self.assertIn(D.NEW_REPRESENTATIVE, [r.decision for r in records])
        self.assertEqual(len({r.group_id for r in records}), 1)
        self.assertEqual(p.periodic().decision, D.NEW_GROUP)
        for record in records:
            self.assertTrue((self.root / record.image_path).is_file())

    def test_case_four_uses_new_group_even_when_representative_matches(self):
        comp = SSIMComparator(.9)
        # Explicit scores isolate Case 4, whose score geometry is not guaranteed
        # for an arbitrary synthetic pixel sequence.
        scores = iter([.95, .95, .2, 1.0])
        comp.score = lambda a, b: next(scores)
        p = self.pipeline(self.white, self.white, self.white, comparator=comp)
        first, reused, returned = p.periodic(), p.periodic(), p.periodic()
        self.assertEqual(reused.decision, D.REUSED)
        self.assertEqual(returned.decision, D.NEW_GROUP)
        self.assertNotEqual(first.group_id, returned.group_id)
        self.assertNotEqual(first.frame_id, returned.frame_id)

    def test_timestamp_is_captured_before_classification_and_storage(self):
        p = self.pipeline(self.white)
        def classify(image):
            self.clock.value = 9999
            return FrameType.VISUAL
        self.classifier.classify = classify
        self.assertEqual(p.periodic().timestamp_ms, 123)

    def test_classifier_failure_abstains(self):
        self.classifier.result = ValueError("unrecognized")
        p = self.pipeline(self.white)
        with self.assertLogs("frame_processing.pipeline", level="WARNING"):
            self.assertEqual(p.periodic().frame_type, FrameType.UNKNOWN)

    def test_save_failure_does_not_advance_periodic_state(self):
        class FailingStore:
            def save(self, frame_id, image):
                raise OSError("disk full")
        p = self.pipeline(self.white, self.white, store=FailingStore())
        with self.assertRaises(FrameProcessingError) as error:
            p.periodic()
        self.assertEqual(error.exception.code, "SAVE_FAILED")
        p.store = JPEGFrameStore(self.root)
        recovered = p.periodic()
        self.assertEqual(recovered.decision, D.NEW_GROUP)
        self.assertEqual(recovered.capture_seq, 2)

    def test_capture_error_and_wrong_shape(self):
        for value in [OSError("capture unavailable"), self.white[:10]]:
            with self.subTest(value=type(value)), self.assertRaises(FrameProcessingError) as error:
                self.pipeline(value).periodic()
            self.assertEqual(error.exception.code, "CAPTURE_FAILED")

    def test_concurrent_same_millisecond_snapshots_have_unique_sequences(self):
        p = self.pipeline(*([self.white] * 16))
        with ThreadPoolExecutor(max_workers=8) as executor:
            records = list(executor.map(lambda _: p.snapshot(S.QUESTION_START), range(16)))
        self.assertEqual(len({r.capture_seq for r in records}), 16)
        self.assertEqual(len({r.frame_id for r in records}), 16)
        self.assertEqual({r.timestamp_ms for r in records}, {123})

    def test_region_change_starts_new_group_without_resetting_sequence(self):
        p = self.pipeline(self.white, self.white)
        first = p.periodic()
        p.set_region(CaptureRegion(0, 0, 128, 96))
        second = p.periodic()
        self.assertEqual(second.decision, D.NEW_GROUP)
        self.assertNotEqual(second.group_id, first.group_id)
        self.assertEqual(second.capture_seq, 2)

    def test_periodic_source_is_rejected_for_snapshot(self):
        with self.assertRaises(ValueError):
            self.pipeline(self.white).snapshot(S.PERIODIC)
