from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from inspect import signature
from queue import Empty, Queue
from threading import Event
import tempfile
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from core import SessionManager
from frame_processing import FrameProcessor, FrameProcessingError
from frame_processing.classification import LayoutHeuristicClassifier
from interface_spec.contracts import (
    CaptureRegion, FrameDecision, FrameProcessorAPI, FrameRecord, FrameSource, FrameType,
    ModuleName,
)


class Backend:
    def __init__(self):
        self.image = np.full((120, 160, 3), 255, np.uint8)
        self.calls = 0
        self.block = False
        self.entered = Event()
        self.release = Event()
        self.regions = []

    def grab(self, region):
        self.calls += 1
        self.regions.append(region)
        if self.block:
            self.entered.set()
            if not self.release.wait(3):
                raise TimeoutError("test did not release backend")
        return self.image.copy()


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.manager = SessionManager(self.temp.name)
        self.session = self.manager.start_session()
        self.backend = Backend()
        self.errors = Queue()
        self.processor = FrameProcessor(ssim_threshold=.9, backend=self.backend,
                                        interval_s=.03, queue_timeout_s=.02)
        self.processor.set_region(CaptureRegion(-160, -120, 160, 120))
        self.addCleanup(self.cleanup_processor)

    def cleanup_processor(self):
        self.backend.release.set()
        # A failing saturation test must not deadlock the cleanup path.
        self.processor.pause()
        while True:
            try:
                self.session.output_queue.get_nowait()
            except Empty:
                break
        self.processor.stop()

    def start(self, callback=None):
        s = self.session
        self.processor.start(s.session_id, str(s.session_dir), s.clock, s.output_queue,
                             callback or self.errors.put)

    def test_contract_signatures_match(self):
        for name in ["set_region", "start", "pause", "prepare_resume", "resume", "stop", "capture_now"]:
            self.assertEqual(list(signature(getattr(FrameProcessor, name)).parameters),
                             list(signature(getattr(FrameProcessorAPI, name)).parameters))

    def test_periodic_worker_reuses_files_and_emits_every_observation(self):
        self.start()
        records = [self.session.output_queue.get(timeout=2) for _ in range(3)]
        self.processor.pause()
        self.assertEqual([r.capture_seq for r in records], [1, 2, 3])
        self.assertEqual([r.decision for r in records],
                         [FrameDecision.NEW_GROUP, FrameDecision.REUSED, FrameDecision.REUSED])
        self.assertEqual(len({r.frame_id for r in records}), 1)
        self.assertTrue(all(isinstance(r, FrameRecord) for r in records))
        self.assertLess(records[0].timestamp_ms, records[-1].timestamp_ms)
        self.assertTrue(self.errors.empty())

    def test_pause_drains_capture_before_clock_pause_and_resume_keeps_ids(self):
        self.backend.block = True
        self.start()
        self.assertTrue(self.backend.entered.wait(2))
        with ThreadPoolExecutor(1) as executor:
            paused = executor.submit(self.processor.pause)
            time.sleep(.02)
            self.assertFalse(paused.done())
            self.backend.block = False
            self.backend.release.set()
            paused.result(timeout=2)
        first = self.session.output_queue.get(timeout=1)
        before = self.backend.calls
        self.session.clock.pause()
        frozen = self.session.clock.now_ms()
        time.sleep(.07)
        self.assertEqual(self.backend.calls, before)
        self.assertEqual(self.session.clock.now_ms(), frozen)
        self.processor.prepare_resume()
        self.session.clock.resume()
        self.processor.resume()
        resumed = self.session.output_queue.get(timeout=2)
        self.assertGreater(resumed.capture_seq, first.capture_seq)
        self.assertEqual(resumed.frame_id, first.frame_id)

    def test_stop_drains_inflight_capture_and_emits_before_return(self):
        self.backend.block = True
        self.start()
        self.assertTrue(self.backend.entered.wait(2))
        with ThreadPoolExecutor(1) as executor:
            stopped = executor.submit(self.processor.stop)
            time.sleep(.02)
            self.assertFalse(stopped.done())
            self.backend.release.set()
            stopped.result(timeout=2)
        record = self.session.output_queue.get_nowait()
        self.assertTrue((self.session.session_dir / record.image_path).exists())
        count = self.backend.calls
        time.sleep(.05)
        self.assertEqual(self.backend.calls, count)
        self.processor.stop()

    def test_question_snapshot_bypasses_queue_and_does_not_change_periodic_state(self):
        # A long interval isolates the snapshots from the second periodic tick.
        self.processor._interval = 30
        self.start()
        first = self.session.output_queue.get(timeout=2)
        start = self.processor.capture_now(FrameSource.QUESTION_START)
        submit = self.processor.capture_now(FrameSource.QUESTION_SUBMIT)
        self.assertTrue(self.session.output_queue.empty())
        self.assertEqual(len({first.frame_id, start.frame_id, submit.frame_id}), 3)
        self.assertEqual([first.capture_seq, start.capture_seq, submit.capture_seq], [1, 2, 3])

    def test_question_acquisition_can_run_during_periodic_classification(self):
        entered, release = Event(), Event()
        class BlockingClassifier:
            def classify(self, image):
                entered.set()
                if not release.wait(2):
                    raise TimeoutError("classifier gate not released")
                return FrameType.TEXT
        self.processor._classifier = BlockingClassifier()
        try:
            self.start()
            self.assertTrue(entered.wait(1))
            record = self.processor.capture_now(FrameSource.QUESTION_SUBMIT)
            self.assertEqual(record.capture_seq, 2)
            self.assertTrue(self.session.output_queue.empty())
        finally:
            release.set()

    def test_queue_saturation_halts_and_prepare_retries_exactly_once(self):
        queue = self.session.output_queue
        queue.maxsize = 1
        self.start()
        error = self.errors.get(timeout=2)
        self.processor.pause()
        self.assertEqual(error.code, "QUEUE_FULL")
        self.assertEqual(error.module, ModuleName.FRAME)
        self.assertTrue(error.recoverable)
        pending = self.processor.pending_record
        self.assertIsNotNone(pending)
        count = self.backend.calls
        time.sleep(.06)
        self.assertEqual(self.backend.calls, count)
        with self.assertRaises(FrameProcessingError):
            self.processor.prepare_resume()
        first = queue.get_nowait()
        self.processor.prepare_resume()
        retried = queue.get_nowait()
        self.assertEqual(retried, pending)
        self.assertNotEqual(first.capture_seq, retried.capture_seq)
        self.assertIsNone(self.processor.pending_record)
        self.processor.prepare_resume()
        self.assertTrue(queue.empty())

    def test_stop_reports_undelivered_record_then_can_be_retried(self):
        queue = self.session.output_queue
        queue.maxsize = 1
        self.start()
        self.errors.get(timeout=2)
        with self.assertRaises(FrameProcessingError):
            self.processor.stop()
        pending = self.processor.pending_record
        queue.get_nowait()
        self.processor.stop()
        self.assertEqual(queue.get_nowait(), pending)
        self.assertIsNone(self.processor.pending_record)

    def test_capture_failure_reports_callback_without_queue_record(self):
        self.backend.grab = lambda region: (_ for _ in ()).throw(OSError("capture failed"))
        self.start()
        error = self.errors.get(timeout=2)
        self.assertEqual(error.code, "CAPTURE_FAILED")
        self.assertFalse(error.recoverable)
        self.assertTrue(self.session.output_queue.empty())
        with self.assertRaises(RuntimeError):
            self.processor.prepare_resume()

    def test_storage_failure_reports_callback_without_queue_record(self):
        with patch("frame_processing.processor.JPEGFrameStore.save", side_effect=OSError("disk full")):
            self.start()
            error = self.errors.get(timeout=2)
            self.assertEqual(error.code, "SAVE_FAILED")
            self.assertTrue(self.session.output_queue.empty())

    def test_invalid_transitions_and_region_change_policy(self):
        with self.assertRaises(RuntimeError):
            self.processor.resume()
        self.start()
        self.session.output_queue.get(timeout=2)
        with self.assertRaises(RuntimeError):
            self.processor.set_region(CaptureRegion(0, 0, 160, 120))
        with self.assertRaises(ValueError):
            self.processor.capture_now(FrameSource.PERIODIC)
        self.processor.pause()
        with self.assertRaises(RuntimeError):
            self.processor.capture_now(FrameSource.QUESTION_START)
        self.processor.set_region(CaptureRegion(0, 0, 160, 120))
        self.processor.prepare_resume()
        self.session.clock.resume()
        self.processor.resume()
        # Drain a possible final old-region observation before looking for the new group.
        records = []
        while not records or records[-1].decision != FrameDecision.NEW_GROUP:
            records.append(self.session.output_queue.get(timeout=2))
        self.assertEqual(self.backend.regions[-1].left, 0)

    def test_callback_can_stop_worker_without_self_join(self):
        self.backend.grab = lambda region: (_ for _ in ()).throw(OSError("capture failed"))
        completed = Event()
        def callback(error):
            self.processor.stop()
            completed.set()
        self.start(callback)
        self.assertTrue(completed.wait(2))
        self.processor._thread.join(timeout=2)
        self.assertFalse(self.processor._thread.is_alive())

    def test_snapshot_failure_reports_error_and_halts_new_captures(self):
        self.processor._interval = 30
        self.start()
        self.session.output_queue.get(timeout=2)
        self.backend.grab = lambda region: (_ for _ in ()).throw(OSError("snapshot failed"))
        with self.assertRaises(FrameProcessingError):
            self.processor.capture_now(FrameSource.QUESTION_SUBMIT)
        error = self.errors.get(timeout=1)
        self.assertEqual((error.code, error.session_id), ("CAPTURE_FAILED", self.session.session_id))
        self.assertTrue(self.session.output_queue.empty())
        with self.assertRaises(RuntimeError):
            self.processor.capture_now(FrameSource.QUESTION_START)

    def test_pause_waits_for_admitted_question_snapshot(self):
        self.processor._interval = 30
        self.start()
        first = self.session.output_queue.get(timeout=2)
        self.backend.block = True
        with ThreadPoolExecutor(2) as executor:
            snapshot = executor.submit(self.processor.capture_now, FrameSource.QUESTION_START)
            self.assertTrue(self.backend.entered.wait(1))
            paused = executor.submit(self.processor.pause)
            time.sleep(.02)
            self.assertFalse(paused.done())
            self.backend.release.set()
            record = snapshot.result(timeout=2)
            paused.result(timeout=2)
        self.assertGreater(record.capture_seq, first.capture_seq)
        self.assertTrue((self.session.session_dir / record.image_path).exists())
        self.assertTrue(self.session.output_queue.empty())

    def test_callback_exception_does_not_prevent_stop(self):
        self.backend.grab = lambda region: (_ for _ in ()).throw(OSError("capture failed"))
        notified = Event()
        def callback(error):
            notified.set()
            raise RuntimeError("UI adapter failure")
        with self.assertLogs("frame_processing.processor", level="ERROR"):
            self.start(callback)
            self.assertTrue(notified.wait(1))
            self.processor.stop()
        self.assertFalse(self.processor._thread.is_alive())

    def test_start_refuses_existing_session_images(self):
        (self.session.session_dir / "frames" / "past.jpg").write_bytes(b"preserve")
        with self.assertRaises(ValueError):
            self.start()
        self.assertEqual((self.session.session_dir / "frames" / "past.jpg").read_bytes(), b"preserve")


class ClassifierTests(unittest.TestCase):
    def test_text_visual_unknown_and_small_logo(self):
        classifier = LayoutHeuristicClassifier()
        text = np.full((360, 640, 3), 255, np.uint8)
        for y in [50, 100, 150]:
            cv2.putText(text, "Linear models and code example", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 0), 1)
        self.assertEqual(classifier.classify(text), FrameType.TEXT)
        logo = text.copy()
        cv2.rectangle(logo, (590, 15), (620, 40), (20, 40, 150), -1)
        self.assertEqual(classifier.classify(logo), FrameType.TEXT)
        visual = text.copy()
        cv2.rectangle(visual, (200, 190), (580, 330), (0, 0, 0), 2)
        cv2.line(visual, (200, 330), (570, 200), (0, 0, 0), 3)
        self.assertEqual(classifier.classify(visual), FrameType.VISUAL)
        self.assertEqual(classifier.classify(np.full_like(text, 255)), FrameType.UNKNOWN)
