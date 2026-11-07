"""Contract and lifecycle checks for the initial desktop integration."""

from __future__ import annotations

import tempfile
import unittest
from inspect import signature
from pathlib import Path
from threading import Event

from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer

from core import SessionManager
from desktop_app.adapters import (
    MockAudioProcessor,
    MockContextQA,
    MockFrameProcessor,
    enqueue_event,
    protocol_adapters,
)
from desktop_app.controller import DesktopController
from desktop_app.controller import default_controller
from interface_spec.contracts import (
    AudioProcessorAPI,
    CaptureRegion,
    ContextQAAPI,
    FrameProcessorAPI,
    SessionState,
)


class DesktopControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QCoreApplication.instance() or QCoreApplication([])

    def make_controller(self, frame=None, audio=None, context=None):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        frame = frame or MockFrameProcessor()
        audio = audio or MockAudioProcessor()
        context = context or MockContextQA()
        controller = DesktopController(
            frame,
            audio,
            context,
            SessionManager(Path(directory.name) / "sessions"),
        )
        self.addCleanup(controller.deleteLater)
        return controller, frame, audio, context

    def wait_operation(self, controller: DesktopController, operation: str) -> None:
        loop = QEventLoop()
        poll = QTimer()
        poll.setInterval(5)
        poll.timeout.connect(
            lambda: loop.quit()
            if controller._pending_operation != operation
            else None
        )
        poll.start()
        QTimer.singleShot(3000, loop.quit)
        if controller._pending_operation == operation:
            loop.exec()
        poll.stop()
        self.assertIsNone(controller._pending_operation)

    def test_mock_adapters_match_protocol_method_names_and_parameter_order(self) -> None:
        frame, audio, context = protocol_adapters()
        for instance, protocol in (
            (frame, FrameProcessorAPI),
            (audio, AudioProcessorAPI),
            (context, ContextQAAPI),
        ):
            for name in protocol.__dict__:
                if name.startswith("_"):
                    continue
                expected = getattr(protocol, name, None)
                actual = getattr(type(instance), name, None)
                if callable(expected):
                    self.assertTrue(callable(actual), name)
                    self.assertEqual(
                        list(signature(expected).parameters),
                        list(signature(actual).parameters),
                        name,
                    )
                    self.assertEqual(
                        signature(expected).return_annotation,
                        signature(actual).return_annotation,
                        f"{name} return type",
                    )

    def test_controller_matches_app_controller_protocol(self) -> None:
        controller = default_controller()
        self.addCleanup(controller.deleteLater)
        from interface_spec.contracts import AppControllerAPI

        for name, expected in AppControllerAPI.__dict__.items():
            if name.startswith("_") or not callable(expected):
                continue
            actual = getattr(type(controller), name)
            self.assertEqual(list(signature(expected).parameters), list(signature(actual).parameters))
            self.assertEqual(signature(expected).return_annotation, signature(actual).return_annotation)

    def test_session_start_propagates_one_shared_resource_set(self) -> None:
        controller, frame, audio, context = self.make_controller()
        session_id = controller.start_session(CaptureRegion(10, 20, 800, 600), "dev-1")
        self.wait_operation(controller, "start")
        session = controller.session
        self.assertEqual(controller.state, SessionState.RUNNING)
        self.assertEqual(session_id, session.session_id)
        self.assertEqual(frame.session_id, audio.session_id)
        self.assertEqual(frame.session_id, context.session_id)
        self.assertEqual(frame.session_dir, str(session.session_dir))
        self.assertEqual(context.session_dir, str(session.session_dir))
        self.assertIs(frame.clock, session.clock)
        self.assertIs(audio.clock, session.clock)
        self.assertIs(frame.output_queue, session.output_queue)
        self.assertIs(audio.output_queue, session.output_queue)
        self.assertIs(context.input_queue, session.output_queue)
        with self.assertRaises(RuntimeError):
            controller.start_session(CaptureRegion(0, 0, 10, 10), "dev-1")

    def test_pause_resume_and_stop_follow_contract_state_order(self) -> None:
        controller, _frame, _audio, context = self.make_controller()
        controller.start_session(CaptureRegion(0, 0, 10, 10), "default")
        self.wait_operation(controller, "start")
        before_pause = controller.session.clock.now_ms()
        controller.pause_session()
        self.assertEqual(controller.state, SessionState.PAUSING)
        self.wait_operation(controller, "pause")
        self.assertEqual(controller.state, SessionState.PAUSED)
        frozen = controller.session.clock.now_ms()
        self.assertGreaterEqual(frozen, before_pause)
        controller.resume_session()
        self.wait_operation(controller, "resume")
        self.assertEqual(controller.state, SessionState.RUNNING)
        controller.stop_session()
        self.wait_operation(controller, "stop")
        self.assertEqual(controller.state, SessionState.IDLE)
        self.assertIsNone(controller.session_manager.active_session)
        self.assertFalse(context.running)

    def test_context_drain_finishes_before_session_reference_is_released(self) -> None:
        order: list[str] = []

        class TrackingManager(SessionManager):
            def finish_session(self, session_id: str) -> None:
                order.append("finish_session")
                super().finish_session(session_id)

        class TrackingContext(MockContextQA):
            def drain_and_stop(self) -> None:
                order.append("drain_and_stop")
                super().drain_and_stop()

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        context = TrackingContext()
        controller = DesktopController(
            MockFrameProcessor(),
            MockAudioProcessor(),
            context,
            TrackingManager(Path(directory.name) / "sessions"),
        )
        self.addCleanup(controller.deleteLater)
        controller.start_session(CaptureRegion(0, 0, 10, 10), "default")
        self.wait_operation(controller, "start")
        controller.stop_session()
        self.wait_operation(controller, "stop")
        self.assertEqual(order, ["drain_and_stop", "finish_session"])

    def test_stop_queued_during_pause_wins_final_state(self) -> None:
        class DelayedPauseFrame(MockFrameProcessor):
            def pause(self) -> None:
                Event().wait(0.05)
                super().pause()

        controller, _frame, _audio, _context = self.make_controller(frame=DelayedPauseFrame())
        controller.start_session(CaptureRegion(0, 0, 10, 10), "default")
        self.wait_operation(controller, "start")
        controller.pause_session()
        controller.stop_session()
        self.assertEqual(controller.state, SessionState.STOPPING)
        loop = QEventLoop()
        controller.state_changed.connect(
            lambda state: loop.quit() if state == SessionState.IDLE else None
        )
        QTimer.singleShot(3000, loop.quit)
        loop.exec()
        self.assertEqual(controller.state, SessionState.IDLE)

    def test_partial_initialization_failure_rolls_back_started_modules(self) -> None:
        class FailingAudio(MockAudioProcessor):
            def start(self, session_id, clock, output_queue, on_error) -> None:
                raise RuntimeError("audio initialization failed")

        controller, frame, _audio, context = self.make_controller(audio=FailingAudio())
        failed = []
        controller.error_raised.connect(failed.append)
        controller.start_session(CaptureRegion(0, 0, 10, 10), "default")
        loop = QEventLoop()
        controller.error_raised.connect(lambda _message: loop.quit())
        QTimer.singleShot(3000, loop.quit)
        loop.exec()
        self.assertEqual(controller.state, SessionState.ERROR)
        self.assertIsNone(controller.session)
        self.assertIsNone(controller.session_manager.active_session)
        self.assertFalse(frame.running)
        self.assertFalse(context.running)
        self.assertTrue(any("audio initialization failed" in item for item in failed))

    def test_queued_stop_completes_after_pause_failure(self) -> None:
        class FailingPauseFrame(MockFrameProcessor):
            def pause(self) -> None:
                Event().wait(0.03)
                raise RuntimeError("pause failed")

        controller, _frame, _audio, _context = self.make_controller(frame=FailingPauseFrame())
        controller.start_session(CaptureRegion(0, 0, 10, 10), "default")
        self.wait_operation(controller, "start")
        controller.pause_session()
        controller.stop_session()
        loop = QEventLoop()
        timer = QTimer()
        timer.setInterval(5)
        timer.timeout.connect(
            lambda: loop.quit() if controller.state == SessionState.IDLE else None
        )
        timer.start()
        QTimer.singleShot(3000, loop.quit)
        loop.exec()
        timer.stop()
        self.assertEqual(controller.state, SessionState.IDLE)
        self.assertIsNone(controller.session_manager.active_session)

    def test_gui_event_loop_keeps_processing_while_start_is_blocked(self) -> None:
        entered = Event()

        class SlowFrame(MockFrameProcessor):
            def start(self, session_id, session_dir, clock, output_queue, on_error) -> None:
                entered.set()
                Event().wait(0.1)
                super().start(session_id, session_dir, clock, output_queue, on_error)

        controller, *_ = self.make_controller(frame=SlowFrame())
        timer_fired = []
        controller.start_session(CaptureRegion(0, 0, 10, 10), "default")
        QTimer.singleShot(10, lambda: timer_fired.append(True))
        self.wait_operation(controller, "start")
        self.assertTrue(entered.is_set())
        self.assertTrue(timer_fired)

    def test_bounded_queue_reports_full_to_caller_without_dropping_silently(self) -> None:
        from queue import Queue
        from interface_spec.contracts import FrameDecision, FrameRecord, FrameSource, FrameType

        queue = Queue(maxsize=1)
        event = FrameRecord(
            "s", "f", 1, 1, "frames/f.jpg", FrameType.UNKNOWN,
            FrameSource.PERIODIC, "g", FrameDecision.NEW_GROUP,
        )
        self.assertTrue(enqueue_event(queue, event))
        self.assertFalse(enqueue_event(queue, event))
        self.assertEqual(queue.qsize(), 1)


if __name__ == "__main__":
    unittest.main()
