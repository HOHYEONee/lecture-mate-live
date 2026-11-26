from __future__ import annotations

import os
import tempfile
import threading
import unittest
from queue import Queue
from unittest.mock import patch

from PySide6.QtCore import QEventLoop, QPoint, QTimer, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from audio_processing import AudioProcessor
from audio_processing.backends import FasterWhisperTranscriber
from audio_processing.chunks import AudioBlock, SpeechSpan
from context_qa import ContextQAService, FakeEmbedder, FakeLLM, create_context_qa
from core import SessionManager
from desktop_app.controller import DesktopController, production_controller
from desktop_app.disabled import DisabledAudioProcessor
from desktop_app.main_window import MainWindow
from frame_processing import FrameProcessor
from frame_processing.capture import WindowsWindowCapture
from frame_processing.classification import PlusLClassifier
from frame_processing.experiments.heuristic import LayoutHeuristicClassifier
from frame_processing.synthetic import SyntheticCapture
from interface_spec.contracts import (
    AnswerDelta,
    AnswerResult,
    CaptureRegion,
    FrameSource,
    FrameType,
    TranscriptSegment,
)
from rag import HybridRAGIndex


class ModuleIntegrationTests(unittest.TestCase):
    def test_production_controller_defaults_to_gpu_without_loading_model(self) -> None:
        with patch.dict(os.environ, clear=True), patch.object(PlusLClassifier, "__init__") as init:
            controller = production_controller()
        self.addCleanup(controller.deleteLater)

        self.assertIs(controller.frame_processor._classifier_factory, PlusLClassifier)
        self.assertIsInstance(controller.window_capture_backend, WindowsWindowCapture)
        self.assertIs(controller.frame_processor._backend, controller.window_capture_backend)
        self.assertIsInstance(controller.audio_processor, DisabledAudioProcessor)
        init.assert_not_called()

    def test_production_controller_composes_contract_modules_without_loading_models(self) -> None:
        values = {
            "CLUE_FRAME_CLASSIFIER": "cpu",
            "CLUE_ASR_DEVICE": "cpu",
            "CLUE_ENABLE_AUDIO": "1",
            "CLUE_USE_KO_RERANKER": "0",
            "CLUE_FAKE_LLM": "1",
        }
        with patch.dict(os.environ, values):
            controller = production_controller()
        self.addCleanup(controller.deleteLater)

        self.assertIsInstance(controller, DesktopController)
        self.assertIsInstance(controller.frame_processor, FrameProcessor)
        self.assertIs(controller.frame_processor._classifier_factory, LayoutHeuristicClassifier)
        self.assertIsInstance(controller.audio_processor, AudioProcessor)
        self.assertIsInstance(controller.audio_processor._transcriber, FasterWhisperTranscriber)
        self.assertEqual(controller.audio_processor._transcriber.device, "cpu")
        self.assertIsInstance(controller.context_qa._rag, HybridRAGIndex)

        with patch.dict(os.environ, {"CLUE_ENABLE_AUDIO": "0"}):
            video_only = production_controller()
        self.addCleanup(video_only.deleteLater)
        self.assertIsInstance(video_only.audio_processor, DisabledAudioProcessor)
        self.assertFalse(video_only.audio_enabled)

        app = QApplication.instance() or QApplication([])
        window = MainWindow(video_only)
        self.addCleanup(window.close)
        window.show()
        QTest.qWait(10)
        window._choose_region()
        selector = window._selector
        self.assertIsNotNone(selector)
        QTest.mousePress(selector, Qt.MouseButton.LeftButton, pos=QPoint(30, 30))
        QTest.mouseMove(selector, QPoint(230, 150), delay=10)
        QTest.mouseRelease(selector, Qt.MouseButton.LeftButton, pos=QPoint(230, 150))
        QTest.qWait(20)
        self.assertIsNotNone(window._region)
        self.assertTrue(window.start_button.isEnabled())

    def test_audio_processor_emits_transcript_to_shared_ingest_queue(self) -> None:
        class Clock:
            def now_ms(self):
                return 100

        class Source:
            def __init__(self):
                self.stopped = threading.Event()
                self.blocks = [AudioBlock([0.1] * 800, 0)]

            def prepare(self, _device_id):
                pass

            def start(self, _clock):
                pass

            def read(self):
                if self.blocks:
                    return self.blocks.pop(0)
                self.stopped.wait(0.01)
                return None

            def has_pending(self):
                return bool(self.blocks) or not self.stopped.is_set()

            def stop(self):
                self.stopped.set()

            def close(self):
                self.stop()

        class Transcriber:
            def prepare(self):
                pass

            def transcribe(self, _samples):
                return (SpeechSpan(0.0, 0.04, "offline lecture transcript"),)

        output: Queue = Queue()
        errors = []
        audio = AudioProcessor(
            source=Source(),
            transcriber=Transcriber(),
            period_seconds=0.05,
            overlap_seconds=0,
        )
        audio.start("session", Clock(), output, errors.append)
        try:
            event = output.get(timeout=3)
            self.assertIsInstance(event, TranscriptSegment)
            self.assertEqual(event.session_id, "session")
            self.assertEqual(event.text, "offline lecture transcript")
        finally:
            audio.stop()
        self.assertFalse(errors)

    def test_frame_processor_captures_and_emits_contract_records_on_cpu(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SessionManager(temp_dir)
            session = manager.start_session()
            processor = FrameProcessor(
                backend=SyntheticCapture(),
                classifier_factory=LayoutHeuristicClassifier,
                interval_s=0.02,
            )
            region = CaptureRegion(0, 0, 640, 360)
            processor.set_region(region)
            errors = []
            processor.start(
                session.session_id,
                str(session.session_dir),
                session.clock,
                session.output_queue,
                errors.append,
            )
            try:
                periodic = session.output_queue.get(timeout=3)
                self.assertEqual(periodic.session_id, session.session_id)
                self.assertEqual(periodic.source, FrameSource.PERIODIC)
                self.assertIn(periodic.frame_type, tuple(FrameType))
                snapshot = processor.capture_now(FrameSource.QUESTION_START)
                self.assertEqual(snapshot.source, FrameSource.QUESTION_START)
                self.assertEqual(snapshot.frame_type, FrameType.UNKNOWN)
                self.assertTrue((session.session_dir / snapshot.image_path).is_file())
                self.assertFalse(errors)
            finally:
                processor.stop()
                manager.finish_session(session.session_id)

    def test_hybrid_rag_indexes_and_retrieves_transcript_with_cpu_test_embedder(self) -> None:
        index = HybridRAGIndex(embedder=FakeEmbedder(), chunk_chars=300, max_chunk_ms=1000)
        index.start("session", ())
        try:
            chunks = index.index_transcript(
                TranscriptSegment(
                    session_id="session",
                    segment_id="segment-1",
                    start_ms=0,
                    end_ms=2500,
                    text="Linear regression adjusts model parameters with gradient descent.",
                )
            )
            self.assertTrue(chunks)
            hits = index.search("linear regression", top_k=3)
            self.assertTrue(hits)
            self.assertEqual(hits[0].segment_ids, ("segment-1",))
        finally:
            if index._pending:
                index.flush_pending()
            index.stop()

    def test_desktop_controller_runs_cpu_modules_and_streams_mock_answer(self) -> None:
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as temp_dir:
            frame = FrameProcessor(
                backend=SyntheticCapture(),
                classifier_factory=LayoutHeuristicClassifier,
                interval_s=0.2,
            )
            audio = DisabledAudioProcessor()
            context = ContextQAService(
                llm=FakeLLM(text="The model estimates a linear relationship.", chunk_size=8),
                rag=HybridRAGIndex(embedder=FakeEmbedder()),
            )
            controller = DesktopController(frame, audio, context, SessionManager(temp_dir))
            self.addCleanup(controller.deleteLater)
            window = MainWindow(controller)
            self.addCleanup(window.close)
            window.show()
            app.processEvents()

            def wait_for(signal, predicate, timeout_ms=4000):
                if predicate():
                    return
                loop = QEventLoop()

                def on_signal(*args):
                    if predicate(*args):
                        loop.quit()

                signal.connect(on_signal)
                timer = QTimer()
                timer.setSingleShot(True)
                timer.timeout.connect(loop.quit)
                timer.start(timeout_ms)
                loop.exec()
                signal.disconnect(on_signal)
                self.assertTrue(predicate(), "timed out waiting for the integration event")

            operations = []
            controller.operation_completed.connect(operations.append)
            controller.start_session(CaptureRegion(0, 0, 640, 360), "default")
            wait_for(controller.operation_completed, lambda _operation=None: "start" in operations)
            self.assertEqual(controller.session.session_id, context._session_id)

            frame_loop = QEventLoop()
            frame_timer = QTimer()
            frame_timer.setInterval(20)
            frame_timer.timeout.connect(window._refresh_preview)
            frame_timer.timeout.connect(
                lambda: frame_loop.quit() if not window._frame_pixmap.isNull() else None
            )
            frame_timer.start()
            QTimer.singleShot(3000, frame_loop.quit)
            frame_loop.exec()
            frame_timer.stop()
            self.assertFalse(window._frame_pixmap.isNull())

            events = []
            controller.answer_event_received.connect(events.append)
            question_id = controller.begin_question()
            controller.submit_question(question_id, "Explain the linear model")
            wait_for(
                controller.answer_event_received,
                lambda event=None: any(isinstance(item, AnswerResult) for item in events),
            )
            self.assertTrue(any(isinstance(item, AnswerDelta) for item in events))
            self.assertTrue(any(isinstance(item, AnswerResult) for item in events))

            controller.stop_session()
            wait_for(controller.operation_completed, lambda _operation=None: "stop" in operations)
            self.assertIsNone(controller.session)

    def test_context_factory_injects_offline_embedder_and_reranker(self) -> None:
        class NoopReranker:
            def score(self, query, texts):
                return tuple(1.0 for _ in texts)

        qa = create_context_qa(
            fake_llm=True,
            rag="hybrid",
            embedder=FakeEmbedder(),
            reranker=NoopReranker(),
        )
        index = qa._rag
        self.assertIsInstance(index, HybridRAGIndex)
        self.assertIs(index._embedder.__class__, FakeEmbedder)
        self.assertIsInstance(index._reranker, NoopReranker)


if __name__ == "__main__":
    unittest.main()
