from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from dotenv import dotenv_values
from PySide6.QtWidgets import QApplication, QLineEdit

from context_qa.llm import LLMError, LLMRequest, OpenAIClient
from desktop_app.controller import production_controller
from desktop_app.main_window import MainWindow
from desktop_app.settings import save_openai_api_key
from scripts import prepare_demo_models


class DemoReadinessTests(unittest.TestCase):
    def test_real_qa_without_key_reports_configuration_required(self):
        with patch.dict(os.environ, clear=True):
            controller = production_controller()
            self.addCleanup(controller.deleteLater)
            self.assertTrue(controller.needs_openai_api_key)
            self.assertIsNone(controller.context_qa._llm)
            with self.assertRaises(LLMError) as exc:
                OpenAIClient()
            self.assertEqual(exc.exception.code, "llm_api_key_required")

    def test_gui_saves_a_masked_key_in_local_settings(self):
        app = QApplication.instance() or QApplication([])
        with TemporaryDirectory() as directory, patch.dict(os.environ, clear=True):
            path = Path(directory) / ".env"
            path.write_text("CLUE_ASR_DEVICE=cpu\n", encoding="utf-8")
            controller = production_controller()
            window = MainWindow(controller)
            try:
                with (
                    patch("desktop_app.main_window.QInputDialog.getText",
                          return_value=("unit-test-value", True)) as input_dialog,
                    patch("desktop_app.settings.PROJECT_ENV_FILE", path),
                ):
                    self.assertTrue(window._edit_openai_api_key())
                self.assertIn(QLineEdit.EchoMode.Password, input_dialog.call_args.args)
                self.assertFalse(controller.needs_openai_api_key)
                self.assertEqual(dotenv_values(path), {
                    "CLUE_ASR_DEVICE": "cpu", "OPENAI_API_KEY": "unit-test-value",
                })
                self.assertNotIn("unit-test-value", window.conversation.toPlainText())
            finally:
                window.close()
                controller.deleteLater()

    def test_unusable_key_rejected_without_changing_file(self):
        with TemporaryDirectory() as directory, patch.dict(os.environ, clear=True):
            path = Path(directory) / ".env"
            with self.assertRaises(ValueError):
                save_openai_api_key("one two", path)
            self.assertFalse(path.exists())
            self.assertNotIn("OPENAI_API_KEY", os.environ)

    def test_openai_sdk_key_provider_reads_the_latest_saved_value(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "initial-test-value"}, clear=True):
            with patch("openai.OpenAI") as factory:
                OpenAIClient()
            provider = factory.call_args.kwargs["api_key"]
            self.assertEqual(provider(), "initial-test-value")
            os.environ["OPENAI_API_KEY"] = "rotated-test-value"
            self.assertEqual(provider(), "rotated-test-value")

    def test_api_authentication_failure_hides_credential(self):
        class Rejected(Exception):
            status_code = 401

        responses = SimpleNamespace(create=Mock(side_effect=Rejected("unit-test-value")))
        client = OpenAIClient(client=SimpleNamespace(responses=responses))
        with self.assertRaises(LLMError) as exc:
            client.stream(LLMRequest("", ()))
        self.assertEqual(exc.exception.code, "llm_auth_failed")
        self.assertNotIn("unit-test-value", str(exc.exception))

    def test_connection_failure_is_reported_as_transient(self):
        class APIConnectionError(Exception):
            pass

        responses = SimpleNamespace(create=Mock(side_effect=APIConnectionError()))
        client = OpenAIClient(client=SimpleNamespace(responses=responses))
        with self.assertRaises(LLMError) as exc:
            client.stream(LLMRequest("", ()))
        self.assertTrue(exc.exception.transient)
        self.assertIn("인터넷", str(exc.exception))

    def test_prepared_models_bypass_downloads(self):
        with TemporaryDirectory() as directory:
            frame_dir = Path(directory)
            for name in prepare_demo_models.MODEL_FILES:
                (frame_dir / name).write_text("fixture", encoding="utf-8")
            with (
                patch.object(prepare_demo_models, "DEFAULT_MODEL_DIR", frame_dir),
                patch.object(prepare_demo_models, "local_whisper_model", return_value=frame_dir),
                patch.object(prepare_demo_models, "default_embedding_is_ready", return_value=True),
                patch("huggingface_hub.snapshot_download") as download,
                patch.object(prepare_demo_models, "SentenceTransformerEmbedder") as embedder,
                patch.dict(os.environ),
                redirect_stdout(StringIO()),
            ):
                prepare_demo_models.prepare_models(gpu=True)
            download.assert_not_called()
            embedder.assert_not_called()

    def test_model_download_failure_includes_retry_instructions(self):
        output = StringIO()
        with (
            patch.object(prepare_demo_models, "prepare_models", side_effect=OSError("offline")),
            patch("sys.argv", ["prepare_demo_models", "--gpu"]),
            redirect_stderr(output),
        ):
            self.assertEqual(prepare_demo_models.main(), 1)
        self.assertIn("다시 실행", output.getvalue())
