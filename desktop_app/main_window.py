"""User-facing PySide6 desktop window for CLUE session control."""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QTimer, Qt, Slot
from PySide6.QtGui import QFont, QFontDatabase, QTextCursor
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from interface_spec.contracts import (
    AnswerCancelled,
    AnswerDelta,
    AnswerFailed,
    AnswerResult,
    CaptureRegion,
    SessionState,
)

from .controller import DesktopController
from .region_selector import RegionSelector


class MainWindow(QMainWindow):
    def __init__(self, controller: DesktopController) -> None:
        super().__init__()
        self.controller = controller
        self._region: CaptureRegion | None = None
        self._selector: RegionSelector | None = None
        self._draft_id: str | None = None
        self._awaiting_question_id: str | None = None
        self._answer_text = ""
        self._busy = False

        self.setWindowTitle("CLUE — Lecture Mate Live")
        self.resize(1180, 760)
        self.setMinimumSize(900, 600)
        korean_font = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "malgun.ttf"
        if korean_font.is_file() and "Malgun Gothic" not in QFontDatabase.families():
            QFontDatabase.addApplicationFont(str(korean_font))
        self.setFont(QFont("Malgun Gothic", 10))
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #f4f7fc; color: #1d2a3a; }
            QFrame#panel { background: white; border: 1px solid #dce5f1; border-radius: 12px; }
            QLabel#sectionTitle { font-size: 15px; font-weight: 600; }
            QLabel#muted { color: #708198; }
            QPushButton { background: white; border: 1px solid #cbd7e6; border-radius: 8px; padding: 9px 14px; }
            QPushButton:hover:enabled { border-color: #4d78df; color: #315fce; }
            QPushButton#primary { background: #416fe0; color: white; border: none; font-weight: 600; }
            QPushButton#stop { background: #ed5151; color: white; border: none; font-weight: 600; }
            QPushButton:disabled { color: #9aa8b9; background: #edf1f6; }
            QLineEdit { background: white; border: 1px solid #cfdaea; border-radius: 8px; padding: 10px; }
            QPlainTextEdit { background: white; border: 1px solid #e2e9f2; border-radius: 8px; padding: 8px; }
            """
        )

        self.region_button = QPushButton("강의 화면 영역 선택")
        self.region_button.clicked.connect(self._choose_region)
        self.region_label = QLabel("선택된 영역 없음")
        self.region_label.setObjectName("muted")
        self.preview = QLabel("강의 화면 영역을 선택해 주세요")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setWordWrap(True)
        self.preview.setMinimumHeight(360)
        self.preview.setStyleSheet(
            "background:#152231;color:#dbe6f4;border-radius:10px;font-size:17px;padding:24px;"
        )
        self.conversation = QPlainTextEdit()
        self.conversation.setReadOnly(True)
        self.conversation.setPlaceholderText("질문과 답변이 여기에 표시됩니다.")

        left_panel = QFrame()
        left_panel.setObjectName("panel")
        left_layout = QVBoxLayout(left_panel)
        left_title = QLabel("강의 화면")
        left_title.setObjectName("sectionTitle")
        left_layout.addWidget(left_title)
        left_layout.addWidget(self.preview, 1)
        region_row = QHBoxLayout()
        region_row.addWidget(self.region_button)
        region_row.addWidget(self.region_label, 1)
        left_layout.addLayout(region_row)

        right_panel = QFrame()
        right_panel.setObjectName("panel")
        right_layout = QVBoxLayout(right_panel)
        chat_title = QLabel("CLUE 학습 도우미")
        chat_title.setObjectName("sectionTitle")
        right_layout.addWidget(chat_title)
        right_layout.addWidget(self.conversation, 1)
        question_row = QHBoxLayout()
        self.question_input = QLineEdit()
        self.question_input.setPlaceholderText("강의 내용에 대해 질문해 보세요…")
        self.send_button = QPushButton("전송")
        self.send_button.setObjectName("primary")
        self.cancel_answer_button = QPushButton("답변 취소")
        self.cancel_answer_button.setEnabled(False)
        question_row.addWidget(self.question_input, 1)
        question_row.addWidget(self.send_button)
        question_row.addWidget(self.cancel_answer_button)
        right_layout.addLayout(question_row)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        self.state_label = QLabel(SessionState.IDLE.value.upper())
        self.elapsed_label = QLabel("00:00:00")
        self.collection_label = QLabel("준비됨 — 영역을 선택해 시작하세요")
        self.start_button = QPushButton("수집 시작")
        self.start_button.setObjectName("primary")
        self.pause_button = QPushButton("일시정지")
        self.resume_button = QPushButton("재개")
        self.stop_button = QPushButton("수집 종료")
        self.stop_button.setObjectName("stop")

        toolbar = QHBoxLayout()
        toolbar.addWidget(self.state_label)
        toolbar.addWidget(self.elapsed_label)
        toolbar.addStretch(1)
        toolbar.addWidget(self.collection_label)
        controls = QHBoxLayout()
        controls.addWidget(self.start_button)
        controls.addWidget(self.pause_button)
        controls.addWidget(self.resume_button)
        controls.addWidget(self.stop_button)
        controls.addStretch(1)
        controls.addWidget(QLabel("시스템 오디오 · 기본 장치"))

        root = QVBoxLayout()
        root.addLayout(toolbar)
        root.addWidget(splitter, 1)
        root.addLayout(controls)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

        self.start_button.clicked.connect(self._start)
        self.pause_button.clicked.connect(self._pause)
        self.resume_button.clicked.connect(self._resume)
        self.stop_button.clicked.connect(self._stop)
        self.send_button.clicked.connect(self._submit_question)
        self.question_input.textEdited.connect(self._begin_question_draft)
        self.question_input.returnPressed.connect(self._submit_question)
        self.cancel_answer_button.clicked.connect(self._cancel_answer)
        controller.state_changed.connect(self._set_state)
        controller.busy_changed.connect(self._set_busy)
        controller.error_raised.connect(self._show_error)
        controller.answer_event_received.connect(self._handle_answer_event)

        self._display_timer = QTimer(self)
        self._display_timer.setInterval(250)
        self._display_timer.timeout.connect(self._refresh_elapsed)
        self._display_timer.start()
        self._set_state(controller.state)
        self._update_buttons()

    @Slot()
    def _choose_region(self) -> None:
        if self.controller.state not in (SessionState.IDLE, SessionState.PAUSED):
            return
        selector = RegionSelector(self)
        self._selector = selector
        selector.region_selected.connect(self._region_selected)
        selector.cancelled.connect(lambda: self.collection_label.setText("영역 선택 취소됨"))
        selector.show()
        selector.activateWindow()
        selector.setFocus()

    @Slot(object)
    def _region_selected(self, region: CaptureRegion) -> None:
        self._region = region
        self.controller.set_capture_region(region)
        self.region_label.setText(
            f"{region.left}, {region.top}  ·  {region.width} × {region.height}px"
        )
        self.preview.setText(
            "캡처 영역을 선택했습니다. 프레임 모듈 연결 후 미리보기가 표시됩니다."
        )
        self.collection_label.setText("영역 선택 완료")
        self._update_buttons()

    @Slot()
    def _start(self) -> None:
        if self._region is None:
            self._show_error("수집을 시작하기 전에 강의 화면 영역을 선택하세요.")
            return
        try:
            self.controller.start_session(self._region, "default")
        except Exception as exc:
            self._show_error(str(exc))

    @Slot()
    def _pause(self) -> None:
        self._invoke(self.controller.pause_session)

    @Slot()
    def _resume(self) -> None:
        self._invoke(self.controller.resume_session)

    @Slot()
    def _stop(self) -> None:
        self._invoke(self.controller.stop_session)

    def _invoke(self, operation) -> None:
        try:
            operation()
        except Exception as exc:
            self._show_error(str(exc))

    @Slot()
    def _begin_question_draft(self, text: str) -> None:
        if not text.strip():
            if self._draft_id is not None:
                self.controller.discard_question_draft(self._draft_id)
                self._draft_id = None
            return
        if self._draft_id is not None:
            return
        if self.controller.state != SessionState.RUNNING:
            return
        try:
            self._draft_id = self.controller.begin_question()
        except Exception as exc:
            self._show_error(str(exc))

    @Slot()
    def _submit_question(self) -> None:
        text = self.question_input.text().strip()
        if not text or self._awaiting_question_id is not None:
            return
        try:
            if self._draft_id is None:
                self._draft_id = self.controller.begin_question()
            question_id = self._draft_id
            self.controller.submit_question(question_id, text)
        except Exception as exc:
            self._show_error(str(exc))
            return
        self._draft_id = None
        self._awaiting_question_id = question_id
        self._answer_text = ""
        self._append_text(f"\n나: {text}\nCLUE: ")
        self.question_input.clear()
        self._set_question_busy(True)

    @Slot()
    def _cancel_answer(self) -> None:
        if self._awaiting_question_id is None:
            return
        self._invoke(lambda: self.controller.cancel_question(self._awaiting_question_id))

    @Slot(object)
    def _handle_answer_event(self, event: object) -> None:
        if getattr(event, "session_id", None) != getattr(self.controller.session, "session_id", None):
            return
        if getattr(event, "question_id", None) != self._awaiting_question_id:
            return
        if isinstance(event, AnswerDelta):
            self._answer_text += event.text_delta
            self._append_text(event.text_delta)
        elif isinstance(event, AnswerResult):
            if not self._answer_text:
                self._append_text(event.answer)
            self._append_text("\n")
            self._awaiting_question_id = None
            self._set_question_busy(False)
        elif isinstance(event, AnswerCancelled):
            self._append_text("\nAnswer cancelled.\n")
            self._awaiting_question_id = None
            self._set_question_busy(False)
        elif isinstance(event, AnswerFailed):
            self._append_text(f"\nAnswer unavailable: {event.message}\n")
            self._awaiting_question_id = None
            self._set_question_busy(False)

    def _append_text(self, text: str) -> None:
        cursor = self.conversation.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.conversation.setTextCursor(cursor)
        self.conversation.ensureCursorVisible()

    def _set_question_busy(self, busy: bool) -> None:
        self.question_input.setEnabled(not busy)
        self.send_button.setEnabled(not busy)
        self.cancel_answer_button.setEnabled(busy)

    @Slot(object)
    def _set_state(self, state: SessionState) -> None:
        self.state_label.setText(state.value.upper())
        if state in (SessionState.STOPPING, SessionState.IDLE, SessionState.ERROR):
            if self._draft_id is not None:
                self.controller.discard_question_draft(self._draft_id)
                self._draft_id = None
            if self._awaiting_question_id is not None:
                self._awaiting_question_id = None
                self._set_question_busy(False)
        labels = {
            SessionState.IDLE: "준비됨 — 영역을 선택해 시작하세요",
            SessionState.RUNNING: "화면·시스템 오디오 수집 중",
            SessionState.PAUSING: "수집 일시정지 중…",
            SessionState.PAUSED: "수집 일시정지됨",
            SessionState.STOPPING: "남은 작업 정리 중…",
            SessionState.ERROR: "수집 오류 — 대화창 메시지를 확인하세요",
        }
        self.collection_label.setText(labels[state])
        self._update_buttons()

    @Slot(bool)
    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._update_buttons()

    def _update_buttons(self) -> None:
        state = self.controller.state
        self.region_button.setEnabled(state in (SessionState.IDLE, SessionState.PAUSED) and not self._busy)
        self.start_button.setEnabled(
            state == SessionState.IDLE and self._region is not None and not self._busy
        )
        self.pause_button.setEnabled(state == SessionState.RUNNING and not self._busy)
        self.resume_button.setEnabled(state == SessionState.PAUSED and not self._busy)
        self.stop_button.setEnabled(
            state in (SessionState.RUNNING, SessionState.PAUSING, SessionState.PAUSED, SessionState.ERROR)
            and not (self._busy and state != SessionState.PAUSING)
        )
        self.question_input.setEnabled(state == SessionState.RUNNING and self._awaiting_question_id is None)
        self.send_button.setEnabled(state == SessionState.RUNNING and self._awaiting_question_id is None)

    @Slot()
    def _refresh_elapsed(self) -> None:
        session = self.controller.session
        if session is None:
            elapsed_ms = 0
        else:
            elapsed_ms = session.clock.now_ms()
        seconds = elapsed_ms // 1000
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        self.elapsed_label.setText(f"{hours:02d}:{minutes:02d}:{seconds:02d}")

    @Slot(str)
    def _show_error(self, message: str) -> None:
        self._append_text(f"\n[Notice] {message}\n")

