"""User-facing PySide6 desktop window for CLUE session control."""

from __future__ import annotations

import os
import re
from pathlib import Path

from PySide6.QtCore import QByteArray, QTimer, Qt, QSize, Slot
from PySide6.QtGui import QColor, QFont, QFontDatabase, QIcon, QPainter, QPixmap, QTextCursor
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextBrowser,
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


class ConversationView(QTextBrowser):
    """Rich text browser that renders locally without a Chromium subprocess."""

    source_markdown = ""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._user_bubbles: list[tuple[int, QLabel]] = []

    def setHtml(self, text: str) -> None:
        for _position, label in self._user_bubbles:
            label.deleteLater()
        self._user_bubbles.clear()
        bubble_contents = re.findall(
            r'<span class="clueUserMessage"[^>]*>(.*?)</span>', text, flags=re.S
        )
        super().setHtml(text)
        search_position = 0
        import html
        for content in bubble_contents:
            plain = html.unescape(re.sub(r"<br\s*/?>", "\n", content, flags=re.I))
            plain = re.sub(r"<[^>]+>", "", plain)
            first_line = plain.splitlines()[0] if plain.splitlines() else plain
            cursor = QTextCursor(self.document())
            cursor.setPosition(search_position)
            cursor = self.document().find(first_line, cursor)
            if cursor.isNull():
                continue
            position = cursor.selectionStart()
            search_position = max(cursor.selectionEnd(), position + 1)
            label = RoundedUserBubble(plain, self.viewport())
            label.setMaximumWidth(max(160, int(self.viewport().width() * 0.78)))
            label.adjustSize()
            label.show()
            self._user_bubbles.append((position, label))
        QTimer.singleShot(0, self._position_user_bubbles)

    def _position_user_bubbles(self) -> None:
        for position, label in self._user_bubbles:
            cursor = QTextCursor(self.document())
            cursor.setPosition(min(position, self.document().characterCount() - 1))
            rect = self.cursorRect(cursor)
            label.setMaximumWidth(max(160, int(self.viewport().width() * 0.78)))
            label.adjustSize()
            x = max(8, self.viewport().width() - label.width() - 12)
            y = rect.top() - 6
            visible = y + label.height() > 0 and y < self.viewport().height()
            label.move(x, y)
            label.setVisible(visible)
            if visible:
                label.raise_()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._position_user_bubbles()

    def scrollContentsBy(self, dx: int, dy: int) -> None:
        super().scrollContentsBy(dx, dy)
        self._position_user_bubbles()

    def toPlainText(self) -> str:
        visible = super().toPlainText()
        source_math = re.findall(
            r"\\\[(.*?)\\\]|\$\$(.*?)\$\$|\\\((.*?)\\\)|(?<!\\)\$(?!\$)(.+?)(?<!\\)\$",
            MainWindow._normalize_markdown(self.source_markdown),
            re.S,
        )
        math_source = " ".join(next((part for part in group if part), "") for group in source_math)
        return visible + (" " + math_source if math_source else "")


class RoundedUserBubble(QLabel):
    """Selectable user message painted over a true rounded background."""

    def __init__(self, text: str, parent=None) -> None:
        super().__init__(text, parent)
        self.setObjectName("userBubble")
        self.setWordWrap(True)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setStyleSheet("QLabel#userBubble { color:#20242b; padding:10px 16px; }")

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#f1f3f5"))
        painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 22, 22)
        painter.setPen(QColor("#20242b"))
        painter.setFont(self.font())
        text_rect = self.rect().adjusted(16, 10, -16, -10)
        flags = int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextWordWrap)
        painter.drawText(text_rect, flags, self.text())


class MainWindow(QMainWindow):
    def __init__(self, controller: DesktopController) -> None:
        super().__init__()
        self.controller = controller
        self._region: CaptureRegion | None = None
        self._selector: RegionSelector | None = None
        self._draft_id: str | None = None
        self._awaiting_question_id: str | None = None
        self._answer_text = ""
        self._latest_answer = ""
        self._answer_render_start: int | None = None
        self._answer_render_end: int | None = None
        self._answer_status_range: tuple[int, int] | None = None
        self._conversation_markdown = ""
        self._busy = False
        self._frame_preview_path: Path | None = None
        self._frame_pixmap = QPixmap()
        self._chat_only = False
        self._regular_size = self.size()
        self._regular_minimum_size = (900, 600)
        self._scroll_restore_value = 0
        self._scroll_follow_output = True
        self._capture_window: tuple[int, str] | None = None
        self._capture_aspect_ratio: float | None = None

        self.setWindowTitle("CLUE — Lecture Mate Live")
        self.resize(1180, 760)
        self.setMinimumSize(900, 600)
        korean_font = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "malgun.ttf"
        if korean_font.is_file() and "Malgun Gothic" not in QFontDatabase.families():
            QFontDatabase.addApplicationFont(str(korean_font))
        self.setFont(QFont("Malgun Gothic", 10))
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #ffffff; color: #20242b; }
            QFrame#panel { background: white; border: none; }
            QFrame#lecturePanel { background: #fafbfc; border: none; }
            QFrame#chatPanel { background: #ffffff; border: none; }
            QSplitter::handle:horizontal { width: 2px; background: #e2e6eb; }
            QSplitter::handle:horizontal:hover { background: #cbd3de; }
            QFrame#toolbar { background: white; border: none; border-bottom: 1px solid #eceef1; }
            QLabel#sectionTitle { font-size: 16px; font-weight: 600; color: #20242b; }
            QLabel#muted { color: #858b95; }
            QLabel#sessionState { color: #416fe0; font-size: 12px; font-weight: 700; }
            QLabel#runtime { color: #626975; font-size: 12px; font-weight: 600; }
            QLabel#collectionStatus { color: #858b95; font-size: 11px; }
            QPushButton { background: transparent; border: none; border-radius: 9px; padding: 6px; }
            QPushButton:hover:enabled { background: #f2f4f7; }
            QPushButton#primary { background: #416fe0; color: white; border: none; font-weight: 600; }
            QPushButton#primary:hover:enabled { background: #315fce; }
            QPushButton#stop { background: transparent; color: #4b5563; border: none; }
            QPushButton#sendCircle { background: #416fe0; color: white; border: none; border-radius: 18px; }
            QPushButton#sendCircle:hover:enabled { background: #315fce; }
            QPushButton#sendCircle[busy="true"] { background: #e05252; }
            QPushButton#sendCircle[busy="true"]:hover:enabled { background: #cc3f3f; }
            QPushButton#copyButton { color: #77808c; font-size: 12px; }
            QPushButton:disabled { color: #b7bdc7; background: transparent; }
            QLineEdit { background: white; border: 1px solid #e2e5e9; border-radius: 20px; padding: 11px 15px; selection-background-color: #dce6ff; }
            QLineEdit:focus { border-color: #b6c8f4; }
            QTextBrowser { background: white; border: none; padding: 6px 2px; selection-background-color: #dce6ff; }
            QLabel#lecturePreview { background: #f6f7f8; color: #87909c; border-radius: 14px; padding: 20px; }
            """
        )

        self.region_button = QPushButton("강의 화면 영역 선택")
        self.region_button.hide()
        self.region_button.clicked.connect(self._choose_region)
        self.region_label = QLabel("선택된 영역 없음")
        self.region_label.setObjectName("muted")
        self.preview = QLabel("강의 화면 영역을 선택해 주세요")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setWordWrap(True)
        self.preview.setMinimumHeight(360)
        self.preview.setObjectName("lecturePreview")
        self.preview.setStyleSheet("font-size:16px;")
        self.conversation = ConversationView()
        self.conversation.setOpenExternalLinks(True)
        chat_scrollbar = self.conversation.verticalScrollBar()
        chat_scrollbar.setFixedWidth(6)
        chat_scrollbar.setStyleSheet(
            "QScrollBar:vertical { width:6px; background:transparent; margin:0; border:none; }"
            "QScrollBar::handle:vertical { min-height:24px; background:#cbd5e1; border-radius:3px; border:none; }"
            "QScrollBar::handle:vertical:hover { background:#94a3b8; }"
            "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; border:none; background:transparent; }"
            "QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical { width:0; height:0; border:none; background:transparent; }"
            "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background:transparent; }"
        )
        self.conversation.setHtml(self._render_conversation(""))

        left_panel = QFrame()
        left_panel.setObjectName("lecturePanel")
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
        right_panel.setObjectName("chatPanel")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(12, 10, 12, 12)
        right_layout.setSpacing(8)
        chat_title = QLabel("CLUE 학습 도우미")
        chat_title.setObjectName("sectionTitle")
        chat_title.hide()
        right_layout.addWidget(chat_title)
        right_layout.addWidget(self.conversation, 1)
        answer_tools_row = QHBoxLayout()
        answer_tools_row.addStretch(1)
        self.copy_button = QPushButton()
        self.copy_button.setObjectName("copyButton")
        self.copy_button.setToolTip("마지막 답변 복사")
        self._set_icon_button(self.copy_button, "copy")
        self.copy_button.setFixedSize(32, 30)
        self.copy_button.setIconSize(QSize(16, 16))
        self.copy_button.setEnabled(False)
        self.copy_button.clicked.connect(self._copy_latest_answer)
        answer_tools_row.addWidget(self.copy_button)
        right_layout.addLayout(answer_tools_row)
        question_row = QHBoxLayout()
        self.question_input = QLineEdit()
        self.question_input.setMinimumHeight(58)
        self.question_input.setPlaceholderText("강의 내용에 대해 질문해 보세요…")
        self.send_button = QPushButton("전송")
        self.send_button.setObjectName("sendCircle")
        self._set_icon_button(self.send_button, "arrow-up")
        self.send_button.setFixedSize(36, 36)
        self.send_button.setIconSize(QSize(18, 18))
        self.send_button.setToolTip("질문 전송")
        self.send_button.setProperty("busy", False)
        # Retain the previous attribute for integrations that inspect it.
        self.cancel_answer_button = self.send_button
        question_row.addWidget(self.question_input, 1)
        question_row.addWidget(self.send_button, 0, Qt.AlignmentFlag.AlignBottom)
        right_layout.addLayout(question_row)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter = splitter
        self.left_panel = left_panel
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setHandleWidth(2)

        self.state_label = QLabel(SessionState.IDLE.value.upper())
        self.state_label.setObjectName("sessionState")
        self.elapsed_label = QLabel("00:00:00")
        self.elapsed_label.setObjectName("runtime")
        self.collection_label = QLabel("준비됨 — 영역을 선택해 시작하세요")
        self.collection_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self.collection_label)
        self.collection_label.setObjectName("collectionStatus")
        self.collection_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.region_button_top = QPushButton()
        self.region_button_top.setToolTip("화면 영역 캡처")
        self._set_icon_button(self.region_button_top, "scan")
        self.region_button_top.clicked.connect(self._choose_area_capture)
        self.window_button = QPushButton()
        self.window_button.setToolTip("프로그램 창 선택")
        self._set_icon_button(self.window_button, "app-window")
        self.window_button.clicked.connect(self._choose_window)
        self.mode_button = QPushButton()
        self.mode_button.setToolTip("채팅 전용 창으로 전환")
        self._set_icon_button(self.mode_button, "panel-left-close")
        self.mode_button.clicked.connect(self._toggle_chat_only)
        self.session_button = QPushButton()
        self.session_button.setObjectName("primary")
        self.start_button = self.pause_button = self.resume_button = self.session_button
        self.stop_button = QPushButton()
        self.stop_button.setObjectName("stop")
        self.session_button.setToolTip("시작")
        self._set_icon_button(self.session_button, "play")
        self._set_icon_button(self.stop_button, "square")
        self.settings_button = QPushButton()
        self.settings_button.setToolTip("설정 보기")
        self._set_icon_button(self.settings_button, "settings")
        self.settings_button.clicked.connect(self._show_settings)
        self.mic_button = QPushButton()
        self.mic_button.setToolTip("오디오 입력 상태")
        self._set_icon_button(self.mic_button, "mic")
        self.mic_button.setEnabled(False)
        audio_enabled = bool(getattr(controller, "audio_enabled", False))
        self.mic_button.setToolTip(
            "마이크 · 오디오 입력 켜짐" if audio_enabled else "마이크 · 오디오 입력 꺼짐"
        )
        self.stop_button.setToolTip("수집 종료")

        toolbar_frame = QFrame()
        toolbar_frame.setObjectName("toolbar")
        toolbar = QVBoxLayout(toolbar_frame)
        toolbar.setContentsMargins(12, 7, 12, 7)
        toolbar.setSpacing(2)
        runtime_row = QHBoxLayout()
        runtime_row.setSpacing(8)
        runtime_row.addWidget(self.state_label)
        runtime_row.addWidget(self.elapsed_label)
        runtime_row.addStretch(1)
        actions_row = QHBoxLayout()
        actions_row.setSpacing(4)
        actions_row.addWidget(self.region_button_top)
        actions_row.addWidget(self.window_button)
        actions_row.addWidget(self.session_button)
        actions_row.addWidget(self.stop_button)
        actions_row.addWidget(self.mic_button)
        actions_row.addWidget(self.settings_button)
        actions_row.addStretch(1)
        actions_row.addWidget(self.mode_button)
        toolbar.addLayout(runtime_row)
        toolbar.addLayout(actions_row)

        root = QVBoxLayout()
        root.addWidget(toolbar_frame)
        root.addWidget(splitter, 1)
        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

        self.session_button.clicked.connect(self._toggle_session)
        self.stop_button.clicked.connect(self._stop)
        self.send_button.clicked.connect(self._send_or_cancel)
        self.question_input.textEdited.connect(self._begin_question_draft)
        self.question_input.returnPressed.connect(self._submit_question)
        controller.state_changed.connect(self._set_state)
        controller.busy_changed.connect(self._set_busy)
        controller.operation_completed.connect(self._on_operation_completed)
        controller.error_raised.connect(self._show_error)
        controller.answer_event_received.connect(self._handle_answer_event)

        self._display_timer = QTimer(self)
        self._display_timer.setInterval(250)
        self._display_timer.timeout.connect(self._refresh_elapsed)
        self._display_timer.start()
        self._frame_timer = QTimer(self)
        self._frame_timer.setInterval(750)
        self._frame_timer.timeout.connect(self._refresh_preview)
        self._frame_timer.start()
        self._scroll_restore_timer = QTimer(self)
        self._scroll_restore_timer.setSingleShot(True)
        self._scroll_restore_timer.timeout.connect(self._restore_conversation_scroll)
        self._stream_render_timer = QTimer(self)
        self._stream_render_timer.setSingleShot(False)
        self._stream_render_timer.setInterval(80)
        self._stream_render_timer.timeout.connect(self._render_streaming_answer)
        self._set_state(controller.state)
        self._update_buttons()

    @Slot()
    def _choose_region(self) -> None:
        if self.controller.state not in (SessionState.IDLE, SessionState.PAUSED):
            return
        selector = RegionSelector()
        self._selector = selector
        selector.region_selected.connect(self._region_selected)
        selector.cancelled.connect(self._selection_cancelled)
        selector.finished.connect(self._restore_after_selection)
        self.hide()
        try:
            selector.show()
            selector.activateWindow()
            selector.setFocus()
        except Exception:
            self.show()
            self.activateWindow()
            raise

    @Slot()
    def _selection_cancelled(self) -> None:
        self.collection_label.setText("영역 선택 취소됨")

    @Slot()
    def _restore_after_selection(self) -> None:
        self._selector = None
        self.show()
        self.raise_()
        self.activateWindow()

    @Slot(object)
    def _region_selected(self, region: CaptureRegion) -> None:
        self._region = region
        self._capture_aspect_ratio = region.width / region.height
        self.controller.set_capture_region(region)
        self._frame_preview_path = None
        self._frame_pixmap = QPixmap()
        self.preview.setPixmap(self._frame_pixmap)
        self.region_label.setText(
            f"{region.left}, {region.top}  ·  {region.width} × {region.height}px"
        )
        self.preview.setText(
            "영역 선택 완료 — 수집을 시작하면 대표 프레임이 세션에 저장됩니다."
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
    def _toggle_session(self) -> None:
        if self.controller.state == SessionState.IDLE:
            self._start()
        elif self.controller.state == SessionState.RUNNING:
            self._pause()
        elif self.controller.state == SessionState.PAUSED:
            self._resume()

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
        self.copy_button.setEnabled(False)
        self._append_text(f"\n\n**YOU:** {text}\n\n**CLUE:**\u00a0")
        self._insert_answer_status()
        self._scroll_follow_output = True
        scrollbar = self.conversation.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        self._scroll_restore_timer.start(35)
        self.question_input.clear()
        self._set_question_busy(True)

    @Slot()
    def _cancel_answer(self) -> None:
        if self._awaiting_question_id is None:
            return
        self._invoke(lambda: self.controller.cancel_question(self._awaiting_question_id))

    @Slot()
    def _send_or_cancel(self) -> None:
        if self._awaiting_question_id is not None:
            self._cancel_answer()
        else:
            self._submit_question()

    @Slot()
    def _copy_latest_answer(self) -> None:
        if self._latest_answer:
            from PySide6.QtWidgets import QApplication

            QApplication.clipboard().setText(self._latest_answer)

    @Slot(object)
    def _handle_answer_event(self, event: object) -> None:
        if getattr(event, "session_id", None) != getattr(self.controller.session, "session_id", None):
            return
        if getattr(event, "question_id", None) != self._awaiting_question_id:
            return
        if isinstance(event, AnswerDelta):
            self._answer_text += event.text_delta
            self._conversation_markdown += event.text_delta
            self.conversation.source_markdown = self._conversation_markdown
            if not self._stream_render_timer.isActive():
                self._render_streaming_answer()
                self._stream_render_timer.start()
        elif isinstance(event, AnswerResult):
            self._stream_render_timer.stop()
            if not self._answer_text:
                self._append_text(event.answer)
            else:
                self._render_streaming_answer(final=True)
            self._append_text("\n")
            self._latest_answer = event.answer or self._answer_text
            self.copy_button.setEnabled(bool(self._latest_answer))
            self._clear_answer_render_range()
            self._awaiting_question_id = None
            self._set_question_busy(False)
        elif isinstance(event, AnswerCancelled):
            self._stream_render_timer.stop()
            if self._answer_text:
                self._render_streaming_answer(final=True)
            self._append_text("\n\n응답이 취소되었습니다.\n")
            self._latest_answer = self._answer_text
            self.copy_button.setEnabled(bool(self._latest_answer))
            self._clear_answer_render_range()
            self._awaiting_question_id = None
            self._set_question_busy(False)
        elif isinstance(event, AnswerFailed):
            self._stream_render_timer.stop()
            if self._answer_text:
                self._render_streaming_answer(final=True)
            self._append_text(f"\n\n답변을 가져오지 못했습니다: {event.message}\n")
            self._latest_answer = self._answer_text
            self.copy_button.setEnabled(bool(self._latest_answer))
            self._clear_answer_render_range()
            self._awaiting_question_id = None
            self._set_question_busy(False)

    def _append_text(self, text: str) -> None:
        scrollbar = self.conversation.verticalScrollBar()
        self._scroll_restore_value = scrollbar.value()
        self._scroll_follow_output = scrollbar.value() >= scrollbar.maximum() - 2
        self._conversation_markdown += text
        self.conversation.source_markdown = self._conversation_markdown
        self.conversation.setHtml(self._render_conversation(self._conversation_markdown))
        self._scroll_restore_timer.start(35)

    def _insert_answer_status(self) -> None:
        cursor = QTextCursor(self.conversation.document())
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._answer_render_start = cursor.position()
        cursor.insertHtml('&nbsp;&nbsp;<span style="color:#9aa8b9"><i>답변 생성 중…</i></span>')
        self._answer_render_end = cursor.position()
        self._answer_status_range = (self._answer_render_start, self._answer_render_end)

    def _clear_answer_render_range(self) -> None:
        self._answer_render_start = None
        self._answer_render_end = None
        self._answer_status_range = None

    @staticmethod
    def _unfinished_math_start(text: str) -> int | None:
        pattern = re.compile(r"\$\$|(?<!\\)\$|\\\[|\\\]|\\\(|\\\)")
        closing = {"$$": "$$", "$": "$", r"\[": r"\]", r"\(": r"\)"}
        opening = None
        for match in pattern.finditer(text):
            token = match.group(0)
            if opening is None:
                if token in closing:
                    opening = (closing[token], match.start())
            elif token == opening[0]:
                opening = None
        return opening[1] if opening else None

    def _render_streaming_answer(self, final: bool = False) -> None:
        if self._answer_render_start is None or self._answer_render_end is None:
            return
        start, end = self._answer_render_start, self._answer_render_end
        stream_text = self._normalize_markdown(self._answer_text)
        pending_math = None if final else self._unfinished_math_start(stream_text)
        complete_text = stream_text if pending_math is None else stream_text[:pending_math]
        fragment = self._render_conversation(complete_text)
        body = re.search(r"<body>(.*)</body>", fragment, re.S)
        rendered_body = body.group(1) if body else complete_text
        if pending_math is not None:
            rendered_body += '<p><span style="color:#9aa8b9"><i>수식 생성 중…</i></span></p>'

        scrollbar = self.conversation.verticalScrollBar()
        old_value = scrollbar.value()
        follow_output = old_value >= scrollbar.maximum() - 2
        cursor = QTextCursor(self.conversation.document())
        cursor.setPosition(start)
        cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
        cursor.removeSelectedText()
        cursor.insertHtml(rendered_body)
        self._answer_render_end = cursor.position()
        self._answer_status_range = None
        scrollbar.setValue(scrollbar.maximum() if follow_output else min(old_value, scrollbar.maximum()))

    def _restore_conversation_scroll(self) -> None:
        scrollbar = self.conversation.verticalScrollBar()
        if self._scroll_follow_output:
            scrollbar.setValue(scrollbar.maximum())
        else:
            scrollbar.setValue(min(self._scroll_restore_value, scrollbar.maximum()))

    @staticmethod
    def _set_icon_button(button: QPushButton, name: str) -> None:
        icon_path = Path(__file__).parent / "icons" / f"{name}.svg"
        svg = icon_path.read_text(encoding="utf-8")
        if button.objectName() in {"primary", "sendCircle"}:
            svg = svg.replace("#334155", "#ffffff")
        renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
        pixmap = QPixmap(24, 24)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        renderer.render(painter)
        painter.end()
        button.setText("")
        button.setIcon(QIcon(pixmap))
        button.setIconSize(QSize(18, 18))
        button.setFixedSize(34, 34)

    @staticmethod
    def _normalize_markdown(text: str) -> str:
        # Model output sometimes over-escapes Markdown punctuation and TeX
        # delimiters. Normalize those while preserving ordinary backslashes.
        text = re.sub(r"\\+([*])", r"\1", text)
        text = re.sub(r"\\+_", "_", text)
        text = re.sub(r"\\{2,}([()\[\]])", r"\\\1", text)
        return text.replace("&#x20;", " ")

    @staticmethod
    def _render_conversation(markdown: str) -> str:
        """Render Markdown and common LaTeX notation using local HTML only."""
        import html
        text = MainWindow._normalize_markdown(markdown)
        user_messages: list[tuple[str, str]] = []
        clue_headers: list[str] = []

        def hold_user_message(match):
            key = f"CLUEUSERMESSAGE{len(user_messages)}TOKEN"
            content = html.escape(match.group(1).strip()).replace("\n", "<br>")
            bubble = (
                '<p align="right" style="margin:8px 0 14px">'
                f'<span class="clueUserMessage" style="color:#ffffff">{content}</span><br></p>'
            )
            user_messages.append((key, bubble))
            return key + "\n\n**CLUE:**"

        text = re.sub(r"\*\*YOU:\*\*\s*(.*?)\n\n\*\*CLUE:\*\*", hold_user_message, text, flags=re.S)

        def hold_clue_header(_match):
            key = f"CLUEASSISTANTHEADER{len(clue_headers)}TOKEN"
            clue_headers.append(key)
            return key

        text = re.sub(r"\*\*CLUE:\*\*", hold_clue_header, text)
        formulas: list[tuple[str, bool]] = []
        delimiters = re.compile(r"\\\[(.*?)\\\]|\$\$(.*?)\$\$|\\\((.*?)\\\)|(?<!\\)\$(?!\$)(.+?)(?<!\\)\$", re.S)

        def hold_formula(match):
            formula = next((part for part in match.groups() if part is not None), "")
            display = match.group(1) is not None or match.group(2) is not None
            key = f"CLUEFORMULA{len(formulas)}TOKEN"
            formulas.append((key, MainWindow._latex_html(formula, display)))
            return key

        text = delimiters.sub(hold_formula, text)
        try:
            from markdown_it import MarkdownIt
            renderer = MarkdownIt("default", {"html": False}).enable("table")
            rendered = renderer.render(text)
            rendered = rendered.replace("<table>", '<table class="markdownTable">')
        except ImportError:
            rendered = "<pre>" + html.escape(text) + "</pre>"
        for key, value in user_messages:
            rendered = rendered.replace(f"<p>{key}</p>", value)
        sparkles_uri = (Path(__file__).parent / "icons" / "sparkles.svg").resolve().as_uri()
        for key in clue_headers:
            rendered = rendered.replace(
                f"<p>{key}",
                f'<p style="margin:18px 0 10px"><img src="{sparkles_uri}" width="18" height="18" align="middle"> '
                '<strong>CLUE</strong><br>',
            )
        for key, value in formulas:
            rendered = rendered.replace(key, value)
        return """<!doctype html><html><head><meta charset='utf-8'>
        <style>body{font:15px 'Malgun Gothic',sans-serif;color:#20242b;background:#fff;margin:12px 16px;line-height:1.7}
        p{margin:0 0 12px}h1,h2,h3,h4{line-height:1.35;margin:20px 0 10px;color:#20242b}ul,ol{margin:6px 0 14px;padding-left:26px}
        table.markdownTable{border-collapse:collapse;margin:12px 0;width:100%}.markdownTable th,.markdownTable td{border:1px solid #e5e7eb;padding:7px 9px;text-align:left}.markdownTable th{background:#f7f8fa;font-weight:600}
        pre{background:#f6f7f8;border-radius:8px;padding:12px;white-space:pre-wrap}code{background:#f2f4f7;border-radius:4px;padding:2px 5px}blockquote{border-left:3px solid #cad6e8;padding-left:12px;color:#52647c;margin-left:0}
        .math-inline{font-family:'Cambria Math','Times New Roman',serif;font-size:1.12em;white-space:nowrap}.math-display{text-align:center;margin:14px 0;font-family:'Cambria Math','Times New Roman',serif;font-size:1.2em}
        .frac{display:inline-table;vertical-align:middle;text-align:center;line-height:1.15;margin:0 2px;border-collapse:collapse}.frac td:first-child{border-bottom:1px solid;padding:0 3px 2px}.frac td:last-child{padding:2px 3px 0}
        sup,sub{font-size:.75em;line-height:0} </style>
        </head><body>""" + rendered + "</body></html>"

    @staticmethod
    def _latex_html(source: str, display: bool) -> str:
        import html
        value = html.escape(source.strip())
        # Resolve nested fractions from the inside out.
        fraction = re.compile(r"\\(?:d?frac)\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
        for _ in range(8):
            value, count = fraction.subn(
                r'<table cellspacing="0" cellpadding="0" style="border-collapse:collapse;margin:0 2px"><tr><td align="center" style="border-bottom:1px solid #1d2a3a;padding:0 3px 2px">\1</td></tr><tr><td align="center" style="padding:2px 3px 0">\2</td></tr></table>',
                value,
            )
            if not count:
                break
        value = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"√<span>(\1)</span>", value)
        value = re.sub(r"\\(?:text|mathrm|mathbf)\s*\{([^{}]*)\}", r"\1", value)
        value = re.sub(r"([A-Za-z0-9)])\^\{([^{}]*)\}", r"\1<sup>\2</sup>", value)
        value = re.sub(r"([A-Za-z0-9)])_\{([^{}]*)\}", r"\1<sub>\2</sub>", value)
        value = re.sub(r"([A-Za-z0-9)])\^([A-Za-z0-9+-])", r"\1<sup>\2</sup>", value)
        value = re.sub(r"([A-Za-z0-9)])_([A-Za-z0-9+-])", r"\1<sub>\2</sub>", value)
        value = value.replace("\\\\", "<br>")
        symbols = {"\\times": "×", "\\cdot": "·", "\\pm": "±", "\\le": "≤", "\\leq": "≤",
                   "\\ge": "≥", "\\geq": "≥", "\\neq": "≠", "\\approx": "≈", "\\infty": "∞",
                   "\\sum": "∑", "\\prod": "∏", "\\int": "∫", "\\pi": "π", "\\theta": "θ",
                   "\\alpha": "α", "\\beta": "β", "\\gamma": "γ", "\\Delta": "Δ", "\\lambda": "λ",
                   "\\mu": "μ", "\\sigma": "σ", "\\omega": "ω", "\\rightarrow": "→", "\\to": "→"}
        for command, symbol in symbols.items():
            value = value.replace(command, symbol)
        value = value.replace("\\left", "").replace("\\right", "").replace("\\,", " ").replace("\\;", " ")
        value = re.sub(r"\\[a-zA-Z]+", "", value).replace("{", "").replace("}", "")
        return f'<div class="math-display">{value}</div>' if display else f'<span class="math-inline">{value}</span>'

    @Slot()
    def _toggle_chat_only(self) -> None:
        self._chat_only = not self._chat_only
        if self._chat_only:
            self._regular_size = self.size()
            self._regular_minimum_size = (self.minimumWidth(), self.minimumHeight())
            self.setMinimumSize(620, 480)
            self.resize(650, min(700, self.height()))
        else:
            self.setMinimumSize(*self._regular_minimum_size)
            self.resize(self._regular_size)
        self.left_panel.setVisible(not self._chat_only)
        self.splitter.setSizes([0, 1] if self._chat_only else [3, 2])
        self._set_icon_button(self.mode_button, "panel-left-close")
        self.mode_button.setToolTip("좌우 화면으로 복원" if self._chat_only else "채팅 전용 창으로 전환")

    @Slot()
    def _choose_area_capture(self) -> None:
        self._capture_window = None
        backend = getattr(self.controller, "window_capture_backend", None)
        if backend is not None:
            backend.select(None)
        self._choose_region()

    @Slot()
    def _choose_window(self) -> None:
        if self.controller.state not in (SessionState.IDLE, SessionState.PAUSED):
            return
        from frame_processing.capture import WindowsWindowCapture
        try:
            windows = WindowsWindowCapture.list_windows()
        except Exception as exc:
            self._show_error(str(exc))
            return
        if not windows:
            QMessageBox.warning(self, "창을 찾을 수 없음", "캡처할 수 있는 프로그램 창이 없습니다.")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("프로그램 창 선택")
        layout = QVBoxLayout(dialog)
        window_list = QComboBox()
        for hwnd, title in windows:
            window_list.addItem(title, hwnd)
        layout.addWidget(window_list)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        backend = getattr(self.controller, "window_capture_backend", None)
        if backend is None:
            QMessageBox.critical(self, "창 캡처 오류", "Windows 창 캡처 기능을 사용할 수 없습니다.")
            return
        self._capture_window = (int(window_list.currentData()), window_list.currentText())
        backend.select(self._capture_window[0])
        self._region_selected(CaptureRegion(0, 0, max(1, self.preview.width()), max(1, self.preview.height())))
        # A live HWND can change dimensions; fit its own source ratio each frame.
        self._capture_aspect_ratio = None
        self.region_label.setText(f"창: {self._capture_window[1]}")

    @Slot()
    def _show_settings(self) -> None:
        audio = "켜짐" if bool(getattr(self.controller, "audio_enabled", False)) else "꺼짐"
        capture = "프로그램 창" if self._capture_window else "화면 영역"
        QMessageBox.information(
            self,
            "CLUE 설정",
            f"캡처 방식: {capture}\n오디오 입력: {audio}",
        )

    def _set_question_busy(self, busy: bool) -> None:
        self.question_input.setEnabled(not busy)
        self.send_button.setEnabled(busy or self.controller.state == SessionState.RUNNING)
        self.send_button.setProperty("busy", busy)
        self.send_button.setToolTip("답변 중지" if busy else "질문 전송")
        self._set_icon_button(self.send_button, "square" if busy else "arrow-up")
        self.send_button.setFixedSize(36, 36)
        self.send_button.setIconSize(QSize(18, 18))
        self.send_button.style().unpolish(self.send_button)
        self.send_button.style().polish(self.send_button)

    @Slot(object)
    def _set_state(self, state: SessionState) -> None:
        self.state_label.setText(state.value.upper())
        if state in (SessionState.STOPPING, SessionState.IDLE, SessionState.ERROR):
            if self._draft_id is not None:
                self.controller.discard_question_draft(self._draft_id)
                self._draft_id = None
            if self._awaiting_question_id is not None:
                self._stream_render_timer.stop()
                self._clear_answer_render_range()
                self._awaiting_question_id = None
                self._set_question_busy(False)
        labels = {
            SessionState.IDLE: "준비됨 — 영역을 선택해 시작하세요",
            SessionState.RUNNING: (
                "화면 수집 중 · 오디오 꺼짐"
                if not getattr(self.controller, "audio_enabled", True)
                else "화면·시스템 오디오 수집 중"
            ),
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

    @Slot(str)
    def _on_operation_completed(self, operation: str) -> None:
        if operation != "stop":
            return
        self.controller.clear_capture_region()
        self._region = None
        self._frame_preview_path = None
        self._frame_pixmap = QPixmap()
        self._capture_aspect_ratio = None
        self.preview.setPixmap(self._frame_pixmap)
        self.preview.setText("강의 화면 영역을 선택해 주세요")
        self.region_label.setText("선택된 영역 없음")
        self._conversation_markdown = ""
        self.conversation.source_markdown = ""
        self.conversation.setHtml(self._render_conversation(""))
        self._stream_render_timer.stop()
        self._clear_answer_render_range()
        self.question_input.clear()
        self._draft_id = None
        self._awaiting_question_id = None
        self._answer_text = ""
        self._set_question_busy(False)
        self.collection_label.setText("대기 중 · 영역을 선택해 시작하세요")
        self._refresh_elapsed()
        self._update_buttons()

    def _update_buttons(self) -> None:
        state = self.controller.state
        can_change_capture = state in (SessionState.IDLE, SessionState.PAUSED) and not self._busy
        self.region_button.setEnabled(can_change_capture)
        self.region_button_top.setEnabled(can_change_capture)
        self.window_button.setEnabled(can_change_capture)
        if state == SessionState.RUNNING:
            self.session_button.setToolTip("Pause session")
            self._set_icon_button(self.session_button, "pause")
            self.session_button.setEnabled(not self._busy)
        elif state == SessionState.PAUSED:
            self.session_button.setToolTip("Resume session")
            self._set_icon_button(self.session_button, "play")
            self.session_button.setEnabled(not self._busy)
        else:
            self.session_button.setToolTip("Start session")
            self._set_icon_button(self.session_button, "play")
            self.session_button.setEnabled(state == SessionState.IDLE and self._region is not None and not self._busy)
        self.stop_button.setEnabled(
            state in (SessionState.RUNNING, SessionState.PAUSING, SessionState.PAUSED, SessionState.ERROR)
            and not (self._busy and state != SessionState.PAUSING)
        )
        self.question_input.setEnabled(state == SessionState.RUNNING and self._awaiting_question_id is None)
        busy = self._awaiting_question_id is not None
        self.send_button.setEnabled(state == SessionState.RUNNING)
        self.send_button.setProperty("busy", busy)
        self.send_button.setToolTip("Stop answer" if busy else "Send question")
        self._set_icon_button(self.send_button, "square" if busy else "arrow-up")
        self.send_button.setFixedSize(36, 36)
        self.send_button.setIconSize(QSize(18, 18))
        self.send_button.style().unpolish(self.send_button)
        self.send_button.style().polish(self.send_button)

    @Slot()
    def _refresh_preview(self) -> None:
        session = self.controller.session
        if session is None:
            return
        frame_dir = session.session_dir / "frames"
        try:
            latest = max(frame_dir.glob("*.jpg"), key=lambda path: path.stat().st_mtime_ns)
        except (OSError, ValueError):
            if self._frame_pixmap.isNull():
                self.preview.setText("대표 프레임 수집을 기다리는 중…")
            return
        if latest != self._frame_preview_path:
            pixmap = QPixmap(str(latest))
            if pixmap.isNull():
                return
            self._frame_preview_path = latest
            self._frame_pixmap = pixmap
        if not self._frame_pixmap.isNull():
            self.preview.setText("")
            target_size = self.preview.contentsRect().size()
            ratio = self._capture_aspect_ratio
            if ratio and target_size.width() > 0 and target_size.height() > 0:
                if target_size.width() / target_size.height() > ratio:
                    target_size.setWidth(round(target_size.height() * ratio))
                else:
                    target_size.setHeight(round(target_size.width() / ratio))
            self.preview.setPixmap(
                self._frame_pixmap.scaled(
                    target_size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

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

