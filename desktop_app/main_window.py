"""Minimal PySide6 window for session lifecycle control."""

from __future__ import annotations

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from interface_spec.contracts import CaptureRegion, SessionState

from .controller import DesktopController


class MainWindow(QMainWindow):
    def __init__(self, controller: DesktopController) -> None:
        super().__init__()
        self.controller = controller
        self.setWindowTitle("CLUE — Lecture Mate Live")
        self.resize(520, 360)

        self.state_label = QLabel(SessionState.IDLE.value.upper())
        self.start_button = QPushButton("Start")
        self.pause_button = QPushButton("Pause")
        self.resume_button = QPushButton("Resume")
        self.stop_button = QPushButton("Stop")
        self.error_log = QTextEdit()
        self.error_log.setReadOnly(True)
        self.error_log.setPlaceholderText("Errors and module notifications appear here.")

        buttons = QHBoxLayout()
        for button in (
            self.start_button,
            self.pause_button,
            self.resume_button,
            self.stop_button,
        ):
            buttons.addWidget(button)
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Session state"))
        layout.addWidget(self.state_label)
        layout.addLayout(buttons)
        layout.addWidget(QLabel("Activity and errors"))
        layout.addWidget(self.error_log)
        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        self.start_button.clicked.connect(self._start)
        self.pause_button.clicked.connect(self._pause)
        self.resume_button.clicked.connect(self._resume)
        self.stop_button.clicked.connect(self._stop)
        controller.state_changed.connect(self._set_state)
        controller.busy_changed.connect(self._set_busy)
        controller.error_raised.connect(self._show_error)
        controller.state_changed.emit(controller.state)
        self._set_busy(False)

    @Slot()
    def _start(self) -> None:
        try:
            # Region selection and device discovery are outside the initial UI scope.
            self.controller.start_session(CaptureRegion(0, 0, 1280, 720), "default")
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

    def _invoke(self, operation: object) -> None:
        try:
            operation()  # type: ignore[operator]
        except Exception as exc:
            self._show_error(str(exc))

    @Slot(object)
    def _set_state(self, state: SessionState) -> None:
        self.state_label.setText(state.value.upper())
        self._update_buttons()

    @Slot(bool)
    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._update_buttons()

    def _update_buttons(self) -> None:
        state = self.controller.state
        busy = getattr(self, "_busy", False)
        self.start_button.setEnabled(state == SessionState.IDLE and not busy)
        self.pause_button.setEnabled(state == SessionState.RUNNING and not busy)
        self.resume_button.setEnabled(state == SessionState.PAUSED and not busy)
        self.stop_button.setEnabled(
            state in (
                SessionState.RUNNING,
                SessionState.PAUSING,
                SessionState.PAUSED,
                SessionState.ERROR,
            )
            and not (busy and state != SessionState.PAUSING)
        )

    @Slot(str)
    def _show_error(self, message: str) -> None:
        self.error_log.append(message)

