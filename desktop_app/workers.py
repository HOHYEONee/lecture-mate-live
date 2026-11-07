"""Small QRunnable wrapper for blocking lifecycle calls."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QRunnable, Signal, Slot


class WorkerSignals(QObject):
    completed = Signal(str, object)
    failed = Signal(str, str)


class FunctionWorker(QRunnable):
    """Run one callable in QThreadPool and report its result through signals."""

    def __init__(self, operation_id: str, function: Callable[[], Any]) -> None:
        super().__init__()
        self.operation_id = operation_id
        self.function = function
        self.signals = WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            result = self.function()
        except Exception as exc:  # surfaced to the controller on the GUI thread
            self.signals.failed.emit(self.operation_id, f"{type(exc).__name__}: {exc}")
        else:
            self.signals.completed.emit(self.operation_id, result)
