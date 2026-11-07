"""Launch the initial CLUE desktop application."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from .controller import default_controller
from .main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    controller = default_controller()
    window = MainWindow(controller)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
