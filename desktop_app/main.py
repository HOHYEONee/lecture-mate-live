"""Launch the initial CLUE desktop application."""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

from PySide6.QtWidgets import QApplication

from .controller import production_controller
from .main_window import MainWindow


def load_project_environment(project_root: Path | None = None) -> None:
    root = project_root or Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env", override=False)


def main() -> int:
    load_project_environment()
    app = QApplication(sys.argv)
    controller = production_controller()
    window = MainWindow(controller)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
