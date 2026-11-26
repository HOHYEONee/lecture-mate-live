"""Launch the initial CLUE desktop application."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from PySide6.QtCore import QThreadPool
from PySide6.QtWidgets import QApplication

from .controller import production_controller
from .main_window import MainWindow
from .runtime import hardware_inventory
from .workers import FunctionWorker


def load_project_environment(project_root: Path | None = None) -> None:
    root = project_root or Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env", override=False)


def main() -> int:
    load_project_environment()
    cache = Path(__file__).resolve().parents[1] / ".cache"
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(cache / "paddlex"))
    os.environ.setdefault("HF_HOME", str(cache / "huggingface"))
    if "--diagnose" in sys.argv[1:]:
        from .runtime import diagnostics

        print(json.dumps(diagnostics(), indent=2, ensure_ascii=True))
        return 0
    app = QApplication(sys.argv)
    controller = production_controller()
    window = MainWindow(controller)
    window.show()

    def report_hardware() -> None:
        inventory = hardware_inventory()
        for gpu in inventory["nvidia"]:
            memory = (
                f" · VRAM 여유 {gpu['free_mib']}/{gpu['total_mib']} MiB"
                if gpu["free_mib"] is not None and gpu["total_mib"] is not None else ""
            )
            controller.runtime_notice.emit(f"{gpu['name']} · 드라이버 {gpu['driver']}{memory}")
        if not inventory["nvidia"]:
            adapters = ", ".join(adapter["Name"] for adapter in inventory["adapters"] if adapter.get("Name"))
            controller.runtime_notice.emit(f"{adapters or '그래픽 장치 정보 없음'} · 세션 시작 시 CUDA 호환성을 확인합니다.")

    hardware_worker = FunctionWorker("hardware", report_hardware)
    QThreadPool.globalInstance().start(hardware_worker)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
