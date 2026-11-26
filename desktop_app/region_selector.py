"""Transparent desktop overlay for selecting a physical-pixel capture region."""

from __future__ import annotations

import os
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QCloseEvent, QGuiApplication, QMouseEvent, QPainter, QPen
from PySide6.QtWidgets import QWidget

from interface_spec.contracts import CaptureRegion


class RegionSelector(QWidget):
    region_selected = Signal(object)
    cancelled = Signal()
    finished = Signal()

    @staticmethod
    def _monitor_key(name: str) -> str:
        return name.rsplit("\\", 1)[-1].casefold()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._origin_global: QPoint | None = None
        self._current_global: QPoint | None = None
        self._selection_completed = False
        self._physical_monitor_rects = self._load_physical_monitor_rects()
        screens = QGuiApplication.screens()
        if not screens:
            raise RuntimeError("No display is available for region selection")
        virtual_rect = screens[0].geometry()
        for screen in screens[1:]:
            virtual_rect = virtual_rect.united(screen.geometry())
        self.setGeometry(virtual_rect)

    @staticmethod
    def _load_physical_monitor_rects() -> dict[str, tuple[int, int, int, int]]:
        """Read Windows monitor pixel bounds so Qt logical coordinates can be mapped."""
        if os.name != "nt":
            return {}
        try:
            import ctypes
            from ctypes import wintypes

            class WinRect(ctypes.Structure):
                _fields_ = [
                    ("left", wintypes.LONG),
                    ("top", wintypes.LONG),
                    ("right", wintypes.LONG),
                    ("bottom", wintypes.LONG),
                ]

            class MonitorInfoEx(ctypes.Structure):
                _fields_ = [
                    ("cbSize", wintypes.DWORD),
                    ("rcMonitor", WinRect),
                    ("rcWork", WinRect),
                    ("dwFlags", wintypes.DWORD),
                    ("szDevice", wintypes.WCHAR * 32),
                ]

            callback_type = ctypes.WINFUNCTYPE(
                wintypes.BOOL,
                wintypes.HMONITOR,
                wintypes.HDC,
                ctypes.POINTER(WinRect),
                wintypes.LPARAM,
            )
            user32 = ctypes.windll.user32
            get_info = user32.GetMonitorInfoW
            get_info.argtypes = [wintypes.HMONITOR, ctypes.POINTER(MonitorInfoEx)]
            get_info.restype = wintypes.BOOL
            rects: dict[str, tuple[int, int, int, int]] = {}

            def collect(handle, _dc, _rect, _data):
                info = MonitorInfoEx()
                info.cbSize = ctypes.sizeof(MonitorInfoEx)
                if get_info(handle, ctypes.byref(info)):
                    bounds = info.rcMonitor
                    rects[RegionSelector._monitor_key(info.szDevice)] = (
                        bounds.left,
                        bounds.top,
                        bounds.right,
                        bounds.bottom,
                    )
                return True

            callback = callback_type(collect)
            user32.EnumDisplayMonitors(0, None, callback, 0)
            return rects
        except (AttributeError, OSError, TypeError, ValueError):
            return {}

    def _physical_point(self, point: QPoint) -> QPoint:
        screen = QGuiApplication.screenAt(point)
        if screen is None:
            screen = min(
                QGuiApplication.screens(),
                key=lambda item: (item.geometry().center() - point).manhattanLength(),
            )
        geometry = screen.geometry()
        native = self._physical_monitor_rects.get(self._monitor_key(screen.name()))
        return self._logical_to_physical(
            point,
            geometry,
            native,
            screen.devicePixelRatio(),
        )

    @staticmethod
    def _logical_to_physical(
        point: QPoint,
        geometry: QRect,
        native: tuple[int, int, int, int] | None,
        fallback_ratio: float,
    ) -> QPoint:
        if native is None:
            physical_left = round(geometry.left() * fallback_ratio)
            physical_top = round(geometry.top() * fallback_ratio)
            scale_x = scale_y = fallback_ratio
        else:
            physical_left, physical_top, right, bottom = native
            # Derive each axis independently. Windows can use non-uniform
            # logical-to-native geometry across mixed-DPI monitor layouts.
            scale_x = (right - physical_left) / geometry.width()
            scale_y = (bottom - physical_top) / geometry.height()
        return QPoint(
            physical_left + round((point.x() - geometry.left()) * scale_x),
            physical_top + round((point.y() - geometry.top()) * scale_y),
        )

    def paintEvent(self, _event) -> None:  # noqa: N802 - Qt virtual method name
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(10, 20, 35, 105))
        if self._origin_global is not None and self._current_global is not None:
            local_origin = self.mapFromGlobal(self._origin_global)
            local_current = self.mapFromGlobal(self._current_global)
            selection = QRect(local_origin, local_current).normalized()
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.fillRect(selection, Qt.GlobalColor.transparent)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            painter.setPen(QPen(QColor("#3478f6"), 3))
            painter.drawRect(selection)
        painter.setPen(QColor("white"))
        painter.drawText(
            self.rect().adjusted(24, 20, -24, -20),
            Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter,
            "드래그해 강의 화면 영역을 선택하세요  ·  Esc 취소",
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._origin_global = event.globalPosition().toPoint()
            self._current_global = self._origin_global
            self.update()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._origin_global is not None:
            self._current_global = event.globalPosition().toPoint()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() != Qt.MouseButton.LeftButton or self._origin_global is None:
            return
        end = event.globalPosition().toPoint()
        physical_start = self._physical_point(self._origin_global)
        physical_end = self._physical_point(end)
        left = min(physical_start.x(), physical_end.x())
        top = min(physical_start.y(), physical_end.y())
        width = abs(physical_end.x() - physical_start.x())
        height = abs(physical_end.y() - physical_start.y())
        if width > 1 and height > 1:
            self._selection_completed = True
            self.region_selected.emit(
                CaptureRegion(left, top, width, height)
            )
            self.close()
        else:
            self._origin_global = None
            self._current_global = None
            self.update()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self.close()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if not self._selection_completed:
            self.cancelled.emit()
        self.finished.emit()
        event.accept()
