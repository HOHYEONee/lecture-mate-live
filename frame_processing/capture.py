"""Capture adapters. The pipeline normalizes raw pixels to owned uint8 BGR."""

from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

from interface_spec.contracts import CaptureRegion


class CaptureBackend(Protocol):
    def grab(self, region: CaptureRegion) -> np.ndarray: ...


def normalize_bgr(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.size == 0:
        raise ValueError("capture must be a nonempty uint8 array")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("capture must have gray, BGR, or BGRA channels")
    return np.array(image, copy=True, order="C")


class MSSCapture:
    """Open/close native resources in the calling thread, including snapshots.

    No cross-thread MSS handle is retained. Cursor capture remains disabled.
    Region coordinates are physical pixels; negative monitor origins are valid.
    """

    def grab(self, region: CaptureRegion) -> np.ndarray:
        import mss

        with mss.mss() as screen:
            shot = screen.grab({
                "left": region.left, "top": region.top,
                "width": region.width, "height": region.height,
            })
            # Keep conversion/copy outside grab so the pipeline timestamps first.
            # numpy retains the ScreenShot's backing byte buffer after close().
            return np.asarray(shot)


class WindowsWindowCapture:
    """Capture a tracked top-level HWND, including when another window covers it.

    PrintWindow asks the target window to paint into an owned bitmap. Some
    protected or GPU-only windows may reject this request; those errors are
    surfaced through the normal frame capture error path.
    """

    def __init__(self) -> None:
        self._hwnd: int | None = None

    @property
    def dynamic_dimensions(self) -> bool:
        return self._hwnd is not None

    def select(self, hwnd: int | None) -> None:
        self._hwnd = hwnd

    @staticmethod
    def list_windows() -> list[tuple[int, str]]:
        if __import__("sys").platform != "win32":
            raise RuntimeError("프로그램 창 캡처는 Windows에서만 사용할 수 있습니다.")
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        results: list[tuple[int, str]] = []
        callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

        def visit(hwnd, _):
            if user32.IsWindowVisible(hwnd) and user32.GetWindowTextLengthW(hwnd):
                length = user32.GetWindowTextLengthW(hwnd)
                title = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, title, length + 1)
                if title.value:
                    results.append((int(hwnd), title.value))
            return True

        user32.EnumWindows(callback_type(visit), 0)
        return results

    def grab(self, region: CaptureRegion) -> np.ndarray:
        if self._hwnd is None:
            return MSSCapture().grab(region)
        if __import__("sys").platform != "win32":
            raise RuntimeError("프로그램 창 캡처는 Windows에서만 사용할 수 있습니다.")
        import ctypes
        from ctypes import wintypes

        user32, gdi32 = ctypes.windll.user32, ctypes.windll.gdi32
        user32.GetWindowDC.restype = ctypes.c_void_p
        gdi32.CreateCompatibleDC.restype = ctypes.c_void_p
        gdi32.CreateDIBSection.restype = ctypes.c_void_p
        gdi32.SelectObject.restype = ctypes.c_void_p
        hwnd = wintypes.HWND(self._hwnd)
        user32.IsWindow.argtypes = [wintypes.HWND]
        user32.IsIconic.argtypes = [wintypes.HWND]
        user32.GetWindowDC.argtypes = [wintypes.HWND]
        user32.GetWindowDC.restype = ctypes.c_void_p
        user32.ReleaseDC.argtypes = [wintypes.HWND, ctypes.c_void_p]
        user32.PrintWindow.argtypes = [wintypes.HWND, ctypes.c_void_p, wintypes.UINT]
        user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
        dwmapi = ctypes.windll.dwmapi
        dwmapi.DwmGetWindowAttribute.argtypes = [
            wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
        ]
        dwmapi.DwmGetWindowAttribute.restype = ctypes.c_long
        gdi32.CreateCompatibleDC.argtypes = [ctypes.c_void_p]
        gdi32.SelectObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        gdi32.DeleteObject.argtypes = [ctypes.c_void_p]
        gdi32.DeleteDC.argtypes = [ctypes.c_void_p]
        if not user32.IsWindow(hwnd):
            raise RuntimeError("선택한 창이 닫혔습니다. 프로그램 창을 다시 선택하세요.")
        if user32.IsIconic(hwnd):
            raise RuntimeError("선택한 창이 최소화되어 캡처할 수 없습니다.")
        rect = wintypes.RECT()
        # DWM frame bounds follow window movement and resize without shadows.
        try:
            result = dwmapi.DwmGetWindowAttribute(
                hwnd, 9, ctypes.byref(rect), ctypes.sizeof(rect)
            )
            if result != 0:
                raise OSError("DwmGetWindowAttribute failed")
        except (AttributeError, OSError):
            if not user32.GetWindowRect(hwnd, ctypes.byref(rect)):
                raise RuntimeError("선택한 창의 위치를 확인할 수 없습니다.")
        width, height = rect.right - rect.left, rect.bottom - rect.top
        if width <= 0 or height <= 0:
            raise RuntimeError("선택한 창이 최소화되어 캡처할 수 없습니다.")

        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
                        ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
                        ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                        ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                        ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
                        ("biClrImportant", wintypes.DWORD)]

        class BITMAPINFO(ctypes.Structure):
            _fields_ = [("bmiHeader", BITMAPINFOHEADER), ("bmiColors", wintypes.DWORD * 3)]

        gdi32.CreateDIBSection.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(BITMAPINFO), wintypes.UINT,
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, wintypes.DWORD,
        ]
        dc = user32.GetWindowDC(hwnd)
        if not dc:
            raise RuntimeError("선택한 창의 장치 컨텍스트를 가져오지 못했습니다.")
        memdc = gdi32.CreateCompatibleDC(dc)
        if not memdc:
            user32.ReleaseDC(hwnd, dc)
            raise RuntimeError("창 캡처용 메모리 컨텍스트를 만들지 못했습니다.")
        info = BITMAPINFO()
        info.bmiHeader = BITMAPINFOHEADER(ctypes.sizeof(BITMAPINFOHEADER), width, -height,
                                           1, 32, 0, width * height * 4, 0, 0, 0, 0)
        bits = ctypes.c_void_p()
        bitmap = gdi32.CreateDIBSection(memdc, ctypes.byref(info), 0, ctypes.byref(bits), None, 0)
        if not bitmap or not bits:
            gdi32.DeleteDC(memdc)
            user32.ReleaseDC(hwnd, dc)
            raise RuntimeError("창 캡처용 비트맵을 만들지 못했습니다.")
        old = gdi32.SelectObject(memdc, bitmap)
        try:
            if not user32.PrintWindow(hwnd, memdc, 2) and not user32.PrintWindow(hwnd, memdc, 0):
                raise RuntimeError("Windows가 선택한 창의 화면을 제공하지 못했습니다.")
            raw = ctypes.string_at(bits, width * height * 4)
            return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4).copy()
        finally:
            gdi32.SelectObject(memdc, old)
            gdi32.DeleteObject(bitmap)
            gdi32.DeleteDC(memdc)
            user32.ReleaseDC(hwnd, dc)
