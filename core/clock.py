"""일시정지 시간을 제외한 공통 세션 시계.

프레임·음성 모듈에는 SessionClockView로 읽기 권한만 전달한다.
pause()/resume()은 앱 통합 계층에서만 호출해야 한다.
"""

from __future__ import annotations

from threading import Lock
from time import monotonic_ns


class SessionClock:
    """생성 시점부터 흐르며, pause 중에는 값이 고정되는 스레드 안전 시계."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._elapsed_ns = 0
        self._running_since_ns: int | None = monotonic_ns()

    def now_ms(self) -> int:
        """현재까지의 누적 활성 수집 시간(밀리초)을 반환한다."""
        with self._lock:
            elapsed = self._elapsed_ns
            if self._running_since_ns is not None:
                elapsed += monotonic_ns() - self._running_since_ns
            return elapsed // 1_000_000

    def pause(self) -> None:
        """수집 모듈 두 곳의 '새 수집 중지' 확인 후 앱이 호출한다."""
        with self._lock:
            if self._running_since_ns is None:
                return
            self._elapsed_ns += monotonic_ns() - self._running_since_ns
            self._running_since_ns = None

    def resume(self) -> None:
        """모듈 재개 준비가 끝난 후 앱이 호출한다."""
        with self._lock:
            if self._running_since_ns is not None:
                return
            self._running_since_ns = monotonic_ns()
