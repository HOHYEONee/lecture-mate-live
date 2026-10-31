"""세션 ID·저장 디렉터리·공통 시계·수집 큐를 한 번에 준비한다.

앱 상태 전이, 수집 워커 제어, SQLite 열기/닫기는 이 모듈의 책임이 아니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Lock
from uuid import uuid4

from interface_spec.contracts import IngestEvent

from .clock import SessionClock


@dataclass(frozen=True, slots=True)
class SessionResources:
    session_id: str
    session_dir: Path
    clock: SessionClock
    output_queue: Queue[IngestEvent]


class SessionManager:
    """동시에 하나의 활성 세션만 유지하는 최소 리소스 관리자."""

    def __init__(
        self, sessions_root: str | Path = "sessions", queue_maxsize: int = 1000
    ) -> None:
        if queue_maxsize <= 0:
            raise ValueError("queue_maxsize must be positive")
        # 세션 시작 시 절대 경로를 확정해 모듈별 경로 해석 차이를 방지한다.
        self._sessions_root = Path(sessions_root).resolve()
        self._queue_maxsize = queue_maxsize
        self._active: SessionResources | None = None
        self._lock = Lock()

    @property
    def active_session(self) -> SessionResources | None:
        with self._lock:
            return self._active

    def start_session(self) -> SessionResources:
        """새 UUID 세션 및 sessions/<id>/frames 디렉터리를 준비한다.

        metadata.db는 맥락 관리가 필요할 때 생성한다.
        """
        with self._lock:
            if self._active is not None:
                raise RuntimeError("an active session already exists")

            session_id = uuid4().hex
            session_dir = self._sessions_root / session_id
            session_dir.mkdir(parents=True, exist_ok=False)
            (session_dir / "frames").mkdir()

            session = SessionResources(
                session_id=session_id,
                session_dir=session_dir,
                clock=SessionClock(),
                output_queue=Queue(maxsize=self._queue_maxsize),
            )
            self._active = session
            return session

    def finish_session(self, session_id: str) -> None:
        """수집 중단·큐 비우기·DB 종료가 끝난 뒤 활성 참조만 해제한다.

        기존 세션 데이터와 디렉터리는 삭제하지 않는다.
        """
        with self._lock:
            if self._active is None or self._active.session_id != session_id:
                raise RuntimeError("no matching active session")
            self._active = None
