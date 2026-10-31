"""CLUE P-실무프로젝트: 모듈 간 공통 데이터/API 계약 v1.5.

이 파일은 인터페이스 정의만 포함합니다. 실제 캡처, 전사, DB, UI 구현은
담당 모듈에서 수행합니다. Python 3.11 이상을 기준으로 합니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from queue import Queue
from typing import Callable, Protocol, TypeAlias


class SessionState(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSING = "pausing"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"


class QuestionState(StrEnum):
    READY = "ready"
    GENERATING = "generating"
    CANCELLING = "cancelling"


class ModuleName(StrEnum):
    FRAME = "frame"
    AUDIO = "audio"
    CONTEXT_QA = "context_qa"
    APP = "app"


class FrameType(StrEnum):
    TEXT = "Text"
    VISUAL = "Visual"
    UNKNOWN = "Unknown"


class FrameSource(StrEnum):
    PERIODIC = "periodic"
    QUESTION_START = "question_start"
    QUESTION_SUBMIT = "question_submit"


class FrameDecision(StrEnum):
    NEW_GROUP = "new_group"
    NEW_REPRESENTATIVE = "new_representative"
    REUSED = "reused"
    QUESTION_SNAPSHOT = "question_snapshot"


@dataclass(frozen=True, slots=True)
class CaptureRegion:
    """물리 화면 픽셀 좌표. 멀티 모니터에서는 left/top이 음수일 수 있음."""

    left: int
    top: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("CaptureRegion width/height must be positive")


@dataclass(frozen=True, slots=True)
class FrameRecord:
    """캡처 관측 결과. 중복 시 기존 프레임 ID/경로를 재사용하고 파일은 저장하지 않음.

    frame_id는 실제 저장된 이미지의 식별자이며 중복 관측에서는 반복될 수 있다.
    capture_seq는 질문 스냅샷까지 포함해 세션 내 캡처마다 고유한 일련번호다.
    timestamp_ms는 이 Record에 대응하는 현재 캡처의 취득 시각이다.
    """

    session_id: str
    frame_id: str
    capture_seq: int
    timestamp_ms: int
    image_path: str  # 세션 폴더를 기준으로 한 상대 경로, / 구분자 사용
    frame_type: FrameType
    source: FrameSource
    group_id: str | None
    decision: FrameDecision

    def __post_init__(self) -> None:
        if self.capture_seq < 0:
            raise ValueError("capture_seq must be non-negative")
        if self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be non-negative")
        path = PurePosixPath(self.image_path)
        if (
            not self.image_path
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in self.image_path
            or ":" in self.image_path
            or not path.parts
            or path.parts[0] != "frames"
        ):
            raise ValueError("image_path must be relative to session_dir/frames")
        if self.source == FrameSource.PERIODIC:
            if not self.group_id or self.decision == FrameDecision.QUESTION_SNAPSHOT:
                raise ValueError("periodic frame needs group_id and periodic decision")
        elif self.group_id is not None or self.decision != FrameDecision.QUESTION_SNAPSHOT:
            raise ValueError("question snapshot has no group_id and uses QUESTION_SNAPSHOT")


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """전사가 확정된 하나의 발화. 처리 완료 시간이 아니라 실제 발화 구간."""

    session_id: str
    segment_id: str
    start_ms: int
    end_ms: int
    text: str

    def __post_init__(self) -> None:
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("transcript interval must be positive")
        if not self.text.strip():
            raise ValueError("final transcript text must not be empty")


@dataclass(frozen=True, slots=True)
class RAGChunkRecord:
    """RAG가 생성한 검색 청크. SQLite 영구 저장은 맥락 관리가 담당한다.

    임베딩/청킹 알고리즘은 RAG 구현 담당자가 선택한다.
    하나의 청크가 여러 TranscriptSegment를 포함할 수 있다.
    """

    session_id: str
    chunk_id: str
    start_ms: int
    end_ms: int
    text: str
    segment_ids: tuple[str, ...]
    embedding: tuple[float, ...]
    embedding_model: str

    def __post_init__(self) -> None:
        if self.start_ms < 0 or self.end_ms <= self.start_ms:
            raise ValueError("RAG chunk interval must be positive")
        if not self.text.strip() or not self.segment_ids:
            raise ValueError("RAG chunk needs text and source segment ids")
        if not self.embedding or not self.embedding_model.strip():
            raise ValueError("RAG chunk needs an embedding and model name")


@dataclass(frozen=True, slots=True)
class RAGSearchHit:
    """RAG의 하이브리드 검색 결과. score는 같은 검색 내의 정렬에 사용한다."""

    session_id: str
    chunk_id: str
    start_ms: int
    end_ms: int
    text: str
    segment_ids: tuple[str, ...]
    score: float


@dataclass(frozen=True, slots=True)
class QuestionRequest:
    """앱이 만든 질문. 두 즉시 캡처 Record를 직접 포함하여 큐 지연을 회피."""

    session_id: str
    question_id: str
    text: str
    input_started_ms: int
    submitted_ms: int
    input_frame: FrameRecord
    submit_frame: FrameRecord

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("question text must not be empty")
        if self.input_started_ms < 0 or self.submitted_ms < self.input_started_ms:
            raise ValueError("invalid question timing")
        if self.input_frame.source != FrameSource.QUESTION_START:
            raise ValueError("input_frame must have QUESTION_START source")
        if self.submit_frame.source != FrameSource.QUESTION_SUBMIT:
            raise ValueError("submit_frame must have QUESTION_SUBMIT source")
        if self.input_frame.session_id != self.session_id:
            raise ValueError("input_frame session mismatch")
        if self.submit_frame.session_id != self.session_id:
            raise ValueError("submit_frame session mismatch")


@dataclass(frozen=True, slots=True)
class AnswerDelta:
    """스트리밍으로 추가된 텍스트 조각. index는 질문당 0부터 시작."""

    session_id: str
    question_id: str
    index: int
    text_delta: str


@dataclass(frozen=True, slots=True)
class AnswerResult:
    """정상 완료된 답변. used_*는 '모델에 제공한' 자료이며 실제 활용 검증은 아님."""

    session_id: str
    question_id: str
    answer: str
    context_tag: int | None
    used_frame_ids: tuple[str, ...]
    used_segment_ids: tuple[str, ...]
    first_response_latency_ms: int | None
    elapsed_ms: int


@dataclass(frozen=True, slots=True)
class AnswerCancelled:
    """사용자 중단으로 완료되지 않은 답변. 검색용 Q&A 이력에서는 제외."""

    session_id: str
    question_id: str
    partial_answer: str
    elapsed_ms: int


@dataclass(frozen=True, slots=True)
class AnswerFailed:
    """질의응답 실패로 질문 상태를 해제하는 종료 이벤트."""

    session_id: str
    question_id: str
    code: str
    message: str
    elapsed_ms: int


@dataclass(frozen=True, slots=True)
class ModuleError:
    session_id: str | None
    module: ModuleName
    code: str
    message: str
    recoverable: bool
    timestamp_ms: int | None = None
    question_id: str | None = None


IngestEvent: TypeAlias = FrameRecord | TranscriptSegment
AnswerEvent: TypeAlias = AnswerDelta | AnswerResult | AnswerCancelled | AnswerFailed
AnswerEventCallback: TypeAlias = Callable[[AnswerEvent], None]
ErrorCallback: TypeAlias = Callable[[ModuleError], None]


class SessionClockView(Protocol):
    """작업 모듈에는 읽기 전용 공통 시간 참조만 제공한다."""

    def now_ms(self) -> int: ...


class SessionClockControl(SessionClockView, Protocol):
    """공통 시계의 변경 권한은 앱/시스템 통합에만 부여한다."""

    def pause(self) -> None: ...

    def resume(self) -> None: ...


class FrameProcessorAPI(Protocol):
    def set_region(self, region: CaptureRegion) -> None: ...

    def start(
        self,
        session_id: str,
        clock: SessionClockView,
        output_queue: Queue[IngestEvent],
        on_error: ErrorCallback,
    ) -> None: ...

    def pause(self) -> None:
        """새 화면 수집을 중단한다. 기존 수집분의 처리는 계속할 수 있다."""
        ...

    def prepare_resume(self) -> None: ...

    def resume(self) -> None: ...

    def stop(self) -> None: ...

    def capture_now(self, source: FrameSource) -> FrameRecord:
        """QUESTION_START/SUBMIT 전용. P/C/R 정기 처리에는 영향 없음."""
        ...


class AudioProcessorAPI(Protocol):
    def set_device(self, device_id: str) -> None: ...

    def start(
        self,
        session_id: str,
        clock: SessionClockView,
        output_queue: Queue[IngestEvent],
        on_error: ErrorCallback,
    ) -> None: ...

    def pause(self) -> None:
        """새 오디오 수집을 중단한다. 기존 수집분의 전사는 계속할 수 있다."""
        ...

    def prepare_resume(self) -> None: ...

    def resume(self) -> None: ...

    def stop(self) -> None: ...


class RAGIndexAPI(Protocol):
    """맥락 관리 모듈 내부에서 사용하는 독립 RAG 검색 하위 구성요소.

    호출·인덱싱은 UI와 수집 큐를 차단하지 않는 백그라운드 작업에서 수행한다.
    실제 SQLite 쓰기와 세션 DB 소유권은 맥락 관리 모듈에 있다.
    """

    def start(
        self, session_id: str, restored_chunks: tuple[RAGChunkRecord, ...]
    ) -> None:
        """기존 저장 청크로 메모리 검색 인덱스를 복원한다."""
        ...

    def index_transcript(self, segment: TranscriptSegment) -> tuple[RAGChunkRecord, ...]:
        """전사를 내부 버퍼/인덱스에 반영하고 완성된 청크만 반환한다.

        호출자는 반환된 청크와 임베딩을 SQLite에 저장한다.
        청크가 아직 완성되지 않았다면 빈 튜플을 반환할 수 있다.
        """
        ...

    def search(
        self, query: str, top_k: int, until_ms: int | None = None
    ) -> tuple[RAGSearchHit, ...]:
        """키워드+의미 기반 검색 결과를 점수 내림차순으로 반환한다.

        until_ms가 있으면 그 시점 이후의 청크를 제외한다.
        인덱싱 작업과 동시 호출되어도 일관된 결과를 반환해야 한다.
        """
        ...

    def flush_pending(self) -> tuple[RAGChunkRecord, ...]:
        """정상 종료 시 미완성 텍스트 버퍼를 청크로 확정하여 반환한다."""
        ...

    def stop(self) -> None: ...


class ContextQAAPI(Protocol):
    def start(
        self,
        session_id: str,
        session_dir: str,
        input_queue: Queue[IngestEvent],
        on_answer_event: AnswerEventCallback,
        on_error: ErrorCallback,
    ) -> None: ...

    def ask_async(self, request: QuestionRequest) -> None:
        """즉시 반환. 질문은 하나씩만 처리하며 AnswerEvent로 결과 전달."""
        ...

    def cancel_question(self, question_id: str) -> None: ...

    def drain_and_stop(self) -> None:
        """수집 이벤트/임베딩 작업을 처리하고 저장을 커밋한 뒤 종료."""
        ...


class AppControllerAPI(Protocol):
    def start_session(self, region: CaptureRegion, audio_device_id: str) -> str: ...

    def set_capture_region(self, region: CaptureRegion) -> None: ...

    def pause_session(self) -> None: ...

    def resume_session(self) -> None: ...

    def stop_session(self) -> None: ...

    def begin_question(self) -> str:
        """첫 비어 있지 않은 입력 시 question_id 생성 및 시작 프레임 캡처."""
        ...

    def discard_question_draft(self, question_id: str) -> None: ...

    def submit_question(self, question_id: str, text: str) -> None:
        """사용자 제출 순간의 프레임을 캡처한 뒤 비동기 질의 요청."""
        ...

    def cancel_question(self, question_id: str) -> None: ...
