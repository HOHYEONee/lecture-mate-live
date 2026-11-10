from __future__ import annotations

import base64
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Protocol, TypeAlias

DEFAULT_MODEL = "gpt-6.1-sol"

_MIME_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


@dataclass(frozen=True, slots=True)
class TextPart:
    text: str


@dataclass(frozen=True, slots=True)
class ImagePart:
    path: Path


ContentPart: TypeAlias = TextPart | ImagePart


@dataclass(frozen=True, slots=True)
class LLMRequest:
    instructions: str
    parts: tuple[ContentPart, ...]


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int
    output_tokens: int


class LLMError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class LLMStream(Protocol):
    usage: Usage | None

    def __iter__(self) -> Iterator[str]: ...

    def close(self) -> None: ...


class LLMClient(Protocol):
    def stream(self, request: LLMRequest) -> LLMStream: ...


class _GeneratorStream:
    def __init__(self) -> None:
        self.usage: Usage | None = None
        self._gen: Iterator[str] | None = None

    def _bind(self, gen: Iterator[str]) -> _GeneratorStream:
        self._gen = gen
        return self

    def __iter__(self) -> Iterator[str]:
        assert self._gen is not None
        return self._gen

    def close(self) -> None:
        close = getattr(self._gen, "close", None)
        if close is not None:
            close()


class FakeLLM:
    def __init__(
        self,
        text: str = "화면과 전사를 근거로 한 테스트 답변입니다.",
        chunk_size: int = 4,
        delay_s: float = 0.0,
        error: Exception | None = None,
        fail_after_chunks: int = 0,
        hold: threading.Event | None = None,
    ) -> None:
        self.text = text
        self.chunk_size = chunk_size
        self.delay_s = delay_s
        self.error = error
        self.fail_after_chunks = fail_after_chunks
        self.hold = hold
        self.requests: list[LLMRequest] = []

    def stream(self, request: LLMRequest) -> LLMStream:
        self.requests.append(request)
        stream = _GeneratorStream()
        return stream._bind(self._generate(stream))

    def _generate(self, stream: _GeneratorStream) -> Iterator[str]:
        chunks = [self.text[i : i + self.chunk_size] for i in range(0, len(self.text), self.chunk_size)]
        for n, chunk in enumerate(chunks):
            if self.error is not None and n == self.fail_after_chunks:
                raise self.error
            if n == 1 and self.hold is not None:
                self.hold.wait(timeout=5)
            if self.delay_s:
                time.sleep(self.delay_s)
            yield chunk
        if self.error is not None and self.fail_after_chunks >= len(chunks):
            raise self.error
        stream.usage = Usage(input_tokens=len(self.requests[-1].parts), output_tokens=len(chunks))


class OpenAIClient:
    def __init__(
        self,
        model: str | None = None,
        reasoning_effort: str | None = None,
        image_detail: str = "auto",
        max_output_tokens: int | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model or os.environ.get("CLUE_MODEL") or DEFAULT_MODEL
        self.reasoning_effort = reasoning_effort or os.environ.get("CLUE_REASONING_EFFORT") or None
        self.image_detail = image_detail
        self.max_output_tokens = max_output_tokens
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise LLMError("llm_unavailable", "openai package is not installed") from exc
            try:
                client = OpenAI()
            except Exception as exc:
                raise LLMError("llm_unavailable", str(exc)) from exc
        self._client = client

    def build_input(self, request: LLMRequest) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        for part in request.parts:
            if isinstance(part, TextPart):
                content.append({"type": "input_text", "text": part.text})
            else:
                content.append(
                    {"type": "input_image", "image_url": _data_url(part.path), "detail": self.image_detail}
                )
        return [{"role": "user", "content": content}]

    def stream(self, request: LLMRequest) -> LLMStream:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "instructions": request.instructions,
            "input": self.build_input(request),
            "stream": True,
        }
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        if self.max_output_tokens:
            kwargs["max_output_tokens"] = self.max_output_tokens
        try:
            events = self._client.responses.create(**kwargs)
        except Exception as exc:
            raise LLMError("llm_request_failed", f"{type(exc).__name__}: {exc}") from exc
        stream = _GeneratorStream()
        return stream._bind(self._read(events, stream))

    def _read(self, events: Any, stream: _GeneratorStream) -> Iterator[str]:
        try:
            for event in events:
                kind = getattr(event, "type", "")
                if kind == "response.output_text.delta":
                    yield event.delta
                elif kind == "response.completed":
                    usage = getattr(event.response, "usage", None)
                    if usage is not None:
                        stream.usage = Usage(usage.input_tokens, usage.output_tokens)
                elif kind == "response.failed":
                    error = getattr(event.response, "error", None)
                    raise LLMError("llm_response_failed", getattr(error, "message", "response failed"))
                elif kind == "response.incomplete":
                    details = getattr(event.response, "incomplete_details", None)
                    raise LLMError("llm_incomplete", str(getattr(details, "reason", "incomplete")))
                elif kind == "error":
                    raise LLMError("llm_stream_error", getattr(event, "message", "stream error"))
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError("llm_stream_error", f"{type(exc).__name__}: {exc}") from exc
        finally:
            close = getattr(events, "close", None)
            if close is not None:
                close()


def _data_url(path: Path) -> str:
    mime = _MIME_TYPES.get(path.suffix.lower())
    if mime is None:
        raise LLMError("unsupported_image", f"unsupported image type: {path.name}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"
