"""Save the OpenAI credential in this installation's local configuration."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import set_key


PROJECT_ENV_FILE = Path(__file__).resolve().parents[1] / ".env"


def save_openai_api_key(key: str, path: Path | None = None) -> None:
    key = key.strip()
    if not key or any(character.isspace() for character in key):
        raise ValueError("공백 없는 OpenAI API 키를 입력하세요.")
    saved, _, _ = set_key(path or PROJECT_ENV_FILE, "OPENAI_API_KEY", key, quote_mode="always")
    if not saved:
        raise OSError("이 설치 폴더의 .env에 API 키를 저장할 수 없습니다.")
    os.environ["OPENAI_API_KEY"] = key
