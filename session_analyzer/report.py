"""Portable static HTML, copied images and UTF-8 exports; no web services."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import shutil

from .analysis import image_file, redact

ASSETS = Path(__file__).with_name("assets")


def stamp(ms) -> str:
    if ms is None:
        return "미기록"
    seconds, milliseconds = divmod(int(ms), 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"


def csv_cell(value):
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False)
    # A transcript/question must not become an Excel formula when exported.
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None):
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: csv_cell(row.get(k)) for k in fields})


def safe_name(value: str) -> str:
    name = re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE).strip(". ")[:80]
    if not name or name.upper().split(".")[0] in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
        name = "session"
    return name + "-" + hashlib.sha256(value.encode()).hexdigest()[:8]


def write_report(data: dict, session: Path, output_root: Path) -> Path:
    session, output_root = session.resolve(), output_root.resolve()
    if output_root.is_relative_to(session) or session.is_relative_to(output_root):
        raise ValueError("분석 출력 폴더는 원본 세션 폴더와 분리해 주세요.")
    output_root.mkdir(parents=True, exist_ok=True)
    name = safe_name(data["summary"]["session_id"])
    # Never erase an earlier report or a user-created directory.
    for suffix in range(10000):
        output = output_root / (name + (f"-{suffix:03}" if suffix else ""))
        try:
            output.mkdir()
            break
        except FileExistsError:
            continue
    else:
        raise ValueError("출력 이름을 생성할 수 없습니다.")
    images = output / "images"
    images.mkdir()
    copied = {}
    for frame in data["frames"]:
        path = image_file(session, frame["image_path"])
        if path:
            if path not in copied:
                target = images / (hashlib.sha256(str(path.relative_to(session)).encode()).hexdigest()[:24] + path.suffix.lower())
                try:
                    shutil.copyfile(path, target)
                    copied[path] = "images/" + target.name
                except OSError:
                    data["warnings"].append("분석 중 이미지 복사 실패 · 누락 표시로 대체")
                    frame["image_missing"] = True
            frame["asset"] = copied.get(path)
        # Do not expose laptop/desktop absolute image paths in exports.
        frame["image_path"] = frame["asset"]
    data = redact(data)
    data["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    for name in ("dashboard.css", "dashboard.js"):
        shutil.copyfile(ASSETS / name, output / name)
    (output / "summary.json").write_text(json.dumps(data["summary"], ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (output / "session_analysis.json").write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    exports = {
        "frame_analysis.csv": (data["history"], "capture_seq timestamp_ms frame_id group_id source decision"),
        "frame_metadata.csv": (data["frames"], "frame_id first_timestamp_ms first_capture_seq group_id source decision frame_type image_path file_bytes image_missing"),
        "transcript_analysis.csv": (data["transcripts"], "segment_id start_ms end_ms duration_ms text observed_frame_id observation_ms observation_decision"),
        "transcript_gaps.csv": (data["gaps"], "start_ms end_ms duration_ms"),
        "rag_chunks.csv": (data["chunks"], "chunk_id start_ms end_ms duration_ms text segment_ids embedding_exists embedding_model dim"),
        "question_context.csv": (data["questions"], "order question_id text question_type input_started_ms submitted_ms input_capture_seq submit_capture_seq context_tag used_frame_ids used_segment_ids related_previous_ids rag_performed rag_search_ms answer status"),
        "response_analysis.csv": (data["questions"], "question_id question_type submitted_ms answer_recorded status first_response_latency_ms elapsed_ms rag_search_ms input_tokens output_tokens model strategy retries error_code"),
    }
    for filename, (rows, fields) in exports.items():
        tagged = [{"synthetic": data["summary"]["synthetic"], "data_origin": data["summary"]["data_origin"], **row} for row in rows]
        write_csv(output / filename, tagged, ["synthetic", "data_origin", *fields.split()])
    write_csv(output / "frame_summary.csv", [{k: v for k, v in data["summary"].items() if k in {"synthetic", "data_origin"} or k.startswith(("periodic", "snapshot", "storage", "capture", "image", "jpeg", "reused", "new_", "classification"))}])
    (output / "transcripts.txt").write_text(data["summary"]["data_origin"] + "\n\n" + "\n".join(
        f"[{stamp(s['start_ms'])} → {stamp(s['end_ms'])}] {s['text'] or ''}" for s in data["transcripts"]), encoding="utf-8-sig")
    (output / "README.txt").write_text(
        f"CLUE Session Analyzer\n{data['summary']['data_origin']}\n"
        "index.html을 브라우저에서 여세요. 폴더 전체를 함께 옮기면 오프라인으로 열립니다.\n"
        "시각 단위: 세션 공통 시계 ms (일시정지 제외). 실행 시간과 관측 범위를 구분하세요.\n"
        "빈 CSV 셀/JSON null은 미기록입니다. 손상된 ID 배열도 null로 표시합니다.\n"
        "텍스트의 일반적인 키·이메일·휴대전화 패턴은 마스킹합니다. 이미지와 그 밖의 개인정보는 자동 익명화하지 않습니다.\n"
        "공유 전 이미지·전사·답변을 확인하세요. 보고서는 Git에 포함하지 마세요.\n", encoding="utf-8")
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    payload = payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    template = (ASSETS / "index.html").read_text(encoding="utf-8")
    document = template.replace("__SESSION__", html.escape(str(data["summary"]["session_id"]))).replace("__DATA__", payload)
    (output / "index.html").write_text(document, encoding="utf-8")
    return output
