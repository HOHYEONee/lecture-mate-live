"""Evidence-based metrics and ID joins, independent of the application runtime."""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter
import json
import math
from pathlib import Path
import re
import statistics

from .source import code_defaults


def number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0 else None


def stats(values) -> dict:
    values = [v for x in values if (v := number(x)) is not None]
    return {"count": len(values), "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None, "max": max(values) if values else None}


def time_range(rows, *keys):
    values = [v for r in rows for k in keys if (v := number(r.get(k))) is not None]
    return {"start_ms": min(values) if values else None, "end_ms": max(values) if values else None,
            "span_ms": max(values) - min(values) if values else None}


def array_ids(value, warnings, label):
    if value is None:
        return None
    try:
        result = json.loads(value)
        if not isinstance(result, list) or any(not isinstance(x, str) for x in result):
            raise ValueError
        return result
    except (ValueError, TypeError):
        warnings.append(f"{label}: ID JSON 손상 · 선택 기록 확인 불가")
        return None


def redact(value):
    """Mask common credentials/contact details in exported text, never read .env."""
    if isinstance(value, str):
        value = re.sub(r"\b(?:sk-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9_]{15,}|github_pat_[A-Za-z0-9_]{15,})", "[KEY REDACTED]", value)
        value = re.sub(r"(?i)(api[_ -]?key\s*[:=]\s*|bearer\s+)[\w.\-]{8,}", r"\1[REDACTED]", value)
        value = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", "[EMAIL REDACTED]", value)
        return re.sub(r"(?<!\d)01[016789][- ]?\d{3,4}[- ]?\d{4}(?!\d)", "[PHONE REDACTED]", value)
    if isinstance(value, list):
        return [redact(x) for x in value]
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items()}
    return value


def image_file(folder: Path, relative) -> Path | None:
    if not isinstance(relative, str) or not relative:
        return None
    # Resolve Windows paths as well as POSIX paths; reject traversal and symlinks out.
    path = (folder / relative.replace("\\", "/")).resolve()
    if not path.is_relative_to(folder.resolve()) or not path.is_file():
        return None
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".svg"}:
        return None
    return path


def analyze(raw: dict) -> dict:
    t = raw["tables"]
    warnings = list(raw["warnings"])
    frames = sorted(t["frames"], key=lambda r: (number(r["first_timestamp_ms"]) or 0, number(r["first_capture_seq"]) or 0))
    history = sorted(t["frame_history"], key=lambda r: (number(r["timestamp_ms"]) or 0, number(r["capture_seq"]) or 0))
    transcripts = sorted(t["transcripts"], key=lambda r: (number(r["start_ms"]) or 0, str(r["segment_id"])))
    chunks = sorted(t["text_chunks"], key=lambda r: (number(r["start_ms"]) or 0, str(r["chunk_id"])))
    frame_map = {r["frame_id"]: r for r in frames}
    segment_map = {r["segment_id"]: r for r in transcripts}
    periodic = [r for r in frames if r["source"] == "periodic"]
    observations = [r for r in history if r["source"] == "periodic"]
    timed = [r for r in observations if number(r["timestamp_ms"]) is not None]
    times = [r["timestamp_ms"] for r in timed]
    intervals = [b - a for a, b in zip(times, times[1:])]
    asset_paths, jpeg_paths = set(), set()
    for frame in frames:
        path = image_file(raw["folder"], frame["image_path"])
        frame["file_bytes"] = path.stat().st_size if path else None
        frame["image_missing"] = path is None
        frame["asset"] = None
        if path:
            asset_paths.add(path)
            if path.suffix.lower() in {".jpg", ".jpeg"}:
                jpeg_paths.add(path)
    # Count physical JPEG files, including files without a corresponding DB row.
    frame_folder = raw["folder"] / "frames"
    if frame_folder.is_dir():
        for candidate in frame_folder.rglob("*"):
            path = candidate.resolve()
            if path.is_relative_to(raw["folder"]) and path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}:
                jpeg_paths.add(path)
    assets = sum(path.stat().st_size for path in asset_paths)
    jpeg_bytes = sum(path.stat().st_size for path in jpeg_paths)
    jpeg_count = len(jpeg_paths)
    missing_images = sum(r["image_missing"] for r in frames)
    if missing_images:
        warnings.append(f"이미지 {missing_images}개 누락 또는 세션 외부/지원하지 않는 경로 · 메타데이터는 유지")
    groups = {}
    for i, row in enumerate(observations):
        key = row["group_id"]
        if key is None:
            continue
        group = groups.setdefault(key, {"group_id": key, "created_ms": None, "observations": 0,
                                       "frame_ids": [], "changes": [], "reuse_intervals": []})
        group["observations"] += 1
        if row["frame_id"] not in group["frame_ids"]:
            group["frame_ids"].append(row["frame_id"])
        if row["decision"] == "new_group":
            group["created_ms"] = row["timestamp_ms"]
        if row["decision"] in {"new_group", "new_representative"}:
            group["changes"].append({"timestamp_ms": row["timestamp_ms"], "frame_id": row["frame_id"], "decision": row["decision"]})
        if row["decision"] == "reused":
            group["reuse_intervals"].append({"start_ms": row["timestamp_ms"],
                "end_ms": observations[i + 1]["timestamp_ms"] if i + 1 < len(observations) else None,
                "frame_id": row["frame_id"]})
    gaps, covered_until = [], None
    for segment in transcripts:
        start, end = number(segment["start_ms"]), number(segment["end_ms"])
        valid = start is not None and end is not None and end >= start
        segment["duration_ms"] = end - start if valid else None
        if not valid:
            warnings.append(f"전사 {segment['segment_id']}: 유효하지 않은 발화 시간 · 지속 시간 미산출")
        if valid:
            if covered_until is not None and start > covered_until:
                gaps.append({"start_ms": covered_until, "end_ms": start, "duration_ms": start - covered_until})
            covered_until = max(covered_until or end, end)
        index = bisect_right(times, start) - 1 if start is not None else -1
        observation = timed[index] if index >= 0 else None
        segment["observed_frame_id"] = observation["frame_id"] if observation else None
        segment["observation_ms"] = observation["timestamp_ms"] if observation else None
        segment["observation_decision"] = observation["decision"] if observation else None
    embeddings = {r["chunk_id"]: r for r in t["chunk_embeddings"]}
    for chunk in chunks:
        chunk["segment_ids"] = array_ids(chunk["segment_ids"], warnings, f"청크 {chunk['chunk_id']}")
        start, end = number(chunk["start_ms"]), number(chunk["end_ms"])
        chunk["duration_ms"] = end - start if start is not None and end is not None and end >= start else None
        embedding = embeddings.get(chunk["chunk_id"])
        chunk["embedding_exists"] = bool(embedding) if raw["schema"]["chunk_embeddings"] else None
        chunk["embedding_model"] = embedding["embedding_model"] if embedding else None
        chunk["dim"] = embedding["dim"] if embedding else None
    answers = {r["question_id"]: r for r in t["answers"]}
    questions = sorted(t["questions"], key=lambda r: (number(r["submitted_ms"]) or 0, str(r["question_id"])))
    known = {r["question_id"] for r in questions}
    for qid in answers.keys() - known:
        warnings.append(f"답변 {qid}: 대응 질문 누락 · 별도 표시")
        questions.append({"question_id": qid, "text": None, "question_type": None,
                          "submitted_ms": None, "input_started_ms": None})
    settings = code_defaults()
    for index, q in enumerate(questions, 1):
        a = answers.get(q["question_id"])
        q["order"] = index
        q["answer_recorded"] = a is not None
        for field in CARRY:
            q[field] = a.get(field) if a else None
        q["status"] = (a["status"] or "unknown") if a else "unanswered"
        for key, lookup in (("used_frame_ids", frame_map), ("used_segment_ids", segment_map)):
            q[key] = array_ids(a[key], warnings, f"질문 {q['question_id']} {key}") if a else None
            missing = [uid for uid in (q[key] or []) if uid not in lookup]
            if missing:
                warnings.append(f"질문 {q['question_id']}: {key} 참조 {len(missing)}개 누락")
        q["rag_performed"] = True if q["rag_search_ms"] is not None else None
        # NULL is not proof of 'not searched': older schemas and interrupted work exist.
        q["related_previous_ids"] = [p["question_id"] for p in questions[:index - 1]
            if q["context_tag"] is not None and p.get("context_tag") == q["context_tag"]
            and p.get("status") == "completed" and number(p.get("submitted_ms")) is not None
            and number(q.get("submitted_ms")) is not None and p["submitted_ms"] < q["submitted_ms"]]
    completed = [q for q in questions if q["status"] == "completed"]
    statuses = dict(Counter(q["status"] for q in questions))
    overall_rows = ([{"start": r.get("first_timestamp_ms")} for r in frames]
        + [{"start": r.get("timestamp_ms")} for r in history]
        + [{"start": r.get("start_ms"), "end": r.get("end_ms")} for r in transcripts + chunks]
        + [{"start": r.get("input_started_ms"), "end": r.get("submitted_ms")} for r in questions])
    available_periodic = "source" in raw["schema"]["frames"]
    available_history = "source" in raw["schema"]["frame_history"]
    reduction = (1 - len(periodic) / len(observations)) * 100 if available_periodic and available_history and observations else None
    if reduction is not None and not 0 <= reduction <= 100:
        warnings.append("정기 대표 프레임 수가 관측 수보다 많습니다. 저장 감소율 산출 불가")
        reduction = None
    marker = raw["folder"] / "SYNTHETIC_SESSION.json"
    synthetic = False
    if marker.is_file():
        try:
            synthetic = json.loads(marker.read_text(encoding="utf-8")).get("synthetic") is True
        except (ValueError, OSError):
            warnings.append("합성 데이터 표식 파일 손상")
    summary = {
        "session_id": raw["session_id"], "synthetic": synthetic,
        "data_origin": "합성 데이터 · 실제 측정 아님" if synthetic else "사용자 제공 세션 기록 · 실측 여부 별도 확인",
        "observed_range": time_range(overall_rows, "start", "end"),
        "range_basis": "관측된 데이터 범위 · 세션 실행 시간/시작/종료는 미기록 · 일시정지 제외 세션 공통 시계",
        "periodic_observations": len(observations) if available_history else None,
        "periodic_frames": len(periodic) if available_periodic else None,
        "snapshot_start": sum(r["source"] == "question_start" for r in frames) if available_periodic else None,
        "snapshot_submit": sum(r["source"] == "question_submit" for r in frames) if available_periodic else None,
        "frame_count": len(frames) if raw["schema"]["frames"] else None,
        "reused_observations": sum(r["decision"] == "reused" for r in observations) if available_history else None,
        "new_groups": sum(r["decision"] == "new_group" for r in observations) if available_history else None,
        "new_representatives": sum(r["decision"] == "new_representative" for r in observations) if available_history else None,
        "storage_reduction_pct": reduction, "capture_interval_ms": stats(intervals),
        "image_bytes": assets, "jpeg_bytes": jpeg_bytes, "jpeg_count": jpeg_count, "missing_images": missing_images,
        "classification": dict(Counter(r["frame_type"] or "미기록" for r in periodic)),
        "transcript_count": len(transcripts) if raw["schema"]["transcripts"] else None,
        "transcript_range": time_range(transcripts, "start_ms", "end_ms"),
        "utterance_duration_ms": stats(r["duration_ms"] for r in transcripts),
        "transcript_gaps": len(gaps), "chunk_count": len(chunks) if raw["schema"]["text_chunks"] else None,
        "question_count": len(t["questions"]) if raw["schema"]["questions"] else None,
        "orphan_answer_count": len(answers.keys() - known), "statuses": statuses,
        "completed_first_ms": stats(q["first_response_latency_ms"] for q in completed),
        "completed_total_ms": stats(q["elapsed_ms"] for q in completed),
        "models": sorted({q["model"] for q in questions if q["model"]}),
        "response_by_type": {kind: {"first_ms": stats(q["first_response_latency_ms"] for q in completed if (q["question_type"] or "미기록") == kind),
            "total_ms": stats(q["elapsed_ms"] for q in completed if (q["question_type"] or "미기록") == kind)}
            for kind in sorted({q["question_type"] or "미기록" for q in completed})},
    }
    return {"summary": summary, "defaults": settings, "frames": frames, "history": history,
            "transcripts": transcripts, "chunks": chunks, "questions": questions,
            "groups": list(groups.values()), "gaps": gaps, "schema": raw["schema"],
            "warnings": list(dict.fromkeys(warnings))}


CARRY = ("answer", "context_tag", "first_response_latency_ms", "elapsed_ms", "error_code",
         "input_tokens", "output_tokens", "model", "strategy", "rag_search_ms", "retries")
