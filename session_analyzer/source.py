"""Read source literals without importing the CLUE runtime or AI dependencies."""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def literals(relative: str) -> dict:
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8-sig"))
    result = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        result[target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
    return result


def defaults(relative: str, owner: str, method: str) -> dict:
    tree = ast.parse((ROOT / relative).read_text(encoding="utf-8-sig"))
    scope = next(n for n in tree.body if getattr(n, "name", None) == owner) if owner else tree
    function = next(n for n in scope.body if getattr(n, "name", None) == method)
    pairs = list(zip(function.args.args[-len(function.args.defaults):], function.args.defaults))
    pairs += list(zip(function.args.kwonlyargs, function.args.kw_defaults))
    result = {}
    for arg, value in pairs:
        try:
            result[arg.arg] = ast.literal_eval(value)
        except (ValueError, TypeError):
            pass
    return result


def code_defaults() -> dict:
    """These are current checkout defaults, never a session configuration claim."""
    try:
        comparison = literals("frame_processing/comparison.py")
        audio = defaults("audio_processing/processor.py", "AudioProcessor", "__init__")
        backend = literals("audio_processing/backends.py")
        selection = defaults("context_qa/selector.py", "", "select")
        return {
            "provenance": "현재 체크아웃의 코드 기본값 · 세션 실행 설정은 DB에 미기록",
            "pc_threshold": comparison.get("DEFAULT_GROUP_THRESHOLD"),
            "rc_threshold": comparison.get("DEFAULT_REPRESENTATIVE_THRESHOLD"),
            "audio_period_seconds": audio.get("period_seconds"),
            "audio_overlap_seconds": audio.get("overlap_seconds"),
            "boundary_holdback_ms": audio.get("boundary_holdback_ms"),
            "asr_model": backend.get("DEFAULT_MODEL"),
            "recent_window_ms": selection.get("recent_window_ms"),
            "references": ["frame_processing/comparison.py", "audio_processing/processor.py",
                           "audio_processing/backends.py", "context_qa/selector.py"],
        }
    except (OSError, SyntaxError, StopIteration):
        return {"provenance": "코드 기본값 확인 불가"}
