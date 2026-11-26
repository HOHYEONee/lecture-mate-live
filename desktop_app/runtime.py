"""Inspect host hardware and select the demo's independent inference devices."""

from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
import subprocess
import sys


GPU_WHISPER = "large-v3-turbo"
CPU_WHISPER = "base"
MODEL_ROOT = Path(__file__).resolve().parents[1] / "models"
PROBE_PREFIX = "CLUE_PROBE="
log = logging.getLogger(__name__)


def _command(args: list[str], timeout: int) -> subprocess.CompletedProcess:
    environment = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    environment.pop("OPENAI_API_KEY", None)
    return subprocess.run(
        args, capture_output=True, encoding="utf-8", errors="replace", timeout=timeout,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        env=environment,
    )


def nvidia_gpus() -> list[dict]:
    fields = "index,uuid,name,driver_version,memory.total,memory.free"
    try:
        result = _command([
            "nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits",
        ], timeout=4)
    except (OSError, subprocess.TimeoutExpired):
        return []
    gpus = []
    if result.returncode == 0:
        for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
            if len(row) == 6:
                index, uuid, name, driver, total, free = (field.strip() for field in row)
                gpus.append(dict(
                    index=int(index) if index.isdecimal() else None, uuid=uuid,
                    vendor="NVIDIA", name=name, driver=driver,
                    total_mib=int(total) if total.isdecimal() else None,
                    free_mib=int(free) if free.isdecimal() else None,
                ))
    return gpus


def hardware_inventory() -> dict:
    nvidia = nvidia_gpus()
    adapters = []
    if os.name == "nt":
        try:
            result = _command([
                "powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                "[Console]::OutputEncoding=[Text.Encoding]::UTF8; "
                "Get-CimInstance Win32_VideoController | "
                "Select-Object Name,AdapterCompatibility,DriverVersion | ConvertTo-Json -Compress",
            ], timeout=6)
            if result.returncode == 0 and result.stdout.strip():
                values = json.loads(result.stdout)
                adapters = values if isinstance(values, list) else [values]
        except (OSError, ValueError, subprocess.TimeoutExpired):
            log.debug("Windows video adapter inventory unavailable", exc_info=True)
    return {"adapters": adapters, "nvidia": nvidia}


def has_model_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def local_whisper_model(name: str) -> Path:
    custom = Path(name).expanduser()
    under_project = MODEL_ROOT.parent / custom
    if name in {CPU_WHISPER, GPU_WHISPER}:
        model = MODEL_ROOT / name
    elif custom.is_dir():
        model = custom.resolve()
    elif under_project.is_dir() or custom.is_absolute():
        model = under_project
    else:
        model = MODEL_ROOT / custom
    required = ("config.json", "model.bin", "tokenizer.json")
    if not all(has_model_file(model / item) for item in required) or not (
        has_model_file(model / "vocabulary.txt") or has_model_file(model / "vocabulary.json")
    ):
        raise FileNotFoundError(f"Whisper 모델 파일이 없습니다: {model}")
    return model


def run_probe(kind: str, model: Path | None = None) -> dict:
    args = [sys.executable, "-m", "desktop_app.runtime_probe", kind]
    if model is not None:
        args.append(str(model))
    try:
        result = _command(args, timeout=90)
    except subprocess.TimeoutExpired:
        return {"ok": False, "reason": "워밍업 검사 90초 제한 초과"}
    except OSError as exc:
        return {"ok": False, "reason": str(exc)}
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(PROBE_PREFIX) and result.returncode == 0:
            try:
                return json.loads(line[len(PROBE_PREFIX):])
            except ValueError:
                break
    details = result.stderr.strip().splitlines()
    suffix = f" · {details[-1][:280]}" if details else ""
    return {"ok": False, "reason": f"검사 프로세스 종료 코드 {result.returncode}{suffix}"}


class AutoFrameClassifier:
    """Select once per frame session and keep the selected classifier consistent."""

    def __init__(self, notice, *, require_gpu: bool = False) -> None:
        self._notice = notice
        self._require_gpu = require_gpu
        self._classifier = None

    def warm_up(self, width: int, height: int) -> None:
        from frame_processing.classification import PlusLClassifier
        from frame_processing.experiments.heuristic import LayoutHeuristicClassifier

        self._notice("프레임 GPU 호환성을 확인하는 중입니다.")
        probe = run_probe("frame")
        reason = probe.get("reason")
        if probe["ok"]:
            try:
                classifier = PlusLClassifier(f"gpu:{probe['index']}")
                classifier.warm_up(width, height)
            except Exception as exc:
                if self._require_gpu:
                    raise
                reason = str(exc)
            else:
                self._classifier = classifier
                self._notice(f"프레임 GPU · {probe['name']} · PP-DocLayout_plus-L")
                return
        if self._require_gpu:
            raise RuntimeError(f"GPU 프레임 분류 초기화 실패: {reason}")
        log.warning("Frame GPU unavailable: %s", reason)
        self._classifier = LayoutHeuristicClassifier()
        self._notice(f"프레임 CPU 호환 모드 · {reason}")

    def classify(self, bgr):
        return self._classifier.classify(bgr)


def automatic_audio_options(gpu_model: str, notice):
    from audio_processing import AudioProcessor
    from audio_processing.backends import FasterWhisperTranscriber

    for kind, model_name in (("asr-gpu", gpu_model), ("asr-cpu", CPU_WHISPER)):
        try:
            model = local_whisper_model(model_name)
        except FileNotFoundError as exc:
            probe = {"ok": False, "reason": str(exc)}
        else:
            notice(f"음성 {model_name} 실행 환경을 확인하는 중입니다.")
            probe = run_probe(kind, model)
        if not probe["ok"]:
            log.warning("%s unavailable: %s", kind, probe["reason"])
            notice(f"음성 {'GPU' if kind == 'asr-gpu' else 'CPU'} 사용 제한 · {probe['reason']}")
            continue
        device = "cuda" if kind == "asr-gpu" else "cpu"
        yield AudioProcessor(transcriber=FasterWhisperTranscriber(
            str(model), device=device, device_index=probe.get("index", 0),
            compute_type=probe["compute_type"], local_files_only=True, project_cuda_only=True,
        )), f"음성 {'GPU' if device == 'cuda' else 'CPU'} · {model_name} · {probe['compute_type']}"


def diagnostics() -> dict:
    from importlib.util import find_spec

    from frame_processing.classification import DEFAULT_MODEL_DIR, MODEL_DIR_ENV, MODEL_FILES
    from rag.embeddings import default_embedding_is_ready

    result = hardware_inventory()
    frame_path = Path(os.environ.get(MODEL_DIR_ENV) or DEFAULT_MODEL_DIR)
    result["packages"] = {name: find_spec(name) is not None for name in
                          ("paddle", "paddleocr", "ctranslate2", "faster_whisper", "torch", "openai", "pyaudiowpatch")}
    result["frame_model_missing"] = [name for name in MODEL_FILES if not has_model_file(frame_path / name)]
    result["embedding_model_ready"] = (
        find_spec("huggingface_hub") is not None and default_embedding_is_ready()
    )
    result["frame"] = run_probe("frame")
    for kind, name in (("asr-gpu", os.environ.get("CLUE_ASR_MODEL") or GPU_WHISPER),
                       ("asr-cpu", CPU_WHISPER)):
        try:
            model = local_whisper_model(name)
        except FileNotFoundError as exc:
            result[kind] = {"ok": False, "reason": str(exc)}
        else:
            result[kind] = run_probe(kind, model)
    result["real_qa_key_configured"] = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    return result
