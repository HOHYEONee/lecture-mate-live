"""Run real inference in a disposable process before selecting a native runtime."""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
from time import monotonic
from uuid import UUID

from .runtime import PROBE_PREFIX, nvidia_gpus


def _ranked_indices(count: int) -> list[int]:
    indices = list(range(count))
    if sys.platform != "win32" or count < 2:
        return indices
    free_by_uuid = {gpu["uuid"].lower(): gpu["free_mib"] or 0 for gpu in nvidia_gpus()}
    if not free_by_uuid:
        return indices
    try:
        driver = ctypes.WinDLL("nvcuda.dll")
        get_uuid = getattr(driver, "cuDeviceGetUuid_v2", None) or driver.cuDeviceGetUuid
        if driver.cuInit(0) != 0:
            return indices
        free = {}
        for index in indices:
            device = ctypes.c_int()
            raw = (ctypes.c_ubyte * 16)()
            if driver.cuDeviceGet(ctypes.byref(device), index) == 0 and get_uuid(ctypes.byref(raw), device) == 0:
                free[index] = free_by_uuid.get(f"gpu-{UUID(bytes=bytes(raw))}", 0)
        return sorted(indices, key=lambda index: -free.get(index, 0))
    except (OSError, AttributeError):
        return indices


def _frame() -> dict:
    from frame_processing.classification import PlusLClassifier, preload_torch_on_windows

    preload_torch_on_windows()
    import paddle

    if not paddle.is_compiled_with_cuda():
        raise RuntimeError("PaddlePaddle GPU 빌드가 필요합니다.")
    failures = []
    for index in _ranked_indices(paddle.device.cuda.device_count()):
        try:
            classifier = PlusLClassifier(f"gpu:{index}")
            classifier.warm_up(640, 360)
            return dict(index=index, name=paddle.device.cuda.get_device_name(index),
                        cuda=paddle.version.cuda(), cudnn=paddle.device.get_cudnn_version())
        except Exception as exc:
            failures.append(f"gpu:{index} {exc}")
    raise RuntimeError(" · ".join(failures) or "CUDA 장치가 없습니다. NVIDIA 드라이버를 확인하세요.")


def _asr(model: str, gpu: bool) -> dict:
    import numpy as np
    from audio_processing.backends import FasterWhisperTranscriber

    if gpu:
        from audio_processing.runtime import prepare_cuda_libraries

        prepare_cuda_libraries(include_system=False)
        import ctranslate2

        indices = _ranked_indices(ctranslate2.get_cuda_device_count())
    else:
        indices = [0]
    duration = 2 if gpu else 5
    samples = np.zeros(duration * 16_000, dtype=np.float32)
    failures = []
    for index in indices:
        try:
            if gpu:
                supported = ctranslate2.get_supported_compute_types("cuda", device_index=index)
                compute = next((choice for choice in ("float16", "int8_float32", "float32")
                                if choice in supported), None)
                if compute is None:
                    raise RuntimeError(f"CTranslate2 지원 연산 형식 없음: {supported}")
            else:
                compute = "int8"
            worker = FasterWhisperTranscriber(
                model, device="cuda" if gpu else "cpu", device_index=index,
                compute_type=compute, local_files_only=True, project_cuda_only=True,
            )
            worker.prepare()
            started = monotonic()
            # Force the encoder and decoder even with a silent synthetic warm-up.
            segments, _ = worker._model.transcribe(
                samples, language="ko", beam_size=5, word_timestamps=True, vad_filter=False,
            )
            tuple(segments)
            elapsed = monotonic() - started
            if not gpu and elapsed >= duration:
                raise RuntimeError(f"CPU 사전 검사 {duration}초 입력에 {elapsed:.1f}초 소요")
            return dict(index=index, compute_type=compute, seconds=round(elapsed, 3))
        except Exception as exc:
            failures.append(f"{'cuda:' + str(index) if gpu else 'cpu'} {exc}")
    raise RuntimeError(" · ".join(failures) or "CTranslate2에서 CUDA 장치를 사용할 수 없습니다.")


def main() -> None:
    if sys.platform == "win32":
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("frame", "asr-gpu", "asr-cpu"))
    parser.add_argument("model", nargs="?")
    args = parser.parse_args()
    try:
        details = _frame() if args.kind == "frame" else _asr(args.model, gpu=args.kind == "asr-gpu")
    except Exception as exc:
        answer = {"ok": False, "reason": str(exc)}
    else:
        answer = {"ok": True, **details}
    print(PROBE_PREFIX + json.dumps(answer, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
