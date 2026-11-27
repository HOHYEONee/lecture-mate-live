"""Prepare public model files before the demo starts."""

from __future__ import annotations

import argparse
import os
import sys

from desktop_app.runtime import CPU_WHISPER, GPU_WHISPER, MODEL_ROOT, has_model_file, local_whisper_model
from frame_processing.classification import DEFAULT_MODEL_DIR, MODEL_FILES
from rag.embeddings import DEFAULT_EMBEDDING_REVISION, SentenceTransformerEmbedder, default_embedding_is_ready


WHISPER_CHECKPOINTS = {
    CPU_WHISPER: ("Systran/faster-whisper-base", "ebe41f70d5b6dfa9166e2c581c45c9c0cfc57b66"),
    GPU_WHISPER: ("mobiuslabsgmbh/faster-whisper-large-v3-turbo", "0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf"),
}


def prepare_models(*, gpu: bool) -> None:
    os.environ.setdefault("HF_HOME", str(MODEL_ROOT.parent / ".cache" / "huggingface"))
    from huggingface_hub import snapshot_download

    names = (CPU_WHISPER, GPU_WHISPER) if gpu else (CPU_WHISPER,)
    for name in names:
        try:
            local_whisper_model(name)
        except FileNotFoundError:
            pass
        else:
            print(f"{name} 로컬 가중치 재사용")
            continue
        repo, revision = WHISPER_CHECKPOINTS[name]
        print(f"{name} 가중치를 준비합니다.", flush=True)
        snapshot_download(repo, revision=revision, local_dir=MODEL_ROOT / name, allow_patterns=[
            "config.json", "model.bin", "tokenizer.json", "vocabulary.*", "preprocessor_config.json",
        ])
        local_whisper_model(name)
    if gpu:
        if all(has_model_file(DEFAULT_MODEL_DIR / name) for name in MODEL_FILES):
            print("PP-DocLayout_plus-L 로컬 가중치 재사용")
        else:
            print("PP-DocLayout_plus-L 가중치를 준비합니다.", flush=True)
            snapshot_download("PaddlePaddle/PP-DocLayout_plus-L",
                              local_dir=DEFAULT_MODEL_DIR, allow_patterns=list(MODEL_FILES))
            missing = [name for name in MODEL_FILES if not has_model_file(DEFAULT_MODEL_DIR / name)]
            if missing:
                raise FileNotFoundError(f"PP-DocLayout_plus-L 모델 파일 누락 · {', '.join(missing)}")
    if default_embedding_is_ready():
        print("MiniLM 로컬 가중치 재사용")
    else:
        print("MiniLM 가중치를 준비합니다.", flush=True)
        SentenceTransformerEmbedder(revision=DEFAULT_EMBEDDING_REVISION).prepare()
        if not default_embedding_is_ready():
            raise FileNotFoundError("MiniLM 로컬 가중치가 완전하지 않습니다.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true", help="also prepare the GPU classifiers and ASR model")
    args = parser.parse_args()
    try:
        prepare_models(gpu=args.gpu)
    except Exception as exc:
        print(f"모델 준비 실패 ({type(exc).__name__}): {exc}", file=sys.stderr)
        print("연결 및 디스크 여유를 확인한 뒤 같은 준비 명령을 다시 실행하세요.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
