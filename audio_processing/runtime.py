"""Make optional project-local NVIDIA runtime wheels visible on Windows."""

import os
from pathlib import Path
import sys
import sysconfig

_DIRECTORY_HANDLES = []
_ADDED = set()


# [내부 / 외부(os.add_dll_directory)] Windows의 NVIDIA 라이브러리 경로를 현재 프로세스에 등록한다.
def prepare_cuda_libraries(*, include_system=True):
    if sys.platform != "win32":
        return
    folders = []
    required = {"cublas64_12.dll", "cudnn64_9.dll"}
    found = set()
    paths = sys.path if include_system else {sysconfig.get_path(key) for key in ("purelib", "platlib")}
    for entry in paths:
        root = Path(entry) / "nvidia"
        if root.is_dir():
            for name in sorted(required):
                for library in root.glob(f"*/bin/{name}"):
                    folders.append(library.parent)
                    found.add(name)
            folders.extend(p.parent for p in root.glob("*/bin/nvrtc64*.dll"))
    if not include_system and required - found:
        missing = ", ".join(sorted(required - found))
        raise FileNotFoundError(f"프로젝트 NVIDIA 런타임 DLL 누락: {missing}")
    toolkit = os.environ.get("CUDA_PATH") if include_system else None
    if toolkit:
        folders.append(Path(toolkit)/"bin")
    for folder in dict.fromkeys(folders):
        path = str(folder.resolve())
        if path in _ADDED or not folder.is_dir():
            continue
        _DIRECTORY_HANDLES.append(os.add_dll_directory(path))
        os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
        _ADDED.add(path)
