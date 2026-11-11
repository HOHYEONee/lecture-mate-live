import os
from pathlib import Path
import sys

_DIRECTORY_HANDLES = []
_ADDED = set()


# [내부 / 외부(os.add_dll_directory)] Windows의 NVIDIA 라이브러리 경로를 현재 프로세스에 등록한다.
def prepare_cuda_libraries():
    if sys.platform != "win32":
        return
    folders = []
    for entry in sys.path:
        root = Path(entry) / "nvidia"
        if root.is_dir():
            folders.extend(p.parent for p in root.glob("*/bin/cublas64_12.dll"))
            folders.extend(p.parent for p in root.glob("*/bin/cudnn64_9.dll"))
            folders.extend(p.parent for p in root.glob("*/bin/nvrtc64*.dll"))
    toolkit = os.environ.get("CUDA_PATH")
    if toolkit:
        folders.append(Path(toolkit)/"bin")
    for folder in dict.fromkeys(folders):
        path = str(folder.resolve())
        if path in _ADDED or not folder.is_dir():
            continue
        _DIRECTORY_HANDLES.append(os.add_dll_directory(path))
        os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
        _ADDED.add(path)
