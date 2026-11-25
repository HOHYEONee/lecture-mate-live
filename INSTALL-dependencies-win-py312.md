# Windows integration dependency setup (candidate configuration)

This guide describes the proposed dependency set for Windows x64 and Python 3.12.
The versions in `constraints-win-py312.txt` are candidate pins. They have not been
installed or tested together, and this file is not a verified lock.

## Install order

Use a clean virtual environment. Install the framework wheels from their vendor
indexes first, then resolve the project requirements with the shared constraints.
The Paddle and PyTorch wheels are deliberately installed separately because they
come from different CUDA-specific indexes. The audio branch's CUDA 12.9 NVIDIA
runtime wheel pins are kept in `requirements-audio-cuda-runtime-candidate-win-py312.txt`
and are intentionally excluded from the aggregate: PyTorch's CUDA wheel also
depends on versioned `nvidia-*` runtime wheels, and the two exact sets have not
been reconciled. First test whether the libraries supplied with the chosen
PyTorch wheel satisfy CTranslate2's Windows runtime needs.

```powershell
py -3.12 -m venv .venv-integration
$python = '.\.venv-integration\Scripts\python.exe'

& $python -m pip install --upgrade pip

# Candidate PyTorch build: Windows x64, CUDA 12.6.
& $python -m pip install 'torch==2.14.1' --index-url https://download.pytorch.org/whl/cu126

# Candidate Paddle GPU build: CUDA 12.6 wheel from Paddle's own index.
& $python -m pip install 'paddlepaddle-gpu==3.3.1' --index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/

# Resolve the rest of the proposed runtime set.
& $python -m pip install -r requirements-integration-win-py312.txt
```

Do not install a second OpenCV wheel flavor such as `opencv-python` alongside
`opencv-contrib-python`; OpenCV distributions share the `cv2` namespace. If a
transitive Paddle dependency requires another OpenCV distribution by package name,
stop and resolve that dependency before using the environment.

## Dependency resolver check

The following is a non-installing resolver preview to run before the actual
aggregate installation. It still requires access to package indexes. The two GPU
framework wheels should already be present in the environment so pip can resolve
their pinned requirements without choosing a wheel from PyPI.

```powershell
& $python -m pip install --dry-run --no-cache-dir -r requirements-integration-win-py312.txt
```

After installation, check metadata and importability:

```powershell
& $python -m pip check
& $python -c "import numpy, cv2, PySide6, torch, paddle, ctranslate2, faster_whisper, av; print('numpy', numpy.__version__); print('opencv', cv2.__version__); print('PySide6', PySide6.__version__); print('torch', torch.__version__, torch.version.cuda); print('paddle', paddle.__version__, paddle.is_compiled_with_cuda()); print('CTranslate2', ctranslate2.__version__); print('Faster-Whisper', faster_whisper.__version__); print('PyAV', av.__version__)"
& $python -c "import torch, paddle, ctranslate2; print('torch CUDA', torch.cuda.is_available()); print('Paddle CUDA devices', paddle.device.cuda.device_count() if paddle.is_compiled_with_cuda() else 0); print('CTranslate2 CUDA devices', ctranslate2.get_cuda_device_count())"
nvidia-smi
```

## Runtime checks still required

1. Initialize and run the Paddle layout classifier by itself.
2. Initialize CTranslate2 and transcribe a short local audio sample with
   Faster-Whisper. Confirm the installed CUDA and cuDNN DLLs are the intended ones.
3. Load the Sentence Transformers embedder and Ko-Reranker. Start with the
   reranker on CPU, then separately measure its GPU path.
4. In one process, initialize Paddle, PyTorch, and CTranslate2 in the application's
   actual startup order. Check for DLL load errors and compare `nvidia-smi` before
   and after model loading and inference.
5. Confirm that one OpenCV distribution is installed and that PySide6 starts
   without Qt plugin conflicts.
6. Confirm WASAPI loopback device discovery with PyAudioWPatch.
7. Run the application's integration tests only after the candidate environment
   passes the package, import, GPU, and model-load checks.

## Candidate choices and unresolved compatibility

- `numpy==2.2.6` is the conservative common candidate. The audio branch currently
  pins `2.5.4` for Python 3.12; choose between them only after resolving the exact
  PaddleX 3.3.13 dependency metadata and testing the binary wheels.
- `opencv-contrib-python==4.10.0.84` is the one intended OpenCV distribution.
  Confirm that PaddleOCR/PaddleX do not force an incompatible second distribution.
- PyTorch 2.14.1 cu126 and PaddlePaddle GPU 3.3.1 cu126 align on the CUDA minor
  family, but their combined Windows DLL loading and VRAM use are not verified.
- The audio requirements pin CUDA 12.9/cuDNN 9.28 NVIDIA runtime wheels. This is
  distinct from the Paddle CUDA 12.6 wheel family. They are isolated in the
  optional candidate file because PyTorch may pin different versions of the same
  `nvidia-*` distributions; do not install them into the aggregate environment
  until the exact PyTorch wheel metadata and CTranslate2 DLL needs are checked.
- CTranslate2 4.9.1 is a candidate pin. CTranslate2 and Faster-Whisper upstream
  installation notes differ on the cuDNN major version required by recent builds;
  verify the actual Windows wheel before treating the pin as final.
- Sentence Transformers 6.2.0 requires `huggingface-hub>=1.3.0,<3.0.0`; the Hub
  pin is therefore `1.3.0`, not the earlier incompatible `1.0.1` candidate.
  Transformers 5.0.0 satisfies Sentence Transformers' `>=5.0.0,<6.0.0` range.
  Re-resolve and lock the transitive dependencies after the GPU wheel smoke tests.
- `requirements-dev-win-py312.txt` is separate from the runtime aggregate and is
  for the frame module's declared scikit-image test/benchmark dependency.
- The frame model weights are not Python packages and remain a separate setup item.
