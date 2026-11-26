param(
    [switch]$CpuOnly,
    [switch]$SkipModels,
    [string]$PythonExecutable,
    [string]$PaddleWheel
)

$ErrorActionPreference = 'Stop'
$clueRoot = Split-Path -Parent $PSScriptRoot
if ($PaddleWheel) {
    if ($CpuOnly) { throw '-CpuOnly와 -PaddleWheel은 함께 사용할 수 없습니다.' }
    $PaddleWheel = (Resolve-Path -LiteralPath $PaddleWheel).ProviderPath
}
Push-Location -LiteralPath $clueRoot
try {
    $cluePython = Join-Path $clueRoot '.venv312\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $cluePython -PathType Leaf)) {
        if ($PythonExecutable) {
            & $PythonExecutable -m venv .venv312
        } else {
            & py -3.12 -m venv .venv312
        }
        if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 가상환경을 만들 수 없습니다.' }
    }

    function Invoke-CluePython {
        & $cluePython @args
        if ($LASTEXITCODE -ne 0) { throw "CLUE Python 명령 실패 (exit $LASTEXITCODE)." }
    }

    Invoke-CluePython -c "import sys,struct; assert sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8, 'Python 3.12 x64 required'"
    Invoke-CluePython -m pip install --upgrade pip
    Invoke-CluePython -m pip install 'torch==2.14.1+cpu' --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
    Invoke-CluePython -m pip install -r requirements-integration-cpu-win-py312.txt

    $clueGpu = [bool]$PaddleWheel
    if (-not $CpuOnly -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        $clueNvidiaList = & nvidia-smi --list-gpus 2>$null
        $clueGpu = $clueGpu -or ($LASTEXITCODE -eq 0 -and $clueNvidiaList -match '^GPU ')
    }
    if ($clueGpu) {
        try {
            $cluePaddle = $PaddleWheel
            if (-not $cluePaddle) {
                $clueWheelName = 'paddlepaddle_gpu-3.3.1-cp312-cp312-win_amd64.whl'
                $clueWheelCache = (New-Item -ItemType Directory -Force .cache/wheels).FullName
                $cluePaddle = Join-Path $clueWheelCache $clueWheelName
                if (-not (Test-Path -LiteralPath $cluePaddle -PathType Leaf)) {
                    $clueUrl = "https://paddle-whl.cdn.bcebos.com/stable/cu129/paddlepaddle-gpu/$clueWheelName"
                    Write-Host 'Paddle CUDA 12.9 휠을 다운로드합니다 (약 810 MB).'
                    & curl.exe --silent --show-error --fail --location --connect-timeout 15 `
                        --max-time 1800 --speed-time 45 --speed-limit 4096 --retry 2 `
                        --output "$cluePaddle.part" $clueUrl
                    if ($LASTEXITCODE -ne 0) { throw 'Paddle CUDA 휠 다운로드가 완료되지 않았습니다.' }
                    Move-Item -LiteralPath "$cluePaddle.part" -Destination $cluePaddle
                }
            }
            Invoke-CluePython -m pip install --timeout 25 --retries 2 -c constraints-win-py312.txt $cluePaddle
            Invoke-CluePython -m pip install -r requirements-integration-win-py312.txt
            Invoke-CluePython -c "from importlib.metadata import requires; assert 'nvidia-cudnn-cu12==9.9.0.52' in requires('paddlepaddle-gpu'), 'Paddle cu129 wheel required'"
        } catch {
            $clueGpu = $false
            Write-Warning "GPU 패키지를 준비하지 못했습니다. CPU 설치로 계속합니다. $_"
        }
    }
    Invoke-CluePython -m pip check

    if (-not $SkipModels) {
        $clueModelArgs = @()
        if ($clueGpu) { $clueModelArgs += '--gpu' }
        Invoke-CluePython -m scripts.prepare_demo_models @clueModelArgs
    }
    Write-Host '설치 완료. .\scripts\run-demo.ps1 로 실행하세요.'
} finally {
    Pop-Location
}
