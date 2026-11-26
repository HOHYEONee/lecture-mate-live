param([switch]$Diagnose)

$ErrorActionPreference = 'Stop'
$clueRoot = Split-Path -Parent $PSScriptRoot
$cluePython = Join-Path $clueRoot '.venv312\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $cluePython -PathType Leaf)) {
    throw '먼저 .\scripts\setup-demo.ps1을 실행하세요.'
}
Push-Location -LiteralPath $clueRoot
try {
    $clueAppArgs = @()
    if ($Diagnose) { $clueAppArgs += '--diagnose' }
    & $cluePython -m desktop_app.main @clueAppArgs
    if ($LASTEXITCODE -ne 0) { throw "CLUE 실행 실패 (exit $LASTEXITCODE)." }
} finally {
    Pop-Location
}
