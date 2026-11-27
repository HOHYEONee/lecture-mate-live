@echo off
setlocal
title CLUE
pushd "%~dp0"
if errorlevel 1 (
    echo Could not open the CLUE folder.
    pause
    exit /b 1
)

if not exist "scripts\setup-demo.ps1" goto :missing_files
if not exist "scripts\run-demo.ps1" goto :missing_files

set "CLUE_PYTHON=.venv312\Scripts\python.exe"
if not exist "%CLUE_PYTHON%" goto :setup
"%CLUE_PYTHON%" -c "import sys; from desktop_app.runtime import CPU_WHISPER, local_whisper_model; from rag.embeddings import default_embedding_is_ready; local_whisper_model(CPU_WHISPER); sys.exit(0 if default_embedding_is_ready() else 1)" >nul 2>&1
if "%errorlevel%"=="0" goto :launch

:setup
if not exist "%CLUE_PYTHON%" (
    py -3.12 -c "import struct, sys; sys.exit(0 if struct.calcsize('P') == 8 else 1)" >nul 2>&1
    if errorlevel 1 goto :missing_python
)
echo Preparing CLUE. The first setup downloads libraries and models.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\setup-demo.ps1"
if not "%errorlevel%"=="0" goto :failed

:launch
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\run-demo.ps1"
if not "%errorlevel%"=="0" goto :failed
popd
exit /b 0

:missing_files
echo Keep this launcher in the extracted CLUE folder, alongside the scripts folder.
goto :stop

:missing_python
echo Install Python 3.12 x64 with the Python Launcher, then double-click this file again.
goto :stop

:failed
echo CLUE could not set up or start. See the error above.

:stop
pause
popd
exit /b 1
