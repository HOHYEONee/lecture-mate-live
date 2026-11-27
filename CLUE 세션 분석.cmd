@echo off
chcp 65001 >nul
setlocal
set PYTHONUTF8=1
cd /d "%~dp0"
if exist ".venv312\Scripts\python.exe" (
  ".venv312\Scripts\python.exe" -m scripts.analyze_session %*
  goto finish
) else if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m scripts.analyze_session %*
  goto finish
)
python -c "import sys; raise SystemExit(sys.version_info < (3, 10))" >nul 2>&1
if not errorlevel 1 (
  python -m scripts.analyze_session %*
  goto finish
)
py -3 -c "import sys; raise SystemExit(sys.version_info < (3, 10))" >nul 2>&1
if not errorlevel 1 (
  py -3 -m scripts.analyze_session %*
  goto finish
)
echo Python 3.10 이상을 찾을 수 없습니다. Python을 설치한 뒤 다시 실행해 주세요.
:finish
echo.
pause
