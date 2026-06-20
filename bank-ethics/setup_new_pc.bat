@echo off
setlocal

cd /d "%~dp0"

echo ================================================================
echo Setup bank-ethics on a new Windows PC
echo ================================================================
echo.

where python >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python is not installed or not in PATH.
  echo Install Python 3.11+ and run this file again.
  exit /b 1
)

where npm >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Node.js/npm is not installed or not in PATH.
  echo Install Node.js LTS and run this file again.
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [1/5] Creating virtual environment...
  python -m venv .venv
  if errorlevel 1 exit /b 1
) else (
  echo [1/5] Virtual environment already exists.
)

echo [2/5] Upgrading pip...
.\.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 exit /b 1

echo [3/5] Installing Python dependencies from requirements.txt...
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo [4/5] Installing frontend dependencies (npm install)...
npm install
if errorlevel 1 exit /b 1

echo [5/5] Setup complete.
echo.
echo Next:
echo   - Run start_no_training.bat  ^(if you already have a trained model^)
echo   - Run train_and_start.bat    ^(to generate/train first, then start app^)
echo.
exit /b 0
