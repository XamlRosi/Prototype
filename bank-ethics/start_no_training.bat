@echo off
setlocal

cd /d "%~dp0"

echo ================================================================
echo Start app WITHOUT training
echo ================================================================
echo.

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Missing .venv. Run setup_new_pc.bat first.
  exit /b 1
)

if not exist "node_modules" (
  echo [ERROR] Missing node_modules. Run setup_new_pc.bat first.
  exit /b 1
)

if not exist "data\models\tfidf_ovr_logreg_7labels.joblib" (
  echo [WARNING] Default model not found: data\models\tfidf_ovr_logreg_7labels.joblib
  echo The UI will start, but predictions may fail until you select/copy a model.
  echo.
)

echo Starting backend at http://127.0.0.1:8000 ...
start "bank-ethics-backend" cmd /k "cd /d %CD% && .venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload"

echo Starting frontend at http://localhost:5173 ...
start "bank-ethics-frontend" cmd /k "cd /d %CD% && npm run dev"

echo.
echo Open: http://localhost:5173
echo Health: http://127.0.0.1:8000/health
echo.
exit /b 0
