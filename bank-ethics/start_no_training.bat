@echo off
setlocal

cd /d "%~dp0"

title bank-ethics launcher (no training)

echo ================================================================
echo Start app WITHOUT training
echo ================================================================
echo.
echo Project dir: %CD%
echo.

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Missing .venv. Run setup_new_pc.bat first.
  pause
  exit /b 1
)

if not exist "node_modules" (
  echo [ERROR] Missing node_modules. Run setup_new_pc.bat first.
  pause
  exit /b 1
)

if not exist "package.json" (
  echo [ERROR] Missing package.json in project root. Frontend cannot start.
  pause
  exit /b 1
)

if not exist "index.html" (
  echo [ERROR] Missing index.html in project root. Vite needs this entry file.
  echo Copy index.html from the original project and try again.
  pause
  exit /b 1
)

where npm >nul 2>nul
if errorlevel 1 (
  echo [ERROR] npm is not available in PATH. Install Node.js LTS.
  pause
  exit /b 1
)

if not exist "data\models\m1_tfidf_logreg\m1_tfidf_logreg.joblib" (
  echo [WARNING] Default model not found: data\models\m1_tfidf_logreg\m1_tfidf_logreg.joblib
  echo The UI will start, but predictions may fail until you select/copy a model.
  echo.
)

echo Starting backend at http://127.0.0.1:8000 ...
start "bank-ethics-backend" cmd /k "cd /d %CD% && .venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload"

echo Starting frontend at http://localhost:5173 ...
start "bank-ethics-frontend" cmd /k "cd /d %CD% && npm run dev"

echo Waiting a few seconds and checking backend health...
timeout /t 4 /nobreak >nul

powershell -NoProfile -Command "try { $r = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 5; if($r.StatusCode -eq 200){ Write-Host '[OK] Backend health endpoint responded (200).' } else { Write-Host '[WARN] Backend returned status' $r.StatusCode } } catch { Write-Host '[WARN] Backend is not ready yet. Check the backend window for errors.' }"

echo.
echo Open: http://localhost:5173
echo Health: http://127.0.0.1:8000/health
echo.
echo If UI does not open, check the two new terminal windows: bank-ethics-backend and bank-ethics-frontend.
echo.
pause
exit /b 0
