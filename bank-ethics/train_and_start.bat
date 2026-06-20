@echo off
setlocal

cd /d "%~dp0"

echo ================================================================
echo Train model and start app
echo ================================================================
echo.

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Missing .venv. Run setup_new_pc.bat first.
  exit /b 1
)

if "%OPENAI_API_KEY%"=="" (
  echo [ERROR] OPENAI_API_KEY is not set.
  echo Set it in your shell or .env before running this script.
  exit /b 1
)

echo [1/8] Initialize DB...
.\.venv\Scripts\python.exe scripts\00_init_db.py
if errorlevel 1 goto error

echo [2/8] Generate transparency prompts...
.\.venv\Scripts\python.exe scripts\01_generate_prompts.py --family transparency --out data\prompts_transparency.csv
if errorlevel 1 goto error

echo [3/8] Simulate risky answers...
.\.venv\Scripts\python.exe scripts\02_simulate_bot.py --csv data\prompts_transparency.csv --limit 30 --mode risky --seed 42
if errorlevel 1 goto error

echo [4/8] Simulate compliant answers...
.\.venv\Scripts\python.exe scripts\02_simulate_bot.py --csv data\prompts_transparency.csv --limit 30 --mode compliant --seed 43
if errorlevel 1 goto error

echo [5/8] Judge labels (uses OpenAI API)...
.\.venv\Scripts\python.exe scripts\03_judge_labels.py --limit 60 --judge-version judge_all_v2
if errorlevel 1 goto error

echo [6/8] Export training dataset...
.\.venv\Scripts\python.exe scripts\05_export_training_data.py --out data\training_dataset.csv
if errorlevel 1 goto error

echo [7/8] Train model...
.\.venv\Scripts\python.exe scripts\06_train.py --csv data\training_dataset.csv
if errorlevel 1 goto error

echo [8/8] Start app...
call start_no_training.bat
exit /b 0

:error
echo.
echo [ERROR] Training pipeline failed. Check logs above.
exit /b 1
