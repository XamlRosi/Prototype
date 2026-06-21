# Setup and Run on a New Windows PC

This guide gives exact steps for both modes:
- install/run **without .bat files** (manual)
- install/run **with .bat files** (automated)

It also covers:
- run **without training**
- run **with training**

---

## 0) What to copy to the new PC

### Minimum copy for app run (no training)

Copy these:
- whole `src/` folder
- whole `scripts/` folder
- whole `data/models/` folder (at least one trained model)
- `server.py`
- `requirements.txt`
- `package.json`
- `package-lock.json` (if present)
- `vite.config.js`
- `.env` (optional)

### Additional copy required for training

Also copy these:
- whole `prompts/` folder (**required**)
- whole `data/` folder (DB/CSVs used by pipeline)

Why: training scripts `02_simulate_bot.py` and `03_judge_labels.py` read system/judge prompt templates from `prompts/`.

---

## 1) Prerequisites

Install these first:
- Python 3.11+
- Node.js LTS (includes npm)
- Git (optional, recommended)

Open PowerShell in the project folder:

```powershell
cd C:\path\to\bank-ethics
```

---

## 2) Manual Install (without .bat)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
npm install
```

If PowerShell blocks script execution:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

---

## 3) Automated Install (with .bat)

Run:

```bat
setup_new_pc.bat
```

This script:
- creates `.venv` (if missing)
- installs Python dependencies from `requirements.txt`
- runs `npm install`

---

## 4) Run WITHOUT training

Use this mode if you already have a trained model in `data/models/`.

`prompts/` folder is **not required** for this mode.

### 4.1 Manual run

Terminal 1 (backend):

```powershell
.\.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload
```

Terminal 2 (frontend):

```powershell
npm run dev
```

Open:
- UI: http://localhost:5173
- API health: http://127.0.0.1:8000/health

### 4.2 With .bat

Run:

```bat
start_no_training.bat
```

This starts backend and frontend in separate terminal windows.

---

## 5) Run WITH training

Use this when you want to generate data and train a model on the new machine.

`prompts/` folder is **required** for this mode.

### Important

Set `OPENAI_API_KEY` first (because simulation/judging scripts call OpenAI):

```powershell
$env:OPENAI_API_KEY="your_key_here"
```

(or put it in `.env`)

### 5.1 Manual training pipeline

```powershell
.\.venv\Scripts\python.exe scripts\00_init_db.py
.\.venv\Scripts\python.exe scripts\01_generate_prompts.py --family transparency --out data\prompts_transparency.csv
.\.venv\Scripts\python.exe scripts\02_simulate_bot.py --csv data\prompts_transparency.csv --limit 30 --mode risky --seed 42
.\.venv\Scripts\python.exe scripts\02_simulate_bot.py --csv data\prompts_transparency.csv --limit 30 --mode compliant --seed 43
.\.venv\Scripts\python.exe scripts\03_judge_labels.py --limit 60 --judge-version judge_all_v2
.\.venv\Scripts\python.exe scripts\05_export_training_data.py --out data\training_dataset.csv
.\.venv\Scripts\python.exe scripts\06_train.py --csv data\training_dataset.csv
```

Then run app (Section 4).

### 5.2 With .bat

Run:

```bat
train_and_start.bat
```

This script runs the same training flow and then starts the app.

---

## 6) Verify everything quickly

1. API health returns `ok: true` and `model_loaded: true`:

```powershell
curl http://127.0.0.1:8000/health
```

2. In UI:
- add prompt + answer
- click `Оцени отговора`
- click `Explain with LIME`

---

## 7) Typical issues

### LIME error
Install/fix dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### `model_loaded: false`
- copy/select a model in `data/models/`
- check active model in UI (`/api/models`)

### HTTP 500 in UI
- check backend terminal logs
- check browser Network response `detail`

---

## 8) Files added for new-machine setup

- `setup_new_pc.bat`
- `start_no_training.bat`
- `train_and_start.bat`
- `SETUP_NEW_PC.md`
