# Bank Ethics - Guide

## About This Project

This is a **multi-label classification system for detecting ethical issues in banking conversations**. The system uses AI to identify problems like:

- **Safety issues** - unsafe or harmful recommendations
- **Privacy violations** - improper handling of personal data
- **Bias & discrimination** - unfair treatment based on demographics
- **Manipulation** - pressure tactics or misleading information
- **Transparency issues** - lack of clear disclosure
- **Policy violations** - non-compliance with regulations
- **Financial risk** - recommendations that could harm customers
- **Missing escalation** - cases that should involve humans

### Three Models Available:
- **M1 (TF-IDF + Logistic Regression)** - Fast, lightweight, good accuracy
- **M2 (mBERT)** - Multilingual transformer, better accuracy
- **M3 (XLM-RoBERTa)** - Multilingual transformer, best accuracy (but slower)

### Three Usage Scenarios:
1. **RUN ONLY** - Use pre-trained models to evaluate conversations
2. **RETRAIN ONLY** - Improve models with new data
3. **FULL PIPELINE** - Generate prompts, simulate bot, label, export, train from scratch

---

## Project Structure

- `data/models/` - generated models, evaluation reports, metrics, and saved artifacts
- `scripts/` - database setup, prompt generation, bot simulation, judge labeling, export, and training scripts
- `src/` - application source code, including the backend package and the frontend interface
- `prompts/` - system prompts, judge prompts, and prompt templates used during generation and labeling
- `data/` - exported datasets, reports, and other pipeline outputs

---

## System Requirements

**Before you start, install:**
- **Python 3.12+** - [Download](https://www.python.org/downloads/)
- **Node.js 18+ (LTS)** - [Download](https://nodejs.org/)

Verify installation:
```powershell
python --version
node --version
npm --version
```

---

## Setup Prerequisites

### First time on a new PC
If you already installed **Python 3.12+** and **Node.js 18+**, you can use the automated setup script:

```powershell
.\setup_new_pc.bat
```

This will create the virtual environment and install the Python and Node dependencies.

```powershell
# Create virtual environment
python -m venv .venv

# Activate it
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

---

## Scenario 1: RUN ONLY (no training)

> Предпоставка: моделите вече са тренирани в `data/models/`

### Quick Start (recommended):
```powershell
.\start_no_training.bat
```
This batch file will automatically install dependencies and start both backend and frontend.

### Manual Start:

#### Install dependencies:
```powershell
pip install -r requirements.txt
npm install
```

#### Start Backend:
```powershell
python server.py
```

Backend ще слуша на `http://127.0.0.1:8000`

#### Start Frontend (new terminal):
```powershell
npm run dev
```

Frontend ще слуша на `http://0.0.0.0:5173` (достъпно на `localhost:5173`)

#### Open Browser:
```
http://localhost:5173
```

---

## Scenario 2: RETRAIN ONLY (existing data)

> Предпоставка: имаш готови prompts и labels в `data/training_dataset.csv` (или `data/bank_ethics.db`)

### Export Dataset (if only have DB):
```powershell
python scripts\05_export_training_data.py --out data\training_dataset.csv
```

### Retrain Models:

**M1 (TF-IDF + Logistic Regression):**
```powershell
python scripts\06_train_m1.py --csv data\training_dataset.csv
```

**M2 (mBERT - multilingual BERT):**
```powershell
python scripts\06_train-m23.py ^
  --csv data\training_dataset.csv ^
  --model-id M2 ^
  --model-name bert-base-multilingual-cased ^
  --outdir data\models\m2_mbert
```

**M3 (XLM-RoBERTa):**
```powershell
python scripts\06_train-m23.py ^
  --csv data\training_dataset.csv ^
  --model-id M3 ^
  --model-name xlm-roberta-base ^
  --outdir data\models\m3_xlm_roberta
```

---

## Scenario 3: FULL PIPELINE (from scratch)

> Генериране на prompts → Bot симулация → Judge labeling → Metrics → Export → Train models

### Step 1: Initialize Database
```powershell
python scripts\00_init_db.py
```

### Step 2: Generate Prompts
```powershell
python scripts\01_generate_prompts.py
```

### Step 3: Simulate Bot Responses
```powershell
python scripts\02_simulate_bot.py
```

### Step 4: Judge & Label (AI labeling)
```powershell
python scripts\03_judge_labels.py --judge-version judge_v2
```

### Step 5: Export Training Dataset
```powershell
python scripts\05_export_training_data.py --out data\training_dataset.csv
```

### Step 6: Train Models

**M1 (TF-IDF + Logistic Regression):**
```powershell
python scripts\06_train_m1.py --csv data\training_dataset.csv
```

**M2 (mBERT - multilingual BERT):**
```powershell
python scripts\06_train-m23.py ^
  --csv data\training_dataset.csv ^
  --model-id M2 ^
  --model-name bert-base-multilingual-cased ^
  --outdir data\models\m2_mbert
```

**M3 (XLM-RoBERTa):**
```powershell
python scripts\06_train-m23.py ^
  --csv data\training_dataset.csv ^
  --model-id M3 ^
  --model-name xlm-roberta-base ^
  --outdir data\models\m3_xlm_roberta
```

---

## Scenario 2: RETRAIN ONLY (existing data)

> Предпоставка: имаш готови prompts и labels в `data/training_dataset.csv` (или `data/bank_ethics.db`)

### Export Dataset (if only have DB):
```powershell
python scripts\05_export_training_data.py --out data\training_dataset.csv
```

### Retrain Models:

**M1:**
```powershell
python scripts\06_train_m1.py --csv data\training_dataset.csv
```

**M2:**
```powershell
python scripts\06_train-m23.py ^
  --csv data\training_dataset.csv ^
  --model-id M2 ^
  --model-name bert-base-multilingual-cased ^
  --outdir data\models\m2_mbert
```

**M3:**
```powershell
python scripts\06_train-m23.py ^
  --csv data\training_dataset.csv ^
  --model-id M3 ^
  --model-name xlm-roberta-base ^
  --outdir data\models\m3_xlm_roberta
```

---

## Scenario 3: RUN ONLY (no training)

> Предпоставка: моделите вече са тренирани в `data/models/`

### Install dependencies:
```powershell
pip install -r requirements.txt
npm install
```

### Start Backend:
```powershell
python server.py
```

Backend ще слуша на `http://127.0.0.1:8000`

### Start Frontend (new terminal):
```powershell
npm run dev
```

Frontend ще слуша на `http://0.0.0.0:5173` (достъпно на `localhost:5173`)

### Open Browser:
```
http://localhost:5173
```

---

## Model Comparison

| Model | File | Type | Speed | Accuracy | Dependencies |
|-------|------|------|-------|----------|---|
| **M1** | `06_train_m1.py` | sklearn (TF-IDF + LogReg) | Fast | Good | numpy, scikit-learn |
| **M2** | `06_train-m23.py` | mBERT (multilingual) | Slow | Better | torch, transformers |
| **M3** | `06_train-m23.py` | XLM-RoBERTa | Slow | Best | torch, transformers |

---

## Files Needed for Each Scenario

### Full Pipeline
- `scripts/00_init_db.py` through `06_train*.py`
- `data/` (for output)
- `prompts/` (judge prompts)
- `requirements.txt`, `src/`

### Retrain Only
- `scripts/05_export_training_data.py`
- `scripts/06_train.py`, `scripts/06_train-m23.py`
- `data/training_dataset.csv` OR `data/bank_ethics.db`
- `requirements.txt`, `src/`

### Run Only
- `server.py`, `index.html`, `vite.config.js`
- `package.json`, `src/` (frontend)
- `data/models/` (at least one trained model)
- `requirements.txt`, `src/bank_ethics/` (backend)

---

## Troubleshooting

**Models not loading?**
- Check `data/models/` folder exists
- Check model file exists: `data/models/m1_tfidf_logreg/m1_tfidf_logreg.joblib`

**Training takes too long?**
- Use M1 (fastest)
- M2/M3 need GPU or will be very slow on CPU

**Database not found?**
- Run `00_init_db.py` first
- Or copy existing `data/bank_ethics.db` from another machine
