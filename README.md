# ClinicalAI — Hospital Readmission Risk & Discharge Decision Support

> **Research Demonstration Notice**: This application is a research and hackathon demonstration for predicting 30-day readmission risk in diabetic inpatients using historical, de-identified data from the UCI Diabetes 130-US Hospitals dataset (1999–2008). It is **not** a cleared Medical Device or validated clinical decision support tool and must not be used for individual clinical diagnostic or treatment decisions.

---

## 🏥 Product Overview & Core Modules

**ClinicalAI** is a full-stack clinical decision support application built to reduce avoidable 30-day hospital readmissions for diabetic inpatient encounters. It connects machine learning risk stratification with post-discharge transition planning and governance across **8 complete end-to-end modules**:

1. **Clinical Overview Dashboard (`/`)**: Real-time cohort volume, readmission rate, high-risk count, risk-tier distribution (Low, Elevated, High), model status, and live facility/risk-tier filters.
2. **Discharge-Planning Worklist (`/worklist`)**: Search, filter, sort, and paginate inpatient encounters. Includes risk tier badges, calibrated probability, priority indicators, and an interactive side-drawer showing feature attributions and immediate care transitions.
3. **Explainable Risk Calculator (`/calculator`)**: Bedside risk simulator allowing clinicians to adjust modifiable clinical inputs (length of stay, insulin dosage, medication counts, glucose/HbA1c tests) to compute live probabilities with model version provenance and SHAP-based feature attributions.
4. **Follow-Up and Care-Plan Tracker (`/careplans`)**: Dedicated transitional care coordinator view tracking multidisciplinary tasks (medication reconciliation, follow-up scheduling within 7 days, telehealth outreach in 48–72h, diabetes education). Supports real-time task toggles, assignee management, and completion timestamps persisted in the database.
5. **Model Comparison and Evaluation (`/models`)**: Leakage-free benchmark comparing **Logistic Regression**, **Random Forest**, and **Calibrated XGBoost (Isotonic)** on the same patient-grouped holdout partition. Displays ROC-AUC, PR-AUC, Recall/Sensitivity, Precision, Accuracy, Brier score, calibration analysis, and confusion matrices.
6. **Fairness and Governance Dashboard (`/fairness`)**: Subgroup audit across protected attributes (**Age Group**, **Gender**, and **Race**). Reports TPR, FPR, selection rate, demographic parity differences, and evaluates Fairlearn's `ThresholdOptimizer` mitigation benchmark.
7. **Data Ingestion & Quality Audit (`/datasets`)**: Backend-enforced CSV upload validation pipeline. Inspects required clinical schema fields, flags statistical outliers (>30-day stay, >80 medications), audits missingness, presents a human-readable quality report, and requires explicit user confirmation before database ingestion.
8. **AI Model Card & System Provenance (`/provenance`)**: Standardized AI Model Card detailing intended use boundaries, target population, training cohort description, leakage safeguards, and known limitations. Includes an immutable compliance audit trail logging administrative events.

---

## 🏗️ Architecture & Technology Stack

```mermaid
graph TD
    Client["React 18 + Vite Frontend<br/>(Light Clinical Healthcare Theme, Recharts, Lucide)"]
    FastAPI["FastAPI REST Backend<br/>(Async REST API, Pydantic v2)"]
    Inference["Inference Engine<br/>(Calibrated XGBoost, SHAP TreeExplainer)"]
    DB[("PostgreSQL / Supabase<br/>(Local SQLite data/demo.db fallback)")]

    Client -->|HTTP / JSON via /api| FastAPI
    FastAPI --> Inference
    FastAPI --> DB
```

- **Frontend**: React 18, Vite, Vanilla CSS design system (clinical light theme: white cards, soft slate-50 canvas `#f8fafc`, slate-900 navy typography, and dark navy sidebar `#0f172a`), Recharts data visualization, Lucide React icons, React `ErrorBoundary`.
- **Backend**: Python 3.10+, FastAPI, Pydantic v2, SQLAlchemy 2.0 (Async), `asyncpg`, `aiosqlite`.
- **Machine Learning**: Scikit-Learn 1.9+, XGBoost 3.4, Imbalanced-Learn (SMOTE), SHAP (TreeExplainer), Fairlearn (subgroup metrics & ThresholdOptimizer).
- **Database**: PostgreSQL (Supabase compatible) with automatic SQLite local fallback (`data/demo.db`).

---

## 🧠 Machine Learning Pipeline

### 1. Dataset & Preprocessing
Derived from the **UCI Diabetes 130-US Hospitals dataset** (101,766 inpatient encounters, 1999–2008):
- **Cohort Eligibility**: Excluded terminal discharges (`discharge_disposition_id` in `{11, 13, 14, 19, 20, 21}` corresponding to expired patients and hospice transfers).
- **Patient Leakage Prevention**: Split using `GroupShuffleSplit` on `patient_nbr` (80/20 train/test split). Encounters from the same patient never appear in both train and test partitions.
- **Missing Value Handling**: Missing indicators (`'?'`, `'Unknown/Invalid'`) preserved as sentinel categories for categoricals; median imputation for numerical features.
- **Clinical ICD-9 Categorization**: Grouped diagnosis codes (`diag_1`, `diag_2`, `diag_3`) into 17 high-level clinical categories (Circulatory, Respiratory, Endocrine/Metabolic, Genitourinary, Digestive, Neoplasms, etc.).
- **Outlier Capping**: IQR-based capping fit exclusively on training data for continuous clinical variables (`time_in_hospital`, `num_lab_procedures`, `num_medications`, `number_diagnoses`).
- **Class Imbalance**: SMOTE applied strictly to the training fold after splitting. The test set preserves the natural 9.0% readmission prevalence.

### 2. Model Comparison

Three model families evaluated on the identical holdout split:

| Model | ROC-AUC | Recall | Precision | Accuracy | Brier Score | Clinical Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression** | 0.620 | 0.952 | 0.096 | 0.187 | 0.2319 | Linear baseline, sensitive screening |
| **Random Forest** | 0.608 | 0.910 | 0.100 | 0.250 | 0.1759 | Non-linear tree ensemble |
| **XGBoost (Calibrated)** | **0.635** | 0.057 | **0.333** | **0.905** | **0.0806** | **Selected**: Gradient boosting with isotonic calibration |

### 3. Calibration & Threshold Rationale
- **Calibration**: XGBoost probabilities are calibrated using `CalibratedClassifierCV` with isotonic regression, producing well-calibrated posterior probabilities with a minimized Brier score (0.0806).
- **Threshold Selection**: An operational decision threshold of **0.15 (15%)** fits a screening workflow (base readmission prevalence ~9%).
- **Risk Tiers**:
  - **Low Risk**: `< 12%` (at or below general population base rate)
  - **Elevated Risk**: `12% – 20%` (1.5× – 2.5× average baseline risk)
  - **High Risk**: `≥ 20%` (> 2.5× average risk; qualifies for intensive care management)

---

## 🔍 Explainability & Decision Support

- **SHAP Feature Attributions**: Deterministic attributions computed via `shap.TreeExplainer` on the underlying tree ensemble, ranking local factors that push risk upward or downward.
- **Modifiable vs. Fixed Taxonomy**:
  - *Modifiable Factors* (intervention targets): medication counts, regimen changes, glucose monitoring, HbA1c testing, discharge destination.
  - *Fixed Historical Factors* (risk context): age, prior emergency/inpatient visits, primary admission diagnosis.
- **Deterministic Action Recommendations**: Generates tailored, non-prescriptive transitional care suggestions:
  - Medication Reconciliation before discharge
  - Early Follow-up Appointment (within 7 days)
  - Telehealth Check-in (48–72 hours post-discharge)
  - Diabetes Self-Management Education (DSME) referral
  - Pharmacy Consult for active regimen changes

---

## 🚀 Quick Start (Local Run)

### Prerequisites
- Python 3.10+ (tested with Python 3.13)
- Node.js 18+ and npm

### 1. Setup Python Environment & Dependencies
```bash
# Install Python dependencies
pip install -r requirements.txt

# Run ML pipeline, train models, and seed demo database
python src/train.py
```

### 2. Run Automated Test Suite
```bash
pytest -v
```
*Expected: 17/17 tests passing across API endpoints, data validation, preprocessing, explainability, and care plans.*

### 3. Launch Development Servers

**Terminal 1 — FastAPI Backend:**
```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
```
API Documentation available at: `http://127.0.0.1:8000/api/docs`

**Terminal 2 — React/Vite Frontend:**
```bash
cd frontend
npm install
npm run dev
```
Access the application in your browser at: **`http://127.0.0.1:5173`**

---

## 🗄️ Database & PostgreSQL Migrations

The application runs out-of-the-box using the SQLite database (`data/demo.db`). For PostgreSQL / Supabase deployment:

1. Create a Supabase or PostgreSQL database.
2. Run the migration scripts in order:
   ```bash
   psql $DATABASE_URL -f supabase/migrations/001_initial_schema.sql
   psql $DATABASE_URL -f supabase/migrations/002_add_dataset_imports_and_task_fields.sql
   ```
3. Configure your `.env` file (see `.env.example`):
   ```env
   DATABASE_URL=postgresql+asyncpg://user:password@host:port/dbname
   CORS_ORIGINS=http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173
   ```
4. Seed the database with safe, de-identified demonstration data:
   ```bash
   python scripts/seed_database.py
   ```

---

## 📁 Repository Structure

```
├── backend/
│   ├── app/
│   │   ├── database.py         # Async SQLAlchemy engine & SQLite fallback
│   │   ├── inference.py        # Model loading, scoring, SHAP explainability
│   │   ├── models_db.py        # SQLAlchemy ORM models (Encounters, Care Plans, Datasets, Audit)
│   │   ├── schemas.py          # Pydantic v2 request/response schemas
│   │   └── routers/            # API endpoints: summary, encounters, predict, care_plans, governance, datasets
│   ├── artifacts/              # Persisted joblib models, metrics, and governance reports
│   └── main.py                 # FastAPI application entrypoint with lifespan startup
├── data/
│   ├── diabetic_data.csv       # UCI Diabetes 130-US Hospitals dataset (101,766 rows)
│   └── dataset_analysis.json   # UCI dataset distributions & clinical analysis
├── frontend/
│   ├── src/
│   │   ├── components/         # EncounterDrawer, ErrorBoundary, StatCards
│   │   ├── pages/              # Overview, Worklist, RiskCalculator, CarePlanTracker, Governance, DataImport, Provenance
│   │   ├── utils/              # API client (api.js) & helpers
│   │   ├── App.jsx             # Shell & 8-module sidebar navigation
│   │   └── styles.css          # Light clinical healthcare design system
│   ├── package.json
│   └── vite.config.js
├── scripts/
│   ├── analyze_dataset.py      # UCI dataset analysis script
│   └── seed_database.py        # 5,000 encounters & 120 care plans seeding script
├── src/
│   ├── preprocessing.py        # Cleaning, ICD-9 mapping, outlier capping, splits, SMOTE
│   ├── models.py               # Logistic Regression, Random Forest, Calibrated XGBoost
│   ├── explainability.py       # SHAP TreeExplainer & action recommendation logic
│   ├── fairness.py             # Subgroup fairness audit & Fairlearn mitigation
│   └── train.py                # End-to-end training and DB seeding pipeline
├── supabase/
│   └── migrations/             # PostgreSQL DDL migrations (001 and 002)
├── tests/
│   ├── test_preprocessing.py   # Data pipeline unit tests
│   ├── test_explainability.py  # SHAP & modifiable factor tests
│   └── test_api.py             # FastAPI integration tests (17 passed)
├── requirements.txt            # Python dependencies
└── README.md
```

---

## 📜 Research & Ethical Disclaimers
This project is built as an open research and hackathon demonstration. It adheres to ethical AI governance:
- Demographic features are audited for parity and excluded from direct risk scoring.
- Feature importance attributions represent statistical association, not medical causation.
- Care plans and action checklists serve as multidisciplinary administrative suggestions, not clinical prescriptions.
- All data shown is de-identified historical data; no real patient identifiers are stored or processed.
