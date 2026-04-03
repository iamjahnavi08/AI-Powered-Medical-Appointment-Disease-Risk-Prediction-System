# Healthcare Risk Prediction Engine

This project builds a healthcare risk prediction pipeline from raw patient data to a live API.

## Quickstart (2 commands)

1. Install deps:
   - `python -m pip install -r requirements.txt`
2. Run the UI + API server:
   - `python -m uvicorn backend.api_backend:app --reload --port 8001`

Open `http://127.0.0.1:8001/`.

## Project Flow

1. `scripts/preprocessing.py`
   Cleans the raw healthcare dataset and saves `data/ml/Healthcare_Cleaned.csv`.

2. `scripts/Feature engineering.py`
   Creates derived medical features and saves `data/ml/Healthcare_FeatureEngineered.csv`.

3. `scripts/Model Training.py`
   Trains the Logistic Regression risk model and saves the final model and evaluation files.

4. `api.py`
   Runs the FastAPI backend for live risk prediction during appointment booking.

5. `api_backend.py`
   Runs the full stack demo backend (login + role dashboards + booking) and serves the HTML UI pages.

## Main Files

- `data/ml/Healthcare.csv`
- `data/ml/Healthcare_Cleaned.csv`
- `data/ml/Healthcare_FeatureEngineered.csv`
- `models/trained_risk_stratification_model.pkl`
- `reports/risk_model_evaluation_summary.csv`
- `backend/risk_engine.py`
- `backend/api.py`
- `backend/api_backend.py`

## Run the Server (UI + API)

Use the backend that serves the HTML pages (patient/nurse/doctor + login):

- PowerShell: `./start_backend.ps1`
- CMD: `start_backend.bat`
- Direct: `python -m uvicorn backend.api_backend:app --reload --port 8001`

Pages:

- `http://127.0.0.1:8001/` (role selection)
- `http://127.0.0.1:8001/login`
- `http://127.0.0.1:8001/patient` / `nurse` / `doctor`

## Configuration (.env)

- Copy `.env.example` to `.env` to customize settings locally.
- In production, set `APP_ENV=production` and set a strong `APP_SESSION_SECRET` (>= 32 chars).

## Initialize the SQLite DB (optional)

This project stores app data in SQLite (default: `data/app/app.db`).

Run once to create tables/migrations:

- `python scripts/init_db.py`

Demo accounts (development only):

- `patient@example.com` / `DemoPass1!`
- `nurse@example.com` / `DemoPass1!`
- `doctor@example.com` / `DemoPass1!`

You can override the demo password via `APP_DEMO_PASSWORD`.

## Smoke test

- `python -m unittest discover -s tests -p "test_*.py" -v`

## Model Output

The current model predicts:

- `Low`
- `Moderate`
- `High`
- `Critical`

The final live API combines:

- trained Logistic Regression prediction
- clinical blood pressure rules
- clinical glucose rules
- BMI rules
- booking-priority recommendations

## Current Performance

From the latest evaluation:

- Train Accuracy: `94.19%`
- Validation Accuracy: `94.34%`
- Test Accuracy: `94.04%`
- Test Recall: `95.46%`

## Important Caution

The current live API is more clinically realistic than the original training dataset.

Current limitation:

- the trained dataset originally had only one `BloodPressure` field
- the trained dataset did not include `Glucose_Type`
- the live API now accepts:
  - `systolic_bp`
  - `diastolic_bp`
  - `glucose_type`

What this means:

- the backend is suitable for demo, prototype, and end-to-end testing
- the live engine uses better real-world clinical rules
- but the trained model still comes from an older feature format
- therefore, the system is not yet the final medically aligned training version

For a stronger medical version, the training dataset should be upgraded to include:

- `SystolicBP`
- `DiastolicBP`
- `Glucose_Type`

## Cookies

The UI uses an HTTP session cookie (signed) for login sessions. The frontend also shows a one-time cookie notice banner.

## Clinical Inputs Required

The live API expects:

- `age`
- `gender`
- `symptoms`
- `glucose`
- `glucose_type`
- `systolic_bp`
- `diastolic_bp`
- `bmi`

## Run The API

Use either command:

```bash
python api.py
```

or

```bash
uvicorn api:app --reload
```

After starting, open:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`

## API Endpoints

- `GET /`
  Basic API information and sample payload

- `GET /health`
  Health check for the service

- `GET /model-info`
  Model details, classes, and supported API inputs

- `POST /predict-risk`
  Returns live patient risk prediction

## Example Request

```json
{
  "age": 47,
  "gender": "Female",
  "symptoms": ["fatigue", "shortness of breath", "cough", "dizziness"],
  "glucose": 145.0,
  "glucose_type": "fasting",
  "systolic_bp": 148.0,
  "diastolic_bp": 96.0,
  "bmi": 31.2
}
```

## Example Response

```json
{
  "risk_group": "High",
  "model_risk_group": "High",
  "clinical_risk_group": "High",
  "risk_probabilities": {
    "High": 1.0,
    "Low": 0.0,
    "Moderate": 0.0
  },
  "booking_priority": "Urgent review recommended"
}
```

## Clinical Benchmarks Used

### Blood Pressure

- `Low Blood Pressure`: below `90/60`
- `Normal`: below `120/80`
- `Elevated`: `120-129` and below `80`
- `Stage 1 Hypertension`: `130-139` or `80-89`
- `Stage 2 Hypertension`: `140+` or `90+`
- `Critical`: above `180/120`

### Glucose

- `Severe Hypoglycemia`: below `54 mg/dL`
- `Low Glucose`: below `70 mg/dL`
- `Fasting Normal`: up to `99 mg/dL`
- `Fasting Prediabetes`: `100-125 mg/dL`
- `Fasting Diabetes Range`: `126+ mg/dL`
- `Random Elevated`: `140-199 mg/dL`
- `Random Diabetes Range`: `200+ mg/dL`

### BMI

- `Underweight`: below `18.5`
- `Healthy Weight`: `18.5` to below `25`
- `Overweight`: `25` to below `30`
- `Class 1 Obesity`: `30` to below `35`
- `Class 2 Obesity`: `35` to below `40`
- `Class 3 Obesity`: `40+`

## Notes

- The model was trained on the available project dataset, while the live API adds clinically grounded rule checks on top.
- For a medically stronger retraining pipeline in the future, the dataset should ideally contain:
  - `SystolicBP`
  - `DiastolicBP`
  - `Glucose_Type`
- See `DATA_SCHEMA_V2.md` and `Healthcare_Medical_Schema_Template.csv` for the recommended upgraded dataset format.
- See `FRONTEND_READINESS_CHECKLIST.md` for the checks to complete before frontend integration.
