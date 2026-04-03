# Frontend Readiness Checklist

Use this checklist before building or connecting the frontend.

## Accuracy And Backend Recheck

- Re-run `preprocessing.py`
- Re-run `Feature engineering.py`
- Re-run `Model Training.py`
- Confirm `risk_model_evaluation_summary.csv` still shows the expected benchmark:
  - Train Accuracy around `94%`
  - Validation Accuracy around `94%`
  - Test Accuracy around `94%`
  - Test Recall around `95%`

## API Health Checks

- Start the backend with `python api.py` or `uvicorn api:app --reload`
- Confirm `GET /health` returns status `ok`
- Confirm `GET /model-info` returns the supported inputs
- Confirm `GET /` shows the example payload
- Confirm `POST /predict-risk` works for:
  - one normal or moderate case
  - one high-risk case
  - one critical case

## Required Frontend Input Fields

The frontend must send:

- `age`
- `gender`
- `symptoms`
- `glucose`
- `glucose_type`
- `systolic_bp`
- `diastolic_bp`
- `bmi`

## Response Fields To Display

At minimum, show:

- `risk_group`
- `model_risk_group`
- `clinical_risk_group`
- `booking_priority`
- `clinical_assessment.alerts`

Recommended additional display:

- blood pressure category
- glucose category
- BMI category
- class probabilities

## Validation Checks For The Frontend

- Age should be between `0` and `120`
- `glucose_type` should be either `fasting` or `random`
- `systolic_bp` should be greater than `0`
- `diastolic_bp` should be greater than `0`
- `bmi` should be greater than `0`
- symptoms should allow either:
  - free text
  - chips/tags list

## Important Product Decision

The current project is ready for:

- demo
- prototype
- end-to-end frontend testing

The current project is not yet the final medically aligned training version because the original training dataset does not contain:

- `SystolicBP`
- `DiastolicBP`
- `Glucose_Type`

## Before Production-Style Use

Before treating this as a stronger clinical version, do these checks:

1. Upgrade the dataset to Schema V2
2. Retrain the model with the upgraded schema
3. Recheck accuracy and recall
4. Revalidate API outputs against known medical scenarios
5. Add audit logging and error handling in the backend
