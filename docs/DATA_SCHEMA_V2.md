# Medical Dataset Schema V2

This file defines the recommended upgraded dataset structure for the medically aligned version of the project.

## Why V2 Is Needed

The current model was trained with:

- one `BloodPressure` field
- no `Glucose_Type`

That is enough for a prototype, but it is not the best structure for real clinical interpretation.

## Recommended Columns

### Core Patient Fields

- `Patient_ID`
- `Age`
- `Gender`

### Symptoms

- `Symptoms`
- `Symptom_Count`

### Diagnosis Or Label Fields

- `Disease`
- `Risk_Factor_Group`

### Vitals And Medical Inputs

- `Glucose`
- `Glucose_Type`
- `SystolicBP`
- `DiastolicBP`
- `BMI`

## Column Definitions

- `Patient_ID`
  Unique patient identifier.

- `Age`
  Age in years.

- `Gender`
  Recommended values: `Male`, `Female`, `Other`

- `Symptoms`
  Comma-separated symptom list or normalized symptom string.

- `Symptom_Count`
  Count of symptoms in the `Symptoms` field.

- `Disease`
  Disease label if disease classification is still needed for secondary tasks.

- `Risk_Factor_Group`
  Recommended values: `Low`, `Moderate`, `High`
  This should be created from a well-defined medical risk policy during training.

- `Glucose`
  Glucose reading in `mg/dL`

- `Glucose_Type`
  Recommended values:
  - `fasting`
  - `random`

- `SystolicBP`
  Systolic blood pressure in `mmHg`

- `DiastolicBP`
  Diastolic blood pressure in `mmHg`

- `BMI`
  Body mass index

## Recommended Clinical Benchmarks

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

## Recommended Next Pipeline Changes

When V2 data becomes available:

1. Update `preprocessing.py` to validate:
   - `Glucose_Type`
   - `SystolicBP`
   - `DiastolicBP`
2. Update `Feature engineering.py` to create:
   - systolic and diastolic blood pressure features
   - glucose-type-aware glucose categories
   - improved risk-rule features
3. Retrain in `Model Training.py`
4. Recheck model accuracy
5. Replace the current inference-time mapping layer with a fully aligned feature pipeline
