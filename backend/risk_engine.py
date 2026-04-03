import re
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_FILE = PROJECT_ROOT / "models" / "trained_risk_stratification_model.pkl"
REFERENCE_DATA_FILE = PROJECT_ROOT / "data" / "ml" / "Healthcare_FeatureEngineered.csv"

NULL_LIKE_VALUES = {"", "nan", "none", "null", "na", "n/a"}
GENDER_MAP = {
    "male": "Male",
    "m": "Male",
    "female": "Female",
    "f": "Female",
    "other": "Other",
    "non-binary": "Other",
    "non binary": "Other",
    "unknown": "Other",
    "prefer not to say": "Other",
}
GENDER_CODE_MAP = {"Female": 0, "Male": 1, "Other": 2}
BMI_SCORE_MAP = {"Underweight": 1, "Normal": 0, "Overweight": 1, "Obese": 2}
GLUCOSE_SCORE_MAP = {"Low": 1, "Normal": 0, "Prediabetes": 1, "High": 2}
BP_SCORE_MAP = {"Low": 1, "Normal": 0, "Prehypertension": 1, "High": 2}
SYMPTOM_SCORE_MAP = {"Minimal": 0, "Moderate": 1, "High": 2, "Severe": 3}
RISK_RANK = {"Low": 0, "Moderate": 1, "High": 2, "Critical": 3}


def normalize_text(value: object, lowercase: bool = False) -> str | None:
    if value is None or pd.isna(value):
        return None

    text = re.sub(r"\s+", " ", str(value)).strip()
    if text.lower() in NULL_LIKE_VALUES:
        return None

    return text.lower() if lowercase else text


def standardize_gender(value: object) -> str:
    normalized = normalize_text(value, lowercase=True)
    if normalized is None:
        return "Other"
    return GENDER_MAP.get(normalized, "Other")


def split_symptoms(value: object) -> list[str]:
    normalized = normalize_text(value, lowercase=True)
    if normalized is None or normalized == "unknown":
        return []
    return [part.strip() for part in re.split(r"[,;|]+", normalized) if part.strip()]


def standardize_symptoms(value: object) -> str:
    if isinstance(value, list):
        parts = [normalize_text(part, lowercase=True) for part in value]
        cleaned_parts = [part for part in parts if part]
    else:
        cleaned_parts = split_symptoms(value)

    if not cleaned_parts:
        return "unknown"

    return ", ".join(dict.fromkeys(cleaned_parts))


def make_safe_name(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", text.lower().replace(" ", "_")).strip("_")


def age_group(value: float) -> str:
    if value <= 18:
        return "Child"
    if value <= 35:
        return "Young"
    if value <= 50:
        return "Adult"
    if value <= 65:
        return "Middle_Age"
    return "Senior"


def bmi_category(value: float) -> str:
    if value < 18.5:
        return "Underweight"
    if value < 25:
        return "Normal"
    if value < 30:
        return "Overweight"
    return "Obese"


def bmi_clinical_category(value: float) -> str:
    if value < 18.5:
        return "Underweight"
    if value < 25:
        return "Healthy Weight"
    if value < 30:
        return "Overweight"
    if value < 35:
        return "Class 1 Obesity"
    if value < 40:
        return "Class 2 Obesity"
    return "Class 3 Obesity"


def legacy_glucose_category(value: float) -> str:
    if value < 70:
        return "Low"
    if value < 100:
        return "Normal"
    if value < 126:
        return "Prediabetes"
    return "High"


def symptom_load_group(count: int) -> str:
    if count <= 1:
        return "Minimal"
    if count <= 3:
        return "Moderate"
    if count <= 5:
        return "High"
    return "Severe"


def categorize_blood_pressure(systolic_bp: float, diastolic_bp: float) -> dict[str, Any]:
    if systolic_bp < 90 or diastolic_bp < 60:
        clinical_category = "Low Blood Pressure"
        legacy_category = "Low"
        severity_points = 2
        urgent_flag = False
        medical_alerts = ["Blood pressure is below 90/60 mmHg."]
    elif systolic_bp > 180 or diastolic_bp > 120:
        clinical_category = "Severe Hypertension"
        legacy_category = "High"
        severity_points = 4
        urgent_flag = True
        medical_alerts = ["Blood pressure is above 180/120 mmHg."]
    elif systolic_bp >= 140 or diastolic_bp >= 90:
        clinical_category = "Stage 2 Hypertension"
        legacy_category = "High"
        severity_points = 3
        urgent_flag = False
        medical_alerts = ["Blood pressure is in the Stage 2 hypertension range."]
    elif systolic_bp >= 130 or diastolic_bp >= 80:
        clinical_category = "Stage 1 Hypertension"
        legacy_category = "Prehypertension"
        severity_points = 2
        urgent_flag = False
        medical_alerts = ["Blood pressure is in the Stage 1 hypertension range."]
    elif systolic_bp >= 120 and diastolic_bp < 80:
        clinical_category = "Elevated Blood Pressure"
        legacy_category = "Prehypertension"
        severity_points = 1
        urgent_flag = False
        medical_alerts = ["Blood pressure is elevated above normal."]
    else:
        clinical_category = "Normal"
        legacy_category = "Normal"
        severity_points = 0
        urgent_flag = False
        medical_alerts = []

    return {
        "clinical_category": clinical_category,
        "legacy_model_category": legacy_category,
        "severity_points": severity_points,
        "urgent_flag": urgent_flag,
        "medical_alerts": medical_alerts,
    }


def categorize_glucose(glucose: float, glucose_type: str) -> dict[str, Any]:
    normalized_type = normalize_text(glucose_type, lowercase=True)
    if normalized_type not in {"fasting", "random"}:
        raise ValueError("glucose_type must be either 'fasting' or 'random'.")

    medical_alerts: list[str] = []

    if glucose < 54:
        clinical_category = "Severe Hypoglycemia"
        legacy_category = "Low"
        severity_points = 4
        urgent_flag = True
        medical_alerts.append("Glucose is below 54 mg/dL.")
    elif glucose < 70:
        clinical_category = "Low Glucose"
        legacy_category = "Low"
        severity_points = 3
        urgent_flag = False
        medical_alerts.append("Glucose is below 70 mg/dL.")
    elif normalized_type == "fasting":
        if glucose <= 99:
            clinical_category = "Normal Fasting Glucose"
            legacy_category = "Normal"
            severity_points = 0
            urgent_flag = False
        elif glucose <= 125:
            clinical_category = "Prediabetes Range"
            legacy_category = "Prediabetes"
            severity_points = 1
            urgent_flag = False
            medical_alerts.append("Fasting glucose is in the prediabetes range.")
        else:
            clinical_category = "Diabetes Range"
            legacy_category = "High"
            severity_points = 3
            urgent_flag = False
            medical_alerts.append("Fasting glucose is in the diabetes range.")
    else:
        if glucose < 140:
            clinical_category = "Normal Random Glucose"
            legacy_category = "Normal"
            severity_points = 0
            urgent_flag = False
        elif glucose < 200:
            clinical_category = "Elevated Random Glucose"
            legacy_category = "Prediabetes"
            severity_points = 1
            urgent_flag = False
            medical_alerts.append(
                "Random glucose is elevated but not by itself diagnostic for diabetes."
            )
        else:
            clinical_category = "Diabetes Range"
            legacy_category = "High"
            severity_points = 3
            urgent_flag = False
            medical_alerts.append("Random glucose is in the diabetes range.")

    return {
        "clinical_category": clinical_category,
        "legacy_model_category": legacy_category,
        "severity_points": severity_points,
        "urgent_flag": urgent_flag,
        "medical_alerts": medical_alerts,
        "glucose_type": normalized_type,
    }


def assess_bmi(bmi: float) -> dict[str, Any]:
    clinical_category = bmi_clinical_category(bmi)
    legacy_category = bmi_category(bmi)

    if clinical_category == "Healthy Weight":
        severity_points = 0
        medical_alerts: list[str] = []
    elif clinical_category in {"Underweight", "Overweight", "Class 1 Obesity"}:
        severity_points = 1 if clinical_category != "Class 1 Obesity" else 2
        medical_alerts = [f"BMI is in the {clinical_category} range."]
    else:
        severity_points = 3
        medical_alerts = [f"BMI is in the {clinical_category} range."]

    return {
        "clinical_category": clinical_category,
        "legacy_model_category": legacy_category,
        "severity_points": severity_points,
        "medical_alerts": medical_alerts,
    }


def determine_clinical_risk(
    *,
    bp_assessment: dict[str, Any],
    glucose_assessment: dict[str, Any],
    bmi_assessment: dict[str, Any],
    symptom_count: int,
) -> dict[str, Any]:
    symptom_points = 2 if symptom_count >= 4 else 1 if symptom_count >= 2 else 0
    total_points = (
        bp_assessment["severity_points"]
        + glucose_assessment["severity_points"]
        + bmi_assessment["severity_points"]
        + symptom_points
    )

    alerts = (
        bp_assessment["medical_alerts"]
        + glucose_assessment["medical_alerts"]
        + bmi_assessment["medical_alerts"]
    )

    urgent_flag = bool(bp_assessment["urgent_flag"] or glucose_assessment["urgent_flag"])

    if urgent_flag:
        risk_group = "Critical"
        booking_priority = "Immediate emergency review recommended"
    elif total_points >= 6:
        risk_group = "High"
        booking_priority = "Urgent clinical review recommended"
    elif total_points >= 3:
        risk_group = "Moderate"
        booking_priority = "Priority follow-up recommended"
    else:
        risk_group = "Low"
        booking_priority = "Standard scheduling is acceptable"

    return {
        "risk_group": risk_group,
        "clinical_score": total_points,
        "symptom_points": symptom_points,
        "booking_priority": booking_priority,
        "urgent_flag": urgent_flag,
        "alerts": alerts,
    }


class RiskPredictionEngine:
    def __init__(
        self,
        model_file: Path = MODEL_FILE,
        reference_data_file: Path = REFERENCE_DATA_FILE,
    ) -> None:
        self.model = joblib.load(model_file)
        self.reference_df = pd.read_csv(reference_data_file)
        self.expected_columns = list(self.model.feature_names_in_)
        self.symptom_feature_columns = [column for column in self.expected_columns if column.startswith("SYM_")]
        self.pair_feature_columns = [column for column in self.expected_columns if column.startswith("PAIR_")]
        self._build_reference_tables()

    def _build_reference_tables(self) -> None:
        reference = self.reference_df.copy()
        reference["Symptoms"] = reference["Symptoms"].apply(standardize_symptoms)
        reference["Symptoms_List"] = reference["Symptoms"].apply(split_symptoms)

        self.symptom_frequency = reference["Symptoms"].value_counts().to_dict()
        self.age_symptom_frequency = (
            reference.groupby(["Age_Group", "Symptoms"], observed=False).size().to_dict()
        )
        self.gender_symptom_frequency = reference.groupby(["Gender", "Symptoms"]).size().to_dict()

        pair_counter: Counter[tuple[str, str]] = Counter()
        for symptoms in reference["Symptoms_List"]:
            pair_counter.update(combinations(sorted(set(symptoms)), 2))

        self.expected_pairs: dict[str, tuple[str, str]] = {}
        for first_symptom, second_symptom in pair_counter:
            column_name = f"PAIR_{make_safe_name(first_symptom)}_{make_safe_name(second_symptom)}"
            if column_name in self.pair_feature_columns:
                self.expected_pairs[column_name] = (first_symptom, second_symptom)

    def build_feature_row(
        self,
        *,
        age: int,
        gender: str,
        symptoms: str | list[str],
        glucose: float,
        glucose_type: str,
        systolic_bp: float,
        diastolic_bp: float,
        bmi: float,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        standardized_gender = standardize_gender(gender)
        standardized_symptoms = standardize_symptoms(symptoms)
        symptoms_list = split_symptoms(standardized_symptoms)
        symptom_name_set = {make_safe_name(symptom) for symptom in symptoms_list}

        bp_assessment = categorize_blood_pressure(systolic_bp, diastolic_bp)
        glucose_assessment = categorize_glucose(glucose, glucose_type)
        bmi_assessment = assess_bmi(bmi)

        symptom_count = len(symptoms_list)
        unique_symptom_count = len(set(symptoms_list))
        patient_age_group = age_group(age)
        patient_symptom_load_group = symptom_load_group(symptom_count)

        bmi_severity = BMI_SCORE_MAP[bmi_assessment["legacy_model_category"]]
        glucose_severity = GLUCOSE_SCORE_MAP[glucose_assessment["legacy_model_category"]]
        bp_severity = BP_SCORE_MAP[bp_assessment["legacy_model_category"]]
        symptom_severity = SYMPTOM_SCORE_MAP[patient_symptom_load_group]
        vital_severity = bmi_severity + glucose_severity + bp_severity
        overall_severity = vital_severity + symptom_severity

        symptom_profile_frequency = int(self.symptom_frequency.get(standardized_symptoms, 0))
        age_group_profile_frequency = int(
            self.age_symptom_frequency.get((patient_age_group, standardized_symptoms), 0)
        )
        gender_profile_frequency = int(
            self.gender_symptom_frequency.get((standardized_gender, standardized_symptoms), 0)
        )
        repeated_symptom_pattern = int(symptom_profile_frequency >= 3)
        recurring_demographic_pattern = int(
            age_group_profile_frequency >= 2 or gender_profile_frequency >= 2
        )
        historical_pattern_score = repeated_symptom_pattern + recurring_demographic_pattern

        row: dict[str, Any] = {
            "Age": int(age),
            "Gender": standardized_gender,
            "Gender_Encoded": GENDER_CODE_MAP[standardized_gender],
            "Symptom_Count": symptom_count,
            "Glucose": float(glucose),
            # The training dataset's single BloodPressure field is diastolic-like by value range.
            "BloodPressure": float(diastolic_bp),
            "BMI": float(bmi),
            "Unique_Symptom_Count": unique_symptom_count,
            "Symptom_Load_Group": patient_symptom_load_group,
            "Multi_System_Symptoms": int(unique_symptom_count >= 4),
            "Age_Group": patient_age_group,
            "BMI_Category": bmi_assessment["legacy_model_category"],
            "Glucose_Category": glucose_assessment["legacy_model_category"],
            "BP_Category": bp_assessment["legacy_model_category"],
            "BMI_Severity_Score": bmi_severity,
            "Glucose_Severity_Score": glucose_severity,
            "BP_Severity_Score": bp_severity,
            "Symptom_Severity_Score": symptom_severity,
            "Vital_Severity_Score": vital_severity,
            "Overall_Severity_Score": overall_severity,
            "Symptom_Profile_Frequency": symptom_profile_frequency,
            "Age_Group_Profile_Frequency": age_group_profile_frequency,
            "Gender_Profile_Frequency": gender_profile_frequency,
            "Repeated_Symptom_Pattern": repeated_symptom_pattern,
            "Recurring_Demographic_Pattern": recurring_demographic_pattern,
            "Historical_Pattern_Score": historical_pattern_score,
        }

        for column_name in self.symptom_feature_columns:
            symptom_key = column_name.replace("SYM_", "", 1)
            row[column_name] = int(symptom_key in symptom_name_set)

        symptom_set = set(symptoms_list)
        for column_name in self.pair_feature_columns:
            first_symptom, second_symptom = self.expected_pairs.get(column_name, ("", ""))
            row[column_name] = int(first_symptom in symptom_set and second_symptom in symptom_set)

        clinical_risk = determine_clinical_risk(
            bp_assessment=bp_assessment,
            glucose_assessment=glucose_assessment,
            bmi_assessment=bmi_assessment,
            symptom_count=symptom_count,
        )

        clinical_assessment = {
            "blood_pressure": {
                "systolic_bp": float(systolic_bp),
                "diastolic_bp": float(diastolic_bp),
                "clinical_category": bp_assessment["clinical_category"],
            },
            "glucose": {
                "value": float(glucose),
                "glucose_type": glucose_assessment["glucose_type"],
                "clinical_category": glucose_assessment["clinical_category"],
            },
            "bmi": {
                "value": float(bmi),
                "clinical_category": bmi_assessment["clinical_category"],
            },
            "symptoms": {
                "standardized_symptoms": standardized_symptoms,
                "symptom_count": symptom_count,
            },
            "clinical_risk_group": clinical_risk["risk_group"],
            "clinical_score": clinical_risk["clinical_score"],
            "alerts": clinical_risk["alerts"],
            "urgent_flag": clinical_risk["urgent_flag"],
        }

        features = pd.DataFrame(
            [[row.get(column_name, 0) for column_name in self.expected_columns]],
            columns=self.expected_columns,
        )
        return features, clinical_assessment

    def predict(
        self,
        *,
        age: int,
        gender: str,
        symptoms: str | list[str],
        glucose: float,
        glucose_type: str,
        systolic_bp: float,
        diastolic_bp: float,
        bmi: float,
    ) -> dict[str, Any]:
        features, clinical_assessment = self.build_feature_row(
            age=age,
            gender=gender,
            symptoms=symptoms,
            glucose=glucose,
            glucose_type=glucose_type,
            systolic_bp=systolic_bp,
            diastolic_bp=diastolic_bp,
            bmi=bmi,
        )
        model_prediction = str(self.model.predict(features)[0])
        probabilities = self.model.predict_proba(features)[0]
        model_probabilities = {
            label: round(float(score), 4)
            for label, score in zip(self.model.classes_, probabilities)
        }

        clinical_risk_group = str(clinical_assessment["clinical_risk_group"])
        final_risk_group = (
            model_prediction
            if RISK_RANK[model_prediction] >= RISK_RANK[clinical_risk_group]
            else clinical_risk_group
        )

        if clinical_assessment["urgent_flag"]:
            booking_priority = "Immediate emergency review recommended"
        elif final_risk_group == "High":
            booking_priority = "Urgent review recommended"
        elif final_risk_group == "Moderate":
            booking_priority = "Priority follow-up recommended"
        else:
            booking_priority = "Standard scheduling is acceptable"

        return {
            "risk_group": final_risk_group,
            "model_risk_group": model_prediction,
            "clinical_risk_group": clinical_risk_group,
            "risk_probabilities": model_probabilities,
            "booking_priority": booking_priority,
            "clinical_assessment": clinical_assessment,
            "engineered_features": features.iloc[0].to_dict(),
        }


engine = RiskPredictionEngine()
