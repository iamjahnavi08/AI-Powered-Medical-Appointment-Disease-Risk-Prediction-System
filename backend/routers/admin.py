from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request

from backend.services.shared import (
    NurseRuleAssessment,
    RiskPredictionRequest,
    RiskPredictionResponse,
    engine,
    nurse_rule_assessment_from_inputs,
    quick_risk_from_age_symptoms,
    require_role,
    require_session,
)

router = APIRouter()


@router.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "healthcare-risk-prediction-api",
        "model": "trained_risk_stratification_model.pkl",
        "rules": "adult clinical glucose, blood pressure, and BMI thresholds enabled",
    }


@router.get("/model-info")
def model_info() -> dict[str, Any]:
    return {
        "model_name": "Logistic Regression",
        "target": "Risk_Factor_Group",
        "classes": list(engine.model.classes_),
        "final_risk_benchmarks": ["Low", "Moderate", "High", "Critical"],
        "api_inputs": [
            "age",
            "gender",
            "symptoms",
            "glucose",
            "glucose_type",
            "systolic_bp",
            "diastolic_bp",
            "bmi",
        ],
        "notes": [
            "Blood pressure uses adult clinical systolic/diastolic thresholds.",
            "Glucose uses adult fasting or random glucose thresholds.",
            "The historical ML model was trained on legacy tabular features, so the engine maps live clinical inputs into the model's expected feature schema.",
        ],
    }


@router.post("/predict-risk", response_model=RiskPredictionResponse)
def predict_risk(request: Request, payload: RiskPredictionRequest) -> RiskPredictionResponse:
    require_role("patient", "nurse", "doctor")(require_session(request))
    result = engine.predict(
        age=payload.age,
        gender=payload.gender,
        symptoms=payload.symptoms,
        glucose=payload.glucose,
        glucose_type=payload.glucose_type,
        systolic_bp=payload.systolic_bp,
        diastolic_bp=payload.diastolic_bp,
        bmi=payload.bmi,
    )

    # Symptom-only high-risk keyword override (e.g., "difficulty in breathing" should still map to High).
    try:
        symptom_text = (
            ", ".join([str(s) for s in payload.symptoms]) if isinstance(payload.symptoms, list) else str(payload.symptoms)
        )
        symptom_quick = quick_risk_from_age_symptoms(age=payload.age, symptoms=symptom_text)
        if str(symptom_quick.get("risk_level") or "").strip() == "High":
            current = str(result.get("risk_group") or "").strip()
            if current in {"Low", "Moderate"}:
                result["risk_group"] = "High"
                result["booking_priority"] = "Urgent review recommended"
    except Exception:
        pass

    result["nurse_rules"] = nurse_rule_assessment_from_inputs(
        age=payload.age,
        gender=payload.gender,
        symptoms=payload.symptoms,
        glucose=payload.glucose,
        glucose_type=payload.glucose_type,
        systolic_bp=payload.systolic_bp,
        diastolic_bp=payload.diastolic_bp,
        bmi=payload.bmi,
    )
    return RiskPredictionResponse(**result)


@router.post("/nurse/triage", response_model=NurseRuleAssessment)
def nurse_triage(request: Request, payload: RiskPredictionRequest) -> NurseRuleAssessment:
    require_role("nurse", "doctor")(require_session(request))
    assessment = nurse_rule_assessment_from_inputs(
        age=payload.age,
        gender=payload.gender,
        symptoms=payload.symptoms,
        glucose=payload.glucose,
        glucose_type=payload.glucose_type,
        systolic_bp=payload.systolic_bp,
        diastolic_bp=payload.diastolic_bp,
        bmi=payload.bmi,
    )
    return NurseRuleAssessment(**assessment)


@router.get("/risk/quick")
def quick_risk(age: int = Query(..., ge=0, le=120), symptoms: str = Query(default="")) -> dict[str, Any]:
    return quick_risk_from_age_symptoms(age=age, symptoms=symptoms)
