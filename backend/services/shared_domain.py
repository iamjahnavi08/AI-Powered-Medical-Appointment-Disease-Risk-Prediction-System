from __future__ import annotations

from .shared_core import *  # noqa: F403
from .shared_models import *  # noqa: F403

def parse_iso_datetime(value: str) -> datetime:
    # Accept "Z" suffix and naive strings; store normalized UTC.
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def appointment_to_payload(appointment: dict[str, Any]) -> dict[str, Any]:
    patient = get_user_by_email(appointment["patient_email"])
    risk = compute_patient_risk(patient.get("health_details") if patient else None)
    return {
        "id": int(appointment["id"]),
        "patient_email": appointment["patient_email"],
        "patient_name": patient["full_name"] if patient else appointment["patient_email"],
        "scheduled_for": appointment["scheduled_for"],
        "status": appointment.get("status", "Pending"),
        "appointment_type": appointment.get("appointment_type"),
        "doctor_email": appointment.get("doctor_email"),
        "contact_info": appointment.get("contact_info"),
        "priority": appointment.get("priority"),
        "predicted_risk_level": risk.get("risk_level"),
        "predicted_risk_source": risk.get("source"),
        "booking_priority": risk.get("booking_priority"),
        "health_details_submitted_at": (patient.get("health_details") or {}).get("submitted_at") if patient else None,
        "reason": appointment.get("reason") or appointment.get("appointment_type") or "",
        "notes": appointment.get("notes"),
        "created_at": appointment.get("created_at") or "",
    }


def parse_blood_pressure(value: object) -> tuple[float, float] | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    if len(numbers) < 2:
        return None

    try:
        systolic = float(numbers[0])
        diastolic = float(numbers[1])
    except ValueError:
        return None

    if systolic <= 0 or diastolic <= 0:
        return None
    return (systolic, diastolic)


def compute_patient_risk(health_details: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(health_details, dict) or not health_details:
        return {}

    age = health_details.get("age")
    gender = health_details.get("gender") or "Other"
    symptoms = health_details.get("symptoms") or ""
    bmi = health_details.get("bmi")
    glucose = health_details.get("glucose")
    bp = parse_blood_pressure(health_details.get("blood_pressure"))

    quick: dict[str, Any] | None = None
    try:
        if isinstance(age, int) and symptoms:
            quick = quick_risk_from_age_symptoms(age=age, symptoms=str(symptoms))
    except Exception:
        quick = None

    # Prefer the full model only when we have the required vitals.
    if (
        isinstance(age, int)
        and isinstance(bmi, (int, float))
        and isinstance(glucose, (int, float))
        and bp is not None
        and symptoms
    ):
        try:
            systolic_bp, diastolic_bp = bp
            model_result = engine.predict(
                age=age,
                gender=str(gender),
                symptoms=str(symptoms),
                glucose=float(glucose),
                glucose_type="fasting",
                systolic_bp=float(systolic_bp),
                diastolic_bp=float(diastolic_bp),
                bmi=float(bmi),
            )

            group = (model_result.get("risk_group") or "").strip()
            # Normalize to the 3-level UI.
            normalized = "Low"
            if group in ("Critical", "High"):
                normalized = "High"
            elif group == "Moderate":
                normalized = "Medium"
            elif group == "Low":
                normalized = "Low"
            else:
                normalized = quick.get("risk_level") if quick else "Low"

            # Symptom-only high-risk keyword override: if the quick symptom scan is High,
            # prefer High even when the full model predicts lower.
            booking_priority = model_result.get("booking_priority")
            if quick and str(quick.get("risk_level") or "").strip() == "High" and normalized != "High":
                normalized = "High"
                booking_priority = "Urgent review recommended"

            return {
                "risk_level": normalized,
                "source": "model",
                "booking_priority": booking_priority,
            }
        except Exception:
            # Fall back to quick estimate.
            pass

    if quick:
        risk_level = str(quick.get("risk_level") or "Low")
        if risk_level == "High":
            booking_priority = "Urgent review recommended"
        elif risk_level == "Medium":
            booking_priority = "Priority follow-up recommended"
        else:
            booking_priority = "Standard scheduling is acceptable"
        return {"risk_level": risk_level, "source": "quick", "booking_priority": booking_priority}

    return {}


def normalize_symptom_tokens(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        text = ", ".join([str(v) for v in value if v is not None])
    else:
        text = str(value)
    parts = [re.sub(r"\s+", " ", part).strip().lower() for part in re.split(r"[,;|\n]+", text)]
    return [part for part in parts if part]


def nurse_rule_assessment_from_inputs(
    *,
    age: int,
    gender: str,
    symptoms: str | list[str],
    glucose: float | None,
    glucose_type: str | None,
    systolic_bp: float | None,
    diastolic_bp: float | None,
    bmi: float | None,
) -> dict[str, Any]:
    validation_errors: list[str] = []

    # Nurse-facing validation (prevents obviously unsafe / impossible entries).
    if age <= 0 or age > 120:
        validation_errors.append("Invalid age (must be 1â€“120).")

    if systolic_bp is None or diastolic_bp is None:
        validation_errors.append("Blood pressure is missing.")
    else:
        if systolic_bp < 50 or systolic_bp > 250:
            validation_errors.append("Invalid systolic BP (must be 50â€“250).")
        if diastolic_bp < 30 or diastolic_bp > 150:
            validation_errors.append("Invalid diastolic BP (must be 30â€“150).")

    if glucose is None:
        validation_errors.append("Glucose is missing.")
    else:
        if glucose <= 0:
            validation_errors.append("Invalid glucose (must be > 0).")
        if glucose > 1200:
            validation_errors.append("Invalid glucose (value too high).")

    if bmi is None:
        validation_errors.append("BMI is missing.")
    else:
        if bmi <= 10 or bmi > 60:
            validation_errors.append("Invalid BMI (must be 10â€“60).")

    symptom_tokens = normalize_symptom_tokens(symptoms)
    symptom_count = len(symptom_tokens)

    stop_words = {"in", "on", "of", "the", "a", "an", "and", "with", "to", "for", "at"}

    def symptom_has(phrase: str) -> bool:
        target = str(phrase or "").strip().lower()
        if not target:
            return False

        target_words = [w for w in re.findall(r"[a-z0-9]+", target) if w and w not in stop_words]
        for token in symptom_tokens:
            if target in token:
                return True
            token_words = [w for w in re.findall(r"[a-z0-9]+", token) if w and w not in stop_words]
            token_set = set(token_words)
            if target_words and all(w in token_set for w in target_words):
                return True
        return False

    symptom_quick: dict[str, Any] | None = None
    try:
        symptom_quick = quick_risk_from_age_symptoms(age=age, symptoms=str(symptoms))
    except Exception:
        symptom_quick = None
    symptom_keyword_high = bool(symptom_quick and str(symptom_quick.get("risk_level") or "").strip() == "High")

    emergency_alerts: list[str] = []
    # Symptom-only emergency triggers (still valid even before vitals are recorded).
    if symptom_has("unconscious") or symptom_has("unconsciousness"):
        emergency_alerts.append("Critical condition: unconscious.")
    if symptom_has("chest pain") and symptom_has("shortness of breath"):
        emergency_alerts.append("Possible heart attack: chest pain + shortness of breath.")
    if symptom_count >= 5:
        emergency_alerts.append("Severe symptom load (5+ symptoms).")

    tags: list[str] = []
    if symptom_has("fever") and symptom_has("cough"):
        tags.append("Infection suspected")

    # If vitals are invalid/missing, return a minimal rule response without engine feature mapping.
    if validation_errors:
        triage_level = "Critical" if emergency_alerts else ("High" if symptom_keyword_high else "Needs vitals")
        show_on_top = triage_level in {"Critical", "High"}
        if triage_level == "Critical":
            appointment_recommendation = "Immediate (0â€“10 mins)"
        elif triage_level == "High":
            appointment_recommendation = "Within 1 hour (vitals ASAP)"
        else:
            appointment_recommendation = "Vitals required"
        return {
            "valid": False,
            "validation_errors": validation_errors,
            "triage_level": triage_level,
            "priority": "Immediate attention"
            if triage_level == "Critical"
            else ("Urgent consultation (symptoms)" if triage_level == "High" else "Vitals required"),
            "show_on_top": show_on_top,
            "emergency_alerts": emergency_alerts,
            "tags": tags,
            "appointment_recommendation": appointment_recommendation,
            "suggestions": [
                "Confirm vitals using correct equipment and technique.",
                "If the patient looks unwell or symptoms are worrying, escalate per clinic protocol.",
            ],
            "nurse_comment": (
                "Escalate immediately and follow emergency protocol."
                if triage_level == "Critical"
                else (
                    "High-risk symptoms present. Capture vitals urgently and fast-track for review."
                    if triage_level == "High"
                    else "Complete vitals entry to enable full triage."
                )
            ),
            "derived": {
                "symptom_count": symptom_count,
            },
        }

    # Full triage using rule-based clinical thresholds already implemented in the engine.
    features, clinical = engine.build_feature_row(
        age=age,
        gender=gender,
        symptoms=symptoms,
        glucose=float(glucose),
        glucose_type=str(glucose_type or "fasting"),
        systolic_bp=float(systolic_bp),
        diastolic_bp=float(diastolic_bp),
        bmi=float(bmi),
    )
    clinical_bp_category = str(clinical["blood_pressure"]["clinical_category"])
    clinical_glucose_category = str(clinical["glucose"]["clinical_category"])
    clinical_bmi_category = str(clinical["bmi"]["clinical_category"])
    clinical_risk_group = str(clinical["clinical_risk_group"])

    # Emergency / critical conditions (top priority)
    if float(systolic_bp) > 180 or float(diastolic_bp) > 120:
        emergency_alerts.append("Hypertensive crisis (BP > 180/120).")
    if float(glucose) < 54:
        emergency_alerts.append("Severe hypoglycemia (glucose < 54 mg/dL).")
    if float(glucose) > 300:
        emergency_alerts.append("Severe hyperglycemia (glucose > 300 mg/dL).")

    triage_level = "Critical" if emergency_alerts else clinical_risk_group

    # High risk priority (fast track)
    if triage_level != "Critical":
        if clinical_bp_category in {"Stage 2 Hypertension", "Severe Hypertension"}:
            triage_level = "High"
        if clinical_glucose_category == "Diabetes Range":
            triage_level = "High"
        if symptom_keyword_high:
            triage_level = "High"

    # Symptom-based tags
    if symptom_has("headache") and clinical_bp_category in {
        "Stage 1 Hypertension",
        "Stage 2 Hypertension",
        "Severe Hypertension",
    }:
        tags.append("Hypertension-related")
    if symptom_has("fatigue") and clinical_glucose_category == "Diabetes Range":
        tags.append("Diabetes-related")
    if symptom_count >= 4:
        tags.append("Multi-system issue")

    # Appointment scheduling rules
    if triage_level == "Critical":
        appointment_recommendation = "Immediate (0â€“10 mins)"
        priority = "Immediate emergency review recommended"
    elif triage_level == "High":
        appointment_recommendation = "Within 1 hour"
        priority = "Urgent consultation"
    elif triage_level == "Moderate":
        appointment_recommendation = "Same day"
        priority = "Priority follow-up"
    else:
        appointment_recommendation = "Normal booking"
        priority = "Normal booking"

    feature_row = features.iloc[0].to_dict()
    repeated_symptom_pattern = int(feature_row.get("Repeated_Symptom_Pattern") or 0)
    historical_pattern_score = int(feature_row.get("Historical_Pattern_Score") or 0)

    notes: list[str] = []
    if repeated_symptom_pattern == 1:
        notes.append("Recurring symptom pattern seen in the reference dataset (pattern match).")
    if historical_pattern_score >= 2:
        notes.append("Reference patterns suggest a possible chronic trend (not a diagnosis).")

    suggestions: list[str] = []
    if float(glucose) < 70:
        suggestions.append("If confirmed low glucose, provide fast-acting carbohydrate per protocol and recheck.")
    if symptom_has("fever"):
        suggestions.append("Supportive care and temperature monitoring per protocol.")
    if clinical_bp_category in {"Stage 1 Hypertension", "Stage 2 Hypertension", "Severe Hypertension"}:
        suggestions.append("Recheck BP after rest using correct cuff size and positioning.")

    if triage_level == "Critical":
        nurse_comment = "Escalate immediately and follow emergency protocol."
    elif triage_level == "High":
        nurse_comment = "Fast-track for urgent evaluation and document vitals + key symptoms."
    elif triage_level == "Moderate":
        nurse_comment = "Schedule same-day review; monitor symptoms and recheck vitals if needed."
    else:
        nurse_comment = "Proceed with normal workflow; provide standard precautions and follow-up guidance."

    show_on_top = bool(emergency_alerts) or bool(clinical.get("urgent_flag"))

    return {
        "valid": True,
        "validation_errors": [],
        "triage_level": triage_level,
        "priority": priority,
        "show_on_top": show_on_top,
        "emergency_alerts": emergency_alerts,
        "tags": sorted(set(tags)),
        "appointment_recommendation": appointment_recommendation,
        "suggestions": suggestions,
        "nurse_comment": nurse_comment,
        "derived": {
            "symptom_count": int(clinical["symptoms"]["symptom_count"]),
            "blood_pressure_category": clinical_bp_category,
            "glucose_category": clinical_glucose_category,
            "bmi_category": clinical_bmi_category,
            "clinical_risk_group": clinical_risk_group,
            "repeated_symptom_pattern": repeated_symptom_pattern,
            "historical_pattern_score": historical_pattern_score,
            "notes": notes,
        },
    }


def nurse_rule_assessment_from_health_details(health_details: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(health_details, dict) or not health_details:
        return None

    bp = parse_blood_pressure(health_details.get("blood_pressure"))
    systolic_bp = bp[0] if bp else None
    diastolic_bp = bp[1] if bp else None

    return nurse_rule_assessment_from_inputs(
        age=int(health_details.get("age") or 0),
        gender=str(health_details.get("gender") or "Other"),
        symptoms=str(health_details.get("symptoms") or ""),
        glucose=(float(health_details.get("glucose")) if health_details.get("glucose") is not None else None),
        glucose_type=str(health_details.get("glucose_type") or "fasting"),
        systolic_bp=(float(systolic_bp) if systolic_bp is not None else None),
        diastolic_bp=(float(diastolic_bp) if diastolic_bp is not None else None),
        bmi=(float(health_details.get("bmi")) if health_details.get("bmi") is not None else None),
    )

def split_symptoms_text(symptoms: str) -> list[str]:
    return [part.strip() for part in re.split(r"[,;|\n]+", symptoms or "") if part.strip()]


def quick_risk_from_age_symptoms(age: int, symptoms: str) -> dict[str, Any]:
    # NOTE: By user request, this quick risk is symptom-based only.
    symptom_list = split_symptoms_text(symptoms or "")
    normalized = [s.lower().strip() for s in symptom_list if s.strip()]

    stop_words = {"in", "on", "of", "the", "a", "an", "and", "with", "to", "for", "at"}

    def matches(symptom_text: str, keyword: str) -> bool:
        text = str(symptom_text or "").strip().lower()
        target = str(keyword or "").strip().lower()
        if not text or not target:
            return False

        # Fast path (also catches variants like "feverish" for keyword "fever").
        if target in text:
            return True

        text_words = [w for w in re.findall(r"[a-z0-9]+", text) if w and w not in stop_words]
        target_words = [w for w in re.findall(r"[a-z0-9]+", target) if w and w not in stop_words]
        if not target_words:
            return False
        text_set = set(text_words)
        return all(w in text_set for w in target_words)

    high_keywords = [
        "chest pain",
        "difficulty breathing",
        "unconsciousness",
        "unconscious",
        "severe bleeding",
        "seizure",
        "seizures",
        "slurred speech",
        "confusion",
        "bluish lips",
        "bluish face",
        "blue lips",
        "blue face",
        "severe head injury",
        "head injury",
        "uncontrolled vomiting",
        "sudden weakness",
        "one side weakness",
        "weakness one side",
        "stroke",
    ]

    medium_keywords = [
        "moderate fever",
        "persistent cough",
        "shortness of breath",
        "dehydration",
        "abdominal pain",
        "dizziness",
        "extreme fatigue",
        "fatigue (extreme)",
        "swelling",
        "rash with fever",
        "frequent vomiting",
        "painful urination",
    ]

    low_keywords = [
        "mild fever",
        "runny nose",
        "sore throat",
        "mild cough",
        "headache",
        "body aches",
        "mild fatigue",
        "sneezing",
        "slight nausea",
        "minor cuts",
        "bruises",
        "minor cuts/bruises",
    ]

    def has_any(substr: str) -> bool:
        return any(matches(s, substr) for s in normalized)

    # Special handling for fever/cough/vomiting intensity words.
    has_fever = has_any("fever")
    fever_is_high = has_any("high fever") or has_any("very high") or has_any("persistent fever")
    fever_is_moderate = has_any("moderate fever")
    fever_is_mild = has_any("mild fever")

    cough_is_persistent = has_any("persistent cough")
    cough_is_mild = has_any("mild cough")

    vomiting_is_uncontrolled = has_any("uncontrolled vomiting")
    vomiting_is_frequent = has_any("frequent vomiting")

    # Compute matches.
    matched_high: list[str] = []
    matched_medium: list[str] = []
    matched_low: list[str] = []

    for kw in high_keywords:
        if kw and has_any(kw):
            matched_high.append(kw)

    for kw in medium_keywords:
        if kw and has_any(kw):
            matched_medium.append(kw)

    for kw in low_keywords:
        if kw and has_any(kw):
            matched_low.append(kw)

    # Map intensity-derived symptoms into buckets.
    if has_fever:
        if fever_is_high:
            matched_high.append("high fever")
        elif fever_is_moderate:
            matched_medium.append("moderate fever")
        elif fever_is_mild:
            matched_low.append("mild fever")
        else:
            # Default fever severity if no qualifier provided.
            matched_medium.append("fever")

    if cough_is_persistent:
        matched_medium.append("persistent cough")
    elif cough_is_mild:
        matched_low.append("mild cough")

    if vomiting_is_uncontrolled:
        matched_high.append("uncontrolled vomiting")
    elif vomiting_is_frequent or has_any("vomiting"):
        matched_medium.append("frequent vomiting")

    # Combination rule requested:
    # - high + low => high
    # - medium + low => medium
    # - any high => high
    # - else any medium => medium
    # - else any low => low
    severity = "Low"
    if matched_high:
        severity = "High"
    elif matched_medium:
        severity = "Medium"
    elif matched_low:
        severity = "Low"
    else:
        severity = "Low"

    # Score is only for display; severity is the decision.
    score = 0
    score += 3 * len(set(matched_high))
    score += 1 * len(set(matched_medium))
    score += 0 * len(set(matched_low))

    triggers: list[str] = []
    if matched_high:
        triggers.append("Matched emergency warning sign symptom(s).")
    elif matched_medium:
        triggers.append("Matched symptoms needing medical attention soon.")
    elif matched_low:
        triggers.append("Matched mild/self-care symptoms.")

    # Include the exact matched keywords (deduped).
    seen_kw: set[str] = set()
    for kw in matched_high + matched_medium + matched_low:
        if kw in seen_kw:
            continue
        seen_kw.add(kw)
        triggers.append(kw)

    if severity == "High":
        advice = "High risk: consider prompt medical evaluation, especially if symptoms are severe or worsening."
    elif severity == "Medium":
        advice = "Medium risk: book a medical review soon and monitor symptoms."
    else:
        advice = "Low risk: try basic self-care and monitor. Book a visit if symptoms persist or worsen."

    return {
        "risk_level": severity,
        "score": score,
        "symptom_count": len(symptom_list),
        "symptoms": ", ".join(symptom_list),
        "triggers": triggers[:10],
        "advice": advice,
        "note": "Quick estimate based only on symptoms (not a medical diagnosis).",
        "matched": {
            "high": sorted(set(matched_high)),
            "medium": sorted(set(matched_medium)),
            "low": sorted(set(matched_low)),
        },
    }
