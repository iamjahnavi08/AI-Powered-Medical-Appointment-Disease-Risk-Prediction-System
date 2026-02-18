from __future__ import annotations

import json
import re
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.pkl"
LABEL_ENCODER_PATH = BASE_DIR / "label_encoder.pkl"
FEATURE_DATA_PATH = BASE_DIR / "Healthcare_FeatureEngineered.csv"


class RiskPredictionRequest(BaseModel):
    patient_features: Dict[str, Any] = Field(
        ...,
        description="Feature dictionary expected by the trained model.",
    )


class AppointmentBookingRequest(BaseModel):
    patient_id: str = Field(..., description="Unique patient identifier")
    doctor_id: str = Field(..., description="Assigned doctor identifier")
    appointment_time: datetime = Field(..., description="Requested appointment date-time")
    patient_features: Optional[Dict[str, Any]] = Field(
        None,
        description="Feature dictionary expected by the trained model.",
    )


class RiskPredictionResponse(BaseModel):
    predicted_class: str
    risk_probability: float
    confidence_breakdown: Dict[str, float]
    risk_level: str


class AppointmentBookingResponse(BaseModel):
    booking_status: str
    patient_id: str
    doctor_id: str
    appointment_time: datetime
    risk_assessment: RiskPredictionResponse


class RiskEngine:
    HIGH_RISK_SYMPTOMS = {"chest_pain", "diarrhea", "diarrhoea", "insomnia", "dizziness"}
    MEDIUM_RISK_SYMPTOMS = {
        "blurred_vision",
        "swelling",
        "depression",
        "sore_throat",
        "joint_pain",
        "anxiety",
        "muscle_pain",
        "appetite_loss",
        "runny_nose",
    }

    def __init__(self, model_path: Path, label_encoder_path: Optional[Path] = None) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = joblib.load(model_path)
        self.label_encoder = None

        if label_encoder_path and label_encoder_path.exists():
            self.label_encoder = joblib.load(label_encoder_path)
        self.patient_feature_map = self._load_patient_feature_map(FEATURE_DATA_PATH)

    @staticmethod
    def _normalize_patient_id(value: Any) -> str:
        text = str(value).strip()
        try:
            return str(int(float(text)))
        except ValueError:
            return text

    def _load_patient_feature_map(self, csv_path: Path) -> Dict[str, Dict[str, Any]]:
        if not csv_path.exists():
            return {}
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            return {}
        if "Patient_ID" not in df.columns:
            return {}

        expected_cols = list(getattr(self.model, "feature_names_in_", []))
        if not expected_cols:
            return {}

        missing_expected = [c for c in expected_cols if c not in df.columns]
        if missing_expected:
            return {}

        feature_map: Dict[str, Dict[str, Any]] = {}
        for _, row in df.iterrows():
            pid = self._normalize_patient_id(row["Patient_ID"])
            features = {}
            for col in expected_cols:
                val = row[col]
                features[col] = None if pd.isna(val) else val
            feature_map[pid] = features
        return feature_map

    def get_patient_features(self, patient_id: str) -> Dict[str, Any]:
        pid = self._normalize_patient_id(patient_id)
        if pid not in self.patient_feature_map:
            raise ValueError(f"Patient_ID not found in dataset: {patient_id}")
        return self.patient_feature_map[pid]

    def _normalize_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(features)

        # Common client typo support.
        if "Sympton_Count" in normalized and "Symptom_Count" not in normalized:
            normalized["Symptom_Count"] = normalized.pop("Sympton_Count")

        # Model was trained with numeric gender values (male=1, female=0, unknown=-1).
        if "Gender" in normalized:
            g = normalized["Gender"]
            if isinstance(g, str):
                g_clean = g.strip().lower()
                gender_map = {
                    "male": 1,
                    "m": 1,
                    "female": 0,
                    "f": 0,
                    "other": -1,
                    "unknown": -1,
                }
                if g_clean in gender_map:
                    normalized["Gender"] = gender_map[g_clean]

        # Try to coerce expected numeric columns from string payloads.
        preprocessor = getattr(self.model, "named_steps", {}).get("preprocessor")
        if preprocessor is not None:
            for name, _, cols in getattr(preprocessor, "transformers", []):
                if name != "num":
                    continue
                for col in cols:
                    if col not in normalized:
                        continue
                    value = normalized[col]
                    if isinstance(value, str):
                        v = value.strip()
                        if v == "":
                            continue
                        try:
                            normalized[col] = float(v)
                        except ValueError:
                            continue

        return normalized

    def _risk_level(self, probability: float) -> str:
        if probability >= 0.75:
            return "high"
        if probability >= 0.4:
            return "medium"
        return "low"

    @staticmethod
    def _normalize_symptom_name(value: str) -> str:
        token = value.strip().lower()
        token = token.replace(" ", "_").replace("-", "_")
        token = re.sub(r"[^a-z0-9_]+", "", token)
        return token

    def _extract_reported_symptoms(self, features: Dict[str, Any]) -> set[str]:
        found: set[str] = set()

        raw_symptoms = features.get("Symptoms")
        if isinstance(raw_symptoms, str):
            parts = [p.strip() for p in re.split(r"[,;|]+", raw_symptoms) if p.strip()]
            found.update(self._normalize_symptom_name(p) for p in parts)

        for key, value in features.items():
            if not str(key).startswith("SYM_"):
                continue
            symptom_name = self._normalize_symptom_name(str(key)[4:])
            try:
                is_present = float(value) == 1.0
            except (TypeError, ValueError):
                is_present = str(value).strip().lower() in {"true", "yes", "y"}
            if is_present:
                found.add(symptom_name)

        return found

    def _rule_based_risk_class(self, features: Dict[str, Any]) -> str:
        symptoms = self._extract_reported_symptoms(features)
        if symptoms & self.HIGH_RISK_SYMPTOMS:
            return "High"
        if symptoms & self.MEDIUM_RISK_SYMPTOMS:
            return "Medium"
        return "Low"

    def predict(self, features: Dict[str, Any]) -> RiskPredictionResponse:
        if not features:
            raise ValueError("patient_features cannot be empty")

        normalized_features = self._normalize_features(features)
        row = pd.DataFrame([normalized_features])

        # Align with model training schema when available.
        expected_cols = getattr(self.model, "feature_names_in_", None)
        if expected_cols is not None:
            missing = [col for col in expected_cols if col not in row.columns]
            if missing:
                raise ValueError(f"Missing required feature(s): {missing}")
            row = row[list(expected_cols)]

        # Predict using explicit symptom-priority rules requested by user.
        predicted_class = self._rule_based_risk_class(normalized_features)
        confidence_breakdown = {"Low": 0.0, "Medium": 0.0, "High": 0.0}
        confidence_breakdown[predicted_class] = 1.0
        risk_probability = 1.0

        return RiskPredictionResponse(
            predicted_class=str(predicted_class),
            risk_probability=risk_probability,
            confidence_breakdown=confidence_breakdown,
            risk_level=predicted_class.lower(),
        )


app = FastAPI(
    title="Disease Risk Prediction Engine",
    version="1.0.0",
    description="Live disease risk estimation integrated with appointment booking.",
)


try:
    risk_engine = RiskEngine(MODEL_PATH, LABEL_ENCODER_PATH)
except Exception as exc:  # pragma: no cover - startup guard
    raise RuntimeError(f"Failed to initialize risk engine: {exc}") from exc

APPOINTMENTS: List[Dict[str, Any]] = []


def _to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if pd.isna(value):
        return None
    return value


def _default_features_json() -> str:
    def default_value_for_feature(feature_name: str) -> Any:
        numeric_defaults = {
            "Age": 45,
            "Symptom_Count": 1,
            "Glucose": 95,
            "BloodPressure": 120,
            "BMI": 24.5,
        }
        categorical_defaults = {
            "Gender": 1,
            "Symptoms": "none",
            "Age_Group": "Adult",
            "BMI_Category": "Normal",
            "BP_Category": "Normal",
        }

        if feature_name in numeric_defaults:
            return numeric_defaults[feature_name]
        if feature_name in categorical_defaults:
            return categorical_defaults[feature_name]
        if feature_name.startswith("SYM_"):
            return 0
        return 0

    expected_cols = getattr(risk_engine.model, "feature_names_in_", None)
    default_features: Dict[str, Any]
    if expected_cols is not None:
        default_features = {col: default_value_for_feature(str(col)) for col in expected_cols}
    else:
        default_features = {
            "Age": 45,
            "Gender": 1,
            "Symptoms": "none",
            "Symptom_Count": 1,
            "Glucose": 95,
            "BloodPressure": 120,
            "BMI": 24.5,
            "Age_Group": "Adult",
            "BMI_Category": "Normal",
            "BP_Category": "Normal",
        }
    return escape(json.dumps(default_features, indent=2))


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Risk Prediction Portal</title>
  <style>
    body { font-family: Consolas, monospace; margin: 0; padding: 28px; background: #f5f8ff; color: #1b1f2a; }
    .wrap { max-width: 860px; margin: 0 auto; }
    .card { background: #fff; border: 1px solid #d9dce7; border-radius: 12px; padding: 18px; }
    a { display: inline-block; margin-right: 12px; margin-top: 8px; text-decoration: none; color: #0b7a75; font-weight: 700; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Risk Prediction Portal</h1>
      <p>Use separate pages for patient booking and doctor review.</p>
      <a href="/patient">Open Patient Page</a>
      <a href="/doctor">Open Doctor Page</a>
      <a href="/docs">Open API Docs</a>
    </div>
  </div>
</body>
</html>
"""


@app.get("/patient", response_class=HTMLResponse)
def patient_page() -> str:
    default_features_json = _default_features_json()
    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Patient Portal</title>
  <style>
    :root {
      --ink: #1b1f2a;
      --cloud: #f6f7fb;
      --teal: #0b7a75;
      --sun: #ff9f1c;
      --mint: #2ec4b6;
      --paper: #ffffff;
      --line: #d9dce7;
      --danger: #9f1239;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      font-family: "Consolas", "Lucida Console", monospace;
      color: var(--ink);
      background:
        radial-gradient(circle at 12% 18%, rgba(46, 196, 182, 0.15), transparent 42%),
        radial-gradient(circle at 88% 8%, rgba(255, 159, 28, 0.18), transparent 34%),
        linear-gradient(150deg, #f1f5ff, #fcfcff 48%, #f8fbff);
      min-height: 100vh;
    }

    .shell {
      max-width: 1080px;
      margin: 0 auto;
      padding: 28px 18px 36px;
      animation: rise 500ms ease-out;
    }

    h1, h2 {
      font-family: "Palatino Linotype", "Book Antiqua", serif;
      margin: 0;
      letter-spacing: 0.02em;
    }

    h1 {
      font-size: clamp(1.7rem, 3.5vw, 2.5rem);
    }

    .subhead {
      margin: 10px 0 22px;
      color: #394058;
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
    }

    .card {
      background: var(--paper);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 10px 24px rgba(29, 39, 71, 0.08);
    }

    label {
      display: block;
      margin: 10px 0 6px;
      font-weight: 600;
      font-size: 0.95rem;
    }

    input, textarea, button {
      width: 100%;
      border-radius: 10px;
      border: 1px solid var(--line);
      font-family: inherit;
      font-size: 0.95rem;
    }

    input, textarea {
      padding: 10px 12px;
      background: #fbfcff;
    }

    textarea {
      min-height: 120px;
      resize: vertical;
    }

    button {
      margin-top: 14px;
      padding: 10px 12px;
      border: 0;
      background: linear-gradient(120deg, var(--teal), #1164a3);
      color: #fff;
      cursor: pointer;
      font-weight: 700;
      transition: transform 160ms ease, filter 160ms ease;
    }

    button:hover {
      transform: translateY(-1px);
      filter: brightness(1.05);
    }

    .output {
      margin-top: 14px;
      padding: 12px;
      border-radius: 10px;
      background: #f7f9ff;
      border: 1px solid #d5dcf0;
      white-space: pre-wrap;
      overflow-x: auto;
      min-height: 58px;
    }

    .quicklinks {
      margin-top: 14px;
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }

    .quicklinks a {
      color: var(--teal);
      text-decoration: none;
      font-weight: 700;
    }

    .error {
      color: var(--danger);
    }

    @keyframes rise {
      from { opacity: 0; transform: translateY(8px); }
      to { opacity: 1; transform: translateY(0); }
    }

    @media (max-width: 900px) {
      .grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <main class="shell">
    <h1>Patient Portal</h1>
    <p class="subhead">Book appointments only.</p>

    <section class="grid">
      <article class="card">
        <h2>Book Appointment</h2>
        <label for="patientId">Patient ID</label>
        <input id="patientId" placeholder="e.g. 1520" />
        <button id="loadPatientBtn" type="button">Load Features From Patient ID</button>
        <label for="doctorId">Doctor ID</label>
        <input id="doctorId" placeholder="D-209" />
        <label for="appointmentTime">Appointment Time</label>
        <input id="appointmentTime" type="datetime-local" />
        <label for="bookFeatures">Patient Features (JSON)</label>
        <textarea id="bookFeatures">__DEFAULT_FEATURES_JSON__</textarea>
        <button id="bookBtn" type="button">Book Appointment</button>
        <pre class="output" id="bookOutput">Waiting for input...</pre>
      </article>
    </section>

    <div class="quicklinks">
      <a href="/doctor">Open Doctor Page</a>
      <a href="/docs" target="_blank" rel="noreferrer">Open Swagger Docs</a>
      <a href="/health" target="_blank" rel="noreferrer">Check Health</a>
    </div>
  </main>

  <script>
    async function postJson(url, payload) {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!res.ok) {
        const detail = data && data.detail ? data.detail : "Request failed";
        throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      }
      return data;
    }

    function parseFeatures(textareaId) {
      const raw = document.getElementById(textareaId).value;
      try {
        return JSON.parse(raw);
      } catch (err) {
        throw new Error("Invalid JSON in patient features.");
      }
    }

    function renderOutput(targetId, value, isError) {
      const el = document.getElementById(targetId);
      el.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
      el.className = isError ? "output error" : "output";
    }

    async function loadPatientFeatures() {
      const patientId = document.getElementById("patientId").value.trim();
      if (!patientId) {
        throw new Error("Enter Patient ID first.");
      }
      renderOutput("bookOutput", "Loading patient features...", false);
      const data = await fetch(`/patient-features/${encodeURIComponent(patientId)}`);
      const payload = await data.json();
      if (!data.ok) {
        const detail = payload && payload.detail ? payload.detail : "Patient lookup failed";
        throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      }
      const featuresPretty = JSON.stringify(payload.patient_features, null, 2);
      document.getElementById("bookFeatures").value = featuresPretty;
      renderOutput("bookOutput", payload, false);
    }

    document.getElementById("loadPatientBtn").addEventListener("click", async () => {
      try {
        await loadPatientFeatures();
      } catch (err) {
        renderOutput("bookOutput", err.message, true);
      }
    });

    document.getElementById("bookBtn").addEventListener("click", async () => {
      try {
        let patient_features;
        try {
          patient_features = parseFeatures("bookFeatures");
        } catch (_) {
          patient_features = null;
        }
        const payload = {
          patient_id: document.getElementById("patientId").value.trim(),
          doctor_id: document.getElementById("doctorId").value.trim(),
          appointment_time: document.getElementById("appointmentTime").value,
          patient_features
        };
        renderOutput("bookOutput", "Submitting...", false);
        const data = await postJson("/book-appointment", payload);
        const bookingSummary = {
          booking_status: data.booking_status,
          patient_id: data.patient_id,
          doctor_id: data.doctor_id,
          appointment_time: data.appointment_time
        };
        renderOutput("bookOutput", bookingSummary, false);
      } catch (err) {
        renderOutput("bookOutput", err.message, true);
      }
    });
  </script>
</body>
</html>
"""
    return html_template.replace("__DEFAULT_FEATURES_JSON__", default_features_json)


@app.get("/doctor", response_class=HTMLResponse)
def doctor_page() -> str:
    default_features_json = _default_features_json()
    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Doctor Dashboard</title>
  <style>
    body { margin: 0; font-family: Consolas, monospace; background: #f6f9ff; color: #1b1f2a; }
    .shell { max-width: 1100px; margin: 0 auto; padding: 24px 18px; }
    .card { background: #fff; border: 1px solid #d9dce7; border-radius: 12px; padding: 14px; margin-top: 14px; }
    table { width: 100%; border-collapse: collapse; font-size: 0.92rem; }
    th, td { border-bottom: 1px solid #e5e9f5; text-align: left; padding: 10px 8px; vertical-align: top; }
    th { background: #eef4ff; }
    .pill { padding: 2px 8px; border-radius: 999px; font-weight: 700; }
    .high { background: #fee2e2; color: #991b1b; }
    .medium { background: #fef3c7; color: #92400e; }
    .low { background: #dcfce7; color: #166534; }
    .links a { color: #0b7a75; text-decoration: none; font-weight: 700; margin-right: 10px; }
    pre { margin: 0; white-space: pre-wrap; }
    textarea, button {
      width: 100%;
      border-radius: 10px;
      font-family: inherit;
      font-size: 0.95rem;
    }
    textarea {
      min-height: 130px;
      border: 1px solid #d9dce7;
      padding: 10px 12px;
      background: #fbfcff;
    }
    button {
      margin-top: 10px;
      padding: 10px 12px;
      border: 0;
      background: linear-gradient(120deg, #0b7a75, #1164a3);
      color: #fff;
      cursor: pointer;
      font-weight: 700;
    }
    .output {
      margin-top: 10px;
      padding: 10px 12px;
      border-radius: 10px;
      background: #f7f9ff;
      border: 1px solid #d5dcf0;
      min-height: 58px;
    }
    .error { color: #9f1239; }
  </style>
</head>
<body>
  <main class="shell">
    <h1>Doctor Dashboard</h1>
    <p>New bookings appear automatically.</p>
    <div class="links">
      <a href="/patient">Open Patient Page</a>
      <a href="/docs">Open API Docs</a>
    </div>
    <div class="card">
      <h2>Predict Risk</h2>
      <label for="riskFeatures">Patient Features (JSON)</label>
      <textarea id="riskFeatures">__DEFAULT_FEATURES_JSON__</textarea>
      <button id="predictBtn" type="button">Run Prediction</button>
      <pre class="output" id="predictOutput">Waiting for input...</pre>
    </div>
    <div class="card">
      <table>
        <thead>
          <tr>
            <th>Booked At</th>
            <th>Patient ID</th>
            <th>Doctor ID</th>
            <th>Appointment Time</th>
            <th>Predicted Risk</th>
            <th>Patient Details</th>
          </tr>
        </thead>
        <tbody id="rows">
          <tr><td colspan="6">No appointments yet.</td></tr>
        </tbody>
      </table>
    </div>
  </main>
  <script>
    async function postJson(url, payload) {
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!res.ok) {
        const detail = data && data.detail ? data.detail : "Request failed";
        throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      }
      return data;
    }

    function parseFeatures() {
      const raw = document.getElementById("riskFeatures").value;
      try {
        return JSON.parse(raw);
      } catch (_) {
        throw new Error("Invalid JSON in patient features.");
      }
    }

    function renderOutput(targetId, value, isError) {
      const el = document.getElementById(targetId);
      el.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
      el.className = isError ? "output error" : "output";
    }

    function riskPill(level) {
      const cls = (level || "low").toLowerCase();
      return `<span class="pill ${cls}">${level || "-"}</span>`;
    }

    function toSafeText(value) {
      return value == null ? "" : String(value);
    }

    async function refreshAppointments() {
      const res = await fetch("/appointments");
      const data = await res.json();
      const rows = document.getElementById("rows");
      if (!data.appointments || data.appointments.length === 0) {
        rows.innerHTML = '<tr><td colspan="6">No appointments yet.</td></tr>';
        return;
      }
      rows.innerHTML = data.appointments.map((item) => `
        <tr>
          <td>${toSafeText(item.booked_at)}</td>
          <td>${toSafeText(item.patient_id)}</td>
          <td>${toSafeText(item.doctor_id)}</td>
          <td>${toSafeText(item.appointment_time)}</td>
          <td>${riskPill(item.risk_assessment && item.risk_assessment.risk_level)}</td>
          <td><pre>${toSafeText(JSON.stringify(item.patient_features, null, 2))}</pre></td>
        </tr>
      `).join("");
    }

    document.getElementById("predictBtn").addEventListener("click", async () => {
      try {
        const patient_features = parseFeatures();
        renderOutput("predictOutput", "Submitting...", false);
        const data = await postJson("/predict-risk", { patient_features });
        renderOutput("predictOutput", data, false);
      } catch (err) {
        renderOutput("predictOutput", err.message, true);
      }
    });

    refreshAppointments();
    setInterval(refreshAppointments, 4000);
  </script>
</body>
</html>
"""
    return html_template.replace("__DEFAULT_FEATURES_JSON__", default_features_json)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "service": "risk_prediction_engine"}


@app.post("/predict-risk", response_model=RiskPredictionResponse)
def predict_risk(payload: RiskPredictionRequest) -> RiskPredictionResponse:
    try:
        return risk_engine.predict(payload.patient_features)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.post("/book-appointment", response_model=AppointmentBookingResponse)
def book_appointment(payload: AppointmentBookingRequest) -> AppointmentBookingResponse:
    features = payload.patient_features
    if features is None:
        try:
            features = risk_engine.get_patient_features(payload.patient_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        risk_assessment = risk_engine.predict(features)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Risk assessment failed: {exc}") from exc

    booking_status = "confirmed"
    response = AppointmentBookingResponse(
        booking_status=booking_status,
        patient_id=payload.patient_id,
        doctor_id=payload.doctor_id,
        appointment_time=payload.appointment_time,
        risk_assessment=risk_assessment,
    )
    APPOINTMENTS.append(
        {
            "booked_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "patient_id": payload.patient_id,
            "doctor_id": payload.doctor_id,
            "appointment_time": payload.appointment_time.isoformat(),
            "patient_features": _to_jsonable(features),
            "risk_assessment": response.risk_assessment.model_dump(),
        }
    )
    return response


@app.get("/patient-features/{patient_id}")
def patient_features(patient_id: str) -> Dict[str, Any]:
    try:
        features = risk_engine.get_patient_features(patient_id)
        return {"patient_id": patient_id, "patient_features": features}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/appointments")
def list_appointments() -> Dict[str, Any]:
    return {"appointments": APPOINTMENTS}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("risk_prediction_engine:app", host="0.0.0.0", port=8000, reload=True)
