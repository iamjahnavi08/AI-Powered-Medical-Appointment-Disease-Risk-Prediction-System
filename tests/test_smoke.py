import importlib
import os
import tempfile
import unittest
import uuid
from datetime import datetime, timedelta, timezone


def build_client():
    tmpdir = tempfile.TemporaryDirectory()
    os.environ["APP_ENV"] = "development"
    os.environ["APP_SESSION_SECRET"] = "test-secret-change-me-please-32chars"
    os.environ["APP_DB_PATH"] = os.path.join(tmpdir.name, "test.db")

    import backend.config  # noqa: WPS433
    import backend.api_backend  # noqa: WPS433

    importlib.reload(backend.config)
    importlib.reload(backend.api_backend)

    from fastapi.testclient import TestClient  # noqa: WPS433

    return TestClient(backend.api_backend.app), tmpdir, backend.api_backend


class SmokeTests(unittest.TestCase):
    def test_basic_flows(self):
        client, tmpdir, backend_api = build_client()
        try:
            res = client.get("/health")
            self.assertEqual(res.status_code, 200)

            # Register a fresh patient user.
            token = uuid.uuid4().hex[:8]
            email = f"smoke_{token}@example.com"
            digits = str(uuid.uuid4().int)[-10:]
            phone = f"+91{digits}"
            password = "StrongPass1!"

            res = client.post(
                "/auth/register",
                json={
                    "full_name": "Smoke Patient",
                    "email": email,
                    "phone": phone,
                    "password": password,
                    "role": "patient",
                },
            )
            self.assertEqual(res.status_code, 200, res.text)
            csrf = res.json().get("csrf_token")
            self.assertTrue(csrf)
            headers = {"X-CSRF-Token": csrf}

            # Session user lookup.
            res = client.get("/auth/user", params={"email": email})
            self.assertEqual(res.status_code, 200, res.text)

            # Submit health details (required before booking UX flows).
            res = client.post(
                "/auth/health-details",
                headers=headers,
                json={
                    "email": email,
                    "age": 30,
                    "gender": "Female",
                    "symptoms": "cough, fatigue",
                    "symptom_count": 2,
                    "glucose": None,
                    "blood_pressure": None,
                    "height_cm": 165,
                    "weight_kg": 60,
                    "bmi": 22.0,
                    "smoking_habit": "No",
                    "alcohol_habit": "No",
                    "average_sleep_hours": 7.5,
                    "medical_history": "",
                    "family_history_major_chronic_disease": "No",
                },
            )
            self.assertEqual(res.status_code, 200, res.text)

            # Authenticated prediction.
            res = client.post(
                "/predict-risk",
                headers=headers,
                json={
                    "age": 30,
                    "gender": "Female",
                    "symptoms": ["cough", "fatigue"],
                    "glucose": 95.0,
                    "glucose_type": "fasting",
                    "systolic_bp": 120.0,
                    "diastolic_bp": 80.0,
                    "bmi": 22.0,
                },
            )
            self.assertEqual(res.status_code, 200, res.text)
            self.assertIn("risk_group", res.json())

            # Booking should work with session + CSRF.
            scheduled_for = (datetime.now(timezone.utc) + timedelta(days=2)).isoformat()
            res = client.post(
                "/appointments/book",
                headers=headers,
                json={
                    "patient_email": email,
                    "scheduled_for": scheduled_for,
                    "appointment_type": "General checkup",
                    "contact_info": phone,
                    "doctor_email": None,
                    "reason": "Routine",
                    "notes": "",
                },
            )
            self.assertEqual(res.status_code, 200, res.text)
            appt_id = res.json().get("id")
            self.assertTrue(appt_id)

            # Logout should require CSRF, and succeed.
            res = client.post("/auth/logout", headers=headers)
            self.assertEqual(res.status_code, 200, res.text)
        finally:
            # Ensure SQLite file handles are released on Windows.
            try:
                if getattr(backend_api, "DB_CONN", None) is not None:
                    backend_api.DB_CONN.close()
                    backend_api.DB_CONN = None
            except Exception:
                pass
            tmpdir.cleanup()


if __name__ == "__main__":
    unittest.main()
