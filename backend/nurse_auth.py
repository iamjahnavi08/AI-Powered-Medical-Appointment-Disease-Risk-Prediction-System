from __future__ import annotations

import csv
import hashlib
import hmac
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

from pydantic import BaseModel, field_validator

from paths import NURSE_ACCOUNTS_CSV


PASSWORD_POLICY_PATTERN = r"^(?=.*[A-Z])(?=.*[a-z])(?=.*\d)(?=.*[^A-Za-z0-9]).{8,}$"
PASSWORD_POLICY_MESSAGE = (
    "Password must contain minimum 8 characters, including uppercase (A-Z), lowercase (a-z), "
    "number (0-9), and special character (@,!,#,$,%,&,*)."
)


class NurseSignupRequest(BaseModel):
    nurse_id: str
    password: str

    @field_validator("nurse_id", "password")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("field cannot be empty")
        return cleaned


class NurseLoginRequest(BaseModel):
    nurse_id: str
    password: str

    @field_validator("nurse_id", "password")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("field cannot be empty")
        return cleaned


class NurseAuthManager:
    def __init__(self, csv_path: Path = NURSE_ACCOUNTS_CSV) -> None:
        self.csv_path = Path(csv_path)
        self._ensure_csv_exists()

    def _ensure_csv_exists(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if self.csv_path.exists():
            return
        with self.csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=["nurse_id", "salt_hex", "password_hash", "created_at"],
            )
            writer.writeheader()

    @staticmethod
    def _hash_password(password: str, salt: bytes) -> str:
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
        return digest.hex()

    def _read_accounts(self) -> Dict[str, Dict[str, str]]:
        self._ensure_csv_exists()
        accounts: Dict[str, Dict[str, str]] = {}
        with self.csv_path.open("r", newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                nurse_id = str(row.get("nurse_id", "")).strip()
                if nurse_id:
                    accounts[nurse_id] = row
        return accounts

    def signup(self, nurse_id: str, password: str) -> None:
        nid = nurse_id.strip()
        pwd = password.strip()
        if not nid:
            raise ValueError("Nurse ID is required.")
        if not re.fullmatch(PASSWORD_POLICY_PATTERN, pwd):
            raise ValueError(PASSWORD_POLICY_MESSAGE)

        accounts = self._read_accounts()
        if nid in accounts:
            raise ValueError("Nurse ID already exists. Please use login.")

        salt = os.urandom(16)
        password_hash = self._hash_password(pwd, salt)
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        with self.csv_path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=["nurse_id", "salt_hex", "password_hash", "created_at"],
            )
            writer.writerow(
                {
                    "nurse_id": nid,
                    "salt_hex": salt.hex(),
                    "password_hash": password_hash,
                    "created_at": created_at,
                }
            )

    def login(self, nurse_id: str, password: str) -> str:
        nid = nurse_id.strip()
        pwd = password.strip()
        accounts = self._read_accounts()

        if nid not in accounts:
            raise ValueError("Invalid Nurse ID or password.")

        row = accounts[nid]
        try:
            salt = bytes.fromhex(str(row.get("salt_hex", "")))
        except ValueError as exc:
            raise ValueError("Stored credential format is invalid.") from exc

        expected_hash = str(row.get("password_hash", ""))
        provided_hash = self._hash_password(pwd, salt)
        if not hmac.compare_digest(provided_hash, expected_hash):
            raise ValueError("Invalid Nurse ID or password.")

        return nid
