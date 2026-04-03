import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
INPUT_FILE = PROJECT_ROOT / "data" / "ml" / "Healthcare.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "ml" / "Healthcare_Cleaned.csv"

NUMERIC_LIMITS = {
    "Age": (0, 120),
    "Glucose": (40, 400),
    "BloodPressure": (30, 250),
    "BMI": (10, 80),
}
NUMERIC_COLUMNS = ["Patient_ID", "Age", "Glucose", "BloodPressure", "BMI"]
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
ORDERED_COLUMNS = [
    "Patient_ID",
    "Age",
    "Gender",
    "Gender_Encoded",
    "Symptoms",
    "Symptom_Count",
    "Disease",
    "Glucose",
    "BloodPressure",
    "BMI",
]


@dataclass
class PreprocessingReport:
    input_rows: int
    output_rows: int
    duplicates_removed: int
    saved_file: Path


def normalize_text(value: object, lowercase: bool = False) -> object:
    if pd.isna(value):
        return pd.NA

    text = re.sub(r"\s+", " ", str(value)).strip()
    if text.lower() in NULL_LIKE_VALUES:
        return pd.NA

    return text.lower() if lowercase else text


def standardize_gender(value: object) -> str:
    normalized = normalize_text(value, lowercase=True)
    if pd.isna(normalized):
        return "Other"
    return GENDER_MAP.get(normalized, "Other")


def standardize_symptoms(value: object) -> str:
    normalized = normalize_text(value, lowercase=True)
    if pd.isna(normalized):
        return "unknown"

    parts = [part.strip() for part in re.split(r"[,;|]+", normalized) if part.strip()]
    if not parts:
        return "unknown"

    return ", ".join(dict.fromkeys(parts))


def count_symptoms(value: object) -> int:
    standardized = standardize_symptoms(value)
    if standardized == "unknown":
        return 0
    return len(standardized.split(", "))


def save_csv_with_fallback(df: pd.DataFrame, output_file: Path) -> Path:
    try:
        df.to_csv(output_file, index=False)
        return output_file
    except PermissionError:
        for counter in range(1, 100):
            fallback_file = output_file.with_name(
                f"{output_file.stem}_new_{counter}{output_file.suffix}"
            )
            try:
                df.to_csv(fallback_file, index=False)
                print(
                    f"\n'{output_file.name}' is open or locked. "
                    f"Saved the updated file as '{fallback_file.name}' instead."
                )
                return fallback_file
            except PermissionError:
                continue
        raise


def load_dataset(input_file: Path) -> pd.DataFrame:
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    return pd.read_csv(input_file)


def clean_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    for column, (lower, upper) in NUMERIC_LIMITS.items():
        if column in df.columns:
            df.loc[~df[column].between(lower, upper), column] = np.nan

    for column in ["Age", "Glucose", "BloodPressure", "BMI"]:
        if column in df.columns:
            median_value = float(df[column].median())
            df[column] = df[column].fillna(median_value)

    if "Age" in df.columns:
        df["Age"] = df["Age"].round().astype(int)

    return df


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].apply(standardize_gender)

    if "Symptoms" in df.columns:
        df["Symptoms"] = df["Symptoms"].apply(standardize_symptoms)
        df["Symptom_Count"] = df["Symptoms"].apply(count_symptoms).astype(int)

    if "Disease" in df.columns:
        df["Disease"] = df["Disease"].apply(normalize_text)
        disease_mode = df["Disease"].mode(dropna=True)
        fill_value = disease_mode.iloc[0] if not disease_mode.empty else "Unknown"
        df["Disease"] = df["Disease"].fillna(fill_value)

    return df


def rebuild_patient_ids(df: pd.DataFrame) -> pd.DataFrame:
    if "Patient_ID" not in df.columns:
        return df

    missing_ids = df["Patient_ID"].isna()
    if missing_ids.any():
        start_id = int(df["Patient_ID"].max()) + 1 if df["Patient_ID"].notna().any() else 1
        df.loc[missing_ids, "Patient_ID"] = range(start_id, start_id + int(missing_ids.sum()))

    df["Patient_ID"] = df["Patient_ID"].round().astype(int)
    return df


def add_encoded_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "Gender" in df.columns:
        gender_code_map = {"Female": 0, "Male": 1, "Other": 2}
        df["Gender_Encoded"] = df["Gender"].map(gender_code_map).fillna(2).astype(int)
    return df


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    remaining_columns = [column for column in df.columns if column not in ORDERED_COLUMNS]
    final_columns = [column for column in ORDERED_COLUMNS if column in df.columns] + remaining_columns
    return df[final_columns]


def preprocess_medical_data(
    input_file: Path = INPUT_FILE,
    output_file: Path = OUTPUT_FILE,
) -> tuple[pd.DataFrame, PreprocessingReport]:
    df = load_dataset(input_file)
    input_rows = len(df)

    print(f"Loaded raw dataset: {input_file.name}")
    print("Original shape:", df.shape)
    print("Original columns:", list(df.columns))

    df = clean_numeric_columns(df)
    df = clean_text_columns(df)
    df = rebuild_patient_ids(df)
    df = add_encoded_columns(df)

    rows_before_dedup = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    duplicates_removed = rows_before_dedup - len(df)

    df = reorder_columns(df)
    saved_file = save_csv_with_fallback(df, output_file)

    report = PreprocessingReport(
        input_rows=input_rows,
        output_rows=len(df),
        duplicates_removed=duplicates_removed,
        saved_file=saved_file,
    )

    print("\nCleaned shape:", df.shape)
    print(f"Duplicates removed: {report.duplicates_removed}")
    print("\nMissing values after preprocessing:\n", df.isna().sum().to_string())
    print("\nFinal dtypes:\n", df.dtypes.to_string())
    print("\nSample output:")
    print(df.head())
    print(f"\nSaved cleaned file as: {report.saved_file}")

    return df, report


def main() -> None:
    preprocess_medical_data()


if __name__ == "__main__":
    main()
