import re
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
INPUT_FILE = PROJECT_ROOT / "data" / "ml" / "Healthcare_Cleaned.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "ml" / "Healthcare_FeatureEngineered.csv"
TOP_SYMPTOM_COUNT = 15
TOP_PAIR_COUNT = 10


@dataclass
class FeatureEngineeringReport:
    input_rows: int
    output_rows: int
    total_columns: int
    saved_file: Path


def split_symptoms(text: object) -> list[str]:
    if pd.isna(text):
        return []

    normalized = str(text).strip().lower()
    if not normalized or normalized == "unknown":
        return []

    return [part.strip() for part in re.split(r"[,;|]+", normalized) if part.strip()]


def make_safe_name(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", text.lower().replace(" ", "_")).strip("_")


def bmi_category(value: float) -> str:
    if value < 18.5:
        return "Underweight"
    if value < 25:
        return "Normal"
    if value < 30:
        return "Overweight"
    return "Obese"


def glucose_category(value: float) -> str:
    if value < 70:
        return "Low"
    if value < 100:
        return "Normal"
    if value < 126:
        return "Prediabetes"
    return "High"


def blood_pressure_category(value: float) -> str:
    if value < 60:
        return "Low"
    if value < 80:
        return "Normal"
    if value < 90:
        return "Prehypertension"
    return "High"


def symptom_load_group(count: int) -> str:
    if count <= 1:
        return "Minimal"
    if count <= 3:
        return "Moderate"
    if count <= 5:
        return "High"
    return "Severe"


def risk_group(score: int) -> str:
    if score <= 2:
        return "Low"
    if score <= 4:
        return "Moderate"
    if score <= 6:
        return "High"
    return "Very_High"


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


def add_symptom_combination_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Symptoms_List"] = df["Symptoms"].apply(split_symptoms)
    df["Symptom_Count"] = df["Symptoms_List"].apply(len).astype(int)
    df["Unique_Symptom_Count"] = df["Symptoms_List"].apply(lambda symptoms: len(set(symptoms))).astype(int)
    df["Symptom_Load_Group"] = df["Symptom_Count"].apply(symptom_load_group)

    all_symptoms = [symptom for symptoms in df["Symptoms_List"] for symptom in symptoms]
    if all_symptoms:
        top_symptoms = pd.Series(all_symptoms).value_counts().head(TOP_SYMPTOM_COUNT).index.tolist()
    else:
        top_symptoms = []

    for symptom in top_symptoms:
        column_name = f"SYM_{make_safe_name(symptom)}"
        df[column_name] = df["Symptoms_List"].apply(lambda symptoms, symptom=symptom: int(symptom in symptoms))

    pair_counter: Counter[tuple[str, str]] = Counter()
    for symptoms in df["Symptoms_List"]:
        pair_counter.update(combinations(sorted(set(symptoms)), 2))

    top_pairs = [pair for pair, _ in pair_counter.most_common(TOP_PAIR_COUNT)]
    for first_symptom, second_symptom in top_pairs:
        column_name = f"PAIR_{make_safe_name(first_symptom)}_{make_safe_name(second_symptom)}"
        df[column_name] = df["Symptoms_List"].apply(
            lambda symptoms, pair=(first_symptom, second_symptom): int(set(pair).issubset(symptoms))
        )

    df["Multi_System_Symptoms"] = (df["Unique_Symptom_Count"] >= 4).astype(int)
    return df


def add_severity_indicator_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Age_Group"] = pd.cut(
        df["Age"],
        bins=[0, 18, 35, 50, 65, 120],
        labels=["Child", "Young", "Adult", "Middle_Age", "Senior"],
        include_lowest=True,
    )
    df["BMI_Category"] = df["BMI"].apply(bmi_category)
    df["Glucose_Category"] = df["Glucose"].apply(glucose_category)
    df["BP_Category"] = df["BloodPressure"].apply(blood_pressure_category)

    bmi_score_map = {"Underweight": 1, "Normal": 0, "Overweight": 1, "Obese": 2}
    glucose_score_map = {"Low": 1, "Normal": 0, "Prediabetes": 1, "High": 2}
    bp_score_map = {"Low": 1, "Normal": 0, "Prehypertension": 1, "High": 2}
    symptom_score_map = {"Minimal": 0, "Moderate": 1, "High": 2, "Severe": 3}

    df["BMI_Severity_Score"] = df["BMI_Category"].map(bmi_score_map).astype(int)
    df["Glucose_Severity_Score"] = df["Glucose_Category"].map(glucose_score_map).astype(int)
    df["BP_Severity_Score"] = df["BP_Category"].map(bp_score_map).astype(int)
    df["Symptom_Severity_Score"] = df["Symptom_Load_Group"].map(symptom_score_map).astype(int)
    df["Vital_Severity_Score"] = (
        df["BMI_Severity_Score"] + df["Glucose_Severity_Score"] + df["BP_Severity_Score"]
    ).astype(int)
    df["Overall_Severity_Score"] = (df["Vital_Severity_Score"] + df["Symptom_Severity_Score"]).astype(int)
    df["Critical_Vitals_Flag"] = (df["Vital_Severity_Score"] >= 4).astype(int)
    return df


def add_risk_factor_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Senior_Risk"] = (df["Age"] >= 60).astype(int)
    df["Pediatric_Risk"] = (df["Age"] <= 12).astype(int)
    df["High_Symptom_Load"] = (df["Symptom_Count"] >= 4).astype(int)
    df["High_Glucose_Risk"] = (df["Glucose"] >= 126).astype(int)
    df["High_BP_Risk"] = (df["BloodPressure"] >= 90).astype(int)
    df["Obesity_Risk"] = (df["BMI"] >= 30).astype(int)

    df["Metabolic_Risk_Count"] = (
        df[["High_Glucose_Risk", "High_BP_Risk", "Obesity_Risk"]].sum(axis=1)
    ).astype(int)
    df["Risk_Score"] = (
        df[
            [
                "Senior_Risk",
                "Pediatric_Risk",
                "High_Symptom_Load",
                "High_Glucose_Risk",
                "High_BP_Risk",
                "Obesity_Risk",
                "Critical_Vitals_Flag",
            ]
        ].sum(axis=1)
    ).astype(int)
    df["Risk_Factor_Group"] = df["Risk_Score"].apply(risk_group)
    return df


def add_historical_pattern_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Symptom_Profile_Frequency"] = df.groupby("Symptoms")["Symptoms"].transform("size").astype(int)
    df["Age_Group_Profile_Frequency"] = (
        df.groupby(["Age_Group", "Symptoms"], observed=False)["Symptoms"].transform("size").astype(int)
    )
    df["Gender_Profile_Frequency"] = df.groupby(["Gender", "Symptoms"])["Symptoms"].transform("size").astype(int)
    df["Repeated_Symptom_Pattern"] = (df["Symptom_Profile_Frequency"] >= 3).astype(int)
    df["Recurring_Demographic_Pattern"] = (
        (df["Age_Group_Profile_Frequency"] >= 2) | (df["Gender_Profile_Frequency"] >= 2)
    ).astype(int)
    df["Historical_Pattern_Score"] = (
        df["Repeated_Symptom_Pattern"] + df["Recurring_Demographic_Pattern"]
    ).astype(int)
    return df


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    base_columns = [
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
    feature_priority = [
        "Unique_Symptom_Count",
        "Symptom_Load_Group",
        "Multi_System_Symptoms",
        "Age_Group",
        "BMI_Category",
        "Glucose_Category",
        "BP_Category",
        "BMI_Severity_Score",
        "Glucose_Severity_Score",
        "BP_Severity_Score",
        "Symptom_Severity_Score",
        "Vital_Severity_Score",
        "Overall_Severity_Score",
        "Critical_Vitals_Flag",
        "Senior_Risk",
        "Pediatric_Risk",
        "High_Symptom_Load",
        "High_Glucose_Risk",
        "High_BP_Risk",
        "Obesity_Risk",
        "Metabolic_Risk_Count",
        "Risk_Score",
        "Risk_Factor_Group",
        "Symptom_Profile_Frequency",
        "Age_Group_Profile_Frequency",
        "Gender_Profile_Frequency",
        "Repeated_Symptom_Pattern",
        "Recurring_Demographic_Pattern",
        "Historical_Pattern_Score",
    ]

    other_columns = [
        column for column in df.columns if column not in base_columns + feature_priority + ["Symptoms_List"]
    ]
    final_columns = [column for column in base_columns if column in df.columns]
    final_columns += [column for column in feature_priority if column in df.columns]
    final_columns += other_columns
    return df[final_columns]


def feature_engineering(
    input_file: Path = INPUT_FILE,
    output_file: Path = OUTPUT_FILE,
) -> tuple[pd.DataFrame, FeatureEngineeringReport]:
    df = load_dataset(input_file)
    input_rows = len(df)

    print(f"Loaded cleaned dataset: {input_file.name}")
    print("Input shape:", df.shape)

    df = add_symptom_combination_features(df)
    df = add_severity_indicator_features(df)
    df = add_risk_factor_features(df)
    df = add_historical_pattern_features(df)

    df = df.drop(columns=["Symptoms_List"], errors="ignore")
    df = reorder_columns(df)

    saved_file = save_csv_with_fallback(df, output_file)
    report = FeatureEngineeringReport(
        input_rows=input_rows,
        output_rows=len(df),
        total_columns=len(df.columns),
        saved_file=saved_file,
    )

    print("\nFeature groups created:")
    print("- Symptom combinations")
    print("- Severity indicators")
    print("- Risk-factor groupings")
    print("- Historical illness patterns")
    print("\nOutput shape:", df.shape)
    print(f"Saved feature-engineered file as: {report.saved_file}")

    return df, report


def main() -> None:
    featured_df, report = feature_engineering()
    print("\nSample columns:")
    print(featured_df.columns[:40].tolist())
    print(f"Total derived dataset columns: {report.total_columns}")


if __name__ == "__main__":
    main()
