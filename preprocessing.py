import pandas as pd  
import numpy as np  
import re     
import warnings
warnings.filterwarnings("ignore")

# Load the dataset

file_path = "Healthcare.csv"   
df = pd.read_csv(file_path)

print(" Original shape:", df.shape)
print("\n Original dtypes:\n", df.dtypes)

# Basic cleaning (preserve missing values as NaN/<NA>)
if "Patient_ID" in df.columns:
    df["Patient_ID"] = df["Patient_ID"].astype("string").str.strip()
    
    if df["Patient_ID"].dropna().str.fullmatch(r"\d+").all():
        df["Patient_ID"] = df["Patient_ID"].astype("Int64")

if "Age" in df.columns:
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce").round().astype("Int64")

if "Symptoms" in df.columns:
    df["Symptoms"] = df["Symptoms"].astype("string").str.strip()

if "Disease" in df.columns:
    df["Disease"] = df["Disease"].astype("string").str.strip()

#before encoding 
if "Gender" in df.columns:
    df["Gender"] = df["Gender"].astype("string").str.strip()

df = df.drop_duplicates()

if "Gender" in df.columns:
    df["Gender"] = df["Gender"].astype("string").str.strip().str.lower()
    gender_map = {"male": 1, "female": 0, "m": 1, "f": 0}
    df["Gender"] = df["Gender"].map(gender_map).fillna(-1).astype("Int64")


# Symptom_Count (only if needed / correct it)
if "Symptoms" in df.columns:
    def count_symptoms(x: str):
        if pd.isna(x):
            return pd.NA
        x = str(x).strip()
        if x == "":
            return pd.NA
        parts = [p.strip() for p in re.split(r"[,;|]+", x) if p.strip()]
        return len(parts)

    df["Symptom_Count"] = df["Symptoms"].apply(count_symptoms).astype("Int64")

# Show preprocessing output (like you wanted)

print("\n Cleaned shape:", df.shape)

print("\n Missing values after preprocessing (output style like you asked):")
missing_report = df.isna().sum().sort_values(ascending=True)
for col, cnt in missing_report.items():
    print(f"{col:<15} {cnt}")

print("\n Final dtypes:\n", df.dtypes)

print("\n Sample output (first 5 rows):")
print(df.head())

# Save cleaned dataset

out_path = "Healthcare_Cleaned.csv"
df.to_csv(out_path, index=False)
print(f"\n Saved cleaned file as: {out_path}")
