# ======================================================
# STEP 5 & 6: Model Selection, Training & Evaluation
# Accuracy shown in Percentage (%)
# ======================================================

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------------------------------------
# 1️⃣ Load Dataset
# ------------------------------------------------------

df = pd.read_csv("Healthcare_FeatureEngineered.csv")
print("Dataset Loaded:", df.shape)

# ------------------------------------------------------
# 2️⃣ Create Risk_Level if missing
# ------------------------------------------------------

if "Risk_Level" not in df.columns:

    def risk_category(row):
        score = 0
        if row["Age"] > 60:
            score += 2
        elif row["Age"] > 40:
            score += 1

        if row["Symptom_Count"] >= 6:
            score += 2
        elif row["Symptom_Count"] >= 3:
            score += 1

        if score <= 2:
            return "Low"
        elif score <= 4:
            return "Medium"
        else:
            return "High"

    df["Risk_Level"] = df.apply(risk_category, axis=1)

# ------------------------------------------------------
# 3️⃣ Features & Target
# ------------------------------------------------------

y = df["Risk_Level"]
X = df.drop(columns=["Risk_Level", "Disease", "Patient_ID"], errors="ignore")

# ------------------------------------------------------
# 4️⃣ Encode Target
# ------------------------------------------------------

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
joblib.dump(label_encoder, "label_encoder.pkl")

# ------------------------------------------------------
# 5️⃣ Identify Column Types
# ------------------------------------------------------

cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns
num_cols = X.select_dtypes(include=["number"]).columns

# ------------------------------------------------------
# 6️⃣ Preprocessing Pipeline
# ------------------------------------------------------

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)

# ------------------------------------------------------
# 7️⃣ Train-Test Split
# ------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\nTrain size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

# ------------------------------------------------------
# 8️⃣ Define Models
# ------------------------------------------------------

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

best_model = None
best_accuracy = 0
best_model_name = ""

# ------------------------------------------------------
# 9️⃣ Train & Evaluate Models
# ------------------------------------------------------

for name, model in models.items():

    print(f"\n==============================")
    print(f"Training {name}...")

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    acc_percent = acc * 100

    # Manual Formula
    correct = (y_test == y_pred).sum()
    total = len(y_test)

    print(f"{name} Accuracy Formula: {correct} / {total}")
    print(f"{name} Accuracy: {acc_percent:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save individual model
    filename = name.lower().replace(" ", "_") + "_model.pkl"
    joblib.dump(pipeline, filename)
    print(f"Saved: {filename}")

    # Track best model
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = pipeline
        best_model_name = name

# ------------------------------------------------------
# 🔟 Save Best Model
# ------------------------------------------------------

joblib.dump(best_model, "best_model.pkl")

print("\n==============================")
print(f"Best Model: {best_model_name}")
print(f"Best Accuracy: {best_accuracy * 100:.2f}%")
print("Saved: best_model.pkl")
print("==============================")
