import pandas as pd
import numpy as np
import joblib
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ModuleNotFoundError:
    HAS_PLOTTING = False
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore") 

#  Load Dataset
df = pd.read_csv("Healthcare_FeatureEngineered.csv")
print("Dataset Loaded:", df.shape)

#  Create Risk_Level if missing
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

#  Features & Target

y = df["Risk_Level"]
X = df.drop(columns=["Risk_Level", "Disease", "Patient_ID"], errors="ignore")

#  Encode Target

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
joblib.dump(label_encoder, "label_encoder.pkl")

#  Identify Column Types
cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns
num_cols = X.select_dtypes(include=["number"]).columns

#  Preprocessing Pipeline


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

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

print("\nTrain size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

#  Define Models

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

best_model = None
best_accuracy = 0
best_model_name = ""
model_accuracies = {}
conf_matrices = {}

#  Train & Evaluate Models

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
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    conf_matrices[name] = cm
    model_accuracies[name] = acc_percent

    # Save individual model
    filename = name.lower().replace(" ", "_") + "_model.pkl"
    joblib.dump(pipeline, filename)
    print(f"Saved: {filename}")

    # Track best model
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = pipeline
        best_model_name = name

#  Save Best Model

joblib.dump(best_model, "best_model.pkl")

print("\n==============================")
print(f"Best Model: {best_model_name}")
print(f"Best Accuracy: {best_accuracy * 100:.2f}%")
print("Saved: best_model.pkl")
print("==============================")

#  Graphical Representation
if HAS_PLOTTING:
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(model_accuracies.keys()), y=list(model_accuracies.values()), palette="viridis")
    plt.title("Model Accuracy Comparison")
    plt.xlabel("Models")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("model_accuracy_comparison.png", dpi=300)
    plt.show()

    class_names = label_encoder.classes_
    for model_name, cm in conf_matrices.items():
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        output_name = model_name.lower().replace(" ", "_") + "_confusion_matrix.png"
        plt.savefig(output_name, dpi=300)
        plt.show()
else:
    print(
        "Plotting skipped: install packages with "
        "\"python -m pip install matplotlib seaborn\""
    )
