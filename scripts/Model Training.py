from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
INPUT_FILE = PROJECT_ROOT / "data" / "ml" / "Healthcare_FeatureEngineered.csv"
TARGET_COLUMN = "Risk_Factor_Group"
MODEL_NAME = "Logistic Regression"
MODEL_APPROACH = "Leakage-Safe Risk Stratification"
MODEL_OUTPUT = PROJECT_ROOT / "models" / "trained_risk_stratification_model.pkl"
SUMMARY_OUTPUT = PROJECT_ROOT / "reports" / "risk_model_evaluation_summary.csv"
CONFUSION_OUTPUT = PROJECT_ROOT / "reports" / "risk_model_confusion_matrix.csv"
ACCURACY_GRAPH_OUTPUT = PROJECT_ROOT / "reports" / "risk_model_accuracy_comparison.svg"
CONFUSION_GRAPH_OUTPUT = PROJECT_ROOT / "reports" / "risk_model_confusion_matrix.svg"
RANDOM_STATE = 42

# These columns directly encode the target-building rule and would make evaluation misleading.
DROP_COLUMNS = [
    "Disease",
    "Patient_ID",
    "Symptoms",
    "Risk_Factor_Group",
    "Risk_Score",
    "Senior_Risk",
    "Pediatric_Risk",
    "High_Symptom_Load",
    "High_Glucose_Risk",
    "High_BP_Risk",
    "Obesity_Risk",
    "Critical_Vitals_Flag",
    "Metabolic_Risk_Count",
]


@dataclass
class DatasetSplit:
    X_train: pd.DataFrame
    X_validation: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_validation: pd.Series
    y_test: pd.Series


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


def save_text_with_fallback(text: str, output_file: Path) -> Path:
    try:
        output_file.write_text(text, encoding="utf-8")
        return output_file
    except PermissionError:
        for counter in range(1, 100):
            fallback_file = output_file.with_name(
                f"{output_file.stem}_new_{counter}{output_file.suffix}"
            )
            try:
                fallback_file.write_text(text, encoding="utf-8")
                print(
                    f"\n'{output_file.name}' is open or locked. "
                    f"Saved the graph as '{fallback_file.name}' instead."
                )
                return fallback_file
            except PermissionError:
                continue
        raise


def save_model_with_fallback(obj: object, output_file: Path) -> Path:
    try:
        joblib.dump(obj, output_file)
        return output_file
    except PermissionError:
        for counter in range(1, 100):
            fallback_file = output_file.with_name(
                f"{output_file.stem}_new_{counter}{output_file.suffix}"
            )
            try:
                joblib.dump(obj, fallback_file)
                print(
                    f"\n'{output_file.name}' is open or locked. "
                    f"Saved the trained model as '{fallback_file.name}' instead."
                )
                return fallback_file
            except PermissionError:
                continue
        raise


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical_columns = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    numeric_columns = [column for column in X.columns if column not in categorical_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )


def build_model() -> LogisticRegression:
    return LogisticRegression(
        solver="saga",
        C=1.0,
        max_iter=5000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def prepare_dataset(input_file: Path, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(input_file)
    original_rows = len(df)
    df = df.dropna(subset=[target_column]).copy()
    dropped_rows = original_rows - len(df)

    print(f"Feature-engineered dataset loaded from: {input_file}")
    print("Dataset shape:", df.shape)
    print(f"Rows dropped due to missing target values: {dropped_rows}")
    print(f"Target column: {target_column}")
    print("Target distribution:")
    print(df[target_column].value_counts().to_string())

    X = df.drop(columns=DROP_COLUMNS, errors="ignore")
    y = df[target_column]
    return X, y


def create_splits(X: pd.DataFrame, y: pd.Series) -> DatasetSplit:
    X_dev, X_test, y_dev, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_dev,
        y_dev,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_dev,
    )
    return DatasetSplit(
        X_train=X_train,
        X_validation=X_validation,
        X_test=X_test,
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
    )


def metric_bundle(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float | int]:
    correct_predictions = int((y_pred == y_true).sum())
    total_predictions = int(len(y_true))
    missed_cases = total_predictions - correct_predictions
    return {
        "Accuracy": round(correct_predictions / total_predictions, 4),
        "Precision": round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "Correct_Predictions": correct_predictions,
        "Total_Predictions": total_predictions,
        "Missed_Cases": missed_cases,
    }


def build_pipeline(preprocessor: ColumnTransformer) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", build_model()),
        ]
    )


def build_accuracy_svg(summary_df: pd.DataFrame) -> str:
    width = 760
    height = 460
    left = 90
    bottom = 70
    top = 40
    plot_height = height - top - bottom
    bar_width = 140
    gap = 40
    max_value = max(0.1, float(summary_df[["Train_Accuracy", "Validation_Accuracy", "Test_Accuracy"]].max().max()))
    labels = ["Train_Accuracy", "Validation_Accuracy", "Test_Accuracy"]
    colors = ["#7a9e7e", "#2f6fed", "#f28e2b"]
    x_positions = [150, 330, 510]

    bars = []
    texts = []
    for label, color, x in zip(labels, colors, x_positions):
        value = float(summary_df[label].iloc[0])
        bar_height = (value / max_value) * (plot_height - 20)
        y = top + plot_height - bar_height
        bars.append(
            f"<rect x='{x}' y='{y:.1f}' width='{bar_width}' height='{bar_height:.1f}' fill='{color}' rx='8' />"
        )
        texts.append(
            f"<text x='{x + bar_width / 2}' y='{height - 30}' text-anchor='middle' font-size='14' fill='#222'>{label.replace('_', ' ')}</text>"
        )
        texts.append(
            f"<text x='{x + bar_width / 2}' y='{max(y - 10, 20):.1f}' text-anchor='middle' font-size='14' fill='#222'>{value:.4f}</text>"
        )

    grid_lines = []
    axis_labels = []
    for tick in range(6):
        value = max_value * tick / 5
        tick_y = top + plot_height - ((plot_height - 20) * tick / 5)
        grid_lines.append(
            f"<line x1='{left - 10}' y1='{tick_y:.1f}' x2='{width - 40}' y2='{tick_y:.1f}' stroke='#d9d9d9' />"
        )
        axis_labels.append(
            f"<text x='{left - 18}' y='{tick_y + 5:.1f}' text-anchor='end' font-size='12' fill='#555'>{value:.2f}</text>"
        )

    return f"""<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>
<rect width='100%' height='100%' fill='white' />
<text x='{width / 2}' y='24' text-anchor='middle' font-size='22' font-weight='bold' fill='#222'>
Risk Stratification Accuracy Summary
</text>
<line x1='{left}' y1='{top}' x2='{left}' y2='{top + plot_height}' stroke='#333' stroke-width='2' />
<line x1='{left}' y1='{top + plot_height}' x2='{width - 40}' y2='{top + plot_height}' stroke='#333' stroke-width='2' />
{''.join(grid_lines)}
{''.join(axis_labels)}
{''.join(bars)}
{''.join(texts)}
</svg>"""


def build_confusion_svg(confusion_df: pd.DataFrame, labels: list[str], title: str) -> str:
    cell_size = 110
    margin_left = 180
    margin_top = 110
    width = margin_left + cell_size * len(labels) + 80
    height = margin_top + cell_size * len(labels) + 120
    max_value = max(1, int(confusion_df.to_numpy().max()))

    cells = []
    row_labels = []
    col_labels = []
    values = []

    for row_index, actual_label in enumerate(labels):
        row_y = margin_top + row_index * cell_size
        row_labels.append(
            f"<text x='{margin_left - 15}' y='{row_y + 60}' text-anchor='end' font-size='14' fill='#222'>{actual_label}</text>"
        )
        for col_index, predicted_label in enumerate(labels):
            col_x = margin_left + col_index * cell_size
            value = int(confusion_df.loc[actual_label, predicted_label])
            shade = 245 - int((value / max_value) * 170)
            fill = f"rgb({shade},{shade},{255 if shade < 245 else 245})"
            cells.append(
                f"<rect x='{col_x}' y='{row_y}' width='{cell_size}' height='{cell_size}' fill='{fill}' stroke='#cfcfcf' />"
            )
            values.append(
                f"<text x='{col_x + cell_size / 2}' y='{row_y + 60}' text-anchor='middle' font-size='18' fill='#111'>{value}</text>"
            )

    for col_index, predicted_label in enumerate(labels):
        col_x = margin_left + col_index * cell_size + cell_size / 2
        col_labels.append(
            f"<text x='{col_x}' y='{margin_top - 18}' text-anchor='middle' font-size='14' fill='#222'>{predicted_label}</text>"
        )

    return f"""<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>
<rect width='100%' height='100%' fill='white' />
<text x='{width / 2}' y='30' text-anchor='middle' font-size='20' font-weight='bold' fill='#222'>
{title}
</text>
<text x='{width / 2}' y='{margin_top - 55}' text-anchor='middle' font-size='12' fill='#444'>Predicted Label</text>
<text x='28' y='{margin_top + (cell_size * len(labels)) / 2}' text-anchor='middle' font-size='12' fill='#444'
 transform='rotate(-90 28 {margin_top + (cell_size * len(labels)) / 2})'>Actual Label</text>
{''.join(cells)}
{''.join(values)}
{''.join(row_labels)}
{''.join(col_labels)}
</svg>"""


def train_model(input_file: Path = INPUT_FILE, target_column: str = TARGET_COLUMN) -> None:
    X, y = prepare_dataset(input_file, target_column)
    splits = create_splits(X, y)

    print(f"\nTraining samples: {len(splits.X_train)}")
    print(f"Validation samples: {len(splits.X_validation)}")
    print(f"Test samples: {len(splits.X_test)}")
    print(f"Selected model: {MODEL_NAME}")
    print(f"Approach: {MODEL_APPROACH}")
    print("Target goal: predict Low / Moderate / High risk from patient inputs.")

    preprocessor = build_preprocessor(X)
    pipeline = build_pipeline(preprocessor)
    pipeline.fit(splits.X_train, splits.y_train)

    train_predictions = pipeline.predict(splits.X_train)
    validation_predictions = pipeline.predict(splits.X_validation)

    train_metrics = metric_bundle(splits.y_train, train_predictions)
    validation_metrics = metric_bundle(splits.y_validation, validation_predictions)

    X_dev = pd.concat([splits.X_train, splits.X_validation], axis=0)
    y_dev = pd.concat([splits.y_train, splits.y_validation], axis=0)
    final_pipeline = build_pipeline(preprocessor)
    final_pipeline.fit(X_dev, y_dev)

    test_predictions = final_pipeline.predict(splits.X_test)
    test_metrics = metric_bundle(splits.y_test, test_predictions)
    labels = sorted(splits.y_test.unique())
    confusion_df = pd.DataFrame(
        confusion_matrix(splits.y_test, test_predictions, labels=labels),
        index=labels,
        columns=labels,
    )
    confusion_df.index.name = "Actual"
    report_text = classification_report(splits.y_test, test_predictions, zero_division=0)

    summary_df = pd.DataFrame(
        [
            {
                "Model": MODEL_NAME,
                "Approach": MODEL_APPROACH,
                "Target": TARGET_COLUMN,
                "Train_Accuracy": train_metrics["Accuracy"],
                "Validation_Accuracy": validation_metrics["Accuracy"],
                "Validation_Precision": validation_metrics["Precision"],
                "Validation_Recall": validation_metrics["Recall"],
                "Validation_Missed_Cases": validation_metrics["Missed_Cases"],
                "Test_Accuracy": test_metrics["Accuracy"],
                "Test_Precision": test_metrics["Precision"],
                "Test_Recall": test_metrics["Recall"],
                "Test_Missed_Cases": test_metrics["Missed_Cases"],
            }
        ]
    )

    saved_summary = save_csv_with_fallback(summary_df, SUMMARY_OUTPUT)
    saved_confusion = save_csv_with_fallback(confusion_df.reset_index(), CONFUSION_OUTPUT)
    saved_model = save_model_with_fallback(final_pipeline, MODEL_OUTPUT)
    saved_accuracy_graph = save_text_with_fallback(build_accuracy_svg(summary_df), ACCURACY_GRAPH_OUTPUT)
    saved_confusion_graph = save_text_with_fallback(
        build_confusion_svg(confusion_df, labels, f"{MODEL_NAME} Risk Confusion Matrix"),
        CONFUSION_GRAPH_OUTPUT,
    )

    summary_print_df = summary_df.copy()
    for column in [
        "Train_Accuracy",
        "Validation_Accuracy",
        "Validation_Precision",
        "Validation_Recall",
        "Test_Accuracy",
        "Test_Precision",
        "Test_Recall",
    ]:
        summary_print_df[column] = summary_print_df[column].apply(lambda value: f"{value * 100:.2f}%")

    print("\nModel Evaluation Summary:")
    print(summary_print_df.to_string(index=False))
    print("\nFinal Classification Report:")
    print(report_text)
    print("Final Confusion Matrix:")
    print(confusion_df.to_numpy())
    print(f"\nSaved evaluation summary: {saved_summary}")
    print(f"Saved confusion matrix: {saved_confusion}")
    print(f"Saved trained model: {saved_model}")
    print(f"Saved accuracy graph: {saved_accuracy_graph}")
    print(f"Saved confusion matrix graph: {saved_confusion_graph}")


def main() -> None:
    train_model()


if __name__ == "__main__":
    main()
