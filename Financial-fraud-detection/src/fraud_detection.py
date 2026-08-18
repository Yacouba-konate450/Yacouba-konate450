"""
fraud_detection.py
--------------------
End-to-end fraud detection pipeline:
  1. Load transaction data
  2. Feature engineering (behavioural + time-based features)
  3. Train a Random Forest classifier on an imbalanced dataset
  4. Evaluate with ROC-AUC, precision/recall, and a confusion matrix
  5. Export results (charts + scored transactions) for a Power BI dashboard

Run from the project root:
    python src/fraud_detection.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report,
    confusion_matrix, precision_recall_curve
)
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "data/transactions.csv"
OUTPUT_DIR = "outputs"

sns.set_style("whitegrid")


def load_data(path):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def engineer_features(df):
    """Create behavioural and time-based features that help distinguish
    fraudulent transactions from normal customer behaviour."""
    df = df.copy()

    # Time-based features
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_night"] = df["hour"].apply(lambda h: 1 if (h < 6 or h > 22) else 0)

    # Behavioural feature: how unusual is this amount vs the customer's average spend?
    df["amount_to_avg_ratio"] = df["amount"] / (df["customer_avg_monthly_spend"] + 1)

    # Location mismatch: is the transaction happening away from the customer's home region?
    df["location_mismatch"] = (df["location"] != df["customer_home_location"]).astype(int)

    # Encode categorical variables
    le_merchant = LabelEncoder()
    le_location = LabelEncoder()
    df["merchant_category_enc"] = le_merchant.fit_transform(df["merchant_category"])
    df["location_enc"] = le_location.fit_transform(df["location"])

    return df


FEATURES = [
    "amount", "hour", "day_of_week", "is_night",
    "amount_to_avg_ratio", "location_mismatch",
    "merchant_category_enc", "location_enc"
]


def train_model(df):
    X = df[FEATURES]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=5,
        class_weight="balanced",  # important: dataset is imbalanced (~2% fraud)
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 5-fold cross-validation on ROC-AUC to check stability of the score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
    print(f"Cross-validated ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return model, X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    print(f"\nTest ROC-AUC: {auc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    return y_pred, y_proba, auc


def save_charts(model, X_test, y_test, y_pred, y_proba):
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc_score(y_test, y_proba):.3f})", color="#0F6E56")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Fraud Detection Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/roc_curve.png", dpi=150)
    plt.close()

    # 2. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=150)
    plt.close()

    # 3. Feature importance
    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values()
    plt.figure(figsize=(7, 5))
    importances.plot(kind="barh", color="#0F6E56")
    plt.title("Feature Importance — What the Model Uses to Detect Fraud")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/feature_importance.png", dpi=150)
    plt.close()

    print(f"\nCharts saved to {OUTPUT_DIR}/")


def export_for_powerbi(df, model):
    """Score the full dataset and export a CSV ready to import into Power BI
    for the interactive fraud-monitoring dashboard."""
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = df.copy()
    df["fraud_probability"] = model.predict_proba(df[FEATURES])[:, 1]
    df["predicted_fraud"] = model.predict(df[FEATURES])

    export_cols = [
        "transaction_id", "customer_id", "timestamp", "amount",
        "merchant_category", "location", "hour", "is_night",
        "location_mismatch", "fraud_probability", "predicted_fraud", "is_fraud"
    ]
    df[export_cols].to_csv(f"{OUTPUT_DIR}/scored_transactions.csv", index=False)
    print(f"Scored transactions exported to {OUTPUT_DIR}/scored_transactions.csv "
          f"(ready for Power BI import)")


if __name__ == "__main__":
    print("Loading data...")
    df = load_data(DATA_PATH)

    print("Engineering features...")
    df = engineer_features(df)

    print("Training model...")
    model, X_train, X_test, y_train, y_test = train_model(df)

    print("Evaluating model...")
    y_pred, y_proba, auc = evaluate_model(model, X_test, y_test)

    print("Saving charts...")
    save_charts(model, X_test, y_test, y_pred, y_proba)

    print("Exporting scored data for Power BI...")
    export_for_powerbi(df, model)

    print("\nDone. See the outputs/ folder for charts and the Power BI-ready CSV.")
