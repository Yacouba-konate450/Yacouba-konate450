"""
analysis.py
------------
End-to-end academic performance pipeline:
  1. Load a messy, real-world-style dataset (missing values, duplicates,
     inconsistent formatting)
  2. Clean and standardise it
  3. Explore relationships between study habits and final GPA
  4. Train a regression model to predict final GPA
  5. Export cleaned data + charts for a Power BI dashboard

Run from the project root:
    python src/analysis.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder

RAW_PATH = "data/students_raw.csv"
OUTPUT_DIR = "outputs"

sns.set_style("whitegrid")


# ---------------------------------------------------------------------------
# 1. Cleaning
# ---------------------------------------------------------------------------

def clean_data(df):
    df = df.copy()

    # Standardise text columns: strip whitespace, fix casing
    df["major"] = df["major"].str.strip().str.title()
    df["gender"] = df["gender"].str.strip().str.upper().replace({
        "MALE": "M", "FEMALE": "F"
    })

    # Drop exact duplicate rows (data export error)
    before = len(df)
    df = df.drop_duplicates(subset="student_id", keep="first")
    print(f"Removed {before - len(df)} duplicate student records.")

    # Handle missing values: impute with median (robust to outliers)
    for col in ["attendance_rate", "study_hours_per_week", "prior_gpa"]:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"Imputed {missing_count} missing values in '{col}' with median ({median_val:.2f}).")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Exploratory analysis
# ---------------------------------------------------------------------------

def exploratory_charts(df):
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Study hours vs final GPA
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=df, x="study_hours_per_week", y="final_gpa",
                     alpha=0.4, color="#0F6E56")
    sns.regplot(data=df, x="study_hours_per_week", y="final_gpa",
                scatter=False, color="#A32D2D")
    plt.title("Study Hours per Week vs Final GPA")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/study_hours_vs_gpa.png", dpi=150)
    plt.close()

    # Attendance vs final GPA
    plt.figure(figsize=(6, 5))
    sns.scatterplot(data=df, x="attendance_rate", y="final_gpa",
                     alpha=0.4, color="#185FA5")
    sns.regplot(data=df, x="attendance_rate", y="final_gpa",
                scatter=False, color="#A32D2D")
    plt.title("Attendance Rate (%) vs Final GPA")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/attendance_vs_gpa.png", dpi=150)
    plt.close()

    # Average GPA by major
    plt.figure(figsize=(7, 5))
    avg_by_major = df.groupby("major")["final_gpa"].mean().sort_values()
    avg_by_major.plot(kind="barh", color="#0F6E56")
    plt.title("Average Final GPA by Major")
    plt.xlabel("Average GPA")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/gpa_by_major.png", dpi=150)
    plt.close()

    print(f"Exploratory charts saved to {OUTPUT_DIR}/")


# ---------------------------------------------------------------------------
# 3. Predictive model
# ---------------------------------------------------------------------------

def train_predictive_model(df):
    features = [
        "study_hours_per_week", "attendance_rate", "prior_gpa",
        "extracurricular_activities", "part_time_job"
    ]
    X = df[features]
    y = df["final_gpa"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Baseline: linear regression
    lin_model = LinearRegression()
    lin_model.fit(X_train, y_train)
    lin_pred = lin_model.predict(X_test)

    # Random forest for comparison
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    print("\n--- Model comparison ---")
    print(f"Linear Regression — MAE: {mean_absolute_error(y_test, lin_pred):.3f}, "
          f"R²: {r2_score(y_test, lin_pred):.3f}")
    print(f"Random Forest     — MAE: {mean_absolute_error(y_test, rf_pred):.3f}, "
          f"R²: {r2_score(y_test, rf_pred):.3f}")

    return rf_model, features, X_test, y_test, rf_pred


def save_prediction_chart(y_test, y_pred):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.4, color="#0F6E56")
    plt.plot([0, 4], [0, 4], linestyle="--", color="gray")
    plt.xlabel("Actual GPA")
    plt.ylabel("Predicted GPA")
    plt.title("Predicted vs Actual Final GPA (Random Forest)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/predicted_vs_actual.png", dpi=150)
    plt.close()


def export_for_powerbi(df):
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(f"{OUTPUT_DIR}/students_clean.csv", index=False)
    print(f"Cleaned dataset exported to {OUTPUT_DIR}/students_clean.csv (ready for Power BI import)")


if __name__ == "__main__":
    print("Loading raw data...")
    raw_df = pd.read_csv(RAW_PATH)
    print(f"Raw dataset: {len(raw_df)} rows, {raw_df.isna().sum().sum()} missing values")

    print("\nCleaning data...")
    clean_df = clean_data(raw_df)

    print("\nGenerating exploratory charts...")
    exploratory_charts(clean_df)

    print("\nTraining predictive model...")
    model, features, X_test, y_test, y_pred = train_predictive_model(clean_df)
    save_prediction_chart(y_test, y_pred)

    print("\nExporting cleaned data for Power BI...")
    export_for_powerbi(clean_df)

    print("\nDone. See the outputs/ folder for charts and the cleaned CSV.")
