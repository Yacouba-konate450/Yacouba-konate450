"""
generate_data.py
-----------------
Generates a realistic, intentionally messy synthetic dataset of student
academic records - including missing values, inconsistent formatting, and
duplicate rows - to practice real-world data cleaning before prediction.

Run:
    python generate_data.py
Output:
    students_raw.csv (in this same folder)
"""

import numpy as np
import pandas as pd

np.random.seed(7)

N_STUDENTS = 1200

MAJORS = ["Computer Engineering", "Business Administration", "Economics",
          "Electrical Engineering", "Mathematics", "Data Science"]

GENDERS = ["M", "F", "m", "f", "Male", "Female"]  # intentionally inconsistent


def generate_students(n):
    study_hours = np.random.gamma(shape=3.0, scale=2.0, size=n)  # hours/week
    attendance_rate = np.clip(np.random.normal(0.82, 0.15, n), 0.2, 1.0)
    prior_gpa = np.clip(np.random.normal(2.8, 0.6, n), 0.5, 4.0)
    extracurricular = np.random.choice([0, 1], size=n, p=[0.55, 0.45])
    part_time_job = np.random.choice([0, 1], size=n, p=[0.65, 0.35])

    # Final GPA depends on study hours, attendance, prior GPA, with noise
    final_gpa = (
        0.35 * prior_gpa
        + 0.04 * study_hours
        + 1.1 * attendance_rate
        - 0.15 * part_time_job
        + 0.05 * extracurricular
        + np.random.normal(0, 0.3, n)
    )
    final_gpa = np.clip(final_gpa, 0.0, 4.0).round(2)

    df = pd.DataFrame({
        "student_id": [f"STU{str(i).zfill(5)}" for i in range(n)],
        "gender": np.random.choice(GENDERS, size=n),
        "major": np.random.choice(MAJORS, size=n),
        "study_hours_per_week": study_hours.round(1),
        "attendance_rate": (attendance_rate * 100).round(1),  # as percentage
        "prior_gpa": prior_gpa.round(2),
        "extracurricular_activities": extracurricular,
        "part_time_job": part_time_job,
        "final_gpa": final_gpa
    })

    return df


def add_realistic_mess(df):
    """Introduce missing values, duplicates and inconsistent formatting -
    the kind of mess a real academic dataset export usually contains."""
    df = df.copy()

    # Missing values in a few columns
    for col in ["attendance_rate", "study_hours_per_week", "prior_gpa"]:
        missing_idx = df.sample(frac=0.04, random_state=1).index
        df.loc[missing_idx, col] = np.nan

    # Some duplicate rows (data export error)
    duplicates = df.sample(frac=0.02, random_state=2)
    df = pd.concat([df, duplicates], ignore_index=True)

    # Inconsistent whitespace / casing in major names
    df.loc[df.sample(frac=0.03, random_state=3).index, "major"] = df["major"].str.upper()
    df.loc[df.sample(frac=0.02, random_state=4).index, "major"] = " " + df["major"] + "  "

    return df.sample(frac=1, random_state=5).reset_index(drop=True)  # shuffle


if __name__ == "__main__":
    df = generate_students(N_STUDENTS)
    df = add_realistic_mess(df)

    df.to_csv("students_raw.csv", index=False)
    print(f"Generated {len(df)} student records (with intentional data-quality issues).")
    print(f"Missing values: {df.isna().sum().sum()}")
    print(f"Saved to students_raw.csv")
