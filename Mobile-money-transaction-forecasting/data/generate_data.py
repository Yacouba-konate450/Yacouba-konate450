"""
generate_data.py
-----------------
Generates a realistic synthetic daily time series of mobile money transaction
volume - with trend, weekly seasonality, monthly (salary-day) spikes, and
noise - mirroring the structure of real telecom/banking transaction data.

This mirrors the kind of series studied in macroeconomic and financial
forecasting (e.g. CPI inflation, transaction volume, deposit flows): a
non-stationary series with trend + seasonality + noise, the classic setting
for comparing statistical, machine learning and deep learning forecasting
methods.

Run:
    python generate_data.py
Output:
    transaction_volume.csv (in this same folder)
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N_DAYS = 730  # 2 years of daily data


def generate_series(n_days):
    dates = pd.date_range(start="2024-01-01", periods=n_days, freq="D")

    t = np.arange(n_days)

    # 1. Long-term upward trend (adoption growth over time)
    trend = 50000 + t * 45

    # 2. Weekly seasonality (higher activity on weekends / market days)
    day_of_week = dates.dayofweek
    weekly_pattern = np.where(day_of_week >= 5, 1.18, 1.0)  # +18% on Sat/Sun

    # 3. Monthly seasonality (salary-day spikes around the 1st and 28th)
    day_of_month = dates.day
    salary_spike = np.where((day_of_month <= 3) | (day_of_month >= 27), 1.35, 1.0)

    # 4. Yearly seasonality (Ramadan / holiday season effect - simplified
    #    as a smooth sinusoidal boost mid-year and around December)
    yearly_pattern = 1 + 0.10 * np.sin(2 * np.pi * t / 365 + 1.2)

    # 5. Random noise
    noise = np.random.normal(0, 2500, n_days)

    volume = trend * weekly_pattern * salary_spike * yearly_pattern + noise
    volume = np.clip(volume, 5000, None).round(0)

    df = pd.DataFrame({
        "date": dates,
        "transaction_volume": volume
    })
    return df


if __name__ == "__main__":
    df = generate_series(N_DAYS)
    df.to_csv("transaction_volume.csv", index=False)
    print(f"Generated {len(df)} days of transaction volume data "
          f"({df['date'].min().date()} to {df['date'].max().date()})")
    print(f"Mean daily volume: {df['transaction_volume'].mean():,.0f}")
    print("Saved to transaction_volume.csv")
