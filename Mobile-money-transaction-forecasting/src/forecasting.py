"""
forecasting.py
---------------
Comparative time series forecasting pipeline: Statistical (SARIMA) vs
Machine Learning (XGBoost) vs Deep Learning (LSTM) - the same three-family
comparison methodology used in academic time series forecasting studies
(e.g. inflation forecasting, transaction/demand forecasting).

Pipeline:
  1. Load and visualise the daily transaction volume series
  2. Time-aware train/test split (last 60 days held out - NEVER shuffle
     time series data randomly, this is a common and serious mistake)
  3. Train and forecast with:
       - SARIMA (statistical baseline, captures trend + seasonality explicitly)
       - XGBoost (ML, using lag + calendar features)
       - LSTM (deep learning, using a sliding window of past values)
  4. Compare all three on MAE, RMSE and MAPE
  5. Export a comparison chart and a Power BI-ready forecast CSV

Run from the project root:
    python src/forecasting.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
tf.random.set_seed(42)

DATA_PATH = "data/transaction_volume.csv"
OUTPUT_DIR = "outputs"
TEST_DAYS = 60  # forecast horizon

sns.set_style("whitegrid")


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

def load_data(path):
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date")
    return df


def train_test_split_timeseries(df, test_days):
    """Time-aware split: the test set is always the LAST N days, never a
    random sample - shuffling a time series before splitting leaks future
    information into training and produces misleadingly good scores."""
    train = df.iloc[:-test_days]
    test = df.iloc[-test_days:]
    return train, test


# ---------------------------------------------------------------------------
# 2a. SARIMA - statistical model
# ---------------------------------------------------------------------------

def run_sarima(train, test):
    print("\n--- Training SARIMA ---")
    model = SARIMAX(
        train["transaction_volume"],
        order=(2, 1, 2),
        seasonal_order=(1, 1, 1, 7),  # weekly seasonality
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    fitted = model.fit(disp=False)
    forecast = fitted.forecast(steps=len(test))
    return forecast.values


# ---------------------------------------------------------------------------
# 2b. XGBoost - ML model with lag + calendar features
# ---------------------------------------------------------------------------

def make_features(df):
    df = df.copy()
    df["day_of_week"] = df.index.dayofweek
    df["day_of_month"] = df.index.day
    df["month"] = df.index.month
    for lag in [1, 2, 3, 7, 14]:
        df[f"lag_{lag}"] = df["transaction_volume"].shift(lag)
    df["rolling_mean_7"] = df["transaction_volume"].shift(1).rolling(7).mean()
    return df


def run_xgboost(train, test):
    print("\n--- Training XGBoost ---")
    full = pd.concat([train, test])
    featured = make_features(full)
    featured = featured.dropna()

    feature_cols = [c for c in featured.columns if c != "transaction_volume"]

    train_feat = featured.loc[featured.index.isin(train.index)]
    test_feat = featured.loc[featured.index.isin(test.index)]

    model = XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.9, random_state=42
    )
    model.fit(train_feat[feature_cols], train_feat["transaction_volume"])

    # Iterative forecasting: predict one step, feed it back in for the next lag features
    history = train["transaction_volume"].copy()
    preds = []
    for date in test.index:
        row = pd.DataFrame(index=[date])
        row["day_of_week"] = date.dayofweek
        row["day_of_month"] = date.day
        row["month"] = date.month
        for lag in [1, 2, 3, 7, 14]:
            row[f"lag_{lag}"] = history.iloc[-lag]
        row["rolling_mean_7"] = history.iloc[-7:].mean()

        pred = model.predict(row[feature_cols])[0]
        preds.append(pred)
        history.loc[date] = pred  # feed forward

    return np.array(preds)


# ---------------------------------------------------------------------------
# 2c. LSTM - deep learning model
# ---------------------------------------------------------------------------

def run_lstm(train, test, window=14):
    print("\n--- Training LSTM ---")
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train[["transaction_volume"]])

    X_train, y_train = [], []
    for i in range(window, len(train_scaled)):
        X_train.append(train_scaled[i - window:i, 0])
        y_train.append(train_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    model = Sequential([
        LSTM(32, activation="tanh", input_shape=(window, 1)),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X_train, y_train, epochs=25, batch_size=16, verbose=0)

    # Iterative multi-step forecasting
    full_scaled = scaler.transform(train[["transaction_volume"]])
    window_data = full_scaled[-window:, 0].tolist()
    preds_scaled = []

    for _ in range(len(test)):
        x_input = np.array(window_data[-window:]).reshape(1, window, 1)
        pred = model.predict(x_input, verbose=0)[0, 0]
        preds_scaled.append(pred)
        window_data.append(pred)

    preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
    return preds


# ---------------------------------------------------------------------------
# 3. Evaluation & comparison
# ---------------------------------------------------------------------------

def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mp = mape(y_true, y_pred)
    print(f"{name:10s} — MAE: {mae:9,.0f} | RMSE: {rmse:9,.0f} | MAPE: {mp:5.2f}%")
    return {"model": name, "MAE": mae, "RMSE": rmse, "MAPE": mp}


def save_comparison_chart(test, forecasts, output_dir):
    import os
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(11, 6))
    plt.plot(test.index, test["transaction_volume"], label="Actual",
              color="black", linewidth=2)
    colors = {"SARIMA": "#185FA5", "XGBoost": "#BA7517", "LSTM": "#0F6E56"}
    for name, preds in forecasts.items():
        plt.plot(test.index, preds, label=name, color=colors[name],
                  linestyle="--", linewidth=1.6)
    plt.title("60-Day Forecast Comparison — SARIMA vs XGBoost vs LSTM")
    plt.xlabel("Date")
    plt.ylabel("Transaction Volume")
    plt.legend()
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/forecast_comparison.png", dpi=150)
    plt.close()


def save_metrics_chart(results_df, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    metrics = ["MAE", "RMSE", "MAPE"]
    colors = ["#0F6E56", "#185FA5", "#BA7517"]
    for ax, metric in zip(axes, metrics):
        ax.bar(results_df["model"], results_df[metric], color=colors)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/metrics_comparison.png", dpi=150)
    plt.close()


def export_for_powerbi(test, forecasts, output_dir):
    export_df = test.copy()
    export_df = export_df.rename(columns={"transaction_volume": "actual"})
    for name, preds in forecasts.items():
        export_df[f"forecast_{name.lower()}"] = preds
    export_df.to_csv(f"{output_dir}/forecast_results.csv")
    print(f"\nForecast results exported to {output_dir}/forecast_results.csv "
          f"(ready for Power BI import)")


if __name__ == "__main__":
    print("Loading data...")
    df = load_data(DATA_PATH)

    train, test = train_test_split_timeseries(df, TEST_DAYS)
    print(f"Train: {len(train)} days | Test: {len(test)} days (last {TEST_DAYS} days held out)")

    sarima_preds = run_sarima(train, test)
    xgb_preds = run_xgboost(train, test)
    lstm_preds = run_lstm(train, test)

    forecasts = {"SARIMA": sarima_preds, "XGBoost": xgb_preds, "LSTM": lstm_preds}

    print("\n--- Evaluation on held-out 60-day test set ---")
    results = []
    y_true = test["transaction_volume"].values
    for name, preds in forecasts.items():
        results.append(evaluate(name, y_true, preds))
    results_df = pd.DataFrame(results)

    print("\nSaving charts...")
    save_comparison_chart(test, forecasts, OUTPUT_DIR)
    save_metrics_chart(results_df, OUTPUT_DIR)

    export_for_powerbi(test, forecasts, OUTPUT_DIR)
    results_df.to_csv(f"{OUTPUT_DIR}/model_comparison_metrics.csv", index=False)

    print("\nDone. See the outputs/ folder for charts and CSVs.")
    print("\nBest model by MAPE:", results_df.loc[results_df["MAPE"].idxmin(), "model"])
