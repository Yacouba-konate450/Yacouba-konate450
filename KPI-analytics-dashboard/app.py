"""
app.py
------
Real-Time KPI Analytical Dashboard — Flask backend.

Serves a live-updating dashboard of key business KPIs (revenue, transaction
volume, active customers, average transaction value) computed from a
simulated data feed, with a REST API that the front-end polls for updates.

Run:
    python app.py
Then open:
    http://127.0.0.1:5000
"""

from flask import Flask, jsonify, render_template
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Simulated data generation (in a real deployment, this would query a
# database or a live transaction feed instead).
# ---------------------------------------------------------------------------

REGIONS = ["Bamako", "Sikasso", "Kayes", "Segou", "Mopti"]
PRODUCTS = ["Savings Account", "Mobile Banking", "Loan", "Card Payment", "Transfer"]


def generate_live_data(n_records=500):
    """Generates a rolling window of simulated transactions for the last 24h."""
    now = datetime.now()
    records = []
    for i in range(n_records):
        ts = now - timedelta(minutes=random.randint(0, 1440))
        records.append({
            "timestamp": ts,
            "region": random.choice(REGIONS),
            "product": random.choice(PRODUCTS),
            "amount": round(np.random.gamma(2.0, 8000), 2),
            "customer_id": f"CUST{random.randint(1, 300):04d}"
        })
    return pd.DataFrame(records)


# Data is regenerated on each API call to simulate a live feed.
def get_current_snapshot():
    df = generate_live_data()

    total_revenue = df["amount"].sum()
    total_transactions = len(df)
    active_customers = df["customer_id"].nunique()
    avg_transaction_value = df["amount"].mean()

    revenue_by_region = (
        df.groupby("region")["amount"].sum().round(2).sort_values(ascending=False).to_dict()
    )
    revenue_by_product = (
        df.groupby("product")["amount"].sum().round(2).sort_values(ascending=False).to_dict()
    )

    # Hourly trend for the last 24h
    df["hour"] = df["timestamp"].dt.floor("h")
    hourly = df.groupby("hour")["amount"].sum().sort_index()
    hourly_trend = {
        "labels": [h.strftime("%H:%M") for h in hourly.index],
        "values": [round(v, 2) for v in hourly.values]
    }

    return {
        "kpis": {
            "total_revenue": round(total_revenue, 2),
            "total_transactions": total_transactions,
            "active_customers": active_customers,
            "avg_transaction_value": round(avg_transaction_value, 2)
        },
        "revenue_by_region": revenue_by_region,
        "revenue_by_product": revenue_by_product,
        "hourly_trend": hourly_trend,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def dashboard():
    return render_template("index.html")


@app.route("/api/kpis")
def api_kpis():
    """REST endpoint the front-end polls every few seconds for fresh KPI data."""
    return jsonify(get_current_snapshot())


if __name__ == "__main__":
    app.run(debug=True, port=5000)
