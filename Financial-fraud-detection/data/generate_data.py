"""
generate_data.py
-----------------
Generates a realistic synthetic dataset of financial transactions with a
small, imbalanced fraction of fraudulent transactions - mirroring the
structure of real banking transaction data (amount, timestamp, merchant
category, customer behaviour) without using any real customer data.

Run:
    python generate_data.py
Output:
    transactions.csv (in this same folder)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

N_CUSTOMERS = 500
N_TRANSACTIONS = 20000
FRAUD_RATE = 0.02  # ~2% fraud, similar to real-world imbalance

MERCHANT_CATEGORIES = [
    "Groceries", "Electronics", "Fuel", "Restaurant", "Online Retail",
    "Utilities", "Travel", "Cash Withdrawal", "Mobile Money Transfer", "Other"
]

LOCATIONS = [
    "Bamako", "Sikasso", "Kayes", "Segou", "Mopti", "Abroad - Europe",
    "Abroad - USA", "Abroad - Other"
]


def generate_customers(n):
    # A subset of customers are frequent legitimate travelers (e.g. diaspora
    # Malians, business travelers) - this is what makes "foreign location"
    # an imperfect fraud signal, forcing the model to combine several weak
    # signals rather than rely on one obvious rule.
    frequent_travelers = np.random.rand(n) < 0.12

    return pd.DataFrame({
        "customer_id": [f"CUST{str(i).zfill(5)}" for i in range(n)],
        "avg_monthly_spend": np.random.gamma(shape=2.0, scale=150, size=n).round(2),
        "home_location": np.random.choice(LOCATIONS[:5], size=n),
        "frequent_traveler": frequent_travelers
    })


def generate_transactions(customers, n_transactions):
    rows = []
    start_date = datetime(2025, 1, 1)

    for i in range(n_transactions):
        customer = customers.sample(1).iloc[0]
        is_fraud = np.random.rand() < FRAUD_RATE

        timestamp = start_date + timedelta(
            days=int(np.random.rand() * 240),
            hours=int(np.random.rand() * 24),
            minutes=int(np.random.rand() * 60)
        )

        if is_fraud:
            # ~65% of fraud is "obvious" (unusual amount/location/hour),
            # ~35% is a harder case that mimics normal behaviour more closely
            # (a more sophisticated fraud pattern) - this keeps the problem
            # realistically hard instead of trivially separable.
            subtle_fraud = np.random.rand() < 0.35

            if subtle_fraud:
                amount = round(np.random.gamma(shape=2.2, scale=customer["avg_monthly_spend"] / 4 + 12), 2)
                location = customer["home_location"] if np.random.rand() < 0.6 else np.random.choice(LOCATIONS)
                merchant = np.random.choice(MERCHANT_CATEGORIES)
            else:
                amount = round(np.random.gamma(shape=4.0, scale=customer["avg_monthly_spend"] / 2 + 40), 2)
                location = np.random.choice(["Abroad - Europe", "Abroad - USA", "Abroad - Other"])
                merchant = np.random.choice(["Online Retail", "Cash Withdrawal", "Mobile Money Transfer"])

        else:
            amount = round(np.random.gamma(shape=2.0, scale=customer["avg_monthly_spend"] / 4 + 10), 2)
            # Frequent travelers legitimately transact abroad ~40% of the time,
            # which is what makes "location_mismatch" an imperfect signal.
            if customer["frequent_traveler"] and np.random.rand() < 0.4:
                location = np.random.choice(LOCATIONS)
            else:
                location = customer["home_location"]
            merchant = np.random.choice(MERCHANT_CATEGORIES)

        # Small amount of random noise on every transaction amount,
        # reflecting natural variability in real spending data.
        amount = max(1.0, round(amount * np.random.normal(1.0, 0.08), 2))

        rows.append({
            "transaction_id": f"TXN{str(i).zfill(7)}",
            "customer_id": customer["customer_id"],
            "timestamp": timestamp,
            "amount": amount,
            "merchant_category": merchant,
            "location": location,
            "customer_home_location": customer["home_location"],
            "customer_avg_monthly_spend": customer["avg_monthly_spend"],
            "is_fraud": int(is_fraud)
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    customers = generate_customers(N_CUSTOMERS)
    transactions = generate_transactions(customers, N_TRANSACTIONS)
    transactions = transactions.sort_values("timestamp").reset_index(drop=True)

    output_path = "transactions.csv"
    transactions.to_csv(output_path, index=False)

    print(f"Generated {len(transactions)} transactions.")
    print(f"Fraud cases: {transactions['is_fraud'].sum()} "
          f"({transactions['is_fraud'].mean()*100:.2f}%)")
    print(f"Saved to {output_path}")
