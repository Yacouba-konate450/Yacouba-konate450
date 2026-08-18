# Financial Fraud Detection System

> Machine learning system for detecting fraudulent financial transactions, with an interactive Power BI dashboard for non-technical stakeholders.

## Overview

This project builds an end-to-end fraud detection pipeline on transaction-level financial data: synthetic data generation (mirroring real banking transaction patterns), feature engineering, model training on an imbalanced dataset, and evaluation with metrics appropriate for rare-event detection.

The dataset is **synthetically generated** to resemble real banking transaction data without using any real customer information, and is deliberately designed with overlapping fraud/legitimate patterns (e.g. legitimate frequent travelers, subtle fraud cases) rather than an artificially "easy" separation — reflecting the real-world difficulty of fraud detection.

## Key results

- **ROC-AUC: 0.914** on held-out test data
- **Precision: 0.83** / **Recall: 0.59** on the fraud class (rare-event detection with ~1.9% fraud rate)
- 5-fold cross-validated ROC-AUC: 0.872 (± 0.030), confirming the model generalises rather than overfitting to one split

## Why precision/recall trade-off matters here

With only ~1.9% of transactions being fraudulent, accuracy is a misleading metric — a model that predicts "not fraud" every time would already be 98% accurate. Instead, this project optimises for **ROC-AUC** and reports **precision and recall separately** for the fraud class, since in a real banking context:
- **Low precision** → too many legitimate customers flagged, creating friction and false alerts
- **Low recall** → real fraud goes undetected

The current model favours precision (0.83) — when it flags a transaction as fraud, it's right 83% of the time — while catching 59% of actual fraud cases. This trade-off point (the classification threshold) can be tuned depending on the bank's risk appetite.

## Tech stack

`Python` `pandas` `scikit-learn` `NumPy` `Matplotlib` `Seaborn` `Power BI`

## Methods used

- Synthetic data generation with realistic behavioural noise and overlapping classes
- Feature engineering: time-based features (hour, day of week, night flag), behavioural features (amount-to-average-spend ratio), and location-mismatch detection
- Random Forest classifier with `class_weight="balanced"` to handle class imbalance
- 5-fold cross-validation for score stability
- ROC-AUC, precision/recall, confusion matrix, and feature importance analysis
- Export of scored transactions to CSV for Power BI dashboard integration

## Project structure

```
financial-fraud-detection/
├── data/
│   ├── generate_data.py       # generates the synthetic transaction dataset
│   └── transactions.csv       # generated dataset (20,000 transactions)
├── src/
│   └── fraud_detection.py     # main pipeline: features, training, evaluation
├── outputs/
│   ├── roc_curve.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── scored_transactions.csv  # ready for Power BI import
├── requirements.txt
└── README.md
```

## How to run

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/financial-fraud-detection
cd financial-fraud-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the synthetic dataset
python data/generate_data.py

# 4. Run the full pipeline (training, evaluation, chart generation)
python src/fraud_detection.py
```

All outputs (charts + the Power BI-ready CSV) are saved to the `outputs/` folder.

## Power BI dashboard

The `outputs/scored_transactions.csv` file is structured for direct import into Power BI, with columns for transaction amount, merchant category, location, computed fraud probability, and the model's prediction — enabling a dashboard with:
- Fraud rate trends over time
- Fraud probability distribution by merchant category and location
- High-risk transaction drill-down for investigators

## Results

**ROC Curve** — shows the model's ability to separate fraud from legitimate transactions across all classification thresholds:

![ROC Curve](outputs/roc_curve.png)

**Confusion Matrix** — shows exactly how many fraud cases were caught vs missed on the test set:

![Confusion Matrix](outputs/confusion_matrix.png)

**Feature Importance** — shows which signals the model relies on most:

![Feature Importance](outputs/feature_importance.png)

## Notes & limitations

- This project uses **synthetic data** designed to mirror realistic transaction patterns; it is not trained on real banking data.
- In a production banking environment, additional features (device fingerprinting, IP geolocation, transaction velocity across a rolling time window, merchant risk scoring) would further improve detection.
- The classification threshold (currently the default 0.5) can be tuned based on the institution's tolerance for false positives vs missed fraud.

---

*Built by [Yacouba Konaté](https://github.com/YOUR_USERNAME) — Computer Engineer, MSc AI Engineering candidate.*
