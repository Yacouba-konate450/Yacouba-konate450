# Real-Time KPI Analytical Dashboard

> Full-stack web dashboard for real-time visualisation of key business performance indicators — Python (Flask) backend, live-updating front-end.

## Overview

A production-style architecture for monitoring business KPIs in real time: a Flask backend simulates a live transaction feed and exposes it through a REST API, while a responsive front-end (Chart.js) polls that API every 5 seconds and updates the dashboard without a page reload.

This mirrors the kind of internal operations dashboard used in banking and telecom environments to monitor daily transaction volume, revenue, and customer activity at a glance.

## Features

- **4 live KPI cards**: total revenue, transaction count, active customers, average transaction value
- **24-hour revenue trend** line chart
- **Revenue breakdown by region** (doughnut chart)
- **Revenue breakdown by product line** (horizontal bar chart)
- Auto-refreshes every 5 seconds via a lightweight REST API (`/api/kpis`)
- Fully responsive layout, no page reload required

## Tech stack

`Python` `Flask` `Pandas` `NumPy` `Chart.js` `HTML/CSS/JavaScript`

## Architecture

```
Browser (index.html)
      │
      │  fetch('/api/kpis') every 5s
      ▼
Flask backend (app.py)
      │
      │  generates / aggregates transaction data
      ▼
   JSON response → Chart.js renders live charts
```

## Project structure

```
kpi-analytics-dashboard/
├── app.py                 # Flask backend + REST API
├── templates/
│   └── index.html         # Front-end dashboard (HTML/CSS/JS + Chart.js)
├── requirements.txt
└── README.md
```

## How to run

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/kpi-analytics-dashboard
cd kpi-analytics-dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Flask server
python app.py

# 4. Open in your browser
http://127.0.0.1:5000
```

The dashboard will start updating automatically — no further action needed.

## Design decisions

- **Why Flask over Django here**: the goal is a lightweight, single-purpose API + dashboard, where Flask's minimal footprint keeps the codebase easy to read and extend.
- **Why polling instead of WebSockets**: a 5-second REST poll is simple, framework-agnostic, and sufficient for a KPI dashboard (as opposed to a use case needing sub-second updates); it can be swapped for WebSockets/Server-Sent Events with minimal changes to `app.py`.
- **Why simulated data**: in a production deployment, `get_current_snapshot()` in `app.py` would query a real database or transaction stream instead of `generate_live_data()` — the rest of the architecture (API, front-end, charts) would not need to change.

## Extending this project

- Swap the simulated data generator for a real database connection (PostgreSQL/MySQL)
- Add authentication (Flask-Login) to restrict dashboard access to authorised staff
- Add a date-range filter and export-to-CSV button for reports
- Deploy behind Gunicorn + Nginx for production use

---

*Built by [Yacouba Konaté](https://github.com/YOUR_USERNAME) — Computer Engineer, MSc AI Engineering candidate.*
