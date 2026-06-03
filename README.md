
# ML Sentinel 🤖📊

An automated MLOps system that continuously monitors ML model health, detects data drift, and automatically retrains models when performance drops.

---

## Problem

Machine learning models lose accuracy over time due to changing data patterns — a phenomenon called **Data Drift**. Without monitoring, this leads to unreliable predictions and business losses.

## Solution

ML Sentinel solves this by running a continuous automated pipeline that:
- Monitors model performance on new incoming data
- Detects when data distribution has shifted (drift)
- Sends alerts when model health degrades
- Automatically retrains and re-evaluates the model

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python, Pandas, NumPy | Data processing and core development |
| Scikit-learn | Model training and evaluation |
| Random Forest | Customer churn prediction model |
| Evidently AI | Data drift detection and monitoring |
| MLflow | Model versioning and experiment tracking |
| Streamlit | Interactive monitoring dashboard |
| Plotly | Visualizing drift trends and model health |
| SQLite | Storing logs and monitoring data |
| APScheduler | Automated monitoring and retraining schedules |

---

## Features

- 📊 Real-time monitoring dashboard built with Streamlit
- 🔍 Automated data drift detection using Evidently AI
- 🔁 Auto-retraining triggered when model accuracy drops
- 📦 Experiment tracking and model versioning with MLflow
- ⏰ Scheduled monitoring runs via APScheduler
- 🗄️ All logs and metrics stored in SQLite

---

## How to Run

### 1. Install dependencies
```
pip install -r requirements.txt
```

### 2. Run the monitoring dashboard
```
streamlit run app.py
```

### 3. Start the automated scheduler
```
python scheduler.py
```

---

## Project Structure

```
ml-sentinel/
├── app.py              # Streamlit dashboard
├── model.py            # Random Forest training
├── monitor.py          # Drift detection logic
├── scheduler.py        # APScheduler automation
├── database.py         # SQLite logging
├── requirements.txt
└── README.md
```

---

## Author

**Khan Ayaan Amjad**
📧 Khanayaan52576@gmail.com
🔗 github.com/khanayaan5465
  
