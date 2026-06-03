import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys
sys.path.append("..")

from scipy import stats

# ── Page Config ──────────────────────────────
st.set_page_config(
    page_title="ML Sentinel",
    page_icon="🤖",
    layout="wide"
)

# ── Helper Functions ──────────────────────────
def load_data():
    ref = pd.read_csv("data/reference_data.csv") \
          if os.path.exists("data/reference_data.csv") else None
    cur = pd.read_csv("data/current_data.csv") \
          if os.path.exists("data/current_data.csv") else None
    return ref, cur

def load_log():
    path = "data/drift_log.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return []

def get_drift_scores(ref_df, cur_df):
    results = {}
    features = ["feature_1","feature_2","feature_3",
                "feature_4","feature_5"]
    for feat in features:
        _, p_value = stats.ks_2samp(
            ref_df[feat].values,
            cur_df[feat].values
        )
        results[feat] = round(p_value, 4)
    return results

# ── Main Dashboard ────────────────────────────
st.title("🤖 ML Sentinel — Live Monitor")
st.caption("Automated Drift Detection & Auto Retraining System")

st.divider()

# ── Sidebar Controls ──────────────────────────
st.sidebar.title("⚙️ Controls")
drift_level = st.sidebar.slider(
    "Drift Level Simulate Karo",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.1
)

if st.sidebar.button("🔄 Naya Data Generate Karo"):
    sys.path.insert(0, "src")
    from data_simulator import generate_current_data
    generate_current_data(drift_level=drift_level)
    st.sidebar.success("✅ Data ready!")

if st.sidebar.button("🚀 Retrain Model"):
    sys.path.insert(0, "src")
    from auto_retrainer import retrain_model
    result = retrain_model()
    if result["deployed"]:
        st.sidebar.success(f"✅ Deployed! {result['new_accuracy']*100:.1f}%")
    else:
        st.sidebar.warning("⚠️ Old model rakha gaya")

st.sidebar.divider()
if st.sidebar.button("🏋️ Initial Setup Chalaao"):
    sys.path.insert(0, "src")
    from data_simulator import generate_reference_data, generate_current_data
    from model_trainer import train_model
    generate_reference_data()
    generate_current_data(drift_level=0.0)
    train_model()
    st.sidebar.success("✅ Setup complete!")

# ── Load Data ─────────────────────────────────
ref_df, cur_df = load_data()
log            = load_log()

if ref_df is None:
    st.warning("⚠️ Pehle sidebar mein 'Initial Setup' button dabao!")
    st.stop()

# ── Row 1: Status Cards ───────────────────────
drift_scores = get_drift_scores(ref_df, cur_df)
drifted      = [f for f, p in drift_scores.items() if p < 0.05]
overall_ok   = len(drifted) <= 2

col1, col2, col3, col4 = st.columns(4)

with col1:
    status = "✅ Healthy" if overall_ok else "🚨 Drifted"
    color  = "normal" if overall_ok else "inverse"
    st.metric("Model Status", status)

with col2:
    st.metric("Drifted Features", f"{len(drifted)} / 5")

with col3:
    st.metric("Reference Data", f"{len(ref_df):,} rows")

with col4:
    st.metric("Current Data", f"{len(cur_df):,} rows")

st.divider()

# ── Row 2: Drift Table + Distribution ─────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📊 Feature Drift Status")

    rows = []
    for feat, p_val in drift_scores.items():
        rows.append({
            "Feature" : feat,
            "P-Value" : p_val,
            "Status"  : "🚨 DRIFT" if p_val < 0.05 else "✅ OK"
        })

    drift_df = pd.DataFrame(rows)
    st.dataframe(drift_df, use_container_width=True, hide_index=True)

with col_right:
    st.subheader("📈 Feature Distribution Comparison")

    selected = st.selectbox("Feature chuno:", drift_scores.keys())

    ref_vals = ref_df[selected].values
    cur_vals = cur_df[selected].values

    chart_df = pd.DataFrame({
        "Reference (Training)" : pd.Series(ref_vals).sample(200, replace=True).values,
        "Current (Production)" : pd.Series(cur_vals).values[:200]
    })
    st.line_chart(chart_df)

st.divider()

# ── Row 3: Drift History ──────────────────────
st.subheader("🕐 Drift History")

if log:
    history = []
    for i, entry in enumerate(log):
        history.append({
            "Check #"       : i + 1,
            "Time"          : entry["timestamp"][11:19],
            "Drift?"        : "🚨 YES" if entry["overall_drift"] else "✅ NO",
            "Features Affected": entry["drifted_count"]
        })
    st.dataframe(pd.DataFrame(history),
                 use_container_width=True,
                 hide_index=True)
else:
    st.info("Abhi koi history nahi hai. Drift check chalao!")

st.divider()
st.caption("ML Sentinel | Built with Python + Streamlit | "
           "KS Test Drift Detection")