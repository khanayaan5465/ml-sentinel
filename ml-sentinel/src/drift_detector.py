import pandas as pd
import json
import os
import sys

sys.path.append("..")

from scipy import stats
from datetime import datetime

from config import (
    REFERENCE_DATA_PATH,
    CURRENT_DATA_PATH,
    DRIFT_LOG_PATH,
    FEATURE_COLUMNS,
    DRIFT_THRESHOLD
)


def check_drift():
    ref_df = pd.read_csv(REFERENCE_DATA_PATH)
    cur_df = pd.read_csv(CURRENT_DATA_PATH)

    print("Drift Checking in Progress...")
    print(f"{'Feature':<15}{'P-value':>10}{'Status':>12}")
    print("-" * 40)

    drifted = []

    for feature in FEATURE_COLUMNS:
        ref_vals = ref_df[feature].values
        cur_vals = cur_df[feature].values

        _, p_value = stats.ks_2samp(ref_vals, cur_vals)

        is_drifted = p_value < DRIFT_THRESHOLD
        status = "DRIFT" if is_drifted else "OK"

        if is_drifted:
            drifted.append(feature)

        print(f"{feature:<15}{p_value:>10.4f}{status:>12}")

    print("-" * 40)

    overall_drift = len(drifted) > 2

    if overall_drift:
        print(f"Drift Detected! {len(drifted)}/5 features affected")
    else:
        print(f"Model Healthy. {len(drifted)}/5 features drifted")

    result = {
        "timestamp": datetime.now().isoformat(),
        "overall_drift": overall_drift,
        "drifted_count": len(drifted),
        "drifted_features": drifted
    }

    save_log(result)

    return result


def save_log(result):
    os.makedirs("data", exist_ok=True)

    log = []

    if os.path.exists(DRIFT_LOG_PATH):
        with open(DRIFT_LOG_PATH, "r") as f:
            try:
                log = json.load(f)
            except:
                log = []

    log.append(result)

    with open(DRIFT_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


if __name__ == "__main__":
    print("\n=== TEST 1: NORMAL DATA ===")
    
    from src.data_simulator import generate_current_data

    generate_current_data(drift_level=0.0)
    check_drift()

    print("\n=== TEST 2: DRIFTED DATA ===")

    generate_current_data(drift_level=0.9)
    check_drift()