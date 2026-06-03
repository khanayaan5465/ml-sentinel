import numpy as np
import pandas as pd
import os
import sys

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, project_root)

from config import FEATURE_COLUMNS, TARGET_COLUMN
from config import REFERENCE_DATA_PATH, CURRENT_DATA_PATH


def generate_reference_data():
    np.random.seed(42)

    data = {}

    for col in FEATURE_COLUMNS:
        data[col] = np.random.normal(loc=0.0, scale=1.0, size=1000)

    score = data["feature_1"] + data["feature_2"]
    data[TARGET_COLUMN] = (score > 0).astype(int)

    df = pd.DataFrame(data)

    os.makedirs("data", exist_ok=True)

    df.to_csv(REFERENCE_DATA_PATH, index=False)

    print(f"Reference data ready: {len(df)} rows")

    return df


def generate_current_data(drift_level=0.0):
    np.random.seed(99)

    data = {}

    for col in FEATURE_COLUMNS:
        data[col] = np.random.normal(
            loc=drift_level * 2.0,
            scale=1.0,
            size=200
        )

    score = data["feature_1"] + data["feature_2"]
    data[TARGET_COLUMN] = (score > 0).astype(int)

    df = pd.DataFrame(data)

    df.to_csv(CURRENT_DATA_PATH, index=False)

    status = (
        "Normal" if drift_level == 0
        else "Mild" if drift_level < 0.5
        else "Heavy"
    )

    print(f"Current data ready: {len(df)} rows | {status}")

    return df

if __name__ == "__main__":
    generate_reference_data()
    generate_current_data(drift_level=0.0)