import pandas as pd
import pickle
import os
import sys

# Project root path add karo
project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
sys.path.insert(0, project_root)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from config import (
    REFERENCE_DATA_PATH,
    CURRENT_DATA_PATH,
    MODEL_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE
)


def retrain_model():
    print("\n🔄 Auto Retraining Shuru...")

    # Purane model ki accuracy
    old_accuracy = get_old_accuracy()
    print(f"📊 Purana model accuracy: {old_accuracy * 100:.2f}%")

    # Models folder ensure karo
    os.makedirs("models", exist_ok=True)

    # Backup banao
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            backup = f.read()

        with open("models/model_old.pkl", "wb") as f:
            f.write(backup)

        print("💾 Purana model backup ho gaya")

    # Data load
    ref_df = pd.read_csv(REFERENCE_DATA_PATH)
    cur_df = pd.read_csv(CURRENT_DATA_PATH)

    # Current data ko double weight do
    combined = pd.concat(
        [ref_df, cur_df, cur_df],
        ignore_index=True
    )

    print(f"📦 Combined data: {len(combined)} rows")

    X = combined[FEATURE_COLUMNS]
    y = combined[TARGET_COLUMN]

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # Train new model
    new_model = RandomForestClassifier(
        random_state=RANDOM_STATE
    )

    new_model.fit(X_train, y_train)

    predictions = new_model.predict(X_test)

    new_accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(f"🤖 Naya model accuracy: {new_accuracy * 100:.2f}%")

    # Deploy decision
    if new_accuracy >= old_accuracy - 0.02:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(new_model, f)

        print("🚀 Naya model deploy ho gaya!")
        print(
            f"📈 {old_accuracy * 100:.2f}% → "
            f"{new_accuracy * 100:.2f}%"
        )

        deployed = True

    else:
        print("⚠️ Naya model worse hai, purana model retain kiya gaya")
        deployed = False

    return {
        "old_accuracy": round(old_accuracy, 4),
        "new_accuracy": round(new_accuracy, 4),
        "deployed": deployed
    }


def get_old_accuracy():
    if not os.path.exists(MODEL_PATH):
        return 0.0

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv(REFERENCE_DATA_PATH)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    predictions = model.predict(X)

    return accuracy_score(y, predictions)


if __name__ == "__main__":
    from src.data_simulator import generate_current_data

    print("🧪 Drift simulate kar rahe hain...")

    generate_current_data(drift_level=0.7)

    result = retrain_model()

    print("\n📋 Result")
    print(f"Old Accuracy : {result['old_accuracy'] * 100:.2f}%")
    print(f"New Accuracy : {result['new_accuracy'] * 100:.2f}%")
    print(f"Deployed     : {result['deployed']}")