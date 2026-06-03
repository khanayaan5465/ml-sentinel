import pandas as pd
import pickle
import os
import sys

# Project root path add karo
sys.path.append("..")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from config import (
    REFERENCE_DATA_PATH,
    MODEL_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE
)


def train_model():
    print("Data loading...")

    # Reference data load karo
    df = pd.read_csv(REFERENCE_DATA_PATH)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    print("Model training in progress...")

    # Model create aur train
    model = RandomForestClassifier(
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    # Prediction aur accuracy
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Models folder create karo agar nahi hai
    os.makedirs("models", exist_ok=True)

    # Model save karo
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved at: {MODEL_PATH}")

    return model, accuracy


if __name__ == "__main__":
    model, acc = train_model()