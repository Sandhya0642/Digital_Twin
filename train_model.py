"""
train_model.py
Generates a synthetic labeled dataset from the Virtual Motor simulator and
trains a Random Forest classifier to diagnose machine condition from 8
sensor readings (Temperature, Vibration, RPM, Current, Voltage, Bearing
Temperature, Torque, Lubrication Level) into one of seven fault types:
Healthy, Early Warning, Overheating, Bearing Failure, Electrical Fault,
Lubrication Failure, Motor Misalignment.

Run this once before starting the dashboard:
    python train_model.py
"""

import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

from simulator import VirtualMotor, SCENARIOS, SENSOR_NAMES

N_SAMPLES_PER_SCENARIO = 900


def build_dataset() -> pd.DataFrame:
    motor = VirtualMotor(seed=42)
    rows = []
    for scenario in SCENARIOS:
        for _ in range(N_SAMPLES_PER_SCENARIO):
            reading = motor.read(scenario)
            reading["status"] = scenario
            rows.append(reading)
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=7).reset_index(drop=True)


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    df = build_dataset()
    df.to_csv("data/training_data.csv", index=False)

    X = df[SENSOR_NAMES]
    y = df["status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=7, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=250, max_depth=8, random_state=7
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("Model evaluation on held-out test data:\n")
    print(classification_report(y_test, preds))

    joblib.dump(model, "models/predictive_maintenance_model.pkl")
    print("\nModel saved to models/predictive_maintenance_model.pkl")


if __name__ == "__main__":
    main()
