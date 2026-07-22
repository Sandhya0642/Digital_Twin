"""
anomaly.py
Unsupervised anomaly detection layer that complements the supervised
Random Forest classifier. An Isolation Forest is trained only on
"Healthy" operating data, so it can flag sensor readings that look
statistically unusual even if they don't match a fault signature the
classifier was explicitly trained on — a common pattern in real
condition-monitoring systems (supervised model for known faults,
unsupervised model for the unknown/unseen ones).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from simulator import VirtualMotor, SENSOR_NAMES

N_HEALTHY_SAMPLES = 1500


def build_reference_dataset(seed: int = 11) -> pd.DataFrame:
    """Sample a large batch of healthy-operation readings to define
    what 'normal' looks like for this machine."""
    motor = VirtualMotor(seed=seed)
    rows = [motor.read("Healthy") for _ in range(N_HEALTHY_SAMPLES)]
    return pd.DataFrame(rows)[SENSOR_NAMES]


def train_anomaly_model() -> IsolationForest:
    df = build_reference_dataset()
    model = IsolationForest(
        n_estimators=200,
        contamination=0.03,
        random_state=7,
    )
    model.fit(df)
    return model


def score_reading(model: IsolationForest, reading: dict) -> dict:
    """Return an anomaly score (higher = more normal, matching sklearn's
    convention) rescaled to an intuitive 0-100 'Normalcy Index', plus a
    boolean flag for whether the point is classified as an outlier."""
    X = pd.DataFrame([reading])[SENSOR_NAMES]
    raw_score = float(model.decision_function(X)[0])   # roughly -0.5..0.5
    is_outlier = bool(model.predict(X)[0] == -1)
    normalcy_index = float(np.clip((raw_score + 0.5) * 100, 0, 100))
    return {
        "raw_score": round(raw_score, 4),
        "normalcy_index": round(normalcy_index, 1),
        "is_anomaly": is_outlier,
    }
