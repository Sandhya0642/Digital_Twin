"""
fleet.py
Multi-machine fleet monitoring. Extends the single-motor digital twin to
a small factory floor of independent virtual machines, so the dashboard
can demonstrate fleet-level thinking (availability, OEE, machines needing
attention) rather than only a single asset — this is closer to how a
real plant-level monitoring system is scoped.
"""

import pandas as pd

from simulator import VirtualMotor, SCENARIOS

FLEET_LAYOUT = [
    {"machine": "Motor-01 · Line A", "scenario": "Healthy"},
    {"machine": "Motor-02 · Line A", "scenario": "Healthy"},
    {"machine": "Motor-03 · Line B", "scenario": "Early Warning"},
    {"machine": "Motor-04 · Line B", "scenario": "Bearing Failure"},
    {"machine": "Motor-05 · Line C", "scenario": "Healthy"},
    {"machine": "Motor-06 · Line C", "scenario": "Overheating"},
]


def generate_fleet_snapshot(diagnose_fn, seed: int = None) -> pd.DataFrame:
    """Generate one simulated reading per machine in the fleet and run it
    through the supplied diagnose function (health + ML prediction)."""
    motor = VirtualMotor(seed=seed)
    rows = []
    for unit in FLEET_LAYOUT:
        reading = motor.read(unit["scenario"])
        health, prediction, proba, recommendation = diagnose_fn(reading)
        rows.append({
            "machine": unit["machine"],
            "scenario": unit["scenario"],
            "health_score": health["health_score"],
            "status": health["status"],
            "prediction": prediction,
            "urgency": recommendation["urgency"],
            "temperature": reading["temperature"],
            "vibration": reading["vibration"],
            "rpm": reading["rpm"],
        })
    return pd.DataFrame(rows)


def fleet_kpis(fleet_df: pd.DataFrame) -> dict:
    """Compute simple plant-level KPIs from a fleet snapshot, loosely
    modeled on OEE (Overall Equipment Effectiveness) concepts:
    Availability = share of machines not Critical
    Performance  = average health score
    Quality      = share of machines with a Healthy diagnosis
    """
    total = len(fleet_df)
    if total == 0:
        return {"availability": 0, "performance": 0, "quality": 0, "oee": 0, "attention": 0}

    availability = (fleet_df["status"] != "Critical").mean() * 100
    performance = fleet_df["health_score"].mean()
    quality = (fleet_df["prediction"] == "Healthy").mean() * 100
    oee = (availability / 100) * (performance / 100) * (quality / 100) * 100
    attention = int((fleet_df["status"] != "Healthy").sum())

    return {
        "availability": round(availability, 1),
        "performance": round(performance, 1),
        "quality": round(quality, 1),
        "oee": round(oee, 1),
        "attention": attention,
    }
