"""
health.py
Rule-based Machine Health Score and Remaining Useful Life (RUL) estimation
for the Digital Twin, now covering all 8 simulated sensors.
"""

# (warning_threshold, critical_threshold, direction, penalty_warning, penalty_critical)
# direction: "high" means values above threshold are bad, "low" means values below are bad
THRESHOLDS = {
    "temperature": (75.0, 88.0, "high", 15, 28),
    "vibration": (4.0, 7.0, "high", 15, 28),
    "bearing_temperature": (72.0, 85.0, "high", 15, 25),
    "current": (13.0, 15.5, "high", 10, 18),
    "voltage": (405.0, 400.0, "low", 8, 15),  # deviation below nominal is bad
    "torque": (28.0, 32.0, "high", 6, 12),
    "lubrication_level": (60.0, 40.0, "low", 12, 20),
    "rpm": (1350.0, 1200.0, "low", 8, 15),
}

REASON_LABELS = {
    "temperature": "Elevated temperature",
    "vibration": "Elevated vibration",
    "bearing_temperature": "Elevated bearing temperature",
    "current": "High motor current",
    "voltage": "Voltage deviation",
    "torque": "Torque irregularity",
    "lubrication_level": "Low lubrication level",
    "rpm": "RPM instability / drop",
}


def calculate_health(**readings) -> dict:
    """Compute a 0-100 health score, broad severity status, and a simple
    Remaining Useful Life (RUL) estimate from the full sensor reading."""
    score = 100.0
    reasons = []

    for sensor, (warn_t, crit_t, direction, pen_w, pen_c) in THRESHOLDS.items():
        if sensor not in readings:
            continue
        value = readings[sensor]

        if direction == "high":
            if value >= crit_t:
                score -= pen_c
                reasons.append(REASON_LABELS[sensor])
            elif value >= warn_t:
                score -= pen_w
                reasons.append(REASON_LABELS[sensor])
        else:  # "low" — below threshold is bad
            if value <= crit_t:
                score -= pen_c
                reasons.append(REASON_LABELS[sensor])
            elif value <= warn_t:
                score -= pen_w
                reasons.append(REASON_LABELS[sensor])

    score = max(0.0, min(100.0, score))

    if score >= 80:
        status = "Healthy"
    elif score >= 50:
        status = "Warning"
    else:
        status = "Critical"

    # Simple heuristic RUL: healthier machines have proportionally more
    # estimated running hours left before intervention is recommended.
    rul_hours = round((score / 100.0) * 500.0)

    return {
        "health_score": round(score, 1),
        "status": status,
        "reasons": reasons if reasons else ["All parameters within normal range"],
        "estimated_rul_hours": rul_hours,
    }
