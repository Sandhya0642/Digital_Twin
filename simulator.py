"""
simulator.py
Virtual Industrial Motor v2 — generates simulated multi-sensor readings
(Temperature, Vibration, RPM, Current, Voltage, Bearing Temperature, Torque,
Lubrication Level) for seven diagnostic scenarios, and supports a
progressive "lifecycle degradation" mode that mimics a motor slowly
developing a specific fault over its operating hours.
"""

import numpy as np

SENSOR_NAMES = [
    "temperature",
    "vibration",
    "rpm",
    "current",
    "voltage",
    "bearing_temperature",
    "torque",
    "lubrication_level",
]

SENSOR_UNITS = {
    "temperature": "°C",
    "vibration": "mm/s",
    "rpm": "RPM",
    "current": "A",
    "voltage": "V",
    "bearing_temperature": "°C",
    "torque": "Nm",
    "lubrication_level": "%",
}

SENSOR_LABELS = {
    "temperature": "Temperature",
    "vibration": "Vibration",
    "rpm": "Rotational Speed",
    "current": "Motor Current",
    "voltage": "Supply Voltage",
    "bearing_temperature": "Bearing Temperature",
    "torque": "Motor Torque",
    "lubrication_level": "Lubrication Level",
}

# Diagnostic labels the motor / ML model can report.
FAULT_TYPES = [
    "Healthy",
    "Early Warning",
    "Overheating",
    "Bearing Failure",
    "Electrical Fault",
    "Lubrication Failure",
    "Motor Misalignment",
]

# Each scenario maps every sensor to a (mean, std_dev) profile used to draw
# a realistic noisy reading. Values are illustrative but internally
# consistent with the physical fault they represent.
SCENARIOS = {
    "Healthy": {
        "temperature": (60.0, 3.0),
        "vibration": (1.8, 0.3),
        "rpm": (1500.0, 20.0),
        "current": (10.0, 0.5),
        "voltage": (415.0, 3.0),
        "bearing_temperature": (55.0, 3.0),
        "torque": (25.0, 2.0),
        "lubrication_level": (85.0, 3.0),
    },
    "Early Warning": {
        "temperature": (75.0, 3.0),
        "vibration": (3.5, 0.5),
        "rpm": (1440.0, 30.0),
        "current": (11.0, 0.5),
        "voltage": (413.0, 3.0),
        "bearing_temperature": (70.0, 3.0),
        "torque": (24.0, 3.0),
        "lubrication_level": (70.0, 3.0),
    },
    "Overheating": {
        "temperature": (95.0, 4.0),
        "vibration": (2.2, 0.4),
        "rpm": (1480.0, 25.0),
        "current": (11.0, 0.6),
        "voltage": (414.0, 3.0),
        "bearing_temperature": (90.0, 4.0),
        "torque": (26.0, 2.0),
        "lubrication_level": (80.0, 3.0),
    },
    "Bearing Failure": {
        "temperature": (70.0, 4.0),
        "vibration": (8.5, 1.0),
        "rpm": (1400.0, 50.0),
        "current": (11.5, 0.7),
        "voltage": (413.0, 3.0),
        "bearing_temperature": (88.0, 5.0),
        "torque": (23.0, 3.0),
        "lubrication_level": (55.0, 5.0),
    },
    "Electrical Fault": {
        "temperature": (68.0, 4.0),
        "vibration": (2.5, 0.5),
        "rpm": (1460.0, 40.0),
        "current": (16.5, 1.2),
        "voltage": (395.0, 8.0),
        "bearing_temperature": (60.0, 4.0),
        "torque": (20.0, 3.0),
        "lubrication_level": (82.0, 3.0),
    },
    "Lubrication Failure": {
        "temperature": (72.0, 4.0),
        "vibration": (6.0, 0.8),
        "rpm": (1420.0, 35.0),
        "current": (11.0, 0.6),
        "voltage": (414.0, 3.0),
        "bearing_temperature": (82.0, 4.0),
        "torque": (22.0, 3.0),
        "lubrication_level": (30.0, 5.0),
    },
    "Motor Misalignment": {
        "temperature": (68.0, 3.0),
        "vibration": (5.5, 0.9),
        "rpm": (1380.0, 70.0),
        "current": (11.0, 0.6),
        "voltage": (414.0, 3.0),
        "bearing_temperature": (65.0, 3.0),
        "torque": (30.0, 5.0),
        "lubrication_level": (78.0, 3.0),
    },
}

DEFAULT_LIFECYCLE_TOTAL_HOURS = 1200.0


class VirtualMotor:
    """A virtual industrial motor whose 8 sensor values can be simulated
    under different diagnostic scenarios, either directly, as a smooth
    random walk, or as a progressive lifecycle degradation toward a
    chosen target fault."""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Direct scenario sampling
    # ------------------------------------------------------------------
    def read(self, scenario: str = "Healthy") -> dict:
        """Return one simulated sensor reading for the given scenario."""
        profile = SCENARIOS.get(scenario, SCENARIOS["Healthy"])
        reading = {}
        for sensor in SENSOR_NAMES:
            mean, std = profile[sensor]
            value = float(self.rng.normal(mean, std))
            if sensor == "vibration":
                value = max(0.0, value)
            if sensor == "lubrication_level":
                value = float(np.clip(value, 0.0, 100.0))
            reading[sensor] = round(value, 2)
        return reading

    # ------------------------------------------------------------------
    # Smooth random walk (used in "Manual scenario" live mode)
    # ------------------------------------------------------------------
    def read_random_walk(self, previous: dict, scenario: str = "Healthy", drift: float = 0.35) -> dict:
        """Generate the next reading as a small drift from the previous one,
        pulled toward the target scenario profile — produces smoother,
        more realistic live charts than fully independent samples."""
        profile = SCENARIOS.get(scenario, SCENARIOS["Healthy"])
        reading = {}
        for sensor in SENSOR_NAMES:
            target_mean, target_std = profile[sensor]
            prev_val = previous.get(sensor, target_mean)
            pull = (target_mean - prev_val) * 0.15
            noise = self.rng.normal(0, target_std * drift)
            value = prev_val + pull + noise
            if sensor == "vibration":
                value = max(0.0, value)
            if sensor == "lubrication_level":
                value = float(np.clip(value, 0.0, 100.0))
            reading[sensor] = round(value, 2)
        return reading

    # ------------------------------------------------------------------
    # Progressive lifecycle degradation
    # ------------------------------------------------------------------
    def read_lifecycle(
        self,
        operating_hours: float,
        target_fault: str = "Bearing Failure",
        total_hours: float = DEFAULT_LIFECYCLE_TOTAL_HOURS,
    ) -> dict:
        """Simulate a motor that starts Healthy at hour 0 and gradually
        degrades toward `target_fault` as operating_hours accumulates,
        reaching full fault severity at `total_hours`. This mimics a
        realistic equipment lifecycle instead of instantly-random data."""
        healthy_profile = SCENARIOS["Healthy"]
        fault_profile = SCENARIOS.get(target_fault, SCENARIOS["Bearing Failure"])
        frac = float(np.clip(operating_hours / total_hours, 0.0, 1.0))

        reading = {}
        for sensor in SENSOR_NAMES:
            h_mean, h_std = healthy_profile[sensor]
            f_mean, f_std = fault_profile[sensor]
            blended_mean = h_mean * (1 - frac) + f_mean * frac
            blended_std = h_std * (1 - frac) + f_std * frac
            value = float(self.rng.normal(blended_mean, blended_std * 0.6))
            if sensor == "vibration":
                value = max(0.0, value)
            if sensor == "lubrication_level":
                value = float(np.clip(value, 0.0, 100.0))
            reading[sensor] = round(value, 2)
        return reading
