"""
maintenance.py
Maps a diagnosed fault type (from the ML model) to a plain-language
maintenance recommendation and an urgency level, similar to how a real
CMMS (Computerized Maintenance Management System) would present alerts.
"""

RECOMMENDATIONS = {
    "Healthy": {
        "urgency": "None",
        "action": "No action needed. Continue routine monitoring.",
    },
    "Early Warning": {
        "urgency": "Low",
        "action": "Schedule a general inspection within the next maintenance window.",
    },
    "Overheating": {
        "urgency": "High",
        "action": "Check cooling fan and ventilation. Verify ambient temperature and airflow around the motor.",
    },
    "Bearing Failure": {
        "urgency": "High",
        "action": "Inspect and re-lubricate bearings; schedule bearing replacement if vibration stays elevated.",
    },
    "Electrical Fault": {
        "urgency": "Critical",
        "action": "Inspect wiring, insulation, and power supply. Check for voltage imbalance or a failing phase.",
    },
    "Lubrication Failure": {
        "urgency": "High",
        "action": "Top up or replace lubricant immediately. Check the lubrication delivery system for blockages.",
    },
    "Motor Misalignment": {
        "urgency": "Medium",
        "action": "Check shaft alignment and coupling. Realign the motor with the driven equipment.",
    },
}


def get_recommendation(fault_type: str) -> dict:
    return RECOMMENDATIONS.get(
        fault_type,
        {"urgency": "Unknown", "action": "Inspect the machine manually to determine the cause."},
    )
