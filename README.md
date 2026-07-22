<<<<<<< HEAD
# Digital Twin for Predictive Maintenance & Industrial Equipment Health Monitoring

A **simulation-based Digital Twin** of an industrial motor — no physical hardware is
used. The app simulates **8 industrial sensors**, diagnoses the machine into one of
**7 specific conditions** (not just Healthy/Warning/Critical), flags **statistical
anomalies** independent of the classifier, recommends maintenance actions, estimates
Remaining Useful Life, logs alerts, monitors a **6-machine fleet** with OEE-style
metrics, persists history in **SQLite**, and can simulate a motor **progressively
degrading over its operating lifetime** — all on a clean, responsive, tabbed
dashboard.

![status](https://img.shields.io/badge/status-simulation--only-2F6F63)

---

## What's new in v3

| Area | Description |
|---|---|
| **Login system** | The dashboard now sits behind a sign-in page (`auth.py`) with two roles — **Admin** (full access, incl. Model Insights and data-clearing controls) and **Operator** (day-to-day monitoring: live readings, fleet view, alerts, but no destructive actions). Demo accounts: `admin` / `admin@123` and `operator` / `operator@123`. |
| **Interactive motor twin** | Live Monitor now shows an animated motor graphic (`motor_visual.py`) instead of only charts — the housing colour shifts from cool teal to amber to red as temperature rises, it glows/pulses once the machine is Critical, the internal rotor spins faster or slower with live RPM, and the whole graphic gets a subtle shake at high vibration. |
| **Live Alerts** | The most recent alerts now surface directly on the Live Monitor tab (a "Live Alerts" strip), in addition to the full Alerts Center tab, and a toast notification pops up the instant a new non-healthy reading is logged. |

---

## What's new in v2

| Area | v1 | v2 |
|---|---|---|
| Sensors | 3 (Temp, Vibration, RPM) | **8** (+ Current, Voltage, Bearing Temp, Torque, Lubrication Level) |
| Diagnosis | Healthy / Warning / Critical | **7 specific conditions**: Healthy, Early Warning, Overheating, Bearing Failure, Electrical Fault, Lubrication Failure, Motor Misalignment |
| Data realism | Random scenario jumps | **Lifecycle Degradation mode** — motor starts Healthy and gradually develops a chosen fault over operating hours |
| Anomaly detection | — | **Isolation Forest** trained on healthy-only data flags statistically abnormal readings, independent of the supervised classifier |
| Fleet monitoring | Single machine | **6-machine fleet view** with Availability / Performance / Quality / OEE metrics |
| Alerting | — | **Alerts Center** — every non-healthy diagnosis is automatically logged with urgency and timestamp |
| Explainability | — | **Model Insights tab** — feature-importance chart + live anomaly/normalcy score |
| Maintenance | — | **Maintenance recommendation card** with urgency level, per diagnosed fault |
| Remaining life | — | Simple **Remaining Useful Life (RUL)** estimate in hours |
| Storage | Session-only | **SQLite** (`motor_data.db`) — readings, alerts, daily summaries, and failure history persist across restarts |
| Layout | Single page | **Tabbed dashboard** — Live Monitor / Fleet Overview / Alerts Center / Model Insights, with responsive styling for smaller screens |

---

## Architecture

```
Virtual Industrial Motor (simulator.py)
   ├── Manual Scenario mode       → instant condition jump + smooth random walk
   ├── Lifecycle Degradation mode → gradual drift from Healthy → chosen fault
   └── Fleet snapshot (fleet.py)  → 6 independent simulated machines
            │
            ▼
8 Simulated Sensors: Temperature, Vibration, RPM, Current, Voltage,
                     Bearing Temperature, Torque, Lubrication Level
            │
     ┌──────┼────────────────┬─────────────────────┐
     ▼      ▼                ▼                      ▼
 Rule-Based  Random Forest    Isolation Forest       Fleet KPIs
 Health/RUL  (7-class fault   (anomaly.py — flags    (fleet.py — OEE-style
             diagnosis)       abnormal readings)     Availability/Performance/
     │           │                 │                 Quality)
     └───────────┼─────────────────┘
                 ▼
   Maintenance Recommendation (maintenance.py)
                 │
                 ▼
   SQLite Storage (db.py) ── readings / alerts / daily summary / failure history
                 │
                 ▼
   Streamlit Dashboard (app.py) — Live Monitor · Fleet Overview ·
                                   Alerts Center · Model Insights
```

---

## Project structure

```
digital-twin-pm/
├── app.py                   # Streamlit dashboard (main entry point, tabbed layout)
├── simulator.py              # Virtual motor: 8 sensors, 7 scenarios, lifecycle mode
├── health.py                  # Rule-based health scoring + RUL estimate
├── maintenance.py             # Fault → maintenance recommendation + urgency
├── anomaly.py                 # Isolation Forest unsupervised anomaly detection
├── fleet.py                   # Multi-machine fleet simulation + OEE-style KPIs
├── db.py                       # SQLite persistence (readings, alerts, summaries)
├── train_model.py             # Generates data + trains the Random Forest model
├── requirements.txt
├── .streamlit/
│   └── config.toml            # Theme + minimal toolbar configuration
├── data/                      # Synthetic training data (generated)
├── models/                    # Saved trained model (generated)
├── motor_data.db               # SQLite database (generated on first run)
└── README.md
```

---

## Setup instructions (Windows / VS Code)

1. **Extract the zip** and open the folder in VS Code.

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the classification model** (creates `models/predictive_maintenance_model.pkl`):
   ```bash
   python train_model.py
   ```
   > Optional — `app.py` auto-trains the model on first run if it's missing. Running
   > it manually first just lets you see the model's classification report for all
   > 7 fault types. The anomaly-detection model trains automatically in the background
   > the first time the dashboard loads.

5. **Run the dashboard:**
   ```bash
   streamlit run app.py
   ```

6. Your browser opens automatically at `http://localhost:8501`.

`motor_data.db` (SQLite) is created automatically the first time you take a reading,
and keeps growing across sessions — restart the app any time and your history is
still there.

---

## How to use the dashboard

**Live Monitor tab**
- **Manual Scenario mode** — instantly jump the motor to any of the 7 conditions
  (Healthy, Early Warning, Overheating, Bearing Failure, Electrical Fault, Lubrication
  Failure, Motor Misalignment) and watch it settle there smoothly.
- **Lifecycle Degradation mode** — pick a target fault and a "lifespan" in hours; each
  reading advances operating hours and the motor gradually drifts from Healthy toward
  that fault, just like a real machine wearing down over time.
- Toggle **"Live auto-simulation"** to stream readings automatically, or use
  **"Take single reading"** to step manually.
- Watch the **gauges**, **trend charts**, **Machine Health %**, **RUL estimate**, the
  **model diagnosis** with confidence bars across all 7 conditions, and the
  **maintenance recommendation card** update in real time.
- The **Historical Data** section (backed by SQLite) shows a daily summary, a failure
  event history, and a recent readings log — all persisted in `motor_data.db`.

**Fleet Overview tab** — six independent simulated machines shown as cards, with
plant-level Availability / Performance / Quality / OEE metrics and a health
distribution chart. Click "Refresh fleet snapshot" to resample.

**Alerts Center tab** — every non-healthy diagnosis on the live twin is automatically
logged here with a timestamp, status, urgency, and recommended action — a lightweight
stand-in for a real notification pipeline.

**Model Insights tab** — a feature-importance chart showing which sensors the
classifier relies on most, a live anomaly/normalcy score from the Isolation Forest,
dataset/model summary stats, and a roadmap of how this would scale to a full
Industry 4.0 deployment.

---

## Resume bullet point

> Developed a Digital Twin simulation of an industrial motor in Python, modeling 8
> sensor parameters (temperature, vibration, RPM, current, voltage, bearing
> temperature, torque, lubrication level) across 7 diagnostic conditions. Built a
> Random Forest classifier for fault diagnosis, an Isolation Forest for unsupervised
> anomaly detection, a rule-based health scoring and Remaining Useful Life engine, a
> multi-machine fleet view with OEE-style metrics, and an alerting system — all
> surfaced on an interactive, tabbed Streamlit dashboard with SQLite-backed historical
> analytics.

**Technologies:** Python · Streamlit · scikit-learn · Plotly · Pandas · NumPy · SQLite ·
Digital Twin · Predictive Maintenance · Anomaly Detection · Industry 4.0

---

## What to say in the interview

> "I developed a simulation-based Digital Twin of an industrial motor because
> physical hardware wasn't available. It models eight sensor parameters and can
> either jump directly to a fault condition or simulate the motor progressively
> degrading over its operating hours, the way a real machine would. A Random Forest
> model diagnoses the specific fault type, and I paired it with an Isolation Forest
> trained only on healthy data so the system can also flag readings that just look
> abnormal, even if they don't match a known fault signature — that's the difference
> between supervised and unsupervised approaches to condition monitoring. I extended
> it to a six-machine fleet view with OEE-style metrics, and every non-healthy
> diagnosis gets logged to an Alerts Center automatically. All of it is persisted in
> SQLite so I can show trends over time, and I mapped out how this would scale to a
> full IoT deployment with real sensors, MQTT, and an LSTM for time-series
> prediction."

---

## Note

This project is a **simulation only** — it does not use real sensors, real motors, or
real hardware. State this plainly in interviews; it's a deliberate, technically honest
framing. The underlying skills — simulation modeling, multi-class classification,
unsupervised anomaly detection, rule-based scoring, database design, and dashboard
engineering — are the same skills a real IoT-based predictive maintenance system would
require.
=======
# Digital_Twin
Simulation-Based Digital Twin for Predictive Maintenance and Industrial Motor Health Monitoring
>>>>>>> 7acc4120078dccd05e0160b04c91d48f0c74dfbb
