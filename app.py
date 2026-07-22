"""
app.py
Digital Twin Dashboard — Predictive Maintenance & Industrial
Equipment Health Monitoring (v2)

Run with:
    streamlit run app.py
"""

import os
import joblib
import time
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from simulator import VirtualMotor, SCENARIOS, FAULT_TYPES, SENSOR_NAMES, SENSOR_LABELS, SENSOR_UNITS
from health import calculate_health
from maintenance import get_recommendation
from anomaly import train_anomaly_model, score_reading
from fleet import generate_fleet_snapshot, fleet_kpis, FLEET_LAYOUT
import db
import auth
from motor_visual import render_motor_visual

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Motor Digital Twin | Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

db.init_db()

# ----------------------------------------------------------------------
# Login gate — everything below this line only runs once signed in
# ----------------------------------------------------------------------
if not auth.is_authenticated():
    auth.render_login_page()
    st.stop()

CURRENT_USER = auth.current_user()
CURRENT_ROLE = auth.current_role()
IS_ADMIN = auth.has_permission("can_clear_data")

MODEL_PATH = "models/predictive_maintenance_model.pkl"
N_TRAIN_SAMPLES = 900 * len(SCENARIOS)  # matches train_model.py sampling

# ----------------------------------------------------------------------
# Styling — light theme, soft palette, clean type
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --ink: #17221D;
        --ink-soft: #4A5A52;
        --muted: #8B9A92;
        --line: #E5EAE7;
        --surface: #FFFFFF;
        --canvas: #F6F8F6;
        --accent: #2F6F63;
        --accent-soft: #E4F0EC;
        --radius: 14px;
        --shadow: 0 1px 2px rgba(23,34,29,0.04), 0 4px 16px rgba(23,34,29,0.05);
        --shadow-hover: 0 4px 10px rgba(23,34,29,0.05), 0 10px 28px rgba(23,34,29,0.09);
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background: var(--canvas); }
    h1, h2, h3 { font-family: 'Poppins', sans-serif !important; color: var(--ink); }

    /* Overall page rhythm */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3.5rem !important;
        max-width: 1320px;
    }
    div[data-testid="stVerticalBlock"] { gap: 0.9rem; }
    div[data-testid="stHorizontalBlock"] { gap: 22px; align-items: stretch; }
    div[data-testid="column"] > div { height: 100%; }

    /* ---------------- Header ---------------- */
    .app-eyebrow {
        font-family: 'Poppins', sans-serif; font-weight: 600; font-size: 12px;
        letter-spacing: 0.14em; text-transform: uppercase; color: var(--accent);
        display: flex; align-items: center; gap: 10px; margin-bottom: 6px;
    }
    .app-eyebrow::before { content: ""; width: 22px; height: 3px; background: var(--accent); border-radius: 2px; }
    .app-header-wrap h1 {
        font-weight: 700; font-size: 32px; margin: 0 0 6px 0; letter-spacing: -0.01em;
    }
    .app-subtitle { color: var(--ink-soft); font-size: 15px; margin-bottom: 22px; max-width: 720px; line-height: 1.55; }
    .app-header-wrap { border-bottom: 1px solid var(--line); padding-bottom: 18px; margin-bottom: 22px; }

    /* ---------------- Status / motor visual ---------------- */
    .motor-visual {
        display: flex; align-items: center; gap: 14px;
        background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius);
        padding: 16px 22px; margin-bottom: 24px;
        box-shadow: var(--shadow);
    }
    .motor-dot { width: 20px; height: 20px; border-radius: 50%; animation: pulse 1.6s infinite; flex-shrink: 0; }
    .dot-healthy { background: #2E9A5D; box-shadow: 0 0 0 0 rgba(46,154,93,0.5); }
    .dot-warning { background: #D9A441; box-shadow: 0 0 0 0 rgba(217,164,65,0.5); }
    .dot-critical { background: #C1443C; box-shadow: 0 0 0 0 rgba(193,68,60,0.5); }
    @keyframes pulse {
        0%   { box-shadow: 0 0 0 0 rgba(0,0,0,0.22); }
        70%  { box-shadow: 0 0 0 12px rgba(0,0,0,0); }
        100% { box-shadow: 0 0 0 0 rgba(0,0,0,0); }
    }
    .motor-visual-text { font-family: 'Poppins', sans-serif; font-weight: 600; font-size: 16px; }
    .motor-visual-sub { font-size: 12.5px; color: var(--muted); margin-top: 2px; }

    /* ---------------- Cards ---------------- */
    .kpi-card {
        background: var(--surface); border-radius: var(--radius); padding: 20px 22px;
        border: 1px solid var(--line); box-shadow: var(--shadow);
        text-align: left; height: 100%; transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-label {
        font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.07em;
        color: var(--muted); font-weight: 600; margin-bottom: 10px;
    }
    .kpi-value { font-family: 'Poppins', sans-serif; font-size: 26px; font-weight: 700; color: var(--ink); }
    .kpi-value-sm { font-family: 'Poppins', sans-serif; font-size: 19px; font-weight: 600; color: var(--ink); }
    .kpi-unit { font-size: 12.5px; color: var(--muted); font-weight: 500; }

    .status-pill {
        display: inline-block; padding: 6px 16px; border-radius: 999px;
        font-family: 'Poppins', sans-serif; font-weight: 600; font-size: 13px; letter-spacing: 0.02em;
    }
    .status-healthy { background: #E4F3EB; color: #1E7A4C; }
    .status-warning { background: #FDF1DC; color: #A8681B; }
    .status-critical { background: #FBE6E6; color: #B3261E; }

    .urgency-none { background:#E4F3EB; color:#1E7A4C; }
    .urgency-low { background:#E7F0FA; color:#2A5C8A; }
    .urgency-medium { background:#FDF1DC; color:#A8681B; }
    .urgency-high { background:#FBE9DE; color:#B85C1E; }
    .urgency-critical { background:#FBE6E6; color:#B3261E; }

    /* ---------------- Section headings ---------------- */
    .section-title {
        font-family: 'Poppins', sans-serif; font-weight: 600; font-size: 18px;
        color: var(--ink); margin-top: 42px; margin-bottom: 4px; letter-spacing: -0.005em;
    }
    .section-note { color: var(--muted); font-size: 13px; margin-bottom: 18px; line-height: 1.5; }

    .reason-chip {
        display: inline-block; background: var(--accent-soft); color: #2B4A41; border-radius: 999px;
        padding: 5px 13px; margin: 3px 6px 3px 0; font-size: 12.5px; font-weight: 500;
    }
    .rec-card {
        background: var(--surface); border-radius: var(--radius); padding: 22px 24px;
        border: 1px solid var(--line); box-shadow: var(--shadow);
    }
    .mono-note { font-family: 'JetBrains Mono', monospace; font-size: 12.5px; color: #7A8A82; line-height: 1.6; }

    /* ---------------- Sidebar ---------------- */
    section[data-testid="stSidebar"] {
        background: #FBFCFB; border-right: 1px solid var(--line);
    }
    section[data-testid="stSidebar"] .block-container { padding-top: 2rem; padding-right: 1.1rem; }
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] { gap: 0.7rem; }

    .side-brand {
        font-family: 'Poppins', sans-serif; font-weight: 700; font-size: 16.5px; color: var(--ink);
        letter-spacing: -0.01em; margin-bottom: 2px;
    }
    .side-brand-sub {
        font-size: 11.5px; color: var(--muted); letter-spacing: 0.03em; margin-bottom: 20px;
        padding-bottom: 18px; border-bottom: 1px solid var(--line);
    }
    .side-section-label {
        font-family: 'Poppins', sans-serif; font-weight: 600; font-size: 11px;
        letter-spacing: 0.1em; text-transform: uppercase; color: var(--accent);
        margin-top: 22px; margin-bottom: 10px;
    }
    .side-section-label.first { margin-top: 4px; }
    .side-hint { font-size: 12px; color: var(--muted); line-height: 1.55; margin-top: 6px; margin-bottom: 4px; }
    .side-divider { height: 1px; background: var(--line); margin: 22px 0; border: none; }
    .side-readout {
        font-family: 'JetBrains Mono', monospace; font-size: 12px; color: var(--ink-soft);
        background: var(--accent-soft); border-radius: 8px; padding: 8px 12px; display: inline-block;
    }

    section[data-testid="stSidebar"] .stButton button {
        border-radius: 10px; border: 1px solid var(--line); font-weight: 500;
        background: var(--surface); color: var(--ink);
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        border-color: var(--accent); color: var(--accent);
    }
    section[data-testid="stSidebar"] .stDownloadButton button {
        border-radius: 10px; font-weight: 500;
    }

    div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; border: 1px solid var(--line); }

    /* Hide the hosted-app chrome (Deploy button / three-dot menu) */
    .stDeployButton, [data-testid="stAppDeployButton"],
    [data-testid="stToolbarActionButton"], #MainMenu { display: none !important; }
    header[data-testid="stHeader"] { background: transparent; }

    /* Card lift + hover interaction */
    .kpi-card:hover, .fleet-card:hover, .rec-card:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-hover);
    }

    /* ---------------- Tabs ---------------- */
    div[data-testid="stTabs"] div[role="tablist"] {
        gap: 26px; border-bottom: 1px solid var(--line); margin-bottom: 14px;
    }
    button[data-testid="stTab"] {
        font-family: 'Poppins', sans-serif; font-weight: 600; font-size: 14.5px;
        color: var(--ink-soft); padding: 10px 16px; border-radius: 10px 10px 0 0;
    }
    button[data-testid="stTab"][aria-selected="true"] {
        color: var(--accent); background: var(--accent-soft);
    }

    /* ---------------- Fleet cards ---------------- */
    .fleet-card {
        background: var(--surface); border-radius: var(--radius); padding: 20px 22px;
        border: 1px solid var(--line); box-shadow: var(--shadow);
        height: 100%; transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .fleet-name { font-family: 'Poppins', sans-serif; font-weight: 600; font-size: 15px; color: var(--ink); }
    .fleet-sub { font-size: 12px; color: var(--muted); margin-bottom: 10px; margin-top: 2px; }

    .metric-strip { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 10px; }
    .metric-chip {
        flex: 1 1 160px; background: var(--surface); border: 1px solid var(--line); border-radius: 12px;
        padding: 16px 18px; box-shadow: var(--shadow);
    }
    .metric-chip .kpi-label { margin-bottom: 6px; }

    .alert-row {
        border-left: 4px solid #C1443C; background: var(--surface); border-radius: 10px;
        padding: 14px 18px; margin-bottom: 10px; box-shadow: var(--shadow);
        line-height: 1.5;
    }
    .alert-row.medium { border-left-color: #D9A441; }
    .alert-row.low { border-left-color: #2A5C8A; }

    /* ---------------- Footer ---------------- */
    .app-footer { margin-top: 48px; padding-top: 20px; border-top: 1px solid var(--line); color: var(--muted); font-size: 12.5px; }

    /* ---------------- Responsive ---------------- */
    @media (max-width: 1024px) {
        .block-container { padding-left: 1.4rem !important; padding-right: 1.4rem !important; }
    }
    @media (max-width: 700px) {
        .app-header-wrap h1 { font-size: 23px !important; }
        .kpi-value { font-size: 20px; }
        .motor-visual { flex-wrap: wrap; }
        .metric-chip { flex: 1 1 100%; }
        .section-title { margin-top: 30px; }
        div[data-testid="stHorizontalBlock"] { gap: 14px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Load / train ML model (auto-trains on first run if missing)
# ----------------------------------------------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        import train_model
        train_model.main()
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_anomaly_model():
    return train_anomaly_model()


model = load_model()
anomaly_model = load_anomaly_model()
db.init_db()

# ----------------------------------------------------------------------
# Session state initialisation
# ----------------------------------------------------------------------
DEFAULT_READING = {
    "temperature": 60.0, "vibration": 1.8, "rpm": 1500.0, "current": 10.0,
    "voltage": 415.0, "bearing_temperature": 55.0, "torque": 25.0, "lubrication_level": 85.0,
}

if "motor" not in st.session_state:
    st.session_state.motor = VirtualMotor(seed=None)
if "history" not in st.session_state:
    st.session_state.history = []
if "last_reading" not in st.session_state:
    st.session_state.last_reading = dict(DEFAULT_READING)
if "running" not in st.session_state:
    st.session_state.running = False
if "operating_hours" not in st.session_state:
    st.session_state.operating_hours = 0.0


def diagnose(reading: dict):
    health = calculate_health(**reading)
    X = pd.DataFrame([reading])[SENSOR_NAMES]
    prediction = model.predict(X)[0]
    proba = dict(zip(model.classes_, model.predict_proba(X)[0]))
    recommendation = get_recommendation(prediction)
    return health, prediction, proba, recommendation


def log_alert_if_needed(reading: dict, health: dict, prediction: str, recommendation: dict):
    """Write an entry to the alert log whenever a reading isn't Healthy —
    mirrors how a CMMS/SCADA alerting pipeline would notify engineers."""
    if health["status"] == "Healthy":
        return
    anomaly = score_reading(anomaly_model, reading)
    db.insert_alert({
        "timestamp": pd.Timestamp.now().isoformat(),
        "machine": "Motor-01 · Live Twin",
        "status": health["status"],
        "prediction": prediction,
        "urgency": recommendation["urgency"],
        "health_score": health["health_score"],
        "anomaly_score": anomaly["normalcy_index"],
        "message": recommendation["action"],
    })
    toast_icon = "🔴" if health["status"] == "Critical" else "🟠"
    st.toast(f"{toast_icon} New alert — {health['status']}: {prediction}", icon=toast_icon)


def take_reading(mode: str, scenario: str, target_fault: str, total_hours: float, hours_step: float):
    if mode == "Lifecycle Degradation":
        st.session_state.operating_hours += hours_step
        new_reading = st.session_state.motor.read_lifecycle(
            st.session_state.operating_hours, target_fault, total_hours
        )
    else:
        new_reading = st.session_state.motor.read_random_walk(
            st.session_state.last_reading, scenario
        )

    st.session_state.last_reading = new_reading
    health, prediction, proba, recommendation = diagnose(new_reading)

    record = {
        "timestamp": pd.Timestamp.now().strftime("%H:%M:%S"),
        **new_reading,
        "health_score": health["health_score"],
        "status": health["status"],
        "ml_prediction": prediction,
    }
    st.session_state.history.append(record)
    st.session_state.history = st.session_state.history[-150:]

    db.insert_reading({
        "timestamp": pd.Timestamp.now().isoformat(),
        "mode": mode,
        "operating_hours": round(st.session_state.operating_hours, 1),
        **new_reading,
        "health_score": health["health_score"],
        "status": health["status"],
        "ml_prediction": prediction,
    })

    log_alert_if_needed(new_reading, health, prediction, recommendation)

    return new_reading, health, prediction, proba, recommendation


def fast_forward(mode, scenario, target_fault, total_hours, hours_step, steps=20):
    """Run several simulation steps back-to-back (no sleep) so the motor
    converges close to the selected condition immediately, instead of
    requiring many manual clicks. Each intermediate step is still recorded,
    so the trend charts show a nice realistic convergence curve."""
    result = None
    for _ in range(steps):
        result = take_reading(mode, scenario, target_fault, total_hours, hours_step)
    return result


# ----------------------------------------------------------------------
# Sidebar controls
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="side-brand">Digital Twin</div>
        <div class="side-brand-sub">Predictive Maintenance Simulator</div>
        """,
        unsafe_allow_html=True,
    )

    role_pill_class = "status-healthy" if IS_ADMIN else "status-warning"
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:14px;">
            <div>
                <div style="font-family:'Poppins',sans-serif; font-weight:600; font-size:13.5px; color:var(--ink);">{CURRENT_USER['display_name']}</div>
                <span class="status-pill {role_pill_class}" style="font-size:11px; padding:3px 10px; margin-top:2px; display:inline-block;">{CURRENT_ROLE}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Log out", use_container_width=True):
        auth.logout()
        st.rerun()

    st.markdown('<hr class="side-divider">', unsafe_allow_html=True)

    st.markdown('<div class="side-section-label first">Simulation Mode</div>', unsafe_allow_html=True)

    mode = st.radio(
        "Simulation mode",
        options=["Manual Scenario", "Lifecycle Degradation"],
        label_visibility="collapsed",
        help=(
            "Manual: jump the motor to any condition instantly.\n"
            "Lifecycle: motor starts Healthy and gradually degrades toward a "
            "chosen fault as operating hours accumulate — mimics a real "
            "equipment lifecycle instead of random jumps."
        ),
    )

    scenario = "Healthy"
    target_fault = "Bearing Failure"
    total_hours = 1200.0
    hours_step = 15.0

    if mode == "Manual Scenario":
        scenario = st.selectbox("Simulate condition", options=list(SCENARIOS.keys()))
    else:
        target_fault = st.selectbox(
            "Fault developing over time", options=[f for f in FAULT_TYPES if f != "Healthy"]
        )
        total_hours = st.slider("Motor lifespan until full failure (hours)", 400, 2000, 1200, 100)
        hours_step = st.slider("Hours advanced per reading", 5, 50, 15, 5)
        st.markdown(
            f'<span class="side-readout">Operating hours &nbsp;{st.session_state.operating_hours:.0f} / {total_hours:.0f}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="side-hint">Starts Healthy by design, like a real new motor. It takes several '
            'readings — or Live auto-simulation — to degrade. Use Fast-forward below to jump ahead.</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="side-section-label">Playback</div>', unsafe_allow_html=True)
    st.session_state.running = st.toggle("Live auto-simulation", value=st.session_state.running)
    refresh_rate = st.slider("Update interval (seconds)", 0.5, 3.0, 1.0, 0.5)

    st.markdown('<hr class="side-divider">', unsafe_allow_html=True)

    st.markdown('<div class="side-section-label first">Readings</div>', unsafe_allow_html=True)
    if st.button("Take single reading", use_container_width=True):
        take_reading(mode, scenario, target_fault, total_hours, hours_step)

    fast_forward_clicked = st.button(
        "Fast-forward to this condition",
        use_container_width=True,
        help="Jumps ahead several steps instantly so you don't have to click repeatedly to see the selected condition develop.",
    )

    st.markdown('<div class="side-section-label">Data</div>', unsafe_allow_html=True)
    if IS_ADMIN:
        if st.button("Reset session", use_container_width=True):
            st.session_state.history = []
            st.session_state.last_reading = dict(DEFAULT_READING)
            st.session_state.operating_hours = 0.0
            st.rerun()

        if st.button("Clear stored history", use_container_width=True):
            db.clear_all()
            st.rerun()
    else:
        st.markdown(
            '<div class="side-hint">Resetting the session and clearing stored history requires an '
            '<b>Admin</b> login.</div>',
            unsafe_allow_html=True,
        )

    if st.session_state.history:
        df_download = pd.DataFrame(st.session_state.history)
        st.download_button(
            "Download session readings (CSV)",
            data=df_download.to_csv(index=False).encode("utf-8"),
            file_name="motor_digital_twin_readings.csv",
            mime="text/csv",
            use_container_width=True,
        )

# ----------------------------------------------------------------------
# Detect scenario / fault / mode changes so the dashboard doesn't keep
# showing a stale (often "Healthy") reading after you pick something new
# ----------------------------------------------------------------------
if "prev_mode" not in st.session_state:
    st.session_state.prev_mode = mode
if "prev_scenario" not in st.session_state:
    st.session_state.prev_scenario = scenario
if "prev_target_fault" not in st.session_state:
    st.session_state.prev_target_fault = target_fault

selection_changed = (
    mode != st.session_state.prev_mode
    or (mode == "Manual Scenario" and scenario != st.session_state.prev_scenario)
    or (mode == "Lifecycle Degradation" and target_fault != st.session_state.prev_target_fault)
)
st.session_state.prev_mode = mode
st.session_state.prev_scenario = scenario
st.session_state.prev_target_fault = target_fault

# ----------------------------------------------------------------------
# Auto-simulation step
# ----------------------------------------------------------------------
if fast_forward_clicked:
    reading, health, prediction, proba, recommendation = fast_forward(
        mode, scenario, target_fault, total_hours, hours_step, steps=25
    )
elif selection_changed and mode == "Manual Scenario":
    # Snap quickly to the newly selected condition instead of requiring
    # ~15-20 manual clicks to drift there one small step at a time.
    reading, health, prediction, proba, recommendation = fast_forward(
        mode, scenario, target_fault, total_hours, hours_step, steps=20
    )
elif st.session_state.running or not st.session_state.history:
    reading, health, prediction, proba, recommendation = take_reading(
        mode, scenario, target_fault, total_hours, hours_step
    )
else:
    reading = st.session_state.last_reading
    health, prediction, proba, recommendation = diagnose(reading)

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header-wrap">
        <h1>Digital Twin Control Center</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

status_class = {"Healthy": "status-healthy", "Warning": "status-warning", "Critical": "status-critical"}[health["status"]]
dot_class = {"Healthy": "dot-healthy", "Warning": "dot-warning", "Critical": "dot-critical"}[health["status"]]


# ----------------------------------------------------------------------
# Main navigation — Live Monitor / Fleet Overview / Alerts / Model Insights
# ----------------------------------------------------------------------
if IS_ADMIN:
    tab_live, tab_fleet, tab_alerts, tab_insights = st.tabs(
        ["Live Monitor", "Fleet Overview", "Alerts Center", "Model Insights"]
    )
else:
    tab_live, tab_fleet, tab_alerts = st.tabs(
        ["Live Monitor", "Fleet Overview", "Alerts Center"]
    )
    tab_insights = None

with tab_live:
    # ----------------------------------------------------------------------
    # Digital Twin visual status indicator
    # ----------------------------------------------------------------------
    mode_note = (
        f"Lifecycle mode · target fault: {target_fault} · {st.session_state.operating_hours:.0f} operating hours"
        if mode == "Lifecycle Degradation"
        else f"Manual scenario · {scenario}"
    )
    st.markdown(
        f"""
        <div class="motor-visual">
            <div class="motor-dot {dot_class}"></div>
            <div>
                <div class="motor-visual-text">Motor status: {health['status']} — {prediction}</div>
                <div class="motor-visual-sub">{mode_note}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ----------------------------------------------------------------------
    # Interactive motor graphic — housing colour reacts to temperature,
    # rotor speed reacts to RPM, instead of only showing numbers/charts.
    # ----------------------------------------------------------------------
    st.markdown(
        render_motor_visual(
            temperature=reading["temperature"],
            rpm=reading["rpm"],
            vibration=reading["vibration"],
            status=health["status"],
            prediction=prediction,
        ),
        unsafe_allow_html=True,
    )

    # ----------------------------------------------------------------------
    # Live Alerts strip — the most recent alerts surface right here on the
    # live monitor (not only inside the Alerts Center tab), plus a toast
    # notification fires the instant a new non-healthy reading is logged.
    # ----------------------------------------------------------------------
    live_alerts_df = db.get_alerts(3)
    if not live_alerts_df.empty:
        st.markdown('<div class="section-title" style="margin-top:26px;">Live Alerts</div>', unsafe_allow_html=True)
        urgency_css_map = {"Critical": "", "High": "", "Medium": "medium", "Low": "low"}
        for _, a in live_alerts_df.iterrows():
            css = urgency_css_map.get(a["urgency"], "")
            st.markdown(
                f"""<div class="alert-row {css}">
                <b>{a['timestamp'][:19].replace('T', ' ')}</b> — {a['machine']} ·
                <span class="mono-note">{a['status']} / {a['prediction']}</span> ·
                Urgency: <b>{a['urgency']}</b><br>
                <span style="font-size:13px;">{a['message']}</span>
                </div>""",
                unsafe_allow_html=True,
            )
        st.caption("Showing the 3 most recent alerts — full history lives in the Alerts Center tab.")

    # ----------------------------------------------------------------------
    # KPI row 1 — core sensors + health
    # ----------------------------------------------------------------------
    st.markdown('<div class="section-title" style="margin-top:6px;">Core Metrics</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Temperature</div>
            <div class="kpi-value">{reading['temperature']}°<span class="kpi-unit">C</span></div></div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Vibration</div>
            <div class="kpi-value">{reading['vibration']} <span class="kpi-unit">mm/s</span></div></div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Rotational speed</div>
            <div class="kpi-value">{reading['rpm']} <span class="kpi-unit">RPM</span></div></div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-label">Machine health</div>
            <div class="kpi-value">{health['health_score']}<span class="kpi-unit">%</span></div>
            <span class="status-pill {status_class}">{health['status']}</span></div>""", unsafe_allow_html=True)

    # ----------------------------------------------------------------------
    # KPI row 2 — additional industrial sensors
    # ----------------------------------------------------------------------
    st.markdown('<div class="section-title">Additional Industrial Sensors</div>', unsafe_allow_html=True)
    a1, a2, a3, a4, a5 = st.columns(5)
    extra_sensors = [
        (a1, "current", "A"),
        (a2, "voltage", "V"),
        (a3, "bearing_temperature", "°C"),
        (a4, "torque", "Nm"),
        (a5, "lubrication_level", "%"),
    ]
    for col, sensor, unit in extra_sensors:
        with col:
            st.markdown(
                f"""<div class="kpi-card"><div class="kpi-label">{SENSOR_LABELS[sensor]}</div>
                <div class="kpi-value-sm">{reading[sensor]} <span class="kpi-unit">{unit}</span></div></div>""",
                unsafe_allow_html=True,
            )

    # ----------------------------------------------------------------------
    # Gauges — 3 primary sensors
    # ----------------------------------------------------------------------
    st.markdown('<div class="section-title">Live Sensor Gauges</div>', unsafe_allow_html=True)
    gauge_colors = {"bar": "#2F6F63"}


    def make_gauge(value, title, value_range, steps, suffix=""):
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=value,
            number={"suffix": suffix, "font": {"size": 26, "family": "Poppins"}},
            title={"text": title, "font": {"size": 15, "family": "Inter", "color": "#445048"}},
            gauge={"axis": {"range": value_range, "tickcolor": "#8B9A92"},
                   "bar": {"color": gauge_colors["bar"], "thickness": 0.28},
                   "bgcolor": "white", "borderwidth": 0, "steps": steps},
        ))
        fig.update_layout(height=220, margin=dict(l=20, r=20, t=50, b=10),
                           paper_bgcolor="rgba(0,0,0,0)", font={"color": "#1F2A24"})
        return fig


    g1, g2, g3 = st.columns(3)
    with g1:
        st.plotly_chart(make_gauge(reading["temperature"], "Temperature (°C)", [0, 110],
            [{"range": [0, 75], "color": "#E4F3EB"}, {"range": [75, 88], "color": "#FDF1DC"}, {"range": [88, 110], "color": "#FBE6E6"}]),
            use_container_width=True)
    with g2:
        st.plotly_chart(make_gauge(reading["vibration"], "Vibration (mm/s)", [0, 10],
            [{"range": [0, 4], "color": "#E4F3EB"}, {"range": [4, 7], "color": "#FDF1DC"}, {"range": [7, 10], "color": "#FBE6E6"}]),
            use_container_width=True)
    with g3:
        st.plotly_chart(make_gauge(reading["rpm"], "Speed (RPM)", [1000, 1650],
            [{"range": [1000, 1150], "color": "#FBE6E6"}, {"range": [1150, 1350], "color": "#FDF1DC"}, {"range": [1350, 1650], "color": "#E4F3EB"}]),
            use_container_width=True)

    # ----------------------------------------------------------------------
    # Historical trend charts (session)
    # ----------------------------------------------------------------------
    st.markdown('<div class="section-title">Sensor Trends (this session)</div>', unsafe_allow_html=True)

    if st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)

        def trend_chart(y_col, name, color, unit):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist_df.index, y=hist_df[y_col], mode="lines", name=name,
                line=dict(color=color, width=2.5, shape="spline"),
                fill="tozeroy", fillcolor=color.replace(")", ", 0.08)").replace("rgb", "rgba"),
            ))
            fig.update_layout(height=210, margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                title=dict(text=f"{name} ({unit})", font=dict(size=14, family="Inter", color="#445048")),
                xaxis=dict(showgrid=False, showticklabels=False),
                yaxis=dict(showgrid=True, gridcolor="#EFF3F1"), showlegend=False)
            return fig

        t1, t2, t3 = st.columns(3)
        with t1:
            st.plotly_chart(trend_chart("temperature", "Temperature", "rgb(47,111,99)", "°C"), use_container_width=True)
        with t2:
            st.plotly_chart(trend_chart("vibration", "Vibration", "rgb(168,104,27)", "mm/s"), use_container_width=True)
        with t3:
            st.plotly_chart(trend_chart("rpm", "RPM", "rgb(58,93,143)", "rpm"), use_container_width=True)

        with st.expander("Show additional sensor trends (Current, Voltage, Bearing Temp, Torque, Lubrication)"):
            e1, e2, e3 = st.columns(3)
            with e1:
                st.plotly_chart(trend_chart("current", "Current", "rgb(122,58,150)", "A"), use_container_width=True)
            with e2:
                st.plotly_chart(trend_chart("voltage", "Voltage", "rgb(58,130,140)", "V"), use_container_width=True)
            with e3:
                st.plotly_chart(trend_chart("bearing_temperature", "Bearing Temp", "rgb(183,71,42)", "°C"), use_container_width=True)
            e4, e5 = st.columns(2)
            with e4:
                st.plotly_chart(trend_chart("torque", "Torque", "rgb(90,110,60)", "Nm"), use_container_width=True)
            with e5:
                st.plotly_chart(trend_chart("lubrication_level", "Lubrication Level", "rgb(60,90,150)", "%"), use_container_width=True)

    # ----------------------------------------------------------------------
    # ML Diagnosis + Maintenance recommendation panel
    # ----------------------------------------------------------------------
    st.markdown('<div class="section-title">Predictive Maintenance — Model Diagnosis</div>', unsafe_allow_html=True)

    p1, p2 = st.columns([1, 1.4])
    with p1:
        pred_class = {
            "Healthy": "status-healthy", "Early Warning": "status-warning",
        }.get(prediction, "status-critical")
        st.markdown(
            f"""<div class="kpi-card"><div class="kpi-label">Diagnosed condition</div>
            <span class="status-pill {pred_class}" style="font-size:18px;">{prediction}</span>
            <div style="margin-top:14px;">{''.join(f'<span class="reason-chip">{r}</span>' for r in health['reasons'])}</div>
            <div style="margin-top:14px;" class="mono-note">Estimated Remaining Useful Life: ~{health['estimated_rul_hours']} hours</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with p2:
        fig = go.Figure(go.Bar(
            x=list(proba.values()), y=list(proba.keys()), orientation="h",
            marker_color="#2F6F63",
            text=[f"{v*100:.0f}%" for v in proba.values()], textposition="outside",
        ))
        fig.update_layout(height=260, margin=dict(l=10, r=30, t=30, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(range=[0, 1], showgrid=False, showticklabels=False),
            title=dict(text="Prediction confidence across all conditions", font=dict(size=14, family="Inter", color="#445048")))
        st.plotly_chart(fig, use_container_width=True)

    urgency_class = {
        "None": "urgency-none", "Low": "urgency-low", "Medium": "urgency-medium",
        "High": "urgency-high", "Critical": "urgency-critical",
    }.get(recommendation["urgency"], "urgency-low")

    st.markdown(
        f"""<div class="rec-card">
        <div class="kpi-label">Recommended action</div>
        <span class="status-pill {urgency_class}" style="margin-bottom:8px; display:inline-block;">Urgency: {recommendation['urgency']}</span>
        <div style="font-size:15px; margin-top:6px;">{recommendation['action']}</div>
        </div>""",
        unsafe_allow_html=True,
    )

    # ----------------------------------------------------------------------
    # Historical data from SQLite
    # ----------------------------------------------------------------------
    st.markdown('<div class="section-title">Historical Data</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-note">{db.get_total_count()} readings stored in motor_data.db</div>',
        unsafe_allow_html=True,
    )

    hist_tab1, hist_tab2, hist_tab3 = st.tabs(["Daily Summary", "Failure History", "Recent Log"])
    with hist_tab1:
        daily = db.get_daily_summary()
        if daily.empty:
            st.info("No historical data yet.")
        else:
            st.dataframe(daily, use_container_width=True, hide_index=True)
    with hist_tab2:
        failures = db.get_failure_history(20)
        if failures.empty:
            st.info("No non-healthy events recorded yet.")
        else:
            st.dataframe(
                failures[["timestamp", "mode", "operating_hours", "health_score", "status", "ml_prediction"]],
                use_container_width=True, hide_index=True,
            )
    with hist_tab3:
        recent = db.get_recent(20)
        if recent.empty:
            st.info("No readings yet.")
        else:
            st.dataframe(recent, use_container_width=True, hide_index=True)


with tab_fleet:
    st.markdown('<div class="section-title" style="margin-bottom:22px;">Multi-Machine Fleet Overview</div>', unsafe_allow_html=True)

    fcol1, fcol2 = st.columns([1, 5])
    with fcol1:
        refresh_fleet = st.button("Refresh fleet snapshot", use_container_width=True)

    if "fleet_snapshot" not in st.session_state or refresh_fleet:
        st.session_state.fleet_snapshot = generate_fleet_snapshot(diagnose)

    fleet_df = st.session_state.fleet_snapshot
    kpis = fleet_kpis(fleet_df)

    st.markdown(
        f"""
        <div class="metric-strip">
            <div class="metric-chip"><div class="kpi-label">Availability</div>
                <div class="kpi-value-sm">{kpis['availability']}<span class="kpi-unit">%</span></div></div>
            <div class="metric-chip"><div class="kpi-label">Performance</div>
                <div class="kpi-value-sm">{kpis['performance']}<span class="kpi-unit">%</span></div></div>
            <div class="metric-chip"><div class="kpi-label">Quality</div>
                <div class="kpi-value-sm">{kpis['quality']}<span class="kpi-unit">%</span></div></div>
            <div class="metric-chip"><div class="kpi-label">Fleet OEE</div>
                <div class="kpi-value-sm">{kpis['oee']}<span class="kpi-unit">%</span></div></div>
            <div class="metric-chip"><div class="kpi-label">Machines needing attention</div>
                <div class="kpi-value-sm">{kpis['attention']} / {len(fleet_df)}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title" style="margin-top:20px;">Machine Cards</div>', unsafe_allow_html=True)
    status_class_map = {"Healthy": "status-healthy", "Warning": "status-warning", "Critical": "status-critical"}
    cols = st.columns(3)
    for i, row in fleet_df.iterrows():
        with cols[i % 3]:
            st.markdown(
                f"""<div class="fleet-card">
                <div class="fleet-name">{row['machine']}</div>
                <div class="fleet-sub">Scenario: {row['scenario']}</div>
                <span class="status-pill {status_class_map.get(row['status'], 'status-warning')}">{row['status']}</span>
                <div style="margin-top:10px;" class="kpi-value-sm">{row['health_score']}<span class="kpi-unit">% health</span></div>
                <div class="mono-note" style="margin-top:6px;">Diagnosis: {row['prediction']} · Urgency: {row['urgency']}</div>
                <div class="mono-note">{row['temperature']}°C · {row['vibration']} mm/s · {row['rpm']} RPM</div>
                </div>""",
                unsafe_allow_html=True,
            )
            st.write("")

    st.markdown('<div class="section-title">Fleet Health Distribution</div>', unsafe_allow_html=True)
    fig_fleet = go.Figure(go.Bar(
        x=fleet_df["machine"], y=fleet_df["health_score"],
        marker_color=["#2E9A5D" if s == "Healthy" else "#D9A441" if s == "Warning" else "#C1443C"
                      for s in fleet_df["status"]],
        text=fleet_df["health_score"], textposition="outside",
    ))
    fig_fleet.update_layout(height=280, margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0, 105], gridcolor="#EFF3F1", title="Health score (%)"))
    st.plotly_chart(fig_fleet, use_container_width=True)


with tab_alerts:
    st.markdown('<div class="section-title" style="margin-bottom:22px;">Alerts Center</div>', unsafe_allow_html=True)

    counts = db.get_alert_counts()
    ac1, ac2, ac3, ac4 = st.columns(4)
    with ac1:
        st.markdown(f"""<div class="metric-chip"><div class="kpi-label">Critical</div>
            <div class="kpi-value-sm">{counts.get('Critical', 0)}</div></div>""", unsafe_allow_html=True)
    with ac2:
        st.markdown(f"""<div class="metric-chip"><div class="kpi-label">High</div>
            <div class="kpi-value-sm">{counts.get('High', 0)}</div></div>""", unsafe_allow_html=True)
    with ac3:
        st.markdown(f"""<div class="metric-chip"><div class="kpi-label">Medium</div>
            <div class="kpi-value-sm">{counts.get('Medium', 0)}</div></div>""", unsafe_allow_html=True)
    with ac4:
        st.markdown(f"""<div class="metric-chip"><div class="kpi-label">Low</div>
            <div class="kpi-value-sm">{counts.get('Low', 0)}</div></div>""", unsafe_allow_html=True)

    st.write("")
    if IS_ADMIN:
        if st.button("Clear alert log"):
            db.clear_alerts()
            st.rerun()
    else:
        st.caption("Clearing the alert log requires an Admin login.")

    alerts_df = db.get_alerts(50)
    if alerts_df.empty:
        st.info("No alerts yet — alerts appear here once the live twin reports a non-healthy condition.")
    else:
        urgency_css = {"Critical": "", "High": "", "Medium": "medium", "Low": "low"}
        for _, a in alerts_df.iterrows():
            css = urgency_css.get(a["urgency"], "")
            st.markdown(
                f"""<div class="alert-row {css}">
                <b>{a['timestamp'][:19].replace('T', ' ')}</b> — {a['machine']} ·
                <span class="mono-note">{a['status']} / {a['prediction']}</span> ·
                Urgency: <b>{a['urgency']}</b><br>
                <span style="font-size:13px;">{a['message']}</span>
                </div>""",
                unsafe_allow_html=True,
            )


if IS_ADMIN:
 with tab_insights:
    st.markdown('<div class="section-title" style="margin-bottom:26px;">Model Insights &amp; Explainability</div>', unsafe_allow_html=True)

    ic1, ic2 = st.columns(2)
    with ic1:
        st.markdown('<div class="section-title" style="margin-top:6px;">Feature Importance (Random Forest)</div>', unsafe_allow_html=True)
        importances = getattr(model, "feature_importances_", None)
        if importances is not None:
            imp_df = pd.DataFrame({
                "sensor": [SENSOR_LABELS[s] for s in SENSOR_NAMES],
                "importance": importances,
            }).sort_values("importance", ascending=True)
            fig_imp = go.Figure(go.Bar(
                x=imp_df["importance"], y=imp_df["sensor"], orientation="h",
                marker_color="#2F6F63",
            ))
            fig_imp.update_layout(height=320, margin=dict(l=10, r=20, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="Relative importance", gridcolor="#EFF3F1"))
            st.plotly_chart(fig_imp, use_container_width=True)

    with ic2:
        st.markdown('<div class="section-title" style="margin-top:6px;">Anomaly Detection (Isolation Forest)</div>', unsafe_allow_html=True)
        anomaly_now = score_reading(anomaly_model, reading)
        anomaly_status = "Anomalous" if anomaly_now["is_anomaly"] else "Normal"
        anomaly_pill = "status-critical" if anomaly_now["is_anomaly"] else "status-healthy"
        st.markdown(
            f"""<div class="kpi-card">
            <div class="kpi-label">Current live-twin reading</div>
            <span class="status-pill {anomaly_pill}" style="font-size:16px;">{anomaly_status}</span>
            <div style="margin-top:12px;" class="kpi-value-sm">{anomaly_now['normalcy_index']}<span class="kpi-unit">/ 100 normalcy index</span></div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-title">Dataset &amp; Model Summary</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="metric-strip">
            <div class="metric-chip"><div class="kpi-label">Training samples</div>
                <div class="kpi-value-sm">{N_TRAIN_SAMPLES:,}</div></div>
            <div class="metric-chip"><div class="kpi-label">Fault classes</div>
                <div class="kpi-value-sm">{len(FAULT_TYPES)}</div></div>
            <div class="metric-chip"><div class="kpi-label">Sensors tracked</div>
                <div class="kpi-value-sm">{len(SENSOR_NAMES)}</div></div>
            <div class="metric-chip"><div class="kpi-label">Classifier</div>
                <div class="kpi-value-sm" style="font-size:15px;">Random Forest</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Roadmap — how this would scale to a full Industry 4.0 deployment"):
        st.markdown(
            """
- **IoT architecture:** Motor → ESP32/PLC → MQTT broker → Cloud ingestion → ML model → Dashboard.
- **Time-series deep learning:** replace/augment the Random Forest with an LSTM over rolling sensor windows for sequence-aware failure prediction.
- **SHAP-based explainability:** per-prediction contribution values instead of only global feature importance.
- **Real notification channels:** wire the Alerts Center up to email/SMS/Slack instead of an in-app log.
- **Cloud deployment:** host on Render/Railway/Azure so the dashboard is reachable outside the local machine.
            """
        )


# ----------------------------------------------------------------------
# Auto-refresh loop for live mode
# ----------------------------------------------------------------------
if st.session_state.running:
    time.sleep(refresh_rate)
    st.rerun()
