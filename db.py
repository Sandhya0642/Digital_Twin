"""
db.py
SQLite persistence layer for the Digital Twin. Every simulated reading is
stored so the dashboard can show historical trends (e.g. daily averages,
failure history) across app restarts, instead of losing data when the
session ends.
"""

import sqlite3
import os
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motor_data.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    mode TEXT,
    operating_hours REAL,
    temperature REAL,
    vibration REAL,
    rpm REAL,
    current REAL,
    voltage REAL,
    bearing_temperature REAL,
    torque REAL,
    lubrication_level REAL,
    health_score REAL,
    status TEXT,
    ml_prediction TEXT
);
"""

ALERTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    machine TEXT,
    status TEXT,
    prediction TEXT,
    urgency TEXT,
    health_score REAL,
    anomaly_score REAL,
    message TEXT,
    acknowledged INTEGER DEFAULT 0
);
"""

USERS_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL,
    display_name TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn


def init_db():
    conn = get_connection()
    conn.execute(SCHEMA)
    conn.execute(ALERTS_SCHEMA)
    conn.execute(USERS_SCHEMA)
    conn.commit()
    conn.close()


def insert_reading(record: dict):
    """Insert one reading dict (must contain all schema columns except id)."""
    conn = get_connection()
    columns = [
        "timestamp", "mode", "operating_hours", "temperature", "vibration", "rpm",
        "current", "voltage", "bearing_temperature", "torque", "lubrication_level",
        "health_score", "status", "ml_prediction",
    ]
    values = [record.get(col) for col in columns]
    placeholders = ", ".join(["?"] * len(columns))
    conn.execute(
        f"INSERT INTO readings ({', '.join(columns)}) VALUES ({placeholders})",
        values,
    )
    conn.commit()
    conn.close()


def get_recent(n: int = 50) -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM readings ORDER BY id DESC LIMIT ?", conn, params=(n,)
    )
    conn.close()
    return df


def get_daily_summary() -> pd.DataFrame:
    """Average sensor values and dominant status per calendar day."""
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM readings", conn)
    conn.close()
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["timestamp"]).dt.date
    summary = (
        df.groupby("date")
        .agg(
            avg_temperature=("temperature", "mean"),
            avg_vibration=("vibration", "mean"),
            avg_rpm=("rpm", "mean"),
            avg_health_score=("health_score", "mean"),
            readings_count=("id", "count"),
        )
        .reset_index()
        .sort_values("date", ascending=False)
    )
    return summary.round(2)


def get_failure_history(n: int = 20) -> pd.DataFrame:
    """Most recent readings where the ML model predicted anything other
    than Healthy — a simple failure/event history log."""
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM readings WHERE ml_prediction != 'Healthy' "
        "ORDER BY id DESC LIMIT ?",
        conn,
        params=(n,),
    )
    conn.close()
    return df


def get_total_count() -> int:
    conn = get_connection()
    cur = conn.execute("SELECT COUNT(*) FROM readings")
    count = cur.fetchone()[0]
    conn.close()
    return count


def clear_all():
    conn = get_connection()
    conn.execute("DELETE FROM readings")
    conn.commit()
    conn.close()


# ----------------------------------------------------------------------
# Alerts / notification log
# ----------------------------------------------------------------------
def insert_alert(record: dict):
    conn = get_connection()
    columns = ["timestamp", "machine", "status", "prediction", "urgency",
               "health_score", "anomaly_score", "message"]
    values = [record.get(col) for col in columns]
    placeholders = ", ".join(["?"] * len(columns))
    conn.execute(
        f"INSERT INTO alerts ({', '.join(columns)}) VALUES ({placeholders})",
        values,
    )
    conn.commit()
    conn.close()


def get_alerts(n: int = 100) -> pd.DataFrame:
    conn = get_connection()
    df = pd.read_sql_query(
        "SELECT * FROM alerts ORDER BY id DESC LIMIT ?", conn, params=(n,)
    )
    conn.close()
    return df


def get_alert_counts() -> dict:
    conn = get_connection()
    df = pd.read_sql_query("SELECT urgency, COUNT(*) as n FROM alerts GROUP BY urgency", conn)
    conn.close()
    return dict(zip(df["urgency"], df["n"])) if not df.empty else {}


def acknowledge_alert(alert_id: int):
    conn = get_connection()
    conn.execute("UPDATE alerts SET acknowledged = 1 WHERE id = ?", (alert_id,))
    conn.commit()
    conn.close()


def clear_alerts():
    conn = get_connection()
    conn.execute("DELETE FROM alerts")
    conn.commit()
    conn.close()


# ----------------------------------------------------------------------
# Registered users (accounts created from the sign-up page)
# ----------------------------------------------------------------------
def username_exists(username: str) -> bool:
    conn = get_connection()
    cur = conn.execute(
        "SELECT 1 FROM users WHERE lower(username) = lower(?)", (username,)
    )
    row = cur.fetchone()
    conn.close()
    return row is not None


def insert_user(username: str, password_hash: str, role: str, display_name: str, created_at: str):
    conn = get_connection()
    conn.execute(
        "INSERT INTO users (username, password_hash, role, display_name, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (username.strip().lower(), password_hash, role, display_name, created_at),
    )
    conn.commit()
    conn.close()


def get_user_by_username(username: str):
    conn = get_connection()
    cur = conn.execute(
        "SELECT username, password_hash, role, display_name FROM users WHERE lower(username) = lower(?)",
        (username,),
    )
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    return {
        "username": row[0],
        "password_hash": row[1],
        "role": row[2],
        "display_name": row[3],
    }
