"""
auth.py
Simple role-based login system for the Digital Twin Console.

Two roles are supported out of the box:
  - Admin    : full access — can view Model Insights, clear alert log,
               clear stored history, reset the live session.
  - Operator : day-to-day monitoring — can take readings, run live
               simulation, view Live Monitor / Fleet / Alerts, but
               cannot wipe data or open Model Insights.

This is a lightweight, self-contained login layer (no external identity
provider) — a good fit for a demo / college project deployment. Passwords
are never kept in plain text in memory comparisons; they're hashed with
SHA-256 before checking.
"""

import hashlib
from datetime import datetime

import streamlit as st

import db

# ----------------------------------------------------------------------
# User directory (demo accounts). In a real deployment this would live
# in a database table instead of a source file.
# ----------------------------------------------------------------------
def _hash(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


USERS = {
    "admin": {
        "password_hash": _hash("admin@123"),
        "role": "Admin",
        "display_name": "Plant Administrator",
    },
    "operator": {
        "password_hash": _hash("operator@123"),
        "role": "Operator",
        "display_name": "Shift Operator",
    },
}

ROLE_PERMISSIONS = {
    "Admin": {
        "can_clear_data": True,
        "can_view_insights": True,
    },
    "Operator": {
        "can_clear_data": False,
        "can_view_insights": False,
    },
}


# ----------------------------------------------------------------------
# Session helpers
# ----------------------------------------------------------------------
def _init_state():
    if "auth_user" not in st.session_state:
        st.session_state.auth_user = None


def is_authenticated() -> bool:
    _init_state()
    return st.session_state.auth_user is not None


def current_user():
    _init_state()
    return st.session_state.auth_user


def current_role():
    user = current_user()
    return user["role"] if user else None


def has_permission(perm: str) -> bool:
    role = current_role()
    if role is None:
        return False
    return ROLE_PERMISSIONS.get(role, {}).get(perm, False)


def attempt_login(username: str, password: str) -> bool:
    clean_username = (username or "").strip().lower()

    # 1) Built-in demo accounts
    user = USERS.get(clean_username)
    if user and user["password_hash"] == _hash(password or ""):
        st.session_state.auth_user = {
            "username": clean_username,
            "role": user["role"],
            "display_name": user["display_name"],
        }
        return True

    # 2) Accounts created via the registration page (stored in SQLite)
    db_user = db.get_user_by_username(clean_username)
    if db_user and db_user["password_hash"] == _hash(password or ""):
        st.session_state.auth_user = {
            "username": db_user["username"],
            "role": db_user["role"],
            "display_name": db_user["display_name"],
        }
        return True

    return False


def register_user(username: str, password: str, confirm_password: str, role: str, display_name: str):
    """Create a new account (Admin or Operator). Returns (success, message)."""
    clean_username = (username or "").strip().lower()
    clean_display = (display_name or "").strip()

    if not clean_username or not password:
        return False, "Username and password are required."
    if len(clean_username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    if password != confirm_password:
        return False, "Passwords do not match."
    if role not in ("Admin", "Operator"):
        return False, "Please select a valid role."
    if clean_username in USERS or db.get_user_by_username(clean_username):
        return False, "That username is already taken."

    db.insert_user(
        username=clean_username,
        password_hash=_hash(password),
        role=role,
        display_name=clean_display or clean_username.title(),
        created_at=datetime.now().isoformat(timespec="seconds"),
    )
    return True, "Account created successfully. You can now sign in."


def logout():
    st.session_state.auth_user = None
    # Clear the rest of the simulation state too, so the next login
    # starts from a clean slate rather than someone else's session.
    for key in ["motor", "history", "last_reading", "running", "operating_hours",
                "prev_mode", "prev_scenario", "prev_target_fault", "fleet_snapshot"]:
        st.session_state.pop(key, None)


# ----------------------------------------------------------------------
# Login page UI
# ----------------------------------------------------------------------
def render_login_page():
    """Full-page login gate. Caller should st.stop() right after this
    if the user is still not authenticated."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
        .main { background: #F6F8F6; }
        .login-wrap { max-width: 430px; margin: 48px auto 0 auto; }
        .login-title { font-family:'Poppins',sans-serif; font-weight:700; font-size:26px; color:#17221D;
            margin-bottom:6px; text-align:center; }
        .login-sub { font-family:'Inter',sans-serif; font-size:13.5px; color:#4A5A52; margin-bottom:6px;
            line-height:1.5; text-align:center; }
        .login-demo {
            margin-top:16px; background:#FFFFFF; border:1px solid #E5EAE7; border-radius:12px;
            padding:14px 16px; font-family:'JetBrains Mono',monospace; font-size:12px; color:#4A5A52;
            line-height:1.8; box-shadow: 0 1px 2px rgba(23,34,29,0.04), 0 4px 16px rgba(23,34,29,0.05);
        }
        .login-demo b { color:#17221D; }
        div[data-testid="stForm"] {
            background:#FFFFFF; border:1px solid #E5EAE7; border-radius:16px; padding:26px 26px 12px 26px;
            box-shadow: 0 4px 10px rgba(23,34,29,0.05), 0 10px 28px rgba(23,34,29,0.09);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if "auth_page_mode" not in st.session_state:
        st.session_state.auth_page_mode = "signin"

    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        tab_signin, tab_register = st.tabs(["Sign in", "Create account"])

        # ------------------------------------------------------------
        # Sign in
        # ------------------------------------------------------------
        with tab_signin:
            st.markdown(
                """
                <div class="login-wrap" style="margin-top:20px;">
                    <div class="login-title">Sign in to continue</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("Username", placeholder="admin or operator")
                password = st.text_input("Password", type="password", placeholder="••••••••")
                submitted = st.form_submit_button("Sign in", use_container_width=True)

            if submitted:
                if attempt_login(username, password):
                    st.rerun()
                else:
                    st.error("Invalid username or password. Please try again.")

            st.markdown(
                """
                <div class="login-demo">
                <b>Demo accounts</b><br>
                Admin&nbsp;&nbsp;&nbsp;→ admin / admin@123<br>
                Operator → operator / operator@123
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ------------------------------------------------------------
        # Create account (Admin or Operator)
        # ------------------------------------------------------------
        with tab_register:
            st.markdown(
                """
                <div class="login-wrap" style="margin-top:20px;">
                    <div class="login-title">Create an account</div>
                    <div class="login-sub">Register as <b>Admin</b> or <b>Operator</b>.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.form("register_form", clear_on_submit=False):
                reg_display_name = st.text_input("Full name", placeholder="e.g. Priya Sharma")
                reg_username = st.text_input("Choose a username", placeholder="e.g. priya.sharma")
                reg_role = st.selectbox("Role", ["Operator", "Admin"])
                reg_password = st.text_input("Password", type="password", placeholder="At least 6 characters")
                reg_confirm = st.text_input("Confirm password", type="password", placeholder="Re-enter password")
                reg_submitted = st.form_submit_button("Create account", use_container_width=True)

            if reg_submitted:
                success, message = register_user(
                    username=reg_username,
                    password=reg_password,
                    confirm_password=reg_confirm,
                    role=reg_role,
                    display_name=reg_display_name,
                )
                if success:
                    st.success(message)
                else:
                    st.error(message)
