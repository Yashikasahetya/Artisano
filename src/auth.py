"""
src/auth.py - login / sign-up for Artisano (SQLite + Streamlit session state).

Usage in app.py (after set_page_config and the CSS block):

    from src.auth import require_login, render_account_sidebar
    user = require_login(CRAFTS, TIERS)   # shows the login card and stops the script until logged in
    render_account_sidebar(user)          # profile summary + Logout button

Security notes
  * Passwords are never stored: only a per-user random salt + PBKDF2-HMAC-SHA256 hash
    (600k iterations, standard library only, so no new dependencies).
  * All SQL uses parameter binding (no string-built queries).
  * Hashes are compared in constant time; unknown e-mails still cost one hash so
    response time does not reveal which accounts exist.
  * Login attempts are throttled per browser session (5 failures -> 60 s pause).
"""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import sqlite3
import time
from contextlib import closing
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "users.db"

PBKDF2_ITERATIONS = 600_000
MIN_PASSWORD_LEN = 8
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_SECONDS = 60
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

_DUMMY_SALT = secrets.token_bytes(16)  # used to equalise timing for unknown e-mails


# ----------------------------------------------------------------- data layer
def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with closing(_connect()) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name     TEXT    NOT NULL,
                email         TEXT    NOT NULL UNIQUE,
                password_hash TEXT    NOT NULL,
                salt          TEXT    NOT NULL,
                iterations    INTEGER NOT NULL,
                craft         TEXT    NOT NULL,
                location      TEXT    NOT NULL,
                market_tier   INTEGER NOT NULL,
                created_at    TEXT    NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def _hash_password(password: str, salt: bytes, iterations: int | None = None) -> bytes:
    iterations = iterations or PBKDF2_ITERATIONS   # read the setting at call time
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)


def _public(row: sqlite3.Row) -> dict:
    """Profile fields that are safe to keep in session state (no hash / salt)."""
    return {
        "id": row["id"],
        "full_name": row["full_name"],
        "email": row["email"],
        "craft": row["craft"],
        "location": row["location"],
        "market_tier": row["market_tier"],
    }


def create_user(full_name: str, email: str, password: str, craft: str,
                location: str, market_tier: int) -> dict:
    """Validate and insert a new user. Raises ValueError with a friendly message."""
    full_name, email, location = full_name.strip(), email.strip().lower(), location.strip()
    if not full_name:
        raise ValueError("Please enter your full name.")
    if not EMAIL_RE.match(email):
        raise ValueError("Please enter a valid email address.")
    if len(password) < MIN_PASSWORD_LEN:
        raise ValueError(f"Password must be at least {MIN_PASSWORD_LEN} characters.")
    if not location:
        raise ValueError("Please enter your location (village / town / city).")

    salt = secrets.token_bytes(16)
    iterations = PBKDF2_ITERATIONS
    pw_hash = _hash_password(password, salt, iterations)
    try:
        with closing(_connect()) as conn:
            cur = conn.execute(
                "INSERT INTO users (full_name, email, password_hash, salt, iterations, "
                "craft, location, market_tier) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (full_name, email, pw_hash.hex(), salt.hex(), iterations,
                 craft, location, int(market_tier)),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM users WHERE id = ?", (cur.lastrowid,)).fetchone()
    except sqlite3.IntegrityError:
        raise ValueError("An account with this email already exists. Please log in.") from None
    return _public(row)


def authenticate(email: str, password: str) -> dict | None:
    """Return the user's public profile if credentials are valid, else None."""
    email = email.strip().lower()
    with closing(_connect()) as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    if row is None:
        _hash_password(password, _DUMMY_SALT)          # same cost as a real check
        return None
    expected = bytes.fromhex(row["password_hash"])
    actual = _hash_password(password, bytes.fromhex(row["salt"]), row["iterations"])
    return _public(row) if hmac.compare_digest(expected, actual) else None


# ----------------------------------------------------------------- session helpers
def _login(user: dict) -> None:
    # Start from a clean slate so no catalogue/edits from a previous user leak in.
    st.session_state.clear()
    st.session_state["user"] = user
    st.session_state["authenticated"] = True


def logout() -> None:
    st.session_state.clear()
    st.rerun()


def _locked_seconds() -> int:
    return max(0, int(st.session_state.get("auth_locked_until", 0) - time.time()))


def _register_failure() -> None:
    fails = st.session_state.get("auth_failures", 0) + 1
    st.session_state["auth_failures"] = fails
    if fails >= MAX_FAILED_ATTEMPTS:
        st.session_state["auth_locked_until"] = time.time() + LOCKOUT_SECONDS
        st.session_state["auth_failures"] = 0


# ----------------------------------------------------------------- UI
def _login_form() -> None:
    with st.form("login_form", clear_on_submit=False):
        email = st.text_input("Email", placeholder="you@example.com")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in", type="primary")

    if not submitted:
        return
    wait = _locked_seconds()
    if wait:
        st.error(f"Too many failed attempts. Please try again in {wait} seconds.")
        return
    if not email.strip() or not password:
        st.warning("Enter your email and password.")
        return
    user = authenticate(email, password)
    if user is None:
        _register_failure()
        st.error("Incorrect email or password.")
        return
    _login(user)
    st.rerun()


def _signup_form(crafts: list[str], tiers: list[str]) -> None:
    with st.form("signup_form", clear_on_submit=False):
        full_name = st.text_input("Full name", placeholder="e.g. Meera Devi")
        email = st.text_input("Email (used to log in)", placeholder="you@example.com", key="su_email")
        p1, p2 = st.columns(2)
        password = p1.text_input(f"Password (min {MIN_PASSWORD_LEN} characters)", type="password", key="su_pw")
        confirm = p2.text_input("Confirm password", type="password", key="su_pw2")
        c1, c2 = st.columns(2)
        craft = c1.selectbox("Your craft", crafts)
        tier_name = c2.selectbox("Where do you mainly sell?", tiers, index=min(1, len(tiers) - 1))
        location = st.text_input("Location", placeholder="Village / town / city, e.g. Sanganer, Jaipur")
        submitted = st.form_submit_button("Create my account", type="primary")

    if not submitted:
        return
    if password != confirm:
        st.error("The two passwords do not match.")
        return
    try:
        user = create_user(full_name, email, password, craft, location, tiers.index(tier_name))
    except ValueError as err:
        st.error(str(err))
        return
    _login(user)
    st.rerun()


def require_login(crafts: list[str], tiers: list[str]) -> dict:
    """Return the logged-in user's profile; otherwise draw the login card and stop the script."""
    init_db()                                  # CREATE TABLE IF NOT EXISTS - cheap and safe
    user = st.session_state.get("user")
    if st.session_state.get("authenticated") and user:
        return user

    _, mid, _ = st.columns([1, 2, 1])          # centred card; stacks on narrow screens
    with mid:
        st.markdown("# 🧵 Artisano")
        st.caption("Sign in to plan your production and price your craft.")
        tab_login, tab_signup = st.tabs(["🔑 Log in", "✨ Sign up"])
        with tab_login:
            _login_form()
        with tab_signup:
            _signup_form(crafts, tiers)
    st.stop()


def render_account_sidebar(user: dict, tiers: list[str]) -> None:
    with st.sidebar:
        st.markdown("### 👤 My account")
        st.markdown(f"**{user['full_name']}**")
        st.caption(
            f"{user['craft']} · {user['location']}\n\n"
            f"Sells at: {tiers[user['market_tier']]}\n\n{user['email']}"
        )
        if st.button("🚪 Log out"):
            logout()
