"""Local accounts and role checks for the dashboard.

The seeded passwords are for this machine only. They are not production
credentials and should be replaced before the application is shared.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from werkzeug.security import check_password_hash, generate_password_hash

ROLES = ("admin", "analyst", "reviewer")

PERMISSIONS = {
    "view": frozenset(ROLES),
    "run": frozenset({"admin", "analyst"}),
    "edit_book": frozenset({"admin"}),
    "manage_users": frozenset({"admin"}),
}

# Local demonstration accounts, created only when the user table is empty.
DEMO_ACCOUNTS = (
    ("admin", "Administrator", "admin", "demo-admin"),
    ("analyst", "Analyst", "analyst", "demo-analyst"),
    ("reviewer", "Reviewer", "reviewer", "demo-reviewer"),
)


def role_can(role: str, permission: str) -> bool:
    return role in PERMISSIONS.get(permission, frozenset())


def ensure_users(db_path: Path) -> None:
    """Create the account table and seed the local demonstration accounts once."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                role TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_utc TEXT NOT NULL
            )
            """
        )
        count = connection.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if count:
            return
        now = datetime.now(timezone.utc).isoformat()
        connection.executemany(
            """
            INSERT INTO users (username, display_name, role, password_hash, created_utc)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (username, display_name, role, generate_password_hash(password), now)
                for username, display_name, role, password in DEMO_ACCOUNTS
            ],
        )


def authenticate(db_path: Path, username: str, password: str) -> dict | None:
    key = username.strip().lower()
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT username, display_name, role, password_hash FROM users WHERE username = ?",
            (key,),
        ).fetchone()
    if row is None or not check_password_hash(row["password_hash"], password):
        return None
    return {"username": row["username"], "display_name": row["display_name"], "role": row["role"]}


def list_users(db_path: Path) -> list[dict]:
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT username, display_name, role, created_utc FROM users ORDER BY username"
        ).fetchall()
    return [dict(row) for row in rows]


def add_user(db_path: Path, username: str, display_name: str, role: str, password: str) -> None:
    key = username.strip().lower()
    name = display_name.strip()
    if not key or not key.replace("_", "").isalnum() or not key[0].isalpha():
        raise ValueError("Username must start with a letter and use letters, numbers, or underscores.")
    if len(key) > 32:
        raise ValueError("Username must be 32 characters or fewer.")
    if not name:
        raise ValueError("Display name is required.")
    if role not in ROLES:
        raise ValueError("Choose a role of admin, analyst, or reviewer.")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")
    with sqlite3.connect(db_path) as connection:
        existing = connection.execute("SELECT 1 FROM users WHERE username = ?", (key,)).fetchone()
        if existing:
            raise ValueError(f"User '{key}' already exists.")
        connection.execute(
            """
            INSERT INTO users (username, display_name, role, password_hash, created_utc)
            VALUES (?, ?, ?, ?, ?)
            """,
            (key, name, role, generate_password_hash(password), datetime.now(timezone.utc).isoformat()),
        )
