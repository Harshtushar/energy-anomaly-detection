from pathlib import Path
import hashlib
import hmac
import os
import re
import sqlite3


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "auth_users.db"
PASSWORD_ITERATIONS = 100_000
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{3,50}$")


def get_db_path(db_path=None):
    resolved_db_path = Path(db_path) if db_path else DEFAULT_DB_PATH

    if resolved_db_path.exists() and resolved_db_path.stat().st_size == 0:
        suffix = resolved_db_path.suffix or ".db"
        return resolved_db_path.with_name(f"{resolved_db_path.stem}_live{suffix}")

    return resolved_db_path


def get_connection(db_path=None):
    resolved_db_path = get_db_path(db_path)
    resolved_db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(resolved_db_path), timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=MEMORY")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA temp_store=MEMORY")
    return connection


def init_auth_db(db_path=None):
    with get_connection(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS run_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                source_name TEXT NOT NULL,
                result_path TEXT NOT NULL,
                processed_rows INTEGER NOT NULL,
                building_count INTEGER NOT NULL,
                anomaly_count INTEGER NOT NULL,
                anomaly_rate REAL NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        existing_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(run_history)").fetchall()
        }
        if "model_dir" not in existing_columns:
            connection.execute("ALTER TABLE run_history ADD COLUMN model_dir TEXT")


def validate_new_user(username, password, confirm_password):
    cleaned_username = username.strip()

    if not cleaned_username:
        return False, "Username is required."
    if not USERNAME_PATTERN.fullmatch(cleaned_username):
        return (
            False,
            "Username must be 3-50 characters and use only letters, numbers, dot, dash, or underscore.",
        )
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    if password != confirm_password:
        return False, "Passwords do not match."

    return True, None


def hash_password(password, salt=None):
    salt_bytes = salt if salt is not None else os.urandom(16)
    password_hash = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt_bytes,
        PASSWORD_ITERATIONS,
    )
    return salt_bytes.hex(), password_hash.hex()


def verify_password(password, password_salt, password_hash):
    salt_bytes = bytes.fromhex(password_salt)
    _, computed_hash = hash_password(password, salt=salt_bytes)
    return hmac.compare_digest(computed_hash, password_hash)


def create_user(username, password, confirm_password, db_path=None):
    is_valid, error_message = validate_new_user(username, password, confirm_password)
    if not is_valid:
        return False, error_message

    cleaned_username = username.strip()
    password_salt, password_hash = hash_password(password)

    try:
        with get_connection(db_path) as connection:
            connection.execute(
                """
                INSERT INTO users (username, password_salt, password_hash)
                VALUES (?, ?, ?)
                """,
                (cleaned_username, password_salt, password_hash),
            )
    except sqlite3.IntegrityError:
        return False, "That username already exists."

    return True, None


def authenticate_user(username, password, db_path=None):
    cleaned_username = username.strip()
    if not cleaned_username or not password:
        return False

    with get_connection(db_path) as connection:
        user_row = connection.execute(
            """
            SELECT username, password_salt, password_hash
            FROM users
            WHERE username = ?
            """,
            (cleaned_username,),
        ).fetchone()

    if user_row is None:
        return False

    return verify_password(
        password,
        user_row["password_salt"],
        user_row["password_hash"],
    )


def save_run_history(
    username,
    source_name,
    result_path,
    model_dir,
    processed_rows,
    building_count,
    anomaly_count,
    anomaly_rate,
    db_path=None,
):
    with get_connection(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO run_history (
                username,
                source_name,
                result_path,
                model_dir,
                processed_rows,
                building_count,
                anomaly_count,
                anomaly_rate
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                username.strip(),
                source_name,
                str(result_path),
                str(model_dir) if model_dir else None,
                int(processed_rows),
                int(building_count),
                int(anomaly_count),
                float(anomaly_rate),
            ),
        )
        return cursor.lastrowid


def get_user_history(username, limit=20, db_path=None):
    with get_connection(db_path) as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                source_name,
                result_path,
                model_dir,
                processed_rows,
                building_count,
                anomaly_count,
                anomaly_rate,
                created_at
            FROM run_history
            WHERE username = ?
            ORDER BY datetime(created_at) DESC, id DESC
            LIMIT ?
            """,
            (username.strip(), int(limit)),
        ).fetchall()

    return [dict(row) for row in rows]


def delete_run_history(history_id, username, db_path=None):
    with get_connection(db_path) as connection:
        existing_row = connection.execute(
            """
            SELECT id, result_path, model_dir
            FROM run_history
            WHERE id = ? AND username = ?
            """,
            (int(history_id), username.strip()),
        ).fetchone()

        if existing_row is None:
            return None

        connection.execute(
            """
            DELETE FROM run_history
            WHERE id = ? AND username = ?
            """,
            (int(history_id), username.strip()),
        )

    return dict(existing_row)
