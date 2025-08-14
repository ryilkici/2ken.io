import sqlite3
from pathlib import Path
from typing import List, Optional, Dict

DB_PATH = Path("data.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
);
"""


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_connection()
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


def list_models() -> List[Dict[str, str]]:
    conn = get_connection()
    try:
        rows = conn.execute("SELECT id, name FROM models ORDER BY id DESC").fetchall()
        return [dict(row) for row in rows]

    finally:
        conn.close()


def add_model(name: str) -> int:
    conn = get_connection()
    try:
        cur = conn.execute(
            "INSERT INTO models (name) VALUES (?)",
            (name,),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def delete_model(name: str) -> None:
    conn = get_connection()
    try:
        conn.execute("DELETE FROM models WHERE name = ?", (name,))
        conn.commit()
    finally:
        conn.close()


