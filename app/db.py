import sqlite3
from pathlib import Path
from datetime import datetime

ROOT_DIR = Path(__file__).resolve().parents[1]   # repo root
DB_PATH = ROOT_DIR / "data" / "predictions.sqlite3"


def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            title TEXT,
            body TEXT,
            source TEXT,
            date_str TEXT,
            input_text TEXT NOT NULL,
            label TEXT NOT NULL,
            confidence_pct REAL NOT NULL,
            prob_fake REAL NOT NULL,
            prob_real REAL NOT NULL,
            category TEXT NOT NULL
        )
        """)
        conn.commit()


def save_prediction(payload: dict):
    """
    Expected keys:
    title, body, source, date_str, input_text, label, confidence_pct,
    prob_fake, prob_real, category
    """
    with get_conn() as conn:
        conn.execute("""
            INSERT INTO predictions (
                created_at, title, body, source, date_str, input_text,
                label, confidence_pct, prob_fake, prob_real, category
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            payload.get("title"),
            payload.get("body"),
            payload.get("source"),
            payload.get("date_str"),
            payload["input_text"],
            payload["label"],
            float(payload["confidence_pct"]),
            float(payload["prob_fake"]),
            float(payload["prob_real"]),
            payload["category"],
        ))
        conn.commit()

def fetch_recent(limit: int = 50):
    with get_conn() as conn:
        cur = conn.execute("""
            SELECT *
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        return [dict(r) for r in cur.fetchall()]

def fetch_by_label(label: str, limit: int = 50):
    with get_conn() as conn:
        cur = conn.execute("""
            SELECT *
            FROM predictions
            WHERE label = ?
            ORDER BY id DESC
            LIMIT ?
        """, (label, limit))
        return [dict(r) for r in cur.fetchall()]

def fetch_sources():
    with get_conn() as conn:
        cur = conn.execute("""
            SELECT DISTINCT source
            FROM predictions
            WHERE source IS NOT NULL AND TRIM(source) <> ''
            ORDER BY source ASC
        """)
        return [r["source"] for r in cur.fetchall()]


def fetch_filtered(label: str | None, source: str | None, min_conf: float, date_from: str | None, date_to: str | None, limit: int = 50):
    q = """
        SELECT *
        FROM predictions
        WHERE 1=1
    """
    params = []

    if label and label != "All":
        q += " AND label = ?"
        params.append(label)

    if source and source != "All":
        q += " AND source = ?"
        params.append(source)

    if min_conf is not None:
        q += " AND confidence_pct >= ?"
        params.append(float(min_conf))

    # created_at se guarda como "YYYY-MM-DD HH:MM:SS" -> podemos filtrar por prefijo de fecha
    if date_from:
        q += " AND substr(created_at,1,10) >= ?"
        params.append(date_from)

    if date_to:
        q += " AND substr(created_at,1,10) <= ?"
        params.append(date_to)

    q += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with get_conn() as conn:
        cur = conn.execute(q, tuple(params))
        return [dict(r) for r in cur.fetchall()]


def fetch_stats():
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) as c FROM predictions").fetchone()["c"]
        real = conn.execute("SELECT COUNT(*) as c FROM predictions WHERE label='Real'").fetchone()["c"]
        fake = conn.execute("SELECT COUNT(*) as c FROM predictions WHERE label='Fake'").fetchone()["c"]
        avg_conf = conn.execute("SELECT AVG(confidence_pct) as a FROM predictions").fetchone()["a"]
        return {
            "total": int(total),
            "real": int(real),
            "fake": int(fake),
            "avg_conf": float(avg_conf) if avg_conf is not None else 0.0
        }

def delete_prediction(pred_id: int) -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM predictions WHERE id = ?", (pred_id,))
        conn.commit()
    finally:
        conn.close()