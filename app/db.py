"""
All state lives here, on disk, in SQLite. Nothing that matters is ever kept
only in memory -- that's the difference between "the process restarted and
we lost a pending retry" and "the process restarted and picked up exactly
where it left off."

Concurrency model: SQLite + WAL mode, with a single asyncio.Lock guarding
every write. This isn't the fastest possible design, but it's the simplest
one that's actually correct -- the uniqueness constraints below are what
prevent duplicate DMs, and they only work if writes are serialized enough
for "INSERT OR IGNORE" to be race-free. At real LinkPlease scale you'd swap
this for Postgres with the same schema and drop the app-level lock in favor
of the DB's own row locking. Documented as a known limit in FAILURES.md.
"""
import asyncio
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from typing import Any, Optional

from app.config import DB_PATH

_lock = asyncio.Lock()
_conn: Optional[sqlite3.Connection] = None


SCHEMA = """
CREATE TABLE IF NOT EXISTS rules (
    rule_id     TEXT PRIMARY KEY,
    keyword     TEXT NOT NULL,
    dm_message  TEXT NOT NULL,
    created_at  REAL NOT NULL
);

-- Every webhook event we've ever received, raw. event_id is unique so a
-- redelivered event (the mock API resends ~8% of events) is recorded once.
-- `processed` flips to 1 once we've run matching logic against it.
CREATE TABLE IF NOT EXISTS events (
    event_id    TEXT PRIMARY KEY,
    event_type  TEXT NOT NULL,
    comment_id  TEXT,
    payload     TEXT NOT NULL,
    received_at REAL NOT NULL,
    processed   INTEGER NOT NULL DEFAULT 0
);

-- Comments we've been told are deleted. Checked before sending a DM for a
-- match, and also relevant if a comment.deleted arrives BEFORE its
-- comment.created (order is not guaranteed) -- we record the deletion
-- regardless of whether we've seen the comment yet.
CREATE TABLE IF NOT EXISTS deleted_comments (
    comment_id  TEXT PRIMARY KEY,
    deleted_at  REAL NOT NULL
);

-- The core dedupe guard. UNIQUE(user_id, rule_id) means "this user has been
-- claimed for this rule" can only ever happen once, no matter how many
-- times the matching event is redelivered or how many times the user
-- re-comments the keyword. This constraint -- not application logic -- is
-- what makes duplicate sends impossible under concurrency.
CREATE TABLE IF NOT EXISTS dm_sends (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         TEXT NOT NULL,
    rule_id         TEXT NOT NULL,
    comment_id      TEXT NOT NULL,
    message         TEXT NOT NULL,
    dm_id           TEXT,
    status          TEXT NOT NULL DEFAULT 'pending',
        -- pending -> attempting/retrying send, not yet accepted by API
        -- queued  -> API returned 202, waiting on delivery confirmation
        -- delivered / failed -> terminal
    attempts        INTEGER NOT NULL DEFAULT 0,
    last_error      TEXT,
    next_attempt_at REAL NOT NULL DEFAULT 0,
    last_checked_at REAL NOT NULL DEFAULT 0,
    created_at      REAL NOT NULL,
    updated_at      REAL NOT NULL,
    UNIQUE(user_id, rule_id)
);

CREATE INDEX IF NOT EXISTS idx_dm_sends_status ON dm_sends(status);
CREATE INDEX IF NOT EXISTS idx_dm_sends_comment ON dm_sends(comment_id);

-- Single-row counter for stats. Kept separate from dm_sends because a
-- "blocked duplicate" by definition never gets its own dm_sends row.
CREATE TABLE IF NOT EXISTS counters (
    name  TEXT PRIMARY KEY,
    value INTEGER NOT NULL DEFAULT 0
);
"""


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute("PRAGMA journal_mode=WAL;")
        _conn.execute("PRAGMA synchronous=NORMAL;")
        _conn.executescript(SCHEMA)
        _conn.execute(
            "INSERT OR IGNORE INTO counters(name, value) VALUES ('duplicates_blocked', 0)"
        )
        _conn.commit()
    return _conn


@contextmanager
def _cursor():
    conn = _get_conn()
    cur = conn.cursor()
    try:
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


# ---------------------------------------------------------------- rules ----

async def create_rule(keyword: str, dm_message: str) -> dict:
    rule_id = str(uuid.uuid4())
    async with _lock:
        with _cursor() as cur:
            cur.execute(
                "INSERT INTO rules(rule_id, keyword, dm_message, created_at) VALUES (?,?,?,?)",
                (rule_id, keyword, dm_message, time.time()),
            )
    return {"rule_id": rule_id, "keyword": keyword, "dm_message": dm_message}


async def get_all_rules() -> list[dict]:
    async with _lock:
        with _cursor() as cur:
            rows = cur.execute("SELECT rule_id, keyword, dm_message FROM rules").fetchall()
    return [dict(r) for r in rows]


# --------------------------------------------------------------- events ----

async def record_event(event_id: str, event_type: str, comment_id: Optional[str], payload: dict) -> bool:
    """Returns True if this is the first time we've seen this event_id."""
    async with _lock:
        with _cursor() as cur:
            try:
                cur.execute(
                    "INSERT INTO events(event_id, event_type, comment_id, payload, received_at) "
                    "VALUES (?,?,?,?,?)",
                    (event_id, event_type, comment_id, json.dumps(payload), time.time()),
                )
                return True
            except sqlite3.IntegrityError:
                return False  # redelivery of an event_id we already recorded


async def fetch_unprocessed_events(limit: int = 25) -> list[dict]:
    async with _lock:
        with _cursor() as cur:
            rows = cur.execute(
                "SELECT event_id, event_type, comment_id, payload FROM events "
                "WHERE processed = 0 ORDER BY received_at ASC LIMIT ?",
                (limit,),
            ).fetchall()
    return [dict(r) for r in rows]


async def mark_event_processed(event_id: str) -> None:
    async with _lock:
        with _cursor() as cur:
            cur.execute("UPDATE events SET processed = 1 WHERE event_id = ?", (event_id,))


async def mark_comment_deleted(comment_id: str) -> None:
    async with _lock:
        with _cursor() as cur:
            cur.execute(
                "INSERT OR IGNORE INTO deleted_comments(comment_id, deleted_at) VALUES (?,?)",
                (comment_id, time.time()),
            )
            # Cancel (delete) any pending DMs for this comment so they aren't sent
            cur.execute(
                "DELETE FROM dm_sends WHERE comment_id=? AND status='pending'",
                (comment_id,),
            )


async def is_comment_deleted(comment_id: str) -> bool:
    async with _lock:
        with _cursor() as cur:
            row = cur.execute(
                "SELECT 1 FROM deleted_comments WHERE comment_id = ?", (comment_id,)
            ).fetchone()
    return row is not None


# ------------------------------------------------------------- dm_sends ----

async def claim_send(user_id: str, rule_id: str, comment_id: str, message: str) -> Optional[int]:
    """
    Atomically claim (user_id, rule_id). Returns the new row id if this
    caller won the claim, or None if a row already existed (duplicate --
    someone already has, or is already getting, this DM).
    """
    async with _lock:
        with _cursor() as cur:
            try:
                now = time.time()
                cur.execute(
                    "INSERT INTO dm_sends(user_id, rule_id, comment_id, message, status, "
                    "created_at, updated_at, next_attempt_at) VALUES (?,?,?,?,'pending',?,?,?)",
                    (user_id, rule_id, comment_id, message, now, now, now),
                )
                return cur.lastrowid
            except sqlite3.IntegrityError:
                return None


async def increment_duplicates_blocked() -> None:
    async with _lock:
        with _cursor() as cur:
            cur.execute("UPDATE counters SET value = value + 1 WHERE name = 'duplicates_blocked'")


async def fetch_sendable(limit: int = 10) -> list[dict]:
    """Rows ready to attempt (or retry) sending, oldest-due first."""
    now = time.time()
    async with _lock:
        with _cursor() as cur:
            rows = cur.execute(
                "SELECT * FROM dm_sends WHERE status = 'pending' AND next_attempt_at <= ? "
                "ORDER BY next_attempt_at ASC LIMIT ?",
                (now, limit),
            ).fetchall()
    return [dict(r) for r in rows]


async def record_send_accepted(row_id: int, dm_id: str, attempts: int) -> None:
    async with _lock:
        with _cursor() as cur:
            cur.execute(
                "UPDATE dm_sends SET status='queued', dm_id=?, attempts=?, updated_at=?, last_checked_at=? "
                "WHERE id=?",
                (dm_id, attempts, time.time(), 0, row_id),
            )


async def record_reconcile_failed_retry(row_id: int, next_attempt_at: float) -> None:
    async with _lock:
        with _cursor() as cur:
            cur.execute(
                "UPDATE dm_sends SET status='pending', dm_id=NULL, next_attempt_at=?, updated_at=?, last_checked_at=? "
                "WHERE id=?",
                (next_attempt_at, time.time(), 0, row_id),
            )


async def record_send_cancelled(row_id: int) -> None:
    async with _lock:
        with _cursor() as cur:
            cur.execute("DELETE FROM dm_sends WHERE id=?", (row_id,))


async def record_send_retry(row_id: int, attempts: int, next_attempt_at: float, error: str) -> None:
    async with _lock:
        with _cursor() as cur:
            cur.execute(
                "UPDATE dm_sends SET attempts=?, next_attempt_at=?, last_error=?, updated_at=? "
                "WHERE id=?",
                (attempts, next_attempt_at, error, time.time(), row_id),
            )


async def record_send_failed(row_id: int, error: str) -> None:
    async with _lock:
        with _cursor() as cur:
            cur.execute(
                "UPDATE dm_sends SET status='failed', last_error=?, updated_at=? WHERE id=?",
                (error, time.time(), row_id),
            )


async def fetch_queued_for_reconcile(limit: int = 20) -> list[dict]:
    cutoff = time.time() - 5.0  # see config.RECONCILE_MIN_RECHECK_SECONDS
    async with _lock:
        with _cursor() as cur:
            rows = cur.execute(
                "SELECT * FROM dm_sends WHERE status = 'queued' AND last_checked_at <= ? "
                "ORDER BY last_checked_at ASC LIMIT ?",
                (cutoff, limit),
            ).fetchall()
    return [dict(r) for r in rows]


async def record_reconcile_result(row_id: int, status: str) -> None:
    """status is 'delivered', 'failed', or 'queued' (still pending -> just bump last_checked_at)."""
    async with _lock:
        with _cursor() as cur:
            if status in ("delivered", "failed"):
                cur.execute(
                    "UPDATE dm_sends SET status=?, updated_at=?, last_checked_at=? WHERE id=?",
                    (status, time.time(), time.time(), row_id),
                )
            else:
                cur.execute(
                    "UPDATE dm_sends SET last_checked_at=? WHERE id=?", (time.time(), row_id)
                )


async def get_stats() -> dict:
    async with _lock:
        with _cursor() as cur:
            sent = cur.execute(
                "SELECT COUNT(*) c FROM dm_sends WHERE status='delivered'"
            ).fetchone()["c"]
            failed = cur.execute(
                "SELECT COUNT(*) c FROM dm_sends WHERE status='failed'"
            ).fetchone()["c"]
            queued = cur.execute(
                "SELECT COUNT(*) c FROM dm_sends WHERE status IN ('pending','queued')"
            ).fetchone()["c"]
            dup = cur.execute(
                "SELECT value v FROM counters WHERE name='duplicates_blocked'"
            ).fetchone()["v"]
    return {"sent": sent, "failed": failed, "queued": queued, "duplicates_blocked": dup}
