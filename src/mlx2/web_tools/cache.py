"""
Disk-backed page cache — 7-day TTL, SQLite storage.

Stores the processed output (markdown content + metadata) so cache hits
skip both the network fetch and the HTML parsing pipeline entirely.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

TTL = 7 * 24 * 3600  # seconds

if sys.platform == "win32":
    _APP_DATA = Path(os.environ.get("APPDATA", str(Path.home() / "AppData/Roaming"))) / "mcp_web_tool"
else:
    # Default must be writable without root (the service sets MCP_WEB_TOOL_DATA;
    # /var/db silently broke the cache when the env var was missing).
    _APP_DATA = Path(os.environ.get(
        "MLX2_WEB_DATA", str(Path.home() / "Library/Application Support/MLX2/web")))

_DB_PATH = _APP_DATA / "page_cache.db"
_conn: sqlite3.Connection | None = None

_CREATE = """
CREATE TABLE IF NOT EXISTS page_cache (
    url        TEXT PRIMARY KEY,
    stored_at  REAL NOT NULL,
    status     TEXT NOT NULL,
    final_url  TEXT NOT NULL,
    via_browser INTEGER NOT NULL,
    content    TEXT NOT NULL,
    metadata   TEXT NOT NULL
)
"""

# Ledger of every URL any tool has actually surfaced, so verify_citations can
# distinguish a real tool-returned URL from a fabricated one without re-fetching.
_CREATE_SEEN = """
CREATE TABLE IF NOT EXISTS seen_urls (
    url      TEXT PRIMARY KEY,
    seen_at  REAL NOT NULL
)
"""


def _db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _APP_DATA.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.execute(_CREATE)
        _conn.execute(_CREATE_SEEN)
        _conn.commit()
    return _conn


def _norm(url: str) -> str:
    """Normalise for match: strip whitespace and a single trailing slash."""
    u = (url or "").strip()
    return u[:-1] if u.endswith("/") and u.count("/") > 3 else u


def mark_seen(urls) -> None:
    """Record URLs a tool actually returned to the model (search hits, fetched
    pages, arxiv_search results) so verify_citations can tell them from
    fabricated URLs without re-fetching."""
    rows = [(_norm(u), time.time()) for u in urls
            if u and str(u).startswith("http")]
    if not rows:
        return
    db = _db()
    db.executemany(
        "INSERT OR REPLACE INTO seen_urls (url, seen_at) VALUES (?, ?)", rows)
    db.commit()


def was_seen(url: str) -> bool:
    """True if this URL was surfaced by a tool within the TTL window."""
    u = _norm(url)
    row = _db().execute(
        "SELECT seen_at FROM seen_urls WHERE url = ?", (u,)).fetchone()
    if row is None:
        return False
    if time.time() - row[0] > TTL:
        _db().execute("DELETE FROM seen_urls WHERE url = ?", (u,))
        _db().commit()
        return False
    return True


def get(url: str) -> dict | None:
    """Return a fresh cached entry or None."""
    row = _db().execute(
        "SELECT stored_at, status, final_url, via_browser, content, metadata "
        "FROM page_cache WHERE url = ?",
        (url,),
    ).fetchone()
    if row is None:
        return None
    stored_at, status, final_url, via_browser, content, metadata_json = row
    age = time.time() - stored_at
    if age > TTL:
        _db().execute("DELETE FROM page_cache WHERE url = ?", (url,))
        _db().commit()
        return None
    return {
        "status": status,
        "final_url": final_url,
        "via_browser": bool(via_browser),
        "content": content,
        "metadata": json.loads(metadata_json),
        "age_seconds": int(age),
    }


def put(url: str, status: str, final_url: str, via_browser: bool,
        content: str, metadata: dict) -> None:
    """Store a successful fetch result. Overwrites any existing entry."""
    _db().execute(
        "INSERT OR REPLACE INTO page_cache "
        "(url, stored_at, status, final_url, via_browser, content, metadata) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (url, time.time(), status, final_url, int(via_browser),
         content, json.dumps(metadata)),
    )
    _db().commit()


def stats() -> dict:
    """Return basic cache statistics."""
    db = _db()
    total = db.execute("SELECT COUNT(*) FROM page_cache").fetchone()[0]
    fresh = db.execute(
        "SELECT COUNT(*) FROM page_cache WHERE stored_at > ?",
        (time.time() - TTL,),
    ).fetchone()[0]
    return {"total_entries": total, "fresh_entries": fresh, "ttl_seconds": TTL}
