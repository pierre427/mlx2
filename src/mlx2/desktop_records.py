"""Durable desktop jobs and source/artifact/host/profile-bound evidence.

These UI levels do not replace the serving runtime's qualification receipts.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import threading
import time
import uuid
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


class RecordStore:
    def __init__(self, home: Path):
        self.lock = threading.RLock()
        self.db = sqlite3.connect(home / "desktop-records.sqlite3", check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, created REAL NOT NULL, updated REAL NOT NULL,
                state TEXT NOT NULL, kind TEXT NOT NULL, spec TEXT NOT NULL,
                progress TEXT NOT NULL, result TEXT);
            CREATE TABLE IF NOT EXISTS evidence (
                id TEXT PRIMARY KEY, created REAL NOT NULL, binding TEXT NOT NULL,
                level INTEGER NOT NULL, passed INTEGER NOT NULL, record TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS evidence_binding ON evidence(binding, level);
        """)
        # An interrupted job is retained, never silently relaunched after reboot.
        self.db.execute("UPDATE jobs SET state='interrupted' WHERE state IN ('running','waiting')")
        self.db.commit()

    def enqueue(self, kind, spec):
        identity = uuid.uuid4().hex
        now = time.time()
        with self.lock, self.db:
            self.db.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?)",
                (identity, now, now, "queued", kind, canonical(spec), "Queued", None))
        return identity

    def update(self, identity, state, progress, result=None):
        with self.lock, self.db:
            self.db.execute("UPDATE jobs SET updated=?,state=?,progress=?,result=? WHERE id=?",
                (time.time(), state, progress, canonical(result) if result is not None else None, identity))

    def jobs(self, limit=100):
        with self.lock:
            rows = self.db.execute("SELECT * FROM jobs ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
        return [{**dict(row), "spec": json.loads(row["spec"]),
                 "result": json.loads(row["result"]) if row["result"] else None} for row in rows]

    def evidence(self, record):
        required = {"checkpoint", "source_revision", "source_sha256", "host", "runtime", "profile"}
        binding = record.get("binding")
        if not isinstance(binding, dict) or not required <= binding.keys():
            raise ValueError("Evidence needs artifact, source, host, runtime and profile bindings")
        if "@" not in binding["checkpoint"] or binding["checkpoint"].startswith("*@"):
            raise ValueError("Evidence must name an exact artifact checkpoint")
        if type(record.get("level")) is not int or record["level"] not in (1, 2, 3):
            raise ValueError("Evidence level must be 1, 2 or 3")
        if type(record.get("passed")) is not bool:
            raise ValueError("Evidence must carry an explicit verdict")
        if record["level"] == 3 and record["passed"]:
            samples = record.get("samples")
            if not isinstance(samples, list) or not samples:
                raise ValueError("Level 3 requires measured performance samples")
            if any(type(row.get(k)) not in (int, float) or not math.isfinite(row[k]) or row[k] <= 0
                   for row in samples for k in ("prefill_tps", "decode_tps")):
                raise ValueError("Performance samples need positive prefill and decode measurements")
        identity = uuid.uuid4().hex
        with self.lock, self.db:
            self.db.execute("INSERT INTO evidence VALUES (?,?,?,?,?,?)",
                (identity, time.time(), digest(binding), record["level"], int(record["passed"]), canonical(record)))
        return identity

    def records(self, limit=500):
        with self.lock:
            rows = self.db.execute("SELECT * FROM evidence ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
        return [{"id": r["id"], "created": r["created"], **json.loads(r["record"])} for r in rows]

    def summary(self, checkpoint, *, source_sha256=None):
        rows = [r for r in self.records() if r["binding"]["checkpoint"] == checkpoint]
        if source_sha256:
            rows = [r for r in rows if r["binding"]["source_sha256"] == source_sha256]
        groups = {}
        for row in rows:
            group = groups.setdefault(digest(row["binding"]), {})
            group.setdefault(row["level"], row)  # newest result wins, including failures
        def route_coverage_complete(group):
            coverage = group.get(2, {}).get("coverage", {})
            required = set(coverage.get("required_routes", []))
            tested = set(coverage.get("tested_routes", []))
            return not required or required <= tested
        complete = [g for g in groups.values() if all(g.get(i, {}).get("passed") for i in (1, 2, 3))
                    and route_coverage_complete(g)]
        passing = max((sum(bool(g.get(i, {}).get("passed")) for i in (1, 2, 3)) for g in groups.values()), default=0)
        label = "Qualified" if complete else f"{passing}/3 levels" if rows else "Untested"
        if passing == 3 and not complete:
            label += " · route coverage pending"
        return {"label": label,
                "qualified": bool(complete), "records": len(rows)}

    def close(self):
        with self.lock:
            self.db.close()
