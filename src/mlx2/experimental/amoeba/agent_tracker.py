"""Work tracker and experience store for ``mlx2 amoeba auto``; experimental, unqualified.

Design borrowed from mlx2-explore (user/mlx2-explore@6730535: src/mlx2_explore/db.py, services.py, lessons.py,
AGPL-3.0, ideas only, no code copied): one SQLite file in WAL mode; jobs claimed by heartbeat leases that expire, so an
interrupted run's work is picked up again; outcomes recorded as facts linked to their jobs; a cross-run lessons store
recalled into prompts. Changes for this harness: jobs are keyed by exact identities (host and donor fingerprints, topic,
profile), so dedup is exact rather than by text similarity; facts are deterministic pipeline outcomes only, never brain
text; lessons dedup by token Jaccard (no embedding service) and are credited against oracle-labelled decisions, so a
lesson that does not help is retired instead of being recalled forever.

The ledger (``<out>/ledger.jsonl``) stays the per-run record; this file is the memory across runs.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, started REAL, finished REAL, mode TEXT, goal TEXT,
    brain TEXT, commit_rev TEXT, out TEXT, outcome TEXT);
CREATE TABLE IF NOT EXISTS jobs(id INTEGER PRIMARY KEY, key TEXT UNIQUE, kind TEXT, state TEXT, owner TEXT,
    heartbeat REAL, attempts INTEGER DEFAULT 0, payload TEXT, result TEXT, run_id TEXT, created REAL, updated REAL,
    attempt_run TEXT);
CREATE TABLE IF NOT EXISTS facts(id INTEGER PRIMARY KEY, job_key TEXT, kind TEXT, value TEXT, run_id TEXT,
    commit_rev TEXT, created REAL);
CREATE TABLE IF NOT EXISTS decisions(id INTEGER PRIMARY KEY, run_id TEXT, stage TEXT, brain TEXT, prompt TEXT,
    options TEXT, scores TEXT, chosen TEXT, lessons TEXT, oracle TEXT, correct INTEGER, created REAL, arm TEXT);
CREATE TABLE IF NOT EXISTS lesson_credit(lesson INTEGER, stage TEXT, brain TEXT, helped INTEGER DEFAULT 0,
    hurt INTEGER DEFAULT 0, PRIMARY KEY(lesson, stage, brain));
CREATE TABLE IF NOT EXISTS lessons(id INTEGER PRIMARY KEY, text TEXT, tags TEXT, source_run TEXT, created REAL,
    shown INTEGER DEFAULT 0, helped INTEGER DEFAULT 0, hurt INTEGER DEFAULT 0, retired INTEGER DEFAULT 0);
"""

JOB_STATES = ("queued", "leased", "done", "refused", "failed")
TERMINAL = ("done", "refused")              # failed jobs may be leased again (bounded by attempts)
LEASE_S = 3 * 3600.0                       # renewed before each child (timeout 7200 s): a live job never expires
MAX_ATTEMPTS = 2                           # per run: a later run may try a failed job again
RETIRE_AFTER_SHOWN = 3                     # labelled showings (same stage and brain) before a lesson is judged
CONTROL_MIN = 5                            # labelled control decisions (same stage and brain) needed to compare
RETIRE_MARGIN = 0.1                        # retired when its rate is below control - margin
RECALL_K = 3                               # lessons per prompt: the best K-1 by record plus the least-shown one
LESSON_DUP_JACCARD = 0.6


def default_path() -> Path:
    root = os.environ.get("MLX2_AMOEBA_AGENT_DB")
    if root:
        return Path(root)
    return Path.home() / "Library/Application Support/mlx2/research-artifacts/amoeba/agent/tracker.sqlite"


def job_key(kind: str, **ident) -> str:
    body = json.dumps({"kind": kind, **ident}, sort_keys=True)
    return kind + ":" + hashlib.sha256(body.encode()).hexdigest()[:20]


def _tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9_.]+", text.lower()) if len(w) > 2}


def jaccard(a: str, b: str) -> float:
    x, y = _tokens(a), _tokens(b)
    return len(x & y) / len(x | y) if x and y else 0.0


class Tracker:
    def __init__(self, path: Path | str | None = None, *, now=time.time):
        self.path = Path(path) if path else default_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        for table, col in (("jobs", "attempt_run TEXT"), ("decisions", "arm TEXT")):   # trackers made before v2
            have = {r["name"] for r in self.db.execute(f"PRAGMA table_info({table})")}
            if col.split()[0] not in have:
                self.db.execute(f"ALTER TABLE {table} ADD COLUMN {col}")
        self.now = now

    def close(self):
        self.db.close()

    # runs ---------------------------------------------------------------------------------------------------------
    def start_run(self, run_id, *, mode, goal, brain, commit, out):
        self.db.execute("INSERT INTO runs(id, started, mode, goal, brain, commit_rev, out) VALUES(?,?,?,?,?,?,?)",
                        (run_id, self.now(), mode, goal, brain, commit, str(out)))

    def finish_run(self, run_id, outcome):
        self.db.execute("UPDATE runs SET finished=?, outcome=? WHERE id=?", (self.now(), json.dumps(outcome), run_id))

    # jobs ---------------------------------------------------------------------------------------------------------
    def expire(self, lease_s: float = LEASE_S) -> int:
        """Leases whose heartbeat is older than lease_s return to the queue (an interrupted run's work resumes)."""
        cur = self.db.execute("UPDATE jobs SET state='queued', owner=NULL, updated=? WHERE state='leased' AND "
                              "heartbeat < ?", (self.now(), self.now() - lease_s))
        return cur.rowcount

    def job(self, key):
        r = self.db.execute("SELECT * FROM jobs WHERE key=?", (key,)).fetchone()
        return None if r is None else {**dict(r), "payload": json.loads(r["payload"] or "null"),
                                       "result": json.loads(r["result"] or "null")}

    def enqueue(self, kind, key, payload, run_id):
        """Exact dedup: an existing job with this key is returned as it is (terminal outcomes are reused)."""
        t = self.now()
        self.db.execute("INSERT OR IGNORE INTO jobs(key, kind, state, payload, run_id, created, updated) "
                        "VALUES(?,?,?,?,?,?,?)", (key, kind, "queued", json.dumps(payload), run_id, t, t))
        return self.job(key)

    def lease(self, key, owner, run_id="") -> bool:
        """Claim one job; False when another live owner holds it, it is terminal, or it ran out of attempts in this
        run (attempts count per run, so one run's crashes never retire a job for every later run)."""
        self.expire()
        t = self.now()
        self.db.execute("UPDATE jobs SET attempts=0, attempt_run=? WHERE key=? AND state IN ('queued','failed') AND "
                        "(attempt_run IS NULL OR attempt_run != ?)", (run_id, key, run_id))
        cur = self.db.execute("UPDATE jobs SET state='leased', owner=?, heartbeat=?, attempts=attempts+1, updated=? "
                              "WHERE key=? AND state IN ('queued','failed') AND attempts < ?",
                              (owner, t, t, key, MAX_ATTEMPTS))
        return cur.rowcount == 1

    def reset(self, key):
        """Operator or harness: send a job back to the queue (a stale reuse, a fixed bug). Its facts stay."""
        cur = self.db.execute("UPDATE jobs SET state='queued', owner=NULL, attempts=0, attempt_run=NULL, updated=? "
                              "WHERE key=? AND state != 'leased'", (self.now(), key))
        return cur.rowcount == 1

    def heartbeat(self, key, owner):
        self.db.execute("UPDATE jobs SET heartbeat=? WHERE key=? AND owner=? AND state='leased'",
                        (self.now(), key, owner))

    def finish(self, key, owner, state, result):
        if state not in JOB_STATES or state in ("queued", "leased"):
            raise ValueError(f"bad terminal state {state}")
        cur = self.db.execute("UPDATE jobs SET state=?, result=?, owner=NULL, updated=? WHERE key=? AND owner=?",
                              (state, json.dumps(result), self.now(), key, owner))
        if cur.rowcount != 1:
            raise RuntimeError(f"job {key} is not leased by {owner}")

    def release(self, key, owner):
        """Give a lease back without spending the attempt (the job was blocked before it ran, e.g. by admission)."""
        self.db.execute("UPDATE jobs SET state='queued', owner=NULL, attempts=MAX(attempts-1, 0), updated=? "
                        "WHERE key=? AND owner=? AND state='leased'", (self.now(), key, owner))

    def jobs(self, state=None):
        q = "SELECT key FROM jobs" + (" WHERE state=?" if state else "") + " ORDER BY id"
        return [self.job(r["key"]) for r in self.db.execute(q, (state,) if state else ())]

    # facts --------------------------------------------------------------------------------------------------------
    def add_fact(self, job_key_, kind, value, run_id, commit):
        self.db.execute("INSERT INTO facts(job_key, kind, value, run_id, commit_rev, created) VALUES(?,?,?,?,?,?)",
                        (job_key_, kind, json.dumps(value), run_id, commit, self.now()))

    def facts(self, job_key_=None, kind=None):
        q, args = "SELECT * FROM facts WHERE 1=1", []
        if job_key_:
            q, args = q + " AND job_key=?", args + [job_key_]
        if kind:
            q, args = q + " AND kind=?", args + [kind]
        return [{**dict(r), "value": json.loads(r["value"])} for r in self.db.execute(q + " ORDER BY id", args)]

    # decisions ----------------------------------------------------------------------------------------------------
    def next_arm(self, stage, brain):
        """Alternate lessons / control per (stage, brain), counting only decisions where lessons were available: a
        decision with no matching lesson is arm "none" and is left out of both the alternation and the comparison,
        so control rows are made at the same time, commit and brain as the lessons rows they are judged against."""
        n = self.db.execute("SELECT COUNT(*) FROM decisions WHERE stage=? AND brain=? AND arm IN ('lessons','control')",
                            (stage, brain)).fetchone()[0]
        return "lessons" if n % 2 else "control"

    def record_decision(self, run_id, stage, brain, prompt, options, scores, chosen, lessons, arm="control"):
        cur = self.db.execute("INSERT INTO decisions(run_id, stage, brain, prompt, options, scores, chosen, lessons, "
                              "created, arm) VALUES(?,?,?,?,?,?,?,?,?,?)",
                              (run_id, stage, brain, prompt, json.dumps(options), json.dumps(scores),
                               json.dumps(chosen), json.dumps(lessons), self.now(), arm))
        return cur.lastrowid

    def arm_accuracy(self, stage, arm, brain=None):
        q = "SELECT COUNT(*) n, SUM(correct) ok FROM decisions WHERE stage=? AND arm=? AND correct IS NOT NULL"
        args = [stage, arm]
        if brain is not None:
            q, args = q + " AND brain=?", args + [brain]
        r = self.db.execute(q, args).fetchone()
        return ((r["ok"] / r["n"]) if r["n"] else None), r["n"]

    def label(self, decision_id, oracle, correct: bool):
        """Attach the deterministic oracle to a decision and credit the lessons it was shown."""
        row = self.db.execute("SELECT stage, brain, lessons, correct FROM decisions WHERE id=?",
                              (decision_id,)).fetchone()
        if row is None or row["correct"] is not None:
            return
        self.db.execute("UPDATE decisions SET oracle=?, correct=? WHERE id=?",
                        (json.dumps(oracle), int(bool(correct)), decision_id))
        stage, brain = row["stage"], row["brain"]
        control, n_control = self.arm_accuracy(stage, "control", brain)
        col = "helped" if correct else "hurt"
        for lid in json.loads(row["lessons"] or "[]"):
            # shown counts labelled showings only: an unlabelled decision says nothing about the lesson
            self.db.execute(f"UPDATE lessons SET {col}={col}+1, shown=shown+1 WHERE id=?", (lid,))
            self.db.execute("INSERT OR IGNORE INTO lesson_credit(lesson, stage, brain) VALUES(?,?,?)",
                            (lid, stage, brain))
            self.db.execute(f"UPDATE lesson_credit SET {col}={col}+1 WHERE lesson=? AND stage=? AND brain=?",
                            (lid, stage, brain))
            c = self.db.execute("SELECT helped, hurt FROM lesson_credit WHERE lesson=? AND stage=? AND brain=?",
                                (lid, stage, brain)).fetchone()
            shown = c["helped"] + c["hurt"]
            # judged per stage and brain against that stage's no-lesson arm, with a margin, once both sides have
            # enough labelled decisions; nothing is retired on a handful of noisy labels
            # retirement is global by design: a lesson shown to do harm on one (stage, brain) is not recalled anywhere
            if shown >= RETIRE_AFTER_SHOWN and n_control >= CONTROL_MIN and \
                    c["helped"] / shown < control - RETIRE_MARGIN:
                self.db.execute("UPDATE lessons SET retired=1 WHERE id=?", (lid,))

    def decisions(self, labelled_only=False):
        q = "SELECT * FROM decisions" + (" WHERE correct IS NOT NULL" if labelled_only else "") + " ORDER BY id"
        out = []
        for r in self.db.execute(q):
            d = dict(r)
            for k in ("options", "scores", "chosen", "lessons", "oracle"):
                d[k] = json.loads(d[k]) if d[k] else None
            out.append(d)
        return out

    # lessons ------------------------------------------------------------------------------------------------------
    def add_lesson(self, text, tags, run_id):
        text = " ".join(str(text).split())[:300]
        if len(_tokens(text)) < 3:
            return None
        dup = [dict(r) for r in self.db.execute("SELECT id, text, retired FROM lessons")
               if jaccard(r["text"], text) >= LESSON_DUP_JACCARD]
        live = [r for r in dup if not r["retired"]]
        if live:
            return live[0]["id"]
        if dup:
            return None            # a retired lesson does not come back as a new one
        cur = self.db.execute("INSERT INTO lessons(text, tags, source_run, created) VALUES(?,?,?,?)",
                              (text, json.dumps(sorted(set(tags))), run_id, self.now()))
        return cur.lastrowid

    def recall(self, tags, k=RECALL_K):
        """Live lessons sharing a tag: the best k-1 by record (helped rate, then newest) plus the least-shown other
        one, so a new lesson always gets showings and the early ones cannot lock the slots."""
        tags = set(tags)
        rows = [dict(r) for r in self.db.execute("SELECT * FROM lessons WHERE retired=0")]
        rows = [r for r in rows if tags & set(json.loads(r["tags"]))]
        rows.sort(key=lambda r: (-(r["helped"] / r["shown"] if r["shown"] else 0.5), -r["created"]))
        best = rows[:max(k - 1, 0)]
        rest = sorted(rows[len(best):], key=lambda r: (r["shown"], -r["created"]))
        pick = best + rest[:1] if k else []
        return [{"id": r["id"], "text": r["text"]} for r in pick]

    def lessons(self):
        return [dict(r) for r in self.db.execute("SELECT * FROM lessons ORDER BY id")]

    # competence ---------------------------------------------------------------------------------------------------
    def competence(self):
        """The harness's own ingestion ability, from deterministic records only."""
        by_stage = {}
        for r in self.db.execute("SELECT stage, brain, arm, COUNT(*) n, SUM(correct) ok FROM decisions "
                                 "WHERE correct IS NOT NULL GROUP BY stage, brain, arm"):
            s = by_stage.setdefault(r["stage"], {}).setdefault(r["brain"], {"labelled": 0, "correct": 0, "arms": {}})
            s["labelled"] += r["n"]
            s["correct"] += r["ok"]
            s["accuracy"] = round(s["correct"] / s["labelled"], 3)
            s["arms"][r["arm"] or "control"] = {"labelled": r["n"], "accuracy": round(r["ok"] / r["n"], 3)}
        jobs = {}
        for r in self.db.execute("SELECT kind, state, COUNT(*) n FROM jobs GROUP BY kind, state"):
            jobs.setdefault(r["kind"], {})[r["state"]] = r["n"]
        verdicts = {}
        for f in self.facts(kind="box_verdict"):
            v = f["value"].get("verdict")
            verdicts[v] = verdicts.get(v, 0) + 1
        installs = {}
        for f in self.facts(kind="install_verdict"):
            for route, v in (f["value"].get("verdicts") or {}).items():
                installs.setdefault(route, {}).setdefault(v, 0)
                installs[route][v] += 1
        runs = [dict(r) for r in self.db.execute("SELECT id, started, finished, outcome FROM runs ORDER BY started")]
        return {"decision_accuracy": by_stage, "jobs": jobs, "box_verdicts": verdicts, "install_verdicts": installs,
                "lessons": {"live": sum(1 for x in self.lessons() if not x["retired"]),
                            "retired": sum(1 for x in self.lessons() if x["retired"])},
                "runs": len(runs)}
