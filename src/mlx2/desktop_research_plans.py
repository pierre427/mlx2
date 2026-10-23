"""Durable, profile-scoped observations of research work in a chat turn.

This is a progress ledger, not human verification or a fabricated research plan.
The explicit planning graph remains available for model-authored tasks and leases.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid

RESEARCH_TOOLS = {"research_search", "research_read", "web_search", "fetch_page"}
NOTICE = "Observed tool activity and response lifecycle only; sources and conclusions are not independently verified."


class ResearchPlans:
    def __init__(self, controller):
        self.controller = controller
        self.lock = threading.RLock()
        path = controller.home / "research-plans.sqlite3"
        self.db = sqlite3.connect(path, check_same_thread=False)
        os.chmod(path, 0o600)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS research_plans (
                id TEXT PRIMARY KEY, session_id TEXT NOT NULL, parent_id TEXT NOT NULL,
                profile_id TEXT NOT NULL, goal TEXT NOT NULL, status TEXT NOT NULL,
                created REAL NOT NULL, updated REAL NOT NULL,
                UNIQUE(profile_id, session_id));
            CREATE TABLE IF NOT EXISTS research_events (
                id TEXT PRIMARY KEY, plan_id TEXT NOT NULL, tool TEXT NOT NULL,
                stage TEXT NOT NULL, status TEXT NOT NULL, started REAL NOT NULL,
                ended REAL, evidence TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS research_children (
                job_id TEXT PRIMARY KEY, plan_id TEXT NOT NULL, status TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS research_consumptions (
                job_id TEXT PRIMARY KEY, session_id TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS research_parent ON research_plans(profile_id, parent_id, created);
        """)
        now = time.time()
        self.db.execute("UPDATE research_plans SET status='interrupted', updated=? WHERE status='active'", (now,))
        self.db.execute("UPDATE research_events SET status='interrupted', ended=? WHERE status='running'", (now,))
        self.db.execute("UPDATE research_children SET status='interrupted' WHERE status IN ('queued', 'running')")
        self.db.commit()

    def _profile(self):
        return self.controller.user_profiles.active

    @staticmethod
    def _text(value, maximum=512):
        return value[:maximum] if isinstance(value, str) else ""

    def begin(self, session_id, session, operation, arguments):
        if operation not in RESEARCH_TOOLS:
            raise ValueError("Not a research operation")
        profile = self._profile()
        if session.get("user_profile") != profile:
            raise ValueError("Research plan belongs to a different profile")
        parent = session.get("chat_session_id") or session_id
        if not isinstance(session_id, str) or not session_id or not isinstance(parent, str):
            raise ValueError("Research needs a tool session and conversation")
        now = time.time()
        with self.lock, self.db:
            plan = self.db.execute("SELECT * FROM research_plans WHERE profile_id=? AND session_id=?", (profile, session_id)).fetchone()
            if plan is None:
                count = self.db.execute("SELECT COUNT(*) FROM research_plans").fetchone()[0]
                if count >= 4096:
                    raise ValueError("Research plan archive reached its 4096-plan limit")
                identity = uuid.uuid4().hex
                goal = self._text(arguments.get("query") or arguments.get("url") or arguments.get("source_id"), 300)
                self.db.execute("INSERT INTO research_plans VALUES(?,?,?,?,?,?,?,?)",
                                (identity, session_id, parent, profile, goal, "active", now, now))
            else:
                if plan["parent_id"] != parent or plan["status"] != "active":
                    raise ValueError("Research plan is no longer active for this conversation")
                identity = plan["id"]
            count = self.db.execute("SELECT COUNT(*) FROM research_events WHERE plan_id=?", (identity,)).fetchone()[0]
            if count >= 256:
                raise ValueError("Research plan reached its 256-call observation limit")
            event_id = uuid.uuid4().hex
            stage = "search" if operation.endswith("search") else "read"
            detail = {"request": self._text(arguments.get("query") or arguments.get("url") or arguments.get("source_id"))}
            self.db.execute("INSERT INTO research_events VALUES(?,?,?,?,?,?,?,?)",
                            (event_id, identity, operation, stage, "running", now, None, json.dumps(detail)))
            self.db.execute("UPDATE research_plans SET updated=? WHERE id=?", (now, identity))
            return event_id

    @staticmethod
    def _failed_result(result):
        if not isinstance(result, dict):
            return False
        status = str(result.get("status", ""))
        return bool(result.get("error")) or (status.isdecimal() and int(status) >= 400)

    @classmethod
    def _evidence(cls, result):
        if not isinstance(result, dict):
            return {"observation": "Tool returned a result; content not retained in the plan."}
        evidence = {}
        for key in ("provider", "url", "source_id", "title", "status"):
            if key in result:
                evidence[key] = cls._text(result[key])
        if cls._failed_result(result):
            # Do not persist arbitrary provider payloads, credentials or page text.
            evidence["observation"] = "Tool reported an error; inspect its retained result."
            return evidence
        rows = result.get("results")
        if isinstance(rows, list):
            evidence["returned_references"] = len(rows)
            evidence["references"] = [{key: cls._text(row[key]) for key in ("id", "url", "title") if key in row}
                                      for row in rows[:8] if isinstance(row, dict)]
        if isinstance(result.get("revision"), dict):
            evidence["revision"] = {key: value for key, value in result["revision"].items()
                                    if key in {"revid", "parentid", "timestamp"} and isinstance(value, (str, int))}
        evidence["observation"] = "Tool returned successfully; source truth and synthesis are not verified."
        return evidence

    def record(self, event_id, *, result=None, error=None):
        now = time.time()
        evidence = self._evidence(result)
        status = "failed" if error is not None or self._failed_result(result) else "completed"
        if error is not None:
            evidence = {"error_type": type(error).__name__, "observation": "Tool raised an error; no successful read/search is claimed."}
        with self.lock, self.db:
            event = self.db.execute("SELECT * FROM research_events WHERE id=?", (event_id,)).fetchone()
            if event is None:
                raise ValueError("Unknown research observation")
            if event["stage"] == "read" and isinstance(result, dict):
                text = result.get("content", result.get("extract"))
                if isinstance(text, str):
                    evidence["returned_characters"] = len(text)
                    if not text.strip():
                        status = "failed"
                        evidence["observation"] = "Reader returned no usable source text; no successful read is claimed."
            evidence = {**json.loads(event["evidence"]), **evidence}
            self.db.execute("UPDATE research_events SET status=?, ended=?, evidence=? WHERE id=? AND status='running'",
                            (status, now, json.dumps(evidence, allow_nan=False), event_id))
            self.db.execute("UPDATE research_plans SET updated=? WHERE id=?", (now, event["plan_id"]))
            plan = self.db.execute("SELECT * FROM research_plans WHERE id=?", (event["plan_id"],)).fetchone()
            return self._receipt(plan)

    def execute(self, session_id, session, operation, arguments, action):
        event = self.begin(session_id, session, operation, arguments)
        try:
            result = action()
        except Exception as error:
            self.record(event, error=error)
            raise
        receipt = self.record(event, result=result)
        return result, receipt

    @staticmethod
    def _receipt(plan):
        return {"id": plan["id"], "status": plan["status"], "parent_id": plan["parent_id"],
                "guidance": "A research progress plan records this turn's actual search/read activity. Review returned evidence, adapt the next query or read if needed, and distinguish source claims from your inference. Do not repeat a completed query without a reason. Finish with an answer or an explicit limitation; tool success does not verify a claim.",
                "notice": NOTICE}

    def attach_child(self, session_id, session, job_id, goal):
        """Record delegation before it runs; do not invent a successful search."""
        profile = self._profile()
        if session.get("user_profile") != profile or not isinstance(job_id, str) or not job_id:
            raise ValueError("Research child requires this profile's admitted session")
        parent = session.get("chat_session_id") or session_id
        now = time.time()
        with self.lock, self.db:
            plan = self.db.execute("SELECT * FROM research_plans WHERE profile_id=? AND session_id=?", (profile, session_id)).fetchone()
            if plan is None:
                if self.db.execute("SELECT COUNT(*) FROM research_plans").fetchone()[0] >= 4096:
                    raise ValueError("Research plan archive reached its 4096-plan limit")
                identity = uuid.uuid4().hex
                self.db.execute("INSERT INTO research_plans VALUES(?,?,?,?,?,?,?,?)",
                                (identity, session_id, parent, profile, self._text(goal, 300), "active", now, now))
                plan = self.db.execute("SELECT * FROM research_plans WHERE id=?", (identity,)).fetchone()
            if plan["status"] != "active" or plan["parent_id"] != parent:
                raise ValueError("Research plan is no longer active for this conversation")
            existing = self.db.execute("SELECT * FROM research_children WHERE job_id=?", (job_id,)).fetchone()
            if existing and existing["plan_id"] != plan["id"]:
                raise ValueError("Research child already belongs to another plan")
            self.db.execute("INSERT OR IGNORE INTO research_children VALUES(?,?,?)", (job_id, plan["id"], "queued"))
            return {**self._receipt(plan), "child_job_id": job_id}

    def execute_child(self, job_id, operation, arguments, action):
        with self.lock, self.db:
            child = self.db.execute("SELECT * FROM research_children WHERE job_id=?", (job_id,)).fetchone()
            if child is None:
                raise ValueError("Research child is not attached to a plan")
            plan = self.db.execute("SELECT * FROM research_plans WHERE id=?", (child["plan_id"],)).fetchone()
            if plan["profile_id"] != self._profile() or plan["status"] != "active":
                raise ValueError("Research child plan is inactive or belongs to another profile")
            self.db.execute("UPDATE research_children SET status='running' WHERE job_id=?", (job_id,))
            session_id = plan["session_id"]
            session = {"user_profile": plan["profile_id"], "chat_session_id": plan["parent_id"]}
        return self.execute(session_id, session, operation, arguments, action)

    def child_finished(self, job_id, outcome):
        if outcome not in {"completed", "failed", "cancelled"}:
            raise ValueError("Invalid research child outcome")
        with self.lock, self.db:
            child = self.db.execute("SELECT * FROM research_children WHERE job_id=?", (job_id,)).fetchone()
            if child is None:
                return {"status": "not_started"}
            self.db.execute("UPDATE research_children SET status=? WHERE job_id=?", (outcome, job_id))
            self.db.execute("UPDATE research_plans SET updated=? WHERE id=?", (time.time(), child["plan_id"]))
            plan = self.db.execute("SELECT * FROM research_plans WHERE id=?", (child["plan_id"],)).fetchone()
            return self._receipt(plan)

    def mark_child_consumed(self, job_id, session_id):
        """A completed child report was returned to an admitted parent tool call."""
        profile = self._profile()
        session = getattr(self.controller, "tool_sessions", {}).get(session_id)
        with self.lock, self.db:
            child = self.db.execute("SELECT * FROM research_children WHERE job_id=?", (job_id,)).fetchone()
            if child is None:
                return
            plan = self.db.execute("SELECT * FROM research_plans WHERE id=?", (child["plan_id"],)).fetchone()
            if (not session or session.get("user_profile") != profile or profile != plan["profile_id"]
                    or session.get("chat_session_id") != plan["parent_id"]):
                raise ValueError("Research report consumption is outside this conversation's scope")
            if child["status"] not in {"queued", "running"}:
                self.db.execute("INSERT OR REPLACE INTO research_consumptions VALUES(?,?)", (job_id, session_id))

    def finish(self, session_id, outcome, *, parent_id=None):
        if not isinstance(session_id, str) or not session_id or outcome not in {"completed", "failed", "cancelled"}:
            raise ValueError("Research completion needs session_id and completed, failed or cancelled outcome")
        profile = self._profile()
        if parent_id is not None:
            session = getattr(self.controller, "tool_sessions", {}).get(session_id)
            if not session or session.get("user_profile") != profile or session.get("chat_session_id") != parent_id:
                raise ValueError("Research completion is outside this conversation's scope")
        with self.lock, self.db:
            if parent_id is None:
                plans = self.db.execute("SELECT * FROM research_plans WHERE profile_id=? AND session_id=?", (profile, session_id)).fetchall()
            else:
                plans = self.db.execute("SELECT * FROM research_plans WHERE profile_id=? AND parent_id=? AND status='active'", (profile, parent_id)).fetchall()
            if not plans:
                if self.db.execute("SELECT 1 FROM research_plans WHERE session_id=? AND profile_id<>?", (session_id, profile)).fetchone():
                    raise ValueError("Research plan belongs to a different profile")
                return {"status": "not_started"}
            values = []
            for plan in plans:
                pending = self.db.execute("SELECT COUNT(*) FROM research_children WHERE plan_id=? AND status IN ('queued','running')", (plan["id"],)).fetchone()[0]
                unconsumed = self.db.execute("""SELECT COUNT(*) FROM research_children c
                    LEFT JOIN research_consumptions r ON c.job_id=r.job_id
                    WHERE c.plan_id=? AND r.job_id IS NULL""", (plan["id"],)).fetchone()[0]
                awaiting_report = plan["session_id"] == session_id and unconsumed > 0
                if plan["status"] == "active" and not (outcome == "completed" and (pending or awaiting_report)):
                    now = time.time()
                    self.db.execute("UPDATE research_plans SET status=?, updated=? WHERE id=?", (outcome, now, plan["id"]))
                    self.db.execute("UPDATE research_events SET status='interrupted', ended=? WHERE plan_id=? AND status='running'", (now, plan["id"]))
                values.append(self._value(self.db.execute("SELECT * FROM research_plans WHERE id=?", (plan["id"],)).fetchone()))
            return {"plans": values, "status": "active" if any(value["status"] == "active" for value in values) else outcome}

    def _value(self, row):
        value = dict(row)
        events = [dict(event) for event in self.db.execute("SELECT * FROM research_events WHERE plan_id=? ORDER BY started LIMIT 256", (row["id"],))]
        for event in events:
            event["evidence"] = json.loads(event["evidence"])
        steps = []
        for stage, label in (("search", "Search for source references"), ("read", "Read source material")):
            matches = [event for event in events if event["stage"] == stage]
            status = ("running" if any(event["status"] == "running" for event in matches) else
                      "completed" if any(event["status"] == "completed" for event in matches) else
                      "interrupted" if any(event["status"] == "interrupted" for event in matches) else
                      "failed" if matches else "pending" if row["status"] == "active" else "not_observed")
            steps.append({"id": stage, "label": label, "status": status, "observed_calls": len(matches)})
        steps.append({"id": "synthesize", "label": "Conclude response (not independently verified)",
                      "status": "completed" if row["status"] == "completed" else "pending" if row["status"] == "active" else "not_observed", "observed_calls": 0})
        children = [dict(child) for child in self.db.execute("SELECT * FROM research_children WHERE plan_id=?", (row["id"],))]
        waiting = ""
        if row["status"] == "active" and children:
            waiting = ("Research agent is working; the parent may continue independent work." if any(child["status"] in {"queued", "running"} for child in children)
                       else "Research agent finished; awaiting the parent response.")
        value.update(steps=steps, events=events[-40:], total_events=len(events), children=children, waiting=waiting, notice=NOTICE)
        return value

    def snapshot(self, *, parent_id=None):
        with self.lock:
            query = "SELECT * FROM research_plans WHERE profile_id=?"
            parameters = [self._profile()]
            if parent_id is not None:
                query += " AND parent_id=?"
                parameters.append(parent_id)
            query += " ORDER BY updated DESC LIMIT 100"
            return {"plans": [self._value(row) for row in self.db.execute(query, parameters)], "authority": NOTICE}

    def close(self):
        with self.lock:
            self.db.close()
