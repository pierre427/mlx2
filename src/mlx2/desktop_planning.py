"""Small durable planning graph; model claims never constitute verification.

Original implementation. Explicit session IDs, immutable goals, dependency
ordering, monotonically fenced leases and append-only events replace implicit
shared current-session state. This graph is context, never execution authority.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid


def bounded_text(value, name, maximum=65536):
    if not isinstance(value, str) or not value.strip() or len(value.encode()) > maximum:
        raise ValueError(f"{name} must be nonempty text of at most {maximum} bytes")
    return value


class Planning:
    def __init__(self, controller):
        self.controller = controller
        self.lock = threading.RLock()
        path = controller.home / "planning.sqlite3"
        self.db = sqlite3.connect(path, check_same_thread=False)
        os.chmod(path, 0o600)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA foreign_keys=ON;
            CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY, goal TEXT NOT NULL, status TEXT NOT NULL, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS tasks(id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id), goal TEXT NOT NULL,
                dependencies TEXT NOT NULL, status TEXT NOT NULL, owner TEXT, fence INTEGER NOT NULL DEFAULT 0,
                expires REAL, result TEXT, verification TEXT);
            CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
                task_id TEXT, action TEXT NOT NULL, payload TEXT NOT NULL, time REAL NOT NULL);
        """)
        self.db.commit()

    def _event(self, session, action, task=None, **payload):
        self.db.execute("INSERT INTO events(session_id,task_id,action,payload,time) VALUES(?,?,?,?,?)",
                        (session, task, action, json.dumps(payload, allow_nan=False), time.time()))

    def _session(self, identity, active=False):
        row = self.db.execute("SELECT * FROM sessions WHERE id=?", (identity,)).fetchone()
        if row is None:
            raise ValueError("Unknown explicit planning session")
        if active and row["status"] != "active":
            raise ValueError("Planning session is not active")
        return row

    def _task(self, session, identity):
        row = self.db.execute("SELECT * FROM tasks WHERE session_id=? AND id=?", (session, identity)).fetchone()
        if row is None:
            raise ValueError("Task does not belong to this session")
        return row

    def _expire(self, session):
        for row in self.db.execute("SELECT id,fence FROM tasks WHERE session_id=? AND status='running' AND expires<=?", (session, time.time())).fetchall():
            self.db.execute("UPDATE tasks SET status='pending',owner=NULL,expires=NULL,fence=fence+1 WHERE id=?", (row["id"],))
            self._event(session, "lease_expired", row["id"], fence=row["fence"] + 1)

    def snapshot(self):
        with self.lock:
            return {"sessions": [dict(row) for row in self.db.execute("SELECT * FROM sessions ORDER BY created DESC LIMIT 100")],
                    "authority": "context_only; reports are claims until explicit human verification"}

    def dispatch(self, request):
        if request["command"] not in {"plan_get", "plan_cancel"} and not self.controller.preferences.get("built_in_planning_enabled", True):
            raise ValueError("Built-in planning is disabled")
        command = request["command"]
        with self.lock, self.db:
            if command == "plan_create":
                identity = uuid.uuid4().hex
                goal = bounded_text(request.get("goal"), "Session goal", 16384)
                self.db.execute("INSERT INTO sessions VALUES(?,?,?,?)", (identity, goal, "active", time.time()))
                self._event(identity, "session_created")
                return {"session_id": identity, "goal": goal}
            session = request.get("session_id")
            if not isinstance(session, str):
                raise ValueError("An explicit session_id is required")  # noqa: TRY004 - missing explicit session is invalid operation
            self._session(session, active=command not in {"plan_get", "plan_cancel"})
            self._expire(session)
            if command == "plan_get":
                tasks = []
                for row in self.db.execute("SELECT * FROM tasks WHERE session_id=? ORDER BY rowid", (session,)):
                    task = dict(row)
                    for field in ("dependencies", "result", "verification"):
                        task[field] = json.loads(task[field]) if task[field] is not None else None
                    tasks.append(task)
                return {"session": dict(self._session(session)), "tasks": tasks,
                        "events": [dict(row) for row in self.db.execute("SELECT * FROM events WHERE session_id=? ORDER BY seq DESC LIMIT 200", (session,))]}
            if command == "plan_cancel":
                self.db.execute("UPDATE sessions SET status='cancelled' WHERE id=?", (session,))
                self.db.execute("UPDATE tasks SET status='cancelled',owner=NULL,expires=NULL,fence=fence+1 WHERE session_id=? AND status IN ('pending','running')", (session,))
                self._event(session, "session_cancelled")
                return {"cancelled": session}
            if command == "plan_add":
                goal = bounded_text(request.get("goal"), "Worker goal", 16384)
                dependencies = request.get("dependencies", [])
                if not isinstance(dependencies, list) or len(dependencies) > 32 or any(not isinstance(d, str) for d in dependencies) or len(set(dependencies)) != len(dependencies):
                    raise ValueError("Dependencies must be unique task IDs (at most 32)")
                for dependency in dependencies:
                    self._task(session, dependency)
                if self.db.execute("SELECT count(*) FROM tasks WHERE session_id=?", (session,)).fetchone()[0] >= 128:
                    raise ValueError("Planning session task limit reached")
                identity = uuid.uuid4().hex
                self.db.execute("INSERT INTO tasks(id,session_id,goal,dependencies,status) VALUES(?,?,?,?,?)", (identity, session, goal, json.dumps(dependencies), "pending"))
                self._event(session, "task_added", identity)
                return {"task_id": identity, "goal": goal}
            task = self._task(session, request.get("task_id"))
            if command == "plan_claim":
                owner = bounded_text(request.get("owner"), "Owner", 128)
                lease = request.get("lease_seconds", 300)
                if type(lease) is not int or not 30 <= lease <= 3600:
                    raise ValueError("Lease must be 30–3600 seconds")
                if task["status"] != "pending":
                    raise ValueError("Task is not available")
                for dependency in json.loads(task["dependencies"]):
                    if self._task(session, dependency)["status"] != "verified":
                        raise ValueError("Dependencies require verified results")
                fence = task["fence"] + 1
                expires = time.time() + lease
                self.db.execute("UPDATE tasks SET status='running',owner=?,fence=?,expires=? WHERE id=?", (owner, fence, expires, task["id"]))
                self._event(session, "task_claimed", task["id"], owner=owner, fence=fence, expires=expires)
                return {"task_id": task["id"], "goal": task["goal"], "owner": owner, "fence": fence, "expires": expires}
            if command == "plan_report":
                if task["status"] != "running" or task["owner"] != request.get("owner") or type(request.get("fence")) is not int or task["fence"] != request["fence"] or task["expires"] <= time.time():
                    raise ValueError("Stale, expired or foreign task lease")
                result = request.get("result")
                encoded = json.dumps(result, allow_nan=False)
                if result is None or len(encoded.encode()) > 65536:
                    raise ValueError("Result must be bounded JSON of at most 64 KiB")
                self.db.execute("UPDATE tasks SET status='claimed',result=?,owner=NULL,expires=NULL WHERE id=?", (encoded, task["id"]))
                self._event(session, "result_claimed", task["id"], fence=task["fence"])
                return {"task_id": task["id"], "status": "claimed", "verified": False}
            if command == "plan_verify":
                if request.get("_user_verified") is not True:
                    raise ValueError("Verification requires a visible native user approval")
                if task["status"] != "claimed":
                    raise ValueError("Only a claimed result can be verified")
                evidence = bounded_text(request.get("evidence"), "Reviewed evidence", 16384)
                proof = {"reviewer": "native_user", "evidence": evidence, "time": time.time(), "kind": "human_review_not_automated_checker"}
                self.db.execute("UPDATE tasks SET status='verified',verification=? WHERE id=?", (json.dumps(proof), task["id"]))
                self._event(session, "result_verified", task["id"], reviewer="native_user")
                return {"task_id": task["id"], "status": "verified"}
            raise ValueError("Unknown planning operation")

    def close(self):
        with self.lock:
            self.db.close()
