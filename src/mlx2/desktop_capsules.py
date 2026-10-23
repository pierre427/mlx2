"""Scoped exact semantic-capsule snapshots and operator-selected autonomy.

Desktop notes are stored as evaluation capsules, never fabricated neural or
activation payloads. Runtime application requires a separate, actually exposed
qualified operation; this bridge has no such operation and refuses it.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

from .desktop import atomic_json, source_identity
from .runtime.semantic_capsules import CapsuleStore

AUTONOMY = {"off", "situational", "always"}
SITUATIONS = {"user_requested", "correction", "tool_failure", "task_completed", "useful_feedback", "contradiction"}
DEFAULT_POLICY = {"revision": 1, "autonomy": "off", "allowed_situations": ["user_requested", "correction", "tool_failure", "task_completed"],
                  "allow_promote": False, "max_operations_per_session": 8, "max_capsules_per_scope": 256,
                  "maximum_envelope_bytes": 262144}


class CapsuleController:
    def __init__(self, controller):
        self.controller = controller
        self.home = controller.home / "agent-capsules"
        self.home.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.policy_path = self.home / "policy.json"
        self.policy = {**DEFAULT_POLICY, **(json.loads(self.policy_path.read_text()) if self.policy_path.exists() else {})}
        self.validate(self.policy)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.home / "index.sqlite3", check_same_thread=False)
        (self.home / "index.sqlite3").chmod(0o600)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS capsules (
                digest TEXT PRIMARY KEY, scope TEXT NOT NULL, created REAL NOT NULL,
                author TEXT NOT NULL, purpose TEXT NOT NULL, note_id TEXT,
                note_revision INTEGER, authority TEXT NOT NULL, policy_revision INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS operations (
                principal TEXT NOT NULL, session_id TEXT NOT NULL, policy_revision INTEGER NOT NULL,
                count INTEGER NOT NULL, PRIMARY KEY(principal,session_id,policy_revision));
        """)
        self.db.commit()

    @staticmethod
    def validate(policy):
        if set(policy) != set(DEFAULT_POLICY) or policy.get("autonomy") not in AUTONOMY:
            raise ValueError("Capsule autonomy must be off, situational or always with known policy fields")
        situations = policy.get("allowed_situations")
        if not isinstance(situations, list) or any(s not in SITUATIONS for s in situations):
            raise ValueError("Unknown autonomous memory-review situation")
        if type(policy["allow_promote"]) is not bool:
            raise ValueError("Allow promotion must be explicitly boolean")
        for key, low, high in (("revision", 1, 2**31), ("max_operations_per_session", 1, 64),
                               ("max_capsules_per_scope", 1, 4096), ("maximum_envelope_bytes", 4096, 1048576)):
            if type(policy[key]) is not int or not low <= policy[key] <= high:
                raise ValueError("Capsule policy limit is invalid: " + key)

    def configure(self, changes, *, context=None):
        if context is not None and context.get("policy_approved") is not True:
            raise PermissionError("Changing autonomy policy requires explicit operator approval")
        if not isinstance(changes, dict) or set(changes) - (set(DEFAULT_POLICY) - {"revision"}):
            raise ValueError("Unknown capsule policy changes")
        with self.lock:
            policy = {**self.policy, **changes, "revision": self.policy["revision"] + 1}
            self.validate(policy)
            atomic_json(self.policy_path, policy)
            self.policy = policy
            return {"saved": True, "policy": dict(policy)}

    def actor(self, context):
        return self.controller.memory._actor(context)

    def authorize(self, scope, actor):
        self.controller.memory._authorize(scope, actor)

    def store(self, scope):
        identity = hashlib.sha256(scope.encode()).hexdigest()
        return CapsuleStore(self.home / "scopes" / identity, maximum_envelope_bytes=self.policy["maximum_envelope_bytes"])

    def allowed(self, actor, context, request, scope, *, promotion=False):
        self.authorize(scope, actor)
        if actor["operator"] or (context or {}).get("capsule_operation_approved") is True:
            return "operator-confirmed"
        autonomy = self.policy["autonomy"]
        situation = request.get("situation")
        if autonomy == "off":
            raise PermissionError("Autonomous capsule and memory review is off; request operator approval")
        if autonomy == "situational" and situation not in self.policy["allowed_situations"]:
            raise PermissionError("This situation is not allowed by the saved memory policy")
        if promotion and not self.policy["allow_promote"]:
            raise PermissionError("The saved autonomy policy does not permit model-driven promotion")
        # Policy authorizes the attempt, not the truth of the model's declared
        # situation or the resulting claim. Receipts preserve that distinction.
        return "model-policy-unverified"

    def consume(self, actor, context):
        if actor["operator"]:
            return
        session = (context or {}).get("operation_session_id")
        sessions = actor["session_ids"]
        if not session:
            session = sessions[0] if len(sessions) == 1 else None
        if not session or session not in sessions:
            raise PermissionError("Autonomous operations require one explicitly admitted session identity")
        row = self.db.execute("SELECT count FROM operations WHERE principal=? AND session_id=? AND policy_revision=?", (actor["principal_id"], session, self.policy["revision"])).fetchone()
        count = row[0] if row else 0
        if count >= self.policy["max_operations_per_session"]:
            raise ValueError("Autonomous capsule/review operation budget exhausted for this session")
        self.db.execute("INSERT INTO operations VALUES (?,?,?,?) ON CONFLICT(principal,session_id,policy_revision) DO UPDATE SET count=excluded.count", (actor["principal_id"], session, self.policy["revision"], count + 1))

    def put(self, note, actor, authority, purpose, data, *, parents=(), context=None):
        scope = note["scope"]
        store = self.store(scope)
        for parent in parents:
            row = self.db.execute("SELECT scope FROM capsules WHERE digest=?", (parent,)).fetchone()
            if not row or row["scope"] != scope:
                raise PermissionError("Capsule parents must be admitted objects in the exact same scope")
            store.get(parent)
        count = self.db.execute("SELECT count(*) FROM capsules WHERE scope=?", (scope,)).fetchone()[0]
        if count >= self.policy["max_capsules_per_scope"]:
            raise ValueError("Capsule scope capacity reached; inspect and remove unreferenced snapshots")
        identity = store.put(kind="evaluation", data=data, parents=parents,
            model_binding=(context or {}).get("model_binding"), tokenizer_binding=(context or {}).get("tokenizer_binding"),
            runtime_binding=(context or {}).get("runtime_binding") or hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            provenance={"schema": "mlx2-desktop-capsule-provenance-v1", "scope": scope,
                "author": actor["principal_id"], "authority": authority, "created_at": time.time(),
                "note_id": note["id"], "note_revision": note["revision"], "purpose": purpose,
                "policy_revision": self.policy["revision"], "source": source_identity(),
                "activation_applied": False, "approximate_operation": None})
        self.db.execute("INSERT OR IGNORE INTO capsules VALUES (?,?,?,?,?,?,?,?,?)", (identity.digest, scope, time.time(), actor["principal_id"], purpose, note["id"], note["revision"], authority, self.policy["revision"]))
        return {"digest": identity.digest, "scope": scope, "kind": "evaluation", "purpose": purpose,
                "note_id": note["id"], "note_revision": note["revision"], "authority": authority,
                "activation_applied": False, "qualified_operation": None}

    def get(self, digest, actor):
        row = self.db.execute("SELECT * FROM capsules WHERE digest=?", (digest,)).fetchone()
        if not row:
            raise ValueError("Capsule does not exist in this desktop index")
        self.authorize(row["scope"], actor)
        value = self.store(row["scope"]).get(digest)
        if not actor["operator"]:
            # Metadata is exact; large source text is retrieved through the
            # bounded memory tools instead of dumping a capsule into context.
            value["data"] = {k: v for k, v in value["data"].items() if k not in {"note", "body"}}
            value["reference_only"] = True
        return value

    def dispatch(self, request, *, context=None):
        if any(k in request for k in ("context", "principal_id", "allowed_scopes", "can_promote", "policy_approved", "capsule_operation_approved")):
            raise ValueError("Capsule tool arguments cannot grant authority")
        actor = self.actor(context)
        command = request["command"]
        if command not in {"capsule_policy", "capsule_list", "capsule_get"} and self.controller.preferences.get("memory_enabled", True) is not True:
            raise ValueError("Agent memory is disabled")
        with self.lock, self.db:
            if command == "capsule_policy":
                return self.configure(request["changes"], context=context)
            if command in {"capsule_apply", "capsule_apply_approximate", "capsule_compile_activation"}:
                raise ValueError("No qualified runtime activation/approximate operation is exposed by this desktop bridge. A note snapshot is not an activation capsule.")
            if command == "capsule_list":
                return self.snapshot(context=context)
            if command == "capsule_get":
                return self.get(request["digest"], actor)
            if command == "capsule_delete":
                value = self.get(request["digest"], actor)
                scope = value["provenance"]["scope"]
                authority = self.allowed(actor, context, request, scope)
                self.consume(actor, context)
                store = self.store(scope)
                for row in self.db.execute("SELECT digest FROM capsules WHERE scope=?", (scope,)):
                    if request["digest"] in store.get(row["digest"])["parents"]:
                        raise ValueError("Capsule is referenced by a child snapshot; delete the child first")
                store.delete(request["digest"])
                self.db.execute("DELETE FROM capsules WHERE digest=?", (request["digest"],))
                return {"deleted": True, "digest": request["digest"], "authority": authority}
            if command not in {"capsule_capture", "capsule_review_note"}:
                raise ValueError("Unknown capsule operation")
            with self.controller.memory.lock:
                note = self.controller.memory._note(request["id"], actor, request.get("revision"))
            if type(request.get("revision")) is not int or request["revision"] != note["revision"]:
                raise ValueError("Capsule operations require an exact note revision")
            authority = self.allowed(actor, context, request, note["scope"], promotion=request.get("status") == "promoted")
            count = self.db.execute("SELECT count(*) FROM capsules WHERE scope=?", (note["scope"],)).fetchone()[0]
            if count >= self.policy["max_capsules_per_scope"]:
                raise ValueError("Capsule scope capacity reached before memory review")
            self.consume(actor, context)
            if command == "capsule_review_note":
                status = request.get("status")
                if status not in {"candidate", "reviewed", "promoted"}:
                    raise ValueError("Unknown note review status")
                review_context = None if actor["operator"] else {**context, "can_promote": True}
                result = self.controller.memory.dispatch({"command": "memory_review", "id": note["id"], "revision": note["revision"],
                    "status": status, "reason": request.get("reason", "") + " [authority: " + authority + "]"}, context=review_context)
                with self.controller.memory.lock:
                    updated = self.controller.memory._note(note["id"], actor, result["revision"])
                return self.put(updated, actor, authority, "memory-review-receipt", {"schema": "mlx2-desktop-memory-review-v1",
                    "status": updated["status"], "authority": authority, "operator_confirmed": authority == "operator-confirmed",
                    "declared_situation": request.get("situation"), "reason": request.get("reason", "")}, context=context)
            parents = request.get("parents", [])
            if not isinstance(parents, list) or len(parents) > 8:
                raise ValueError("Capsules accept at most eight exact parents")
            return self.put(note, actor, authority, "exact-note-snapshot", {"schema": "mlx2-desktop-note-snapshot-v1", "note": note,
                "declared_situation": request.get("situation"), "activation_applied": False}, parents=parents, context=context)

    def snapshot(self, *, context=None):
        actor = self.actor(context)
        with self.lock:
            rows = self.db.execute("SELECT * FROM capsules ORDER BY created DESC LIMIT 256").fetchall()
            admitted = [dict(r) for r in rows if actor["operator"] or r["scope"] in actor["allowed_scopes"]]
        return {"policy": dict(self.policy), "capsules": admitted[:100], "store": "Exact content-addressed CapsuleStore evaluation envelopes",
                "activation_applied": False, "approximate_operation": "unavailable",
                "runtime_application": "Existing engine semantic/activation routes require separate artifact-bound configuration and runtime receipts; this bridge does not apply them"}

    def close(self):
        with self.lock:
            self.db.close()


def tool_definitions():
    string, integer = {"type": "string"}, {"type": "integer"}
    schemas = {
        "capsule_list": ("Inspect scoped exact capsule metadata and saved autonomy policy.", {}, []),
        "capsule_get": ("Read one scoped exact capsule; large note text stays behind bounded memory retrieval.", {"digest": string}, ["digest"]),
        "capsule_capture": ("Capture an exact note revision into a scoped evaluation capsule. This does not apply activation state.", {"id": string, "revision": integer, "parents": {"type": "array", "items": string}, "situation": {"enum": sorted(SITUATIONS)}}, ["id", "revision", "situation"]),
        "capsule_review_note": ("Review a scoped exact note under saved operator-selected autonomy. Model-driven review is explicitly unverified; it is not operator confirmation.", {"id": string, "revision": integer, "status": {"enum": ["candidate", "reviewed", "promoted"]}, "reason": string, "situation": {"enum": sorted(SITUATIONS)}}, ["id", "revision", "status", "reason", "situation"]),
        "capsule_delete": ("Delete an admitted exact capsule only if no child snapshot references it.", {"digest": string, "situation": {"enum": sorted(SITUATIONS)}}, ["digest", "situation"]),
        "capsule_policy": ("Request an operator-approved change to autonomous capsule/review policy and bounded limits.", {"changes": {"type": "object"}}, ["changes"]),
    }
    return [{"type": "function", "function": {"name": name, "description": description,
             "parameters": {"type": "object", "properties": properties, "required": required, "additionalProperties": False}}}
            for name, (description, properties, required) in schemas.items()]
