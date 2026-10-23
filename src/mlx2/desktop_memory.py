"""Local, scoped, revisioned agent notes and explicit worker messages.

Original SQLite/FTS baseline inspired by A-MEM's linked notes and Hermes'
curation lifecycle. It does not implement model-generated semantic embeddings
or claim the paper's results. Model text remains reference material, never a
source of authorization. No background model call or external fetch occurs.
"""
from __future__ import annotations

import json
import math
import re
import socket
import sqlite3
import threading
import time
import uuid

KINDS = {"note", "wiki", "forum", "tip", "procedure", "lesson"}
STATES = {"candidate", "reviewed", "promoted"}
SCOPE_KINDS = {"private", "session", "project", "machine", "community"}


def _text(value, name, maximum, *, empty=False):
    if not isinstance(value, str) or (not empty and not value.strip()) or len(value.encode("utf8")) > maximum:
        raise ValueError(f"{name} must be {'a bounded' if empty else 'a nonempty bounded'} string ({maximum} UTF-8 bytes)")
    return value


def _scope(value):
    _text(value, "Scope", 512)
    kind, separator, identity = value.partition(":")
    if not separator or kind not in SCOPE_KINDS or not identity.strip():
        raise ValueError("Scope must be private/session/project/machine/community:<identity>")
    return value


def _list(value, name, count=32, maximum=256):
    if not isinstance(value, list) or len(value) > count:
        raise ValueError(f"{name} must be a list of at most {count} strings")
    return list(dict.fromkeys(_text(v, name, maximum) for v in value))


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _clip(value, maximum):
    return value.encode("utf8")[:max(0, maximum)].decode("utf8", errors="ignore")


class MemoryController:
    def __init__(self, controller):
        self.controller = controller
        self.lock = threading.RLock()
        self.path = controller.home / "agent-memory.sqlite3"
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.path.chmod(0o600)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY, kind TEXT NOT NULL, scope TEXT NOT NULL,
                created REAL NOT NULL, author TEXT NOT NULL, current_revision INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS revisions (
                note_id TEXT NOT NULL REFERENCES notes(id), revision INTEGER NOT NULL,
                created REAL NOT NULL, author TEXT NOT NULL, title TEXT NOT NULL,
                body TEXT NOT NULL, tags TEXT NOT NULL, sources TEXT NOT NULL,
                status TEXT NOT NULL, reason TEXT NOT NULL,
                PRIMARY KEY(note_id,revision));
            CREATE TABLE IF NOT EXISTS links (
                id TEXT PRIMARY KEY, created REAL NOT NULL, author TEXT NOT NULL,
                source_id TEXT NOT NULL, source_revision INTEGER NOT NULL,
                target_id TEXT NOT NULL, target_revision INTEGER NOT NULL,
                relation TEXT NOT NULL, status TEXT NOT NULL, rationale TEXT NOT NULL,
                FOREIGN KEY(source_id,source_revision) REFERENCES revisions(note_id,revision),
                FOREIGN KEY(target_id,target_revision) REFERENCES revisions(note_id,revision));
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY, note_id TEXT NOT NULL, revision INTEGER NOT NULL,
                created REAL NOT NULL, author TEXT NOT NULL, useful INTEGER NOT NULL,
                reason TEXT NOT NULL, FOREIGN KEY(note_id,revision) REFERENCES revisions(note_id,revision));
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY, session_id TEXT NOT NULL, sender TEXT NOT NULL,
                recipient TEXT NOT NULL, created REAL NOT NULL, body TEXT NOT NULL,
                reply_to TEXT, capsule TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS acknowledgements (
                message_id TEXT NOT NULL REFERENCES messages(id), recipient TEXT NOT NULL,
                created REAL NOT NULL, PRIMARY KEY(message_id,recipient));
            CREATE INDEX IF NOT EXISTS message_recipient ON messages(session_id,recipient,created);
            CREATE INDEX IF NOT EXISTS note_scope ON notes(scope,kind);
            CREATE VIRTUAL TABLE IF NOT EXISTS note_search USING fts5(note_id UNINDEXED,title,body,tags,tokenize='unicode61');
        """)
        self.db.commit()

    def _actor(self, context):
        if context is None:
            return {"principal_id": "operator", "operator": True, "allowed_scopes": [],
                    "session_ids": [], "worker_ids": [], "can_promote": True}
        if not isinstance(context, dict):
            raise TypeError("Memory requires a trusted actor context")
        principal = _text(context.get("principal_id"), "Principal", 256)
        scopes = [_scope(s) for s in _list(context.get("allowed_scopes", []), "Scope grants", 128, 512)]
        return {"principal_id": principal, "operator": False, "allowed_scopes": scopes,
                "session_ids": _list(context.get("session_ids", []), "Session grants", 128),
                "worker_ids": _list(context.get("worker_ids", []), "Worker grants", 128),
                "can_promote": context.get("can_promote") is True,
                "context_tokens": context.get("context_tokens")}

    def _authorize(self, scope, actor):
        _scope(scope)
        if not actor["operator"] and scope not in actor["allowed_scopes"]:
            raise PermissionError("Memory scope is not granted to this session")

    def _note(self, identity, actor, revision=None):
        row = self.db.execute("SELECT * FROM notes WHERE id=?", (identity,)).fetchone()
        if not row:
            raise ValueError("Memory record does not exist")
        self._authorize(row["scope"], actor)
        version = self.db.execute("SELECT * FROM revisions WHERE note_id=? AND revision=?",
            (identity, row["current_revision"] if revision is None else revision)).fetchone()
        if not version:
            raise ValueError("Memory revision does not exist")
        value = {**dict(row), **dict(version), "id": row["id"], "created": row["created"], "revised_at": version["created"],
                 "created_by": row["author"], "author": version["author"], "tags": json.loads(version["tags"]), "sources": json.loads(version["sources"])}
        value.pop("note_id", None)
        return value

    def _enabled(self):
        return self.controller.preferences.get("memory_enabled", True) is True

    def add(self, request, actor):
        scope = request.get("scope") or ("project:mlx2" if actor["operator"] else "private:" + actor["principal_id"])
        self._authorize(scope, actor)
        kind = request.get("kind", "note")
        if kind not in KINDS:
            raise ValueError("Choose note, wiki, forum, tip, procedure or lesson")
        title = _text(request.get("title"), "Title", 256)
        body = _text(request.get("body"), "Body", 65536)
        tags = _list(request.get("tags", []), "Tags")
        sources = _list(request.get("sources", []), "Source pointers", 32, 2048)
        identity, now = uuid.uuid4().hex, time.time()
        # Captures begin as candidates even when the author is the operator.
        # Review is a separate explicit, revisioned operation.
        self.db.execute("INSERT INTO notes VALUES (?,?,?,?,?,?)", (identity, kind, scope, now, actor["principal_id"], 1))
        self.db.execute("INSERT INTO revisions VALUES (?,?,?,?,?,?,?,?,?,?)", (identity, 1, now, actor["principal_id"], title, body, _json(tags), _json(sources), "candidate", "Captured"))
        self.db.execute("INSERT INTO note_search VALUES (?,?,?,?)", (identity, title, body, " ".join(tags)))
        return self._note(identity, actor)

    def refine(self, request, actor, *, review=False):
        old = self._note(request["id"], actor)
        expected = request.get("revision")
        if type(expected) is not int or expected != old["revision"]:
            raise ValueError("Memory changed; read its current revision before updating")
        if review:
            if not actor["can_promote"]:
                raise PermissionError("Memory review/promotion requires explicit operator approval")
            status = request.get("status")
            if status not in STATES:
                raise ValueError("Unknown memory review state")
        else:
            status = "candidate"  # a rewritten claim must be reviewed again
        title = _text(request.get("title", old["title"]), "Title", 256)
        body = _text(request.get("body", old["body"]), "Body", 65536)
        tags = _list(request.get("tags", old["tags"]), "Tags")
        sources = _list(request.get("sources", old["sources"]), "Source pointers", 32, 2048)
        reason = _text(request.get("reason"), "Revision reason", 2048)
        revision = old["revision"] + 1
        self.db.execute("INSERT INTO revisions VALUES (?,?,?,?,?,?,?,?,?,?)", (old["id"], revision, time.time(), actor["principal_id"], title, body, _json(tags), _json(sources), status, reason))
        self.db.execute("UPDATE notes SET current_revision=? WHERE id=?", (revision, old["id"]))
        self.db.execute("DELETE FROM note_search WHERE note_id=?", (old["id"],))
        self.db.execute("INSERT INTO note_search VALUES (?,?,?,?)", (old["id"], title, body, " ".join(tags)))
        return self._note(old["id"], actor)

    def get(self, request, actor):
        value = self._note(request["id"], actor, request.get("revision"))
        # Explicit get is bounded too; history contains metadata only.
        maximum = 65536 if actor["operator"] else 4096
        value["body"] = _clip(value["body"], maximum)
        history = self.db.execute("SELECT revision,created,author,status,reason FROM revisions WHERE note_id=? ORDER BY revision DESC LIMIT 100", (value["id"],)).fetchall()
        value["history"] = [dict(r) for r in history]
        links = self.db.execute("SELECT * FROM links WHERE (source_id=? AND source_revision=?) OR (target_id=? AND target_revision=?) ORDER BY created DESC LIMIT 100", (value["id"], value["revision"], value["id"], value["revision"])).fetchall()
        value["links"] = []
        for link in links:
            try:
                self._note(link["source_id"], actor, link["source_revision"])
                self._note(link["target_id"], actor, link["target_revision"])
            except PermissionError:
                continue
            value["links"].append(dict(link))
        value["feedback"] = [dict(r) for r in self.db.execute("SELECT created,author,useful,reason FROM feedback WHERE note_id=? AND revision=? ORDER BY created DESC LIMIT 8", (value["id"], value["revision"]))]
        if not actor["operator"]:
            value["history"] = [{**r, "reason": _clip(r["reason"], 128)} for r in value["history"][:8]]
            value["links"] = [{**r, "rationale": _clip(r["rationale"], 128)} for r in value["links"][:8]]
            value["feedback"] = [{**r, "reason": _clip(r["reason"], 128)} for r in value["feedback"]]
            value["sources"] = [_clip(v, 256) for v in value["sources"][:3]]
            value["tags"] = [_clip(v, 64) for v in value["tags"][:8]]
            value["reason"] = _clip(value["reason"], 256)
            if len(_json(value).encode()) > 8192:
                value["history"], value["links"], value["feedback"] = [], [], []
                value["body"] = _clip(value["body"], 2048)
        value["reference_only"] = True
        return value

    def _budget(self, request, actor):
        configured = self.controller.preferences.get("memory_retrieval_tokens", 1024)
        if type(configured) is not int or not 64 <= configured <= 16384:
            configured = 1024
        budget = request.get("token_budget", configured)
        if type(budget) is not int or not 64 <= budget <= configured:
            raise ValueError("Retrieval budget exceeds the configured bound")
        context = actor.get("context_tokens")
        percent = self.controller.preferences.get("memory_context_percent", 10)
        if type(percent) not in (int, float) or not math.isfinite(percent) or not 0 < percent <= 10:
            percent = 10
        if type(context) is int and context > 0:
            budget = min(budget, int(context * percent / 100))
        return max(0, budget)

    def search(self, request, actor):
        query = _text(request.get("query", ""), "Query", 2048, empty=actor["operator"])
        limit = request.get("limit", 12)
        if type(limit) is not int or not 1 <= limit <= 50:
            raise ValueError("Memory retrieval limit must be 1..50")
        scope = request.get("scope")
        if scope:
            self._authorize(scope, actor)
        scopes = [scope] if scope else actor["allowed_scopes"]
        states = request.get("states", ["reviewed", "promoted"])
        if not isinstance(states, list) or not states or any(s not in STATES for s in states):
            raise ValueError("Unknown memory retrieval states")
        clauses, parameters = ["r.revision=n.current_revision"], []
        if not actor["operator"] or scope:
            if not scopes:
                return {"records": [], "retrieval": "fts5", "reference_only": True}
            clauses.append("n.scope IN (" + ",".join("?" for _ in scopes) + ")")
            parameters.extend(scopes)
        clauses.append("r.status IN (" + ",".join("?" for _ in states) + ")")
        parameters.extend(states)
        if request.get("kind"):
            if request["kind"] not in KINDS:
                raise ValueError("Unknown note kind")
            clauses.append("n.kind=?")
            parameters.append(request["kind"])
        terms = re.findall(r"[^\W_]+", query, flags=re.UNICODE)[:32]
        if terms:
            match = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
            sql = "SELECT n.id FROM note_search JOIN notes n ON n.id=note_search.note_id JOIN revisions r ON r.note_id=n.id WHERE note_search MATCH ? AND " + " AND ".join(clauses) + " ORDER BY bm25(note_search),n.created DESC LIMIT ?"
            args = [match, *parameters, limit]
        elif actor["operator"]:
            sql = "SELECT n.id FROM notes n JOIN revisions r ON r.note_id=n.id WHERE " + " AND ".join(clauses) + " ORDER BY n.created DESC LIMIT ?"
            args = [*parameters, limit]
        else:
            return {"records": [], "retrieval": "fts5", "reference_only": True}
        rows = self.db.execute(sql, args).fetchall()
        budget = self._budget(request, actor)
        response = {"records": [], "retrieval": "fts5", "reference_only": True,
                    "token_bound": budget, "budget_unit": "UTF-8 bytes (conservative token upper bound)", "encoded_bytes": 0}
        for row in rows:
            note = self._note(row["id"], actor)
            snippet = {k: note[k] for k in ("id", "revision", "kind", "scope", "title", "status", "revised_at")}
            snippet["sources"] = note["sources"][:3]
            snippet["tags"] = note["tags"][:8]
            remaining = budget - len(_json(response).encode()) - len(_json(snippet).encode()) - 24
            if remaining < 32:
                break
            snippet["snippet"] = _clip(note["body"], min(remaining, 768))
            response["records"].append(snippet)
            if len(_json(response).encode()) > budget:
                response["records"].pop()
                break
        for _ in range(3):
            response["encoded_bytes"] = len(_json(response).encode())
        if len(_json(response).encode()) > budget:
            return {"records": [], "token_bound": budget}
        return response

    def link(self, request, actor):
        source = self._note(request["source_id"], actor, request.get("source_revision"))
        target = self._note(request["target_id"], actor, request.get("target_revision"))
        relation = request.get("relation")
        if not isinstance(relation, str) or not re.fullmatch(r"[a-z][a-z0-9_-]{0,47}", relation):
            raise ValueError("Use an explicit typed relation such as supports, contradicts, supersedes or related")
        if source["id"] == target["id"] and source["revision"] == target["revision"]:
            raise ValueError("A memory cannot link to itself at the same revision")
        status = "accepted" if actor["can_promote"] and request.get("accepted") is True else "suggested"
        rationale = _text(request.get("rationale"), "Link rationale", 2048)
        identity = uuid.uuid4().hex
        self.db.execute("INSERT INTO links VALUES (?,?,?,?,?,?,?,?,?,?)", (identity, time.time(), actor["principal_id"], source["id"], source["revision"], target["id"], target["revision"], relation, status, rationale))
        return {"id": identity, "source_id": source["id"], "source_revision": source["revision"], "target_id": target["id"], "target_revision": target["revision"], "relation": relation, "status": status}

    def useful(self, request, actor):
        note = self._note(request["id"], actor, request.get("revision"))
        if type(request.get("useful")) is not bool:
            raise ValueError("Useful must be an explicit boolean")
        reason = _text(request.get("reason"), "Feedback reason", 2048)
        identity = uuid.uuid4().hex
        self.db.execute("INSERT INTO feedback VALUES (?,?,?,?,?,?,?)", (identity, note["id"], note["revision"], time.time(), actor["principal_id"], int(request["useful"]), reason))
        return {"id": identity, "recorded": True, "note_id": note["id"], "revision": note["revision"]}

    def message(self, request, actor):
        session = _text(request.get("session_id"), "Message session", 256)
        if not actor["operator"] and session not in actor["session_ids"]:
            raise PermissionError("Message session is not granted")
        command = request["command"]
        if command == "message_post":
            recipient = _text(request.get("recipient"), "Recipient", 256)
            if not actor["operator"] and recipient not in actor["worker_ids"]:
                raise PermissionError("Recipient is not an explicitly admitted sibling worker")
            body = _text(request.get("body"), "Message", 8192)
            reply = request.get("reply_to")
            if reply is not None:
                original = self.db.execute("SELECT * FROM messages WHERE id=? AND session_id=?", (reply, session)).fetchone()
                if not original or (not actor["operator"] and actor["principal_id"] not in {original["sender"], original["recipient"]}):
                    raise PermissionError("Reply target is not visible in this message session")
            capsule = request.get("capsule", {})
            if not isinstance(capsule, dict) or set(capsule) - {"id", "revision", "engaged"} or len(_json(capsule)) > 1024:
                raise ValueError("Capsule provenance accepts only id, revision and explicit engaged state")
            if capsule and (not isinstance(capsule.get("id"), str) or type(capsule.get("revision")) is not int or type(capsule.get("engaged")) is not bool):
                raise ValueError("Capsule provenance requires id, integer revision and boolean actual engagement")
            identity = uuid.uuid4().hex
            self.db.execute("INSERT INTO messages VALUES (?,?,?,?,?,?,?,?)", (identity, session, actor["principal_id"], recipient, time.time(), body, reply, _json(capsule)))
            return {"id": identity, "session_id": session, "sender": actor["principal_id"], "recipient": recipient, "posted": True}
        if command == "message_ack":
            row = self.db.execute("SELECT * FROM messages WHERE id=? AND session_id=?", (request["id"], session)).fetchone()
            if not row or (not actor["operator"] and row["recipient"] != actor["principal_id"]):
                raise PermissionError("Only the admitted recipient can acknowledge a message")
            self.db.execute("INSERT OR IGNORE INTO acknowledgements VALUES (?,?,?)", (row["id"], row["recipient"], time.time()))
            return {"id": row["id"], "acknowledged": True}
        if command == "message_read":
            limit = request.get("limit", 20)
            if type(limit) is not int or not 1 <= limit <= 50:
                raise ValueError("Message limit must be 1..50")
            args = [session]
            where = "m.session_id=?"
            if not actor["operator"]:
                where += " AND (m.recipient=? OR m.sender=?)"
                args.extend([actor["principal_id"], actor["principal_id"]])
            if request.get("unread_only", False):
                where += " AND a.message_id IS NULL"
            rows = self.db.execute("SELECT m.*,a.created AS acknowledged_at FROM messages m LEFT JOIN acknowledgements a ON a.message_id=m.id AND a.recipient=m.recipient WHERE " + where + " ORDER BY m.created DESC LIMIT ?", [*args, limit]).fetchall()
            remaining, result = 8192, []
            for row in rows:
                item = {**dict(row), "capsule": json.loads(row["capsule"]), "reference_only": True}
                item["body"] = _clip(item["body"], min(2048, remaining))
                size = len(_json(item).encode())
                if size > remaining:
                    break
                result.append(item)
                remaining -= size
            return {"messages": list(reversed(result)), "delivery": "local scoped mailbox"}
        raise ValueError("Unknown message operation")

    def dispatch(self, request, *, context=None):
        if not self._enabled():
            raise ValueError("Agent memory is disabled")
        actor = self._actor(context)
        if any(key in request for key in ("context", "principal_id", "allowed_scopes", "session_ids", "worker_ids", "can_promote")):
            raise ValueError("Actor privileges cannot be supplied as tool arguments")
        command = request.get("command")
        handlers = {"memory_add": self.add, "memory_refine": self.refine, "memory_get": self.get,
                    "memory_search": self.search, "memory_context": self.search, "memory_link": self.link, "memory_useful": self.useful}
        with self.lock, self.db:
            if command == "memory_review":
                result = self.refine(request, actor, review=True)
                return result if actor["operator"] else {key: result[key] for key in ("id", "revision", "status", "scope") }
            if isinstance(command, str) and command.startswith("message_"):
                return self.message(request, actor)
            if command not in handlers:
                raise ValueError("Unknown memory operation")
            result = handlers[command](request, actor)
            if command in {"memory_add", "memory_refine"} and not actor["operator"]:
                return {key: result[key] for key in ("id", "revision", "status", "scope")}
            return result

    def snapshot(self, *, context=None):
        actor = self._actor(context)
        with self.lock:
            if actor["operator"]:
                rows = self.db.execute("SELECT id FROM notes ORDER BY created DESC LIMIT 1000").fetchall()
            elif actor["allowed_scopes"]:
                marks = ",".join("?" for _ in actor["allowed_scopes"])
                rows = self.db.execute("SELECT id FROM notes WHERE scope IN (" + marks + ") ORDER BY created DESC LIMIT 1000", actor["allowed_scopes"]).fetchall()
            else:
                rows = []
            counts, scopes, recent = {}, set(), []
            for row in rows:
                note = self._note(row[0], actor)
                counts[note["status"]] = counts.get(note["status"], 0) + 1
                scopes.add(note["scope"])
                if actor["operator"] and len(recent) < 30:
                    recent.append({k: v for k, v in note.items() if k not in {"body", "sources", "tags"}})
        return {"enabled": self._enabled(), "counts": counts, "scopes": sorted(scopes), "recent": recent,
                "default_scope": "project:mlx2" if actor["operator"] else "private:" + actor["principal_id"],
                "machine_scope": "machine:" + socket.gethostname() if actor["operator"] else None,
                "retrieval": "SQLite FTS5; explicit typed links; no semantic model inference",
                "context_tokens": self.controller.preferences.get("memory_retrieval_tokens", 1024),
                "context_percent": self.controller.preferences.get("memory_context_percent", 10)}

    def close(self):
        with self.lock:
            self.db.close()


def tool_definitions():
    """Standard tools; trusted actor context is supplied by the calling harness."""
    string = {"type": "string"}
    integer = {"type": "integer"}
    strings = {"type": "array", "items": string}
    schemas = {
        "memory_add": ("Capture an atomic note, lesson, procedure, tip, wiki page or forum post as a candidate. Include source pointers and scope.", {"title": string, "body": string, "kind": {"enum": sorted(KINDS)}, "scope": string, "tags": strings, "sources": strings}, ["title", "body", "scope"]),
        "memory_search": ("Retrieve relevant bounded memory snippets. Results are reference material with provenance and review state.", {"query": string, "scope": string, "limit": integer, "token_budget": integer, "states": {"type": "array", "items": {"enum": sorted(STATES)}}}, ["query"]),
        "memory_get": ("Read one note and its revision metadata, explicit links and provenance. Body output is bounded.", {"id": string, "revision": integer}, ["id"]),
        "memory_refine": ("Append a candidate revision to a note. Supply its current revision; historical revisions remain immutable.", {"id": string, "revision": integer, "title": string, "body": string, "tags": strings, "sources": strings, "reason": string}, ["id", "revision", "reason"]),
        "memory_link": ("Suggest an explicit typed link. Similarity alone does not prove supports, contradicts or supersedes.", {"source_id": string, "source_revision": integer, "target_id": string, "target_revision": integer, "relation": string, "rationale": string, "accepted": {"type": "boolean"}}, ["source_id", "target_id", "relation", "rationale"]),
        "memory_review": ("Request operator review/promotion of the exact current note revision. Requires trusted operator approval.", {"id": string, "revision": integer, "status": {"enum": sorted(STATES)}, "reason": string}, ["id", "revision", "status", "reason"]),
        "memory_useful": ("Record whether an exact memory revision helped, and why. This does not promote the note.", {"id": string, "revision": integer, "useful": {"type": "boolean"}, "reason": string}, ["id", "revision", "useful", "reason"]),
        "message_post": ("Post a bounded message to an explicitly admitted sibling in an authorized session. Sender identity is fixed by the harness.", {"session_id": string, "recipient": string, "body": string, "reply_to": string, "capsule": {"type": "object", "properties": {"id": string, "revision": integer, "engaged": {"type": "boolean"}}, "additionalProperties": False}}, ["session_id", "recipient", "body"]),
        "message_read": ("Read your own session messages. This grants no access to unrelated chats or workers.", {"session_id": string, "limit": integer, "unread_only": {"type": "boolean"}}, ["session_id"]),
        "message_ack": ("Acknowledge a session message addressed to your admitted worker identity.", {"session_id": string, "id": string}, ["session_id", "id"]),
    }
    return [{"type": "function", "function": {"name": name, "description": description,
             "parameters": {"type": "object", "properties": properties, "required": required, "additionalProperties": False}}}
            for name, (description, properties, required) in schemas.items()]
