"""Private lossless chat archives with scoped, bounded model retrieval.

Compaction checkpoints are model-authored candidates linked to immutable exact
input revisions. They never replace or erase original tool/media documents.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import uuid

FIELDS = {"goal", "constraints", "decisions", "evidence", "open_tasks", "next_steps", "uncertainties"}
MAX_DOCUMENT = 64 << 20


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf8")


def identifier(value):
    if not isinstance(value, str) or not value or len(value.encode()) > 256 or any(ord(c) < 32 for c in value):
        raise ValueError("Profile and session identities must be nonempty bounded strings")
    return value


def integer(value, minimum, maximum, name):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def text_leaves(document):
    stack = [("", document)]
    while stack:
        pointer, value = stack.pop()
        if isinstance(value, dict):
            stack.extend((pointer + "/" + str(key).replace("~", "~0").replace("/", "~1"), item) for key, item in reversed(list(value.items())))
        elif isinstance(value, list):
            stack.extend((pointer + "/" + str(index), item) for index, item in reversed(list(enumerate(value))))
        elif isinstance(value, str):
            # Media bytes remain lossless in the immutable blob. Searching their
            # base64 representation is neither meaningful nor bounded context.
            if value.startswith("data:") and ";base64," in value[:256]:
                value = value[:256].split(",", 1)[0] + ",[binary content retained in archive]"
            for offset in range(0, max(1, len(value)), 3968):
                yield pointer[:4096], offset, value[offset:offset + 4096]


class SessionArchive:
    def __init__(self, controller):
        self.controller = controller
        self.path = controller.home / "session-archive.sqlite3"
        self.lock = threading.RLock()
        self.uploads = {}
        self.upload_root = controller.home / "session-uploads"
        if self.upload_root.is_symlink():
            raise ValueError("Session upload directory must be private and local")
        self.upload_root.mkdir(mode=0o700, exist_ok=True)
        for path in self.upload_root.glob("*.upload"):
            if path.is_file() and not path.is_symlink() and time.time() - path.stat().st_mtime > 86400:
                path.unlink()
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.path.chmod(0o600)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS sessions(profile TEXT,session TEXT,parent TEXT,parent_revision INTEGER,
                created REAL NOT NULL,PRIMARY KEY(profile,session));
            CREATE TABLE IF NOT EXISTS blobs(digest TEXT PRIMARY KEY,payload BLOB NOT NULL,bytes INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS revisions(profile TEXT,session TEXT,revision INTEGER,digest TEXT NOT NULL,
                created REAL NOT NULL,PRIMARY KEY(profile,session,revision),FOREIGN KEY(digest) REFERENCES blobs(digest),
                FOREIGN KEY(profile,session) REFERENCES sessions(profile,session));
            CREATE TABLE IF NOT EXISTS checkpoints(id TEXT PRIMARY KEY,profile TEXT,session TEXT,input_revision INTEGER,
                input_digest TEXT,summary BLOB NOT NULL,created REAL NOT NULL,authority TEXT NOT NULL,
                FOREIGN KEY(profile,session,input_revision) REFERENCES revisions(profile,session,revision));
            CREATE VIRTUAL TABLE IF NOT EXISTS session_search USING fts5(profile UNINDEXED,session UNINDEXED,
                revision UNINDEXED,pointer UNINDEXED,character_offset UNINDEXED,body,tokenize='unicode61');
        """)
        self.db.commit()

    def latest(self, profile_id, session_id):
        return self.db.execute("SELECT r.*,b.bytes FROM revisions r JOIN blobs b USING(digest) WHERE profile=? AND session=? ORDER BY revision DESC LIMIT 1", (profile_id, session_id)).fetchone()

    def archive(self, profile_id, session_id, document, parent_id=None, *, parent_revision=None, expected_revision=None):
        profile_id, session_id = identifier(profile_id), identifier(session_id)
        if not isinstance(document, (dict, list)):
            raise TypeError("Archive document must be a complete JSON object or array")
        payload = canonical(document)
        if len(payload) > MAX_DOCUMENT:
            raise ValueError("Session document exceeds 64 MiB; preserve it using explicit revision-bound archive parts")
        digest = hashlib.sha256(payload).hexdigest()
        if expected_revision is not None:
            integer(expected_revision, 0, 2**31 - 1, "Expected revision")
        with self.lock, self.db:
            previous = self.latest(profile_id, session_id)
            current = previous["revision"] if previous else 0
            if expected_revision is not None and expected_revision != current:
                raise ValueError("Session revision changed; archive refused")
            session = self.db.execute("SELECT * FROM sessions WHERE profile=? AND session=?", (profile_id, session_id)).fetchone()
            if session:
                if parent_id is not None and parent_id != session["parent"] or parent_revision is not None and parent_revision != session["parent_revision"]:
                    raise ValueError("A branch's parent and fork revision are immutable")
            else:
                if parent_id is not None:
                    parent_id = identifier(parent_id)
                    if parent_id == session_id:
                        raise ValueError("A session cannot be its own parent")
                    parent = self.latest(profile_id, parent_id)
                    if parent is None:
                        raise ValueError("Parent session is not archived in this user profile")
                    parent_revision = parent["revision"] if parent_revision is None else integer(parent_revision, 1, parent["revision"], "Parent revision")
                elif parent_revision is not None:
                    raise ValueError("Parent revision needs a parent session")
                if parent_id is not None and not self.db.execute("SELECT 1 FROM session_search WHERE profile=? AND session=? AND CAST(revision AS INTEGER)=? LIMIT 1", (profile_id, parent_id, parent_revision)).fetchone():
                    frozen = self.db.execute("SELECT payload FROM revisions JOIN blobs USING(digest) WHERE profile=? AND session=? AND revision=?", (profile_id, parent_id, parent_revision)).fetchone()
                    for pointer, offset, value in text_leaves(json.loads(frozen[0])):
                        self.db.execute("INSERT INTO session_search VALUES(?,?,?,?,?,?)", (profile_id, parent_id, parent_revision, pointer, offset, value))
                self.db.execute("INSERT INTO sessions VALUES(?,?,?,?,?)", (profile_id, session_id, parent_id, parent_revision, time.time()))
            if previous and previous["digest"] == digest:
                return self.metadata(previous)
            revision = current + 1
            self.db.execute("INSERT OR IGNORE INTO blobs VALUES(?,?,?)", (digest, payload, len(payload)))
            self.db.execute("INSERT INTO revisions VALUES(?,?,?,?,?)", (profile_id, session_id, revision, digest, time.time()))
            self.db.execute("DELETE FROM session_search WHERE profile=? AND session=? AND CAST(revision AS INTEGER) NOT IN (SELECT parent_revision FROM sessions WHERE profile=? AND parent=? AND parent_revision IS NOT NULL)", (profile_id, session_id, profile_id, session_id))
            for pointer, offset, value in text_leaves(document):
                self.db.execute("INSERT INTO session_search VALUES(?,?,?,?,?,?)", (profile_id, session_id, revision, pointer, offset, value))
            return self.metadata(self.latest(profile_id, session_id))

    @staticmethod
    def metadata(row):
        return {"profile_id": row["profile"], "session_id": row["session"], "revision": row["revision"], "digest": row["digest"], "bytes": row["bytes"], "created": row["created"]}

    def scope(self, context):
        if not isinstance(context, dict):
            raise TypeError("Session tools require a trusted active profile and conversation binding")
        profile, session = identifier(context.get("profile_id")), identifier(context.get("session_id"))
        row = self.latest(profile, session)
        if row is None:
            raise ValueError("Archive this conversation's complete state before using session tools")
        scope, seen = [], set()
        revision = row["revision"]
        while session:
            if session in seen or len(seen) >= 64:
                raise ValueError("Invalid or overly deep conversation ancestry")
            seen.add(session)
            scope.append((session, revision))
            parent = self.db.execute("SELECT parent,parent_revision FROM sessions WHERE profile=? AND session=?", (profile, session)).fetchone()
            if parent is None:
                raise ValueError("Conversation ancestry is incomplete")
            session, revision = parent["parent"], parent["parent_revision"]
        return profile, scope

    def get(self, *, context, session_id=None, revision=None, offset=0, max_bytes=8192):
        budget = integer(max_bytes, 2048, 16384, "Response byte budget")
        integer(offset, 0, MAX_DOCUMENT, "Byte offset")
        with self.lock:
            profile, scope = self.scope(context)
            requested = session_id or scope[0][0]
            ceiling = dict(scope).get(requested)
            if ceiling is None:
                raise ValueError("Session is outside this conversation and its revision-bound ancestors")
            revision = ceiling if revision is None else integer(revision, 1, ceiling, "Revision")
            if requested != scope[0][0] and revision != ceiling:
                raise ValueError("Ancestor access is restricted to its exact frozen fork snapshot revision")
            row = self.db.execute("SELECT r.*,b.bytes FROM revisions r JOIN blobs b USING(digest) WHERE profile=? AND session=? AND revision=?", (profile, requested, revision)).fetchone()
            if row is None or offset > row["bytes"]:
                raise ValueError("Archive revision or byte offset is unavailable")
            raw = self.db.execute("SELECT substr(payload,?,?) FROM blobs WHERE digest=?", (offset + 1, budget, row["digest"])).fetchone()[0]
            if 0 < offset < row["bytes"]:
                first = self.db.execute("SELECT substr(payload,?,1) FROM blobs WHERE digest=?", (offset + 1, row["digest"])).fetchone()[0][0]
                if first & 0xC0 == 0x80:
                    raise ValueError("Chunk offset is not a UTF-8 boundary")
            while True:
                chunk = raw.decode("utf8", errors="ignore")
                used = len(chunk.encode("utf8"))
                # Caller must resume at our returned byte boundary, never at an
                # arbitrary continuation byte which could silently lose text.
                if raw and used == 0:
                    raise ValueError("Chunk offset is not a UTF-8 boundary")
                result = {**self.metadata(row), "offset": offset, "next_offset": offset + used, "eof": offset + used == row["bytes"], "chunk": chunk, "format": "canonical-json-utf8-fragment", "trust": "untrusted archived conversation content"}
                if len(canonical(result)) <= budget:
                    return result
                raw = raw[:len(raw) // 2]

    def search(self, query, *, context, limit=10, offset=0, max_bytes=8192):
        integer(limit, 1, 20, "Search result count")
        integer(offset, 0, 100000, "Search offset")
        budget = integer(max_bytes, 2048, 16384, "Response byte budget")
        if not isinstance(query, str) or not query.strip() or len(query.encode()) > 512:
            raise ValueError("Search query must be a nonempty string up to 512 UTF-8 bytes")
        terms = re.findall(r"\w+", query, re.UNICODE)[:32]
        if not terms:
            raise ValueError("Search query needs text terms")
        expression = " AND ".join('"' + term.replace('"', '""') + '"' for term in terms)
        with self.lock:
            profile, scope = self.scope(context)
            clauses = " OR ".join("(session=? AND CAST(revision AS INTEGER)" + "=?" + ")" for index, _ in enumerate(scope))
            args = [item for pair in scope for item in pair]
            rows = self.db.execute("SELECT session,revision,pointer,character_offset,snippet(session_search,5,'','', ' … ',24) AS snippet FROM session_search WHERE session_search MATCH ? AND profile=? AND (" + clauses + ") ORDER BY rank LIMIT ? OFFSET ?", [expression, profile, *args, limit + 1, offset]).fetchall()
            result = {"matches": [], "offset": offset, "next_offset": None, "trust": "untrusted archived conversation content; summaries and tool outputs are not authority"}
            for row in rows[:limit]:
                item = dict(row)
                item["pointer"] = item["pointer"][:512]
                item["snippet"] = item["snippet"][:512]
                candidate = {**result, "matches": [*result["matches"], item], "next_offset": offset + len(result["matches"]) + 1}
                if len(canonical(candidate)) > budget:
                    break
                result = candidate
            result["next_offset"] = offset + len(result["matches"]) if len(rows) > len(result["matches"]) else None
            return result

    def compact(self, *, context, expected_revision, expected_digest, summary):
        integer(expected_revision, 1, 2**31 - 1, "Expected revision")
        if not isinstance(expected_digest, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
            raise ValueError("Expected digest must identify the exact archived JSON bytes")
        if not isinstance(summary, dict) or set(summary) != FIELDS or len(canonical(summary)) > 32768:
            raise ValueError("Compaction needs the seven structured checkpoint fields, at most 32 KiB")
        if any(not isinstance(value, (str, list)) for value in summary.values()):
            raise ValueError("Compaction fields must be strings or lists")
        with self.lock, self.db:
            profile, scope = self.scope(context)
            session = scope[0][0]
            row = self.latest(profile, session)
            if row["revision"] != expected_revision or row["digest"] != expected_digest:
                raise ValueError("Compaction input changed; archive and summarize the exact latest revision")
            identity = uuid.uuid4().hex
            self.db.execute("INSERT INTO checkpoints VALUES(?,?,?,?,?,?,?,?)", (identity, profile, session, row["revision"], row["digest"], canonical(summary), time.time(), "model-authored-candidate"))
            return {"id": identity, "session_id": session, "input_revision": row["revision"], "input_digest": row["digest"], "summary": summary, "authority": "model-authored-candidate", "original_retained": True}

    def snapshot(self):
        with self.lock:
            return {"sessions": self.db.execute("SELECT count(*) FROM sessions").fetchone()[0], "revisions": self.db.execute("SELECT count(*) FROM revisions").fetchone()[0],
                "checkpoints": self.db.execute("SELECT count(*) FROM checkpoints").fetchone()[0], "lossless": True, "raw_history_in_snapshot": False}

    def private_dispatch(self, request, context):
        if not isinstance(context, dict) or context.get("archive_operator") is not True:
            raise ValueError("Only the native operator may archive full session documents")
        profile, session = identifier(context.get("profile_id")), identifier(context.get("session_id"))
        if request.get("session_id", session) != session:
            raise ValueError("Native archive session binding differs")
        command = request["command"]
        with self.lock:
            if command == "session_archive_status":
                row = self.latest(profile, session)
                checkpoint = self.db.execute("SELECT * FROM checkpoints WHERE profile=? AND session=? ORDER BY created DESC LIMIT 1", (profile, session)).fetchone()
                return {"archive": self.metadata(row) if row else None, "checkpoint": {**{key: checkpoint[key] for key in ("id", "input_revision", "input_digest", "created", "authority")}, "summary": json.loads(checkpoint["summary"])} if checkpoint else None}
            if command == "session_archive":
                if len(canonical(request["document"])) > 8 << 20:
                    raise ValueError("Use the private chunk upload for archives over 8 MiB")
                return self.archive(profile, session, request["document"], request.get("parent_id"), parent_revision=request.get("parent_revision"), expected_revision=request.get("expected_revision"))
            if command == "session_archive_begin":
                total = integer(request.get("total_bytes"), 1, MAX_DOCUMENT, "Archive upload bytes")
                digest = request.get("sha256")
                if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                    raise ValueError("Archive upload requires a SHA256 digest")
                for stale_id, stale in list(self.uploads.items()):
                    if time.time() - stale["created"] > 3600:
                        self.abort_upload(stale_id)
                if len(self.uploads) >= 4:
                    raise ValueError("Too many active session uploads; finish or abort one")
                identity = uuid.uuid4().hex
                reservation = "session-upload-" + identity
                disk = getattr(self.controller, "disk", None)
                if disk is not None:
                    disk.reserve(reservation, self.upload_root, total * 5, reserve=64 << 20)
                path = self.upload_root / (identity + ".upload")
                try:
                    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                    os.close(descriptor)
                    row = self.latest(profile, session)
                    self.uploads[identity] = {"profile": profile, "session": session, "path": path, "total": total, "digest": digest,
                        "written": 0, "index": 0, "hasher": hashlib.sha256(), "created": time.time(), "reservation": reservation,
                        "parent_id": request.get("parent_id"), "parent_revision": request.get("parent_revision"), "expected_revision": row["revision"] if row else 0}
                except BaseException:
                    if disk is not None:
                        disk.release(reservation)
                    raise
                return {"upload_id": identity, "next_index": 0, "max_chunk_bytes": 1 << 20}
            identity = request.get("upload_id")
            upload = self.uploads.get(identity)
            if upload is None or upload["profile"] != profile or upload["session"] != session:
                raise ValueError("No session upload admitted for this active profile and conversation")
            if time.time() - upload["created"] > 3600:
                self.abort_upload(identity)
                raise ValueError("Session upload expired; retry the archive")
            if command == "session_archive_abort":
                self.abort_upload(identity)
                return {"aborted": True}
            if command == "session_archive_chunk":
                if type(request.get("index")) is not int or request["index"] != upload["index"]:
                    raise ValueError("Archive chunks must arrive once in sequential index order")
                encoded = request.get("data_base64")
                if not isinstance(encoded, str) or len(encoded) > 1_398_104:
                    raise ValueError("Archive chunk exceeds 1 MiB")
                try:
                    chunk = base64.b64decode(encoded, validate=True)
                except ValueError:
                    raise ValueError("Archive chunk is not valid base64") from None
                if not chunk or len(chunk) > 1 << 20 or upload["written"] + len(chunk) > upload["total"]:
                    raise ValueError("Archive chunk does not fit the declared upload")
                if upload["path"].stat().st_size != upload["written"]:
                    raise ValueError("Private archive staging changed")
                disk = getattr(self.controller, "disk", None)
                if disk is not None:
                    disk.check(upload["reservation"], len(chunk))
                descriptor = os.open(upload["path"], os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
                with os.fdopen(descriptor, "ab") as stream:
                    stream.write(chunk)
                    stream.flush()
                    os.fsync(stream.fileno())
                upload["hasher"].update(chunk)
                upload["written"] += len(chunk)
                upload["index"] += 1
                if disk is not None:
                    disk.used(upload["reservation"], upload["written"])
                return {"upload_id": identity, "next_index": upload["index"], "received_bytes": upload["written"]}
            if command == "session_archive_commit":
                if upload["written"] != upload["total"] or upload["hasher"].hexdigest() != upload["digest"]:
                    raise ValueError("Archive upload length or digest does not match; original conversation remains unchanged")
                try:
                    descriptor = os.open(upload["path"], os.O_RDONLY | os.O_NOFOLLOW)
                    with os.fdopen(descriptor, "rb") as stream:
                        stat = os.fstat(stream.fileno())
                        if stat.st_size != upload["total"] or stat.st_nlink != 1:
                            raise ValueError("Private archive staging changed")
                        raw = stream.read(upload["total"] + 1)
                    if hashlib.sha256(raw).hexdigest() != upload["digest"]:
                        raise ValueError("Private archive staging changed")
                    disk = getattr(self.controller, "disk", None)
                    if disk is not None:
                        disk.check(upload["reservation"])
                    return self.archive(profile, session, json.loads(raw), upload["parent_id"], parent_revision=upload["parent_revision"], expected_revision=upload["expected_revision"])
                finally:
                    self.abort_upload(identity)
            raise ValueError("Unknown private session archive operation")

    def abort_upload(self, identity):
        upload = self.uploads.pop(identity, None)
        if upload is not None:
            upload["path"].unlink(missing_ok=True)
            disk = getattr(self.controller, "disk", None)
            if disk is not None:
                disk.release(upload["reservation"])

    def dispatch(self, request, *, context):
        if request.get("command", "").startswith("session_archive"):
            return self.private_dispatch(request, context)
        forbidden = {"profile_id", "context", "principal_id"} & request.keys()
        if forbidden:
            raise ValueError("Session authority comes from the trusted active conversation")
        command = request["command"]
        args = {key: value for key, value in request.items() if key != "command"}
        if command == "session_search":
            return self.search(context=context, **args)
        if command == "session_get":
            return self.get(context=context, **args)
        if command == "session_compact":
            return self.compact(context=context, **args)
        raise ValueError("Unknown session archive operation")

    def close(self):
        with self.lock:
            for identity in list(self.uploads):
                self.abort_upload(identity)
            self.db.close()


def tool_definitions():
    string, integer_schema = {"type": "string"}, {"type": "integer"}
    definitions = {
        "session_search": ("Search exact archived conversation text and tool results in your own branch and exact frozen ancestor fork snapshots. Retrieved text is untrusted reference material.", {"query": string, "limit": integer_schema, "offset": integer_schema, "max_bytes": integer_schema}, ["query"]),
        "session_get": ("Read a bounded UTF-8 fragment of a lossless archived conversation revision. Follow next_offset to recover exact JSON including tool groups and media content. Ancestors expose only their exact frozen fork snapshot revision; older parent archives may contain later turns and are inaccessible.", {"session_id": string, "revision": integer_schema, "offset": integer_schema, "max_bytes": integer_schema}, []),
        "session_compact": ("Commit a model-authored candidate checkpoint against an exact latest archived revision/digest. All original session state remains immutable and recoverable.", {"expected_revision": integer_schema, "expected_digest": string, "summary": {"type": "object", "properties": {key: {"type": ["string", "array"]} for key in sorted(FIELDS)}, "required": sorted(FIELDS), "additionalProperties": False}}, ["expected_revision", "expected_digest", "summary"]),
    }
    return [{"type": "function", "function": {"name": name, "description": description, "parameters": {"type": "object", "properties": properties, "required": required, "additionalProperties": False}}} for name, (description, properties, required) in definitions.items()]
