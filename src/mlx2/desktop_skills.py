"""Private, revisioned procedural memory; skill content never grants execution authority."""
from __future__ import annotations

import contextlib
import copy
import json
import re
import sqlite3
import threading
import time
from pathlib import PurePosixPath


class Skills:
    def __init__(self, controller):
        self.controller = controller
        self.lock = threading.RLock()
        self.bound_profile = None
        path = controller.home / "skills.sqlite3"
        if path.is_symlink():
            raise ValueError("Skills database cannot be a symbolic link")
        self.db = sqlite3.connect(path, check_same_thread=False)
        path.chmod(0o600)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS skills (
                profile TEXT, name TEXT, revision INTEGER, document TEXT,
                enabled INTEGER, deleted INTEGER, updated REAL,
                PRIMARY KEY(profile,name,revision));
            CREATE TABLE IF NOT EXISTS policy (
                profile TEXT PRIMARY KEY, enabled INTEGER, autonomous_writes INTEGER);
        """)

    def _profile(self):
        return self.bound_profile or self.controller.user_profiles.active

    @contextlib.contextmanager
    def _binding(self, profile_id=None):
        with self.lock:
            active = self.controller.user_profiles.active
            expected = profile_id if profile_id is not None else (self.bound_profile or active)
            if not isinstance(expected, str) or not expected or expected != active:
                raise ValueError("Active user profile changed; refresh this skill operation")
            previous, self.bound_profile = self.bound_profile, expected
            try:
                yield
            finally:
                self.bound_profile = previous


    @staticmethod
    def _name(name):
        if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", name):
            raise ValueError("Skill names use 1–64 lowercase letters, digits, underscores or hyphens")
        return name

    @classmethod
    def _document(cls, value):
        if not isinstance(value, dict) or set(value) - {"name", "description", "body", "files", "enabled"}:
            raise ValueError("Invalid skill document fields")
        name = cls._name(value.get("name"))
        description, body, files = value.get("description"), value.get("body"), value.get("files", {})
        if not isinstance(description, str) or not description.strip() or not 1 <= len(description.encode()) <= 512:
            raise ValueError("Supply a description of at most 512 bytes")
        if not isinstance(body, str) or not body.strip() or len(body.encode()) > 65536:
            raise ValueError("Skill instructions must contain 1–65536 bytes")
        if not isinstance(files, dict) or len(files) > 64:
            raise ValueError("A skill supports up to 64 text reference files")
        paths = set()
        for path, content in files.items():
            if (not isinstance(path, str) or len(path) > 256 or "\\" in path or "\x00" in path
                    or any(ord(character) < 32 or ord(character) == 127 for character in path)
                    or path.split("/", 1)[0].casefold() == "skill.md" or PurePosixPath(path).is_absolute()
                    or any(part in {"", ".", ".."} for part in path.split("/"))):
                raise ValueError("Reference files need safe relative paths other than SKILL.md")
            if any(len(part.encode()) > 255 for part in path.split("/")):
                raise ValueError("Reference path components exceed portable filesystem limits")
            folded = path.casefold()
            if folded in paths or any(folded.startswith(previous + "/") or previous.startswith(folded + "/") for previous in paths):
                raise ValueError("Reference file identities collide on a portable filesystem")
            paths.add(folded)
            if not isinstance(content, str) or len(content.encode()) > 131072:
                raise ValueError("Reference files are text, each at most 128 KiB")
        if len(json.dumps(files).encode()) > 512 * 1024:
            raise ValueError("Skill reference files exceed 512 KiB")
        if type(value.get("enabled", True)) is not bool:
            raise TypeError("Skill enabled must be boolean")
        return {"name": name, "description": description, "body": body, "files": copy.deepcopy(files)}

    def policy(self, *, profile_id=None):
        with self._binding(profile_id):
            row = self.db.execute("SELECT * FROM policy WHERE profile=?", (self._profile(),)).fetchone()
            return {"enabled": bool(row["enabled"]) if row else True,
                    "autonomous_writes": bool(row["autonomous_writes"]) if row else False}

    def _latest(self, name):
        return self.db.execute("SELECT * FROM skills WHERE profile=? AND name=? ORDER BY revision DESC LIMIT 1",
                               (self._profile(), self._name(name))).fetchone()

    @staticmethod
    def _metadata(row):
        description = row["description"] if "description" in row.keys() else json.loads(row["document"])["description"]  # noqa: SIM118 -- sqlite3.Row membership tests values, not column names
        return {"name": row["name"], "description": description, "revision": row["revision"],
                "enabled": bool(row["enabled"]), "deleted": bool(row["deleted"]), "updated": row["updated"]}

    def snapshot(self, *, includeall=True, offset=0, limit=100, profile_id=None):
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 200:
            raise ValueError("Choose a nonnegative offset and 1–200 skills")
        with self._binding(profile_id):
            source = """ FROM skills s JOIN
                (SELECT name,MAX(revision) revision FROM skills WHERE profile=? GROUP BY name) latest
                ON s.name=latest.name AND s.revision=latest.revision WHERE s.profile=?"""
            if not includeall:
                source += " AND s.enabled=1 AND s.deleted=0"
            arguments = (self._profile(), self._profile())
            count = self.db.execute("SELECT count(*)" + source, arguments).fetchone()[0]
            rows = self.db.execute("SELECT s.name,s.revision,s.enabled,s.deleted,s.updated,json_extract(s.document,'$.description') AS description" + source + " ORDER BY s.name LIMIT ? OFFSET ?", (*arguments, limit, offset)).fetchall()
            return {"policy": self.policy(), "skills": [self._metadata(row) for row in rows], "total": count,
                    "offset": offset, "next_offset": offset + limit if offset + limit < count else None}

    def dispatch(self, request, *, operator=False, profile_id=None):
        with self._binding(profile_id):
            if request.get("profile_id", self._profile()) != self._profile():
                raise ValueError("Active user profile changed; refresh this skill operation")
            return self._dispatch(request, operator=operator)

    def _dispatch(self, request, *, operator=False):
        command = request.get("command")
        with self.lock:
            if command == "skill_policy":
                if not operator:
                    raise ValueError("Skill policy requires operator approval")
                changes = request.get("changes")
                if not isinstance(changes, dict) or set(changes) - {"enabled", "autonomous_writes"} or any(type(v) is not bool for v in changes.values()):
                    raise ValueError("Skill policy fields must be boolean")
                policy = {**self.policy(), **changes}
                with self.db:
                    self.db.execute("INSERT OR REPLACE INTO policy VALUES (?,?,?)", (self._profile(), policy["enabled"], policy["autonomous_writes"]))
                return policy
            if not operator and not self.policy()["enabled"]:
                raise ValueError("Skills are disabled")
            if command == "skills_list":
                return self.snapshot(includeall=operator and request.get("includeall", False), offset=request.get("offset", 0), limit=request.get("limit", 100))
            if command == "skill_import":
                if not operator:
                    raise ValueError("Import requires the operator")
                document = self._document(request.get("document"))
                return self.dispatch({"command": "skill_manage", "action": "create", **document,
                                      "expected_revision": 0, "enabled": False}, operator=True)
            name = self._name(request.get("name"))
            row = self._latest(name)
            if command == "skill_manage":
                if not operator and not self.policy()["autonomous_writes"]:
                    raise ValueError("Skill writes require approval")
                expected = request.get("expected_revision")
                if type(expected) is not int or expected != (row["revision"] if row else 0):
                    raise ValueError("Skill changed; reread its current revision")
                action = request.get("action")
                if action == "create":
                    if row is not None:
                        raise ValueError("Skill already exists; update or restore it")
                    document = self._document({key: request[key] for key in ("name", "description", "body", "files", "enabled") if key in request})
                    enabled, deleted = request.get("enabled", True), False
                else:
                    if row is None:
                        raise ValueError("Unknown skill")
                    document, enabled, deleted = json.loads(row["document"]), bool(row["enabled"]), bool(row["deleted"])
                    if deleted and action not in {"archive", "restore"}:
                        raise ValueError("Restore the archived skill before editing or enabling it")
                    if action == "update":
                        if "enabled" in request:
                            if type(request["enabled"]) is not bool:
                                raise TypeError("Skill enabled must be boolean")
                            enabled = request["enabled"]
                        document = self._document({**document, **{key: request[key] for key in ("description", "body", "files") if key in request}})
                    elif action == "patch":
                        old, new = request.get("old_text"), request.get("new_text")
                        if not isinstance(old, str) or not old or not isinstance(new, str) or document["body"].count(old) != 1:
                            raise ValueError("A targeted patch must match exactly once")
                        document = self._document({**document, "body": document["body"].replace(old, new, 1)})
                    elif action == "set_enabled":
                        enabled = request.get("enabled")
                        if type(enabled) is not bool:
                            raise TypeError("Skill enabled must be boolean")
                    elif action in {"archive", "restore"}:
                        deleted = action == "archive"
                        enabled = False  # explicit enable after restore
                    else:
                        raise ValueError("Unknown skill action")
                with self.db:
                    self.db.execute("INSERT INTO skills VALUES (?,?,?,?,?,?,?)", (self._profile(), name, expected + 1,
                        json.dumps(document, ensure_ascii=False), enabled, deleted, time.time()))
                return self._metadata(self._latest(name))
            if row is None or (not operator and (row["deleted"] or not row["enabled"])):
                raise ValueError("Skill unavailable")
            if command == "skill_export":
                if not operator:
                    raise ValueError("Export requires the operator")
                return {"document": {**json.loads(row["document"]), "enabled": bool(row["enabled"])}}
            if command == "skill_view":
                revision = request.get("revision", row["revision"])
                if type(revision) is not int:
                    raise TypeError("Skill revision must be an integer")
                row = self.db.execute("SELECT * FROM skills WHERE profile=? AND name=? AND revision=?", (self._profile(), name, revision)).fetchone()
                if row is None:
                    raise ValueError("Unknown skill revision")
                document = json.loads(row["document"])
                path = request.get("path", "SKILL.md")
                text = (json.dumps(sorted(document["files"]), ensure_ascii=False, separators=(",", ":")) if path == "."
                        else document["body"] if path == "SKILL.md" else document["files"].get(path))
                if text is None:
                    raise ValueError("Unknown skill reference")
                offset, maximum = request.get("offset", 0), request.get("max_bytes", 8192)
                if type(offset) is not int or offset < 0 or type(maximum) is not int or not 256 <= maximum <= 8192:
                    raise ValueError("Invalid skill page budget")
                # Character offsets avoid splitting UTF-8; byte budgets remain exact.
                content = text[offset:offset + maximum]
                while len(content.encode()) > maximum:
                    content = content[:-1]
                return {**self._metadata(row), "content": content, "path": path, "reference_count": len(document["files"]), "reference_catalog_path": ".",
                        "offset": offset, "next_offset": offset + len(content) if offset + len(content) < len(text) else None,
                        "reference_only": True, "authority": "procedural guidance; does not grant tool permissions"}
            raise ValueError("Unknown skill operation")

    def close(self):
        with self.lock:
            self.db.close()
