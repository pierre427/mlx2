"""Bounded tenant-scoped resources for OpenAI-compatible HTTP contracts.

Stores remain process-local unless a root directory is configured.  Durable
mode uses opaque tenant directories and atomic same-directory replacement so a
crash exposes either the old record or the new one, never a partial JSON file.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import threading
import time
import uuid
from collections import Counter, OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path


class ResourceNotFound(KeyError):
    """A tenant-scoped API resource does not exist."""


class CapabilityUnavailable(RuntimeError):
    """The loaded route does not implement the requested capability."""


def _tenant_name(tenant_id):
    value = str(tenant_id or "default")
    return hashlib.sha256(value.encode()).hexdigest()


def _atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _json_bytes(value):
    return json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode()


def _restore_candidates(root, counts):
    """Sort only file metadata; payloads are admitted and evicted one at a time."""
    candidates = []
    for path in root.glob("*/*.json"):
        try:
            info = path.lstat()
            if path.parent.is_symlink() or not stat.S_ISREG(info.st_mode):
                raise ValueError("stored metadata must be a regular tenant-local file")
            candidates.append((info.st_mtime_ns, str(path), path))
        except (OSError, ValueError):
            counts["restore_failures"] += 1
    return (path for _, _, path in sorted(candidates))


def _read_restore_file(path, limit):
    """Reject oversized files before allocation, including growth after stat."""
    if path.parent.is_symlink():
        raise ValueError("stored files must stay in their tenant directory")
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("stored file exceeds its restore bound")
        raw = stream.read(info.st_size + 1)
    if len(raw) > info.st_size:
        raise ValueError("stored file grew during restore")
    return raw


def _restore_identity(path, tenant_id, identifier):
    if (
        not isinstance(tenant_id, str)
        or not tenant_id
        or not isinstance(identifier, str)
        or not identifier
        or identifier in {".", ".."}
        or any(character in identifier for character in ("/", "\\", "\x00"))
        or path.parent.name != _tenant_name(tenant_id)
        or path.name != f"{identifier}.json"
    ):
        raise ValueError("stored resource identity does not match its path")


def _invalid_json_constant(value):
    raise ValueError(f"invalid JSON constant: {value}")


def _restore_json(raw):
    value = json.loads(raw, parse_constant=_invalid_json_constant)
    if not isinstance(value, dict):
        raise TypeError("stored resource must be an object")
    return value


def _unlink_restored(path, counts):
    try:
        path.unlink(missing_ok=True)
    except OSError:
        # Disk cleanup failure must not defeat the resident-memory bound.
        counts["restore_cleanup_failures"] += 1


def _page(records, *, limit=20, after=None):
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
        raise ValueError("limit must be an integer from 1 to 100")
    records = list(records)
    if after is not None:
        positions = [index for index, item in enumerate(records) if item["id"] == after]
        if not positions:
            raise ValueError("after cursor does not exist")
        records = records[positions[0] + 1 :]
    data = records[:limit]
    return {
        "object": "list",
        "data": deepcopy(data),
        "first_id": data[0]["id"] if data else None,
        "last_id": data[-1]["id"] if data else None,
        "has_more": len(records) > limit,
    }


class ResponseStore:
    """Bounded response payloads plus replayable chat context."""

    def __init__(self, *, max_entries=128, max_bytes=32 << 20, root=None):
        if type(max_entries) is not int or type(max_bytes) is not int:
            raise ValueError("response-store bounds must be integers")
        if max_entries < 1 or max_bytes < 1:
            raise ValueError("response-store bounds must be positive")
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self.root = Path(root).expanduser().resolve() if root is not None else None
        self._entries = OrderedDict()
        self._bytes = 0
        self._counts = Counter()
        self._lock = threading.Lock()
        self._restore()

    @staticmethod
    def _key(tenant_id, response_id):
        return str(tenant_id or "default"), str(response_id)

    def _path(self, tenant_id, response_id):
        if self.root is None:
            return None
        return self.root / _tenant_name(tenant_id) / f"{response_id}.json"

    def _restore(self):
        if self.root is None or not self.root.exists():
            return
        for path in _restore_candidates(self.root, self._counts):
            try:
                raw = _read_restore_file(path, self.max_bytes)
                record = _restore_json(raw)
                tenant_id = record.pop("tenant_id")
                if not isinstance(record.get("payload"), dict):
                    raise TypeError("stored response payload must be an object")
                response_id = record["payload"]["id"]
                _restore_identity(path, tenant_id, response_id)
                context = record.get("context_messages")
                if not isinstance(context, list) or any(
                    not isinstance(message, dict)
                    or not isinstance(message.get("role"), str)
                    for message in context
                ):
                    raise ValueError("stored response context must contain messages")
                payload = record["payload"]
                if payload.get("model") is not None and not isinstance(payload["model"], str):
                    raise ValueError("stored response model must be text")
                receipt = payload.get("mlx2")
                if receipt is not None and (
                    not isinstance(receipt, dict)
                    or (
                        receipt.get("agent_compat") is not None
                        and not isinstance(receipt["agent_compat"], dict)
                    )
                ):
                    raise ValueError("stored response receipt must be an object")
                key = self._key(tenant_id, response_id)
                if key in self._entries:
                    raise ValueError("duplicate stored response identity")
                self._entries[key] = (len(raw), record)
                self._bytes += len(raw)
            except (OSError, ValueError, KeyError, TypeError, RecursionError):
                self._counts["restore_failures"] += 1
                continue
            while len(self._entries) > self.max_entries or self._bytes > self.max_bytes:
                key, (size, _) = self._entries.popitem(last=False)
                self._bytes -= size
                _unlink_restored(self._path(*key), self._counts)
                self._counts["evictions"] += 1
        self._counts["restored"] += len(self._entries)

    def put(self, tenant_id, payload, context_messages):
        tenant_id = str(tenant_id or "default")
        record = {
            "payload": deepcopy(payload),
            "context_messages": deepcopy(context_messages),
        }
        encoded = _json_bytes({**record, "tenant_id": tenant_id})
        size = len(encoded)
        if size > self.max_bytes:
            raise ValueError("stored response exceeds the local response-store bound")
        key = self._key(tenant_id, payload["id"])
        with self._lock:
            path = self._path(*key)
            if path is not None:
                _atomic_write(path, encoded)
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._bytes -= previous[0]
            self._entries[key] = (size, record)
            self._bytes += size
            while (
                len(self._entries) > self.max_entries or self._bytes > self.max_bytes
            ):
                removed_key, (removed_size, _) = self._entries.popitem(last=False)
                self._bytes -= removed_size
                removed_path = self._path(*removed_key)
                if removed_path is not None:
                    removed_path.unlink(missing_ok=True)
                self._counts["evictions"] += 1
            self._counts["stores"] += 1

    def get(self, tenant_id, response_id):
        key = self._key(tenant_id, response_id)
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is None:
                self._counts["misses"] += 1
                raise ResourceNotFound(response_id)
            self._entries[key] = entry
            self._counts["retrievals"] += 1
            return deepcopy(entry[1]["payload"])

    def agent_compat_enabled(self, tenant_id, response_id):
        """The stored response's agent-compat mode (no counters touched).

        Recorded in the payload's ``mlx2.agent_compat`` receipt; responses
        without one were produced with translation off, as on main.
        Returns None for an unknown id.
        """
        key = self._key(tenant_id, response_id)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            receipt = entry[1]["payload"].get("mlx2") or {}
            return bool((receipt.get("agent_compat") or {}).get("enabled"))

    def context(self, tenant_id, response_id):
        key = self._key(tenant_id, response_id)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._counts["continuation_misses"] += 1
                raise ResourceNotFound(response_id)
            self._counts["continuations"] += 1
            return deepcopy(entry[1]["context_messages"])

    def input_record(self, tenant_id, response_id):
        """Snapshot input-item context and model without copying output payloads."""
        key = self._key(tenant_id, response_id)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._counts["input_item_misses"] += 1
                raise ResourceNotFound(response_id)
            self._counts["input_item_lists"] += 1
            return {
                "payload": {"model": deepcopy(entry[1]["payload"].get("model"))},
                "context_messages": deepcopy(entry[1]["context_messages"]),
            }

    def delete(self, tenant_id, response_id):
        key = self._key(tenant_id, response_id)
        with self._lock:
            entry = self._entries.pop(key, None)
            if entry is None:
                self._counts["delete_misses"] += 1
                raise ResourceNotFound(response_id)
            self._bytes -= entry[0]
            path = self._path(*key)
            if path is not None:
                path.unlink(missing_ok=True)
            self._counts["deletes"] += 1
        return {"id": str(response_id), "object": "response.deleted", "deleted": True}

    def status(self):
        with self._lock:
            return {
                "entries": len(self._entries),
                "bytes": self._bytes,
                "max_entries": self.max_entries,
                "max_bytes": self.max_bytes,
                "durable": self.root is not None,
                "counts": dict(self._counts),
            }


@dataclass(frozen=True)
class StoredFile:
    id: str
    tenant_id: str
    filename: str
    purpose: str
    content_type: str
    content: bytes
    created_at: int = field(default_factory=lambda: int(time.time()))

    def object(self, *, status="processed"):
        return {
            "id": self.id,
            "object": "file",
            "bytes": len(self.content),
            "created_at": self.created_at,
            "filename": self.filename,
            "purpose": self.purpose,
            "status": status,
        }


class FileStore:
    """Bounded tenant-scoped file content used by Files and Batch APIs."""

    PURPOSES = frozenset({"batch", "user_data"})

    def __init__(
        self,
        *,
        max_files=128,
        max_file_bytes=32 << 20,
        max_bytes=128 << 20,
        root=None,
    ):
        self.max_files = int(max_files)
        self.max_file_bytes = int(max_file_bytes)
        self.max_bytes = int(max_bytes)
        if min(self.max_files, self.max_file_bytes, self.max_bytes) < 1:
            raise ValueError("file-store bounds must be positive")
        self.root = Path(root).expanduser().resolve() if root is not None else None
        self._files = OrderedDict()
        self._bytes = 0
        self._counts = Counter()
        self._lock = threading.Lock()
        self._restore()

    def _paths(self, tenant_id, file_id):
        if self.root is None:
            return None, None
        directory = self.root / _tenant_name(tenant_id)
        return directory / f"{file_id}.json", directory / f"{file_id}.bin"

    def _restore(self):
        if self.root is None or not self.root.exists():
            return
        for metadata_path in _restore_candidates(self.root, self._counts):
            try:
                # Filename/type fields are small; do not let corrupt metadata
                # consume the content budget before its size is checked.
                metadata = _restore_json(_read_restore_file(metadata_path, 64 << 10))
                _restore_identity(metadata_path, metadata["tenant_id"], metadata["id"])
                if (
                    type(metadata.get("bytes")) is not int
                    or not 0 < metadata["bytes"] <= min(self.max_file_bytes, self.max_bytes)
                    or type(metadata.get("created_at")) is not int
                    or metadata["created_at"] < 0
                    or not isinstance(metadata.get("filename"), str)
                    or not 1 <= len(metadata["filename"]) <= 255
                    or not isinstance(metadata.get("content_type"), str)
                    or len(metadata["content_type"]) > 128
                    or metadata.get("purpose") not in self.PURPOSES
                ):
                    raise ValueError("invalid stored file metadata")
                content_path = metadata_path.with_suffix(".bin")
                content = _read_restore_file(content_path, metadata["bytes"])
                if len(content) != metadata["bytes"]:
                    raise ValueError("file byte count mismatch")
                item = StoredFile(
                    id=metadata["id"],
                    tenant_id=metadata["tenant_id"],
                    filename=metadata["filename"],
                    purpose=metadata["purpose"],
                    content_type=metadata["content_type"],
                    content=content,
                    created_at=metadata["created_at"],
                )
                key = (item.tenant_id, item.id)
                if key in self._files:
                    raise ValueError("duplicate stored file identity")
                self._files[key] = item
                self._bytes += len(content)
            except (OSError, ValueError, KeyError, TypeError, RecursionError):
                self._counts["restore_failures"] += 1
                continue
            while len(self._files) > self.max_files or self._bytes > self.max_bytes:
                key, removed = self._files.popitem(last=False)
                self._bytes -= len(removed.content)
                for path in self._paths(*key):
                    _unlink_restored(path, self._counts)
                self._counts["evictions"] += 1
        self._counts["restored"] += len(self._files)

    def create(self, tenant_id, *, filename, purpose, content_type, content):
        if purpose not in self.PURPOSES:
            raise ValueError("file purpose must be batch or user_data")
        if not isinstance(filename, str) or not filename or len(filename) > 255:
            raise ValueError("filename must contain 1 to 255 characters")
        if not isinstance(content, bytes) or not content:
            raise ValueError("file content must be nonempty bytes")
        if len(content) > min(self.max_file_bytes, self.max_bytes):
            raise ValueError("file exceeds the local per-file byte bound")
        if not isinstance(content_type, str) or len(content_type) > 128:
            raise ValueError("invalid file content type")
        item = StoredFile(
            id="file-" + uuid.uuid4().hex,
            tenant_id=str(tenant_id or "default"),
            filename=filename,
            purpose=purpose,
            content_type=content_type or "application/octet-stream",
            content=content,
        )
        key = (item.tenant_id, item.id)
        with self._lock:
            metadata_path, content_path = self._paths(*key)
            if metadata_path is not None:
                _atomic_write(content_path, content)
                _atomic_write(
                    metadata_path,
                    _json_bytes(
                        {
                            **item.object(),
                            "tenant_id": item.tenant_id,
                            "content_type": item.content_type,
                        }
                    ),
                )
            self._files[key] = item
            self._bytes += len(content)
            while len(self._files) > self.max_files or self._bytes > self.max_bytes:
                removed_key, removed = self._files.popitem(last=False)
                self._bytes -= len(removed.content)
                for path in self._paths(*removed_key):
                    if path is not None:
                        path.unlink(missing_ok=True)
                self._counts["evictions"] += 1
            self._counts["uploads"] += 1
        return item.object()

    def _get(self, tenant_id, file_id):
        key = (str(tenant_id or "default"), str(file_id))
        with self._lock:
            item = self._files.get(key)
            if item is None:
                self._counts["misses"] += 1
                raise ResourceNotFound(file_id)
            self._counts["retrievals"] += 1
            return item

    def get(self, tenant_id, file_id):
        return self._get(tenant_id, file_id).object()

    def content(self, tenant_id, file_id):
        item = self._get(tenant_id, file_id)
        return item.content, item.content_type, item.filename

    def list(self, tenant_id, *, limit=20, after=None, purpose=None):
        if purpose is not None and purpose not in self.PURPOSES:
            raise ValueError("unsupported file purpose filter")
        tenant_id = str(tenant_id or "default")
        with self._lock:
            records = [
                item.object()
                for (owner, _), item in self._files.items()
                if owner == tenant_id and (purpose is None or item.purpose == purpose)
            ]
            records.sort(key=lambda item: (-item["created_at"], item["id"]))
            self._counts["lists"] += 1
            return _page(records, limit=limit, after=after)

    def delete(self, tenant_id, file_id):
        key = (str(tenant_id or "default"), str(file_id))
        with self._lock:
            item = self._files.pop(key, None)
            if item is None:
                self._counts["delete_misses"] += 1
                raise ResourceNotFound(file_id)
            self._bytes -= len(item.content)
            for path in self._paths(*key):
                if path is not None:
                    path.unlink(missing_ok=True)
            self._counts["deletes"] += 1
        return {"id": str(file_id), "object": "file", "deleted": True}

    def status(self):
        with self._lock:
            return {
                "files": len(self._files),
                "bytes": self._bytes,
                "max_files": self.max_files,
                "max_file_bytes": self.max_file_bytes,
                "max_bytes": self.max_bytes,
                "durable": self.root is not None,
                "counts": dict(self._counts),
            }


class BatchManager:
    """Bounded asynchronous JSONL batches over local inference endpoints."""

    ENDPOINTS = frozenset(
        {"/v1/chat/completions", "/v1/completions", "/v1/responses", "/v1/embeddings"}
    )

    def __init__(
        self,
        file_store,
        executor,
        *,
        max_batches=32,
        max_lines=1000,
        root=None,
    ):
        self.file_store = file_store
        self.executor = executor
        self.max_batches = int(max_batches)
        self.max_lines = int(max_lines)
        if min(self.max_batches, self.max_lines) < 1:
            raise ValueError("batch bounds must be positive")
        self.root = Path(root).expanduser().resolve() if root is not None else None
        self._batches = OrderedDict()
        self._lock = threading.Lock()
        self._restore()

    def _path(self, tenant_id, batch_id):
        if self.root is None:
            return None
        return self.root / _tenant_name(tenant_id) / f"{batch_id}.json"

    def _persist(self, record):
        path = self._path(record["tenant_id"], record["id"])
        if path is not None:
            _atomic_write(path, _json_bytes({**self._public(record), "tenant_id": record["tenant_id"]}))

    def _restore(self):
        if self.root is None or not self.root.exists():
            return
        candidates = sorted(
            self.root.glob("*/*.json"), key=lambda path: path.stat().st_mtime_ns
        )
        for path in candidates:
            try:
                record = json.loads(path.read_bytes())
                if record["status"] in {"validating", "in_progress", "cancelling"}:
                    record["status"] = "failed"
                    record["failed_at"] = int(time.time())
                    record["errors"] = {
                        "data": [
                            {
                                "code": "server_restarted",
                                "message": "batch execution was interrupted by server restart",
                            }
                        ]
                    }
                record["cancel"] = threading.Event()
                key = (record["tenant_id"], record["id"])
                self._batches[key] = record
                self._persist(record)
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                continue
        while len(self._batches) > self.max_batches:
            key, _ = self._batches.popitem(last=False)
            path = self._path(*key)
            if path is not None:
                path.unlink(missing_ok=True)

    def create(self, tenant_id, body):
        if not isinstance(body, dict):
            raise ValueError("batch request must be an object")
        unknown = set(body) - {"input_file_id", "endpoint", "completion_window", "metadata"}
        if unknown:
            raise ValueError("unsupported batch fields: " + ", ".join(sorted(unknown)))
        endpoint = body.get("endpoint")
        if endpoint not in self.ENDPOINTS:
            raise ValueError("unsupported batch endpoint")
        if body.get("completion_window") != "24h":
            raise ValueError("completion_window must be '24h'")
        metadata = body.get("metadata", {})
        if (
            not isinstance(metadata, dict)
            or len(metadata) > 16
            or any(
                not isinstance(key, str)
                or not isinstance(value, str)
                or not 1 <= len(key) <= 64
                or len(value) > 512
                for key, value in metadata.items()
            )
        ):
            raise ValueError("batch metadata must contain at most 16 entries")
        file_id = body.get("input_file_id")
        content, _, _ = self.file_store.content(tenant_id, file_id)
        lines = [line for line in content.splitlines() if line.strip()]
        if not 1 <= len(lines) <= self.max_lines:
            raise ValueError(f"batch input must contain 1 to {self.max_lines} JSONL rows")
        batch_id = "batch_" + uuid.uuid4().hex
        now = int(time.time())
        record = {
            "id": batch_id,
            "object": "batch",
            "endpoint": endpoint,
            "input_file_id": file_id,
            "completion_window": "24h",
            "status": "validating",
            "created_at": now,
            "in_progress_at": None,
            "completed_at": None,
            "failed_at": None,
            "cancelling_at": None,
            "cancelled_at": None,
            "expires_at": now + 86400,
            "output_file_id": None,
            "error_file_id": None,
            "request_counts": {"total": len(lines), "completed": 0, "failed": 0},
            "metadata": dict(metadata),
            "errors": None,
            "tenant_id": str(tenant_id or "default"),
            "cancel": threading.Event(),
            "abort_reason": None,
        }
        key = (record["tenant_id"], batch_id)
        with self._lock:
            active = sum(
                item["status"] in {"validating", "in_progress", "cancelling"}
                for item in self._batches.values()
            )
            if active >= self.max_batches:
                raise CapabilityUnavailable("maximum active local batches reached")
            while len(self._batches) >= self.max_batches:
                removable = next(
                    (
                        key
                        for key, item in self._batches.items()
                        if item["status"]
                        in {"completed", "failed", "cancelled"}
                    ),
                    None,
                )
                if removable is None:
                    raise CapabilityUnavailable("maximum retained local batches reached")
                del self._batches[removable]
                path = self._path(*removable)
                if path is not None:
                    path.unlink(missing_ok=True)
            self._batches[key] = record
            self._persist(record)
        threading.Thread(
            target=self._run,
            args=(key, tuple(lines)),
            name=f"mlx2-{batch_id}",
            daemon=True,
        ).start()
        return self._public(record)

    def _run(self, key, lines):
        with self._lock:
            record = self._batches[key]
            record["status"] = "in_progress"
            record["in_progress_at"] = int(time.time())
            self._persist(record)
        outputs, errors = [], []
        for raw in lines:
            with self._lock:
                record = self._batches[key]
                cancelled = record["cancel"].is_set()
            if cancelled:
                break
            custom_id = None
            try:
                row = json.loads(raw)
                if not isinstance(row, dict) or set(row) != {"custom_id", "method", "url", "body"}:
                    raise ValueError("each batch row requires custom_id, method, url, and body")
                custom_id = row["custom_id"]
                if not isinstance(custom_id, str) or not custom_id:
                    raise ValueError("batch custom_id must be nonempty text")
                if row["method"] != "POST" or row["url"] != record["endpoint"]:
                    raise ValueError("batch row method/url must match POST and the batch endpoint")
                status, response = self.executor(
                    row["url"], row["body"], record["tenant_id"]
                )
                result = {
                    "id": "batch_req_" + uuid.uuid4().hex,
                    "custom_id": custom_id,
                    "response": {
                        "status_code": status,
                        "request_id": "req_" + uuid.uuid4().hex,
                        "body": response,
                    },
                    "error": None,
                }
                outputs.append(json.dumps(result, allow_nan=False).encode())
                with self._lock:
                    record["request_counts"]["completed"] += 1
                    self._persist(record)
            except Exception as error:  # one malformed row must not abort siblings
                status = getattr(error, "status", None)
                if (
                    isinstance(status, int)
                    and not isinstance(status, bool)
                    and 400 <= status <= 599
                ):
                    result = {
                        "id": "batch_req_" + uuid.uuid4().hex,
                        "custom_id": custom_id,
                        "response": {
                            "status_code": status,
                            "request_id": "req_" + uuid.uuid4().hex,
                            "body": {
                                "error": {
                                    "code": getattr(error, "code", "server_error"),
                                    "message": str(error),
                                    "type": "server_error",
                                }
                            },
                        },
                        "error": None,
                    }
                    outputs.append(json.dumps(result, allow_nan=False).encode())
                else:
                    result = {
                        "id": "batch_req_" + uuid.uuid4().hex,
                        "custom_id": custom_id,
                        "response": None,
                        "error": {"code": "invalid_request", "message": str(error)},
                    }
                    errors.append(json.dumps(result, allow_nan=False).encode())
                with self._lock:
                    record["request_counts"]["failed"] += 1
                    self._persist(record)
        with self._lock:
            record = self._batches[key]
            cancelled = record["cancel"].is_set()
            abort_reason = record.get("abort_reason")
        try:
            if outputs:
                object_ = self.file_store.create(
                    record["tenant_id"],
                    filename=f"{record['id']}-output.jsonl",
                    purpose="batch",
                    content_type="application/jsonl",
                    content=b"\n".join(outputs) + b"\n",
                )
                output_file_id = object_["id"]
            else:
                output_file_id = None
            if errors:
                object_ = self.file_store.create(
                    record["tenant_id"],
                    filename=f"{record['id']}-errors.jsonl",
                    purpose="batch",
                    content_type="application/jsonl",
                    content=b"\n".join(errors) + b"\n",
                )
                error_file_id = object_["id"]
            else:
                error_file_id = None
            with self._lock:
                record["output_file_id"] = output_file_id
                record["error_file_id"] = error_file_id
                if abort_reason:
                    remaining = (
                        record["request_counts"]["total"]
                        - record["request_counts"]["completed"]
                        - record["request_counts"]["failed"]
                    )
                    record["request_counts"]["failed"] += max(0, remaining)
                    record["status"] = "failed"
                    record["failed_at"] = int(time.time())
                    record["errors"] = {
                        "data": [
                            {
                                "code": "drain_timeout",
                                "message": abort_reason,
                                "status_code": 503,
                            }
                        ]
                    }
                elif cancelled:
                    record["status"] = "cancelled"
                    record["cancelled_at"] = int(time.time())
                else:
                    record["status"] = "completed"
                    record["completed_at"] = int(time.time())
                self._persist(record)
        except Exception as error:
            with self._lock:
                record["status"] = "failed"
                record["failed_at"] = int(time.time())
                record["errors"] = {"data": [{"code": "storage_error", "message": str(error)}]}
                self._persist(record)

    @staticmethod
    def _public(record):
        return {
            key: deepcopy(value)
            for key, value in record.items()
            if key not in {"tenant_id", "cancel", "abort_reason"}
        }

    def get(self, tenant_id, batch_id):
        key = (str(tenant_id or "default"), str(batch_id))
        with self._lock:
            record = self._batches.get(key)
            if record is None:
                raise ResourceNotFound(batch_id)
            return self._public(record)

    def list(self, tenant_id, *, limit=20, after=None):
        tenant_id = str(tenant_id or "default")
        with self._lock:
            records = [
                self._public(record)
                for (owner, _), record in self._batches.items()
                if owner == tenant_id
            ]
            records.sort(key=lambda item: (-item["created_at"], item["id"]))
            return _page(records, limit=limit, after=after)

    def cancel(self, tenant_id, batch_id):
        key = (str(tenant_id or "default"), str(batch_id))
        with self._lock:
            record = self._batches.get(key)
            if record is None:
                raise ResourceNotFound(batch_id)
            if record["status"] in {"validating", "in_progress"}:
                record["status"] = "cancelling"
                record["cancelling_at"] = int(time.time())
                record["cancel"].set()
                self._persist(record)
            return self._public(record)

    def abort_for_drain_timeout(self):
        """Stop every accepted batch and mark unstarted rows as HTTP 503 work."""
        with self._lock:
            aborted = 0
            for record in self._batches.values():
                if record["status"] not in {"validating", "in_progress", "cancelling"}:
                    continue
                record["abort_reason"] = "drain timeout"
                record["cancel"].set()
                aborted += 1
            return aborted

    def status(self):
        with self._lock:
            states = Counter(record["status"] for record in self._batches.values())
            completed = sum(
                record["request_counts"]["completed"]
                for record in self._batches.values()
            )
            failed = sum(
                record["request_counts"]["failed"]
                for record in self._batches.values()
            )
            return {
                "batches": len(self._batches),
                "max_batches": self.max_batches,
                "max_lines": self.max_lines,
                "durable": self.root is not None,
                "states": dict(states),
                "requests_completed": completed,
                "requests_failed": failed,
            }
