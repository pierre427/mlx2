"""Bounded, opt-in exact-prefix lookup for logical paged APCv2 checkpoints.

This is a directory policy around :mod:`paged_apc_bridge`, not another cache
engine. The index holds no physical page references and never admits a prefix
until the checkpoint's own digest and identity checks pass on restore.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from pathlib import Path

from .apc_v2 import APCKey
from .paged_apc_bridge import ExactPagedAPCCheckpoint, _canonical, _digest, _identity
from .paged_kv_cache import PagedKVPrivateCache
from .paged_kv_token import PagedKVTokenOwner
from .paged_kv_write import PagedKVWriteOwner, WriteTicket


class PagedAPCIndex:
    """Single-owner, bounded index; callers serialize access to its directory."""

    SCHEMA = "mlx2-paged-apcv2-index-v1"
    FILE = "paged-apcv2-index.json"

    def __init__(self, directory: Path, *, max_entries: int = 64,
                 max_tokens_per_entry: int = 8192,
                 max_payload_bytes: int = 1 << 30,
                 permit_candidate: bool = False) -> None:
        if not permit_candidate:
            raise RuntimeError("paged APCv2 index requires explicit candidate enablement")
        for label, value in (("max_entries", max_entries),
                             ("max_tokens_per_entry", max_tokens_per_entry),
                             ("max_payload_bytes", max_payload_bytes)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        self.directory = Path(directory)
        if self.directory.is_symlink():
            raise ValueError("paged APCv2 index directory cannot be a symlink")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_entries = max_entries
        self.max_tokens_per_entry = max_tokens_per_entry
        self.max_payload_bytes = max_payload_bytes
        self._entries: list[dict] = []
        path = self.directory / self.FILE
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_file() or path.stat().st_size > 8_000_000:
                raise ValueError("paged APCv2 index file is unsafe or oversized")
            document = json.loads(path.read_text())
            if type(document) is not dict or set(document) != {"schema", "entries"} or document["schema"] != self.SCHEMA:
                raise ValueError("paged APCv2 index schema mismatch")
            entries = document["entries"]
            if type(entries) is not list or len(entries) > max_entries:
                raise ValueError("paged APCv2 index entry bound exceeded")
            names: set[str] = set()
            used_bytes = 0
            for entry in entries:
                self._validate_entry(entry)
                if len(entry["tokens"]) > max_tokens_per_entry or entry["name"] in names:
                    raise ValueError("paged APCv2 index token bound or duplicate")
                names.add(entry["name"])
                used_bytes += entry["payload_bytes"]
                self._checkpoint_path(entry)
            if used_bytes > max_payload_bytes:
                raise ValueError("paged APCv2 index byte bound exceeded")
            self._entries = entries
        indexed = {entry["name"] for entry in self._entries}
        if any(re.fullmatch(r"paged-apcv2-[0-9a-f]{64}", path.name) and
               path.name not in indexed for path in self.directory.glob("paged-apcv2-*")):
            raise ValueError("paged APCv2 directory has an unindexed checkpoint")

    @staticmethod
    def _validate_entry(entry: object) -> None:
        if (type(entry) is not dict or set(entry) != {"name", "identity", "tokens", "payload_bytes"} or
                type(entry["name"]) is not str or
                re.fullmatch(r"paged-apcv2-[0-9a-f]{64}", entry["name"]) is None or
                type(entry["identity"]) is not dict or
                type(entry["tokens"]) is not list or not entry["tokens"] or
                any(type(t) is not int or t < 0 for t in entry["tokens"]) or
                type(entry["payload_bytes"]) is not int or entry["payload_bytes"] <= 0):
            raise ValueError("paged APCv2 index entry mismatch")
        identity = {"key": entry["identity"], "tokens": entry["tokens"]}
        if entry["name"] != "paged-apcv2-" + _digest(_canonical(identity).encode()):
            raise ValueError("paged APCv2 index digest mismatch")

    def _checkpoint_path(self, entry: dict) -> Path:
        path = self.directory / entry["name"]
        if path.is_symlink() or not path.is_dir():
            raise ValueError("paged APCv2 checkpoint path is missing or unsafe")
        for leaf in ("manifest.json", "keys.bin", "values.bin"):
            item = path / leaf
            if item.is_symlink() or not item.is_file():
                raise ValueError("paged APCv2 checkpoint file is missing or unsafe")
        manifest_path = path / "manifest.json"
        if manifest_path.stat().st_size > 1_000_000:
            raise ValueError("paged APCv2 checkpoint manifest is oversized")
        manifest = json.loads(manifest_path.read_text())
        if (type(manifest) is not dict or
                manifest.get("schema") != "mlx2-paged-apcv2-exact-v1" or
                manifest.get("identity") != {"key": entry["identity"],
                                             "tokens": entry["tokens"]} or
                type(manifest.get("profile")) is not dict or
                manifest["profile"].get("tokens") != len(entry["tokens"]) or
                (path / "keys.bin").stat().st_size + (path / "values.bin").stat().st_size != entry["payload_bytes"]):
            raise ValueError("paged APCv2 index checkpoint mismatch")
        return path

    def _write(self, entries: list[dict]) -> None:
        path = self.directory / self.FILE
        temporary = self.directory / ("." + self.FILE + "." + uuid.uuid4().hex + ".tmp")
        try:
            with temporary.open("xb") as output:
                output.write(_canonical({"schema": self.SCHEMA, "entries": entries}).encode())
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
            fd = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        finally:
            temporary.unlink(missing_ok=True)

    def _indexed_on_disk(self, name: str) -> bool:
        try:
            document = json.loads((self.directory / self.FILE).read_text())
            return any(entry.get("name") == name for entry in document["entries"])
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return False

    def store(self, *, key: APCKey, tokens: tuple[int, ...],
              source: PagedKVPrivateCache) -> Path:
        identity = _identity(key, tokens)
        if len(tokens) > self.max_tokens_per_entry:
            raise ValueError("paged APCv2 index token bound exceeded")
        payload = source.export_exact()
        if payload.tokens != len(tokens):
            raise ValueError("checkpoint tokens do not cover the accepted KV")
        payload_bytes = len(payload.key_bytes) + len(payload.value_bytes)
        if (len(self._entries) >= self.max_entries or
                sum(entry["payload_bytes"] for entry in self._entries) + payload_bytes > self.max_payload_bytes):
            raise MemoryError("paged APCv2 index capacity exceeded")
        name = "paged-apcv2-" + _digest(_canonical(identity).encode())
        if any(entry["name"] == name for entry in self._entries):
            raise FileExistsError("paged APCv2 checkpoint already indexed")
        path = ExactPagedAPCCheckpoint.save(self.directory, key=key, tokens=tokens,
                                            source=source, permit_candidate=True)
        entry = {"name": name, "identity": identity["key"],
                 "tokens": list(tokens), "payload_bytes": payload_bytes}
        updated = [*self._entries, entry]
        try:
            self._write(updated)
        except BaseException:
            # Not indexed, so never admitted: remove it, or the retry finds
            # the name taken and a restart finds an unindexed checkpoint.  A
            # failure after the index file was replaced (its directory fsync)
            # keeps the checkpoint the file now names.
            if not self._indexed_on_disk(name):
                shutil.rmtree(path, ignore_errors=True)
            raise
        self._entries = updated
        return path

    def lookup(self, *, key: APCKey, tokens: tuple[int, ...]) -> tuple[Path, int] | None:
        identity = _identity(key, tokens)
        matches = [entry for entry in self._entries
                   if entry["identity"] == identity["key"] and
                   len(entry["tokens"]) <= len(tokens) and
                   tuple(entry["tokens"]) == tokens[:len(entry["tokens"])]]
        if not matches:
            return None
        best = max(matches, key=lambda entry: len(entry["tokens"]))
        return self._checkpoint_path(best), len(best["tokens"])

    def restore_longest(self, *, key: APCKey, tokens: tuple[int, ...],
                        writer: PagedKVWriteOwner,
                        staging_headroom_pages: int = 0) -> tuple[PagedKVPrivateCache, tuple[WriteTicket, ...], int] | None:
        found = self.lookup(key=key, tokens=tokens)
        if found is None:
            return None
        path, count = found
        owner, tickets = ExactPagedAPCCheckpoint.restore(
            path, key=key, tokens=tokens[:count], writer=writer,
            staging_headroom_pages=staging_headroom_pages, permit_candidate=True)
        return owner, tickets, count

    def restore_longest_token_owner(
        self, *, key: APCKey, tokens: tuple[int, ...], writer: PagedKVWriteOwner,
    ) -> tuple[PagedKVTokenOwner, tuple[WriteTicket, ...], int] | None:
        """Restore the longest exact prefix to a fresh native token owner.

        No page table becomes readable until its write completions succeed.
        """
        found = self.lookup(key=key, tokens=tokens)
        if found is None:
            return None
        path, count = found
        owner, tickets = ExactPagedAPCCheckpoint.restore_token_owner(
            path, key=key, tokens=tokens[:count], writer=writer,
            permit_candidate=True)
        return owner, tickets, count


__all__ = ["PagedAPCIndex"]
