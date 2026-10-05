"""Default-off APCv2 identity bridge for exact logical paged KV checkpoints.

Only accepted token-major bytes cross the process boundary. Physical page IDs,
generations, and completion epochs are never serialized. A restored leaf still
needs its write completions before attention or publication.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from dataclasses import asdict
from pathlib import Path

from .apc_v2 import APCKey, _json_identity_value
from .paged_attention_plan import PAGE_SIZE
from .paged_kv_cache import PagedKVPrivateCache
from .paged_kv_token import PagedKVTokenOwner, TokenKVProfile
from .paged_kv_write import PagedKVWriteOwner, WriteTicket


def _identity(key: APCKey, tokens: tuple[int, ...]) -> dict:
    if type(key) is not APCKey or key.revision is None:
        raise ValueError("paged APCv2 checkpoint requires a revision-bound APCKey")
    if type(tokens) is not tuple or not tokens or any(type(t) is not int or t < 0 for t in tokens):
        raise ValueError("paged APCv2 checkpoint requires exact token IDs")
    return {"key": _json_identity_value(asdict(key)), "tokens": list(tokens)}


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


class ExactPagedAPCCheckpoint:
    """A single logical checkpoint; caller owns admission and directory policy."""

    @staticmethod
    def save(directory: Path, *, key: APCKey, tokens: tuple[int, ...],
             source: PagedKVPrivateCache, permit_candidate: bool = False) -> Path:
        if not permit_candidate:
            raise RuntimeError("paged APCv2 persistence requires explicit candidate enablement")
        identity = _identity(key, tokens)
        payload = source.export_exact()
        if payload.tokens != len(tokens):
            raise ValueError("checkpoint tokens do not cover the accepted KV")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        name = "paged-apcv2-" + _digest(_canonical(identity).encode())
        target = directory / name
        if target.exists():
            raise FileExistsError("paged APCv2 checkpoint already exists")
        temporary = directory / ("." + name + "." + uuid.uuid4().hex + ".tmp")
        temporary.mkdir(mode=0o700)
        try:
            manifest = {
                "schema": "mlx2-paged-apcv2-exact-v1", "identity": identity,
                "profile": {"tokens": payload.tokens, "kv_heads": payload.kv_heads,
                            "head_dim": payload.head_dim, "dtype": payload.dtype},
                "files": {"keys.bin": _digest(payload.key_bytes),
                          "values.bin": _digest(payload.value_bytes)},
            }
            for name, data in (("keys.bin", payload.key_bytes),
                               ("values.bin", payload.value_bytes),
                               ("manifest.json", json.dumps(manifest, sort_keys=True).encode())):
                with (temporary / name).open("xb") as output:
                    output.write(data)
                    output.flush()
                    os.fsync(output.fileno())
            os.replace(temporary, target)
            fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            return target
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    @staticmethod
    def _read_payload(path: Path, *, key: APCKey, tokens: tuple[int, ...],
                      writer: PagedKVWriteOwner,
                      staging_headroom_pages: int) -> tuple[dict, dict[str, bytes]]:
        if type(writer) is not PagedKVWriteOwner:
            raise TypeError("paged APCv2 restore requires a PagedKVWriteOwner")
        identity = _identity(key, tokens)
        path = Path(path)
        if path.is_symlink() or not path.is_dir():
            raise ValueError("paged APCv2 checkpoint path is missing or unsafe")
        manifest_path = path / "manifest.json"
        if (manifest_path.is_symlink() or not manifest_path.is_file() or
                manifest_path.stat().st_size > 1_000_000):
            raise ValueError("paged APCv2 checkpoint manifest is missing, unsafe or oversized")
        with manifest_path.open("rb") as stream:
            manifest_bytes = stream.read(1_000_001)
        if len(manifest_bytes) > 1_000_000:
            raise ValueError("paged APCv2 checkpoint manifest is oversized")
        manifest = json.loads(manifest_bytes)
        if type(manifest) is not dict or set(manifest) != {"schema", "identity", "profile", "files"}:
            raise ValueError("paged APCv2 checkpoint manifest mismatch")
        if (manifest["schema"] != "mlx2-paged-apcv2-exact-v1" or
                _canonical(manifest["identity"]) != _canonical(identity)):
            raise ValueError("paged APCv2 checkpoint identity mismatch")
        profile = manifest["profile"]
        if (type(profile) is not dict or
                set(profile) != {"tokens", "kv_heads", "head_dim", "dtype"} or
                type(profile["tokens"]) is not int or profile["tokens"] != len(tokens) or
                type(profile["kv_heads"]) is not int or profile["kv_heads"] <= 0 or
                type(profile["head_dim"]) is not int or profile["head_dim"] not in (128, 256) or
                profile["dtype"] not in ("float16", "bfloat16")):
            raise ValueError("paged APCv2 checkpoint geometry mismatch")
        files = manifest["files"]
        if (type(files) is not dict or set(files) != {"keys.bin", "values.bin"} or
                any(type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None
                    for value in files.values())):
            raise ValueError("paged APCv2 checkpoint file manifest mismatch")
        token_bytes = profile["kv_heads"] * profile["head_dim"] * 2
        if writer.page_bytes != PAGE_SIZE * token_bytes:
            raise ValueError("paged APCv2 checkpoint writer geometry mismatch")
        if type(staging_headroom_pages) is not int or staging_headroom_pages < 0:
            raise ValueError("staging headroom must be a nonnegative page count")
        required = (len(tokens) + PAGE_SIZE - 1) // PAGE_SIZE
        if writer.pool.free_count < required + staging_headroom_pages:
            raise MemoryError("paged APCv2 restore lacks page and staging headroom")
        data = {}
        for name in ("keys.bin", "values.bin"):
            leaf = path / name
            if leaf.is_symlink() or not leaf.is_file() or leaf.stat().st_size != len(tokens) * token_bytes:
                raise ValueError("paged APCv2 checkpoint payload mismatch")
            value = leaf.read_bytes()
            if _digest(value) != files[name] or len(value) != len(tokens) * token_bytes:
                raise ValueError("paged APCv2 checkpoint payload mismatch")
            data[name] = value
        return profile, data

    @staticmethod
    def restore(path: Path, *, key: APCKey, tokens: tuple[int, ...],
                writer: PagedKVWriteOwner, staging_headroom_pages: int = 0,
                permit_candidate: bool = False) -> tuple[PagedKVPrivateCache, tuple[WriteTicket, ...]]:
        if not permit_candidate:
            raise RuntimeError("paged APCv2 restore requires explicit candidate enablement")
        profile, data = ExactPagedAPCCheckpoint._read_payload(
            path, key=key, tokens=tokens, writer=writer,
            staging_headroom_pages=staging_headroom_pages)
        reserved = writer.pool.reserve(staging_headroom_pages) if staging_headroom_pages else ()
        owner: PagedKVPrivateCache | None = None
        try:
            owner = PagedKVPrivateCache(writer, kv_heads=profile["kv_heads"],
                                        head_dim=profile["head_dim"], dtype=profile["dtype"],
                                        permit_candidate=True)
            owner._staging_reservation = reserved
            tickets = owner.append(data["keys.bin"], data["values.bin"])
        except Exception:
            if owner is not None:
                owner.close()
            elif reserved:
                writer.pool.release(reserved, after_epoch=writer.ledger.completed_epoch)
                writer.pool.retire(writer.ledger.completed_epoch)
            raise
        return owner, tickets

    @staticmethod
    def restore_token_owner(path: Path, *, key: APCKey, tokens: tuple[int, ...],
                            writer: PagedKVWriteOwner,
                            staging_headroom_pages: int = 0,
                            permit_candidate: bool = False) -> tuple[PagedKVTokenOwner, tuple[WriteTicket, ...]]:
        """Restore token-major logical bytes to fresh native head-local pages.

        This starts a private write transaction. The returned owner has offset
        zero until its tickets receive successful terminal completions through
        ``poll_completions``. The caller serializes writer and pool operations.
        """
        if not permit_candidate:
            raise RuntimeError("paged APCv2 token restore requires explicit candidate enablement")
        # A token owner cannot retain arbitrary staging pages. Refuse a promise
        # of reserved future headroom rather than silently releasing it.
        if staging_headroom_pages:
            raise ValueError("token restore cannot retain staging headroom")
        profile, data = ExactPagedAPCCheckpoint._read_payload(
            path, key=key, tokens=tokens, writer=writer,
            staging_headroom_pages=0)
        token_profile = TokenKVProfile(profile["kv_heads"], profile["head_dim"],
                                       profile["dtype"])
        owner = PagedKVTokenOwner(writer, token_profile, permit_candidate=True)
        try:
            spans = owner.planned_spans(len(tokens))
            import mlx.core as mx
            import numpy as np

            head_bytes = token_profile.head_token_bytes
            token_bytes = token_profile.token_bytes

            def chunks(payload: bytes) -> tuple[object, ...]:
                values = []
                for span in spans:
                    chunk = bytearray(span.byte_count)
                    for row in range(span.token_count):
                        begin = ((span.source_token_offset + row) * token_bytes +
                                 span.kv_head * head_bytes)
                        chunk[row * head_bytes:(row + 1) * head_bytes] = payload[begin:begin + head_bytes]
                    values.append(mx.array(np.frombuffer(chunk, dtype=np.uint8).copy(),
                                           dtype=mx.uint8))
                return tuple(values)

            tickets = owner.append(chunks(data["keys.bin"]), chunks(data["values.bin"]),
                                   token_count=len(tokens))
            return owner, tickets
        except BaseException:
            owner.close()
            raise


__all__ = ["ExactPagedAPCCheckpoint"]
