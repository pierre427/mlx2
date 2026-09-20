# SPDX-License-Identifier: Apache-2.0
# Adapted from unified; see provenance/.
from __future__ import annotations

_ubc_evicted_bytes_total: int = 0
_ubc_evict_calls_total: int = 0
_ubc_evict_failed_total: int = 0
import ctypes
import ctypes.util
import logging
import os
import sys
import threading
import time
from collections.abc import Callable, Iterable

logger = logging.getLogger(__name__)
_PROT_READ: int = 1
_MAP_SHARED: int = 1
_MS_INVALIDATE: int = 2
_MMAP_FAILED: int = ctypes.c_void_p(-1).value
_libc_lock = threading.Lock()
_libc: ctypes.CDLL | None = None


def _get_libc() -> ctypes.CDLL | None:
    """Return the cached libc handle with the three syscalls typed, or None."""
    global _libc
    if sys.platform != "darwin":
        return None
    if _libc is not None:
        return _libc
    with _libc_lock:
        if _libc is not None:
            return _libc
        try:
            libname = ctypes.util.find_library("c") or "libSystem.dylib"
            lib = ctypes.CDLL(libname, use_errno=True)
            lib.mmap.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_longlong,
            ]
            lib.mmap.restype = ctypes.c_void_p
            lib.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            lib.msync.restype = ctypes.c_int
            lib.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            lib.munmap.restype = ctypes.c_int
            _libc = lib
        except OSError as e:
            logger.debug("ubc_evict: libc load failed: %s", e)
            _libc = None
    return _libc


_counter_lock = threading.Lock()


def _bump_counter(evicted: int, *, failed: bool) -> None:
    global _ubc_evicted_bytes_total, _ubc_evict_calls_total, _ubc_evict_failed_total
    with _counter_lock:
        _ubc_evict_calls_total += 1
        if failed:
            _ubc_evict_failed_total += 1
        elif evicted > 0:
            _ubc_evicted_bytes_total += int(evicted)


def ubc_evict(path: str) -> int:
    """Evict the UBC mirror of ``path`` via ``msync(MS_INVALIDATE)``.

    Returns the file size on success (upper bound on bytes asked to discard),
    0 in every error path and on non-Darwin platforms. Never raises.
    """
    if sys.platform != "darwin":
        logger.debug("ubc_evict no-op on %s", sys.platform)
        _bump_counter(0, failed=False)
        return 0
    libc = _get_libc()
    if libc is None:
        logger.debug("ubc_evict: libc unavailable, no-op")
        _bump_counter(0, failed=True)
        return 0
    try:
        size = os.path.getsize(path)
    except OSError as e:
        logger.warning("ubc_evict: stat %s failed: %s", path, e)
        _bump_counter(0, failed=True)
        return 0
    if size <= 0:
        logger.debug("ubc_evict: %s is empty, no-op", path)
        _bump_counter(0, failed=False)
        return 0
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError as e:
        logger.warning("ubc_evict: open %s failed: %s", path, e)
        _bump_counter(0, failed=True)
        return 0
    try:
        ctypes.set_errno(0)
        addr = libc.mmap(None, size, _PROT_READ, _MAP_SHARED, fd, 0)
        if addr is None or addr == 0 or addr == _MMAP_FAILED:
            err = ctypes.get_errno()
            logger.warning(
                "ubc_evict: mmap %s failed errno=%d (%s)",
                path,
                err,
                os.strerror(err) if err else "unknown",
            )
            _bump_counter(0, failed=True)
            return 0
        msync_ok = False
        munmap_ok = False
        try:
            ctypes.set_errno(0)
            rc = libc.msync(addr, size, _MS_INVALIDATE)
            if rc != 0:
                err = ctypes.get_errno()
                logger.warning(
                    "ubc_evict: msync(MS_INVALIDATE) %s rc=%d errno=%d (%s)",
                    path,
                    rc,
                    err,
                    os.strerror(err) if err else "unknown",
                )
            else:
                msync_ok = True
        finally:
            ctypes.set_errno(0)
            munmap_rc = libc.munmap(addr, size)
            if munmap_rc != 0:
                err = ctypes.get_errno()
                logger.warning(
                    "ubc_evict: munmap %s rc=%d errno=%d (%s)",
                    path,
                    munmap_rc,
                    err,
                    os.strerror(err) if err else "unknown",
                )
            else:
                munmap_ok = True
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
    if not (msync_ok and munmap_ok):
        _bump_counter(0, failed=True)
        return 0
    _bump_counter(size, failed=False)
    return size


def load_shards_evicting(
    files: Iterable[object],
    *,
    sanitize: "Callable[[dict], dict] | None" = None,
    keep_lazy: "Callable[[str], bool] | None" = None,
    deferred: "list[str] | None" = None,
) -> dict:
    """Load safetensors shards, evicting each shard's UBC mirror as it lands.

    ``mx.load`` is lazy: it costs ~1 MiB and ~2 ms for a multi-GiB shard and
    reads nothing. The file's bytes enter the unified buffer cache only when
    the tensors are *materialised* by ``mx.eval``. A loader that loads every
    shard, materialises the whole model in one ``mx.eval``, and only then
    evicts therefore peaks at ``materialised weights + a UBC mirror of the
    same bytes`` -- roughly twice the model size.

    This helper materialises one shard at a time and evicts that shard's
    mirror immediately, so the mirror never exceeds a single shard.

    The ordering invariant is the whole point and must not be inverted:
    **eviction of a file always follows materialisation of that file's
    tensors, never precedes it.** Evicting first would discard a cache the
    subsequent ``mx.eval`` has to refill from disk -- a pessimisation, not a
    saving.

    ``sanitize`` is an optional per-shard ``dict -> dict`` transform applied
    *before* materialisation, so tensors the model discards (a dropped vision
    tower, for instance) are never read at all. It must be key-local: it sees
    one shard, not the whole checkpoint.

    ``keep_lazy`` names tensors that must stay unevaluated -- typically ones a
    later step prunes or rebinds to a sidecar. They are returned lazily, and
    because their bytes are still unread, the file holding them cannot be
    evicted yet: its path is appended to ``deferred`` for the caller to evict
    once it is done with them.
    """
    import mlx.core as mx

    if keep_lazy is not None and deferred is None:
        raise ValueError("keep_lazy requires a deferred list to record held paths")

    weights: dict = {}
    for file in files:
        path = str(file)
        shard = mx.load(path)
        if sanitize is not None:
            shard = sanitize(shard)
        if keep_lazy is None:
            eager = list(shard.values())
            held = False
        else:
            eager = [value for name, value in shard.items() if not keep_lazy(name)]
            held = len(eager) != len(shard)
        if eager:
            # Materialise first -- eviction below depends on this having run.
            mx.eval(eager)
        if held:
            deferred.append(path)
        else:
            ubc_evict_paths([path])
        weights.update(shard)
    return weights


def ubc_evict_paths(paths: Iterable[str]) -> int:
    """Evict each path; aggregate bytes evicted; never raise."""
    if sys.platform != "darwin":
        logger.debug("ubc_evict_paths no-op on %s", sys.platform)
        return 0
    total = 0
    t0 = time.monotonic()
    for p in paths:
        bytes_evicted = ubc_evict(str(p))
        if bytes_evicted > 0:
            logger.info(
                "ubc_evict: evicted %.1f MB from UBC for %s",
                bytes_evicted / (1024 * 1024),
                p,
            )
        total += bytes_evicted
    if total > 0:
        logger.info(
            "ubc_evict: pass complete total_mb=%.1f elapsed_s=%.2f",
            total / (1024 * 1024),
            time.monotonic() - t0,
        )
    return total
