# SPDX-License-Identifier: Apache-2.0
"""Bounded immutable tensor pages with evaluation-owner leases.

The durable source is the original verified file. A page is one tensor payload,
not an OS page or an approximate copy. One host staging allocation plus a bounded
read chunk is reserved separately from resident device pages. No device work
runs on the I/O worker. See provenance/deepseek-v41-dense-paging.json.
"""

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
import stat
import threading
from types import MappingProxyType

import numpy as np

from .io_ownership import CloseAction, ReadPool, storage_owner


def file_stamp(value):
    return dict(size=value.st_size, mtime_ns=value.st_mtime_ns,
                ctime_ns=value.st_ctime_ns, inode=value.st_ino, device=value.st_dev)


# A volume's st_dev is assigned at mount and changes across reboots on macOS (seen
# 2026-11-09: 16777230 -> 16777234 for an unchanged file), so a seal written to disk
# must not bind it. Inode, size, mtime and ctime still catch replacement or edits.
REBOOT_VOLATILE_STAMP_KEYS = frozenset(("device", "dev"))


def same_stamp(live, sealed):
    """True when two file stamps match on every key except the reboot-volatile device id."""
    if not isinstance(live, dict) or not isinstance(sealed, dict):
        return False
    keys = (set(live) | set(sealed)) - REBOOT_VOLATILE_STAMP_KEYS
    return all(k in live and k in sealed and live[k] == sealed[k] for k in keys)


@dataclass(frozen=True)
class ImmutablePage:
    offset: int
    source_bytes: int
    resident_bound_bytes: int
    layout: str

    def __post_init__(self):
        if (type(self.offset) is not int or self.offset < 0
                or type(self.source_bytes) is not int or self.source_bytes < 1
                or type(self.resident_bound_bytes) is not int
                or self.resident_bound_bytes < self.source_bytes
                or not isinstance(self.layout, str) or not self.layout):
            raise ValueError("invalid immutable page geometry")


def read_page(fd, offset, size, chunk_bytes):
    """One owned allocation; no mmap, whole-model view or unbounded read call."""
    result = np.empty(size, np.uint8)
    destination = memoryview(result)
    positional_read_into = getattr(os, "preadv", None)
    cursor = 0
    while cursor < size:
        stop = min(cursor+chunk_bytes, size)
        if positional_read_into is not None:
            # Keep the destination owned by this call/ReadPool during a late
            # syscall. No intermediate bytes object or host-to-host copy.
            count = positional_read_into(fd, [destination[cursor:stop]], offset+cursor)
            if count <= 0 or count > stop-cursor:
                raise OSError("short or invalid immutable weight page read")
        else:
            raw = os.pread(fd, stop-cursor, offset+cursor)
            if not raw:
                raise OSError("short immutable weight page read")
            count = len(raw)
            result[cursor:cursor+count] = np.frombuffer(raw, np.uint8)
        cursor += count
    return result


class ImmutableWeightPages:
    """One source, one evaluation owner and one explicit active page window.

    ``materialize`` and ``evict`` are evaluation-thread callbacks. The former
    must finish the upload before returning; its returned object owns exact
    immutable data. The latter drops *all* provider-specific prepared aliases.
    Callers evaluate every dependent output before leaving ``window``. The
    synchronizer then completes scheduled work before those leases are released.
    """

    def __init__(self, path, pages, *, binding, expected_stat, cache_bytes,
                 materialize, evict, synchronize, minimum_bytes=None,
                 chunk_bytes=16 << 20):
        self.pages = MappingProxyType(dict(pages))
        if (not self.pages or any(not isinstance(k,str) or not k or not isinstance(v,ImmutablePage)
                                  for k,v in self.pages.items())
                or not isinstance(binding,str) or not binding
                or type(cache_bytes) is not int or cache_bytes < 1
                or type(chunk_bytes) is not int or not 1 <= chunk_bytes <= 16 << 20
                or any(not callable(fn) for fn in (materialize,evict,synchronize))):
            raise ValueError("invalid immutable weight page provider")
        largest = max(p.resident_bound_bytes for p in self.pages.values())
        self.minimum_bytes = largest if minimum_bytes is None else minimum_bytes
        if type(self.minimum_bytes) is not int or self.minimum_bytes < largest or cache_bytes < self.minimum_bytes:
            raise ValueError("dense page budget cannot hold an execution stage")
        self.path, self.binding = str(path), binding
        self.cache_bytes = self.maximum_bytes = cache_bytes
        self.chunk_bytes = chunk_bytes
        self.staging_bound_bytes = max(p.source_bytes for p in self.pages.values())+chunk_bytes
        manifest={name:dict(offset=p.offset,source_bytes=p.source_bytes,resident_bound_bytes=p.resident_bound_bytes,layout=p.layout)
                  for name,p in self.pages.items()}
        self.manifest_sha256=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
        self.materialize, self.evict, self.synchronize = materialize, evict, synchronize
        self.expected_stat = dict(expected_stat)
        self._cache, self._leased = OrderedDict(), set()
        self._owner = None
        self._active = self._closed = self._parked = False
        self._error = None
        self._lock = threading.RLock()
        self.resident_bound_bytes = 0
        self.stats = dict(hits=0, misses=0, read_bytes=0, evictions=0, windows=0, parks=0, activations=0)
        self.fd = os.open(self.path, os.O_RDONLY | getattr(os,"O_NOFOLLOW",0))
        try:
            self.stat_binding=file_stamp(os.fstat(self.fd))
            # A stat holding only reboot-volatile keys binds nothing: refuse it.
            if (not set(self.expected_stat) - REBOOT_VOLATILE_STAMP_KEYS
                    or any(key not in self.stat_binding or self.stat_binding[key] != value
                                             for key,value in self.expected_stat.items()
                                             if key not in REBOOT_VOLATILE_STAMP_KEYS)):
                raise ValueError("immutable weight source differs from admitted stat")
            self.verify()
            if any(p.offset+p.source_bytes > os.fstat(self.fd).st_size for p in self.pages.values()):
                raise ValueError("immutable page outside source")
            if os.uname().sysname == "Darwin":
                import fcntl
                fcntl.fcntl(self.fd,48,1)  # F_NOCACHE: bounded explicit reads.
                fcntl.fcntl(self.fd,45,0)  # F_RDAHEAD: no file-wide read-ahead.
            self.pool = ReadPool(1,"immutable-weights",timeout=60.0,max_pending=1,
                                 owner=storage_owner("immutable-weights",self.path))
        except BaseException:
            os.close(self.fd)
            raise

    def verify(self):
        if self._closed:
            raise ValueError("immutable pages are closed")
        source = os.stat(self.path,follow_symlinks=False)
        if (not stat.S_ISREG(source.st_mode) or file_stamp(source) != self.stat_binding
                or file_stamp(os.fstat(self.fd)) != self.stat_binding):
            raise ValueError("immutable weight source changed")

    def bind_owner(self):
        owner = threading.get_ident()
        if self._owner is not None and self._owner != owner:
            raise ValueError("immutable pages belong to another evaluation thread")
        self._owner = owner

    def _ready(self):
        self.bind_owner()
        if self._closed or self._parked or self._error is not None:
            raise ValueError("immutable pages are closed, parked or failed")

    @contextmanager
    def window(self):
        with self._lock:
            self._ready()
            if self._active:
                raise ValueError("immutable page window already owned")
            self.verify()
            self._active = True
        try:
            yield
        except BaseException as error:
            self._error = f"{type(error).__name__}: {error}"
            raise
        finally:
            fenced = False
            try:
                self.synchronize()
                fenced = True
                self.verify()
            except BaseException as error:
                self._error = f"{type(error).__name__}: {error}"
                raise
            finally:
                # A failed completion fence retains both leases and active
                # ownership. Cleanup may retry the fence but never drop them
                # merely because the caller unwound an exception.
                if fenced:
                    self._leased.clear()
                    self._active = False
                self.stats["windows"] += 1

    def acquire(self, name):
        with self._lock:
            self._ready()
            if not self._active:
                raise ValueError("dense weight acquisition requires an evaluation window")
            page = self.pages[name]
            if name in self._cache:
                self._cache.move_to_end(name)
                self._leased.add(name)
                self.stats["hits"] += 1
                return self._cache[name]
            required = page.resident_bound_bytes
            while self.resident_bound_bytes+required > self.cache_bytes:
                victim = next((key for key in self._cache if key not in self._leased),None)
                if victim is None:
                    raise ValueError("dense page window exceeds its admitted budget")
                self._drop(victim)
            self.verify()
            try:
                raw = self.pool.call(read_page,self.fd,page.offset,page.source_bytes,self.chunk_bytes)
                self.verify()
                value = self.materialize(name,raw)
                # Publish ownership before the post-upload source check. If
                # that check fails, cleanup still owns every uploaded buffer
                # and prepared alias until the window completion fence.
                self._cache[name] = value
                self.resident_bound_bytes += required
                self._leased.add(name)
                self.verify()
            except BaseException as error:
                self._error = f"{type(error).__name__}: {error}"
                raise
            self.stats["misses"] += 1
            self.stats["read_bytes"] += page.source_bytes
            return value

    def _drop(self, name):
        if name in self._leased:
            raise ValueError("cannot evict a leased immutable page")
        self.evict(name)
        del self._cache[name]
        self.resident_bound_bytes -= self.pages[name].resident_bound_bytes
        self.stats["evictions"] += 1

    def resize(self, size_bytes):
        with self._lock:
            self._ready()
            if self._active or type(size_bytes) is not int or not self.minimum_bytes <= size_bytes <= self.maximum_bytes:
                raise ValueError("cannot resize owned pages or below the stage floor")
            self.synchronize()
            while self.resident_bound_bytes > size_bytes:
                self._drop(next(iter(self._cache)))
            self.cache_bytes = size_bytes
            return self.cache_bytes

    def park(self):
        with self._lock:
            self.bind_owner()
            if self._closed or self._active:
                raise ValueError("cannot park closed or owned immutable pages")
            if self.pool.counters()["weight_read_inflight"]:
                raise ValueError("cannot park while a required immutable page read still owns staging memory")
            self.synchronize()
            while self._cache:
                self._drop(next(iter(self._cache)))
            self._parked = True
            self.stats["parks"] += 1
            return self.receipt()

    def activate(self):
        with self._lock:
            self.bind_owner()
            if self._closed or self._active or self._error is not None:
                raise ValueError("cannot activate closed, failed or owned immutable pages")
            self.verify()
            self._parked = False
            self.stats["activations"] += 1
            return self.receipt()

    def receipt(self):
        return dict(schema="mlx2.immutable-weight-pages.v1",binding=self.binding,
                    manifest_sha256=self.manifest_sha256,
                    page_count=len(self.pages),source_bytes=sum(p.source_bytes for p in self.pages.values()),
                    cache_bytes=self.cache_bytes,minimum_cache_bytes=self.minimum_bytes,
                    staging_bound_bytes=self.staging_bound_bytes,resident_bound_bytes=self.resident_bound_bytes,
                    resident_pages=len(self._cache),leased_pages=len(self._leased),active=self._active,
                    parked=self._parked,closed=self._closed,error=self._error,
                    read_chunk_bytes=self.chunk_bytes,counters=dict(self.stats),
                    read_pool=self.pool.counters(),qualification="unqualified")

    def close(self):
        with self._lock:
            if self._closed:return
            if self._active:
                if self._error is None:raise ValueError("cannot close an owned immutable page window")
                self.synchronize()
                self._leased.clear()
                self._active = False
            self._leased.clear()
            while self._cache:self._drop(next(iter(self._cache)))
            self._closed = True
            self.pool.close(CloseAction(lambda:os.close(self.fd)))
