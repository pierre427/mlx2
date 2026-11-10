"""Exact bounded file-row reads for a native F16 GGUF embedding.

This is ordinary embedding offload, not PLE. It never maps or uploads the
whole table. Original implementation; no external backend code copied.
"""
from __future__ import annotations
from collections import OrderedDict
import os
from pathlib import Path
import stat
import threading
import numpy as np
from .reader import GGUFError
from ..immutable_weight_pages import REBOOT_VOLATILE_STAMP_KEYS


def _stamp(value):
    return dict(size=value.st_size, mtime_ns=value.st_mtime_ns,
                ctime_ns=value.st_ctime_ns, inode=value.st_ino, device=value.st_dev)


class F16EmbeddingRows:
    """Own a read-only descriptor and a bounded LRU of immutable native rows.

    ``data_offset`` is GGUFFile.data_offset; tensor.offset is relative to it.
    ``expected_stat`` accepts the artifact seal's size/mtime_ns/inode/device
    mapping. A stat binding detects ordinary mutation, not hostile same-stat
    changes; content SHA verification belongs to the artifact acquisition gate.
    """
    def __init__(self, path, tensor, *, data_offset, expected_stat=None,
                 max_cache_bytes=8 << 20, max_output_bytes=64 << 20):
        self._fd = None
        self._lock = threading.RLock()
        self._cache = OrderedDict()
        self.cache_bytes = self.read_bytes = self.read_calls = self.cache_hits = 0
        if (tensor.ggml_type != 1 or len(tensor.shape) != 2
                or any(type(n) is not int or n <= 0 for n in tensor.shape)
                or type(data_offset) is not int or data_offset < 0
                or type(tensor.offset) is not int or tensor.offset < 0):
            raise GGUFError('file embedding requires a native rank2 F16 tensor')
        if (type(max_cache_bytes) is not int or max_cache_bytes < 0
                or type(max_output_bytes) is not int or max_output_bytes <= 0):
            raise GGUFError('invalid file embedding memory bounds')
        self.width, self.vocab_size = tensor.shape
        self.row_bytes = self.width * 2
        self.tensor_bytes = self.row_bytes * self.vocab_size
        if tensor.nbytes != self.tensor_bytes or self.row_bytes > max_output_bytes:
            raise GGUFError('embedding byte geometry exceeds bounds')
        self.start = data_offset + tensor.offset
        self.path = Path(path).absolute()
        self.max_cache_bytes, self.max_output_bytes = max_cache_bytes, max_output_bytes
        try:
            self._fd = os.open(self.path, os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0) | getattr(os, 'O_NOFOLLOW', 0))
            descriptor = os.fstat(self._fd)
            if not stat.S_ISREG(descriptor.st_mode):
                raise GGUFError('embedding source must be a regular file')
            self.stat_binding = _stamp(descriptor)
            if expected_stat is not None:
                # A stat holding only reboot-volatile keys binds nothing: refuse it.
                if not set(expected_stat) - REBOOT_VOLATILE_STAMP_KEYS or any(
                        k not in self.stat_binding or self.stat_binding[k] != v
                                           for k, v in expected_stat.items()
                                           if k not in REBOOT_VOLATILE_STAMP_KEYS):
                    raise GGUFError('embedding source differs from expected artifact stat')
            if self.start + self.tensor_bytes > descriptor.st_size:
                raise GGUFError('embedding tensor extends past file')
            self._verify()
        except BaseException:
            self.close()
            raise

    @property
    def closed(self):
        return self._fd is None

    def _verify(self):
        if self._fd is None:
            raise GGUFError('file embedding reader is closed')
        path_stat = os.stat(self.path, follow_symlinks=False)
        if (not stat.S_ISREG(path_stat.st_mode) or _stamp(path_stat) != self.stat_binding
                or _stamp(os.fstat(self._fd)) != self.stat_binding):
            raise GGUFError('embedding file changed after binding')

    def _read_row(self, token):
        if token in self._cache:
            self.cache_hits += 1
            self._cache.move_to_end(token)
            return self._cache[token]
        chunks, consumed = [], 0
        while consumed < self.row_bytes:
            raw = os.pread(self._fd, self.row_bytes - consumed,
                           self.start + token * self.row_bytes + consumed)
            self.read_calls += 1
            if not raw:
                raise GGUFError('short embedding row read')
            chunks.append(raw)
            consumed += len(raw)
            self.read_bytes += len(raw)
        raw = b''.join(chunks)
        if self.row_bytes <= self.max_cache_bytes:
            while self.cache_bytes + self.row_bytes > self.max_cache_bytes:
                self._cache.popitem(last=False)
                self.cache_bytes -= self.row_bytes
            self._cache[token] = raw
            self.cache_bytes += self.row_bytes
        return raw

    def rows(self, ids):
        ids = np.asarray(ids)
        if ids.dtype.kind not in ('i', 'u'):
            raise GGUFError('embedding token IDs must be integers')
        if ids.size and (np.any(ids < 0) or np.any(ids >= self.vocab_size)):
            raise GGUFError('embedding token ID out of range')
        if ids.size * self.row_bytes > self.max_output_bytes:
            raise GGUFError('embedding request exceeds bounded output bytes')
        with self._lock:
            try:
                self._verify()
                output = np.empty((ids.size, self.width), dtype='<f2')
                seen = {}
                for row, token in enumerate(ids.flat):
                    token = int(token)
                    if token in seen:
                        output[row] = output[seen[token]]
                    else:
                        output[row] = np.frombuffer(self._read_row(token), dtype='<f2')
                        seen[token] = row
                self._verify()
                return output.reshape(*ids.shape, self.width)
            except BaseException:
                self.close()
                raise

    def diagnostics(self):
        return dict(route='native-F16-file-rows', ple='not_applicable',
                    table_bytes=self.tensor_bytes, row_bytes=self.row_bytes,
                    cache_bytes=self.cache_bytes, max_cache_bytes=self.max_cache_bytes,
                    read_bytes=self.read_bytes, read_calls=self.read_calls,
                    cache_hits=self.cache_hits, closed=self.closed,
                    stat_binding=dict(self.stat_binding))

    def close(self):
        with self._lock:
            fd, self._fd = self._fd, None
            self._cache.clear()
            self.cache_bytes = 0
            if fd is not None:
                os.close(fd)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
