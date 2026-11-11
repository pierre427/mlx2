# SPDX-License-Identifier: MIT
"""Parent-indexed Nemotron-H (Mamba-2 hybrid) tree verify: topology and commit.

The structure follows Qwen3.8's TensorFold tree15 (recurrent layers branched
per node, one target dispatch per layer) and Ling's packed tree (this module's
transaction shape), with a stricter exactness bar: every node's target row is
bit-equal to one-token decode along that node's root-to-node path on the same
route, and the committed planes equal what an exact chain verify of the
accepted path commits.

Ownership:

* the model (``nemotron_h``) owns every tensor op.  Row-local work runs once
  per layer across all nodes through the chain's exact strip ops; the Mamba
  mixer branches its conv window and SSM recurrence per node
  (``NemotronHMamba2Mixer.tree_forward``); attention runs one query per node
  over ``[prefix; ancestors; self]`` keys with the one-token decode mask law.
* this tree owns parents, paths, conv windows, the request-private KV
  descriptors and the transaction material.  Nothing in the live cache below
  its committed fill level is written before ``publish``.

Commit replays the accepted path's recurrence from the committed state with
the decode kernel itself (``NemotronHMamba2Mixer.tree_commit_state``), gathers
the conv tail of the path, and compacts the path's KV rows.  All
layers are prepared before any descriptor is published; ``abort`` leaves the
live cache at its committed boundary.  Provenance:
provenance/nemotron-h-mamba-tree.json.
"""

from __future__ import annotations

import copy
from collections import deque

import mlx.core as mx

from .cache import ArraysCache, KVCache

NEMOTRON_TREE_MAX_ROWS = 16


class NemotronTree:
    """One request's verify tile: rooted parent-first rows, private state."""

    def __init__(self, parents, cache):
        parents = tuple(parents)
        if not 1 <= len(parents) <= NEMOTRON_TREE_MAX_ROWS:
            raise ValueError("Nemotron tree requires 1..16 rows")
        self.parents = parents
        self.paths = []
        self.depths = []
        for row, parent in enumerate(parents):
            if type(parent) is not int or (
                (row == 0 and parent != -1) or (row and not 0 <= parent < row)
            ):
                raise ValueError("Nemotron tree requires one parent-first root")
            path = [] if parent < 0 else self.paths[parent]
            self.paths.append([*path, row])
            self.depths.append(len(path))
        # A node whose path is rows 0..d reads a contiguous KV view.
        self.chain_prefix = [path == list(range(len(path))) for path in self.paths]
        for item in cache:
            if type(item) is KVCache:
                if item.keys is None or item.values is None or item.offset <= 0:
                    raise ValueError("Nemotron tree requires a prefilled KV cache")
                continue
            if not isinstance(item, ArraysCache):
                raise ValueError("Nemotron tree requires plain KV / Mamba caches")
            if (
                item.speculating
                or item.lengths is not None
                or item.left_padding is not None
                or item._rollbacks
                or item._rollback_position
                or item._rollback_positions is not None
            ):
                raise ValueError("Nemotron tree requires an unpadded committed cache")
            if item.cache[0] is None or item.cache[1] is None:
                raise ValueError("Nemotron tree requires prefilled Mamba state")
        self.live = cache
        self._live_ids = {id(item) for item in self.live}
        self._parents_array = None
        self._windows = {}
        self._mamba = {}
        self._kv = {}
        self.backends = {}

    # -- topology -----------------------------------------------------------

    def parents_array(self) -> mx.array:
        if self._parents_array is None:
            self._parents_array = mx.array(self.parents, dtype=mx.int32)
        return self._parents_array

    def window_rows(self, kernel_size: int) -> list[list[int]]:
        """Each node's K-1 path predecessors in ``[state rows; tile rows]``."""
        count = kernel_size - 1
        windows = []
        for path in self.paths:
            history = [*range(count), *(count + row for row in path[:-1])]
            windows.append(history[-count:])
        return windows

    def window_array(self, kernel_size: int) -> mx.array:
        cached = self._windows.get(kernel_size)
        if cached is None:
            cached = mx.array(self.window_rows(kernel_size), dtype=mx.int32)
            self._windows[kernel_size] = cached
        return cached

    def note_backend(self, stage: str, backend: str) -> None:
        bucket = self.backends.setdefault(stage, {})
        bucket[backend] = bucket.get(backend, 0) + 1

    def _require_live(self, cache) -> None:
        if id(cache) not in self._live_ids:
            raise ValueError("Nemotron tree layer cache is not part of this tile")

    # -- Mamba material ---------------------------------------------------------

    def record_mamba(self, cache, *, mixer, source, x, B, C, dt, gate, state) -> None:
        """Keep what the commit replay needs; never per-node states."""
        self._require_live(cache)
        if id(cache) in self._mamba:
            raise RuntimeError("Nemotron tree Mamba layer ran twice")
        self._mamba[id(cache)] = {
            "mixer": mixer, "source": source, "x": x, "B": B, "C": C, "dt": dt,
            "gate": gate, "state": state, "kernel": int(mixer.conv_kernel_size),
            "ssm_backend": None,
        }

    def set_mamba_backend(self, cache, backend: str) -> None:
        self._mamba[id(cache)]["ssm_backend"] = backend

    # -- attention --------------------------------------------------------------

    def kv_append(self, cache, keys, values):
        """Append all tile rows once to a private descriptor lending ``cache``.

        The private KVCache shares the live K/V buffer objects (the
        ``lend_append_only_kv`` contract): it writes only at and past the live
        fill level, so an abort leaves the committed prefix untouched.  A
        buffer that grows is rebound on the private descriptor only.
        """
        self._require_live(cache)
        if type(cache) is not KVCache:
            raise ValueError("Nemotron tree attention requires a plain KVCache")
        if id(cache) in self._kv:
            raise RuntimeError("Nemotron tree attention layer ran twice")
        if keys.shape[2] != len(self.parents) or values.shape[2] != len(self.parents):
            raise ValueError("Nemotron tree K/V rows differ from the tile")
        private = copy.copy(cache)
        start = int(cache.offset)
        private.update_and_fetch(keys, values)
        self._kv[id(cache)] = {"private": private, "start": start,
                               "keys": keys, "values": values}
        return private, start

    def node_keys_values(self, cache, row: int):
        """``[prefix; ancestors; self]`` K/V for one node, in path order."""
        entry = self._kv[id(cache)]
        private, start = entry["private"], entry["start"]
        end = start + len(self.paths[row])
        if self.chain_prefix[row]:
            # Same slice geometry as one-token decode's keys_and_values().
            if end < private.keys.shape[2]:
                return private.keys[..., :end, :], private.values[..., :end, :]
            return private.keys, private.values
        indices = mx.array(self.paths[row], dtype=mx.int32)
        keys = mx.concatenate(
            [private.keys[..., :start, :], mx.take(entry["keys"], indices, axis=2)], axis=2
        )
        values = mx.concatenate(
            [private.values[..., :start, :], mx.take(entry["values"], indices, axis=2)],
            axis=2,
        )
        return keys, values

    def private_kv(self, cache):
        return self._kv[id(cache)]["private"]

    # -- commit -------------------------------------------------------------------

    def validate_path(self, path):
        path = list(path)
        if (
            not path
            or any(type(row) is not int or not 0 <= row < len(self.parents) for row in path)
            or path != self.paths[path[-1]]
        ):
            raise ValueError("Nemotron tree commit must follow one complete root path")
        return path

    def prepare_commit(self, path):
        """Build every layer's committed descriptor; publish none of them."""
        path = self.validate_path(path)
        indices = mx.array(path, dtype=mx.int32)
        prepared = []
        for live in self.live:
            if type(live) is KVCache:
                entry = self._kv.get(id(live))
                if entry is None:
                    raise RuntimeError("Nemotron tree attention layer did not run")
                private, start = entry["private"], entry["start"]
                if not self.chain_prefix[path[-1]]:
                    for field, rows in (("keys", entry["keys"]), ("values", entry["values"])):
                        buffer = getattr(private, field)
                        buffer[..., start : start + len(path), :] = mx.take(
                            rows, indices, axis=2
                        )
                        setattr(private, field, buffer)
                private.offset = start + len(path)
                prepared.append((live, dict(private.__dict__)))
                continue
            material = self._mamba.get(id(live))
            if material is None:
                raise RuntimeError("Nemotron tree Mamba layer did not run")
            state = material["mixer"].tree_commit_state(material, path, indices)
            count = material["kernel"] - 1
            history = [*range(count), *(count + row for row in path)]
            conv = mx.take(
                material["source"], mx.array(history[-count:], dtype=mx.int32), axis=1
            )
            private = copy.copy(live)
            private.__dict__ = dict(live.__dict__)
            private.cache = [conv, state]
            private._rollbacks = deque()
            private.advance(len(path))
            prepared.append((live, dict(private.__dict__)))
        return prepared

    @staticmethod
    def publish(prepared):
        for live, fields in prepared:
            live.__dict__.clear()
            live.__dict__.update(fields)

    def commit(self, path):
        self.publish(self.prepare_commit(path))
        return self.live

    def release(self) -> None:
        """Drop tile material (views would pin the shared KV buffers)."""
        self._mamba.clear()
        self._kv.clear()


__all__ = ["NEMOTRON_TREE_MAX_ROWS", "NemotronTree"]
