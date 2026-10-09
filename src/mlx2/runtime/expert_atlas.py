"""Expert access atlas: COLLECT ONLY, NEVER PIN.

The atlas accumulates decay-weighted per-(layer, expert) access counts while a
streamed model serves, persists them atomically, and validates them against the
checkpoint they were built from.  In this iteration it **must not influence
residency**: the cache in :mod:`mlx2.runtime.weight_stream` runs a plain LRU
whatever the atlas says.

The reason the counts are collected at all is
:func:`replay_counterfactual`, which replays a recorded access trace and
reports the hit rate an atlas-pinned resident set *would* have achieved against
the LRU that actually ran.  That number is the gate for ever building pinning,
and it settles the question on our own models instead of on someone else's
benchmark.

Validation fails closed in the weak sense the design specifies: a stale,
partial or mismatched atlas is *ignored* (the run continues unpinned), never
fatal.  It cannot change output either way -- nothing here touches arithmetic.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ATLAS_FORMAT = "mlx2-weight-atlas"
ATLAS_VERSION = 1
ATLAS_KIND = "moe_expert"
DATA_OFFSET = 4096
COUNTER_ITEMSIZE = 8
DEFAULT_HALF_LIFE = 2_000_000
DEFAULT_MIN_SAMPLES = 20_000
# Trace v2: uint32 ``(layer, id)`` rows, each ``observe()`` call preceded by
# a ``(_TRACE_CALL, n)`` row, so calls replay exactly.  v1 had no call rows:
# its calls can only be inferred, and its replay is approximate.
TRACE_MAGIC = b"MLX2XTR2"
TRACE_MAGIC_V1 = b"MLX2XTR1"
# Marker rows (first column; a layer index never reaches these values).
_TRACE_CALL = 0xFFFFFFFF  # (CALL, n): the next n rows are one call
_TRACE_LOST = 0xFFFFFFFE  # (LOST, layer): an acquire that failed, unrecorded
_TRACE_END = 0xFFFFFFFD  # (END, dropped calls): written once, at close
ATLAS_MAGIC = b"MLX2ATL1"
ATLAS_SENTINEL = b"ENDATLAS"


def index_digest(model_path) -> str:
    """SHA-256 binding an atlas to one checkpoint.

    Prefers ``model.safetensors.index.json``; a single-shard checkpoint has no
    index, so its shard headers are hashed instead.
    """
    root = Path(model_path).expanduser()
    index = root / "model.safetensors.index.json"
    digest = hashlib.sha256()
    if index.is_file():
        digest.update(index.read_bytes())
        return digest.hexdigest()
    from .weight_stream import read_safetensors_header

    for shard in sorted(root.glob("*.safetensors")):
        header = dict(read_safetensors_header(shard))
        header.pop("__data_offset__", None)
        digest.update(shard.name.encode("utf-8"))
        digest.update(json.dumps(header, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# collection
# ---------------------------------------------------------------------------


class AtlasCollector:
    """Decay-weighted access counts, plus an optional replay trace.

    Counting happens in the paging path, not in the model's forward: the cache
    already resolves every unit, so this is one vectorised increment per layer
    per step.
    """

    def __init__(
        self,
        model_path=None,
        *,
        trace_path=None,
        trace_limit: int = 1 << 22,
        checkpoint_every: int = 100_000,
        sink=None,
    ):
        self.model_path = model_path
        self.num_layers = 0
        self.num_units = 0
        self.counts = None
        # Counts already merged into the sink. ``counts`` is cumulative for
        # the collector's lifetime, so each flush merges only the difference;
        # merging the whole array would re-add every earlier checkpoint.
        self._persisted_counts = None
        # The atlas the last flush wrote, or before the first flush the one
        # the sink held at bind: decayed counts, total and history. A sink
        # that has since gone missing is rebuilt from this rather than from
        # ``counts``, whose lifetime totals never decay.
        self._persisted_atlas = None
        self.observations = 0
        self.persisted_observations = 0
        self.checkpoint_every = int(checkpoint_every)
        self.sink = Path(sink).expanduser() if sink else None
        self.trace_path = Path(trace_path).expanduser() if trace_path else None
        self.trace_limit = int(trace_limit)
        self._trace = None
        self._trace_written = 0
        # Set at the first call the record limit drops: no later call is
        # written either, so the trace stays an exact prefix of the run.
        self._trace_full = False
        self._trace_dropped_calls = 0
        self._steps = 0
        self._dropped = 0
        self._closed = False

    def bind(self, *, num_layers: int, num_units: int) -> None:
        import numpy as np

        self.num_layers = int(num_layers)
        self.num_units = int(num_units)
        self.counts = np.zeros((self.num_layers, self.num_units), dtype=np.uint64)
        self._persisted_counts = np.zeros_like(self.counts)
        if self.sink is not None:
            # The first flush merges onto the sink as it then stands. Keep
            # the atlas it holds now, validated as that flush validates it,
            # so a sink lost before the first flush is rebuilt from it too.
            self._persisted_atlas = load_atlas(
                self.sink, expect_digest=self._digest(), geometry=self.counts.shape
            )
        if self.trace_path is not None and self._trace is None:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            self._trace = open(self.trace_path, "wb")
            self._trace.write(
                TRACE_MAGIC
                + struct.pack("<II", self.num_layers, self.num_units)
            )

    def observe(self, layer: int, units) -> None:
        if self.counts is None:
            return
        import numpy as np

        ids = np.asarray(units, dtype=np.int64)
        if ids.size == 0 or layer >= self.num_layers:
            return
        np.add.at(self.counts[layer], ids, np.uint64(1))
        self.observations += int(ids.size)
        self._steps += 1
        if self._trace is not None:
            if not self._trace_full and self._trace_written + ids.size <= self.trace_limit:
                payload = np.empty((ids.size + 1, 2), dtype=np.uint32)
                payload[0] = (_TRACE_CALL, ids.size)
                payload[1:, 0] = layer
                payload[1:, 1] = ids
                self._trace.write(payload.tobytes())
                self._trace_written += int(ids.size)
            else:
                self._trace_full = True
                self._trace_dropped_calls += 1
                self._dropped += int(ids.size)
        if (
            self.sink is not None
            and self.checkpoint_every > 0
            and self.observations - self.persisted_observations
            >= self.checkpoint_every
        ):
            self.flush()

    def note_lost_call(self, layer: int) -> None:
        """An ``acquire`` that raised: it changed the LRU (it evicts before it
        reads) but no call was observed, so the trace marks the gap."""
        if self._trace is not None and not self._trace_full:
            self._trace.write(struct.pack("<II", _TRACE_LOST, int(layer)))

    def counters(self) -> Dict[str, int]:
        return {
            "atlas_observations_total": self.observations,
            "atlas_trace_records_total": self._trace_written,
            "atlas_trace_dropped_total": self._dropped,
        }

    def flush(self) -> Optional[Path]:
        """Merge into the persisted atlas and rewrite it atomically."""
        if self.sink is None or self.counts is None:
            return None
        prior = load_atlas(
            self.sink, expect_digest=self._digest(), geometry=self.counts.shape
        )
        if prior is None:
            # Merging only the difference assumes the sink still holds what
            # was persisted before. A sink removed, rotated or replaced by an
            # incompatible one holds none of it, so merge onto the atlas the
            # last flush wrote. Rewriting the lifetime counts instead would
            # undo the decay earlier checkpoints applied, and drop the counts
            # and generations of earlier runs that only the sink carried.
            # Before the first flush that is the atlas the sink held at bind;
            # with none, nothing is persisted and the difference is every
            # observation.
            prior = self._persisted_atlas
        snapshot = self.counts.copy()
        new_observations = self.observations - self.persisted_observations
        merged = merge_counts(
            prior.counts if prior is not None else None,
            snapshot - self._persisted_counts,
            observations=new_observations,
        )
        manifest = build_manifest(
            digest=self._digest(),
            num_layers=self.num_layers,
            num_units=self.num_units,
            total_observations=(
                (prior.total_observations if prior is not None else 0)
                + new_observations
            ),
            generations=(prior.generations if prior is not None else [])
            + [
                {
                    "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "source": "serving",
                    "observations": new_observations,
                }
            ],
        )
        write_atlas(self.sink, manifest, merged)
        self._persisted_atlas = Atlas(
            counts=merged,
            total_observations=manifest["total_observations"],
            generations=manifest["generations"],
            manifest=manifest,
        )
        self._persisted_counts = snapshot
        self.persisted_observations = self.observations
        return self.sink

    def close(self) -> None:
        """Persist the trace tail and the sink once; later calls do nothing.

        A repeated flush would append an empty generation to the sink.
        """
        if self._closed:
            return
        self._closed = True
        try:
            if self._trace is not None:
                try:
                    # The end row: the trace closed cleanly, and how many
                    # calls the record limit dropped.
                    dropped = min(self._trace_dropped_calls, 0xFFFFFFFF)
                    self._trace.write(struct.pack("<II", _TRACE_END, dropped))
                    self._trace.flush()
                finally:
                    self._trace.close()
                    self._trace = None
        finally:
            self.flush()

    def _digest(self) -> str:
        if self.model_path is None:
            return "0" * 64
        try:
            return index_digest(self.model_path)
        except Exception:  # noqa: BLE001 - never break serving for a profile
            return "0" * 64


def merge_counts(prior, observed, *, observations: int, half_life: int = DEFAULT_HALF_LIFE):
    """``new = old * 0.5 ** (observations / half_life) + observed``."""
    import numpy as np

    observed = np.asarray(observed, dtype=np.uint64)
    if prior is None:
        return observed.copy()
    prior = np.asarray(prior, dtype=np.float64)
    if prior.shape != observed.shape:
        return observed.copy()
    decay = 0.5 ** (float(max(observations, 0)) / float(max(half_life, 1)))
    faded = np.floor(prior * decay).astype(np.uint64)
    return faded + observed


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------


@dataclass
class Atlas:
    counts: object
    total_observations: int
    generations: List[dict]
    manifest: dict


def build_manifest(
    *,
    digest: str,
    num_layers: int,
    num_units: int,
    total_observations: int,
    generations: Sequence[dict],
    min_samples: int = DEFAULT_MIN_SAMPLES,
    half_life: int = DEFAULT_HALF_LIFE,
) -> dict:
    return {
        "format": ATLAS_FORMAT,
        "version": ATLAS_VERSION,
        "kind": ATLAS_KIND,
        "source_index_sha256": digest,
        "num_layers": int(num_layers),
        "num_units_per_layer": int(num_units),
        "counter_dtype": "u64",
        "data_offset": DATA_OFFSET,
        "total_observations": int(total_observations),
        "min_samples_for_trust": int(min_samples),
        "decay_half_life_observations": int(half_life),
        "generations": [dict(generation) for generation in generations][-16:],
        # Collection only.  A future iteration may pin; this one may not, and
        # the flag is recorded so a replayed trace says which regime produced
        # it rather than leaving a reader to guess.
        "pinning": False,
    }


def _paths(sink) -> Tuple[Path, Path]:
    sink = Path(sink).expanduser()
    if sink.suffix == ".json":
        return (sink, sink.with_suffix(".bin"))
    return (sink.with_suffix(".json"), sink.with_suffix(".bin"))


def write_atlas(sink, manifest: dict, counts) -> None:
    """tmp + fsync + ``os.replace`` -- never written in place."""
    import numpy as np

    counts = np.ascontiguousarray(np.asarray(counts, dtype=np.uint64))
    (manifest_path, data_path) = _paths(sink)
    payload = counts.tobytes()
    manifest = dict(manifest)
    manifest["crc32"] = zlib.crc32(payload) & 0xFFFFFFFF
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    body = (
        ATLAS_MAGIC
        + b"\0" * (DATA_OFFSET - len(ATLAS_MAGIC))
        + payload
        + ATLAS_SENTINEL
    )
    for (path, blob) in (
        (data_path, body),
        (manifest_path, json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")),
    ):
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as handle:
            handle.write(blob)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)


def load_atlas(sink, *, expect_digest=None, geometry=None) -> Optional[Atlas]:
    """Return the persisted atlas, or ``None`` for any reason at all.

    Stale, torn, truncated, malformed, geometry-mismatched or
    checkpoint-mismatched: all of them degrade to "no atlas".  The caller
    runs unpinned, which is the baseline path and must work on its own.
    """
    try:
        return _load_atlas(sink, expect_digest=expect_digest, geometry=geometry)
    except (TypeError, ValueError, OverflowError):
        # A manifest field of the wrong type: malformed, so no atlas.
        return None


def _load_atlas(sink, *, expect_digest, geometry) -> Optional[Atlas]:
    import numpy as np

    (manifest_path, data_path) = _paths(sink)
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(manifest, dict):
        return None
    if manifest.get("format") != ATLAS_FORMAT:
        return None
    if manifest.get("version") != ATLAS_VERSION:
        return None
    if manifest.get("kind") != ATLAS_KIND:
        return None
    if manifest.get("counter_dtype") != "u64":
        return None
    layers = int(manifest.get("num_layers") or 0)
    units = int(manifest.get("num_units_per_layer") or 0)
    if layers <= 0 or units <= 0:
        return None
    if geometry is not None and tuple(geometry) != (layers, units):
        return None
    if expect_digest is not None and manifest.get("source_index_sha256") != expect_digest:
        return None
    offset = int(manifest.get("data_offset") or 0)
    if offset != DATA_OFFSET:
        return None
    try:
        blob = data_path.read_bytes()
    except Exception:  # noqa: BLE001
        return None
    expected = DATA_OFFSET + layers * units * COUNTER_ITEMSIZE + len(ATLAS_SENTINEL)
    if len(blob) != expected:
        return None
    if not blob.startswith(ATLAS_MAGIC) or not blob.endswith(ATLAS_SENTINEL):
        return None
    payload = blob[DATA_OFFSET : DATA_OFFSET + layers * units * COUNTER_ITEMSIZE]
    if (zlib.crc32(payload) & 0xFFFFFFFF) != int(manifest.get("crc32", -1)):
        return None
    counts = np.frombuffer(payload, dtype=np.uint64).reshape(layers, units).copy()
    generations = manifest.get("generations") or []
    # The next checkpoint copies every generation as a dict: keep only those.
    if not isinstance(generations, list):
        generations = []
    return Atlas(
        counts=counts,
        total_observations=int(manifest.get("total_observations") or 0),
        generations=[dict(item) for item in generations if isinstance(item, dict)],
        manifest=manifest,
    )


# ---------------------------------------------------------------------------
# the counterfactual
# ---------------------------------------------------------------------------


def _read_rows(path) -> Tuple[int, int, int, object]:
    import numpy as np

    blob = Path(path).expanduser().read_bytes()
    if blob.startswith(TRACE_MAGIC):
        version = 2
    elif blob.startswith(TRACE_MAGIC_V1):
        version = 1
    else:
        raise ValueError("not an mlx2 expert access trace")
    (layers, units) = struct.unpack("<II", blob[8:16])
    records = np.frombuffer(blob[16:], dtype=np.uint32)
    usable = (records.size // 2) * 2
    return (version, layers, units, records[:usable].reshape(-1, 2))


def read_trace(path) -> Tuple[int, int, object]:
    """``(layers, units, rows)``: the ``(layer, id)`` row of every recorded
    access, in order (a v2 trace's marker rows removed)."""
    (version, layers, units, records) = _read_rows(path)
    if version == 2:
        records = records[records[:, 0] < _TRACE_END]
    return (layers, units, records)


def _trace_calls(records, layers: int) -> List[List[List[int]]]:
    """Infer a legacy v1 trace's ``observe()`` calls, per layer.

    v1 recorded no call boundaries.  Every call writes one layer's
    ``np.unique`` ids, strictly ascending, so a call is taken to end where the
    layer changes or the id stops increasing.  That is ambiguous whenever two
    consecutive recorded calls share a layer and ascend across the boundary:
    a single streamed projection, or calls the record limit dropped between
    them.  Its replay is therefore approximate.
    """
    import numpy as np

    calls: List[List[List[int]]] = [[] for _ in range(layers)]
    if len(records) == 0:
        return calls
    layer_ids = records[:, 0].astype(np.int64)
    units = records[:, 1].astype(np.int64)
    breaks = np.flatnonzero(
        (layer_ids[1:] != layer_ids[:-1]) | (units[1:] <= units[:-1])
    ) + 1
    starts = np.concatenate(([0], breaks))
    stops = np.concatenate((breaks, [len(records)]))
    for (start, stop) in zip(starts.tolist(), stops.tolist()):
        layer = int(layer_ids[start])
        if layer < layers:
            calls[layer].append(units[start:stop].tolist())
    return calls


def _trace_calls_v2(records, layers: int) -> Tuple[List[List[List[int]]], dict]:
    """A v2 trace's recorded calls, per layer, and what the trace covers."""
    calls: List[List[List[int]]] = [[] for _ in range(layers)]
    info = {"calls": 0, "closed": False, "dropped_calls": 0, "lost_calls": 0,
            "torn": False}
    (pos, rows) = (0, len(records))
    while pos < rows:
        if info["closed"]:
            raise ValueError("expert access trace has rows after its end row")
        (kind, value) = (int(records[pos, 0]), int(records[pos, 1]))
        if kind == _TRACE_CALL:
            stop = pos + 1 + value
            if stop > rows:
                # The process ended mid-write: keep the complete calls.
                info["torn"] = True
                break
            block = records[pos + 1 : stop]
            if value == 0 or (block[:, 0] != block[0, 0]).any() or block[0, 0] >= _TRACE_END:
                raise ValueError("malformed call in expert access trace")
            layer = int(block[0, 0])
            if layer < layers:
                calls[layer].append(block[:, 1].tolist())
            info["calls"] += 1
            pos = stop
        elif kind == _TRACE_LOST:
            info["lost_calls"] += 1
            pos += 1
        elif kind == _TRACE_END:
            info["closed"] = True
            info["dropped_calls"] = value
            pos += 1
        else:
            raise ValueError("expert access trace row outside a call")
    return (calls, info)


def load_trace_calls(path) -> Tuple[int, int, List[List[List[int]]], dict]:
    """``(layers, units, calls per layer, info)`` of a recorded trace.

    ``info["baseline"]`` says what replaying the calls with the engine's rule
    reproduces: ``"exact"`` -- a complete v2 trace (closed, nothing dropped or
    lost), the LRU that ran; ``"exact_prefix"`` -- a v2 trace cut short by its
    record limit or by the process ending, the LRU that ran up to its last
    recorded call; ``"approximate"`` -- a legacy v1 trace (calls inferred) or
    one with a failed acquire it could not record.
    """
    (version, layers, units, records) = _read_rows(path)
    if version == 1:
        calls = _trace_calls(records, int(layers))
        info = {"calls": sum(len(layer) for layer in calls), "closed": None,
                "dropped_calls": None, "lost_calls": None, "torn": None,
                "baseline": "approximate"}
    else:
        (calls, info) = _trace_calls_v2(records, int(layers))
        if info["lost_calls"]:
            info["baseline"] = "approximate"
        elif info["closed"] and not info["dropped_calls"]:
            info["baseline"] = "exact"
        else:
            info["baseline"] = "exact_prefix"
    info["version"] = version
    return (layers, units, calls, info)


def _lru_hit_rate(calls, capacity: int, pinned=frozenset()) -> Tuple[int, int]:
    """Replay one layer's calls.  Returns ``(hits, accesses)``.

    Each call is replayed as ``weight_stream.ExpertLRU.acquire`` serves it:
    the call's whole set is held, so a miss evicts only entries outside it;
    hits are refreshed in request order, misses inserted after them, and the
    cache is trimmed back to capacity (a call wider than it overflows).
    ``pinned`` occupies capacity permanently; the rest is LRU.  The actual
    engine always runs with ``pinned`` empty -- that is the whole point of the
    collect-only rule -- so a non-empty set here is strictly counterfactual.
    """
    from collections import OrderedDict

    lru_capacity = max(capacity - len(pinned), 0)
    resident: "OrderedDict[int, None]" = OrderedDict()
    hits = 0
    total = 0
    for call in calls:
        wanted = list(dict.fromkeys(int(unit) for unit in call))
        total += len(wanted)
        hits += sum(1 for unit in wanted if unit in pinned)
        wanted = [unit for unit in wanted if unit not in pinned]
        if not wanted or lru_capacity == 0:
            continue
        missing = []
        for unit in wanted:
            if unit in resident:
                hits += 1
                resident.move_to_end(unit)
            else:
                missing.append(unit)
        keep = set(wanted)
        while len(resident) + len(missing) > lru_capacity:
            victim = next((unit for unit in resident if unit not in keep), None)
            if victim is None:
                break
            del resident[victim]
        for unit in missing:
            resident[unit] = None
        while len(resident) > lru_capacity:
            resident.popitem(last=False)
    return (hits, total)


def replay_counterfactual(
    trace_path,
    *,
    capacity: int,
    atlas=None,
    pin_fractions: Iterable[float] = (0.0, 0.1, 0.2, 0.33, 0.5),
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> dict:
    """What an atlas-pinned resident set *would* have achieved, per layer.

    The ``pin_fraction = 0`` row replays the trace call by call with the
    engine's eviction rule.  ``baseline`` (see :func:`load_trace_calls`) says
    whether that is the LRU that actually ran (``"exact"``), the run up to the
    trace's last recorded call (``"exact_prefix"``), or an estimate
    (``"approximate"``).  Every other row is hypothetical: nothing in the
    serving path consults the atlas.
    """
    import numpy as np

    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
        raise ValueError("counterfactual capacity must be a positive integer")
    (layers, units, per_layer, trace_info) = load_trace_calls(trace_path)
    effective_capacity = min(capacity, int(units))
    counts = None
    if atlas is not None:
        counts = np.asarray(atlas.counts if isinstance(atlas, Atlas) else atlas)
        if counts.shape != (layers, units):
            counts = None

    # The actual plain-LRU policy is the comparison baseline even when a
    # caller requests only hypothetical pin fractions.
    fractions = [float(max(0.0, min(0.5, value))) for value in pin_fractions]
    if 0.0 not in fractions:
        fractions.insert(0, 0.0)
    fractions = list(dict.fromkeys(fractions))

    rows = []
    for fraction in fractions:
        pin_budget = int(effective_capacity * fraction)
        hits = 0
        total = 0
        layer_rows = []
        for (layer, calls) in enumerate(per_layer):
            pinned = frozenset()
            if counts is not None and pin_budget > 0:
                eligible = np.where(counts[layer] >= min_samples)[0]
                order = eligible[
                    np.argsort(counts[layer][eligible], kind="stable")[::-1]
                ]
                pinned = frozenset(int(unit) for unit in order[:pin_budget])
            (layer_hits, layer_total) = _lru_hit_rate(
                calls, effective_capacity, pinned
            )
            hits += layer_hits
            total += layer_total
            layer_rows.append(
                {
                    "layer": layer,
                    "accesses": layer_total,
                    "hits": layer_hits,
                    "hit_rate": (layer_hits / layer_total) if layer_total else 0.0,
                    "pinned_units": len(pinned),
                }
            )
        rows.append(
            {
                "pin_fraction": fraction,
                "pinned_per_layer": pin_budget,
                "accesses": total,
                "hits": hits,
                "hit_rate": (hits / total) if total else 0.0,
                "page_ins": total - hits,
                "layers": layer_rows,
            }
        )

    baseline = next((row for row in rows if row["pin_fraction"] == 0.0), rows[0])
    for row in rows:
        row["page_ins_vs_lru"] = row["page_ins"] - baseline["page_ins"]
        row["hit_rate_delta"] = row["hit_rate"] - baseline["hit_rate"]
    baseline_note = {
        "exact": "The pin_fraction 0 row reproduces the LRU that ran.",
        "exact_prefix": (
            "The trace stops before the run did (record limit or no clean "
            "close): the pin_fraction 0 row reproduces the LRU that ran up to "
            "the last recorded call."
        ),
        "approximate": (
            "The pin_fraction 0 row is approximate: a legacy trace without "
            "call boundaries, or a failed acquire the trace could not record."
        ),
    }[trace_info["baseline"]]
    return {
        "schema": "mlx2.expert-atlas-counterfactual.v1",
        "note": (
            "Counterfactual replay only. The serving path runs plain LRU; the "
            "atlas is collect-only and never influenced residency. Wall-clock "
            "numbers from a streamed run are not benchmarks. " + baseline_note
        ),
        "baseline": trace_info["baseline"],
        "trace_format": {
            key: trace_info[key]
            for key in ("version", "calls", "closed", "dropped_calls", "lost_calls", "torn")
        },
        "capacity_experts": effective_capacity,
        "requested_capacity_experts": capacity,
        "layers": layers,
        "units_per_layer": units,
        "actual_policy": "lru",
        "rows": rows,
        "verdict": _verdict(rows),
    }


def _verdict(rows) -> str:
    baseline = next((row for row in rows if row["pin_fraction"] == 0.0), None)
    if baseline is None or not baseline["accesses"]:
        return "inconclusive: no accesses replayed"
    best = max(rows, key=lambda row: row["hit_rate"])
    if best["pin_fraction"] == 0.0:
        return (
            "pinning would not have helped: plain LRU matched or beat every "
            "pinned fraction. Do not build pinning on this evidence."
        )
    gain = best["hit_rate"] - baseline["hit_rate"]
    if gain < 0.02:
        return (
            f"pinning would have gained {gain:.2%} hit rate at "
            f"pin_fraction={best['pin_fraction']:.2f} -- inside the noise of a "
            "single trace. Not enough to justify building pinning."
        )
    return (
        f"pinning at pin_fraction={best['pin_fraction']:.2f} would have raised "
        f"the hit rate by {gain:.2%} ({-best['page_ins_vs_lru']} fewer page-ins). "
        "That is a candidate, not a result: confirm on a second independent "
        "trace before building residency control."
    )
