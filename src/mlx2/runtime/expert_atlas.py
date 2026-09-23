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
TRACE_MAGIC = b"MLX2XTR1"
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
        # The atlas the last flush wrote: decayed counts, total and history.
        # A sink that has since gone missing is rebuilt from this rather than
        # from ``counts``, whose lifetime totals never decay.
        self._persisted_atlas = None
        self.observations = 0
        self.persisted_observations = 0
        self.checkpoint_every = int(checkpoint_every)
        self.sink = Path(sink).expanduser() if sink else None
        self.trace_path = Path(trace_path).expanduser() if trace_path else None
        self.trace_limit = int(trace_limit)
        self._trace = None
        self._trace_written = 0
        self._steps = 0
        self._dropped = 0

    def bind(self, *, num_layers: int, num_units: int) -> None:
        import numpy as np

        self.num_layers = int(num_layers)
        self.num_units = int(num_units)
        self.counts = np.zeros((self.num_layers, self.num_units), dtype=np.uint64)
        self._persisted_counts = np.zeros_like(self.counts)
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
            if self._trace_written + ids.size <= self.trace_limit:
                payload = np.empty((ids.size, 2), dtype=np.uint32)
                payload[:, 0] = layer
                payload[:, 1] = ids
                self._trace.write(payload.tobytes())
                self._trace_written += int(ids.size)
            else:
                self._dropped += int(ids.size)
        if (
            self.sink is not None
            and self.checkpoint_every > 0
            and self.observations - self.persisted_observations
            >= self.checkpoint_every
        ):
            self.flush()

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
            # Before any flush has succeeded nothing is persisted, and the
            # difference is every observation.
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
        if self._trace is not None:
            self._trace.flush()
            self._trace.close()
            self._trace = None
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

    Stale, torn, truncated, geometry-mismatched or checkpoint-mismatched: all
    of them degrade to "no atlas".  The caller runs unpinned, which is the
    baseline path and must work on its own.
    """
    import numpy as np

    (manifest_path, data_path) = _paths(sink)
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:  # noqa: BLE001
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
    return Atlas(
        counts=counts,
        total_observations=int(manifest.get("total_observations") or 0),
        generations=list(manifest.get("generations") or []),
        manifest=manifest,
    )


# ---------------------------------------------------------------------------
# the counterfactual
# ---------------------------------------------------------------------------


def read_trace(path) -> Tuple[int, int, object]:
    import numpy as np

    blob = Path(path).expanduser().read_bytes()
    if not blob.startswith(TRACE_MAGIC):
        raise ValueError("not an mlx2 expert access trace")
    (layers, units) = struct.unpack("<II", blob[8:16])
    records = np.frombuffer(blob[16:], dtype=np.uint32)
    usable = (records.size // 2) * 2
    return (layers, units, records[:usable].reshape(-1, 2))


def _lru_hit_rate(sequence, capacity: int, pinned=frozenset()) -> Tuple[int, int]:
    """Replay one layer's access sequence.  Returns ``(hits, accesses)``.

    ``pinned`` occupies capacity permanently; the rest is LRU.  The actual
    engine always runs with ``pinned`` empty -- that is the whole point of the
    collect-only rule -- so a non-empty set here is strictly counterfactual.
    """
    from collections import OrderedDict

    lru_capacity = max(capacity - len(pinned), 0)
    resident: "OrderedDict[int, None]" = OrderedDict()
    hits = 0
    total = 0
    for unit in sequence:
        unit = int(unit)
        total += 1
        if unit in pinned:
            hits += 1
            continue
        if unit in resident:
            hits += 1
            resident.move_to_end(unit)
            continue
        if lru_capacity == 0:
            continue
        while len(resident) >= lru_capacity:
            resident.popitem(last=False)
        resident[unit] = None
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

    The ``pin_fraction = 0`` row is the LRU that actually ran.  Every other row
    is hypothetical: nothing in the serving path consults the atlas.
    """
    import numpy as np

    (layers, units, records) = read_trace(trace_path)
    counts = None
    if atlas is not None:
        counts = np.asarray(atlas.counts if isinstance(atlas, Atlas) else atlas)
        if counts.shape != (layers, units):
            counts = None
    per_layer = [records[records[:, 0] == layer][:, 1] for layer in range(layers)]

    rows = []
    for fraction in pin_fractions:
        fraction = float(max(0.0, min(0.5, fraction)))
        pin_budget = int(capacity * fraction)
        hits = 0
        total = 0
        layer_rows = []
        for (layer, sequence) in enumerate(per_layer):
            pinned = frozenset()
            if counts is not None and pin_budget > 0:
                eligible = np.where(counts[layer] >= min_samples)[0]
                if eligible.size == 0:
                    # Nothing has enough observations behind it to be trusted;
                    # the honest counterfactual is "no pinning happened".
                    eligible = np.argsort(counts[layer])[::-1][:pin_budget]
                    eligible = eligible[counts[layer][eligible] > 0]
                order = eligible[np.argsort(counts[layer][eligible])[::-1]]
                pinned = frozenset(int(unit) for unit in order[:pin_budget])
            (layer_hits, layer_total) = _lru_hit_rate(sequence, capacity, pinned)
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
    return {
        "schema": "mlx2.expert-atlas-counterfactual.v1",
        "note": (
            "Counterfactual replay only. The serving path runs plain LRU; the "
            "atlas is collect-only and never influenced residency. Wall-clock "
            "numbers from a streamed run are not benchmarks."
        ),
        "capacity_experts": int(capacity),
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
