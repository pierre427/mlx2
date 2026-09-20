"""Disk streaming of MoE expert weights.

Experts of a routed MoE layer are contiguous rows of a stacked ``[E, ...]``
tensor inside the model's own safetensors shards.  This module reads one
expert's rows by byte range on demand, keeps a bounded per-layer LRU of the
experts a run actually touches, and remaps ``rhs_indices`` so the untouched
``mx.gather_qmm`` kernel computes on the resident subset.

Deliberate non-goals, each measured or argued in the design note:

* **No cross-layer speculative prefetch.**  Adjacent-layer expert overlap sits
  at chance on three architectures and an end-to-end A/B gave -2.9 % decode.
  Fetch is reactive: a miss is resolved when the router names it.
* **No mmap demand paging.**  Metal wires the whole buffer on first kernel use,
  so a 2.6 MiB expert would fault its entire ~1.2 GiB stacked tensor.
* **No pinned hot set.**  The atlas in :mod:`mlx2.runtime.expert_atlas` collects
  access counts but must not influence residency in this iteration; the
  offline counterfactual replay is the gate for ever building pinning.

Correctness invariant, and the reason this is safe at all: paging changes only
*where* a byte sits, never its value.  The same quantized rows reach the same
kernel with the same ``group_size``/``bits``/``mode``; only the index vector
differs, and the remap is order preserving so a sorted gather stays sorted.
Streamed and fully-resident execution must therefore be bit-identical, and
``tests/test_moe_expert_streaming.py`` proves it rather than assuming it.

Any throughput measured on a streamed configuration is NOT a benchmark.
"""

from __future__ import annotations

import gc
import json
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Concurrency sweet spot measured on the M3 Pro's internal NVMe: random
# ~900 KiB reads run 2.14 GB/s at one thread and 7.46 GB/s at sixteen.  Never
# issue single-threaded small reads; never multi-thread one sequential stream.
DEFAULT_READ_WORKERS = 16

_SAFETENSORS_DTYPES = {
    "BOOL": ("u1", "bool_"),
    "U8": ("u1", "uint8"),
    "I8": ("i1", "int8"),
    "U16": ("u2", "uint16"),
    "I16": ("i2", "int16"),
    "F16": ("f2", "float16"),
    "BF16": ("u2", "bfloat16"),
    "U32": ("u4", "uint32"),
    "I32": ("i4", "int32"),
    "F32": ("f4", "float32"),
    "U64": ("u8", "uint64"),
    "I64": ("i8", "int64"),
    "F64": ("f8", "float64"),
}


class StreamingUnavailable(RuntimeError):
    """Streaming cannot be installed for this model; run fully resident."""


class WorkingSetTooSmall(RuntimeError):
    """The configured ceiling cannot hold one step's experts.

    Refusing with a printed budget beats a run that thrashes for six hours and
    then fails; Splash's ``WorkingSetTooSmall`` is the precedent.
    """


# ---------------------------------------------------------------------------
# safetensors byte-range addressing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TensorLocation:
    path: Path
    dtype: str
    shape: Tuple[int, ...]
    begin: int
    end: int

    @property
    def nbytes(self) -> int:
        return self.end - self.begin

    def row_bytes(self) -> int:
        if not self.shape or self.shape[0] <= 0:
            raise StreamingUnavailable(f"not a stacked tensor: {self.shape}")
        nbytes = self.nbytes
        rows = int(self.shape[0])
        if nbytes % rows:
            raise StreamingUnavailable("stacked tensor rows are not byte aligned")
        return nbytes // rows

    def row_range(self, row: int) -> Tuple[int, int]:
        stride = self.row_bytes()
        start = self.begin + row * stride
        return (start, stride)


def read_safetensors_header(path: Path) -> Dict[str, dict]:
    with open(path, "rb") as handle:
        raw = handle.read(8)
        if len(raw) != 8:
            raise StreamingUnavailable(f"truncated safetensors header: {path}")
        length = int.from_bytes(raw, "little")
        header = json.loads(handle.read(length).decode("utf-8"))
    header.pop("__metadata__", None)
    return {"__data_offset__": 8 + length, **header}


class SafetensorsIndex:
    """Name -> byte range across every shard of a checkpoint."""

    def __init__(self, tensors: Dict[str, TensorLocation]):
        self._tensors = tensors

    @classmethod
    def from_model_path(cls, model_path) -> "SafetensorsIndex":
        root = Path(model_path).expanduser()
        shards = sorted(root.glob("*.safetensors"))
        if not shards:
            raise StreamingUnavailable(f"no safetensors shards under {root}")
        tensors: Dict[str, TensorLocation] = {}
        for shard in shards:
            header = read_safetensors_header(shard)
            base = header.pop("__data_offset__")
            for name, spec in header.items():
                offsets = spec.get("data_offsets") or [0, 0]
                tensors[name] = TensorLocation(
                    path=shard,
                    dtype=str(spec.get("dtype")),
                    shape=tuple(int(dim) for dim in spec.get("shape") or ()),
                    begin=base + int(offsets[0]),
                    end=base + int(offsets[1]),
                )
        return cls(tensors)

    def __contains__(self, name: str) -> bool:
        return name in self._tensors

    def __getitem__(self, name: str) -> TensorLocation:
        return self._tensors[name]

    def get(self, name: str) -> Optional[TensorLocation]:
        return self._tensors.get(name)

    def names(self) -> Iterable[str]:
        return self._tensors.keys()


class _FileHandles:
    """Per-thread read-only descriptors, re-opened after a fork.

    A forked child inherits the dict and the descriptors; keying on the owning
    pid makes the child open its own rather than share a file offset (``pread``
    is positional, but a shared fd across a fork is still a footgun worth
    removing).
    """

    def __init__(self):
        self._local = threading.local()
        self._owner = os.getpid()

    def get(self, path: Path) -> int:
        if os.getpid() != self._owner:
            self._local = threading.local()
            self._owner = os.getpid()
        table = getattr(self._local, "table", None)
        if table is None:
            table = {}
            self._local.table = table
        fd = table.get(path)
        if fd is None:
            fd = os.open(str(path), os.O_RDONLY)
            table[path] = fd
        return fd

    def pread(self, path: Path, offset: int, length: int) -> bytes:
        fd = self.get(path)
        chunks = []
        remaining = length
        while remaining > 0:
            chunk = os.pread(fd, remaining, offset)
            if not chunk:
                raise StreamingUnavailable(f"short read on {path} at {offset}")
            chunks.append(chunk)
            offset += len(chunk)
            remaining -= len(chunk)
        return chunks[0] if len(chunks) == 1 else b"".join(chunks)


# ---------------------------------------------------------------------------
# per-expert slice reader
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpertSliceSpec:
    """Where one projection's experts live, and how to rebuild one.

    ``parts`` is normally a single stacked checkpoint tensor.  It holds more
    than one when the live module's projection is a load-time *fusion* of
    several checkpoint tensors -- Qwen3-Next-family adapters concatenate
    ``gate_proj`` and ``up_proj`` into ``gate_up_proj`` along the output axis,
    and that lever defaults to on.  Fusing one expert is still pure byte-range
    addressing: read each part's row for that expert and concatenate them in
    the same order, which reproduces the fused row exactly.
    """

    key: str
    parts: Tuple[TensorLocation, ...]
    concat_axis: int = 0

    @classmethod
    def single(cls, key: str, location: TensorLocation) -> "ExpertSliceSpec":
        return cls(key=key, parts=(location,))

    @property
    def location(self) -> TensorLocation:
        return self.parts[0]

    @property
    def num_experts(self) -> int:
        return int(self.parts[0].shape[0])

    def row_bytes(self) -> int:
        return sum(part.row_bytes() for part in self.parts)

    @property
    def expert_shape(self) -> Tuple[int, ...]:
        return tuple(self.location.shape[1:])


class ExpertSliceReader:
    """Byte-range reads of one MoE projection's expert rows.

    Disk I/O is split from ``mx.array`` construction on purpose: MLX arrays
    must be built on the thread that runs the model, so the pool returns plain
    bytes and :meth:`materialize` runs on the caller's thread.
    """

    def __init__(
        self,
        specs: Sequence[ExpertSliceSpec],
        *,
        handles: _FileHandles,
        pool: Optional[ThreadPoolExecutor] = None,
    ):
        if not specs:
            raise StreamingUnavailable("expert reader needs at least one tensor")
        self.specs = tuple(specs)
        self._handles = handles
        self._pool = pool
        counts = {part.shape[0] for spec in self.specs for part in spec.parts}
        if len(counts) != 1:
            raise StreamingUnavailable("projection tensors disagree on expert count")
        self.num_experts = int(counts.pop())
        self.expert_bytes = sum(spec.row_bytes() for spec in self.specs)

    def read_bytes(self, expert: int) -> Tuple[Tuple[bytes, ...], ...]:
        """Raw rows for ``expert``.  Safe to run off the model thread."""
        out = []
        for spec in self.specs:
            blobs = []
            for part in spec.parts:
                (offset, length) = part.row_range(expert)
                blobs.append(self._handles.pread(part.path, offset, length))
            out.append(tuple(blobs))
        return tuple(out)

    def read_many(self, experts: Sequence[int]) -> Dict[int, Tuple[bytes, ...]]:
        if self._pool is None or len(experts) <= 1:
            return {expert: self.read_bytes(expert) for expert in experts}
        futures = {
            expert: self._pool.submit(self.read_bytes, expert) for expert in experts
        }
        return {expert: future.result() for (expert, future) in futures.items()}

    def materialize(self, payload: Sequence[bytes]) -> Tuple["object", ...]:
        """Build the expert's ``mx.array`` rows.  Model thread only."""
        import mlx.core as mx
        import numpy as np

        arrays = []
        for (spec, blobs) in zip(self.specs, payload):
            pieces = []
            for (part, blob) in zip(spec.parts, blobs):
                dtype = _SAFETENSORS_DTYPES.get(part.dtype)
                if dtype is None:
                    raise StreamingUnavailable(
                        f"unsupported safetensors dtype {part.dtype}"
                    )
                (numpy_code, mlx_name) = dtype
                flat = np.frombuffer(blob, dtype=np.dtype(numpy_code))
                array = mx.array(flat)
                target = getattr(mx, mlx_name)
                if array.dtype != target:
                    array = array.view(target)
                pieces.append(array.reshape(tuple(part.shape[1:])))
            arrays.append(
                pieces[0]
                if len(pieces) == 1
                else mx.concatenate(pieces, axis=spec.concat_axis)
            )
        return tuple(arrays)


# ---------------------------------------------------------------------------
# the cache
# ---------------------------------------------------------------------------


@dataclass
class StreamStats:
    page_ins: int = 0
    page_in_bytes: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    resident_bytes: int = 0
    refusals: int = 0
    overflows: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "stream_page_ins_total": self.page_ins,
            "stream_page_in_bytes_total": self.page_in_bytes,
            "stream_expert_hits_total": self.hits,
            "stream_expert_misses_total": self.misses,
            "stream_evictions_total": self.evictions,
            "stream_resident_bytes": self.resident_bytes,
            "stream_admission_refusals_total": self.refusals,
            "stream_working_set_overflows_total": self.overflows,
        }


class ExpertLRU:
    """Bounded, dict-keyed LRU over one layer's experts.

    Dict-keyed, **not** a fixed slot array.  Every index in a forward pass is
    resolved before the gather runs, so an eviction triggered later in the same
    pass would otherwise clobber a slot an earlier expert already resolved to.
    Keying on the expert id and holding the pass's working set removes that
    class of bug entirely.
    """

    def __init__(self, reader: ExpertSliceReader, *, capacity_experts: int, stats: StreamStats):
        if capacity_experts < 1:
            raise WorkingSetTooSmall("expert cache capacity must be at least 1")
        self.reader = reader
        self.capacity = int(min(capacity_experts, reader.num_experts))
        self.stats = stats
        self._entries: "OrderedDict[int, Tuple[object, ...]]" = OrderedDict()

    @property
    def resident(self) -> int:
        return len(self._entries)

    def acquire(self, experts: Sequence[int]) -> Dict[int, Tuple["object", ...]]:
        """Resident rows for every id in ``experts``, fetching what is missing.

        The whole request set is held for the duration of the call: eviction
        only ever considers entries outside it.
        """
        wanted = list(dict.fromkeys(int(expert) for expert in experts))
        # A wide batch can route to more experts in one call than the steady
        # ceiling holds.  Every index must resolve before the gather runs, so
        # the only correct answers are "fetch them all" or "fail the request".
        # We fetch, count the overshoot, and trim back to the ceiling before
        # returning: the caller keeps the rows it needs alive through the
        # returned mapping, so the transient peak is one call's working set
        # and steady-state residency still respects the ceiling.
        overflow = len(wanted) > self.capacity
        if overflow:
            self.stats.overflows += 1
        missing = []
        for expert in wanted:
            if expert in self._entries:
                self._entries.move_to_end(expert)
                self.stats.hits += 1
            else:
                self.stats.misses += 1
                missing.append(expert)
        if missing:
            if not overflow:
                self._evict_for(len(missing), keep=set(wanted))
            payloads = self.reader.read_many(missing)
            for expert in missing:
                # Array construction stays on this thread by contract.
                self._entries[expert] = self.reader.materialize(payloads[expert])
                self.stats.page_ins += 1
                self.stats.page_in_bytes += self.reader.expert_bytes
        resolved = {expert: self._entries[expert] for expert in wanted}
        if overflow:
            self._trim_to_capacity()
        self.stats.resident_bytes = self.resident * self.reader.expert_bytes
        return resolved

    def _evict_for(self, incoming: int, *, keep) -> None:
        while self.resident + incoming > self.capacity:
            victim = None
            for expert in self._entries:
                if expert not in keep:
                    victim = expert
                    break
            if victim is None:  # pragma: no cover - guarded by the overflow path
                break
            self._entries.pop(victim)
            self.stats.evictions += 1

    def _trim_to_capacity(self) -> None:
        while self.resident > self.capacity:
            self._entries.popitem(last=False)
            self.stats.evictions += 1

    def seed(self, experts: Sequence[int]) -> None:
        """Warm the cache through the fetch path.

        Never by slicing an already-loaded stacked weight: a prefix slice keeps
        the whole parent buffer alive and frees nothing.
        """
        for expert in experts:
            self.acquire([expert])


# ---------------------------------------------------------------------------
# the streamed module
# ---------------------------------------------------------------------------


_STREAMED_CLASS = None


def _streamed_class():
    """Build the streamed module class once, at module scope.

    Deliberately NOT a closure over the module being replaced.  A class
    defined per call captures the original ``QuantizedSwitchLinear`` in a
    closure cell, which pins its stacked weight/scales/biases for the life of
    the process -- so installing streaming would free nothing at all.  That
    is not hypothetical: it was measured on an M3, where active memory after
    install stayed at 19.68 GiB instead of dropping by the table.
    """
    global _STREAMED_CLASS
    if _STREAMED_CLASS is not None:
        return _STREAMED_CLASS

    import mlx.core as mx
    import numpy as np

    from .models.switch_layers import (
        QuantizedSwitchLinear,
        _quantized_gather_tail_policy,
    )

    class StreamedQuantizedSwitchLinear(QuantizedSwitchLinear):
        def __init__(self, *, group_size, bits, mode, bias, input_dims,
                     output_dims, reader, cache, layer_index, collector):
            # Bypass the base class' random-init constructor entirely.
            object.__setattr__(self, "_no_grad", set())
            object.__setattr__(self, "_training", False)
            dict.__init__(self)
            self.group_size = group_size
            self.bits = bits
            self.mode = mode
            if bias is not None:
                self["bias"] = bias
            self._stream_reader = reader
            self._stream_cache = cache
            self._stream_layer = layer_index
            self._stream_collector = collector
            self._stream_num_experts = reader.num_experts
            self._stream_input_dims = int(input_dims)
            self._stream_output_dims = int(output_dims)
            self.freeze(recurse=False)

        @property
        def input_dims(self):
            return self._stream_input_dims

        @property
        def output_dims(self):
            return self._stream_output_dims

        @property
        def num_experts(self):
            return self._stream_num_experts

        def __call__(self, x, indices, sorted_indices=False):
            host = np.asarray(indices, dtype=np.int64)
            unique = np.unique(host)  # ascending: the remap stays monotonic,
            # so a sorted gather is still sorted afterwards.
            entries = self._stream_cache.acquire([int(e) for e in unique])
            if self._stream_collector is not None:
                self._stream_collector.observe(self._stream_layer, unique)
            table = np.zeros(self._stream_num_experts, dtype=np.uint32)
            table[unique] = np.arange(unique.size, dtype=np.uint32)
            remapped = mx.array(table[host].reshape(host.shape))
            rows = [entries[int(e)] for e in unique]
            weight = mx.stack([row[0] for row in rows])
            scales = mx.stack([row[1] for row in rows])
            biases = mx.stack([row[2] for row in rows]) if len(rows[0]) > 2 else None
            tail_policy = _quantized_gather_tail_policy(
                self.mode, int(x.shape[-1]), sorted_indices
            )
            if tail_policy == "dense":
                quantized = [weight, scales]
                if biases is not None:
                    quantized.append(biases)
                dense = mx.dequantize(
                    *quantized,
                    group_size=self.group_size,
                    bits=self.bits,
                    mode=self.mode,
                )
                out = mx.gather_mm(
                    x,
                    dense.swapaxes(-1, -2),
                    rhs_indices=remapped,
                    sorted_indices=sorted_indices,
                )
            else:
                out = mx.gather_qmm(
                    x,
                    weight,
                    scales,
                    biases,
                    rhs_indices=remapped,
                    transpose=True,
                    group_size=self.group_size,
                    bits=self.bits,
                    mode=self.mode,
                    sorted_indices=(
                        False if tail_policy == "unsorted" else sorted_indices
                    ),
                )
            if "bias" in self:
                out = out + mx.expand_dims(self["bias"][indices], -2)
            return out

    _STREAMED_CLASS = StreamedQuantizedSwitchLinear
    return _STREAMED_CLASS


def _streamed_switch_linear(base, reader, cache, *, layer_index, unit_offset, collector):
    """Return a drop-in replacement for one ``QuantizedSwitchLinear``.

    Everything the replacement needs is copied out of ``base`` here; no
    reference to ``base`` survives this call, so its stacked tensors become
    collectable the moment the parent drops it.
    """
    return _streamed_class()(
        group_size=base.group_size,
        bits=base.bits,
        mode=base.mode,
        bias=base["bias"] if "bias" in base else None,
        input_dims=base.input_dims,
        output_dims=base.output_dims,
        reader=reader,
        cache=cache,
        layer_index=layer_index,
        collector=collector,
    )


# ---------------------------------------------------------------------------
# sizing
# ---------------------------------------------------------------------------

# Resident fraction of the expert table, as a function of table/RAM ratio.
# Community measurement, reproduced in the design note: decode throughput
# peaks and then collapses as the cache squeezes the kernel page cache
# (60 GB -> 35 GB was +65 % decode on one 128 GB box).  Bigger is not better,
# so the budget is sized from this curve rather than from "as much as fits".
_RESIDENT_FRACTION_CURVE = ((1.05, 0.45), (1.8, 0.30), (2.6, 0.20))


def resident_fraction(table_bytes: int, budget_bytes: int) -> float:
    """Fraction of the expert table worth holding resident."""
    if budget_bytes <= 0 or table_bytes <= 0:
        return 0.0
    ratio = table_bytes / float(budget_bytes)
    points = _RESIDENT_FRACTION_CURVE
    if ratio <= 1.0:
        # The whole table fits inside the budget: there is nothing to stream
        # and no page cache to squeeze.  The curve starts at the first ratio
        # where the table does *not* fit (1.05 -> 0.45), and the step between
        # the two is real, not an artefact: it is the measured collapse.
        return 1.0
    if ratio <= points[0][0]:
        return points[0][1]
    if ratio >= points[-1][0]:
        # Extrapolating a collapse curve is how you get a made-up constant;
        # clamp to the last measured point instead.
        return points[-1][1]
    for ((x0, y0), (x1, y1)) in zip(points, points[1:]):
        if x0 <= ratio <= x1:
            span = x1 - x0
            return y0 + (y1 - y0) * ((ratio - x0) / span)
    return points[-1][1]


def advisory_budget_bytes() -> int:
    """Metal's recommended working set, not physical RAM."""
    try:
        import mlx.core as mx

        return int(mx.device_info().get("max_recommended_working_set_size", 0) or 0)
    except Exception:  # noqa: BLE001 - a sizing probe must not break startup
        return 0


def plan_cache_experts(
    *,
    table_bytes: int,
    expert_bytes: int,
    num_layers: int,
    top_k: int,
    ceiling_bytes: int,
) -> int:
    """Per-layer expert capacity for an enforced ``ceiling_bytes``."""
    if expert_bytes <= 0 or num_layers <= 0:
        raise WorkingSetTooSmall("cannot size a cache for an empty model")
    floor_bytes = num_layers * max(top_k, 1) * expert_bytes
    if ceiling_bytes < floor_bytes:
        raise WorkingSetTooSmall(
            f"expert cache ceiling {ceiling_bytes} B is below the floor of one "
            f"step's experts ({floor_bytes} B = {num_layers} layers x {top_k} "
            f"experts x {expert_bytes} B)"
        )
    fraction = resident_fraction(table_bytes, ceiling_bytes)
    by_curve = int((table_bytes * fraction) // (num_layers * expert_bytes))
    by_ceiling = int(ceiling_bytes // (num_layers * expert_bytes))
    return max(max(top_k, 1), min(by_curve, by_ceiling))


# ---------------------------------------------------------------------------
# the manager
# ---------------------------------------------------------------------------


@dataclass
class ExpertStreamPlan:
    layers: int = 0
    experts_per_layer: int = 0
    expert_bytes: int = 0
    table_bytes: int = 0
    capacity_experts: int = 0
    ceiling_bytes: int = 0
    projections: Tuple[str, ...] = ()

    def as_dict(self) -> Dict[str, object]:
        return {
            "layers": self.layers,
            "experts_per_layer": self.experts_per_layer,
            "expert_bytes": self.expert_bytes,
            "table_bytes": self.table_bytes,
            "capacity_experts": self.capacity_experts,
            "ceiling_bytes": self.ceiling_bytes,
            "resident_ceiling_bytes": self.capacity_experts
            * self.layers
            * self.expert_bytes,
            "projections": list(self.projections),
        }


class ExpertStreamManager:
    """Installs and owns per-layer expert caches for one model."""

    def __init__(
        self,
        *,
        plan: ExpertStreamPlan,
        stats: StreamStats,
        pool: Optional[ThreadPoolExecutor],
        caches: Dict[str, ExpertLRU],
        collector=None,
    ):
        self.plan = plan
        self.stats = stats
        self.caches = caches
        self.collector = collector
        self._pool = pool

    def counters(self) -> Dict[str, int]:
        self.stats.resident_bytes = sum(
            cache.resident * cache.reader.expert_bytes for cache in self.caches.values()
        )
        counters = self.stats.as_dict()
        if self.collector is not None:
            counters.update(self.collector.counters())
        return counters

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None


# Load-time fusions this addressing layer can reassemble from checkpoint
# rows.  The fused projection is a concatenation along the output axis, so
# one fused expert row is the concatenation of each part's row for that same
# expert, in this order.
_FUSED_PROJECTIONS = {"gate_up_proj": ("gate_proj", "up_proj")}


def _resolve_spec(index: SafetensorsIndex, path: str, key: str):
    """Byte ranges backing ``<path>.<key>``, direct or reassembled."""
    location = index.get(f"{path}.{key}")
    if location is not None:
        return ExpertSliceSpec.single(key, location)
    (prefix, _, leaf) = path.rpartition(".")
    parts = _FUSED_PROJECTIONS.get(leaf)
    if not parts:
        return None
    located = [index.get(f"{prefix}.{part}.{key}") for part in parts]
    if any(part is None for part in located):
        return None
    # Concatenation is along the output axis (-2 of the stacked tensor,
    # which is axis 0 once the leading expert axis is dropped).
    return ExpertSliceSpec(key=key, parts=tuple(located), concat_axis=0)


def _module_parent(model, path: str):
    parts = path.split(".")
    node = model
    for part in parts[:-1]:
        node = node[int(part)] if isinstance(node, list) else node[part]
    return (node, parts[-1])


def install_expert_streaming(
    model,
    model_path,
    *,
    ceiling_bytes: int,
    top_k: int = 1,
    read_workers: int = DEFAULT_READ_WORKERS,
    collector=None,
) -> ExpertStreamManager:
    """Replace every streamable ``QuantizedSwitchLinear`` with a streamed one.

    Raises :class:`StreamingUnavailable` when the checkpoint's tensor names do
    not match the live module tree (an adapter that fuses or renames expert
    tensors during ``sanitize`` cannot be addressed by byte range), and
    :class:`WorkingSetTooSmall` when the ceiling cannot hold one step.
    """
    import mlx.core as mx

    from .models.switch_layers import QuantizedSwitchLinear

    index = SafetensorsIndex.from_model_path(model_path)
    handles = _FileHandles()

    targets: List[Tuple[str, object, List[ExpertSliceSpec]]] = []
    for (path, module) in model.named_modules():
        if not isinstance(module, QuantizedSwitchLinear):
            continue
        specs = []
        for key in ("weight", "scales", "biases"):
            if module.get(key) is None:
                continue
            spec = _resolve_spec(index, path, key)
            if spec is None:
                specs = []
                break
            specs.append(spec)
        if not specs:
            raise StreamingUnavailable(
                f"checkpoint has no byte-addressable expert rows for {path!r}; "
                "this model's expert tensors are renamed, or fused from parts "
                "this addressing layer does not know how to reassemble"
            )
        expected = int(module.num_experts)
        for spec in specs:
            if spec.num_experts != expected:
                raise StreamingUnavailable(
                    f"{path!r} has {expected} experts but the checkpoint rows "
                    f"for {spec.key} have {spec.num_experts}; a load-time "
                    "transform (for example folding the shared expert into "
                    "the gather) has changed the table and byte ranges no "
                    "longer address it"
                )
        targets.append((path, module, specs))

    if not targets:
        raise StreamingUnavailable("model has no quantized routed experts")

    expert_bytes = max(
        sum(spec.row_bytes() for spec in specs) for (_, _, specs) in targets
    )
    experts_per_layer = max(specs[0].num_experts for (_, _, specs) in targets)
    table_bytes = sum(
        specs[0].num_experts * sum(spec.row_bytes() for spec in specs)
        for (_, _, specs) in targets
    )
    capacity = plan_cache_experts(
        table_bytes=table_bytes,
        expert_bytes=expert_bytes,
        num_layers=len(targets),
        top_k=top_k,
        ceiling_bytes=ceiling_bytes,
    )

    pool = ThreadPoolExecutor(
        max_workers=max(1, int(read_workers)), thread_name_prefix="expert-stream"
    )
    stats = StreamStats()
    caches: Dict[str, ExpertLRU] = {}
    for (layer_index, (path, module, specs)) in enumerate(targets):
        reader = ExpertSliceReader(specs, handles=handles, pool=pool)
        cache = ExpertLRU(reader, capacity_experts=capacity, stats=stats)
        caches[path] = cache
        replacement = _streamed_switch_linear(
            module,
            reader,
            cache,
            layer_index=layer_index,
            unit_offset=0,
            collector=collector,
        )
        (parent, leaf) = _module_parent(model, path)
        if isinstance(parent, list):
            parent[int(leaf)] = replacement
        else:
            parent[leaf] = replacement
    projections = tuple(spec.key for spec in targets[0][2])
    # Drop the stacked tables; the streamed modules own their bytes now.
    # The replaced modules sit in reference cycles (nn.Module holds its
    # parameters in a dict that the parent module also references), so
    # refcounting alone does not release them and the tables stay resident
    # for an arbitrary time.  Collect explicitly before clearing, or the
    # whole point of installing -- dropping the table -- silently does not
    # happen.  Measured on an M3: without this, active memory after install
    # was unchanged at 19.68 GiB.
    del targets
    gc.collect()
    mx.clear_cache()

    plan = ExpertStreamPlan(
        layers=len(caches),
        experts_per_layer=experts_per_layer,
        expert_bytes=expert_bytes,
        table_bytes=table_bytes,
        capacity_experts=capacity,
        ceiling_bytes=int(ceiling_bytes),
        projections=projections,
    )
    if collector is not None:
        collector.bind(num_layers=plan.layers, num_units=plan.experts_per_layer)
    return ExpertStreamManager(
        plan=plan, stats=stats, pool=pool, caches=caches, collector=collector
    )
