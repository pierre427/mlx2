"""Early-load weight streaming: replace streamable modules before materializing.

The ordinary adapter loader (:func:`mlx2.runtime.ubc_evict.load_shards_evicting`)
evaluates every shard before ``sanitize`` runs, so streaming installed after
it bounds only steady-state residency.  :func:`load_streamed` instead:

1. binds each indexed shard (descriptor, ``fstat`` pin, header) once;
2. ``mx.load``s every shard *lazily* and records each raw array's origin by
   object identity;
3. runs the adapter's own ``sanitize`` with a concat ledger active, then its
   quantization and a strict ``load_weights`` (names and shapes checked, no
   evaluation);
4. resolves every streamable tensor to proven checkpoint bytes and installs
   all replacements;
5. refuses before evaluation when the retained remainder plus the streaming
   reservation plus a lane's headroom exceeds the admission limit;
6. only then evaluates the retained remainder and verifies the paths still
   name the bound files.

The adapter declares support with a class attribute read from the class's own
``__dict__`` (:func:`declared_stream_modes`), so subclasses never inherit it.
The model-specific choices -- sanitize, quantization predicate, which dense
projections stream, the routed ``top_k`` -- stay in the adapter.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence

from .weight_stream import (
    DEFAULT_READ_WORKERS,
    STREAM_MODES,
    BoundShards,
    StreamingUnavailable,
    TensorOrigins,
    WorkingSetTooSmall,
    concat_ledger,
    install_dense_streaming,
    install_expert_streaming,
    is_mtp_path,
)

DECLARATION = "weight_streaming_modes"
# A lane needs at least this much beside the weights and the streaming
# reservation before the pre-load budget admits a streamed model.  A floor,
# not a measurement: real lanes are costed by admission after load.
MIN_LANE_HEADROOM_BYTES = 1 << 30


@dataclass(frozen=True, eq=False)
class WeightStreamRequest:
    """What the engine asks a declaring adapter to stream, before it loads."""

    mode: str
    budget_bytes: int
    read_workers: int = DEFAULT_READ_WORKERS
    mtp_resident: bool = False
    collector: object = None
    admission_limit_bytes: Optional[int] = None
    headroom_bytes: int = MIN_LANE_HEADROOM_BYTES

    def __post_init__(self):
        if self.mode not in STREAM_MODES:
            raise ValueError(f"unknown weight streaming mode {self.mode!r}")
        for (name, value, low) in (
            ("budget_bytes", self.budget_bytes, 1),
            ("read_workers", self.read_workers, 1),
            ("headroom_bytes", self.headroom_bytes, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < low:
                raise ValueError(f"weight streaming {name} must be an integer >= {low}")
        if type(self.mtp_resident) is not bool:
            raise ValueError("weight streaming mtp_resident must be boolean")
        limit = self.admission_limit_bytes
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 0
        ):
            raise ValueError("weight streaming admission_limit_bytes must be a non-negative int")


def declared_stream_modes(adapter_class) -> frozenset:
    """Modes the class itself declares -- never a parent's declaration."""
    try:
        own = vars(adapter_class)
    except TypeError:
        return frozenset()
    value = own.get(DECLARATION, ())
    return frozenset(value) & frozenset(STREAM_MODES)


def require_declared(adapter_class, request: Optional[WeightStreamRequest]):
    """Validate a request against the adapter's own declaration, pre-load."""
    if request is None:
        return None
    if not isinstance(request, WeightStreamRequest):
        raise ValueError("weight_streaming must be a WeightStreamRequest")
    if request.mode not in declared_stream_modes(adapter_class):
        raise ValueError(
            f"{adapter_class.__name__} does not declare {request.mode} weight "
            "streaming; refusing before any weight is loaded"
        )
    return request


def _tree_nbytes(tree) -> int:
    from mlx.utils import tree_flatten

    return sum(int(value.nbytes) for (_, value) in tree_flatten(tree))


@dataclass
class StreamedLoad:
    weights: Dict[str, object]
    manager: object
    remainder_bytes: int


def load_streamed(
    model,
    root,
    names: Sequence[str],
    *,
    request: WeightStreamRequest,
    sanitize: Callable[[dict], dict],
    quantize: Callable[[dict], None],
    records=None,
    weight_map=None,
    shard_prune: Optional[Callable[[dict], dict]] = None,
    top_k: int = 1,
    dense_targets: Optional[Callable[[object], Sequence[str]]] = None,
    on_installed: Optional[Callable[[object], None]] = None,
) -> StreamedLoad:
    """Build, stream and materialize ``model`` without evaluating streamed tables.

    ``quantize(weights)`` applies the adapter's own ``nn.quantize`` predicate;
    ``dense_targets(model)`` returns module paths for ``dense_mlp``.
    ``on_installed(manager)`` runs after the swap and before evaluation (the
    adapter forces stock arithmetic there).  Every failure closes what was
    opened; on success the returned manager owns the bound descriptors.
    """
    import mlx.core as mx

    from .ubc_evict import ubc_evict_paths

    request = require_declared_mode(request)
    shards = BoundShards(root, names, records=records, weight_map=weight_map)
    manager = None
    try:
        origins = TensorOrigins(
            shards.index, weight_map=weight_map, name_of=shards.name_of
        )
        raw: dict = {}
        for path in shards.paths:
            arrays = mx.load(str(path))
            if shard_prune is not None:
                arrays = shard_prune(arrays)
            origins.record_shard(path, arrays)
            raw.update(arrays)
        arrays = None  # no loop-local reference to a raw shard dict survives
        with concat_ledger(origins.concats):
            weights = sanitize(dict(raw))
        del raw
        quantize(weights)
        model.load_weights(list(weights.items()), strict=True)
        model.eval()
        if request.mode == "moe_experts":
            manager = install_expert_streaming(
                model,
                shards.root,
                ceiling_bytes=request.budget_bytes,
                top_k=top_k,
                read_workers=request.read_workers,
                collector=request.collector,
                mtp_resident=request.mtp_resident,
                shards=shards,
                resolver=origins.resolve_expert,
            )
        else:
            if dense_targets is None:
                raise StreamingUnavailable("dense streaming needs adapter-selected targets")
            manager = install_dense_streaming(
                model,
                targets=tuple(dense_targets(model)),
                shards=shards,
                resolver=origins.resolve_tensor,
                staging_bytes=request.budget_bytes,
                read_workers=request.read_workers,
            )
        for key in manager.streamed_keys:
            weights.pop(key, None)
        # Drop every out-of-tree reference to a streamed raw array.
        del origins
        gc.collect()
        if on_installed is not None:
            on_installed(manager)
        remainder = _tree_nbytes(model.parameters())
        manager.plan.resident_remainder_bytes = remainder
        limit = request.admission_limit_bytes
        if limit is not None:
            need = remainder + manager.reserved_bytes() + request.headroom_bytes
            if need > limit:
                raise WorkingSetTooSmall(
                    f"streamed model needs {need} B before evaluation (resident "
                    f"remainder {remainder} + streaming reserve "
                    f"{manager.reserved_bytes()} + lane headroom "
                    f"{request.headroom_bytes}) but admission allows {limit} B"
                )
        mx.eval(model.parameters())
        shards.verify_paths()
        ubc_evict_paths([str(path) for path in shards.paths])
        return StreamedLoad(weights=weights, manager=manager, remainder_bytes=remainder)
    except BaseException:
        if manager is not None:
            manager.close()
        shards.close()
        raise


def require_declared_mode(request) -> WeightStreamRequest:
    if not isinstance(request, WeightStreamRequest):
        raise ValueError("weight_streaming must be a WeightStreamRequest")
    return request


def trunk_mlp_targets(model, *, layers_prefix: str,
                      projections=("gate_proj", "up_proj", "down_proj")) -> list:
    """Trunk MLP projection paths under ``layers_prefix`` (never an MTP head)."""
    import mlx.nn as nn

    selected = []
    for (path, module) in model.named_modules():
        if not path.startswith(layers_prefix) or is_mtp_path(path):
            continue
        (parent, _, leaf) = path.rpartition(".")
        if leaf in projections and parent.endswith(".mlp") and isinstance(
            module, nn.QuantizedLinear
        ):
            selected.append(path)
    return sorted(selected)


# Live setters on routed MoE blocks whose non-stock modes read expert tables.
_STOCK_SETTERS = (
    ("set_moe_routed_decode_mode", "off", "routed_decode"),
    ("set_moe_window_consumers", (), "window_consumers"),
    ("set_fused_expert_kernel_mode", "stock", "fused_expert_kernel"),
    ("set_moe_routed_candidate_mode", "off", "routed_candidate"),
)


def force_stock_expert_arithmetic(model) -> Dict[str, int]:
    """Select stock expert arithmetic on every routed block that exposes it.

    Streamed tables make the table-reading kernels decline per call; this
    makes the selection explicit, so the route runs (and records) the same
    arithmetic as a resident stock reference rather than a counted fallback.
    """
    counts = {label: 0 for (_, _, label) in _STOCK_SETTERS}
    for (_, module) in model.named_modules():
        for (setter, value, label) in _STOCK_SETTERS:
            method = getattr(module, setter, None)
            if not callable(method):
                continue
            if setter == "set_moe_routed_candidate_mode" and not hasattr(
                getattr(module, "switch_mlp", None), "routed_candidate_mode"
            ):
                continue
            method(value)
            counts[label] += 1
    return counts
