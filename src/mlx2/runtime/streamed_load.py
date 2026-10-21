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
from typing import Callable, Dict, Mapping, Optional, Protocol, Sequence, runtime_checkable

from .weight_stream import (
    DEFAULT_READ_WORKERS,
    PREFILL_ADMISSION_POLICIES,
    ROUTER_HINT_POLICIES,
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
BUDGET_MODES = ("explicit", "auto")
ROUTER_HINT_DECLARATION = "weight_stream_router_hints"
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
    expert_overlap: bool = False
    prefill_admission: str = "lru"
    router_hint: str = "off"
    # "explicit": ``budget_bytes`` is the operator's cache ceiling.  "auto":
    # the engine selected streaming because the resident load does not fit;
    # the loader sizes the ceiling from the admission limit once the resident
    # remainder is known, and ``budget_bytes`` must be 0.
    budget_mode: str = "explicit"
    # Read streamed rows with ``F_NOCACHE`` (see ``BoundShards``).  The
    # engine sets it for the automatic selection; explicit routes keep the
    # kernel page cache as before.
    bypass_page_cache: bool = False

    def __post_init__(self):
        if self.mode not in STREAM_MODES:
            raise ValueError(f"unknown weight streaming mode {self.mode!r}")
        if self.budget_mode not in BUDGET_MODES:
            raise ValueError(f"unknown weight streaming budget mode {self.budget_mode!r}")
        auto = self.budget_mode == "auto"
        if auto and (self.mode != "moe_experts" or self.admission_limit_bytes is None):
            raise ValueError(
                "automatic weight streaming is MoE-only and needs a measured admission limit"
            )
        for (name, value, low) in (
            ("budget_bytes", self.budget_bytes, 0 if auto else 1),
            ("read_workers", self.read_workers, 1),
            ("headroom_bytes", self.headroom_bytes, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < low:
                raise ValueError(f"weight streaming {name} must be an integer >= {low}")
        if type(self.bypass_page_cache) is not bool:
            raise ValueError("weight streaming bypass_page_cache must be boolean")
        if type(self.mtp_resident) is not bool:
            raise ValueError("weight streaming mtp_resident must be boolean")
        if type(self.expert_overlap) is not bool or (self.expert_overlap and self.mode != "moe_experts"):
            raise ValueError("expert_overlap is a boolean MoE-only option")
        if self.prefill_admission not in PREFILL_ADMISSION_POLICIES:
            raise ValueError(f"unknown expert prefill admission {self.prefill_admission!r}")
        if self.prefill_admission != "lru" and self.mode != "moe_experts":
            raise ValueError("prefill_admission is a MoE-only option")
        if self.router_hint not in ROUTER_HINT_POLICIES:
            raise ValueError(f"unknown expert router hint {self.router_hint!r}")
        if self.router_hint != "off" and self.mode != "moe_experts":
            raise ValueError("router_hint is a MoE-only option")
        limit = self.admission_limit_bytes
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 0
        ):
            raise ValueError("weight streaming admission_limit_bytes must be a non-negative int")
        if auto and self.budget_bytes != 0:
            raise ValueError("automatic weight streaming sizes its own budget; pass 0")


def declared_stream_modes(adapter_class) -> frozenset:
    """Modes the class itself declares -- never a parent's declaration."""
    try:
        own = vars(adapter_class)
    except TypeError:
        return frozenset()
    value = own.get(DECLARATION, ())
    return frozenset(value) & frozenset(STREAM_MODES)


def declared_router_hints(adapter_class) -> frozenset:
    """Router hints the class itself declares -- never a parent's declaration."""
    try:
        own = vars(adapter_class)
    except TypeError:
        return frozenset()
    value = own.get(ROUTER_HINT_DECLARATION, ())
    return frozenset(value) & frozenset(ROUTER_HINT_POLICIES)


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
    if request.router_hint != "off":
        hints = declared_router_hints(adapter_class)
        if request.router_hint not in hints:
            raise ValueError(
                f"{adapter_class.__name__} does not declare {request.router_hint} "
                "weight-stream router hints"
            )
    return request


def _tree_nbytes(tree) -> int:
    from mlx.utils import tree_flatten

    return sum(int(value.nbytes) for (_, value) in tree_flatten(tree))


def _auto_ceiling(model, request: WeightStreamRequest):
    """The cache room an automatically selected stream may take.

    Everything is counted against the same admission limit the pre-evaluation
    refusal below uses: the resident remainder (the whole lazy tree minus the
    expert tables leaving it), the lane headroom floor and the worst-case
    gather transient.  What is left is offered to the cache; the planner then
    sizes the cache from the resident-fraction curve inside it.
    """
    limit = int(request.admission_limit_bytes)

    def ceiling(shape: Dict[str, int]) -> int:
        remainder = _tree_nbytes(model.parameters()) - int(shape["target_bytes"])
        available = (
            limit - remainder - int(request.headroom_bytes) - int(shape["transient_bytes"])
        )
        if available < int(shape["floor_bytes"]):
            raise WorkingSetTooSmall(
                "automatic expert streaming cannot fit: admission limit "
                f"{limit} B - resident remainder {remainder} B - lane headroom "
                f"{request.headroom_bytes} B - worst gather transient "
                f"{shape['transient_bytes']} B leaves {available} B, below one "
                f"routed step ({shape['floor_bytes']} B = {shape['layers']} layers x "
                f"{shape['top_k']} experts x {shape['expert_bytes']} B)"
            )
        return available

    return ceiling


def artifact_weight_bytes(model_path) -> int:
    """Bytes of the safetensors shards a resident load reads.

    The indexed shard set when ``model.safetensors.index.json`` exists (a
    stray or stale file beside it is never counted), otherwise every
    top-level ``*.safetensors``.  File sizes include the small JSON headers:
    an upper bound on the weights, read without opening a shard.
    """
    import json
    from pathlib import Path

    root = Path(model_path).expanduser()
    index = root / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text()).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"{index} has no weight map")
        names = sorted(set(str(name) for name in weight_map.values()))
        paths = []
        for name in names:
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"shard path escapes the artifact: {name}")
            paths.append(root / relative)
    else:
        paths = sorted(root.glob("*.safetensors"))
    if not paths:
        raise ValueError(f"no safetensors shards under {root}")
    return sum(int(path.stat().st_size) for path in paths)


AUTO_REASONS = (
    "explicit_policy",
    "undeclared",
    "admission_limit_unmeasured",
    "artifact_unmeasured",
    "fits_resident",
    "resident_exceeds_admission",
)


def auto_expert_streaming_decision(
    adapter_class,
    model_path,
    *,
    admission_limit_bytes: Optional[int],
    explicit: bool,
    headroom_bytes: int = MIN_LANE_HEADROOM_BYTES,
) -> Dict[str, object]:
    """Whether a resident load fits; select expert streaming when it does not.

    The rule (Pierre, 2026-10-21): an adapter that declares ``moe_experts`` on
    its own class, whose artifact weights plus one lane's headroom floor
    exceed the admission limit (Metal advisory minus the service and driver
    reserves), streams its experts.  Everything else keeps the resident path.
    An explicit ``moe_expert_streaming`` policy always wins; ``fits`` is still
    reported so the engine can fail closed when streaming is explicitly
    disabled for an artifact that cannot be resident.
    """
    decision: Dict[str, object] = {
        "selected": False,
        "admission_limit_bytes": admission_limit_bytes,
        "headroom_bytes": int(headroom_bytes),
        "weight_bytes": None,
        "fits": None,
    }
    declared = "moe_experts" in declared_stream_modes(adapter_class)
    if declared:
        try:
            decision["weight_bytes"] = artifact_weight_bytes(model_path)
        except (OSError, ValueError, TypeError):
            decision["weight_bytes"] = None
    weight = decision["weight_bytes"]
    if weight is not None and admission_limit_bytes is not None:
        decision["fits"] = weight + int(headroom_bytes) <= int(admission_limit_bytes)
    if explicit:
        reason = "explicit_policy"
    elif not declared:
        reason = "undeclared"
    elif admission_limit_bytes is None:
        reason = "admission_limit_unmeasured"
    elif weight is None:
        reason = "artifact_unmeasured"
    elif decision["fits"]:
        reason = "fits_resident"
    else:
        reason = "resident_exceeds_admission"
        decision["selected"] = True
    decision["reason"] = reason
    return decision


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
    shards = BoundShards(root, names, records=records, weight_map=weight_map,
                         bypass_page_cache=request.bypass_page_cache)
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
        with concat_ledger(origins.concats, origins.stacks):
            weights = sanitize(dict(raw))
        del raw
        quantize(weights)
        model.load_weights(list(weights.items()), strict=True)
        model.eval()
        if request.mode == "moe_experts":
            auto_ceiling = None
            if request.budget_mode == "auto":
                auto_ceiling = _auto_ceiling(model, request)
            manager = install_expert_streaming(
                model,
                shards.root,
                ceiling_bytes=request.budget_bytes,
                auto_ceiling=auto_ceiling,
                top_k=top_k,
                read_workers=request.read_workers,
                collector=request.collector,
                mtp_resident=request.mtp_resident,
                overlap=request.expert_overlap,
                prefill_admission=request.prefill_admission,
                router_hint=request.router_hint,
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


@runtime_checkable
class ExpertStreamingArithmeticGuard(Protocol):
    """A routed block that can atomically select stream-safe arithmetic."""

    def force_stock_for_expert_streaming(self) -> Mapping[str, int]: ...


def force_stock_expert_arithmetic(model) -> Dict[str, int]:
    """Select stock expert arithmetic on every routed block that exposes it.

    Streamed tables make the table-reading kernels decline per call; this
    makes the selection explicit, so the route runs (and records) the same
    arithmetic as a resident stock reference rather than a counted fallback.
    """
    counts: Dict[str, int] = {}
    for (_, module) in model.named_modules():
        if not isinstance(module, ExpertStreamingArithmeticGuard):
            continue
        selected = module.force_stock_for_expert_streaming()
        if not isinstance(selected, Mapping):
            raise TypeError("expert streaming arithmetic guard must return a mapping")
        for label, count in selected.items():
            if not isinstance(label, str) or not label:
                raise TypeError("expert streaming arithmetic guard labels must be strings")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise TypeError("expert streaming arithmetic guard counts must be non-negative ints")
            counts[label] = counts.get(label, 0) + count
    return counts
