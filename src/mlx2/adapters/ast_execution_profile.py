"""Build AST's cold cost prior from the loaded exact kernels and tensors.

No model-name dispatch and no invented FLOP/s or bandwidth specifications.
Each kernel publishes how many full weight reads a row count costs it
(``weight_passes`` on the row-exact QMV, the lane matmul and the lane
installer); this module consumes that contract and never re-derives a
kernel's tiling. Other weight paths are conservatively priced as independent
row work. These relative-work estimates are not timings.
"""

from __future__ import annotations

import hashlib
import json

from ..runtime.ast_planner import ExecutionProfile


def build_execution_profile(
    model,
    draft,
    *,
    binding,
    tile_cap,
    max_lanes,
    num_draft,
    alternative_provider=None,
    fidelity="exact",
    automatic=False,
):
    import mlx.core as mx
    from mlx import nn

    from ..runtime import row_exact_verify
    from ..runtime.lane import installer as lane_installer
    from ..runtime.mlx_build import installed_mlx_build
    from ..runtime.models import row_exact_qmv as qmv
    from ..runtime.models.switch_layers import QuantizedSwitchLinear, SwitchLinear
    from ..runtime.row_exact.ops import installed_row_exact_base

    if fidelity not in {"exact", "near_tie_bounded"}:
        raise ValueError("unsupported AST execution-profile fidelity")
    exact = fidelity == "exact"
    if not exact and alternative_provider is not None:
        raise ValueError("near-tie AST does not admit an exact alternative backend")
    declared_max = (
        getattr(model, "progressive_external_exact_max_lanes", max_lanes)
        if exact else max_lanes
    )
    maximum = min(max_lanes, int(declared_max))

    def widths(name, fallback):
        value = getattr(model, name, fallback) if exact else fallback
        if callable(value):
            value = value(max_lanes=maximum, num_draft=num_draft)
        return tuple(value)

    tiles = widths("progressive_external_exact_tile_widths", range(1, tile_cap + 1))
    terminal = widths("progressive_external_exact_terminal_input_widths", range(2, tile_cap + 2))
    if not exact and automatic:
        from .external_draft_policy import progressive_external_approximate_geometry

        tiles, terminal = progressive_external_approximate_geometry(
            model, max_lanes=maximum, num_draft=num_draft
        )
    tiles = tuple(t for t in tiles if t <= tile_cap)
    terminal = tuple(t for t in terminal if t <= tile_cap + 1)
    max_rows = maximum * max(terminal)
    alternative_capacity = getattr(
        model, "progressive_external_alternative_row_capacity", None
    )
    if alternative_capacity is not None:
        if type(alternative_capacity) is not int or not 1 <= alternative_capacity <= 64:
            raise ValueError("AST alternative physical row capacity must be within1..64")
        # A tree packs independent proposal nodes, not a longer trained draft
        # sequence. Extend only the cost table; retain legal primary widths.
        if exact:
            max_rows = max(max_rows, alternative_capacity)
    costs = [0.0] * max_rows
    compact_costs = [0.0] * max_rows
    shapes, seen = [], set()
    total_bytes = 0
    row_capacity = None
    compact = exact and bool(getattr(model, "progressive_external_compact_exact_rows", False))
    args = getattr(model, "speculative_args", getattr(model, "args", None))
    active_experts = getattr(args, "num_experts_per_tok", None)
    named = getattr(model, "named_modules", None)
    named = list(named()) if callable(named) else []
    roots = getattr(model, "progressive_external_target_modules", None)
    if roots is not None:
        active_ids = {id(module) for root in roots for _, module in root.named_modules()}
        named = [(name, module) for name, module in named if id(module) in active_ids]
    compact_modules = [name for name, module in named
                       if row_exact_verify.compacts_valid_rows(module)]
    declared_compact = getattr(model, "progressive_external_compact_projection_names", ())
    for name, module in named:
        weight = getattr(module, "weight", None)
        if weight is None or id(weight) in seen or getattr(weight, "ndim", 0) < 2:
            continue
        seen.add(id(weight))
        # Embeddings are indexed lookups, not a full weight pass per token.
        if isinstance(module, nn.Embedding) or "embed_tokens" in name:
            continue
        size = int(weight.nbytes)
        if isinstance(module, (QuantizedSwitchLinear, SwitchLinear)) and active_experts:
            # Expert-bank traffic is bounded by the adapter's active expert
            # geometry, never charged as a dense pass over every expert.
            size = size * min(int(active_experts), weight.shape[0]) / weight.shape[0]
        bits = getattr(module, "bits", None)
        groups = getattr(module, "group_size", None)
        # Which exact law serves this projection inside the verify window:
        # a row-exact lane or guarded QMV kernel (the QMV module's own
        # dispatch), an installed lane module declared by the adapter and
        # exact at every width (it serves a one-row call), else one stock
        # one-row call per row.
        row_exact_lane = qmv.lane_preparation(module) if exact else None
        installed_lane = None
        if (exact and row_exact_lane is None and name in declared_compact
                and lane_installer.module_weight_passes(module, 1) is not None):
            installed_lane = lane_installer.installed_lane(module)
            cap = lane_installer.module_row_capacity(module)
            if cap is not None:
                row_capacity = cap if row_capacity is None else min(row_capacity, cap)
        lane = row_exact_lane if row_exact_lane is not None else installed_lane
        shapes.append((name, tuple(weight.shape), str(weight.dtype), bits, groups,
                       None if lane is None else (lane.backend, lane.split_k)))
        total_bytes += size
        # Only a module the row-exact route swapped in (or one the adapter
        # declares compact) runs the guarded QMV kernel inside the window;
        # a stock module is priced as independent rows.
        exact_quantized = exact and (row_exact_lane is not None or (
            isinstance(module, nn.QuantizedLinear)
            and (installed_row_exact_base(module) is nn.QuantizedLinear or name in declared_compact)
        ))
        for index in range(max_rows):
            rows = index + 1
            if installed_lane is not None:
                passes = lane_installer.module_weight_passes(module, rows)
                # Past an unchunked lane window the module would take the
                # stock law; the planner refuses that row count through
                # ``max_position_rows``, so price it as independent rows.
                passes = rows if passes is None else passes
            elif exact_quantized:
                passes = qmv.weight_passes(module, rows)
            else:
                passes = rows
            work = size * passes
            costs[index] += work
            if compact and (name in declared_compact or any(
                    name == parent or name.startswith(parent + ".") for parent in compact_modules)):
                compact_costs[index] += work
    # No tensor geometry (e.g. protocol-only host doubles): expose that lack
    # of evidence and use independent-row work, not hardware speed claims.
    if not total_bytes:
        costs = list(range(1, max_rows + 1))
        total_bytes = 1
    else:
        costs = [cost / total_bytes for cost in costs]
        compact_costs = [cost / total_bytes for cost in compact_costs]
    try:
        info = mx.device_info()
    except RuntimeError:
        info = {}
    build = installed_mlx_build()
    device = str(info.get("device_name", "unknown"))
    backend = getattr(draft, "backend", draft)
    draft_named = getattr(backend, "named_modules", None)
    draft_roots = getattr(backend, "ast_execution_modules", None)
    draft_modules = ([(name, module) for root in draft_roots for name, module in root.named_modules()]
                     if draft_roots is not None else list(draft_named()) if callable(draft_named) else [])
    draft_bytes, draft_seen = 0, set()
    for _, module in draft_modules:
        weight = getattr(module, "weight", None)
        if (weight is not None and getattr(weight, "ndim", 0) >= 2
                and id(weight) not in draft_seen and not isinstance(module, nn.Embedding)):
            draft_seen.add(id(weight))
            size = int(weight.nbytes)
            if isinstance(module, (QuantizedSwitchLinear, SwitchLinear)) and active_experts:
                size *= min(int(active_experts), weight.shape[0]) / weight.shape[0]
            draft_bytes += size
    # KV traffic per valid query/context token. An adapter that declares its
    # attention geometry is authoritative; otherwise the module tree is
    # scanned for KV attention modules (fused or unconventional projections
    # would be missed, which the cost authority records). Recurrent layers do
    # not incur linear context work.
    element_bytes = 4 if str(getattr(args, "dtype", "")) == "float32" else 2
    kv_bytes, windows, attention_modules = 0, {}, 0
    declared_attention = getattr(model, "progressive_external_attention_geometry", None)
    if callable(declared_attention):
        declared_attention = declared_attention()
    attention_authority = "scanned_attention_modules"
    if isinstance(declared_attention, dict):
        kv_bytes = int(declared_attention["kv_bytes_per_token"])
        windows = {int(w): int(b) for w, b in dict(declared_attention.get("windows", {})).items()}
        if kv_bytes < 0 or any(w < 1 or b < 0 for w, b in windows.items()):
            raise ValueError("declared attention geometry must be nonnegative with positive windows")
        attention_modules = 1 + len(windows)
        attention_authority = "declared_attention_geometry"
    for _, module in (() if attention_authority == "declared_attention_geometry" else named):
        heads = next((int(getattr(module, name)) for name in
                      ("num_key_value_heads", "num_kv_heads", "n_kv_heads")
                      if getattr(module, name, None) is not None), 0)
        dim = int(getattr(module, "head_dim", 0) or 0)
        # KV attention modules own both projections. Recurrent/GDN modules
        # and configuration objects cannot masquerade as attention layers.
        if not heads or not dim or not hasattr(module, "k_proj") or not hasattr(module, "v_proj"):
            continue
        attention_modules += 1
        size = 2 * heads * dim * element_bytes
        if getattr(module, "is_sliding", False):
            window = int(module.window)
            windows[window] = windows.get(window, 0) + size
        else:
            kv_bytes += size
    if not attention_modules and roots is None and attention_authority != "declared_attention_geometry":
        # Protocol-only fixtures / undeclared architectures retain an
        # explicitly weaker config estimate, never count linear attention.
        kinds = getattr(args, "layer_types", ()) or ()
        count = (sum("attention" in str(k) and "linear" not in str(k) for k in kinds)
                 if kinds else int(getattr(args, "num_hidden_layers", 0)))
        kv_bytes = count * 2 * int(getattr(args, "num_key_value_heads", 0) or 0) * int(getattr(args, "head_dim", 0) or 0) * element_bytes
    identity = hashlib.sha256(json.dumps(
        [binding, getattr(getattr(draft, "session", None), "session_revision", None),
         getattr(backend, "revision", None), build, info, shapes, tiles, terminal,
         maximum, compact, active_experts, kv_bytes, draft_bytes,
         row_capacity, windows, attention_authority,
         costs, compact_costs,
         getattr(draft, "policy", None),
         *([] if exact else [fidelity])],
        sort_keys=True, default=str,
    ).encode()).hexdigest()
    profile = ExecutionProfile(
        binding=identity, device=device, build=build, max_lanes=maximum,
        tile_widths=tiles, terminal_widths=terminal,
        packed_min_width=(
            max(1, int(getattr(model, "progressive_external_lane_exact_min_width", 1)))
            if exact else 1
        ),
        row_costs=tuple(costs),
        # Dispatch latency is unknown until measured. Do not fabricate a
        # time from the tensor size; the cold prior prices memory work only.
        launch_cost=0,
        context_row_cost=kv_bytes / total_bytes,
        draft_cost=draft_bytes / total_bytes if draft_bytes else 1.0,
        cost_authority=(
            ("device_build_kernel_geometry_prior" if shapes else "undeclared_geometry_independent_row_prior")
            if exact else "stock_unmeasured_independent_row_prior"
        )
        + "+" + attention_authority,
        backend=(
            ("padded_state_compact_exact_stages" if compact else "padded_row_exact")
            if exact else "padded_stock_near_tie_bounded"
        ),
        valid_row_costs=tuple(compact_costs) if compact else (),
        reusable_draft_lattice=bool(getattr(backend, "supports_continuation_tree_reuse", False)
                                   or getattr(backend, "continuation_source_reuses_backbone", False)),
        max_position_rows=row_capacity,
        context_windows=tuple((window, cost / total_bytes) for window, cost in sorted(windows.items())),
        sequential_draft=bool(getattr(backend, "ast_sequential_draft", False)),
    )
    if alternative_provider is None:
        return profile
    from ..runtime.target_tree_executor import ASTTargetTreeExecutorProvider

    if not isinstance(alternative_provider, ASTTargetTreeExecutorProvider):
        raise TypeError("AST target-tree backend requires a complete adapter provider")
    alternative_provider.validate_revision()
    candidate = alternative_provider.ast_execution_backend(profile)
    if candidate is None:
        raise ValueError(
            "AST target-tree backend lacks a passing device/build/source-bound "
            "numerical oracle"
        )
    return ExecutionProfile(
        **{
            **profile.__dict__,
            "alternative_backends": (candidate,),
        }
    )
