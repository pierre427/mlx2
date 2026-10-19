"""Adapter-owned geometry priors; CPU fixtures are not Metal qualification."""
import math
from types import SimpleNamespace

import mlx.core as mx
from mlx import nn

from mlx2.adapters.ast_execution_profile import build_execution_profile
from mlx2.runtime.models.row_exact_components import (
    RowExactQuantizedLinearMixin,
    row_exact_subclass,
)


def test_profile_uses_installed_kernel_geometry_and_binds_draft_source(monkeypatch):
    from mlx2.runtime.models import row_exact_qmv as qmv

    mx.set_default_device(mx.cpu)
    projection = nn.QuantizedLinear(256, 128, bias=False, group_size=64, bits=4)
    projection.__class__ = row_exact_subclass(RowExactQuantizedLinearMixin, nn.QuantizedLinear)
    calls = []

    def groups(bits, fast, n, rows, k):
        calls.append((bits, n, rows, k))
        return min(rows, 4)

    monkeypatch.setattr(qmv, "_row_groups", groups)
    target = SimpleNamespace(
        named_modules=lambda: [("projection", projection)],
        args=SimpleNamespace(num_hidden_layers=4, num_key_value_heads=2, head_dim=8),
        progressive_external_compact_exact_rows=True,
        progressive_external_exact_max_lanes=4,
        progressive_external_exact_tile_widths=(1, 2, 3),
        progressive_external_exact_terminal_input_widths=(2, 3, 4),
    )
    draft = SimpleNamespace(revision="draft-a", supports_continuation_tree_reuse=True)
    a = build_execution_profile(target, draft, binding="target-a", tile_cap=3, max_lanes=8, num_draft=7)
    assert a.max_lanes == 4
    assert a.row_costs[7] == 2
    assert a.cost([1, 3]) == 1  # four valid projection rows instead of six
    assert a.context_row_cost > 0
    assert a.reusable_draft_lattice
    assert calls and all((b, n, k) == (4, 128, 256) for b, n, _, k in calls)
    draft.revision = "draft-b"
    b = build_execution_profile(target, draft, binding="target-a", tile_cap=3, max_lanes=8, num_draft=7)
    assert a.binding != b.binding


def test_muse_auto_resolves_before_generator_construction():
    from test_external_dflash2_cpu import tiny

    from mlx2.adapters.external_draft_policy import resolve_progressive_external_tile
    from mlx2.adapters.muse_glimmer import normalize_external_policy

    model, _ = tiny()
    model.configure_target_verify_row_exact(True)
    policy = normalize_external_policy({
        "draft_model": "/test/draft", "num_draft": 7,
        "target_verify_row_exact": True, "progressive_verification_tile": "auto",
        "progressive_alternative_branches": 4,
        "progressive_alternative_strategy": "confidence_tree",
        "continuation_pool": {"sources": ["external"]},
    })
    tile = resolve_progressive_external_tile(policy, num_draft=7, target=model, max_lanes=4)
    assert tile in model.progressive_external_exact_tile_widths
    assert tile < 7


def test_planner_auto_ceiling_is_not_a_future_cohort_depth_cap():
    from mlx2.adapters.external_draft_policy import resolve_progressive_external_tile

    target = SimpleNamespace(progressive_external_exact_tile_widths=tuple(range(1, 16)),
                             progressive_external_exact_terminal_input_widths=tuple(range(2, 17)),
                             progressive_external_exact_max_lanes=8)
    policy = {"num_draft": 15, "progressive_verification_tile": "auto",
              "progressive_multilane_draft_cap": 4, "progressive_alternative_branches": 4,
              "progressive_cross_session_pack_rows": 8, "continuation_pool": {"ast_planner": True}}
    assert resolve_progressive_external_tile(policy, num_draft=15, target=target, max_lanes=4) == 14
    policy["continuation_pool"]["ast_planner"] = False
    assert resolve_progressive_external_tile(policy, num_draft=15, target=target, max_lanes=4) == 3
    policy["progressive_verification_tile"] = 3
    policy["continuation_pool"]["ast_planner"] = True
    assert resolve_progressive_external_tile(policy, num_draft=15, target=target, max_lanes=4) == 3


def test_installed_mpp_lane_uses_real_row_block_and_refuses_unavailable_chunking(monkeypatch):
    from mlx2.runtime.ast_planner import ASTExecutionPlanner, ReadyRow
    from mlx2.runtime.lane import installer
    from mlx2.runtime.lane.matmul import ROW_BLOCK

    projection = nn.QuantizedLinear(256, 128, bias=False, group_size=64, bits=8)
    object.__setattr__(projection, "_lane_prepared", SimpleNamespace(backend="mpp", split_k=2))
    object.__setattr__(projection, "_lane_min_rows", 1)
    object.__setattr__(projection, "_lane_max_rows", 8)
    object.__setattr__(projection, "_lane_chunk_above_max", False)
    monkeypatch.setattr(installer, "ENABLED", [True])
    # The contract mirrors the mixin's dispatch, which also needs a backend.
    monkeypatch.setattr(installer, "available", lambda: True)
    target = SimpleNamespace(named_modules=lambda: [("projection", projection)],
                             progressive_external_compact_projection_names=("projection",),
                             progressive_external_compact_exact_rows=True)
    p = build_execution_profile(target, SimpleNamespace(), binding="lane-target",
                                tile_cap=3, max_lanes=4, num_draft=7)
    # Eight rows share one tensor-unit pass inside the lane window; past the
    # window the mixin would take the stock law, so the lane is not credited
    # there and the planner refuses that row count through the capacity.
    assert ROW_BLOCK > 8 and p.row_costs[7] == 1 and p.row_costs[15] == 16
    assert p.max_position_rows == 8
    assert math.isinf(p.cost([4, 4, 4]))
    plan = ASTExecutionPlanner(p, tile_cap=3).plan_launch(
        [ReadyRow(i, i, 3, 0) for i in range(3)], terminal=True, max_lanes=4)
    assert len(plan.indices) <= 2


def test_target_scope_excludes_resident_head_and_prices_only_real_kv_attention():
    body = nn.Module()
    body.projection = nn.Linear(8, 8, bias=False)
    body.conv = nn.Conv1d(8, 8, 3, bias=False)
    body.attention = nn.Module()
    body.attention.num_key_value_heads = 2
    body.attention.head_dim = 4
    body.attention.k_proj = nn.Linear(8, 8, bias=False)
    body.attention.v_proj = nn.Linear(8, 8, bias=False)
    body.attention.is_sliding = True
    body.attention.window = 64
    dormant = nn.Linear(128, 128, bias=False)
    target = SimpleNamespace(
        named_modules=lambda: [("body." + n, m) for n, m in body.named_modules()] + [("mtp", dormant)],
        progressive_external_target_modules=(body,),
        args=SimpleNamespace(num_hidden_layers=100, num_key_value_heads=64, head_dim=128,
                             num_experts_per_tok=0),
    )
    p = build_execution_profile(target, SimpleNamespace(), binding="target", tile_cap=3, max_lanes=4, num_draft=7)
    total = sum(m.weight.nbytes for _, m in body.named_modules() if hasattr(m, "weight"))
    assert p.context_row_cost == 0
    assert p.context_windows == ((64, 2 * 2 * 4 * 2 / total),)
    dormant = nn.Linear(256, 256, bias=False)
    again = build_execution_profile(target, SimpleNamespace(), binding="target", tile_cap=3, max_lanes=4, num_draft=7)
    assert again.binding == p.binding


def test_profile_consumes_the_kernels_published_passes_not_private_tiling(monkeypatch):
    """A kernel that changes its row geometry changes the prior through its
    own ``weight_passes``; the profile holds no copy of the tiling."""
    from mlx2.runtime.lane import installer
    from mlx2.runtime.models import row_exact_qmv as qmv

    projection = nn.QuantizedLinear(256, 128, bias=False, group_size=64, bits=4)
    projection.__class__ = row_exact_subclass(RowExactQuantizedLinearMixin, nn.QuantizedLinear)
    target = SimpleNamespace(named_modules=lambda: [("projection", projection)],
                             progressive_external_compact_exact_rows=True)
    monkeypatch.setattr(qmv, "weight_passes", lambda linear, rows: 1)
    shared = build_execution_profile(target, SimpleNamespace(), binding="t", tile_cap=3,
                                     max_lanes=4, num_draft=7)
    assert shared.row_costs == (1.0,) * len(shared.row_costs)
    monkeypatch.setattr(qmv, "weight_passes", lambda linear, rows: rows)
    serial = build_execution_profile(target, SimpleNamespace(), binding="t", tile_cap=3,
                                     max_lanes=4, num_draft=7)
    assert serial.row_costs == tuple(float(n) for n in range(1, len(serial.row_costs) + 1))
    assert shared.binding != serial.binding
    # A stock QuantizedLinear the route never swapped runs the stock kernel,
    # so the guarded kernel's grouping is not credited to it.
    monkeypatch.setattr(qmv, "weight_passes", lambda linear, rows: 1)
    stock_target = SimpleNamespace(
        named_modules=lambda: [("projection", nn.QuantizedLinear(256, 128, bias=False, group_size=64, bits=4))],
        progressive_external_compact_exact_rows=True)
    stock = build_execution_profile(stock_target, SimpleNamespace(), binding="t", tile_cap=3,
                                    max_lanes=4, num_draft=7)
    assert stock.row_costs == tuple(float(n) for n in range(1, len(stock.row_costs) + 1))
    monkeypatch.setattr(qmv, "weight_passes", lambda linear, rows: rows)
    # An installed lane the mixin would not serve at one row is not credited.
    lane_module = nn.QuantizedLinear(256, 128, bias=False, group_size=64, bits=8)
    object.__setattr__(lane_module, "_lane_prepared", SimpleNamespace(backend="mpp", split_k=1))
    monkeypatch.setattr(installer, "module_weight_passes",
                        lambda module, rows: None if rows < 4 else 1)
    stock = build_execution_profile(
        SimpleNamespace(named_modules=lambda: [("lane", lane_module)],
                        progressive_external_compact_projection_names=("lane",),
                        progressive_external_compact_exact_rows=True),
        SimpleNamespace(), binding="t", tile_cap=3, max_lanes=4, num_draft=7)
    assert stock.row_costs[3] == 4.0 and stock.max_position_rows is None


def test_declared_attention_geometry_overrides_the_module_scan():
    from mlx2.runtime.attention_geometry import declared_attention_geometry, geometry_from_layer_types

    body = nn.Module()
    body.projection = nn.Linear(8, 8, bias=False)
    total = body.projection.weight.nbytes
    declared = declared_attention_geometry(full_layers=3, kv_heads=2, head_dim=4,
                                           sliding_layers=2, sliding_window=64)
    target = SimpleNamespace(
        named_modules=lambda: [("body." + n, m) for n, m in body.named_modules()],
        progressive_external_target_modules=(body,),
        progressive_external_attention_geometry=declared,
        args=SimpleNamespace(num_hidden_layers=100, num_key_value_heads=64, head_dim=128,
                             num_experts_per_tok=0),
    )
    p = build_execution_profile(target, SimpleNamespace(), binding="t", tile_cap=3, max_lanes=4, num_draft=7)
    assert p.context_row_cost == 3 * 2 * 2 * 4 * 2 / total
    assert p.context_windows == ((64, 2 * 2 * 2 * 4 * 2 / total),)
    assert p.cost_authority.endswith("+declared_attention_geometry")
    undeclared = SimpleNamespace(**{k: v for k, v in vars(target).items()
                                    if k != "progressive_external_attention_geometry"})
    q = build_execution_profile(undeclared, SimpleNamespace(), binding="t", tile_cap=3, max_lanes=4, num_draft=7)
    assert q.cost_authority.endswith("+scanned_attention_modules") and q.binding != p.binding
    kinds = ["linear_attention", "full_attention", "sliding_attention", "linear_attention"]
    g = geometry_from_layer_types(kinds, kv_heads=2, head_dim=4, sliding_window=32)
    assert (g["full_attention_layers"], g["sliding_attention_layers"], g["windows"]) == (1, 1, {32: 32})
