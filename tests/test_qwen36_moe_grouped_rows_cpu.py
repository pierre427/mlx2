"""CPU tests for the expert-grouped Qwen3.6 multi-row routed experts.

The grouped kernels are Metal-only.  Here the kernel's grouping is replaced
by a composed stand-in that follows ``plan`` exactly (leader election,
members in flat order, one expert table read per group) and evaluates each
member with the module's own one-assignment gather, so these tests pin the
plumbing and the row-invariance contract: which calls the hook serves, the
counted refusals, shapes, and that a row's output never depends on which
other rows share its tile.  The Metal numerics are checked against the stock
gathers by ``scripts/check_qwen36_moe_grouped_rows.py`` (synthetic tables)
and ``scripts/probe_qwen36_tree_moe_rows.py`` (real weights).
"""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest

from mlx2.runtime.models import qwen3_next as QN
from mlx2.runtime.models import qwen36_moe_decode as M
from mlx2.runtime.models import qwen36_moe_grouped as G

E, H, I, K = 16, 256, 128, 8


def _block(monkeypatch, seed=0):
    monkeypatch.setattr(QN, "_MOE_FUSED_GATE_UP", False)
    mx.random.seed(seed)
    args = SimpleNamespace(
        hidden_size=H, moe_intermediate_size=I, shared_expert_intermediate_size=I,
        norm_topk_prob=True, num_experts=E, num_experts_per_tok=K,
    )
    block = M.Qwen36SparseMoeBlock(args)
    for proj, shape in (("gate_proj", (E, I, H)), ("up_proj", (E, I, H)), ("down_proj", (E, H, I))):
        getattr(block.switch_mlp, proj).weight = mx.random.normal(shape) * 0.05
    nn.quantize(block, group_size=64, bits=4,
                class_predicate=lambda p, m: hasattr(m, "to_quantized") and "switch_mlp" in p)
    block.set_dtype(mx.bfloat16)
    block.set_fused_expert_kernel_mode("stock")  # the Qwen3.6 profile pins stock
    block.eval()
    return block


def _standin(x, indices, gate, up, down):
    """The grouped kernels' contract, composed: per group of ``plan``, each
    member assignment evaluated alone through the module's own gathers."""
    hidden = x.shape[-1]
    rows = x.size // hidden
    flat = x.reshape(rows, 1, 1, hidden)
    k = indices.shape[-1]
    out = [None] * (rows * k)
    for _leader, expert, members in G.plan(indices):
        idx = mx.array([[expert]], dtype=mx.uint32)
        for a in members:
            xr = flat[a // k][None]                           # [1, 1, 1, H]
            from mlx2.runtime.models.activations import swiglu

            h = swiglu(gate(xr, idx), up(xr, idx))
            out[a] = down(h, idx).reshape(hidden)
    return mx.stack(out).reshape(*indices.shape, hidden).astype(x.dtype)


@pytest.fixture
def grouped(monkeypatch):
    calls = []

    def admit(x, indices, gate, up, down):
        rows = x.size // x.shape[-1]
        return None if 2 <= rows <= G.MAX_ROWS else "width"

    def run(*args):
        calls.append(args[0].shape)
        return _standin(*args)

    monkeypatch.setattr(G, "admission_refusal", admit)
    monkeypatch.setattr(G, "grouped_rows", run)
    return calls


def _routing(rows, pool=E, seed=1):
    mx.random.seed(seed)
    return mx.argpartition(mx.random.uniform(shape=(1, rows, pool)), kth=-K, axis=-1)[..., -K:].astype(mx.uint32)


def test_plan_partitions_assignments_by_expert():
    inds = mx.array([[[3, 1, 2], [1, 4, 3], [5, 3, 0]]], dtype=mx.uint32)
    groups = G.plan(inds)
    assert [(lead, e) for lead, e, _ in groups] == [(0, 3), (1, 1), (2, 2), (4, 4), (6, 5), (8, 0)]
    assert dict((e, m) for _, e, m in groups)[3] == [0, 5, 7]
    covered = sorted(a for _, _, members in groups for a in members)
    assert covered == list(range(9))
    # A row picks an expert once, so no group outgrows the window.
    assert max(len(m) for _, _, m in groups) <= 3


def test_default_off_and_cpu_refusal_is_counted(monkeypatch):
    block = _block(monkeypatch)
    assert block.grouped_rows_selected() == "off"
    x = (mx.random.normal((1, 4, H)) * 0.5).astype(mx.bfloat16)
    stock = block(x)
    block.set_grouped_rows(True)
    assert block.grouped_rows_selected() == "qmv"
    out = block(x)
    stats = block.grouped_rows_stats(reset=True)
    assert mx.array_equal(out, stock)
    assert stats["calls"] == 0 and stats["fallbacks"] == 1
    assert stats["reasons"] == {"Metal runtime unavailable": 1}
    assert block.grouped_rows_stats()["fallbacks"] == 0
    block.set_grouped_rows(False)
    assert "grouped_rows_hook" not in block.switch_mlp.__dict__


@pytest.mark.parametrize("rows", [2, 8, 16])
def test_grouped_slots_equal_stock_and_block_output(monkeypatch, grouped, rows):
    block = _block(monkeypatch, seed=rows)
    sw = block.switch_mlp
    x = (mx.random.normal((1, rows, H)) * 0.5).astype(mx.bfloat16)
    inds = _routing(rows, pool=10)                   # shared experts: real groups
    stock = sw(x, inds)
    stock_block = block(x)
    block.set_grouped_rows(True)
    assert mx.array_equal(sw(x, inds), stock)
    assert mx.array_equal(block(x), stock_block)
    stats = block.grouped_rows_stats()
    assert stats["calls"] == 2 and stats["rows"] == 2 * rows and stats["fallbacks"] == 0
    assert grouped and all(shape == x.shape for shape in grouped)


def test_one_row_keeps_the_stock_path(monkeypatch, grouped):
    block = _block(monkeypatch)
    block.set_grouped_rows(True)
    x = (mx.random.normal((1, 1, H)) * 0.5).astype(mx.bfloat16)
    block(x)
    assert not grouped
    assert block.grouped_rows_stats()["calls"] == 0


@pytest.mark.parametrize("tile", [8, 16])
def test_row_output_independent_of_tile_neighbours(monkeypatch, grouped, tile):
    block = _block(monkeypatch, seed=tile)
    sw = block.switch_mlp
    target = (mx.random.normal((1, 1, H)) * 0.5).astype(mx.bfloat16)
    target_inds = _routing(1, pool=10, seed=7)
    alone = sw(target, target_inds)                  # the module's own one-row output
    block.set_grouped_rows(True)
    for seed, position in ((11, 0), (12, tile // 2), (13, tile - 1)):
        mx.random.seed(seed)
        neighbours = (mx.random.normal((1, tile, H)) * 0.5).astype(mx.bfloat16)
        inds = _routing(tile, pool=10, seed=seed)
        x = mx.concatenate([neighbours[:, :position], target, neighbours[:, position + 1:]], axis=1)
        inds = mx.concatenate([inds[:, :position], target_inds, inds[:, position + 1:]], axis=1)
        out = sw(x, inds)
        assert mx.array_equal(out[:, position:position + 1], alone)


def test_hook_declines_weighted_and_variant_calls(monkeypatch, grouped):
    block = _block(monkeypatch)
    sw = block.switch_mlp
    block.set_grouped_rows(True)
    x = (mx.random.normal((1, 4, H)) * 0.5).astype(mx.bfloat16)
    inds = _routing(4)
    scores = mx.ones(inds.shape, dtype=mx.bfloat16) / K
    sw(x, inds, scores=scores, variant="stock")
    sw(x, inds, variant="stock")
    assert not grouped


def test_admission_refuses_unsupported_formats(monkeypatch):
    block = _block(monkeypatch)
    sw = block.switch_mlp
    x = (mx.random.normal((1, 4, H)) * 0.5).astype(mx.bfloat16)
    inds = _routing(4)
    monkeypatch.setattr(G.RD, "runtime_supported", lambda: True)
    monkeypatch.setattr(G.RD, "served_swiglu_refusal", lambda: None)
    assert G.admission_refusal(x, inds, sw.gate_proj, sw.up_proj, sw.down_proj) is None
    assert "bf16" in G.admission_refusal(x.astype(mx.float16), inds, sw.gate_proj, sw.up_proj, sw.down_proj)
    one = x[:, :1]
    assert "outside" in G.admission_refusal(one, inds[:, :1], sw.gate_proj, sw.up_proj, sw.down_proj)
    monkeypatch.setattr(sw.down_proj, "bits", 8)
    assert "differ" in G.admission_refusal(x, inds, sw.gate_proj, sw.up_proj, sw.down_proj)


def test_adapter_policy_key_is_an_explicit_target_enum():
    from mlx2.adapters import qwen36_35b as A

    assert A.KERNEL_POLICY_ENV["moe_grouped_rows"] == "MLX_QWEN36_MOE_GROUPED_ROWS"
    assert "moe_grouped_rows" in A.EXPLICIT_ONLY_KERNELS
    assert "MLX_QWEN36_MOE_GROUPED_ROWS" in A.LIVE_APPLIED_ENV
    for law in ("off", "qmv", "lane"):
        assert A.validate_kernel_choice("moe_grouped_rows", law) == law
    with pytest.raises(ValueError):
        A.validate_kernel_choice("moe_grouped_rows", True)
    external, target = A.partition_external_target_policy(
        {"draft_model": "/drafter", "moe_grouped_rows": "qmv"})
    assert target == {"moe_grouped_rows": "qmv"} and "moe_grouped_rows" not in external
    assert not A.choice_selected("moe_grouped_rows", "off")


def test_setter_takes_laws_and_moves_the_routing_launch_bound(monkeypatch):
    block = _block(monkeypatch)
    assert block.__dict__["moe_topk_launch_max_rows"] == M.MULTI_ROW_GLUE_ROWS
    block.set_grouped_rows("lane")
    assert block.grouped_rows_selected() == "lane"
    assert block.__dict__["moe_topk_launch_max_rows"] == M.GROUPED_TOPK_LAUNCH_ROWS
    block.set_grouped_rows("off")
    assert block.grouped_rows_selected() == "off"
    assert block.__dict__["moe_topk_launch_max_rows"] == M.MULTI_ROW_GLUE_ROWS
    with pytest.raises(ValueError):
        block.set_grouped_rows("mma")


@pytest.mark.parametrize("bits", [4, 6, 8])
def test_lane_expert_source_addresses_one_expert_and_one_group(bits):
    src = G.lane_expert_source(bits, 64)
    assert "const device uchar* We = (const device uchar*)W + (int64_t)expert" in src
    assert "We = We" not in src and "mdims" not in src and "SBt" not in src
    assert src.count("Y[(int64_t)(gstart + m) * N + n + j]") == 1
    assert "Se[at]" in src and "Be[at]" in src
    # The early exit precedes every barrier of the body.
    assert src.index("if (expert < 0)") < src.index("threadgroup_barrier")
