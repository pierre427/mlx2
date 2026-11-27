"""CPU checks for the opt-in row-invariant Flash tree tile law (prototype)."""
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx2.runtime.models import flash_tree_mma_law as L
from mlx2.runtime.models.activations import swiglu
from mlx2.runtime.models.switch_layers import QuantizedSwitchLinear


def test_groups_cover_every_pair_once_with_fixed_height():
    rng = np.random.default_rng(0)
    inds = np.stack([rng.choice(40, 10, replace=False) for _ in range(16)])
    ids, experts, position, distinct = L._groups(inds, 8)
    assert ids.shape[1] == 8 and len(experts) == ids.shape[0]
    assert distinct == len(np.unique(inds))
    flat = position.reshape(-1)
    assert len(set(flat.tolist())) == flat.size
    for (row, slot), pos in np.ndenumerate(position):
        group, lane = divmod(int(pos), 8)
        assert experts[group] == inds[row, slot] and ids[group, lane] == row


def _block(hidden=64, inter=32, experts=12):
    mx.random.seed(1)
    sw = SimpleNamespace(
        gate_proj=QuantizedSwitchLinear(hidden, inter, experts, bias=False, group_size=32, bits=4),
        up_proj=QuantizedSwitchLinear(hidden, inter, experts, bias=False, group_size=32, bits=4),
        down_proj=QuantizedSwitchLinear(inter, hidden, experts, bias=False, group_size=32, bits=4),
    )
    shared = SimpleNamespace(**{name: nn.QuantizedLinear(i, o, bias=False, group_size=32, bits=4)
                                for name, i, o in (("gate_proj", hidden, inter), ("up_proj", hidden, inter),
                                                   ("down_proj", inter, hidden))})
    return SimpleNamespace(switch_mlp=sw, shared_expert=shared)


def _reference_row(block, x, inds, scores, gate):
    sw = block.switch_mlp
    out = mx.zeros((x.shape[-1],), mx.float32)
    for slot in range(inds.shape[-1]):
        e = mx.array([int(inds[slot])], mx.uint32)
        h = swiglu(L._gather(x[None, None], sw.gate_proj, e), L._gather(x[None, None], sw.up_proj, e))
        out = out + L._gather(h, sw.down_proj, e).reshape(-1).astype(mx.float32) * scores[slot].astype(mx.float32)
    sh = block.shared_expert
    s = sh.down_proj(swiglu(sh.gate_proj(x[None]), sh.up_proj(x[None]))).reshape(-1)
    return out + mx.sigmoid(gate.astype(mx.float32)) * s.astype(mx.float32)


@pytest.mark.parametrize("group_rows", [8, 16])
def test_moe_matches_rowwise_reference_and_ignores_co_rows(group_rows):
    with mx.stream(mx.cpu):
        block = _block()
        rng = np.random.default_rng(2)
        rows, k = 6, 3
        x = mx.random.normal((rows, 64)).astype(mx.float32)
        inds = mx.array(np.stack([rng.choice(12, k, replace=False) for _ in range(rows)]).astype(np.uint32))
        scores = mx.softmax(mx.random.normal((rows, k)), axis=-1)
        gate = mx.random.normal((rows, 1))
        full = L.moe(block, x, inds, scores, gate, group_rows=group_rows)
        for r in range(rows):
            ref = _reference_row(block, x[r], inds[r], scores[r], gate[r, 0])
            assert mx.allclose(full[r], ref, atol=1e-4).item()
        sub = L.moe(block, x[:2], inds[:2], scores[:2], gate[:2], group_rows=group_rows)
        assert mx.array_equal(sub, full[:2]).item()


def test_rejects_unknown_group_height():
    with pytest.raises(ValueError):
        L.moe(_block(), mx.zeros((1, 64)), mx.zeros((1, 2), mx.uint32), mx.zeros((1, 2)),
              mx.zeros((1, 1)), group_rows=12)


def test_executor_rejects_unknown_tile_law():
    from mlx2.adapters.flash_next_tree import FlashNextTargetTreeExecutor

    with pytest.raises(ValueError, match="tile law"):
        FlashNextTargetTreeExecutor(object(), tile_law="fast")
    with pytest.raises(ValueError, match="group_rows"):
        FlashNextTargetTreeExecutor(object(), tile_law="row_invariant_mma", group_rows=12)
    with pytest.raises(ValueError, match="CPU reference"):
        FlashNextTargetTreeExecutor(object(), tile_law="row_invariant_mma", cpu_reference=True)


def test_device_moe_sort_and_unsort_match_host_grouping(monkeypatch):
    """The GPU path's sort / row map / inverse permutation, with the NAX
    kernels replaced by the stock gathers they are bit-identical to."""
    from mlx2.runtime.models import moe_nax_gather as NAX

    def split(x, gate, up, indices, *, row_map, group_size, bits, mode, plan=None):
        rows = mx.take(x, row_map, axis=0)
        g = mx.gather_qmm(rows, *gate, rhs_indices=indices, transpose=True,
                          group_size=group_size, bits=bits, mode=mode)
        u = mx.gather_qmm(rows, *up, rhs_indices=indices, transpose=True,
                          group_size=group_size, bits=bits, mode=mode)
        return swiglu(g, u)

    def plain(x, w, scales, biases, indices, *, group_size, bits, mode, plan=None):
        return mx.gather_qmm(x, w, scales, biases, rhs_indices=indices, transpose=True,
                             group_size=group_size, bits=bits, mode=mode)

    monkeypatch.setattr(NAX, "sorted_gather_qmm_swiglu_split", split)
    monkeypatch.setattr(NAX, "sorted_gather_qmm", plain)
    with mx.stream(mx.cpu):
        block = _block()
        rng = np.random.default_rng(3)
        rows, k = 5, 3
        x = mx.random.normal((rows, 64)).astype(mx.float32)
        inds = mx.array(np.stack([rng.choice(12, k, replace=False) for _ in range(rows)]).astype(np.uint32))
        scores = mx.softmax(mx.random.normal((rows, k)), axis=-1)
        gate = mx.random.normal((rows, 1))
        device = L.moe_device(block, x, inds, scores, gate)
        host = L.moe(block, x, inds, scores, gate, group_rows=8)
        assert mx.allclose(device, host, atol=1e-5).item()
        sub = L.moe_device(block, x[:2], inds[:2], scores[:2], gate[:2])
        assert mx.array_equal(sub, device[:2]).item()


def test_device_moe_declines_cleanly(monkeypatch):
    from mlx2.runtime.models import moe_nax_gather as NAX

    monkeypatch.setattr(NAX, "sorted_gather_qmm_swiglu_split", lambda *a, **k: None)
    with mx.stream(mx.cpu):
        assert L.moe_device(_block(), mx.zeros((2, 64)), mx.zeros((2, 2), mx.uint32),
                            mx.zeros((2, 2)), mx.zeros((2, 1))) is None


def test_compiled_glue_matches_plain():
    with mx.stream(mx.cpu):
        picked = mx.random.normal((5, 3, 64)).astype(mx.bfloat16)
        weights = mx.softmax(mx.random.normal((5, 3)), axis=-1)
        shared = mx.random.normal((5, 64)).astype(mx.bfloat16)
        gate = mx.random.normal((5, 1))
        a = L._combine(picked, weights, shared, gate)
        b = L._combine_compiled(picked, weights, shared, gate)
        assert a.dtype == mx.bfloat16 and mx.allclose(a.astype(mx.float32), b.astype(mx.float32), atol=1e-2).item()
        linear = nn.QuantizedLinear(64, 32, bias=False, group_size=32, bits=4)
        x = mx.random.normal((7, 64)).astype(mx.float32)
        L.FUSED_GLUE = False
        try:
            plain = L.project(linear, x)
        finally:
            L.FUSED_GLUE = True
        fused = L.project(linear, x)
        assert mx.allclose(plain, fused, atol=1e-4).item()


def test_gate_project_dense_and_one_column():
    with mx.stream(mx.cpu):
        router = nn.Linear(64, 12, bias=False)
        shared_gate = nn.Linear(64, 1, bias=False)
        x = mx.random.normal((5, 64))
        assert mx.allclose(L.gate_project(router, x), router(x), atol=1e-4).item()
        out = L.gate_project(shared_gate, x)
        assert out.shape == (5, 1) and mx.allclose(out, shared_gate(x), atol=1e-4).item()
        assert mx.array_equal(L.gate_project(router, x[:2]), L.gate_project(router, x)[:2]).item()


def test_cached_padding_matches_invariant_lane():
    from mlx2.runtime.models import invariant_prefill as inv

    with mx.stream(mx.cpu):
        linear = nn.QuantizedLinear(64, 32, bias=False, group_size=32, bits=4)
        x = mx.random.normal((1, 7, 64))
        L.begin_forward()
        a = L.project(linear, x)
        b = L.project(linear, x)  # memo hit
        ref = inv.quantized_matmul(x, linear["weight"], linear["scales"], linear["biases"],
                                   group_size=32, bits=4)
        assert mx.array_equal(a, ref).item() and mx.array_equal(b, ref).item()
        assert a.shape == (1, 7, 32)


def test_gdn_chain_walk_source_guard():
    from mlx2.runtime.models import qwen4_fused_gdn as F, qwen4_tree_gdn as G

    assert F._SOURCE.count(G._HEAD_SERIAL) == 1
    assert "chains[chain * MAXD + depth]" in G._CHAIN_PREFIX


class _HC(dict):
    """A tiny GatedResidual stand-in: dict membership like nn.Module."""

    def __init__(self, hc=4, hidden=32, lowrank=32, inject=True):
        super().__init__()
        self.hc_count, self.hidden_size = hc, hidden
        width = hc * hidden
        self.hc_norm = lambda x: mx.fast.rms_norm(x.reshape(*x.shape[:-1], hc, hidden), None, 1e-6).reshape(x.shape)
        self.input_mix_weight_down = nn.QuantizedLinear(width, lowrank, bias=False, group_size=32, bits=4)
        self.input_mix_weight_up = nn.QuantizedLinear(lowrank, width, bias=False, group_size=32, bits=4)
        if inject:
            self.block_inject_weight = nn.Linear(width, hc, bias=False)
            self["block_inject_weight"] = self.block_inject_weight


def test_hc_rows_matches_composed_math_and_is_row_invariant():
    with mx.stream(mx.cpu):
        module = _HC()
        x = mx.random.normal((6, 128)).astype(mx.float32)
        L.begin_forward()
        mixed, inject = L.hc_rows(module, x)
        normed = module.hc_norm(x)
        g = module.input_mix_weight_down(normed)
        up = module.input_mix_weight_up(nn.silu(g / 4))
        ref = mx.mean(mx.sigmoid(up).reshape(6, 4, 32) * normed.reshape(6, 4, 32), axis=-2)
        ref_inject = 2 * mx.sigmoid(module.block_inject_weight(normed) / 4)
        assert mx.allclose(mixed, ref, atol=1e-4).item()
        assert mx.allclose(inject, ref_inject, atol=1e-4).item()
        L.begin_forward()
        sub_mixed, sub_inject = L.hc_rows(module, x[:2])
        assert mx.array_equal(sub_mixed, mixed[:2]).item() and mx.array_equal(sub_inject, inject[:2]).item()
        assert L.hc_rows(_HC(inject=False), x)[1] is None


def test_hc_multirow_refusals_and_source_shape():
    from mlx2.runtime.models import qwen4_hc_decode as HC

    plan = {"inject_mode": "quantized"}
    assert HC.multirow_refusal(plan, 16) is None
    assert HC.multirow_refusal(plan, 1) is not None and HC.multirow_refusal(plan, 17) is not None
    assert HC.multirow_refusal({"inject_mode": "dense"}, 8) is not None
    for source in (HC.MROW_DOWN_SOURCE, HC.MROW_UP_SOURCE):
        assert "ROWS" in source and "threadgroup_position_in_grid.z" not in source


def test_hc_matrix_refusal_and_source():
    from mlx2.runtime.models import qwen4_hc_decode as HC

    assert HC.matrix_refusal({}, 16) is None and HC.matrix_refusal({}, 66) is None
    assert HC.matrix_refusal({}, 1) is not None and HC.matrix_refusal({}, 67) is not None
    assert "hcd_sigmoid_unary" in HC.MATRIX_MIX_SOURCE
