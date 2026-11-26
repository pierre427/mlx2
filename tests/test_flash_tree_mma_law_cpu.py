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

    def split(x, gate, up, indices, *, row_map, group_size, bits, mode):
        rows = mx.take(x, row_map, axis=0)
        g = mx.gather_qmm(rows, *gate, rhs_indices=indices, transpose=True,
                          group_size=group_size, bits=bits, mode=mode)
        u = mx.gather_qmm(rows, *up, rhs_indices=indices, transpose=True,
                          group_size=group_size, bits=bits, mode=mode)
        return swiglu(g, u)

    def plain(x, w, scales, biases, indices, *, group_size, bits, mode):
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
