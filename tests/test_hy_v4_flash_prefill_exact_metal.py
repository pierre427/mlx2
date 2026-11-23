"""Real-Metal gate for the Hy4-flash exact prefill defaults (Pierre 2026-11-21).

Run with ``MLX2_RUN_METAL_TESTS=1``.  Small tensors (well under 1 GB in one
process) and the tiny random checkpoint: no model checkpoint, no speed or
qualification claim.

* ``qwen4_moe_weighted_sum`` at Hy4's top-16 with fp32 router weights over
  bf16 latent rows is bitwise the eager unsort + product + sum, including
  the real latent width (1536) and B=2; and the MoE layer with the sorted
  combine is bitwise the eager layer (dense and 4-bit affine expert banks).
* The head row floor: a padded checkpoint row is bitwise the same row of a
  2048-row chunk at a real-width 8-bit fp32-contract head, and the tiny
  model's trimmed rows are bitwise its full head's.
* The tail split leaves every cache plane bitwise unchanged on Metal.
"""

import os
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx2.runtime.models import hy_v4_flash as H
from mlx2.runtime.models import qwen4_moe_weighted_sum as WS
from mlx2.runtime.models.switch_layers import _scatter_unsort
from mlx2.runtime.prefill_checkpoint_rows import prefill_checkpoint_rows
# Imported here (it sets the CPU default device on import); the fixture selects the GPU.
from test_hy_v4_flash_cpu import LONG, build, checkpoint, tiny_config  # noqa: E402

pytestmark = pytest.mark.skipif(
    os.environ.get("MLX2_RUN_METAL_TESTS") != "1" or not mx.metal.is_available(),
    reason="set MLX2_RUN_METAL_TESTS=1 for the real-Metal gate",
)


@pytest.fixture(autouse=True)
def gpu(monkeypatch):
    monkeypatch.delenv("MLX2_PREFILL_TAIL_SKIP", raising=False)
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    yield
    mx.set_default_device(previous)


@pytest.mark.parametrize("B,T,D", [(1, 64, 24), (1, 300, 1024), (1, 2048, 1536), (2, 512, 1536)])
def test_weighted_sum_kernel_bitwise_at_top16(B, T, D):
    K = 16
    for seed in range(3):
        mx.random.seed(seed)
        idx = mx.argpartition(mx.random.normal((B, T, 512)), kth=K - 1, axis=-1)[..., :K].astype(mx.int32)
        rows = (mx.random.normal((B * T * K, 1, D)) * 3).astype(mx.bfloat16)
        inv = mx.argsort(mx.argsort(idx.flatten()))
        w = mx.random.uniform(shape=(B, T, K)) * 4
        eager = (_scatter_unsort(rows, inv, idx.shape).squeeze(-2) * w[..., None]).sum(axis=-2)
        admission = WS.admit_moe_weighted_sum(x_sorted=rows, inv_order=inv, scores=w, indices=idx,
                                              do_sort=True, training=False,
                                              supported_top_k=H.SORTED_SUM_TOP_K)
        assert admission.accepted, admission.reason
        got = WS.moe_weighted_sum(rows, inv, w)
        assert got.dtype == eager.dtype == mx.float32
        assert mx.array_equal(got, eager).item()
        assert mx.array_equal(got.astype(mx.bfloat16), eager.astype(mx.bfloat16)).item()


def _tiny_moe_model(quantized):
    config = tiny_config(n_routed_experts=32, num_experts_per_tok=16, routed_expert_hidden_size=32,
                         moe_intermediate_size=32)
    model, _ = build(config, checkpoint(config))
    model.eval()  # as the adapter loads it
    if quantized:
        nn.quantize(model, group_size=32, bits=4,
                    class_predicate=lambda p, m: p.endswith("mlp.experts.gate_proj")
                    or p.endswith("mlp.experts.up_proj") or p.endswith("mlp.experts.down_proj"))
    return model


@pytest.mark.parametrize("quantized", [False, True], ids=["dense", "affine4"])
@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_moe_layer_sorted_combine_bitwise(quantized, dtype):
    model = _tiny_moe_model(quantized)
    moe = model.model.layers[1].mlp
    x = (mx.random.normal((2, 48, 32), key=mx.random.key(3))).astype(dtype)
    moe.sorted_weighted_sum = False
    eager = moe(x)
    moe.sorted_weighted_sum = True
    got = moe(x)
    assert moe.sorted_sum_calls == 1, moe.sorted_sum_declines
    assert mx.array_equal(got, eager).item()


def _stub_head(V=16384, D=3072, seed=0):
    lin = nn.Linear(D, V, bias=False)
    lin.weight = mx.random.normal((V, D), key=mx.random.key(seed)) * 0.02
    head = nn.QuantizedLinear.from_linear(lin, group_size=64, bits=8)
    head.scales = head.scales.astype(mx.float32)
    head.biases = head.biases.astype(mx.float32)
    stub = SimpleNamespace(args=SimpleNamespace(hidden_size=D, enable_lm_head_fp32=True), lm_head=head,
                           model=SimpleNamespace(embed_tokens=nn.Embedding(4, D)))
    stub.model.embed_tokens.weight = stub.model.embed_tokens.weight.astype(mx.bfloat16)
    stub.logits = lambda h: H.Model.logits(stub, h)
    stub.HEAD_ROW_CANDIDATES, stub.HEAD_ROW_REFERENCES = H.Model.HEAD_ROW_CANDIDATES, H.Model.HEAD_ROW_REFERENCES
    return stub


def test_head_row_floor_at_real_width_is_bitwise_against_2048_rows():
    stub = _stub_head()
    floor = H.Model.calibrate_head_rows(stub)
    assert floor > 1  # one row through the head is not the chunk's row on Metal
    x = mx.random.normal((1, 2048, 3072), key=mx.random.key(11)).astype(mx.bfloat16)
    full = stub.logits(x)
    for lanes in (1, 2, 4):
        picks = mx.array([2047 - 517 * i for i in range(lanes)], dtype=mx.int32)
        kept = x[0, picks]
        rows = max(floor, lanes)
        padded = mx.concatenate([kept, mx.zeros((rows - lanes, 3072), dtype=kept.dtype)])
        out = stub.logits(padded[None])[0, :lanes]
        assert mx.array_equal(out, full[0, picks]).item(), (floor, lanes)
    # Without the floor, the one-row head is not bitwise the chunk's row.
    one = stub.logits(x[:, -1:])
    assert not mx.array_equal(one[0, 0], full[0, -1]).item()


def _tiny():
    config = tiny_config()
    model, drafter = build(config, checkpoint(config))
    model.eval()
    return model, LONG


@pytest.mark.parametrize("head", ["dense", "q8"])
def test_tiny_head_trim_bitwise_on_metal(head):
    model, LONG = _tiny()
    if head == "q8":
        nn.quantize(model, group_size=32, bits=8, class_predicate=lambda p, m: p == "lm_head")
    model.prefill_tail_skip = False
    for lengths in ([80], [80, 61]):
        width = max(lengths)
        tokens = mx.array([LONG[i:i + n] + [0] * (width - n) for i, n in enumerate(lengths)])

        def caches():
            if len(lengths) == 1:
                return model.make_cache()
            out = [H.HY4FlashBatchLayerCache(list(r)) for r in zip(*[model.make_cache() for _ in lengths])]
            for c in out:
                c.prepare(lengths=lengths, right_padding=[width - n for n in lengths])
            return out

        full = model(tokens, caches())
        with prefill_checkpoint_rows(lengths, width=width) as request:
            out = model(tokens, caches())
        assert request.applied == 1 and out.shape[1] == 1
        for b, n in enumerate(lengths):
            assert mx.array_equal(out[b, 0], full[b, n - 1]).item(), (head, lengths, b)


def test_tiny_tail_state_bitwise_on_metal():
    model, LONG = _tiny()
    tokens = mx.array([LONG[:60]])
    states = {}
    for tail in (True, False):
        model.prefill_tail_skip = tail
        cache = model.make_cache()
        for s in range(0, 60, 20):
            with prefill_checkpoint_rows([20], width=20):
                model(tokens[:, s:s + 20], cache)
        mx.eval([c.state for c in cache])
        nxt, toks = mx.array([[LONG[59]]]), []
        for _ in range(6):
            nxt = mx.argmax(model(nxt, cache)[:, -1:], axis=-1).astype(mx.int32)
            toks.append(int(nxt.item()))
        states[tail] = ([[np.array(a) for a in c.state] for c in cache], toks)
    assert model.prefill_exact_stats["tail_forwards"] == 3
    for a, b in zip(states[True][0], states[False][0]):
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
    assert states[True][1] == states[False][1]
