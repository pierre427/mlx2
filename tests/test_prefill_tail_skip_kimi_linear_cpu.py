"""Kimi Linear tier-2 prompt-chunk tail (MLX2_PREFILL_TAIL_SKIP=1), CPU.

Tiny random-weight model; no checkpoint qualification claim. The last layer
is global MLA: its latent/shared cache rows are written for every row, the
query, attention output, MoE/MLP, final norm and head run on the last row.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx2.adapters.kimi_linear import parse_policy
from mlx2.runtime import prefill_tail_skip as pts
from mlx2.runtime import row_exact_verify, verify_scope
from mlx2.runtime.models.cache import ArraysCache, BatchKVCache, KVCache
from mlx2.runtime.models.kimi_linear import DecoderLayer, Model, ModelArgs, MultiLatentAttention
from mlx2.runtime.prefill_checkpoint_rows import prefill_checkpoint_rows

VOCAB = 64


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _model(full=(2, 5), layers=5, dense_until=1, mla="auto"):
    mx.random.seed(11)
    args = ModelArgs(
        model_type="kimi_linear", vocab_size=VOCAB, hidden_size=32,
        intermediate_size=64, moe_intermediate_size=32, num_hidden_layers=layers,
        num_attention_heads=2, num_key_value_heads=2, num_experts=4,
        num_experts_per_tok=2, num_shared_experts=1, n_group=1, topk_group=1,
        first_k_dense_replace=dense_until, layer_group_size=4,
        max_position_embeddings=1048576, rms_norm_eps=1e-5, rope_theta=10000.,
        routed_scaling_factor=2.446, head_dim=72, kv_lora_rank=16, qk_rope_head_dim=8,
        qk_nope_head_dim=16, v_head_dim=16,
        linear_attn_config={"full_attn_layers": list(full),
                            "kda_layers": [i for i in range(1, layers + 1) if i not in full],
                            "num_heads": 2, "head_dim": 128, "short_conv_kernel_size": 4})
    model = Model(args)
    model.eval()
    model.configure(parse_policy({"kda": "reference", "mla": mla}))
    mx.eval(model.parameters())
    return model


def _batch_cache(model, padding):
    return [BatchKVCache(padding) if layer.is_global else ArraysCache(size=4, left_padding=padding)
            for layer in model.layers]


def _state(cache):
    out = []
    for entry in cache:
        if isinstance(entry, ArraysCache):
            out.extend(np.array(v) for v in entry.cache if v is not None)
        else:
            filled = entry.offset if type(entry) is KVCache else entry._idx
            out.append(np.array(entry.keys[..., :filled, :]))
            out.append(np.array(entry.values[..., :filled, :]))
    return out


def _chunks(model, chunks, cache, lengths=None):
    outs = []
    for chunk in chunks:
        tokens = mx.array(chunk, dtype=mx.int32)
        with prefill_checkpoint_rows(lengths or [tokens.shape[1]] * tokens.shape[0],
                                     width=tokens.shape[1]) as request:
            logits = model(tokens, cache=cache)
            mx.eval(logits)
        outs.append((logits, request.applied))
    return outs


def _compare(monkeypatch, model, chunks, make_cache, step):
    monkeypatch.setenv(pts.ENV, "0")
    full_cache = make_cache()
    full = _chunks(model, chunks, full_cache)
    monkeypatch.setenv(pts.ENV, "1")
    before = pts.STATS["kimi_linear_tail_split"]
    tail_cache = make_cache()
    tail = _chunks(model, chunks, tail_cache)
    assert pts.STATS["kimi_linear_tail_split"] - before == len(chunks)
    batch = len(chunks[0])
    for (want, want_applied), (got, applied) in zip(full, tail):
        # Off: the ordinary head trim; on: the claimed tail. Both one row.
        assert want_applied == applied == 1
        assert want.shape == got.shape == (batch, 1, VOCAB)
        np.testing.assert_allclose(np.array(got), np.array(want), rtol=1e-4, atol=1e-4)
    full_state, tail_state = _state(full_cache), _state(tail_cache)
    assert len(full_state) == len(tail_state) > 0
    for a, b in zip(full_state, tail_state):
        np.testing.assert_array_equal(a, b)
    monkeypatch.setenv(pts.ENV, "1")  # decode (S=1) never takes the tail
    np.testing.assert_array_equal(np.array(model(step, full_cache)),
                                  np.array(model(step, tail_cache)))
    return full_cache, tail_cache


@pytest.mark.parametrize("mla", ["auto", "expanded", "latent_tiled", "latent_single"])
@pytest.mark.parametrize("dense_until", [1, 5], ids=["moe_last", "dense_last"])
def test_two_chunks_state_bitwise_and_checkpoint_row(monkeypatch, mla, dense_until):
    rng = np.random.default_rng(0)
    chunks = [rng.integers(0, VOCAB, (1, 21)).tolist(), rng.integers(0, VOCAB, (1, 9)).tolist()]
    model = _model(mla=mla, dense_until=dense_until)
    _compare(monkeypatch, model, chunks, model.make_cache, mx.array([[5]], dtype=mx.int32))


@pytest.mark.parametrize("mla", ["auto", "expanded", "latent_tiled"])
def test_left_padded_batch_two_chunks(monkeypatch, mla):
    """B=2, left padding [0, 3] on BatchKVCache (MLA) + ArraysCache (KDA)."""
    rng = np.random.default_rng(5)
    chunks = [rng.integers(1, VOCAB, (2, 7)).tolist(), rng.integers(1, VOCAB, (2, 20)).tolist()]
    model = _model(mla=mla)
    layer = model.layers[-1].attention
    _compare(monkeypatch, model, chunks, lambda: _batch_cache(model, [0, 3]),
             mx.array([[5], [7]], dtype=mx.int32))
    assert layer.calls["multirow_calls"] > 0


def test_unpadded_batch_cache(monkeypatch):
    rng = np.random.default_rng(6)
    chunks = [rng.integers(1, VOCAB, (2, 18)).tolist(), rng.integers(1, VOCAB, (2, 5)).tolist()]
    model = _model()
    _compare(monkeypatch, model, chunks, lambda: _batch_cache(model, [0, 0]),
             mx.array([[5], [7]], dtype=mx.int32))


def test_no_cache_forward(monkeypatch):
    model = _model()
    tokens = mx.array([[3, 9, 1, 7, 2, 5, 8]], dtype=mx.int32)
    monkeypatch.setenv(pts.ENV, "0")
    full = model(tokens)
    monkeypatch.setenv(pts.ENV, "1")
    with prefill_checkpoint_rows([7], width=7) as request:
        got = model(tokens)
    assert request.applied == 1 and got.shape == (1, 1, VOCAB)
    np.testing.assert_allclose(np.array(got[:, 0]), np.array(full[:, -1]), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("value", [None, "0", "head"])
def test_env_unset_zero_or_head_does_not_split(monkeypatch, value):
    model = _model()
    if value is None:
        monkeypatch.delenv(pts.ENV, raising=False)
    else:
        monkeypatch.setenv(pts.ENV, value)
    before = pts.STATS["kimi_linear_tail_split"]
    (logits, applied), = _chunks(model, [[[1, 2, 3, 4, 5]]], model.make_cache())
    assert applied == 1 and logits.shape == (1, 1, VOCAB)  # ordinary head trim
    assert pts.STATS["kimi_linear_tail_split"] == before


def test_no_request_and_inner_checkpoint_stay_full(monkeypatch):
    model = _model()
    monkeypatch.setenv(pts.ENV, "1")
    before = pts.STATS["kimi_linear_tail_split"]
    tokens = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    assert model(tokens, cache=model.make_cache()).shape == (1, 4, VOCAB)
    # Lane 1's checkpoint row is not the chunk's last: not claimed.
    tokens = mx.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=mx.int32)
    with prefill_checkpoint_rows([4, 2], width=4) as request:
        out = model(tokens, cache=_batch_cache(model, [0, 0]))
    assert request.applied == 1 and out.shape == (2, 1, VOCAB)
    assert pts.STATS["kimi_linear_tail_split"] == before


def test_taps_and_prefill_body_never_engage(monkeypatch):
    model = _model()
    monkeypatch.setenv(pts.ENV, "1")
    tokens = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
    before = pts.STATS["kimi_linear_tail_split"]
    with prefill_checkpoint_rows([5], width=5) as request:
        logits, taps = model.forward_with_taps(tokens, model.make_cache(), (1, 4))
    assert request.applied == 0 and logits.shape == (1, 5, VOCAB) and taps.shape[1] == 5
    with prefill_checkpoint_rows([5], width=5) as request:
        features = model.prefill_body(tokens, model.make_cache(), (4,))
    assert request.applied == 0 and features.shape[1] == 5
    assert pts.STATS["kimi_linear_tail_split"] == before


def test_verify_and_row_exact_windows_decline(monkeypatch):
    model = _model()
    monkeypatch.setenv(pts.ENV, "1")
    tokens = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
    before = pts.STATS["kimi_linear_tail_split"]
    declined = pts.STATS["declined_kimi_verify_scope"]
    with verify_scope.verify_forward(), prefill_checkpoint_rows([5], width=5) as request:
        model(tokens, cache=model.make_cache())
    assert pts.STATS["declined_kimi_verify_scope"] == declined + 1
    monkeypatch.setattr(row_exact_verify, "active", lambda: True)
    declined = pts.STATS["declined_kimi_row_exact_window"]
    with prefill_checkpoint_rows([5], width=5) as request:
        out = model(tokens, cache=model.make_cache())
    assert request.applied == 1 and out.shape == (1, 1, VOCAB)  # head trim only
    assert pts.STATS["declined_kimi_row_exact_window"] == declined + 1
    assert pts.STATS["kimi_linear_tail_split"] == before


def test_wrapped_last_layer_declines(monkeypatch):
    model = _model()
    monkeypatch.setenv(pts.ENV, "1")
    tokens = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
    layer = model.layers[-1]

    class Hooked(DecoderLayer):  # residual-ablation style __class__ swap
        pass

    class HookedMLA(MultiLatentAttention):
        pass

    for module, cls in ((layer, Hooked), (layer.attention, HookedMLA)):
        original = module.__class__
        module.__class__ = cls
        try:
            before = dict(pts.STATS)
            with prefill_checkpoint_rows([5], width=5) as request:
                out = model(tokens, cache=model.make_cache())
            assert request.applied == 1 and out.shape == (1, 1, VOCAB)
            assert pts.STATS["kimi_linear_tail_split"] == before.get("kimi_linear_tail_split", 0)
            assert (pts.STATS["declined_kimi_last_layer_wrapped"]
                    == before.get("declined_kimi_last_layer_wrapped", 0) + 1)
        finally:
            module.__class__ = original


def test_kda_last_layer_training_law_and_cache_layout_refusals():
    tokens = mx.array([[1, 2, 3]], dtype=mx.int32)
    kda_last = _model(full=(2,), layers=3)
    assert kda_last._tail_refusal(tokens, kda_last.make_cache()) == "kimi_last_layer_kda"
    model = _model()
    assert model._tail_refusal(tokens, model.make_cache()) is None
    assert model._tail_refusal(tokens, None) is None
    model.layers[-1].attention.training_mla_policy = "padded_sdpa"
    assert model._tail_refusal(tokens, model.make_cache()) == "kimi_training_mla"
    model.layers[-1].attention.training_mla_policy = "expanded"
    cache = model.make_cache()
    assert model._tail_refusal(tokens, cache[:-1]) == "kimi_cache_layout"
    cache[-1] = object()  # e.g. a segmented verify or quantized cache
    assert model._tail_refusal(tokens, cache) == "kimi_cache_layout"
    cache = model.make_cache()
    cache[0].speculating = True
    assert model._tail_refusal(tokens, cache) == "kimi_speculating"


def test_kda_last_layer_runs_full_chunk(monkeypatch):
    model = _model(full=(2,), layers=3)
    monkeypatch.setenv(pts.ENV, "1")
    before = pts.STATS["declined_kimi_last_layer_kda"]
    (logits, applied), = _chunks(model, [[[1, 2, 3, 4, 5]]], model.make_cache())
    assert applied == 1 and logits.shape == (1, 1, VOCAB)
    assert pts.STATS["declined_kimi_last_layer_kda"] == before + 1
