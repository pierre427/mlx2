"""GLM-5.3 GGUF prompt-chunk post-mixer tail (MLX2_PREFILL_TAIL_SKIP=1), CPU only.

The last layer is KDA: its HC collapse and recurrent mixer run on every row
(conv window and state need them); the KDA output projection, HC expand, MoE
and head run on the last row. Cache words must equal the full forward's, and
so must the next decode step; the checkpoint row's logits are allclose.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from test_glm53_layer_tree_cpu import TreeWeights as _TreeWeights
from test_glm53_layer_tree_cpu import cache_bits

from mlx2.runtime import prefill_tail_skip as pts
from mlx2.runtime.models.glm53_gguf import Glm53GGUFModel
from mlx2.runtime.prefill_checkpoint_rows import prefill_checkpoint_rows

# Last layer KDA at index 4 takes the MoE FFN; at index 1 the dense FFN.
MOE_TAIL = ["linear_attention", "deepseek_sparse_attention", "linear_attention",
            "deepseek_sparse_attention", "linear_attention"]
DENSE_TAIL = ["deepseek_sparse_attention", "linear_attention"]
MLA_LAST = ["linear_attention", "deepseek_sparse_attention"]


class TreeWeights(_TreeWeights):
    """The layer-tree test's seeded weights plus the prompt-scope laws."""

    expert_prefill_policy = "reference"
    dense_prefill_policy = "reference"


@pytest.fixture
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def model_for(layer_types, weights=None):
    config = {"layer_types": layer_types, "rms_norm_eps": 1e-5,
              "max_position_embeddings": 65536, "vocab_size": 16}
    return Glm53GGUFModel(config, weights or TreeWeights(), identity="synthetic-glm53-tail",
                          math_policy="reference")


def run_chunks(model, chunks, cache, lengths=None):
    outs = []
    for chunk in chunks:
        tokens = mx.array([chunk], dtype=mx.int32)
        with prefill_checkpoint_rows(lengths or [len(chunk)], width=len(chunk)) as request:
            logits = model.prefill_forward(tokens, cache)
            mx.eval(logits, [c.state for c in cache])
        outs.append((logits, request.applied))
    return outs


CHUNKS = [[1, 4, 2, 9, 7, 3, 11], [5, 8, 6, 2, 13]]


@pytest.mark.parametrize("layer_types", [MOE_TAIL, DENSE_TAIL], ids=["moe", "dense"])
def test_tail_keeps_cache_words_and_next_decode_bitwise(cpu, monkeypatch, layer_types):
    weights = TreeWeights()
    model = model_for(layer_types, weights)
    monkeypatch.setenv(pts.ENV, "0")
    full_cache = model.make_cache()
    full = run_chunks(model, CHUNKS, full_cache)
    monkeypatch.setenv(pts.ENV, "1")
    before = dict(pts.STATS)
    tail_cache = model.make_cache()
    tail = run_chunks(model, CHUNKS, tail_cache)
    assert pts.STATS["glm53_tail_split"] - before.get("glm53_tail_split", 0) == 2
    assert pts.STATS["tail_engaged"] - before.get("tail_engaged", 0) == 2
    for (want, want_applied), (got, applied), chunk in zip(full, tail, CHUNKS):
        # Without the switch the ordinary head trim still narrows to one row.
        assert want_applied == 1 and want.shape == (1, 1, 16)
        assert applied == 1 and got.shape == (1, 1, 16)
        np.testing.assert_allclose(np.array(got), np.array(want), rtol=1e-4, atol=1e-5)
    assert cache_bits(tail_cache) == cache_bits(full_cache)
    step = mx.array([[3]], dtype=mx.int32)
    a, b = model(step, full_cache), model(step, tail_cache)
    mx.eval(a, b)
    assert np.array(a).tobytes() == np.array(b).tobytes()
    assert cache_bits(tail_cache) == cache_bits(full_cache)
    assert type(model.layers[-1]) is SimpleNamespace  # never mutated


def test_tail_equals_unrequested_full_width_last_row(cpu, monkeypatch):
    """Against the full ``[1, S, vocab]`` logits of a forward with no request."""
    weights = TreeWeights()
    model = model_for(MOE_TAIL, weights)
    tokens = mx.array([CHUNKS[0]], dtype=mx.int32)
    monkeypatch.setenv(pts.ENV, "1")
    full = model.prefill_forward(tokens, model.make_cache())  # no request: full width
    assert full.shape == (1, len(CHUNKS[0]), 16)
    (got, applied), = run_chunks(model, CHUNKS[:1], model.make_cache())
    assert applied == 1
    np.testing.assert_allclose(np.array(got[:, -1]), np.array(full[:, -1]), rtol=1e-4, atol=1e-5)


def _no_engage(before):
    return pts.STATS["glm53_tail_split"] == before.get("glm53_tail_split", 0)


@pytest.mark.parametrize("value", [None, "0", "head"])
def test_switch_off_or_head_mode_never_splits(cpu, monkeypatch, value):
    model = model_for(DENSE_TAIL)
    if value is None:
        monkeypatch.delenv(pts.ENV, raising=False)
    else:
        monkeypatch.setenv(pts.ENV, value)
    before = dict(pts.STATS)
    (logits, applied), = run_chunks(model, CHUNKS[:1], model.make_cache())
    assert applied == 1 and logits.shape == (1, 1, 16)  # ordinary head trim
    assert _no_engage(before)


def test_no_request_and_inner_checkpoint_row_stay_full(cpu, monkeypatch):
    model = model_for(DENSE_TAIL)
    monkeypatch.setenv(pts.ENV, "1")
    before = dict(pts.STATS)
    tokens = mx.array([CHUNKS[0]], dtype=mx.int32)
    assert model(tokens, model.make_cache()).shape == (1, len(CHUNKS[0]), 16)
    # A checkpoint row that is not the chunk's last (a short lane): no claim,
    # and select_checkpoint_rows narrows to that row as before.
    (logits, applied), = run_chunks(model, CHUNKS[:1], model.make_cache(), lengths=[3])
    assert applied == 1 and logits.shape == (1, 1, 16)
    assert _no_engage(before)
    # Decode (S=1) inside a request is never claimed.
    with prefill_checkpoint_rows([1], width=1):
        assert model(mx.array([[2]], dtype=mx.int32), model.make_cache()).shape == (1, 1, 16)
    assert _no_engage(before)


def test_taps_prefill_body_and_explicit_keep_never_engage(cpu, monkeypatch):
    weights = TreeWeights()
    model = model_for(MOE_TAIL, weights)
    tokens = mx.array([CHUNKS[0]], dtype=mx.int32)
    width = tokens.shape[1]
    monkeypatch.setenv(pts.ENV, "0")
    ref_cache = model.make_cache()
    _, ref_taps = model.forward_with_taps(tokens, ref_cache, (1, 4))
    model.external_feature_source = "mtp_seed"  # the MTP seed reader
    ref_seed_cache = model.make_cache()
    ref_seed = model.prefill_body(tokens, ref_seed_cache, (4,))
    model.external_feature_source = "layer_hc_mean"
    monkeypatch.setenv(pts.ENV, "1")
    before = dict(pts.STATS)
    with prefill_checkpoint_rows([width], width=width) as request:
        cache = model.make_cache()
        _, taps = model.forward_with_taps(tokens, cache, (1, 4))
    assert request.applied == 0 and taps.shape[1] == width
    assert np.array(taps).tobytes() == np.array(ref_taps).tobytes()
    assert cache_bits(cache) == cache_bits(ref_cache)
    model.external_feature_source = "mtp_seed"
    with prefill_checkpoint_rows([width], width=width) as request:
        cache = model.make_cache()
        seed = model.prefill_body(tokens, cache, (4,))
    model.external_feature_source = "layer_hc_mean"
    assert request.applied == 0 and seed.shape[1] == width
    assert np.array(seed).tobytes() == np.array(ref_seed).tobytes()
    assert cache_bits(cache) == cache_bits(ref_seed_cache)
    with prefill_checkpoint_rows([width], width=width) as request:
        out = model(tokens, model.make_cache(), logits_to_keep=2)
    assert request.applied == 0 and out.shape == (1, 2, 16)
    assert _no_engage(before)


def test_mla_last_layer_and_replaced_layer_record_decline(cpu, monkeypatch):
    monkeypatch.setenv(pts.ENV, "1")
    before = dict(pts.STATS)
    model = model_for(MLA_LAST)
    (logits, applied), = run_chunks(model, CHUNKS[:1], model.make_cache())
    assert applied == 1 and logits.shape == (1, 1, 16)  # ordinary head trim
    model = model_for(DENSE_TAIL)

    class Wrapped:  # a layer-replacement record, not the model's own
        is_linear = True

    model.layers[-1] = Wrapped()
    tokens = mx.array([CHUNKS[0]], dtype=mx.int32)
    with prefill_checkpoint_rows([tokens.shape[1]], width=tokens.shape[1]) as request:
        assert model._claim_prompt_tail(tokens, None) is False
    assert request.applied == 0
    assert (pts.STATS["declined_glm53_last_layer_not_kda"]
            - before.get("declined_glm53_last_layer_not_kda", 0)) == 2
    assert _no_engage(before)


def test_compiled_region_runner_declines(cpu, monkeypatch):
    monkeypatch.setenv(pts.ENV, "1")
    model = model_for(DENSE_TAIL)
    tokens = mx.array([CHUNKS[0]], dtype=mx.int32)
    before = pts.STATS["declined_glm53_compile_regions"]
    with prefill_checkpoint_rows([tokens.shape[1]], width=tokens.shape[1]) as request:
        assert model._claim_prompt_tail(tokens, object()) is False
    assert request.applied == 0
    assert pts.STATS["declined_glm53_compile_regions"] == before + 1


def test_batch_of_two_is_still_refused(cpu, monkeypatch):
    """The packed GLM-5.3 forward is B1 only; the switch does not change that."""
    monkeypatch.setenv(pts.ENV, "1")
    model = model_for(DENSE_TAIL)
    tokens = mx.array([CHUNKS[0], CHUNKS[0]], dtype=mx.int32)
    with (prefill_checkpoint_rows([7, 4], width=7) as request,
          pytest.raises(ValueError, match="B1")):
        model.prefill_forward(tokens, model.make_cache())
    assert request.applied == 0
