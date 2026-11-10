"""Flash-Next (qwen4_exp) prompt-chunk tier-2 tail split, CPU.

With ``MLX2_PREFILL_TAIL_SKIP=1`` and a claimable checkpoint-row request, the
last layer writes its state (PLE history, GDN conv/recurrent state, QSA index
keys, block summaries, MTP top-k share, K/V) over every row and computes the
rest (q, SDPA, o_proj, HC injects, MoE, mixer, head) over the last row only.
Every cache plane must be bitwise the full forward's, the next decode step
bitwise identical, and the checkpoint row's logits close.  Random tiny
weights; no real checkpoint.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx2.runtime import prefill_tail_skip as pts  # noqa: E402
from mlx2.runtime.models import qwen4_exp as q4  # noqa: E402
from mlx2.runtime.prefill_checkpoint_rows import prefill_checkpoint_rows  # noqa: E402

ENV = "MLX2_PREFILL_TAIL_SKIP"
COUNTER = "qwen4_exp_tail_split"
VOCAB = 64


def _model(layer_types=("linear_attention", "full_attention"), ple_layer_ids=(1,)):
    text_config = dict(
        model_type="qwen4_exp_text", hidden_size=32, intermediate_size=0,
        num_hidden_layers=len(layer_types), num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, vocab_size=VOCAB,
        linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=8,
        linear_value_head_dim=8, linear_conv_kernel_dim=4,
        layer_types=list(layer_types), num_experts=4, num_experts_per_tok=2,
        moe_intermediate_size=16, shared_expert_intermediate_size=16, hc_count=2,
        hc_lowrank=8, ple_layer_ids=list(ple_layer_ids), ple_embed_dim=32,
        ple_conv_kernel_size=4, ngram_size=3, heads_per_ngram=2,
        ngram_vocab_size_base=128, make_ngram_vocab_size_divisible_by=128,
        split_ngram_parts=1, indexer_n_heads=2, indexer_kv_heads=1,
        indexer_head_dim=8, indexer_budget=8, indexer_compress_ratio=2,
        mtp_num_hidden_layers=1,
        rope_parameters={"type": "default", "rope_theta": 10000,
                         "partial_rotary_factor": 0.25},
    )
    mx.random.seed(11)
    model = q4.Model(q4.ModelArgs(model_type="qwen4_exp", text_config=text_config))
    mx.eval(model.parameters())
    return model


def _ids(n, salt):
    rng = np.random.default_rng(salt)
    return rng.integers(1, VOCAB, n).tolist()


def _planes(cache):
    """Every array a cache list holds: KV/index/summary state, GDN/PLE slots,
    and the QSA cycle fields (pooled summaries, ragged tail, MTP share)."""
    out = []
    for entry in cache:
        out.extend(entry.state if isinstance(entry.state, (list, tuple)) else [entry.state])
        out.extend(getattr(entry, "cache", None) or [])
        for name, _blank in q4._QSA_CYCLE_STATE:
            value = getattr(entry, name, None)
            if isinstance(value, tuple):
                out.extend(v for v in value if isinstance(v, mx.array))
            elif isinstance(value, mx.array):
                out.append(value)
        for name in ("offset", "left_padding", "_idx"):
            value = getattr(entry, name, None)
            if isinstance(value, mx.array):
                out.append(value)
    return [p for p in out if isinstance(p, mx.array)]


def _assert_bitwise(left, right):
    a, b = _planes(left), _planes(right)
    assert len(a) == len(b) and a
    for x, y in zip(a, b):
        assert x.shape == y.shape and x.dtype == y.dtype
        np.testing.assert_array_equal(np.array(x), np.array(y))
    for x, y in zip(left, right):
        offset = getattr(x, "offset", None)
        if offset is not None and not isinstance(offset, mx.array):
            assert offset == y.offset


def _chunk(model, tokens, cache, lengths=None, *, request=True):
    tokens = mx.array(tokens, dtype=mx.int32)
    if not request:
        logits = model(tokens, cache=cache)
        mx.eval(logits, _planes(cache))
        return logits, None
    with prefill_checkpoint_rows(
        lengths or [tokens.shape[1]] * tokens.shape[0], width=tokens.shape[1]
    ) as req:
        logits = model(tokens, cache=cache)
        mx.eval(logits, _planes(cache))
    return logits, req.applied


def _close(got, want):
    np.testing.assert_allclose(
        np.array(got.astype(mx.float32)), np.array(want.astype(mx.float32)),
        rtol=1e-4, atol=1e-4,
    )


def _arms(monkeypatch, model, make_cache, chunks, lengths=None, arm_share=False):
    """Run ``chunks`` with the switch off and on from identical caches."""
    results = []
    base = make_cache()
    for mode in ("0", "1"):
        monkeypatch.setenv(ENV, mode)
        cache = copy.deepcopy(base)
        if arm_share:
            cache[-1]._mtp_share_topk = True
        outs = [_chunk(model, chunk, cache, lengths) for chunk in chunks]
        results.append((outs, cache))
    return results


def _engaged():
    return pts.STATS[COUNTER]


@pytest.mark.parametrize(
    "layer_types,ple_layer_ids",
    [
        (("linear_attention", "full_attention"), (1,)),  # QSA last layer
        (("full_attention", "linear_attention"), (2,)),  # GDN+PLE last layer
    ],
    ids=["qsa_last", "gdn_ple_last"],
)
def test_two_chunks_state_bitwise_decode_bitwise_logits_close(
    monkeypatch, layer_types, ple_layer_ids
):
    with mx.stream(mx.cpu):
        model = _model(layer_types, ple_layer_ids)
        # 24 + 20 rows: the second chunk has 22 blocks > the 4-block budget,
        # so the indexer takes an explicit sparse selection.
        chunks = [[_ids(24, 1)], [_ids(20, 2)]]
        before = _engaged()
        (full_outs, full_cache), (tail_outs, tail_cache) = _arms(
            monkeypatch, model, model.make_cache, chunks
        )
        assert _engaged() - before == 2
        for (want, want_applied), (got, applied) in zip(full_outs, tail_outs):
            assert want_applied == applied == 1
            assert got.shape == want.shape == (1, 1, VOCAB)
            _close(got, want)
        _assert_bitwise(full_cache, tail_cache)
        step = mx.array([[5]], dtype=mx.int32)
        np.testing.assert_array_equal(
            np.array(model(step, cache=full_cache)), np.array(model(step, cache=tail_cache))
        )
        _assert_bitwise(full_cache, tail_cache)


def _visibility(name):
    return q4._qsa_visibility.status()["counts"].get(name, 0)


@pytest.mark.parametrize("flag", ["_QSA_LAZY_CAUSAL_MASK", "_QSA_FUSED_PROJ"])
@pytest.mark.parametrize("batched", [False, True], ids=["b1", "b2_left_padded"])
def test_lazy_visibility_and_fused_projection_modes(monkeypatch, flag, batched):
    """The policy's lazy causal visibility (QSAVisibility masks) and the
    opt-in fused QSA projection table keep the same guarantees."""
    monkeypatch.setattr(q4, flag, True)
    with mx.stream(mx.cpu):
        model = _model()
        if batched:
            monkeypatch.setenv(ENV, "0")
            base = _merged_left_padded(model)
            make, chunks = (lambda: base), [[_ids(12, 7), _ids(12, 8)], [_ids(10, 9), _ids(10, 10)]]
        else:
            make, chunks = model.make_cache, [[_ids(24, 1)], [_ids(20, 2)]]
        visibility_before = _visibility("metadata_forwards")
        before = _engaged()
        (full_outs, full_cache), (tail_outs, tail_cache) = _arms(monkeypatch, model, make, chunks)
        assert _engaged() - before == 2
        if flag == "_QSA_LAZY_CAUSAL_MASK":
            assert _visibility("metadata_forwards") > visibility_before
        for (want, _), (got, applied) in zip(full_outs, tail_outs):
            assert applied == 1
            _close(got, want)
        _assert_bitwise(full_cache, tail_cache)
        step = mx.array([[5], [9]] if batched else [[5]], dtype=mx.int32)
        np.testing.assert_array_equal(
            np.array(model(step, cache=full_cache)), np.array(model(step, cache=tail_cache))
        )


def test_explicit_selection_is_exercised():
    """The QSA chunk above really runs an explicit sparse selection."""
    with mx.stream(mx.cpu):
        model = _model()
        cache = model.make_cache()
        _chunk(model, [_ids(24, 1)], cache, request=False)
        attn = model.language_model.model.layers[-1].self_attn
        x = mx.random.normal((1, 20, 32))
        mask = q4.qsa_attention_mask(x, cache[-1])
        selection = attn.indexer(copy.deepcopy(x), mask, copy.deepcopy(cache[-1]))
        assert selection.kind == "explicit"
        last = q4._last_row_selection(selection)
        assert last.length == 1 and last.raw_block_ids.shape[1] == 1
        np.testing.assert_array_equal(
            np.array(last.dense_mask()), np.array(selection.dense_mask()[..., -1:, :])
        )


def test_mtp_share_topk_state_is_kept(monkeypatch):
    with mx.stream(mx.cpu):
        model = _model()
        chunks = [[_ids(24, 3)], [_ids(18, 4)]]
        (_full_outs, full_cache), (_tail_outs, tail_cache) = _arms(
            monkeypatch, model, model.make_cache, chunks, arm_share=True
        )
        assert tail_cache[-1]._mtp_shared_topk is not None
        _assert_bitwise(full_cache, tail_cache)


def _merged_left_padded(model):
    """A B=2 cohort with left padding: two lanes prefilled separately (17 and
    26 tokens), merged the way the scheduler joins lanes."""
    lanes = []
    for n, salt in ((17, 5), (26, 6)):
        cache = model.make_cache()
        ids = _ids(n, salt)
        for start in range(0, n, 16):
            _chunk(model, [ids[start:start + 16]], cache, request=False)
        lanes.append(cache)
    merged = [type(lanes[0][i]).merge([lane[i] for lane in lanes])
              for i in range(len(lanes[0]))]
    mx.eval([c.state for c in merged])
    return merged


@pytest.mark.parametrize(
    "layer_types,ple_layer_ids",
    [
        (("linear_attention", "full_attention"), (1,)),
        (("full_attention", "linear_attention"), (2,)),
    ],
    ids=["qsa_last", "gdn_ple_last"],
)
def test_left_padded_batch(monkeypatch, layer_types, ple_layer_ids):
    with mx.stream(mx.cpu):
        model = _model(layer_types, ple_layer_ids)
        monkeypatch.setenv(ENV, "0")
        base = _merged_left_padded(model)
        attn_index = list(layer_types).index("full_attention")
        assert type(base[attn_index]) is q4.BatchQSAKVCache
        assert np.array(base[attn_index].left_padding).tolist() == [9, 0]
        chunks = [[_ids(12, 7), _ids(12, 8)], [_ids(10, 9), _ids(10, 10)]]
        before = _engaged()
        (full_outs, full_cache), (tail_outs, tail_cache) = _arms(
            monkeypatch, model, lambda: base, chunks
        )
        assert _engaged() - before == 2
        for (want, _), (got, applied) in zip(full_outs, tail_outs):
            assert applied == 1 and got.shape == want.shape == (2, 1, VOCAB)
            _close(got, want)
        _assert_bitwise(full_cache, tail_cache)
        step = mx.array([[5], [9]], dtype=mx.int32)
        np.testing.assert_array_equal(
            np.array(model(step, cache=full_cache)), np.array(model(step, cache=tail_cache))
        )


# --- declines -------------------------------------------------------------


def _declines(monkeypatch, model, cache, tokens, lengths=None, *, mode="1"):
    if mode is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, mode)
    before = _engaged()
    logits, applied = _chunk(model, tokens, cache, lengths)
    assert _engaged() == before
    return logits, applied


@pytest.mark.parametrize("mode", [None, "0", "head"])
def test_declines_when_switch_is_off(monkeypatch, mode):
    with mx.stream(mx.cpu):
        model = _model()
        _logits, applied = _declines(
            monkeypatch, model, model.make_cache(), [_ids(9, 1)], mode=mode
        )
        assert applied == 1  # the ordinary head narrowing still applied


def test_declines_without_an_open_request(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    with mx.stream(mx.cpu):
        model = _model()
        before = _engaged()
        logits, _ = _chunk(model, [_ids(9, 1)], model.make_cache(), request=False)
        assert _engaged() == before and logits.shape == (1, 9, VOCAB)


def test_declines_when_a_lane_checkpoint_is_not_last(monkeypatch):
    with mx.stream(mx.cpu):
        model = _model()
        monkeypatch.setenv(ENV, "0")
        cache = _merged_left_padded(model)
        tokens = [_ids(8, 11), _ids(8, 12)]
        _logits, applied = _declines(monkeypatch, model, cache, tokens, lengths=[8, 5])
        assert applied == 1


def test_decode_step_never_engages(monkeypatch):
    with mx.stream(mx.cpu):
        model = _model()
        cache = model.make_cache()
        _chunk(model, [_ids(9, 1)], cache, request=False)
        _declines(monkeypatch, model, cache, [[4]])


def test_declines_a_wrapped_last_layer(monkeypatch):
    with mx.stream(mx.cpu):
        model = _model()
        layers = model.language_model.model.layers
        original = layers[-1]

        class Wrapped(q4.DecoderLayer):
            pass

        original.__class__ = Wrapped
        try:
            before = pts.STATS["declined_qwen4_exp_wrapped_last_layer"]
            _declines(monkeypatch, model, model.make_cache(), [_ids(9, 1)])
            assert pts.STATS["declined_qwen4_exp_wrapped_last_layer"] == before + 1
        finally:
            original.__class__ = q4.DecoderLayer


def test_declines_a_swapped_attention_class(monkeypatch):
    """Row-exact verify swaps module classes at install."""
    with mx.stream(mx.cpu):
        model = _model()
        attn = model.language_model.model.layers[-1].self_attn

        class Swapped(q4.Attention):
            pass

        attn.__class__ = Swapped
        try:
            _declines(monkeypatch, model, model.make_cache(), [_ids(9, 1)])
        finally:
            attn.__class__ = q4.Attention


def test_declines_row_exact_and_invariant_modes(monkeypatch):
    with mx.stream(mx.cpu):
        model = _model()
        before = pts.STATS["declined_qwen4_exp_row_exact"]
        monkeypatch.setattr(q4, "_row_exact_window_active", lambda: True)
        _declines(monkeypatch, model, model.make_cache(), [_ids(9, 1)])
        monkeypatch.setattr(q4, "_row_exact_window_active", lambda: False)
        monkeypatch.setattr(q4._invariant, "active", lambda: True)
        assert q4._invariant.active()
        refusal = model.language_model._tail_row_refusal(model.make_cache(), None)
        assert refusal == "qwen4_exp_row_exact"
        assert pts.STATS["declined_qwen4_exp_row_exact"] == before + 1


def test_declines_input_embeddings_and_quantized_cache(monkeypatch):
    with mx.stream(mx.cpu):
        model = _model()
        lm = model.language_model
        cache = model.make_cache()
        embeds = mx.zeros((1, 9, 32))
        assert lm._tail_row_refusal(cache, embeds) == "qwen4_exp_input_embeddings"
        cache[-1] = q4.SinkWindowKVCache(8, 4)
        assert lm._tail_row_refusal(cache, None) == "qwen4_exp_last_cache_type"
        assert lm._tail_row_refusal(None, None) == "qwen4_exp_cache_layout"
        speculating = model.make_cache()
        speculating[0].speculating = True
        assert lm._tail_row_refusal(speculating, None) == "qwen4_exp_verify_cache"


def test_taps_prefill_body_and_mtp_backbone_never_engage(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    with mx.stream(mx.cpu):
        model = _model()
        tokens = mx.array([_ids(9, 1)], dtype=mx.int32)
        before = _engaged()
        with prefill_checkpoint_rows([9], width=9) as req:
            logits, taps = model.forward_with_taps(tokens, model.make_cache(), (0, 1))
            body = model.prefill_body(tokens, model.make_cache(), (1,))
            mixed, hyper = model.mtp_backbone(tokens, model.make_cache())
            mx.eval(logits, taps, body, mixed, hyper)
        assert _engaged() == before and req.applied == 0
        assert logits.shape == (1, 9, VOCAB) and mixed.shape[1] == hyper.shape[1] == 9
        with pytest.raises(ValueError):
            model.language_model.model(
                tokens, model.make_cache(), return_hyper=True, tail_row=True
            )


def test_fused_attention_rows_policy_declines_only_short_single_lane(monkeypatch):
    """Flash-Next enables fused rows by default; only short B=1 chunks use them."""
    from mlx2.runtime.models import qwen4_attn_rows as attn_rows

    monkeypatch.setattr(attn_rows, "_ENABLED", True)
    monkeypatch.setenv(ENV, "1")
    short = attn_rows.PREP_MAX_ROWS
    with mx.stream(mx.cpu):
        model = _model()
        before = pts.STATS["declined_qwen4_exp_attn_rows"]
        _logits, applied = _declines(monkeypatch, model, model.make_cache(), [_ids(short, 3)])
        assert applied == 1
        assert pts.STATS["declined_qwen4_exp_attn_rows"] == before + 1
        engaged = _engaged()
        _chunk(model, [_ids(short + 3, 4)], model.make_cache())
        assert _engaged() == engaged + 1


def test_declines_are_not_counted_without_a_claimable_request(monkeypatch):
    monkeypatch.setenv(ENV, "1")
    with mx.stream(mx.cpu):
        model = _model()
        cache = model.make_cache()
        before = {k: v for k, v in pts.STATS.items() if k.startswith("declined_")}
        model(mx.array([_ids(9, 5)]), cache=cache, input_embeddings=None)
        after = {k: v for k, v in pts.STATS.items() if k.startswith("declined_")}
        assert after == before
