"""Prompt-chunk tier-2 tail (MLX2_PREFILL_TAIL_SKIP) for Ripple (ricursive), CPU.

The last layer computes K and the value blend over every row through the
same law its selected mode uses (eager, fused regions, Metal glue), and the
query, gate, attention output, o_proj, MLP and post norms for the last row
only.  Cache state and the next decode step must be bitwise identical to the
full forward; the checkpoint row's logits are close.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx2.runtime import prefill_tail_skip as pts  # noqa: E402
from mlx2.runtime.models import ricursive  # noqa: E402
from mlx2.runtime.models import ricursive_glue  # noqa: E402
from mlx2.runtime.models.cache import BatchKVCache, BatchRotatingKVCache  # noqa: E402
from mlx2.runtime.prefill_checkpoint_rows import prefill_checkpoint_rows  # noqa: E402

VOCAB = 64
WINDOW = 4
HIDDEN = 32


def _model(last="full_attention", *, mode="eager", quantize=False):
    mx.random.seed(4)
    layer_types = ["sliding_attention", "sliding_attention", "full_attention",
                   "sliding_attention", last]
    args = ricursive.ModelArgs(
        hidden_size=HIDDEN, intermediate_size=64, num_hidden_layers=5,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8, vocab_size=VOCAB,
        sliding_window=WINDOW, embed_scale=71.55417527999327, ffn_clamp_threshold=20.0,
        layer_types=layer_types,
    )
    model = ricursive.Model(args, nvfp4=False)
    for layer, value in zip(model.layers, [-2.75, -.3, .25, 1.1, 4.0], strict=True):
        layer.self_attn.value_res_lambda = mx.array([value])
    if quantize:
        from mlx import nn

        nn.quantize(model, group_size=32, bits=4,
                    class_predicate=lambda _p, m: isinstance(m, nn.Linear) and m.weight.shape[-1] % 32 == 0)
    model.eval()
    mx.eval(model.parameters())
    if mode == "fused":
        model.set_fused_elementwise(True)
    elif mode in ("glue", "glue_declines"):
        model.set_metal_glue(True)
    elif mode == "gqa_fold":
        model.set_gqa_folded_attention(16)
    return model


@pytest.fixture
def simulated_glue(monkeypatch):
    """Stand-in Metal glue kernels on CPU (the eager-equal fused law), so the
    tail's glue branch and its all-or-nothing fallback are exercised."""
    calls = {"qk": 0, "blend": 0, "decline_blend": False}

    def qk_norm_rope(raw, w, eps, cos=None, sin=None):
        calls["qk"] += 1
        return ricursive.fused_qk(raw, w, eps, cos, sin)

    def blend(seed32, x, value):
        calls["blend"] += 1
        if calls["decline_blend"]:
            return None
        return ricursive.fused_blend(seed32, x, value)

    monkeypatch.setattr(ricursive_glue, "qk_norm_rope", qk_norm_rope)
    monkeypatch.setattr(ricursive_glue, "blend", blend)
    return calls


def _batch_cache(model, padding):
    return [
        BatchKVCache(list(padding)) if kind == "full_attention"
        else BatchRotatingKVCache(WINDOW, list(padding))
        for kind in model.args.layer_types
    ]


def _state(cache):
    out = []
    for entry in cache:
        for name, value in sorted(vars(entry).items()):
            if isinstance(value, mx.array):
                out.append((name, np.array(value)))
            elif isinstance(value, (int, float)):
                out.append((name, value))
    return out


def _assert_same_state(a, b):
    sa, sb = _state(a), _state(b)
    assert [n for n, _ in sa] == [n for n, _ in sb]
    assert sa
    for (name, x), (_, y) in zip(sa, sb):
        if isinstance(x, np.ndarray):
            np.testing.assert_array_equal(x, y, err_msg=name)
        else:
            assert x == y, name


def _prefill(model, chunks, cache, lengths=None):
    outs = []
    for chunk in chunks:
        tokens = mx.array(chunk, dtype=mx.int32)
        rows = lengths or [tokens.shape[1]] * tokens.shape[0]
        with prefill_checkpoint_rows(rows, width=tokens.shape[1]) as request:
            logits = model(tokens, cache=cache)
            mx.eval(logits)
        outs.append((logits, request.applied))
    return outs


def _split_count():
    return pts.STATS["ricursive_tail_split"]


def _check_two_chunks(monkeypatch, model, chunks):
    monkeypatch.setenv(pts.ENV, "0")
    full_cache = model.make_cache()
    full = _prefill(model, chunks, full_cache)
    monkeypatch.setenv(pts.ENV, "1")
    before = _split_count()
    tail_cache = model.make_cache()
    tail = _prefill(model, chunks, tail_cache)
    assert _split_count() - before == len(chunks)
    for (want, _), (got, applied) in zip(full, tail):
        assert applied == 1 and got.shape == want.shape == (1, 1, VOCAB)
        np.testing.assert_allclose(np.array(got), np.array(want), rtol=1e-4, atol=1e-4)
    _assert_same_state(full_cache, tail_cache)
    step = mx.array([[5]], dtype=mx.int32)
    np.testing.assert_array_equal(np.array(model(step, cache=full_cache)),
                                  np.array(model(step, cache=tail_cache)))
    _assert_same_state(full_cache, tail_cache)


@pytest.mark.parametrize("last", ["full_attention", "sliding_attention"])
@pytest.mark.parametrize("mode", ["eager", "fused", "gqa_fold"])
@pytest.mark.parametrize("quantize", [False, True])
def test_two_chunks_bitwise_state_and_close_logits(monkeypatch, last, mode, quantize):
    rng = np.random.default_rng(0)
    # The second chunk outgrows the sliding window (array masks).
    chunks = [rng.integers(0, VOCAB, (1, 3)).tolist(), rng.integers(0, VOCAB, (1, 9)).tolist()]
    with mx.stream(mx.cpu):
        _check_two_chunks(monkeypatch, _model(last, mode=mode, quantize=quantize), chunks)


@pytest.mark.parametrize("last", ["full_attention", "sliding_attention"])
@pytest.mark.parametrize("decline_blend", [False, True])
def test_metal_glue_branch_and_its_fallback(monkeypatch, simulated_glue, last, decline_blend):
    simulated_glue["decline_blend"] = decline_blend
    rng = np.random.default_rng(5)
    chunks = [rng.integers(0, VOCAB, (1, 6)).tolist(), rng.integers(0, VOCAB, (1, 7)).tolist()]
    with mx.stream(mx.cpu):
        _check_two_chunks(monkeypatch, _model(last, mode="glue"), chunks)
    assert simulated_glue["qk"] > 0 and simulated_glue["blend"] > 0


@pytest.mark.parametrize("last", ["full_attention", "sliding_attention"])
def test_left_padded_batch(monkeypatch, last):
    rng = np.random.default_rng(1)
    padding = (3, 0)
    chunks = [rng.integers(0, VOCAB, (2, 8)).tolist(), rng.integers(0, VOCAB, (2, 4)).tolist()]
    with mx.stream(mx.cpu):
        model = _model(last)
        monkeypatch.setenv(pts.ENV, "0")
        full_cache = _batch_cache(model, padding)
        full = _prefill(model, chunks, full_cache)
        monkeypatch.setenv(pts.ENV, "1")
        before = _split_count()
        tail_cache = _batch_cache(model, padding)
        tail = _prefill(model, chunks, tail_cache)
        assert _split_count() - before == 2
        for (want, _), (got, applied) in zip(full, tail):
            assert applied == 1 and got.shape == want.shape == (2, 1, VOCAB)
            np.testing.assert_allclose(np.array(got), np.array(want), rtol=1e-4, atol=1e-4)
        _assert_same_state(full_cache, tail_cache)
        step = mx.array([[5], [7]], dtype=mx.int32)
        np.testing.assert_array_equal(np.array(model(step, cache=full_cache)),
                                      np.array(model(step, cache=tail_cache)))


def test_batch_generator_greedy_matches_with_tail(monkeypatch):
    """End to end through generate's checkpoint-row requests: the tail engages
    on prompt chunks and the batched greedy tokens are unchanged (FP32)."""
    from mlx2.runtime.generate import BatchGenerator

    prompts = [[3, 7, 11, 5, 9, 2, 8, 1, 4, 6, 10, 12, 13], [5, 1, 9], [9, 9, 4, 2, 7, 3, 8]]
    count = 6

    def run():
        generator = BatchGenerator(model, max_tokens=count, prefill_step_size=4,
                                   completion_batch_size=4, prefill_batch_size=4)
        uids = generator.insert(prompts, max_tokens=[count] * len(prompts))
        emitted = {uid: [] for uid in uids}
        for _ in range(400):
            responses = generator.next()
            responses = responses[1] if isinstance(responses, tuple) else responses
            for response in responses:
                emitted[response.uid].append(response.token)
            if all(len(tokens) >= count for tokens in emitted.values()):
                break
        return [emitted[uid][:count] for uid in uids]

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        model = _model()
        monkeypatch.setenv(pts.ENV, "0")
        expected = run()
        monkeypatch.setenv(pts.ENV, "1")
        before = _split_count()
        assert run() == expected
        assert _split_count() > before
    finally:
        mx.set_default_device(previous)


@pytest.mark.parametrize("value", [None, "0", "head"])
def test_declines_when_switch_is_off(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(pts.ENV, raising=False)
    else:
        monkeypatch.setenv(pts.ENV, value)
    with mx.stream(mx.cpu):
        model = _model()
        before = _split_count()
        (logits, applied), = _prefill(model, [[[1, 2, 3, 4]]], model.make_cache())
        assert _split_count() == before
        assert applied == 1 and logits.shape == (1, 1, VOCAB)


def test_declines_without_open_request(monkeypatch):
    monkeypatch.setenv(pts.ENV, "1")
    with mx.stream(mx.cpu):
        model = _model()
        before = _split_count()
        logits = model(mx.array([[1, 2, 3, 4]], dtype=mx.int32), cache=model.make_cache())
        assert _split_count() == before and logits.shape == (1, 4, VOCAB)


def test_declines_when_a_lane_checkpoint_is_not_last(monkeypatch):
    monkeypatch.setenv(pts.ENV, "1")
    with mx.stream(mx.cpu):
        model = _model()
        before = _split_count()
        tokens = np.random.default_rng(2).integers(0, VOCAB, (2, 6)).tolist()
        (logits, applied), = _prefill(model, [tokens], _batch_cache(model, (0, 0)),
                                      lengths=[6, 4])
        assert _split_count() == before
        assert applied == 1 and logits.shape == (2, 1, VOCAB)


def test_decode_step_never_takes_the_tail(monkeypatch):
    monkeypatch.setenv(pts.ENV, "1")
    with mx.stream(mx.cpu):
        model = _model()
        cache = model.make_cache()
        _prefill(model, [[[1, 2, 3]]], cache)
        before = _split_count()
        (logits, _), = _prefill(model, [[[4]]], cache)
        assert _split_count() == before and logits.shape == (1, 1, VOCAB)


def test_taps_and_prefill_body_keep_every_row(monkeypatch):
    rng = np.random.default_rng(3)
    tokens = mx.array(rng.integers(0, VOCAB, (1, 7)), dtype=mx.int32)
    with mx.stream(mx.cpu):
        model = _model()
        monkeypatch.setenv(pts.ENV, "0")
        ref_logits, ref_features = model.forward_with_taps(tokens, model.make_cache(), (1, 4))
        ref_body = model.prefill_body(tokens, model.make_cache(), (1, 4))
        monkeypatch.setenv(pts.ENV, "1")
        before = _split_count()
        with prefill_checkpoint_rows([7], width=7) as request:
            logits, features = model.forward_with_taps(tokens, model.make_cache(), (1, 4))
            body = model.prefill_body(tokens, model.make_cache(), (1, 4))
        assert _split_count() == before and request.applied == 0
        assert logits.shape == (1, 7, VOCAB)
        np.testing.assert_array_equal(np.array(logits), np.array(ref_logits))
        np.testing.assert_array_equal(np.array(features), np.array(ref_features))
        np.testing.assert_array_equal(np.array(body), np.array(ref_body))


def test_backbone_refuses_tail_with_taps():
    with mx.stream(mx.cpu):
        model = _model()
        tokens = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
        with pytest.raises(ValueError, match="no taps"):
            model.model(tokens, model.make_cache(), capture={4: None}, tail_row=True)


def test_declines_input_embeddings(monkeypatch):
    monkeypatch.setenv(pts.ENV, "1")
    with mx.stream(mx.cpu):
        model = _model()
        tokens = mx.array([[1, 2, 3, 4, 5]], dtype=mx.int32)
        embeddings = model.model.embed_tokens(tokens)
        before = _split_count()
        declined = pts.STATS["declined_ricursive_input_embeddings"]
        with prefill_checkpoint_rows([5], width=5) as request:
            logits = model(tokens, cache=model.make_cache(), input_embeddings=embeddings)
        assert _split_count() == before and request.applied == 1
        assert pts.STATS["declined_ricursive_input_embeddings"] == declined + 1
        assert logits.shape == (1, 1, VOCAB)


def test_declines_inside_attention_tree_scope(monkeypatch):
    monkeypatch.setenv(pts.ENV, "1")
    with mx.stream(mx.cpu):
        model = _model()
        tokens = mx.array([[1, 2, 3]], dtype=mx.int32)
        before = _split_count()
        declined = pts.STATS["declined_ricursive_tree_scope"]
        with model.attention_tree_position_scope(mx.array([0, 1, 2], dtype=mx.int32)):
            with prefill_checkpoint_rows([3], width=3):
                model(tokens, cache=model.make_cache())
        assert _split_count() == before
        assert pts.STATS["declined_ricursive_tree_scope"] == declined + 1


def test_declines_a_wrapped_last_layer(monkeypatch):
    monkeypatch.setenv(pts.ENV, "1")
    with mx.stream(mx.cpu):
        model = _model()
        original = model.model.layers[-1]

        class Wrapped:
            def __call__(self, *args, **kwargs):
                return original(*args, **kwargs)

        model.model.layers[-1] = Wrapped()
        before = _split_count()
        declined = pts.STATS["declined_ricursive_wrapped_last_layer"]
        (logits, applied), = _prefill(model, [[[1, 2, 3, 4, 5]]], model.make_cache())
        assert _split_count() == before
        assert pts.STATS["declined_ricursive_wrapped_last_layer"] == declined + 1
        assert applied == 1 and logits.shape == (1, 1, VOCAB)
