"""CPU tests for cross-layer top-k index reuse (experimental, unqualified)."""

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models.cache import KVCache
from mlx2.runtime.topk_index_reuse import (
    DecodeTraffic,
    ReuseProbe,
    ReuseProbeKVCache,
    Variant,
    attention_probs,
    gathered_attention,
    indices_from_mask,
    install_probe,
    mass_recall,
    masked_attention,
    relative_error,
    select_mask,
    selection_scores,
    window_mask,
)


def _qkv(B=1, Hq=4, Hkv=2, L=1, T=40, D=16, seed=0):
    mx.random.seed(seed)
    q = mx.random.normal((B, Hq, L, D))
    k = mx.random.normal((B, Hkv, T, D))
    v = mx.random.normal((B, Hkv, T, D))
    return q, k, v


# --- selection -------------------------------------------------------------


def test_token_selection_keeps_window_and_exact_top_budget():
    scores = mx.array(np.random.default_rng(1).random((1, 1, 50)), dtype=mx.float32)
    mask = select_mask(scores, budget=6, window=4)
    got = np.array(mask)[0, 0]
    assert got.sum() == 10
    assert got[-4:].all()
    head = np.array(scores)[0, 0, :46]
    expected = set(np.argsort(-head)[:6].tolist())
    assert set(np.flatnonzero(got[:46]).tolist()) == expected


def test_context_within_budget_keeps_everything():
    scores = mx.zeros((2, 3, 12))
    assert bool(select_mask(scores, budget=8, window=4).all().item())


def test_block_selection_takes_whole_highest_mass_blocks():
    s = np.zeros((1, 1, 70), dtype=np.float32)
    s[0, 0, 16:24] = 1.0  # block 2 of size 8
    s[0, 0, 40] = 5.0  # block 5, a single strong token
    mask = np.array(select_mask(mx.array(s), budget=16, window=6, granularity="block", block_size=8))[0, 0]
    assert mask[16:24].all() and mask[40:48].all()
    assert mask[-6:].all()
    assert mask.sum() == 16 + 6


def test_block_reaching_into_window_adds_only_pre_window_positions():
    s = np.zeros((1, 1, 30), dtype=np.float32)
    s[0, 0, 24:26] = 1.0  # last partial block before the window, [24, 26)
    mask = np.array(select_mask(mx.array(s), budget=8, window=4, granularity="block", block_size=8))[0, 0]
    assert mask[24:26].all() and mask[26:].all()
    assert mask.sum() < 8 + 4


def test_per_kv_head_sharing_selects_per_group():
    probs = np.full((1, 4, 1, 20), 1e-3, dtype=np.float32)
    probs[0, :2, 0, 3] = 1.0  # KV head 0's query heads look at position 3
    probs[0, 2:, 0, 11] = 1.0  # KV head 1's query heads look at position 11
    scores = selection_scores(mx.array(probs), n_kv_heads=2, sharing="kv_head")
    mask = np.array(select_mask(scores, budget=1, window=2))
    assert mask.shape == (1, 2, 20)
    assert mask[0, 0, 3] and not mask[0, 0, 11]
    assert mask[0, 1, 11] and not mask[0, 1, 3]


def test_selection_rejects_empty_budget():
    with pytest.raises(ValueError):
        select_mask(mx.zeros((1, 1, 8)), budget=0, window=0)


# --- attention over a selection --------------------------------------------


@pytest.mark.parametrize("L", [1, 3])
def test_attention_probs_match_repeat_reference(L):
    q, k, _ = _qkv(Hq=6, Hkv=2, L=L, T=17)
    probs = attention_probs(q, k, scale=0.25)
    kr = mx.repeat(k, 3, axis=1)
    logits = (q * 0.25) @ kr.swapaxes(-1, -2)
    if L > 1:
        rows = mx.arange(17 - L, 17)[:, None]
        logits = mx.where(mx.arange(17)[None, :] <= rows, logits, -mx.inf)
    assert mx.allclose(probs, mx.softmax(logits, axis=-1), atol=1e-6).item()


def test_chunked_attention_probs_match_unchunked():
    q, k, _ = _qkv(Hq=4, Hkv=2, L=2, T=37)
    whole = attention_probs(q, k, 0.25, chunk=1 << 20)
    assert mx.allclose(attention_probs(q, k, 0.25, chunk=8), whole, atol=1e-6).item()


def test_all_true_mask_is_dense_attention():
    q, k, v = _qkv()
    dense = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.25)
    full = masked_attention(q, k, v, mx.ones((1, 1, 40), dtype=mx.bool_), 0.25)
    assert mx.allclose(dense, full, atol=1e-5).item()


def test_gathered_attention_equals_masked_attention():
    q, k, v = _qkv(T=64)
    probs = attention_probs(q, k, 0.25)
    mask = select_mask(selection_scores(probs, 2), budget=10, window=6)
    indices = indices_from_mask(mask)
    assert indices.shape == (1, 1, 16)
    assert mx.allclose(
        gathered_attention(q, k, v, indices, 0.25),
        masked_attention(q, k, v, mask, 0.25),
        atol=1e-5,
    ).item()


def test_indices_from_mask_refuses_ragged_rows():
    mask = mx.array([[[True, False, True], [True, False, False]]])
    with pytest.raises(ValueError):
        indices_from_mask(mask)


def test_mass_recall_bounds_and_complement():
    q, k, _ = _qkv(T=30)
    probs = attention_probs(q, k, 0.25)
    mask = select_mask(selection_scores(probs, 2), budget=5, window=3)
    inside = mass_recall(probs, mask)
    outside = mass_recall(probs, mx.logical_not(mask))
    assert mx.allclose(inside + outside, mx.ones_like(inside), atol=1e-5).item()
    assert mx.allclose(mass_recall(probs, mx.ones_like(mask)), mx.ones_like(inside), atol=1e-5).item()


def test_own_topk_maximizes_mean_recall_at_equal_size():
    q, k, _ = _qkv(T=80, seed=3)
    probs = attention_probs(q, k, 0.25)
    own = select_mask(selection_scores(probs, 2), budget=12, window=4)
    other_probs = attention_probs(_qkv(T=80, seed=4)[0], k, 0.25)
    other = select_mask(selection_scores(other_probs, 2), budget=12, window=4)
    assert mass_recall(probs, own).mean().item() >= mass_recall(probs, other).mean().item()


def test_relative_error_zero_on_identity():
    x = mx.random.normal((1, 2, 1, 8))
    assert relative_error(x, x).max().item() == 0.0


# --- traffic ceiling -------------------------------------------------------


def test_traffic_ceiling_is_one_when_nothing_is_skipped():
    traffic = DecodeTraffic(weight_bytes=1e9, kv_row_bytes=[1e3] * 4)
    assert traffic.ceiling(1000, budget=1024, window=128, score_pass=0.0) == pytest.approx(1.0)


def test_traffic_ceiling_matches_hand_calculation():
    traffic = DecodeTraffic(weight_bytes=100.0, kv_row_bytes=[1.0, 1.0])
    # dense: 100 + 2*1000; reused: 100 + 1000*1.5 + 10
    assert traffic.ceiling(1000, budget=6, window=4) == pytest.approx(2100 / 1610)


# --- probe on tiny models --------------------------------------------------


def _tiny_qwen():
    from mlx2.runtime.models.qwen3_5 import TextModelArgs
    from mlx2.runtime.models.qwen38_27b import TextModel

    args = TextModelArgs(
        model_type="qwen3_5", hidden_size=64, intermediate_size=64,
        num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, vocab_size=128, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=3, full_attention_interval=2,
        mtp_num_hidden_layers=0, partial_rotary_factor=0.5,
        rope_parameters=None, max_position_embeddings=4096,
    )
    mx.random.seed(7)
    model = TextModel(args)
    model.eval()
    # Sharpen attention so a small budget is visibly lossy on random weights.
    for layer in model.model.layers:
        attention = getattr(layer, "self_attn", None)
        if attention is not None:
            attention.v_proj.weight = attention.v_proj.weight * 8.0
            attention.o_proj.weight = attention.o_proj.weight * 8.0
    mx.eval(model.parameters())
    return model


def _tiny_muse():
    from mlx2.adapters.muse_glimmer_config import ModelArgs
    from mlx2.runtime.models.muse_glimmer import Model

    args = ModelArgs(
        hidden_size=64, num_hidden_layers=8, intermediate_size=64,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        vocab_size=128, sliding_window=8, max_position_embeddings=4096,
    )
    mx.random.seed(11)
    model = Model(args)
    model.eval()
    mx.eval(model.parameters())
    return model


def _run(model, tokens, prompt_len, probe_kwargs=None):
    """Prefill ``prompt_len`` tokens, then teacher-force the rest one at a time."""
    cache = model.make_cache()
    probe = None
    if probe_kwargs is not None:
        cache, probe = install_probe(cache, lambda n: ReuseProbe(n_layers=n, **probe_kwargs))
    ids = mx.array([tokens], dtype=mx.uint32)
    mx.eval(model(ids[:, :prompt_len], cache=cache))
    if probe is not None:
        probe.armed = True
    logits = []
    for pos in range(prompt_len, len(tokens)):
        out = model(ids[:, pos : pos + 1], cache=cache)
        mx.eval(out)
        logits.append(out[0, -1])
        if probe is not None:
            probe.flush()
    return mx.stack(logits), probe, cache


TOKENS = [int(x) for x in np.random.default_rng(5).integers(1, 120, size=72)]


@pytest.mark.parametrize("factory", [_tiny_qwen, _tiny_muse], ids=["qwen", "muse"])
def test_install_swaps_only_full_attention_caches(factory):
    model = factory()
    fresh = model.make_cache()
    swapped, probe = install_probe(fresh, lambda n: ReuseProbe(mode="shadow", n_layers=n))
    fa = [i for i, c in enumerate(fresh) if type(c) is KVCache]
    assert probe.n_layers == len(fa) >= 2
    for i, (before, after) in enumerate(zip(fresh, swapped)):
        if i in fa:
            assert isinstance(after, ReuseProbeKVCache)
        else:
            assert after is before


@pytest.mark.parametrize("factory", [_tiny_qwen, _tiny_muse], ids=["qwen", "muse"])
def test_substitute_with_everything_kept_equals_dense(factory):
    model = factory()
    dense, _, _ = _run(model, TOKENS, 48)
    sub, probe, _ = _run(model, TOKENS, 48, dict(mode="substitute", budget=64, window=64))
    steps = len(TOKENS) - 48
    assert probe.counts["substituted"] == len(probe.targets) * steps
    assert probe.counts["source"] == len(probe.sources) * steps
    assert mx.allclose(dense, sub, atol=1e-4).item()


@pytest.mark.parametrize("factory", [_tiny_qwen, _tiny_muse], ids=["qwen", "muse"])
def test_substitute_with_small_budget_changes_logits(factory):
    model = factory()
    dense, _, _ = _run(model, TOKENS, 48)
    sub, probe, _ = _run(model, TOKENS, 48, dict(mode="substitute", budget=4, window=4))
    assert probe.counts["substituted"] == len(probe.targets) * (len(TOKENS) - 48)
    assert not mx.allclose(dense, sub, atol=1e-3).item()


@pytest.mark.parametrize("factory", [_tiny_qwen, _tiny_muse], ids=["qwen", "muse"])
def test_shadow_leaves_logits_bit_identical_and_records_metrics(factory):
    model = factory()
    dense, _, _ = _run(model, TOKENS, 48)
    variants = (Variant("token", "shared"), Variant("token", "kv_head"), Variant("block", "shared"))
    shadow, probe, _ = _run(
        model, TOKENS, 48,
        dict(mode="shadow", budget=8, window=4, variants=variants, block_size=4),
    )
    assert mx.array_equal(dense, shadow).item()
    steps = len(TOKENS) - 48
    assert probe.counts["shadowed"] == probe.n_layers * steps
    assert len(probe.rows) == probe.n_layers * steps
    targets = [r for r in probe.rows if r["role"] == "target"]
    assert targets
    for row in targets:
        for name in ("recall/token/shared", "recall/token/kv_head", "recall/block/shared",
                     "oracle_recall", "window_recall"):
            assert 0.0 <= row[name] <= 1.0 + 1e-5
        # Own top-k maximizes mean recall at equal size; a reused set cannot beat it.
        assert row["oracle_recall"] + 1e-5 >= row["recall/token/shared"]
        assert row["recall/token/shared"] + 1e-5 >= row["window_recall"]
        assert row["relerr"] >= 0.0
        assert all(f"from/{s}" in row for s in range(row["layer"]))


def _peaked_kv(T=64, D=16, peak=10, seed=0):
    """Keys where position ``peak`` dominates attention for a matching query."""
    mx.random.seed(seed)
    k = mx.random.normal((1, 2, T, D)) * 0.1
    v = mx.random.normal((1, 2, T, D))
    direction = mx.ones((D,)) / D**0.5
    k = mx.concatenate([k[:, :, :peak], mx.broadcast_to(direction * 12.0, (1, 2, 1, D)), k[:, :, peak + 1 :]], axis=2)
    q = mx.broadcast_to(direction * 12.0, (1, 4, 1, D))
    return q, k, v


def test_probe_selection_follows_the_source_attention_peak():
    q, k, v = _peaked_kv()
    probe = ReuseProbe(mode="shadow", n_layers=2, budget=4, window=4)
    probe.armed = True
    assert probe.attend(0, q, k, v, 0.25) is None
    assert probe.attend(1, q, k, v, 0.25) is None
    probe.flush()
    target = [r for r in probe.rows if r["role"] == "target"][0]
    assert target["recall/token/shared"] > 0.9  # the reused set holds the peak
    assert target["window_recall"] < 0.1  # the window alone does not
    assert target["relerr"] < 0.1

    sub = ReuseProbe(mode="substitute", n_layers=2, budget=4, window=4)
    sub.armed = True
    assert sub.attend(0, q, k, v, 0.25) is None
    out = sub.attend(1, q, k, v, 0.25)
    dense = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.25)
    assert relative_error(dense, out).max().item() < 0.1
    assert sub.counts == {"source": 1, "substituted": 1}


def test_probe_passes_prefill_through_untouched():
    model = _tiny_qwen()
    _, probe, _ = _run(model, TOKENS, 48, dict(mode="substitute", budget=4, window=4))
    assert probe.counts["passthrough"] == probe.n_layers  # one prefill call per FA layer


def test_probe_rejects_sources_without_first_layer():
    with pytest.raises(ValueError):
        ReuseProbe(mode="shadow", n_layers=4, sources=(1,))


def test_install_refuses_used_caches():
    model = _tiny_qwen()
    cache = model.make_cache()
    model(mx.array([[1, 2, 3]], dtype=mx.uint32), cache=cache)
    with pytest.raises(ValueError):
        install_probe(cache, lambda n: ReuseProbe(mode="shadow", n_layers=n))


def test_window_mask_marks_recent_positions():
    mask = np.array(window_mask((1, 1, 10), 3))[0, 0]
    assert mask.tolist() == [False] * 7 + [True] * 3
