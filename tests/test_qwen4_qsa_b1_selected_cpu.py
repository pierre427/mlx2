"""CPU tests for the one-token selected-block QSA prototype.

The kernel is Metal-only (its numerics are a GPU check in
qualification/runs/fn-lily-b1-selected-20261029).  These pin the switch, the
admission refusals and the visible-token rule the kernel implements: complete
selected blocks plus the incomplete tail, which must equal
``QSASelection.dense_mask`` for one row without left padding.
"""

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models import qwen4_exp as Q
from mlx2.runtime.models import qwen4_qsa_b1_selected as B1


def test_env_parsing_and_default_off(monkeypatch):
    monkeypatch.delenv(B1.B1_SELECTED_ENV, raising=False)
    assert B1.enabled_from_env() is False
    monkeypatch.setenv(B1.B1_SELECTED_ENV, "1")
    assert B1.enabled_from_env() is True
    monkeypatch.setenv(B1.B1_SELECTED_ENV, "sometimes")
    with pytest.raises(ValueError):
        B1.enabled_from_env()


def _arrays(q_heads=24, kv_heads=2, cap=64, dtype=mx.bfloat16, rows=1, dim=256):
    q = mx.zeros((1, q_heads, rows, dim), dtype=dtype)
    keys = mx.zeros((1, kv_heads, cap, dim), dtype=dtype)
    return q, keys, keys, mx.zeros((1, 1, 8), dtype=mx.uint32)


def test_refusal_accepts_flash_next_geometry():
    q, k, v, ids = _arrays()
    assert B1.refusal(q, k, v, ids, 40) is None


def test_refusal_handles_empty_cache_geometry_and_wrong_index_dtype():
    q, k, v, ids = _arrays()
    assert B1.refusal(q, mx.array(0), mx.array(0), ids, 40) == "cache layout"
    assert "multiple" in B1.refusal(q, k[:, :0], v[:, :0], ids, 40)
    assert B1.refusal(q, k, v, ids.astype(mx.int64), 40) == "block ids dtype"


def test_selection_requires_four_token_blocks_and_proven_causal_masks():
    from types import SimpleNamespace
    from mlx2.runtime.models.qsdpa_verify_metal import register_causal_mask

    selection = SimpleNamespace(
        block_size=4, batch=1, length=1, physical_width=41, causal_mask=None,
    )
    assert B1.selection_refusal(selection) is None
    selection.block_size = 8
    assert B1.selection_refusal(selection) == "block size"
    selection.block_size = 4
    selection.causal_mask = mx.ones((1, 1, 1, 41), dtype=mx.bool_)
    assert B1.selection_refusal(selection) == "custom attention mask"
    register_causal_mask(selection.causal_mask, None)
    assert B1.selection_refusal(selection) is None


@pytest.mark.parametrize(
    ("kwargs", "q_pos", "needle"),
    [
        ({"rows": 2}, 40, "one query row"),
        ({"dim": 128}, 40, "head dim"),
        ({"q_heads": 25}, 40, "multiple"),
        ({"dtype": mx.float32}, 40, "dtype"),
        ({}, 64, "query position"),
    ],
)
def test_refusal_reasons(kwargs, q_pos, needle):
    q, k, v, ids = _arrays(**kwargs)
    assert needle in B1.refusal(q, k, v, ids, q_pos)


def _kernel_visible(selected, q_pos, block=B1.BLOCK_SIZE):
    """The kernel's rule, in numpy: complete selected blocks + the tail."""
    visible = set()
    for b in selected:
        if b * block + block - 1 <= q_pos:
            visible.update(range(b * block, b * block + block))
    visible.update(range((q_pos + 1) // block * block, q_pos + 1))
    return visible


@pytest.mark.parametrize("seed", range(12))
def test_kernel_visible_set_equals_dense_mask(seed):
    rng = np.random.default_rng(seed)
    q_pos = int(rng.integers(40, 400))
    total = q_pos + 1
    n_blocks = total // 4
    k = min(16, n_blocks)
    selected = rng.choice(n_blocks, size=k, replace=False)
    selection = Q.QSASelection(
        kind="explicit", batch=1, length=1, block_size=4,
        raw_block_ids=mx.array(selected[None, None, :], dtype=mx.uint32),
        valid_blocks=((mx.arange(n_blocks) * 4 + 3)[None, None, :] <= q_pos),
        q_positions=mx.array([[q_pos]]),
        token_positions=mx.arange(total)[None, :],
        offset=q_pos, physical_width=total, n_blocks=n_blocks,
    )
    mask = np.array(selection.dense_mask()).reshape(-1)
    assert set(np.nonzero(mask)[0].tolist()) == _kernel_visible(selected.tolist(), q_pos)


def test_policy_selects_the_candidate_through_its_environment():
    from mlx2.adapters.flash_next_policy import FlashNextPolicy

    off = FlashNextPolicy()
    assert "MLX_QWEN4_QSA_B1_SELECTED" not in off.environment()
    assert "qsa_b1_selected" not in off.as_dict()
    on = FlashNextPolicy.from_mapping({"qsa_b1_selected": True})
    assert on.environment()["MLX_QWEN4_QSA_B1_SELECTED"] == "1"
    assert on.as_dict()["qsa_b1_selected"] is True


def _qual_settings(max_context, **adapter_policy):
    from mlx2.adapters.flash_next_policy import FlashNextPolicy

    return {"environment": FlashNextPolicy().environment(), "max_context": max_context,
            "max_lanes": 4, "mtp": False, "speculation": "ordinary",
            "adapter_policy": adapter_policy}


def test_qualifier_requires_engagement_past_the_indexer_budget():
    from mlx2.qualification import required_feature_checks, selected_not_observed_features

    assert "feature_qsa_b1_selected" in required_feature_checks(
        _qual_settings(65536, qsa_b1_selected=True))
    assert "feature_qsa_b1_selected" not in required_feature_checks(_qual_settings(65536))
    short = _qual_settings(2304, qsa_b1_selected=True)
    assert "feature_qsa_b1_selected" not in required_feature_checks(short)
    assert "indexer budget" in selected_not_observed_features(short)["feature_qsa_b1_selected"]


@pytest.mark.parametrize("seed", range(8))
def test_kernel_visible_set_with_left_padding(seed):
    # A batch cache row: logical position = physical - left padding.  The
    # kernel applies its logical rule and reads physical row logical + pad.
    rng = np.random.default_rng(100 + seed)
    pad = int(rng.integers(1, 9))
    q_pos = int(rng.integers(40, 400))          # logical
    total = q_pos + 1 + pad                     # physical width
    logical_width = q_pos + 1
    n_blocks = total // 4
    valid_n = logical_width // 4
    k = min(16, valid_n)
    selected = rng.choice(valid_n, size=k, replace=False)
    left = mx.array([pad])
    selection = Q.QSASelection(
        kind="explicit", batch=1, length=1, block_size=4,
        raw_block_ids=mx.array(selected[None, None, :], dtype=mx.uint32),
        valid_blocks=((mx.arange(n_blocks) * 4 + 3)[None, None, :] <= q_pos),
        q_positions=mx.array([[q_pos]]),
        token_positions=mx.arange(total)[None, :] - left[:, None],
        left_padding=left, offset=mx.array([q_pos]),
        physical_width=total, n_blocks=n_blocks,
    )
    mask = np.array(selection.dense_mask()).reshape(-1)
    physical = {t + pad for t in _kernel_visible(selected.tolist(), q_pos)}
    assert set(np.nonzero(mask)[0].tolist()) == physical


def test_meta_keeps_device_offsets_on_the_device():
    meta = B1._meta(mx.array([123]), mx.array([5]), 4096, 31)
    assert meta.dtype == mx.int32 and meta.tolist() == [123, 4096, 5, 31]
    assert B1._meta(7, 0, 64, 1).tolist() == [7, 64, 0, 1]


def test_selected_count_and_chunks_are_runtime_values_not_templates(monkeypatch):
    """Below the indexer budget the selected count grows every block; a
    template value per count compiled a Metal library per count (omlx#5025).
    Every count must share one partials and one combine instantiation."""
    seen = {"partials": set(), "combine": set()}

    def recorder(kind):
        def call(**kw):
            seen[kind].add(tuple(kw["template"]))
            return [mx.zeros(shape, dtype) for shape, dtype in zip(kw["output_shapes"], kw["output_dtypes"])]
        return call

    monkeypatch.setattr(B1, "_kernel", lambda variant: recorder("partials"))
    monkeypatch.setattr(B1, "_combine_kernel", lambda: recorder("combine"))
    q, keys, values = _arrays(cap=2112)[:3]
    for nsel in (1, 2, 7, 8, 9, 63, 64, 511, 512):
        ids = mx.arange(nsel, dtype=mx.uint32).reshape(1, 1, nsel)
        m, l, o = B1.partials(q, keys, values, ids, q_pos=4 * nsel + 2, scale=1 / 16)
        B1.combine(m, l, o, dtype=q.dtype)
    assert len(seen["partials"]) == 1 and len(seen["combine"]) == 1
    names = {name for template in seen["partials"] | seen["combine"] for name, _ in template}
    assert not names & {"NSEL", "NCH"}


@pytest.mark.parametrize("pad", [0, 3])
def test_own_position_is_the_row_update_and_fetch_just_wrote(pad):
    # fn-b1-coldwarm-20261029: a batch cache's offset is advanced in place by
    # update_and_fetch, so an alias taken before the update (the selection's
    # offset) already counts the query token.  The query is row _idx - 1.
    from mlx2.runtime.models.cache import BatchKVCache

    cache = BatchKVCache([pad])
    kv = mx.zeros((1, 2, 10 + pad, 8))
    cache.update_and_fetch(kv, kv)
    alias = cache.offset
    one = mx.zeros((1, 2, 1, 8))
    cache.update_and_fetch(one, one)
    own = B1.own_position(cache, cache.left_padding)
    assert own.tolist() == [10]                 # 11 logical tokens: positions 0..10
    assert cache._idx - 1 == 10 + pad           # physical row = logical + pad
    assert int(alias.item()) == 11              # the alias counts the query token


def test_own_position_of_a_plain_cache():
    class Plain:
        offset = 41

    assert B1.own_position(Plain()) == 40


def test_module_snapshots_its_import_environment():
    """import_env.assert_profile_applied only sees modules that snapshot; a
    module that reads its switch at import without one could serve the other
    decode law while the receipt says selected."""
    from mlx2.runtime.models import import_env

    assert B1.__name__ in import_env._SNAPSHOTS


def test_mask_builders_do_not_run_on_a_call_the_b1_kernel_serves(monkeypatch):
    """The fused-rows / lazy-mask / scatter counters are the qualifier's
    observed-used evidence; a mask built before the B1 decision and then
    discarded advanced them for work that never ran."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from qsa_oracle import tiny_args

    from mlx2.runtime import round_levers as levers
    from mlx2.runtime.models import qwen4_attn_rows as rows

    mx.random.seed(3)
    attn = Q.Attention(tiny_args(indexer_budget=8))  # explicit selection past 2 blocks
    attn.eval()
    mx.eval(attn.parameters())
    cache = Q.QSAKVCache(attn.indexer.summary_identity)
    hidden = mx.random.normal((1, 40, 16))
    out = attn(hidden, Q.qsa_attention_mask(hidden, cache), cache)
    mx.eval(out, cache.state)

    calls = []

    def fake_attention(q, keys, values, block_ids, *, q_pos, scale, gate=None, left_pad=0, **kw):
        calls.append(int(q.shape[1]))
        return mx.zeros((1, 1, q.shape[1] * q.shape[-1]), dtype=q.dtype)

    # The kernel is Metal-only and refuses the tiny fp32 geometry; only the
    # order of the decision and the mask build is under test.
    monkeypatch.setattr(B1, "attention", fake_attention)
    monkeypatch.setattr(B1, "refusal", lambda *a, **k: None)
    previous = B1.set_enabled(True)
    try:
        monkeypatch.setattr(Q.Attention, "_attn_rows_admit", lambda self, x, cache, length: True)
        monkeypatch.setattr(Q.Attention, "_attn_rows_prep", lambda self, qg, k_flat, offset: None)
        monkeypatch.setattr(rows, "qsa_mask_supported", lambda selection: None)
        monkeypatch.setattr(rows, "qsa_mask", lambda selection: selection.dense_mask())
        rows_was = rows.enabled()
        rows.set_enabled(True)
        try:
            before = (
                rows.status()["counts"].get("mask_rows", 0),
                levers.counters().get("qsa_scatter_chosen_calls", 0),
            )
            step = mx.random.normal((1, 1, 16))
            out = attn(step, Q.qsa_attention_mask(step, cache), cache)
            mx.eval(out, cache.state)
            after = (
                rows.status()["counts"].get("mask_rows", 0),
                levers.counters().get("qsa_scatter_chosen_calls", 0),
            )
        finally:
            rows.set_enabled(rows_was)
    finally:
        B1.set_enabled(previous)
    assert len(calls) == 1, "the B1 kernel path was not taken"
    assert after == before


def test_selected_candidate_preserves_custom_token_exclusions(monkeypatch):
    from qsa_oracle import tiny_args

    mx.random.seed(7)
    attn = Q.Attention(tiny_args(indexer_budget=8))
    attn.eval()
    hidden = mx.random.normal((1, 40, 16))
    step = mx.random.normal((1, 1, 16))
    # Only the newly appended token is visible. A block-id-only kernel
    # cannot represent the exclusions inside its selected complete blocks.
    mask = (mx.arange(41) == 40)[None, None, None, :]

    def run():
        cache = Q.QSAKVCache(attn.indexer.summary_identity)
        mx.eval(attn(hidden, Q.qsa_attention_mask(hidden, cache), cache), cache.state)
        result = attn(step, mask, cache)
        mx.eval(result, cache.state)
        return np.array(result)

    previous = B1.set_enabled(False)
    try:
        reference = run()
        monkeypatch.setattr(B1, "refusal", lambda *args, **kwargs: None)

        def forbidden(*args, **kwargs):
            raise AssertionError("selected-block kernel discarded a custom mask")

        monkeypatch.setattr(B1, "attention", forbidden)
        B1.set_enabled(True)
        before = B1.status()["declines"].get("custom attention mask", 0)
        actual = run()
        assert B1.status()["declines"]["custom attention mask"] == before + 1
        np.testing.assert_array_equal(actual, reference)
    finally:
        B1.set_enabled(previous)
