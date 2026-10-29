"""Synthetic CPU contracts, not checkpoint or Metal qualification.

Base source 89e6938ea58a0060026815c94e0ed4d33729d775 plus this patch.
Tiny random fixtures use qsa_oracle.tiny_args and the seeds below.
"""

from dataclasses import replace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from qsa_oracle import tiny_args

from mlx2.adapters.flash_next_policy import FlashNextPolicy
from mlx2.runtime.models import qwen4_attn_rows as rows
from mlx2.runtime.models import qwen4_exp as Q
from mlx2.runtime.models import qwen4_qsa_visibility as visibility
from mlx2.runtime.models.base import create_causal_mask
from mlx2.runtime.models.qwen4_qsa_nax import compact_token_validity


@pytest.fixture(autouse=True)
def _mode(monkeypatch):
    monkeypatch.setattr(Q, "_QSA_LAZY_CAUSAL_MASK", True)
    visibility.status(reset=True)
    yield
    visibility.status(reset=True)


def test_policy_is_explicit_unqualified_and_default_off():
    base = FlashNextPolicy()
    assert "qsa_lazy_causal_mask" not in base.as_dict()
    assert "MLX_QWEN4_QSA_LAZY_CAUSAL_MASK" not in base.environment()
    chosen = FlashNextPolicy(qsa_lazy_causal_mask=True)
    assert chosen.as_dict()["qsa_lazy_causal_mask"] is True
    assert chosen.environment()["MLX_QWEN4_QSA_LAZY_CAUSAL_MASK"] == "1"
    with pytest.raises(ValueError, match="boolean"):
        FlashNextPolicy(qsa_lazy_causal_mask=1)
    assert visibility.status()["qualification"] == "unqualified"


def _selection(size, offset, length, padding, stride=1):
    batch = len(padding)
    total = offset + max(padding) + length
    blocks = total // size
    qpos = offset + max(padding) - np.array(padding)[:, None] + np.arange(length)
    logical = np.arange(total)[None] - np.array(padding)[:, None]
    ids = np.arange(0, blocks, stride, dtype=np.uint32)
    ids = np.broadcast_to(ids, (batch, length, ids.size)).copy()
    valid = ((np.arange(blocks) + 1) * size - 1)[None, None] <= qpos[..., None]
    causal = ((logical[:, None] >= 0) & (logical[:, None] <= qpos[..., None]))[:, None]
    pad = mx.array(padding, dtype=mx.int32)
    selection = Q.QSASelection(
        kind="explicit",
        batch=batch,
        length=length,
        block_size=size,
        raw_block_ids=mx.array(ids),
        valid_blocks=mx.array(valid),
        q_positions=mx.array(qpos, dtype=mx.int32),
        token_positions=mx.array(logical, dtype=mx.int32),
        causal_mask=visibility.QSAVisibility(length, total, batch, pad),
        left_padding=pad,
        offset=mx.array(qpos[:, 0], dtype=mx.int32),
        physical_width=total,
        n_blocks=blocks,
        scatter_chosen=True,
    )
    return selection, mx.array(causal)


@pytest.mark.parametrize("size", [2, 4, 8, 64])
@pytest.mark.parametrize("length", [1, 3, 9])
@pytest.mark.parametrize("padding", [[0], [0, 1, 5]])
@pytest.mark.parametrize("stride", [1, 2])
def test_dense_and_compact_visibility_match_reference(size, length, padding, stride):
    # Misaligned total, unequal row lengths and incomplete tails are deliberate.
    selection, causal = _selection(size, 65, length, padding, stride)
    reference = replace(selection, causal_mask=causal)
    assert mx.array_equal(selection.dense_mask(), reference.dense_mask()).item()
    compact = selection.compact_blocks()
    assert compact.causal_mask is None
    *_, positions, valid = compact_token_validity(compact)
    actual = np.zeros((len(padding), length, selection.physical_width), dtype=bool)
    for b in range(len(padding)):
        for q in range(length):
            actual[b, q, np.asarray(positions[b, q])[np.asarray(valid[b, q])]] = True
    np.testing.assert_array_equal(actual[:, None], np.asarray(reference.dense_mask()))
    assert visibility.status()["counts"].get("dense_materializations", 0) == 0


def test_unfinished_block_cannot_expose_future_tokens():
    selection, causal = _selection(4, 7, 3, [0])
    selection = replace(selection, raw_block_ids=mx.ones((1, 3, 1), dtype=mx.uint32))
    naive = replace(selection, causal_mask=None).dense_mask()
    assert naive[0, 0, 0, 8:].tolist() == [True, True]
    assert selection.dense_mask()[0, 0, 0, 8:].tolist() == [False, False]
    assert mx.array_equal(
        selection.dense_mask(), replace(selection, causal_mask=causal).dense_mask()
    ).item()


@pytest.mark.parametrize("length", [1, 3, 9])
def test_implicit_all_materializes_once_and_preserves_registration(length):
    pad = mx.array([0, 3], dtype=mx.int32)
    proof = visibility.QSAVisibility(length, 13 + length, 2, pad)
    selection = Q.QSASelection(
        kind="implicit_all",
        batch=2,
        length=length,
        block_size=4,
        physical_width=13 + length,
        n_blocks=(13 + length) // 4,
        causal_mask=proof,
        left_padding=pad,
    )
    expected = create_causal_mask(length, offset=13, left_padding=pad)
    assert mx.array_equal(selection.dense_mask(), expected).item()
    assert selection.dense_mask() is proof.materialize()
    assert visibility.status()["counts"]["dense_materializations"] == 1
    from mlx2.runtime.models.qsdpa_verify_metal import registered_mask_kind

    assert registered_mask_kind(selection.dense_mask()) == "left_padded"


def test_mask_only_and_custom_masks_keep_exact_exclusions():
    selection, _ = _selection(4, 7, 3, [0])
    custom = mx.zeros((1, 1, 3, 10), dtype=mx.bool_)
    assert not mx.any(replace(selection, causal_mask=custom).dense_mask()).item()
    assert replace(selection, causal_mask=custom).compact_blocks().causal_mask is custom
    only = Q.QSASelection(
        kind="mask_only", batch=1, length=3, block_size=4, passthrough_mask=custom
    )
    assert only.dense_mask() is custom


def test_unknown_special_and_quantized_cache_masks_are_not_reinterpreted():
    sentinel = mx.zeros((1, 1, 3, 7), dtype=mx.bool_)

    class Custom(Q.QSAKVCache):
        def make_mask(self, *args, **kwargs):
            return sentinel

    hidden = mx.zeros((1, 3, 16))
    assert Q.qsa_attention_mask(hidden, Custom()) is sentinel
    for cache in (Q.QSAQuantizedKVCache(), Q.QSAKVCache()):
        if type(cache) is Q.QSAKVCache:
            cache._pld_ordinary_mask_padding = mx.array([0])
        assert not isinstance(
            Q.qsa_attention_mask(hidden, cache), visibility.QSAVisibility
        )


def test_producer_and_disable_switch(monkeypatch):
    hidden = mx.zeros((1, 3, 16))
    assert isinstance(
        Q.qsa_attention_mask(hidden, Q.QSAKVCache()), visibility.QSAVisibility
    )
    assert isinstance(Q.qsa_attention_mask(hidden, None), visibility.QSAVisibility)
    assert Q.qsa_attention_mask(hidden[:, :1], Q.QSAKVCache()) is None
    monkeypatch.setattr(Q, "_QSA_LAZY_CAUSAL_MASK", False)
    assert isinstance(Q.qsa_attention_mask(hidden, Q.QSAKVCache()), mx.array)


def test_visibility_preserves_nax_admission_and_invariant_fallback(monkeypatch):
    from types import SimpleNamespace

    hidden = mx.zeros((1, 3, 16))
    for cache in (None, Q.QSAKVCache()):
        monkeypatch.setattr(Q, "_QSA_LAZY_CAUSAL_MASK", False)
        reference = Q.qsa_attention_mask(hidden, cache)
        monkeypatch.setattr(Q, "_QSA_LAZY_CAUSAL_MASK", True)
        proof = Q.qsa_attention_mask(hidden, cache)
        assert Q._qsa_nax_plain_mask(
            SimpleNamespace(causal_mask=proof)
        ) == Q._qsa_nax_plain_mask(SimpleNamespace(causal_mask=reference, batch=1))
    monkeypatch.setattr(Q._invariant, "active", lambda: True)
    assert isinstance(Q.qsa_attention_mask(hidden, Q.QSAKVCache()), mx.array)


def test_visibility_refuses_mismatched_selection_geometry():
    selection, _ = _selection(4, 7, 3, [0])
    with pytest.raises(ValueError, match="batch"):
        replace(selection, causal_mask=visibility.QSAVisibility(3, 10, 2))
    with pytest.raises(ValueError, match="physical width"):
        replace(selection, causal_mask=visibility.QSAVisibility(2, 10, 1))


def test_fused_kernel_gets_positions_without_a_causal_array(monkeypatch):
    selection, _ = _selection(4, 7, 3, [0])
    captured = {}

    def kernel(**kwargs):
        captured.update(kwargs)

        def launch(**call):
            captured["call"] = call
            return [mx.zeros(call["output_shapes"][0], dtype=mx.bool_)]

        return launch

    monkeypatch.setattr(mx.fast, "metal_kernel", kernel)
    monkeypatch.setattr(rows, "_KERNELS", {})
    monkeypatch.setattr(rows, "metal_ready", lambda: True)
    assert rows.qsa_mask_supported(selection) is None
    rows.qsa_mask(selection)
    assert "v = v && (tl <= qp);" in captured["source"]
    assert "@VISIBILITY@" not in captured["source"]
    assert "causal" not in captured["input_names"]
    assert len(captured["call"]["inputs"]) == 4
    assert visibility.status()["counts"].get("dense_materializations", 0) == 0


@pytest.mark.parametrize("shortcircuit", [False, True])
def test_tiny_model_chunked_prefill_and_decode_match_reference(
    monkeypatch, shortcircuit
):
    mx.random.seed(27)
    model = Q.TextModel(tiny_args(indexer_budget=8))
    model.eval()
    mx.eval(model.parameters())
    caches = [model.make_cache(), model.make_cache()]
    monkeypatch.setattr(Q, "_QSA_DENSE_SHORTCIRCUIT", shortcircuit)
    for width, start in ((3, 0), (7, 3), (9, 10), (1, 19), (3, 20)):
        tokens = (mx.arange(start, start + width) % 63)[None]
        outputs = []
        for enabled, cache in zip((False, True), caches):
            monkeypatch.setattr(Q, "_QSA_LAZY_CAUSAL_MASK", enabled)
            out = model(tokens, cache)
            mx.eval(out, [entry.state for entry in cache])
            outputs.append(out)
        assert mx.array_equal(*outputs).item()
        left = tree_flatten([entry.state for entry in caches[0]])
        right = tree_flatten([entry.state for entry in caches[1]])
        assert [key for key, _ in left] == [key for key, _ in right]
        for (_, a), (_, b) in zip(left, right):
            assert mx.array_equal(a, b).item()


def test_selected_feature_requires_run_local_engagement():
    from mlx2.qualification import required_feature_checks

    assert "feature_qsa_lazy_causal_mask" in required_feature_checks(
        {"adapter_policy": {"qsa_lazy_causal_mask": True}}
    )


@pytest.mark.parametrize("gather", [False, True])
def test_ragged_batch_attention_and_cache_match_reference(monkeypatch, gather):
    import copy

    mx.random.seed(31)
    attention = Q.Attention(tiny_args(indexer_budget=8))
    attention.eval()
    mx.eval(attention.parameters())
    row_caches = []
    for length in (13, 21):
        cache = Q.QSAKVCache(attention.indexer.summary_identity)
        hidden = mx.random.normal((1, length, 16))
        out = attention(hidden, Q.qsa_attention_mask(hidden, cache), cache)
        mx.eval(out, cache.state)
        row_caches.append(cache)
    batch = Q.BatchQSAKVCache.merge(row_caches)
    caches = [batch, copy.deepcopy(batch)]
    monkeypatch.setattr(Q, "_QSA_GATHER_KV", gather)
    monkeypatch.setattr(Q, "_QSA_GATHER_MIN_CONTEXT", 0)
    monkeypatch.setattr(Q, "_QSA_GATHER_MIN_QUERY", 1)
    monkeypatch.setattr(Q, "_QSA_GATHER_MAX_QUERY", 16)
    visibility.status(reset=True)
    for width in (3, 1, 7):
        hidden = mx.random.normal((2, width, 16))
        outputs = []
        for enabled, cache in zip((False, True), caches):
            monkeypatch.setattr(Q, "_QSA_LAZY_CAUSAL_MASK", enabled)
            mask = Q.qsa_attention_mask(hidden, cache)
            out = attention(hidden, mask, cache)
            mx.eval(out, cache.state)
            outputs.append(out)
        assert mx.array_equal(*outputs).item()
        for field in ("keys", "values", "index_keys", "left_padding", "offset"):
            assert mx.array_equal(
                getattr(caches[0], field), getattr(caches[1], field)
            ).item()
    counts = visibility.status()["counts"]
    assert counts.get("dense_materializations", 0) == 0
    assert counts["compact_calls" if gather else "dense_mask_calls"] > 0


def test_qualification_observation_uses_deltas_not_metadata():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "qsa_lazy_qualifier",
        Path(__file__).resolve().parents[1] / "scripts" / "qualify_serving.py",
    )
    qualifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qualifier)

    def receipt(**counts):
        return {"execution": {"qsa_lazy_causal_mask": {"counts": counts}}}

    initial = receipt(dense_mask_calls=5, metadata_forwards=20)
    unchanged = receipt(dense_mask_calls=5, metadata_forwards=30)
    engaged = receipt(dense_mask_calls=6, compact_calls=2)
    assert (
        qualifier.default_on_observations(unchanged, initial)["qsa_lazy_causal_mask"]
        == 0
    )
    assert (
        qualifier.default_on_observations(engaged, initial)["qsa_lazy_causal_mask"] == 3
    )


def test_tiny_native_mtp_head_matches_reference(monkeypatch):
    # Synthetic MTP fixture (test_batched_mtp._tiny_qwen4_model), seed 41.
    from test_batched_mtp import _tiny_qwen4_model

    mx.random.seed(41)
    model = _tiny_qwen4_model()
    model.eval()
    caches = [model.make_mtp_cache(), model.make_mtp_cache()]
    for width in (3, 1, 5):
        hidden = mx.random.normal((1, width, 64))
        tokens = mx.arange(width)[None]
        outputs = []
        for enabled, cache in zip((False, True), caches):
            monkeypatch.setattr(Q, "_QSA_LAZY_CAUSAL_MASK", enabled)
            out = model.mtp_step(hidden, tokens, cache)
            mx.eval(out, [entry.state for entry in cache])
            outputs.append(out)
        for a, b in zip(*outputs):
            assert mx.array_equal(a, b).item()
        for (_, a), (_, b) in zip(
            tree_flatten(caches[0][0].state), tree_flatten(caches[1][0].state)
        ):
            assert mx.array_equal(a, b).item()


def test_qualification_harness_pin_matches_reviewed_producer():
    import hashlib
    from pathlib import Path

    from mlx2.qualification import APPROVED_QUALIFICATION_HARNESS

    path = Path(__file__).resolve().parents[1] / "scripts" / "qualify_serving.py"
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == APPROVED_QUALIFICATION_HARNESS["sha256"]
    )
