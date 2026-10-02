# oMLX #4070 batched one-token sparse QSA arm (Apache-2.0 source, mlx2 port);
# see provenance/omlx-4070-batched-qsa.json.
import mlx.core as mx
import pytest

from mlx2.adapters.flash_next_policy import FlashNextPolicy
from mlx2.runtime.models import qwen4_exp as Q
from mlx2.runtime.models.qwen4_exp import Attention, BatchQSAKVCache, QSAKVCache


@pytest.fixture(autouse=True)
def _restore_mode():
    saved = (Q._QSA_BATCH_DECODE_SPARSE, Q._QSA_BATCH_DECODE_SPARSE_MIN_CONTEXT)
    Q.qsa_batch_decode_sparse_status(reset=True)
    yield
    Q._QSA_BATCH_DECODE_SPARSE, Q._QSA_BATCH_DECODE_SPARSE_MIN_CONTEXT = saved
    Q.qsa_batch_decode_sparse_status(reset=True)


def _attention():
    from qsa_oracle import tiny_args

    mx.random.seed(7)
    attention = Attention(tiny_args(indexer_budget=8))  # top-2 blocks of 4
    attention.eval()
    mx.eval(attention.parameters())
    return attention


def _batched_cache(attention, lengths):
    """Prefill each row as B1 and merge them into one left-padded batch.

    A row's prompt depends only on its length, so a row prefilled alone
    matches the same row inside a batch."""
    rows = []
    for length in lengths:
        mx.random.seed(1000 + length)
        row = QSAKVCache(attention.indexer.summary_identity)
        hidden = mx.random.normal((1, length, 16))
        out = attention(hidden, row.make_mask(length, return_array=True, window_size=None), row)
        mx.eval(out, row.state)
        rows.append(row)
    cache = BatchQSAKVCache.merge(rows)
    mx.eval(cache.state)
    return cache


def _decode(attention, cache, steps, seed=23):
    mx.random.seed(seed)
    batch = cache.left_padding.shape[0]
    outputs = []
    for _ in range(steps):
        hidden = mx.random.normal((batch, 1, 16))
        mask = cache.make_mask(1, return_array=True)
        out = attention(hidden, mask, cache)
        mx.eval(out, cache.state)
        outputs.append(out)
    return outputs


def test_default_is_off_and_absent_from_policy_receipts():
    assert Q._env_batch_decode_sparse_mode() in Q._QSA_BATCH_DECODE_SPARSE_MODES
    policy = FlashNextPolicy()
    assert policy.qsa_batch_decode_sparse == "off"
    assert "qsa_batch_decode_sparse" not in policy.as_dict()
    assert "qsa_batch_decode_sparse_min_context" not in policy.as_dict()
    env = policy.environment()
    assert "MLX_QWEN4_QSA_BATCH_DECODE_SPARSE" not in env
    chosen = FlashNextPolicy(qsa_batch_decode_sparse="gather",
                             qsa_batch_decode_sparse_min_context=8192)
    assert chosen.as_dict()["qsa_batch_decode_sparse"] == "gather"
    assert chosen.environment()["MLX_QWEN4_QSA_BATCH_DECODE_SPARSE"] == "gather"
    assert chosen.environment()["MLX_QWEN4_QSA_BATCH_DECODE_SPARSE_MIN_CONTEXT"] == "8192"


@pytest.mark.parametrize("field,value", [
    ("qsa_batch_decode_sparse", "on"),
    ("qsa_batch_decode_sparse", True),
    ("qsa_batch_decode_sparse_min_context", -1),
    ("qsa_batch_decode_sparse_min_context", 1.5),
])
def test_policy_refuses_invalid_values(field, value):
    with pytest.raises(ValueError):
        FlashNextPolicy(**{field: value})


def test_env_parse_and_setter_refuse_unknown_modes(monkeypatch):
    monkeypatch.setenv("MLX_QWEN4_QSA_BATCH_DECODE_SPARSE", "0")
    assert Q._env_batch_decode_sparse_mode() == "off"
    monkeypatch.setenv("MLX_QWEN4_QSA_BATCH_DECODE_SPARSE", "Gather")
    assert Q._env_batch_decode_sparse_mode() == "gather"
    monkeypatch.setenv("MLX_QWEN4_QSA_BATCH_DECODE_SPARSE", "dense")
    with pytest.raises(ValueError):
        Q._env_batch_decode_sparse_mode()
    with pytest.raises(ValueError):
        Q.set_qsa_batch_decode_sparse("dense")
    with pytest.raises(ValueError):
        Q.set_qsa_batch_decode_sparse("gather", min_context=-4)


def test_off_mode_records_nothing():
    attention = _attention()
    cache = _batched_cache(attention, [24, 37])
    Q.set_qsa_batch_decode_sparse("off")
    _decode(attention, cache, 3)
    assert Q.qsa_batch_decode_sparse_status()["attempts"] == 0


def test_gather_matches_masked_path_with_identical_cache_state():
    attention = _attention()
    reference_cache = _batched_cache(attention, [24, 37])
    gather_cache = _batched_cache(attention, [24, 37])
    Q.set_qsa_batch_decode_sparse("off")
    reference = _decode(attention, reference_cache, 6)
    Q.set_qsa_batch_decode_sparse("gather", min_context=0)
    gathered = _decode(attention, gather_cache, 6)
    for expected, got in zip(reference, gathered):
        assert mx.allclose(expected, got, atol=2e-5, rtol=2e-5).item()
    for a, b in zip(reference_cache.state, gather_cache.state):
        assert mx.array_equal(a, b).item()
    status = Q.qsa_batch_decode_sparse_status()
    assert status["counts"]["engaged_gather"] == 6
    assert status["counts"]["engaged_rows"] == 12
    assert status["fallbacks"] == 0


def test_each_row_matches_its_batch_one_reference():
    attention = _attention()
    lengths = [24, 37]
    batch_cache = _batched_cache(attention, lengths)
    Q.set_qsa_batch_decode_sparse("gather", min_context=0)
    mx.random.seed(5)
    hidden = mx.random.normal((2, 1, 16))
    out = attention(hidden, batch_cache.make_mask(1, return_array=True), batch_cache)
    mx.eval(out)
    Q.set_qsa_batch_decode_sparse("off")
    for index, length in enumerate(lengths):
        row_cache = _batched_cache(attention, [length])
        row = attention(hidden[index : index + 1], row_cache.make_mask(1, return_array=True), row_cache)
        assert mx.allclose(out[index : index + 1], row, atol=2e-5, rtol=2e-5).item()


def test_fallback_reasons_are_counted_and_output_unchanged():
    attention = _attention()
    # Context below the floor.
    Q.set_qsa_batch_decode_sparse("gather", min_context=1 << 20)
    _decode(attention, _batched_cache(attention, [24, 37]), 2)
    counts = Q.qsa_batch_decode_sparse_status(reset=True)["counts"]
    assert counts == {"attempts": 2, "fallback_context_below_min": 2}
    # Every block selected: the masked arm is already exact and dense.
    Q.set_qsa_batch_decode_sparse("gather", min_context=0)
    _decode(attention, _batched_cache(attention, [3, 6]), 1)
    counts = Q.qsa_batch_decode_sparse_status(reset=True)["counts"]
    assert set(counts) <= {"attempts", "fallback_dense_by_construction",
                           "fallback_selection_not_explicit"}
    assert counts["attempts"] == 1 and len(counts) == 2
    # The indexed arm declines on CPU (no Metal kernel) and keeps the masked
    # arm's exact output.
    reference_cache = _batched_cache(attention, [24, 37])
    indexed_cache = _batched_cache(attention, [24, 37])
    Q.set_qsa_batch_decode_sparse("off")
    reference = _decode(attention, reference_cache, 2)
    Q.set_qsa_batch_decode_sparse("indexed", min_context=0)
    got = _decode(attention, indexed_cache, 2)
    for a, b in zip(reference, got):
        assert mx.array_equal(a, b).item()
    status = Q.qsa_batch_decode_sparse_status()
    assert status["engagements"] == 0 and status["attempts"] == 2
    assert all(key.startswith("fallback_indexed_") for key in status["counts"] if key != "attempts")


def test_batch_one_and_multi_row_calls_are_not_candidates():
    attention = _attention()
    Q.set_qsa_batch_decode_sparse("gather", min_context=0)
    _decode(attention, _batched_cache(attention, [37]), 2)  # B1
    cache = _batched_cache(attention, [24, 37])
    hidden = mx.random.normal((2, 3, 16))  # a 3-row window
    mx.eval(attention(hidden, cache.make_mask(3, return_array=True), cache))
    assert Q.qsa_batch_decode_sparse_status()["attempts"] == 0


def test_indexed_admission_override_respects_kill_switch(monkeypatch):
    from mlx2.runtime.models import qwen4_qsa_indexed as I

    attention = _attention()
    cache = _batched_cache(attention, [24, 37])
    hidden = mx.random.normal((2, 1, 16))
    selection = attention.indexer(hidden, cache.make_mask(1, return_array=True), cache)
    monkeypatch.setattr(I, "_QSA_INDEXED_ENABLED", False)
    assert I.decide_qsa_indexed_admission(
        selection, length=1, training=False, layout_ok=True, cache=cache,
        min_context_override=0,
    ) == (False, "disabled")
    monkeypatch.setattr(I, "_QSA_INDEXED_ENABLED", None)
    assert I.decide_qsa_indexed_admission(
        selection, length=1, training=False, layout_ok=True, cache=cache,
        min_context_override=1 << 20,
    ) == (False, "context_out_of_range")
    # Without the override the auto one-token floor (64K) still applies.
    assert I.decide_qsa_indexed_admission(
        selection, length=1, training=False, layout_ok=True, cache=cache,
    ) == (False, "auto_context_out_of_range")
