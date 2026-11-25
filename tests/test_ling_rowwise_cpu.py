"""Ling per-row verify laws (ling_rowwise): diagnostic attribution opt-ins.

CPU only, synthetic tiny model (no checkpoint).  The exact control
(``verify_rowwise``: dense + moe + kda + mla) must make a speculative verify
forward's rows bit-identical to the ordinary L == 1 decode sequence, including
the KDA state after a partial-accept rollback.
"""

import copy
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters.ling3_flash import Ling3FlashAdapter, optimization_policy
from mlx2.runtime.ling_tree_executor import LingTreeExecutor
from mlx2.runtime.models import bailing_hybrid as bh
from mlx2.runtime.models import ling_rowwise

from test_bailing_hybrid_ling32_math import _tiny

PROMPT = [3, 7, 11, 2, 5, 9]
ROWS = [4, 13, 8, 21, 6]


def _bind(model, components):
    """The adapter's binding (Ling3FlashAdapter._bind_optimizations) on a bare model."""
    for layer in model.model.layers:
        if "mla" in components and layer.is_global:
            layer.attention.ling_rowwise = True
        if "kda" in components and not layer.is_global:
            layer.attention.ling_rowwise = True
        if "moe" in components and isinstance(layer.mlp, bh.SparseMoeBlock):
            layer.mlp.ling_rowwise = True
        if "dense" in components:
            ling_rowwise.bind_dense(m for _n, m in layer.named_modules())
    if "dense" in components:
        ling_rowwise.bind_dense([model.model.norm, model.lm_head])


def _prefilled(model):
    cache = model.make_cache()
    mx.eval(model(mx.array([PROMPT]), cache))
    return cache


def _decode_sequence(model, cache, tokens):
    # Evaluated step by step, as served decode is.
    rows = []
    for token in tokens:
        rows.append(model(mx.array([[token]]), cache)[:, -1])
        mx.eval(rows[-1], [item.state for item in cache])
    return mx.stack(rows, axis=1)


def _speculate(cache):
    for item in cache:
        item.start_speculation()
    return cache


def _calls(model, key):
    return sum(layer.attention.ling_calls[key] for layer in model.model.layers)


def _bitwise(a, b):
    a, b = np.asarray(a.astype(mx.float32)), np.asarray(b.astype(mx.float32))
    assert a.shape == b.shape
    assert np.array_equal(a, b), float(np.abs(a - b).max())


def test_policy_keys_default_off_and_expand():
    policy = optimization_policy(None)
    for key in ("verify_rowwise", "rowwise_dense", "rowwise_moe", "rowwise_kda", "rowwise_mla"):
        assert policy[key] is False
    expand = Ling3FlashAdapter.rowwise_components
    assert expand(SimpleNamespace(optimizations=policy)) == ()
    assert expand(SimpleNamespace(optimizations=optimization_policy({"verify_rowwise": True}))) \
        == ("dense", "moe", "kda", "mla")
    assert expand(SimpleNamespace(optimizations=optimization_policy({"rowwise_kda": True}))) == ("kda",)


def test_tree_executor_binds_rowwise_keys_by_source():
    plain = LingTreeExecutor(optimizations={})
    bound = LingTreeExecutor(optimizations={"verify_rowwise": True})
    assert "ling_rowwise.py" in {name for name, _digest in bound.source_identity}
    assert bound.source_revision_token != plain.source_revision_token


def test_default_model_never_engages():
    ling_rowwise.STATS.clear()
    model = _tiny()
    cache = _speculate(_prefilled(model))
    mx.eval(model(mx.array([ROWS]), cache))
    assert ling_rowwise.STATS["verify_forwards"] == 1
    assert ling_rowwise.STATS["dense_rows"] == ling_rowwise.STATS["moe_rows"] == 0
    assert _calls(model, "kda_rowwise_rows") == _calls(model, "mla_rowwise_rows") == 0


def test_exact_control_matches_decode_sequence_bitwise():
    model = _tiny()
    _bind(model, ling_rowwise.COMPONENTS)
    base = _prefilled(model)
    expected = _decode_sequence(model, copy.deepcopy(base), ROWS)
    ling_rowwise.STATS.clear()
    verify = model(mx.array([ROWS]), _speculate(copy.deepcopy(base)))
    _bitwise(verify, expected)
    assert ling_rowwise.STATS["dense_rows"] > 0 and ling_rowwise.STATS["moe_rows"] > 0
    assert _calls(model, "kda_rowwise_rows") > 0 and _calls(model, "mla_rowwise_rows") > 0
    assert _calls(model, "mla_rowwise_declined") == 0


@pytest.mark.parametrize("accepted", [1, 3, 5])
def test_exact_control_rollback_state_matches_decode(accepted):
    model = _tiny()
    _bind(model, ling_rowwise.COMPONENTS)
    base = _prefilled(model)
    ordinary = copy.deepcopy(base)
    _decode_sequence(model, ordinary, ROWS[:accepted])
    following = model(mx.array([[17]]), ordinary)

    cache = _speculate(copy.deepcopy(base))
    mx.eval(model(mx.array([ROWS]), cache))
    for item in cache:
        item.trim(len(ROWS) - accepted)
        item.stop_speculation()
    _bitwise(model(mx.array([[17]]), cache), following)


@pytest.mark.parametrize("component", ling_rowwise.COMPONENTS)
def test_single_component_engages_and_stays_close(component):
    model = _tiny()
    base = _prefilled(model)
    reference = model(mx.array([ROWS]), _speculate(copy.deepcopy(base)))
    _bind(model, (component,))
    ling_rowwise.STATS.clear()
    candidate = model(mx.array([ROWS]), _speculate(copy.deepcopy(base)))
    engaged = {"dense": ling_rowwise.STATS["dense_rows"], "moe": ling_rowwise.STATS["moe_rows"],
               "kda": _calls(model, "kda_rowwise_rows"), "mla": _calls(model, "mla_rowwise_rows")}
    assert engaged.pop(component) > 0
    assert not any(engaged.values())
    assert mx.allclose(candidate, reference, atol=1e-4, rtol=1e-4)


def test_inactive_for_prompt_chunks_and_wide_forwards():
    model = _tiny()
    _bind(model, ling_rowwise.COMPONENTS)
    ling_rowwise.STATS.clear()
    mx.eval(model(mx.array([ROWS]), model.make_cache()))  # prompt chunk: not speculating
    wide = _speculate(_prefilled(model))
    mx.eval(model(mx.array([list(range(1, 18))]), wide))  # 17 rows: above the verify window
    assert ling_rowwise.STATS["verify_forwards"] == 0
    assert ling_rowwise.STATS["dense_rows"] == ling_rowwise.STATS["moe_rows"] == 0
    assert _calls(model, "kda_rowwise_rows") == _calls(model, "mla_rowwise_rows") == 0


def test_bind_dense_keeps_parameters_and_is_idempotent():
    model = _tiny()
    before = dict(model.lm_head.parameters())
    assert ling_rowwise.bind_dense([model.lm_head]) == 1
    assert ling_rowwise.bind_dense([model.lm_head]) == 0
    assert type(model.lm_head).__name__ == "RowwiseLinear"
    after = dict(model.lm_head.parameters())
    assert before.keys() == after.keys()
    assert all(before[k] is after[k] for k in before)


def test_exact_control_engages_through_segmented_self_mtp(monkeypatch):
    """The served native-MTP verify path (segmented true-batched self-MTP)
    opens the verify scope, takes every component law without a decline and
    still emits the ordinary greedy stream."""
    from test_bailing_hybrid_ling32_math import _self_mtp_tokens

    new_stream = mx.new_stream
    monkeypatch.setattr(mx, "new_stream", lambda device: new_stream(mx.cpu))
    model = _tiny()
    ordinary, _ = _self_mtp_tokens(model, monkeypatch, self_mtp=False)
    _bind(model, ling_rowwise.COMPONENTS)
    ling_rowwise.STATS.clear()
    tokens, stats = _self_mtp_tokens(model, monkeypatch)
    assert stats["true_batched_engaged"] > 0
    assert ling_rowwise.STATS["verify_forwards"] > 0
    assert ling_rowwise.STATS["dense_rows"] > 0 and ling_rowwise.STATS["moe_rows"] > 0
    assert _calls(model, "kda_rowwise_rows") > 0 and _calls(model, "mla_rowwise_rows") > 0
    assert _calls(model, "mla_rowwise_declined") == 0
    # The MTP head drafts outside the verify scope and is never bound.
    assert model.mtp.layers[0].attention.ling_calls["mla_rowwise_rows"] == 0
    assert tokens == ordinary
