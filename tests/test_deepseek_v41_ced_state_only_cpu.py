"""Synthetic CPU contracts for CED state-only chunks; no checkpoint or GPU evidence."""

import math

import mlx.core as mx
import numpy as np
import pytest
from test_deepseek_v41_cpu import Weights, config

from mlx2.adapters.deepseek_v41_execution import validate_csa_policy
from mlx2.runtime.generate import PromptChunkCorruption, _validate_prefill_checkpoint_output
from mlx2.runtime.models.deepseek_v41 import DeepSeekV41Model
from mlx2.runtime.models.deepseek_v41_ced import state_only_admitted
from mlx2.runtime.prefill_lookahead import StateOnlyPrefillOutput, prefill_lookahead


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    def forbidden(*args, **kwargs):
        raise AssertionError("synthetic CPU test attempted Metal")
    monkeypatch.setattr(mx.fast, "metal_kernel", forbidden)
    yield
    mx.set_default_device(previous)


def _bits(x):
    return np.array(x).view(np.uint32)


def _config(**kwargs):
    # Barrier 3 (last KV source), decoder layers 4..7, window 3: the first
    # decoder layer's cone is 3 + 3 * 2 = 9 rows.
    values = dict(num_hidden_layers=8, compress_ratios=(0, 2, 2, 1, 1, 1, 1, 1),
                  kv_source_layer_ids=(1, 3), index_source_layer_ids=(1, 3, 5, 7), sliding_window=3)
    values.update(kwargs)
    return config(**values)


def _packed(**kwargs):
    return _config(head_dim=32, index_head_dim=32, simulate_quantization=True,
                   activation_dtype="bfloat16", **kwargs)


def _prompt(model, cache, ids, step):
    """The prompt loop's contract: one lookahead scope per plain chunk."""
    outputs = []
    for first in range(0, ids.shape[1], step):
        chunk = ids[:, first:first + step]
        with prefill_lookahead([ids.shape[1] - first - chunk.shape[1]], step=step):
            outputs.append(model.prefill_forward(chunk, cache))
    return outputs


def _states(cache):
    return [tuple(None if v is None else np.array(v) for v in (s.keys, s.buf_kv, s.buf_gate, s.pooled,
                                                               s.index_keys, s.hash_tail)) + (s.offset,)
            for s in cache]


@pytest.mark.parametrize("cfg,options", [
    (_config, {}),
    (_packed, {"kv_storage_policy": "native_packed", "candidate_score_policy": "candidate_only"}),
])
@pytest.mark.parametrize("length,step", [(40, 4), (37, 5), (23, 3)])
def test_state_only_chunks_match_dependency_ced_bits_and_reference(cfg, options, length, step):
    c = cfg()
    ref = DeepSeekV41Model(c, Weights(c), identity="synthetic@ced-so")
    ced = DeepSeekV41Model(c, Weights(c), identity="synthetic@ced-so", ced_policy="dependency_exact", **options)
    so = DeepSeekV41Model(c, Weights(c), identity="synthetic@ced-so", ced_policy="dependency_exact",
                          ced_state_only=True, **options)
    ids = mx.array([[(7 * i + 3) % c.vocab_size for i in range(length)]])
    rc, cc, sc = ref.make_cache(), ced.make_cache(), so.make_cache()
    expected = ref(ids, rc, logits_to_keep=1)
    dependency = _prompt(ced, cc, ids, step)[-1]
    outputs = _prompt(so, sc, ids, step)
    skipped = [isinstance(o, StateOnlyPrefillOutput) for o in outputs]
    # A chunk is state-only exactly while at least one cone (9 rows) follows it.
    assert skipped == [length - first - min(step, length - first) >= 9 for first in range(0, length, step)]
    assert so.scheduled_kv_calls["ced_state_only_chunks"] == sum(skipped)
    assert not isinstance(outputs[-1], StateOnlyPrefillOutput)
    np.testing.assert_array_equal(_bits(outputs[-1]), _bits(dependency))
    tolerance = dict(rtol=1e-5, atol=1e-6) if not options else dict(rtol=2e-2, atol=2e-2)
    np.testing.assert_allclose(np.array(outputs[-1]), np.array(expected), **tolerance)
    for left, right in zip(_states(cc), _states(sc)):
        assert left[-1] == right[-1]
        for a, b in zip(left[:-1], right[:-1]):
            assert (a is None) == (b is None)
            if a is not None:
                np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8))
    assert not any(s.ced_local_deferred or s.pending for s in sc)
    token = mx.array([[1]])
    np.testing.assert_array_equal(_bits(ced(token, cc)), _bits(so(token, sc)))
    sc[0].snapshot()


def test_deferred_rows_refuse_decode_snapshot_publication_and_ordinary_forward():
    c = _config()
    model = DeepSeekV41Model(c, Weights(c), identity="synthetic@ced-so-refuse",
                             ced_policy="dependency_exact", ced_state_only=True)
    cache = model.make_cache()
    with prefill_lookahead([20], step=4):
        output = model.prefill_forward(mx.array([[1, 2, 3, 4]]), cache)
    assert isinstance(output, StateOnlyPrefillOutput) and output.hidden.shape[0] == 1
    deferred = [s for s in cache if s.ced_local_deferred]
    assert [cache.index(s) for s in deferred] == [4, 5, 6, 7]
    assert all(s.offset == 4 and s.keys is None and s.pending for s in deferred)
    assert deferred[0].checkpoint_publication_refusal()
    with pytest.raises(ValueError, match="snapshot frontier"):
        deferred[0].snapshot()
    # Only a dependency prefill chunk may enter a row with deferred local keys.
    with pytest.raises(ValueError, match="frontier mismatch"):
        model(mx.array([[1]]), cache)
    with pytest.raises(ValueError, match="frontier mismatch"):
        model.verify_rows(mx.array([[1]]), cache)
    model.ced_policy = "reference"
    with pytest.raises(ValueError, match="frontier mismatch"):
        model.prefill_forward(mx.array([[1]]), cache)


@pytest.mark.parametrize("rows_after,step,expected", [(9, 3, True), (8, 3, False), (9, 2, False), (100, 3, True)])
def test_admission_needs_one_cone_after_and_a_window_wide_step(rows_after, step, expected):
    assert state_only_admitted(_config(), rows_after, step) is expected


def test_admission_refuses_engram_above_barrier_and_models_without_decoder():
    assert not state_only_admitted(_config(engram_layer_ids=(5,), engram_num_embeddings=(7,)), 100, 4)
    assert not state_only_admitted(config(), 100, 4)  # barrier is the last layer


def test_no_lookahead_or_flag_means_full_dependency_prefill():
    c = _config()
    model = DeepSeekV41Model(c, Weights(c), identity="synthetic@ced-so-off",
                             ced_policy="dependency_exact", ced_state_only=True)
    assert not isinstance(model.prefill_forward(mx.array([[1, 2, 3, 4]]), model.make_cache()), StateOnlyPrefillOutput)
    plain = DeepSeekV41Model(c, Weights(c), identity="synthetic@ced-so-off", ced_policy="dependency_exact")
    with prefill_lookahead([100], step=4):
        assert not isinstance(plain.prefill_forward(mx.array([[1, 2, 3, 4]]), plain.make_cache()),
                              StateOnlyPrefillOutput)
    with pytest.raises(ValueError, match="require dependency_exact"):
        DeepSeekV41Model(c, Weights(c), identity="synthetic@ced-so-off", ced_state_only=True)


def test_policy_requires_dependency_ced():
    with pytest.raises(ValueError, match="require deepseek_ced=dependency_exact"):
        validate_csa_policy({"deepseek_ced_state_only": "after_cone_v1"})
    validate_csa_policy({"deepseek_ced": "dependency_exact", "deepseek_ced_state_only": "after_cone_v1"})


def test_prompt_loop_accepts_state_only_output_only_before_a_chunk_of_the_same_call():
    finite = StateOnlyPrefillOutput(mx.zeros((1, 6)))
    _validate_prefill_checkpoint_output(finite, [4], [11], rows_after=[9])
    with pytest.raises(PromptChunkCorruption, match="followed by a chunk"):
        _validate_prefill_checkpoint_output(finite, [4], [11], rows_after=[0])
    with pytest.raises(PromptChunkCorruption, match="followed by a chunk"):
        _validate_prefill_checkpoint_output(finite, [4], [11])
    with pytest.raises(PromptChunkCorruption, match="non-finite hidden"):
        _validate_prefill_checkpoint_output(StateOnlyPrefillOutput(mx.array([[0.0, math.nan]])), [4], [11],
                                            rows_after=[9])


def test_receipt_states_that_sliced_serving_prompts_do_not_engage():
    """The serving scheduler feeds one prefill-step slice per prompt call, so
    the lookahead sees rows_after 0 and state-only never engages there; the
    route receipt must not imply otherwise, and counts what was observed."""
    from mlx2.adapters.deepseek_v41_execution import execution_receipt

    c = _config()
    model = DeepSeekV41Model(c, Weights(c), identity="synthetic@ced-so-slice",
                             ced_policy="dependency_exact", ced_state_only=True)
    cache, ids = model.make_cache(), mx.array([[1 + i % 17 for i in range(16)]])
    for first in range(0, 16, 4):  # one slice per call: nothing follows in the call
        with prefill_lookahead([0], step=4):
            assert not isinstance(model.prefill_forward(ids[:, first:first + 4], cache), StateOnlyPrefillOutput)
    receipt = execution_receipt(model)
    assert receipt["ced_state_only"] == "after_cone_v1"
    assert receipt["ced_state_only_observed_chunks"] == 0
    assert "does not engage on served requests" in receipt["ced_state_only_scope"]
    _prompt(model, model.make_cache(), ids, 4)  # a whole-prompt call does engage
    assert execution_receipt(model)["ced_state_only_observed_chunks"] > 0
    plain = DeepSeekV41Model(c, Weights(c), identity="synthetic@ced-so-slice", ced_policy="dependency_exact")
    assert "ced_state_only_scope" not in execution_receipt(plain)
