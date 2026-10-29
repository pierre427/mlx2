"""Candidate arithmetic/ownership fixtures, not Nanbeige model qualification."""
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx2.adapters.nanbeige43_cache import (
    approximate_operations,
    candidate_q8_cache,
    install_reservation,
)
from mlx2.runtime.generate import GenerationBatch, StopSequenceMatcher, _corrupt_rows
from mlx2.runtime.greedy_scores import (
    LazyLogprobs,
    admitted,
    greedy_scores,
    score_eval_targets,
)
from mlx2.runtime.models.base import scaled_dot_product_attention
from mlx2.runtime.models.cache import KVCache
from mlx2.runtime.sample_utils import make_sampler


@pytest.fixture(autouse=True)
def cpu():
    before = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(before)


def test_greedy_preserves_normalized_tie_law_and_lazy_full_scores():
    # Large negative normalizers can coalesce adjacent raw logits; retain exact
    # normalized first-index argmax rather than silently selecting raw argmax.
    logits = mx.array([[0., 1., 1.], [-100000000., -100000000., -100000000.]])
    ids, scores = greedy_scores(logits)
    reference = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    assert ids.tolist() == mx.argmax(reference, axis=-1).tolist()
    assert all(isinstance(row, LazyLogprobs) for row in scores)
    assert mx.array_equal(scores[0].materialize_logprobs(), reference[0]).item()
    assert len(score_eval_targets(scores)) == 4
    vocab, valid = _corrupt_rows(ids, scores)
    assert vocab == 3 and valid.tolist() == [True, True]


def test_nonfinite_lazy_scores_fail_closed():
    ids, scores = greedy_scores(mx.array([[-float('inf')]*3, [1., 2., 3.]]))
    _, valid = _corrupt_rows(ids, scores)
    assert valid.tolist() == [False, True]


def test_only_trusted_greedy_sampler_admitted():
    model = SimpleNamespace(_mlx2_lean_greedy=True)
    greedy = make_sampler(temp=0)
    assert admitted(model, [greedy, None], greedy)
    assert not admitted(model, [make_sampler(temp=.6)], greedy)
    custom = lambda x: mx.argmax(x, axis=-1)
    custom.deterministic = True
    assert not admitted(model, [custom], greedy)


class Model:
    def __init__(self, lean):
        self._mlx2_lean_greedy = lean
    def __call__(self, inputs, cache):
        return mx.broadcast_to(mx.array([[[0., 2., 1.]]]), (inputs.shape[0], 1, 3))


def make_batch(lean):
    return GenerationBatch(Model(lean), [1, 2], mx.array([0, 0]), [], [[], []],
                           [], make_sampler(temp=0), [],
                           [StopSequenceMatcher(), StopSequenceMatcher()], [3, 3])


@pytest.mark.parametrize('mode', ['current', 'deferred'])
def test_actual_generation_lazy_scores_match_reference(mode, monkeypatch):
    monkeypatch.setenv('MLX2_STEP_VALIDITY', mode)
    normal, lean = make_batch(False), make_batch(True)
    for _ in range(3):
        a, b = normal.next(), lean.next()
        assert [(r.uid, r.token, r.finish_reason) for r in a] == [(r.uid, r.token, r.finish_reason) for r in b]
        for x, y in zip(a, b):
            assert mx.array_equal(x.logprobs, y.logprobs.materialize_logprobs()).item()


def test_reservation_preserves_values_offset_and_trim():
    normal, reserved = KVCache(), KVCache().reserve(1024)
    for n in [255, 2, 260]:
        k = mx.arange(n * 64).reshape(1, 1, n, 64).astype(mx.float32)
        for cache in [normal, reserved]:
            cache.update_and_fetch(k, k + 1)
        assert reserved.offset == normal.offset
        assert mx.array_equal(normal.keys_and_values()[0], reserved.keys_and_values()[0]).item()
    assert reserved.keys.shape[2] == 1024
    reserved.reserve(1300)
    assert reserved.offset == 517 and reserved.keys.shape[2] == 1536
    assert reserved.trim(5) == 5
    with pytest.raises(ValueError): reserved.reserve(True)


def test_factory_reservation_and_approximate_declaration():
    model = SimpleNamespace(make_cache=lambda: [KVCache()], args=SimpleNamespace(num_hidden_layers=1, num_key_value_heads=8, head_dim=128), layers=[1])
    receipt = install_reservation(model, 2048)
    cache = model.make_cache()[0]
    assert cache.offset == 0 and cache.keys is None and cache._reserve_tokens == 2048
    assert receipt['planned_bf16_kv_bytes'] == 2048*8*128*4
    with pytest.raises(ValueError): install_reservation(model, 9000)
    assert approximate_operations()['kv_q8'].key_bits == 8
    with pytest.raises(ValueError): candidate_q8_cache(model)
    assert candidate_q8_cache(model, allow_approximate=True)[0].bits == 8


def test_packed_q8_attention_without_full_history_dequantization(monkeypatch):
    # Same-representation ordinary reference isolates storage/math. This is
    # not a BF16 model quality assertion and uses the CPU composed QMM path.
    model = SimpleNamespace(layers=[1])
    cache = candidate_q8_cache(model, allow_approximate=True)[0]
    keys = mx.random.normal((1, 8, 17, 128))
    values = mx.random.normal((1, 8, 17, 128))
    qk, qv = cache.update_and_fetch(keys, values)
    query = mx.random.normal((1, 56, 1, 128))
    original = mx.dequantize
    def refuse(*a, **kw): raise AssertionError('full KV dequantization called')
    monkeypatch.setattr(mx, 'dequantize', refuse)
    actual = scaled_dot_product_attention(query, qk, qv, cache, 128**-.5, None)
    mx.eval(actual)
    monkeypatch.setattr(mx, 'dequantize', original)
    reference = mx.fast.scaled_dot_product_attention(query, original(*qk, group_size=64, bits=8), original(*qv, group_size=64, bits=8), scale=128**-.5)
    assert mx.max(mx.abs(actual - reference)).item() < 1e-4


def test_reservation_descriptor_clone_keeps_append_ownership():
    cache = KVCache().reserve(1024)
    k = mx.ones((1, 1, 3, 64))
    cache.update_and_fetch(k, k)
    clone = cache.committed_descriptor_clone()
    assert clone is not None
    assert clone.offset == cache.offset == 3
    assert clone._reserve_tokens == 1024
    clone.update_and_fetch(k, k)
    assert cache.offset == 3 and clone.offset == 6
    assert mx.array_equal(cache.keys_and_values()[0], k).item()


def test_lazy_logprob_api_serializes_exact_probabilities():
    from mlx2.logprobs import token_logprob
    class Tokenizer:
        def convert_ids_to_tokens(self, ids): return [str(i) for i in ids]
    _, scores = greedy_scores(mx.array([[0., 2., 1.]]))
    actual = token_logprob(scores[0], 1, Tokenizer(), top_n=2, array_module=mx)
    expected = token_logprob(scores[0].materialize_logprobs(), 1, Tokenizer(), top_n=2, array_module=mx)
    assert actual == expected


def test_merge_reservation_never_lends_source_spare_storage():
    from mlx2.runtime.models.cache import BatchKVCache
    source = KVCache().reserve(1024)
    k = mx.ones((1, 1, 3, 64))
    source.update_and_fetch(k, k)
    batch = BatchKVCache.merge([source])
    assert batch.keys.shape[2] == 3  # not the source's reserved spare slab
    assert batch._reserve_tokens == 1024
    batch.update_and_fetch(k, k + 1)
    assert batch.keys.shape[2] == 1024
    assert source.offset == 3 and batch._idx == 6
    assert mx.array_equal(source.keys_and_values()[1], k).item()
    empty = BatchKVCache.merge([KVCache().reserve(512)])
    empty.update_and_fetch(k, k)
    assert empty.keys.shape[2] == 512


def test_reservation_applied_to_restored_decode_cache():
    from mlx2.runtime.models.cache import BatchKVCache
    model = SimpleNamespace(make_cache=lambda: [KVCache()], args=SimpleNamespace(num_hidden_layers=1, num_key_value_heads=8, head_dim=128), layers=[1])
    install_reservation(model, 512)
    # Simulate a restored exact prefix whose serialization omitted capacity plan.
    warm = KVCache()
    k = mx.ones((1, 1, 7, 64))
    warm.update_and_fetch(k, k)
    merged = BatchKVCache.merge([warm])
    model._mlx2_prepare_decode_cache([merged])
    merged.update_and_fetch(k, k)
    assert merged.keys.shape[2] == 512
    assert warm.offset == 7 and merged._idx == 14
    with pytest.raises(ValueError, match='exact KV'):
        model._mlx2_prepare_decode_cache([candidate_q8_cache(model, allow_approximate=True)[0]])


def test_small_reserved_warm_growth_fits_rounded_total_not_prefix_plus_tail():
    from mlx2.runtime.models.cache import BatchKVCache
    source = KVCache().reserve(256)
    source.update_and_fetch(mx.ones((1, 1, 3, 64)), mx.ones((1, 1, 3, 64)))
    batch = BatchKVCache.merge([source])
    batch.update_and_fetch(mx.ones((1, 1, 2, 64)), mx.ones((1, 1, 2, 64)))
    assert batch.keys.shape[2] == 256
    assert batch._idx == 5


@pytest.mark.parametrize('reserved', [False, True])
def test_actual_executor_checkpoint_preserves_lazy_pending_and_continuation(reserved):
    from test_exact_decode_checkpoint_cpu import _batch, _model, _next_output

    from mlx2.runtime.greedy_scores import materialize_logprobs
    model = _model()
    model._mlx2_lean_greedy = True
    if reserved:
        install_reservation(model, 256)
    control, _ = _batch(model)
    expected = [_next_output(control) for _ in range(14)]
    batch, uid = _batch(model)
    actual = [_next_output(batch) for _ in range(4)]
    assert isinstance(batch._generation_batch._next_logprobs[0], LazyLogprobs)
    pending = batch._generation_batch._next_tokens.tolist()
    checkpoint = batch.suspend_ordinary_checkpoint(uid, ('synthetic', 'lean-reserve'))
    try:
        assert not checkpoint.control.prompt_cache
        assert checkpoint.control._next_tokens.tolist() == pending
        assert batch.resume_ordinary_checkpoint(checkpoint, ('synthetic', 'lean-reserve')) == uid
        if reserved:
            assert all(c._reserve_tokens == 256 for c in batch._generation_batch.prompt_cache)
        actual.extend(_next_output(batch) for _ in range(10))
        assert [r.token for r in actual] == [r.token for r in expected]
        for a, b in zip(actual, expected):
            assert mx.array_equal(materialize_logprobs(a.logprobs), materialize_logprobs(b.logprobs)).item()
    finally:
        checkpoint.close()
        batch.close()
        control.close()


def test_actual_serving_greedy_sampler_is_admitted_and_sampled_is_not():
    from mlx2.serving import _greedy_batch_sampler
    model = SimpleNamespace(_mlx2_lean_greedy=True)
    assert admitted(model, [_greedy_batch_sampler], make_sampler(temp=0))
    assert not admitted(model, [make_sampler(temp=.6)], _greedy_batch_sampler)


def test_normalization_created_top_tie_changes_greedy_winner_and_keeps_api_row():
    from mlx2.logprobs import token_logprob
    class Tokenizer:
        def convert_ids_to_tokens(self, ids):
            return [str(index) for index in ids]

    logits = mx.array([[-2e-8, 0.] + [-.1] * 14], dtype=mx.float32)
    reference = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    assert mx.argmax(logits, axis=-1).tolist() == [1]
    assert mx.argmax(reference, axis=-1).tolist() == [0]
    assert reference[0, 0].item() == reference[0, 1].item()
    ids, scores = greedy_scores(logits)
    assert ids.tolist() == [0]
    assert mx.array_equal(scores[0].materialize_logprobs(), reference[0]).item()
    actual = token_logprob(scores[0], 0, Tokenizer(), top_n=2, array_module=mx)
    expected = token_logprob(reference[0], 0, Tokenizer(), top_n=2, array_module=mx)
    assert actual == expected


def test_optional_prefill_hook_preserves_scheduler_without_model():
    from mlx2.runtime.generate import BatchGenerator
    scheduler = SimpleNamespace(prefill_step_size=512, prefill_step_autoscale=False)
    assert BatchGenerator._prompt_prefill_step(scheduler, 1024) == 512


def test_dynamic_model_proxy_cannot_invent_optional_callbacks():
    from unittest.mock import MagicMock

    from mlx2.runtime.generate import BatchGenerator, _declared_model_callback
    model = MagicMock()
    # Even already-created dynamic proxy children are not declarations.
    phantom = model._mlx2_prefill_plan
    prepare = model._mlx2_prepare_decode_cache
    scheduler = SimpleNamespace(model=model, prefill_step_size=512,
                                prefill_step_autoscale=False, _generation_batch=[])
    assert BatchGenerator._prompt_prefill_step(scheduler, 1024) == 512
    assert _declared_model_callback(model, '_mlx2_prepare_decode_cache') is None
    phantom.assert_not_called()
    prepare.assert_not_called()


def test_explicit_instance_and_class_prefill_callbacks_remain_admitted():
    from unittest.mock import Mock

    from mlx2.runtime.generate import BatchGenerator
    instance = SimpleNamespace(_mlx2_prefill_plan=Mock(return_value=128))
    scheduler = SimpleNamespace(model=instance, prefill_step_size=512,
                                prefill_step_autoscale=False, _generation_batch=[1, 2])
    assert BatchGenerator._prompt_prefill_step(scheduler, 1024) == 128
    instance._mlx2_prefill_plan.assert_called_once_with(
        total_tokens=1024, active_decode_lanes=2, maximum=512)
    class Declared:
        def _mlx2_prefill_plan(self, **kwargs):
            assert kwargs['active_decode_lanes'] == 2
            return 64
    scheduler.model = Declared()
    assert BatchGenerator._prompt_prefill_step(scheduler, 1024) == 64
    instance._mlx2_prefill_plan.return_value = 513
    scheduler.model = instance
    with pytest.raises(ValueError, match='configured positive maximum'):
        BatchGenerator._prompt_prefill_step(scheduler, 1024)
