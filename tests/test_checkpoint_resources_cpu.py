"""Optional resource callbacks cannot authorize or invalidate numeric KV."""
from types import SimpleNamespace
import pytest

from mlx2.runtime.checkpoint_resources import publication_hints, resource_binding, restore_resources
from mlx2.runtime.expert_residency_hints import ExpertResidencyHints, prefix_token_digest


TOKENS = [11, 12, 13, 14]
BINDING = dict(checkpoint='synthetic-model@123', source_revision='source-v1',
               quantization='original-q8', layout='packed-v1', resource_abi='expert-v1')


def hint(*, tokens=TOKENS, **changes):
    fields = dict(**BINDING, snapshot_revision='resource-observation-1',
                  token_digest=prefix_token_digest(tokens), covered_tokens=len(tokens),
                  layer_expert_counts=(8, 16), experts=((0, 1), (1, 7)),
                  phase='resident_snapshot', coverage='advisory', truncated=True)
    fields.update(changes)
    return ExpertResidencyHints(**fields)


def adapter(value=None, **changes):
    value = hint() if value is None else value
    fields = dict(checkpoint_resource_binding=lambda: BINDING,
                  checkpoint_resource_hints=lambda cache, tokens: value,
                  restore_checkpoint_resources=lambda h, cache: dict(prefetched=1, resident=1))
    fields.update(changes)
    return SimpleNamespace(**fields)


def lookup(value=None, *, covered=4):
    return SimpleNamespace(hit=True, cache=[SimpleNamespace(offset=covered)],
                           cached_tokens=covered, expert_hints=hint() if value is None else value)


def test_unsupported_adapter_keeps_all_optional_work_absent():
    assert resource_binding(None) is None
    assert publication_hints(None, [object()], TOKENS) == (None, {'action': 'not_supported'})
    assert restore_resources(None, lookup(), TOKENS)['action'] == 'not_supported'


def test_publication_keeps_inventory_advisory_without_owning_buffers():
    h = hint()
    got, receipt = publication_hints(adapter(h), [object()], TOKENS)
    assert got is h
    assert receipt['phase'] == 'resident_snapshot' and receipt['coverage'] == 'advisory'
    assert receipt['truncated'] and not receipt['physical_buffers_owned']


@pytest.mark.parametrize('changed', [dict(checkpoint='foreign'), dict(quantization='other'),
                                   dict(layout='other'), dict(source_revision='other'), dict(resource_abi='other')])
def test_foreign_resource_binding_declines_independently(changed):
    other = dict(BINDING, **changed)
    a = adapter(checkpoint_resource_binding=lambda: other)
    assert publication_hints(a, [object()], TOKENS)[0] is None
    hit = lookup()
    cache = hit.cache
    result = restore_resources(a, hit, TOKENS)
    assert result['reason'] == 'resource_binding_mismatch'
    assert hit.hit and hit.cache is cache and hit.cached_tokens == 4


def test_same_frontier_requires_actual_token_digest_even_with_matching_resource_identity():
    a = adapter()
    assert publication_hints(a, [object()], [11, 12, 13, 99])[0] is None
    hit = lookup()
    result = restore_resources(a, hit, [11, 12, 13, 99])
    assert result['reason'] == 'prefix_digest_mismatch' and hit.hit


def test_shorter_branch_retains_original_observation_without_fabricating_new_digest():
    observed = []
    a = adapter(restore_checkpoint_resources=lambda h, cache: observed.append(h) or dict(prefetched=0))
    hit = lookup(covered=2)
    result = restore_resources(a, hit, [11, 12, 99, 100])
    assert result['action'] == 'adapter'
    assert observed[0].covered_tokens == 4 and observed[0].token_digest == prefix_token_digest(TOKENS)
    assert observed[0].coverage == 'advisory' and observed[0].truncated
    assert hit.cached_tokens == 2 and hit.hit


def test_prefetch_failure_does_not_mutate_or_discard_numeric_hit():
    def fail(*args):
        raise RuntimeError('prefetch unavailable')
    a = adapter(restore_checkpoint_resources=fail)
    hit = lookup()
    cache = hit.cache
    result = restore_resources(a, hit, TOKENS)
    assert result['reason'] == 'resource_prepare_error' and result['numeric_hit_unchanged']
    assert hit.hit and hit.cache is cache


@pytest.mark.parametrize('value', [dict(buffer=object()), dict(number=float('nan')), dict(payload='x' * (17 << 10))])
def test_resource_telemetry_cannot_leak_live_objects_or_unbounded_payloads(value):
    result = restore_resources(adapter(restore_checkpoint_resources=lambda *a: value), lookup(), TOKENS)
    assert result['action'] == 'declined'


def test_invalid_binding_or_absent_hint_does_not_call_prefetch():
    def fail(*args):
        pytest.fail('No prefetch for missing or invalid metadata')
    a = adapter(checkpoint_resource_binding=lambda: {'checkpoint': 'x'}, restore_checkpoint_resources=fail)
    assert resource_binding(a) is None
    assert restore_resources(a, lookup(), TOKENS)['action'] == 'not_supported'
    empty = lookup()
    empty.expert_hints = None
    assert restore_resources(adapter(restore_checkpoint_resources=fail), empty, TOKENS)['reason'] == 'no_expert_hint'


def test_publication_hook_error_is_optional_and_speculative_metadata_never_publishes():
    def fail(*args):
        raise ValueError('unsupported checkpoint resources')
    assert publication_hints(adapter(checkpoint_resource_hints=fail), [object()], TOKENS)[0] is None
    speculative = hint(phase='speculative', coverage='exact', truncated=False)
    assert publication_hints(adapter(speculative), [object()], TOKENS)[0] is None


@pytest.mark.parametrize('name', ['checkpoint_resource_binding', 'checkpoint_resource_hints',
                                 'restore_checkpoint_resources'])
def test_optional_descriptor_failure_preserves_numeric_operations(name):
    class Broken:
        checkpoint_resource_binding = lambda self: BINDING
        checkpoint_resource_hints = lambda self, cache, tokens: hint()
        restore_checkpoint_resources = lambda self, h, cache: dict(prefetched=0)
    setattr(Broken, name, property(lambda self: (_ for _ in ()).throw(ValueError('descriptor failed'))))
    a = Broken()
    if name == 'checkpoint_resource_binding':
        assert resource_binding(a) is None
    if name != 'restore_checkpoint_resources':
        assert publication_hints(a, [object()], TOKENS)[0] is None
    hit = lookup()
    result = restore_resources(a, hit, TOKENS)
    if name != 'checkpoint_resource_hints':
        assert result['action'] in ('declined', 'not_supported')
    assert hit.hit and hit.cached_tokens == len(TOKENS)


def test_engine_publication_materializes_one_shot_tokens_once(monkeypatch):
    from mlx2.serving import ServingEngine
    import mlx2.serving as serving
    import mlx2.runtime.approximate_kv as approximate
    monkeypatch.setattr(approximate, 'prompt_cache_is_approximate', lambda cache: False)
    monkeypatch.setattr(serving, 'detach_checkpoint_graph', lambda *args: None)
    stored = []
    apc = SimpleNamespace(store=lambda key, tokens, cache, **kwargs: stored.append((list(tokens), kwargs)) or SimpleNamespace(stored=True))
    fake = SimpleNamespace(apc_reuse_disabled_reason=None, counts={}, adapter=adapter())
    assert ServingEngine._publish_checkpoint(fake, apc, 'key', iter(TOKENS), [object()])
    assert stored[0][0] == TOKENS and stored[0][1]['expert_hints'].phase == 'resident_snapshot'
    assert fake.counts['apc_expert_hints_offered'] == 1
    assert 'apc_expert_hints_published' not in fake.counts
