"""CPU-only law/reproducibility checks; no target checkpoint or model loads."""
import importlib.util
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

# Before any MLX array creation, keep this fixture entirely on CPU.
mx.set_default_device(mx.cpu)
PATH = Path(__file__).resolve().parents[1] / 'scripts/research/kimi_lora_sampling.py'
spec = importlib.util.spec_from_file_location('kimi_lora_sampling', PATH)
sampling = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sampling)


def rows():
    # Realistic vocab > top_k; a flat tail and close leaders exercise filters.
    base = np.linspace(-4, 1, 64, dtype=np.float32)
    return [mx.array(np.roll(base, i % 13)) for i in range(32)]


def test_server_defaults_and_missing_fields_resolve_exactly():
    from mlx2.sampling_defaults import resolve_sampling
    expected, _ = resolve_sampling({}, None, thinking=None)
    assert sampling.resolved_profile() == expected
    assert sampling.resolved_profile({'temperature': .7}) == expected
    assert expected['top_p'] == .8 and expected['top_k'] == 20


def test_seed_stable_independent_of_order_and_distinct_replicates():
    a = {cid: sampling.case_seed(20261103, cid) for cid in ('case-a', 'case-b', 'case-c')}
    b = {cid: sampling.case_seed(20261103, cid) for cid in reversed(tuple(a))}
    assert a == b and len(set(a.values())) == 3
    assert sampling.case_seed(20261103, 'case-a', 1) != a['case-a']
    assert all(0 <= value < 2**32 for value in a.values())


def test_same_case_repeats_despite_other_cases_and_global_rng_draws():
    logits = rows()
    first = sampling.CaseSampler(suite_seed=7, case_id='a')
    second = sampling.CaseSampler(suite_seed=7, case_id='a')
    unrelated = sampling.CaseSampler(suite_seed=7, case_id='b')
    expected = [first.draw(row) for row in logits]
    actual = []
    for row in logits:
        unrelated.draw(row)
        mx.eval(mx.random.uniform(shape=(9,)))
        actual.append(second.draw(row))
    assert actual == expected
    assert first.receipt()['rng_draws'] == second.receipt()['rng_draws'] == len(logits)


def test_matches_existing_serving_seeded_law_exactly():
    from mlx2.runtime.sample_utils import LaneRNG, draw_key, make_transformed_logprobs
    sampler = sampling.CaseSampler(suite_seed=99, case_id='serving-law')
    rng = LaneRNG(sampler.seed)
    transform = make_transformed_logprobs(.7, top_p=.8, top_k=20, min_p=0.)
    for row in rows():
        expected = int(mx.random.categorical(transform(row), key=draw_key(rng)).item())
        assert sampler.draw(row) == expected


def test_sampled_draws_preserve_global_training_rng_state():
    mx.random.seed(812)
    before = [np.array(key) for key in mx.random.state]
    sampler = sampling.CaseSampler(suite_seed=6, case_id='isolated')
    for row in rows():
        sampler.draw(row)
    after = [np.array(key) for key in mx.random.state]
    assert len(before) == len(after)
    assert all(np.array_equal(a, b) for a, b in zip(before, after))


@pytest.mark.parametrize('dtype', [mx.float32, mx.float16, mx.bfloat16])
def test_greedy_exact_argmax_ties_and_zero_rng_draws(dtype):
    sampler = sampling.CaseSampler(sampling.GREEDY_PROFILE, suite_seed=3, case_id='greedy')
    before = [np.array(key) for key in mx.random.state]
    for data in ([0., 1., 1., -3.], [-2., 0., 2., 1.]):
        row = mx.array(data, dtype=dtype)
        assert sampler.draw(row) == int(mx.argmax(row).item())
    assert sampler.receipt()['rng_draws'] == 0
    assert all(np.array_equal(a, np.array(b)) for a, b in zip(before, mx.random.state))


@pytest.mark.parametrize('profile', [
    {'temperature': True}, {'temperature': float('nan')}, {'temperature': -1},
    {'top_p': 1.1}, {'min_p': -.1}, {'top_k': True}, {'top_k': -1},
    {'repetition_penalty': 1.1}, {'presence_penalty': .5}, {'frequency_penalty': .5},
    {'xtc_probability': .1}, {'seed': 3},
])
def test_unsupported_or_invalid_profiles_refused(profile):
    with pytest.raises(ValueError):
        sampling.resolved_profile(profile)


@pytest.mark.parametrize('kwargs', [
    {'suite_seed': True, 'case_id': 'x'}, {'suite_seed': -1, 'case_id': 'x'},
    {'suite_seed': 2**32, 'case_id': 'x'}, {'suite_seed': 1, 'case_id': ''},
    {'suite_seed': 1, 'case_id': 'x', 'replicate': -1},
])
def test_invalid_seed_identity_refused(kwargs):
    with pytest.raises(ValueError):sampling.case_seed(**kwargs)


def test_invalid_shape_and_top_k_refuse_without_consuming_draw():
    sampler = sampling.CaseSampler(suite_seed=1, case_id='shape')
    with pytest.raises(ValueError):sampler.draw(mx.array([[1., 2.]]))
    with pytest.raises(ValueError):sampler.draw(mx.array([1., 2.]))
    assert sampler.receipt()['rng_draws'] == 0
