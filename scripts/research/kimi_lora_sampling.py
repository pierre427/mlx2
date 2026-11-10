"""Isolated evaluation draws using mlx2's existing ordinary-serving law.

No model load, global RNG reseed, or runner hook. The caller owns one instance
per case/replicate and passes the raw one-dimensional next-token logits.
The unchanged greedy operation is mx.argmax(logits). Sampled draws use the
same LaneRNG/transformed-logprobs/keyed-categorical APIs as serving.py.
"""
from __future__ import annotations

import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src

import hashlib
import json
import math
from types import MappingProxyType

from mlx2.sampling_defaults import LEGACY_FALLBACK, SAMPLING_FIELDS, resolve_sampling

SEED_VERSION = 'kimi-evaluation-case-seed-v1'
SERVING_PROFILE = MappingProxyType(dict(LEGACY_FALLBACK))
GREEDY_PROFILE = MappingProxyType({**LEGACY_FALLBACK, 'temperature': 0.0, 'top_p': 1.0, 'top_k': 0})


def case_seed(suite_seed: int, case_id: str, replicate: int = 0) -> int:
    """Stable unsigned 32-bit seed; deliberately excludes model/adapter/arm/order."""
    for name, value in (('suite_seed', suite_seed), ('replicate', replicate)):
        if type(value) is not int or not 0 <= value < 2**32:
            raise ValueError(name + ' must be an unsigned 32-bit integer')
    if not isinstance(case_id, str) or not 1 <= len(case_id) <= 512:
        raise ValueError('case_id must be a nonempty bounded string')
    payload = json.dumps([SEED_VERSION, suite_seed, case_id, replicate],
                         ensure_ascii=False, separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], 'big')


def resolved_profile(profile=None):
    """Resolve explicitly to the current Kimi server fallback, with no penalties."""
    request = dict(SERVING_PROFILE if profile is None else profile)
    if set(request) - set(SAMPLING_FIELDS):
        raise ValueError('Unsupported evaluation sampling field')
    for name, value in request.items():
        if name == 'top_k':
            if type(value) is not int or not 0 <= value < 2**31:
                raise ValueError('top_k must be a nonnegative bounded integer')
        elif type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(name + ' must be finite numeric, not boolean')
    effective, _ = resolve_sampling(request, None, thinking=None)
    if not 0 <= effective['temperature'] <= 2:
        raise ValueError('temperature outside 0..2')
    if not 0 <= effective['top_p'] <= 1 or not 0 <= effective['min_p'] <= 1:
        raise ValueError('top_p/min_p outside 0..1')
    if (effective['repetition_penalty'], effective['presence_penalty'], effective['frequency_penalty']) != (1.0, 0.0, 0.0):
        raise ValueError('This minimal evaluation profile supports neutral penalties only')
    return effective


class CaseSampler:
    """Per-case stream; reconstruct with the same identity to replay a case."""

    def __init__(self, profile=None, *, suite_seed: int, case_id: str, replicate: int = 0):
        # Lazy MLX import lets CPU-only profile/receipt tooling inspect this module.
        from mlx2.runtime.sample_utils import LaneRNG, make_transformed_logprobs
        self.profile = MappingProxyType(resolved_profile(profile))
        self.suite_seed, self.case_id, self.replicate = suite_seed, case_id, replicate
        self.seed = case_seed(suite_seed, case_id, replicate)
        self._rng = self._transform = None
        if self.profile['temperature']:
            self._rng = LaneRNG(self.seed)
            self._transform = make_transformed_logprobs(
                self.profile['temperature'], top_p=self.profile['top_p'],
                top_k=self.profile['top_k'], min_p=self.profile['min_p'])

    def draw(self, logits) -> int:
        import mlx.core as mx
        from mlx2.runtime.sample_utils import draw_key
        if logits.ndim != 1 or not logits.size:
            raise ValueError('Expected one nonempty raw next-token logits row')
        if self._rng is None:
            # Preserve the original runner's exact dtype/tie/argmax behavior.
            return int(mx.argmax(logits).item())
        if self.profile['top_k'] >= logits.size:
            raise ValueError('top_k must be smaller than the logits vocabulary')
        return int(mx.random.categorical(self._transform(logits), key=draw_key(self._rng)).item())

    def receipt(self):
        return {'schema': 'kimi-evaluation-sampling-v1', 'profile': dict(self.profile),
                'mode': 'greedy' if self._rng is None else 'sampled',
                'suite_seed': self.suite_seed, 'case_id': self.case_id,
                'replicate': self.replicate, 'derived_seed': self.seed,
                'seed_derivation': SEED_VERSION,
                'rng_draws': 0 if self._rng is None else self._rng.draws,
                'global_training_rng_used': False,
                'law': 'mlx2 ordinary LaneRNG + make_transformed_logprobs + keyed categorical',
                'greedy_law': 'unchanged mx.argmax on raw logits; zero draws'}


SAMPLING_CODE = ('scripts/research/kimi_lora_sampling.py',
                 'src/mlx2/runtime/sample_utils.py', 'src/mlx2/sampling_defaults.py',
                 'src/mlx2/sampling_contract.py')


def generation_profile(spec):
    """Pure receipt law; absent options mean the historical raw-argmax path."""
    explicit = 'sampling_profile' in spec
    if not explicit and any(key in spec for key in ('sampling_seed', 'replicate')):
        raise ValueError('sampling_seed/replicate require an explicit sampling_profile')
    seed_strategy = None
    profile = dict(GREEDY_PROFILE)
    if explicit:
        if not isinstance(spec['sampling_profile'], dict):
            raise ValueError('sampling_profile must be an explicit object')
        profile = resolved_profile(spec['sampling_profile'])
        if 'sampling_seed' not in spec:
            raise ValueError('explicit sampling_profile requires sampling_seed')
        seed, replicate = spec['sampling_seed'], spec.get('replicate', 0)
        case_seed(seed, 'validation', replicate)
        seed_strategy = {'version': SEED_VERSION, 'suite_seed': seed, 'replicate': replicate}
    return {'schema': 'kimi-generation-profile-v1',
            'sampling': 'seeded_categorical' if profile['temperature'] else 'greedy_argmax',
            'profile': profile, 'seed_strategy': seed_strategy,
            'prefill_chunk': 128, 'budget_source': 'exact case records and bound runner source'}


def expected_case_receipt(generation, case_id, draws):
    """Expected immutable sampling fields, also usable without importing MLX."""
    strategy = generation['seed_strategy']
    if strategy is None:
        return {'schema': 'kimi-legacy-greedy-case-v1', 'case_id': case_id, 'rng_draws': 0}
    seed, replicate = strategy['suite_seed'], strategy['replicate']
    return {'schema': 'kimi-evaluation-sampling-v1', 'profile': generation['profile'],
            'mode': 'sampled' if generation['profile']['temperature'] else 'greedy',
            'suite_seed': seed, 'case_id': case_id, 'replicate': replicate,
            'derived_seed': case_seed(seed, case_id, replicate),
            'seed_derivation': SEED_VERSION, 'rng_draws': draws,
            'global_training_rng_used': False,
            'law': 'mlx2 ordinary LaneRNG + make_transformed_logprobs + keyed categorical',
            'greedy_law': 'unchanged mx.argmax on raw logits; zero draws'}
