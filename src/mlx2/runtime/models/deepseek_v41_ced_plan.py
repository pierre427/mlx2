# SPDX-License-Identifier: MIT
"""CPU-only CED dependency planning; never an authority to execute/publish.

The planner describes a suffix cone for decoder-local attention when complete
encoder-owned global pools already exist. It does not establish numerical
equivalence, construct model state, or implement a CED serving route.
Provenance: provenance/deepseek-v41-ced-dependency-plan.json.
"""
from dataclasses import asdict, dataclass
import hashlib
import json

REVISION = 'source_ced_dependency_plan_v1'


@dataclass(frozen=True)
class Interval:
    first: int
    stop: int

    @property
    def rows(self):
        return self.stop-self.first


@dataclass(frozen=True)
class DecoderStep:
    layer: int
    required_output: Interval
    aligned_output: Interval
    required_input: Interval
    warmed_input: Interval
    final_local_kv_input: Interval


@dataclass(frozen=True)
class PoolPreparation:
    layer: int
    source_input: Interval
    phase: str
    required_frontiers: tuple
    publication: str = 'complete KV and index pools before decoder suffix consumption'
    execution_supported: bool = False


@dataclass(frozen=True)
class IndexRefresh:
    layer: int
    pool_source_layer: int
    consume_complete_global_pool: bool = True
    execution_supported: bool = False


@dataclass(frozen=True)
class CedPlan:
    mode: str
    reason: str
    prompt_tokens: int
    encoder_layers: int
    total_layers: int
    sliding_window: int
    reference_chunk_rows: int
    requested_output_rows: int
    encoder_input: Interval
    global_pool_input: Interval
    decoder_steps: tuple
    proposed_decoder_steps: tuple
    pool_preparations: tuple
    index_refreshes: tuple
    pool_source_layers: tuple
    index_source_layers: tuple
    engram_layers: tuple
    row_evaluations: int
    ordinary_row_evaluations: int
    boundary_projection_rows: int
    dependency_only: bool = True
    execution_supported: bool = False
    qualified: bool = False

    def receipt(self):
        result = asdict(self)
        result['revision'] = REVISION
        result['numerical_gate'] = 'pending; dependency closure is not numerical/state equivalence'
        result['global_pool_contract'] = 'full encoder/boundary-owned pools and complete index-key refreshes remain required'
        result['identity'] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
        return result


def _integer(name, value, minimum, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f'CED {name} is outside the admitted integer extent')


def _layers(name, values, total):
    if not isinstance(values, tuple) or tuple(sorted(set(values))) != values or any(
            type(i) is not int or not 0 <= i < total for i in values):
        raise ValueError(f'CED {name} must be unique ordered layer IDs')


def build_plan(*, prompt_tokens, encoder_layers, total_layers, sliding_window,
               reference_chunk_rows, output_rows=1, pool_source_layers=(),
               index_source_layers=(), engram_layers=(), prompt_logprobs=False,
               all_hidden_states=False, cached_prefix_tokens=0):
    """Plan full encoder rows and dependency-complete aligned decoder suffixes.

    Alignment is an explicit reference graph input, not a universal 2048-row
    rule. Each decoder's warm *input* interval precedes its output interval.
    The required input also covers the final W local-KV rows. Unsupported
    global ownership, resumed seams or all-row capabilities choose ordinary
    execution as a recorded fallback. No returned plan is executable yet.
    """
    _integer('prompt tokens', prompt_tokens, 1)
    _integer('total layers', total_layers, 2)
    _integer('encoder layers', encoder_layers, 1, total_layers-1)
    _integer('sliding window', sliding_window, 1)
    _integer('reference chunk rows', reference_chunk_rows, 1)
    _integer('output rows', output_rows, 1, prompt_tokens)
    _integer('cached prefix tokens', cached_prefix_tokens, 0)
    if type(prompt_logprobs) is not bool or type(all_hidden_states) is not bool:
        raise ValueError('CED all-row capabilities must be boolean')
    for name, values in (('pool sources', pool_source_layers),
                         ('index sources', index_source_layers), ('Engram layers', engram_layers)):
        _layers(name, values, total_layers)
    if not set(pool_source_layers).issubset(index_source_layers):
        raise ValueError('CED every global KV owner requires an index source')
    if any(not any(owner <= layer for owner in pool_source_layers) for layer in index_source_layers):
        raise ValueError('CED index refresh has no preceding global KV owner')
    reason = 'dependency_complete_suffix'
    if prompt_logprobs or all_hidden_states:
        reason = 'all_row_capability_requires_ordinary'
    elif cached_prefix_tokens:
        reason = 'resumed_prefix_requires_ordinary_until_seam_validation'
    elif any(i > encoder_layers for i in pool_source_layers):
        reason = 'decoder_owned_global_pools_unsupported'
    elif any(i >= encoder_layers for i in engram_layers):
        reason = 'decoder_owned_engram_unsupported'
    elif encoder_layers in pool_source_layers:
        reason = 'boundary_source_preparation_unimplemented'
    ordinary = reason != 'dependency_complete_suffix'
    full = Interval(0, prompt_tokens)
    cache = Interval(max(0, prompt_tokens-sliding_window), prompt_tokens)
    def suffix_steps(full_depth):
        steps = []
        required = prompt_tokens-output_rows
        for layer in reversed(range(encoder_layers, total_layers)):
            wanted = 0 if full_depth else required
            aligned = 0 if full_depth else wanted//reference_chunk_rows*reference_chunk_rows
            # A sliding window includes the current token, hence W-1 predecessors.
            warm = max(0, aligned-(sliding_window-1))
            input_first = min(warm, cache.first)
            steps.append(DecoderStep(layer, Interval(wanted, prompt_tokens),
                Interval(aligned, prompt_tokens), Interval(input_first, prompt_tokens),
                Interval(input_first, aligned), cache))
            required = input_first
        return tuple(reversed(steps))
    steps = suffix_steps(ordinary)
    proposed = suffix_steps(False) if reason in (
        'dependency_complete_suffix', 'boundary_source_preparation_unimplemented') else ()
    preparations = tuple(PoolPreparation(i, full,
        'encoder_full_prefix' if i < encoder_layers else
        ('boundary_full_prefix_before_suffix' if i == encoder_layers else 'unsupported_decoder_owner'),
        ('hidden', 'staggered_pre', 'attention_input_mix_and_norm')) for i in pool_source_layers)
    refreshes = tuple(IndexRefresh(i, max(owner for owner in pool_source_layers if owner <= i))
                      for i in index_source_layers)
    row_work = encoder_layers*prompt_tokens+sum(s.aligned_output.rows for s in steps)
    return CedPlan('ordinary' if ordinary else 'dependency_suffix', reason,
        prompt_tokens, encoder_layers, total_layers, sliding_window,
        reference_chunk_rows, output_rows, full, full, steps, proposed, preparations, refreshes,
        pool_source_layers, index_source_layers, engram_layers,
        row_work, total_layers*prompt_tokens,
        prompt_tokens if encoder_layers in pool_source_layers else 0)
