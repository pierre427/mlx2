"""Pure CPU dependency graph oracle; this does not run a CED executor."""
import json
import random
import pytest

from mlx2.runtime.models.deepseek_v41_ced_plan import build_plan


def independent_graph_oracle(tokens, first_decoder, layers, window, chunk, output_rows):
    """Enumerate token nodes and edges rather than repeat interval recursion."""
    needed = {i: set() for i in range(first_decoder, layers+1)}
    needed[layers].update(range(tokens-output_rows, tokens))
    # Local state at each decoder needs the last W input activations even if
    # a caller only requests final logits. These are separate graph roots.
    for layer in range(first_decoder, layers):
        needed[layer].update(range(max(0, tokens-window), tokens))
    outputs, inputs = {}, {}
    for layer in reversed(range(first_decoder, layers)):
        wanted = needed[layer+1]
        aligned = set()
        for row in wanted:
            block = row//chunk
            aligned.update(range(block*chunk, min(tokens, (block+1)*chunk)))
        # The required sets are suffixes, but derive all causal edges directly.
        for row in aligned:
            for source in range(max(0, row-window+1), row+1):
                needed[layer].add(source)
        outputs[layer] = aligned
        inputs[layer] = set(needed[layer])
    return outputs, inputs


def plan(**kwargs):
    args = dict(prompt_tokens=16384, encoder_layers=20, total_layers=40,
                sliding_window=128, reference_chunk_rows=2048,
                pool_source_layers=(2, 8, 14), index_source_layers=(2, 8, 14, 24, 28),
                engram_layers=(1, 14))
    args.update(kwargs)
    return build_plan(**args)


def test_random_small_graphs_have_every_output_and_cache_dependency():
    rng = random.Random(41026)
    for _ in range(100):
        n = rng.randint(1, 80); layers = rng.randint(2, 8); enc = rng.randint(1, layers-1)
        w = rng.randint(1, 12); chunk = rng.randint(1, 13); outputs = rng.randint(1, n)
        p = build_plan(prompt_tokens=n, encoder_layers=enc, total_layers=layers,
                       sliding_window=w, reference_chunk_rows=chunk, output_rows=outputs)
        expected_out, expected_in = independent_graph_oracle(n, enc, layers, w, chunk, outputs)
        for step in p.decoder_steps:
            assert set(range(step.aligned_output.first, n)) == expected_out[step.layer]
            assert set(range(step.required_input.first, n)) == expected_in[step.layer]
            assert step.aligned_output.first % chunk == 0
            assert step.warmed_input.first == step.required_input.first
            assert step.warmed_input.stop == step.aligned_output.first
            assert set(range(step.final_local_kv_input.first, n)) <= expected_in[step.layer]


def test_unaligned_short_prompt_keeps_full_depth_and_long_prompt_has_cone():
    short = plan(prompt_tokens=100)
    assert short.row_evaluations == short.ordinary_row_evaluations == 4000
    for length in (127, 128, 129, 2047, 2048, 2049, 8192, 8406, 16384, 32768):
        p = plan(prompt_tokens=length)
        assert p.encoder_input.first == p.global_pool_input.first == 0
        assert p.encoder_input.stop == p.global_pool_input.stop == length
        assert p.row_evaluations <= p.ordinary_row_evaluations
        assert all(s.required_input.first <= s.aligned_output.first <= s.required_output.first for s in p.decoder_steps)
    long = plan(prompt_tokens=32768)
    assert long.row_evaluations < long.ordinary_row_evaluations
    assert long.decoder_steps[-1].required_output.rows == 1
    assert long.decoder_steps[0].required_input.rows > 128


def test_dependency_cone_is_not_universal_last_128_replay():
    p = plan(prompt_tokens=8192, reference_chunk_rows=1)
    assert p.decoder_steps[-1].aligned_output.rows == 1
    assert p.decoder_steps[-1].warmed_input.rows == 127
    assert p.decoder_steps[0].aligned_output.rows == 1+19*127
    assert p.decoder_steps[0].required_input.rows == 1+20*127


@pytest.mark.parametrize('capability', ['prompt_logprobs', 'all_hidden_states'])
def test_all_row_capabilities_choose_ordinary(capability):
    p = plan(**{capability: True})
    assert p.mode == 'ordinary' and p.reason == 'all_row_capability_requires_ordinary'
    assert p.row_evaluations == p.ordinary_row_evaluations
    assert all(s.aligned_output.first == 0 for s in p.decoder_steps)


def test_canonical_decoder_pool_owner_and_engram_are_explicit_ordinary_fallbacks():
    p = plan(pool_source_layers=(2, 8, 14, 20), index_source_layers=(2, 8, 14, 20, 24, 28, 32, 36))
    assert p.mode == 'ordinary' and p.reason == 'boundary_source_preparation_unimplemented'
    assert p.row_evaluations == p.ordinary_row_evaluations
    assert p.proposed_decoder_steps and p.proposed_decoder_steps[-1].required_output.rows == 1
    boundary = [stage for stage in p.pool_preparations if stage.layer == 20]
    assert len(boundary) == 1 and boundary[0].phase == 'boundary_full_prefix_before_suffix'
    assert boundary[0].source_input.first == 0 and boundary[0].source_input.stop == p.prompt_tokens
    assert boundary[0].required_frontiers == ('hidden', 'staggered_pre', 'attention_input_mix_and_norm')
    assert not boundary[0].execution_supported and p.boundary_projection_rows == p.prompt_tokens
    assert all(stage.source_input.rows == p.prompt_tokens for stage in p.pool_preparations)
    assert [(step.layer, step.pool_source_layer) for step in p.index_refreshes[-4:]] == [(24, 20), (28, 20), (32, 20), (36, 20)]
    assert all(step.consume_complete_global_pool for step in p.index_refreshes)
    p = plan(pool_source_layers=(2, 8, 14, 24), index_source_layers=(2, 8, 14, 24, 28))
    assert p.mode == 'ordinary' and p.reason == 'decoder_owned_global_pools_unsupported'
    assert not p.proposed_decoder_steps
    p = plan(engram_layers=(1, 24))
    assert p.mode == 'ordinary' and p.reason == 'decoder_owned_engram_unsupported'
    p = plan(pool_source_layers=(2, 8, 14, 20), index_source_layers=(2, 8, 14, 20), engram_layers=(1, 24))
    assert p.reason == 'decoder_owned_engram_unsupported' and not p.proposed_decoder_steps
    p = plan(cached_prefix_tokens=256)
    assert p.mode == 'ordinary' and p.reason.startswith('resumed_prefix')


def test_receipt_is_dependency_only_identity_bound_and_nonexecutable():
    a = plan(); b = plan(reference_chunk_rows=1024)
    assert a.dependency_only and not a.execution_supported and not a.qualified
    assert a.receipt()['identity'] != b.receipt()['identity']
    assert json.loads(json.dumps(a.receipt()))['dependency_only'] is True


@pytest.mark.parametrize('kwargs', [dict(prompt_tokens=0), dict(prompt_tokens=True),
 dict(encoder_layers=0), dict(encoder_layers=40), dict(sliding_window=0),
 dict(reference_chunk_rows=0), dict(reference_chunk_rows=True), dict(output_rows=0),
 dict(output_rows=20000), dict(prompt_logprobs=1), dict(cached_prefix_tokens=-1),
 dict(pool_source_layers=(2, 2)), dict(pool_source_layers=[2]),
 dict(pool_source_layers=(2,), index_source_layers=()), dict(engram_layers=(40,)),
 dict(pool_source_layers=(2,), index_source_layers=(1, 2))])
def test_invalid_topology_or_extent_refused(kwargs):
    with pytest.raises(ValueError): plan(**kwargs)
