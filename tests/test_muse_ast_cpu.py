"""Muse AST: mixed positional topology, mask-row DSpark, real tree commits."""
import copy
from dataclasses import asdict

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters.muse_glimmer import (MUSE_AST_ROUTE, MuseGlimmerAdapter,
    configure_environment, normalize_external_policy)
from mlx2.adapters.muse_dspark import config_from_hf
from mlx2.runtime.attention_tree_executor import AttentionTreeExecutor
from mlx2.runtime.drafters.muse_dspark import MuseDSparkConfig, MuseDSparkDraftModel
from test_muse_dflash_assistant_cpu import tiny, greedy_reference, drain


@pytest.fixture(autouse=True)
def cpu():
    mx.set_default_device(mx.cpu)


def test_policy_is_ast_only_and_rejects_chain_parking():
    policy = normalize_external_policy({'draft_model': '/draft', 'batch_size_route': MUSE_AST_ROUTE})
    assert policy['num_draft'] == 15
    assert policy['cost_per_commit_handoff'] is False
    assert policy['proposal_composition'] is False
    for key, value in [('cost_per_commit_handoff', True), ('proposal_composition', {}),
                       ('progressive_verification_tile', 2)]:
        with pytest.raises(ValueError):
            normalize_external_policy({**policy, key: value})


@pytest.mark.parametrize('prompt_length', [3, 13])
def test_muse_local_rope_tree_matches_each_serial_path_and_commit(prompt_length):
    target, draft = tiny()
    initial = target.make_cache()
    target(mx.array([[1] * prompt_length]), cache=initial)
    mx.eval([c.state for c in initial])
    tokens, parents = [4, 5, 6, 7, 8, 9], [-1, 0, 1, 0, 3, 4]
    cache = copy.deepcopy(initial)
    executor = AttentionTreeExecutor.create()
    logits, taps, transaction = executor.forward(target, tokens, parents, cache, [1, 3], cached=False)
    mx.eval(logits, taps)
    for position in range(len(tokens)):
        node, path = position, []
        while node >= 0:
            path.append(tokens[node]); node = parents[node]
        reference = copy.deepcopy(initial)
        for token in reversed(path):
            expected = target(mx.array([[token]]), cache=reference)
        np.testing.assert_allclose(np.asarray(logits[:, position].astype(mx.float32)),
            np.asarray(expected[:, -1].astype(mx.float32)), rtol=2e-4, atol=2e-4)
    transaction.abort()
    assert [c.offset for c in cache] == [c.offset for c in initial]


@pytest.mark.parametrize('kind', ['assistant', 'dspark'])
def test_ast_executor_generates_like_ordinary_without_chain_or_parking(kind):
    configure_environment(ast=True)
    target, draft = tiny()
    if kind == 'dspark':
        draft = MuseDSparkDraftModel(MuseDSparkConfig(**asdict(draft.config), markov_rank=4)).bind(target)
        mx.eval(draft.parameters())
    adapter = object.__new__(MuseGlimmerAdapter)
    adapter.model, adapter.draft_model = target, draft
    adapter.identity = {'fingerprint': 'tiny-muse-ast'}
    adapter.external_policy = normalize_external_policy({'draft_model': '/draft',
        'batch_size_route': MUSE_AST_ROUTE, 'num_draft': 5})
    batch = adapter.create_external_batch(completion_batch_size=2, prefill_step_size=4)
    prompts = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10]]
    uids = batch.insert(prompts, max_tokens=[12, 12])
    outputs, final = drain(batch)
    for uid, prompt in zip(uids, prompts):
        assert outputs[uid] == greedy_reference(target, prompt, 12)
    assert batch.dynamic_tree_overflow == 'cohort'
    assert batch.scheduler_stats['external_auto_tree_rounds'] > 0
    assert batch.scheduler_stats['external_auto_chain_rounds'] == 0
    assert draft.stats['tree_lattices'] > 0


def test_dspark_false_anchor_and_absent_confidence_are_exact_contract():
    config = {'architectures': ['Qwen3DSparkModel'], 'sample_from_anchor': False,
        'markov_rank': 4, 'hidden_size': 16, 'intermediate_size': 32,
        'num_hidden_layers': 2, 'num_attention_heads': 4, 'num_key_value_heads': 2,
        'head_dim': 4, 'vocab_size': 96, 'target_layer_ids': [1, 3],
        'mask_token_id': 95, 'block_size': 6, 'sliding_window': 8}
    target = {'hidden_size': 16, 'vocab_size': 96, 'num_hidden_layers': 4}
    args = config_from_hf(config, target_text=target)
    assert args.markov_rank == 4 and args.block_size == 6
    for alteration in ({'sample_from_anchor': True}, {'enable_confidence_head': True},
                       {'quantization': {'bits': 4}}):
        with pytest.raises(ValueError):
            config_from_hf({**config, **alteration}, target_text=target)


def test_dspark_lattice_uses_anchor_and_each_actual_parent_markov_factor():
    target, assistant = tiny()
    draft = MuseDSparkDraftModel(MuseDSparkConfig(**asdict(assistant.config), markov_rank=4)).bind(target)
    mx.random.seed(74)
    base = mx.random.normal((2, 3, draft.config.vocab_size))
    anchors = [3, 7]
    candidates, unary, root, pair = draft._tree_lattice(base, anchors)
    base, candidates, unary, root, pair, w1, w2 = [np.asarray(x.astype(mx.float32)) for x in
        (base, candidates, unary, root, pair, draft.markov_head.markov_w1.weight,
         draft.markov_head.markov_w2.weight)]
    candidates = candidates.astype(int)
    for row in range(2):
        previous = anchors[row]
        for depth in range(3):
            # Candidate narrowing follows the greedy conditional chain, while
            # every tree edge below uses its own predecessor (not that chain).
            conditional = base[row, depth] + w2 @ w1[previous]
            assert set(candidates[row, depth]) == set(np.argsort(conditional)[-8:])
            previous = int(np.argmax(conditional))
            np.testing.assert_array_equal(unary[row, depth], base[row, depth, candidates[row, depth]])
            if depth == 0:
                np.testing.assert_allclose(root[row], w2[candidates[row, depth]] @ w1[anchors[row]], atol=1e-6)
            else:
                expected = w2[candidates[row, depth]] @ w1[candidates[row, depth - 1]].T
                np.testing.assert_allclose(pair[row, depth - 1], expected, atol=1e-6)
    with pytest.raises(ValueError, match='AST'):
        draft.draft_distributions()


def test_cli_selects_muse_owned_tree_route_for_each_learned_drafter(tmp_path):
    import json
    from types import SimpleNamespace
    from mlx2.ast_observability import route_status
    from mlx2.server import apply_ast_invocation

    for kind, architecture in [('dflash', 'MuseGlimmerAssistantModel'),
                              ('dspark', 'Qwen3DSparkModel'), ('dflash2', 'DFlash2DraftModel')]:
        path = tmp_path / kind
        path.mkdir()
        (path / 'config.json').write_text(json.dumps({'architectures': [architecture]}))
        args = SimpleNamespace(model='/target', ast=True, ast_drafter=str(path),
            ast_topology=None, ast_graft_pld=False, draft_model=None, external_draft=False)
        resolution = SimpleNamespace(adapter_type=MuseGlimmerAdapter, artifact={}, default_route='ordinary')
        policy = normalize_external_policy(apply_ast_invocation(args, None, resolution))
        assert policy['batch_size_route'] == MUSE_AST_ROUTE and policy['draft_model'] == str(path)
        assert args.ast_route['tree_tiles'] and args.external_draft
        view = route_status(args.ast_route, policy, MuseGlimmerAdapter)
        assert view['drafter_kind'] == kind and view['tile_rows_mode'] == 'fixed'


def test_discovery_binds_dspark_through_its_strict_muse_inspector(monkeypatch):
    from mlx2.adapters import drafter_pairing, muse_dspark

    calls = []
    def inspect(draft, target):
        calls.append((draft, target))
        return {'fingerprint': 'bound'}
    monkeypatch.setattr(muse_dspark, 'inspect_drafter', inspect)
    pairing = drafter_pairing.pair_drafter({'path': '/target', 'adapter': 'MuseGlimmerAdapter'},
        {'path': '/draft', 'name': 'dspark', 'architectures': ['Qwen3DSparkModel']})
    assert calls == [('/draft', '/target')]
    assert pairing['execution_policy']['batch_size_route'] == MUSE_AST_ROUTE
    assert pairing['identity'] == 'bound'


def test_inventory_separates_muse_auto_pld_from_all_three_explicit_ast_pairs():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / 'scripts/model_matrix/inventory.py'
    spec = importlib.util.spec_from_file_location('muse_inventory', path)
    inventory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inventory)
    assert inventory.AST_ROUTE_KIND['MuseGlimmerAdapter'] == 'prompt_lookup'
    for kind in ('MuseGlimmerAssistantModel', 'Qwen3DSparkModel', 'DFlash2DraftModel'):
        route = inventory.ast_drafter_route({'adapter': 'MuseGlimmerAdapter'},
            {'draft_kind': kind, 'draft_path': '/draft', 'draft': 'artifact@pin'}, 'test')
        assert route['server_flags'] == ['--ast', '--ast-drafter', '/draft']
        assert route['expected_route_kind'] == 'external_draft'
