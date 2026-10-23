"""Host regressions for the 2026-10-23 handover; no checkpoint is loaded."""
from types import SimpleNamespace

import pytest

from mlx2.server import apply_ast_invocation
from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator, TreeDraftRow


def args(**kw):
    values = dict(model='/target', ast=True, ast_drafter=None, ast_topology=None,
                  ast_graft_pld=False, draft_model=None, external_draft=False,
                  native_mtp=False, prompt_lookup=False)
    values.update(kw)
    return SimpleNamespace(**values)


def resolution(adapter=None):
    adapter = adapter or type('Adapter', (), {'ast_tree_tiles': True, 'ast_tree_tile_dtype_gate': False})
    return SimpleNamespace(adapter_type=adapter, artifact={}, default_route='ordinary')


@pytest.mark.parametrize('drafter', [None, 'pld'])
def test_asta1_ling_model_free_drafter_selects_supported_chain(drafter):
    from mlx2.adapters.ling3_flash import Ling3FlashAdapter
    a = args(ast_drafter=drafter)
    policy = apply_ast_invocation(a, None, resolution(Ling3FlashAdapter))
    assert policy == {} and a.prompt_lookup and not a.external_draft
    assert a.ast_route['tree_tiles'] is False


def test_asta2_native_numerical_defaults_reach_status(monkeypatch):
    from mlx2.adapters.qwen36_35b import Qwen3635BA3BAdapter
    from mlx2.adapters import tensorfold_dtype_gate
    from mlx2.ast_observability import route_status
    monkeypatch.setattr(tensorfold_dtype_gate, 'artifact_tree_tile_dtype_issue', lambda *a, **k: None)
    r = resolution(Qwen3635BA3BAdapter)
    r.artifact = {'has_mtp': True}
    a = args()
    policy = apply_ast_invocation(a, None, r)
    assert policy['ast_adaptive_rows'] is False
    assert policy['cost_per_commit_handoff'] is False
    assert a.ast_route['drafter_policy_defaults']['ast_adaptive_rows'] is False
    assert route_status(a.ast_route, policy, Qwen3635BA3BAdapter)['tile_rows_mode'] == 'fixed'


class Base:
    receipt_kind = 'external_dflash2'
    receipt_settings = {'kind': 'dflash2'}
    config = SimpleNamespace(block_size=16)
    def propose_tree(self, anchors, hidden, cache, max_nodes, **kw):
        return [([900 + i for i in range(7)], [-1] + list(range(6))) for _ in anchors]


def graft():
    from mlx2.runtime.drafters.prompt_lookup_draft import PromptLookupDraftModel
    target = SimpleNamespace(args=SimpleNamespace(vocab_size=1000, hidden_size=8), layers=[None])
    return PromptLookupDraftModel(target, num_draft=15, base=Base())


def test_asta3_graft_copy_depth_uses_available_node_budget():
    span = list(range(100, 112))
    history = [1, 2, 3] + span + [4, 5, 6] + span[:-1]
    tokens, parents = graft().propose_tree([span[-1]], None, None, 15,
                                          processor_histories=[history], topology='3x5')[0]
    assert tokens[:8] == [4, 5, 6, 100, 101, 102, 103, 104]
    assert parents[:8] == [-1, 0, 1, 2, 3, 4, 5, 6]


def test_asta4_graft_preserves_learned_drafter_receipt():
    from mlx2.adapters.qwen38_27b import _external_speculation_label
    model = graft()
    assert model.receipt_kind == 'external_dflash2'
    assert model.receipt_settings['graft']['kind'] == 'prompt_lookup'
    assert _external_speculation_label(model) == 'external-dflash2-implemented-unqualified'


def test_asta5_graft_onto_model_free_drafter_refused_before_load():
    with pytest.raises(ValueError, match='graft.*PLD|graft.*pld|graft.*prompt.lookup'):
        apply_ast_invocation(args(ast_drafter='pld', ast_graft_pld=True), None, resolution())


@pytest.mark.parametrize('knob,value', [('ast_tiles', 3), ('ast_tile_rows', 2)])
def test_asta6_invalid_tile_knobs_refused_before_load(knob, value):
    with pytest.raises(ValueError, match=knob.replace('_', '[-_]')):
        apply_ast_invocation(args(ast_drafter='/draft', **{knob: value}), None, resolution())


def test_astc1_asta7_dtype_refusal_precedes_feedback_and_executor(monkeypatch):
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter
    from mlx2.adapters import tensorfold_dtype_gate
    from mlx2.adapters.qwen38_target_tree_executor import Qwen38TargetTreeExecutor
    adapter = object.__new__(Qwen3827BAdapter)
    adapter.draft_model = object()
    adapter.model = object()
    adapter.external_policy = {'ast_target_tree_oracle': '/must-not-read.json'}
    calls = []
    adapter._initialize_external_feedback = lambda: calls.append('feedback')
    monkeypatch.setattr(Qwen38TargetTreeExecutor, 'create', lambda **kw: calls.append('executor'))
    monkeypatch.setattr(tensorfold_dtype_gate, 'model_tree_tile_dtype_issue', lambda *a, **k: 'requires bf16')
    with pytest.raises(ValueError, match='route refused.*bf16'):
        adapter.create_external_batch(completion_batch_size=1)
    assert calls == []
    assert not getattr(adapter, '_external_execution_started', False)


def test_astc2_confirmed_stop_is_accepted_without_committing_stop_state():
    g = object.__new__(ExternalDraftBatchGenerator)
    g.stops = {0}
    g.tree_gates = {'logprobs_on_request': True}
    g.scheduler_stats = {}
    g._target_law = lambda *a, **kw: None
    lane = SimpleNamespace(anchor=9, history=[], sampling={'emit_logprobs': False}, rng=SimpleNamespace(sample=lambda law: next(samples)))
    samples = iter([5, 0])
    block = TreeDraftRow([5, 0], [-1, 0])
    import numpy as np
    decision = g._verify_tree(lane, block, np.zeros((1, 3, 10)))
    assert decision.accepted == 2
    assert decision.commit_rows == [0, 1]
    assert decision.emitted == [5, 0]
    from mlx2.runtime.ast_tree_calibration import tree_records
    records = tree_records(block.parents, [1, 2], [-.1, -.2], decision.commit_rows,
                           accepted_rows=decision.accepted_rows)
    assert records == [(1, -.1, True), (2, -.2, True)]


def _host_tree_round(monkeypatch, sizes):
    import numpy as np
    from mlx2.runtime import external_speculative as mod
    g = object.__new__(ExternalDraftBatchGenerator)
    g.target_execution = 'provider'
    g.target_tree_cohort_limit = 4
    g.round_timing = False
    g.mx = SimpleNamespace(eval=lambda *a: None)
    g.tree_gates = {'batched_laws': False, 'pipeline_draft': False, 'single_fence': False}
    g.draft = object()
    g._snapshot_round = lambda lanes: []
    g._snapshot_scheduler_stats = lambda: dict(g.scheduler_stats)
    g.scheduler_stats = {'recovery_checkpoint_captures': 0}
    blocks = [TreeDraftRow(list(range(n)), [-1] + list(range(n - 1))) for n in sizes]
    g._propose = lambda lanes: blocks
    g._target_tree_forward_many = lambda *a: (np.zeros((len(sizes), max(sizes)+1, 1)), None, None)
    g._presnapshot_draft = lambda lanes: None
    g._verify_tree = lambda *a: mod.RoundDecision(0, [99], None, commit_rows=[0])
    g._discard_prelaunched = lambda uids: None
    g._commit = lambda *a, **kw: clock.__setitem__(0, 10.)
    costs, rates, acceptances = [], [], []
    g.tile_budget = SimpleNamespace(snapshot=lambda: None,
        observe=lambda *a: acceptances.append(a), receipt=lambda: {},
        observe_cost=lambda *a: costs.append(a))
    g.tree_calibration = SimpleNamespace(sizes=(), snapshot=lambda: None,
        observe_rate=lambda *a: rates.append(a), receipt=lambda: {})
    g._observe_tree_calibration = lambda lanes, blocks, decisions, seconds: (
        rates.append((len(decisions), seconds)) if seconds is not None else None)
    clock = [1.]
    monkeypatch.setattr(mod.time, 'perf_counter', lambda: clock[0])
    lanes = [SimpleNamespace(uid=i, anchor=50, external_rounds=2) for i in range(len(sizes))]
    g._tree_round(lanes)
    return g, costs, rates, acceptances


def test_astc3_calibration_rate_and_cost_include_the_same_commit_span(monkeypatch):
    _, costs, rates, _ = _host_tree_round(monkeypatch, [3])
    assert costs == [(1, 3, 9.)]
    assert rates == [(1, 9.)]


def test_astc4_padded_cost_uses_physical_width_but_acceptance_is_lane_local(monkeypatch):
    g, costs, _, acceptances = _host_tree_round(monkeypatch, [3, 15])
    assert costs == [(2, 15, 9.), (2, 15, 9.)]
    assert [a[1] for a in acceptances] == [3, 15]
    assert g.scheduler_stats['external_tree_verified_nodes'] == {15: 2}
    assert g.scheduler_stats['external_tree_nodes'] == 30


def test_mod1_flint_one_token_prefill_keeps_ordinary_blocks():
    import mlx.core as mx
    from test_flint_ast_tiles_cpu import tiny
    from mlx2.runtime.generate import prefill_model_forward
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        model = tiny()
        calls = []
        class Block:
            layers = model.layers
            def __call__(self, hidden, caches):
                calls.append(tuple(c.offset for c in caches))
                return hidden
        model.model.fused_blocks = [Block()]
        cache = model.make_cache()
        prefill_model_forward(model)(mx.array([[3]]), cache=cache)
        assert calls == []
        prefill_model_forward(model)(mx.array([[4, 5, 6]]), cache=cache)
        prefill_model_forward(model)(mx.array([[7]]), cache=cache)
        assert calls == []
        model(mx.array([[8]]), cache=cache)
        assert len(calls) == 1
    finally:
        mx.set_default_device(previous)


def test_astc2_stop_child_descendants_are_not_calibration_trials():
    from mlx2.runtime.ast_tree_calibration import tree_records
    assert tree_records([-1, 0, 1], [1, 2, 3], [-.1, -.2, -.3], [0, 1],
                        accepted_rows=[1, 2]) == [(1, -.1, True), (2, -.2, True)]


def test_asta2_explicit_numerical_policy_overrides_defaults(monkeypatch):
    from mlx2.adapters.qwen36_35b import Qwen3635BA3BAdapter
    from mlx2.adapters import tensorfold_dtype_gate
    monkeypatch.setattr(tensorfold_dtype_gate, 'artifact_tree_tile_dtype_issue', lambda *a, **k: None)
    r = resolution(Qwen3635BA3BAdapter)
    a = args(ast_drafter='pld')
    policy = apply_ast_invocation(a, {'ast_adaptive_rows': True}, r)
    assert policy['ast_adaptive_rows'] is True
    assert policy['cost_per_commit_handoff'] is False


def test_asta6_inspection_rejects_bad_knobs_before_drafter_io():
    from mlx2.adapters.qwen38_27b import inspect_external_policy
    with pytest.raises(ValueError, match='ast_tiles'):
        inspect_external_policy({'draft_model': '/does-not-exist', 'ast_tiles': 3}, '/target')


def test_asta6_ling_declares_its_larger_packed_tile_limit():
    from mlx2.adapters.ling3_flash import Ling3FlashAdapter
    a = args(ast_drafter='/ubtree', ast_tile_rows=32)
    policy = apply_ast_invocation(a, None, resolution(Ling3FlashAdapter))
    assert policy['ast_tile_rows'] == 32
    with pytest.raises(ValueError, match='ast_tile_rows'):
        apply_ast_invocation(args(ast_drafter='/ubtree', ast_tile_rows=33), None,
                             resolution(Ling3FlashAdapter))


@pytest.mark.parametrize('length,cached,expected_prefills', [(1, 0, [1]), (6, 0, [4, 1, 1]), (6, 4, [1, 1])])
def test_mod1_scheduler_marks_every_prompt_chunk_and_tail(monkeypatch, length, cached, expected_prefills):
    import mlx.core as mx
    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.models.flint import Model
    from test_flint_ast_tiles_cpu import tiny
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    generator = None
    try:
        model = tiny()
        prefix = [1, 2, 3, 4][:cached]
        prompt = (prefix + [5, 6]) if cached else list(range(1, length + 1))
        cache = model.make_cache()
        if prefix:
            model.prefill_forward(mx.array([prefix]), cache=cache)
        fused_offsets, prefills = [], []
        ordinary_prefill = Model.prefill_forward
        def recorded_prefill(self, inputs, cache=None):
            prefills.append(inputs.shape[1])
            return ordinary_prefill(self, inputs, cache=cache)
        monkeypatch.setattr(Model, 'prefill_forward', recorded_prefill)
        class Block:
            layers = model.layers
            def __call__(self, hidden, caches):
                fused_offsets.append(int(caches[0].offset))
                return hidden
        model.model.fused_blocks = [Block()]
        generator = BatchGenerator(model, prefill_step_size=4, completion_batch_size=1,
                                   prefill_batch_size=1, sampler=lambda x: mx.argmax(x, axis=-1))
        generator.insert([prompt[cached:]], max_tokens=[2], caches=[cache], all_tokens=[prefix])
        finished = False
        for _ in range(20):
            _, responses = generator.next()
            if any(r.finish_reason for r in responses):
                finished = True
                break
        assert finished
        assert prefills == expected_prefills
        assert fused_offsets and all(offset == length for offset in fused_offsets)
    finally:
        if generator is not None:
            generator.close()
        mx.set_default_device(previous)
