"""Device-round tree rounds with the DFlash2 drafter (``device_round.tree_lazy``,
``tree_decide``), CPU only.

Tiny random DFlash2 pairs (no real weights): the muse-glimmer pair of
``test_dflash_pair_select`` on the reference tree forward, and the Nemotron-H
pair of ``test_nemotron35_dflash2_cpu`` on the Nemotron target-tree provider
(``device_tree_inputs``) with the bounded-tree gates on (the Qwen3.8 27B
route's shape: batched laws, single fence, draft pipelining).  Pins:
``DFlash2DraftModel.draft_tree_lazy`` is ``propose_tree``'s tree (tokens,
parents, node order, draft cache) with its tokens on the device; with the
switches on, every round's proposal, decision, RNG draws, cache planes,
outputs and tree counters equal the switch-off round's, pipelined or serial;
pipelined lattices are adopted by the device round (counted), and a drafter
that does not declare them declines; sampled lanes keep the host decision,
and a greedy-only declaration declines them; host-twin overrides decline.
"""

from contextlib import closing

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from test_dflash_pair_select import generator, taps, tiny

from mlx2.runtime.drafters.dflash2 import DFlash2DraftModel
from mlx2.runtime.external_speculative import _TREE_GATES
from mlx2.runtime.speculative_device_tree import COUNTERS, DECIDE_SKIPS, DECLINES

mx.set_default_device(mx.cpu)

LAZY = {"tree_lazy": True}
DECIDE = {"tree_lazy": True, "tree_decide": True}
PROMPTS = ([1, 2, 3, 4], [9, 4, 33, 17, 2, 40, 5], [7, 7, 12])


@pytest.fixture(autouse=True)
def tree15(monkeypatch):
    for name, _enabled in _TREE_GATES.values():
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("MLX2_QWEN_TARGET_EXECUTION", raising=False)
    monkeypatch.delenv("MLX2_TARGET_TREE_EXECUTION", raising=False)
    monkeypatch.setenv("MLX2_DFLASH_TOPOLOGY", "tree15")


def _gates(monkeypatch, **gates):
    for gate, value in gates.items():
        name, enabled = _TREE_GATES[gate]
        monkeypatch.setenv(name, enabled if value else "0")


def _planes(cache):
    planes = []
    for entry in cache:
        offset = getattr(entry, "offset", None)
        if isinstance(offset, int):
            planes.append(np.array([offset]))
        for _key, value in tree_flatten(getattr(entry, "state", None) or []):
            if value is None or not isinstance(value, mx.array):
                continue
            if isinstance(offset, int) and value.ndim == 4:
                value = value[..., :offset, :]
            planes.append(np.asarray(value))
    return planes


def _decision(decision):
    laws, rows = decision.target_laws, decision.response_logprobs
    return {
        "accepted": decision.accepted, "accepted_rows": decision.accepted_rows,
        "emitted": list(decision.emitted), "commit_rows": decision.commit_rows,
        "laws": None if laws is None else [np.asarray(law).tolist() for law in laws],
        "rows": None if rows is None else [
            None if row is None else np.asarray(row).view(np.uint32).tolist() for row in rows],
    }


def _run(batch, prompt, steps, sampling):
    rounds = []
    commit = batch._commit

    def recorded(cohort, decisions, features, *, blocks, **kwargs):
        result = commit(cohort, decisions, features, blocks=blocks, **kwargs)
        for lane, block, decision in zip(cohort, blocks, decisions, strict=True):
            rounds.append({
                "tokens": None if block is None else [int(t) for t in block.tokens],
                "parents": list(getattr(block, "parents", None) or []),
                "decision": _decision(decision),
                "draws": lane.rng.draws,
                "cache": _planes(lane.cache),
                "draft": _planes(lane.draft_cache),
            })
        return result

    batch._commit = recorded
    uid = batch.insert([list(prompt)], max_tokens=[steps],
                       sampling_configs=[sampling or {"sampling_temp": 0}])[0]
    out = []
    for _ in range(400):
        _, responses = batch.next()
        out.extend(r.token for r in responses if r.uid == uid)
        if not batch.lanes:
            break
    else:
        raise AssertionError("scheduler stalled")
    return out, rounds, dict(batch.scheduler_stats)


def _glimmer(prompt, *, steps=16, sampling=None, draft_vocab=False, **options):
    model, draft = tiny(vocab=64, top_k=16, block_size=8)
    if draft_vocab:
        from mlx2.runtime.drafters.dflash2 import configure_draft_vocab

        configure_draft_vocab(draft, 40, special_from=60)
    with closing(generator(model, draft, **options)) as batch:
        result = _run(batch, prompt, steps, sampling)
        vocab = draft.__dict__.get("_draft_vocab")
        result[2]["_draft_vocab"] = None if vocab is None else (
            vocab["reduced_rounds"], vocab["full_rounds"])
        result[2]["_draft_stats"] = dict(getattr(draft, "stats", None) or {})
        return result


def _nemotron(prompt=(1, 2, 3, 4, 5, 6), *, steps=24, sampling=None, **options):
    from test_nemotron35_dflash2_cpu import _model, _tiny_draft

    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
    from mlx2.runtime.nemotron_tree_executor import NemotronTreeExecutor

    model = _model()
    draft = _tiny_draft(model)
    with closing(ExternalDraftBatchGenerator(
        model, draft_model=draft, binding="nemotron-dflash2-tree-cpu", num_draft=7,
        completion_batch_size=1, prefill_step_size=8,
        dynamic_singleton_tree=True, dynamic_tree_max_width=1,
        dynamic_tree_overflow="chain", tree_node_budget_by_lanes={1: 7},
        target_tree_executor_provider=NemotronTreeExecutor(), cost_per_commit_handoff=False,
        ast_adaptive_rows=False, **options,
    )) as batch:
        return _run(batch, list(prompt), steps, sampling)


def _assert_rounds_equal(host, device):
    assert len(host) == len(device) > 0
    for left, right in zip(host, device, strict=True):
        for key in ("tokens", "parents", "decision", "draws"):
            assert left[key] == right[key], key
        for name in ("cache", "draft"):
            assert len(left[name]) == len(right[name])
            for a, b in zip(left[name], right[name], strict=True):
                assert a.dtype == b.dtype and a.shape == b.shape
                assert np.array_equal(a.view(np.uint8), b.view(np.uint8))


_TREE_KEYS = ("external_tree_rounds", "external_tree_nodes", "external_tree_accepted_edges",
              "external_tree_verified_nodes", "external_tree_pipelined_drafts",
              "external_tree_pipeline_discards", "external_tree_early_prelaunches")
# (``external_tree_batched_law_rounds`` is the host verifier's: checked per arm.)


def _assert_same_run(host, device):
    assert device[0] == host[0]
    _assert_rounds_equal(host[1], device[1])
    for key in _TREE_KEYS:
        assert device[2].get(key) == host[2].get(key), key


# -- the lazy proposal ----------------------------------------------------------
@pytest.mark.parametrize("pair", ["glimmer", "nemotron"])
def test_draft_tree_lazy_is_propose_tree_on_the_device(pair):
    """Same tokens, parents, node order and draft cache as the host
    ``propose_tree`` on several prompts and tree shapes (node budgets,
    lattice widths, with and without the codebook cache)."""
    if pair == "glimmer":
        model, draft = tiny(vocab=64, top_k=16, block_size=8)

        def hidden_of(prompt):
            return taps(model, [prompt])
    else:
        from test_nemotron35_dflash2_cpu import _model, _tiny_draft

        model = _model()
        draft = _tiny_draft(model)

        def hidden_of(prompt):
            return model.prefill_body(mx.array([prompt]), model.make_cache(),
                                      tuple(draft.config.target_layer_ids))
    shapes = set()
    for prompt in PROMPTS:
        prompt = [token % 15 for token in prompt] if pair == "nemotron" else prompt
        hidden = hidden_of(prompt[:-1])
        anchor = prompt[-1]
        for budget, options in ((15, {}), (9, {"lattice_positions": 5}), (6, {}), (3, {}),
                                (1, {}), (15, {"codebooks": True})):
            options = dict(options)
            if options.pop("codebooks", False):
                pred, succ, _hit = draft.tree_codebooks()
                options["codebooks"] = (pred, succ)
            host_cache, lazy_cache = draft.make_cache(), draft.make_cache()
            host_marks, lazy_marks = [], []
            host = draft.propose_tree([anchor], hidden, draft.batch_caches([host_cache]), budget,
                                      mark=host_marks.append, **options)[0]
            proposal = draft.draft_tree_lazy(anchor, hidden, draft.batch_caches([lazy_cache]),
                                             budget, mark=lazy_marks.append, **options)
            assert isinstance(proposal.tokens, mx.array) and proposal.tokens.dtype == mx.int32
            assert proposal.host_search is True
            assert [int(t) for t in proposal.tokens.tolist()] == list(host[0])
            assert list(proposal.parents) == list(host[1])
            assert lazy_marks == host_marks
            assert 1 <= len(host[0]) <= budget
            shapes.add(tuple(host[1]))
            for a, b in zip(_planes(host_cache), _planes(lazy_cache), strict=True):
                assert np.array_equal(a, b)
    # The trees are data-dependent: more than one branching topology here.
    assert len(shapes) >= 4
    assert any(sum(parent == -1 for parent in shape) > 1 for shape in shapes)
    assert any(any(parent >= 0 for parent in shape) for shape in shapes)


def test_draft_tree_lazy_refuses_wide_batches():
    model, draft = tiny(vocab=64, top_k=16, block_size=8)
    hidden = taps(model, [[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError, match="one lane"):
        draft.draft_tree_lazy(4, hidden, draft.batch_caches([draft.make_cache()] * 2), 7)


# -- reference execution --------------------------------------------------------
@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("policy", ["lazy", "decide"])
def test_reference_rounds_equal_the_host_round(monkeypatch, policy, batched, pipeline):
    _gates(monkeypatch, batched_laws=batched, single_fence=batched, pipeline_draft=pipeline)
    switches = LAZY if policy == "lazy" else DECIDE
    for prompt in PROMPTS[:2]:
        host = _glimmer(prompt)
        device = _glimmer(prompt, device_round=switches)
        _assert_same_run(host, device)
        stats, rounds = device[2], host[2]["external_tree_rounds"]
        assert stats["external_device_tree_rounds"] == rounds > 0
        assert stats["external_device_tree_declines"] == 0 and stats[DECLINES] == {}
        assert stats["external_device_tree_host_reads"] == rounds
        adopted = stats["external_device_tree_adopted_lanes"]
        assert stats["external_device_tree_lazy_lanes"] == \
            stats["external_device_tree_host_search_lanes"] == rounds - adopted
        if pipeline:
            assert adopted == host[2]["external_tree_pipelined_drafts"] > 0
        else:
            assert adopted == 0
        if policy == "decide":
            assert stats["external_device_tree_decide_lanes"] == rounds
            assert stats["external_device_tree_host_decide_lanes"] == 0
            assert stats.get("external_tree_batched_law_rounds", 0) == 0
        else:
            assert stats["external_device_tree_host_decide_lanes"] == rounds


# -- the Nemotron target-tree provider (the bounded-tree route shape) -----------------
@pytest.mark.parametrize("pipeline", [True, False])
@pytest.mark.parametrize("policy", ["lazy", "decide"])
def test_provider_rounds_equal_the_host_round(monkeypatch, policy, pipeline):
    """``dynamic_singleton_tree`` turns every tree gate on, draft pipelining
    included; ``MLX2_TREE_PIPELINE_DRAFT=0`` is the serial control arm."""
    if not pipeline:
        _gates(monkeypatch, pipeline_draft=False)
    switches = LAZY if policy == "lazy" else DECIDE
    host = _nemotron()
    device = _nemotron(device_round=switches)
    _assert_same_run(host, device)
    stats, rounds = device[2], host[2]["external_tree_rounds"]
    assert stats["external_device_tree_rounds"] == rounds > 0
    assert stats[DECLINES] == {}
    assert host[2]["external_tree_accepted_edges"] > 0
    commits = [item["decision"]["commit_rows"] or [0] for item in host[1] if item["parents"]]
    # Accepts on a later root branch (not the first root's row 1) and root rejects.
    assert any(len(rows) > 1 and rows[1] != 1 for rows in commits)
    assert any(len(rows) == 1 for rows in commits)
    adopted = stats["external_device_tree_adopted_lanes"]
    assert stats["external_device_tree_host_search_lanes"] == rounds - adopted
    assert (adopted > 0) == pipeline
    if pipeline:
        assert adopted == host[2]["external_tree_pipelined_drafts"]
    if policy == "decide":
        assert stats["external_device_tree_decide_lanes"] == rounds
        assert stats["external_tree_batched_law_rounds"] == 0
    else:
        assert stats["external_tree_batched_law_rounds"] == host[2][
            "external_tree_batched_law_rounds"] == rounds


def test_provider_rounds_serve_ordinary_greedy_tokens():
    from test_nemotron35_dflash2_cpu import _model, _ordinary

    expected = _ordinary(_model(), [1, 2, 3, 4, 5, 6], 24)
    out, _rounds, stats = _nemotron(device_round=DECIDE)
    assert out == expected
    assert stats["external_device_tree_rounds"] == stats["external_tree_rounds"] > 0
    assert stats[DECLINES] == {}


# -- sampled lanes ----------------------------------------------------------------
def test_sampled_lanes_get_the_same_tree_and_the_host_decision(monkeypatch):
    _gates(monkeypatch, pipeline_draft=True)
    sampling = {"sampling_temp": 0.8, "seed": 3}
    host = _glimmer(PROMPTS[0], sampling=sampling)
    lazy = _glimmer(PROMPTS[0], sampling=sampling, device_round=LAZY)
    _assert_same_run(host, lazy)
    assert lazy[2]["external_device_tree_rounds"] == host[2]["external_tree_rounds"] > 0
    decide = _glimmer(PROMPTS[0], sampling=sampling, device_round=DECIDE)
    _assert_same_run(host, decide)
    stats = decide[2]
    assert stats[DECIDE_SKIPS] == {"sampled_lane": host[2]["external_tree_rounds"]}
    assert stats["external_device_tree_decide_lanes"] == 0


def test_a_greedy_only_declaration_declines_sampled_lanes(monkeypatch):
    sampling = {"sampling_temp": 0.8, "seed": 3}
    host = _glimmer(PROMPTS[0], sampling=sampling)
    monkeypatch.setattr(DFlash2DraftModel, "draft_tree_lazy_greedy_only", True)
    device = _glimmer(PROMPTS[0], sampling=sampling, device_round=DECIDE)
    _assert_same_run(host, device)
    stats = device[2]
    assert stats["external_device_tree_rounds"] == 0
    assert stats[DECLINES] == {"sampled_lane": host[2]["external_tree_rounds"]}
    greedy = _glimmer(PROMPTS[0], device_round=DECIDE)[2]
    assert greedy["external_device_tree_rounds"] > 0 and greedy[DECLINES] == {}


# -- declines ---------------------------------------------------------------------
def test_pipelining_declines_without_the_declaration(monkeypatch):
    _gates(monkeypatch, pipeline_draft=True)
    host = _glimmer(PROMPTS[0])
    monkeypatch.setattr(DFlash2DraftModel, "draft_tree_lazy_pipelines", False)
    device = _glimmer(PROMPTS[0], device_round=LAZY)
    _assert_same_run(host, device)
    assert device[2]["external_device_tree_rounds"] == 0
    assert device[2][DECLINES] == {"pipeline_draft": host[2]["external_tree_rounds"]}


def test_switch_off_publishes_nothing_and_counters_install_at_zero():
    off = _glimmer(PROMPTS[0])[2]
    assert not any(key in off for key in COUNTERS) and DECLINES not in off
    model, draft = tiny(vocab=64, top_k=16, block_size=8)
    with closing(generator(model, draft, device_round=LAZY)) as batch:
        assert all(batch.scheduler_stats[key] == 0 for key in COUNTERS)


def test_a_subclass_overriding_a_host_twin_declines():
    from types import SimpleNamespace

    from mlx2.runtime.speculative_device_tree import _twin_overridden, engine_reason

    class OwnTree(DFlash2DraftModel):
        def propose_tree(self, *args, **kwargs):
            return super().propose_tree(*args, **kwargs)

    class OwnFinish(DFlash2DraftModel):
        def finish_tree(self, *args, **kwargs):
            return super().finish_tree(*args, **kwargs)

    assert not _twin_overridden(DFlash2DraftModel.__new__(DFlash2DraftModel))
    assert _twin_overridden(OwnTree.__new__(OwnTree))
    assert _twin_overridden(OwnFinish.__new__(OwnFinish))
    owner = SimpleNamespace(target_execution="reference", draft=OwnTree.__new__(OwnTree))
    assert engine_reason(owner, [object()]) == "lazy_tree_twin_overridden"


def test_an_unknown_tree_option_declines_and_redoes_on_the_host(monkeypatch):
    host = _glimmer(PROMPTS[0])
    monkeypatch.setattr(DFlash2DraftModel, "draft_tree_lazy_options", frozenset({"mark"}))
    _gates(monkeypatch, codebook_cache=True)
    with_codebooks = _glimmer(PROMPTS[0])
    device = _glimmer(PROMPTS[0], device_round=LAZY)
    assert host[0] == with_codebooks[0] == device[0]
    _assert_rounds_equal(with_codebooks[1], device[1])
    assert device[2]["external_device_tree_rounds"] == 0
    assert device[2][DECLINES] == {"tree_option_codebooks": host[2]["external_tree_rounds"]}


@pytest.mark.parametrize("pipeline", [False, True])
def test_a_decline_after_the_lazy_draft_restores_the_round(monkeypatch, pipeline):
    """A ``LazyDecline`` raised after ``draft_tree_lazy`` ran (it appended the
    draft context and counted head rounds) restores the round: the host redo
    decodes the same tokens, caches and draft counters as the switch-off run.
    The reduced draft head is on, so its round counters are checked too."""
    from mlx2.runtime.proposal_composition import LazyDecline

    _gates(monkeypatch, pipeline_draft=pipeline)
    prompt = [1, 2, 3, 41, 4]  # anchors inside and outside the reduced head
    host = _glimmer(prompt, steps=20, draft_vocab=True)
    original = DFlash2DraftModel.draft_tree_lazy
    calls = []

    def drafted_then_declined(self, *args, **kwargs):
        proposal = original(self, *args, **kwargs)
        calls.append(1)
        if len(calls) % 2:
            raise LazyDecline("test_decline")
        return proposal

    monkeypatch.setattr(DFlash2DraftModel, "draft_tree_lazy", drafted_then_declined)
    device = _glimmer(prompt, steps=20, draft_vocab=True, device_round=LAZY)
    assert device[0] == host[0]
    _assert_rounds_equal(host[1], device[1])
    assert device[2]["_draft_vocab"] == host[2]["_draft_vocab"]
    assert sum(host[2]["_draft_vocab"]) > 0
    assert device[2]["_draft_stats"] == host[2]["_draft_stats"]
    stats = device[2]
    declined = stats[DECLINES]["test_decline"]
    assert declined > 0 and stats["external_device_tree_rounds"] > 0
    assert declined + stats["external_device_tree_rounds"] == host[2]["external_tree_rounds"]
