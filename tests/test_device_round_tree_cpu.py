"""Device-round tree rounds (``device_round.tree_lazy``, slice 1b-a), CPU only.

Covers slice 1b-b (``tree_decide``) too: the device path equals the host walk.

Tiny random Nemotron-H targets with a native MTP head (no real weights) on the
``tree15`` route, with the generic reference tree forward and with the
Nemotron target-tree provider.  Pins: the switch is default off and
validated; ``NativeMTPDraftModel.draft_tree_lazy`` is ``propose_tree``'s tree
with its tokens left on the device; with the switch on, every round's
proposal, decision (commit rows, emitted tokens, accepted rows, laws,
response rows), lane RNG, target and draft cache planes, output tokens and
tree counters equal the switch-off round's, with accepts on every branch and
rejects at the root; one host read per device tree round; unsupported rounds
decline with a counted reason and still decode the host round's output.
"""

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten
from test_nemotron_native_mtp_external_cpu import PROMPT, aligned, engine, seeded

from mlx2.runtime.drafters.native_mtp import NativeMTPDraftModel
from mlx2.runtime.speculative_device_round import DeviceRoundPolicy
from mlx2.runtime.speculative_device_tree import (
    COUNTERS,
    DECLINES,
    chain_branch_parents,
    geometry,
)

LAZY = {"tree_lazy": True}
PROMPTS = (PROMPT, [7, 1, 30, 4, 4, 22, 13])
# Mixing 0.3 accepts on the first, second and third branch and rejects at
# the root in these 30-token decodes (checked below, not assumed).
CASES = [(0.3, 28, 0), (0.3, 28, 1), (0.3, 5, 0), (0.1, 5, 1), (1.0, 5, 0)]


@pytest.fixture(autouse=True)
def tree15(monkeypatch):
    monkeypatch.setenv("MLX2_DFLASH_TOPOLOGY", "tree15")
    monkeypatch.delenv("MLX2_TARGET_TREE_EXECUTION", raising=False)
    monkeypatch.delenv("MLX2_QWEN_TARGET_EXECUTION", raising=False)
    monkeypatch.setenv("MLX2_AST_CONTROLLER_MODE", "deterministic")
    monkeypatch.setenv("MLX2_AST_CONTROLLER_COST",
                       '{"a_ms": 5.0, "b_ms": 0.5, "ordinary_ms": 4.0}')


def _planes(cache):
    """Every committed cache plane (KV rows past ``offset`` are scratch)."""
    planes = []
    for entry in cache:
        offset = getattr(entry, "offset", None)
        if isinstance(offset, int):
            planes.append(np.array([offset]))
        for _key, value in tree_flatten(entry.state):
            if value is None:
                continue
            if isinstance(offset, int) and getattr(entry, "keys", None) is not None \
                    and value.ndim == 4:
                value = value[..., :offset, :]
            planes.append(np.asarray(value))
    return planes


def _decision(decision):
    laws = decision.target_laws
    rows = decision.response_logprobs
    return {
        "accepted": decision.accepted, "accepted_rows": decision.accepted_rows,
        "emitted": list(decision.emitted), "commit_rows": decision.commit_rows,
        "laws": None if laws is None else [np.asarray(law).tolist() for law in laws],
        "rows": None if rows is None else [
            None if row is None else np.asarray(row).view(np.uint32).tolist() for row in rows],
        "window": type(decision.verify_window).__name__,
    }


def _decode(build, prompt, steps=30, *, sampling=None, provider=None, processors=None,
            **options):
    from mlx2.runtime.nemotron_tree_executor import NemotronTreeExecutor

    extra = {}
    if provider:
        extra = {"dynamic_singleton_tree": True, "dynamic_tree_max_width": 1,
                 "dynamic_tree_overflow": "chain", "tree_node_budget_by_lanes": {1: 15},
                 "target_tree_executor_provider": NemotronTreeExecutor(),
                 "cost_per_commit_handoff": False}
    batch = engine(seeded(build), num_draft=4, **extra, **options)
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
    try:
        uid = batch.insert([list(prompt)], max_tokens=[steps],
                           sampling_configs=[sampling or {"sampling_temp": 0}],
                           logits_processors=None if processors is None else [processors()])[0]
        out = []
        for _ in range(400):
            _, responses = batch.next()
            out.extend(response.token for response in responses if response.uid == uid)
            if not batch.lanes:
                break
        else:
            raise AssertionError("scheduler stalled")
        return out, rounds, dict(batch.scheduler_stats), dict(batch.draft.stats)
    finally:
        batch.close()


def _assert_rounds_equal(host, device):
    assert len(host) == len(device)
    for left, right in zip(host, device, strict=True):
        assert left["tokens"] == right["tokens"]
        assert left["parents"] == right["parents"]
        assert left["decision"] == right["decision"]
        assert left["draws"] == right["draws"]
        for name in ("cache", "draft"):
            assert len(left[name]) == len(right[name])
            for a, b in zip(left[name], right[name], strict=True):
                assert a.dtype == b.dtype and a.shape == b.shape
                assert np.array_equal(a.view(np.uint8), b.view(np.uint8))


_TREE_KEYS = ("external_tree_rounds", "external_tree_nodes", "external_tree_accepted_edges",
              "external_tree_verified_nodes", "external_tree_node_budget_histogram")


def _branch_of(row, parents):
    """The root branch index of a target row (0 for the anchor)."""
    roots = [node + 1 for node, parent in enumerate(parents) if parent < 0]
    if row == 0:
        return None
    node = row - 1
    while parents[node] >= 0:
        node = parents[node]
    return roots.index(node + 1)


# -- policy -----------------------------------------------------------------
def test_switch_is_default_off_validated_and_counted_only_when_on():
    assert DeviceRoundPolicy.from_value({"tree_lazy": False}) is None
    assert DeviceRoundPolicy.from_value(LAZY).tree_lazy
    with pytest.raises(ValueError):
        DeviceRoundPolicy.from_value({"tree_lazy": 1})
    off = engine(seeded())
    try:
        assert not any(key in off.scheduler_stats for key in COUNTERS)
        assert DECLINES not in off.scheduler_stats
    finally:
        off.close()
    on = engine(seeded(), device_round=LAZY)
    try:
        assert all(on.scheduler_stats[key] == 0 for key in COUNTERS)
        assert on.device_round.tree and on.device_round.settings()["policy"]["tree_lazy"]
    finally:
        on.close()


def test_geometry_is_the_target_tree_of_the_parents():
    parents = chain_branch_parents((3, 2, 1))
    assert parents == (-1, 0, 1, -1, 3, -1)
    geo = geometry(parents)
    assert geo.target_parents == (-1, 0, 1, 2, 0, 4, 0)
    assert geo.paths == ((0,), (0, 1), (0, 1, 2), (0, 1, 2, 3), (0, 4), (0, 4, 5), (0, 6))
    assert geo.max_depth == 3 and geo.rows == 7
    assert geo.parent_rows.tolist() == [0, 1, 2, 0, 4, 0]
    # Roots 0, 3 and 5 are siblings in that order.
    assert geo.earlier_siblings[3].tolist() == [True, False, False, False, False, False]
    assert geo.earlier_siblings[5].tolist() == [True, False, False, True, False, False]
    assert geo.ancestors[2].tolist() == [True, True, True, False, False, False]
    with pytest.raises(ValueError):
        geometry((0,))


# -- the lazy proposal --------------------------------------------------------
def test_draft_tree_lazy_is_propose_tree_on_the_device():
    model = seeded()
    draft = NativeMTPDraftModel(model, num_draft=4, revision="a" * 64)
    last = len(model.layers) - 1
    hidden = model.prefill_body(mx.array([PROMPT[:6]]), model.make_cache(), (last,))
    pairing = [*PROMPT[1:6], 19]  # seed t pairs with token t + 1, ending with the anchor
    for budget, topology in ((15, None), (9, "3x5"), (6, "4/4/4/3"), (4, "chain")):
        host_cache, lazy_cache = draft.make_cache(), draft.make_cache()
        host = draft.propose_tree([19], hidden, draft.batch_caches([host_cache]), budget,
                                  topology=topology, context_tokens=[pairing])[0]
        proposal = draft.draft_tree_lazy(19, hidden, draft.batch_caches([lazy_cache]), budget,
                                         topology=topology, context_tokens=[pairing])
        assert isinstance(proposal.tokens, mx.array) and proposal.tokens.dtype == mx.int32
        assert [int(t) for t in proposal.tokens.tolist()] == list(host[0])
        assert list(proposal.parents) == list(host[1])
        assert len(host[0]) <= budget
        for a, b in zip(_planes(host_cache), _planes(lazy_cache), strict=True):
            assert np.array_equal(a, b)


# -- exactness against the host tree round -------------------------------------
@pytest.mark.parametrize(("mixing", "seed", "prompt"), CASES)
def test_reference_tree_rounds_equal_the_host_round(mixing, seed, prompt):
    def build():
        return aligned(seed, mixing)

    host = _decode(build, PROMPTS[prompt])
    device = _decode(build, PROMPTS[prompt], device_round=LAZY)
    assert device[0] == host[0]
    _assert_rounds_equal(host[1], device[1])
    for key in _TREE_KEYS:
        assert device[2].get(key) == host[2].get(key), key
    assert device[3] == host[3]  # drafter counters (draft and chain steps)
    stats = device[2]
    assert stats["external_device_tree_rounds"] == host[2]["external_tree_rounds"] > 0
    assert stats["external_device_tree_lazy_lanes"] == stats["external_device_tree_rounds"]
    assert stats["external_device_tree_host_reads"] == stats["external_device_tree_rounds"]
    assert stats["external_device_tree_declines"] == 0


def test_cases_cover_every_branch_and_root_rejects():
    branches, root_rejects = set(), 0
    for mixing, seed, prompt in CASES[:3]:
        _out, rounds, _stats, _draft = _decode(lambda m=mixing, s=seed: aligned(s, m),
                                               PROMPTS[prompt], device_round=LAZY)
        for item in rounds:
            if not item["parents"]:
                continue
            rows = item["decision"]["commit_rows"] or [0]
            if len(rows) == 1:
                root_rejects += 1
            else:
                branches.add(_branch_of(rows[1], item["parents"]))
    assert branches == {0, 1, 2}
    assert root_rejects > 0


def _nemotron_tree_model(seed=28, mixing=0.3):
    model = aligned(seed, mixing)
    model.set_short_segment_policy("layerwise_short_exact_routed_batch")
    for layer in model.layers:
        if layer.block_type == "M":
            layer.mixer.set_conv_policy("fused_short")
            layer.mixer.set_ssm_policy("fused_short_gate")
            layer.mixer.set_projection_policy("short_row_exact")
    return model


@pytest.mark.parametrize("emit_logprobs", [True, False])
def test_provider_tree_rounds_equal_the_host_round(emit_logprobs):
    sampling = {"sampling_temp": 0, "emit_logprobs": emit_logprobs}
    host = _decode(_nemotron_tree_model, PROMPT, sampling=sampling, provider=True)
    device = _decode(_nemotron_tree_model, PROMPT, sampling=sampling, provider=True,
                     device_round=LAZY)
    assert device[0] == host[0]
    _assert_rounds_equal(host[1], device[1])
    for key in _TREE_KEYS:
        assert device[2].get(key) == host[2].get(key), key
    # The provider route turns the batched-law gate on: the host decision
    # served here is ``_verify_tree_batched``'s, at one fence.
    assert device[2]["external_tree_batched_law_rounds"] == host[2][
        "external_tree_batched_law_rounds"] > 0
    assert device[2]["external_device_tree_rounds"] == host[2]["external_tree_rounds"]


def test_sampled_lanes_keep_the_host_law_and_tokens():
    sampling = {"sampling_temp": 0.8, "seed": 3}
    host = _decode(lambda: aligned(28, 0.3), PROMPT, sampling=sampling)
    device = _decode(lambda: aligned(28, 0.3), PROMPT, sampling=sampling, device_round=LAZY)
    assert device[0] == host[0]
    _assert_rounds_equal(host[1], device[1])


# -- declines ---------------------------------------------------------------
def _presence():
    from mlx2.runtime.sample_utils import make_logits_processors

    return make_logits_processors(presence_penalty=1.5, presence_context_size=20,
                                  penalty_generation_start=len(PROMPT))


def test_processor_lanes_decline_and_decode_the_host_output():
    host = _decode(lambda: aligned(28, 0.3), PROMPT, processors=_presence)
    device = _decode(lambda: aligned(28, 0.3), PROMPT, processors=_presence, device_round=LAZY)
    assert host[2]["external_tree_rounds"] > 0
    assert device[0] == host[0]
    stats = device[2]
    assert stats["external_device_tree_rounds"] == 0
    assert stats[DECLINES] == {"processors": host[2]["external_tree_rounds"]}


def test_drafter_and_provider_without_device_hooks_decline(monkeypatch):
    from mlx2.runtime.nemotron_tree_executor import NemotronTreeExecutor

    host = _decode(lambda: aligned(28, 0.3), PROMPT)
    monkeypatch.setattr(NativeMTPDraftModel, "draft_tree_lazy", None)
    device = _decode(lambda: aligned(28, 0.3), PROMPT, device_round=LAZY)
    assert device[0] == host[0]
    assert device[2][DECLINES] == {"drafter_without_lazy_tree": host[2]["external_tree_rounds"]}
    monkeypatch.undo()
    monkeypatch.setenv("MLX2_DFLASH_TOPOLOGY", "tree15")
    monkeypatch.setenv("MLX2_AST_CONTROLLER_MODE", "deterministic")
    monkeypatch.setenv("MLX2_AST_CONTROLLER_COST",
                       '{"a_ms": 5.0, "b_ms": 0.5, "ordinary_ms": 4.0}')
    monkeypatch.setattr(NemotronTreeExecutor, "device_tree_inputs", False)
    provided = _decode(_nemotron_tree_model, PROMPT, provider=True, device_round=LAZY)
    assert provided[2]["external_device_tree_rounds"] == 0
    assert set(provided[2][DECLINES]) == {"provider_without_device_inputs"}


def test_lazy_decline_restores_the_round_and_redoes_it_on_the_host(monkeypatch):
    from mlx2.runtime.proposal_composition import LazyDecline

    host = _decode(lambda: aligned(28, 0.3), PROMPT)
    original = NativeMTPDraftModel.draft_tree_lazy
    calls = []

    def sometimes(self, *args, **kwargs):
        calls.append(1)
        if len(calls) % 2:
            raise LazyDecline("test_decline")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(NativeMTPDraftModel, "draft_tree_lazy", sometimes)
    device = _decode(lambda: aligned(28, 0.3), PROMPT, device_round=LAZY)
    assert device[0] == host[0]
    _assert_rounds_equal(host[1], device[1])
    assert device[3] == host[3]
    stats = device[2]
    declined = stats[DECLINES]["test_decline"]
    assert declined > 0 and stats["external_device_tree_rounds"] > 0
    assert declined + stats["external_device_tree_rounds"] == host[2]["external_tree_rounds"]


# -- slice 1b-b: the device path selection (``tree_decide``) ----------------------
DECIDE = {"tree_lazy": True, "tree_decide": True}


def _random_tree(rng, nodes, vocab):
    parents = [-1]
    for node in range(1, nodes):
        parents.append(int(rng.integers(-1, node)))
    tokens = [int(rng.integers(0, vocab)) for _ in range(nodes)]
    return tokens, parents


def _target_logits(rng, tokens, parents, vocab, *, follow, bad_row=None):
    """Logits whose row argmaxes follow the tree with probability ``follow``."""
    geo = geometry(tuple(parents))
    rows = np.asarray(rng.normal(size=(geo.rows, vocab)), dtype=np.float32)
    for row in range(geo.rows):
        children = [node for node, parent in enumerate(parents) if geo.target_parents[node + 1]
                    == row]
        if children and rng.random() < follow:
            choice = tokens[children[int(rng.integers(0, len(children)))]]
        else:
            choice = int(rng.integers(0, vocab))
        rows[row, choice] = 8.0 + float(rng.random())
    if bad_row is not None:
        rows[bad_row, int(rng.integers(0, vocab))] = rng.choice([np.nan, np.inf])
    return mx.array(rows[None])


def _lane(uid, anchor, emit, seed):
    from types import SimpleNamespace

    from mlx2.runtime.speculative_sampling import RequestRNG

    return SimpleNamespace(uid=uid, anchor=anchor, history=[1, 2, 3], processors=[],
                           sampling={"sampling_temp": 0, "emit_logprobs": emit},
                           rng=RequestRNG(seed))


def _outcome(call, lane):
    from mlx2.runtime.external_speculative import LaneFailure

    try:
        decision = call()
    except LaneFailure as error:
        return ("failure", str(error), lane.rng.draws)
    return (_decision(decision), lane.rng.draws)


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("emit", [True, False])
def test_device_path_equals_the_host_walk_on_random_trees(batched, emit):
    """Random parent-first trees over a 5-token vocabulary (duplicate
    siblings, so the first-match tie-break matters), argmaxes that follow,
    leave or reject the tree, stops, and non-finite rows on and off the path:
    the device decision equals ``_verify_tree`` / ``_verify_tree_batched``."""
    from mlx2.runtime.external_speculative import TreeDraftRow
    from mlx2.runtime.speculative_device_tree import DeviceTree

    batch = engine(seeded(), device_round=DECIDE)
    rng = np.random.default_rng(7 + batched + 2 * emit)
    gates = dict(batch.tree_gates)
    try:
        batch.tree_gates.update(batched_laws=batched, single_fence=batched)
        runner = DeviceTree(batch.device_round)
        lengths, kinds = set(), set()
        for trial in range(160):
            vocab = 5
            tokens, parents = _random_tree(rng, int(rng.integers(1, 15)), vocab)
            bad = int(rng.integers(0, len(tokens) + 1)) if trial % 7 == 3 else None
            logits = _target_logits(rng, tokens, parents, vocab, follow=0.8, bad_row=bad)
            batch.stops = set() if trial % 3 else {int(rng.integers(0, vocab))}
            block = TreeDraftRow(list(tokens), list(parents), 1)
            host_lane, device_lane = _lane(1, 0, emit, trial), _lane(1, 0, emit, trial)
            host = _outcome(lambda lane=host_lane, block=block, logits=logits: batch._tree_verify(
                [lane], [block], logits, mx.zeros((1,)), None)[0], host_lane)
            lazy = TreeDraftRow(mx.array(tokens, dtype=mx.int32), list(parents), 1)

            def device(lane=device_lane, lazy=lazy, logits=logits, block=block):
                plan = runner._plan(batch, lane, lazy, logits)
                mx.eval(plan.packet)
                return runner._finish(batch, lane, block, plan, logits)

            got = _outcome(device, device_lane)
            assert got == host, (trial, tokens, parents)
            kinds.add(host[0] if host[0] == "failure" else "ok")
            if host[0] != "failure":
                lengths.add(len(host[0]["commit_rows"]))
        assert "failure" in kinds and max(lengths) >= 4 and 1 in lengths
    finally:
        batch.tree_gates.clear()
        batch.tree_gates.update(gates)
        batch.close()


@pytest.mark.parametrize(("mixing", "seed", "prompt"), CASES)
def test_reference_device_decided_rounds_equal_the_host_round(mixing, seed, prompt):
    def build():
        return aligned(seed, mixing)

    host = _decode(build, PROMPTS[prompt])
    device = _decode(build, PROMPTS[prompt], device_round=DECIDE)
    assert device[0] == host[0]
    _assert_rounds_equal(host[1], device[1])
    for key in _TREE_KEYS:
        assert device[2].get(key) == host[2].get(key), key
    stats = device[2]
    assert stats["external_device_tree_decide_lanes"] == host[2]["external_tree_rounds"] > 0
    assert stats["external_device_tree_host_decide_lanes"] == 0
    assert stats["external_device_tree_host_reads"] == stats["external_device_tree_rounds"]


@pytest.mark.parametrize("emit_logprobs", [True, False])
def test_provider_device_decided_rounds_equal_the_host_round(emit_logprobs):
    sampling = {"sampling_temp": 0, "emit_logprobs": emit_logprobs}
    host = _decode(_nemotron_tree_model, PROMPT, sampling=sampling, provider=True)
    device = _decode(_nemotron_tree_model, PROMPT, sampling=sampling, provider=True,
                     device_round=DECIDE)
    assert device[0] == host[0]
    _assert_rounds_equal(host[1], device[1])
    for key in (*_TREE_KEYS, "external_tree_logprob_rows_skipped"):
        assert device[2].get(key) == host[2].get(key), key
    assert device[2]["external_device_tree_decide_lanes"] == host[2]["external_tree_rounds"]
    # The device decided every round: the host verifier never ran.
    assert device[2]["external_tree_batched_law_rounds"] == 0


def test_a_stop_inside_an_accepted_path_truncates_as_on_the_host():
    def build():
        return aligned(28, 0.3)

    _out, rounds, _stats, _draft = _decode(build, PROMPT, device_round=DECIDE)
    long = next(item for item in rounds if len(item["decision"]["commit_rows"] or ()) >= 3)
    stop = long["decision"]["emitted"][1]
    host = _decode(build, PROMPT, stop_tokens=((stop,),))
    device = _decode(build, PROMPT, stop_tokens=((stop,),), device_round=DECIDE)
    assert device[0] == host[0] and device[0][-1] == stop
    _assert_rounds_equal(host[1], device[1])


def test_tree_decide_needs_tree_lazy_and_sampled_lanes_decline():
    with pytest.raises(ValueError, match="tree_decide requires tree_lazy"):
        DeviceRoundPolicy.from_value({"tree_decide": True})
    sampling = {"sampling_temp": 0.8, "seed": 3}
    host = _decode(lambda: aligned(28, 0.3), PROMPT, sampling=sampling)
    device = _decode(lambda: aligned(28, 0.3), PROMPT, sampling=sampling, device_round=DECIDE)
    assert device[0] == host[0]
    _assert_rounds_equal(host[1], device[1])
    stats = device[2]
    assert stats[DECLINES] == {}
    assert stats["external_device_tree_decide_skips"] == {"sampled_lane": host[2]["external_tree_rounds"]}
    assert stats["external_device_tree_host_decide_lanes"] == host[2]["external_tree_rounds"]
    assert stats["external_device_tree_decide_lanes"] == 0


def test_a_subclass_overriding_a_host_twin_declines():
    """``CascadedMTPDraftModel`` overrides ``_lazy_tree_paths`` and inherits
    ``draft_tree_lazy``: such a drafter keeps the host tree round."""
    from types import SimpleNamespace

    from mlx2.runtime.drafters.inkling_mtp import CascadedMTPDraftModel
    from mlx2.runtime.speculative_device_tree import _twin_overridden, engine_reason

    class OwnPaths(NativeMTPDraftModel):
        def _lazy_tree_paths(self, *args, **kwargs):
            return super()._lazy_tree_paths(*args, **kwargs)

    class OwnLazy(OwnPaths):
        def draft_tree_lazy(self, *args, **kwargs):
            return super().draft_tree_lazy(*args, **kwargs)

    assert not _twin_overridden(NativeMTPDraftModel.__new__(NativeMTPDraftModel))
    assert _twin_overridden(OwnPaths.__new__(OwnPaths))
    assert not _twin_overridden(OwnLazy.__new__(OwnLazy))
    assert _twin_overridden(CascadedMTPDraftModel.__new__(CascadedMTPDraftModel))
    owner = SimpleNamespace(target_execution="reference",
                            draft=CascadedMTPDraftModel.__new__(CascadedMTPDraftModel))
    assert engine_reason(owner, [object()]) == "lazy_tree_twin_overridden"


@pytest.mark.parametrize("lazy", [False, True])
def test_reference_transaction_rejects_sibling_replay_before_cache_mutation(lazy):
    from types import SimpleNamespace

    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
    from mlx2.runtime.speculative_device_tree import DeviceTree

    def forward(inputs, cache, _layers):
        cache[0].tokens.extend(inputs[0].tolist())
        features = mx.cumsum(inputs, axis=1)[:, :, None]
        return features, features

    batch = object.__new__(ExternalDraftBatchGenerator)
    batch.mx, batch.layers = mx, ()
    batch.model = SimpleNamespace(forward_with_taps=forward)
    lane = SimpleNamespace(uid=1, cache=[SimpleNamespace(tokens=[7])])
    tokens, parents = [9, 11, 13], [-1, 0, 0]
    if lazy:
        _, _, transaction = DeviceTree._reference_forward(
            batch, lane, mx.array(tokens), geometry((-1, -1)))
        transaction.inputs = tokens  # populated by DeviceTree.decide after the joint read
    else:
        _, _, transaction = batch._reference_tree_forward(lane, tokens, parents)
    with pytest.raises(ValueError, match="root path"):
        transaction.commit_paths([[0, 1, 2]])
    assert lane.cache[0].tokens == [7] and not transaction.closed
    transaction.commit_paths([[0, 2]])
    assert lane.cache[0].tokens == [7, 9, 13]


@pytest.mark.parametrize("override", ["mixin", "instance"])
def test_lazy_admission_honors_the_resolved_host_tree_method(override):
    from types import SimpleNamespace

    from mlx2.runtime.speculative_device_tree import engine_reason

    class Host:
        def propose_tree(self):
            return ([1], [-1])

        def draft_tree_lazy(self):
            return self.proposal

    class DifferentHost:
        def propose_tree(self):
            return ([2], [-1])

    class Draft(DifferentHost, Host):
        pass

    draft = Draft() if override == "mixin" else Host()
    if override == "instance":
        draft.propose_tree = DifferentHost().propose_tree
    draft.proposal = ([1], [-1])
    owner = SimpleNamespace(draft=draft, _draft_pipelines=lambda: False)
    assert draft.propose_tree() != draft.draft_tree_lazy()
    assert engine_reason(owner, [SimpleNamespace(processors=[])]) == "lazy_tree_twin_overridden"
