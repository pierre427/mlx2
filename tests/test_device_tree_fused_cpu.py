"""``device_round.tree_fused``: the fused device tree round, CPU only.

On the CPU the two fused operations (``verify_row``, ``decide_rows``) run
their plain-MLX reference form; ``tests/test_device_tree_fused_metal.py``
checks the Metal kernels against that form on the GPU.  Pins: the switch is
default off, needs ``tree_decide`` and installs its counter only when on;
the native MTP proposal's branch grid and index give its tokens; the fused
verify row is the unfused ``[anchor] + clip(tokens)``; the fused decision
(row argmax and finiteness, host path walk) equals the host verifier on
random trees with duplicate siblings, stops and non-finite rows, batched and
row-wise, with and without response rows; and on the tiny Qwen3.8-shaped
target, every tree round at every node budget the tile chooser uses
(3/5/7/9/11/15), plain and under presence, frequency and repetition
penalties, equals the host tree round (tokens, decisions, RNG draws, cache
planes), with one host read per round and the drafter's token array never
evaluated.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest
from test_device_round_tree_cpu import (
    _lane,
    _outcome,
    _random_tree,
    _target_logits,
)
from test_nemotron_native_mtp_external_cpu import PROMPT as NEMOTRON_PROMPT
from test_nemotron_native_mtp_external_cpu import engine, seeded
from test_qwen38_27b_device_round_cpu import (
    PROMPT,
    OracleLaneTree,
    _assert_rounds_equal,
    _decision,
    _penalties,
    _planes,
    provider,
    tiny,
)

from mlx2.runtime import speculative_device_tree_fused as fused
from mlx2.runtime.speculative_device_round import DeviceRoundPolicy
from mlx2.runtime.speculative_device_tree import DeviceTree, geometry

DECIDE = {"tree_lazy": True, "tree_decide": True}
FUSED = {"tree_lazy": True, "tree_decide": True, "tree_fused": True}
DECIDE_P = {**DECIDE, "processors": True}
FUSED_P = {**FUSED, "processors": True}
# The tile-budget chooser's node budgets on the 27B route (r2 receipts: 3, 5,
# 7, 9, 11, 15) and their 3x5 patterns.
BUDGETS = (3, 5, 7, 9, 11, 15)


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    monkeypatch.setenv("MLX2_DFLASH_TOPOLOGY", "tree15")
    monkeypatch.delenv("MLX2_TARGET_TREE_EXECUTION", raising=False)
    monkeypatch.delenv("MLX2_QWEN_TARGET_EXECUTION", raising=False)
    monkeypatch.setenv("MLX2_AST_CONTROLLER_MODE", "deterministic")
    monkeypatch.setenv("MLX2_AST_CONTROLLER_COST",
                       '{"a_ms": 5.0, "b_ms": 0.5, "ordinary_ms": 4.0}')
    yield
    mx.set_default_device(previous)


# -- policy -----------------------------------------------------------------
def test_switch_is_default_off_needs_tree_decide_and_counts_only_when_on():
    assert DeviceRoundPolicy.from_value(DECIDE).tree_fused is False
    assert DeviceRoundPolicy.from_value(FUSED).tree_fused is True
    with pytest.raises(ValueError, match="tree_fused requires tree_decide"):
        DeviceRoundPolicy.from_value({"tree_lazy": True, "tree_fused": True})
    with pytest.raises(ValueError):
        DeviceRoundPolicy.from_value({**FUSED, "tree_fused": 1})
    off = engine(seeded(), device_round=DECIDE)
    on = engine(seeded(), device_round=FUSED)
    try:
        assert fused.COUNTER not in off.scheduler_stats
        assert on.scheduler_stats[fused.COUNTER] == 0
        assert on.device_round.settings()["policy"]["tree_fused"] is True
    finally:
        off.close()
        on.close()


# -- the two fused operations ------------------------------------------------
def test_reference_decide_rows_is_the_host_law_verdict():
    """Row argmax and finiteness on rows with ties, -inf, +inf, NaN, all -inf."""
    rng = np.random.default_rng(3)
    vocab = 9
    rows = np.asarray(rng.normal(size=(8, vocab)), dtype=np.float32)
    rows[1, [2, 6]] = 5.0  # a tie: the first index
    rows[2, :] = -np.inf  # all masked: not a distribution
    rows[3, [0, 4]] = -np.inf
    rows[4, 5] = np.inf
    rows[5, 3] = np.nan
    rows[6, [1, 7]] = np.nan
    for dtype in (mx.float32, mx.bfloat16):
        values = mx.array(rows).astype(dtype)
        packet = np.asarray(fused.decide_rows(mx, values)).tolist()
        f32 = values.astype(mx.float32)
        want_selected = np.asarray(mx.argmax(f32, axis=-1)).tolist()
        want_finite = np.isfinite(np.asarray(mx.logsumexp(f32, axis=-1))).astype(int).tolist()
        assert packet[:8] == want_selected
        assert packet[8:] == want_finite == [1, 1, 0, 1, 0, 0, 0, 1]


def test_verify_row_is_the_unfused_input_and_reads_the_grid():
    grid = mx.array([[4, 9, 2, 7], [3, 3, 11, 0], [8, 1, 5, 6]], dtype=mx.uint32)
    pattern = (4, 3, 2)
    index = [branch * 4 + step for branch, depth in enumerate(pattern) for step in range(depth)]
    row, raw = fused.verify_row(mx, grid, index, 6, 9)
    tokens = mx.concatenate([grid[b, :d] for b, d in enumerate(pattern)]).astype(mx.int32)
    want = mx.concatenate([mx.array([6], dtype=mx.int32), mx.clip(tokens, 0, 9)])
    assert row.dtype == raw.dtype == mx.int32
    assert np.asarray(row).tolist() == np.asarray(want).tolist()
    assert np.asarray(raw).tolist() == np.asarray(tokens).tolist()  # 11 kept raw
    with pytest.raises(ValueError):
        fused.verify_row(mx, grid, [], 6, 9)


def test_walk_takes_the_first_matching_child():
    parents = (-1, -1, 0, 0, 2)  # roots 0 and 1 (same token), node 2/3 under 0
    target = geometry(parents).target_parents
    tokens = [5, 5, 7, 7, 1]
    assert fused.walk(target, tokens, [5, 7, 1, 1, 9, 3]) == [0, 1, 3, 5]
    assert fused.walk(target, tokens, [5, 7, 1, 9, 9, 3]) == [0, 1, 3]
    assert fused.walk(target, tokens, [4, 0, 0, 0, 0, 0]) == [0]


def test_draft_tree_lazy_grid_and_index_give_the_tokens_at_every_budget():
    from mlx2.runtime.drafters.native_mtp import NativeMTPDraftModel

    model = seeded()
    draft = NativeMTPDraftModel(model, num_draft=5, revision="a" * 64)
    last = len(model.layers) - 1
    hidden = model.prefill_body(mx.array([NEMOTRON_PROMPT[:6]]), model.make_cache(), (last,))
    pairing = [*NEMOTRON_PROMPT[1:6], 19]
    for budget in BUDGETS:
        proposal = draft.draft_tree_lazy(19, hidden, draft.batch_caches([draft.make_cache()]),
                                         budget, topology="3x5", context_tokens=[pairing])
        assert proposal.grid is not None and len(proposal.index) == len(proposal.parents)
        flat = np.asarray(proposal.grid).reshape(-1)
        assert [int(flat[i]) for i in proposal.index] == np.asarray(proposal.tokens).tolist()
        row, raw = fused.verify_row(mx, proposal.grid, proposal.index, 19, 40)
        assert np.asarray(raw).tolist() == np.asarray(proposal.tokens).tolist()
        assert np.asarray(row).tolist()[0] == 19


# -- the fused decision against the host verifier -----------------------------
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("emit", [True, False])
def test_fused_decision_equals_the_host_walk_on_random_trees(batched, emit):
    """As ``test_device_path_equals_the_host_walk_on_random_trees`` (duplicate
    siblings over a 5-token vocabulary, stops, non-finite rows on and off the
    path), for the fused plan and its host path walk."""
    from mlx2.runtime.external_speculative import TreeDraftRow

    batch = engine(seeded(), device_round=FUSED)
    rng = np.random.default_rng(11 + batched + 2 * emit)
    gates = dict(batch.tree_gates)
    try:
        batch.tree_gates.update(batched_laws=batched, single_fence=batched)
        lengths, kinds = set(), set()
        for trial in range(200):
            vocab = 5
            tokens, parents = _random_tree(rng, int(rng.integers(1, 16)), vocab)
            bad = int(rng.integers(0, len(tokens) + 1)) if trial % 7 == 3 else None
            logits = _target_logits(rng, tokens, parents, vocab, follow=0.8, bad_row=bad)
            if trial % 5 == 1:
                logits = logits.astype(mx.bfloat16)
            batch.stops = set() if trial % 3 else {int(rng.integers(0, vocab))}
            block = TreeDraftRow(list(tokens), list(parents), 1)
            host_lane, device_lane = _lane(1, 0, emit, trial), _lane(1, 0, emit, trial)
            host = _outcome(lambda lane=host_lane, block=block, logits=logits: batch._tree_verify(
                [lane], [block], logits, mx.zeros((1,)), None)[0], host_lane)
            lazy = TreeDraftRow(mx.array(tokens, dtype=mx.int32), list(parents), 1)
            runner = DeviceTree(batch.device_round)

            def device(lane=device_lane, lazy=lazy, logits=logits, block=block, runner=runner,
                       nodes=len(tokens)):  # noqa: B008 - bound per trial
                plan = runner._plan(batch, lane, lazy, logits)
                assert plan.fused and int(plan.packet.shape[0]) == 2 * (nodes + 1)
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


# -- end to end on the tiny Qwen3.8 target ------------------------------------
def _decode(monkeypatch, *, budget=15, steps=24, processors=None, sampling=None, **options):
    """``test_qwen38_27b_device_round_cpu.decode`` at one node budget."""
    from mlx2.runtime.drafters.native_mtp import NativeMTPDraftModel
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    model = tiny()
    oracle = OracleLaneTree()
    draft = NativeMTPDraftModel(model, num_draft=5, revision="a" * 64)
    batch = ExternalDraftBatchGenerator(
        model, draft_model=draft, binding="qwen38-27b-device-tree-fused-cpu",
        num_draft=int(draft.config.block_size) - 1, prefill_step_size=4,
        completion_batch_size=1, cost_per_commit_handoff=False, ready_drain="all",
        dynamic_singleton_tree=True, dynamic_tree_max_width=1, dynamic_tree_overflow="chain",
        tree_node_budget_by_lanes={1: budget},
        target_tree_executor_provider=provider(monkeypatch, oracle), **options,
    )
    rounds = []
    commit = batch._commit

    def recorded(cohort, decisions, features, *, blocks, **kwargs):
        result = commit(cohort, decisions, features, blocks=blocks, **kwargs)
        for lane, block, decision in zip(cohort, blocks, decisions, strict=True):
            rounds.append({
                "tokens": None if block is None else [int(t) for t in block.tokens],
                "parents": list(getattr(block, "parents", None) or []),
                "decision": _decision(decision), "draws": lane.rng.draws,
                "cache": _planes(lane.cache), "draft": _planes(lane.draft_cache),
            })
        return result

    batch._commit = recorded
    try:
        (uid,) = batch.insert(
            [list(PROMPT)], max_tokens=[steps],
            sampling_configs=[sampling or {"sampling_temp": 0}],
            logits_processors=None if processors is None else [processors()])
        out = []
        for _ in range(600):
            _, responses = batch.next()
            out.extend(response.token for response in responses if response.uid == uid)
            if not batch.lanes:
                break
        else:
            raise AssertionError("scheduler stalled")
        return out, rounds, dict(batch.scheduler_stats)
    finally:
        batch.close()


def _assert_fused_engaged(stats, rounds):
    assert rounds > 0
    assert stats["external_device_tree_rounds"] == rounds
    assert stats["external_device_tree_decide_lanes"] == rounds
    assert stats[fused.COUNTER] == rounds
    assert stats["external_device_tree_host_reads"] == rounds
    assert stats["external_device_tree_declines"] == 0


@pytest.mark.parametrize("budget", BUDGETS)
def test_fused_rounds_equal_the_host_round_at_every_node_budget(monkeypatch, budget):
    host = _decode(monkeypatch, budget=budget)
    device = _decode(monkeypatch, budget=budget, device_round=FUSED)
    assert device[0] == host[0]
    _assert_rounds_equal(host[1], device[1])
    _assert_fused_engaged(device[2], host[2]["external_tree_rounds"])
    nodes = {len(item["tokens"]) for item in host[1] if item["parents"]}
    assert max(nodes) == budget  # the budget was the round's tree size
    # The unfused decide arm agrees too (same packet semantics).
    decide = _decode(monkeypatch, budget=budget, device_round=DECIDE)
    assert decide[0] == host[0]


@pytest.mark.parametrize("kind", ["presence", "frequency", "repetition", "presence+repetition"])
@pytest.mark.parametrize("budget", [5, 15])
def test_fused_rounds_with_penalties_equal_the_host_round(monkeypatch, kind, budget):
    processors = _penalties(kind, 20)
    host = _decode(monkeypatch, budget=budget, processors=processors)
    device = _decode(monkeypatch, budget=budget, processors=processors, device_round=FUSED_P)
    assert device[0] == host[0]
    _assert_rounds_equal(host[1], device[1])
    _assert_fused_engaged(device[2], host[2]["external_tree_rounds"])
    assert host[0] != _decode(monkeypatch, budget=budget)[0]  # the penalty moved the decode


@pytest.mark.parametrize("emit_logprobs", [True, False])
@pytest.mark.parametrize("batched", ["1", "0"])
def test_fused_rounds_equal_the_host_round_with_and_without_response_rows(
        monkeypatch, emit_logprobs, batched):
    monkeypatch.setenv("MLX2_TREE_BATCHED_TARGET_LAWS", batched)
    monkeypatch.setenv("MLX2_TREE_SINGLE_FENCE", batched)
    sampling = {"sampling_temp": 0, "emit_logprobs": emit_logprobs}
    host = _decode(monkeypatch, sampling=sampling)
    device = _decode(monkeypatch, sampling=sampling, device_round=FUSED)
    assert device[0] == host[0]
    _assert_rounds_equal(host[1], device[1])
    _assert_fused_engaged(device[2], host[2]["external_tree_rounds"])


def test_sampled_lanes_keep_the_host_decision_under_the_fused_switch(monkeypatch):
    sampling = {"sampling_temp": 0.8, "seed": 5}
    host = _decode(monkeypatch, sampling=sampling)
    device = _decode(monkeypatch, sampling=sampling, device_round=FUSED)
    assert device[0] == host[0]
    _assert_rounds_equal(host[1], device[1])
    stats = device[2]
    rounds = host[2]["external_tree_rounds"]
    assert stats["external_device_tree_rounds"] == rounds
    assert stats["external_device_tree_decide_skips"] == {"sampled_lane": rounds}
    assert stats[fused.COUNTER] == 0


def test_the_drafter_token_array_is_never_evaluated(monkeypatch):
    """The fused round reads the verify row's raw drafts, not the drafter's
    concatenated token array (its slices, concatenate and cast never run)."""
    from mlx2.runtime.drafters import native_mtp

    seen = []
    original = native_mtp.NativeMTPDraftModel.draft_tree_lazy

    def spy(self, *args, **kwargs):
        proposal = original(self, *args, **kwargs)
        seen.append(proposal.tokens)
        return proposal

    monkeypatch.setattr(native_mtp.NativeMTPDraftModel, "draft_tree_lazy", spy)
    # The spy is an instance-free class attribute: the twin check sees
    # ``draft_tree_lazy`` on the class, so the device round still engages.
    out, _rounds, stats = _decode(monkeypatch, steps=12, device_round=FUSED)
    assert out and seen and stats[fused.COUNTER] == len(seen)
    import io

    for tokens in seen:
        # Still a lazy graph node: export shows its primitives, an evaluated
        # array exports as a bare leaf.
        buffer = io.StringIO()
        mx.export_to_dot(buffer, tokens)
        assert "Concatenate" in buffer.getvalue() or "AsType" in buffer.getvalue()


def test_the_native_mtp_tree_route_accepts_the_fused_switch(monkeypatch, tmp_path):
    from test_qwen38_27b_device_round_cpu import _load_until_artifact

    target = tmp_path / "target"
    target.mkdir()
    adapter = _load_until_artifact(monkeypatch, target, {
        "draft_model": str(target), "device_round": FUSED_P})
    assert adapter.external_policy["device_round"] == FUSED_P
    with pytest.raises(ValueError, match="tree_fused requires tree_decide"):
        _load_until_artifact(monkeypatch, target, {
            "draft_model": str(target), "device_round": {"tree_lazy": True, "tree_fused": True}})

