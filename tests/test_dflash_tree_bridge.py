"""tree15 round-cost bridge gates (TensorFold parity handoff); CPU only.

Covers the executor cache, the block lattice and codebook cache, batched
target laws against the row reference, on-request logprobs, the single
fence, phase timing, default-off behavior and fault recovery.
"""
import subprocess
import sys
from collections import deque

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)
from mlx2.runtime import qwen38_tensorfold
from mlx2.runtime.external_speculative import (
    Lane,
    TreeDraftRow,
    _TREE_GATES,
)
from mlx2.runtime.speculative_sampling import RequestRNG
from mlx2.serving import minimum_tokens_processor

from test_dflash_pair_select import drain, generator, taps, tiny

ALL_GATES = [name for name, _ in _TREE_GATES.values()]


@pytest.fixture(autouse=True)
def _clean_gates(monkeypatch):
    for name in ALL_GATES + [
        "MLX2_DFLASH_TOPOLOGY",
        "MLX2_EXTERNAL_ROUND_TIMING",
        "MLX2_TENSORFOLD_COHORT_LIMIT",
    ]:
        monkeypatch.delenv(name, raising=False)


def _tree_env(monkeypatch, **gates):
    monkeypatch.setenv("MLX2_DFLASH_TOPOLOGY", "tree15")
    for gate, value in gates.items():
        name, enabled = _TREE_GATES[gate]
        if value is False:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, enabled if value is True else value)


def _serving_processors(prompt_len, minimum=None, presence=None, frequency=None):
    """Processors in serving order: penalties first, then the minimum mask."""
    from mlx2.runtime.sample_utils import make_logits_processors

    processors = make_logits_processors(
        presence_penalty=presence, presence_context_size=0,
        frequency_penalty=frequency, frequency_context_size=0,
        penalty_generation_start=prompt_len,
    )
    if minimum is not None:
        processors.append(minimum_tokens_processor(mx, [0, 7], prompt_len, minimum))
    return processors


def _run(m, d, *, temp=0.0, seeds=(123,), max_tokens=10, minimum=None,
         stops=(), sampling=None, prompt=(1, 2, 3), presence=None, frequency=None):
    """One tree15 lane to completion: tokens, logprobs, draws, stats."""
    b = generator(m, d, pairwise_selection="batched", stop_tokens=[[t] for t in stops])
    processors = [_serving_processors(len(prompt), minimum, presence, frequency)]
    config = {"sampling_temp": temp, **(sampling or {})}
    b.insert(
        [list(prompt)],
        max_tokens=[max_tokens],
        logits_processors=processors,
        sampling_configs=[config],
        lane_rngs=[type("Seed", (), {"key": mx.array(list(seeds))})()],
    )
    lane = next(iter(b.lanes.values()))
    tokens, logprobs = [], []
    for _ in range(200):
        _, responses = b.next()
        for response in responses:
            tokens.append(response.token)
            logprobs.append(
                None if response.logprobs is None else np.asarray(response.logprobs)
            )
        if not b.lanes:
            break
    else:
        raise AssertionError("scheduler stalled")
    return tokens, logprobs, lane.rng.draws, dict(b.scheduler_stats)


# -- default-off and fail-closed --------------------------------------------


def test_default_off_publishes_no_bridge_counters(monkeypatch):
    monkeypatch.setenv("MLX2_DFLASH_TOPOLOGY", "tree15")
    m, d = tiny(vocab=64, top_k=16, block_size=8)
    _, _, _, stats = _run(m, d)
    assert stats["external_tree_rounds"] > 0
    for counters in (
        "external_tensorfold_executor_cache_hits",
        "external_tree_codebook_cache_hits",
        "external_tree_batched_law_rounds",
        "external_tree_logprob_rows_skipped",
        "external_tree_single_fence_rounds",
    ):
        assert counters not in stats
    assert not any(key.startswith("external_phase_") for key in stats)


@pytest.mark.parametrize(
    "env, message",
    [
        ({"MLX2_TREE_BATCHED_TARGET_LAWS": "1"}, "require MLX2_DFLASH_TOPOLOGY=tree15"),
        (
            {"MLX2_DFLASH_TOPOLOGY": "tree15", "MLX2_TREE_SINGLE_FENCE": "1"},
            "requires MLX2_TREE_BATCHED_TARGET_LAWS",
        ),
        (
            {"MLX2_DFLASH_TOPOLOGY": "tree15", "MLX2_TENSORFOLD_CACHE_EXECUTOR": "1"},
            "requires MLX2_QWEN_TARGET_EXECUTION=tensorfold",
        ),
        (
            {"MLX2_DFLASH_TOPOLOGY": "tree15", "MLX2_TREE_PIPELINE_DRAFT": "yes"},
            "must be unset, 0 or 1",
        ),
    ],
)
def test_gates_fail_closed(monkeypatch, env, message):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    m, d = tiny()
    with pytest.raises(ValueError, match=message):
        generator(m, d)


@pytest.mark.parametrize("value", ["0", "5", "yes", "01"])
def test_tensorfold_cohort_limit_is_bounded_and_strict(monkeypatch, value):
    monkeypatch.setenv("MLX2_QWEN_TARGET_EXECUTION", "tensorfold")
    monkeypatch.setenv("MLX2_TENSORFOLD_SOURCE", "/unused")
    monkeypatch.setenv("MLX2_TENSORFOLD_COHORT_LIMIT", value)
    model, draft = tiny()
    with pytest.raises(ValueError, match="must be one of"):
        generator(model, draft)


def test_tensorfold_cohort_limit_defaults_singleton_and_is_target_only(monkeypatch):
    model, draft = tiny()
    monkeypatch.setenv("MLX2_QWEN_TARGET_EXECUTION", "tensorfold")
    monkeypatch.setenv("MLX2_TENSORFOLD_SOURCE", "/unused")
    singleton = generator(model, draft, completion_batch_size=8)
    assert singleton.tensorfold_cohort_limit == 1
    assert singleton.scheduler_stats["external_tensorfold_cohort_limit"] == 1

    monkeypatch.setenv("MLX2_TENSORFOLD_COHORT_LIMIT", "4")
    cohort = generator(model, draft, completion_batch_size=8)
    assert cohort.tensorfold_cohort_limit == 4
    assert cohort.scheduler_stats["external_tensorfold_cohort_limit"] == 4

    monkeypatch.setenv("MLX2_QWEN_TARGET_EXECUTION", "reference")
    with pytest.raises(ValueError, match="requires MLX2_QWEN_TARGET_EXECUTION=tensorfold"):
        generator(model, draft)


# -- task 1: executor cache ---------------------------------------------------


@pytest.fixture
def fake_tensorfold(tmp_path, monkeypatch):
    """Two fake TensorFold roots whose lane_tree records its capture taps."""
    lane_tree_source = '''
import mlx.core as mx
def tree_forward(core, head, tokens, parents, cache, start):
    for layer in core.layers:
        storage = getattr(layer, "_storage", None)
        if storage is not None:
            storage[layer._idx] = mx.full((1, len(tokens), 2), float(layer._idx))
    return mx.zeros((1, len(tokens), 4)), {"start": start, "parents": list(parents)}
def commit_tree(cache, record, path, window, start):
    cache.append(("commit", list(path), window, start, record["parents"]))
'''
    roots = []
    for name in ("a", "b"):
        base = tmp_path / name / "src" / "tensorfold" / "kernels" / "qwen" / "dense" / "v1"
        base.mkdir(parents=True)
        for package in (base.parents[3], base.parents[2], base.parents[1], base.parents[0], base):
            (package / "__init__.py").write_text("")
        for module in qwen38_tensorfold._KERNEL_MODULES:
            leaf = module.rsplit(".", 1)[1]
            (base / f"{leaf}.py").write_text(lane_tree_source if leaf == "lane_tree" else "")
        roots.append(tmp_path / name)
    for key in [k for k in sys.modules if k == "tensorfold" or k.startswith("tensorfold.")]:
        monkeypatch.delitem(sys.modules, key)
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(qwen38_tensorfold, "_EXECUTORS", {})
    calls = []

    def check_output(command, cwd, text):
        calls.append(cwd)
        return getattr(check_output, "revision", qwen38_tensorfold.EXPECTED_REVISION) + "\n"

    monkeypatch.setattr(qwen38_tensorfold.subprocess, "check_output", check_output)
    yield roots, calls, check_output
    for key in [k for k in sys.modules if k == "tensorfold" or k.startswith("tensorfold.")]:
        del sys.modules[key]


@pytest.mark.parametrize("cached", [False, True])
def test_executor_rejects_wrong_revision(fake_tensorfold, cached):
    roots, calls, check_output = fake_tensorfold
    check_output.revision = "0" * 40
    with pytest.raises(RuntimeError, match="revision mismatch"):
        qwen38_tensorfold._modules(roots[0], cached=cached)
    assert not qwen38_tensorfold._EXECUTORS


def _fake_model():
    class Layer:
        pass

    class Core:
        layers = [Layer() for _ in range(4)]

    class Model:
        model = Core()

        @staticmethod
        def logits(hidden):
            return hidden

    return Model()


@pytest.mark.parametrize("cached", [False, True])
def test_repeated_forwards_validate_once_when_cached(fake_tensorfold, cached):
    roots, calls, _ = fake_tensorfold
    model = _fake_model()
    transactions = []
    for _ in range(4):
        logits, features, transaction = _forward_with_offset(model, roots[0], cached)
        transactions.append(transaction)
        assert features.shape == (1, 3, 4)
        assert np.asarray(features)[0, 0].tolist() == [0.0, 0.0, 3.0, 3.0]
    assert len(calls) == (1 if cached else 4)
    assert [t.executor_cached for t in transactions] == (
        [False, True, True, True] if cached else [False] * 4
    )
    if cached:
        storage = model._mlx2_tensorfold_capture[1]
        assert all(model.model.layers[i]._storage is storage for i in (0, 3))


def _forward_with_offset(model, root, cached):
    class Entry:
        offset = 5

    return qwen38_tensorfold.forward(
        model, [1, 2, 3], [-1, 0, 0], [Entry()], (0, 3), root, cached=cached
    )


def test_cached_executor_refuses_module_from_other_root(fake_tensorfold):
    roots, calls, _ = fake_tensorfold
    qwen38_tensorfold._modules(roots[0], cached=True)
    with pytest.raises(RuntimeError, match="not the validated source"):
        qwen38_tensorfold._modules(roots[1], cached=True)
    assert set(qwen38_tensorfold._EXECUTORS) == {roots[0].resolve()}


def test_cached_executor_does_not_spawn_git_per_forward(fake_tensorfold, monkeypatch):
    roots, calls, _ = fake_tensorfold
    model = _fake_model()
    _forward_with_offset(model, roots[0], True)
    spawned = []
    monkeypatch.setattr(
        subprocess, "Popen", lambda *a, **k: spawned.append(a) or (_ for _ in ()).throw(AssertionError)
    )
    for _ in range(3):
        _forward_with_offset(model, roots[0], True)
    assert not spawned and len(calls) == 1


def test_cohort_executor_keeps_parents_cache_and_commit_paths_per_lane(fake_tensorfold):
    roots, calls, _ = fake_tensorfold
    model = _fake_model()

    class Entry:
        def __init__(self, offset):
            self.offset = offset

    caches = [[Entry(5)], [Entry(9)]]
    logits, features, transaction = qwen38_tensorfold.forward_many(
        model,
        [[1, 2, 3], [4, 5, 6]],
        [[-1, 0, 0], [-1, 0, 1]],
        caches,
        (0, 3),
        roots[0],
        cached=True,
    )
    assert logits.shape == (2, 3, 4)
    assert features.shape == (2, 3, 4)
    rows = transaction.commit_paths([[0, 2], [0, 1, 2]])
    assert rows == caches
    assert caches[0][-1] == ("commit", [0, 2], 3, 5, [-1, 0, 0])
    assert caches[1][-1] == ("commit", [0, 1, 2], 3, 9, [-1, 0, 1])
    assert transaction.closed and all(item.closed for item in transaction.transactions)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "tokens, parents, cache_alias, message",
    [
        ([[1]] * 5, [[-1]] * 5, False, "takes 1..4 lanes"),
        ([[1], [2, 3]], [[-1], [-1, 0]], False, "uniform token/parent widths"),
        ([[1], [2]], [[-1], [-1]], True, "one cache owner per lane"),
    ],
)
def test_cohort_executor_fails_closed_on_bounds_and_ownership(
    fake_tensorfold, tokens, parents, cache_alias, message
):
    roots, _, _ = fake_tensorfold

    class Entry:
        offset = 5

    caches = [[Entry()] for _ in tokens]
    if cache_alias:
        caches[1] = caches[0]
    with pytest.raises(ValueError, match=message):
        qwen38_tensorfold.forward_many(
            _fake_model(), tokens, parents, caches, (0, 3), roots[0], cached=True
        )


def _install_reference_cohort(batch, monkeypatch, seen):
    """Exercise cohort scheduling with the generic exact tree as CPU oracle."""

    def target(cohort, inputs, parents):
        seen.append((tuple(lane.uid for lane in cohort), tuple(map(tuple, parents))))
        results = [
            batch._reference_tree_forward(lane, row, parent_row)
            for lane, row, parent_row in zip(cohort, inputs, parents)
        ]
        if len(cohort) > 1:
            batch.scheduler_stats["external_tensorfold_target_rounds"] += 1
            batch.scheduler_stats["external_tensorfold_cohort_rounds"] += 1
            batch.scheduler_stats["external_tensorfold_cohort_lanes"] += len(cohort)
            batch.scheduler_stats["external_tensorfold_cohort_max_width"] = max(
                batch.scheduler_stats["external_tensorfold_cohort_max_width"],
                len(cohort),
            )
        return (
            mx.concatenate([result[0] for result in results]),
            mx.concatenate([result[1] for result in results]),
            qwen38_tensorfold.TensorfoldCohortTransaction(
                [result[2] for result in results]
            ),
        )

    monkeypatch.setattr(batch, "_target_tree_forward_many", target)


def test_tensorfold_tree_cohort_matches_single_lane_reference(monkeypatch):
    prompts = [[1, 2, 3], [4, 5, 6]]
    seeds = [type("Seed", (), {"key": mx.array([17 + row])})() for row in range(2)]

    monkeypatch.setenv("MLX2_DFLASH_TOPOLOGY", "tree15")
    model, draft = tiny(vocab=64, top_k=16, block_size=8)
    reference = generator(model, draft, ready_drain="all")
    reference.insert(
        prompts,
        max_tokens=[12, 12],
        sampling_configs=[{"sampling_temp": 0.8, "top_k": 20}] * 2,
        lane_rngs=seeds,
    )
    reference_lanes = list(reference.lanes.values())
    expected, _ = drain(reference)
    expected_draws = [lane.rng.draws for lane in reference_lanes]

    monkeypatch.setenv("MLX2_QWEN_TARGET_EXECUTION", "tensorfold")
    monkeypatch.setenv("MLX2_TENSORFOLD_SOURCE", "/unused-by-cpu-oracle")
    monkeypatch.setenv("MLX2_TENSORFOLD_COHORT_LIMIT", "4")
    candidate = generator(model, draft, ready_drain="all")
    candidate.insert(
        prompts,
        max_tokens=[12, 12],
        sampling_configs=[{"sampling_temp": 0.8, "top_k": 20}] * 2,
        lane_rngs=seeds,
    )
    candidate_lanes = list(candidate.lanes.values())
    seen = []
    _install_reference_cohort(candidate, monkeypatch, seen)
    actual, receipts = drain(candidate)

    assert actual == expected
    assert [lane.rng.draws for lane in candidate_lanes] == expected_draws
    assert any(len(uids) == 2 for uids, _ in seen)
    assert all(1 <= len(uids) <= 4 for uids, _ in seen)
    stats = candidate.scheduler_stats
    assert stats["external_tensorfold_cohort_rounds"] > 0
    assert stats["external_tensorfold_cohort_lanes"] >= 2
    assert stats["external_tensorfold_cohort_max_width"] == 2
    assert stats["external_tensorfold_cohort_limit"] == 4
    assert all(receipt["target_width"] >= 2 for receipt in receipts.values())
    assert all(
        receipt["tensorfold_target"]["cohort_max_width"] == 2
        for receipt in receipts.values()
    )


def test_tensorfold_cohort_partial_commit_restores_every_lane(monkeypatch):
    _tree_env(monkeypatch)
    monkeypatch.setenv("MLX2_QWEN_TARGET_EXECUTION", "tensorfold")
    monkeypatch.setenv("MLX2_TENSORFOLD_SOURCE", "/unused-by-cpu-oracle")
    monkeypatch.setenv("MLX2_TENSORFOLD_COHORT_LIMIT", "4")
    model, draft = tiny(vocab=64, top_k=16, block_size=8)
    batch = generator(model, draft, ready_drain="all")
    batch.insert(
        [[1, 2, 3], [4, 5, 6]],
        max_tokens=[12, 12],
        sampling_configs=[{"sampling_temp": 0.8, "top_k": 20}] * 2,
        lane_rngs=[
            type("Seed", (), {"key": mx.array([31])})(),
            type("Seed", (), {"key": mx.array([32])})(),
        ],
    )
    lanes = list(batch.lanes.values())
    for lane in lanes:
        while lane.anchor is None:
            batch._prefill(lane, step=3)
    before = [_lane_state(lane) for lane in lanes]
    _install_reference_cohort(batch, monkeypatch, [])

    def fail_after_first_commit(
        self, cohort, decisions, features, *, blocks, transaction
    ):
        transaction.transactions[0].commit_paths([decisions[0].commit_rows])
        raise RuntimeError("injected cohort commit")

    monkeypatch.setattr(type(batch), "_commit", fail_after_first_commit)
    with pytest.raises(RuntimeError, match="injected cohort commit"):
        batch._tree_round(lanes)
    assert [_lane_state(lane) for lane in lanes] == before
    assert all(not lane.ready for lane in lanes)
    assert batch.scheduler_stats["recovery_checkpoint_restores"] == 2


# -- task 2: block lattice and codebook cache ---------------------------------


def _propose(d, m, **options):
    hidden = taps(m, [[1, 2, 3]])
    cache = d.batch_caches([d.make_cache()])
    return d.propose_tree([3], hidden, cache, 15, **options)


def test_lattice_positions_bound_the_draft_block(monkeypatch):
    m, d = tiny(vocab=64, top_k=16, block_size=8)
    shapes = []
    original = type(d)._hidden

    def record(self, inputs, hidden, cache):
        shapes.append(tuple(inputs.shape))
        return original(self, inputs, hidden, cache)

    monkeypatch.setattr(type(d), "_hidden", record)
    full = _propose(d, m)
    block = _propose(d, m, lattice_positions=d.config.block_size)
    assert shapes == [(1, 16), (1, 8)]
    for tokens, parents in (full[0], block[0]):
        assert len(tokens) == len(parents) == 15
        assert all(parent < row for row, parent in enumerate(parents))
    from mlx2.runtime.drafters.dflash_tree import tree_paths

    assert max(map(len, tree_paths(block[0][1]))) <= 7


def test_tree_forbidden_tokens_never_proposed():
    m, d = tiny(vocab=64, top_k=16, block_size=8)
    tokens, _ = _propose(d, m, lattice_positions=8)[0]
    banned = tuple(sorted(set(tokens[:4])))
    masked, parents = _propose(
        d, m, lattice_positions=8, forbidden_token_ids=[banned]
    )[0]
    assert not set(masked) & set(banned)
    assert all(parent < row for row, parent in enumerate(parents))


def test_codebook_cache_is_exact_and_rebuilds_on_new_weights():
    m, d = tiny(vocab=64, top_k=16, block_size=8)
    pred, succ, hit = d.tree_codebooks()
    assert not hit
    again = d.tree_codebooks()
    assert again[2] and again[0] is pred and again[1] is succ
    assert _propose(d, m, codebooks=(pred, succ)) == _propose(d, m)
    selector = d.candidate_selector
    selector.successor_codebook.weight = selector.successor_codebook.weight * 2
    assert not d.tree_codebooks()[2]


def test_codebook_gate_engages(monkeypatch):
    _tree_env(monkeypatch, codebook_cache=True)
    m, d = tiny(vocab=64, top_k=16, block_size=8)
    _, _, _, stats = _run(m, d, max_tokens=12)
    rounds = stats["external_tree_rounds"]
    assert stats["external_tree_codebook_cache_hits"] == rounds - 1 > 0


# -- task 3/4/5: batched target laws, logprobs, single fence ------------------

SAMPLING = [
    (0.0, {}),
    (1.0, {"top_k": 20, "top_p": 0.95}),
    (0.8, {"top_k": 5}),
    (1.0, {"top_p": 0.9}),
]


@pytest.mark.parametrize("temp, sampling", SAMPLING)
@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("fence", [False, True])
@pytest.mark.parametrize("presence", [None, 1.5])
def test_batched_laws_match_row_reference(monkeypatch, temp, sampling, seed, fence, presence):
    m, d = tiny(vocab=64, top_k=16, block_size=8)
    kwargs = dict(temp=temp, sampling=sampling, seeds=(seed, 17), max_tokens=14,
                  minimum=9, stops=(0, 7), presence=presence)
    monkeypatch.setenv("MLX2_DFLASH_TOPOLOGY", "tree15")
    reference = _run(m, d, **kwargs)
    _tree_env(monkeypatch, batched_laws=True, single_fence=fence)
    batched = _run(m, d, **kwargs)
    assert batched[0] == reference[0]
    assert batched[2] == reference[2]
    stats = batched[3]
    assert stats["external_tree_batched_law_rounds"] == stats["external_tree_rounds"] > 0
    assert stats["external_tree_row_law_rounds"] == 0
    assert stats.get("external_tree_single_fence_rounds", 0) == (
        stats["external_tree_rounds"] if fence else 0
    )
    if sampling.get("top_k"):
        assert stats["external_tree_sparse_law_rows"] > 0
    for key in ("external_tree_accepted_edges", "accepted_proposals", "external_tree_nodes"):
        assert stats[key] == reference[3][key]


@pytest.mark.parametrize("temp, sampling", SAMPLING[:2])
def test_requested_logprobs_match_row_reference(monkeypatch, temp, sampling):
    m, d = tiny(vocab=64, top_k=16, block_size=8)
    kwargs = dict(temp=temp, sampling={**sampling, "emit_logprobs": True},
                  seeds=(5, 6), max_tokens=12, minimum=6)
    monkeypatch.setenv("MLX2_DFLASH_TOPOLOGY", "tree15")
    reference = _run(m, d, **kwargs)
    _tree_env(monkeypatch, batched_laws=True, logprobs_on_request=True)
    batched = _run(m, d, **kwargs)
    assert batched[0] == reference[0]
    assert len(batched[1]) == len(reference[1])
    for ours, theirs in zip(batched[1], reference[1]):
        np.testing.assert_array_equal(ours, theirs)
    assert batched[3]["external_tree_logprob_rows_skipped"] == 0


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("temp", [0.0, 1.0])
def test_unrequested_logprobs_are_skipped(monkeypatch, batched, temp):
    m, d = tiny(vocab=64, top_k=16, block_size=8)
    kwargs = dict(temp=temp, sampling={"emit_logprobs": False, "top_k": 20},
                  seeds=(9, 9), max_tokens=12)
    monkeypatch.setenv("MLX2_DFLASH_TOPOLOGY", "tree15")
    reference = _run(m, d, **kwargs)
    _tree_env(monkeypatch, batched_laws=batched, logprobs_on_request=True)
    skipped = _run(m, d, **kwargs)
    assert skipped[0] == reference[0] and skipped[2] == reference[2]
    assert all(value is None for value in skipped[1] + reference[1])
    stats = skipped[3]
    assert stats["external_tree_logprob_rows_skipped"] > 0


def _lane(history, processors, sampling, seed=3):
    return Lane(
        uid=0, history=list(history), remaining=deque(), cache=[], draft_cache=[],
        tail=None, rng=RequestRNG(seed), maximum=64, processors=processors,
        sampling=dict(sampling), anchor=history[-1] if history else 1,
    )


def _chain_logits(vocab, tokens, depth, rng):
    """Target logits whose greedy pick follows ``tokens`` for ``depth`` rows."""
    width = len(tokens) + 1
    logits = rng.normal(size=(1, width, vocab)).astype(np.float32)
    for row in range(width):
        want = tokens[row] if row < depth else (tokens[row] + 1) % vocab if row < len(tokens) else 3
        logits[0, row, want] += 20.0
    return mx.array(logits)


@pytest.mark.parametrize("depth", range(16))
@pytest.mark.parametrize("temp", [0.0, 1.0])
@pytest.mark.parametrize("presence", [None, 1.5])
def test_every_accepted_depth_matches_row_reference(monkeypatch, depth, temp, presence):
    _tree_env(monkeypatch, batched_laws=True)
    m, d = tiny(vocab=64, top_k=16, block_size=8)
    b = generator(m, d, stop_tokens=[[0]])
    vocab = 64
    tokens = [(5 + 3 * i) % 60 + 1 for i in range(15)]
    block = TreeDraftRow(tokens, list(range(-1, 14)))
    logits = _chain_logits(vocab, tokens, depth, np.random.default_rng(depth))
    sampling = {"sampling_temp": temp, "top_k": 20, "top_p": 0.95, "emit_logprobs": True}
    processors = _serving_processors(3, presence=presence)
    processors.append(minimum_tokens_processor(mx, [0], 4, 12))
    row_lane = _lane([1, 2, 3, 4], processors, sampling)
    batch_lane = _lane([1, 2, 3, 4], processors, sampling)
    reference = b._verify_tree(row_lane, block, logits)
    fence, state = b._launch_tree_laws(batch_lane, block, logits)
    mx.eval(*fence)
    batched = b._verify_tree_batched(batch_lane, block, state)
    assert batched.emitted == reference.emitted
    assert batched.commit_rows == reference.commit_rows
    if temp == 0:
        assert reference.accepted == depth
    for ours, theirs in zip(batched.target_laws, reference.target_laws):
        np.testing.assert_array_equal(ours, theirs)
    for ours, theirs in zip(batched.response_logprobs, reference.response_logprobs):
        if theirs is None:
            assert ours is None
        else:
            np.testing.assert_array_equal(np.asarray(ours), np.asarray(theirs))
    assert batch_lane.rng.draws == row_lane.rng.draws == len(reference.emitted)
    assert batch_lane.rng.snapshot() == row_lane.rng.snapshot()


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("temp", [0.0, 1.0])
def test_branching_tree_non_prefix_paths_match(monkeypatch, seed, temp):
    _tree_env(monkeypatch, batched_laws=True)
    m, d = tiny(vocab=64, top_k=16, block_size=8)
    b = generator(m, d, stop_tokens=[[0]])
    rng = np.random.default_rng(seed)
    parents = [-1, -1, -1, 0, 0, 1, 3, 3, 4, 6, 6, 8, 9, 11, 13]
    tokens = [int(t) for t in rng.integers(1, 63, size=15)]
    block = TreeDraftRow(tokens, parents)
    target_parents = b._target_tree_parents(block)
    logits = rng.normal(size=(1, 16, 64)).astype(np.float32)
    # Point each row at one of its children, chosen at random, so the walk
    # takes non-prefix branches of the proposal order.
    for row in range(16):
        kids = [c for c, p in enumerate(target_parents) if p == row]
        if kids:
            logits[0, row, tokens[int(rng.choice(kids)) - 1]] += 6.0
    logits = mx.array(logits)
    sampling = {"sampling_temp": temp, "top_k": 8, "emit_logprobs": True}
    row_lane, batch_lane = _lane([1, 2], [], sampling, seed), _lane([1, 2], [], sampling, seed)
    reference = b._verify_tree(row_lane, block, logits)
    fence, state = b._launch_tree_laws(batch_lane, block, logits)
    mx.eval(*fence)
    batched = b._verify_tree_batched(batch_lane, block, state)
    assert (batched.emitted, batched.commit_rows) == (reference.emitted, reference.commit_rows)
    assert batch_lane.rng.snapshot() == row_lane.rng.snapshot()


def test_minimum_tokens_mask_follows_each_row_history(monkeypatch):
    # Rows of one tree sit at different depths; the mask must expire exactly
    # where each row's own history reaches the minimum.
    _tree_env(monkeypatch, batched_laws=True)
    m, d = tiny(vocab=64, top_k=16, block_size=8)
    b = generator(m, d, stop_tokens=[[0]])
    tokens = list(range(1, 16))
    block = TreeDraftRow(tokens, list(range(-1, 14)))
    logits = np.zeros((1, 16, 64), dtype=np.float32)
    logits[0, :, 0] = 30.0  # EOS dominant everywhere
    for row in range(15):
        logits[0, row, tokens[row]] = 10.0
    logits = mx.array(logits)
    processors = [minimum_tokens_processor(mx, [0], 2, 5)]  # masked until 5 generated
    sampling = {"sampling_temp": 0.0, "emit_logprobs": False}
    row_lane, batch_lane = _lane([1, 2], processors, sampling), _lane([1, 2], processors, sampling)
    reference = b._verify_tree(row_lane, block, logits)
    fence, state = b._launch_tree_laws(batch_lane, block, logits)
    mx.eval(*fence)
    batched = b._verify_tree_batched(batch_lane, block, state)
    assert reference.emitted == batched.emitted == [1, 2, 3, 4, 0]


def test_presence_penalty_changes_the_batched_law(monkeypatch):
    # The penalty must actually bite: a repeated path token loses its lead.
    _tree_env(monkeypatch, batched_laws=True)
    m, d = tiny(vocab=64, top_k=16, block_size=8)
    b = generator(m, d)
    block = TreeDraftRow([5, 5], [-1, 0])
    logits = np.zeros((1, 3, 64), dtype=np.float32)
    logits[0, :, 5] = 1.0
    logits[0, :, 6] = 0.5
    logits = mx.array(logits)
    sampling = {"sampling_temp": 0.0, "emit_logprobs": False}
    for processors, expected in (([], [5, 5, 5]),
                                 (_serving_processors(2, presence=1.5), [5, 6])):
        lanes = [_lane([1, 2], list(processors), sampling) for _ in range(2)]
        reference = b._verify_tree(lanes[0], block, logits)
        fence, state = b._launch_tree_laws(lanes[1], block, logits)
        mx.eval(*fence)
        batched = b._verify_tree_batched(lanes[1], block, state)
        assert reference.emitted == batched.emitted == expected


def test_frequency_penalty_keeps_row_reference(monkeypatch):
    _tree_env(monkeypatch, batched_laws=True)
    m, d = tiny(vocab=64, top_k=16, block_size=8)
    _, _, _, stats = _run(m, d, frequency=0.5, presence=1.5, minimum=4)
    assert stats["external_tree_row_law_rounds"] == stats["external_tree_rounds"] > 0
    assert stats["external_tree_batched_law_rounds"] == 0


def test_other_processors_keep_row_reference(monkeypatch):
    _tree_env(monkeypatch, batched_laws=True)
    m, d = tiny(vocab=64, top_k=16, block_size=8)
    b = generator(m, d, pairwise_selection="batched")

    def bias(tokens, logits):
        return logits

    bias.forbidden_token_ids_at_length = lambda length: ()
    b.insert([[1, 2, 3]], max_tokens=[8], logits_processors=[[bias]],
             sampling_configs=[{"sampling_temp": 0.0}])
    drain(b)
    stats = b.scheduler_stats
    # Not history_pure: no proven batched contract, so every round is row-wise.
    assert stats["external_tree_row_law_rounds"] == stats["external_tree_rounds"] > 0
    assert stats["external_tree_batched_law_rounds"] == 0


def test_single_fence_is_one_explicit_eval_per_target_phase(monkeypatch):
    m, d = tiny(vocab=64, top_k=16, block_size=8)
    counts = {}
    for label, gates in (("two", {"batched_laws": True}),
                         ("one", {"batched_laws": True, "single_fence": True})):
        _tree_env(monkeypatch, **gates)
        b = generator(m, d)
        b.insert([[1, 2, 3]], max_tokens=[6], sampling_configs=[{"sampling_temp": 0.0}])
        lane = next(iter(b.lanes.values()))
        while lane.anchor is None:
            b.next()
        calls = []
        original = b.mx.eval
        monkeypatch.setattr(b, "mx", type("MX", (), {
            "__getattr__": lambda self, name: getattr(mx, name),
            "eval": lambda self, *a: calls.append(len(a)) or original(*a),
        })())
        b.next()
        counts[label] = len(calls)
        monkeypatch.setattr(b, "mx", mx)
    # Draft lattice eval + reference target evals are shared; the fence
    # removes exactly the separate logits/taps eval.
    assert counts["two"] - counts["one"] == 1


# -- phase timing --------------------------------------------------------------


def test_phase_timing_receipt_without_extra_syncs(monkeypatch):
    m, d = tiny(vocab=64, top_k=16, block_size=8)
    evals = {}
    for timing in ("", "1"):
        _tree_env(monkeypatch, batched_laws=True, single_fence=True)
        if timing:
            monkeypatch.setenv("MLX2_EXTERNAL_ROUND_TIMING", "1")
        calls = []
        original = mx.eval
        monkeypatch.setattr(mx, "eval", lambda *a: calls.append(1) or original(*a))
        _, _, _, stats = _run(m, d, max_tokens=12)
        monkeypatch.setattr(mx, "eval", original)
        evals[timing] = len(calls)
    assert evals[""] == evals["1"]
    phases = {k for k in stats if k.startswith("external_phase_")}
    for phase in ("tree_recovery_capture", "tree_draft_setup", "tree_draft_build",
                  "tree_draft_wait", "tree_search", "tree_target_launch",
                  "tree_target_wait", "tree_target_law", "transaction_commit",
                  "emit", "tree_round"):
        assert f"external_phase_{phase}_ns" in phases
        assert isinstance(stats[f"external_phase_{phase}_ns"], int)
    assert stats["external_phase_rounds"] == stats["external_tree_rounds"]


# -- fault injection and exact recovery ---------------------------------------


def _lane_state(lane):
    return (
        list(lane.history), lane.anchor, lane.generated,
        [int(c.offset) for c in lane.cache if hasattr(c, "offset")],
        [int(c.offset) for c in lane.draft_cache],
        tuple(lane.tail.shape), np.asarray(lane.tail).tobytes(),
        lane.rng.snapshot(), lane.rng.draws,
    )


PHASES = ["proposal", "target_launch", "sampling", "commit", "emit"]


@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("mode", ["row", "batched", "pipelined"])
def test_fault_at_each_phase_restores_committed_boundary(monkeypatch, phase, mode):
    m, d = tiny(vocab=64, top_k=16, block_size=8)
    batched = mode != "row"
    gates = {"batched_laws": True, "single_fence": True} if batched else {}
    if mode == "pipelined":
        gates["pipeline_draft"] = True
    _tree_env(monkeypatch, **gates)
    clean = _run(m, d, temp=1.0, sampling={"top_k": 20}, max_tokens=14, minimum=8)

    b = generator(m, d, pairwise_selection="batched")
    b.insert([[1, 2, 3]], max_tokens=[14],
             logits_processors=[[minimum_tokens_processor(mx, [0, 7], 3, 8)]],
             sampling_configs=[{"sampling_temp": 1.0, "top_k": 20}],
             lane_rngs=[type("Seed", (), {"key": mx.array([123])})()])
    lane = next(iter(b.lanes.values()))
    tokens = []
    while lane.anchor is None:
        tokens += [r.token for r in b.next()[1]]
    tokens += [r.token for r in b.next()[1]]  # one clean tree round first
    while lane.ready:
        tokens += [r.token for r in b.next()[1]]
    rounds = b.scheduler_stats["external_tree_rounds"]
    assert rounds >= 1
    before = _lane_state(lane)

    def boom(*_args, **_kwargs):
        raise RuntimeError(f"injected {phase}")

    with monkeypatch.context() as patch:
        if phase == "proposal":
            patch.setattr(b, "_target_tree_forward", boom)
        elif phase == "target_launch":
            patch.setattr(b, "_launch_tree_laws" if batched else "_verify_tree", boom)
        elif phase == "sampling":
            # Class-level: the round's recovery snapshot deep-copies the RNG.
            patch.setattr(RequestRNG, "sample", boom)
        elif phase == "commit":
            # The accepted path is committed to the target, then the commit
            # step fails before any lane field is published.
            original = type(b)._commit

            def failing_commit(self, cohort, decisions, features, *, blocks, transaction):
                transaction.commit_paths([decisions[0].commit_rows])
                boom()

            patch.setattr(type(b), "_commit", failing_commit)
        else:
            # Commit and lane updates land; the first response publish fails.
            patch.setattr(b, "_verification_receipt", boom)
        with pytest.raises(RuntimeError, match="injected"):
            b.next()
    assert _lane_state(lane) == before
    assert not lane.ready
    assert b.scheduler_stats["recovery_checkpoint_restores"] == 1
    assert b.scheduler_stats["external_tree_rounds"] == rounds
    for _ in range(100):
        tokens += [r.token for r in b.next()[1]]
        if not b.lanes:
            break
    assert tokens == clean[0]


# -- task 7: pipelined next-round draft --------------------------------------

COMPOSED = dict(batched_laws=True, single_fence=True, codebook_cache=True,
                logprobs_on_request=True)


@pytest.mark.parametrize("temp, sampling", SAMPLING[:2])
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("presence", [None, 1.5])
@pytest.mark.parametrize("others", [{}, COMPOSED])
def test_pipelined_draft_matches_serial(monkeypatch, temp, sampling, seed, presence, others):
    m, d = tiny(vocab=64, top_k=16, block_size=8)
    kwargs = dict(temp=temp, sampling=sampling, seeds=(seed, 3), max_tokens=16,
                  minimum=10, stops=(0, 7), presence=presence)
    _tree_env(monkeypatch, **others)
    reference = _run(m, d, **kwargs)
    _tree_env(monkeypatch, pipeline_draft=True, **others)
    piped = _run(m, d, **kwargs)
    assert piped[0] == reference[0] and piped[2] == reference[2]
    stats = piped[3]
    rounds = stats["external_tree_rounds"]
    # Every round after the first adopts the lattice its predecessor queued.
    assert stats["external_tree_pipelined_drafts"] == rounds - 1 > 0
    assert stats["external_tree_pipeline_discards"] == 0
    for key in ("external_tree_accepted_edges", "external_tree_nodes", "accepted_proposals"):
        assert stats[key] == reference[3][key]


def _started(b, tokens=None):
    lane = next(iter(b.lanes.values()))
    while lane.anchor is None:
        responses = b.next()[1]
        if tokens is not None:
            tokens += [r.token for r in responses]
    return lane


def _draft_offsets(lane):
    return [int(c.offset) for c in lane.draft_cache]


def test_prelaunch_leaves_committed_draft_plane_untouched(monkeypatch):
    _tree_env(monkeypatch, pipeline_draft=True)
    m, d = tiny(vocab=64, top_k=16, block_size=8)
    b = generator(m, d)
    b.insert([[1, 2, 3]], max_tokens=[12], sampling_configs=[{"sampling_temp": 0.0}])
    lane = _started(b)
    seen = []
    original = b._prelaunch_tree

    def watch(lane, decision):
        before = (_draft_offsets(lane), lane.draft_cache[0].keys, lane.tail)
        pending = int(lane.tail.shape[1])
        original(lane, decision)
        after = (_draft_offsets(lane), lane.draft_cache[0].keys, lane.tail)
        seen.append((before, after, b._prelaunched.get(lane.uid), pending))

    monkeypatch.setattr(b, "_prelaunch_tree", watch)
    while lane.ready:
        b.next()
    b.next()
    assert seen
    assert any(record is not None for _, _, record, _ in seen)
    for before, after, record, pending in seen:
        assert before[0] == after[0]
        assert before[1] is after[1] and before[2] is after[2]
        if record is not None:
            # Only the copy carries the queued context append.
            assert record.draft_cache is not lane.draft_cache
            assert [int(c.offset) for c in record.draft_cache] == [
                offset + pending for offset in before[0]
            ]


@pytest.mark.parametrize("depth", range(16))
def test_prelaunched_lattice_equals_fresh_at_every_accepted_depth(depth):
    # A commit of accepted depth d leaves a d+1-row tail; the lattice queued
    # on a copy must equal the one the next round would build in place, and
    # the adopted copy must end where the in-place draft plane would.
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator as G

    m, d = tiny(vocab=64, top_k=16, block_size=8)
    prompt = list(range(1, 20))
    context = taps(m, [prompt[:-(depth + 1)]])
    tail = taps(m, [prompt])[:, -(depth + 1):]
    live = d.make_cache()
    d.append_context(context, live)
    copy_cache, copy_tail = G._snapshot_draft_state(live, tail)
    state = d.start_tree([5], copy_tail, d.batch_caches([copy_cache]), 15,
                         forbidden_token_ids=[(7,)])
    mx.async_eval(*state["pending"])
    before = [int(c.offset) for c in live]
    queued = d.finish_tree(state)
    assert [int(c.offset) for c in live] == before
    fresh = d.propose_tree([5], tail, d.batch_caches([live]), 15, forbidden_token_ids=[(7,)])
    assert queued == fresh
    assert [int(c.offset) for c in copy_cache] == [int(c.offset) for c in live]
    for ours, theirs in zip(copy_cache, live):
        np.testing.assert_array_equal(np.asarray(ours.keys), np.asarray(theirs.keys))


def test_fault_after_adoption_restores_and_discards(monkeypatch):
    _tree_env(monkeypatch, pipeline_draft=True)
    m, d = tiny(vocab=64, top_k=16, block_size=8)
    clean = _run(m, d, temp=1.0, sampling={"top_k": 20}, max_tokens=14, minimum=8)
    b = generator(m, d, pairwise_selection="batched")
    b.insert([[1, 2, 3]], max_tokens=[14],
             logits_processors=[[minimum_tokens_processor(mx, [0, 7], 3, 8)]],
             sampling_configs=[{"sampling_temp": 1.0, "top_k": 20}],
             lane_rngs=[type("Seed", (), {"key": mx.array([123])})()])
    tokens = []
    lane = _started(b, tokens)
    while lane.ready:
        tokens += [r.token for r in b.next()[1]]
    tokens += [r.token for r in b.next()[1]]
    while lane.ready:
        tokens += [r.token for r in b.next()[1]]
    assert lane.uid in b._prelaunched
    before = _lane_state(lane)
    committed_plane = lane.draft_cache

    def boom(*_a, **_k):
        raise RuntimeError("injected after adoption")

    with monkeypatch.context() as patch:
        patch.setattr(b, "_target_tree_forward", boom)
        with pytest.raises(RuntimeError, match="after adoption"):
            b.next()
    assert _lane_state(lane) == before
    assert lane.uid not in b._prelaunched
    assert [int(c.offset) for c in lane.draft_cache] == [int(c.offset) for c in committed_plane]
    for _ in range(100):
        tokens += [r.token for r in b.next()[1]]
        if not b.lanes:
            break
    assert tokens == clean[0]
    assert b.scheduler_stats["recovery_checkpoint_restores"] == 1


def test_changed_boundary_discards_queued_draft(monkeypatch):
    _tree_env(monkeypatch, pipeline_draft=True)
    m, d = tiny(vocab=64, top_k=16, block_size=8)
    clean = _run(m, d, max_tokens=12)
    b = generator(m, d, pairwise_selection="batched")
    b.insert([[1, 2, 3]], max_tokens=[12], sampling_configs=[{"sampling_temp": 0.0}],
             lane_rngs=[type("Seed", (), {"key": mx.array([123])})()])
    tokens = []
    lane = _started(b, tokens)
    while lane.ready:
        tokens += [r.token for r in b.next()[1]]
    assert lane.uid in b._prelaunched
    lane.tail = lane.tail + 0  # same values, another object: not the queued boundary
    for _ in range(100):
        tokens += [r.token for r in b.next()[1]]
        if not b.lanes:
            break
    assert tokens == clean[0]
    assert b.scheduler_stats["external_tree_pipeline_discards"] == 1


@pytest.mark.parametrize("action", ["remove", "disable"])
def test_membership_and_route_changes_discard_queued_draft(monkeypatch, action):
    _tree_env(monkeypatch, pipeline_draft=True)
    m, d = tiny(vocab=64, top_k=16, block_size=8)
    b = generator(m, d)
    b.insert([[1, 2, 3]], max_tokens=[12], sampling_configs=[{"sampling_temp": 0.0}])
    lane = _started(b)
    while lane.ready:
        b.next()
    assert lane.uid in b._prelaunched
    if action == "remove":
        b.remove([lane.uid])
    else:
        b.disable_speculation(lane.uid)
    assert lane.uid not in b._prelaunched
    assert b.scheduler_stats["external_tree_pipeline_discards"] == 1
