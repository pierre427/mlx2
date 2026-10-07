"""SPEC-01: default-on MLX2_TREE_PIPELINE_DRAFT wasted a draft lattice per
lane per round at B2-B4 and never counted the discard.

The selected Qwen3.8 default (tree15_b1_b4_chain_b5plus_v1) turns on every
tree gate (``_tree_gates(default_on=dynamic_singleton_tree)``).  After each
tree commit ``_tree_round`` prelaunches the next lattice for EVERY lane, but
``_propose`` only adopts a prelaunched lattice when its tail-shape group has
exactly one lane.  At B2-B4 the queued lattice is computed (async_eval),
never adopted, and silently popped by the next ``_prelaunch_tree``.  Now a
multi-lane cohort does not prelaunch, and every queued lattice is either
adopted or counted as a discard.
"""
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx2.runtime.external_speculative import (
    ExternalDraftBatchGenerator,
    TreeDraftRow,
)


class StubDraft:
    def __init__(self):
        self.start_calls = 0
        self.finish_calls = 0
        self.fresh_calls = 0

    def draft_distributions(self, *a, **k):
        raise AssertionError("chain path not expected")

    def batch_caches(self, rows):
        return rows

    def start_tree(self, anchors, tail, caches, count, **_):
        self.start_calls += 1
        return {"pending": [mx.zeros((1,))], "n": count}

    def finish_tree(self, state, mark=None):
        self.finish_calls += 1
        return [(list(range(state["n"])), [-1] + list(range(state["n"] - 1)))]

    def propose_tree(self, anchors, tail, caches, count, **_):
        self.fresh_calls += 1
        return [(list(range(count)), [-1] + list(range(count - 1))) for _ in anchors]


def make_gen(width):
    gen = ExternalDraftBatchGenerator.__new__(ExternalDraftBatchGenerator)
    gen.mx = mx
    gen.draft = StubDraft()
    gen.draft_topology = "tree15"
    gen.dynamic_tree_max_width = 4
    gen.tree_node_budget_by_lanes = {1: 15, 2: 7, 3: 4, 4: 3}
    gen._auto_active_width = width
    gen.num_draft = 7
    gen.pair_context_tokens = False
    gen.pairwise_selection = "host"
    gen.tree_gates = {g: True for g in (
        "cache_executor", "codebook_cache", "batched_laws",
        "logprobs_on_request", "single_fence", "pipeline_draft")}
    gen.tree_gates["codebook_cache"] = False
    gen.scheduler_stats = {"external_tree_node_budget_histogram": {},
                           "external_draft_masked_positions": 0,
                           "draft_max_width": 1,
                           "external_tree_pipeline_discards": 0,
                           "external_tree_pipelined_drafts": 0}
    gen._tree_clock = None
    gen.adaptive_policy = None
    gen.continuation_policy = None
    gen.stops = set()
    gen._prelaunched = {}
    gen._snapshot_draft_state = lambda cache, tail: (cache, tail)
    return gen


def lane(uid):
    return SimpleNamespace(
        uid=uid, anchor=5, history=[1, 2, 3], maximum=100, generated=10,
        ordinary=False, cancelled=False, processors=[], tail=mx.zeros((1, 1, 4)),
        draft_cache=[object()], rng=None, sampling={},
    )


@pytest.mark.parametrize("width", [2, 4])
def test_prelaunched_lattices_are_adopted_or_not_computed(width):
    gen = make_gen(width)
    cohort = [lane(i) for i in range(width)]
    decision = SimpleNamespace(emitted=[7])
    for _round in range(3):
        for ln in cohort:                    # what _tree_round does post-commit
            gen._prelaunch_tree(ln, decision)
        blocks = gen._propose(cohort)        # next round's draft phase
        assert all(isinstance(b, TreeDraftRow) for b in blocks)
    prelaunched = gen.draft.start_calls
    adopted = gen.scheduler_stats["external_tree_pipelined_drafts"]
    discarded = gen.scheduler_stats["external_tree_pipeline_discards"]
    print(f"B{width}: prelaunched={prelaunched} adopted={adopted} "
          f"counted_discards={discarded} fresh_proposals={gen.draft.fresh_calls}")
    # Expected: every prelaunched lattice is either adopted or counted.
    assert adopted + discarded == prelaunched


def test_b1_adopts():
    gen = make_gen(1)
    ln = lane(0)
    gen._prelaunch_tree(ln, SimpleNamespace(emitted=[7]))
    gen._propose([ln])
    assert gen.scheduler_stats["external_tree_pipelined_drafts"] == 1


def test_serial_singleton_lanes_adopt_or_count_every_prelaunch(monkeypatch):
    """Two lanes on the CPU reference target run as singleton cohorts."""
    from test_dflash_tree_bridge import _tree_env, generator, tiny

    _tree_env(monkeypatch, pipeline_draft=True)
    m, d = tiny(vocab=64, top_k=16, block_size=8)
    b = generator(m, d, pairwise_selection="batched", completion_batch_size=2)
    starts = []
    real = d.start_tree
    monkeypatch.setattr(d, "start_tree", lambda *a, **k: starts.append(1) or real(*a, **k))
    b.insert([[1, 2, 3], [4, 5, 6]], max_tokens=[12, 12],
             sampling_configs=[{"sampling_temp": 0.0}] * 2)
    for _ in range(200):
        b.next()
        if not b.lanes:
            break
    stats = b.scheduler_stats
    assert stats["external_tree_pipelined_drafts"] > 0
    assert (
        stats["external_tree_pipelined_drafts"] + stats["external_tree_pipeline_discards"]
        == len(starts)
    )
