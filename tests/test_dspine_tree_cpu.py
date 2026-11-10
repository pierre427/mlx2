"""DSpine best-first tree over the exact-sender tables, on tiny CPU models.

Search properties are checked against an independent brute-force reference
over the same edge laws; the route test runs the tree through the tree15
target-tree route on the tiny hybrid Qwen3.8 target and compares with
ordinary greedy decoding.
"""
import math
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from mlx2.runtime.drafters.dspine import dspine_best_first_tree

from test_dspine_drafter_cpu import context, hybrid_pair, tiny_pair


def _laws(draft, cfg, length=7, anchors=(5,), depth=None):
    depth = draft.proposal_rows if depth is None else depth
    hidden = context(cfg, length, batch=len(anchors))
    cache = draft.batch_caches([draft.make_cache() for _ in anchors])
    state = draft.start_tree(list(anchors), hidden, cache, depth)
    cand, law = state["pending"]
    mx.eval(cand, law)
    return np.asarray(cand), np.asarray(law).astype(np.float64)


def _reference_tree(cand, law, max_nodes, children, forbidden=()):
    """Exhaustive: every path whose every edge is among its parent's top
    ``children`` (root: among the allowed roots), ranked by summed log-law."""
    n, k = cand.shape
    paths = []

    def top(row, skip=()):
        order = [int(r) for r in np.argsort(-row, kind="stable") if int(cand_row_token[0][r]) not in skip]
        return order[:children]

    cand_row_token = [cand[0]]
    frontier = [((r,), law[0, 0, r]) for r in top(law[0, 0], set(forbidden))]
    while frontier:
        nxt = []
        for path, score in frontier:
            paths.append((path, score))
            d = len(path)
            if d < n:
                row = law[d, path[-1]]
                order = [int(r) for r in np.argsort(-row, kind="stable")][:children]
                nxt.extend(((*path, r), score + row[r]) for r in order)
        frontier = nxt
    return sorted(paths, key=lambda item: -item[1])


def _f64(x):
    return np.asarray(x.astype(mx.float32)).astype(np.float64)


def test_edge_law_equals_numpy_logits_of_the_refined_row_per_predecessor():
    """Independent second pass: redo the last bridge write with each actual
    predecessor's codebook code, apply the final RMSNorm and the target
    head, and take the log-softmax over the row's candidates."""
    cfg, draft, target = tiny_pair(seed=6)
    anchor, depth = 5, draft.proposal_rows
    hidden = context(cfg, 7)
    inputs = mx.array([[anchor] + [cfg.mask_token_id] * depth])
    _normed, stash = draft.block_forward(inputs, hidden, draft.batch_caches([draft.make_cache()]))
    cand, law = _laws(draft, cfg, anchors=(anchor,))
    cand, law = cand[0], law[0]
    bridge = draft.selected_prefix_bridge
    R, C = bridge.rank, bridge.code_size
    raw, rms = _f64(stash["raw"])[0], _f64(stash["rms"])[0]
    q = _f64(stash["packed"])[0, :, :R]
    gp, a, b = _f64(bridge.predecessor_gate.weight), _f64(bridge.prev.weight), _f64(bridge.gate.bias)
    write = _f64(bridge.feature_write_weights[-1])
    book = _f64(bridge.codebook) * math.sqrt(C)
    norm_w, eps = _f64(draft.norm.weight), draft.norm.eps
    head = _f64(target.lm_head.weight)
    for i in range(depth):
        t = i + 1  # block row
        preds = [anchor] * cand.shape[1] if i == 0 else cand[i - 1]
        for s_index, pred in enumerate(preds):
            code = book[int(pred)]
            message = 1 / (1 + np.exp(-(q[t] + gp @ code + b))) * (a @ code)
            out = raw[t] + rms[t] * (write @ message)
            normed = out / np.sqrt(np.mean(out * out) + eps) * norm_w
            logits = head[cand[i]] @ normed
            want = logits - logits.max() - np.log(np.exp(logits - logits.max()).sum())
            np.testing.assert_allclose(law[i, s_index], want, atol=0.08, rtol=0.02)
            assert int(np.argmax(law[i, s_index])) == int(np.argmax(want))
    np.testing.assert_allclose(np.exp(law).sum(-1), 1.0, atol=1e-5)


def test_single_child_tree_is_the_greedy_exact_sender_chain():
    cfg, draft, _ = tiny_pair()
    hidden = context(cfg, 7)
    chain = np.asarray(draft.propose_chain(mx.array([5]), hidden, draft.batch_caches([draft.make_cache()])))[0]
    draft.tree_children = 1
    (tokens, parents, info), = draft.propose_tree([5], hidden, draft.batch_caches([draft.make_cache()]), 31)
    assert tokens == chain.tolist()
    assert parents == list(range(-1, len(tokens) - 1))
    assert info["depths"] == list(range(1, len(tokens) + 1))


@pytest.mark.parametrize("children", [2, 3, 4])
def test_best_first_tree_equals_exhaustive_reference(children):
    cfg, draft, _ = tiny_pair(seed=11)
    cand, law = _laws(draft, cfg)
    cand, law = cand[0], law[0]
    ref = _reference_tree(cand, law, 31, children)
    tokens, parents, info = dspine_best_first_tree(cand, law, 31, children=children)
    # Node set and order: the first m popped paths are the m best paths.
    paths = []
    for row, parent in enumerate(parents):
        paths.append(([] if parent < 0 else paths[parent]) + [row])
    got = []
    for row, path in enumerate(paths):
        idx = []
        for depth, node in enumerate(path):
            idx.append(int(np.flatnonzero(cand[depth] == tokens[node])[0]))
        got.append((tuple(idx), info["path_log_q"][row]))
    want = ref[: len(got)]
    np.testing.assert_allclose([s for _, s in got], [s for _, s in want], atol=1e-9)
    assert {p for p, _ in got} == {p for p, _ in want} or len({round(s, 9) for _, s in want}) < len(want)


def test_tree_order_is_budget_prefix_invariant_and_path_law_non_increasing():
    cfg, draft, _ = tiny_pair(seed=5)
    hidden = context(cfg, 7)

    def tree(budget):
        return draft.propose_tree([5], hidden, draft.batch_caches([draft.make_cache()]), budget)[0]

    full = tree(31)
    # Smaller budgets shorten the block itself (fewer mask rows).
    for budget in (1, 2, 3, 4, 9, 15):
        small = tree(budget)
        assert small[0] == full[0][:budget] and small[1] == full[1][:budget]
    path = full[2]["path_log_q"]
    assert all(a >= b - 1e-12 for a, b in zip(path, path[1:]))
    assert all(p < row for row, p in enumerate(full[1]))
    assert max(full[2]["depths"]) <= draft.proposal_rows


def test_lattice_positions_bounds_the_tree_depth():
    cfg, draft, _ = tiny_pair(seed=5)
    hidden = context(cfg, 7)
    (tokens, parents, info), = draft.propose_tree(
        [5], hidden, draft.batch_caches([draft.make_cache()]), 31, lattice_positions=3,
    )
    assert max(info["depths"]) <= 2 and len(tokens) > 2
    with pytest.raises(ValueError, match="draft position"):
        draft.propose_tree([5], hidden, draft.batch_caches([draft.make_cache()]), 31,
                           lattice_positions=1)


def test_tree_feeds_the_prompt_lookup_graft():
    from mlx2.runtime.drafters.prompt_lookup_draft import PromptLookupDraftModel

    cfg, draft, target = tiny_pair(seed=5)
    hidden = context(cfg, 7)
    stub = SimpleNamespace(args=SimpleNamespace(vocab_size=cfg.vocab_size, hidden_size=cfg.hidden_size),
                           layers=[None])
    graft = PromptLookupDraftModel(stub, num_draft=5, base=draft)
    (tokens, parents, *_), = graft.propose_tree(
        [5], hidden, draft.batch_caches([draft.make_cache()]), 15, processor_histories=[[1, 2, 3]],
    )
    assert tokens and all(p < row for row, p in enumerate(parents))


def test_forbidden_ids_are_removed_from_the_root_only():
    cfg, draft, _ = tiny_pair(seed=5)
    hidden = context(cfg, 7)
    (base, base_parents, _), = draft.propose_tree([5], hidden, draft.batch_caches([draft.make_cache()]), 15)
    root = base[0]
    (tokens, parents, _), = draft.propose_tree(
        [5], hidden, draft.batch_caches([draft.make_cache()]), 15, forbidden_token_ids=[(root,)],
    )
    assert root not in [t for t, p in zip(tokens, parents) if p < 0]


def test_start_finish_equals_propose_and_appends_the_same_draft_context():
    cfg, draft, _ = tiny_pair(seed=4)
    hidden = context(cfg, 9, batch=2)
    rows_a = [draft.make_cache(), draft.make_cache()]
    rows_b = [draft.make_cache(), draft.make_cache()]
    want = draft.propose_tree([5, 7], hidden, draft.batch_caches(rows_a), 15)
    state = draft.start_tree([5, 7], hidden, draft.batch_caches(rows_b), 15)
    mx.async_eval(*state["pending"])
    got = draft.finish_tree(state)
    assert [t[:2] for t in got] == [t[:2] for t in want]
    for row_a, row_b in zip(rows_a, rows_b):
        assert [c.offset for c in row_a] == [c.offset for c in row_b] == [9, 9]
        for x, y in zip(row_a, row_b):
            np.testing.assert_array_equal(np.asarray(x.keys.astype(mx.float32)),
                                          np.asarray(y.keys.astype(mx.float32)))
    # Batched rows equal single rows.
    single = draft.propose_tree([7], hidden[1:], draft.batch_caches([draft.make_cache()]), 15)
    assert single[0][:2] == want[1][:2]


def test_geometry_is_refused():
    cfg, draft, _ = tiny_pair()
    hidden = context(cfg, 7)
    with pytest.raises(ValueError, match="1..31"):
        draft.start_tree([5], hidden, draft.batch_caches([draft.make_cache()]), 0)
    with pytest.raises(ValueError, match="forbidden"):
        draft.start_tree([5], hidden, draft.batch_caches([draft.make_cache()]), 4,
                         forbidden_token_ids=[(), ()])


# -- tree15 route on the tiny hybrid Qwen3.8 target ------------------------------

@pytest.fixture
def _tree_env(monkeypatch):
    for name in ("MLX2_DFLASH_TOPOLOGY", "MLX2_QWEN_TARGET_EXECUTION",
                 "MLX2_EXTERNAL_ROUND_TIMING", "MLX2_TENSORFOLD_COHORT_LIMIT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MLX2_TENSORFOLD_SOURCE", "/unused-by-cpu-oracle")


@pytest.mark.parametrize("width", [1, 4])
def test_tree_route_equals_ordinary_greedy_on_hybrid_target(monkeypatch, _tree_env, width):
    from mlx2.adapters.qwen38_target_tree_executor import Qwen38TargetTreeExecutor
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
    from test_dflash_tree_bridge import _install_reference_cohort
    from test_qwen38_dflash2_cpu import PROMPTS, ordinary_greedy

    _, target, draft = hybrid_pair()
    batch = ExternalDraftBatchGenerator(
        target, draft_model=draft, binding="dspine-tree-cpu", num_draft=5,
        prefill_step_size=4, completion_batch_size=4,
        dynamic_singleton_tree=True, dynamic_tree_max_width=width,
        tree_node_budget_by_lanes={w: n for w, n in zip(range(1, width + 1), (15, 7, 4, 3))},
        target_tree_executor_provider=Qwen38TargetTreeExecutor.create(),
        cost_per_commit_handoff=False,
    )
    _install_reference_cohort(batch, monkeypatch, [])
    prompts = PROMPTS[:1] if width == 1 else PROMPTS
    uids = batch.insert([list(p) for p in prompts], max_tokens=[16] * len(prompts),
                        sampling_configs=[{"sampling_temp": 0.0}] * len(prompts))
    out, final = {u: [] for u in uids}, {}
    for _ in range(800):
        _, responses = batch.next()
        for r in responses:
            out[r.uid].append(r.token)
            if r.finish_reason:
                final[r.uid] = r
        if not batch.lanes:
            break
    else:
        raise AssertionError("scheduler stalled")
    for uid, prompt in zip(uids, prompts):
        assert out[uid] == ordinary_greedy(target, prompt, 16)
        assert final[uid].speculative_receipt["kind"] == "external_dspine"
    assert draft.stats.get("tree_lattices", 0) > 0
    assert batch.scheduler_stats.get("external_auto_tree_rounds", 0) > 0
