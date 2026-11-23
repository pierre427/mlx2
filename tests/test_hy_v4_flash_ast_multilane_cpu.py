"""Hy4-flash AST with two to four concurrent requests, and the PLD graft (CPU).

Tiny random checkpoint with the real tensor names (``test_hy_v4_flash_cpu``).
Every request on the AST route must equal its own single-lane ordinary
greedy decode; the joint (cross-lane packed) chain forward must be the path
that ran, shown by the executor receipt and the engine's packed-round
counters; aborts and partial commits must leave each lane exact.
"""

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from test_hy_v4_flash_cpu import (  # noqa: E402
    LONG,
    _drain,
    _greedy,
    _tiny_artifact,
    build,
    checkpoint,
    tiny_config,
)


@pytest.fixture(scope="module")
def tiny():
    config = tiny_config()
    W = checkpoint(config)
    model, drafter = build(config, W)
    return config, W, model, drafter


# The first prompt's greedy continuation loops (33 33 33 33), so prompt
# lookup finds copies; the third prompt repeats a span.
PROMPTS = [
    LONG[9:21],
    LONG[10:24],
    LONG[50:58] + LONG[50:55],
    LONG[60:81],
]


def _adapter(tmp_path, tiny, policy):
    from mlx2.adapters import hy_v4_flash as A

    config, W, _, _ = tiny
    return A.HYV4FlashAdapter(str(_tiny_artifact(tmp_path, config, W)), execution_policy=policy)


def _run(adapter, width, n=12):
    batch = adapter.create_external_batch(completion_batch_size=width, prefill_step_size=4)
    prompts = PROMPTS[:width]
    uids = batch.insert(prompts, max_tokens=[n] * width,
                        sampling_configs=[{"sampling_temp": 0}] * width)
    out = _drain(batch, uids, n)
    for uid, prompt in zip(uids, prompts):
        assert out[uid][:n] == _greedy(adapter.model, prompt, n), prompt
    return batch


@pytest.mark.parametrize("width", [2, 3, 4])
@pytest.mark.parametrize("drafter", ["embedded_dfly", "prompt_lookup"])
def test_ast_cohort_equals_single_lane_greedy_and_packs_jointly(tmp_path, tiny, drafter, width):
    adapter = _adapter(tmp_path, tiny, {"draft_model": drafter, "num_draft": 3})
    try:
        assert adapter.execution_config(max_lanes=width, prefill_step=4)[
            "segment_aware_cohort_size"] == width
        batch = _run(adapter, width)
        stats = batch.scheduler_stats
        receipt = adapter.diagnostics()["ast"]["executor"]
        # The joint path ran: tree rounds of several lanes went through one
        # packed forward, not one forward per lane.
        assert stats["external_target_tree_cohort_rounds"] > 0
        assert stats["external_target_tree_packed_target_rounds"] > 0
        assert stats["external_target_tree_cohort_max_width"] >= 2
        assert stats["external_target_tree_cohort_limit"] == width
        assert receipt["joint"] is True and receipt["joint_cohort_waves"] > 0
        assert receipt["cohort_dispatches"] > 0
        assert not receipt["qualified"]
    finally:
        adapter.close()


def test_four_lane_cohort_reaches_width_four_with_dfly2(tmp_path, tiny):
    adapter = _adapter(tmp_path, tiny, {"draft_model": "embedded_dfly", "num_draft": 3,
                                        "ast_tile_rows": 16})
    try:
        stats = _run(adapter, 4, n=14).scheduler_stats
        assert stats["external_target_tree_target_width_4_rounds"] > 0
        assert stats["external_target_tree_cohort_max_width"] == 4
        d = adapter.diagnostics()
        assert d["dfly2"]["observed_used"] and d["ast"]["max_width"] == 4
    finally:
        adapter.close()


def test_serial_reference_executor_matches_and_never_packs(tmp_path, tiny):
    adapter = _adapter(tmp_path, tiny, {"draft_model": "embedded_dfly", "num_draft": 3,
                                        "ast_target_executor": "serial"})
    try:
        stats = _run(adapter, 3).scheduler_stats
        receipt = adapter.diagnostics()["ast"]["executor"]
        assert receipt["joint"] is False and receipt["joint_waves"] == 0
        assert receipt["cohort_dispatches"] > 0
        assert stats["external_target_tree_packed_target_rounds"] == 0
        assert adapter.diagnostics()["ast"]["target_executor"] == "serial"
    finally:
        adapter.close()


def test_server_lane_count_is_clamped_to_four(tmp_path, tiny):
    adapter = _adapter(tmp_path, tiny, {"draft_model": "embedded_dfly"})
    try:
        assert adapter.execution_config(max_lanes=16, prefill_step=8)["segment_aware_cohort_size"] == 4
        batch = adapter.create_external_batch(completion_batch_size=16, prefill_step_size=8)
        assert batch.scheduler_stats["external_target_tree_cohort_limit"] == 4
    finally:
        adapter.close()


# ------------------------------------------------------------- PLD graft --

@pytest.mark.parametrize("width", [1, 2])
def test_graft_pld_is_accepted_engages_and_stays_exact(tmp_path, tiny, width):
    # Cost handoff off: on random weights DFly2 accepts little and the
    # controller would park speculation before the first prompt's copy site.
    adapter = _adapter(tmp_path, tiny, {"draft_model": "embedded_dfly", "num_draft": 3,
                                        "ast_graft_prompt_lookup": True,
                                        "cost_per_commit_handoff": False})
    try:
        from mlx2.runtime.drafters.prompt_lookup_draft import PromptLookupDraftModel

        assert isinstance(adapter.draft_model, PromptLookupDraftModel)
        assert adapter.draft_model.requires_target_caches
        _run(adapter, width, n=14)
        d = adapter.diagnostics()
        graft = d["ast"]["graft_pld"]
        assert graft["selected"] and graft["observed_used"] and not graft["qualified"]
        assert graft["stats"]["copy_paths"] > 0
        # The DFly2 head still drafts under the graft.
        assert d["dfly2"]["observed_used"]
    finally:
        adapter.close()


def test_graft_pld_policy_fails_closed():
    from mlx2.adapters.hy_v4_flash import _parse_policy

    with pytest.raises(ValueError, match="onto itself"):
        _parse_policy({"draft_model": "prompt_lookup", "ast_graft_prompt_lookup": True})
    with pytest.raises(ValueError, match="boolean"):
        _parse_policy({"draft_model": "embedded_dfly", "ast_graft_prompt_lookup": "yes"})
    with pytest.raises(ValueError, match="ast_target_executor"):
        _parse_policy({"draft_model": "embedded_dfly", "ast_target_executor": "fast"})
    _, external, kind = _parse_policy({"draft_model": "embedded_dfly",
                                       "ast_graft_prompt_lookup": True})
    assert kind == "dfly" and external["ast_graft_prompt_lookup"] is True
    assert external["ast_target_executor"] == "packed"


# ------------------------------------------- executor: exact commit/abort --

def _lanes(model, prompts):
    caches = []
    for prompt in prompts:
        cache = model.make_cache()
        model(mx.array([prompt]), cache)
        caches.append(cache)
    return caches


def _snapshot(caches):
    return [[(c.offset, c.meta_state, [np.array(a) for a in c.state]) for c in lane] for lane in caches]


def _assert_same(before, after):
    for lane_a, lane_b in zip(before, after, strict=True):
        for (oa, ma, sa), (ob, mb, sb) in zip(lane_a, lane_b, strict=True):
            assert oa == ob and ma == mb and len(sa) == len(sb)
            for x, y in zip(sa, sb, strict=True):
                np.testing.assert_array_equal(x, y)


# Three lanes, three tree shapes: a chain, a branching tree and a bush.
TREES = [
    ([31, 32, 33, 34, 35], [-1, 0, 1, 2, 3]),
    ([41, 42, 43, 44, 45], [-1, 0, 1, 0, 3]),
    ([51, 52, 53, 54, 55], [-1, 0, 0, 0, 1]),
]
PATHS = [[0, 1, 2], [0, 3, 4], [0, 2]]


def _forward(executor, model, caches):
    layers = (model.args.num_hidden_layers - 1,)
    return executor.forward_many(model, [t for t, _ in TREES], [p for _, p in TREES], caches,
                                 layers, cached=False, configured_limit=4)


def test_joint_cohort_logits_match_serial_and_commit_is_exact(tiny):
    from mlx2.adapters.hy_v4_flash_ast import Hy4ChainTreeExecutor

    _, _, model, _ = tiny
    prompts = [LONG[:11], LONG[20:37], LONG[45:52]]
    joint, serial = Hy4ChainTreeExecutor.create(), Hy4ChainTreeExecutor.create(joint=False)
    a, b = _lanes(model, prompts), _lanes(model, prompts)
    la, fa, ta = _forward(joint, model, a)
    lb, fb, tb = _forward(serial, model, b)
    # Same math; joined rows only change float32 accumulation order.
    np.testing.assert_allclose(np.array(la), np.array(lb), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.array(fa), np.array(fb), rtol=1e-4, atol=1e-4)
    assert ta.physical_packed and not tb.physical_packed
    # Waves: depth 0 packs all three root chains; depth 1 packs the branches.
    assert joint.receipt()["joint_cohort_waves"] == 2 and joint.receipt()["chain_forwards"] == 2
    assert serial.receipt()["chain_forwards"] == 1 + 2 + 3
    tb.abort()
    # Row logits equal the ordinary decode of each node's root path.
    for lane, (prompt, (tokens, parents)) in enumerate(zip(prompts, TREES)):
        for node in range(len(tokens)):
            path, at = [], node
            while at >= 0:
                path.insert(0, tokens[at])
                at = parents[at]
            ref = model(mx.array([prompt + path]))[0, -1]
            np.testing.assert_allclose(np.array(la[lane, node]), np.array(ref), rtol=1e-4, atol=1e-4)
    # Partial commits (rollback inside a chain) land each lane exactly.
    ta.commit_paths(PATHS)
    for lane, (prompt, path) in enumerate(zip(prompts, PATHS)):
        committed = [TREES[lane][0][n] for n in path]
        assert a[lane][0].offset == len(prompt) + len(committed)
        ref = model.make_cache()
        model(mx.array([prompt + committed]), ref)
        for _ in range(3):
            got = model(mx.array([[7]]), a[lane])
            want = model(mx.array([[7]]), ref)
            np.testing.assert_allclose(np.array(got), np.array(want), rtol=1e-5, atol=1e-5)


def test_joint_cohort_abort_leaves_every_lane_exact(tiny):
    from mlx2.adapters.hy_v4_flash_ast import Hy4ChainTreeExecutor

    _, _, model, _ = tiny
    prompts = [LONG[:11], LONG[20:37], LONG[45:52]]
    caches = _lanes(model, prompts)
    before = _snapshot(caches)
    _, _, transaction = _forward(Hy4ChainTreeExecutor.create(), model, caches)
    transaction.abort()
    assert transaction.closed
    _assert_same(before, _snapshot(caches))
    with pytest.raises(RuntimeError):
        transaction.commit_paths(PATHS)
    # The lent root buffers were written past the fill level only: decoding
    # on from the aborted lanes equals a fresh reference.
    for lane, prompt in zip(caches, prompts):
        ref = model.make_cache()
        model(mx.array([prompt]), ref)
        np.testing.assert_allclose(np.array(model(mx.array([[9, 10]]), lane)),
                                   np.array(model(mx.array([[9, 10]]), ref)), rtol=1e-5, atol=1e-5)


def test_failure_inside_a_joint_wave_leaves_lanes_exact(tiny, monkeypatch):
    from mlx2.adapters import hy_v4_flash_ast as X

    _, _, model, _ = tiny
    prompts = [LONG[:11], LONG[20:37], LONG[45:52]]
    caches = _lanes(model, prompts)
    before = _snapshot(caches)
    original, calls = X._wave_forward, []

    def failing(*args, **kwargs):
        result = original(*args, **kwargs)  # the root wave wrote lent buffers
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("injected")
        return result

    monkeypatch.setattr(X, "_wave_forward", failing)
    with pytest.raises(RuntimeError, match="injected"):
        _forward(X.Hy4ChainTreeExecutor.create(), model, caches)
    _assert_same(before, _snapshot(caches))


def test_plain_anchor_cohort_is_one_joint_ordinary_step(tiny):
    from mlx2.adapters.hy_v4_flash_ast import Hy4ChainTreeExecutor

    _, _, model, _ = tiny
    prompts = [LONG[:5], LONG[8:21], LONG[30:39], LONG[60:62]]
    caches = _lanes(model, prompts)
    executor = Hy4ChainTreeExecutor.create()
    logits, _, transaction = executor.forward_plain_anchors(
        model, mx.array([[11], [12], [13], [14]]), caches, (model.args.num_hidden_layers - 1,))
    assert transaction.plain and transaction.physical_packed
    receipt = executor.receipt()
    assert receipt["chain_forwards"] == 1 and receipt["plain_anchor_joint_steps"] == 1
    # Anchor-only steps are not counted as joint tree verification.
    assert receipt["joint_cohort_waves"] == 0 and receipt["tree_cohort_packed_dispatches"] == 0
    transaction.commit(accepted_lengths=[1] * 4)
    for lane, (prompt, anchor) in enumerate(zip(prompts, [11, 12, 13, 14])):
        ref = model(mx.array([prompt + [anchor]]))[0, -1]
        np.testing.assert_allclose(np.array(logits[lane, 0]), np.array(ref), rtol=1e-4, atol=1e-4)
        assert caches[lane][0].offset == len(prompt) + 1


def test_executor_refuses_unbounded_or_ragged_cohorts(tiny):
    from mlx2.adapters.hy_v4_flash_ast import Hy4ChainTreeExecutor

    _, _, model, _ = tiny
    executor = Hy4ChainTreeExecutor.create()
    layers = (model.args.num_hidden_layers - 1,)
    caches = _lanes(model, [LONG[:4]] * 5)
    with pytest.raises(ValueError, match="bound"):
        executor.forward_many(model, [[1]] * 5, [[-1]] * 5, caches, layers, cached=False,
                              configured_limit=4)
    with pytest.raises(ValueError, match="ragged"):
        executor.forward_many(model, [[1], [1]], [[-1], [-1]], caches[:2], layers,
                              cached=False, configured_limit=4, ragged=True)
    with pytest.raises(ValueError):
        Hy4ChainTreeExecutor(joint="yes")


def test_cohort_refuses_lanes_sharing_a_lent_buffer(tiny):
    from mlx2.adapters.hy_v4_flash_ast import Hy4ChainTreeExecutor

    _, _, model, _ = tiny
    (lane,) = _lanes(model, [LONG[:6]])
    alias = [c.clone(lend=True) for c in lane]  # same append-only buffer objects
    with pytest.raises(ValueError, match="share an append-only buffer"):
        Hy4ChainTreeExecutor.create().forward_many(
            model, [[1], [2]], [[-1], [-1]], [lane, alias], (model.args.num_hidden_layers - 1,),
            cached=False, configured_limit=4)


def test_unequal_lane_widths_run_in_one_wave_without_filler(tiny):
    from mlx2.adapters.hy_v4_flash_ast import Hy4ChainTreeExecutor

    _, _, model, _ = tiny
    prompts = [LONG[:9], LONG[12:30], LONG[40:44]]
    caches = _lanes(model, prompts)
    executor = Hy4ChainTreeExecutor.create()
    assert executor.accepts_unequal_tree_widths is True
    trees = [[21, 22, 23, 24, 25, 26], [31, 32, 33], [41]]
    logits, features, transaction = executor.forward_many(
        model, trees, [list(range(-1, len(t) - 1)) for t in trees], caches,
        (model.args.num_hidden_layers - 1,), cached=False, configured_limit=4)
    assert logits.shape[:2] == (3, 6) and features.shape[:2] == (3, 6)
    assert executor.receipt()["chain_forwards"] == 1
    for lane, (prompt, tree) in enumerate(zip(prompts, trees)):
        ref = model(mx.array([prompt + tree]))[0]
        np.testing.assert_allclose(np.array(logits[lane, : len(tree)]), np.array(ref[-len(tree):]),
                                   rtol=1e-4, atol=1e-4)
    transaction.commit_paths([[0, 1, 2, 3], [0, 1, 2], [0]])
    assert [c[0].offset for c in caches] == [13, 21, 5]


def test_commit_refuses_a_non_ancestry_path_before_any_lane_commits(tiny):
    from mlx2.adapters.hy_v4_flash_ast import Hy4ChainTreeExecutor

    _, _, model, _ = tiny
    prompts = [LONG[:11], LONG[20:37], LONG[45:52]]
    caches = _lanes(model, prompts)
    before = _snapshot(caches)
    _, _, transaction = _forward(Hy4ChainTreeExecutor.create(), model, caches)
    # Lane 0 is valid; lane 1's [0, 1, 3] skips from node 1 to node 3 (child of 0).
    with pytest.raises(ValueError, match="root path"):
        transaction.commit_paths([[0, 1, 2], [0, 1, 3], [0, 2]])
    assert transaction.closed
    _assert_same(before, _snapshot(caches))
    _, _, transaction = _forward(Hy4ChainTreeExecutor.create(), model, caches)
    with pytest.raises(ValueError, match="root path"):
        transaction.commit(accepted_lengths=[1, 1, 3])  # range(3) is not a root path in a bush
    _assert_same(before, _snapshot(caches))


# ------------------------------------------------------- aMoEba overlays (amoeba_mask) on AST / DFly2

def _mask_bundle(root, out, frac=0.4, seed=0):
    from mlx2.adapters.hy_v4_flash import HYV4FlashAdapter
    from mlx2.runtime import amoeba_neuron_mask as M

    identity, sites = HYV4FlashAdapter.amoeba_mask_sites_for(root)
    rng = np.random.default_rng(seed)
    closed = {}
    for s in sites:
        if s.experts is None:
            continue
        # whole 8-neuron blocks (paged under pageout) plus scattered single neurons (zeroed, resident)
        nb = s.experts * (s.width // 8)
        blocks = rng.choice(nb, size=int(frac * nb), replace=False)
        closed[s.layer] = np.concatenate([np.arange(b * 8, b * 8 + 8) for b in blocks] + [rng.choice(s.units, 5)])
    return M.write_bundle(out, checkpoint="tiny", fingerprint=identity["fingerprint"], sites=sites, closed=closed,
                          boxes_off=["lang.cmn", "lang.jpn", "lang.hin"], source={"note": "AST overlay test"})


@pytest.mark.parametrize("mode", ["zero", "pageout"])
@pytest.mark.parametrize("drafter", ["embedded_dfly", "prompt_lookup"])
def test_ast_and_dfly2_commit_masked_ordinary_decode_under_an_overlay(tmp_path, tiny, drafter, mode):
    """With the box gated off (zero mask) or paged out, AST/DFly2 commit exactly the masked model's own ordinary
    greedy decode (the drafter predicts the unmasked model; every draft is verified by the masked target)."""
    from mlx2.adapters import hy_v4_flash as A

    config, W, base, _ = tiny
    (tmp_path / "art").mkdir()
    root = _tiny_artifact(tmp_path / "art", config, W)
    bundle = _mask_bundle(root, tmp_path / "mask")
    spec = str(bundle) if mode == "zero" else {"bundle": str(bundle), "mode": "pageout", "block": 8}
    adapter = A.HYV4FlashAdapter(str(root), execution_policy={"draft_model": drafter, "num_draft": 3,
                                                              "amoeba_mask": spec})
    try:
        _run(adapter, 2)                    # each lane == _greedy(adapter.model, ...): the masked ordinary decode
        assert any(_greedy(adapter.model, p, 12) != _greedy(base, p, 12) for p in PROMPTS[:2])   # the box matters
        r = adapter.diagnostics()["amoeba_mask"]
        assert r["selected"] and r["observed_used"] and r["mode"] == mode
        if mode == "pageout":
            po = r["pageout"]
            assert po["resident_bytes_freed"] > 0 and po["paged_bank_calls"] > 0
            assert "pageout-b8" in adapter.layout
        assert adapter.diagnostics()["ast"]["selected"]
    finally:
        adapter.close()
