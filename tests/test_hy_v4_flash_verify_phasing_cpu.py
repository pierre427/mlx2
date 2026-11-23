"""Hy4-flash ``verify_phase_align`` and the B>1 verify instrumentation (CPU).

Tiny random checkpoint with the real tensor names (``test_hy_v4_flash_cpu``).

* A one-wave cohort (every DFly2 chain round) takes the unchanged wave path:
  logits and committed caches are bitwise equal with the key on and off.
* A multi-wave cohort (branching trees) runs in ONE forward with the key on;
  its logits equal the wave path within float32 accumulation tolerance (the
  measured maximum difference is asserted small), every node equals the
  ordinary decode of its root path, and partial commits land exactly.
* End to end: greedy outputs at two to four lanes equal single-lane ordinary
  decode with the key on, on the chain and on a branching topology, and the
  receipts show engagement only where a cohort had more than one wave.
* The engine's decode-group split reasons are counted.
"""

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from test_hy_v4_flash_ast_multilane_cpu import (  # noqa: E402
    PATHS,
    PROMPTS,
    TREES,
    _adapter,
    _assert_same,
    _lanes,
    _run,
    _snapshot,
    tiny,  # noqa: F401 - module fixture
)
from test_hy_v4_flash_cpu import LONG  # noqa: E402

LANE_PROMPTS = [LONG[:11], LONG[20:37], LONG[45:52]]


def _forward(executor, model, caches, trees=TREES):
    layers = (model.args.num_hidden_layers - 1,)
    return executor.forward_many(model, [t for t, _ in trees], [p for _, p in trees], caches,
                                 layers, cached=False, configured_limit=4)


def test_policy_key_is_boolean_needs_packed_and_defaults_off():
    from mlx2.adapters.hy_v4_flash import _parse_policy

    _, external, _ = _parse_policy({"draft_model": "embedded_dfly"})
    assert "verify_phase_align" not in external
    _, external, _ = _parse_policy({"draft_model": "embedded_dfly", "verify_phase_align": True})
    assert external["verify_phase_align"] is True
    with pytest.raises(ValueError, match="boolean"):
        _parse_policy({"draft_model": "embedded_dfly", "verify_phase_align": "yes"})
    with pytest.raises(ValueError, match="packed"):
        _parse_policy({"draft_model": "embedded_dfly", "verify_phase_align": True,
                       "ast_target_executor": "serial"})
    from mlx2.adapters.hy_v4_flash_ast import Hy4ChainTreeExecutor

    with pytest.raises(ValueError):
        Hy4ChainTreeExecutor(joint=False, phase_align=True)
    with pytest.raises(ValueError):
        Hy4ChainTreeExecutor(phase_align=1)
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "grid", Path(__file__).resolve().parents[1] / "scripts" / "build_model_feature_grid.py")
    grid = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grid)
    assert "verify_phase_align" in grid.FEATURES
    from mlx2.adapters.hy_v4_flash import numerics_layout

    # Committed child-path rows come from the re-fed chunk: own APCv2 namespace.
    assert numerics_layout("L") == "L"
    assert numerics_layout("L", verify_phase_align=True) == "L+verify-phase-align-v1"


def test_one_wave_cohort_is_bitwise_unchanged(tiny):
    from mlx2.adapters.hy_v4_flash_ast import Hy4ChainTreeExecutor

    _, _, model, _ = tiny
    chains = [([21, 22, 23, 24], [-1, 0, 1, 2]), ([31, 32], [-1, 0]), ([41, 42, 43], [-1, 0, 1])]
    off, on = Hy4ChainTreeExecutor.create(), Hy4ChainTreeExecutor.create(phase_align=True)
    a, b = _lanes(model, LANE_PROMPTS), _lanes(model, LANE_PROMPTS)
    la, fa, ta = _forward(off, model, a, chains)
    lb, fb, tb = _forward(on, model, b, chains)
    np.testing.assert_array_equal(np.array(la), np.array(lb))
    np.testing.assert_array_equal(np.array(fa), np.array(fb))
    ta.commit_paths([[0, 1], [0, 1], [0]])
    tb.commit_paths([[0, 1], [0, 1], [0]])
    _assert_same(_snapshot(a), _snapshot(b))
    r = on.receipt()
    assert r["phase_align"]["selected"] and not r["phase_align"]["observed_used"]
    assert r["multi_wave_dispatches"] == 0 and r["chain_forwards"] == 1


def test_multi_wave_cohort_runs_in_one_forward_and_matches(tiny):
    from mlx2.adapters.hy_v4_flash_ast import Hy4ChainTreeExecutor

    _, _, model, _ = tiny
    off, on = Hy4ChainTreeExecutor.create(), Hy4ChainTreeExecutor.create(phase_align=True)
    a, b = _lanes(model, LANE_PROMPTS), _lanes(model, LANE_PROMPTS)
    la, fa, ta = _forward(off, model, a)
    lb, fb, tb = _forward(on, model, b)
    diff = float(np.max(np.abs(np.array(la) - np.array(lb))))
    # Same math; the re-fed ancestry rows and the joined rows change only the
    # float32 accumulation order.  Quantified, not bitwise.
    assert diff < 1e-4, diff
    np.testing.assert_allclose(np.array(fa), np.array(fb), rtol=1e-4, atol=1e-4)
    assert off.receipt()["chain_forwards"] == 2           # two depth waves
    r = on.receipt()
    assert r["chain_forwards"] == 1 and r["joint_cohort_waves"] == 1
    p = r["phase_align"]
    assert p["observed_used"] and p["dispatches"] == 1 and p["waves_saved"] == 1
    # Lane 1's child chain [3, 4] re-feeds node 0; lane 2's two child chains
    # re-feed node 0 each: three ancestry rows.
    assert p["prefix_rows"] == 3
    assert r["multi_wave_dispatches"] == 1 and off.receipt()["multi_wave_dispatches"] == 1
    # Every node equals the ordinary decode of its root path.
    for lane, (prompt, (tokens, parents)) in enumerate(zip(LANE_PROMPTS, TREES)):
        for node in range(len(tokens)):
            path, at = [], node
            while at >= 0:
                path.insert(0, tokens[at])
                at = parents[at]
            ref = model(mx.array([prompt + path]))[0, -1]
            np.testing.assert_allclose(np.array(lb[lane, node]), np.array(ref), rtol=1e-4, atol=1e-4)
    # Partial commits (including the child chains) land each lane exactly.
    tb.commit_paths(PATHS)
    ta.abort()
    for lane, (prompt, path) in enumerate(zip(LANE_PROMPTS, PATHS)):
        committed = [TREES[lane][0][n] for n in path]
        assert b[lane][0].offset == len(prompt) + len(committed)
        ref = model.make_cache()
        model(mx.array([prompt + committed]), ref)
        for _ in range(3):
            np.testing.assert_allclose(np.array(model(mx.array([[7]]), b[lane])),
                                       np.array(model(mx.array([[7]]), ref)), rtol=1e-5, atol=1e-5)


def test_phase_aligned_abort_leaves_every_lane_exact(tiny):
    from mlx2.adapters.hy_v4_flash_ast import Hy4ChainTreeExecutor

    _, _, model, _ = tiny
    caches = _lanes(model, LANE_PROMPTS)
    before = _snapshot(caches)
    _, _, transaction = _forward(Hy4ChainTreeExecutor.create(phase_align=True), model, caches)
    transaction.abort()
    _assert_same(before, _snapshot(caches))
    for lane, prompt in zip(caches, LANE_PROMPTS):
        ref = model.make_cache()
        model(mx.array([prompt]), ref)
        np.testing.assert_allclose(np.array(model(mx.array([[9, 10]]), lane)),
                                   np.array(model(mx.array([[9, 10]]), ref)), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("width", [2, 4])
@pytest.mark.parametrize("topology", [None, "3x2"])
def test_engine_outputs_stay_exact_with_phase_align(tmp_path, tiny, width, topology):
    policy = {"draft_model": "embedded_dfly", "num_draft": 3, "verify_phase_align": True,
              "cost_per_commit_handoff": False, "ast_tile_rows": 16}
    if topology is not None:
        policy["ast_topology"] = topology
    adapter = _adapter(tmp_path, tiny, policy)
    try:
        _run(adapter, width, n=12)                 # each lane == its own ordinary greedy decode
        d = adapter.diagnostics()["ast"]
        r = d["executor"]
        assert d["verify_phase_align"] is True and r["phase_align"]["selected"]
        assert "+verify-phase-align-v1" in adapter.layout
        if topology is None:
            # DFly2 chains are one wave: nothing to align, the wave path ran.
            assert r["multi_wave_dispatches"] == 0 and not r["phase_align"]["observed_used"]
        else:
            assert r["phase_align"]["observed_used"] == (r["multi_wave_dispatches"] > 0)
    finally:
        adapter.close()


def test_branching_topology_engages_and_matches_the_wave_path(tmp_path, tiny):
    outs = {}
    for align in (False, True):
        (tmp_path / str(align)).mkdir()
        adapter = _adapter(tmp_path / str(align), tiny, {
            "draft_model": "embedded_dfly", "num_draft": 3, "ast_topology": "3x2",
            "verify_phase_align": align, "cost_per_commit_handoff": False, "ast_tile_rows": 16})
        try:
            batch = adapter.create_external_batch(completion_batch_size=3, prefill_step_size=4)
            uids = batch.insert(PROMPTS[:3], max_tokens=[12] * 3,
                                sampling_configs=[{"sampling_temp": 0}] * 3)
            from test_hy_v4_flash_cpu import _drain

            got = _drain(batch, uids, 12)
            outs[align] = [got[u][:12] for u in uids]
            r = adapter.diagnostics()["ast"]["executor"]
            if align:
                assert r["phase_align"]["observed_used"] and r["phase_align"]["waves_saved"] > 0
            else:
                assert r["multi_wave_dispatches"] > 0 and not r["phase_align"]["observed_used"]
        finally:
            adapter.close()
    assert outs[False] == outs[True]


def test_decode_group_split_reasons_are_counted():
    from mlx2.runtime.external_speculative import _decode_group_split_reasons

    assert _decode_group_split_reasons([(3, False, False)]) == []
    assert _decode_group_split_reasons([(3, False, False), (1, False, False)]) == ["depth"]
    # Ordinary and parked lanes carry count 0 by construction: not a depth split.
    assert sorted(_decode_group_split_reasons(
        [(3, False, False), (0, False, True), (0, True, False)])) == ["ordinary", "parked"]
    assert sorted(_decode_group_split_reasons(
        [(3, False, False), (1, False, False), (0, False, True)])) == ["depth", "parked"]


def test_deferred_child_fork_keeps_the_root_write_in_place():
    """Item: a child fork that aliased the lent buffers before the root's
    write would make that write copy the whole history (MLX cannot reuse a
    buffer another array still references).  The deferred fork costs one
    copy (its own), as a wave-path child does; aliasing first costs two."""
    from mlx2.runtime.models import hy_v4_flash as H

    def run(defer):
        live = H.HY4FlashLayerCache(H.INDEXED, ratio=4)
        live.kv_data = mx.arange(1 * 1280 * 1024, dtype=mx.float32).reshape(1, 1280, 1024)
        live.kv_end = live.offset = 1024
        mx.eval(live.kv_data)
        before = np.array(live.kv_data[:, :1024])
        mx.synchronize()
        mx.clear_cache()
        mx.reset_peak_memory()
        base = mx.get_active_memory()
        root = live.clone(lend=True)
        child = live.clone()
        if defer:
            child.defer_append_only_to(root)
            assert child.kv_data is None            # no reference before the root writes
        r, _ = root.append(mx.full((1, 2, 1024), -1.0))
        c, _ = child.append(mx.full((1, 3, 1024), -2.0))
        mx.eval(r, c)
        peak = mx.get_peak_memory() - base
        np.testing.assert_array_equal(np.array(r[:, :1024]), before)
        np.testing.assert_array_equal(np.array(c[:, :1024]), before)
        assert float(r[0, 1024:].max()) == -1.0 and r.shape[1] == 1026
        assert float(c[0, 1024:].max()) == -2.0 and c.shape[1] == 1027
        return peak, live.kv_data.nbytes

    deferred, size = run(True)
    aliased, _ = run(False)
    assert deferred < 1.5 * size <= aliased, (deferred, aliased, size)


def test_refed_rows_are_padding_for_layer_output_readers(tiny):
    from mlx2.adapters.hy_v4_flash_ast import Hy4ChainTreeExecutor

    _, _, model, _ = tiny
    seen = []
    caches = _lanes(model, LANE_PROMPTS)
    readers = model.layer_output_readers
    readers[1] = lambda layer, streams, ids, prefix: seen.append(np.array(ids))
    try:
        _forward(Hy4ChainTreeExecutor.create(phase_align=True), model, caches)
    finally:
        readers.pop(1)
    (ids,) = seen                                   # one forward
    # Rows: lane 0 root; lane 1 root, child [0]+[3, 4]; lane 2 root, children
    # [0]+[2], [0]+[3].  Each child's re-fed node 0 is padding to readers.
    tokens = [t for t, _ in TREES]
    assert ids.shape[0] == 6
    assert list(ids[2, :3]) == [-1, tokens[1][3], tokens[1][4]]
    assert list(ids[4, :2]) == [-1, tokens[2][2]] and list(ids[5, :2]) == [-1, tokens[2][3]]
    assert list(ids[1, :2]) == tokens[1][:2]        # the root row keeps node 0
    assert getattr(model, "_reader_refed_prefix", None) is None


def test_calculator_gate_match_ignores_masked_refed_rows():
    """A re-fed row at the root row's position with another gate value would
    make calc_gate_at fail closed (gate_ambiguous); masked, the root decides."""
    from types import SimpleNamespace

    from mlx2.runtime.amoeba_det_tiles import DetTileSet

    def stash(child_ids):
        ids = mx.array([[5, 6, 7], child_ids])
        gate = mx.array([[False, True, False], [True, False, False]])
        return SimpleNamespace(out_stash={"t": [{"ids": ids, "gate": gate, "host": None,
                                                 "prefix": [[1, 2], [1, 2]]}]})

    history = [1, 2, 5]
    assert DetTileSet.calc_gate_at(stash([5, 9, 9]), "t", history) == (None, "gate_ambiguous")
    assert DetTileSet.calc_gate_at(stash([-1, 9, 9]), "t", history) == (False, None)


# ------------------------------------------- scripts/profile_hy4_batched_verify.py

def _profiler():
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("profile_hy4_batched_verify",
                                                  root / "scripts/profile_hy4_batched_verify.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_profiler_guards(tmp_path):
    B = _profiler()
    with pytest.raises(SystemExit):
        B.parse(["--model", "x", "--out", str(tmp_path / "o")])  # no --cpu / --i-own-the-gpu
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "keep").write_text("evidence")
    with pytest.raises(SystemExit):
        B.parse(["--model", "x", "--out", str(tmp_path / "full"), "--cpu"])
    with pytest.raises(SystemExit):
        B.parse(["--model", "x", "--out", str(tmp_path / "o"), "--cpu", "--widths", "8"])


@pytest.mark.parametrize("policy", [None, {"draft_model": "embedded_dfly", "vision": False,
                                           "ast_topology": "3x2", "verify_phase_align": True,
                                           "ast_tile_rows": 16, "cost_per_commit_handoff": False}])
def test_profiler_receipt_on_the_tiny_artifact(tmp_path, tiny, policy):
    import json

    from test_hy_v4_flash_cpu import _tiny_artifact

    B = _profiler()
    config, W, _, _ = tiny
    (tmp_path / "art").mkdir()
    path = _tiny_artifact(tmp_path / "art", config, W)
    out = tmp_path / "out"
    args = ["--model", str(path), "--out", str(out), "--cpu", "--widths", "1", "2", "4",
            "--rounds", "8", "--max-tokens", "14", "--replay-rounds", "2", "--replay-reps", "1",
            "--prefill-step", "4"]
    if policy is not None:
        args += ["--policy-json", json.dumps(policy)]
    B.main(args)
    doc = json.loads((out / "receipt.json").read_text())
    assert doc["schema"] == B.SCHEMA and "@" in doc["checkpoint"] and doc["mlx2"]["commit"]
    assert doc["finished"] and set(doc["widths"]) == {"1", "2", "4"}
    for width, cell in doc["widths"].items():
        s = cell["summary"]
        assert cell["plain"]["rounds"] >= 1 and cell["plain"]["tok_s"] > 0
        assert s["calls"] >= 1 and s["wall_ms_median"] > 0 and s["gpu_busy_ms_median"] is not None
        assert cell["instrumented"]["accept_length_per_lane_round"]["mean"] > 0
        calls = cell["instrumented"]["per_call"]
        assert all({"shape", "split_reasons", "gpu_busy_ms", "host_ms", "idle_ms"} <= set(c) for c in calls)
        verify = [d for c in calls for d in c["dispatches"] if d["kind"] == "verify"]
        assert verify and all(f["experts"] for d in verify for f in d["forwards"])
        assert s["experts_union_median"] >= 1
        if width == "1":
            assert cell["replay"] == []
        else:
            assert cell["replay"], width
            r = cell["replay"][0]
            assert len(r["alone_ms"]) == r["lanes"] >= 2 and r["joint_ms"] > 0
            jac = [e["jaccard_mean"] for d in verify if d["lanes"] > 1 for f in d["forwards"]
                   for e in f["experts"]]
            assert jac and all(0 <= j <= 1 for j in jac)
    for cell in doc["widths"].values():
        s = cell["summary"]
        assert s["kernel_assignments_median"] >= 1 and s["kernel_union_median"] >= s["experts_union_median"] - 1e-9
    if policy is not None:
        assert doc["receipts"]["ast"]["executor"]["phase_align"]["observed_used"]
        assert any(c["summary"]["kernel_refed_assignments_total"] > 0 for c in doc["widths"].values())
    else:
        assert all(c["summary"]["kernel_refed_assignments_total"] == 0 for c in doc["widths"].values())
