"""Hy4-flash lane cache projection, prefill bound and bounded score blocks (CPU).

The projection is checked against the actual cache arrays the tiny
checkpoint allocates, and against the CPU allocator's retained and peak
bytes.  The bounded score blocks are checked for bit-identical outputs and a
peak of one block.  No real weights, no Metal.  See
docs/hy4-flash/context-128k.md.
"""

import gc
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from mlx2.adapters.hy_v4_flash_memory import HY4FlashCacheBudget, _rows  # noqa: E402
from mlx2.runtime.memory_policy import SelfMTPLaneAdmissionController as C  # noqa: E402
from mlx2.runtime.models import hy_v4_flash as H  # noqa: E402
from test_hy_v4_flash_cpu import (  # noqa: E402
    TOKENS, _tiny_artifact, build, checkpoint, tiny_config)
from test_rolling_prefill_checkpoints import host  # noqa: E402,F401 - fixture

GIB = float(1 << 30)


@pytest.fixture(scope="module")
def tiny():
    config = tiny_config()
    W = checkpoint(config)
    model, drafter = build(config, W)
    return config, W, model, drafter


def _prefill(model, tokens, step):
    caches = model.make_cache()
    for s in range(0, len(tokens), step):
        h = model.hidden(mx.array([tokens[s:s + step]]), caches)
        mx.eval(h, [c.state for c in caches])
    return caches


@pytest.mark.parametrize("record", [False, True])
@pytest.mark.parametrize("T,step", [(100, 16), (300, 64), (700, 256), (1000, 1000)])
def test_projection_matches_actual_cache_arrays(tiny, T, step, record):
    _, _, model, drafter = tiny
    previous = model.record_layers
    model.record_layers = frozenset(drafter.index_layers) if record else frozenset()
    try:
        budget = HY4FlashCacheBudget.from_model(model, prefill_rows=step)
        tokens = [int(t) for t in np.random.default_rng(T).integers(4, 90, size=T)]
        gc.collect()
        mx.clear_cache()
        before = mx.get_active_memory()
        mx.reset_peak_memory()
        caches = _prefill(model, tokens, step)
        gc.collect()
        retained = mx.get_active_memory() - before
        peak = mx.get_peak_memory() - before
        parts = budget.project_parts(T)
        # The append-only planes are exact: 256-row capacity of n tokens.
        indexed = [c for c in caches if c.kind == H.INDEXED]
        assert sum(c.kv_data.nbytes + c.tk_data.nbytes + c.gk_data.nbytes for c in indexed) \
            == parts["append_only"]
        for c in indexed:
            assert c.kv_data.shape[1] == _rows(T) and c.gk_data.shape[1] == _rows(T // c.ratio)
        rings = sum(c.ring_data.nbytes for c in caches if c.kind == H.SLIDING)
        assert rings <= parts["sliding_rings"]
        if step >= T or T <= 300:
            assert rings == parts["sliding_rings"]  # one growth: the bound is attained
        if not record:
            assert parts["index_share_selections"] == 0
        # Everything the lane retains (pinned recall arrays, page rounding
        # and small state included) fits the projection, and the chunk's
        # peak fits the projection plus the prefill bound.
        assert retained <= budget.project(T)
        # The chunk's growth copy is charged once, by the decode transient.
        assert peak <= (budget.project(T) + budget.prefill_transient_bytes(T, step)
                        + budget.decode_transient_bytes(T))
        del caches
    finally:
        model.record_layers = previous


def test_fp8_rows_are_projected_at_packed_width(tiny):
    config, W, _, _ = tiny
    fp8, _ = build(config, W)
    fp8.args.latent_kv = "fp8"
    budget = HY4FlashCacheBudget.from_model(fp8, prefill_rows=64)
    caches = _prefill(fp8, [int(t) for t in np.random.default_rng(3).integers(4, 90, size=200)], 64)
    assert budget.indexed_row_bytes == caches[0].kv_data.shape[-1] == 12 + 4 + 2 * 4
    indexed = [c for c in caches if c.kind == H.INDEXED]
    assert sum(c.kv_data.nbytes + c.tk_data.nbytes + c.gk_data.nbytes for c in indexed) \
        == budget.project_parts(200)["append_only"]


def test_real_geometry_per_token_cost_and_kv_rows():
    """The served artifact's geometry: 12 indexed + 30 sliding layers, bf16 rows."""
    a = H.ModelArgs(layer_types=[
        H.INDEXED if i < 2 or (i - 1) % 4 == 0 else H.SLIDING for i in range(42)])
    budget = HY4FlashCacheBudget.from_args(a, item_bytes=2, record_layers=5, verify_rows=8)
    assert (budget.indexed_layers, budget.sliding_layers) == (12, 30)
    assert budget.indexed_row_bytes == 1152 and budget.index_key_bytes == 256
    assert budget.group_key_bytes == 512 and budget.ratio == 4
    # 18 KiB per token on the append-only planes: 2.25 GiB at 128K.
    assert budget.append_only_bytes(131072) == 131072 * 12 * (1152 + 256 + 512 // 4)
    assert budget.append_only_bytes(131072) / GIB == 2.25
    parts = budget.project_parts(131072)
    assert parts["sliding_rings"] == 30 * _rows(575 + 8192) * 1152
    assert parts["index_share_selections"] == 5 * (8192 + 64 * 8) * 2048 * 4
    assert 2.8 < budget.project(131072) / GIB < 2.9
    # A real indexed cache layer at this geometry allocates the projected rows.
    cache = H.HY4FlashLayerCache(H.INDEXED)
    cache.append(mx.zeros((1, 1000, 576), dtype=mx.bfloat16))
    cache.append(mx.zeros((1, 1, 576), dtype=mx.bfloat16))
    assert cache.kv_data.nbytes == _rows(1001) * 1152


def test_projection_replaces_the_envelope_that_refused_32k():
    """Admission arithmetic of the ladder's refused 32K cell at reserve 8."""
    import os

    a = H.ModelArgs(layer_types=[
        H.INDEXED if i < 2 or (i - 1) % 4 == 0 else H.SLIDING for i in range(42)])
    budget = HY4FlashCacheBudget.from_args(a, item_bytes=2, record_layers=5, prefill_rows=2048,
                                           verify_rows=8)
    saved = os.environ.get("MLX2_SERVICE_RESERVE_GIB")
    os.environ["MLX2_SERVICE_RESERVE_GIB"] = "8"
    try:
        envelope = C(host_memory_gib=128.0, advisory_gib=112.0)
        projected = C(host_memory_gib=128.0, advisory_gib=112.0, cache_estimator=budget.project,
                      context_transient_estimator=budget.decode_transient_bytes)
    finally:
        if saved is None:
            os.environ.pop("MLX2_SERVICE_RESERVE_GIB")
        else:
            os.environ["MLX2_SERVICE_RESERVE_GIB"] = saved
    context = 32995 + 24
    headroom = 26671970544 / GIB  # fresh-server headroom_bytes in the ladder receipt
    old = envelope.hard_reserve_gib + envelope.lane_gib(context, 0)
    assert old == pytest.approx(26.77, abs=0.01) and old > headroom
    new = (projected.hard_reserve_gib + projected.lane_gib(context, 0)
           + budget.prefill_transient_bytes(32995, 2048) / GIB)
    assert projected.context_transient_gib(context) > budget.append_only_bytes(context) / GIB
    assert new < headroom - 4


def test_bounded_score_blocks_are_bit_identical_and_hold_one_block(monkeypatch):
    monkeypatch.setattr(H, "SCORE_BUDGET", 1 << 18)
    rng = np.random.default_rng(0)
    L, T, heads = 1024, 1024, 8
    q = mx.array(rng.standard_normal((1, heads, L, 20)).astype(np.float32))
    kv = mx.array(rng.standard_normal((1, 1, T, 20)).astype(np.float32))
    mask = mx.array(rng.random((1, 1, L, T)) > 0.3)
    sinks = mx.array(rng.standard_normal(heads).astype(np.float32))
    mx.eval(q, kv, mask, sinks)
    peaks, outs = {}, {}
    for bounded in (False, True):
        monkeypatch.setattr(H, "BOUNDED_SCORE_BLOCKS", bounded)
        gc.collect()
        mx.clear_cache()
        base = mx.get_active_memory()
        mx.reset_peak_memory()
        out = H.mla_attend(q, kv, mask, sinks, 0.2, 16)
        mx.eval(out)
        peaks[bounded] = mx.get_peak_memory() - base
        outs[bounded] = np.array(out)
    np.testing.assert_array_equal(outs[True], outs[False])
    full = heads * L * T * 4
    assert peaks[False] >= full  # every block live at once
    block = 4 * H.SCORE_BUDGET  # 1 MiB of fp32 scores
    # One block's scores, masked scores and softmax (the budget's 3 x 4 B per
    # element), plus the chunk outputs and their concatenation.
    outputs = 2 * heads * L * 16 * 4
    assert peaks[True] <= 3 * block + outputs + (256 << 10)
    assert peaks[False] > 6 * peaks[True]


def test_bounded_indexer_blocks_are_bit_identical(tiny, monkeypatch):
    _, _, model, _ = tiny
    tokens = [int(t) for t in np.random.default_rng(9).integers(4, 90, size=120)]
    results = {}
    for bounded in (False, True):
        monkeypatch.setattr(H, "BOUNDED_SCORE_BLOCKS", bounded)
        monkeypatch.setattr(H, "SCORE_BUDGET", 64)  # many indexer and attention blocks
        caches = _prefill(model, tokens, 120)
        results[bounded] = (np.array(model.hidden(mx.array([[5]]), caches).astype(mx.float32)),
                            [np.array(c.state[0]) for c in caches])
    np.testing.assert_array_equal(results[True][0], results[False][0])
    for a, b in zip(results[True][1], results[False][1]):
        np.testing.assert_array_equal(a, b)


def test_unbounded_blocks_are_charged_whole(monkeypatch):
    a = H.ModelArgs(layer_types=[
        H.INDEXED if i < 2 or (i - 1) % 4 == 0 else H.SLIDING for i in range(42)])
    budget = HY4FlashCacheBudget.from_args(a, item_bytes=2, record_layers=5)
    bounded = budget.prefill_transient_parts(131072, 2048)
    monkeypatch.setattr(H, "BOUNDED_SCORE_BLOCKS", False)
    whole = budget.prefill_transient_parts(131072, 2048)
    assert bounded["attention_scores"] == 3 * 4 * H.SCORE_BUDGET
    assert whole["attention_scores"] == 3 * 4 * 32 * 2048 * 131072  # 96 GiB: refused


def test_adapter_declares_budget_and_keeps_rolling_default_off(tmp_path, tiny):
    from mlx2.adapters import hy_v4_flash as A

    config, W, _, _ = tiny
    adapter = A.HYV4FlashAdapter(str(_tiny_artifact(tmp_path, config, W)),
                                 execution_policy={"draft_model": "embedded_dfly", "num_draft": 3})
    try:
        budget = adapter.cache_budget(mtp=False)
        assert budget.record_layers == 2  # DFly2 Index Share layers [4, 5]
        # num_draft 3 on the default 8-row tile: seven nodes plus the anchor.
        assert (budget.verify_rows, budget.child_chains, budget.prefill_rows) == (8, 0, 8192)
        assert adapter.cache_budget(mtp=False, prefill_step=1024).prefill_rows == 1024
        assert budget.item_bytes == 4  # the tiny checkpoint runs fp32 activations
        assert budget.project(0) == 0 and budget.project(64) > 0
        json.dumps(budget.as_dict())
        assert vars(A.HYV4FlashAdapter)["apc_rolling_checkpoint_default"] is False
    finally:
        adapter.close()


# --------------------------------------------------- review round two --

REAL_CONFIG = Path("~/mlx-models/Hy4-flash-MLX-4.6bit-mlx2/config.json").expanduser()


def _real_args():
    if not REAL_CONFIG.is_file():
        pytest.skip("the served Hy4-flash artifact's config.json is not on this host")
    return H.ModelArgs.from_hf_config(json.loads(REAL_CONFIG.read_text()))


def test_real_artifact_layer_layout_from_its_config():
    """The served config's layer-type list (JSON only, no weights)."""
    a = _real_args()
    indexed = [i for i, kind in enumerate(a.layer_types) if kind == H.INDEXED]
    assert indexed == [0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41]
    assert len(a.layer_types) - len(indexed) == 30
    cfg = json.loads(REAL_CONFIG.read_text())
    assert set(cfg["dfly_target_index_layer"]) <= set(indexed)
    b = HY4FlashCacheBudget.from_args(a, item_bytes=2, record_layers=len(cfg["dfly_target_index_layer"]),
                                      prefill_rows=1024, verify_rows=8)
    assert b.append_only_bytes(131072) == 131072 * 18432
    assert b.ring_bytes(131072) == 30 * _rows(575 + 1024) * 1152
    # The served chain route at 128K, --prefill-step 1024 (doc section 6).
    assert b.project(131072) / GIB == pytest.approx(2.367, abs=0.002)
    assert b.decode_transient_bytes(131072) / GIB == pytest.approx(2.334, abs=0.002)
    assert b.prefill_transient_bytes(131048, 1024) / GIB == pytest.approx(4.601, abs=0.002)


def test_route_shape_follows_the_policy():
    from mlx2.adapters.hy_v4_flash_memory import route_verify_shape

    assert route_verify_shape({}, draft_selected=False) == (1, 0)
    base = {"draft_model": "embedded_dfly", "ast_topology": "chain", "ast_tile_rows": 8}
    assert route_verify_shape({**base, "num_draft": 5}, draft_selected=True) == (8, 0)
    assert route_verify_shape({**base, "num_draft": 15}, draft_selected=True) == (16, 0)
    assert route_verify_shape({**base, "num_draft": 5, "ast_tiles": 2, "ast_tile_rows": 16},
                              draft_selected=True) == (32, 0)
    assert route_verify_shape({**base, "num_draft": 5, "ast_topology": "3x5", "ast_tile_rows": 16},
                              draft_selected=True) == (16, 2)
    assert route_verify_shape({**base, "num_draft": 5, "ast_topology": "drafter"},
                              draft_selected=True) == (8, 6)
    assert route_verify_shape({**base, "num_draft": 5, "ast_graft_prompt_lookup": True},
                              draft_selected=True) == (8, 1)


def test_decode_growth_and_child_chain_copies_are_charged(tiny):
    """A verify round at a 256-row boundary, and a tree child chain, on real caches."""
    from mlx2.adapters.hy_v4_flash_ast import clone_caches

    _, _, model, _ = tiny
    T, rows = 512, 8  # capacity exactly 512: the next append regrows every plane
    caches = _prefill(model, [int(t) for t in np.random.default_rng(4).integers(4, 90, size=T)], 256)
    budget = HY4FlashCacheBudget.from_model(model, prefill_rows=256, verify_rows=rows)
    tree = HY4FlashCacheBudget.from_model(model, prefill_rows=256, verify_rows=rows, child_chains=1)
    gc.collect()
    mx.clear_cache()
    before = mx.get_active_memory()
    mx.reset_peak_memory()
    root = clone_caches(caches, lend=True)  # the chain executor's root fork
    mx.eval(model.hidden(mx.array([TOKENS[:rows]]), root), [c.state for c in root])
    child = clone_caches(root)  # a child chain aliases its parent (copy on write)
    mx.eval(model.hidden(mx.array([TOKENS[:rows]]), child), [c.state for c in child])
    peak = mx.get_peak_memory() - before
    # The live cache still holds its buffers: the root's regrowth and the
    # child's first append each allocated one full copy of the planes.
    assert peak >= 2 * budget.append_only_bytes(T)
    assert peak > budget.decode_transient_bytes(T)  # a chain budget misses the child
    assert peak <= tree.decode_transient_bytes(T) + budget.project(T + 2 * rows)
    assert tree.decode_transient_parts(T)["child_chain_copies"] >= budget.append_only_bytes(T + rows)


def test_peak_in_the_blocked_score_regime_fits_the_bound(tiny, monkeypatch):
    """Small, but every attention and indexer forward is split into blocks."""
    _, _, model, _ = tiny
    monkeypatch.setattr(H, "SCORE_BUDGET", 1 << 10)
    T, step = 600, 200
    budget = HY4FlashCacheBudget.from_model(model, prefill_rows=step)
    assert budget.prefill_score_syncs(T, step) > budget.indexed_layers  # blocked
    gc.collect()
    mx.clear_cache()
    before = mx.get_active_memory()
    mx.reset_peak_memory()
    caches = _prefill(model, [int(t) for t in np.random.default_rng(6).integers(4, 90, size=T)], step)
    peak = mx.get_peak_memory() - before
    assert peak <= (budget.project(T) + budget.prefill_transient_bytes(T, step)
                    + budget.decode_transient_bytes(T))
    del caches


def test_block_evaluations_match_the_sync_count_and_spare_decode(request):
    """Runs in its own process: it patches ``mlx.core.eval`` module-wide, and
    sharing a process with other Hy4 suites segfaulted inside MLX's CPU eval
    (flaky, order-dependent, 2026-11-21); alone it is stable."""
    import os
    import subprocess
    import sys

    if os.environ.get("MLX2_ISOLATED_TEST") == request.node.nodeid:
        return _block_evaluations_body(request.getfixturevalue("tiny"), request.getfixturevalue("monkeypatch"))
    env = {**os.environ, "MLX2_ISOLATED_TEST": request.node.nodeid}
    done = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", request.node.nodeid],
                          cwd=str(request.config.rootpath), env=env, capture_output=True, text=True, timeout=1800)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-2000:]


def _block_evaluations_body(tiny, monkeypatch):
    _, _, model, _ = tiny
    calls = []
    real_eval = mx.eval
    monkeypatch.setattr(H, "SCORE_BUDGET", 1 << 10)
    budget = HY4FlashCacheBudget.from_model(model, prefill_rows=200)
    caches = _prefill(model, [int(t) for t in np.random.default_rng(7).integers(4, 90, size=400)], 200)
    monkeypatch.setattr(H.mx, "eval", lambda *a: (calls.append(1), real_eval(*a)))
    h = model.hidden(mx.array([[int(t) for t in range(10, 210)]]), caches)
    real_eval(h)
    assert len(calls) == budget.prefill_score_syncs(600, 200)
    # Decode and an 8-row verify at the real SCORE_BUDGET add none.
    monkeypatch.setattr(H, "SCORE_BUDGET", 1 << 27)
    calls.clear()
    for rows in (1, 8):
        real_eval(model.hidden(mx.array([TOKENS[:rows]]), caches))
    assert calls == []


def test_real_geometry_sync_counts_per_chunk():
    a = H.ModelArgs(layer_types=[
        H.INDEXED if i < 2 or (i - 1) % 4 == 0 else H.SLIDING for i in range(42)])
    b = HY4FlashCacheBudget.from_args(a, item_bytes=2, record_layers=5, verify_rows=8)
    # Per prefill chunk at 128K (doc section 4): the sliding layers stay one
    # graph up to a 1024-row chunk.
    assert [b.prefill_score_syncs(131048, L) for L in (1024, 2048, 8192)] == [432, 954, 4506]
    assert b.prefill_score_syncs(32768, 1024) == 96
    for n in (4096, 131072, 135168):
        for rows in (1, 8, 16, 32):
            assert b.prefill_score_syncs(n, rows) == 0


def test_q2_expert_layers_are_costed_per_format(tiny):
    from mlx2.adapters.hy_v4_flash_memory import expert_row_bytes

    affine = expert_row_bytes("affine", routed_hidden=1536, intermediate=1536, item_bytes=2)
    q2 = expert_row_bytes("q2_0", routed_hidden=1536, intermediate=1536, item_bytes=2)
    assert q2 / affine == pytest.approx(2.57, abs=0.01)
    a = H.ModelArgs()
    plain = HY4FlashCacheBudget.from_args(a, item_bytes=2, record_layers=0)
    mixed = HY4FlashCacheBudget.from_args(a, item_bytes=2, record_layers=0,
                                          expert_formats=("affine", "q2_0"))
    assert mixed.prefill_transient_parts(4096, 1024)["routed_experts"] == 1024 * 16 * q2
    assert plain.prefill_transient_parts(4096, 1024)["routed_experts"] == 1024 * 16 * affine
    # A loaded model with one q2_0 projection is detected.
    _, _, model, _ = tiny
    layer = model.model.layers[1].mlp.experts
    saved = layer.gate_proj
    layer.gate_proj = H.Q2ZeroSwitchLinear(64, 16, 8, group_size=64)
    try:
        assert HY4FlashCacheBudget.from_model(model).expert_bytes_per_row == expert_row_bytes(
            "q2_0", routed_hidden=24, intermediate=16, item_bytes=4)
    finally:
        layer.gate_proj = saved


def test_cold_media_prompt_is_charged_per_slice():
    from mlx2.serving import prefill_transient_gib

    b = HY4FlashCacheBudget.from_args(_real_args(), item_bytes=2, record_layers=5, prefill_rows=1024,
                                      verify_rows=8)
    for n in (16384, 131072):
        media = prefill_transient_gib(b, context_tokens=n, uncached_tokens=n, prefill_step=1024,
                                      single_chunk=True)
        whole = b.prefill_transient_bytes(n, n) / GIB
        sliced = (b.prefill_transient_bytes(n, 2048) + 4 * n * 3072 * 2) / GIB
        assert media == pytest.approx(sliced) and media < whole / 3
    assert prefill_transient_gib(b, context_tokens=16384, uncached_tokens=16384,
                                 prefill_step=1024, single_chunk=True) < 6


def test_vision_tower_peak_fits_its_bound_and_blocks_are_bit_identical(monkeypatch):
    from mlx2.adapters.hy_v4_flash_memory import vision_tower_peak_bytes
    from mlx2.runtime.models import hy_v4_flash_vision as V
    from test_hy_v4_flash_vision_cpu import GRIDS, pixels_for, vision_config, vit_checkpoint

    tower = V.load_vision_tower(vision_config(),
                                {k: mx.array(v) for k, v in vit_checkpoint().items()})
    pixels = mx.array(pixels_for(GRIDS))
    mx.eval(tower.parameters(), pixels)
    monkeypatch.setattr(V, "VISION_SCORE_BUDGET", 64)  # many blocks per frame
    outs = {}
    for bounded in (False, True):
        monkeypatch.setattr(V, "BOUNDED_SCORE_BLOCKS", bounded)
        gc.collect()
        mx.clear_cache()
        base = mx.get_active_memory()
        mx.reset_peak_memory()
        out = tower(pixels, GRIDS)
        mx.eval(out)
        peak = mx.get_peak_memory() - base
        outs[bounded] = np.array(out)
        if bounded:
            item = tower.patch_embedding.weight.dtype.size
            assert peak <= vision_tower_peak_bytes(tower.args, GRIDS, item_bytes=item, score_budget=64)
    np.testing.assert_array_equal(outs[True], outs[False])


def test_engine_keeps_the_rolling_default_off_for_an_opted_out_adapter(host):  # noqa: F811
    from test_apc_hits_hybrid_gdn_self_mtp import tiny_qwen38_mtp
    from test_rolling_default_on_cpu import _default_engine, _long_adapter

    from mlx2.serving import ServingEngine

    model, vocab = tiny_qwen38_mtp()
    base = _long_adapter(model, vocab)

    class OptedOut(base):  # the base declares a cache projection
        apc_rolling_checkpoint_default = False

    engine = ServingEngine("tiny", adapter_factory=OptedOut, qualification_mode=True,
                           max_lanes=1, prefill_step=512)
    try:
        assert engine.ready.wait(60), engine.error
        settings = engine.snapshot["settings"]
        assert engine.apc_rolling_route is None
        assert "apc_rolling_checkpoints" not in settings
        assert "keeps APCv2 rolling checkpoints off by default" in \
            settings["skipped_route_defaults"]["apc_rolling_checkpoints"]
    finally:
        engine.close()
    default = _default_engine(model, vocab, mtp=False)
    try:
        assert default.apc_rolling_route == "hybrid"  # the same route without the opt-out
    finally:
        default.close()
