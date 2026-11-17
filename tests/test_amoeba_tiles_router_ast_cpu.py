"""Restricted amoeba open sets through the real external-draft generator paths (CPU).

The tiny Qwen3.8 + DFlash2 pair of the AST CPU harnesses, with router tiles installed on the
target before the generator binds (as the adapter does). One lane of each run is restricted: every
target forward the generator runs must bind its rows (an unbound tiled forward raises the seam
check), and every lane's committed tokens must equal greedy decoding of the tiled target under
that lane's own open set (all-open lanes: the untiled model, neuron_order).

Exercised: serial prefill, AST packed tree cohort (``_ast_target_tree_forward_many``) and the
anchor-only cohort, the plain generator rounds (cohort verify, ordinary rounds, batched prefill
when lanes start together). Not exercisable on CPU: the B1 TensorFold tile
(``_ast_target_tree_forward``) and the native flat physical-target launch (Metal kernels only; the
latter refuses restricted sets at admission instead).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

import test_ast_fast_tree_policy_cpu as F
import test_qwen38_dflash2_cpu as Q
from test_amoeba_tiles_spec_cpu import _export

TN = 64


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _dense_source(layers, path):
    import json
    path.mkdir(parents=True)
    tensors = {f"language_model.model.layers.{l}.mlp.{p}.weight": getattr(lay.mlp, p).weight
               for l, lay in enumerate(layers) for p in ("gate_proj", "up_proj", "down_proj")}
    mx.save_safetensors(str(path / "model.safetensors"), tensors)
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in tensors}}))


def _install_tiles(target, tmp, layout):
    from mlx2.runtime.amoeba_tiles import install
    from mlx2.runtime.models.activations import swiglu
    layers = list(target.language_model.model.layers)
    _dense_source(layers, tmp / "src")
    owner = np.random.default_rng(9).integers(0, 4, size=(len(layers), TN))
    core = (owner == 0).astype(np.float32)
    boxes = {f"r{j}": (owner == j).astype(np.float32) for j in (1, 2, 3)}
    _export().write_tiles(tmp / "t", tmp / "src", "fp-q38", "fake", core, [0], boxes, {}, len(layers), TN,
                          lambda *a: None, "t", router={b: [] for b in boxes})
    decl = {"mlp": "mlp", "projections": ("gate_proj", "up_proj", "down_proj"), "activation_fn": nn.silu,
            "product_fn": swiglu}
    return install(target, layers, tmp / "t" / "manifest.json", decl, source_fingerprint="fp-q38",
                   layout=layout, route="external_draft")


@pytest.fixture(params=["neuron_order", "fused"])
def tiled(request, monkeypatch, tmp_path):
    original = Q.tiny_pair
    box = {}

    def pair(seed=7):
        target, draft = original(seed)
        box["tiles"] = _install_tiles(target, tmp_path / f"p{len(box)}", request.param)
        box["target"] = target
        return target, draft
    monkeypatch.setattr(F, "tiny_pair", pair)
    monkeypatch.setattr(Q, "tiny_pair", pair)
    return box


def _greedy(target, prompt, n, uid):
    from mlx2.runtime.amoeba_tiles import amoeba_rows
    tokens = list(prompt)
    for _ in range(n):
        with amoeba_rows(target, [uid]):
            logits = target(mx.array([tokens]))
        tokens.append(int(mx.argmax(logits[0, -1]).item()))
    return tokens[len(prompt):]


def _check_lanes(box, lanes, prompt):
    router = box["tiles"].router
    assert router.receipt()["seam_refusals"] == 0
    for lane in lanes:
        committed = list(lane.history)[len(prompt):] + [int(lane.anchor)]
        assert committed == _greedy(box["target"], prompt, len(committed), lane.uid), lane.uid


def _restrict(router, uid, tiles=("r1",)):
    router.bind_uid(uid, frozenset(tiles))


def test_ast_packed_tree_cohort_with_one_restricted_lane(tiled):
    from test_ast_tree_cohort_flow_cpu import _CohortProvider
    from test_qwen38_dflash2_cpu import PROMPTS
    draft, batch, first = F._batch(maximum=32, num_draft=15, capacity=4)
    router = tiled["tiles"].router
    batch.minimum_draft_proposals = 3
    batch.ast_target_tree_executor_provider = _CohortProvider((), "cpu-source", ast_fast_tree=True)
    ids = batch.insert([PROMPTS[0]] * 3, max_tokens=[32] * 3, sampling_configs=[{"sampling_temp": 0}] * 3)
    lanes = [first, *(batch.lanes[uid] for uid in ids)]
    _restrict(router, lanes[1].uid)
    _restrict(router, lanes[3].uid, ("r2", "r3"))
    for lane in lanes:
        while lane.anchor is None:
            batch._prefill(lane)
        lane.ready.clear()
    batch._ast_tree_ready_sessions = 4
    batch._run_round(lanes)
    assert batch.scheduler_stats["external_ast_target_tree_cohort_launches"] == 1
    _check_lanes(tiled, lanes, PROMPTS[0])
    assert router.receipt()["bound_forwards"] > 0


def test_ast_anchor_only_cohort_with_one_restricted_lane(tiled):
    from test_ast_tree_cohort_flow_cpu import _CohortProvider
    from test_qwen38_dflash2_cpu import PROMPTS
    draft, batch, first = F._batch(maximum=32, num_draft=15, capacity=4)
    router = tiled["tiles"].router
    batch.minimum_draft_proposals = 3
    batch.ast_target_tree_executor_provider = _CohortProvider((), "cpu-source", ast_fast_tree=True)
    ids = batch.insert([PROMPTS[0]] * 3, max_tokens=[32] * 3, sampling_configs=[{"sampling_temp": 0}] * 3)
    lanes = [first, *(batch.lanes[uid] for uid in ids)]
    _restrict(router, lanes[2].uid)
    for lane in lanes:
        while lane.anchor is None:
            batch._prefill(lane)
        lane.ready.clear()
        lane.maximum = lane.generated + 1                     # final-budget rows: anchors only
    batch._ast_tree_ready_sessions = 4
    batch._run_round(lanes)
    assert batch.scheduler_stats.get("external_ast_anchor_target_launches", 0) == 1
    for lane in lanes:
        assert lane.generated >= 1
    assert router.receipt()["seam_refusals"] == 0


def test_plain_generator_drain_with_mixed_open_sets(tiled):
    target, draft = Q.tiny_pair()
    router = tiled["tiles"].router
    batch = Q.generator(target, draft, num_draft=3, completion_batch_size=2)
    prompts = [Q.PROMPTS[0], Q.PROMPTS[1]]
    uids = batch.insert(prompts, max_tokens=[12, 12], sampling_configs=[{"sampling_temp": 0}] * 2)
    _restrict(router, uids[0])
    output, _ = Q.drain(batch)
    assert router.receipt()["seam_refusals"] == 0
    for uid, prompt in zip(uids, prompts):
        assert output[uid] == _greedy(target, prompt, len(output[uid]), uid), uid
    stats = batch.scheduler_stats
    assert stats.get("external_rounds", stats.get("prefill_rounds", 0)) > 0


def test_physical_target_refuses_restricted_open_sets_at_admission(tiled, monkeypatch):
    real = F.ExternalDraftBatchGenerator

    def physical(*args, **kwargs):
        args[0].ast_physical_target_provider = object()
        return real(*args, **kwargs, ast_physical_target=True)
    F._batch()
    tiles = tiled["tiles"]
    assert tiles.router.open_set_refusal is None
    assert tiles.open_request({"tiles": ["r1"]}) == frozenset({"r1"})
    monkeypatch.setattr(F, "ExternalDraftBatchGenerator", physical)
    F._batch()
    tiles = tiled["tiles"]
    with pytest.raises(ValueError, match="ast_physical_target"):
        tiles.open_request({"tiles": ["r1"]})
    assert tiles.open_request(None) is None                    # all open stays admitted


@pytest.mark.parametrize("width", [1, 2], ids=["b1_pipeline", "b2_ordinary_round"])
def test_ordinary_rounds_with_a_restricted_lane(tiled, width):
    target, draft = Q.tiny_pair()
    router = tiled["tiles"].router
    batch = Q.generator(target, draft, num_draft=3, completion_batch_size=width)
    prompts = [Q.PROMPTS[0], Q.PROMPTS[1]][:width]
    uids = batch.insert(prompts, max_tokens=[16] * width, sampling_configs=[{"sampling_temp": 0}] * width)
    _restrict(router, uids[0])
    lanes = [batch.lanes[uid] for uid in uids]
    for lane in lanes:
        while lane.anchor is None:
            batch._prefill(lane)
        lane.ordinary = True
    for _ in range(4):
        batch._run_round(lanes)
    abort = getattr(batch, "_abort_ordinary_inflight", None)
    if abort is not None:
        abort()
    assert router.receipt()["seam_refusals"] == 0
    for lane, prompt in zip(lanes, prompts):
        committed = list(lane.history)[len(prompt):]
        assert committed and committed == _greedy(target, prompt, len(committed), lane.uid), lane.uid
