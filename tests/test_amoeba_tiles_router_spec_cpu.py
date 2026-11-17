"""Restricted amoeba open sets on the speculative routes (CPU).

* packed geometry: a TensorFold cohort runs every lane's tree rows as one packed row axis; the
  router binds each lane's run of rows (``amoeba_rows(..., widths)``), so each lane's rows equal
  that lane run alone, in every layout (neuron_order masking; unfused/fused skipping);
* the TensorFold tree verify's MLP step (``lane_multi._mlp_rows``): a neuron-order layer takes the
  tree path's own stock arithmetic (all open: the untiled tree verify), masked when restricted;
* end to end through real ServingEngines: on the external-draft route (DFlash2 drafter, tiled
  target), a restricted request, and a mixed batch of a restricted and an all-open request, give
  the greedy tokens of the ordinary route under the same open sets; every target forward the
  route ran was bound (an unbound tiled forward raises), and all open is the untiled model.

The TensorFold kernels themselves are Metal-only: the packed tree verify on the 27B is the GPU
check's job; here its MLP step and the packed binding are shown on CPU.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from test_amoeba_tiles_fused_cpu import L, _many_boxes, _swiglu_decl, fx  # noqa: F401
from test_amoeba_tiles_spec_cpu import _export, _np

ROOT = Path(__file__).resolve().parents[1]
LAYOUTS = ["neuron_order", "unfused", "fused"]


def _write(fx, name, core, boxes):
    from test_amoeba_tiles_fused_cpu import N
    fx.export.write_tiles(fx.tmp / name, fx.src, "fp-test", "fake", core, [0], boxes, {}, L, N, print, name,
                          router={n: [f"t{j % 2}"] for j, n in enumerate(boxes)})
    return fx.tmp / name / "manifest.json"


def _setup(fx, layout):
    from mlx2.runtime.amoeba_tiles import install
    core, boxes, _ = _many_boxes(partition=True)
    model = nn.Module()
    layers = fx.fresh()
    tiles = install(model, layers, _write(fx, f"p_{layout}", core, boxes), _swiglu_decl(),
                    source_fingerprint="fp-test", layout=layout, route="external_draft")
    return model, layers, tiles, core, boxes


def _x(rows, seed=5):
    from test_amoeba_tiles_fused_cpu import D
    mx.random.seed(seed)
    return (mx.random.normal((1, rows, D)) * 2.0).astype(mx.bfloat16)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_packed_cohort_rows_equal_each_lane_alone(fx, layout):
    from mlx2.runtime.amoeba_tiles import amoeba_rows
    model, layers, tiles, _, _ = _setup(fx, layout)
    router = tiles.router
    router.bind_uid("A", frozenset({"b0", "b2"}))
    router.bind_uid("C", frozenset())
    widths = [3, 5, 2]
    uids = ["A", "B", "C"]                                   # B is all open
    x = _x(sum(widths))
    with amoeba_rows(model, uids, widths):
        packed = [_np(layers[l].mlp(x)) for l in range(1, L)]
    start = 0
    for uid, w in zip(uids, widths):
        with amoeba_rows(model, [uid]):
            alone = [_np(layers[l].mlp(x[:, start:start + w])) for l in range(1, L)]
        for l in range(L - 1):
            assert np.array_equal(packed[l][:, start:start + w], alone[l]), (layout, uid, l)
        start += w
    assert router.state()[-1] is False                       # the scoped seam restored the outer state
    with amoeba_rows(model, uids, [3, 5, 3]), pytest.raises(RuntimeError, match="fail closed"):
        layers[1].mlp(x)                                     # widths disagree with the packed rows
    with pytest.raises(RuntimeError, match="outside the per-row open-set seam"):
        layers[1].mlp(x)                                     # no seam while restricted uids are bound
    assert tiles.receipt()["router"]["last_forward"]["packed"] is True


def test_gathered_partial_rows_refuse_a_packed_forward(fx):
    from mlx2.runtime.amoeba_tiles import amoeba_rows, install
    core, boxes, _ = _many_boxes(partition=True)
    model = nn.Module()
    layers = fx.fresh()
    tiles = install(model, layers, _write(fx, "g", core, boxes), _swiglu_decl(), source_fingerprint="fp-test",
                    layout="fused", route="external_draft", router_partial_rows="gather")
    tiles.router.bind_uid("A", frozenset({"b0"}))
    with amoeba_rows(model, ["A", "B"], [2, 2]), pytest.raises(RuntimeError, match="packed forward"):
        layers[1].mlp(_x(4))


def test_tree_verify_mlp_step_keeps_the_tree_arithmetic(fx):
    from mlx2.runtime.amoeba_tiles import amoeba_rows
    from mlx2.runtime.models.activations import swiglu
    from mlx2.runtime.tensorfold_qwen38.lane_multi import _mlp_rows
    calls = []

    class Fuse:                                              # CPU stand-ins for the Metal kernels
        @staticmethod
        def mlp_gate_up(mlp, x):
            calls.append("gate_up")
            return None

        @staticmethod
        def mlp_act(gu):
            raise AssertionError

    class Glue:
        @staticmethod
        def mlp_act(g, u):
            calls.append("act")
            return swiglu(g, u)
    model, layers, tiles, core, boxes = _setup(fx, "neuron_order")
    source = fx.fresh()
    x = _x(6)
    for l in range(1, L):
        src = source[l].mlp
        stock = _np(src.down_proj(swiglu(src.gate_proj(x), src.up_proj(x))))
        assert np.array_equal(_np(_mlp_rows(layers[l].mlp, x, Fuse, Glue)), stock)   # no restricted request
    assert calls[:2] == ["gate_up", "act"]
    tiles.router.bind_uid("A", frozenset({"b1"}))
    with amoeba_rows(model, ["A", "B"], [4, 2]):
        ys = [_np(_mlp_rows(layers[l].mlp, x, Fuse, Glue)) for l in range(1, L)]
    for l in range(1, L):
        src = source[l].mlp
        a = swiglu(src.gate_proj(x), src.up_proj(x))
        keep = mx.array(core[l] + boxes["b1"][l]).astype(a.dtype)
        ref_a = _np(src.down_proj(a * keep))
        ref_b = _np(src.down_proj(a))
        assert np.array_equal(ys[l - 1][:, :4], ref_a[:, :4]) and np.array_equal(ys[l - 1][:, 4:], ref_b[:, 4:])
    _, f_layers, f_tiles, _, _ = _setup(fx, "fused")
    assert np.array_equal(_np(_mlp_rows(f_layers[1].mlp, x, Fuse, Glue)), _np(f_layers[1].mlp(x)))


# -- end to end: external draft (DFlash2) vs the ordinary route ----------------------------

D2, N2, V2 = 64, 128, 128


def _muse(seed=8):
    from mlx2.adapters.muse_glimmer_config import ModelArgs
    from mlx2.runtime.drafters.dflash2 import DFlash2DraftModel
    from mlx2.runtime.drafters.dflash2_config import DFlash2Config
    from mlx2.runtime.models.muse_glimmer import Model

    mx.random.seed(seed)
    model = Model(ModelArgs(hidden_size=D2, intermediate_size=N2, num_hidden_layers=4, num_attention_heads=2,
                            num_key_value_heads=1, head_dim=16, vocab_size=V2, sliding_window=8,
                            max_position_embeddings=512))
    nn.quantize(model, group_size=64, bits=8,
                class_predicate=lambda p, m: isinstance(m, nn.Linear) and ".mlp." in f".{p}.")
    config = DFlash2Config(hidden_size=D2, intermediate_size=N2, num_hidden_layers=2, num_attention_heads=2,
                           num_key_value_heads=1, head_dim=16, vocab_size=V2, num_target_layers=4,
                           target_layer_ids=[0, 3], conv_kernel_size=2, conv_group_size=2, selector_rank=4,
                           selector_top_k=4, block_size=4, mask_token_id=V2 - 1, max_position_embeddings=512,
                           sliding_window=8, layer_types=["sliding_attention"] * 2)
    draft = DFlash2DraftModel(config).bind(model)
    mx.eval(model.parameters(), draft.parameters())
    return model, draft


def _tile(model, tmp, layout):
    from mlx2.runtime.amoeba_tiles import install
    from test_amoeba_tiles_spec_cpu import _write_source
    layers = list(model.model.layers)
    _write_source(layers, tmp / "src")
    owner = np.random.default_rng(6).integers(0, 4, size=(len(layers), N2))
    core = (owner == 0).astype(np.float32)
    boxes = {f"r{j}": (owner == j).astype(np.float32) for j in (1, 2, 3)}
    _export().write_tiles(tmp / "t", tmp / "src", "fp-muse", "fake", core, [0], boxes, {}, len(layers), N2,
                          print, "t", router={b: [] for b in boxes})
    decl = {"mlp": "mlp", "projections": ("gate_proj", "up_proj", "down_proj"), "activation_fn": nn.silu}
    return install(model, layers, tmp / "t" / "manifest.json", decl, source_fingerprint="fp-muse",
                   layout=layout, route="external_draft")


def _engines(model, draft, tiles):
    import route_harness
    from route_harness import make_engine, make_external_engine
    original = route_harness.make_adapter

    def with_tiles(*args, **kw):
        cls = original(*args, **kw)
        cls.amoeba_tiles = tiles
        return cls
    route_harness.make_adapter = with_tiles
    try:
        tiles.route = "ordinary"
        ordinary = make_engine(model, V2, mtp=False, max_lanes=2)
        yield "ordinary", ordinary
        ordinary.close()
        tiles.route = "external_draft"
        external = make_external_engine(model, draft, V2, num_draft=3, max_lanes=2)
        yield "external", external
        external.close()
    finally:
        route_harness.make_adapter = original


PROMPTS = ([(7 * i + 3) % (V2 - 2) + 1 for i in range(18)], [(5 * i + 11) % (V2 - 2) + 1 for i in range(14)])
OPEN = [{"tiles": ["r1"]}, None]


def _requests(engine, both):
    from route_harness import collect
    jobs = []
    for prompt, opened in zip(PROMPTS if both else PROMPTS[:1], OPEN if both else OPEN[:1]):
        body = {"tokens": prompt, "max_tokens": 10, "temperature": 0}
        if opened is not None:
            body["mlx2_amoeba_open"] = opened
        jobs.append(engine.submit(body))
    out = [collect(job) for job in jobs]
    for result in out:
        assert "error" not in result, result
    return [r["tokens"] for r in out], [r["receipt"] for r in out]


@pytest.mark.parametrize("layout", ["neuron_order", "fused"])
def test_external_draft_restricted_and_mixed_equal_the_ordinary_route(tmp_path, monkeypatch, layout):
    from route_harness import collect, make_engine, patch_host
    patch_host(monkeypatch)
    untiled, _ = _muse()
    model, draft = _muse()
    tiles = _tile(model, tmp_path, layout)
    results = {}
    for name, engine in _engines(model, draft, tiles):
        results[name] = {"single": _requests(engine, False), "mixed": _requests(engine, True)}
    single_o, single_e = results["ordinary"]["single"][0], results["external"]["single"][0]
    mixed_o, mixed_e = results["ordinary"]["mixed"][0], results["external"]["mixed"][0]
    assert single_e == single_o and mixed_e == mixed_o
    assert mixed_e[0] == single_e[0]                         # the restricted request beside an all-open one
    receipt = results["external"]["mixed"][1][0]
    assert receipt["amoeba_open"]["open_tiles"] == ["r1"] and receipt["amoeba_open"]["bound_forwards"] > 0
    assert receipt["speculation"]["kind"] == "external_dflash2"
    assert receipt["speculation"]["external_rounds"] > 0     # draft/verify rounds ran on the tiled target
    r = tiles.receipt()["router"]
    assert r["seam_refusals"] == 0 and r["bound_forwards"] > 0
    if layout == "neuron_order":                             # all open is the untiled model
        plain = make_engine(untiled, V2, mtp=False, max_lanes=2)
        try:
            ref = collect(plain.submit({"tokens": PROMPTS[1], "max_tokens": 10, "temperature": 0}))["tokens"]
        finally:
            plain.close()
        assert mixed_e[1] == ref
    plain = make_engine(untiled, V2, mtp=False, max_lanes=1)
    try:
        full = collect(plain.submit({"tokens": PROMPTS[0], "max_tokens": 10, "temperature": 0}))["tokens"]
    finally:
        plain.close()
    assert single_e != full                                  # the closed tiles really change the output
