"""Amoeba neuron-order layout (runtime/amoeba_tiles.py, layout "neuron_order"): router tiles as index
sets over the source MLP's own modules (CPU).

* all tiles open: the tiled model is the untiled model bit for bit (max |dlogit| == 0), q4 and q8
  hosts, a whole tiny Qwen3.8 model;
* a mixed batch: fully-open rows equal the stock MLP's rows exactly; restricted rows equal the mask
  reference (activations of closed tiles' neurons times 0, ascending neuron order) and their own
  single-row run;
* "auto" picks neuron_order only for a router-only full partition of exact slices; explicit
  neuron_order refuses anything else; the tile-group layouts stay and are labelled near_tie.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from test_amoeba_tiles_fused_cpu import L, N, _many_boxes, _swiglu_decl, _x, fx  # noqa: F401
from test_amoeba_tiles_spec_cpu import TD, TN, _export, _np, _write_source


def _write(fx, name, core, boxes, gates=None, router=None, tile_bits=None):
    fx.export.write_tiles(fx.tmp / name, fx.src, "fp-test", "fake", core, [0], boxes, gates or {}, L, N, print,
                          name, bf16_src=fx.bf16 if tile_bits else None, tile_bits=tile_bits,
                          router=router if router is not None else {n: [f"t{j % 2}"] for j, n in enumerate(boxes)})
    return fx.tmp / name / "manifest.json"


def _install(layers, manifest, layout="neuron_order", fingerprint="fp-test", model=None):
    from mlx2.runtime.amoeba_tiles import install
    return install(model, layers, manifest, _swiglu_decl(), source_fingerprint=fingerprint, layout=layout)


def _bind(tiles, rows):
    router = tiles.router
    for i, opened in enumerate(rows):
        if opened is not None:
            router.bind_uid(("u", i), opened)
    router.bind_rows([("u", i) for i in range(len(rows))])
    return router


def _release(router):
    router.clear_rows()
    router.uids.clear()
    router.stats.clear()


def _tiny(bits):
    from mlx2.runtime.models.qwen38_27b import Model, ModelArgs
    text = dict(model_type="qwen3_5", hidden_size=TD, intermediate_size=TN, num_hidden_layers=8,
                num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=96,
                linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8,
                linear_value_head_dim=8, linear_conv_kernel_dim=4, full_attention_interval=4,
                mtp_num_hidden_layers=0, partial_rotary_factor=0.5, rope_parameters=None,
                max_position_embeddings=512)
    mx.random.seed(11)
    model = Model(ModelArgs(model_type="qwen3_5", text_config=text))
    nn.quantize(model, group_size=64, bits=bits,
                class_predicate=lambda p, m: isinstance(m, nn.Linear) and ".mlp." in f".{p}.")
    model.eval()
    mx.eval(model.parameters())
    return model


@pytest.mark.parametrize("bits", [4, 8], ids=["q4", "q8"])
def test_all_open_whole_model_equals_the_untiled_model(tmp_path, bits):
    from mlx2.runtime.amoeba_tiles import install
    tiled, plain = _tiny(bits), _tiny(bits)
    layers = list(tiled.language_model.model.layers)
    _write_source(layers, tmp_path / "src")
    rng = np.random.default_rng(4)
    owner = rng.integers(0, 4, size=(len(layers), TN))           # 0 core, 1..3 router tiles
    core = (owner == 0).astype(np.float32)
    boxes = {f"r{j}": (owner == j).astype(np.float32) for j in (1, 2, 3)}
    _export().write_tiles(tmp_path / "t", tmp_path / "src", "fp-tiny", "fake", core, [0], boxes, {},
                          len(layers), TN, print, "t", router={b: [] for b in boxes})
    tiles = install(tiled, layers, tmp_path / "t" / "manifest.json", _swiglu_decl(),
                    source_fingerprint="fp-tiny", layout="auto")
    assert tiles.layout == "neuron_order" and tiles.identity["all_open_exactness"] == "stock_exact"
    tokens = mx.array([[(7 * i + 3) % 90 for i in range(12)], [(5 * i + 1) % 90 for i in range(12)]])
    y, ref = tiled(tokens).astype(mx.float32), plain(tokens).astype(mx.float32)
    assert float(mx.abs(y - ref).max()) == 0.0 and np.array_equal(np.array(y), np.array(ref))
    # Restricted: an ordinary forward of one restricted request differs (tiles really close).
    router = _bind(tiles, [frozenset({"r1"}), None])
    try:
        y2 = tiled(tokens).astype(mx.float32)
    finally:
        _release(router)
    assert not np.array_equal(np.array(y2[0]), np.array(ref[0]))
    assert np.array_equal(np.array(y2[1]), np.array(ref[1]))       # the all-open row: still stock


def test_mixed_batch_open_rows_stock_restricted_rows_mask_reference(fx):
    from mlx2.runtime.models.activations import swiglu
    core, boxes, _ = _many_boxes(partition=True)
    layers, source = fx.fresh(), fx.fresh()
    tiles = _install(layers, _write(fx, "no", core, boxes))
    sets = [None, frozenset({"b0"}), frozenset(), frozenset({"b1", "b3", "b4"})]
    x = _x(batch=len(sets))
    router = _bind(tiles, sets)
    try:
        batched = [_np(layers[l].mlp(x)) for l in range(1, L)]
    finally:
        _release(router)
    r = tiles.receipt()["router"]
    assert all(r["rows_skipped"][n] == 0 for n in boxes)              # masked, never skipped
    assert r["rows_masked_closed"]["b0"] == 2 * 6 * (L - 1)           # rows 3 and 4 close b0 (6 tokens)
    for l in range(1, L):
        src = source[l].mlp
        a = swiglu(src.gate_proj(x), src.up_proj(x))
        stock = _np(src.down_proj(a))
        for i, opened in enumerate(sets):
            y = batched[l - 1][i]
            if opened is None:
                assert np.array_equal(y, stock[i]), (l, i)
                continue
            keep = core[l] + sum(boxes[n][l] for n in opened)           # ascending neuron order mask
            ref = _np(src.down_proj(a * mx.array(keep).astype(a.dtype)))
            assert np.array_equal(y, ref[i]), (l, i)
            router = _bind(tiles, [opened])
            try:
                alone = _np(layers[l].mlp(x[i:i + 1]))
            finally:
                _release(router)
            assert np.array_equal(y, alone[0]), (l, i)


def test_auto_and_refusals(fx):
    core, boxes, gates = _many_boxes(partition=True)
    hard = _write(fx, "hard", core, boxes, gates, router={})            # hard-gated boxes
    req = _write(fx, "req", core, boxes, tile_bits={"core": None, **{n: None for n in boxes}, "b2": 4})
    holes = dict(boxes, b5=boxes["b5"] * 0)
    part = _write(fx, "part", core, holes)                                # b5's neurons in no tile
    for manifest, match in ((hard, "router"), (req, "exact host slices"), (part, "full partition")):
        with pytest.raises(ValueError, match=match):
            _install(fx.fresh(), manifest)
        tiles = _install(fx.fresh(), manifest, layout="auto")
        assert tiles.layout == "unfused" and tiles.receipt()["neuron_order_refusal"]
    ok = _write(fx, "ok", core, boxes)
    d = ok.parent
    from test_amoeba_tiles_cpu import _tamper

    def edit(t):
        w = np.array(t["L1.b0.gate.weight"]); w[0, 0] ^= 1
        t["L1.b0.gate.weight"] = mx.array(w)
    _tamper(d, edit)
    with pytest.raises(ValueError, match="not the live module's rows"):
        _install(fx.fresh(), ok)


def test_identity_labels_and_projection_modules(fx):
    core, boxes, _ = _many_boxes(partition=True)
    manifest = _write(fx, "id", core, boxes)
    layers = fx.fresh()
    originals = [lay.mlp for lay in layers]
    tiles = _install(layers, manifest)
    fused = _install(fx.fresh(), manifest, layout="fused")
    assert tiles.identity["layout"] == "neuron-order-masked-v1"
    assert tiles.identity["restricted_exactness"] == "masked_reference_exact"
    assert fused.identity["all_open_exactness"].startswith("near_tie")
    assert tiles.identity["manifest_sha256"] == fused.identity["manifest_sha256"]
    assert {k: v for k, v in tiles.identity.items() if k != "layout"} != fused.identity   # APC namespace
    mlp = layers[1].mlp
    assert mlp.gate_proj is originals[1].gate_proj and mlp.down_proj is originals[1].down_proj   # no copy
    assert [p for p, _ in mlp.projections()] == ["gate_proj", "up_proj", "down_proj"]
    from mlx.utils import tree_flatten
    stock = sum(v.nbytes for lay in originals for _, v in tree_flatten(lay.parameters()))
    owners = sum(layers[l].mlp.owner.nbytes for l in range(1, L))
    assert tiles.resident_bytes == stock + owners                         # one copy of the weights
