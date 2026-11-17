"""Fused amoeba tile layout (runtime/amoeba_tiles.py, layout "fused") against the unfused reference (CPU).

Synthetic tiny quantized SwiGLU layers, tiles written by scripts/amoeba27b_export.write_tiles. What
is shown, per path:

* union path (core + exact-slice boxes scattering into the shared down union, overlapping boxes
  included): fused == unfused bit for bit (CPU kernels; on Metal the fused gate/up has a larger N,
  so a kernel whose per-output result depends on N would make this a near tie: the GPU A/B checks);
* own-down groups with one member per format: bit for bit; with several members of one format:
  allclose, because the fused down accumulates every member in one fp32 reduction where unfused
  rounds each member's output to bf16 and then adds;
* the batched all-box gate equals each box's own gate bit for bit, and is row-invariant 1..16;
* the fused layer is row-invariant 1..16 under the exact lane law (CPU stand-in kernel);
* closed boxes add exactly 0; all boxes open over a full partition = the source MLP;
* the prefill tail path runs the tiled MLP on the last row only; receipts carry the layout.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from test_amoeba_tiles_spec_cpu import (  # noqa: F401  (cpu_lane is a fixture)
    DECL,
    Layer,
    Stack,
    _export,
    _np,
    _quantized_mlp,
    _write_source,
    cpu_lane,
)

D, N, L = 128, 512, 3
NBOX = 6


def _swiglu_decl():
    from mlx2.runtime.models.activations import swiglu
    return {**DECL, "product_fn": swiglu, "product_name": "compiled_swiglu"}


def _bf16_source(layers, path):
    path.mkdir()
    dense = {}
    for l, lay in enumerate(layers):
        for p in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(lay.mlp, p)
            dense[f"model.language_model.layers.{l}.mlp.{p}.weight"] = mx.dequantize(
                lin.weight, lin.scales, lin.biases, group_size=64, bits=lin.bits).astype(mx.bfloat16)
    mx.save_safetensors(str(path / "model.safetensors"), dense)
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in dense}}))


def _many_boxes(seed=0, partition=False, overlap=True):
    rng = np.random.default_rng(seed)
    core = (rng.random((L, N)) < 0.4).astype(np.float32)
    core[:, 64:128] = 0.0                                   # a dropped group (outside the union)
    boxes = {}
    if partition:
        core[:, 64:128] = 1.0
        owner = rng.integers(0, NBOX, size=(L, N))
        for j in range(NBOX):
            boxes[f"b{j}"] = ((owner == j) & (core == 0)).astype(np.float32)
    else:
        free = 1 - core
        owner = np.where(rng.random((L, N)) < 0.9, rng.integers(0, NBOX, size=(L, N)), -1)
        for j in range(NBOX):                               # disjoint boxes over part of the free neurons
            m = free * (owner == j)
            m[:, 64:128] = 0.0
            boxes[f"b{j}"] = m.astype(np.float32)
        if overlap:                                         # shared neurons between boxes b0/b1
            boxes["b1"] = np.maximum(boxes["b1"], boxes["b0"] * (rng.random((L, N)) < 0.5))
    gates = {n: (rng.normal(size=(L, D)).astype(np.float32) * 0.5, rng.normal(size=(L,)).astype(np.float32))
             for n in boxes}
    return core, boxes, gates


@pytest.fixture()
def fx(tmp_path):
    layers = [Layer(_quantized_mlp(l + 1, n=N)) for l in range(L)]
    _write_source(layers, tmp_path / "src")
    _bf16_source(layers, tmp_path / "bf16")
    return SimpleNamespace(export=_export(), tmp=tmp_path, src=tmp_path / "src", bf16=tmp_path / "bf16",
                           fresh=lambda: [Layer(_quantized_mlp(l + 1, n=N)) for l in range(L)])


def _write(fx, name, core, boxes, gates, tile_bits=None):
    fx.export.write_tiles(fx.tmp / name, fx.src, "fp-test", "fake", core, [0], boxes, gates, L, N, print, name,
                          bf16_src=fx.bf16 if tile_bits else None, tile_bits=tile_bits)
    return fx.tmp / name / "manifest.json"


def _pair(fx, manifest, decl=None, **kw):
    """(unfused layers, fused layers, fused tile set) from one tile set."""
    from mlx2.runtime.amoeba_tiles import install
    decl = decl or _swiglu_decl()
    un, fu = fx.fresh(), fx.fresh()
    install(None, un, manifest, decl, source_fingerprint="fp-test", **kw)
    tiles = install(None, fu, manifest, decl, source_fingerprint="fp-test", layout="fused", **kw)
    return un, fu, tiles


def _x(rows=6, batch=2, scale=2.0, seed=3):
    mx.random.seed(seed)
    return (mx.random.normal((batch, rows, D)) * scale).astype(mx.bfloat16)


@pytest.mark.parametrize("overlap", [False, True], ids=["disjoint", "overlapping"])
def test_union_path_fused_equals_unfused_exactly(fx, overlap):
    core, boxes, gates = _many_boxes(overlap=overlap)
    un, fu, tiles = _pair(fx, _write(fx, "u", core, boxes, gates))
    x = _x()
    for l in range(1, L):
        assert np.array_equal(_np(fu[l].mlp(x)), _np(un[l].mlp(x)))
    layout = tiles.layout_detail()
    groups = layout["layers"]["1"]["groups"]
    assert [g["kind"] for g in groups] == ["union"]                     # core + every box: one gate/up
    assert len(groups[0]["segments"]) == 1 + NBOX
    assert groups[0]["scatter_layers"] == (2 if overlap else 1)
    assert layout["projection_launches_per_layer_max"] <= 2 + 2 + 1      # gate, up, <=2 scatters, gate reduce


def test_mixed_formats_group_by_format(fx):
    core, boxes, gates = _many_boxes(overlap=False)
    bits = {"core": None, "b0": None, "b1": None, "b2": 4, "b3": 4, "b4": 16, "b5": 8}
    un, fu, tiles = _pair(fx, _write(fx, "mix", core, boxes, gates, tile_bits=bits))
    x = _x()
    for l in range(1, L):
        y_un, y_fu = _np(un[l].mlp(x)), _np(fu[l].mlp(x))
        # Two q4 own-down members share one fused down: one fp32 accumulation, where unfused rounds
        # each member's output to bf16 before adding. Components of |y| ~ 100 cancel, so the error is
        # a few bf16 ulps of the components (bounded relative to the output scale), not bitwise.
        assert np.max(np.abs(y_fu - y_un)) <= 0.03 * np.max(np.abs(y_un))
    kinds = {(g["kind"], g["format"], len(g["segments"])) for g in tiles.layout_detail()["layers"]["1"]["groups"]}
    assert ("union", "q4g64-affine", 3) in kinds
    assert ("own_down", "q4g64-affine", 2) in kinds and ("own_down", "q8g64-affine", 1) in kinds
    assert ("own_down", "bfloat16", 1) in kinds


def test_single_member_own_down_groups_are_exact(fx):
    core, boxes, gates = _many_boxes(overlap=False)
    bits = {"core": None, "b0": None, "b1": None, "b2": 4, "b3": None, "b4": 16, "b5": 8}
    un, fu, _ = _pair(fx, _write(fx, "own1", core, boxes, gates, tile_bits=bits))
    x = _x()
    for l in range(1, L):
        assert np.array_equal(_np(fu[l].mlp(x)), _np(un[l].mlp(x)))


def test_batched_gates_equal_each_box_gate_and_are_row_invariant(fx):
    from mlx2.runtime.amoeba_tiles import gate_open, gate_open_many
    rng = np.random.default_rng(7)
    W = mx.array(rng.normal(size=(NBOX, D)).astype(np.float32))
    b = mx.array(rng.normal(size=(NBOX,)).astype(np.float32) * 0.1)
    x = _x(rows=16, batch=1, scale=3.0)
    many = gate_open_many(x, W, b)
    each = mx.concatenate([gate_open(x, W[j], b[j:j + 1]) for j in range(NBOX)], axis=-1)
    assert np.array_equal(_np(many), _np(each))
    for k in range(1, 17):
        xk = x[:, :k]
        alone = mx.concatenate([gate_open_many(xk[:, i:i + 1], W, b) for i in range(k)], axis=1)
        assert np.array_equal(_np(gate_open_many(xk, W, b)), _np(alone)), k
    assert np.array_equal(_np(gate_open_many(x, W, b, per_row=True)), _np(many))


def test_gate_chunking_keeps_bits(fx, monkeypatch):
    from mlx2.runtime import amoeba_tiles as T
    rng = np.random.default_rng(8)
    W = mx.array(rng.normal(size=(NBOX, D)).astype(np.float32))
    b = mx.array(np.zeros(NBOX, np.float32))
    x = _x(rows=40, batch=1)
    whole = T.gate_open_many(x, W, b)
    monkeypatch.setattr(T, "GATE_CHUNK_ELEMENTS", 7 * NBOX * D)       # 7-row chunks
    assert np.array_equal(_np(T.gate_open_many(x, W, b)), _np(whole))


def test_fused_layer_row_invariant_under_lane_exact_law(fx, cpu_lane):
    core, boxes, gates = _many_boxes()
    _, fu, tiles = _pair(fx, _write(fx, "lane", core, boxes, gates))
    cpu_lane.install(Stack(fu), min_rows=1)
    assert all(cpu_lane.installed_lane(m) is not None for _, m in tiles.projection_modules())
    x = _x(rows=16, batch=1)
    for l in range(1, L):
        mlp = fu[l].mlp
        for k in range(1, 17):
            xk = x[:, :k]
            alone = mx.concatenate([mlp(xk[:, i:i + 1]) for i in range(k)], axis=1)
            assert np.array_equal(_np(mlp(xk)), _np(alone)), (l, k)
    assert cpu_lane.STATS["group_launches"] > 0                          # gate/up stacked into one launch


def test_closed_boxes_add_exactly_zero(fx):
    core, boxes, gates = _many_boxes()
    closed = {n: (w * 0.0, np.full_like(b, -50.0)) for n, (w, b) in gates.items()}
    _, fu, tiles = _pair(fx, _write(fx, "closed", core, boxes, closed))
    _, core_only, _ = _pair(fx, _write(fx, "core_only", core, {}, {}))
    x = _x()
    for l in range(1, L):
        assert np.array_equal(_np(fu[l].mlp(x)), _np(core_only[l].mlp(x)))
    assert all(v == 0.0 for v in tiles.receipt()["counters"]["box_open_rows"].values())


@pytest.mark.parametrize("layout", ["unfused", "fused"])
def test_all_boxes_open_over_a_partition_is_the_source_mlp(fx, layout):
    from mlx2.runtime.amoeba_tiles import install
    from mlx2.runtime.models.activations import swiglu
    core, boxes, gates = _many_boxes(partition=True)
    total = core + sum(boxes.values())
    assert np.all(total == 1)                                            # every neuron exactly once
    opened = {n: (w * 0.0, np.full_like(b, 50.0)) for n, (w, b) in gates.items()}
    manifest = _write(fx, "part", core, boxes, opened)
    layers, source = fx.fresh(), fx.fresh()
    tiles = install(None, layers, manifest, _swiglu_decl(), source_fingerprint="fp-test", layout=layout)
    x = _x()
    for l in range(1, L):
        src = source[l].mlp
        ref = src.down_proj(swiglu(src.gate_proj(x), src.up_proj(x)))
        assert np.array_equal(_np(layers[l].mlp(x)), _np(ref)), l
    assert tiles.identity["product"] == "compiled_swiglu"


def test_prefill_tail_runs_the_tiled_mlp_on_the_last_row(tmp_path):
    """hybrid_layer_tail (prefill tail skip) calls layer.mlp on the last row only."""
    from test_amoeba_tiles_spec_cpu import TD, TN, _masks, _tiny_model

    from mlx2.runtime.amoeba_tiles import install
    from mlx2.runtime.models.base import create_attention_mask, create_ssm_mask
    model = _tiny_model()
    layers = list(model.language_model.model.layers)
    _write_source(layers, tmp_path / "src")
    core, boxes, gates, dense = _masks(len(layers), TN, TD, seed=6)
    _export().write_tiles(tmp_path / "t", tmp_path / "src", "fp", "fake", core, dense, boxes, gates,
                          len(layers), TN, print, "t")
    tiles = install(model, layers, tmp_path / "t" / "manifest.json", _swiglu_decl(), source_fingerprint="fp",
                    layout="fused")
    cache = model.make_cache()
    h = model.language_model.model.embed_tokens(mx.array([[3, 1, 4, 1, 5, 9, 2]]))
    for index in range(3):                                               # GDN layers 1..2 are tiled
        layer = layers[index]
        mask = create_ssm_mask(h, cache[index]) if layer.is_linear else create_attention_mask(h, cache[index])
        if index == 2:
            before = tiles.counters["rows"]
            out = layer.tail(h, mask, cache[index])
            assert out.shape == (1, 1, TD) and tiles.counters["rows"] - before == 1
        else:
            h = layer(h, mask, cache[index])


def test_layout_identity_and_receipt_fields(fx):
    from mlx2.runtime.amoeba_tiles import LAYOUT_IDS, load_manifest, tile_set_identity
    core, boxes, gates = _many_boxes()
    manifest = _write(fx, "rc", core, boxes, gates)
    _, _, tiles = _pair(fx, manifest)
    m = load_manifest(manifest)
    assert tile_set_identity(m, layout="fused") != tile_set_identity(m, layout="unfused")
    r = tiles.receipt()
    assert r["layout"] == LAYOUT_IDS["fused"] and r["fused_layout"]["layout"] == "fused"
    assert set(r["tile_fingerprints"]) == {"core", *boxes}               # per-box identities kept
    assert r["fused_layout"]["formats"] == ["q4g64-affine"]
    with pytest.raises(ValueError, match="layout must be one of"):
        tile_set_identity(m, layout="packed")
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter
    with pytest.raises(ValueError, match="amoeba_tiles_layout must be one of"):
        Qwen3827BAdapter("/nonexistent", execution_policy={"amoeba_tiles": "/x", "amoeba_tiles_layout": "packed"})
    with pytest.raises(ValueError, match="need amoeba_tiles"):
        Qwen3827BAdapter("/nonexistent", execution_policy={"amoeba_tiles_layout": "fused"})


# -- review follow-ups (2026-11-14, second fused review) ----------------------------------------

def _source_with_dtype(layers, path, dtype):
    path.mkdir()
    dense = {}
    for l, lay in enumerate(layers):
        for p in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(lay.mlp, p)
            dense[f"model.language_model.layers.{l}.mlp.{p}.weight"] = mx.dequantize(
                lin.weight, lin.scales, lin.biases, group_size=64, bits=lin.bits).astype(dtype)
    mx.save_safetensors(str(path / "model.safetensors"), dense)
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in dense}}))


def _resign(d, edit_tensors=None, edit_manifest=None):
    from mlx2.runtime.amoeba_tiles import file_sha256
    if edit_tensors is not None:
        t = dict(mx.load(str(d / "tiles.safetensors")))
        edit_tensors(t)
        mx.save_safetensors(str(d / "tiles.safetensors"), t)
    man = json.loads((d / "manifest.json").read_text())
    if edit_manifest is not None:
        edit_manifest(man)
    man["files"]["tiles.safetensors"] = file_sha256(d / "tiles.safetensors")
    (d / "manifest.json").write_text(json.dumps(man))


def _dense_f32(module):
    w = module.weight
    if isinstance(module, nn.QuantizedLinear):
        w = mx.dequantize(module.weight, module.scales, module.biases, group_size=module.group_size,
                          bits=module.bits)
    lin = nn.Linear(int(w.shape[1]), int(w.shape[0]), bias=False)
    lin.weight = w.astype(mx.float32)
    return lin


def _f32_reference(mlp):
    """The unfused layer with every projection dequantized to f32 (the same gates and maps)."""
    for src in mlp.sources:
        src.gate_proj, src.up_proj = _dense_f32(src.gate_proj), _dense_f32(src.up_proj)
        if src.own_down:
            src.down_proj = _dense_f32(src.down_proj)
    if "down_union" in mlp:
        mlp.down_union = _dense_f32(mlp.down_union)
    return mlp


def test_f16_own_down_group_before_a_q4_group_keeps_each_dtype(fx):
    """The gate mask is cast per group dtype: an f16 own-down group (f32 activation) first must not
    promote a later q4 group to f32."""
    core, boxes, gates = _many_boxes(overlap=False)
    boxes = {k: boxes[k] for k in ("b0", "b1")}
    gates = {k: gates[k] for k in ("b0", "b1")}
    f16 = fx.tmp / "f16"
    _source_with_dtype(fx.fresh(), f16, mx.float16)
    fx.export.write_tiles(fx.tmp / "f16t", fx.src, "fp-test", "fake", core, [0], boxes, gates, L, N, print, "f16t",
                          bf16_src=f16, tile_bits={"core": None, "b0": 16, "b1": 4})

    def bf16_scales(t):                                     # the q4 box requantized from f16: match the host
        for key in list(t):
            if ".b1." in key and key.endswith((".scales", ".biases")):
                t[key] = t[key].astype(mx.bfloat16)
    _resign(fx.tmp / "f16t", bf16_scales)
    un, fu, tiles = _pair(fx, fx.tmp / "f16t" / "manifest.json")
    kinds = [(g["kind"], g["format"]) for g in tiles.layout_detail()["layers"]["1"]["groups"]]
    assert kinds == [("union", "q4g64-affine"), ("own_down", "float16"), ("own_down", "q4g64-affine")]
    x = _x()
    for l in range(1, L):
        y_un, y_fu = un[l].mlp(x), fu[l].mlp(x)
        assert y_fu.dtype == y_un.dtype
        assert np.array_equal(_np(y_fu), _np(y_un))


def test_identity_fields_for_fused_gates_padding_and_product(fx):
    from mlx2.runtime.amoeba_tiles import (FUSED_UNION_ROW_MULTIPLE, GATE_ARITHMETIC, GATE_ARITHMETIC_FUSED,
                                           GATE_ARITHMETIC_FUSED_PER_ROW, GATE_CHUNK_ELEMENTS, load_manifest,
                                           tile_set_identity)
    core, boxes, gates = _many_boxes()
    m = load_manifest(_write(fx, "ids", core, boxes, gates))
    fused = tile_set_identity(m, layout="fused", product="compiled_swiglu")
    assert fused["gate_arithmetic"] == GATE_ARITHMETIC_FUSED and fused["gate_chunk_elements"] == GATE_CHUNK_ELEMENTS
    assert fused["padding"]["union_gate_up_rows_multiple"] == FUSED_UNION_ROW_MULTIPLE
    assert fused["padding"]["union_zero_rows_min"] == 1 and "down block" in fused["padding"]["own_down_gate_up_rows"]
    assert tile_set_identity(m, layout="fused", gate_per_row=True)["gate_arithmetic"] == GATE_ARITHMETIC_FUSED_PER_ROW
    unfused = tile_set_identity(m, product="compiled_swiglu")
    assert unfused["gate_arithmetic"] == GATE_ARITHMETIC and "gate_chunk_elements" not in unfused
    # The compiled-swiglu product changes the unfused identity (APCv2 namespace): older state is a
    # different arm and cannot be restored under it.
    assert unfused != tile_set_identity(m) and unfused["product"] == "compiled_swiglu"


def test_union_rows_padded_to_64_with_exact_zero_rows(fx):
    from mlx2.runtime.amoeba_tiles import make_linear
    core, boxes, gates = _many_boxes()
    _, fu, tiles = _pair(fx, _write(fx, "pad64", core, boxes, gates))
    x = _x()
    for l in range(1, L):
        grp = fu[l].mlp.groups[0]
        info = grp.info
        for proj in (grp.gate_proj, grp.up_proj):
            n = int(proj.weight.shape[0])
            assert n % 64 == 0 and n - info["rows"] == info["zero_rows"] >= 1
            y = proj(x)
            assert bool(mx.all(y[..., info["rows"]:] == 0))
            bare = make_linear({"kind": "quantized", "bits": proj.bits, "group_size": proj.group_size,
                                "mode": proj.mode, "weight": proj.weight[:info["rows"]],
                                "scales": proj.scales[:info["rows"]], "biases": proj.biases[:info["rows"]]})
            assert np.array_equal(_np(y[..., :info["rows"]]), _np(bare(x)))
    compact = tiles.receipt()["fused_layout"]
    assert "layers" not in compact and compact["zero_rows_total"] > 0
    assert compact["tiled_layers"] == L - 1 and compact["boxes_per_layer"]["max"] == NBOX


def _overlap_masks(seed=11):
    rng = np.random.default_rng(seed)
    core = (rng.random((L, N)) < 0.4).astype(np.float32)
    owner = np.where(rng.random((L, N)) < 0.8, rng.integers(0, 4, size=(L, N)), -1)
    boxes = {f"b{j}": ((owner == j) & (core == 0)).astype(np.float32) for j in range(4)}
    shared = (rng.random((L, N)) < 0.05)
    for j in range(3):                                          # three-way overlap b0/b1/b2
        boxes[f"b{j}"] = np.maximum(boxes[f"b{j}"], shared)
    boxes["b3"] = np.maximum(boxes["b3"], core * (rng.random((L, N)) < 0.2))   # core-box overlap
    boxes["b3"][1] = 0.0                                        # b3 absent from layer 1
    gates = {n: (rng.normal(size=(L, D)).astype(np.float32) * 0.5, rng.normal(size=(L,)).astype(np.float32))
             for n in boxes}
    return core, boxes, gates


def test_three_way_and_core_box_overlap_and_absent_box(fx):
    core, boxes, gates = _overlap_masks()
    un, fu, tiles = _pair(fx, _write(fx, "ov", core, boxes, gates))
    x = _x()
    for l in range(1, L):
        assert np.array_equal(_np(fu[l].mlp(x)), _np(un[l].mlp(x)))
    detail = tiles.layout_detail()["layers"]
    seg1 = {sid for sid, _ in detail["1"]["groups"][0]["segments"]}
    seg2 = {sid for sid, _ in detail["2"]["groups"][0]["segments"]}
    assert "b3" not in seg1 and "b3" in seg2
    assert detail["1"]["groups"][0]["scatter_layers"] >= 3                 # core + b0/b1/b2 share neurons
    assert detail["2"]["groups"][0]["scatter_layers"] >= 3
    rows = tiles.receipt()["counters"]["box_layer_rows"]
    assert rows["b3"] == x.shape[0] * x.shape[1] and rows["b0"] == 2 * x.shape[0] * x.shape[1]


class _MLP8(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.QuantizedLinear(D, N, bias=False, group_size=64, bits=8)
        self.up_proj = nn.QuantizedLinear(D, N, bias=False, group_size=64, bits=8)
        self.down_proj = nn.QuantizedLinear(N, D, bias=False, group_size=64, bits=8)


def _q8_layers(seed=0):
    layers = []
    for l in range(L):
        mx.random.seed(seed + l + 1)
        m = _MLP8()
        for name, (o, i) in {"gate_proj": (N, D), "up_proj": (N, D), "down_proj": (D, N)}.items():
            lin = getattr(m, name)
            lin.weight, lin.scales, lin.biases = mx.quantize((mx.random.normal((o, i)) * 0.2).astype(mx.bfloat16),
                                                             group_size=64, bits=8)
        layers.append(Layer(m))
    return layers


def test_two_union_groups_of_different_formats(tmp_path):
    """A q8 core and a q4 box that both scatter into the down union: two union groups, one per format."""
    from mlx2.runtime.amoeba_tiles import install
    _write_source(_q8_layers(), tmp_path / "src")
    core, boxes, gates = _many_boxes(overlap=False)
    boxes, gates = {"b0": boxes["b0"], "b1": boxes["b1"]}, {"b0": gates["b0"], "b1": gates["b1"]}
    d = tmp_path / "two"
    _export().write_tiles(d, tmp_path / "src", "fp-test", "fake", core, [0], boxes, gates, L, N, print, "two")

    def requantize_b1(t):
        for l in range(1, L):
            for proj in ("gate", "up"):
                key = f"L{l}.b1.{proj}"
                if f"{key}.weight" not in t:
                    continue
                w = mx.dequantize(t[f"{key}.weight"], t[f"{key}.scales"], t[f"{key}.biases"], group_size=64, bits=8)
                t[f"{key}.weight"], t[f"{key}.scales"], t[f"{key}.biases"] = mx.quantize(
                    w.astype(mx.bfloat16), group_size=64, bits=4)

    def mark(man):
        for b in man["boxes"]:
            if b["id"] == "b1":
                b["bits"] = 4
    _resign(d, requantize_b1, mark)
    un, fu = _q8_layers(), _q8_layers()
    install(None, un, d / "manifest.json", _swiglu_decl(), source_fingerprint="fp-test")
    tiles = install(None, fu, d / "manifest.json", _swiglu_decl(), source_fingerprint="fp-test", layout="fused")
    kinds = [(g["kind"], g["format"]) for g in tiles.layout_detail()["layers"]["1"]["groups"]]
    assert kinds == [("union", "q8g64-affine"), ("union", "q4g64-affine")]
    x = _x()
    for l in range(1, L):
        assert np.array_equal(_np(fu[l].mlp(x)), _np(un[l].mlp(x)))


@pytest.mark.parametrize("layout", ["unfused", "fused"])
def test_partition_with_computed_gates_forced_open_by_bias(fx, layout):
    """Real, non-trivial gate logits (random w), forced open by a large bias: still the source MLP."""
    from mlx2.runtime.amoeba_tiles import install
    from mlx2.runtime.models.activations import swiglu
    core, boxes, gates = _many_boxes(partition=True)
    forced = {n: (w, np.full_like(b, 1000.0)) for n, (w, b) in gates.items()}
    manifest = _write(fx, f"partb_{layout}", core, boxes, forced)
    layers, source = fx.fresh(), fx.fresh()
    tiles = install(None, layers, manifest, _swiglu_decl(), source_fingerprint="fp-test", layout=layout)
    x = _x()
    for l in range(1, L):
        src = source[l].mlp
        assert np.array_equal(_np(layers[l].mlp(x)), _np(src.down_proj(swiglu(src.gate_proj(x), src.up_proj(x)))))
    counters = tiles.receipt()["counters"]
    assert all(counters["box_open_rows"][n] == counters["box_layer_rows"][n] for n in boxes)


def test_mixed_formats_against_an_f32_reference(fx):
    """Two q4 own-down members in one fused down: the fused error against an f32 reference of the
    same tiles is of the same order as the unfused error (both are bf16 roundings of one sum)."""
    core, boxes, gates = _many_boxes(overlap=False)
    bits = {"core": None, "b0": None, "b1": None, "b2": 4, "b3": 4, "b4": 16, "b5": 8}
    manifest = _write(fx, "mix32", core, boxes, gates, tile_bits=bits)
    un, fu, _ = _pair(fx, manifest)
    ref_layers, _, _ = _pair(fx, manifest)
    x = _x()
    for l in range(1, L):
        ref = _np(_f32_reference(ref_layers[l].mlp)(x.astype(mx.float32)))
        d_un, d_fu = np.abs(_np(un[l].mlp(x)) - ref), np.abs(_np(fu[l].mlp(x)) - ref)
        # Both layouts round bf16 activations the same way (that dominates the error against f32);
        # fusing the own-down members must not add error of its own beyond that order.
        assert d_fu.max() <= 1.25 * d_un.max(), (l, d_fu.max(), d_un.max())
        assert d_fu.mean() <= 1.25 * d_un.mean(), (l, d_fu.mean(), d_un.mean())


@pytest.mark.parametrize("layout", ["unfused", "fused"])
def test_gradient_through_tiles_with_counting_suspended(fx, layout):
    """Research value solves differentiate through a tiled MLP: with gate counting suspended the
    gradient is computed and the counters are untouched; with counting on, MLX refuses the
    counters' async_eval inside the transformation (that is why the switch exists)."""
    from mlx2.runtime.amoeba_tiles import gate_counting_suspended
    core, boxes, gates = _many_boxes()
    un, fu, tiles = _pair(fx, _write(fx, "g", core, boxes, gates))
    lay = (fu if layout == "fused" else un)[1]
    x = _x(rows=4, batch=1).astype(mx.float32)
    before = json.dumps(tiles.receipt().get("gate_open_rows", {}), sort_keys=True, default=str)

    def f(v):
        return lay.mlp(x + v).astype(mx.float32).sum()

    with gate_counting_suspended():
        for _ in range(300):                                 # past GATE_FOLD_EVERY
            val, g = mx.value_and_grad(f)(mx.zeros((D,)))
        mx.eval(val, g)
    assert g.shape == (D,) and bool(mx.all(mx.isfinite(g)))
    after = json.dumps(tiles.receipt().get("gate_open_rows", {}), sort_keys=True, default=str)
    assert after == before
