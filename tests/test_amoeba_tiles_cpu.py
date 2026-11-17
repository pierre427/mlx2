"""Amoeba tiled MLPs (runtime/amoeba_tiles.py): exact packing, gates, fail-closed loading, receipts.

A tiny two-layer stack of quantized SwiGLU MLPs (gate/up 4-bit, down 5-bit, group 64) stands in
for the 27B. Tiles are written by scripts/amoeba27b_export.write_tiles from a fake source
checkpoint with the 27B's key layout, then installed and compared with the research reference
``down(m * act)``, ``m = core + sum_b box_b * [sigmoid(w.x + b) > 0.5]``.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
D, N, L = 128, 256, 3


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.QuantizedLinear(D, N, bias=False, group_size=64, bits=4)
        self.up_proj = nn.QuantizedLinear(D, N, bias=False, group_size=64, bits=4)
        self.down_proj = nn.QuantizedLinear(N, D, bias=False, group_size=64, bits=5)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = MLP()


def _quantized_mlp(seed):
    mx.random.seed(seed)
    m = MLP()
    for name, (o, i, bits) in {"gate_proj": (N, D, 4), "up_proj": (N, D, 4), "down_proj": (D, N, 5)}.items():
        w = (mx.random.normal((o, i)) * 0.2).astype(mx.bfloat16)   # bf16 scales, as in the q8 build
        q, s, b = mx.quantize(w, group_size=64, bits=bits)
        lin = getattr(m, name)
        lin.weight, lin.scales, lin.biases = q, s, b
    return m


@pytest.fixture()
def setup(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    export = importlib.import_module("amoeba27b_export")
    layers = []
    for l in range(L):
        lay = Layer()
        lay.mlp = _quantized_mlp(l + 1)
        layers.append(lay)
    src = tmp_path / "src"
    src.mkdir()
    tensors = {}
    for l, lay in enumerate(layers):
        for p in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(lay.mlp, p)
            for k in ("weight", "scales", "biases"):
                tensors[f"language_model.model.layers.{l}.mlp.{p}.{k}"] = getattr(lin, k)
    mx.save_safetensors(str(src / "model.safetensors"), tensors)
    (src / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "model.safetensors" for k in tensors}}))
    rng = np.random.default_rng(0)
    core = (rng.random((L, N)) < 0.7).astype(np.float32)
    core[:, 64:128] = 0.0                                  # one whole group dropped -> excluded from the union
    free = 1 - core
    box_a = free * (rng.random((L, N)) < 0.3)
    box_a[:, 64:128] = 0.0
    box_b = free * (rng.random((L, N)) < 0.3)
    box_b[:, 64:128] = 0.0
    box_b[:, :] = np.maximum(box_b, box_a * (rng.random((L, N)) < 0.2))   # some neurons shared by both boxes
    gates = {"a": (rng.normal(size=(L, D)).astype(np.float32) * 0.5, rng.normal(size=(L,)).astype(np.float32)),
             "b": (rng.normal(size=(L, D)).astype(np.float32) * 0.5, rng.normal(size=(L,)).astype(np.float32))}
    dense = [0]
    return {"export": export, "layers": layers, "src": src, "core": core, "boxes": {"a": box_a, "b": box_b},
            "gates": gates, "dense": dense, "tmp": tmp_path}


def _write(s, name, boxes=True):
    return s["export"].write_tiles(s["tmp"] / name, s["src"], "fp-test", "fake", s["core"], s["dense"],
                                   s["boxes"] if boxes else {}, s["gates"] if boxes else {}, L, N, print, name)


def _reference(s, mlp, l, x, boxes=True):
    act = nn.silu(mlp.gate_proj(x)) * mlp.up_proj(x)
    m = mx.array(s["core"][l])
    if boxes:
        for n, bm in s["boxes"].items():
            W, b = s["gates"][n]
            g = (mx.sigmoid(x.astype(mx.float32) @ mx.array(W[l])[:, None] + float(b[l])) > 0.5).astype(mx.float32)
            m = m + mx.array(bm[l]) * g
    return mlp.down_proj(m.astype(act.dtype) * act)


def _install(s, manifest_dir, fingerprint="fp-test"):
    from mlx2.runtime.amoeba_tiles import install
    decl = {"mlp": "mlp", "projections": ("gate_proj", "up_proj", "down_proj"), "activation_fn": nn.silu}
    return install(None if False else nn.Module(), s["layers"], manifest_dir / "manifest.json", decl,
                   source_fingerprint=fingerprint)


@pytest.mark.parametrize("boxes", [False, True])
def test_tiled_mlp_matches_masked_reference(setup, boxes):
    s = setup
    originals = [lay.mlp for lay in s["layers"]]
    x = mx.random.normal((2, 5, D)) * 2.0
    refs = [_reference(s, originals[l], l, x, boxes) for l in range(L)]
    m = _write(s, "t_box" if boxes else "t_core", boxes)
    assert m["stats"]["down_group_fraction"] == pytest.approx(3 / 4)
    tiles = _install(s, s["tmp"] / ("t_box" if boxes else "t_core"))
    assert s["layers"][0].mlp is originals[0]               # dense stand-in layer untouched
    for l in range(1, L):
        y = s["layers"][l].mlp(x)
        np.testing.assert_allclose(np.array(y), np.array(refs[l]), atol=2e-4, rtol=1e-4)
    r = tiles.receipt()
    assert r["qualified"] is False and r["observed_used"] is True
    assert r["tiles"] == sorted(["core"] + (["a", "b"] if boxes else []))
    assert set(r["tile_fingerprints"]) == set(r["tiles"])


def test_identity_differs_between_tile_sets(setup):
    from mlx2.runtime.amoeba_tiles import load_manifest, tile_set_identity
    s = setup
    _write(s, "c", False)
    _write(s, "cb", True)
    a = tile_set_identity(load_manifest(s["tmp"] / "c"))
    b = tile_set_identity(load_manifest(s["tmp"] / "cb"))
    assert a["manifest_sha256"] != b["manifest_sha256"] and a["tiles"] != b["tiles"]


def test_fail_closed_on_digest_missing_file_and_fingerprint(setup):
    from mlx2.runtime.amoeba_tiles import load_manifest
    s = setup
    _write(s, "f", True)
    d = s["tmp"] / "f"
    with pytest.raises(ValueError, match="source fingerprint"):
        _install(s, d, fingerprint="other")
    with open(d / "tiles.safetensors", "ab") as f:
        f.write(b"x")
    with pytest.raises(ValueError, match="digest mismatch"):
        load_manifest(d)
    (d / "tiles.safetensors").unlink()
    with pytest.raises(ValueError, match="missing"):
        load_manifest(d)
    with pytest.raises(ValueError, match="not found"):
        load_manifest(s["tmp"] / "nope")


def test_fail_closed_on_bad_partition_and_schema(setup):
    from mlx2.runtime.amoeba_tiles import file_sha256, load_manifest
    s = setup
    _write(s, "p", False)
    d = s["tmp"] / "p"
    man = json.loads((d / "manifest.json").read_text())
    man["dense_layers"] = []                                  # layer 0 now neither tiled nor dense
    (d / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(ValueError, match="partition"):
        _install(s, d)
    man["schema"] = "other"
    (d / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(ValueError, match="schema"):
        load_manifest(d)
    assert file_sha256(d / "tiles.safetensors") == man["files"]["tiles.safetensors"]


def test_request_tile_checks(setup):
    from mlx2.runtime.amoeba_tiles import TileNotLoaded, refuse_request_tiles
    s = setup
    _write(s, "r", True)
    tiles = _install(s, s["tmp"] / "r")
    tiles.check_request(None)
    tiles.check_request(["core", "a", "b"])
    with pytest.raises(TileNotLoaded) as e:
        tiles.check_request(["core", "capitals"])
    assert e.value.http_status == 409
    with pytest.raises(TileNotLoaded):
        tiles.check_request(["core"])                         # subsets are not implemented
    with pytest.raises(ValueError):
        tiles.check_request("core")
    refuse_request_tiles(None)
    with pytest.raises(TileNotLoaded):
        refuse_request_tiles(["core"])


def test_adapter_admits_spec_routes_and_refuses_the_rest():
    """Speculative routes are admitted since 2026-11-14 (tests/test_amoeba_tiles_spec_cpu.py has
    the route evidence); tensorfold_prefill and a verify law without a drafter still refuse."""
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter, _refuse_amoeba_combinations
    assert Qwen3827BAdapter.amoeba_mlp["projections"] == ("gate_proj", "up_proj", "down_proj")
    assert _refuse_amoeba_combinations("/x/manifest.json", {}, require_mtp=False, weight_streaming=None) == "ordinary"
    assert _refuse_amoeba_combinations("/x", {}, require_mtp=True, weight_streaming=None) == "native_mtp"
    assert _refuse_amoeba_combinations("/x", {"draft_model": "/d"}, require_mtp=False,
                                       weight_streaming=None) == "external_draft"
    assert _refuse_amoeba_combinations("/x", {"varlen_dense_mlp": True}, require_mtp=False,
                                       weight_streaming=None) == "ordinary"
    with pytest.raises(ValueError, match="tensorfold_prefill"):
        _refuse_amoeba_combinations("/x", {"tensorfold_prefill": True}, require_mtp=False, weight_streaming=None)
    with pytest.raises(ValueError, match="needs a draft_model"):
        _refuse_amoeba_combinations("/x", {"target_verify_row_exact": True}, require_mtp=False,
                                    weight_streaming=None)
    with pytest.raises(ValueError, match="weight streaming"):
        _refuse_amoeba_combinations("/x", {}, require_mtp=False, weight_streaming="dense_layers")
    with pytest.raises(ValueError, match="manifest path"):
        _refuse_amoeba_combinations("", {}, require_mtp=False, weight_streaming=None)


def test_server_accepts_the_request_field():
    import mlx2.server as server
    src = Path(server.__file__).read_text()
    assert '"mlx2_amoeba_tiles",' in src


def test_q4_boxes_with_own_down_block(setup):
    """q4 boxes requantized from a BF16 source carry their own down block; the core is unchanged."""
    s = setup
    bf = s["tmp"] / "bf16"
    bf.mkdir()
    dense_w = {}
    for l, lay in enumerate(s["layers"]):
        for p in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(lay.mlp, p)
            w = mx.dequantize(lin.weight, lin.scales, lin.biases, group_size=64, bits=lin.bits).astype(mx.bfloat16)
            dense_w[f"model.language_model.layers.{l}.mlp.{p}.weight"] = w
    mx.save_safetensors(str(bf / "model.safetensors"), dense_w)
    (bf / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "model.safetensors" for k in dense_w}}))
    originals = [lay.mlp for lay in s["layers"]]
    x = mx.random.normal((2, 5, D)) * 2.0
    m = s["export"].write_tiles(s["tmp"] / "q4", s["src"], "fp-test", "fake", s["core"], s["dense"], s["boxes"],
                                s["gates"], L, N, print, "q4", box_bits=4, bf16_src=bf)
    assert all(b["bits"] == 4 and b["down"] == "own" for b in m["boxes"])
    tiles = _install(s, s["tmp"] / "q4")
    for l in range(1, L):
        ref = _reference(s, originals[l], l, x, True)
        y = s["layers"][l].mlp(x)
        err = float(mx.abs(y - ref).max()) / float(mx.abs(ref).max())
        assert 0 < err < 0.1                                 # requantized at 4 bits: close, not exact
    assert tiles.identity["tile_bits"]["a"] == 4


def _tamper(d, fn):
    """Rewrite a tile file through fn(tensors) and re-sign the manifest (a well-formed but wrong file)."""
    from mlx2.runtime.amoeba_tiles import file_sha256
    t = dict(mx.load(str(d / "tiles.safetensors")))
    fn(t)
    mx.save_safetensors(str(d / "tiles.safetensors"), t)
    man = json.loads((d / "manifest.json").read_text())
    man["files"]["tiles.safetensors"] = file_sha256(d / "tiles.safetensors")
    (d / "manifest.json").write_text(json.dumps(man))


@pytest.mark.parametrize("case,match", [
    ("ungated", "needs a gate"),
    ("short_down", "packed width"),
    ("bad_map", "map disagrees"),
    ("bad_groups", "strictly increasing"),
])
def test_install_refuses_well_signed_but_wrong_tiles(setup, case, match):
    s = setup
    _write(s, case, True)
    d = s["tmp"] / case

    def edit(t):
        if case == "ungated":
            del t["L1.a.gate_w"], t["L1.a.gate_b"]
        elif case == "short_down":
            t["L1.down.weight"] = t["L1.down.weight"][:, :-10]
        elif case == "bad_map":
            m = np.array(t["L1.core.map"]); m[0] = 10 ** 6
            t["L1.core.map"] = mx.array(m)
        elif case == "bad_groups":
            g = np.array(t["L1.down.groups"]); g[[0, 1]] = g[[1, 0]]
            t["L1.down.groups"] = mx.array(g)
    _tamper(d, edit)
    with pytest.raises(ValueError, match=match):
        _install(s, d)


def test_closed_gates_add_exactly_nothing(setup):
    s = setup
    for n in s["gates"]:
        s["gates"][n] = (s["gates"][n][0] * 0.0, np.full_like(s["gates"][n][1], -50.0))
    x = mx.random.normal((2, 5, D))
    _write(s, "closed", True)
    tiles_box = _install(s, s["tmp"] / "closed")
    y_box = [s["layers"][l].mlp(x) for l in range(1, L)]
    assert tiles_box.receipt()["counters"]["box_open_fraction"] == {"a": 0.0, "b": 0.0}
    for lay, orig in zip(s["layers"], [_quantized_mlp(l + 1) for l in range(L)]):
        lay.mlp = orig
    _write(s, "closed_core", False)
    _install(s, s["tmp"] / "closed_core")
    for l in range(1, L):
        assert np.array_equal(np.array(s["layers"][l].mlp(x)), np.array(y_box[l - 1]))


def test_eight_bit_down_union_is_exact(setup):
    s = setup
    for l, lay in enumerate(s["layers"]):
        w = mx.dequantize(lay.mlp.down_proj.weight, lay.mlp.down_proj.scales, lay.mlp.down_proj.biases,
                          group_size=64, bits=5).astype(mx.bfloat16)
        q, sc, b = mx.quantize(w, group_size=64, bits=8)
        lay.mlp.down_proj = nn.QuantizedLinear(N, D, bias=False, group_size=64, bits=8)
        lay.mlp.down_proj.weight, lay.mlp.down_proj.scales, lay.mlp.down_proj.biases = q, sc, b
    tensors = {}
    for l, lay in enumerate(s["layers"]):
        for p in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(lay.mlp, p)
            for k in ("weight", "scales", "biases"):
                tensors[f"language_model.model.layers.{l}.mlp.{p}.{k}"] = getattr(lin, k)
    mx.save_safetensors(str(s["src"] / "model.safetensors"), tensors)
    originals = [lay.mlp for lay in s["layers"]]
    x = mx.random.normal((2, 5, D))
    refs = [_reference(s, originals[l], l, x, True) for l in range(L)]
    _write(s, "q8down", True)
    _install(s, s["tmp"] / "q8down")
    for l in range(1, L):
        np.testing.assert_allclose(np.array(s["layers"][l].mlp(x)), np.array(refs[l]), atol=2e-4, rtol=1e-4)


def test_numerics_contract_and_empty_requests(setup):
    from mlx2.runtime.amoeba_tiles import load_manifest, numerics_contract_with_tiles, refuse_request_tiles
    s = setup
    _write(s, "nc", True)
    tiles = _install(s, s["tmp"] / "nc")
    base = {"schema": "x", "fused_gdn": {"a": 1}}
    joined = numerics_contract_with_tiles(base, tiles)
    assert joined != base and joined["amoeba_tiles"]["manifest_sha256"] == load_manifest(s["tmp"] / "nc")["_digest"]
    assert numerics_contract_with_tiles(base, None) is base
    assert numerics_contract_with_tiles(None, tiles)["amoeba_tiles"]["projection_law"]
    tiles.check_request([])
    refuse_request_tiles([])


def test_manifest_refuses_dotted_tile_ids(setup):
    from mlx2.runtime.amoeba_tiles import file_sha256, load_manifest
    s = setup
    _write(s, "dots", True)
    d = s["tmp"] / "dots"
    man = json.loads((d / "manifest.json").read_text())
    man["boxes"][0]["id"] = "geo.capitals"
    (d / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(ValueError, match="tile ids"):
        load_manifest(d)


def test_adapter_refuses_prompt_lookup_and_mtp_keys():
    from mlx2.adapters.qwen38_27b import _refuse_amoeba_combinations
    for key in ("prompt_lookup", "self_mtp_copy_draft", "mtp_ordinary_handoff", "draft_models"):
        with pytest.raises(ValueError, match=f"refused with {key}"):
            _refuse_amoeba_combinations("/x", {key: {"on": True}}, require_mtp=False, weight_streaming=None)


@pytest.mark.parametrize("with_boxes", [False, True])
def test_q16_core_from_bf16_source(setup, with_boxes):
    """A BF16 (q16) core on a quantized host: core rows and own down columns gathered from BF16 (exact),
    boxes (if any) scatter into a down union of their own groups."""
    s = setup
    bf = s["tmp"] / "bf16_16"
    bf.mkdir()
    dense_w = {}
    for l, lay in enumerate(s["layers"]):
        for p in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(lay.mlp, p)
            dense_w[f"model.language_model.layers.{l}.mlp.{p}.weight"] = mx.dequantize(
                lin.weight, lin.scales, lin.biases, group_size=64, bits=lin.bits).astype(mx.bfloat16)
    mx.save_safetensors(str(bf / "model.safetensors"), dense_w)
    (bf / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "model.safetensors" for k in dense_w}}))
    originals = [lay.mlp for lay in s["layers"]]
    x = mx.random.normal((2, 5, D)).astype(mx.bfloat16)
    refs = []
    for l in range(L):
        W = {p: dense_w[f"model.language_model.layers.{l}.mlp.{p}.weight"] for p in ("gate_proj", "up_proj", "down_proj")}
        act_core = nn.silu(x @ W["gate_proj"].T) * (x @ W["up_proj"].T) * mx.array(s["core"][l]).astype(mx.bfloat16)
        y = act_core @ W["down_proj"].T
        if with_boxes:
            mlp = originals[l]
            act = nn.silu(mlp.gate_proj(x)) * mlp.up_proj(x)
            m = mx.zeros((N,))
            for n, bm in s["boxes"].items():
                Wg, b = s["gates"][n]
                g = (mx.sigmoid(x.astype(mx.float32) @ mx.array(Wg[l])[:, None] + float(b[l])) > 0.5).astype(mx.float32)
                m = m + mx.array(bm[l]) * g
            y = y + mlp.down_proj(m.astype(act.dtype) * act)
        refs.append(y)
    name = "c16b" if with_boxes else "c16"
    m = s["export"].write_tiles(s["tmp"] / name, s["src"], "fp-test", "fake", s["core"], s["dense"],
                                s["boxes"] if with_boxes else {}, s["gates"] if with_boxes else {}, L, N, print, name,
                                bf16_src=bf, core_bits=16)
    assert m["core"]["bits"] == 16 and m["core"]["down"] == "own"
    _install(s, s["tmp"] / name)
    for l in range(1, L):
        y = s["layers"][l].mlp(x)
        np.testing.assert_allclose(np.array(y.astype(mx.float32)), np.array(refs[l].astype(mx.float32)), atol=0.05, rtol=0.02)


def test_per_tile_quant_mix(setup):
    """Each tile at its own precision: core q4 from BF16, box a BF16, box b exact host slice."""
    s = setup
    bf = s["tmp"] / "bf16_mix"
    bf.mkdir()
    dense_w = {}
    for l, lay in enumerate(s["layers"]):
        for p in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(lay.mlp, p)
            dense_w[f"model.language_model.layers.{l}.mlp.{p}.weight"] = mx.dequantize(
                lin.weight, lin.scales, lin.biases, group_size=64, bits=lin.bits).astype(mx.bfloat16)
    mx.save_safetensors(str(bf / "model.safetensors"), dense_w)
    (bf / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "model.safetensors" for k in dense_w}}))
    originals = [lay.mlp for lay in s["layers"]]
    x = mx.random.normal((2, 5, D)).astype(mx.bfloat16)
    refs = [_reference(s, originals[l], l, x, True) for l in range(L)]
    m = s["export"].write_tiles(s["tmp"] / "mix", s["src"], "fp-test", "fake", s["core"], s["dense"], s["boxes"],
                                s["gates"], L, N, print, "mix", bf16_src=bf, tile_bits={"core": 4, "a": 16, "b": None})
    bits = {t["id"]: (t["bits"], t["down"]) for t in [m["core"], *m["boxes"]]}
    assert bits == {"core": (4, "own"), "a": (16, "own"), "b": (None, "union")}
    tiles = _install(s, s["tmp"] / "mix")
    for l in range(1, L):
        y = s["layers"][l].mlp(x).astype(mx.float32)
        r = refs[l].astype(mx.float32)
        assert float(mx.abs(y - r).max()) / float(mx.abs(r).max()) < 0.1
    assert tiles.identity["tile_bits"] == {"core": 4, "a": 16, "b": None}
