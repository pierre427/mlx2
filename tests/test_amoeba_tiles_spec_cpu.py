"""Amoeba tiles on the speculative routes and prefill laws (CPU evidence, nothing qualified).

What is shown here, on tiny synthetic layers:

* the tile projections are ordinary linear child modules, so an installed projection law reaches
  them: the serving lane law (CPU stand-in kernel, as in test_lane_projection_groups) covers every
  tile projection, and under it a row computed in a batch of k equals the row computed alone
  (the lane all-rows/tree scope; the crossover law without it is the falsifier);
* the box gate is row-invariant (f32 row-wise product-sum) and a closed box adds exact zeros;
* the row-exact verify executor, installed after the tiles on a tiny dense Qwen3.8 model, gives
  every verify row the bits of serial one-token tiled decode (logits and taps), while the stock
  wide forward does not;
* the varlen live-row compaction and the slice-invariant prefill swap cover the tiled MLP;
* refusals for every still-unsupported combination, route checks, receipts with route and drafter.

Exactness of the GPU kernels themselves (lane MPP/simd, row_exact_qmv on Metal, the gate
reduction on Metal) is not shown here: scripts/amoeba27b_spec_check.py is the GPU check.
"""

from __future__ import annotations

import copy
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
D, N, L = 128, 256, 3


def _export():
    sys.path.insert(0, str(ROOT / "scripts"))
    return importlib.import_module("amoeba27b_export")


class MLP(nn.Module):
    def __init__(self, d=D, n=N, down_bits=8):
        super().__init__()
        self.gate_proj = nn.QuantizedLinear(d, n, bias=False, group_size=64, bits=4)
        self.up_proj = nn.QuantizedLinear(d, n, bias=False, group_size=64, bits=4)
        self.down_proj = nn.QuantizedLinear(n, d, bias=False, group_size=64, bits=down_bits)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, mlp):
        super().__init__()
        self.mlp = mlp


class Stack(nn.Module):
    """A container whose named_modules() walk reaches every layer (what an installer walks)."""

    def __init__(self, layers):
        super().__init__()
        self.layers = layers


def _quantized_mlp(seed, d=D, n=N, dtype=mx.bfloat16):
    mx.random.seed(seed)
    m = MLP(d, n)
    for name, (o, i) in {"gate_proj": (n, d), "up_proj": (n, d), "down_proj": (d, n)}.items():
        lin = getattr(m, name)
        w = (mx.random.normal((o, i)) * 0.2).astype(dtype)
        lin.weight, lin.scales, lin.biases = mx.quantize(w, group_size=64, bits=lin.bits)
    return m


def _write_source(layers, src, prefix="language_model.model.layers"):
    src.mkdir()
    tensors = {}
    for l, lay in enumerate(layers):
        for p in ("gate_proj", "up_proj", "down_proj"):
            lin = getattr(lay.mlp, p)
            for k in ("weight", "scales", "biases"):
                tensors[f"{prefix}.{l}.mlp.{p}.{k}"] = getattr(lin, k)
    mx.save_safetensors(str(src / "model.safetensors"), tensors)
    (src / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in tensors}}))


def _masks(n_layers, n, d, seed=0, dense=(0,)):
    rng = np.random.default_rng(seed)
    core = (rng.random((n_layers, n)) < 0.6).astype(np.float32)
    core[:, 64:128] = 0.0
    free = 1 - core
    box_a = free * (rng.random((n_layers, n)) < 0.4)
    box_a[:, 64:128] = 0.0
    gates = {"a": (rng.normal(size=(n_layers, d)).astype(np.float32) * 0.5,
                   rng.normal(size=(n_layers,)).astype(np.float32))}
    return core, {"a": box_a}, gates, list(dense)


@pytest.fixture()
def tiles_setup(tmp_path):
    export = _export()
    layers = [Layer(_quantized_mlp(l + 1)) for l in range(L)]
    _write_source(layers, tmp_path / "src")
    core, boxes, gates, dense = _masks(L, N, D)
    originals = [lay.mlp for lay in layers]
    export.write_tiles(tmp_path / "t", tmp_path / "src", "fp-test", "fake", core, dense, boxes, gates,
                       L, N, print, "t")
    return SimpleNamespace(export=export, layers=layers, originals=originals, core=core, boxes=boxes,
                           gates=gates, dense=dense, tmp=tmp_path, manifest=tmp_path / "t" / "manifest.json")


def _np(a):
    """mx -> numpy (bf16 widened to f32, exactly)."""
    return np.array(a.astype(mx.float32) if a.dtype == mx.bfloat16 else a)


DECL = {"mlp": "mlp", "projections": ("gate_proj", "up_proj", "down_proj"), "activation_fn": nn.silu}


def _install(s, route="ordinary", drafter=None, layers=None):
    from mlx2.runtime.amoeba_tiles import install
    return install(None, s.layers if layers is None else layers, s.manifest, DECL,
                   source_fingerprint="fp-test", route=route, drafter=drafter)


# -- structure ------------------------------------------------------------------

def test_tile_projections_are_padded_linear_modules(tiles_setup):
    s = tiles_setup
    tiles = _install(s)
    modules = tiles.projection_modules()
    assert modules and all(isinstance(m, (nn.QuantizedLinear, nn.Linear)) for _, m in modules)
    for path, m in modules:
        if path.endswith(("gate_proj", "up_proj")):
            assert m.weight.shape[0] % 4 == 0, path
    # The live projection set is what the native-flat provider fingerprints.
    mlp = s.layers[1].mlp
    assert [p for p, _ in mlp.projection_topology()] == [p for p, _ in mlp.projections()]
    # Dense stand-in layers keep the source module.
    assert s.layers[0].mlp is s.originals[0]


def test_padding_rows_add_exact_zeros(tiles_setup):
    """Padding gate/up rows to a multiple of 4 leaves every real output column's bits unchanged."""
    from mlx2.runtime.amoeba_tiles import make_linear
    s = tiles_setup
    tiles = _install(s)
    x = (mx.random.normal((1, 6, D)) * 2).astype(mx.bfloat16)
    padded = 0
    for path, m in tiles.projection_modules():
        if not path.endswith(("gate_proj", "up_proj")):
            continue
        src = s.layers[int(path.split(".")[0])].mlp.sources[int(path.split(".")[3])]
        w = mx.dequantize(m.weight, m.scales, m.biases, group_size=m.group_size, bits=m.bits)
        assert bool(mx.all(w[src.rows:] == 0))                    # padded rows dequantize to 0
        bare = make_linear({"kind": "quantized", "bits": m.bits, "group_size": m.group_size, "mode": m.mode,
                            "weight": m.weight[:src.rows], "scales": m.scales[:src.rows],
                            "biases": m.biases[:src.rows]})
        y = m(x)
        assert np.array_equal(_np(y[..., :src.rows]), _np(bare(x)))
        assert bool(mx.all(y[..., src.rows:] == 0))
        padded += int(m.weight.shape[0] != src.rows)
    assert padded                                                  # the fixture exercises padding


def test_gate_is_row_invariant(tiles_setup):
    from mlx2.runtime.amoeba_tiles import gate_open
    w = mx.array(np.random.default_rng(1).normal(size=(D,)).astype(np.float32))
    b = mx.array([0.1], dtype=mx.float32)
    x = (mx.random.normal((1, 16, D)) * 3).astype(mx.bfloat16)
    many = gate_open(x, w, b)
    alone = mx.concatenate([gate_open(x[:, i:i + 1], w, b) for i in range(16)], axis=1)
    assert np.array_equal(_np(many), _np(alone))


# -- lane law -------------------------------------------------------------------

@pytest.fixture()
def cpu_lane(monkeypatch):
    """The lane kernel replaced by a dequantize-and-f32-matmul stand-in (a law distinct from stock)."""
    from mlx2.runtime import lane
    from mlx2.runtime.lane import installer

    monkeypatch.setattr(installer, "available", lambda: True)

    def reference(x, lw):
        if lw.bits == installer.UNQUANTIZED_BITS:
            y = (x.astype(mx.float32) @ lw.weight.T.astype(mx.float32)).astype(x.dtype)
        else:
            w = mx.dequantize(lw.weight, lw.scale_bias[..., 0].T, lw.scale_bias[..., 1].T,
                              group_size=lw.group_size, bits=lw.bits)
            y = (x.astype(mx.float32) @ w.T.astype(mx.float32)).astype(x.dtype)
        return y if lw.bias is None else y + lw.bias

    monkeypatch.setattr(installer, "lane_matmul", reference)
    installer.STATS.clear()
    yield installer
    installer.STATS.clear()
    lane.set_enabled(True)


def _rows_alone(mlp, x):
    return mx.concatenate([mlp(x[:, i:i + 1]) for i in range(x.shape[1])], axis=1)


@pytest.mark.parametrize("k", [2, 4, 9, 16])
def test_lane_exact_law_covers_tiles_and_is_row_invariant(tiles_setup, cpu_lane, k):
    s = tiles_setup
    tiles = _install(s, route="native_mtp")
    stack = Stack(s.layers)
    receipt = cpu_lane.install(stack, min_rows=1)
    assert all(cpu_lane.installed_lane(m) is not None for _, m in tiles.projection_modules())
    assert tiles.lane_law_refusal({**receipt, "law_id": receipt.get("law_id", "lane-matmul-v1")}) is None
    x = (mx.random.normal((1, k, D)) * 2).astype(mx.bfloat16)
    for l in range(1, L):
        mlp = s.layers[l].mlp
        before = cpu_lane.STATS["lane_calls"]
        many = mlp(x)
        assert cpu_lane.STATS["lane_calls"] > before            # the tile projections ran the lane law
        assert np.array_equal(_np(many), _np(_rows_alone(mlp, x)))


def test_lane_law_matches_masked_reference(tiles_setup, cpu_lane):
    """Under the lane law the tiled layer equals the research reference down(m * act) computed with
    the same lane arithmetic on the full source projections (union restriction: allclose)."""
    from mlx2.runtime.amoeba_tiles import gate_open
    s = tiles_setup
    full = Stack([Layer(_quantized_mlp(l + 1)) for l in range(L)])
    cpu_lane.install(full, min_rows=1)
    _install(s)
    cpu_lane.install(Stack(s.layers), min_rows=1)
    x = (mx.random.normal((1, 5, D)) * 2).astype(mx.bfloat16)
    for l in range(1, L):
        ref_mlp = full.layers[l].mlp
        act = nn.silu(ref_mlp.gate_proj(x)) * ref_mlp.up_proj(x)
        W, b = s.gates["a"]
        g = gate_open(x, mx.array(W[l]), mx.array([float(b[l])], dtype=mx.float32)).astype(mx.float32)
        m = mx.array(s.core[l]) + mx.array(s.boxes["a"][l]) * g
        ref = ref_mlp.down_proj(m.astype(act.dtype) * act)
        np.testing.assert_allclose(_np(s.layers[l].mlp(x).astype(mx.float32)),
                                   _np(ref.astype(mx.float32)), atol=3e-2, rtol=2e-2)


def test_crossover_law_needs_the_all_rows_scope(tiles_setup, cpu_lane):
    """Tree position law: under the crossover (min_rows=4) a 4-row call and a 1-row call take
    different arithmetic (the falsifier); inside ``all_rows()`` every row count takes the lane law."""
    from mlx2.runtime.lane import all_rows
    s = tiles_setup
    _install(s, route="external_draft")
    cpu_lane.install(Stack(s.layers), min_rows=4)
    x = (mx.random.normal((1, 8, D)) * 2).astype(mx.bfloat16)
    mlp = s.layers[2].mlp
    assert not np.array_equal(_np(mlp(x)), _np(_rows_alone(mlp, x)))
    with all_rows():
        assert np.array_equal(_np(mlp(x)), _np(_rows_alone(mlp, x)))


def test_lane_law_refusal_rules(tiles_setup, cpu_lane):
    s = tiles_setup
    tiles = _install(s, route="external_draft")
    receipt = cpu_lane.install(Stack(s.layers), min_rows=1, skip=lambda name, m: ".sources.0.up_proj" in name)
    receipt = {**receipt, "law_id": "lane-matmul-v1"}
    refusal = tiles.lane_law_refusal(receipt)
    assert refusal and "does not cover" in refusal and "sources.0.up_proj" in refusal
    assert tiles.receipt()["projection_laws"]["lane"]["tile_projections_uncovered"] >= 1
    # The ordinary route is its own reference: partial coverage is recorded, not refused.
    tiles.route = "ordinary"
    assert tiles.lane_law_refusal(receipt) is None
    tiles.route = "native_mtp"
    assert "TensorFold" in tiles.lane_law_refusal({"law_id": "tensorfold-lane-qmm-v1+fused-groups",
                                                   "covered": {"g": 1}})
    # No lane law at all (lane off, or stepped aside for another projection owner): stock everywhere.
    assert tiles.lane_law_refusal(None) is None
    assert tiles.lane_law_refusal({"law_id": "stock", "covered": {}}) is None


# -- row-exact verify on a tiny dense Qwen3.8 ----------------------------------------

TD, TN = 64, 256
TLAYERS = (1, 6)


def _tiny_model():
    from mlx2.runtime.models.qwen38_27b import Model, ModelArgs

    text = dict(
        model_type="qwen3_5", hidden_size=TD, intermediate_size=TN,
        num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, vocab_size=96, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=4, full_attention_interval=4,
        mtp_num_hidden_layers=0, partial_rotary_factor=0.5,
        rope_parameters=None, max_position_embeddings=512,
    )
    mx.random.seed(11)
    model = Model(ModelArgs(model_type="qwen3_5", text_config=text))
    nn.quantize(model, group_size=64, bits=4,
                class_predicate=lambda p, m: isinstance(m, nn.Linear) and ".mlp." in f".{p}.")
    model.eval()
    mx.eval(model.parameters())
    return model


@pytest.fixture()
def tiny_tiled(tmp_path):
    export = _export()
    model = _tiny_model()
    layers = list(model.language_model.model.layers)
    _write_source(layers, tmp_path / "src")
    core, boxes, gates, dense = _masks(len(layers), TN, TD, seed=3, dense=(0,))
    export.write_tiles(tmp_path / "t", tmp_path / "src", "fp-tiny", "fake", core, dense, boxes, gates,
                       len(layers), TN, print, "t")
    return model, layers, tmp_path / "t" / "manifest.json"


def _install_tiny(model, layers, manifest, route="external_draft"):
    from mlx2.runtime.amoeba_tiles import install
    return install(model, layers, manifest, DECL, source_fingerprint="fp-tiny", route=route)


def _bases(model, lanes):
    bases = []
    for lane in range(lanes):
        cache = model.make_cache()
        mx.eval(model(mx.array([[1 + lane, 2, 3 + 2 * lane, 5, 8 + lane]]), cache=cache))
        bases.append(cache)
    return bases


def _tokens(lengths):
    width = max(lengths)
    return [[(7 * lane + 3 * pos + 11) % 90 for pos in range(width)] for lane in range(len(lengths))]


def _serial(model, cache, tokens):
    logits, taps = [], []
    for token in tokens:
        row_logits, row_taps = model.forward_with_taps(mx.array([[token]]), cache, TLAYERS)
        mx.eval(row_logits, row_taps)
        logits.append(row_logits)
        taps.append(row_taps)
    return mx.concatenate(logits, axis=1), mx.concatenate(taps, axis=1)


@pytest.mark.parametrize("lengths", [[4], [3, 4], [4, 1, 3, 2]], ids=lambda v: "x".join(map(str, v)))
def test_row_exact_verify_rows_equal_serial_tiled_decode(tiny_tiled, lengths):
    from mlx2.runtime.hybrid_verify_rows import HybridVerifyRows
    from mlx2.runtime.models.row_exact_components import RowExactQuantizedLinearMixin

    model, layers, manifest = tiny_tiled
    tiles = _install_tiny(model, layers, manifest)
    route = model.configure_target_verify_row_exact(True)            # after the tiles (adapter order)
    assert tiles.class_law_refusal("target_verify_row_exact",
                                   lambda m: isinstance(m, RowExactQuantizedLinearMixin)) is None
    assert not any(k.startswith("unswapped_projection") for k in route.static_refusals)
    bases = _bases(model, len(lengths))
    tokens = _tokens(lengths)
    owner = HybridVerifyRows([copy.deepcopy(b) for b in bases])
    transaction = owner.begin(lengths)
    result = model.progressive_external_target_rows(mx.array(tokens), transaction.caches, TLAYERS)
    mx.eval(result.logits, result.features)
    transaction.abort()
    assert result.row_exact is True, result.detail
    for lane, valid in enumerate(lengths):
        logits, taps = _serial(model, copy.deepcopy(bases[lane]), tokens[lane][:valid])
        np.testing.assert_array_equal(np.asarray(result.logits[lane:lane + 1, :valid]), np.asarray(logits))
        np.testing.assert_array_equal(np.asarray(result.features[lane:lane + 1, :valid]), np.asarray(taps))
    assert tiles.receipt()["counters"]["tiled_forwards"] > 0


def test_row_exact_falsifier_and_late_install_refusal(tiny_tiled):
    """The stock wide forward over the tiled trunk is not serial decode (so the test above can
    fail), and tiles installed after the row-exact swap are not covered (the adapter refuses)."""
    from mlx2.runtime.hybrid_verify_rows import HybridVerifyRows
    from mlx2.runtime.models.row_exact_components import RowExactQuantizedLinearMixin

    model, layers, manifest = tiny_tiled
    _install_tiny(model, layers, manifest)
    bases = _bases(model, 1)
    tokens = _tokens([4])[0]
    serial_logits, _ = _serial(model, copy.deepcopy(bases[0]), tokens)
    owner = HybridVerifyRows([copy.deepcopy(bases[0])])
    transaction = owner.begin([4])
    wide, _ = model.forward_with_taps(mx.array([tokens]), transaction.caches, TLAYERS)
    mx.eval(wide)
    transaction.abort()
    assert not np.array_equal(np.asarray(wide), np.asarray(serial_logits))

    late = _tiny_model()
    late.configure_target_verify_row_exact(True)
    tiles = _install_tiny(late, list(late.language_model.model.layers), manifest)
    refusal = tiles.class_law_refusal("target_verify_row_exact",
                                      lambda m: isinstance(m, RowExactQuantizedLinearMixin))
    assert refusal and "does not cover" in refusal


# -- prefill laws ----------------------------------------------------------------

def test_varlen_compaction_matches_plain_tiled_rows(tiles_setup):
    from mlx2.runtime.models import varlen_dense_mlp as varlen
    s = tiles_setup
    _install(s)
    holder = SimpleNamespace(prefill_row_context=lambda *a, **k: None)
    handle = varlen.install(holder, varlen.VarlenDenseMLPPolicy(enabled=True))
    x = (mx.random.normal((2, 5, D)) * 2).astype(mx.bfloat16)
    mlp = s.layers[1].mlp
    with varlen._active_scope((3, 5), 5, handle):
        y = mlp(x)
    assert handle["counters"]["mlp_compaction_calls"] == 1
    # The live rows run as one packed [1, 8, D] call (the stock MLP's compaction contract: the
    # packed call, not each lane alone, is the reference), padding rows are skipped and zero-filled.
    packed = mlp(mx.concatenate([x[:1, :3], x[1:2]], axis=1))
    assert np.array_equal(_np(y[0, :3]), _np(packed[0, :3]))
    assert np.array_equal(_np(y[1]), _np(packed[0, 3:]))
    assert bool(mx.all(y[0, 3:] == 0))
    np.testing.assert_allclose(_np(y[0, :3]), _np(mlp(x[:1, :3])[0]), atol=3e-2, rtol=2e-2)


def test_invariant_prefill_swaps_tile_projections(tiles_setup):
    from mlx2.runtime.models import invariant_prefill as inv
    s = tiles_setup
    tiles = _install(s)
    stack = Stack(s.layers)
    handle = inv.install(stack)
    assert handle.installed and handle.audit() is None
    assert tiles.class_law_refusal(
        "invariant_prefill", lambda m: isinstance(m, (inv.InvariantQuantizedLinear, inv.InvariantLinear))) is None
    x = (mx.random.normal((1, 192, D)) * 2).astype(mx.bfloat16)
    mlp = s.layers[1].mlp
    cuts = (0, 40, 104, 192)                                     # slices of 40, 64 and 88 rows

    def sliced():
        return mx.concatenate([mlp(x[:, a:b]) for a, b in zip(cuts, cuts[1:])], axis=1)

    with inv.scope():
        whole, parts = mlp(x), sliced()
    # Slice invariance (the law's point): a sliced prefill equals the whole prefill bit for bit.
    # Caveat: MLX's CPU kernels are already slice-invariant here (checked 2026-11-14), so on CPU this
    # shows the tiled MLP adds no slice dependence of its own; only Metal can falsify the law.
    assert np.array_equal(_np(whole), _np(parts))
    plain = mlp(x)                                               # outside the scope: stock arithmetic
    np.testing.assert_allclose(_np(whole), _np(plain), atol=3e-2, rtol=2e-2)


# -- adapter admission, route checks, receipts -----------------------------------------

def test_adapter_admits_spec_routes_and_returns_the_route():
    from mlx2.adapters.qwen38_27b import _refuse_amoeba_combinations as refuse
    assert refuse("/x", {}, require_mtp=False, weight_streaming=None) == "ordinary"
    assert refuse("/x", {}, require_mtp=True, weight_streaming=None) == "native_mtp"
    for policy in ({"draft_model": "/d"}, {"draft_model": "/d", "target_verify_row_exact": True},
                   {"draft_model": "/d", "target_verify_row_exact": True, "q4_gate_three_rows": True},
                   {"draft_model": "/d", "varlen_dense_mlp": True}, {"draft_model": "/d", "invariant_prefill": True},
                   {"draft_model": "/d", "mtp_head": "/h"}):
        assert refuse("/x", policy, require_mtp=False, weight_streaming=None) == "external_draft"
    assert refuse("/x", {"varlen_dense_mlp": True, "invariant_prefill": True},
                  require_mtp=False, weight_streaming=None) == "ordinary"


@pytest.mark.parametrize("policy,match", [
    ({"tensorfold_prefill": True}, "tensorfold_prefill"),
    ({"draft_models": ["a"]}, "draft_models"),
    ({"prompt_lookup": {"on": True}}, "prompt_lookup"),
    ({"ast_graft_prompt_lookup": True}, "ast_graft_prompt_lookup"),
    ({"self_mtp_copy_draft": True}, "self_mtp_copy_draft"),
    ({"mtp_ordinary_handoff": True}, "mtp_ordinary_handoff"),
    ({"draft_model": "pld"}, "draft_model=pld"),
    ({"target_verify_row_exact": True}, "needs a draft_model"),
])
def test_adapter_refuses_unsupported_combinations(policy, match):
    from mlx2.adapters.qwen38_27b import _refuse_amoeba_combinations as refuse
    with pytest.raises(ValueError, match=match):
        refuse("/x", policy, require_mtp=False, weight_streaming=None)
    with pytest.raises(ValueError, match="weight streaming"):
        refuse("/x", {}, require_mtp=True, weight_streaming="dense_layers")


def test_drafter_provenance():
    from mlx2.adapters.qwen38_27b import _amoeba_drafter
    assert _amoeba_drafter("ordinary", native_tree=False, mtp_head=None, draft_record=None) is None
    mtp = _amoeba_drafter("native_mtp", native_tree=False, mtp_head=None, draft_record=None)
    assert mtp["kind"] == "native_mtp_head" and "not tiled" in mtp["weights"]
    tree = _amoeba_drafter("external_draft", native_tree=True, mtp_head="/h", draft_record=None)
    assert tree["kind"] == "native_mtp_head_tree_tiles" and tree["mtp_head_artifact"] == "/h"
    ext = _amoeba_drafter("external_draft", native_tree=False, mtp_head=None,
                          draft_record={"draft_revision": "rev1"})
    assert ext == {"kind": "external_draft_model", "draft_revision": "rev1",
                   "weights": "the drafter's own checkpoint, not tiled", "input_hidden": "the tiled trunk's taps"}


def test_receipt_route_identity_and_route_check(tiles_setup):
    from mlx2.runtime.amoeba_tiles import load_manifest, tile_set_identity
    s = tiles_setup
    drafter = {"kind": "native_mtp_head", "weights": "full source MTP head (language_model.mtp), not tiled"}
    tiles = _install(s, route="native_mtp", drafter=drafter)
    r = tiles.receipt()
    assert r["route"] == "native_mtp" and r["drafter"] == drafter and r["qualified"] is False
    assert r["gate_arithmetic"] and r["projection_law"]
    m = load_manifest(s.manifest)
    assert tile_set_identity(m, route="ordinary") != tile_set_identity(m, route="native_mtp")
    tiles.check_route("native_mtp")
    for other in (None, "ordinary", "external_draft", "prompt_lookup"):
        with pytest.raises(ValueError, match="installed for the native_mtp route"):
            tiles.check_route(other)
    with pytest.raises(ValueError, match="route must be one of"):
        tile_set_identity(m, route="tree")


def _fake_adapter(tiles, **attrs):
    base = dict(amoeba_tiles=tiles, tensorfold_prefill=None, invariant_prefill=None,
                target_verify_row_exact=None, q4_gate_three_rows=None, varlen_dense_mlp=None,
                direct_prefill_candidate=False, fast_ast_target=None)
    return SimpleNamespace(**{**base, **attrs})


def test_adapter_law_audit_and_lane_bind(tiles_setup, cpu_lane):
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter as A
    s = tiles_setup
    tiles = _install(s, route="external_draft")
    fake = _fake_adapter(tiles, target_verify_row_exact=object())
    with pytest.raises(ValueError, match="target_verify_row_exact does not cover"):
        A._audit_amoeba_laws(fake)                               # tiles not swapped by a row-exact route
    with pytest.raises(ValueError, match="tensorfold_prefill"):
        A._audit_amoeba_laws(_fake_adapter(tiles, tensorfold_prefill={"installed": 1}))
    A._audit_amoeba_laws(_fake_adapter(tiles, varlen_dense_mlp={"x": 1}))
    assert "varlen_dense_mlp" in tiles.receipt()["projection_laws"]
    receipt = cpu_lane.install(Stack(s.layers), min_rows=1, skip=lambda name, m: "down_union" in name)
    with pytest.raises(ValueError, match="external_draft route refused"):
        A.bind_serving_lane_matmul(_fake_adapter(tiles), {**receipt, "law_id": "lane-matmul-v1"})
    assert A.bind_serving_lane_matmul(_fake_adapter(tiles), None) is None


def test_native_mtp_head_stays_full(tmp_path):
    """Only the trunk's layers are tiled: the MTP head (its own decoder layer and MLP) keeps the
    full source weights, and the row-exact/lane walkers that skip ``mtp`` paths see no tile there."""
    from mlx2.runtime.amoeba_tiles import install
    from mlx2.runtime.models.qwen38_27b import Model, ModelArgs
    from mlx2.runtime.models.qwen3_next import Qwen3NextMLP

    text = dict(
        model_type="qwen3_5", hidden_size=TD, intermediate_size=TN, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, vocab_size=96,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=4, full_attention_interval=4, mtp_num_hidden_layers=1,
        partial_rotary_factor=0.5, rope_parameters=None, max_position_embeddings=512,
    )
    mx.random.seed(5)
    model = Model(ModelArgs(model_type="qwen3_5", text_config=text))
    nn.quantize(model, group_size=64, bits=4,
                class_predicate=lambda p, m: isinstance(m, nn.Linear) and ".mlp." in f".{p}.")
    model.eval()
    mx.eval(model.parameters())
    layers = list(model.language_model.model.layers)
    _write_source(layers, tmp_path / "src")
    core, boxes, gates, dense = _masks(len(layers), TN, TD, seed=4)
    _export().write_tiles(tmp_path / "t", tmp_path / "src", "fp", "fake", core, dense, boxes, gates,
                          len(layers), TN, print, "t")
    head_mlp = model.language_model.mtp.layers[0].mlp
    install(model, layers, tmp_path / "t" / "manifest.json", DECL, source_fingerprint="fp", route="native_mtp")
    assert model.language_model.mtp.layers[0].mlp is head_mlp and isinstance(head_mlp, Qwen3NextMLP)
    tiled = [name for name, m in model.named_modules() if getattr(m, "amoeba_tiled", False)]
    assert tiled and not any("mtp" in name.split(".") for name in tiled)


# -- review follow-ups (2026-11-14): counters, per-row gate, exactness labels, load order -----

def test_gate_counts_fold_bounded_and_match_a_direct_count(tiles_setup):
    from mlx2.runtime.amoeba_tiles import GATE_FOLD_EVERY, gate_open
    s = tiles_setup
    tiles = _install(s)
    xs = [(mx.random.normal((2, 3, D)) * 2).astype(mx.bfloat16) for _ in range(GATE_FOLD_EVERY // 2 + 7)]
    expected = 0.0
    for x in xs:
        for l in range(1, L):
            mx.eval(s.layers[l].mlp(x))
            W, b = s.gates["a"]
            expected += float(gate_open(x, mx.array(W[l]), mx.array([float(b[l])], dtype=mx.float32)).sum())
    assert len(tiles.counters["pending"]) < GATE_FOLD_EVERY            # folded, not growing
    assert tiles.counters["open_totals"]                                # on-device running total
    counters = tiles.receipt()["counters"]
    assert counters["box_open_rows"]["a"] == expected
    assert counters["box_layer_rows"]["a"] == len(xs) * 6 * (L - 1)
    assert counters["box_open_fraction"]["a"] == expected / counters["box_layer_rows"]["a"]
    assert "verify row" in counters["box_open_fraction_denominator"]


def test_per_row_gate_fallback(tiles_setup):
    from mlx2.runtime.amoeba_tiles import (GATE_ARITHMETIC_PER_ROW, gate_open, gate_open_rows, install,
                                           load_manifest, tile_set_identity)
    s = tiles_setup
    w = mx.array(np.random.default_rng(2).normal(size=(D,)).astype(np.float32))
    b = mx.array([0.0], dtype=mx.float32)
    x = (mx.random.normal((2, 7, D)) * 3).astype(mx.bfloat16)
    rows = gate_open_rows(x, w, b)
    alone = mx.stack([mx.concatenate([gate_open(x[i:i + 1, j:j + 1], w, b) for j in range(7)], axis=1)[0]
                      for i in range(2)])
    assert rows.shape == (2, 7, 1) and np.array_equal(_np(rows), _np(alone))
    tiles = _install(s, route="native_mtp")
    batched = [s.layers[l].mlp(x) for l in range(1, L)]
    for lay, orig in zip(s.layers, [_quantized_mlp(l + 1) for l in range(L)]):
        lay.mlp = orig
    per_row = install(None, s.layers, s.manifest, DECL, source_fingerprint="fp-test", route="native_mtp",
                      gate_per_row=True)
    assert per_row.identity["gate_arithmetic"] == GATE_ARITHMETIC_PER_ROW != tiles.identity["gate_arithmetic"]
    for l in range(1, L):                                            # CPU: same gates either way
        assert np.array_equal(_np(s.layers[l].mlp(x)), _np(batched[l - 1]))
    m = load_manifest(s.manifest)
    assert tile_set_identity(m, gate_per_row=True) != tile_set_identity(m)
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter
    with pytest.raises(ValueError, match="need amoeba_tiles"):
        Qwen3827BAdapter("/nonexistent", execution_policy={"amoeba_tiles_gate_per_row": True})
    with pytest.raises(ValueError, match="must be boolean"):
        Qwen3827BAdapter("/nonexistent", execution_policy={"amoeba_tiles": "/x", "amoeba_tiles_gate_per_row": 1})


def test_exactness_labels(tiles_setup, cpu_lane):
    """A speculative route is exact only on a stated basis; the default crossover law is near-tie-only."""
    s = tiles_setup
    tiles = _install(s, route="native_mtp")
    assert tiles.receipt()["exactness"]["exactness"] == "unclassified"
    assert tiles.classify_exactness(None)["exactness"] == "near_tie_only"            # stock, no lane
    assert tiles.classify_exactness(None, row_exact=True)["exactness"] == "row_exact"
    stack = Stack(s.layers)
    crossover = cpu_lane.install(stack, min_rows=8)
    label = tiles.classify_exactness(crossover)
    assert label["exactness"] == "near_tie_only" and "crossover" in label["basis"]
    assert tiles.classify_exactness(crossover, tree_position_law=True)["exactness"] == "tree_position_law"
    exact = cpu_lane.install(stack, min_rows=1)
    assert tiles.classify_exactness(exact)["exactness"] == "lane_exact_min_rows_1"
    assert tiles.classify_exactness(exact, fast_ast=True)["exactness"] == "lane_exact_min_rows_1"
    assert tiles.receipt()["exactness"]["gpu_verified"] is False
    tiles.route = "ordinary"
    assert tiles.classify_exactness(exact)["exactness"] == "reference"


def test_lane_bind_labels_native_mtp_crossover_near_tie_only(tiles_setup, cpu_lane):
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter as A
    s = tiles_setup
    tiles = _install(s, route="native_mtp")
    receipt = cpu_lane.install(Stack(s.layers), min_rows=8)          # the served auto crossover
    A.bind_serving_lane_matmul(_fake_adapter(tiles, external_policy={}), receipt)
    assert tiles.receipt()["exactness"]["exactness"] == "near_tie_only"


def _finish_load_harness(monkeypatch, model, manifest, *, invariant, row_exact):
    """Run the real Qwen3827BAdapter._finish_load on a tiny tiled model, recording installer order."""
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter
    from mlx2.runtime.lane import tiled_prefill
    from mlx2.runtime.models import invariant_prefill, varlen_dense_mlp

    order = []

    def spy(name, fn):
        def wrapped(*a, **k):
            order.append(name)
            return fn(*a, **k)
        return wrapped

    monkeypatch.setattr(tiled_prefill, "admit_model", spy("direct_prefill", tiled_prefill.admit_model))
    monkeypatch.setattr(varlen_dense_mlp, "install", spy("varlen", varlen_dense_mlp.install))
    monkeypatch.setattr(invariant_prefill, "install", spy("invariant_prefill", invariant_prefill.install))
    adapter = object.__new__(Qwen3827BAdapter)
    adapter.model = model
    adapter.identity = {"fingerprint": "fp-tiny"}
    adapter.fused_gdn = adapter.fused_gdn_prefill = adapter.fused_gdn_compact_replay = False
    adapter.fused_gdn_architecture = "qwen38"
    adapter.external_policy = {"target_verify_row_exact": True} if row_exact else {}
    adapter._kernels = {"attention_window_row_exact": False}
    adapter._q4_gate_three_rows_requested = False
    adapter._amoeba_request = {"manifest": str(manifest), "route": "external_draft" if row_exact else "ordinary",
                               "drafter": None}
    adapter._record_load_dtype = lambda: None
    adapter._select_gdn_state = lambda dtype: None
    for name in ("_install_amoeba_tiles", "_select_target_verify_row_exact", "_select_q4_gate_three_rows",
                 "_audit_amoeba_laws"):
        object.__setattr__(adapter, name, spy(name.strip("_"), getattr(adapter, name)))

    class Stop(Exception):
        pass

    def tokenizer(*a, **k):
        order.append("tokenizer")
        raise Stop
    prefill = SimpleNamespace(tensorfold_prefill=False, tensorfold_prefill_backend=None, gdn_prefill_chunk=0,
                              gdn_prefill_segment_rows=None, invariant_prefill=invariant)
    weights = {"held": mx.zeros((1,))}
    with pytest.raises(Stop):
        adapter._finish_load(weights, prefill, False, Path("/nonexistent"), {}, SimpleNamespace(
            from_pretrained=tokenizer), None, None, (0, 0), "float32",
            varlen_dense_mlp.VarlenDenseMLPPolicy(enabled=True))
    assert weights == {}
    return adapter, order


@pytest.mark.parametrize("invariant,row_exact", [(True, False), (False, True)], ids=["invariant", "row_exact"])
def test_real_finish_load_installs_tiles_before_every_projection_installer(tiny_tiled, monkeypatch,
                                                                           invariant, row_exact):
    """The real _finish_load (tiny synthetic model, real tile/varlen/invariant/row-exact installers;
    tokenizer and dtype probes stubbed): tiles install first, the laws then cover them (the audit
    passes), and the source-weight dict is cleared before the tile build."""
    model, layers, manifest = tiny_tiled
    adapter, order = _finish_load_harness(monkeypatch, model, manifest, invariant=invariant, row_exact=row_exact)
    want = ["install_amoeba_tiles", "direct_prefill", "varlen"] + (["invariant_prefill"] if invariant else []) + [
        "select_target_verify_row_exact", "select_q4_gate_three_rows", "audit_amoeba_laws", "tokenizer"]
    assert order == want
    laws = adapter.amoeba_tiles.receipt()["projection_laws"]
    assert ("invariant_prefill" in laws) is invariant and ("target_verify_row_exact" in laws) is row_exact
    assert "varlen_dense_mlp" in laws


def test_drafter_providers_bind_after_the_tile_install():
    """Static half of the ordering check: in _init_qwen38 the drafter/target providers are bound after
    _finish_load (which installs the tiles first), and no tile install happens anywhere else."""
    import ast
    import inspect
    import textwrap
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter

    tree = ast.parse(textwrap.dedent(inspect.getsource(Qwen3827BAdapter._init_qwen38)))
    calls = [(n.lineno, ast.unparse(n.func)) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    first = {name: min(line for line, f in calls if f.endswith(name)) for name in (
        "_finish_load", "_bind_external_drafter", "bind_tree_tile_drafter",
        "install_qwen38_native_flat_physical_target_provider", "graft_prompt_lookup_drafter")}
    assert all(first["_finish_load"] < line for name, line in first.items() if name != "_finish_load")
    assert not any(f.endswith("_install_amoeba_tiles") for _, f in calls)
    finish = ast.parse(textwrap.dedent(inspect.getsource(Qwen3827BAdapter._finish_load)))
    assert any(isinstance(n, ast.Call) and ast.unparse(n.func).endswith("_install_amoeba_tiles")
               for n in ast.walk(finish))
