"""Hy4-flash opt-in small-op fusions on the tiny CPU checkpoint (hy_v4_flash_fusions).

Every fusion is default off; with it on, the CPU forward must equal the
unfused reference bit for bit (the Metal kernels fall back to the reference
ops on a CPU stream; the fused GEMMs and stacked Engram math are exact).  The
Metal kernels themselves are covered by tests/test_hy_v4_flash_fusions_metal.py.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from test_hy_v4_flash_cpu import LONG, TOKENS, _drain, _greedy, build, checkpoint, tiny_config  # noqa: E402

from mlx2.runtime.models import hy_v4_flash as H  # noqa: E402
from mlx2.runtime.models import hy_v4_flash_fusions as F  # noqa: E402

ALL = dict(hc_seam=True, mla_inputs=True, engram_inputs=True, indexer_scores=True)


@pytest.fixture(scope="module")
def tiny():
    config = tiny_config()
    W = checkpoint(config)
    model, _ = build(config, W)
    return config, W, model


def _fused(config, W, **kw):
    model, _ = build(config, W)
    F.install(model, **kw)
    return model


def test_fusions_default_off(tiny):
    _, _, model = tiny
    st = F.status(model)
    assert not any(v["selected"] for v in st.values())
    assert all(getattr(layer, "hc_seam", None) is None for layer in model.model.layers)


@pytest.mark.parametrize("kw", [dict(hc_seam=True), dict(mla_inputs=True), dict(engram_inputs=True),
                                dict(indexer_scores=True), ALL], ids=lambda kw: "+".join(kw))
def test_fused_forward_is_bitwise_reference(tiny, kw):
    config, W, ref = tiny
    model = _fused(config, W, **kw)
    # 40 tokens: past candidate_topk (8), so the indexer selection runs.
    tokens = mx.array([LONG[:40]])
    assert mx.array_equal(model(tokens), ref(tokens)).item()
    assert _greedy(model, TOKENS[:12], 8) == _greedy(ref, TOKENS[:12], 8)


def test_fused_status_counts_use(tiny):
    config, W, _ = tiny
    model = _fused(config, W, **ALL)
    model(mx.array([LONG[:40]]))
    # Decode steps: one row, within every measured bound (prefill rows above
    # a shape's self-checked bound run the split GEMMs).
    _greedy(model, TOKENS[:6], 3)
    st = F.status(model)
    assert st["mla_fused_input_proj"]["observed_used"]
    assert st["engram_fused_proj"]["observed_used"]
    assert set(st["mla_fused_input_proj"]["layers"].values()) == {"fused"}
    # On a CPU stream the seam and indexer kernels run the reference ops.
    assert st["hc_seam_fused"]["calls"]["metal"] == 0 and st["hc_seam_fused"]["calls"]["reference"] > 0
    assert not st["indexer_fused_scores"]["observed_used"]
    # The split projections are weightless views: no doubled parameters.
    attn = model.model.layers[0].self_attn
    assert isinstance(attn.q_a_proj, F.SegmentLinear) and not attn.q_a_proj.parameters()
    assert isinstance(attn.compressed_indexer.token_key_projection, F.SegmentLinear)


def test_fused_lanes_batch_equals_single_lane(tiny):
    from mlx2.runtime.generate import BatchGenerator

    config, W, ref = tiny
    model = _fused(config, W, **ALL)
    prompts = [TOKENS[:10], TOKENS[:17], TOKENS[3:7]]
    batch = BatchGenerator(model, max_tokens=6, completion_batch_size=3, prefill_batch_size=2)
    uids = batch.insert(prompts, max_tokens=[6] * 3)
    out = _drain(batch, uids, 6)
    for uid, prompt in zip(uids, prompts):
        assert out[uid][:6] == _greedy(ref, prompt, 6)
    # Lane slices ran each projection's own segment (exactly the split route).
    assert F.status(model)["mla_fused_input_proj"]["segment_calls"] > 0


def test_mla_fusion_refuses_mixed_formats(tiny):
    config, W, ref = tiny
    model, _ = build(config, W)
    attn = model.model.layers[2].self_attn  # a sliding layer
    attn.q_a_proj = nn.QuantizedLinear.from_linear(attn.q_a_proj, group_size=32, bits=8)
    record = F.install(model, mla_inputs=True)
    assert record["mla_fused_input_proj"][2].startswith("format_mismatch")
    assert record["mla_fused_input_proj"][0] == "fused"
    assert not isinstance(attn.q_a_proj, F.SegmentLinear)


def test_fused_linear_quantized_segments_are_bitwise():
    rng = np.random.default_rng(0)
    mods = []
    for out in (16, 24, 8):
        lin = nn.Linear(64, out, bias=False)
        lin.weight = mx.array(rng.standard_normal((out, 64)).astype(np.float32) * 0.1)
        mods.append(nn.QuantizedLinear.from_linear(lin, group_size=32, bits=4))
    fused = F.FusedLinear(mods, ["a", "b", "c"])
    for rows in (1, 5, 12, 40):
        x = mx.array(rng.standard_normal((1, rows, 64)).astype(np.float32))
        parts = fused.segments(x)
        for part, mod in zip(parts, mods):
            assert mx.array_equal(part, mod(x)).item()
    # Above the measured row bound each segment is its own GEMM.
    assert fused.stats["segment_calls"] == 3


def test_engram_fused_layer_bitwise_bf16():
    config = tiny_config()
    args = H.ModelArgs.from_hf_config(config)
    rng = np.random.default_rng(4)
    from mlx.utils import tree_flatten, tree_unflatten

    def layer():
        lay = H.EngramLayer(args, 2)
        params = [(k, mx.array((rng.standard_normal(v.shape) * 0.3 + (1.0 if "norm" in k else 0.0))
                               .astype(np.float32)).astype(mx.bfloat16))
                  for k, v in tree_flatten(lay.parameters()) if v.dtype != mx.int32]
        lay.update(tree_unflatten(params))
        return lay

    ref = layer()
    fused = layer()
    fused.update(ref.parameters())
    fused.fused = True
    assert F.fuse_engram_inputs(fused) is None
    streams = mx.array(rng.standard_normal((2, 5, 4, 32)).astype(np.float32)).astype(mx.bfloat16)
    rows = mx.array(rng.standard_normal((2, 5, 4, 4)).astype(np.float32))
    mask = mx.array(np.array([[1, 1, 0, 1, 1], [1, 1, 1, 1, 0]], dtype=bool))
    assert mx.array_equal(fused(streams, rows, mask), ref(streams, rows, mask)).item()


def _torch_gate(key, query, hidden):
    """The reference's gate, executed in torch: ``(key * query).sum(-1) * hidden**-0.5`` and on."""
    torch = pytest.importorskip("torch")
    k = torch.tensor(np.array(key.astype(mx.float32))).bfloat16()
    q = torch.tensor(np.array(query.astype(mx.float32))).bfloat16()
    gate = (k * q).sum(dim=-1) * hidden ** -0.5
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    return np.array(gate.sigmoid().float())


def test_engram_gate_torch_rounding_matches_torch():
    pytest.importorskip("torch")
    rng = np.random.default_rng(11)
    D = 3072
    # Normed key and query rows (rms 1), as the layer feeds the gate.
    key = H.torch_rms_norm(mx.array(rng.standard_normal((2048, D)).astype(np.float32)).astype(mx.bfloat16))
    query = H.torch_rms_norm(mx.array((rng.standard_normal((2048, D)) * 1.7 + 0.2).astype(np.float32))
                             .astype(mx.bfloat16))
    want = _torch_gate(key, query, D)

    def mismatches(torch_round):
        got = H.engram_gate(key, query, D, transformers=True, torch_round=torch_round)
        return int((np.array(got.astype(mx.float32)) != want).sum())

    default, torchlike = mismatches(False), mismatches(True)
    # Measured: default ~1000 of 2048 rows, torch rounding 0-2 (fp32 summation order).
    assert torchlike <= 4, torchlike
    assert default > 100 * max(torchlike, 1), (default, torchlike)


def test_engram_gate_rounding_policy_changes_only_transformers_bf16(tiny):
    config, W, ref = tiny
    model = _fused(config, W, engram_gate_rounding="torch")
    # fp32 activations: the torch rounding points are no-ops.
    tokens = mx.array([LONG[:20]])
    assert mx.array_equal(model(tokens), ref(tokens)).item()
    assert F.status(model)["engram_gate_rounding"]["observed_used"]


def test_adapter_policy_keys_parse_and_namespace():
    from mlx2.adapters import hy_v4_flash as A

    keys = {"hc_seam_fused", "mla_fused_input_proj", "engram_fused_proj", "indexer_fused_scores",
            "engram_gate_rounding"}
    assert keys <= A.MODEL_KEYS
    assert keys <= A.HYV4FlashAdapter.route_neutral_policy_keys
    model, _, _ = A._parse_policy({"hc_seam_fused": True, "engram_gate_rounding": "torch"})
    assert model == {"hc_seam_fused": True, "engram_gate_rounding": "torch"}
    for bad in ({"hc_seam_fused": "metal"}, {"mla_fused_input_proj": 1}, {"engram_gate_rounding": "bf16"}):
        with pytest.raises(ValueError):
            A._parse_policy(bad)
    with pytest.raises(ValueError, match="tensorfold"):
        A._parse_policy({"mla_fused_input_proj": True, "draft_model": "embedded_dfly",
                         "ast_tensorfold_rows": True})
    base = A.numerics_layout("L", fusions=A.fusion_policy({}))
    assert base == "L"   # every fusion opt-in (MLA input fusion reverted 2026-11-22: slower on the real model)
    tagged = A.numerics_layout("L", fusions=A.fusion_policy({k: True for k in keys - {"engram_gate_rounding"}}))
    assert tagged == "L+engram-fused-proj-v1+hc-seam-v1+indexer-fused-v1+mla-fused-in-v1"
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "grid", Path(__file__).resolve().parents[1] / "scripts" / "build_model_feature_grid.py")
    grid = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(grid)
    assert keys <= set(grid.FEATURES)


def test_load_model_installs_fusions(tmp_path, tiny):
    from test_hy_v4_flash_cpu import _write_artifact

    from mlx2.adapters import hy_v4_flash as A

    config, W, ref = tiny
    path = _write_artifact(tmp_path, config, W)
    artifact = A.inspect_artifact(path)
    model, _, _ = A.load_model(path, artifact, fusions=A.fusion_policy({"mla_fused_input_proj": True,
                                                                         "engram_fused_proj": True}))
    tokens = mx.array([LONG[:30]])
    assert mx.array_equal(model(tokens), ref(tokens)).item()
    assert _greedy(model, TOKENS[:6], 3) == _greedy(ref, TOKENS[:6], 3)
    st = F.status(model)
    assert st["mla_fused_input_proj"]["observed_used"] and st["engram_fused_proj"]["observed_used"]
    plain, _, _ = A.load_model(path, artifact, fusions=A.fusion_policy({"mla_fused_input_proj": False}))
    assert getattr(plain, "fusion_record", None) is None


def test_engram_fusion_refusal_keeps_split_route(tiny):
    """A refused Engram bank (mixed formats) keeps its split projections and still runs."""
    config, W, _ = tiny
    model, _ = build(config, W)
    ref16, _ = build(config, W)
    for m in (model, ref16):
        proj = m.model.layers[1].engram_layer.key_projs[2]
        proj.weight = proj.weight.astype(mx.float16)
    record = F.install(model, engram_inputs=True)
    assert record["engram_fused_proj"][1].startswith("format_mismatch")
    assert record["engram_fused_proj"][4] == "fused"
    eng = model.model.layers[1].engram_layer
    assert eng.fused is False and "fused_proj" not in eng
    tokens = mx.array([LONG[:30]])
    assert mx.array_equal(model(tokens), ref16(tokens)).item()


def test_fusions_calibrate_row_bounds_at_load(tiny):
    config, W, _ = tiny
    model = _fused(config, W, mla_inputs=True, engram_inputs=True)
    bounds = model.fusion_record["row_bounds"]
    assert bounds["device"] == "cpu" and bounds["mlx"] == mx.__version__
    for shape in bounds["shapes"]:
        assert 0 <= shape["measured_bound"] <= shape["static_bound"] and shape["modules"]
    measured = {s["measured_bound"] for s in bounds["shapes"]}
    for layer in model.model.layers:
        assert layer.self_attn.fused_input_proj.max_rows in measured
    assert F.status(model)["mla_fused_input_proj"]["row_bounds"] == bounds


def test_fused_linear_bound_zero_never_fuses():
    rng = np.random.default_rng(2)
    mods = [nn.Linear(16, 8, bias=True) for _ in range(2)]
    fused = F.FusedLinear(mods, ["a", "b"])
    fused.set_max_rows(0)
    x = mx.array(rng.standard_normal((1, 1, 16)).astype(np.float32))
    for part, mod in zip(fused.segments(x), mods):
        assert mx.array_equal(part, mod(x)).item()
    assert fused.stats == {"fused_calls": 0, "segment_calls": 2}
