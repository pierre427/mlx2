"""Residual ablation on Hy4-flash: CPU contracts on the tiny random checkpoint.

The Hy4 trunk writes each branch into four iHC streams and ends both branches
in an RMSNorm, so the declaration hooks the output linears with the post-norm
writer direction.  These tests check the capture geometry, that hooks equal
the weight edit and restore bit for bit, that a lam = 1 edit removes the
direction from every writer's normed output and from every stream increment,
the bake, the serving overlay, and the route refusals.  No GPU, no model files.
"""

import json
from contextlib import nullcontext

import mlx.core as mx
import numpy as np
import pytest

pytest.importorskip(
    "mlx2.experimental.weight_space.heretic",
    reason="weight-space Heretic module is withheld from the public mirror",
)

import test_hy_v4_flash_cpu as T

from mlx2.adapters.hy_v4_flash import HYV4FlashAdapter
from mlx2.experimental.weight_space import residual_ablation as ra

PROMPT = [5, 7, 9, 11, 13, 4, 6, 8, 10, 12]
DECL = ra.declaration_for(HYV4FlashAdapter)


@pytest.fixture(scope="module")
def tiny():
    config = T.tiny_config()
    W = T.checkpoint(config)
    model, _ = T.build(config, W)
    return config, W, model


def _fresh(tiny):
    config, W, _ = tiny
    return T.build(config, W)[0]


def _logits(model, tokens=(PROMPT,)):
    out = model(mx.array(list(tokens)), cache=model.make_cache())
    mx.eval(out)
    return np.asarray(out)


def _unit(rng, hidden):
    v = rng.standard_normal(hidden).astype(np.float32)
    return v / np.linalg.norm(v)


def _edits(layers, hidden, lam=1.0, seed=3, shared=False, mlp_ratio=0.6):
    rng = np.random.default_rng(seed)
    one = _unit(rng, hidden)
    edits = []
    for layer in range(layers):
        v = one if shared else _unit(rng, hidden)
        edits += [(layer, "attention", lam, v), (layer, "mlp", lam * mlp_ratio, v)]
    return edits


def _record(model, names):
    """Record each layer's ``names`` module call outputs (and layer outputs)."""
    rows = {}
    pairs = []
    for index, layer in enumerate(model.layers):
        for name in names:
            module = layer if name == "layer" else layer[name]

            def hook(out, key=(index, name)):
                rows[key] = np.asarray(out.astype(mx.float32))
                return out

            pairs.append((module, hook))
    return rows, ra._hooked(pairs)


def test_declaration_resolves_hy4_writers(tiny):
    _, _, model = tiny
    linears = ra.resolve_linears(model, DECL)
    paths = {(layer, path) for layer, _component, path, *_ in linears}
    assert (0, "self_attn.o_proj") in paths and (0, "mlp.down_proj") in paths
    for layer in range(1, len(model.layers)):
        assert {(layer, "self_attn.o_proj"), (layer, "mlp.routed_expert_up_proj"),
                (layer, "mlp.shared_experts.down_proj")} <= paths
        assert (layer, "mlp.down_proj") not in paths
    assert all(module["weight"].ndim == 2 for *_, module in linears)
    gammas = ra.post_norm_gammas(model, DECL)
    assert set(gammas) == {(layer, c) for layer in range(len(model.layers)) for c in ra.COMPONENTS}
    np.testing.assert_array_equal(gammas[(3, "mlp")], np.asarray(model.layers[3].mlp.combine_output_norm.weight))
    np.testing.assert_array_equal(gammas[(0, "mlp")], np.asarray(model.layers[0].mlp.post_fc2_norm.weight))


def test_declaration_contract_fails_closed(tiny):
    with pytest.raises(ValueError, match="post_norms must name"):
        ra.AblationDeclaration(**{**HYV4FlashAdapter.residual_ablation,
                                  "post_norms": {"self_attn.o_proj": "self_attn.post_proj_norm"}})
    with pytest.raises(ValueError, match="hook_point 'linears'"):
        ra.AblationDeclaration(**{**HYV4FlashAdapter.residual_ablation, "hook_point": "writers"})
    # A MoE layer whose routed bank would write the residual directly is refused.
    model = _fresh(tiny)
    model.layers[2].mlp.pop("routed_expert_up_proj")
    with pytest.raises(ValueError, match="bypass the declared output projections"):
        ra.resolve_linears(model, DECL)


def test_capture_is_stream_mean_and_default_capture_refuses_streams(tiny):
    _, _, model = tiny
    hidden = model.args.hidden_size
    with ra.LayerOutputCapture(model, DECL.capture).active() as capture:
        _logits(model)
        rows = capture.take()
        streams = capture.take_streams()
    assert rows.shape == (1, len(model.layers), hidden)
    assert streams.shape == (1, len(model.layers), model.args.hc_mult, hidden)
    recorded, hooks = _record(model, ["layer"])
    with hooks:
        _logits(model)
    manual = np.stack([recorded[(i, "layer")][0, -1] for i in range(len(model.layers))])
    np.testing.assert_allclose(streams[0], manual, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(rows[0], manual.mean(axis=1), rtol=1e-5, atol=1e-6)
    # The streams differ (post gains are per stream), so the mean is a real reduction.
    assert not np.allclose(manual[:, 0], manual[:, 1], atol=1e-4)
    with pytest.raises(ValueError, match="rank-3 layer output"):
        with ra.LayerOutputCapture(model).active():
            _logits(model)


def test_hooks_equal_weight_edit_and_restore_bit_exact(tiny):
    _, _, model = tiny
    base = _logits(model)
    classes = [type(m) for *_, m in ra.resolve_linears(model, DECL)]
    edits = _edits(len(model.layers), model.args.hidden_size, lam=0.9)
    with ra.ProjectionHooks(model, DECL).active(edits):
        hooked = _logits(model)
    np.testing.assert_array_equal(_logits(model), base)
    with ra.materialize(model, DECL, edits, "none") as swapped:
        assert swapped == len(ra.resolve_linears(model, DECL))
        edited = _logits(model)
    np.testing.assert_array_equal(_logits(model), base)
    assert [type(m) for *_, m in ra.resolve_linears(model, DECL)] == classes
    assert not np.allclose(hooked, base, atol=1e-3)
    np.testing.assert_allclose(hooked, edited, rtol=5e-4, atol=5e-4)
    with ra.materialize(model, DECL, edits, "full"):
        full = _logits(model)
    np.testing.assert_array_equal(_logits(model), base)
    assert not np.allclose(full, edited, atol=1e-4)


@pytest.mark.parametrize("mode", ["hooks", "materialize"])
def test_planted_direction_leaves_every_writer_output(tiny, mode):
    _, _, model = tiny
    layers, hidden = len(model.layers), model.args.hidden_size
    edits = _edits(layers, hidden, lam=1.0, seed=9, mlp_ratio=1.0)
    by_slot = {(layer, component): v for layer, component, _lam, v in edits}
    names = {"attention": "self_attn", "mlp": "mlp"}

    def components(context):
        recorded, hooks = _record(model, ["self_attn", "mlp"])
        with context(), hooks:
            _logits(model)
        return {(layer, c): recorded[(layer, names[c])] @ by_slot[(layer, c)] for layer, c in by_slot}

    base = components(nullcontext)
    context = ((lambda: ra.ProjectionHooks(model, DECL).active(edits)) if mode == "hooks"
               else (lambda: ra.materialize(model, DECL, edits, "none")))
    after = components(context)
    for slot in by_slot:
        assert np.max(np.abs(base[slot])) > 1e-2, slot
        assert np.max(np.abs(after[slot])) < 2e-5 * max(1.0, np.max(np.abs(base[slot]))), slot


def test_raw_direction_does_not_survive_the_post_norm(tiny):
    """Why the writer direction exists: projecting ``v`` itself before the norm leaks."""
    _, _, model = tiny
    layers, hidden = len(model.layers), model.args.hidden_size
    edits = _edits(layers, hidden, lam=1.0, seed=9)
    plain = ra.AblationDeclaration(**{**HYV4FlashAdapter.residual_ablation, "post_norms": None,
                                      "hook_point": "linears"})
    recorded, hooks = _record(model, ["self_attn"])
    with ra.ProjectionHooks(model, plain).active(edits), hooks:
        _logits(model)
    leaks = [abs(recorded[(layer, "self_attn")] @ v).max()
             for layer, component, _lam, v in edits if component == "attention"]
    assert max(leaks) > 1e-2


def test_shared_direction_leaves_every_stream_increment(tiny):
    """iHC: a branch is written to every stream with a scalar gain, so removing
    ``v`` from each writer removes it from each stream's increment (Engram
    layers add their own, unedited write)."""
    _, _, model = tiny
    layers, hidden = len(model.layers), model.args.hidden_size
    edits = _edits(layers, hidden, lam=1.0, seed=4, shared=True, mlp_ratio=1.0)
    v = edits[0][3]
    engram = set(model.args.engram_layers)
    plain, hooks = _record(model, ["layer"])
    with hooks:
        _logits(model)
    assert max(np.abs((plain[(i, "layer")] - plain[(i - 1, "layer")]) @ v).max()
               for i in range(1, layers) if i not in engram) > 1e-2
    recorded, hooks = _record(model, ["layer"])
    with ra.ProjectionHooks(model, DECL).active(edits), hooks:
        _logits(model)
    checked = 0
    for layer in range(1, layers):
        if layer in engram:
            continue
        delta = recorded[(layer, "layer")] - recorded[(layer - 1, "layer")]  # [1, L, n, D]
        scale = np.abs(recorded[(layer, "layer")] @ v).max()
        assert np.abs(delta @ v).max() < 1e-4 * max(1.0, scale), layer
        checked += 1
    assert checked >= 2


def _derived_source(tmp_path, config, W):
    """A tiny artifact shaped like the q2L38-39 pack: per-module quantization
    overrides and a previous ``mlx2_derivation`` record in config.json, plus
    extra non-shard files that must be hardlinked."""
    import json

    root = tmp_path / "src"
    root.mkdir()
    T._write_artifact(root, config, W)
    cfg = json.loads((root / "config.json").read_text())
    cfg["quantization"] = {"group_size": 64, "bits": 4, "mode": "affine",
                           "model.layers.1.mlp.switch_mlp.down_proj": {"bits": 2, "group_size": 64,
                                                                       "mode": "q2_0"},
                           "model.layers.1.mlp.gate": False}
    cfg["mlx2_derivation"] = {"kind": "routed_expert_requantization", "layers": [38, 39],
                              "source_checkpoint": "pack@sha256:abc", "qualified": False}
    (root / "config.json").write_text(json.dumps(cfg))
    (root / "engram-iq4nl.gguf").write_bytes(b"gguf-bytes")
    (root / "assets").mkdir()
    (root / "assets" / "note.txt").write_text("asset")
    return root, cfg


def test_bake_hardlinks_untouched_files_and_keeps_the_derivation(tmp_path, tiny):
    import json
    import os

    config, W, model = tiny
    tables = {k for k in W if k.endswith("multi_head_embedding.embedding.weight")}
    source, cfg = _derived_source(tmp_path, config, W)
    before = ra.hash_tree(source)
    edits = _edits(len(model.layers), model.args.hidden_size, lam=0.8, seed=21)
    receipt = ra.bake_checkpoint(
        source, tmp_path / "baked", DECL, edits, "none", edited_precision="requant",
        metadata={"source_checkpoint": "tiny-src@sha256:0", "trial": 4, "directions_sha256": "d"})
    assert ra.hash_tree(source) == before and receipt["source_unchanged"]
    baked_dir = tmp_path / "baked"
    # Only the shard holding edited tensors is rewritten; the rest are hardlinks.
    assert receipt["rewritten_shards"] == ["model-00001.safetensors"]
    assert tables  # the Engram tables live in model-00002 (hardlinked below)
    for name in ("model-00002.safetensors", "engram-iq4nl.gguf", "assets/note.txt"):
        assert os.stat(baked_dir / name).st_ino == os.stat(source / name).st_ino, name
    for name in ("config.json", "model.safetensors.index.json", "model-00001.safetensors"):
        assert os.stat(baked_dir / name).st_ino != os.stat(source / name).st_ino, name
    assert receipt["linked"] == 3 and receipt["copied"] == 2
    baked_cfg = json.loads((baked_dir / "config.json").read_text())
    assert baked_cfg["quantization"] == cfg["quantization"]  # per-module overrides kept
    derivation = baked_cfg["mlx2_derivation"]
    assert derivation["kind"] == "residual_ablation" and derivation["previous"] == cfg["mlx2_derivation"]
    assert derivation["source_checkpoint"] == "tiny-src@sha256:0" and derivation["trial"] == 4
    assert derivation["unedited_writers"] == ["embed", "engram"] == receipt["unedited_writers"]
    assert baked_cfg["mlx2_residual_ablation"]["unedited_writers"] == ["embed", "engram"]
    assert json.loads((source / "config.json").read_text()) == cfg  # source config untouched

    gammas = ra.post_norm_gammas(model, DECL)
    rows = {r["key"]: r for r in receipt["edited_tensors"]}
    assert len(rows) == len(ra.resolve_linears(model, DECL))
    from safetensors.numpy import load_file

    baked = load_file(str(baked_dir / "model-00001.safetensors"))
    for layer, component, lam, v in edits:
        u = ra.writer_direction(v, gammas[(layer, component)])
        for path in DECL.linears[component]:
            key = f"model.layers.{layer}.{path}.weight"
            if key not in W:
                continue
            np.testing.assert_allclose(baked[key], ra.edit_weight(W[key], u, lam, "none"), rtol=1e-6, atol=1e-6)
            assert rows[key]["post_norm"].endswith(DECL.post_norms[path] + ".weight")
    untouched = "model.layers.2.self_attn.q_a_proj.weight"
    np.testing.assert_array_equal(baked[untouched], W[untouched])
    # The baked checkpoint behaves like the in-memory edit.
    edited_model, _ = T.build(config, {**W, **{k: baked[k] for k in rows}})
    with ra.ProjectionHooks(model, DECL).active(edits):
        hooked = _logits(model)
    np.testing.assert_allclose(_logits(edited_model), hooked, rtol=5e-4, atol=5e-4)


def test_bake_bf16_clears_quantization_overrides_of_unquantized_tensors(tmp_path):
    """A tensor stored bf16 drops its scales, so its per-module override must go too."""
    import json

    from safetensors.numpy import save_file

    rng = np.random.default_rng(0)
    w = rng.standard_normal((64, 64)).astype(np.float32)
    wq, scales, biases = mx.quantize(mx.array(w), group_size=64, bits=8)
    norm = (1 + 0.1 * rng.standard_normal(64)).astype(np.float32)
    root = tmp_path / "q"
    root.mkdir()
    tensors = {"model.layers.0.self_attn.o_proj.weight": np.asarray(wq),
               "model.layers.0.self_attn.o_proj.scales": np.asarray(scales),
               "model.layers.0.self_attn.o_proj.biases": np.asarray(biases),
               "model.layers.0.self_attn.post_proj_norm.weight": norm,
               "model.layers.0.mlp.down_proj.weight": w,
               "model.layers.0.mlp.post_fc2_norm.weight": norm}
    save_file(tensors, str(root / "model.safetensors"))
    (root / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {k: "model.safetensors" for k in tensors}}))
    quant = {"group_size": 64, "bits": 4, "mode": "affine",
             "model.layers.0.self_attn.o_proj": {"group_size": 64, "bits": 8, "mode": "affine"}}
    (root / "config.json").write_text(json.dumps({"quantization": quant}))
    v = rng.standard_normal(64).astype(np.float32)
    v /= np.linalg.norm(v)
    receipt = ra.bake_checkpoint(root, tmp_path / "out", DECL, [(0, "attention", 1.0, v), (0, "mlp", 1.0, v)],
                                 "none", metadata={}, edited_precision="bf16")
    assert "model.layers.0.self_attn.o_proj.scales" in receipt["dropped_keys"]
    out = json.loads((tmp_path / "out" / "config.json").read_text())
    assert out["quantization"]["model.layers.0.self_attn.o_proj"] is False
    assert "previous" not in out["mlx2_derivation"]
    index = json.loads((tmp_path / "out" / "model.safetensors.index.json").read_text())["weight_map"]
    assert "model.layers.0.self_attn.o_proj.scales" not in index
    source_index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    assert "model.layers.0.self_attn.o_proj.scales" in source_index  # source index untouched


def test_read_safetensor_bf16_and_f32(tmp_path):
    from safetensors.numpy import save_file

    x = np.array([[1.5, -2.25], [3.0, 0.125]], np.float32)
    bf16 = (x.view(np.uint32) >> 16).astype(np.uint16)
    save_file({"a": x}, str(tmp_path / "f.safetensors"))
    np.testing.assert_array_equal(ra.read_safetensor(tmp_path / "f.safetensors", "a"), x)
    import json
    import struct

    header = json.dumps({"b": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, 8]}}).encode()
    (tmp_path / "g.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + bf16.tobytes())
    np.testing.assert_array_equal(ra.read_safetensor(tmp_path / "g.safetensors", "b"), x)


def _overlay_doc(tmp_path, layers, hidden):
    rng = np.random.default_rng(13)
    vectors = np.full((layers + 1, hidden), np.nan, np.float32)
    for i in range(1, layers + 1):
        vectors[i] = _unit(rng, hidden)
    directions = ra.Directions(vectors, [None] * (layers + 1))
    directions.save(tmp_path / "dirs")
    params = ra.AblationParameters(kernels={
        "attention": ra.ComponentKernel(1.0, 3.0, 0.5, 3.0),
        "mlp": ra.ComponentKernel(0.8, 3.0, 0.3, 3.0)})
    doc = ra.overlay_document(directions_path=tmp_path / "dirs", checkpoint="tiny-hy4@0",
                              parameters=params, layers=layers, label="t")
    return doc, params.plan(directions, layers)


def test_serving_overlay_rows_match_hooks_and_restore_base(tmp_path, tiny):
    from mlx2.runtime.residual_ablation_overlays import AblationOverlayManager
    from mlx2.runtime.weight_overlays import (
        bind_weight_overlay_rows,
        clear_weight_overlay_rows,
    )

    model = _fresh(tiny)
    layers, hidden = len(model.layers), model.args.hidden_size
    doc, plan = _overlay_doc(tmp_path, layers, hidden)
    tokens = (PROMPT, [3, 8, 14, 22, 9, 30, 6, 4, 18, 27])
    base = _logits(model, tokens)
    with ra.ProjectionHooks(model, DECL).active(plan):
        hooked = _logits(model, tokens)
    manager = AblationOverlayManager(model, DECL, checkpoint="tiny-hy4@0")
    info = manager.load_document(doc)
    assert manager.status()["direction_space"] == "post_norm_writer"
    np.testing.assert_array_equal(_logits(model, tokens), base)  # wrappers inert while unbound
    manager.bind_uid(7, manager.acquire(info["id"]))
    bound = bind_weight_overlay_rows(model, [7, 8])
    try:
        mixed = _logits(model, tokens)
    finally:
        clear_weight_overlay_rows(bound)
    np.testing.assert_allclose(mixed[0], hooked[0], rtol=5e-4, atol=5e-4)
    np.testing.assert_array_equal(mixed[1], base[1])
    receipt = manager.receipt(7)
    assert receipt["applied"] and receipt["direction_space"] == "post_norm_writer" and not receipt["qualified"]
    assert receipt["unedited_writers"] == ["embed", "engram"]
    manager.release(7, info["id"])
    manager.unload(info["id"])
    manager.restore()
    np.testing.assert_array_equal(_logits(model, tokens), base)
    assert all(type(m).__name__ != "AblationProjectionLinear" for *_, m in ra.resolve_linears(model, DECL))


@pytest.fixture
def host(monkeypatch):
    from mlx2 import memory, serving
    from mlx2.runtime import os_memory

    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "src"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)


def _engine(path, policy):
    from mlx2.serving import ServingEngine

    engine = ServingEngine(str(path), adapter_factory=HYV4FlashAdapter, qualification_mode=True,
                           mtp=False, max_lanes=1, execution_policy=policy)
    assert engine.ready.wait(120), engine.error
    return engine


def test_real_engine_route_gate_reads_the_loaded_route(tmp_path, tiny, host, monkeypatch):
    """The route check reads the snapshot ``_load`` really builds (2026-11-21 review:
    a check on ``settings["backend"]`` never fired; the backend lives under
    ``settings["execution_policy"]``).  DFly2 is admitted for overlays only
    because the Hy4 adapter declares it, and installs in strict-rows mode; an
    adapter that does not declare it is refused."""
    from mlx2 import serving

    config, W, _ = tiny
    path = T._tiny_artifact(tmp_path, config, W)
    engine = _engine(path, None)
    try:
        assert engine.snapshot["settings"]["route"] == "ordinary"
        assert not engine._serving_route_is_external_draft()
        manager = engine.configure_ablation_overlays()
        assert manager.layer_count == len(engine.adapter.model.layers) and manager.gammas
        assert manager.status()["unedited_writers"] == ["embed", "engram"]
        assert manager.strict_rows is False
    finally:
        engine.close()
    engine = _engine(path, {"draft_model": "embedded_dfly", "num_draft": 3})
    try:
        settings = engine.snapshot["settings"]
        assert settings["execution_policy"]["backend"] == "external_draft"
        assert "backend" not in settings  # the shape the old check assumed
        assert engine._serving_route_is_external_draft()
        # An adapter without the external-draft declaration keeps the refusal.
        monkeypatch.setattr(type(engine.adapter), "residual_overlay_routes",
                            property(lambda self: frozenset({"ordinary"})))
        with pytest.raises(ValueError, match="external-draft route the adapter declares"):
            engine.configure_ablation_overlays()
        assert getattr(engine.adapter.model, "_mlx2_ablation_overlays", None) is None
        monkeypatch.undo()
        manager = engine.configure_ablation_overlays()
        assert manager.strict_rows is True
        manager.restore()
        engine.ablation_overlays = None
        # The same resolved-route check now guards weight programs and surgery
        # (the multi-LoRA precondition is stubbed to reach the route check).
        monkeypatch.setattr(serving, "multi_lora_policy", lambda *a, **k: None)
        with pytest.raises(ValueError, match="weight programs require the ordinary route"):
            engine._validate_weight_program_route()
        with pytest.raises(ValueError, match="weight overlays require ordinary"):
            engine._validate_weight_surgery_route(adapter_policy=True)
    finally:
        engine.close()


def test_tensorfold_rows_policy_withdraws_the_external_overlay_route():
    adapter = HYV4FlashAdapter.__new__(HYV4FlashAdapter)
    adapter.external_policy = {"ast_tensorfold_rows": True}
    assert adapter.residual_overlay_routes == frozenset({"ordinary"})
    adapter.external_policy = {}
    assert adapter.residual_overlay_routes == frozenset({"ordinary", "external_draft"})


def test_route_check_reads_every_recorded_form():
    from mlx2.serving import ServingEngine

    engine = ServingEngine.__new__(ServingEngine)
    for settings, expected in (({"execution_policy": {"backend": "external_draft"}}, True),
                               ({"route": "external_draft"}, True),
                               ({"speculation": "external_draft"}, True),
                               ({"route": "ordinary", "execution_policy": {}}, False),
                               ({}, False)):
        engine.snapshot = {"settings": settings}
        assert engine._serving_route_is_external_draft() is expected, settings
    engine.snapshot = {}
    assert engine._serving_route_is_external_draft() is False


def test_requantized_materialize_matches_the_bake_law(tiny):
    """fullab's requant arm stores what ``bake --edited-precision requant`` writes."""
    from mlx import nn

    config, W, _ = tiny
    model = _fresh(tiny)
    targets = {path for paths in DECL.linears.values() for path in paths}

    def predicate(path, module):
        rest = path.split(".", 3)[-1] if path.startswith("model.layers.") else path
        return isinstance(module, nn.Linear) and rest in targets and module.weight.shape[-1] % 32 == 0

    nn.quantize(model, group_size=32, bits=8, class_predicate=predicate)
    quantized = [(layer, c, path, m) for layer, c, path, _p, _a, m in ra.resolve_linears(model, DECL)
                 if "scales" in m]
    assert quantized
    base = _logits(model)
    edits = _edits(len(model.layers), model.args.hidden_size, lam=0.9, seed=5)
    gammas = ra.post_norm_gammas(model, DECL)
    by_slot = {(layer, c): (lam, v) for layer, c, lam, v in edits}
    receipts = []
    with ra.materialize(model, DECL, edits, "none", requantize=True, receipts=receipts):
        requant = _logits(model)
        swapped = {(layer, path): m for layer, _c, path, _p, _a, m in ra.resolve_linears(model, DECL)}
        for layer, component, path, original in quantized:
            lam, v = by_slot[(layer, component)]
            u = ra.writer_direction(v, gammas[(layer, component)])
            edited = ra.edit_weight(ra.dense_weight(original), u, lam, "none")
            packed = mx.quantize(mx.array(edited).astype(original["scales"].dtype), group_size=32, bits=8)
            module = swapped[(layer, path)]
            assert isinstance(module, nn.QuantizedLinear) and module.bits == 8
            np.testing.assert_array_equal(np.asarray(module["weight"]), np.asarray(packed[0]))
            np.testing.assert_array_equal(np.asarray(module["scales"]), np.asarray(packed[1]))
    with ra.materialize(model, DECL, edits, "none"):
        dense = _logits(model)
    np.testing.assert_array_equal(_logits(model), base)
    rows = [r for r in receipts if r["stored"] == "requant"]
    assert len(rows) == len(quantized) and all(0 < r["requant_rms"] < r["edit_rms"] for r in rows)
    assert not np.array_equal(requant, dense)
    np.testing.assert_allclose(requant, dense, rtol=0.05, atol=0.05)


def test_runner_renders_through_the_adapter_and_batches_on_hy4(tiny):
    import importlib.util
    from pathlib import Path

    runner_path = Path(__file__).resolve().parents[1] / "scripts/research/residual_ablation.py"
    if not runner_path.is_file():
        pytest.skip("private residual-ablation runner is not shipped in this tree")
    spec = importlib.util.spec_from_file_location("residual_ablation_runner_hy4", runner_path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    _, _, model = tiny

    class Tok:
        eos_token_ids = [2]

        def decode(self, ids):
            return " ".join(map(str, ids))

        def apply_chat_template(self, *_a, **_k):
            raise AssertionError("Hy4 renders through the adapter")

    seen = []

    class Adapter:
        def prompt_tokens(self, request):
            seen.append(request)
            return [5, 7, 9]

    run = runner.Runner.__new__(runner.Runner)
    run.mx, run.model, run.tokenizer, run.eos = mx, model, Tok(), {2}
    run.adapter, run.declaration, run.ra = Adapter(), DECL, ra
    assert run.tokens("hello") == [5, 7, 9]
    assert seen[0]["messages"][-1] == {"role": "user", "content": "hello"}
    prompts = {"a": PROMPT[:4], "bb": PROMPT, "c": [6, 8, 10, 12, 14, 16, 18]}
    run.tokens = prompts.__getitem__
    sequential = [run.greedy(p, 6) for p in prompts]
    assert run.greedy_batched(list(prompts), 6, batch=3) == sequential
    rows = run.capture(list(prompts)[:2])
    assert rows.shape == (2, len(model.layers), model.args.hidden_size)


# ------------------------------------------------ speculative (AST / DFly2) route


def _strong_overlay(tmp_path, layers, hidden, seed=13, checkpoint="tiny-hy4@0"):
    rng = np.random.default_rng(seed)
    vectors = np.full((layers + 1, hidden), np.nan, np.float32)
    for i in range(1, layers + 1):
        vectors[i] = _unit(rng, hidden)
    directions = ra.Directions(vectors, [None] * (layers + 1))
    directions.save(tmp_path / "strong")
    # Strong enough that the overlay changes the greedy continuation.
    params = ra.AblationParameters(kernels={
        "attention": ra.ComponentKernel(1.0, 2.5, 1.0, 6.0),
        "mlp": ra.ComponentKernel(1.0, 2.5, 1.0, 6.0)})
    return ra.overlay_document(directions_path=tmp_path / "strong", checkpoint=checkpoint,
                               parameters=params, layers=layers, label="strong")


def _overlay_greedy(model, uid, prompt, n):
    """Ordinary greedy with ``uid``'s overlay (or none) bound to every forward."""
    from mlx2.runtime.residual_ablation_overlays import lane_rows

    with lane_rows(model, [uid]):
        return T._greedy(model, prompt, n)


AST_PROMPTS = [T.LONG[9:21], T.LONG[10:24], T.LONG[50:58] + T.LONG[50:55], T.LONG[60:81]]


@pytest.mark.parametrize("width", [1, 2, 3, 4])
@pytest.mark.parametrize("drafter", ["embedded_dfly", "prompt_lookup"])
def test_ast_route_with_overlay_equals_overlay_greedy_and_keeps_other_lanes_exact(
        tmp_path, tiny, drafter, width):
    from mlx2.runtime.residual_ablation_overlays import AblationOverlayManager

    config, W, _ = tiny
    adapter = HYV4FlashAdapter(str(T._tiny_artifact(tmp_path, config, W)),
                               execution_policy={"draft_model": drafter, "num_draft": 3})
    try:
        assert "external_draft" in adapter.residual_overlay_routes
        model = adapter.model
        manager = AblationOverlayManager(model, DECL, checkpoint="tiny-hy4@0", strict_rows=True)
        info = manager.load_document(_strong_overlay(tmp_path, len(model.layers), model.args.hidden_size))
        n = 12
        batch = adapter.create_external_batch(completion_batch_size=width, prefill_step_size=4)
        prompts = AST_PROMPTS[:width]
        uids = batch.insert(prompts, max_tokens=[n] * width,
                            sampling_configs=[{"sampling_temp": 0}] * width)
        # Lane 0 carries the overlay; every other lane runs the base model.
        manager.bind_uid(uids[0], manager.acquire(info["id"]))
        out = T._drain(batch, uids, n)
        expected0 = _overlay_greedy(model, uids[0], prompts[0], n)
        assert expected0 != _overlay_greedy(model, -1, prompts[0], n)  # the overlay matters
        assert out[uids[0]][:n] == expected0
        for uid, prompt in zip(uids[1:], prompts[1:]):
            assert out[uid][:n] == _overlay_greedy(model, uid, prompt, n), prompt  # base-exact
        receipt = manager.receipt(uids[0])
        assert receipt["observed_used"] and receipt["applied"] and not receipt["qualified"]
        assert receipt["projection_executions"] > 0 and receipt["unedited_writers"] == ["embed", "engram"]
        assert manager.counts.get("unbound_forward_refusals", 0) == 0
        assert manager.counts.get("row_geometry_refusals", 0) == 0
        d = adapter.diagnostics()
        assert d["route"] == "external_draft" and d["ast"]["selected"]
        if drafter == "embedded_dfly":
            assert d["dfly2"]["observed_used"]
        stats = batch.scheduler_stats
        if width > 1:
            assert stats["external_target_tree_cohort_rounds"] > 0
        manager.release(uids[0], info["id"])
        manager.restore()
    finally:
        adapter.close()


def test_strict_rows_refuse_an_unbound_target_forward(tiny):
    from mlx2.runtime.residual_ablation_overlays import AblationOverlayManager, lane_rows

    model = _fresh(tiny)
    layers, hidden = len(model.layers), model.args.hidden_size
    manager = AblationOverlayManager(model, DECL, checkpoint="tiny-hy4@0", strict_rows=True)
    import tempfile
    from pathlib import Path

    info = manager.load_document(_strong_overlay(Path(tempfile.mkdtemp()), layers, hidden))
    base = _logits(model)  # nothing bound yet: no refusal
    manager.bind_uid(5, manager.acquire(info["id"]))
    with pytest.raises(RuntimeError, match="without a lane row binding"):
        _logits(model)
    with lane_rows(model, [6]):  # a lane without an overlay: bound, base-exact
        np.testing.assert_array_equal(_logits(model), base)
    with lane_rows(model, [5, 6], packed=True):  # packed scope: the provider must bind
        with pytest.raises(RuntimeError, match="without a lane row binding"):
            _logits(model, (PROMPT, PROMPT))
    assert manager.counts["unbound_forward_refusals"] == 2
    manager.release(5, info["id"])
    manager.restore()


def test_real_engine_dfly2_overlay_request_equals_ordinary_overlay_request(tmp_path, tiny, host):
    """End to end through ServingEngine: the same overlay request on the DFly2
    route and the ordinary route commits the same tokens, with receipts."""
    config, W, model = tiny
    path = _chat_artifact(tmp_path / "art", config, W)
    text = " ".join(f"t{t}" for t in AST_PROMPTS[0])
    base_body = {"messages": [{"role": "user", "content": text}], "max_tokens": 10,
                 "temperature": 0, "logprobs": True}
    results, ids = {}, {}
    for name, policy in (("ordinary", None), ("dfly2", {"draft_model": "embedded_dfly", "num_draft": 3})):
        engine = _engine(path, policy)
        try:
            # Overlays bind to the served checkpoint identity (name@fingerprint).
            (tmp_path / name).mkdir()
            doc = _strong_overlay(tmp_path / name, len(model.layers), model.args.hidden_size,
                                  checkpoint=engine._lora_base_checkpoint())
            ra.save_overlay(tmp_path / name / "overlays" / "strong", doc)
            ids[name] = doc["id"]
            engine.configure_ablation_overlays(tmp_path / name / "overlays")
            results[name] = _collect(engine.submit({**base_body, "mlx2_overlay": doc["id"]}))
            results[name + "_base"] = _collect(engine.submit(dict(base_body)))
            results[name + "_status"] = engine.ablation_overlays.status()
        finally:
            engine.close()
    assert results["dfly2"][0] == results["ordinary"][0]
    assert results["dfly2_base"][0] == results["ordinary_base"][0]
    assert results["ordinary"][0] != results["ordinary_base"][0]
    for name in ("ordinary", "dfly2"):
        receipt = results[name][1]
        assert receipt["mlx2_overlay"]["observed_used"] and receipt["mlx2_overlay"]["id"] == ids[name]
        assert results[name + "_status"]["strict_rows"] is (name == "dfly2")
    receipt = results["dfly2"][1]
    assert receipt["route"] == "external_draft"
    assert receipt["route_receipt"]["weight_space_operation"] == "ablation_overlay"
    speculation = receipt["speculation"]
    # DFly2 verify rounds ran under the overlay (the tiny random drafter may
    # then park to plain anchors, which run through the same provider).
    assert speculation["kind"] == "external_hy4_dfly2" and speculation["external_rounds"] > 0
    assert speculation["verification"] == "exact" and speculation["proposed"] > 0


def _collect(job):
    tokens, receipt = [], None
    while True:
        event = job.events.get(timeout=120)
        if "error" in event:
            raise AssertionError(event)
        if "logprob" in event:
            tokens.append(event["logprob"]["id"])
        if "receipt" in event:
            receipt = event["receipt"]
        if "finish_reason" in event:
            return tokens, receipt or getattr(job, "receipt", None)


CHAT_TEMPLATE = "{% for m in messages %}t5 {{ m.content }} {% endfor %}t6"


def _chat_artifact(path, config, W):
    """The tiny artifact with a whitespace word-level tokenizer and a chat template."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    path.mkdir()
    T._write_artifact(path, config, W)
    vocab = {f"t{i}": i for i in range(config["text_config"]["vocab_size"])}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="t0"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(json.dumps(
        {"tokenizer_class": "PreTrainedTokenizerFast", "eos_token": "t2",
         "chat_template": CHAT_TEMPLATE}))
    return path
