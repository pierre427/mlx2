"""CPU contracts for experimental residual ablation (no checkpoint, no GPU)."""

import json

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

pytest.importorskip(
    "mlx2.experimental.weight_space.heretic",
    reason="weight-space Heretic module is withheld from the public mirror",
)

from mlx2.experimental.weight_space import residual_ablation as ra  # noqa: E402
from mlx2.experimental.weight_space.heretic import compile_abliteration, LayerProfile  # noqa: E402
from test_disk_weight_streaming import _dense_model, _official, _quantize_all  # noqa: E402

DECL = ra.AblationDeclaration(
    writers={"attention": ("linear_attn", "self_attn"), "mlp": ("mlp",)},
    linears={"attention": ("linear_attn.out_proj", "self_attn.o_proj"), "mlp": ("mlp.down_proj",)},
)
TOKENS = [[3, 9, 4, 1, 7, 2]]


def _model(seed=11):
    mx.random.seed(seed)
    model = _dense_model(interval=2)
    model.set_dtype(mx.float32)
    return model


def _logits(model):
    out = model(mx.array(TOKENS, mx.uint32), cache=model.make_cache())
    mx.eval(out)
    return np.asarray(out)


def _edits(model, seed=3, lam=0.9):
    rng = np.random.default_rng(seed)
    hidden = model.layers[0].input_layernorm.weight.shape[0]
    edits = []
    for layer in range(len(model.layers)):
        v = rng.standard_normal(hidden).astype(np.float32)
        v /= np.linalg.norm(v)
        edits += [(layer, "attention", lam, v), (layer, "mlp", lam * 0.5, v)]
    return edits


def test_adapter_declaration_required():
    class Bare:
        pass

    with pytest.raises(ValueError, match="declares no residual-ablation"):
        ra.declaration_for(Bare())
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter

    assert ra.declaration_for(Qwen3827BAdapter).writers["mlp"] == ("mlp",)


def test_hooks_match_none_weight_edit_and_restore_exactly():
    model = _model()
    base = _logits(model)
    classes = [type(layer) for layer in model.layers]
    edits = _edits(model)
    with ra.ProjectionHooks(model, DECL).active(edits):
        hooked = _logits(model)
    assert np.array_equal(_logits(model), base)
    with ra.materialize(model, DECL, edits, "none") as swapped:
        assert swapped == 2 * len(model.layers)
        edited = _logits(model)
    assert np.array_equal(_logits(model), base)
    assert [type(layer) for layer in model.layers] == classes
    assert not np.allclose(hooked, base, atol=1e-3)
    np.testing.assert_allclose(hooked, edited, rtol=2e-4, atol=2e-4)


def test_full_mode_differs_from_none_and_preserves_row_norms():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((16, 24)).astype(np.float32)
    v = rng.standard_normal(16).astype(np.float32)
    v /= np.linalg.norm(v)
    full = ra.edit_weight(w, v, 1.0, "full")
    np.testing.assert_allclose(np.linalg.norm(full, axis=1), np.linalg.norm(w, axis=1), rtol=1e-5)
    assert not np.allclose(full, ra.edit_weight(w, v, 1.0, "none"))


@pytest.mark.parametrize("mode", ["none", "full"])
def test_edit_weight_parity_with_weight_space_heretic(mode):
    rng = np.random.default_rng(1)
    w = rng.standard_normal((12, 20)).astype(np.float32)
    v = rng.standard_normal(12).astype(np.float32)
    v /= np.linalg.norm(v)
    program = compile_abliteration(
        {"m": w}, {"m": v}, {"m": {"layer": 3, "component": "attention_output"}},
        profiles={"attention_output": LayerProfile(peak_layer=3, radius=0, peak_strength=0.7)},
        checkpoint="synthetic@sha256:0", source_revision="test", provenance={"test": True},
        row_normalization=mode, full_rank=12 if mode == "full" else None,
    )
    a, b = program.factor_map["m"]
    reference = w + (a.astype(np.float64) @ b.astype(np.float64)).T
    np.testing.assert_allclose(ra.edit_weight(w, v, 0.7, mode), reference, rtol=1e-4, atol=1e-5)




def test_stacked_experts_match_per_expert_and_moe_output_projection():
    rng = np.random.default_rng(2)
    experts = rng.standard_normal((5, 8, 6)).astype(np.float32)
    v = rng.standard_normal(8).astype(np.float32)
    v /= np.linalg.norm(v)
    for mode in ("none", "full"):
        stacked = ra.edit_weight(experts, v, 0.8, mode)
        for e in range(5):
            np.testing.assert_allclose(stacked[e], ra.edit_weight(experts[e], v, 0.8, mode), rtol=1e-6)
    # "none" on every expert == one projection of the router-weighted sum.
    x = rng.standard_normal(6).astype(np.float32)
    gates = rng.random(5).astype(np.float32)
    edited = ra.edit_weight(experts, v, 0.8, "none")
    lhs = sum(g * (edited[e] @ x) for e, g in enumerate(gates))
    y = sum(g * (experts[e] @ x) for e, g in enumerate(gates))
    np.testing.assert_allclose(lhs, y - 0.8 * v * (v @ y), rtol=1e-5, atol=1e-5)


def test_capture_matches_manual_layer_loop():
    model = _model()
    with ra.LayerOutputCapture(model).active() as capture:
        _logits(model)
        rows = capture.take()
    assert rows.shape == (1, len(model.layers), model.layers[0].input_layernorm.weight.shape[0])
    driver_rows = []
    from mlx2.adapters.qwen35_capture import Qwen35CaptureDriver

    driver = Qwen35CaptureDriver(model)
    state = driver.start(TOKENS[0])
    for index in range(len(model.layers)):
        state = driver.step(state, index)
        driver_rows.append(np.asarray(state["hidden"][0, -1].astype(mx.float32)))
    np.testing.assert_allclose(rows[0], np.stack(driver_rows), rtol=1e-5, atol=1e-5)


def test_directions_recover_planted_direction_and_interpolate():
    rng = np.random.default_rng(4)
    planted = rng.standard_normal(32)
    planted /= np.linalg.norm(planted)
    good = rng.standard_normal((64, 3, 32)).astype(np.float32)
    bad = (rng.standard_normal((64, 3, 32)) + 6 * planted).astype(np.float32)
    directions = ra.compute_directions(good, bad, orthogonalize=False)
    for layer in range(3):
        assert abs(directions.for_layer(layer) @ planted) > 0.9
        assert directions.reports[layer + 1]["separation_d"] > 3
    mixed = directions.for_layer(0, direction_index=0.5)
    assert np.isclose(np.linalg.norm(mixed), 1, atol=1e-5)
    with pytest.raises(ValueError):
        directions.for_layer(0, direction_index=5.0)


def test_parameters_plan_matches_heretic_kernel():
    params = ra.AblationParameters(kernels={
        "attention": ra.ComponentKernel(1.2, 5.0, 0.3, 2.0),
        "mlp": ra.ComponentKernel(0.0, 5.0, 0.0, 2.0),
    })
    directions = ra.Directions(np.ones((9, 4), np.float32) / 2, [None] * 9)
    plan = params.plan(directions, 8)
    lams = {layer: lam for layer, component, lam, _ in plan if component == "attention"}
    assert set(lams) == {3, 4, 5, 6, 7}
    assert lams[5] == pytest.approx(1.2)
    assert lams[4] == pytest.approx(1.2 + 0.5 * (0.3 - 1.2))
    assert all(component == "attention" for _, component, _, _ in plan)


def _write_tiny_checkpoint(root):
    mx.random.seed(5)
    model = _dense_model(interval=2)
    _quantize_all(model)
    weights = {_official(k): v for k, v in tree_flatten(model.parameters())}
    root.mkdir()
    names = sorted(weights)
    half = len(names) // 2
    shards = {"model-00001-of-00002.safetensors": names[:half],
              "model-00002-of-00002.safetensors": names[half:]}
    weight_map = {}
    for shard, keys in shards.items():
        mx.save_safetensors(str(root / shard), {k: weights[k] for k in keys}, metadata={"format": "mlx"})
        weight_map.update({k: shard for k in keys})
    (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    from test_disk_weight_streaming import BITS, GROUP

    (root / "config.json").write_text(json.dumps(
        {"model_type": "qwen3_5", "quantization": {"group_size": GROUP, "bits": BITS, "mode": "affine"}}))
    (root / "tokenizer.json").write_text("{}")
    return model, weights


def test_bake_edits_only_targets_and_leaves_source_untouched(tmp_path):
    source = tmp_path / "source"
    model, weights = _write_tiny_checkpoint(source)
    before = ra.hash_tree(source)
    hidden = model.layers[0].input_layernorm.weight.shape[0]
    v = np.eye(hidden, dtype=np.float32)[0]
    edits = [(1, "attention", 1.0, v), (0, "mlp", 0.5, v)]
    receipt = ra.bake_checkpoint(source, tmp_path / "out", DECL, edits, "none", metadata={"test": 1})
    assert ra.hash_tree(source) == before and receipt["source_unchanged"]
    assert {r["component"] for r in receipt["edited_tensors"]} == {"attention", "mlp"}
    assert len(receipt["edited_tensors"]) == 2
    out = {}
    for shard in sorted((tmp_path / "out").glob("*.safetensors")):
        out.update(mx.load(str(shard)))
    edited_keys = {r["key"] for r in receipt["edited_tensors"]}
    for key, value in weights.items():
        prefix = key.rsplit(".", 1)[0]
        if prefix + ".weight" in edited_keys:
            continue
        assert np.array_equal(np.asarray(out[key]), np.asarray(value)), key
    config = json.loads((tmp_path / "out" / "config.json").read_text())
    assert config["mlx2_residual_ablation"]["edited_tensors"] == 2
    for r in receipt["edited_tensors"]:
        assert r["edit_rms"] > 0 and r["requant_rms"] < r["edit_rms"] * 4
    with pytest.raises(FileExistsError):
        ra.bake_checkpoint(source, tmp_path / "out", DECL, edits, "none", metadata={})
    with pytest.raises(ValueError, match="outside the source"):
        ra.bake_checkpoint(source, source / "nested", DECL, edits, "none", metadata={})
    assert ra.hash_tree(source) == before


def test_batched_greedy_matches_sequential(monkeypatch):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "residual_ablation_runner", Path(__file__).resolve().parents[1] / "scripts/research/residual_ablation.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    mx.random.seed(3)
    model = _dense_model(interval=2)
    model.set_dtype(mx.float32)

    class Tok:
        eos_token_ids = [0]

        def decode(self, ids):
            return " ".join(map(str, ids))

    run = runner.Runner.__new__(runner.Runner)
    run.mx, run.model, run.tokenizer, run.eos = mx, model, Tok(), {0}
    prompts = {"a": [3, 9, 4], "bb": [7, 2, 1, 10, 6, 5, 8], "c": [11], "dddd": [4, 4, 12, 3, 2]}
    run.tokens = prompts.__getitem__
    sequential = [run.greedy(p, 12) for p in prompts]
    assert run.greedy_batched(list(prompts), 12, batch=4) == sequential
    assert run.greedy_batched(list(prompts), 12, batch=3) == sequential


def test_bake_bf16_keeps_edit_exact_and_drops_quant_keys(tmp_path):
    source = tmp_path / "source"
    model, weights = _write_tiny_checkpoint(source)
    before = ra.hash_tree(source)
    hidden = model.layers[0].input_layernorm.weight.shape[0]
    v = np.eye(hidden, dtype=np.float32)[1]
    receipt = ra.bake_checkpoint(source, tmp_path / "out", DECL, [(1, "attention", 1.0, v)], "none",
                                 metadata={}, edited_precision="bf16")
    assert ra.hash_tree(source) == before
    (entry,) = receipt["edited_tensors"]
    prefix = entry["key"][: -len(".weight")]
    assert set(receipt["dropped_keys"]) == {prefix + ".scales", prefix + ".biases"}
    index = json.loads((tmp_path / "out" / "model.safetensors.index.json").read_text())["weight_map"]
    assert prefix + ".scales" not in index and entry["key"] in index
    out = {}
    for shard in (tmp_path / "out").glob("*.safetensors"):
        out.update(mx.load(str(shard)))
    assert prefix + ".scales" not in out and out[entry["key"]].dtype == mx.bfloat16
    w = out[entry["key"]].astype(mx.float32)
    # The edited direction is (almost) gone from the stored weight.
    assert float(mx.abs(mx.array(v) @ w).max().item()) < 2e-2


def _toy_directions(layers, hidden, seed=7):
    rng = np.random.default_rng(seed)
    vectors = np.full((layers + 1, hidden), np.nan, np.float32)
    for i in range(1, layers + 1):
        v = rng.standard_normal(hidden).astype(np.float32)
        vectors[i] = v / np.linalg.norm(v)
    return ra.Directions(vectors, [None] * (layers + 1))


def _toy_overlay(tmp_path, directions, params, layers, checkpoint="m@abc", name="dirs"):
    directions.save(tmp_path / name)
    return ra.overlay_document(directions_path=tmp_path / name, checkpoint=checkpoint,
                               parameters=params, layers=layers, label="t")


def test_overlay_document_binding_and_id(tmp_path):
    directions = _toy_directions(4, 8)
    params = ra.AblationParameters(kernels={
        "attention": ra.ComponentKernel(1.0, 2.0, 0.5, 2.0),
        "mlp": ra.ComponentKernel(0.0, 2.0, 0.0, 1.0)})
    doc = _toy_overlay(tmp_path, directions, params, 4)
    overlay = ra.overlay_from_document(doc, fingerprint="abc", layer_count=4)
    assert overlay.overlay_id == doc["id"] and overlay.lam.shape == (4, 2)
    assert overlay.lam[2, 0] == pytest.approx(1.0) and np.all(overlay.lam[:, 1] == 0)
    assert overlay.edits == doc["edits"]
    with pytest.raises(ra.OverlayBindingError):
        ra.overlay_from_document(doc, fingerprint="other", layer_count=4)
    with pytest.raises(ValueError, match="id does not match"):
        ra.overlay_from_document({**doc, "layers": 5}, fingerprint="abc", layer_count=4)
    relabelled = {**doc, "label": "renamed", "created": 1.0}
    assert ra.overlay_document_id(relabelled) == doc["id"]
    other = ra.overlay_document(directions_path=tmp_path / "dirs", checkpoint="m@abc", layers=4, label="t",
                                parameters=ra.AblationParameters(kernels={
                                    **params.kernels, "mlp": ra.ComponentKernel(0.5, 2.0, 0.0, 1.0)}))
    assert other["id"] != doc["id"]
    ra.save_overlay(tmp_path / "ov", doc)
    with pytest.raises(FileExistsError):
        ra.save_overlay(tmp_path / "ov", doc)


def test_overlay_id_matches_desktop_client(tmp_path):
    from mlx2.desktop_residual_ablation_cases import canonical
    import hashlib

    directions = _toy_directions(4, 8)
    params = ra.AblationParameters(kernels={
        "attention": ra.ComponentKernel(1.0, 2.0, 0.5, 2.0),
        "mlp": ra.ComponentKernel(0.0, 2.0, 0.0, 1.0)})
    doc = _toy_overlay(tmp_path, directions, params, 4)
    body = {k: v for k, v in doc.items() if k not in ("id", "label", "source", "created", "qualification")}
    assert doc["id"] == "ovl-" + hashlib.sha256(canonical(body).encode()).hexdigest()[:20]


def test_row_projection_matches_single_hook_per_row():
    rng = np.random.default_rng(1)
    y = mx.array(rng.standard_normal((3, 5, 8)).astype(np.float32))
    v = rng.standard_normal((3, 8)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    lam = np.array([1.0, 0.0, 0.6], np.float32)
    out = np.asarray(ra.row_projection(y, mx.array(lam), mx.array(v)))
    for b in range(3):
        hook = ra.ProjectionHooks._hook(float(lam[b]), v[b])
        np.testing.assert_allclose(out[b], np.asarray(hook(y[b:b + 1]))[0], rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(out[1], np.asarray(y[1]))


def test_serving_overlay_manager_per_row_matches_hooks(tmp_path):
    from mlx2.runtime.residual_ablation_overlays import AblationOverlayManager
    from mlx2.runtime.weight_overlays import bind_weight_overlay_rows, clear_weight_overlay_rows

    model = _model()
    layers = len(model.layers)
    hidden = model.layers[0].input_layernorm.weight.shape[0]
    directions = _toy_directions(layers, hidden)
    params = ra.AblationParameters(kernels={
        "attention": ra.ComponentKernel(1.1, 1.0, 0.6, 2.0),
        "mlp": ra.ComponentKernel(0.7, 1.0, 0.3, 2.0)})
    doc = _toy_overlay(tmp_path, directions, params, layers, checkpoint="tiny@0")
    ra.save_overlay(tmp_path / "ov", doc)
    tokens = mx.array([[3, 9, 4, 1, 7, 2], [5, 6, 2, 8, 1, 4]], mx.uint32)
    base = np.asarray(model(tokens, cache=model.make_cache()))
    with ra.ProjectionHooks(model, DECL).active(params.plan(directions, layers)):
        hooked = np.asarray(model(tokens, cache=model.make_cache()))
    manager = AblationOverlayManager(model, DECL, checkpoint="tiny@0")
    info = manager.load(tmp_path / "ov")
    with pytest.raises(ra.OverlayBindingError):
        ra.overlay_from_document(doc, fingerprint="1", layer_count=layers)
    manager.bind_uid(101, manager.acquire({"id": info["id"], "branch_id": "b1"}), "b1")
    bound = bind_weight_overlay_rows(model, [101, 202])
    try:
        mixed = np.asarray(model(tokens, cache=model.make_cache()))
    finally:
        clear_weight_overlay_rows(bound)
    np.testing.assert_allclose(mixed[0], hooked[0], rtol=2e-4, atol=2e-4)
    np.testing.assert_array_equal(mixed[1], base[1])
    r = manager.receipt(101)
    assert r["applied"] and r["lane_width"] == 2 and r["branch_id"] == "b1" and manager.receipt(202) is None
    # No bound rows: wrappers are inert.
    np.testing.assert_array_equal(np.asarray(model(tokens, cache=model.make_cache())), base)
    from mlx2.runtime.residual_ablation_overlays import OverlayInUse, OverlayNotLoaded
    with pytest.raises(OverlayInUse):
        manager.unload(info["id"])
    with pytest.raises(OverlayNotLoaded):
        manager.resolve({"id": "ovl-missing"})
    assert manager.load_document(doc)["id"] == info["id"]  # idempotent
    manager.release(101, info["id"])
    manager.unload(info["id"])
    # Temporary (preview) loads: reference counted, never demote a persistent load.
    manager.load_document(doc)
    assert manager.load_document(doc, transient=True)["transient"] is False
    manager.drop_transient(info["id"])
    assert manager.resolve(info["id"]) == info["id"]  # persistent overlay survives a preview
    manager.unload(info["id"])
    assert manager.load_document(doc, transient=True)["transient"] is True
    held = manager.acquire(info["id"])
    manager.drop_transient(held)
    assert manager.resolve(held) == held  # still pinned by a request
    manager.release(None, held)
    with pytest.raises(OverlayNotLoaded):
        manager.resolve(held)  # removed with its last holder
    manager.restore()
    np.testing.assert_array_equal(np.asarray(model(tokens, cache=model.make_cache())), base)


def test_overlay_http_endpoints_follow_spec(tmp_path):
    import threading
    from http.server import ThreadingHTTPServer
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    from test_server_quiesce import AdminEngine
    from mlx2.runtime.residual_ablation_overlays import AblationOverlayManager
    from mlx2.server import handler_for, validate_overlay_selection

    engine = AdminEngine()
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(engine, admin_token="private-admin"))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = "http://127.0.0.1:" + str(server.server_port)
    admin = {"Authorization": "Bearer private-admin", "Content-Type": "application/json"}

    def call(method, path, body=None, headers=None):
        data = None if body is None else json.dumps(body).encode()
        request = Request(base + path, data=data, method=method, headers=headers or {"Content-Type": "application/json"})
        try:
            with urlopen(request) as response:
                return response.status, json.load(response)
        except HTTPError as error:
            return error.code, None

    try:
        assert call("GET", "/v1/overlays")[0] == 404  # not enabled: client fails closed
        model = _model()
        layers = len(model.layers)
        hidden = model.layers[0].input_layernorm.weight.shape[0]
        params = ra.AblationParameters(kernels={
            "attention": ra.ComponentKernel(1.0, 1.0, 0.5, 2.0), "mlp": ra.ComponentKernel(0.0, 1.0, 0.0, 1.0)})
        doc = _toy_overlay(tmp_path, _toy_directions(layers, hidden), params, layers, checkpoint="tiny@fp1")
        engine.ablation_overlays = AblationOverlayManager(model, DECL, checkpoint="tiny@fp1")
        status, listing = call("GET", "/v1/overlays")
        assert status == 200 and listing["object"] == "overlay_list" and listing["api_version"] == 1
        assert listing["checkpoint_fingerprint"] == "fp1" and listing["data"] == []
        assert call("POST", "/v1/admin/overlays", {"overlay": doc})[0] in (401, 403)
        status, loaded = call("POST", "/v1/admin/overlays", {"overlay": doc}, admin)
        assert status == 200 and loaded == {"id": doc["id"], "loaded": True, "edits": doc["edits"]}
        assert call("POST", "/v1/admin/overlays", {"overlay": doc}, admin)[0] == 200  # idempotent
        foreign = ra.overlay_document(directions_path=tmp_path / "dirs", checkpoint="tiny@fp2",
                                      parameters=params, layers=layers, label="x")
        assert call("POST", "/v1/admin/overlays", {"overlay": foreign}, admin)[0] == 409
        assert call("POST", "/v1/admin/overlays", {"overlay": {**doc, "edits": 999}}, admin)[0] == 422
        assert [row["id"] for row in call("GET", "/v1/overlays")[1]["data"]] == [doc["id"]]
        assert call("DELETE", f"/v1/admin/overlays/{doc['id']}", headers=admin)[0] == 200
        assert call("DELETE", f"/v1/admin/overlays/{doc['id']}", headers=admin)[0] == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join(2)
    assert validate_overlay_selection({"id": "ovl-1", "branch_id": "branch:a"})
    for bad in ({"id": "ovl-1", "extra": 1}, {"id": ""}, "bad id", {"id": "x" * 200}):
        with pytest.raises(ValueError):
            validate_overlay_selection(bad)


def test_bake_key_pattern_skips_mtp_and_refuses_duplicates():
    keys = ["model.language_model.layers.0.mlp.down_proj.weight", "mtp.layers.0.mlp.down_proj.weight",
            "language_model.model.layers.1.self_attn.o_proj.weight", "vision_tower.blocks.0.mlp.down_proj.weight"]
    found = ra._target_keys(keys, DECL)
    assert set(found) == {keys[0], keys[2]}
    with pytest.raises(ValueError, match="both map"):
        ra._target_keys(["model.layers.0.mlp.down_proj.weight",
                         "language_model.model.layers.0.mlp.down_proj.weight"], DECL)


def test_overlay_rejects_non_unit_directions(tmp_path):
    directions = _toy_directions(4, 8)
    directions.vectors[2] *= 3.0
    params = ra.AblationParameters(kernels={
        "attention": ra.ComponentKernel(1.0, 1.0, 0.5, 2.0), "mlp": ra.ComponentKernel(0.0, 1.0, 0.0, 1.0)})
    doc = _toy_overlay(tmp_path, directions, params, 4)
    with pytest.raises(ValueError, match="unit vectors"):
        ra.overlay_from_document(doc, fingerprint="abc", layer_count=4)


def _tiny_kimi():
    from test_kimi_linear import args as kimi_args
    from mlx2.adapters.kimi_linear import KimiLinearAdapter, parse_policy
    from mlx2.runtime.models.kimi_linear import Model as KimiModel

    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.random.seed(118)
    model = KimiModel(kimi_args())
    model.eval()
    model.configure(parse_policy({"kda": "reference"}))
    return model, ra.declaration_for(KimiLinearAdapter), previous


def test_kimi_declaration_hooks_equal_weight_edit_on_moe_and_restore():
    model, decl, previous = _tiny_kimi()
    try:
        linears = ra.resolve_linears(model, decl)
        paths = {(layer, path) for layer, _component, path, *_ in linears}
        assert (0, "mlp.down_proj") in paths
        assert {(1, "mlp.switch_mlp.down_proj"), (1, "mlp.shared_experts.down_proj")} <= paths
        assert all((layer, "attention.o_proj") in paths for layer in range(4))
        stacked = [m for layer, _c, path, _p, _a, m in linears if path == "mlp.switch_mlp.down_proj"]
        assert stacked and stacked[0]["weight"].ndim == 3
        tokens = mx.array([[1, 2, 3, 4, 5, 6]])
        base = np.asarray(model(tokens, cache=model.make_cache()))
        hidden = 32
        rng = np.random.default_rng(5)
        edits = []
        for layer in range(4):
            v = rng.standard_normal(hidden).astype(np.float32)
            v /= np.linalg.norm(v)
            edits += [(layer, "attention", 0.9, v), (layer, "mlp", 0.7, v)]
        with ra.ProjectionHooks(model, decl).active(edits):
            hooked = np.asarray(model(tokens, cache=model.make_cache()))
        with ra.materialize(model, decl, edits, "none") as swapped:
            assert swapped == len(linears)
            edited = np.asarray(model(tokens, cache=model.make_cache()))
        restored = np.asarray(model(tokens, cache=model.make_cache()))
        np.testing.assert_array_equal(restored, base)
        assert not np.allclose(hooked, base, atol=1e-3)
        np.testing.assert_allclose(hooked, edited, rtol=5e-4, atol=5e-4)
        with ra.LayerOutputCapture(model).active() as capture:
            model(tokens, cache=model.make_cache())
            assert capture.take().shape == (1, 4, hidden)
    finally:
        mx.set_default_device(previous)


def test_kimi_checkpoint_keys_use_checkpoint_spelling():
    from mlx2.adapters.kimi_linear import KimiLinearAdapter

    decl = ra.declaration_for(KimiLinearAdapter)
    keys = ["model.layers.0.mlp.down_proj.weight", "model.layers.3.self_attn.o_proj.weight",
            "model.layers.3.mlp.switch_mlp.down_proj.weight", "model.layers.3.mlp.shared_experts.down_proj.weight",
            "model.layers.3.mlp.switch_mlp.gate_proj.weight", "model.layers.3.mlp.gate.weight"]
    found = ra._target_keys(keys, decl)
    assert set(found) == set(keys[:4])
    assert found[keys[1]][:2] == (3, "attention") and found[keys[2]][:2] == (3, "mlp")


def test_dense_bf16_bake_requantizes_stacked_experts(tmp_path):
    rng = np.random.default_rng(3)
    source = tmp_path / "src"
    source.mkdir()
    dense = mx.array(rng.standard_normal((32, 64)).astype(np.float32))
    experts = mx.array(rng.standard_normal((4, 32, 64)).astype(np.float32))
    tensors = {}
    for name, value in (("model.layers.1.self_attn.o_proj", dense), ("model.layers.1.mlp.switch_mlp.down_proj", experts)):
        w, scales, biases = mx.quantize(value, group_size=64, bits=8)
        tensors.update({name + ".weight": w, name + ".scales": scales, name + ".biases": biases})
    mx.save_safetensors(str(source / "model.safetensors"), tensors, metadata={"format": "mlx"})
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {k: "model.safetensors" for k in tensors}}))
    (source / "config.json").write_text(json.dumps({"quantization": {"group_size": 64, "bits": 8, "mode": "affine"}}))
    from mlx2.adapters.kimi_linear import KimiLinearAdapter

    decl = ra.declaration_for(KimiLinearAdapter)
    v = np.eye(32, dtype=np.float32)[0]
    receipt = ra.bake_checkpoint(source, tmp_path / "out", decl, [(1, "attention", 1.0, v), (1, "mlp", 1.0, v)],
                                 "none", metadata={}, edited_precision="dense-bf16")
    stored = {r["key"]: r["stored"] for r in receipt["edited_tensors"]}
    assert stored == {"model.layers.1.self_attn.o_proj.weight": "bf16",
                      "model.layers.1.mlp.switch_mlp.down_proj.weight": "requant"}
    out = mx.load(str(tmp_path / "out" / "model.safetensors"))
    assert "model.layers.1.mlp.switch_mlp.down_proj.scales" in out
    assert "model.layers.1.self_attn.o_proj.scales" not in out
    roundtrip = mx.dequantize(out["model.layers.1.mlp.switch_mlp.down_proj.weight"],
                              out["model.layers.1.mlp.switch_mlp.down_proj.scales"],
                              out["model.layers.1.mlp.switch_mlp.down_proj.biases"], group_size=64, bits=8)
    assert float(mx.abs(mx.array(v) @ roundtrip).max().item()) < 0.1  # direction mostly removed at 8-bit


def test_kimi_checkpoint_name_never_labels_a_derived_bake_as_its_source(tmp_path):
    import shutil
    from pathlib import Path

    from mlx2.adapters.kimi_linear import checkpoint_name

    source = Path("~/Models/Kimi-Linear-48B-A3B-Instruct-8bit/config.json")
    if not source.is_file():
        pytest.skip("Kimi q8 config not on this host")
    derived = tmp_path / "Kimi-derived"
    derived.mkdir()
    config = json.loads(source.read_text())
    config["mlx2_residual_ablation"] = {"source_checkpoint": "Kimi@abc"}
    (derived / "config.json").write_text(json.dumps(config))
    assert checkpoint_name(derived) == "Kimi-derived@derived-from:Kimi@abc"
    plain = tmp_path / "plain"
    plain.mkdir()
    shutil.copy(source, plain / "config.json")
    assert checkpoint_name(plain).startswith("mlx-community/Kimi-Linear-48B-A3B-Instruct-8bit@")
