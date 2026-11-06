from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models.exl3 import Exl3Linear, install_exl3_experts, validate_metadata
from mlx2.runtime.models.exl3_reference import decode_inner

META = {
    "version": 1,
    "codebook": "mcg",
    "window_bits": 16,
    "bits_per_weight": 2.5,
    "execution": "reference",
}


def _model_and_weights(
    *, layers=2, seed=21, hidden=128, intermediate=128, experts=4, topk=2
):
    from mlx2.runtime.models.qwen36_35b import Model, ModelArgs
    from mlx2.runtime.models import qwen3_next
    from mlx.utils import tree_flatten

    mx.random.seed(seed)
    text = dict(
        model_type="qwen3_5_moe_text",
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=64,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=3,
        full_attention_interval=2,
        mtp_num_hidden_layers=0,
        partial_rotary_factor=0.5,
        rope_parameters=None,
        max_position_embeddings=256,
        num_experts=experts,
        num_experts_per_tok=topk,
        moe_intermediate_size=intermediate,
        shared_expert_intermediate_size=intermediate,
    )
    old = qwen3_next._MOE_SHARED_IN_GATHER
    qwen3_next._MOE_SHARED_IN_GATHER = False
    try:
        model = Model(
            ModelArgs.from_dict({"model_type": "qwen3_5_moe", "text_config": text})
        )
    finally:
        qwen3_next._MOE_SHARED_IN_GATHER = old
    weights = dict(tree_flatten(model.parameters()))
    roots = [path for path, m in model.named_modules() if path.endswith(".switch_mlp")]
    rng = np.random.default_rng(seed)
    for root in roots:
        for key in list(weights):
            if key.startswith(root + "."):
                del weights[key]
        for proj in ("gate_proj", "up_proj", "down_proj"):
            i, o = (
                (intermediate, hidden)
                if proj == "down_proj"
                else (hidden, intermediate)
            )
            weights[f"{root}.{proj}.trellis"] = mx.array(
                rng.integers(0, 65536, (experts, i // 16, o // 16, 40), np.uint16)
            )
            weights[f"{root}.{proj}.suh"] = mx.full((experts, i), 0.15, mx.float16)
            weights[f"{root}.{proj}.svh"] = mx.full((experts, o), 0.15, mx.float16)
    return model, weights


def test_converter_fixture_inner_and_projection():
    root = Path(__file__).resolve().parents[1]
    f = mx.load(
        str(root / "scripts/sushi_probe/fixtures/exl3_k2p5_mcg_linear.safetensors")
    )
    np.testing.assert_array_equal(
        decode_inner(np.asarray(f["trellis"])), np.asarray(f["inner"])
    )
    p = Exl3Linear(f["trellis"][None], f["suh"][None], f["svh"][None])
    p.eval()
    x = mx.arange(128, dtype=mx.float32)[None] / 10000
    got = p(x, mx.array([0], mx.uint32))
    ref = x @ f["public"]
    np.testing.assert_allclose(
        np.asarray(got), np.asarray(ref), atol=0.0005, rtol=0.005
    )
    with pytest.raises(ValueError, match="outside"):
        p(x, mx.array([1], mx.uint32))


def test_full_moe_load_forward_cache_and_packed_storage():
    model, weights = _model_and_weights()
    receipt = install_exl3_experts(model, weights, META)
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    tokens = mx.array([[3, 4, 1, 8]], mx.uint32)
    cache = model.make_cache()
    logits = model(tokens, cache=cache)
    mx.eval(logits, [c.state for c in cache])
    assert bool(mx.all(mx.isfinite(logits)).item())
    assert receipt["blocks"] == 2 and receipt["qualification"] == "unqualified"
    assert receipt["packed_weight_bytes"] == 2 * 3 * (
        4 * 8 * 8 * 40 * 2 + 2 * 4 * 128 * 2
    )
    next_logits = model(mx.array([[11]], mx.uint32), cache=cache)
    mx.eval(next_logits, [c.state for c in cache])
    assert bool(mx.all(mx.isfinite(next_logits)).item())
    # Repeated independent ordinary decode has identical logits/state.
    again = model(tokens, cache=model.make_cache())
    mx.eval(again)
    np.testing.assert_array_equal(np.asarray(logits), np.asarray(again))
    for _, module in model.named_modules():
        if hasattr(module, "trellis"):
            assert "weight" not in module  # no hidden dense resident bank


def test_bad_metadata_geometry_or_partial_pack_refuses_atomically():
    for mutation in (
        {"window_bits": 12},
        {"bits_per_weight": 3},
        {"version": True},
        {"execution": "auto"},
    ):
        with pytest.raises(ValueError):
            validate_metadata({**META, **mutation})
    model, weights = _model_and_weights()
    original = model.layers[0].mlp.switch_mlp
    key = next(k for k in weights if k.endswith(".trellis"))
    bad = dict(weights)
    bad[key] = bad[key][..., :32]
    with pytest.raises(ValueError):
        install_exl3_experts(model, bad, META)
    assert model.layers[0].mlp.switch_mlp is original
    bad = {
        **weights,
        "language_model.model.layers.0.mlp.switch_mlp.gate_up_proj.weight": mx.zeros(
            (4, 256, 128)
        ),
    }
    with pytest.raises(ValueError, match="mixed"):
        install_exl3_experts(model, bad, META)
    assert model.layers[0].mlp.switch_mlp is original

    bad = {
        k: v
        for k, v in weights.items()
        if not k.startswith("language_model.model.layers.1.mlp.switch_mlp.")
    }
    with pytest.raises(ValueError, match="every"):
        install_exl3_experts(model, bad, META)
    assert model.layers[0].mlp.switch_mlp is original


def test_adapter_load_hook_identity_and_observed_use(tmp_path, monkeypatch):
    from mlx2.adapters import qwen36_35b as adapter_module
    from test_disk_weight_streaming import _stub_tokenizer, _write_checkpoint

    model, weights = _model_and_weights()
    weight_map, records = _write_checkpoint(
        tmp_path, weights, lambda k: ".switch_mlp." in k
    )
    text = dict(vars(model.language_model.args))
    config = {"model_type": "qwen3_5_moe", "text_config": text, "mlx2_exl3": META}
    artifact = {
        "config": config,
        "weight_map": weight_map,
        "has_mtp": False,
        "mtp_tensor_count": 0,
        "identity": {"path": str(tmp_path), "fingerprint": "0" * 64, "files": records},
    }
    monkeypatch.setattr(adapter_module, "inspect_artifact", lambda _: artifact)
    monkeypatch.setattr(
        adapter_module,
        "configure_environment",
        lambda kernels, *a, **k: {
            "MLX_LM_COMPILED_DECODE": "0",
            "MLX_QWEN36_FUSED_GDN_DECODE": "1" if kernels.get("fused_gdn_decode") else "0",
        },
    )
    _stub_tokenizer(monkeypatch, adapter_module)
    adapter = adapter_module.Qwen3635BA3BAdapter(str(tmp_path))
    try:
        assert adapter.identity["fingerprint"] != "0" * 64
        assert adapter.moe_nax_gather == "off"
        assert adapter._kernels.get("moe_routed_decode", "off") == "off"
        assert adapter._kernels.get("moe_topk_fold", "off") == "off"
        assert adapter.diagnostics()["exl3"]["observed_blocks"]
        assert (
            adapter.descriptor.metadata["expert_format_qualification"] == "unqualified"
        )
        tokens = mx.array([[3, 4, 1, 8]], mx.uint32)
        out = adapter.model(tokens, cache=adapter.model.make_cache())
        mx.eval(out)
        assert bool(mx.all(mx.isfinite(out)).item())
        captured = []

        def sink(index, tokens, hidden, cache):
            logits = adapter.model.logits(hidden)
            mx.eval(logits)
            captured.append(np.asarray(logits))

        capture = adapter.capture_teacher_windows([[3, 4, 1, 8]], sink)
        np.testing.assert_array_equal(captured[0], np.asarray(out))
        assert capture["model_revision"] != adapter.identity["fingerprint"]
        assert (
            capture["execution_binding"]["artifact"] == adapter.identity["fingerprint"]
        )
        blocks = [
            m for _, m in adapter.model.named_modules() if hasattr(m, "reference_calls")
        ]
        assert len(blocks) == 2 and all(m.reference_calls > 0 for m in blocks)
        reference_identity = adapter.identity["fingerprint"]
    finally:
        adapter.close()
    artifact["config"] = {**config, "mlx2_exl3": {**META, "execution": "fused"}}
    adapter = adapter_module.Qwen3635BA3BAdapter(str(tmp_path))
    try:
        assert adapter.identity["fingerprint"] != reference_identity
    finally:
        adapter.close()
    for policy in (
        {"moe_routed_decode": "gate_up_down_shared"},
        {"moe_topk_fold": "launch"},
        {"moe_window_batch_decode": True},
        {"moe_window_verify": True},
        {"moe_nax_gather": "fused"},
    ):
        with pytest.raises(ValueError, match="affine MoE"):
            adapter_module.Qwen3635BA3BAdapter(str(tmp_path), execution_policy=policy)
    artifact["config"] = {k: v for k, v in config.items() if k != "mlx2_exl3"}
    with pytest.raises(ValueError, match="explicit"):
        adapter_module.Qwen3635BA3BAdapter(str(tmp_path))


def test_inspector_refuses_unsupported_embedded_mtp_pack(tmp_path):
    import json
    from test_qwen36_35b_port import make_artifact
    from mlx2.adapters.qwen36_35b import inspect_artifact

    root = make_artifact(tmp_path, mtp=True)
    config = json.loads((root / "config.json").read_text())
    config["mlx2_exl3"] = META
    (root / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="ordinary artifacts"):
        inspect_artifact(root)
