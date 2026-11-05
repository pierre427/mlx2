"""Tiny Qwen3.6 MoE from an upstream-layout EXL3 checkpoint, against the
dense model built from the same weights decoded on the host."""

import json

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from mlx2.adapters.qwen36_35b import exl3_module_path
from mlx2.runtime.models.exl3 import Exl3DenseLinear, Exl3SwitchGLU
from mlx2.runtime.models.exl3_reference import effective_weight
from mlx2.runtime.trellis.upstream import (
    CODEBOOKS,
    ShardReader,
    install_upstream,
    load_beside,
    load_stored,
    plan_checkpoint,
)

EXPERTS = 4
FORMAT = {"quant_method": "exl3", "version": "0.0.29", "bits": 3.0, "head_bits": 6,
          "out_scales": "always", "codebook": "mcg"}
# Projections the upstream converter stores as trellis; everything else is stored.
TRELLIS = (".q_proj", ".k_proj", ".v_proj", ".o_proj", ".in_proj_qkv", ".in_proj_z",
           ".out_proj", ".shared_expert.gate_proj", ".shared_expert.up_proj",
           ".shared_expert.down_proj")
WORDS = {"q_proj": 64, "in_proj_qkv": 48, "lm_head": 96, "experts": 32}


def _model():
    from mlx2.runtime.models.qwen36_35b import Model, ModelArgs

    text = dict(
        model_type="qwen3_5_moe_text", hidden_size=128, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=64,
        vocab_size=256, linear_num_key_heads=2, linear_num_value_heads=4,
        linear_key_head_dim=64, linear_value_head_dim=64, linear_conv_kernel_dim=3,
        full_attention_interval=2, mtp_num_hidden_layers=0, partial_rotary_factor=0.5,
        rope_parameters=None, max_position_embeddings=256, num_experts=EXPERTS,
        num_experts_per_tok=2, moe_intermediate_size=128,
        shared_expert_intermediate_size=128, tie_word_embeddings=False,
    )
    return Model(ModelArgs.from_dict({"model_type": "qwen3_5_moe", "text_config": text}))


@pytest.fixture(autouse=True)
def _qwen36_profile(monkeypatch):
    # The Qwen3.6 profile pins split gate/up tables and an unfolded shared
    # expert; construction and sanitize both read these switches.
    from mlx2.runtime.models import qwen3_next

    monkeypatch.setattr(qwen3_next, "_MOE_SHARED_IN_GATHER", False)
    monkeypatch.setattr(qwen3_next, "_MOE_FUSED_GATE_UP", False)


def _upstream(name):
    return "model.language_model." + name.removeprefix("language_model.model.") if name.startswith(
        "language_model.model.") else name.removeprefix("language_model.")


def _trellis(rng, i, o, words):
    marker = CODEBOOKS["mcg"] - 2**32
    return {
        "trellis": rng.integers(-32768, 32768, (i // 16, o // 16, words), np.int16),
        "suh": (rng.normal(0, 1, i) * (0.5 / np.sqrt(i))).astype(np.float16),
        "svh": np.sign(rng.normal(size=o)).astype(np.float16),
        "mcg": np.int32(marker),
    }


def _checkpoints(tmp_path):
    """Write the EXL3 checkpoint and the dense checkpoint decoded from it."""
    rng = np.random.default_rng(3)
    params = dict(tree_flatten(_model().parameters()))
    exl3, dense = {}, {}

    def add_trellis(key, i, o, words):
        t = _trellis(rng, i, o, words)
        exl3.update({f"{key}.{part}": mx.array(v) for part, v in t.items()})
        w = effective_weight(t["trellis"], t["suh"], t["svh"], "mcg")
        dense[f"{key}.weight"] = mx.array(w.T.astype(np.float32))

    for name, value in params.items():
        module = name.rsplit(".", 1)[0]
        if name.endswith(".weight") and (module.endswith(TRELLIS) or module == "language_model.lm_head"):
            o, i = value.shape
            words = next((n for k, n in WORDS.items() if module.endswith(k)), 64)
            add_trellis(_upstream(module), i, o, words)
        elif ".switch_mlp." in name:
            e, o, i = value.shape
            root, proj = module.split(".switch_mlp.")
            for expert in range(e):
                add_trellis(f"{_upstream(root)}.experts.{expert}.{proj}", i, o, WORDS["experts"])
        else:
            # Router, norms, GDN gates/convolution, embeddings: stored as is.
            stored = mx.array(rng.normal(0, 0.05, value.shape).astype(np.float32)).astype(value.dtype)
            if name.endswith("norm.weight") or "layernorm" in name:
                stored = mx.ones_like(value)
            exl3[_upstream(name)] = dense[_upstream(name)] = stored
    for directory, tensors, config in (
        (tmp_path / "exl3", exl3, {"model_type": "qwen3_5_moe", "quantization_config": FORMAT}),
        (tmp_path / "dense", dense, {"model_type": "qwen3_5_moe"}),
    ):
        directory.mkdir()
        mx.save_safetensors(str(directory / "model.safetensors"), tensors)
        (directory / "model.safetensors.index.json").write_text(json.dumps(
            {"metadata": {}, "weight_map": {k: "model.safetensors" for k in tensors}}))
        (directory / "config.json").write_text(json.dumps(config))
    return tmp_path / "exl3", tmp_path / "dense"


def _load_exl3(path, execution="reference"):
    weight_map = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    plan = plan_checkpoint(path, weight_map, num_experts=EXPERTS)
    model = _model()
    read = ShardReader(path, weight_map)
    receipt = install_upstream(model, read, plan, exl3_module_path, execution=execution)
    load_beside(model, receipt, model.sanitize(model.shard_prune(load_stored(read, plan))))
    model.eval()
    return model, receipt


def _load_dense(path):
    model = _model()
    weights = model.sanitize(mx.load(str(path / "model.safetensors")))
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    return model


def _logits(model, tokens):
    cache = model.make_cache()
    return model(mx.array([tokens]), cache=cache).astype(mx.float32)


@pytest.mark.parametrize("execution", ["reference", "rows", "fused", "fused_nax", "nax"])
def test_exl3_model_matches_its_decoded_dense_model(tmp_path, execution):
    exl3_path, dense_path = _checkpoints(tmp_path)
    model, receipt = _load_exl3(exl3_path, execution)
    assert receipt["execution"] == execution
    reference = _load_dense(dense_path)
    assert receipt["installed_switches"] == 2 and receipt["installed_linears"] == 14
    assert receipt["rates"] == {"mcg/k2": 6, "mcg/k3": 1, "mcg/k4": 12, "mcg/k6": 1}
    kinds = {type(m) for _, m in model.named_modules()}
    assert Exl3DenseLinear in kinds and Exl3SwitchGLU in kinds
    tokens = [3, 17, 200, 45, 9, 101, 77, 250]
    got, ref = _logits(model, tokens), _logits(reference, tokens)
    assert bool(mx.all(mx.isfinite(got)).item())
    scale = float(mx.max(mx.abs(ref)).item())
    assert float(mx.max(mx.abs(got - ref)).item()) <= 1e-2 * scale
    assert mx.array_equal(mx.argmax(got, -1), mx.argmax(ref, -1))
    dense = [m for _, m in model.named_modules() if isinstance(m, Exl3DenseLinear)]
    assert all(m.calls >= 1 and m.rows >= len(tokens) for m in dense)
    switches = [m for _, m in model.named_modules() if isinstance(m, Exl3SwitchGLU)]
    calls = "reference_calls" if execution == "reference" else "rows_calls"
    assert all(getattr(m, calls) >= 1 and m.fused_calls == 0 for m in switches)


def test_incremental_decode_matches_the_full_forward(tmp_path):
    exl3_path, _ = _checkpoints(tmp_path)
    model, _ = _load_exl3(exl3_path)
    tokens = [5, 81, 33, 190, 12]
    full = _logits(model, tokens)
    cache = model.make_cache()
    model(mx.array([tokens[:3]]), cache=cache)
    steps = [model(mx.array([[t]]), cache=cache).astype(mx.float32)[0, -1] for t in tokens[3:]]
    for step, row in zip(steps, (3, 4)):
        scale = float(mx.max(mx.abs(full[0, row])).item())
        assert float(mx.max(mx.abs(step - full[0, row])).item()) <= 1e-2 * scale


def test_coverage_gaps_refuse(tmp_path):
    exl3_path, _ = _checkpoints(tmp_path)
    weight_map = json.loads((exl3_path / "model.safetensors.index.json").read_text())["weight_map"]
    plan = plan_checkpoint(exl3_path, weight_map, num_experts=EXPERTS)
    model = _model()
    read = ShardReader(exl3_path, weight_map)
    receipt = install_upstream(model, read, plan, exl3_module_path)
    stored = model.sanitize(model.shard_prune(load_stored(read, plan)))
    gate = next(k for k in stored if k.endswith(".mlp.gate.weight"))
    with pytest.raises(ValueError, match="missing"):
        load_beside(model, receipt, {k: v for k, v in stored.items() if k != gate})
    with pytest.raises(ValueError, match="unexpected"):
        load_beside(model, receipt, {**stored, "language_model.ghost.weight": mx.zeros((1,))})


def test_module_geometry_mismatch_refuses_before_swapping(tmp_path):
    exl3_path, _ = _checkpoints(tmp_path)
    weight_map = json.loads((exl3_path / "model.safetensors.index.json").read_text())["weight_map"]
    plan = plan_checkpoint(exl3_path, weight_map, num_experts=EXPERTS)
    model = _model()
    before = {p: type(m) for p, m in model.named_modules()}

    def wrong(key):
        path = exl3_module_path(key)
        return path.replace(".q_proj", ".o_proj") if path.endswith(".q_proj") else path

    with pytest.raises(ValueError, match="disagrees with model geometry"):
        install_upstream(model, ShardReader(exl3_path, weight_map), plan, wrong)
    assert {p: type(m) for p, m in model.named_modules()} == before
