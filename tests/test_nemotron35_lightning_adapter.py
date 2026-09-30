"""CPU-only artifact, sidecar, and model-key checks for Lightning 3.5."""
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from mlx2.adapters.nemotron35_lightning_memory import LightningCacheBudget

TARGET = Path(os.environ.get(
    "MLX2_NEMOTRON35_TARGET",
    str(Path.home() / "mlx-models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16-mlx-8Bit"),
))


def _require_local_target():
    if not (TARGET / "mlx2-nemotron35-mtp.json").is_file():
        pytest.skip("pinned Lightning 3.5 target and source MTP shard absent")


def test_pinned_target_dispatch_advertises_only_verified_mtp():
    _require_local_target()
    script = """
import sys
from mlx2.adapters.registry import inspect_model, resolve_adapter
from mlx2.contracts import Capability
r = inspect_model(sys.argv[1])
assert r.adapter_type.__name__ == 'Nemotron35LightningAdapter'
assert r.default_route == 'ordinary'
assert r.artifact['mtp_tensor_count'] == 270
assert Capability.MTP in r.descriptor.capabilities
assert resolve_adapter(sys.argv[1], mtp=True) is r.adapter_type
assert 'mlx.core' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(TARGET)],
        capture_output=True, text=True, check=False,
        env={**os.environ, "PYTHONPATH": "src"},
    )
    assert result.returncode == 0, result.stderr


def test_missing_or_wrong_sidecar_fails_closed(tmp_path, monkeypatch):
    _require_local_target()
    from mlx2.adapters import nemotron35_lightning
    from mlx2.adapters.registry import inspect_model, resolve_adapter
    from mlx2.contracts import Capability

    # Keep the pinned target in place; represent an absent attachment without
    # copying 33 GB of shards or loosening the inspector's path boundary.
    with monkeypatch.context() as patch:
        patch.setattr(nemotron35_lightning, "_inspect_mtp", lambda _path: None)
        ordinary = inspect_model(TARGET)
        assert not ordinary.artifact["has_mtp"]
        assert Capability.MTP not in ordinary.descriptor.capabilities
        assert resolve_adapter(TARGET, mtp=False) is ordinary.adapter_type
        with pytest.raises(ValueError, match="no implemented native MTP"):
            resolve_adapter(TARGET, mtp=True)

    bad = json.loads((TARGET / "mlx2-nemotron35-mtp.json").read_text())
    bad["source_sha256"] = "0" * 64
    (tmp_path / "mlx2-nemotron35-mtp.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="provenance mismatch"):
        nemotron35_lightning._inspect_mtp(tmp_path)


def test_cpu_mtp_module_matches_source_keys_and_steps():
    _require_local_target()
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from mlx2.runtime.models.nemotron_h import Model, ModelArgs

    mx.set_default_device(mx.cpu)
    config = json.loads((TARGET / "config.json").read_text())
    config.update(
        vocab_size=64, hidden_size=64, intermediate_size=64,
        layers_block_type=["mamba", "attention", "moe"], num_hidden_layers=3,
        num_attention_heads=2, num_key_value_heads=1, head_dim=32,
        mamba_num_heads=2, mamba_head_dim=32, ssm_state_size=32,
        conv_kernel=3, n_groups=1, n_routed_experts=2,
        num_experts_per_tok=1, moe_intermediate_size=64,
        moe_shared_expert_intermediate_size=64,
    )
    model = Model(ModelArgs.from_dict(config))
    actual = {key for key, _ in tree_flatten(model.parameters()) if key.startswith("mtp.")}
    manifest = json.loads((TARGET / "mlx2-nemotron35-mtp.json").read_text())
    source = (TARGET / manifest["path"]).resolve()
    with source.open("rb") as stream:
        header_length = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_length))
    expected = set(header) - {"__metadata__"}
    for prefix in {key.rsplit(".experts.", 1)[0] for key in expected if ".experts." in key}:
        for old, new in (("up_proj", "fc1"), ("down_proj", "fc2")):
            expected = {
                key for key in expected
                if not (key.startswith(prefix + ".experts.") and key.endswith("." + old + ".weight"))
            }
            expected.add(prefix + ".switch_mlp." + new + ".weight")
    assert actual == expected
    assert len(actual) == 16

    _hidden, seed = model.mtp_backbone(mx.array([[1, 2]], dtype=mx.int32), model.make_cache())
    logits, post = model.mtp_step(
        seed[:, -1:], mx.array([[3]], dtype=mx.int32), model.make_mtp_cache()
    )
    mx.eval(logits, post)
    assert logits.shape == (1, 1, 64)
    assert post.shape == (1, 1, 64)


def test_mamba_time_step_clamp_follows_the_config_like_nvidias_forward():
    """transformers #48989: NVIDIA's forward clamps dt to ``time_step_limit``
    (default (0, inf)); ``time_step_min`` only initializes the dt bias."""
    from mlx2.adapters.nemotron35_lightning import _runtime_config

    config = {"time_step_min": 0.001, "time_step_max": 0.1, "time_step_limit": None}
    assert _runtime_config(config)["time_step_limit"] == (0.0, float("inf"))
    assert _runtime_config({**config, "time_step_limit": [0.0, 5.0]})["time_step_limit"] == (0.0, 5.0)


def test_mamba_time_step_is_not_floored_on_the_checkpoint_cpu():
    _require_local_target()
    import mlx.core as mx

    from mlx2.adapters.nemotron35_lightning import _runtime_config
    from mlx2.runtime.models.nemotron_h import ModelArgs
    from mlx2.runtime.models.ssm import compute_dt

    mx.set_default_device(mx.cpu)
    config = json.loads((TARGET / "config.json").read_text())
    args = ModelArgs.from_dict(_runtime_config(config))
    assert args.time_step_limit == (0.0, float("inf"))
    # A slow head (dt_bias -9.7 on this checkpoint) keeps its ~6e-5 step, and
    # the configured 0.1 maximum does not clamp large steps.
    dt = compute_dt(mx.array([0.0, 2.0]), mx.array([-9.688, 0.0]), args.time_step_limit)
    mx.eval(dt)
    assert 5e-5 < float(dt[0]) < 1e-4
    assert float(dt[1]) > config["time_step_max"]


def test_cpu_target_quantization_matches_pinned_layer_keys():
    _require_local_target()
    import mlx.core as mx
    from mlx import nn
    from mlx.utils import tree_flatten

    from mlx2.runtime.models.nemotron_h import Model, ModelArgs

    mx.set_default_device(mx.cpu)
    config = json.loads((TARGET / "config.json").read_text())
    config.update(
        vocab_size=64, hidden_size=64, intermediate_size=64,
        layers_block_type=["mamba", "attention", "moe"], num_hidden_layers=3,
        num_attention_heads=2, num_key_value_heads=1, head_dim=32,
        mamba_num_heads=2, mamba_head_dim=32, ssm_state_size=32,
        conv_kernel=3, n_groups=1, n_routed_experts=2,
        num_experts_per_tok=1, moe_intermediate_size=64,
        moe_shared_expert_intermediate_size=64,
    )
    model = Model(ModelArgs.from_dict(config))
    index = json.loads((TARGET / "model.safetensors.index.json").read_text())["weight_map"]
    representative_layers = {0: 0, 1: 5, 2: 6}

    def source_path(name):
        for tiny, actual in representative_layers.items():
            prefix = f"backbone.layers.{tiny}."
            if name.startswith(prefix):
                return f"backbone.layers.{actual}." + name[len(prefix):]
        return name

    nn.quantize(
        model, group_size=64, bits=8, mode="affine",
        class_predicate=lambda name, module: hasattr(module, "to_quantized")
        and source_path(name) + ".scales" in index,
    )
    names = {key for key, _ in tree_flatten(model.parameters())}
    for tiny, actual in representative_layers.items():
        target_prefix = f"backbone.layers.{actual}."
        model_prefix = f"backbone.layers.{tiny}."
        expected = {key.removeprefix(target_prefix) for key in index if key.startswith(target_prefix)}
        observed = {key.removeprefix(model_prefix) for key in names if key.startswith(model_prefix)}
        assert observed == expected


def test_lightning_budget_charges_recurrent_and_draft_state():
    from types import SimpleNamespace

    args = SimpleNamespace(
        hybrid_override_pattern=list("M" * 23 + "E" * 23 + "*" * 6),
        num_key_value_heads=2, head_dim=128, mamba_num_heads=64,
        mamba_head_dim=64, ssm_state_size=128, n_groups=8, conv_kernel=4,
    )
    ordinary = LightningCacheBudget.from_config(args, mtp=False)
    draft = LightningCacheBudget.from_config(args, mtp=True)
    assert ordinary.recurrent_live_bytes > 0
    assert draft.project(2048) > ordinary.project(2048)
    assert draft.project_pool(2048, resident_lanes=2, running_lanes=1) > draft.project(2048)
    assert draft.as_dict()["schema"] == "nemotron35-lightning-cache-geometry-v1"
