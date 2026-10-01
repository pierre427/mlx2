"""CPU/static Qwen3.6 port tests; these never load production tensors."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

import pytest

from mlx2.adapters.qwen36_35b import (
    CACHE_LAYOUT,
    Qwen3635BA3BAdapter,
    configure_environment,
    descriptor_for,
    inspect_artifact,
)
from mlx2.adapters.registry import inspect_model
from mlx2.contracts import Capability


CONFIG = {
    "model_type": "qwen3_5_moe",
    "text_config": {
        "num_hidden_layers": 40,
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "full_attention_interval": 4,
        "vocab_size": 248320,
        "num_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 512,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "mtp_num_hidden_layers": 1,
        "max_position_embeddings": 262144,
    },
}


def make_artifact(root: Path, *, mtp: bool = False) -> Path:
    (root / "config.json").write_text(json.dumps(CONFIG))
    weights = {"language_model.model.embed_tokens.weight": "model.safetensors"}
    if mtp:
        names = [
            "fc.weight",
            "norm.weight",
            "pre_fc_norm_embedding.weight",
            "pre_fc_norm_hidden.weight",
            "layers.0.self_attn.q_proj.weight",
            "layers.0.self_attn.k_proj.weight",
            "layers.0.self_attn.v_proj.weight",
            "layers.0.self_attn.o_proj.weight",
            "layers.0.mlp.gate.weight",
            "layers.0.mlp.switch_mlp.gate_proj.weight",
            "layers.0.mlp.switch_mlp.up_proj.weight",
            "layers.0.mlp.switch_mlp.down_proj.weight",
        ]
        weights.update(
            {f"language_model.mtp.{name}": "model.safetensors" for name in names}
        )
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weights})
    )
    (root / "model.safetensors").write_bytes(b"metadata-only")
    return root


def test_import_is_gpu_free():
    code = """import sys
import mlx2.adapters.qwen36_35b
assert not any(k == 'mlx' or k.startswith('mlx.') for k in sys.modules)
assert 'mlx2.runtime.models.qwen36_35b' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_ordinary_artifact_registry_and_descriptor():
    with tempfile.TemporaryDirectory() as directory:
        root = make_artifact(Path(directory))
        artifact = inspect_artifact(root)
        resolved = inspect_model(root)
        assert not artifact["has_mtp"]
        assert resolved.adapter_type is Qwen3635BA3BAdapter
        assert resolved.default_route == "ordinary"
        assert resolved.descriptor.cache_layout == CACHE_LAYOUT
        assert Capability.APC_V2 in resolved.descriptor.capabilities
        assert Capability.MTP not in resolved.descriptor.capabilities


def test_embedded_mtp_requires_complete_sparse_head():
    with tempfile.TemporaryDirectory() as directory:
        root = make_artifact(Path(directory), mtp=True)
        assert inspect_artifact(root)["has_mtp"]
        resolved = inspect_model(root)
        assert Capability.MTP in resolved.descriptor.capabilities
        # Measured default: native MTP, sound only because the wide-cohort
        # ordinary handoff recovers the batched loss (see the adapter note).
        assert resolved.default_route == "native_mtp"
        index_path = root / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        del index["weight_map"][
            "language_model.mtp.layers.0.mlp.switch_mlp.down_proj.weight"
        ]
        index_path.write_text(json.dumps(index))
        with pytest.raises(ValueError, match="incomplete"):
            inspect_artifact(root)


def test_official_fused_mtp_experts_are_accepted_and_still_required():
    """Qwen/Qwen3.6-35B-A3B@995ad96 ships every expert table fused in the hub
    layout (``experts.gate_up_proj`` [E, 2I, H], ``experts.down_proj`` [E, H, I])
    with no ``.weight`` suffix.  Inspection rejected that checkpoint even for
    the ordinary route although ``sanitize`` converts it."""
    with tempfile.TemporaryDirectory() as directory:
        root = make_artifact(Path(directory), mtp=True)
        index_path = root / "model.safetensors.index.json"
        index = json.loads(index_path.read_text())
        weight_map = index["weight_map"]
        for name in ("gate_proj", "up_proj", "down_proj"):
            del weight_map[f"language_model.mtp.layers.0.mlp.switch_mlp.{name}.weight"]
        for name in ("gate_up_proj", "down_proj"):
            weight_map[f"mtp.layers.0.mlp.experts.{name}"] = "model.safetensors"
        index_path.write_text(json.dumps(index))
        artifact = inspect_artifact(root)
        assert artifact["has_mtp"]
        assert Capability.MTP in inspect_model(root).descriptor.capabilities
        del weight_map["mtp.layers.0.mlp.experts.down_proj"]
        index_path.write_text(json.dumps(index))
        with pytest.raises(ValueError, match="incomplete"):
            inspect_artifact(root)


# H=32 and I=24: a layout that confuses H with I or 2I cannot load strictly.
TINY_MTP_MOE = {
    "model_type": "qwen3_5_moe_text",
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "vocab_size": 128,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "linear_key_head_dim": 8,
    "linear_value_head_dim": 8,
    "linear_conv_kernel_dim": 3,
    "full_attention_interval": 4,
    "mtp_num_hidden_layers": 1,
    "partial_rotary_factor": 0.5,
    "rope_parameters": None,
    "max_position_embeddings": 128,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 24,
    "shared_expert_intermediate_size": 32,
}


def _tiny_mtp_model(monkeypatch, fuse_gate_up):
    from mlx2.runtime.models import qwen3_next
    from mlx2.runtime.models.qwen36_35b import Model, ModelArgs

    monkeypatch.setattr(qwen3_next, "_MOE_FUSED_GATE_UP", fuse_gate_up)
    monkeypatch.setattr(qwen3_next, "_MOE_SHARED_IN_GATHER", False)
    config = {"model_type": "qwen3_5_moe", "text_config": TINY_MTP_MOE}
    return Model(ModelArgs.from_dict(config))


def _moe_prefixes():
    body = [
        f"language_model.model.layers.{index}.mlp"
        for index in range(TINY_MTP_MOE["num_hidden_layers"])
    ]
    return body + ["language_model.mtp.layers.0.mlp"]


def _official_name(prefix):
    if prefix.startswith("language_model.model."):
        return prefix.replace("language_model.model.", "model.language_model.", 1)
    return prefix.removeprefix("language_model.")


def _released_layout(stacked, fuse_gate_up, layout):
    """Rewrite MLX-stacked experts the way a released checkpoint stores them."""
    import mlx.core as mx

    weights = dict(stacked)
    for prefix in _moe_prefixes():
        switch = f"{prefix}.switch_mlp"
        if fuse_gate_up:
            gate_up = weights.pop(f"{switch}.gate_up_proj.weight")
            midpoint = gate_up.shape[-2] // 2
            gate, up = gate_up[:, :midpoint], gate_up[:, midpoint:]
        else:
            gate = weights.pop(f"{switch}.gate_proj.weight")
            up = weights.pop(f"{switch}.up_proj.weight")
        down = weights.pop(f"{switch}.down_proj.weight")
        experts = f"{_official_name(prefix)}.experts"
        if layout == "hub":  # Qwen/Qwen3.6-35B-A3B
            weights[f"{experts}.gate_up_proj"] = mx.concatenate([gate, up], axis=1)
            weights[f"{experts}.down_proj"] = down
        elif layout == "bmm":  # transformers' batched-matmul orientation
            weights[f"{experts}.gate_up_proj"] = mx.concatenate(
                [gate, up], axis=1
            ).swapaxes(1, 2)
            weights[f"{experts}.down_proj"] = down.swapaxes(1, 2)
        else:  # one tensor per expert
            for expert in range(gate.shape[0]):
                for name, table in (("gate", gate), ("up", up), ("down", down)):
                    weights[f"{experts}.{expert}.{name}_proj.weight"] = table[expert]
    return weights


@pytest.mark.parametrize("fuse_gate_up", [False, True])
def test_official_fused_mtp_experts_load_strictly_into_the_stacked_tables(
    monkeypatch, fuse_gate_up
):
    """MTPLX#574: a lenient loader silently dropped fused MTP experts.  mlx2
    converts the hub layout for the body and the MTP head, then loads strictly;
    the result must equal loading the MLX-stacked tables directly."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    mx.random.seed(5)
    source = _tiny_mtp_model(monkeypatch, fuse_gate_up)
    stacked = dict(tree_flatten(source.parameters()))
    expected = {
        key: value
        for key, value in stacked.items()
        if key.startswith("language_model.mtp.layers.0.mlp.switch_mlp.")
    }
    assert len(expected) == (2 if fuse_gate_up else 3)

    reference = _tiny_mtp_model(monkeypatch, fuse_gate_up)
    reference.load_weights(list(reference.sanitize(dict(stacked)).items()), strict=True)
    released = _tiny_mtp_model(monkeypatch, fuse_gate_up)
    hub = _released_layout(stacked, fuse_gate_up, "hub")
    assert "mtp.layers.0.mlp.experts.gate_up_proj" in hub
    released.load_weights(list(released.sanitize(hub).items()), strict=True)

    loaded = dict(tree_flatten(released.parameters()))
    assert loaded.keys() == dict(tree_flatten(reference.parameters())).keys()
    for key, value in tree_flatten(reference.parameters()):
        assert mx.array_equal(loaded[key], value).item(), key
    for key, value in expected.items():
        assert mx.array_equal(loaded[key], value).item(), key


@pytest.mark.parametrize("layout", ["bmm", "numbered"])
def test_other_expert_layouts_fail_closed_at_the_strict_load(monkeypatch, layout):
    """Only the hub layout is converted.  The transformers bmm orientation and
    one-tensor-per-expert tables must refuse to load, never load partially."""
    from mlx.utils import tree_flatten

    source = _tiny_mtp_model(monkeypatch, False)
    stacked = dict(tree_flatten(source.parameters()))
    target = _tiny_mtp_model(monkeypatch, False)
    weights = target.sanitize(_released_layout(stacked, False, layout))
    with pytest.raises(ValueError):
        target.load_weights(list(weights.items()), strict=True)


def test_adapter_artifact_loads_are_strict():
    """Every adapter's model load is strict, so a key a sanitize rule missed
    fails the load instead of leaving a randomly initialised tensor."""
    import ast

    adapters = Path(__file__).parents[1] / "src/mlx2/adapters"
    calls = []
    for path in sorted(adapters.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "load_weights"
            ):
                strict = [kw.value for kw in node.keywords if kw.arg == "strict"]
                calls.append((path.name, node.lineno))
                assert (
                    len(strict) == 1
                    and isinstance(strict[0], ast.Constant)
                    and strict[0].value is True
                ), f"{path.name}:{node.lineno} loads weights without strict=True"
    assert any(name == "qwen36_35b.py" for name, _ in calls)
    assert len(calls) >= 10


def test_wrong_topology_rejected():
    with tempfile.TemporaryDirectory() as directory:
        root = make_artifact(Path(directory))
        config = json.loads((root / "config.json").read_text())
        config["text_config"]["num_experts"] = 128
        (root / "config.json").write_text(json.dumps(config))
        with pytest.raises(ValueError, match="topology"):
            inspect_artifact(root)


def test_baseline_environment_selects_stock_moe():
    with patch.dict(os.environ, {"MLX_QWEN4_MOE_ROUTER_KERNEL": "1"}):
        # Other tests exercise explicit kernel switches in this process.  A
        # baseline profile means these three supported overrides are absent.
        for name in (
            "MLX_QWEN36_FUSED_GDN_DECODE", "MLX_QWEN4_MOE_FUSED_GATE_UP",
            "MLX_GDN_CORE",
        ):
            os.environ.pop(name, None)
        profile = configure_environment()
        assert profile["MLX_LM_COMPILED_DECODE"] == "0"
        assert profile["MLX_QWEN36_FUSED_GDN_DECODE"] == "0"
        assert profile["MLX_QWEN4_MOE_ROUTER_KERNEL"] == "0"
        assert profile["MLX_QWEN4_FUSED_EXPERT_KERNEL"] == "stock"
        assert os.environ["MLX_QWEN4_MOE_FUSED_GATE_UP"] == "0"


def test_declared_kernel_env_values_survive_unless_policy_overrides_them():
    with patch.dict(os.environ, {
        "MLX_QWEN36_FUSED_GDN_DECODE": "1",
        "MLX_QWEN4_MOE_FUSED_GATE_UP": "1",
        "MLX_GDN_CORE": "1",
        "MLX_QWEN4_MOE_ROUTER_KERNEL": "1",  # unsupported 512/top-10 kernel
    }):
        effective = configure_environment()
        for name in (
            "MLX_QWEN36_FUSED_GDN_DECODE", "MLX_QWEN4_MOE_FUSED_GATE_UP",
            "MLX_GDN_CORE",
        ):
            assert effective[name] == os.environ[name] == "1"
        assert effective["MLX_QWEN4_MOE_ROUTER_KERNEL"] == "0"
        assert os.environ["MLX_QWEN4_MOE_ROUTER_KERNEL"] == "0"

        overridden = configure_environment({
            "fused_gdn_decode": False, "moe_fused_gate_up": False,
            "gdn_core": False,
        })
        for name in (
            "MLX_QWEN36_FUSED_GDN_DECODE", "MLX_QWEN4_MOE_FUSED_GATE_UP",
            "MLX_GDN_CORE",
        ):
            assert overridden[name] == os.environ[name] == "0"


def test_profile_names_remain_unqualified_candidates():
    adapter = object.__new__(Qwen3635BA3BAdapter)
    adapter._num_draft = 2
    adapter.descriptor = descriptor_for(has_mtp=False)
    assert adapter.profile_name(False) == "qwen36-35b-a3b-apcv2-ordinary"
    with pytest.raises(ValueError, match="embedded head"):
        adapter.profile_name(True)
    adapter.descriptor = descriptor_for(has_mtp=True)
    assert adapter.profile_name(True) == "qwen36-35b-a3b-apcv2-mtp2"


def test_tensor_module_compiles_and_has_no_apple_header():
    path = Path(__file__).parents[1] / "src/mlx2/runtime/models/qwen36_35b.py"
    source = path.read_text()
    compile(source, str(path), "exec")
    assert "Copyright © 2026 Apple" not in source


def test_fused_gdn_geometry_is_qwen36_specific_and_opt_in():
    import mlx.core as mx
    from mlx2.runtime.models.qwen4_fused_gdn import admit_qwen4_fused_gdn_decode

    bf16 = mx.bfloat16
    admission = admit_qwen4_fused_gdn_decode(
        qkv=mx.zeros((1, 1, 8192), dtype=bf16),
        z=mx.zeros((1, 1, 4096), dtype=bf16),
        b=mx.zeros((1, 1, 32), dtype=bf16),
        a=mx.zeros((1, 1, 32), dtype=bf16),
        conv_state=mx.zeros((1, 3, 8192), dtype=bf16),
        recurrent_state=mx.zeros((1, 32, 128, 128), dtype=mx.float32),
        conv_weight=mx.zeros((8192, 4, 1), dtype=bf16),
        A_log=mx.zeros((32,), dtype=mx.float32),
        dt_bias=mx.zeros((32,), dtype=bf16),
        norm_weight=mx.ones((128,), dtype=bf16),
        mask=None,
        spans=(),
        speculating=False,
        training=False,
        sharded=False,
        num_key_heads=16,
        num_value_heads=32,
        key_head_dim=128,
        value_head_dim=128,
        conv_kernel=4,
        gate_activation="swish",
        architecture="qwen35",
    )
    assert admission.accepted

    wrong_architecture = admit_qwen4_fused_gdn_decode(
        qkv=mx.zeros((1, 1, 8192), dtype=bf16),
        z=mx.zeros((1, 1, 4096), dtype=bf16),
        b=mx.zeros((1, 1, 32), dtype=bf16),
        a=mx.zeros((1, 1, 32), dtype=bf16),
        conv_state=mx.zeros((1, 3, 8192), dtype=bf16),
        recurrent_state=mx.zeros((1, 32, 128, 128), dtype=mx.float32),
        conv_weight=mx.zeros((8192, 4, 1), dtype=bf16),
        A_log=mx.zeros((32,), dtype=mx.float32),
        dt_bias=mx.zeros((32,), dtype=bf16),
        norm_weight=mx.ones((128,), dtype=bf16),
        mask=None,
        spans=(),
        speculating=False,
        training=False,
        sharded=False,
        num_key_heads=16,
        num_value_heads=32,
        key_head_dim=128,
        value_head_dim=128,
        conv_kernel=4,
        gate_activation="swish",
        architecture="agnes",
    )
    assert not wrong_architecture.accepted
    assert "geometry" in wrong_architecture.reason


def _chat_tokenizer(root: Path) -> tuple[int, int]:
    """Save a real fast tokenizer whose chat EOS is <|im_end|>, as Qwen ships."""
    tokenizers = pytest.importorskip("tokenizers")
    from transformers import PreTrainedTokenizerFast

    base = tokenizers.Tokenizer(
        tokenizers.models.WordLevel({"a": 0, "b": 1, "c": 2}, unk_token=None)
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=base)
    tokenizer.add_tokens(
        [
            tokenizers.AddedToken("<|endoftext|>", special=True, normalized=False),
            tokenizers.AddedToken("<|im_end|>", special=True, normalized=False),
        ]
    )
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.save_pretrained(root)
    return (
        tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
    )


class _StubTrunk:
    def set_eager_dispatch(self, *_args):
        pass


class _LoadedModel:
    """Stands in for the tensor module: these tests exercise tokenizer setup."""

    apc_v2_layout = "stub-layout"
    model = _StubTrunk()

    def __init__(self, *_args, **_kwargs):
        pass

    def sanitize(self, weights):
        return weights

    def shard_prune(self, *_args, **_kwargs):
        return True

    def load_weights(self, *_args, **_kwargs):
        pass

    def eval(self):
        pass

    def parameters(self):
        return {}

    def named_modules(self):
        return []


@pytest.mark.parametrize("converted", [False, True], ids=["official_config", "converted_config"])
def test_qwen36_stops_on_the_tokenizer_chat_eos(tmp_path, monkeypatch, converted):
    from mlx2.adapters import qwen36_35b
    from mlx2.runtime import ubc_evict
    from mlx2.runtime.models import qwen36_35b as tensors
    from mlx2.serving import generation_stop_token_ids

    endoftext, im_end = _chat_tokenizer(tmp_path)
    make_artifact(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    # The official Qwen/Qwen3.6-35B-A3B config names only <|endoftext|>; some
    # converted artifacts list both terminators at the top level.
    config["text_config"]["eos_token_id"] = endoftext
    if converted:
        config["eos_token_id"] = [endoftext, im_end]
    (tmp_path / "config.json").write_text(json.dumps(config))
    monkeypatch.setattr(qwen36_35b, "configure_environment", lambda: {})
    monkeypatch.setattr(tensors, "Model", _LoadedModel)
    monkeypatch.setattr(ubc_evict, "load_shards_evicting", lambda *_a, **_k: {})
    adapter = Qwen3635BA3BAdapter(str(tmp_path))
    assert generation_stop_token_ids(adapter) == (endoftext, im_end)


def test_flash_next_stops_on_the_tokenizer_chat_eos(tmp_path, monkeypatch):
    import mlx.nn as nn

    from mlx2.adapters import flash_next
    from mlx2.runtime import ubc_evict
    from mlx2.runtime.models import qwen4_exp, qwen4_ple_nvme
    from mlx2.serving import generation_stop_token_ids

    endoftext, im_end = _chat_tokenizer(tmp_path)
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen4_exp",
                "quantization": {"group_size": 64, "bits": 4},
                "text_config": {"eos_token_id": endoftext, "max_position_embeddings": 4096},
            }
        )
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.embed_tokens.weight": "model.safetensors"}})
    )
    (tmp_path / "model.safetensors").write_bytes(b"metadata-only")
    from mlx2.runtime.models import import_env

    monkeypatch.setattr(flash_next, "configure_environment", lambda *_a, **_k: {})
    # No profile is pinned here, so the import-order guard has nothing to check.
    monkeypatch.setattr(import_env, "assert_profile_applied", lambda *_a, **_k: None)
    monkeypatch.setattr(flash_next, "artifact_identity", lambda path: {"path": str(path)})
    monkeypatch.setattr(qwen4_exp, "Model", _LoadedModel)
    monkeypatch.setattr(qwen4_exp.ModelArgs, "from_dict", classmethod(lambda cls, config: config))
    monkeypatch.setattr(ubc_evict, "load_shards_evicting", lambda *_a, **_k: {})
    monkeypatch.setattr(ubc_evict, "ubc_evict_paths", lambda *_a, **_k: None)
    monkeypatch.setattr(
        qwen4_ple_nvme, "install_file_backed_ple", lambda _model, weights, *_a, **_k: weights
    )
    # This test isolates tokenizer EOS behavior. Sidecar integrity has its
    # own tests; supply its now-required metadata boundary without model I/O.
    (tmp_path / "ple_rows.bin").write_bytes(b"test-sidecar")
    monkeypatch.setattr(qwen4_ple_nvme, "verify_sidecar_against_artifact", lambda *_a: {})
    monkeypatch.setattr(qwen4_ple_nvme, "verify_sidecar_content", lambda *_a: "converted")
    monkeypatch.setattr(nn, "quantize", lambda *_a, **_k: None)
    adapter = flash_next.FlashNextAdapter(str(tmp_path))
    assert generation_stop_token_ids(adapter) == (endoftext, im_end)


def test_qwen36_eager_dispatch_defaults_on_and_is_bound_to_route_identity(tmp_path, monkeypatch):
    from mlx2.adapters import qwen36_35b
    from mlx2.runtime import ubc_evict
    from mlx2.runtime.models import qwen36_35b as tensors

    calls = []

    class _Trunk:
        def set_eager_dispatch(self, stride, max_rows):
            calls.append((stride, max_rows))

    class _Model(_LoadedModel):
        model = _Trunk()

    _chat_tokenizer(tmp_path)
    make_artifact(tmp_path)
    monkeypatch.setattr(qwen36_35b, "configure_environment", lambda *_a: {})
    monkeypatch.setattr(tensors, "Model", _Model)
    monkeypatch.setattr(ubc_evict, "load_shards_evicting", lambda *_a, **_k: {})
    default = Qwen3635BA3BAdapter(str(tmp_path))
    # Absent policy: the default stride 2 is installed and is route identity.
    assert calls == [(2, 64)]
    assert default.environment["MLX2_EAGER_DISPATCH_STRIDE"] == "2"
    selected = Qwen3635BA3BAdapter(str(tmp_path), execution_policy={
        "eager_dispatch_stride": 4, "eager_dispatch_max_rows": 8})
    assert calls == [(2, 64), (4, 8)]
    # A selected stride is route identity; an explicit 0 removes it again.
    assert selected.environment["MLX2_EAGER_DISPATCH_STRIDE"] == "4"
    assert selected.environment["MLX2_EAGER_DISPATCH_MAX_ROWS"] == "8"
    stock = Qwen3635BA3BAdapter(str(tmp_path), execution_policy={"eager_dispatch_stride": 0})
    assert calls == [(2, 64), (4, 8)]
    assert "MLX2_EAGER_DISPATCH_STRIDE" not in stock.environment
    assert "MLX2_EAGER_DISPATCH_STRIDE" not in os.environ
    with pytest.raises(ValueError):
        Qwen3635BA3BAdapter(str(tmp_path), execution_policy={"eager_dispatch_stride": "4"})
