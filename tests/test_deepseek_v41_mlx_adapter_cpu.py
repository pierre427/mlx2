"""Synthetic native-artifact admission; no checkpoint/GPU qualification."""

import copy
import hashlib
import json
import os
import struct
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from mlx2.adapters import deepseek_v41_mlx_artifact as artifact
from mlx2.adapters.deepseek_v41_mlx import (
    DEFAULT_EXPERT_CACHE_BYTES,
    DESCRIPTOR,
    DeepSeekV41MLXAdapter,
    admission_plan,
    execution_policy,
    optimized_policy,
)


def _write(root, name, data, *, arrays=None):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    header, base = {}, 0
    if arrays is not None:
        payload = bytearray()
        for key, value in arrays.items():
            dtype = {"float32": "F32", "uint32": "U32", "uint8": "U8"}[str(value.dtype)]
            first = len(payload)
            payload.extend(value.tobytes())
            header[key] = {
                "dtype": dtype,
                "shape": list(value.shape),
                "data_offsets": [first, len(payload)],
            }
        encoded = json.dumps(header).encode()
        encoded += b" " * (-len(encoded) % 8)
        base = 8 + len(encoded)
        data = struct.pack("<Q", len(encoded)) + encoded + payload
    path.write_bytes(data)
    return {
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "stat": artifact.file_stamp(path.stat()),
        "header_bytes": base,
        "tensors": header,
    }


@pytest.fixture
def native_fixture(tmp_path, monkeypatch):
    """Small storage schema fixture; topology validator remains separate."""
    files, tensors = {}, {}
    dense = {"output_norm.weight": np.ones(32, np.float32)}
    tensors["output_norm.weight"] = {
        "shape": [32],
        "numpy_shape": [32],
        "ggml_type": 0,
        "offset": 0,
        "nbytes": 128,
        "storage": {
            "kind": "raw",
            "file": "weights.safetensors",
            "key": "output_norm.weight",
        },
    }
    keys = {key: "token_embd.weight." + key for key in ("qweight", "scales", "biases")}
    dense.update(
        {
            keys["qweight"]: np.zeros((2, 8), np.uint32),
            keys["scales"]: np.ones((2, 1), np.float32),
            keys["biases"]: np.zeros((2, 1), np.float32),
        }
    )
    tensors["token_embd.weight"] = {
        "shape": [32, 2],
        "numpy_shape": [2, 32],
        "ggml_type": 8,
        "offset": 128,
        "nbytes": 68,
        "storage": {
            "kind": "affine",
            "file": "weights.safetensors",
            "keys": keys,
            "bits": 8,
            "group_size": 32,
            "mode": "affine",
        },
    }
    files["weights.safetensors"] = _write(
        tmp_path, "weights.safetensors", b"", arrays=dense
    )
    names = [f"experts/layer-000/expert-{i:03d}.safetensors" for i in range(2)]
    planes = {}
    for plane in ("gate", "up", "down"):
        keys = {key: plane + "." + key for key in ("qweight", "scales", "biases")}
        planes.update(
            {
                keys["qweight"]: np.zeros((32, 2), np.uint32),
                keys["scales"]: np.ones((32, 1), np.float32),
                keys["biases"]: np.zeros((32, 1), np.float32),
            }
        )
        tensors[f"blk.0.ffn_{plane}_exps.weight"] = {
            "shape": [32, 32, 2],
            "numpy_shape": [2, 32, 32],
            "ggml_type": 19,
            "offset": 256,
            "nbytes": 400,
            "storage": {
                "kind": "expert-affine",
                "expert_files": list(names),
                "keys": keys,
                "bits": 2,
                "group_size": 32,
                "mode": "affine",
            },
        }
    for name in names:
        files[name] = _write(tmp_path, name, b"", arrays=planes)
    name = "engram/layer-001.bin"
    files[name] = _write(tmp_path, name, bytes(528))
    tensors["blk.1.engram_embd.weight"] = {
        "shape": [264, 2],
        "numpy_shape": [2, 264],
        "ggml_type": 24,
        "offset": 1024,
        "nbytes": 528,
        "storage": {"kind": "engram", "file": name, "row_bytes": 264, "rows": 2},
    }
    reference = b'{"fixture":"synthetic tokenizer"}'
    files[artifact.REFERENCE_FILE] = _write(
        tmp_path, artifact.REFERENCE_FILE, reference
    )
    monkeypatch.setattr(
        artifact, "TOKENIZER_SHA256", hashlib.sha256(reference).hexdigest()
    )
    config = {"model_type": artifact.MODEL_TYPE, "mlx2_weight_format": artifact.FORMAT}
    files["config.json"] = _write(tmp_path, "config.json", json.dumps(config).encode())
    recipe = {"group_size": 32, "fixture": True}
    manifest = {
        "schema": artifact.SCHEMA,
        "model_type": artifact.MODEL_TYPE,
        "format": artifact.FORMAT,
        "numeric_profile": artifact.NUMERIC_PROFILE,
        "finished": True,
        "source": {
            "gguf_sha256": artifact.WEIGHT_SHA256,
            "size": artifact.WEIGHT_BYTES,
            "fingerprint": "a" * 64,
            "path": "/deleted/original.gguf",
        },
        "recipe": recipe,
        "recipe_sha256": hashlib.sha256(
            json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "metadata": {"fixture": True},
        "files": files,
        "tensors": tensors,
    }
    expected = set(tensors)

    def validate(metadata, descriptions):
        if metadata != {"fixture": True} or set(descriptions) != expected:
            raise ValueError("fixture tensor inventory changed")
        assert descriptions["token_embd.weight"].shape == (32, 2)
        return SimpleNamespace(
            num_hidden_layers=1, n_routed_experts=2, num_experts_per_tok=1
        )

    monkeypatch.setattr(artifact, "validate_header", validate)

    def save():
        (tmp_path / artifact.MANIFEST_FILE).write_text(json.dumps(manifest))

    save()
    return tmp_path, manifest, save


def test_standalone_inspection_never_opens_original(native_fixture, monkeypatch):
    from mlx2.runtime.gguf.reader import GGUFFile

    monkeypatch.setattr(
        GGUFFile, "__init__", lambda *a, **k: pytest.fail("original GGUF dependency")
    )
    root, manifest, _ = native_fixture
    found = artifact.inspect_artifact(root, verify_hashes=True)
    assert found["original_gguf_required"] is False
    assert found["numeric_profile"] == "native_f32_qmm"
    assert found["inventory"]["tensor_count"] == len(manifest["tensors"])
    assert found["inventory"]["storage_bytes"]["engram"] == 528
    assert found["identity"]["fingerprint"] != manifest["source"]["fingerprint"]
    expected = found["inventory"]["storage_bytes"]
    assert (
        DeepSeekV41MLXAdapter.resident_weight_bytes(root)
        == expected["dense"] + expected["expert"]
    )


@pytest.mark.parametrize("finished", [False, None, 1])
def test_partial_conversions_refuse_before_assets(native_fixture, finished):
    root, manifest, save = native_fixture
    manifest["finished"] = finished
    save()
    (root / "weights.safetensors").unlink()
    with pytest.raises(ValueError, match="unfinished"):
        artifact.inspect_artifact(root)


def test_same_size_payload_replacement_refuses(native_fixture):
    root, _, _ = native_fixture
    path = root / "weights.safetensors"
    old = path.stat()
    path.write_bytes(bytes(old.st_size))
    os.utime(path, ns=(old.st_atime_ns, old.st_mtime_ns))
    with pytest.raises(ValueError, match="changed after sealing"):
        artifact.inspect_artifact(root)


def test_symlink_parent_and_traversal_refuse(native_fixture, tmp_path):
    root, manifest, save = native_fixture
    item = manifest["files"].pop("weights.safetensors")
    manifest["files"]["../weights.safetensors"] = item
    save()
    with pytest.raises(ValueError, match="escapes"):
        artifact.inspect_artifact(root)
    with pytest.raises(ValueError, match="escapes"):
        artifact.asset_path(root, "/etc/passwd")


@pytest.mark.parametrize(
    "damage,match",
    [
        ("expert_missing", "incomplete"),
        ("expert_misaligned", "multiply owned|disagree"),
        ("bits", "affine recipe"),
        ("dtype", "storage geometry"),
        ("source_size", "byte extent"),
        ("recipe", "recipe identity"),
        ("foreign_tensor", "inventory changed"),
    ],
)
def test_manifest_storage_corruption_refuses(native_fixture, damage, match):
    root, manifest, save = native_fixture
    gate = manifest["tensors"]["blk.0.ffn_gate_exps.weight"]
    if damage == "expert_missing":
        gate["storage"]["expert_files"].pop()
    elif damage == "expert_misaligned":
        manifest["tensors"]["blk.0.ffn_up_exps.weight"]["storage"][
            "expert_files"
        ].reverse()
    elif damage == "bits":
        gate["storage"]["bits"] = 4
    elif damage == "dtype":
        name = gate["storage"]["expert_files"][0]
        manifest["files"][name]["tensors"]["gate.qweight"]["dtype"] = "F32"
    elif damage == "source_size":
        gate["nbytes"] += 1
    elif damage == "recipe":
        manifest["recipe"]["fixture"] = False
    else:
        manifest["tensors"]["unexpected"] = copy.deepcopy(gate)
    save()
    with pytest.raises(ValueError, match=match):
        artifact.inspect_artifact(root)


def test_hash_mode_detects_wrong_content_even_with_resealed_stat(native_fixture):
    root, manifest, save = native_fixture
    path = root / "weights.safetensors"
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    manifest["files"]["weights.safetensors"]["stat"] = artifact.file_stamp(path.stat())
    save()
    with pytest.raises(ValueError, match="hash changed"):
        artifact.inspect_artifact(root, verify_hashes=True)


def test_native_plan_admits_before_allocations_and_rejects_unsupported_modes(
    native_fixture,
):
    from mlx2.runtime.streamed_load import WeightStreamRequest, declared_stream_modes

    root, _, _ = native_fixture
    found = artifact.inspect_artifact(root)
    request = WeightStreamRequest("moe_experts", 1 << 20, admission_limit_bytes=8 << 30)
    plan = admission_plan(found, request)
    assert (
        plan["phase"] == "pre_materialization" and plan["expert_cache_bytes"] == 1 << 20
    )
    assert plan["required_bytes"] > (5 << 30) + (1 << 20)
    assert declared_stream_modes(DeepSeekV41MLXAdapter) == {"moe_experts"}
    with pytest.raises(MemoryError, match="early admission"):
        admission_plan(
            found,
            WeightStreamRequest("moe_experts", 1 << 20, admission_limit_bytes=1 << 20),
        )
    with pytest.raises(ValueError, match="synchronous"):
        admission_plan(found, WeightStreamRequest("moe_experts", 1 << 20, elastic=True))
    with pytest.raises(ValueError, match="at most 70"):
        admission_plan(found, WeightStreamRequest("moe_experts", 71 << 30))
    automatic = admission_plan(
        found,
        WeightStreamRequest(
            "moe_experts", 0, budget_mode="auto", admission_limit_bytes=8 << 30
        ),
    )
    assert 0 < automatic["expert_cache_bytes"] < DEFAULT_EXPERT_CACHE_BYTES


@pytest.mark.parametrize(
    "value",
    [
        {"deepseek_packed": "cooperative"},
        {"ced_prefill": True},
        {"deepseek_verify_rows": "layerwise"},
        {"draft_model": "dspark"},
        {"num_draft": 5},
        {"ast_prompt_lookup": True},
        {"draft_model": "prompt_lookup", "num_draft": 6},
    ],
)
def test_foreign_math_and_incomplete_copy_policies_refuse(value):
    with pytest.raises(ValueError):
        execution_policy(value)


def test_native_policy_retains_ordinary_and_serial_copy_contract():
    assert execution_policy()["ast_prompt_lookup"] is False
    chosen = execution_policy(optimized_policy(copy_chain=True))
    assert (
        chosen["ast_prompt_lookup"] is True
        and chosen["deepseek_verify_rows"] == "serial"
    )
    assert chosen["deepseek_verify_commit"] == "replay"
    assert DESCRIPTOR.model_type == "deepseek41_mlx"
    assert DESCRIPTOR.metadata["qualification"] == "unqualified"


def test_imports_are_gpu_free():
    code = """
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in ('mlx','torch','transformers','tokenizers'):
   raise RuntimeError('device/runtime import: '+fullname)
sys.meta_path.insert(0, Guard())
from mlx2.adapters.deepseek_v41_mlx import DeepSeekV41MLXAdapter
from mlx2.adapters.deepseek_v41_mlx_weights import native_weight_plan
assert DeepSeekV41MLXAdapter.descriptor.model_type == 'deepseek41_mlx'
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def native_adapter(tmp_path, monkeypatch):
    """Real native weights/model/adapter on tiny CPU geometry; fake host admission."""
    import mlx.core as mx
    from test_deepseek_v41_cpu import Weights, config
    from test_deepseek_v41_mlx_weights_cpu import matrix, write_native_artifact
    from tokenizers import Tokenizer, models

    from mlx2 import process_env
    from mlx2.adapters import deepseek_v4_gguf_tokenizer
    from mlx2.adapters import deepseek_v41_mlx as adapter_module
    from mlx2.runtime import weight_residency
    from mlx2.runtime.gguf.reader import GGUFFile
    from mlx2.runtime.models import deepseek_v41_engram
    from mlx2.runtime.streamed_load import WeightStreamRequest

    mx.set_default_device(mx.cpu)
    c = config(
        vocab_size=32,
        hidden_size=32,
        head_dim=32,
        q_lora_rank=32,
        qk_rope_head_dim=16,
        o_lora_rank=32,
        moe_intermediate_size=32,
        index_head_dim=32,
        activation_dtype="bfloat16",
    )
    original = Weights(c)
    rng = np.random.default_rng(9921)
    dense = {}
    for name, value in original.data.items():
        if "_exps.weight" in name:
            continue
        value = np.array(value)
        if value.ndim == 2 and "hc_" not in name and "ffn_gate_inp" not in name:
            dense[name] = matrix(*value.shape, 8, rng)
        else:
            dense[name] = value
    experts = {
        layer: [
            {kind: matrix(32, 32, 2, rng) for kind in ("gate", "up", "down")}
            for _ in range(c.n_routed_experts)
        ]
        for layer in range(c.num_hidden_layers)
    }
    admitted = write_native_artifact(tmp_path / "native", dense, experts, c)
    admitted["identity"].update(path=str(admitted["root"]), weight_sha256="b" * 64)
    admitted["config"] = {
        "model_type": artifact.MODEL_TYPE,
        "mlx2_weight_format": artifact.FORMAT,
    }
    admitted["metadata"] = {
        "tokenizer.ggml.tokens": [str(i) for i in range(c.vocab_size)]
    }
    backend = Tokenizer(
        models.WordLevel({str(i): i for i in range(c.vocab_size)}, unk_token="3")
    )
    (admitted["root"] / artifact.REFERENCE_FILE).write_text(backend.to_str())
    monkeypatch.setattr(
        adapter_module.DeepSeekV41MLXAdapter,
        "artifact_inspector",
        staticmethod(lambda path: admitted),
    )
    monkeypatch.setattr(
        deepseek_v4_gguf_tokenizer,
        "align_tokenizer",
        lambda raw, meta: (raw, {"synthetic": True}),
    )
    monkeypatch.setattr(deepseek_v41_engram, "EngramHasher", lambda metadata: None)
    monkeypatch.setattr(
        weight_residency,
        "wire_serving_weights",
        lambda: {"wired": False, "fixture": "CPU"},
    )
    for name in (
        "require_process_numerics",
        "clear_inherited_profile",
        "assert_process_profile_applied",
    ):
        monkeypatch.setattr(process_env, name, lambda *args: None)
    for key, value in {
        **process_env.PROCESS_NUMERICS,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        GGUFFile,
        "__init__",
        lambda *a, **k: pytest.fail("native adapter opened original GGUF"),
    )
    adapter = DeepSeekV41MLXAdapter(
        str(admitted["root"]),
        execution_policy={"draft_model": "prompt_lookup", "num_draft": 3},
        weight_streaming=WeightStreamRequest(
            "moe_experts", 3072, admission_limit_bytes=8 << 30, read_workers=1
        ),
    )
    try:
        yield adapter
    finally:
        adapter.close()


def test_native_adapter_full_model_apcv2_and_rejected_suffix(native_adapter, tmp_path):
    import mlx.core as mx

    from mlx2.adapters.deepseek_v4_external import DeepSeekV4VerifyRows
    from mlx2.runtime.models.cache import load_prompt_cache, save_prompt_cache

    adapter, target = native_adapter, native_adapter.model
    cache = target.make_cache()
    target(mx.array([[1, 2, 3]], mx.int32), cache)
    path = tmp_path / "native-cache.safetensors"
    save_prompt_cache(str(path), cache)
    restored = load_prompt_cache(str(path))
    a = target(mx.array([[4]], mx.int32), cache)
    b = target(mx.array([[4]], mx.int32), restored)
    np.testing.assert_array_equal(a, b)
    snapshots = [row.snapshot() for row in restored]
    owner = DeepSeekV4VerifyRows(target, [restored], (4,))
    transaction = owner.begin([3])
    transaction.forward(mx.array([[5, 6, 7]], mx.int32))
    transaction.commit([1])
    actual = target(mx.array([[8]], mx.int32), restored)
    for row, snapshot in zip(cache, snapshots):
        row.restore(snapshot)
    target(mx.array([[5]], mx.int32), cache)
    expected = target(mx.array([[8]], mx.int32), cache)
    np.testing.assert_array_equal(actual, expected)
    assert adapter.diagnostics()["numeric_profile"] == "native_f32_qmm"
    assert adapter.diagnostics()["original_gguf_required"] is False
    assert adapter.weight_stream.receipt()["allocation"] == "global_byte_lru"
    assert adapter.weights.scheduled_dense_calls["native_affine8_f32"] > 0


def test_native_serial_copy_loop_matches_ordinary(native_adapter):
    import mlx.core as mx

    target = native_adapter.model
    prompt = [1, 2, 3, 1, 2, 3, 1, 2, 3]
    cache = target.make_cache()
    logits = target.prefill_forward(mx.array([prompt], mx.int32), cache)
    ordinary = []
    for _ in range(10):
        token = int(mx.argmax(logits[0, -1]).item())
        ordinary.append(token)
        logits = target(mx.array([[token]], mx.int32), cache)
    generator = native_adapter.create_external_batch(
        stop_tokens=set(), prefill_step_size=16
    )
    generator.insert([prompt], max_tokens=[10])
    output = []
    for _ in range(30):
        _, rows = generator.next()
        output.extend(int(row.token) for row in rows)
        if not generator.lanes:
            break
    assert output == ordinary


def test_native_close_retains_failed_weight_owner_and_can_retry(monkeypatch):
    events = []
    adapter = object.__new__(DeepSeekV41MLXAdapter)

    def fail():
        events.append("failed-fence")
        raise RuntimeError("uncertain upload fence")

    weights = SimpleNamespace(close=fail)
    target = SimpleNamespace(weights=weights)
    adapter.weights, adapter.model, adapter.weight_stream = weights, target, object()
    adapter.draft_model = None
    adapter.engram_tables = {}
    adapter.preload_residency = None
    with pytest.raises(RuntimeError, match="uncertain upload"):
        adapter.close()
    assert (
        adapter.weights is weights
        and adapter.model is target
        and adapter.weight_stream is not None
    )
    weights.close = lambda: events.append("drained")
    adapter.close()
    assert adapter.weights is adapter.model is adapter.weight_stream is None
    assert events == ["failed-fence", "drained"]
