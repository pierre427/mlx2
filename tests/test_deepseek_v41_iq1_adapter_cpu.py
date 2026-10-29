"""Compact admission/adapter CPU contracts, not packed Metal execution evidence."""

import copy
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from test_deepseek_v41_mlx_adapter_cpu import (  # noqa: F401 - fixture dependency
    _write,
    native_fixture,
)

from mlx2.adapters import deepseek_v41_iq1_artifact as compact
from mlx2.adapters import deepseek_v41_mlx_artifact as affine
from mlx2.adapters.deepseek_v41_iq1 import (
    DeepSeekV41IQ1Adapter,
    execution_policy,
    optimized_policy,
)
from mlx2.adapters.deepseek_v41_mlx import DeepSeekV41MLXAdapter


@pytest.fixture
def compact_fixture(request):
    root, original, _ = request.getfixturevalue("native_fixture")
    manifest = copy.deepcopy(original)
    manifest.update(
        schema=compact.SCHEMA,
        model_type=compact.MODEL_TYPE,
        format=compact.FORMAT,
        numeric_profile=compact.NUMERIC_PROFILE,
        source_affine={
            "derived_identity_sha256": "c" * 64,
            "fingerprint": "d" * 64,
            "path": "/absent/affine-reference",
        },
    )
    paths = manifest["tensors"]["blk.0.ffn_gate_exps.weight"]["storage"]["expert_files"]
    for path in paths:
        manifest["files"][path] = _write(
            root,
            path,
            b"",
            arrays={
                plane + ".data": np.zeros(256 * 50, np.uint8)
                for plane in ("gate", "up", "down")
            },
        )
    for plane in ("gate", "up", "down"):
        row = manifest["tensors"][f"blk.0.ffn_{plane}_exps.weight"]
        row.update(shape=[256, 256, 2], numpy_shape=[2, 256, 256], nbytes=2 * 256 * 50)
        row["storage"] = {
            "kind": "expert-iq1_s",
            "expert_files": list(paths),
            "keys": {"data": plane + ".data"},
        }
    config = {"model_type": compact.MODEL_TYPE, "mlx2_weight_format": compact.FORMAT}
    manifest["files"]["config.json"] = _write(
        root, "config.json", json.dumps(config).encode()
    )
    (root / affine.MANIFEST_FILE).unlink()

    def save():
        (root / compact.MANIFEST_FILE).write_text(json.dumps(manifest))

    save()
    return root, manifest, save


def test_compact_admission_needs_neither_original_source(compact_fixture, monkeypatch):
    from mlx2.runtime.gguf.reader import GGUFFile

    monkeypatch.setattr(
        GGUFFile, "__init__", lambda *args, **kwargs: pytest.fail("source GGUF access")
    )
    root, manifest, _ = compact_fixture
    found = compact.inspect_artifact(root, verify_hashes=True)
    assert (
        found["weight_format"] == compact.FORMAT
        and found["original_gguf_required"] is False
    )
    assert found["source_affine_required"] is False
    assert found["source_affine_derived_identity_sha256"] == "c" * 64
    assert found["source_affine_fingerprint"] == "d" * 64
    assert found["identity"]["fingerprint"] not in ("c" * 64, "d" * 64)
    payload = sum(
        f["size"]
        for name, f in manifest["files"].items()
        if name.startswith("experts/")
    )
    assert found["inventory"]["storage_bytes"]["expert"] == payload
    assert (
        DeepSeekV41IQ1Adapter.resident_weight_bytes(root)
        == payload + found["inventory"]["storage_bytes"]["dense"]
    )


@pytest.mark.parametrize(
    "damage",
    [
        "unfinished",
        "source_binding",
        "source_fingerprint",
        "affine_profile",
        "bad_extent",
        "bad_key",
        "mixed_affine",
        "malformed_storage",
        "float_extent",
        "reordered_experts",
    ],
)
def test_compact_bad_manifest_refuses(compact_fixture, damage):
    root, manifest, save = compact_fixture
    row = manifest["tensors"]["blk.0.ffn_gate_exps.weight"]
    if damage == "unfinished":
        manifest["finished"] = False
    elif damage == "source_binding":
        manifest["source_affine"]["derived_identity_sha256"] = "wrong"
    elif damage == "source_fingerprint":
        manifest["source_affine"].pop("fingerprint")
    elif damage == "affine_profile":
        manifest["numeric_profile"] = affine.NUMERIC_PROFILE
    elif damage == "bad_extent":
        row["nbytes"] -= 1
    elif damage == "bad_key":
        row["storage"]["keys"] = {"data": "up.data"}
    elif damage == "mixed_affine":
        row["storage"]["kind"] = "expert-affine"
    elif damage == "malformed_storage":
        row["storage"] = []
    elif damage == "float_extent":
        row["nbytes"] = float(row["nbytes"])
    else:
        row["storage"]["expert_files"].reverse()
    save()
    with pytest.raises(ValueError):
        compact.inspect_artifact(root)


def test_compact_payload_replacement_and_hardlinks_refuse(compact_fixture, tmp_path):
    root, manifest, save = compact_fixture
    weight = root / "weights.safetensors"
    os.link(weight, tmp_path / "reference-hardlink")
    # Even a new compact seal must not bless hardlink sharing of the reference.
    manifest["files"]["weights.safetensors"]["stat"] = affine.file_stamp(weight.stat())
    save()
    with pytest.raises(ValueError, match="hardlinks"):
        compact.inspect_artifact(root)


def test_compact_same_size_expert_replacement_refuses(compact_fixture):
    root, manifest, _ = compact_fixture
    name = next(name for name in manifest["files"] if name.startswith("experts/"))
    path = root / name
    path.write_bytes(bytes(path.stat().st_size))
    with pytest.raises(ValueError, match="changed after sealing"):
        compact.inspect_artifact(root)


def test_compact_policy_is_explicit_and_affine_policy_stays_unchanged():
    from mlx2.adapters.deepseek_v41_mlx import execution_policy as affine_policy
    from mlx2.runtime.streamed_load import declared_stream_modes

    assert execution_policy()["deepseek_expert_prefill"] == "reference"
    selected = execution_policy(optimized_policy(copy_chain=True))
    assert (
        selected["deepseek_expert_prefill"] == "segmented_nax_fp16"
        and selected["ast_prompt_lookup"]
    )
    assert "deepseek_expert_prefill" not in affine_policy()
    assert declared_stream_modes(DeepSeekV41IQ1Adapter) == {"moe_experts"}
    assert (
        DeepSeekV41IQ1Adapter.numeric_profile != DeepSeekV41MLXAdapter.numeric_profile
    )
    for value in (
        {"deepseek_expert_prefill": "legacy"},
        {"deepseek_packed": "cooperative"},
        {"ced_prefill": True},
    ):
        with pytest.raises(ValueError):
            execution_policy(value)


def test_compact_inspector_adapter_planner_import_without_device_runtime():
    script = """
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in ('mlx','torch','transformers','tokenizers'):
   raise RuntimeError('device import: '+fullname)
sys.meta_path.insert(0, Guard())
from mlx2.adapters.deepseek_v41_iq1 import DeepSeekV41IQ1Adapter
from mlx2.adapters.deepseek_v41_iq1_weights import compact_weight_plan
assert DeepSeekV41IQ1Adapter.descriptor.model_type == 'deepseek41_iq1'
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("prefill", ["reference", "segmented_nax_fp16"])
@pytest.mark.parametrize(
    "layout,completion,decode",
    [
        ("row_major", "sync", "row_exact"),
        ("tiled64", "sync", "row_exact"),
        ("tiled64", "async", "row_exact"),
        ("tiled64", "sync", "column_f32"),
        ("tiled64", "async", "column_f32"),
    ],
)
def test_compact_full_adapter_loads_native_dense_and_distinct_cache_identity(
    tmp_path, monkeypatch, prefill, layout, completion, decode
):
    """Real CPU construction/preload; packed Metal forward is deliberately not run."""
    import mlx.core as mx
    from test_deepseek_v41_cpu import Weights, config
    from test_deepseek_v41_iq1_weights_cpu import write_compact_artifact
    from test_deepseek_v41_mlx_weights_cpu import matrix
    from tokenizers import Tokenizer, models

    from mlx2 import process_env
    from mlx2.adapters import deepseek_v4_gguf_tokenizer
    from mlx2.runtime import weight_residency
    from mlx2.runtime.gguf.reader import GGUFFile
    from mlx2.runtime.models import deepseek_v41_engram
    from mlx2.runtime.streamed_load import WeightStreamRequest

    mx.set_default_device(mx.cpu)
    c = config(
        vocab_size=32,
        hidden_size=256,
        head_dim=32,
        q_lora_rank=32,
        qk_rope_head_dim=16,
        o_lora_rank=32,
        moe_intermediate_size=256,
        index_head_dim=32,
        activation_dtype="bfloat16",
    )
    rng, dense = np.random.default_rng(917), {}
    for name, value in Weights(c).data.items():
        if "_exps.weight" in name:
            continue
        value = np.array(value)
        dense[name] = (
            matrix(*value.shape, 8, rng)
            if value.ndim == 2 and "hc_" not in name and "ffn_gate_inp" not in name
            else value
        )
    raw = np.zeros(256 * 50, np.uint8)
    packed = {
        layer: [
            {plane: raw.copy() for plane in ("gate", "up", "down")}
            for _ in range(c.n_routed_experts)
        ]
        for layer in range(c.num_hidden_layers)
    }
    admitted = write_compact_artifact(tmp_path / "compact", dense, packed, c)
    admitted["identity"].update(
        path=str(admitted["root"]),
        fingerprint="compact-fixture",
        weight_sha256="b" * 64,
    )
    admitted["config"] = {
        "model_type": compact.MODEL_TYPE,
        "mlx2_weight_format": compact.FORMAT,
    }
    admitted["metadata"] = {
        "tokenizer.ggml.tokens": [str(i) for i in range(c.vocab_size)]
    }
    backend = Tokenizer(
        models.WordLevel({str(i): i for i in range(c.vocab_size)}, unk_token="3")
    )
    (admitted["root"] / affine.REFERENCE_FILE).write_text(backend.to_str())
    monkeypatch.setattr(
        DeepSeekV41IQ1Adapter, "artifact_inspector", staticmethod(lambda path: admitted)
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
        lambda *args, **kwargs: pytest.fail("compact adapter opened original GGUF"),
    )
    adapter = DeepSeekV41IQ1Adapter(
        str(admitted["root"]),
        execution_policy={
            "deepseek_expert_prefill": prefill,
            "deepseek_expert_layout": layout,
            "deepseek_arena_completion": completion,
            "deepseek_tiled_decode": decode,
            "draft_model": "prompt_lookup",
            "num_draft": 3,
        },
        weight_streaming=WeightStreamRequest(
            "moe_experts", 1 << 20, admission_limit_bytes=8 << 30, read_workers=1
        ),
    )
    try:
        assert adapter.descriptor.model_type == compact.MODEL_TYPE
        assert adapter.weights.expert_prefill_policy == prefill
        revision = (
            "mlx2-v41-iq1s-native-f32-dense-v1"
            if layout == "row_major"
            else "mlx2.v41-iq1-tiled-block-end-repair.v1"
        )
        assert revision in adapter.model.identity
        assert adapter.model.make_cache()[0].identity == adapter.model.identity
        expected_profile = (
            compact.NUMERIC_PROFILE if layout == "row_major" else "native_iq1_s_tiled"
        )
        if decode == "column_f32":
            from mlx2.runtime.gguf.v41_iq1_column import REVISION as COLUMN_REVISION

            expected_profile = "native_iq1_s_tiled_column_f32"
            assert COLUMN_REVISION in adapter.model.identity
            assert COLUMN_REVISION in adapter.weights.compute_policy
            assert adapter.diagnostics()["tiled_decode"] == "column_f32"
        assert adapter.diagnostics()["numeric_profile"] == expected_profile
        assert adapter.descriptor.metadata["numeric_profile"] == expected_profile
        assert adapter.diagnostics()["source_affine_required"] is False
        assert (
            adapter.weight_stream.plan.table_bytes
            == c.num_hidden_layers * c.n_routed_experts * 3 * 256 * 50
        )
        assert not adapter.weights.scheduled_expert_calls
        assert adapter.draft_model is not None
        assert adapter.weights.verify_source()
    finally:
        adapter.close()
