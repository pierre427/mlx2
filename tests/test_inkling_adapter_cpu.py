"""CPU-safe artifact/dispatch gates against the pinned Hub config fixture."""

import json
import math
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest

from mlx2.adapters.inkling import inspect_artifact, text_tensor_specs, validate_policy
from mlx2.adapters.registry import inspect_model

FIXTURE = Path(__file__).parent / "fixtures/inkling-small-config.json"


@pytest.fixture
def artifact(tmp_path):
    (tmp_path / "config.json").write_bytes(FIXTURE.read_bytes())
    specs = text_tensor_specs()
    offset = 0
    header = {}
    for key, (dtype, shape) in specs.items():
        count = math.prod(shape) * (2 if dtype == "BF16" else 4)
        header[key] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + count],
        }
        offset += count
    raw = json.dumps(header).encode()
    # The fixture has real header geometry and sparse, unmaterialized payload
    # extents. No tensors or model weights are loaded by these tests.
    with (tmp_path / "model.safetensors").open("wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        f.truncate(8 + len(raw) + offset)
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(specs, "model.safetensors")})
    )
    return tmp_path


def test_dispatch_is_cpu_safe_and_stays_unqualified(artifact):
    resolution = inspect_model(artifact)
    assert resolution.adapter_type.__name__ == "InklingAdapter"
    assert resolution.default_route == "ordinary"
    assert resolution.artifact["has_mtp"] is False
    assert resolution.artifact["identity"]["source_revision"] == "unattested-local-copy"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")}
    code = 'import sys;sys.modules["mlx"]=None;from mlx2.adapters.registry import inspect_model;print(inspect_model(sys.argv[1]).adapter_type.__name__)'
    output = subprocess.check_output(
        [sys.executable, "-c", code, str(artifact)], env=env, text=True
    )
    assert output.strip() == "InklingAdapter"


@pytest.mark.parametrize(
    "mutation",
    [
        "dense_width",
        "quantized_router",
        "missing_projection",
        "escaped_shard",
        "truncated_payload",
    ],
)
def test_malformed_checkpoint_roles_fail_before_tensor_access(artifact, mutation):
    config = json.loads((artifact / "config.json").read_text())
    index = json.loads((artifact / "model.safetensors.index.json").read_text())
    if mutation == "dense_width":
        config["text_config"]["moe_intermediate_size"] = 16384
    if mutation == "quantized_router":
        config["quantization"]["language_model.model.layers.2.mlp.gate"] = {
            "bits": 2,
            "group_size": 128,
            "mode": "affine",
        }
    if mutation == "missing_projection":
        index["weight_map"].pop("language_model.model.layers.2.self_attn.r_proj.weight")
    if mutation == "escaped_shard":
        index["weight_map"][next(iter(index["weight_map"]))] = "../other.safetensors"
    if mutation == "truncated_payload":
        with (artifact / "model.safetensors").open("r+b") as f:
            f.truncate(512000)
    (artifact / "config.json").write_text(json.dumps(config))
    (artifact / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises((ValueError, FileNotFoundError)):
        inspect_artifact(artifact)


def test_kernel_and_ast_policy_validation_is_explicit():
    assert validate_policy(None) == validate_policy({})
    kernels, external = validate_policy(
        {
            "inkling_fused": True,
            "inkling_moe_nax": "fused",
            "draft_model": "prompt_lookup",
            "inkling_tree_executor": "packed",
        }
    )
    assert kernels["inkling_fused"] and external["batch_size_route"]
    for bad in (
        {"inkling_fused": "yes"},
        {"inkling_moe_nax": "bad"},
        {"draft_model": "arbitrary"},
        {"inkling_tree_executor": "packed"},
    ):
        with pytest.raises((ValueError, TypeError)):
            validate_policy(bad)


def test_ast_controls_reach_the_bound_provider(monkeypatch):
    from mlx2.adapters.external_draft_policy import ExternalDraftAdapterMixin
    from mlx2.adapters.inkling import InklingAdapter

    adapter = object.__new__(InklingAdapter)
    adapter.kernel_policy, adapter.external_policy = validate_policy(
        {"draft_model": "prompt_lookup", "inkling_tree_executor": "packed"}
    )
    monkeypatch.setattr(
        ExternalDraftAdapterMixin, "create_external_batch", lambda self, **kw: kw
    )
    value = adapter.create_external_batch()
    assert value["ast_topology"] == "chain"
    assert value["tree_node_budget_by_lanes"] == {1: 15, 2: 7, 3: 4, 4: 3}
    assert (
        value["target_tree_executor_provider"].protocol
        == "inkling_layerwise_packed_tree_v1"
    )
    assert value["dynamic_tree_overflow"] == "cohort"
    for extra in (
        {"ast_physical_target": True},
        {"ast_tiles": 2},
        {"ast_tile_rows": 32},
    ):
        with pytest.raises(ValueError):
            validate_policy({"draft_model": "prompt_lookup", **extra})
