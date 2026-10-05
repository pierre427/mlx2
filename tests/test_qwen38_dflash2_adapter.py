"""Qwen3.8 27B DFlash2 policy, revision pins and tensor mapping; header-only.

No payload is loaded.  The real-artifact checks skip when the local
checkpoints are absent.
"""
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest

from mlx2.adapters.dflash2 import _expected_weight_shapes, validate_runtime_quantization
from mlx2.runtime.drafters.dflash2_config import DFlash2Config

DRAFT = Path.home() / "mlx-models/Qwen3.8-27B-DFlash2"
TARGET = Path.home() / "mlx-models/Qwen3.8-27B-oQ4e-mtp"
POLICY = Path(__file__).parents[1] / "qualification/policies/qwen38-27b-dflash2.json"


def test_varlen_tree_declares_row_stable_lane_geometry():
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter

    adapter = object.__new__(Qwen3827BAdapter)
    adapter.external_policy = {
        "external_varlen_prefill": True,
        "batch_size_route": "tree15_b1_b4_chain_b5plus_v1",
    }
    assert adapter.lane_policy_defaults() == {
        "max_rows": 128,
        "chunk_above_max": True,
    }
    adapter.external_policy = {}
    assert adapter.lane_policy_defaults() is None


def _draft_config(**dflash):
    config = {
        "architectures": ["DFlash2DraftModel"], "model_type": "qwen3", "dtype": "bfloat16",
        "hidden_size": 16, "intermediate_size": 24, "num_hidden_layers": 2,
        "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
        "vocab_size": 64, "num_target_layers": 8, "rms_norm_eps": 1e-6,
        "max_position_embeddings": 512, "sliding_window": 6,
        "layer_types": ["sliding_attention"] * 2,
        "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
        "dflash_config": {
            "block_size": 8, "mask_token_id": 63, "target_layer_ids": [1, 6],
            "conv_group_size": 4, "conv_kernel_size": 2, "selector_rank": 8,
            "selector_top_k": 4, **dflash,
        },
    }
    return config


def _write_artifacts(tmp_path, *, target_layers=8, draft=None):
    target = tmp_path / "target"
    target.mkdir()
    (target / "config.json").write_text(json.dumps({
        "model_type": "qwen3_5",
        "text_config": {"hidden_size": 16, "vocab_size": 64, "num_hidden_layers": target_layers},
    }))
    (target / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"w": "a.safetensors"}}))
    path = tmp_path / "draft"
    path.mkdir()
    config = draft or _draft_config()
    (path / "config.json").write_text(json.dumps(config))
    shapes = _expected_weight_shapes(DFlash2Config.from_dict(config))
    offset, header = 0, {}
    for name, shape in shapes.items():
        count = 1
        for value in shape:
            count *= value
        header[name] = {"dtype": "BF16", "shape": shape, "data_offsets": [offset, offset + 2 * count]}
        offset += 2 * count
    raw = json.dumps(header).encode()
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(offset))
    return target, path


def _pins(target, draft):
    from mlx2.adapters.dflash2 import content_revision, inspect_drafter
    from mlx2.adapters.qwen38_27b import content_revision as target_revision

    return {
        "draft_model": str(draft),
        "draft_revision": content_revision(inspect_drafter(draft, target)),
        "target_revision": target_revision(target),
    }


def test_pinned_policy_inspects_without_payload(tmp_path):
    from mlx2.adapters.qwen38_27b import inspect_external_policy

    target, draft = _write_artifacts(tmp_path)
    record = inspect_external_policy({**_pins(target, draft), "num_draft": 5}, target)
    assert record["args"].target_layer_ids == [1, 6]
    assert record["runtime_quantization"] is None


def test_qwen_external_policy_enforces_two_and_prefers_three_proposals(tmp_path):
    from mlx2.adapters.qwen38_27b import inspect_external_policy

    target, draft = _write_artifacts(tmp_path)
    pins = _pins(target, draft)
    with pytest.raises(ValueError, match=r"min\(2, num_draft\)"):
        inspect_external_policy(
            {**pins, "num_draft": 7, "minimum_draft_proposals": 1}, target
        )
    assert inspect_external_policy(
        {**pins, "num_draft": 7, "minimum_draft_proposals": 2}, target
    )["args"].block_size == 8


def test_external_policy_accepts_target_varlen_selection(tmp_path):
    from mlx2.adapters.qwen38_27b import inspect_external_policy

    target, draft = _write_artifacts(tmp_path)
    record = inspect_external_policy(
        {
            **_pins(target, draft),
            "num_draft": 7,
            "varlen_dense_mlp": True,
            "external_varlen_prefill": True,
        },
        target,
    )
    assert record["args"].block_size == 8


def test_external_policy_rejects_non_boolean_varlen_prefill(tmp_path):
    from mlx2.adapters.qwen38_27b import inspect_external_policy

    target, draft = _write_artifacts(tmp_path)
    with pytest.raises(ValueError, match="external_varlen_prefill must be boolean"):
        inspect_external_policy(
            {**_pins(target, draft), "external_varlen_prefill": 1}, target
        )


def test_external_varlen_prefill_requires_target_varlen_selection(tmp_path):
    from mlx2.adapters.qwen38_27b import inspect_external_policy

    target, draft = _write_artifacts(tmp_path)
    with pytest.raises(ValueError, match="requires varlen_dense_mlp selection"):
        inspect_external_policy(
            {**_pins(target, draft), "external_varlen_prefill": True}, target
        )

    with pytest.raises(ValueError, match="requires varlen_dense_mlp selection"):
        inspect_external_policy(
            {
                **_pins(target, draft),
                "varlen_dense_mlp": {"enabled": False},
                "external_varlen_prefill": True,
            },
            target,
        )


@pytest.mark.parametrize("value", [True, -1, 1001, 1.5])
def test_external_prefill_coalesce_policy_is_bounded_integer(tmp_path, value):
    from mlx2.adapters.qwen38_27b import inspect_external_policy

    target, draft = _write_artifacts(tmp_path)
    with pytest.raises(ValueError, match="integer from 0 to 1000"):
        inspect_external_policy(
            {
                **_pins(target, draft),
                "varlen_dense_mlp": True,
                "external_varlen_prefill": True,
                "external_prefill_coalesce_ms": value,
            },
            target,
        )


@pytest.mark.parametrize("value", [True, 0, -1, 1.5])
def test_external_prefill_coalesce_min_tokens_is_positive_integer(tmp_path, value):
    from mlx2.adapters.qwen38_27b import inspect_external_policy

    target, draft = _write_artifacts(tmp_path)
    with pytest.raises(ValueError, match="must be a positive integer"):
        inspect_external_policy(
            {
                **_pins(target, draft),
                "varlen_dense_mlp": True,
                "external_varlen_prefill": True,
                "external_prefill_coalesce_min_tokens": value,
            },
            target,
        )


@pytest.mark.parametrize("route", ["tree15_b1_chain_b2plus_v1", "tree15_b1_b4_chain_b5plus_v1"])
def test_bounded_route_requires_pinned_tensorfold_source(tmp_path, monkeypatch, route):
    from mlx2.adapters.qwen38_27b import inspect_external_policy
    from mlx2.runtime import qwen38_tensorfold

    target, draft = _write_artifacts(tmp_path)
    policy = {**_pins(target, draft), "batch_size_route": route}
    monkeypatch.delenv("MLX2_TENSORFOLD_SOURCE", raising=False)
    with pytest.raises(ValueError, match="requires MLX2_TENSORFOLD_SOURCE"):
        inspect_external_policy(policy, target)
    source = tmp_path / "tensorfold"
    source.mkdir()
    monkeypatch.setenv("MLX2_TENSORFOLD_SOURCE", str(source))
    checked = []
    monkeypatch.setattr(qwen38_tensorfold, "_validate", lambda path: checked.append(path))
    assert inspect_external_policy(policy, target)["args"].block_size == 8
    assert checked == [source.resolve()]

    def reject_source(path):
        raise RuntimeError("TensorFold revision mismatch")

    monkeypatch.setattr(qwen38_tensorfold, "_validate", reject_source)
    with pytest.raises(RuntimeError, match="revision mismatch"):
        inspect_external_policy(policy, target)


@pytest.mark.parametrize("extra,error", [
    ({"batch_size_route": "other"}, "unsupported"),
    ({"adaptive_verification": {}}, "chain-only"),
    ({"proposal_composition": {}}, "chain-only"),
    ({"continuation_pool": {}}, "chain-only"),
])
def test_bounded_route_refuses_incompatible_policy(tmp_path, monkeypatch, extra, error):
    from mlx2.adapters.qwen38_27b import inspect_external_policy

    target, draft = _write_artifacts(tmp_path)
    monkeypatch.setenv("MLX2_TENSORFOLD_SOURCE", str(tmp_path))
    policy = {**_pins(target, draft), "batch_size_route": "tree15_b1_b4_chain_b5plus_v1", **extra}
    with pytest.raises(ValueError, match=error):
        inspect_external_policy(policy, target)


@pytest.mark.parametrize(
    "budgets,error",
    [
        ({"1": 15, "2": 7, "4": 3}, "every enabled tree width"),
        ({"1": 15, "2": 7, "3": 4, "4": 2}, "integers from 3 to 15"),
        ({"1": 15, "2": 7, "3": 8, "4": 3}, "must not increase"),
    ],
)
def test_bounded_route_validates_lane_pressure_budgets(
    tmp_path, monkeypatch, budgets, error
):
    from mlx2.adapters.qwen38_27b import inspect_external_policy
    from mlx2.runtime import qwen38_tensorfold

    target, draft = _write_artifacts(tmp_path)
    source = tmp_path / "tensorfold"
    source.mkdir()
    monkeypatch.setenv("MLX2_TENSORFOLD_SOURCE", str(source))
    monkeypatch.setattr(qwen38_tensorfold, "_validate", lambda _path: None)
    policy = {
        **_pins(target, draft),
        "batch_size_route": "tree15_b1_b4_chain_b5plus_v1",
        "tree_node_budget_by_lanes": budgets,
    }
    with pytest.raises(ValueError, match=error):
        inspect_external_policy(policy, target)


@pytest.mark.parametrize("limit", [0, 5, True, "2"])
def test_bounded_route_validates_tensorfold_cohort_limit(
    tmp_path, monkeypatch, limit
):
    from mlx2.adapters.qwen38_27b import inspect_external_policy
    from mlx2.runtime import qwen38_tensorfold

    target, draft = _write_artifacts(tmp_path)
    source = tmp_path / "tensorfold"
    source.mkdir()
    monkeypatch.setenv("MLX2_TENSORFOLD_SOURCE", str(source))
    monkeypatch.setattr(qwen38_tensorfold, "_validate", lambda _path: None)
    policy = {
        **_pins(target, draft),
        "batch_size_route": "tree15_b1_b4_chain_b5plus_v1",
        "tensorfold_cohort_limit": limit,
    }
    with pytest.raises(ValueError, match="tensorfold_cohort_limit"):
        inspect_external_policy(policy, target)


def test_tensorfold_cohort_limit_requires_bounded_route(tmp_path):
    from mlx2.adapters.qwen38_27b import inspect_external_policy

    target, draft = _write_artifacts(tmp_path)
    with pytest.raises(ValueError, match="requires a batch_size_route"):
        inspect_external_policy(
            {**_pins(target, draft), "tensorfold_cohort_limit": 1}, target
        )


@pytest.mark.parametrize("name", [
    "MLX2_DFLASH_TOPOLOGY", "MLX2_QWEN_TARGET_EXECUTION", "MLX2_TENSORFOLD_COHORT_LIMIT",
])
def test_bounded_route_refuses_explicit_topology_override(tmp_path, monkeypatch, name):
    from mlx2.adapters.qwen38_27b import inspect_external_policy

    target, draft = _write_artifacts(tmp_path)
    monkeypatch.setenv("MLX2_TENSORFOLD_SOURCE", str(tmp_path))
    monkeypatch.setenv(name, "1")
    with pytest.raises(ValueError, match="explicit topology override"):
        inspect_external_policy({**_pins(target, draft), "batch_size_route": "tree15_b1_b4_chain_b5plus_v1"}, target)


def test_tensorfold_source_alone_does_not_select_tree(monkeypatch):
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter
    from mlx2.runtime import external_speculative

    adapter = object.__new__(Qwen3827BAdapter)
    adapter.model = object()
    adapter.draft_model = type("Draft", (), {"propose_tree": lambda self: None})()
    adapter.external_policy = {}
    adapter.identity = {"fingerprint": "test"}
    monkeypatch.setattr(Qwen3827BAdapter, "_initialize_external_feedback", lambda self: None)
    monkeypatch.setattr(Qwen3827BAdapter, "_external_num_draft", lambda self: 7)
    monkeypatch.setattr(Qwen3827BAdapter, "_external_execution_config",
                        lambda self, **kwargs: {"route": "chain"})
    monkeypatch.setenv("MLX2_TENSORFOLD_SOURCE", "/pinned/source")
    calls = []
    monkeypatch.setattr(external_speculative, "ExternalDraftBatchGenerator",
                        lambda *args, **kwargs: calls.append(kwargs) or object())

    adapter.create_external_batch(completion_batch_size=1)
    assert calls[-1]["dynamic_singleton_tree"] is False
    assert "batch_size_route" not in adapter.execution_config(max_lanes=4, prefill_step=16)
    with pytest.raises(ValueError, match="explicit batch_size_route"):
        adapter.create_external_batch(dynamic_singleton_tree=True)
    monkeypatch.setenv("MLX2_DFLASH_TOPOLOGY", "tree15")
    with pytest.raises(ValueError, match="explicit batch_size_route"):
        adapter.create_external_batch()
    monkeypatch.delenv("MLX2_DFLASH_TOPOLOGY")

    adapter.external_policy = {"batch_size_route": "tree15_b1_b4_chain_b5plus_v1"}
    adapter.create_external_batch()
    assert calls[-1]["dynamic_singleton_tree"] is True
    assert calls[-1]["dynamic_tree_max_width"] == 4
    assert adapter.execution_config(max_lanes=4, prefill_step=16)["batch_size_route"] == "tree15_b1_b4_chain_b5plus_v1"

    adapter.external_policy = {"batch_size_route": "tree15_b1_chain_b2plus_v1"}
    adapter.create_external_batch()
    assert calls[-1]["dynamic_singleton_tree"] is True
    assert calls[-1]["dynamic_tree_max_width"] == 1
    assert adapter.execution_config(max_lanes=4, prefill_step=16)["batch_size_route"] == "tree15_b1_chain_b2plus_v1"
    with pytest.raises(ValueError, match="cannot be overridden"):
        adapter.create_external_batch(dynamic_tree_max_width=4)


def test_external_varlen_prefill_cohort_policy_moves_to_ingress(monkeypatch):
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter
    from mlx2.runtime import external_speculative

    adapter = object.__new__(Qwen3827BAdapter)
    adapter.model = object()
    adapter.draft_model = object()
    adapter.external_policy = {
        "external_varlen_prefill": True,
        "external_prefill_coalesce_ms": 250,
        "external_prefill_coalesce_min_tokens": 256,
    }
    adapter.identity = {"fingerprint": "test"}
    monkeypatch.setattr(Qwen3827BAdapter, "_initialize_external_feedback", lambda self: None)
    monkeypatch.setattr(Qwen3827BAdapter, "_external_num_draft", lambda self: 4)
    monkeypatch.setattr(
        Qwen3827BAdapter,
        "_external_execution_config",
        lambda self, **kwargs: {"route": "chain"},
    )
    calls = []
    monkeypatch.setattr(
        external_speculative,
        "ExternalDraftBatchGenerator",
        lambda *args, **kwargs: calls.append(kwargs) or object(),
    )

    adapter.create_external_batch()
    assert calls[-1]["external_varlen_prefill"] is True
    assert calls[-1]["external_prefill_coalesce_ms"] == 0
    assert calls[-1]["external_prefill_coalesce_min_tokens"] == 256
    config = adapter.execution_config(max_lanes=4, prefill_step=16)
    assert config["external_varlen_prefill"] == {
        "enabled": True,
        "schema": "mlx2.external-varlen-prefill.v1",
        "law": "right-padded-target-prefill-merge-private-extract-live",
    }
    assert config["ingress_cohort"] == {
        "enabled": True,
        "mechanism": "external_varlen_prefill",
        "maximum_wait_ms": 250,
        "minimum_prompt_tokens": 256,
        "target_lanes": 4,
    }


def test_external_execution_config_normalizes_pairwise_and_adaptive_policy():
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter

    adapter = object.__new__(Qwen3827BAdapter)
    adapter.draft_model = object()
    adapter.external_policy = {"num_draft": 2}
    baseline = adapter.execution_config(max_lanes=4, prefill_step=16)
    assert "pairwise_selection" not in baseline
    assert "adaptive_verification" not in baseline
    assert "external_varlen_prefill" not in baseline
    assert "ingress_cohort" not in baseline

    adapter.external_policy = {
        "num_draft": 2,
        "pairwise_selection": "host",
    }
    assert "pairwise_selection" not in adapter.execution_config(
        max_lanes=4, prefill_step=16
    )

    adapter.external_policy = {
        "num_draft": 2,
        "pairwise_selection": "batched",
        "adaptive_verification": {
            "verification_costs": [1, 2, 4],
            "mode": "per_request",
        },
    }
    selected = adapter.execution_config(max_lanes=4, prefill_step=16)
    assert selected["pairwise_selection"] == "batched"
    adaptive = selected["adaptive_verification"]
    assert adaptive["verification_costs"] == [1.0, 2.0, 4.0]
    assert adaptive["verification_costs_by_cohort"] == []
    assert adaptive["continuation_costs"] == []
    assert adaptive["draft_cost"] == 0.0
    assert adaptive["min_gain"] == 0.05
    assert adaptive["mode"] == "per_request"

    adapter.varlen_dense_mlp = object()
    adapter.external_policy["external_varlen_prefill"] = True
    numerics = adapter._external_execution_numerics()
    assert numerics["external_varlen_prefill"] == {
        "schema": "mlx2.external-varlen-prefill.v1",
        "law": "right-padded-target-prefill-merge-private-extract-live",
    }
    assert numerics["external_adaptive_verification"] == {
        "algorithm": "exact-chain-adaptive-target-verify-width-v1",
        "policy": adaptive,
    }


def test_external_varlen_prefill_has_a_distinct_counter_free_identity():
    from mlx2.runtime.prefill_plan import (
        EXTERNAL_VARLEN_PREFILL_IDENTITY,
        apc_prefill_fingerprint,
        execution_identity,
    )

    off = execution_identity()
    on = execution_identity(
        external_varlen_prefill=EXTERNAL_VARLEN_PREFILL_IDENTITY
    )
    assert off is None
    assert on == {
        "version": 1,
        "external_varlen_prefill": {
            "schema": "mlx2.external-varlen-prefill.v1",
            "law": "right-padded-target-prefill-merge-private-extract-live",
        },
    }
    assert apc_prefill_fingerprint("base", off) != apc_prefill_fingerprint(
        "base", on
    )


def test_explicit_b1_route_receipt_names_policy_and_remains_unqualified():
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    route = SimpleNamespace(model=object(), dynamic_singleton_tree=True,
                            dynamic_tree_max_width=1, _auto_active_width=1,
                            _auto_cohort_width=1, draft_topology="tree15",
                            target_execution="tensorfold", tensorfold_cohort_limit=1,
                            scheduler_stats={"external_tensorfold_cohort_max_width": 1,
                                             "external_tensorfold_cohort_rounds": 0})
    receipt = ExternalDraftBatchGenerator._target_execution_receipt(route)["batch_size_route"]
    assert receipt == {"policy": "tree15_b1_chain_b2plus_v1", "max_tree_width": 1,
                       "tree_node_budget_by_lanes": {"1": 15},
                       "current_tree_node_budget": 15,
                       "active_lanes": 1, "cohort_width": 1,
                       "current": "tree15_tensorfold", "qualified": False}


def test_tree_topology_override_requires_explicit_route_before_artifact_load(tmp_path, monkeypatch):
    from mlx2.adapters.qwen38_27b import inspect_external_policy

    target, draft = _write_artifacts(tmp_path)
    policy = _pins(target, draft)
    monkeypatch.setenv("MLX2_TENSORFOLD_SOURCE", str(tmp_path / "unused"))
    assert inspect_external_policy(policy, target)["args"].block_size == 8
    monkeypatch.setenv("MLX2_DFLASH_TOPOLOGY", "tree15")
    with pytest.raises(ValueError, match="explicit batch_size_route"):
        inspect_external_policy(policy, target)


@pytest.mark.parametrize("key", ["draft_revision", "target_revision"])
def test_missing_or_wrong_revision_pin_fails_closed(tmp_path, key):
    from mlx2.adapters.qwen38_27b import inspect_external_policy

    target, draft = _write_artifacts(tmp_path)
    policy = _pins(target, draft)
    with pytest.raises(ValueError, match="must pin"):
        inspect_external_policy({k: v for k, v in policy.items() if k != key}, target)
    with pytest.raises(ValueError, match="revision mismatch"):
        inspect_external_policy({**policy, key: "0" * 64}, target)


def test_changed_target_or_draft_content_breaks_the_pin(tmp_path):
    from mlx2.adapters.qwen38_27b import inspect_external_policy

    target, draft = _write_artifacts(tmp_path)
    policy = _pins(target, draft)
    config = json.loads((draft / "config.json").read_text())
    config["dflash_config"]["mask_token_id"] = 62
    (draft / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="draft revision mismatch"):
        inspect_external_policy(policy, target)
    index = target / "model.safetensors.index.json"
    index.write_text(index.read_text() + " ")
    with pytest.raises(ValueError, match="target revision mismatch"):
        inspect_external_policy(policy, target)


def test_topology_mismatch_fails_before_pins_matter(tmp_path):
    from mlx2.adapters.dflash2 import inspect_drafter

    target, draft = _write_artifacts(tmp_path, target_layers=6)
    with pytest.raises(ValueError, match="layer count"):
        inspect_drafter(draft, target)


@pytest.mark.parametrize(
    "extra, error",
    [
        ({"num_draft": 8}, "below the draft block size"),
        ({"num_draft": 0}, "below the draft block size"),
        ({"pairwise_selection": "gpu"}, "pairwise_selection"),
        ({"gdn_core": True}, "unknown keys"),
        ({"draft_quantization": {"bits": 3, "group_size": 64}}, "draft_quantization"),
    ],
)
def test_policy_negative_cases(tmp_path, extra, error):
    from mlx2.adapters.qwen38_27b import inspect_external_policy

    target, draft = _write_artifacts(tmp_path)
    with pytest.raises(ValueError, match=error):
        inspect_external_policy({**_pins(target, draft), **extra}, target)


def test_runtime_quantization_is_part_of_the_draft_identity(tmp_path):
    from mlx2.adapters.qwen38_27b import inspect_external_policy

    target, draft = _write_artifacts(tmp_path)
    policy = _pins(target, draft)
    plain = inspect_external_policy(policy, target)
    q4 = inspect_external_policy(
        {**policy, "draft_quantization": {"bits": 4, "group_size": 64}}, target
    )
    assert q4["fingerprint"] != plain["fingerprint"]
    assert q4["runtime_quantization"] == {"bits": 4, "group_size": 64}
    assert validate_runtime_quantization(None) is None


def _mapped_parameter_shapes(config):
    """Checkpoint names/shapes after ``sanitize``, against the lazy module tree."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    from mlx2.runtime.drafters.dflash2 import DFlash2DraftModel

    args = DFlash2Config.from_dict(config)
    model = DFlash2DraftModel(args)  # parameters stay lazy: nothing allocates
    module = {name: list(value.shape) for name, value in tree_flatten(model.parameters())}
    fake = {name: mx.zeros((1,)) for name in _expected_weight_shapes(args)}
    renamed = model.sanitize(fake)
    expected = _expected_weight_shapes(args)
    mapped = {}
    for source, target_name in zip(fake, renamed):
        mapped[target_name] = expected[source]
    return module, mapped


def test_tensor_mapping_covers_every_draft_parameter():
    module, mapped = _mapped_parameter_shapes(_draft_config())
    assert module == mapped


def test_real_checkpoint_headers_map_onto_the_drafter():
    if not (DRAFT / "config.json").exists() or not (TARGET / "config.json").exists():
        pytest.skip("local Qwen3.8 27B DFlash2 artifacts are not installed")
    from mlx2.adapters.dflash2 import _read_safetensors_header
    from mlx2.adapters.qwen38_27b import inspect_external_policy

    policy = json.loads(POLICY.read_text())
    record = inspect_external_policy(policy, TARGET)
    args = record["args"]
    assert args.target_layer_ids == [5, 19, 33, 47, 61]
    assert (args.block_size, args.sliding_window, args.selector_top_k, args.selector_rank) == (8, 2048, 16, 256)
    assert (args.conv_kernel_size, args.conv_group_size, args.mask_token_id) == (2, 16, 248070)
    _, header, _ = _read_safetensors_header(DRAFT / "model.safetensors")
    header.pop("__metadata__", None)
    assert len(header) == 81
    config = json.loads((DRAFT / "config.json").read_text())
    module, mapped = _mapped_parameter_shapes(config)
    assert module == mapped
    assert {name: record["shape"] for name, record in header.items()} == _expected_weight_shapes(args)
