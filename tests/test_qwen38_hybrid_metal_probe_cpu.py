"""CPU contract evidence for the Qwen3.8 hybrid Metal probe boundary.

The success cell uses a test-only ready attestation while MLX executes on CPU;
it validates lifecycle and exact arithmetic only and is not Metal evidence.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import mlx.core as mx
import pytest
from mlx import nn

from mlx2.adapters import qwen38_hybrid_metal_probe as probe
from mlx2.runtime import flat_segmented_hybrid as hybrid_owner
from mlx2.adapters.qwen38_hybrid_metal_probe import (
    PHYSICAL_LAYOUT,
    QWEN38_27B_OQ4E_MTP_CHECKPOINT,
    QWEN38_27B_OQ4E_MTP_REVIEWED_MANIFEST,
    Qwen38HybridMetalFlatTarget,
    diagnose_qwen38_hybrid_model_manifest,
    make_reviewed_qwen38_27b_metal_target,
    qwen38_hybrid_model_manifest,
    qwen38_hybrid_model_manifest_sha256,
    reviewed_qwen38_27b_artifact_metadata,
    run_qwen38_hybrid_metal_flat_probe,
)
from mlx2.runtime.flat_segmented_ast import (
    DEVICE_ATTESTATION_SCHEMA,
    FlatSegmentDescriptor,
    query_metal_flat_segment_admission,
)
from mlx2.runtime.flat_segmented_hybrid import (
    FlatSegmentedHybridTransaction,
    HybridLiveStateBoundary,
    MetalFlatSegmentedHybridRows,
    canonical_hybrid_state,
)
from mlx2.runtime.models.cache import KVCache
from mlx2.runtime.models.qwen38_27b import Model, ModelArgs

SOURCE_REVISION = "358df51a55a6c5da1482a7d738c7d31c33108a15"
ARTIFACT = (
    "tiny-qwen38-hybrid-o4-mtp-test-fixture@sha256:"
    "4f8b28418ad81ce85040c7f621dd48418b203eeb6b6aad5bc856350f27777df1"
)
OWNER_SEMANTIC_SHA256 = (
    "9647f7809789677a70de10d22115bd3514da531b6dbece01c71142cb7b26cb0e"
)
STATE_FINGERPRINTS = (
    "5cce324cf8b59feb02ac4362c53872b62b19d06fc9e0ceab478d2a867ea88a2e",
    "8491ce6cf23f5a337f6ec7a052bdfb8cd036c01ecf2b6e51e773d6798a764cbe",
)
MODEL_MANIFEST_SHA256 = (
    "10a88ba89fb90c1cd2d85fa6a8a48623522547f0e0fdd5ba3235f3290bdf1d63"
)
OWNER_SOURCE_PATHS = (
    "src/mlx2/runtime/flat_segmented_hybrid.py",
    "src/mlx2/runtime/flat_segmented_ast.py",
    "src/mlx2/runtime/flat_segmented_kv.py",
    "src/mlx2/runtime/hybrid_verify_rows.py",
    "src/mlx2/runtime/hybrid_speculative.py",
    "src/mlx2/runtime/segmented_batch_cache.py",
    "src/mlx2/runtime/segmented_plain_kv.py",
    "src/mlx2/runtime/ast_execution_capability.py",
    "src/mlx2/runtime/cache_operations.py",
    "src/mlx2/runtime/models/cache.py",
    "src/mlx2/runtime/models/import_env.py",
    "src/mlx2/runtime/models/invariant_prefill.py",
    "src/mlx2/runtime/models/moe_nax_gather.py",
    "src/mlx2/runtime/models/qsdpa_verify_metal.py",
    "src/mlx2/runtime/models/qwen4_exp.py",
    "src/mlx2/runtime/models/qwen4_qsa_indexed.py",
    "src/mlx2/runtime/models/served_exp.py",
    "src/mlx2/runtime/paged_kv_cache.py",
    "src/mlx2/runtime/prefill_checkpoint_rows.py",
    "src/mlx2/runtime/round_levers.py",
)
TARGET_SOURCE_PATHS = (
    "src/mlx2/adapters/qwen38_hybrid_metal_probe.py",
    "src/mlx2/runtime/flat_segmented_hybrid.py",
    "src/mlx2/runtime/flat_segmented_ast.py",
    "src/mlx2/runtime/flat_segmented_kv.py",
    "src/mlx2/runtime/hybrid_verify_rows.py",
    "src/mlx2/runtime/hybrid_speculative.py",
    "src/mlx2/runtime/segmented_batch_cache.py",
    "src/mlx2/runtime/segmented_plain_kv.py",
    "src/mlx2/runtime/models/qwen_hybrid_row_exact.py",
    "src/mlx2/runtime/models/qwen4_attn_rows.py",
    "src/mlx2/runtime/models/qwen4_attn_window.py",
    "src/mlx2/runtime/models/row_exact_components.py",
    "src/mlx2/runtime/models/row_exact_qmv.py",
    "src/mlx2/runtime/models/qwen38_27b.py",
    "src/mlx2/runtime/models/qwen3_5.py",
    "src/mlx2/runtime/models/qwen38_fused_gdn.py",
    "src/mlx2/runtime/models/gdn_kernel_contract.py",
    "src/mlx2/runtime/models/import_env.py",
    "src/mlx2/runtime/models/qwen4_fused_gdn.py",
    "src/mlx2/runtime/models/qwen4_fused_gdn_verify.py",
    "src/mlx2/runtime/models/qwen4_fused_gdn_prefill.py",
    "src/mlx2/runtime/models/qwen_gdn_contracts.py",
    "src/mlx2/runtime/models/gdn_state.py",
    "src/mlx2/runtime/models/gated_delta.py",
    "src/mlx2/runtime/models/invariant_prefill.py",
    "src/mlx2/runtime/models/moe_nax_gather.py",
    "src/mlx2/runtime/models/qwen4_exp.py",
    "src/mlx2/runtime/models/qwen4_routed_decode.py",
    "src/mlx2/runtime/models/qwen3_next.py",
    "src/mlx2/runtime/models/precise_ops.py",
    "src/mlx2/runtime/models/base.py",
    "src/mlx2/runtime/models/rope_utils.py",
    "src/mlx2/runtime/models/activations.py",
    "src/mlx2/runtime/models/pipeline.py",
    "src/mlx2/runtime/models/served_exp.py",
    "src/mlx2/runtime/progressive_external_verify.py",
    "src/mlx2/runtime/prefill_checkpoint_rows.py",
    "src/mlx2/runtime/ragged_verify_observation.py",
    "src/mlx2/runtime/round_levers.py",
    "src/mlx2/runtime/paged_kv_cache.py",
    "src/mlx2/runtime/models/cache.py",
    "src/mlx2/runtime/row_exact/ops.py",
    "src/mlx2/runtime/row_exact/canary.py",
    "src/mlx2/runtime/row_exact/__init__.py",
    "src/mlx2/runtime/row_exact/planner.py",
    "src/mlx2/runtime/row_exact/recipes.py",
    "src/mlx2/runtime/row_exact_verify.py",
    "src/mlx2/adapters/dense_qwen3_flat.py",
)
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def _sources(paths):
    return {
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in paths
    }


def _identity():
    return {
        "source_revision": SOURCE_REVISION,
        "build": f"mlx-{mx.__version__}",
        "device": "mlx-metal:unavailable",
        "artifact": ARTIFACT,
    }


def _tiny_quantized_mtp():
    text = dict(
        model_type="qwen3_5_text",
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=128,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        full_attention_interval=4,
        mtp_num_hidden_layers=1,
        partial_rotary_factor=0.5,
        rope_parameters=None,
        max_position_embeddings=512,
    )
    mx.random.seed(7109)
    model = Model(ModelArgs(model_type="qwen3_5", text_config=text))
    model.eval()
    nn.quantize(model, group_size=32, bits=4)
    # The real artifact keeps the embedded drafter's fusion projection
    # unquantized.  It is outside the target executor's tensor surface.
    model.language_model.mtp.fc = nn.Linear(64, 32, bias=False)
    mx.eval(model.parameters())
    return model


def _pair(lengths=(1, 2, 3, 4)):
    model = _tiny_quantized_mtp()
    identity = _identity()
    target = Qwen38HybridMetalFlatTarget(
        model,
        capture_layers=(1, 6),
        identity=identity,
        expected_model_manifest_sha256=MODEL_MANIFEST_SHA256,
    )
    owner = target.make_bound_rows(
        tuple((1 + lane, 7, 11) for lane in range(len(lengths)))
    )
    descriptor = FlatSegmentDescriptor.from_query_lengths(lengths)
    return descriptor, owner, target


def _expected():
    return {
        "expected_identity": _identity(),
        "expected_owner_semantic_contract_sha256": OWNER_SEMANTIC_SHA256,
        "expected_state_contract_fingerprints": STATE_FINGERPRINTS,
        "expected_owner_source_hashes": _sources(OWNER_SOURCE_PATHS),
        "expected_target_source_hashes": _sources(TARGET_SOURCE_PATHS),
    }


def _ready(target, monkeypatch):
    identity = _identity()
    monkeypatch.setattr(
        target,
        "ast_flat_segment_device_attestation",
        lambda: {
            "schema": DEVICE_ATTESTATION_SCHEMA,
            "backend": "metal",
            "device": identity["device"],
            "build": identity["build"],
            "ready": True,
        },
    )


def _run(owner, target, descriptor):
    return run_qwen38_hybrid_metal_flat_probe(
        owner,
        target,
        descriptor,
        tuple(range(13, 13 + descriptor.logical_rows)),
        (1, 1, 2, 3),
        (41, 42, 43, 44),
        **_expected(),
    )


def test_cpu_declines_metal_without_touching_live_hybrid_state():
    descriptor, owner, target = _pair()
    before = canonical_hybrid_state(owner.rows)
    admission = query_metal_flat_segment_admission(
        owner, target, descriptor, **_expected()
    )
    assert admission.available is False
    assert admission.decline_reasons == ("metal_device_unavailable",)
    with pytest.raises(RuntimeError, match="metal_device_unavailable"):
        _run(owner, target, descriptor)
    assert canonical_hybrid_state(owner.rows) == before
    assert owner.revision == 0 and owner._active is None
    assert target.invocations == 0


def test_b4_1234_host_contract_matches_logits_taps_kv_gdn_and_continuation(
    monkeypatch,
):
    descriptor, owner, target = _pair()
    prior_boundaries = owner.boundaries
    _ready(target, monkeypatch)
    result = _run(owner, target, descriptor)
    assert result.receipt["logical_rows"] == 10
    assert result.receipt["physical_rows"] == 16
    assert result.receipt["padding_rows"] == 6
    assert result.receipt["initial_boundary_positions"] == [3, 3, 3, 3]
    assert result.receipt["initial_boundary_generations"] == [0, 0, 0, 0]
    assert result.receipt["bound_prefix_rows_per_lane_hard_limit"] == 8
    assert result.receipt["target_invocations"] == 1
    assert result.receipt["physical_layout"] == PHYSICAL_LAYOUT
    assert result.receipt["attention_window_row_exact"] is False
    assert result.receipt["embedded_mtp_present"] is True
    assert result.receipt["embedded_mtp_layers"] == 1
    assert result.receipt["embedded_mtp_engaged"] is False
    assert result.receipt["packed_matmul_authority"] is False
    assert result.receipt["ordinary_serial_oracle"] == {
        "full_logits": "bit_exact",
        "capture_taps": "bit_exact",
        "kv_and_gdn_state": "bit_exact_prepublication",
        "continuation_logits_and_taps": "bit_exact",
    }
    assert owner.revision == 1 and owner._active is None
    assert all(boundary.valid is False for boundary in prior_boundaries)
    assert all(boundary.valid is True for boundary in owner.boundaries)
    assert all(
        current.generation == prior.generation + 1
        and current.position == prior.position + accepted
        and current.revision == ARTIFACT
        for current, prior, accepted in zip(
            owner.boundaries, prior_boundaries, (1, 1, 2, 3), strict=True
        )
    )
    assert target.invocations == 1
    route_counts = target._route.status()["counts"]
    assert route_counts["launches"] == 1
    assert route_counts["row_exact"] == 1
    assert route_counts["physical_width_hist"] == {"4": 1}
    compact = route_counts["stages"]["valid_row_compaction"]
    assert sum(
        count
        for route, count in compact.items()
        if route.endswith("_padding_skipped")
    ) > 0


def test_continuation_mismatch_aborts_private_successor(monkeypatch):
    descriptor, owner, target = _pair()
    prior_boundaries = owner.boundaries
    _ready(target, monkeypatch)
    before = canonical_hybrid_state(owner.rows)
    calls = 0

    def comparison(_actual, _expected_outputs):
        nonlocal calls
        calls += 1
        return calls == 1

    monkeypatch.setattr(probe, "_outputs_exact", comparison)
    with pytest.raises(ValueError, match="continuation oracle differs"):
        _run(owner, target, descriptor)
    assert canonical_hybrid_state(owner.rows) == before
    assert owner.boundaries == prior_boundaries
    assert all(boundary.valid for boundary in prior_boundaries)
    assert owner.revision == 1 and owner._active is None


def test_publish_failure_restores_all_live_planes(monkeypatch):
    descriptor, owner, target = _pair()
    prior_boundaries = owner.boundaries
    _ready(target, monkeypatch)
    before = canonical_hybrid_state(owner.rows)
    original = FlatSegmentedHybridTransaction._assign
    calls = 0

    def fail_second(live, successor):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected hybrid publish failure")
        return original(live, successor)

    monkeypatch.setattr(
        FlatSegmentedHybridTransaction,
        "_assign",
        staticmethod(fail_second),
    )
    with pytest.raises(RuntimeError, match="injected hybrid publish failure"):
        _run(owner, target, descriptor)
    assert canonical_hybrid_state(owner.rows) == before
    assert owner.boundaries == prior_boundaries
    assert all(boundary.valid for boundary in prior_boundaries)
    assert owner.revision == 1 and owner._active is None


def test_post_assignment_verification_failure_restores_boundaries(monkeypatch):
    descriptor, owner, target = _pair()
    prior_boundaries = owner.boundaries
    _ready(target, monkeypatch)
    before = canonical_hybrid_state(owner.rows)
    initial_positions = tuple(boundary.position for boundary in prior_boundaries)
    original = hybrid_owner.canonical_bound_hybrid_state

    def fail_after_assignment(rows, boundaries):
        if rows is not None and any(
            cache.offset != initial_positions[lane]
            for lane, row in enumerate(rows)
            for cache in row
            if hasattr(cache, "offset")
        ):
            raise RuntimeError("injected post-assignment verification failure")
        return original(rows, boundaries)

    monkeypatch.setattr(hybrid_owner, "canonical_bound_hybrid_state", fail_after_assignment)
    with pytest.raises(RuntimeError, match="post-assignment verification failure"):
        _run(owner, target, descriptor)
    assert canonical_hybrid_state(owner.rows) == before
    assert owner.boundaries == prior_boundaries
    assert all(boundary.valid for boundary in prior_boundaries)
    assert owner.revision == 1 and owner._active is None


def test_quantized_layout_drift_declines_before_oracle_or_begin(monkeypatch):
    descriptor, owner, target = _pair()
    _ready(target, monkeypatch)
    target.model.language_model.model.layers[0].mlp.up_proj = nn.Linear(32, 64)
    calls = 0

    def forbidden(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("declined pair must not run ordinary oracle")

    monkeypatch.setattr(probe, "_ordinary_serial_oracles", forbidden)
    with pytest.raises(RuntimeError, match="target_capability_invalid"):
        _run(owner, target, descriptor)
    assert calls == 0
    assert owner.revision == 0 and owner._active is None
    assert target.invocations == 0


def test_unquantized_target_head_rejects_before_route_install():
    model = _tiny_quantized_mtp()
    model.language_model.lm_head = nn.Linear(32, 128, bias=False)
    with pytest.raises(ValueError, match="unquantized target tensor surface"):
        Qwen38HybridMetalFlatTarget(
            model,
            capture_layers=(1, 6),
            identity=_identity(),
            expected_model_manifest_sha256=MODEL_MANIFEST_SHA256,
        )
    assert "_mlx2_qwen_hybrid_row_exact" not in model.__dict__


@pytest.mark.parametrize("lengths", [(5,), (4, 4, 4, 4)])
def test_probe_limits_decline_before_tensor_or_state_work(lengths, monkeypatch):
    descriptor, owner, target = _pair(lengths)
    _ready(target, monkeypatch)
    before = canonical_hybrid_state(owner.rows)
    calls = 0

    def forbidden(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("over-limit unit must not run ordinary oracle")

    monkeypatch.setattr(probe, "_ordinary_serial_oracles", forbidden)
    accepted = tuple(1 for _ in lengths)
    continuation = tuple(2 for _ in lengths)
    with pytest.raises(RuntimeError, match="descriptor_exceeds_hard_limits"):
        run_qwen38_hybrid_metal_flat_probe(
            owner,
            target,
            descriptor,
            tuple(3 for _ in range(descriptor.logical_rows)),
            accepted,
            continuation,
            **_expected(),
        )
    assert calls == 0
    assert canonical_hybrid_state(owner.rows) == before
    assert owner.revision == 0 and target.invocations == 0


def test_target_issued_owner_rejects_more_than_four_lanes_before_execution():
    model = _tiny_quantized_mtp()
    target = Qwen38HybridMetalFlatTarget(
        model,
        capture_layers=(1, 6),
        identity=_identity(),
        expected_model_manifest_sha256=MODEL_MANIFEST_SHA256,
    )
    with pytest.raises(ValueError, match="prefixes exceed probe limits"):
        target.make_bound_rows(((1,), (2,), (3,), (4,), (5,)))
    assert target.invocations == 0


def test_reviewed_source_and_artifact_identity_fail_closed(monkeypatch):
    descriptor, owner, target = _pair()
    _ready(target, monkeypatch)
    expected = _expected()
    expected["expected_target_source_hashes"] = {
        **expected["expected_target_source_hashes"],
        "src/mlx2/adapters/unreviewed.py": "0" * 64,
    }
    admission = query_metal_flat_segment_admission(
        owner, target, descriptor, **expected
    )
    assert admission.available is False
    assert "target_source_hashes_mismatch" in admission.decline_reasons
    expected = _expected()
    expected["expected_identity"] = {
        **expected["expected_identity"],
        "artifact": "other@sha256:" + "0" * 64,
    }
    admission = query_metal_flat_segment_admission(
        owner, target, descriptor, **expected
    )
    assert admission.available is False
    assert "target_identity_mismatch" in admission.decline_reasons
    assert "owner_identity_mismatch" in admission.decline_reasons


def test_uninitialized_gdn_and_topology_drift_fail_before_begin():
    model = _tiny_quantized_mtp()
    identity = _identity()
    direct_target = Qwen38HybridMetalFlatTarget(
        model,
        capture_layers=(1, 6),
        identity=identity,
        expected_model_manifest_sha256=MODEL_MANIFEST_SHA256,
    )
    with pytest.raises(ValueError, match="prefixes exceed probe limits"):
        direct_target.make_bound_rows([()])

    descriptor, owner, target = _pair((1,))
    target.model.language_model.mtp = None
    admission = query_metal_flat_segment_admission(
        owner, target, descriptor, **_expected()
    )
    assert admission.available is False
    assert "target_capability_invalid" in admission.decline_reasons
    assert owner.revision == 0 and owner._active is None


def test_externally_reviewed_model_manifest_rejects_another_valid_hybrid():
    model = _tiny_quantized_mtp()
    classes = tuple(type(module) for _, module in model.named_modules())
    assert qwen38_hybrid_model_manifest_sha256(model) == MODEL_MANIFEST_SHA256
    model.language_model.args.max_position_embeddings += 1
    assert qwen38_hybrid_model_manifest_sha256(model) != MODEL_MANIFEST_SHA256
    with pytest.raises(ValueError, match="differs from reviewed manifest"):
        Qwen38HybridMetalFlatTarget(
            model,
            capture_layers=(1, 6),
            identity=_identity(),
            expected_model_manifest_sha256=MODEL_MANIFEST_SHA256,
        )
    assert "_mlx2_qwen_hybrid_row_exact" not in model.__dict__
    assert tuple(type(module) for _, module in model.named_modules()) == classes


def test_reviewed_manifest_rejects_text_model_type_drift_before_install():
    model = _tiny_quantized_mtp()
    assert model.language_model.args.model_type == "qwen3_5_text"
    assert qwen38_hybrid_model_manifest_sha256(model) == MODEL_MANIFEST_SHA256
    model.language_model.args.model_type = "qwen3_5"
    assert qwen38_hybrid_model_manifest_sha256(model) != MODEL_MANIFEST_SHA256
    with pytest.raises(ValueError, match="differs from reviewed manifest"):
        Qwen38HybridMetalFlatTarget(
            model,
            capture_layers=(1, 6),
            identity=_identity(),
            expected_model_manifest_sha256=MODEL_MANIFEST_SHA256,
        )
    assert "_mlx2_qwen_hybrid_row_exact" not in model.__dict__


def test_live_manifest_uses_safetensors_dtype_vocabulary():
    manifest = qwen38_hybrid_model_manifest(_tiny_quantized_mtp())
    dtypes = {
        dtype
        for row in manifest["quantized_modules"]
        for _, _, dtype in row[-1]
    }
    assert dtypes == {"uint32", "float32"}
    assert all(not dtype.startswith("mlx.core.") for dtype in dtypes)


def test_constructor_failure_after_install_rolls_back_new_route(monkeypatch):
    model = _tiny_quantized_mtp()
    prior_target_rows = object()
    object.__setattr__(model, "progressive_external_target_rows", prior_target_rows)
    classes = tuple(type(module) for _, module in model.named_modules())

    def fail_after_install(self):
        assert self.model.__dict__.get("_mlx2_qwen_hybrid_row_exact") is self._route
        raise RuntimeError("injected post-install validation failure")

    monkeypatch.setattr(Qwen38HybridMetalFlatTarget, "_validate_model", fail_after_install)
    with pytest.raises(RuntimeError, match="post-install validation failure"):
        Qwen38HybridMetalFlatTarget(
            model,
            capture_layers=(1, 6),
            identity=_identity(),
            expected_model_manifest_sha256=MODEL_MANIFEST_SHA256,
        )
    assert "_mlx2_qwen_hybrid_row_exact" not in model.__dict__
    assert model.__dict__["progressive_external_target_rows"] is prior_target_rows
    assert tuple(type(module) for _, module in model.named_modules()) == classes


def test_reviewed_real_artifact_metadata_is_header_only_and_not_self_echo():
    path = Path.home() / "mlx-models" / "Qwen3.8-27B-oQ4e-mtp"
    if not path.exists():
        pytest.skip("reviewed Qwen3.8 artifact is not installed")
    result = reviewed_qwen38_27b_artifact_metadata(path)
    assert result == {
        "checkpoint": QWEN38_27B_OQ4E_MTP_CHECKPOINT,
        **QWEN38_27B_OQ4E_MTP_REVIEWED_MANIFEST,
    }


def test_manifest_diagnostic_is_bounded_and_non_authoritative():
    path = Path.home() / "mlx-models" / "Qwen3.8-27B-oQ4e-mtp"
    if not path.exists():
        pytest.skip("reviewed Qwen3.8 artifact is not installed")
    report = diagnose_qwen38_hybrid_model_manifest(
        _tiny_quantized_mtp(), path, max_items=3
    )
    assert report["authoritative"] is False
    assert report["match"] is False
    assert report["live_manifest_sha256"] == MODEL_MANIFEST_SHA256
    assert report["reviewed_manifest_sha256"] == (
        QWEN38_27B_OQ4E_MTP_REVIEWED_MANIFEST["model_manifest_sha256"]
    )
    for group in report["quantized_module_differences"].values():
        assert len(group["first"]) <= 3
        assert group["count"] >= len(group["first"])
    assert report["text_arg_differences"]["count"] > 0


def test_reviewed_real_target_factory_rejects_unbound_fixture_before_install():
    model = _tiny_quantized_mtp()
    identity = {
        **_identity(),
        "artifact": QWEN38_27B_OQ4E_MTP_CHECKPOINT,
    }
    metadata = {
        "checkpoint": QWEN38_27B_OQ4E_MTP_CHECKPOINT,
        **QWEN38_27B_OQ4E_MTP_REVIEWED_MANIFEST,
    }
    with pytest.raises(ValueError, match="differs from reviewed manifest"):
        make_reviewed_qwen38_27b_metal_target(
            model,
            capture_layers=(1, 6),
            identity=identity,
            artifact_metadata=metadata,
        )
    assert "_mlx2_qwen_hybrid_row_exact" not in model.__dict__


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_gdn_lineage",
        "wrong_gdn_state",
        "wrong_kv_state",
        "wrong_boundary_offset",
        "wrong_revision",
    ],
)
def test_common_kv_gdn_boundary_drift_declines_before_begin_or_oracle(mutation):
    descriptor, owner, target = _pair((1, 1))
    if mutation in ("wrong_gdn_state", "wrong_kv_state"):
        if mutation == "wrong_gdn_state":
            recurrent = owner.boundaries[0].caches[0]
            recurrent.cache[0] = recurrent.cache[0] + mx.array(
                1, dtype=recurrent.cache[0].dtype
            )
        else:
            attention = next(
                cache for cache in owner.rows[0] if type(cache) is KVCache
            )
            attention.keys = attention.keys + mx.array(
                1, dtype=attention.keys.dtype
            )
        admission = query_metal_flat_segment_admission(
            owner, target, descriptor, **_expected()
        )
        assert admission.available is False
        assert "target_owner_compatibility_invalid" in admission.decline_reasons
        assert owner.revision == 0 and owner._active is None
        assert target.invocations == 0
        return
    boundaries = list(owner.boundaries)
    first = boundaries[0]
    boundaries[0] = HybridLiveStateBoundary._mint(
        revision=("other-artifact" if mutation == "wrong_revision" else first.revision),
        lane_id=first.lane_id,
        position=first.position + (1 if mutation == "wrong_boundary_offset" else 0),
        generation=first.generation,
        row=(
            owner.rows[1]
            if mutation == "wrong_gdn_lineage"
            else owner.rows[0]
        ),
        issuer_id=id(target),
    )
    owner._boundaries = tuple(boundaries)
    admission = query_metal_flat_segment_admission(
        owner, target, descriptor, **_expected()
    )
    assert admission.available is False
    assert "target_owner_compatibility_invalid" in admission.decline_reasons
    assert owner.revision == 0 and owner._active is None
    assert target.invocations == 0


def test_abort_and_stale_ticket_preserve_authoritative_membership(monkeypatch):
    descriptor, owner, target = _pair((1,))
    _ready(target, monkeypatch)
    admission = query_metal_flat_segment_admission(
        owner, target, descriptor, **_expected()
    )
    assert admission.available
    before = canonical_hybrid_state(owner.rows)
    owner._rows = tuple(tuple(copy.deepcopy(row)) for row in owner.rows)
    with pytest.raises(ValueError, match="GDN lineage"):
        owner.begin_flat(
            descriptor,
            target_executor=target,
            admission=admission,
        )
    assert canonical_hybrid_state(owner.rows) == before
    assert owner.revision == 0 and owner._active is None


def test_receipt_does_not_promote_cpu_contract_to_metal_evidence():
    _descriptor, _owner, target = _pair()
    receipt = target.receipt()
    assert receipt["implemented"] is True
    assert receipt["qualified"] is False
    assert receipt["selected"] is False
    assert receipt["default"] is False
    assert receipt["observed_used"] is False
    assert receipt["device_admitted"] is False
    assert receipt["physical_layout"] == PHYSICAL_LAYOUT
    assert receipt["embedded_mtp_present"] is True
    assert receipt["embedded_mtp_layers"] == 1
    assert receipt["embedded_mtp_engaged"] is False
