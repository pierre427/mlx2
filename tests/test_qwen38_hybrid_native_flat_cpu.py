"""CPU evidence for the default-off Qwen3.8 native-flat Metal boundary."""

from __future__ import annotations

import copy
import hashlib
from contextlib import contextmanager
from pathlib import Path

import mlx.core as mx
import pytest
from mlx import nn

from mlx2.adapters import qwen38_hybrid_native_flat as native
from mlx2.adapters.qwen38_hybrid_metal_probe import (
    Qwen38HybridMetalFlatTarget,
)
from mlx2.adapters.qwen38_hybrid_native_flat import (
    PHYSICAL_LAYOUT,
    Qwen38HybridNativeFlatTarget,
    run_qwen38_hybrid_native_flat_probe,
)
from mlx2.runtime import row_exact_verify as REV
from mlx2.runtime.flat_segmented_ast import (
    DEVICE_ATTESTATION_SCHEMA,
    FlatSegmentDescriptor,
    FlatSegmentError,
    FlatSegmentUnsupported,
    query_native_flat_segment_admission,
    validate_native_flat_segment_execution,
)
from mlx2.runtime.flat_segmented_hybrid import (
    canonical_bound_hybrid_state,
)
from mlx2.runtime.hybrid_verify_rows import HybridVerifyRows
from mlx2.runtime.models import invariant_prefill, varlen_dense_mlp
from mlx2.runtime.models.cache import KVCache
from mlx2.runtime.models.qwen38_27b import Model, ModelArgs
from mlx2.runtime.ragged_verify_observation import RaggedVerifyObserver, activate

ROOT = Path(__file__).resolve().parents[1]
SOURCE_REVISION = "442a49d1de8027dfb57a7935623cef3025e746c1"
ARTIFACT = (
    "tiny-qwen38-native-flat-o4-mtp-test-fixture@sha256:"
    "763f75a0674e6d20e95e8376b4bd8577857806084205c6f926ceb30be337a25c"
)
MODEL_MANIFEST_SHA256 = (
    "10a88ba89fb90c1cd2d85fa6a8a48623522547f0e0fdd5ba3235f3290bdf1d63"
)
STATE_FINGERPRINTS = (
    "5cce324cf8b59feb02ac4362c53872b62b19d06fc9e0ceab478d2a867ea88a2e",
    "8491ce6cf23f5a337f6ec7a052bdfb8cd036c01ecf2b6e51e773d6798a764cbe",
)
OWNER_SEMANTIC_BY_LIMITS = {
    (4, 4, 12): "0dfd0cb4fa755a3043baa488609cc7df7665c2e7b4bedb9ac70d753c9b6317cf",
}
OWNER_SOURCE_PATHS = (
    "src/mlx2/runtime/flat_segmented_hybrid.py",
    "src/mlx2/runtime/flat_segmented_ast.py",
    "src/mlx2/runtime/ast_execution_capability.py",
    "src/mlx2/runtime/hybrid_verify_rows.py",
    "src/mlx2/runtime/hybrid_speculative.py",
    "src/mlx2/runtime/cache_operations.py",
    "src/mlx2/runtime/models/cache.py",
)
TARGET_SOURCE_PATHS = (
    "src/mlx2/adapters/qwen38_hybrid_native_flat.py",
    "src/mlx2/adapters/qwen38_hybrid_metal_probe.py",
    "src/mlx2/adapters/dense_qwen3_flat.py",
    "src/mlx2/runtime/flat_segmented_ast.py",
    "src/mlx2/runtime/flat_segmented_hybrid.py",
    "src/mlx2/runtime/flat_segmented_kv.py",
    "src/mlx2/runtime/ast_execution_capability.py",
    "src/mlx2/runtime/hybrid_verify_rows.py",
    "src/mlx2/runtime/hybrid_speculative.py",
    "src/mlx2/runtime/cache_operations.py",
    "src/mlx2/runtime/segmented_batch_cache.py",
    "src/mlx2/runtime/segmented_plain_kv.py",
    "src/mlx2/runtime/models/cache.py",
    "src/mlx2/runtime/models/qwen38_27b.py",
    "src/mlx2/runtime/models/qwen3_5.py",
    "src/mlx2/runtime/models/qwen3_next.py",
    "src/mlx2/runtime/models/varlen_dense_mlp.py",
    "src/mlx2/runtime/models/qwen_hybrid_row_exact.py",
    "src/mlx2/runtime/models/row_exact_components.py",
    "src/mlx2/runtime/models/row_exact_qmv.py",
    "src/mlx2/runtime/row_exact/ops.py",
    "src/mlx2/runtime/row_exact/canary.py",
    "src/mlx2/runtime/models/qwen38_fused_gdn.py",
    "src/mlx2/runtime/models/qwen4_fused_gdn.py",
    "src/mlx2/runtime/models/qwen4_fused_gdn_verify.py",
    "src/mlx2/runtime/models/qwen4_fused_gdn_prefill.py",
    "src/mlx2/runtime/models/qwen_gdn_contracts.py",
    "src/mlx2/runtime/models/gdn_kernel_contract.py",
    "src/mlx2/runtime/models/gdn_state.py",
    "src/mlx2/runtime/models/gated_delta.py",
    "src/mlx2/runtime/models/base.py",
    "src/mlx2/runtime/models/precise_ops.py",
    "src/mlx2/runtime/models/rope_utils.py",
    "src/mlx2/runtime/models/activations.py",
    "src/mlx2/runtime/models/invariant_prefill.py",
    "src/mlx2/runtime/models/import_env.py",
    "src/mlx2/runtime/models/moe_nax_gather.py",
    "src/mlx2/runtime/models/qwen4_exp.py",
    "src/mlx2/runtime/models/qwen4_routed_decode.py",
    "src/mlx2/runtime/models/pipeline.py",
    "src/mlx2/runtime/models/served_exp.py",
    "src/mlx2/runtime/paged_kv_cache.py",
    "src/mlx2/runtime/prefill_checkpoint_rows.py",
    "src/mlx2/runtime/ragged_verify_observation.py",
    "src/mlx2/runtime/round_levers.py",
)


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
    text = {
        "model_type": "qwen3_5_text",
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 8,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 128,
        "linear_num_key_heads": 2,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_conv_kernel_dim": 4,
        "full_attention_interval": 4,
        "mtp_num_hidden_layers": 1,
        "partial_rotary_factor": 0.5,
        "rope_parameters": None,
        "max_position_embeddings": 512,
    }
    mx.random.seed(7109)
    model = Model(ModelArgs(model_type="qwen3_5", text_config=text))
    model.eval()
    nn.quantize(model, group_size=32, bits=4)
    model.language_model.mtp.fc = nn.Linear(64, 32, bias=False)
    mx.eval(model.parameters())
    return model


def _pair(lengths=(1, 2, 3, 4), prefixes=None):
    descriptor = FlatSegmentDescriptor.from_query_lengths(lengths)
    model = _tiny_quantized_mtp()
    target = Qwen38HybridNativeFlatTarget(
        model,
        descriptor=descriptor,
        capture_layers=(1, 6),
        identity=_identity(),
        expected_model_manifest_sha256=MODEL_MANIFEST_SHA256,
    )
    if prefixes is None:
        prefixes = tuple((1 + lane, 7, 11) for lane in range(len(lengths)))
    owner = target.make_bound_rows(prefixes)
    return descriptor, owner, target


@contextmanager
def _runtime_scope(model, kind):
    if kind == "invariant_prefill":
        with invariant_prefill.scope():
            yield
        return
    if kind == "ragged_observer":
        with activate(RaggedVerifyObserver()):
            yield
        return
    if kind != "varlen_dense_mlp":
        raise AssertionError(f"unknown runtime scope {kind}")
    if getattr(model, "_varlen_dense_mlp", None) is None:
        varlen_dense_mlp.install(
            model,
            varlen_dense_mlp.VarlenDenseMLPPolicy(
                enabled=True, minimum_padding_rows=1
            ),
        )
    with model.prefill_row_context((1, 2), width=2):
        assert varlen_dense_mlp.active()
        yield


def _expected(owner):
    limits = owner.hard_limits
    key = (
        limits["max_lanes"],
        limits["max_segment_rows"],
        limits["max_total_rows"],
    )
    return {
        "expected_identity": _identity(),
        "expected_owner_semantic_contract_sha256": OWNER_SEMANTIC_BY_LIMITS[key],
        "expected_state_contract_fingerprints": STATE_FINGERPRINTS,
        "expected_owner_source_hashes": _sources(OWNER_SOURCE_PATHS),
        "expected_target_source_hashes": _sources(TARGET_SOURCE_PATHS),
        "expected_backend": "metal",
    }


def _ready(target, monkeypatch):
    identity = _identity()
    monkeypatch.setattr(
        target,
        "ast_native_flat_device_attestation",
        lambda: {
            "schema": DEVICE_ATTESTATION_SCHEMA,
            "backend": "metal",
            "device": identity["device"],
            "build": identity["build"],
            "ready": True,
        },
    )


def _admission(owner, target, descriptor):
    return query_native_flat_segment_admission(
        owner, target, descriptor, **_expected(owner)
    )


def _execute(owner, target, descriptor, tokens):
    admission = _admission(owner, target, descriptor)
    assert admission.available, admission.decline_reasons
    transaction = owner.begin_native_flat(
        descriptor, target_executor=target, admission=admission
    )
    execution = target.execute_native_flat_segments(
        tokens, descriptor, transaction
    )
    validate_native_flat_segment_execution(execution, descriptor)
    return transaction, execution


def _assert_array_exact(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert bool(mx.array_equal(actual, expected, equal_nan=True))


def _assert_rows_exact(actual_rows, expected_rows):
    for actual_row, expected_row in zip(actual_rows, expected_rows, strict=True):
        for actual, expected in zip(actual_row, expected_row, strict=True):
            assert type(actual) is type(expected)
            if type(actual) is KVCache:
                assert actual.offset == expected.offset
                assert (actual.keys is None) == (expected.keys is None)
                if actual.keys is not None:
                    actual_keys, actual_values = actual.keys_and_values()
                    expected_keys, expected_values = expected.keys_and_values()
                    _assert_array_exact(actual_keys, expected_keys)
                    _assert_array_exact(actual_values, expected_values)
            else:
                assert actual.speculating is expected.speculating is False
                assert not actual._rollbacks and not expected._rollbacks
                assert actual.lengths is expected.lengths is None
                assert actual.left_padding is expected.left_padding is None
                for actual_value, expected_value in zip(
                    actual.cache, expected.cache, strict=True
                ):
                    _assert_array_exact(actual_value, expected_value)


def _ordinary_committed_rows(model, base_rows, descriptor, tokens, accepted):
    rows = copy.deepcopy(base_rows)
    for lane, count in enumerate(accepted):
        for token in tokens[descriptor.lane_slice(lane)][:count]:
            mx.eval(model(mx.array([[token]], dtype=mx.int32), cache=rows[lane]))
    return rows


def _group_outputs(execution, descriptor):
    return tuple(
        execution.outputs[descriptor.lane_slice(lane)]
        for lane in range(descriptor.lane_count)
    )


def _assert_output_groups_exact(actual, expected):
    for actual_lane, expected_lane in zip(actual, expected, strict=True):
        for (actual_logits, actual_taps), (expected_logits, expected_taps) in zip(
            actual_lane, expected_lane, strict=True
        ):
            _assert_array_exact(actual_logits, expected_logits)
            _assert_array_exact(actual_taps, expected_taps)


def test_b4_native_flat_exact_oracles_and_zero_padding(monkeypatch):
    descriptor, owner, target = _pair()
    _ready(target, monkeypatch)
    tokens = tuple(range(13, 23))
    # K=0, accepted-draft-0, partial, and full/terminal accepted prefixes.
    accepted = (1, 1, 2, 4)
    base_rows = copy.deepcopy(owner.rows)
    expected_rows = _ordinary_committed_rows(
        target.model, base_rows, descriptor, tokens, accepted
    )
    original_classes = tuple(
        type(module) for _, module in target.model.named_modules()
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("native target reached padded row machinery")

    monkeypatch.setattr(HybridVerifyRows, "views", forbidden)
    monkeypatch.setattr(REV, "valid_rows", forbidden)
    result = run_qwen38_hybrid_native_flat_probe(
        owner,
        target,
        descriptor,
        tokens,
        accepted,
        (41, 42, 43, 44),
        **{key: value for key, value in _expected(owner).items() if key != "expected_backend"},
    )

    receipt = result.receipt
    assert receipt["physical_layout"] == PHYSICAL_LAYOUT
    assert receipt["logical_rows"] == receipt["physical_rows"] == 10
    assert receipt["padding_rows"] == 0
    assert receipt["target_invocations"] == 1
    assert receipt["outer_layer_traversals"] == 8
    assert receipt["accepted_lengths"] == list(accepted)
    assert receipt["ordinary_serial_oracle"] == {
        "full_logits": "bit_exact",
        "capture_taps": "bit_exact",
        "kv_and_gdn_state": "bit_exact_prepublication",
        "continuation_logits_and_taps": "bit_exact_prepublication",
    }
    assert receipt["implemented"] is True
    assert receipt["qualified"] is receipt["selected"] is False
    assert receipt["observed_used_real_artifact"] is False
    assert target.receipt()["padding_rows"] == 0
    assert target.receipt()["target_row_exact_installed"] is False
    assert tuple(type(module) for _, module in target.model.named_modules()) == original_classes
    assert target.model.__dict__.get("_mlx2_qwen_hybrid_row_exact") is None
    _assert_rows_exact(owner.rows, expected_rows)
    assert tuple(boundary.position for boundary in owner.boundaries) == (4, 4, 5, 7)
    assert all(boundary.generation == 1 for boundary in owner.boundaries)


def test_reordered_lanes_are_exactly_permuted(monkeypatch):
    lengths = (1, 2, 3, 4)
    prefixes = ((1, 7, 11), (2, 7, 11), (3, 7, 11), (4, 7, 11))
    segments = ((13,), (14, 15), (16, 17, 18), (19, 20, 21, 22))
    descriptor, owner, target = _pair(lengths, prefixes)
    _ready(target, monkeypatch)
    transaction, execution = _execute(
        owner, target, descriptor, tuple(token for segment in segments for token in segment)
    )
    base_outputs = _group_outputs(execution, descriptor)
    prepared = transaction.prepare_commit(lengths)
    base_rows = transaction._prepared_successor_rows_for_oracle(prepared)
    transaction.abort()

    order = (3, 0, 2, 1)
    reordered, other_owner, other_target = _pair(
        tuple(lengths[index] for index in order),
        tuple(prefixes[index] for index in order),
    )
    _ready(other_target, monkeypatch)
    other_transaction, other_execution = _execute(
        other_owner,
        other_target,
        reordered,
        tuple(token for index in order for token in segments[index]),
    )
    reordered_outputs = _group_outputs(other_execution, reordered)
    _assert_output_groups_exact(
        reordered_outputs, tuple(base_outputs[index] for index in order)
    )
    other_prepared = other_transaction.prepare_commit(reordered.query_lengths)
    other_rows = other_transaction._prepared_successor_rows_for_oracle(other_prepared)
    _assert_rows_exact(
        other_rows,
        tuple(base_rows[index] for index in order),
    )
    other_transaction.abort()


def test_neighbor_lanes_do_not_change_one_lane(monkeypatch):
    descriptor, owner, target = _pair()
    _ready(target, monkeypatch)
    tokens = tuple(range(13, 23))
    transaction, execution = _execute(owner, target, descriptor, tokens)
    cohort_outputs = _group_outputs(execution, descriptor)[2]
    prepared = transaction.prepare_commit(descriptor.query_lengths)
    cohort_row = transaction._prepared_successor_rows_for_oracle(prepared)[2]
    transaction.abort()

    single, single_owner, single_target = _pair((3,), ((3, 7, 11),))
    _ready(single_target, monkeypatch)
    single_transaction, single_execution = _execute(
        single_owner, single_target, single, (16, 17, 18)
    )
    _assert_output_groups_exact(
        _group_outputs(single_execution, single), (cohort_outputs,)
    )
    single_prepared = single_transaction.prepare_commit(single.query_lengths)
    single_row = single_transaction._prepared_successor_rows_for_oracle(
        single_prepared
    )[0]
    _assert_rows_exact((single_row,), (cohort_row,))
    single_transaction.abort()


@pytest.mark.parametrize("drift", ["device", "execution", "source", "artifact"])
def test_execution_time_drift_fails_before_invocation(monkeypatch, drift):
    descriptor, owner, target = _pair()
    _ready(target, monkeypatch)
    admission = _admission(owner, target, descriptor)
    assert admission.available
    transaction = owner.begin_native_flat(
        descriptor, target_executor=target, admission=admission
    )
    before = canonical_bound_hybrid_state(owner.rows, owner.boundaries)
    if drift == "device":
        monkeypatch.setattr(
            target,
            "ast_native_flat_device_attestation",
            lambda: {
                **admission.device_attestation,
                "ready": False,
            },
        )
    elif drift == "execution":
        target._capture_layers = (1, 5)
    elif drift == "source":
        original = target.ast_native_flat_target_contract

        def changed_contract():
            result = original()
            result["source_hashes"] = {
                **result["source_hashes"],
                "src/mlx2/adapters/qwen38_hybrid_native_flat.py": "0" * 64,
            }
            return result

        monkeypatch.setattr(target, "ast_native_flat_target_contract", changed_contract)
    else:
        target.identity["artifact"] = "drifted@sha256:" + "0" * 64
    with pytest.raises(FlatSegmentUnsupported):
        target.execute_native_flat_segments(
            tuple(range(13, 23)), descriptor, transaction
        )
    assert target.invocations == 0
    transaction.abort()
    assert canonical_bound_hybrid_state(owner.rows, owner.boundaries) == before


@pytest.mark.parametrize("surface", ["attention_projection", "lm_head"])
def test_post_admission_tensor_membership_replacement_fails_before_begin(
    monkeypatch, surface
):
    descriptor, owner, target = _pair()
    _ready(target, monkeypatch)
    admission = _admission(owner, target, descriptor)
    assert admission.available
    if surface == "attention_projection":
        projection = target.model.language_model.model.layers[3].self_attn.q_proj
        replacement = copy.deepcopy(projection)
        replacement.weight = mx.zeros_like(replacement.weight)
        target.model.language_model.model.layers[3].self_attn.q_proj = replacement
    else:
        projection = target.model.language_model.lm_head
        replacement = copy.deepcopy(projection)
        replacement.weight = mx.zeros_like(replacement.weight)
        target.model.language_model.lm_head = replacement
    with pytest.raises(FlatSegmentUnsupported):
        owner.begin_native_flat(
            descriptor, target_executor=target, admission=admission
        )
    assert owner._active is None
    assert target.invocations == 0


def _install_unsupported_policy(model, policy):
    if policy == "gdn_tensorfold":
        model.language_model.model.layers[0].linear_attn._prefill_counts = {}
    elif policy == "mlp_tensorfold":
        model.language_model.model.layers[0].mlp._prefill_counts = {}
    else:
        model.language_model.model.layers[0].linear_attn.sharding_group = object()


@pytest.mark.parametrize(
    "policy", ["gdn_tensorfold", "mlp_tensorfold", "gdn_tensor_sharded"]
)
def test_unsupported_policy_declines_at_construction(policy):
    descriptor = FlatSegmentDescriptor.from_query_lengths((1, 2, 3, 4))
    model = _tiny_quantized_mtp()
    _install_unsupported_policy(model, policy)
    with pytest.raises(ValueError, match="model binding drifted"):
        Qwen38HybridNativeFlatTarget(
            model,
            descriptor=descriptor,
            capture_layers=(1, 6),
            identity=_identity(),
            expected_model_manifest_sha256=MODEL_MANIFEST_SHA256,
        )


@pytest.mark.parametrize(
    "policy", ["gdn_tensorfold", "mlp_tensorfold", "gdn_tensor_sharded"]
)
def test_post_admission_policy_drift_fails_before_begin(monkeypatch, policy):
    descriptor, owner, target = _pair()
    _ready(target, monkeypatch)
    admission = _admission(owner, target, descriptor)
    assert admission.available
    _install_unsupported_policy(target.model, policy)
    with pytest.raises(FlatSegmentUnsupported):
        owner.begin_native_flat(
            descriptor, target_executor=target, admission=admission
        )
    assert owner._active is None
    assert target.invocations == 0


def test_post_admission_gdn_decode_policy_drift_fails_before_begin(monkeypatch):
    descriptor, owner, target = _pair()
    _ready(target, monkeypatch)
    admission = _admission(owner, target, descriptor)
    assert admission.available
    gdn = target.model.language_model.model.layers[0].linear_attn
    gdn.set_fused_gdn_enabled(not gdn.fused_gdn_enabled)
    with pytest.raises(FlatSegmentUnsupported):
        owner.begin_native_flat(
            descriptor, target_executor=target, admission=admission
        )
    assert owner._active is None
    assert target.invocations == 0


@pytest.mark.parametrize(
    "surface", ["attention_scale", "final_norm_eps", "gdn_conv_padding"]
)
def test_post_admission_arithmetic_scalar_drift_fails_before_begin(
    monkeypatch, surface
):
    descriptor, owner, target = _pair()
    _ready(target, monkeypatch)
    admission = _admission(owner, target, descriptor)
    assert admission.available
    trunk = target.model.language_model.model
    if surface == "attention_scale":
        trunk.layers[3].self_attn.scale *= 0.5
    elif surface == "final_norm_eps":
        trunk.norm.eps *= 2.0
    else:
        trunk.layers[0].linear_attn.conv1d.padding = 1
    with pytest.raises(FlatSegmentUnsupported):
        owner.begin_native_flat(
            descriptor, target_executor=target, admission=admission
        )
    assert owner._active is None
    assert target.invocations == 0


@pytest.mark.parametrize(
    "scope_kind", ["invariant_prefill", "varlen_dense_mlp", "ragged_observer"]
)
def test_ambient_runtime_scope_declines_at_construction(scope_kind):
    descriptor = FlatSegmentDescriptor.from_query_lengths((1, 2, 3, 4))
    model = _tiny_quantized_mtp()
    with (
        _runtime_scope(model, scope_kind),
        pytest.raises(ValueError, match="model binding drifted"),
    ):
        Qwen38HybridNativeFlatTarget(
            model,
            descriptor=descriptor,
            capture_layers=(1, 6),
            identity=_identity(),
            expected_model_manifest_sha256=MODEL_MANIFEST_SHA256,
        )


@pytest.mark.parametrize(
    "scope_kind", ["invariant_prefill", "varlen_dense_mlp", "ragged_observer"]
)
def test_post_admission_runtime_scope_fails_before_begin(monkeypatch, scope_kind):
    descriptor, owner, target = _pair()
    _ready(target, monkeypatch)
    admission = _admission(owner, target, descriptor)
    assert admission.available
    with _runtime_scope(target.model, scope_kind), pytest.raises(
        FlatSegmentUnsupported
    ):
        owner.begin_native_flat(
            descriptor, target_executor=target, admission=admission
        )
    assert owner._active is None
    assert target.invocations == 0


@pytest.mark.parametrize(
    "scope_kind", ["invariant_prefill", "varlen_dense_mlp", "ragged_observer"]
)
def test_post_begin_runtime_scope_fails_before_execution(monkeypatch, scope_kind):
    descriptor, owner, target = _pair()
    _ready(target, monkeypatch)
    admission = _admission(owner, target, descriptor)
    assert admission.available
    transaction = owner.begin_native_flat(
        descriptor, target_executor=target, admission=admission
    )
    before = canonical_bound_hybrid_state(owner.rows, owner.boundaries)
    with (
        _runtime_scope(target.model, scope_kind),
        pytest.raises(ValueError, match="model binding drifted"),
    ):
        target.execute_native_flat_segments(
            tuple(range(13, 23)), descriptor, transaction
        )
    assert target.invocations == 0
    transaction.abort()
    assert canonical_bound_hybrid_state(owner.rows, owner.boundaries) == before
    assert owner._active is None


def test_fixed_capacity_and_prefix_limits_fail_before_target_work(monkeypatch):
    model = _tiny_quantized_mtp()
    over_limit = FlatSegmentDescriptor.from_query_lengths((4, 4, 4, 1))

    def forbidden_manifest(_model):
        raise AssertionError("over-limit target inspected model tensors")

    monkeypatch.setattr(
        native, "qwen38_hybrid_model_manifest_sha256", forbidden_manifest
    )
    with pytest.raises(ValueError, match="exceeds direct-probe limits"):
        Qwen38HybridNativeFlatTarget(
            model,
            descriptor=over_limit,
            capture_layers=(1, 6),
            identity=_identity(),
            expected_model_manifest_sha256=MODEL_MANIFEST_SHA256,
        )
    monkeypatch.undo()

    _descriptor, owner, target = _pair()
    del owner
    prefixes = ((1,) * 9, (2, 7, 11), (3, 7, 11), (4, 7, 11))
    with pytest.raises(ValueError, match="prefix geometry"):
        target.make_bound_rows(prefixes)
    assert target.invocations == 0


def test_fused_gdn_row_is_already_projected_and_rollback_is_per_row(monkeypatch):
    _descriptor, owner, target = _pair()
    layer = target.model.language_model.model.layers[0]
    assert layer.is_linear
    cache = copy.deepcopy(owner.rows[0][0])
    cache.start_speculation()
    token = mx.array([[19]], dtype=mx.int32)
    hidden = target.model.language_model.model.embed_tokens(token)
    initial_conv, initial_state = cache[0], cache[1]
    projected = mx.full(hidden.shape, 0.125, dtype=hidden.dtype)

    def fused_step(_layer, _qkv, _z, _b, _a, scratch, _dtype):
        scratch[0] = scratch[0] + 1
        scratch[1] = scratch[1] + 2
        scratch.advance(1)
        return "fused_decode", projected

    class ForbiddenProjection:
        def __call__(self, _value):
            raise AssertionError("fused GDN output was projected twice")

    monkeypatch.setattr(native, "ordinary_one_token_step", fused_step)
    monkeypatch.setattr(layer.linear_attn, "out_proj", ForbiddenProjection())
    first = target._advance_gdn_row(layer, hidden, cache)
    expected_hidden = hidden + projected
    expected_first = expected_hidden + layer.mlp(
        layer.post_attention_layernorm(expected_hidden)
    )
    _assert_array_exact(first, expected_first)
    first_conv, first_state = cache[0], cache[1]
    assert len(cache._rollbacks) == 1
    target._advance_gdn_row(layer, hidden, cache)
    assert len(cache._rollbacks) == 2
    assert cache.trim(1) == 1
    _assert_array_exact(cache[0], first_conv)
    _assert_array_exact(cache[1], first_state)
    assert not bool(mx.array_equal(cache[0], initial_conv, equal_nan=True))
    assert not bool(mx.array_equal(cache[1], initial_state, equal_nan=True))
    cache.stop_speculation()


def test_mid_execution_failure_aborts_every_private_plane(monkeypatch):
    descriptor, owner, target = _pair()
    _ready(target, monkeypatch)
    admission = _admission(owner, target, descriptor)
    transaction = owner.begin_native_flat(
        descriptor, target_executor=target, admission=admission
    )
    before = canonical_bound_hybrid_state(owner.rows, owner.boundaries)
    original = target._advance_layer_row
    calls = 0

    def fail_after_state(layer, hidden, cache):
        nonlocal calls
        calls += 1
        result = original(layer, hidden, cache)
        if calls == 7:
            raise RuntimeError("injected native row failure")
        return result

    monkeypatch.setattr(target, "_advance_layer_row", fail_after_state)
    with pytest.raises(RuntimeError, match="injected native row failure"):
        target.execute_native_flat_segments(
            tuple(range(13, 23)), descriptor, transaction
        )
    assert target.invocations == 1
    transaction.abort()
    assert canonical_bound_hybrid_state(owner.rows, owner.boundaries) == before
    assert owner._active is None


def test_continuation_mismatch_never_publishes(monkeypatch):
    descriptor, owner, target = _pair()
    _ready(target, monkeypatch)
    before = canonical_bound_hybrid_state(owner.rows, owner.boundaries)
    calls = 0
    original = native._outputs_exact

    def reject_second(actual, expected):
        nonlocal calls
        calls += 1
        return original(actual, expected) if calls == 1 else False

    monkeypatch.setattr(native, "_outputs_exact", reject_second)
    with pytest.raises(FlatSegmentError, match="continuation oracle"):
        run_qwen38_hybrid_native_flat_probe(
            owner,
            target,
            descriptor,
            tuple(range(13, 23)),
            (1, 1, 2, 4),
            (41, 42, 43, 44),
            **{
                key: value
                for key, value in _expected(owner).items()
                if key != "expected_backend"
            },
        )
    assert canonical_bound_hybrid_state(owner.rows, owner.boundaries) == before
    assert owner._active is None
    assert all(boundary.valid for boundary in owner.boundaries)


def test_observed_use_is_latched_to_successful_execution_identity(monkeypatch):
    descriptor, owner, target = _pair()
    _ready(target, monkeypatch)
    transaction, _execution = _execute(
        owner, target, descriptor, tuple(range(13, 23))
    )
    transaction.abort()
    first = target.receipt()
    assert first["observed_used_cpu_fixture"] is True
    assert first["observed_used_real_artifact"] is False
    assert first["checkpoint"] == ARTIFACT
    assert first["source_revision"] == SOURCE_REVISION
    assert first["build"] == _identity()["build"]
    assert first["device"] == _identity()["device"]
    assert first["observed_execution_identity"] == _identity()

    monkeypatch.setattr(native.mx, "default_device", lambda: mx.gpu)
    monkeypatch.setattr(
        target,
        "ast_native_flat_device_attestation",
        lambda: {
            "schema": DEVICE_ATTESTATION_SCHEMA,
            "backend": "metal",
            "device": "mlx-metal:later-device",
            "build": "mlx-later-build",
            "ready": True,
        },
    )
    later = target.receipt()
    assert later["observed_used_cpu_fixture"] is True
    assert later["observed_used_real_artifact"] is False
    assert later["observed_execution_identity"] == _identity()


def test_source_authority_and_distinct_fallback_contract(monkeypatch):
    descriptor, owner, target = _pair()
    _ready(target, monkeypatch)
    assert tuple(TARGET_SOURCE_PATHS) == native.SOURCE_PATHS
    expected = _expected(owner)
    omitted = dict(expected["expected_target_source_hashes"])
    omitted.pop("src/mlx2/adapters/qwen38_hybrid_native_flat.py")
    expected["expected_target_source_hashes"] = omitted
    admission = query_native_flat_segment_admission(
        owner, target, descriptor, **expected
    )
    assert not admission.available
    assert "target_source_hashes_mismatch" in admission.decline_reasons
    assert not hasattr(target, "ast_flat_segment_target_contract")
    assert not hasattr(target, "execute_flat_segments")
    assert hasattr(Qwen38HybridMetalFlatTarget, "ast_flat_segment_target_contract")
    assert not hasattr(Qwen38HybridMetalFlatTarget, "ast_native_flat_target_contract")


def test_rejected_target_never_installs_or_mutates_padded_route():
    descriptor = FlatSegmentDescriptor.from_query_lengths((1, 2, 3, 4))
    model = _tiny_quantized_mtp()
    classes = tuple(type(module) for _, module in model.named_modules())
    with pytest.raises(ValueError, match="model binding drifted"):
        Qwen38HybridNativeFlatTarget(
            model,
            descriptor=descriptor,
            capture_layers=(1, 6),
            identity=_identity(),
            expected_model_manifest_sha256="0" * 64,
        )
    assert tuple(type(module) for _, module in model.named_modules()) == classes
    assert model.__dict__.get("_mlx2_qwen_hybrid_row_exact") is None


def test_default_receipt_is_truthful_before_any_use():
    _descriptor, owner, target = _pair()
    del owner
    receipt = target.receipt()
    assert receipt["implemented"] is True
    assert receipt["qualified"] is False
    assert receipt["selected"] is receipt["default"] is False
    assert receipt["observed_used_cpu_fixture"] is False
    assert receipt["observed_used_real_artifact"] is False
    assert receipt["physical_rows"] == receipt["padding_rows"] == 0
    assert receipt["serving_wired"] is receipt["performance_claim"] is False
