"""Default-off native logical-row Qwen3.8 hybrid AST correctness target.

The adapter accepts one lane-major flat token vector and visits the language
trunk once.  Direct probes retain the literal B1/S1 logical-row executor.
Serving uses the proven segmented row-exact executor so row-independent math
is compacted over valid rows while attention/GDN transitions remain lane-local
and position ordered.  Neither path uses the historical shrinking-depth loop.

The target is an eventual direct Metal correctness probe.  CPU tests exercise
the exact lifecycle and arithmetic under a test-only ready attestation; they
do not qualify either CPU or Metal execution.
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar

import mlx.core as mx
from mlx.utils import tree_flatten

from ..runtime.ast_live_state import (
    ASTLiveLaneStateRegistry,
    ASTLiveStateAuthority,
    ASTPhysicalCommitCoordinator,
    ASTPhysicalResult,
    ASTPhysicalResultPlan,
)
from ..runtime.ast_physical_target import (
    PROVIDER_PROTOCOL,
    ASTPhysicalTargetCandidate,
    ast_serving_admission_receipt,
    validate_ast_serving_admission,
)
from ..runtime.cache_operations import cache_operations
from ..runtime.flat_segmented_ast import (
    ABORT_LAW,
    COMMIT_LAW,
    DESCRIPTOR_SCHEMA,
    EXECUTION_LAW,
    FIDELITY_EXACT,
    LAYOUT_CLASS,
    NATIVE_EXECUTION_SCOPE,
    NATIVE_OUTER_INVOCATION_LAW,
    NATIVE_PHYSICAL_LAYOUT,
    NATIVE_PHYSICAL_ROWS_LAW,
    NATIVE_ROW_ARITHMETIC_LAW,
    NATIVE_STATEFUL_CORE_LAW,
    NATIVE_TARGET_CAPABILITY_SCHEMA,
    NATIVE_TARGET_PROTOCOL,
    STATE_WRITE_LAW,
    TERMINAL_LAW,
    FlatSegmentDescriptor,
    FlatSegmentError,
    FlatSegmentUnsupported,
    NativeFlatSegmentExecution,
    query_native_flat_segment_admission,
    validate_native_flat_segment_admission,
    validate_native_flat_segment_execution,
)
from ..runtime.flat_segmented_hybrid import (
    HybridLiveStateBoundary,
    NativeFlatSegmentedHybridRows,
    NativeFlatSegmentedHybridTransaction,
    successor_bound_hybrid_state,
)
from ..runtime.flat_segmented_kv import METAL_PROBE_HARD_LIMITS
from ..runtime.hybrid_verify_rows import HybridVerifyRows, HybridVerifyTransaction
from ..runtime.models import invariant_prefill as _invariant_prefill
from ..runtime.models import qwen38_27b as qwen38
from ..runtime.models import varlen_dense_mlp as _varlen_dense_mlp
from ..runtime.models.base import create_attention_mask
from ..runtime.models.cache import ArraysCache, KVCache
from ..runtime.models.qwen_hybrid_row_exact import (
    _scratch,
    ordinary_one_token_step,
)
from ..runtime.ragged_verify_observation import current_observer
from .dense_qwen3_flat import dense_qwen3_metal_device_attestation
from .qwen38_hybrid_metal_probe import (
    MAX_BOUND_PREFIX_ROWS,
    QWEN38_27B_OQ4E_MTP_CHECKPOINT,
    QWEN38_27B_OQ4E_MTP_REVIEWED_MANIFEST,
    QWEN38_27B_OQ4E_NO_MTP_MODEL_MANIFEST_SHA256,
    _identity,
    _ordinary_forward_with_taps,
    _outputs_exact,
    _quantized_manifest,
    _sha256,
    qwen38_hybrid_model_manifest_sha256,
)

PROBE_SCHEMA = "mlx2.ast-qwen38-hybrid-native-flat-direct-probe.v1"
PHYSICAL_LAYOUT = NATIVE_PHYSICAL_LAYOUT
_PACKAGE = Path(__file__).resolve().parents[1]
_PACKAGE_PREFIX = "src/mlx2/"

# Exact code surface controlling the native owner, Qwen tensor arithmetic,
# model binding, and ordinary one-token GDN transition.  Tests enumerate this
# independently so dropping an implementation file fails admission.
SOURCE_PATHS = (
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
SERVING_SOURCE_PATHS = SOURCE_PATHS + (
    "src/mlx2/adapters/external_draft_policy.py",
    "src/mlx2/adapters/qwen38_27b.py",
    "src/mlx2/runtime/ast_live_state.py",
    "src/mlx2/runtime/ast_physical_target.py",
    "src/mlx2/runtime/external_speculative.py",
    "src/mlx2/runtime/progressive_external_verify.py",
    "src/mlx2/runtime/row_exact_verify.py",
)
SERVING_TARGET_CAPABILITY_SCHEMA = "mlx2.ast-qwen38-live-target.v2"
SERVING_TARGET_PROTOCOL = "qwen38-segmented-zero-pad-or-native-flat-v1"


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _callable_identity(value):
    if not callable(value):
        return None
    return (
        id(getattr(value, "__self__", None)),
        id(getattr(value, "__func__", value)),
    )


def _source_hashes() -> dict[str, str]:
    return {
        name: hashlib.sha256(
            (_PACKAGE / name[len(_PACKAGE_PREFIX) :]).read_bytes()
        ).hexdigest()
        for name in SOURCE_PATHS
    }


def _serving_source_hashes() -> dict[str, str]:
    return {
        name: hashlib.sha256(
            (_PACKAGE / name[len(_PACKAGE_PREFIX) :]).read_bytes()
        ).hexdigest()
        for name in SERVING_SOURCE_PATHS
    }


def _serving_source_generation():
    """Cheap immutable-byte generation proof for the admitted source set."""

    generation = []
    for name in SERVING_SOURCE_PATHS:
        stat = (_PACKAGE / name[len(_PACKAGE_PREFIX) :]).stat()
        generation.append(
            (
                name,
                stat.st_dev,
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
            )
        )
    return tuple(generation)


def _git_source_revision() -> str:
    revision = subprocess.check_output(
        ["git", "-C", str(_PACKAGE.parents[1]), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise RuntimeError("Qwen3.8 AST source revision is unavailable")
    return revision


def _require_clean_source_revision(revision: str) -> None:
    if _git_source_revision() != revision:
        raise RuntimeError("Qwen3.8 AST source revision changed")
    clean = subprocess.run(
        [
            "git",
            "-C",
            str(_PACKAGE.parents[1]),
            "diff",
            "--quiet",
            revision,
            "--",
            *SERVING_SOURCE_PATHS,
        ],
        check=False,
    )
    if clean.returncode != 0:
        raise RuntimeError("Qwen3.8 AST serving sources differ from revision")


def _target_execution_membership(model) -> tuple[int, ...]:
    """Ordered object identity for every executable target module and tensor."""

    modules = []
    seen = set()
    module_schema = []
    for name, module in model.named_modules():
        if "mtp" in name.split(".") or id(module) in seen:
            continue
        seen.add(id(module))
        modules.append(id(module))
        module_schema.append((name, type(module).__module__, type(module).__qualname__))
    tensors = []
    tensor_schema = []
    for name, value in tree_flatten(model.parameters()):
        if "mtp" in name.split("."):
            continue
        tensors.append(id(value))
        tensor_schema.append((name, tuple(value.shape), str(value.dtype)))
    schema_word = int(
        hashlib.sha256(_canonical((module_schema, tensor_schema))).hexdigest(), 16
    )
    return (len(modules), *modules, len(tensors), *tensors, schema_word)


def _literal_semantic(value):
    if isinstance(value, (bool, int, float, str, type(None))):
        return value
    if isinstance(value, (tuple, list)):
        return tuple(_literal_semantic(item) for item in value)
    if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
        return tuple((key, _literal_semantic(value[key])) for key in sorted(value))
    raise TypeError("value is not a token-free arithmetic literal")


def _target_execution_policy(model) -> tuple[object, ...]:
    """Live arithmetic semantics not represented by module/tensor identity."""

    trunk = model.language_model.model
    training = tuple(
        (name, bool(module.training))
        for name, module in model.named_modules()
        if "mtp" not in name.split(".")
    )
    layers = []
    for index, layer in enumerate(trunk.layers):
        mlp = layer.mlp
        mlp_topology = tuple(
            (
                name,
                type(projection).__module__,
                type(projection).__qualname__,
                tuple(projection.weight.shape),
                tuple(projection.bias.shape)
                if getattr(projection, "bias", None) is not None
                else None,
            )
            for name, projection in (
                ("gate_proj", mlp.gate_proj),
                ("up_proj", mlp.up_proj),
                ("down_proj", mlp.down_proj),
            )
        )
        common = (
            index,
            bool(layer.is_linear),
            float(layer.input_layernorm.eps),
            float(layer.post_attention_layernorm.eps),
            mlp_topology,
        )
        if layer.is_linear:
            recurrent = layer.linear_attn
            contract = recurrent.fused_gdn_contract
            layers.append(
                (
                    *common,
                    "gdn",
                    int(recurrent.hidden_size),
                    int(recurrent.num_k_heads),
                    int(recurrent.num_v_heads),
                    int(recurrent.head_k_dim),
                    int(recurrent.head_v_dim),
                    int(recurrent.key_dim),
                    int(recurrent.value_dim),
                    int(recurrent.conv_kernel_size),
                    int(recurrent.conv_dim),
                    (
                        type(recurrent.conv1d).__module__,
                        type(recurrent.conv1d).__qualname__,
                        _literal_semantic(recurrent.conv1d.padding),
                        _literal_semantic(recurrent.conv1d.dilation),
                        _literal_semantic(recurrent.conv1d.stride),
                        int(recurrent.conv1d.groups),
                    ),
                    float(recurrent.layer_norm_epsilon),
                    float(recurrent.norm.eps),
                    tuple(recurrent.row_exact_decode_route()),
                    (
                        contract.identity,
                        (
                            int(contract.geometry.num_key_heads),
                            int(contract.geometry.num_value_heads),
                            int(contract.geometry.key_head_dim),
                            int(contract.geometry.value_head_dim),
                            int(contract.geometry.conv_kernel),
                        ),
                        contract.output_gate,
                        bool(contract.decay_input_float32),
                        bool(contract.beta_input_float32),
                        contract.qk_normalization.value,
                        contract.state_layout,
                        tuple(contract.recurrent_storage_classes),
                    ),
                    bool(recurrent.training),
                    str(getattr(recurrent, "_gdn_state_dtype", None)),
                    recurrent.sharding_group is not None,
                    bool(recurrent.fused_gdn_prefill_enabled),
                    bool(recurrent.fused_gdn_compact_replay_enabled),
                )
            )
            continue
        attention = layer.self_attn
        rope = attention.rope
        rope_semantics = []
        for name, value in vars(rope).items():
            if name.startswith("_"):
                continue
            try:
                literal = _literal_semantic(value)
            except TypeError:
                continue
            rope_semantics.append((name, literal))
        layers.append(
            (
                *common,
                "attention",
                int(attention.num_attention_heads),
                int(attention.num_key_value_heads),
                int(attention.head_dim),
                float(attention.scale),
                float(attention.q_norm.eps),
                float(attention.k_norm.eps),
                type(rope).__module__,
                type(rope).__qualname__,
                tuple(sorted(rope_semantics)),
            )
        )
    return (
        training,
        (
            int(trunk.pipeline_rank),
            int(trunk.pipeline_size),
            int(trunk.start_idx),
            None if trunk.end_idx is None else int(trunk.end_idx),
            int(trunk.ssm_idx),
            int(trunk.fa_idx),
            float(trunk.norm.eps),
        ),
        tuple(layers),
    )


def _runtime_context_is_ordinary() -> bool:
    """Reject ambient scopes that redirect or instrument ordinary row math."""

    return (
        not _invariant_prefill.active()
        and not _varlen_dense_mlp.active()
        and current_observer() is None
    )


class Qwen38HybridNativeFlatTarget:
    """One adapter invocation over exact lane-major logical rows."""

    execution_scope = NATIVE_EXECUTION_SCOPE
    target_protocol = NATIVE_TARGET_PROTOCOL

    def __init__(
        self,
        model,
        *,
        descriptor: FlatSegmentDescriptor,
        capture_layers: Sequence[int],
        identity: Mapping[str, str],
        expected_model_manifest_sha256: str,
        construction_role: str = "direct_probe",
        source_hashes: Mapping[str, str] | None = None,
    ):
        if type(model) is not qwen38.Model:
            raise TypeError("native Qwen3.8 target requires exact wrapper")
        if construction_role not in {"direct_probe", "serving"}:
            raise ValueError("native Qwen3.8 construction role differs")
        if not isinstance(descriptor, FlatSegmentDescriptor):
            raise TypeError("native Qwen3.8 target requires validated geometry")
        actuals = {
            "max_lanes": descriptor.lane_count,
            "max_segment_rows": descriptor.max_segment_rows,
            "max_total_rows": descriptor.logical_rows,
        }
        if any(
            actuals[name] > METAL_PROBE_HARD_LIMITS[name]
            for name in METAL_PROBE_HARD_LIMITS
        ):
            raise ValueError("native Qwen3.8 descriptor exceeds direct-probe limits")
        self._model = model
        self._descriptor = descriptor
        self._capture_layers = tuple(capture_layers)
        self.identity = _identity(identity)
        self._bound_identity = dict(self.identity)
        self.hard_limits = dict(METAL_PROBE_HARD_LIMITS)
        self._expected_model_manifest_sha256 = _sha256(
            expected_model_manifest_sha256,
            "expected native Qwen3.8 model manifest",
        )
        self._model_manifest_sha256 = qwen38_hybrid_model_manifest_sha256(model)
        self._quantized_manifest = _quantized_manifest(model)
        self._quantized_manifest_sha256 = hashlib.sha256(
            _canonical(self._quantized_manifest)
        ).hexdigest()
        self._mtp_present = model.mtp is not None
        self._mtp_layers = 0 if model.mtp is None else len(model.mtp.layers)
        self._execution_membership = _target_execution_membership(model)
        self._execution_policy = _target_execution_policy(model)
        self._construction_role = construction_role
        if source_hashes is not None and (
            set(source_hashes) != set(SOURCE_PATHS)
            or any(
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                for digest in source_hashes.values()
            )
        ):
            raise ValueError("native Qwen3.8 source-hash binding differs")
        self._source_hash_binding = (
            None if source_hashes is None else dict(source_hashes)
        )
        self._installed_row_exact_route = model.__dict__.get(
            "_mlx2_qwen_hybrid_row_exact"
        )
        self._installed_row_exact_executor = _callable_identity(
            getattr(model, "progressive_external_target_rows", None)
        )
        self._installed_compact_exact_rows = getattr(
            model, "progressive_external_compact_exact_rows", None
        )
        self._installed_progressive_protocol = getattr(
            model, "progressive_external_verify_protocol", None
        )
        if (construction_role == "serving") != (
            self._installed_row_exact_route is not None
        ):
            raise ValueError("native Qwen3.8 row-exact route binding differs")
        self.invocations = 0
        self.successful_invocations = 0
        self.executed_rows = 0
        self.layer_traversals = 0
        self._observed_cpu_fixture = False
        self._observed_real_artifact = False
        self._observed_identity = None
        self._last_serving_execution_mode = None
        # A loaded serving model is immutable for the lifetime of its adapter
        # session.  The complete binding audit below walks every executable
        # module, tensor, and arithmetic policy; doing that again at each
        # discovery/admission/publication seam turns a cheap authority check
        # into a model-sized hot-path tax.  Keep one last-session token so a
        # new session still receives the complete audit, while every round
        # retains the cheap mutable-hook/device/source guards.
        self._validated_session_scope_sha256 = None
        self._validate_model()

    @property
    def model(self):
        return self._model

    @property
    def descriptor(self):
        return self._descriptor

    @property
    def capture_layers(self):
        return self._capture_layers

    def _validate_model(self):
        model = self.model
        trunk = model.language_model.model
        if (
            type(model) is not qwen38.Model
            or self._model_manifest_sha256 != self._expected_model_manifest_sha256
            or qwen38_hybrid_model_manifest_sha256(model)
            != self._expected_model_manifest_sha256
            or _quantized_manifest(model) != self._quantized_manifest
            or trunk.pipeline_size != 1
            or not trunk.layers
            or not any(layer.is_linear for layer in trunk.layers)
            or not any(not layer.is_linear for layer in trunk.layers)
            or not self.capture_layers
            or self.capture_layers != tuple(sorted(set(self.capture_layers)))
            or self.capture_layers[0] < 0
            or self.capture_layers[-1] >= len(trunk.layers)
            or (model.mtp is not None) != self._mtp_present
            or (0 if model.mtp is None else len(model.mtp.layers)) != self._mtp_layers
            or _target_execution_membership(model) != self._execution_membership
            or _target_execution_policy(model) != self._execution_policy
            or not _runtime_context_is_ordinary()
            or model.__dict__.get("_mlx2_qwen_hybrid_row_exact")
            is not self._installed_row_exact_route
            or _callable_identity(
                getattr(model, "progressive_external_target_rows", None)
            )
            != self._installed_row_exact_executor
            or getattr(model, "progressive_external_compact_exact_rows", None)
            != self._installed_compact_exact_rows
            or getattr(model, "progressive_external_verify_protocol", None)
            != self._installed_progressive_protocol
            or (
                self._construction_role == "serving"
                and (
                    self._installed_row_exact_executor is None
                    or self._installed_compact_exact_rows is not True
                    or self._installed_progressive_protocol
                    != "row-exact-b1s1-cache-state-v1"
                )
            )
            or any(
                hasattr(layer.mlp, "_prefill_counts")
                or (
                    layer.is_linear
                    and (
                        hasattr(layer.linear_attn, "_prefill_counts")
                        or layer.linear_attn.sharding_group is not None
                    )
                )
                for layer in trunk.layers
            )
        ):
            raise ValueError("native Qwen3.8 model binding drifted")

    def _validate_hot_model_binding(self):
        """Validate the mutable serving surface without rescanning tensors.

        Tensor/module membership and numerical policy are fully audited when
        a target is constructed and once when a new serving-session scope is
        admitted.  mlx2 does not support mutating a loaded model in place;
        replacing a model creates a new provider/session.  Hooks and runtime
        switches *are* mutable, so keep checking those at every execution and
        publication boundary.
        """

        model = self.model
        trunk = model.language_model.model
        if (
            type(model) is not qwen38.Model
            or trunk.pipeline_size != 1
            or not trunk.layers
            or (model.mtp is not None) != self._mtp_present
            or (0 if model.mtp is None else len(model.mtp.layers)) != self._mtp_layers
            or not _runtime_context_is_ordinary()
            or model.__dict__.get("_mlx2_qwen_hybrid_row_exact")
            is not self._installed_row_exact_route
            or _callable_identity(
                getattr(model, "progressive_external_target_rows", None)
            )
            != self._installed_row_exact_executor
            or getattr(model, "progressive_external_compact_exact_rows", None)
            != self._installed_compact_exact_rows
            or getattr(model, "progressive_external_verify_protocol", None)
            != self._installed_progressive_protocol
            or (
                self._construction_role == "serving"
                and (
                    self._installed_row_exact_executor is None
                    or self._installed_compact_exact_rows is not True
                    or self._installed_progressive_protocol
                    != "row-exact-b1s1-cache-state-v1"
                )
            )
            or any(
                hasattr(layer.mlp, "_prefill_counts")
                or (
                    layer.is_linear
                    and (
                        hasattr(layer.linear_attn, "_prefill_counts")
                        or layer.linear_attn.sharding_group is not None
                    )
                )
                for layer in trunk.layers
            )
        ):
            raise ValueError("native Qwen3.8 model binding drifted")

    def validate_session_model_binding(self, session_scope_sha256):
        """Run one complete model audit per exact serving-session scope."""

        if (
            not isinstance(session_scope_sha256, str)
            or len(session_scope_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in session_scope_sha256
            )
        ):
            raise ValueError("native Qwen3.8 session scope differs")
        if self._validated_session_scope_sha256 != session_scope_sha256:
            self._validate_model()
            self._validated_session_scope_sha256 = session_scope_sha256
        else:
            self._validate_hot_model_binding()

    def ast_native_flat_device_attestation(self):
        return dense_qwen3_metal_device_attestation()

    def ast_native_flat_execution_identity(self):
        self._validate_model()
        trunk = self.model.language_model.model
        geometry_word = int(self.descriptor.geometry_sha256, 16)
        manifest_word = int(self._expected_model_manifest_sha256, 16)
        policy_word = int(
            hashlib.sha256(_canonical(self._execution_policy)).hexdigest(), 16
        )
        return (
            id(self.model),
            self.capture_layers
            + (
                id(trunk),
                id(trunk.embed_tokens),
                *(id(layer) for layer in trunk.layers),
                id(trunk.norm),
                *self._execution_membership,
                manifest_word,
                geometry_word,
                policy_word,
                int(self._mtp_present),
                self._mtp_layers,
            ),
        )

    @property
    def source_hashes(self):
        return (
            _source_hashes()
            if self._source_hash_binding is None
            else dict(self._source_hash_binding)
        )

    def ast_native_flat_target_contract(self):
        # The direct-probe contract is also its admission-time drift gate.
        # Live serving has a separate session audit plus per-boundary hot
        # guard, so it must not repeat the model-sized scan while merely
        # materializing the immutable contract.
        if self._construction_role != "serving":
            self._validate_model()
        return {
            "schema": NATIVE_TARGET_CAPABILITY_SCHEMA,
            "protocol": self.target_protocol,
            "execution_scope": self.execution_scope,
            "fidelity": FIDELITY_EXACT,
            "execution_law": EXECUTION_LAW,
            "descriptor_schema": DESCRIPTOR_SCHEMA,
            "layout_classes": [LAYOUT_CLASS],
            "input_storage": "flat_logical_rows",
            "output_storage": "flat_logical_rows_in_segment_order",
            "position_law": "per_segment_base_plus_local_offset",
            "causal_isolation": "strict_no_cross_segment_reads",
            "terminal_law": TERMINAL_LAW,
            "state_write_law": STATE_WRITE_LAW,
            "commit_law": COMMIT_LAW,
            "abort_law": ABORT_LAW,
            "hard_limits": dict(self.hard_limits),
            "source_hashes": self.source_hashes,
            "identity": dict(self.identity),
            "physical_layout": NATIVE_PHYSICAL_LAYOUT,
            "physical_rows_law": NATIVE_PHYSICAL_ROWS_LAW,
            "outer_invocation_law": NATIVE_OUTER_INVOCATION_LAW,
            "row_arithmetic_law": NATIVE_ROW_ARITHMETIC_LAW,
            "stateful_core_law": NATIVE_STATEFUL_CORE_LAW,
        }

    def ast_live_serving_target_contract(self):
        """Source-bound serving capability with truthful geometry dispatch."""

        native_contract = self.ast_native_flat_target_contract()
        return {
            "schema": SERVING_TARGET_CAPABILITY_SCHEMA,
            "protocol": SERVING_TARGET_PROTOCOL,
            "fidelity": FIDELITY_EXACT,
            "descriptor_schema": DESCRIPTOR_SCHEMA,
            "input_storage": "flat_logical_rows",
            "output_storage": "flat_logical_rows_in_segment_order",
            "physical_rows_law": "logical_rows_only_no_padding_rows",
            "dispatch_law": (
                "equal_length_lanes_use_segmented_zero_padding_executor_"
                "else_literal_native_flat_b1s1_rows"
            ),
            "segmented_zero_padding_law": (
                "one_bxs_row_exact_traversal_with_segmented_kv_gdn_state"
            ),
            "ragged_law": "ordinary_b1s1_exact_per_logical_row",
            "state_write_law": STATE_WRITE_LAW,
            "commit_law": COMMIT_LAW,
            "abort_law": ABORT_LAW,
            "direct_native_contract_manifest_sha256": _manifest(native_contract),
            "hard_limits": dict(self.hard_limits),
            "source_hashes": self.source_hashes,
            "identity": dict(self.identity),
            "physical_layout": NATIVE_PHYSICAL_LAYOUT,
        }

    def validate_live_execution_attestation(self):
        if self._construction_role != "serving":
            raise ValueError("native Qwen3.8 target is not serving-bound")
        attestation = self.ast_native_flat_device_attestation()
        if (
            attestation.get("backend") != "metal"
            or attestation.get("build") != self._bound_identity["build"]
            or attestation.get("device") != self._bound_identity["device"]
        ):
            raise ValueError("native Qwen3.8 serving device attestation drifted")
        if (
            self._bound_identity["artifact"] == QWEN38_27B_OQ4E_MTP_CHECKPOINT
            and attestation.get("ready") is not True
        ):
            raise ValueError("native Qwen3.8 real artifact requires ready Metal")
        return attestation

    def make_bound_rows(self, prefixes: Sequence[Sequence[int]]):
        """Issue exact artifact/lane boundaries from ordinary prefix state."""

        self._validate_model()
        prefixes = tuple(tuple(prefix) for prefix in prefixes)
        if len(prefixes) != self.descriptor.lane_count or any(
            not prefix
            or len(prefix) > MAX_BOUND_PREFIX_ROWS
            or any(type(token) is not int or token < 0 for token in prefix)
            or len(prefix) + length
            > self.model.language_model.args.max_position_embeddings
            for prefix, length in zip(
                prefixes, self.descriptor.query_lengths, strict=True
            )
        ):
            raise ValueError("native Qwen3.8 prefix geometry differs")
        rows = []
        boundaries = []
        for lane, prefix in enumerate(prefixes):
            caches = self.model.make_cache()
            mx.eval(self.model(mx.array([prefix], dtype=mx.int32), cache=caches))
            rows.append(caches)
            boundaries.append(
                HybridLiveStateBoundary._mint(
                    revision=self.identity["artifact"],
                    lane_id=lane,
                    position=len(prefix),
                    generation=0,
                    row=caches,
                    issuer_id=id(self),
                )
            )
        return NativeFlatSegmentedHybridRows(
            rows,
            identity=self.identity,
            boundary_checkpoints=boundaries,
            target_executor=self,
            hard_limits=self.hard_limits,
        )

    def validate_native_flat_owner(self, owner, descriptor):
        self._validate_model()
        if type(owner) is not NativeFlatSegmentedHybridRows:
            raise TypeError("native Qwen3.8 target requires its exact owner")
        if descriptor != self.descriptor:
            raise ValueError("native Qwen3.8 descriptor binding differs")
        owner._validate_rows(
            owner.rows,
            owner.boundaries,
            artifact=self.identity["artifact"],
            issuer_id=id(self),
        )
        if (
            owner._target_executor_id != id(self)
            or owner.identity != self.identity
            or owner.hard_limits != self.hard_limits
        ):
            raise ValueError("native Qwen3.8 owner binding differs")
        layers = self.model.language_model.model.layers
        if descriptor.lane_count != len(owner.rows) or any(
            len(row) != len(layers) for row in owner.rows
        ):
            raise ValueError("native Qwen3.8 layer geometry differs")
        max_position = self.model.language_model.args.max_position_embeddings
        for lane, row in enumerate(owner.rows):
            kv_offsets = []
            for layer, cache in zip(layers, row, strict=True):
                if layer.is_linear:
                    if type(cache) is not ArraysCache:
                        raise TypeError("native Qwen3.8 GDN cache differs")
                    gdn = layer.linear_attn
                    conv, state = cache.cache
                    if conv.shape != (
                        1,
                        gdn.conv_kernel_size - 1,
                        gdn.conv_dim,
                    ) or state.shape != (
                        1,
                        gdn.num_v_heads,
                        gdn.head_v_dim,
                        gdn.head_k_dim,
                    ):
                        raise ValueError("native Qwen3.8 GDN geometry differs")
                else:
                    if type(cache) is not KVCache or cache.keys is None:
                        raise TypeError("native Qwen3.8 KV cache differs")
                    attention = layer.self_attn
                    if (
                        cache.values is None
                        or cache.keys.shape != cache.values.shape
                        or cache.keys.shape[0] != 1
                        or cache.keys.shape[1] != attention.num_key_value_heads
                        or cache.keys.shape[-1] != attention.head_dim
                    ):
                        raise ValueError("native Qwen3.8 KV geometry differs")
                    kv_offsets.append(cache.offset)
            if not kv_offsets or len(set(kv_offsets)) != 1:
                raise ValueError("native Qwen3.8 lane offsets differ")
            if kv_offsets[0] + descriptor.query_lengths[lane] > max_position:
                raise ValueError("native Qwen3.8 segment exceeds context")

    @staticmethod
    def _advance_gdn_row(layer, hidden, cache):
        """Run exactly the layer's ordinary one-token GDN state transition."""

        gdn = layer.linear_attn
        normalized = layer.input_layernorm(hidden)
        qkv, z, b, a = gdn._input_projections(normalized)
        snapshot = [cache[0], cache[1]]
        scratch = _scratch(snapshot[0], snapshot[1])
        route, value = ordinary_one_token_step(
            gdn, qkv, z, b, a, scratch, normalized.dtype
        )
        if route == "stock_decode":
            value = gdn.out_proj(value)
        final = [scratch[0], scratch[1]]

        def replay(count, state=final):
            if int(count) != 1:
                raise ValueError("native Qwen3.8 GDN replay depth differs")
            return list(state)

        cache.record_rollback(1, replay, snapshot)
        cache[0], cache[1] = final
        cache.advance(1)
        h = hidden + value
        return h + layer.mlp(layer.post_attention_layernorm(h))

    @classmethod
    def _advance_layer_row(cls, layer, hidden, cache):
        if layer.is_linear:
            return cls._advance_gdn_row(layer, hidden, cache)
        mask = create_attention_mask(hidden, cache)
        return layer(hidden, mask=mask, cache=cache)

    def _execute_rows(
        self, flat_inputs, descriptor, lane_caches, *, execution_attestation=None
    ):
        """Execute one lane-major logical slab on already-private live rows."""

        tokens = tuple(flat_inputs)
        if len(tokens) != descriptor.logical_rows or any(
            type(token) is not int or token < 0 for token in tokens
        ):
            raise ValueError("native Qwen3.8 inputs must cover logical rows")
        self.invocations += 1

        trunk = self.model.language_model.model
        flat_token_array = mx.array([tokens], dtype=mx.int32)
        embedded = trunk.embed_tokens(flat_token_array)
        hidden_rows = [
            embedded[:, index : index + 1] for index in range(descriptor.logical_rows)
        ]
        tap_rows = [[] for _ in hidden_rows]
        traversed = 0
        for layer_index, layer in enumerate(trunk.layers):
            for lane in range(descriptor.lane_count):
                cache = lane_caches[lane][layer_index]
                for row_index in range(
                    descriptor.cu_query_lengths[lane],
                    descriptor.cu_query_lengths[lane + 1],
                ):
                    hidden_rows[row_index] = self._advance_layer_row(
                        layer, hidden_rows[row_index], cache
                    )
            traversed += 1
            if layer_index in self.capture_layers:
                for row_index, hidden in enumerate(hidden_rows):
                    tap_rows[row_index].append(hidden)
        if traversed != len(trunk.layers):
            raise RuntimeError("native Qwen3.8 layer traversal differs")

        outputs = []
        evaluation = []
        for hidden, taps in zip(hidden_rows, tap_rows, strict=True):
            if len(taps) != len(self.capture_layers):
                raise RuntimeError("native Qwen3.8 tap count differs")
            normalized = trunk.norm(hidden)
            logits = self.model.logits(normalized)
            features = mx.concatenate(taps, axis=-1)
            evaluation.extend((logits, features))
            outputs.append((logits[0, 0], features[0, 0]))
        mx.eval(*evaluation)
        execution_attestation = execution_attestation or {}
        if mx.default_device() == mx.cpu:
            self._observed_cpu_fixture = True
        elif (
            mx.default_device() == mx.gpu
            and self._bound_identity["artifact"] == QWEN38_27B_OQ4E_MTP_CHECKPOINT
            and execution_attestation.get("backend") == "metal"
            and execution_attestation.get("ready") is True
            and execution_attestation.get("device") == self._bound_identity["device"]
            and execution_attestation.get("build") == self._bound_identity["build"]
        ):
            self._observed_real_artifact = True
        if self._observed_identity is None:
            self._observed_identity = dict(self._bound_identity)
        self.successful_invocations += 1
        self.executed_rows += descriptor.logical_rows
        self.layer_traversals += traversed
        return tuple(outputs)

    def execute_native_flat_segments(self, flat_inputs, descriptor, transaction):
        """Execute one logical-row invocation without a padded physical slab."""

        self._validate_model()
        if type(transaction) is not NativeFlatSegmentedHybridTransaction:
            raise TypeError("native Qwen3.8 target requires exact transaction")
        if descriptor != self.descriptor or transaction.descriptor != descriptor:
            raise ValueError("native Qwen3.8 execution geometry differs")
        validate_native_flat_segment_admission(
            transaction.admission,
            transaction.owner,
            self,
            descriptor,
            expected_backend="metal",
        )
        self.validate_native_flat_owner(transaction.owner, descriptor)
        outputs = self._execute_rows(
            flat_inputs,
            descriptor,
            transaction.lane_caches,
            execution_attestation=transaction.admission.device_attestation,
        )
        return NativeFlatSegmentExecution(
            outputs=outputs,
            geometry_sha256=descriptor.geometry_sha256,
            physical_rows=descriptor.logical_rows,
        )

    def execute_live_flat_segments(self, flat_inputs, descriptor, lane_caches):
        """Literal native-flat serving entry for genuinely ragged geometry."""

        attestation = self.validate_live_execution_attestation()
        self._validate_hot_model_binding()
        if descriptor != self.descriptor:
            raise ValueError("native Qwen3.8 serving geometry differs")
        rows = tuple(tuple(row) for row in lane_caches)
        if len(rows) != descriptor.lane_count:
            raise ValueError("native Qwen3.8 serving lane count differs")
        outputs = self._execute_rows(
            flat_inputs,
            descriptor,
            rows,
            execution_attestation=attestation,
        )
        self._last_serving_execution_mode = "native_flat_b1s1_rows"
        return outputs

    def execute_live_rectangular_segments(self, flat_inputs, descriptor, transaction):
        """Run one segmented serving traversal over private KV/GDN successors.

        The direct-probe executor above deliberately proves a literal B1/S1
        arithmetic oracle.  Reusing that implementation in serving serialized
        every logical row through every layer.  The installed row-exact route
        already owns the equivalent fast physical law: large row-independent
        stages compact valid rows, while segmented caches preserve lane-local
        attention and recurrent ordering.  Use that one-traversal executor and
        flatten only its valid result rows back into the physical-result ABI.
        """

        attestation = self.validate_live_execution_attestation()
        self._validate_hot_model_binding()
        if descriptor != self.descriptor:
            raise ValueError("native Qwen3.8 serving geometry differs")
        if descriptor.logical_rows != (
            descriptor.lane_count * descriptor.max_segment_rows
        ):
            raise ValueError("segmented serving requires zero-padding geometry")
        if type(transaction) is not HybridVerifyTransaction or transaction.closed:
            raise TypeError(
                "native Qwen3.8 serving requires an open hybrid transaction"
            )
        if tuple(transaction.lengths) != descriptor.query_lengths:
            raise ValueError("native Qwen3.8 serving transaction geometry differs")
        tokens = tuple(flat_inputs)
        if len(tokens) != descriptor.logical_rows or any(
            type(token) is not int or token < 0 for token in tokens
        ):
            raise ValueError("native Qwen3.8 inputs must cover logical rows")

        rectangular = [
            list(tokens[descriptor.lane_slice(lane)])
            for lane in range(descriptor.lane_count)
        ]

        from ..runtime.progressive_external_verify import (
            PROGRESSIVE_FIDELITY_EXACT,
            PROGRESSIVE_TARGET_PROTOCOL_VERSION,
            execute_progressive_target_rows,
        )

        self.invocations += 1
        logits, features, detail = execute_progressive_target_rows(
            self.model,
            mx.array(rectangular, dtype=mx.int32),
            transaction.caches,
            self.capture_layers,
            protocol=PROGRESSIVE_TARGET_PROTOCOL_VERSION,
            fidelity=PROGRESSIVE_FIDELITY_EXACT,
            input_lengths=list(descriptor.query_lengths),
        )
        if (
            detail.get("execution") != "adapter_owned"
            or detail.get("row_exact") is not True
        ):
            raise RuntimeError("native Qwen3.8 serving left its row-exact executor")
        mx.eval(logits, features)
        outputs = tuple(
            (logits[lane, position], features[lane, position])
            for lane, length in enumerate(descriptor.query_lengths)
            for position in range(length)
        )

        self.successful_invocations += 1
        self.executed_rows += descriptor.logical_rows
        self.layer_traversals += len(self.model.language_model.model.layers)
        if mx.default_device() == mx.cpu:
            self._observed_cpu_fixture = True
        elif (
            mx.default_device() == mx.gpu
            and self._bound_identity["artifact"] == QWEN38_27B_OQ4E_MTP_CHECKPOINT
            and attestation.get("backend") == "metal"
            and attestation.get("ready") is True
            and attestation.get("device") == self._bound_identity["device"]
            and attestation.get("build") == self._bound_identity["build"]
        ):
            self._observed_real_artifact = True
        if self._observed_identity is None:
            self._observed_identity = dict(self._bound_identity)
        self._last_serving_execution_mode = "segmented_compact_valid_rows"
        return outputs

    def receipt(self):
        attestation = self.ast_native_flat_device_attestation()
        serving = self._construction_role == "serving"
        return {
            "route": (
                "qwen38-hybrid-native-flat-serving"
                if serving
                else "qwen38-hybrid-native-flat-direct-probe"
            ),
            "checkpoint": self._bound_identity["artifact"],
            "source_revision": self._bound_identity["source_revision"],
            "build": self._bound_identity["build"],
            "device": self._bound_identity["device"],
            "observed_execution_identity": (
                None
                if self._observed_identity is None
                else dict(self._observed_identity)
            ),
            "execution_scope": self.execution_scope,
            "logical_layout": LAYOUT_CLASS,
            "physical_layout": PHYSICAL_LAYOUT,
            "implemented": True,
            "qualified": False,
            # Construction proves that the serving implementation is wired;
            # only a per-round admitted binding may claim selection.
            "selected": False,
            "default": False,
            "observed_used_cpu_fixture": (
                False if serving else self._observed_cpu_fixture
            ),
            "observed_used_real_artifact": (
                False if serving else self._observed_real_artifact
            ),
            "observed_execution_cpu_fixture": self._observed_cpu_fixture,
            "observed_execution_real_artifact": self._observed_real_artifact,
            "target_invocations": self.invocations,
            "successful_target_invocations": self.successful_invocations,
            "logical_rows": self.executed_rows,
            "physical_rows": self.executed_rows,
            "padding_rows": 0,
            "outer_layer_traversals": self.layer_traversals,
            "row_arithmetic": (
                self._last_serving_execution_mode
                if serving and self._last_serving_execution_mode is not None
                else "ordinary_b1s1_direct"
            ),
            "target_row_exact_installed": self._installed_row_exact_route is not None,
            "embedded_mtp_present": self._mtp_present,
            "embedded_mtp_layers": self._mtp_layers,
            "embedded_mtp_engaged": False,
            "quantized_manifest_sha256": self._quantized_manifest_sha256,
            "model_manifest_sha256": self._expected_model_manifest_sha256,
            "bound_prefix_rows_per_lane_hard_limit": MAX_BOUND_PREFIX_ROWS,
            "device_attestation": attestation,
            "device_admitted": attestation["ready"],
            "serving_wired": serving,
            "direct_probe_only": not serving,
            "performance_claim": False,
        }


def _manifest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _live_positions(rows) -> tuple[int, ...]:
    positions = []
    for row in rows:
        offsets = {
            cache_operations(cache, capability="Qwen3.8 AST live position").position(
                cache
            )
            for cache in row
            if cache_operations(
                cache, capability="Qwen3.8 AST live position"
            ).offset_field
            is not None
        }
        if len(offsets) != 1 or None in offsets:
            raise ValueError("Qwen3.8 AST live lane positions differ")
        positions.append(int(next(iter(offsets))))
    return tuple(positions)


class _Qwen38LiveFlatDelegate:
    """Adapter-owned private successor and transfer publication."""

    def __init__(self, target, base_rows, flat_inputs, descriptor):
        self.target = target
        self.base_rows = tuple(tuple(row) for row in base_rows)
        self.flat_inputs = tuple(flat_inputs)
        self.descriptor = descriptor
        self.successor_rows = None
        self.transactions = ()
        self.execution_mode = None
        self.aggregate_transaction = False
        self._published = False

    def execute_private(self, *, descriptor, result_plan):
        if descriptor != self.descriptor:
            raise ValueError("Qwen3.8 AST live descriptor drifted")
        successor = tuple(
            tuple(copy.deepcopy(cache) for cache in row) for row in self.base_rows
        )
        self.successor_rows = successor
        uniform = descriptor.logical_rows == (
            descriptor.lane_count * descriptor.max_segment_rows
        )
        transactions = []
        try:
            if uniform:
                transaction = HybridVerifyRows(successor).begin(
                    lengths=descriptor.query_lengths
                )
                transactions.append(transaction)
                self.aggregate_transaction = True
                outputs = self.target.execute_live_rectangular_segments(
                    self.flat_inputs,
                    descriptor,
                    transaction,
                )
                self.execution_mode = "segmented_compact_valid_rows"
            else:
                for row, length in zip(
                    successor, descriptor.query_lengths, strict=True
                ):
                    transactions.append(
                        HybridVerifyRows((row,)).begin(lengths=(length,))
                    )
                outputs = self.target.execute_live_flat_segments(
                    self.flat_inputs,
                    descriptor,
                    tuple(transaction.owner.rows[0] for transaction in transactions),
                )
                self.execution_mode = "native_flat_b1s1_rows"
        except BaseException:
            self.transactions = tuple(transactions)
            self.abort()
            raise
        self.transactions = tuple(transactions)
        result = ASTPhysicalResult(
            descriptor,
            outputs,
            None,
            storage="flat_row_pairs",
            invocation_count=1,
        )
        result_plan.validate_result(result)
        return result

    def prepare_commit(self, accepted, *, descriptor, result, prepare_authority):
        if not self.transactions or any(
            transaction.closed for transaction in self.transactions
        ):
            raise RuntimeError("Qwen3.8 AST live transaction is unavailable")
        if self.aggregate_transaction:
            committed = tuple(
                tuple(row) for row in self.transactions[0].commit(accepted)
            )
        else:
            committed = tuple(
                tuple(transaction.commit((lane_accepted,))[0])
                for transaction, lane_accepted in zip(
                    self.transactions, accepted, strict=True
                )
            )
        self.successor_rows = committed
        return prepare_authority.seal(
            successor_rows=committed,
            publication_rows=committed,
            publication_mode="transfer_successor",
            accepted_lengths=accepted,
            successor_positions=_live_positions(committed),
            descriptor_geometry_sha256=descriptor.geometry_sha256,
            result_manifest_sha256=result.manifest_sha256,
            delegate_token=self,
        )

    def publish_commit(self, prepared):
        if prepared.delegate_token is not self or self._published:
            raise RuntimeError("Qwen3.8 AST live publication token differs")
        self._published = True

    def abort(self):
        first = None
        for transaction in self.transactions:
            if transaction.closed:
                continue
            try:
                transaction.abort()
            except BaseException as error:  # noqa: BLE001 - every lane cleans up
                first = first or error
        if first is not None:
            raise first


class Qwen38NativeFlatServingBinding:
    """One admitted live progressive launch; execution begins explicitly."""

    def __init__(
        self,
        provider,
        *,
        rows,
        descriptor,
        capture_layers,
        flat_inputs,
        session_scope_sha256,
        discovery,
        admission,
    ):
        self.provider = provider
        self.capture_layers = tuple(capture_layers)
        self.session_scope_sha256 = session_scope_sha256
        self.discovery = dict(discovery)
        self.rows = tuple(tuple(row) for row in rows)
        self.positions = _live_positions(self.rows)
        self.descriptor = descriptor
        self.target = provider._target(descriptor, capture_layers)
        self.target.validate_live_execution_attestation()
        self.admission = validate_ast_serving_admission(
            admission,
            discovery,
            expected_session_scope_sha256=session_scope_sha256,
        )
        owner_identity = _manifest(
            {
                "positions": self.positions,
                "contracts": [
                    [
                        cache_operations(
                            cache, capability="Qwen3.8 AST owner"
                        ).fingerprint
                        for cache in row
                    ]
                    for row in self.rows
                ],
            }
        )
        identity = self.target.identity
        self.authority = ASTLiveStateAuthority(
            artifact=identity["artifact"],
            session_scope_sha256=session_scope_sha256,
            owner_identity_sha256=owner_identity,
            provider_manifest_sha256=self.admission["manifest_sha256"],
            source_manifest_sha256=self.admission["source_manifest_sha256"],
            build_manifest_sha256=self.admission["build_manifest_sha256"],
            device_manifest_sha256=self.admission["device_manifest_sha256"],
            max_lanes=provider.hard_limits["max_lanes"],
            max_caches_per_lane=max(map(len, self.rows)),
            max_segment_rows=provider.hard_limits["max_segment_rows"],
            max_total_rows=provider.hard_limits["max_total_rows"],
            max_physical_rows=provider.hard_limits["max_total_rows"],
            required_plane_kinds=("attention_kv", "gdn_recurrent"),
            publication_modes=("transfer_successor",),
            result_storage_classes=("flat_row_pairs",),
        )
        self.registry = ASTLiveLaneStateRegistry(
            self.rows, self.positions, authority=self.authority
        )
        self.delegate = _Qwen38LiveFlatDelegate(
            self.target, self.rows, flat_inputs, descriptor
        )
        self.coordinator = None
        self.result = None
        self.published = None
        self._closed = False

    def execute(self):
        if self.coordinator is not None:
            raise RuntimeError("Qwen3.8 AST serving binding already executed")
        self.coordinator = ASTPhysicalCommitCoordinator(
            self.registry,
            self.registry.handles,
            self.rows,
            self.positions,
            descriptor=self.descriptor,
            result_plan=ASTPhysicalResultPlan.from_descriptor(
                self.descriptor, storage="flat_row_pairs"
            ),
            delegate=self.delegate,
            authority=self.authority,
        )
        try:
            self.result = self.coordinator.execute(authority=self.authority)
        except BaseException:
            self._closed = True
            raise
        return self.result

    def commit(self, accepted_lengths):
        if self.coordinator is None:
            raise RuntimeError("Qwen3.8 AST serving binding is unexecuted")
        try:
            self.provider.validate_live_serving_admission(
                descriptor=self.descriptor,
                capture_layers=self.capture_layers,
                session_scope_sha256=self.session_scope_sha256,
                discovery=self.discovery,
                admission=self.admission,
            )
            self.coordinator.prepare(accepted_lengths, authority=self.authority)
            # Target-law and prepare callbacks have run since execution. Rebind
            # every authority immediately before the publication pointer swap.
            self.provider.validate_live_serving_admission(
                descriptor=self.descriptor,
                capture_layers=self.capture_layers,
                session_scope_sha256=self.session_scope_sha256,
                discovery=self.discovery,
                admission=self.admission,
            )
            self.published = self.coordinator.publish(authority=self.authority)
        except BaseException:
            try:
                self.coordinator.abort()
            except BaseException:  # noqa: BLE001,S110 - preserve authority error
                pass
            self._closed = True
            raise
        self._closed = True
        return self.published.publication_rows

    def abort(self):
        if self._closed:
            return
        try:
            if self.coordinator is not None and self.published is None:
                self.coordinator.abort()
        finally:
            self._closed = True

    def receipt(self):
        return {
            "schema": "mlx2.ast-qwen38-native-flat-serving.v1",
            "implemented": True,
            "qualified": False,
            "selected": True,
            "default": False,
            "observed_used": self.published is not None,
            "geometry_sha256": self.descriptor.geometry_sha256,
            "registry_epoch": self.registry.epoch,
            "authority_manifest_sha256": self.authority.manifest_sha256,
            "admission_manifest_sha256": self.admission["manifest_sha256"],
            "result": None if self.result is None else self.result.receipt(),
            "execution_mode": self.delegate.execution_mode,
        }


class Qwen38NativeFlatPhysicalTargetProvider:
    """Reviewed exact-artifact provider for generic serving discovery."""

    protocol = PROVIDER_PROTOCOL
    hard_limits: ClassVar[Mapping[str, int]] = METAL_PROBE_HARD_LIMITS

    def __init__(self, model):
        self.model = model
        self.source_revision_token = _git_source_revision()
        _require_clean_source_revision(self.source_revision_token)
        attestation = dense_qwen3_metal_device_attestation()
        self.identity = MappingProxyType(
            {
                "source_revision": self.source_revision_token,
                "build": attestation["build"],
                "device": attestation["device"],
                "artifact": QWEN38_27B_OQ4E_MTP_CHECKPOINT,
            }
        )
        self.hard_limits = MappingProxyType(dict(type(self).hard_limits))
        self.source_identity = tuple(sorted(_serving_source_hashes().items()))
        self._protocol_binding = self.protocol
        self._identity_binding = tuple(sorted(self.identity.items()))
        self._hard_limits_binding = tuple(sorted(self.hard_limits.items()))
        self._source_revision_binding = self.source_revision_token
        self._source_identity_binding = self.source_identity
        self._source_generation_binding = _serving_source_generation()
        self._expected_model_manifest_sha256 = (
            QWEN38_27B_OQ4E_NO_MTP_MODEL_MANIFEST_SHA256
            if model.mtp is None
            else QWEN38_27B_OQ4E_MTP_REVIEWED_MANIFEST["model_manifest_sha256"]
        )
        self._targets = {}

    def validate_revision(self):
        if (
            self.protocol != self._protocol_binding
            or tuple(sorted(self.identity.items())) != self._identity_binding
            or tuple(sorted(self.hard_limits.items())) != self._hard_limits_binding
            or self.source_revision_token != self._source_revision_binding
            or self.source_identity != self._source_identity_binding
            or _serving_source_generation() != self._source_generation_binding
        ):
            raise RuntimeError("Qwen3.8 AST physical-target binding drifted")

    @staticmethod
    def serving_admission_contract():
        """Truthful owner contract used only for live serving admission."""

        return {
            "schema": "mlx2.ast-qwen38-live-serving-owner.v2",
            "state_write_law": "disjoint_private_successor_until_publish",
            "transaction_law": (
                "aggregate_segmented_transaction_only_for_zero_padding_geometry_"
                "else_one_b1_transaction_per_lane"
            ),
            "segmented_execution_law": (
                "one_row_exact_traversal_for_equal_length_lanes"
            ),
            "ragged_execution_law": "literal_native_flat_b1s1_rows",
            "publication_law": "transfer_successor_after_complete_prepare",
            "fallback_law": "ordinary_padded_before_native_execution_only",
            "abort_law": "base_membership_exact_or_registry_poisoned",
            "serving_wired": True,
            "selection_eligible": True,
            "implemented": True,
            "qualified": False,
            "selected": False,
            "default": False,
            "observed_used": False,
        }

    def _session_source_manifest(self, session_scope_sha256):
        # Provider installation is the model/engine session boundary at which
        # clean Git provenance and exact bytes are established. Per-cohort
        # scopes derive a cheap immutable binding without retaining an
        # unbounded cache or touching the filesystem on the hot path.
        self.validate_revision()
        return _manifest(
            {
                "session_scope_sha256": session_scope_sha256,
                "source_revision": self.source_revision_token,
                "source_hashes": dict(self.source_identity),
            }
        )

    def _target(self, descriptor, capture_layers):
        key = (descriptor.geometry_sha256, tuple(capture_layers))
        target = self._targets.get(key)
        if target is None:
            target = make_reviewed_qwen38_27b_native_flat_target(
                self.model,
                descriptor=descriptor,
                capture_layers=capture_layers,
                identity=self.identity,
                artifact_metadata={
                    "checkpoint": QWEN38_27B_OQ4E_MTP_CHECKPOINT,
                    **QWEN38_27B_OQ4E_MTP_REVIEWED_MANIFEST,
                },
                construction_role="serving",
                expected_model_manifest_sha256=(self._expected_model_manifest_sha256),
                source_hashes={
                    path: digest
                    for path, digest in self.source_identity
                    if path in SOURCE_PATHS
                },
            )
            self._targets[key] = target
        elif target.descriptor != descriptor:
            raise RuntimeError("Qwen3.8 AST target geometry digest collided")
        return target

    def discover(self, request):
        target = self._target(request.descriptor, request.capture_layers)
        target_contract = target.ast_live_serving_target_contract()
        return ASTPhysicalTargetCandidate.create(
            geometry_sha256=request.descriptor.geometry_sha256,
            capture_layers_sha256=request.capture_layers_sha256,
            fidelity=request.fidelity,
            fallback_target_protocol=request.fallback_target_protocol,
            identity=self.identity,
            source_hashes=self.source_identity,
            hard_limits=self.hard_limits,
            owner_protocol="ast-live-state-transfer-successor-v1",
            native_target_protocol=SERVING_TARGET_PROTOCOL,
            owner_capability_manifest_sha256=_manifest(
                self.serving_admission_contract()
            ),
            target_capability_manifest_sha256=_manifest(target_contract),
            decline_reasons=("default_off_policy_required",),
        )

    def admit_serving_physical_target(
        self,
        *,
        descriptor,
        capture_layers,
        session_scope_sha256,
        discovery,
        **_unused,
    ):
        """Revalidate session source and live device before selection."""

        self.validate_revision()
        target = self._target(descriptor, capture_layers)
        target.validate_session_model_binding(session_scope_sha256)
        attestation = target.validate_live_execution_attestation()
        source_manifest = self._session_source_manifest(session_scope_sha256)
        owner_manifest = _manifest(self.serving_admission_contract())
        target_manifest = _manifest(target.ast_live_serving_target_contract())
        if (
            discovery.get("owner_capability_manifest_sha256") != owner_manifest
            or discovery.get("target_capability_manifest_sha256") != target_manifest
        ):
            raise RuntimeError("Qwen3.8 AST serving capabilities drifted")
        return ast_serving_admission_receipt(
            discovery=discovery,
            session_scope_sha256=session_scope_sha256,
            source_manifest_sha256=source_manifest,
            build_manifest_sha256=_manifest(attestation["build"]),
            device_manifest_sha256=_manifest(attestation["device"]),
            owner_capability_manifest_sha256=owner_manifest,
            target_capability_manifest_sha256=target_manifest,
        )

    def bind_serving_physical_target(
        self,
        *,
        rows,
        descriptor,
        capture_layers,
        flat_inputs,
        session_scope_sha256,
        discovery,
        admission,
    ):
        self.validate_live_serving_admission(
            descriptor=descriptor,
            capture_layers=capture_layers,
            session_scope_sha256=session_scope_sha256,
            discovery=discovery,
            admission=admission,
        )
        return Qwen38NativeFlatServingBinding(
            self,
            rows=rows,
            descriptor=descriptor,
            capture_layers=capture_layers,
            flat_inputs=flat_inputs,
            session_scope_sha256=session_scope_sha256,
            discovery=discovery,
            admission=admission,
        )

    def validate_live_serving_admission(
        self,
        *,
        descriptor,
        capture_layers,
        session_scope_sha256,
        discovery,
        admission,
    ):
        """Rebind all mutable authority immediately before publication."""

        self.validate_revision()
        checked = validate_ast_serving_admission(
            admission,
            discovery,
            expected_session_scope_sha256=session_scope_sha256,
        )
        target = self._target(descriptor, capture_layers)
        target._validate_hot_model_binding()
        attestation = target.validate_live_execution_attestation()
        expected = {
            "source_manifest_sha256": self._session_source_manifest(
                session_scope_sha256
            ),
            "build_manifest_sha256": _manifest(attestation["build"]),
            "device_manifest_sha256": _manifest(attestation["device"]),
            "owner_capability_manifest_sha256": _manifest(
                self.serving_admission_contract()
            ),
            "target_capability_manifest_sha256": _manifest(
                target.ast_live_serving_target_contract()
            ),
        }
        if any(checked.get(name) != value for name, value in expected.items()):
            raise RuntimeError("Qwen3.8 AST live admission authority drifted")
        return checked


def install_qwen38_native_flat_physical_target_provider(model):
    """Install only when exact repository provenance is locally available."""

    try:
        return Qwen38NativeFlatPhysicalTargetProvider(model)
    except (OSError, RuntimeError, subprocess.SubprocessError):
        return None


def make_reviewed_qwen38_27b_native_flat_target(
    model,
    *,
    descriptor: FlatSegmentDescriptor,
    capture_layers: Sequence[int],
    identity: Mapping[str, str],
    artifact_metadata: Mapping[str, object],
    construction_role: str = "direct_probe",
    expected_model_manifest_sha256: str | None = None,
    source_hashes: Mapping[str, str] | None = None,
):
    """Bind the native direct target to the independently reviewed 27B record."""

    checked_identity = _identity(identity)
    expected_metadata = {
        "checkpoint": QWEN38_27B_OQ4E_MTP_CHECKPOINT,
        **QWEN38_27B_OQ4E_MTP_REVIEWED_MANIFEST,
    }
    if (
        checked_identity["artifact"] != QWEN38_27B_OQ4E_MTP_CHECKPOINT
        or not isinstance(artifact_metadata, Mapping)
        or dict(artifact_metadata) != expected_metadata
    ):
        raise ValueError("native Qwen3.8 target lacks reviewed artifact metadata")
    return Qwen38HybridNativeFlatTarget(
        model,
        descriptor=descriptor,
        capture_layers=capture_layers,
        identity=checked_identity,
        expected_model_manifest_sha256=(
            QWEN38_27B_OQ4E_MTP_REVIEWED_MANIFEST["model_manifest_sha256"]
            if expected_model_manifest_sha256 is None
            else expected_model_manifest_sha256
        ),
        construction_role=construction_role,
        source_hashes=source_hashes,
    )


def _ordinary_serial_oracles(target, owner, descriptor, tokens, accepted, continuation):
    base_rows = copy.deepcopy(owner.rows)
    outputs = []
    for lane in range(descriptor.lane_count):
        caches = copy.deepcopy(base_rows[lane])
        for token in tokens[descriptor.lane_slice(lane)]:
            logits, taps = _ordinary_forward_with_taps(
                target.model,
                mx.array([[token]], dtype=mx.int32),
                caches,
                target.capture_layers,
            )
            mx.eval(logits, taps)
            outputs.append((logits[0, 0], taps[0, 0]))
    committed_rows = copy.deepcopy(base_rows)
    for lane, count in enumerate(accepted):
        for token in tokens[descriptor.lane_slice(lane)][:count]:
            mx.eval(
                target.model(
                    mx.array([[token]], dtype=mx.int32),
                    cache=committed_rows[lane],
                )
            )
    committed_state = successor_bound_hybrid_state(
        committed_rows, owner.boundaries, accepted
    )
    continuation_outputs = []
    for lane, token in enumerate(continuation):
        logits, taps = _ordinary_forward_with_taps(
            target.model,
            mx.array([[token]], dtype=mx.int32),
            copy.deepcopy(committed_rows[lane]),
            target.capture_layers,
        )
        mx.eval(logits, taps)
        continuation_outputs.append((logits[0, 0], taps[0, 0]))
    return tuple(outputs), committed_state, tuple(continuation_outputs)


@dataclass(frozen=True, slots=True)
class Qwen38HybridNativeFlatProbeResult:
    receipt: dict[str, object]


def run_qwen38_hybrid_native_flat_probe(
    owner,
    target,
    descriptor,
    flat_tokens: Sequence[int],
    accepted_lengths: Sequence[int],
    continuation_tokens: Sequence[int],
    *,
    expected_identity: Mapping[str, str],
    expected_owner_semantic_contract_sha256: str,
    expected_state_contract_fingerprints: Sequence[str],
    expected_owner_source_hashes: Mapping[str, str],
    expected_target_source_hashes: Mapping[str, str],
):
    """Run one direct correctness unit; publication is the final operation."""

    if type(owner) is not NativeFlatSegmentedHybridRows:
        raise TypeError("native Qwen3.8 probe requires exact owner")
    if type(target) is not Qwen38HybridNativeFlatTarget:
        raise TypeError("native Qwen3.8 probe requires exact target")
    if target.invocations != 0:
        raise FlatSegmentUnsupported("native Qwen3.8 target must be fresh")
    tokens = tuple(flat_tokens)
    accepted = tuple(accepted_lengths)
    continuation = tuple(continuation_tokens)
    if (
        descriptor != target.descriptor
        or len(tokens) != descriptor.logical_rows
        or any(type(token) is not int or token < 0 for token in tokens)
        or len(accepted) != descriptor.lane_count
        or any(type(value) is not int for value in accepted)
        or any(
            value < 1 or value > length
            for value, length in zip(accepted, descriptor.query_lengths, strict=True)
        )
        or len(continuation) != descriptor.lane_count
        or any(type(token) is not int or token < 0 for token in continuation)
    ):
        raise FlatSegmentError("native Qwen3.8 probe geometry differs")
    admission = query_native_flat_segment_admission(
        owner,
        target,
        descriptor,
        expected_identity=expected_identity,
        expected_owner_semantic_contract_sha256=(
            expected_owner_semantic_contract_sha256
        ),
        expected_state_contract_fingerprints=expected_state_contract_fingerprints,
        expected_owner_source_hashes=expected_owner_source_hashes,
        expected_target_source_hashes=expected_target_source_hashes,
        expected_backend="metal",
    )
    if not admission.available:
        raise FlatSegmentUnsupported(
            "native Qwen3.8 admission declined: " + ",".join(admission.decline_reasons)
        )
    expected_outputs, expected_state, expected_continuation = _ordinary_serial_oracles(
        target, owner, descriptor, tokens, accepted, continuation
    )
    initial_positions = [boundary.position for boundary in owner.boundaries]
    initial_generations = [boundary.generation for boundary in owner.boundaries]
    transaction = None
    published = False
    try:
        transaction = owner.begin_native_flat(
            descriptor, target_executor=target, admission=admission
        )
        execution = target.execute_native_flat_segments(tokens, descriptor, transaction)
        validate_native_flat_segment_execution(execution, descriptor)
        if target.invocations != 1 or not _outputs_exact(
            execution.outputs, expected_outputs
        ):
            raise FlatSegmentError("native Qwen3.8 full-logit/tap oracle differs")
        prepared = transaction.prepare_commit(accepted)
        if prepared.state != expected_state:
            raise FlatSegmentError("native Qwen3.8 prepared state differs")
        prepared_rows = transaction._prepared_successor_rows_for_oracle(prepared)
        actual_continuation = []
        for lane, token in enumerate(continuation):
            logits, taps = _ordinary_forward_with_taps(
                target.model,
                mx.array([[token]], dtype=mx.int32),
                copy.deepcopy(prepared_rows[lane]),
                target.capture_layers,
            )
            mx.eval(logits, taps)
            actual_continuation.append((logits[0, 0], taps[0, 0]))
        if not _outputs_exact(tuple(actual_continuation), expected_continuation):
            raise FlatSegmentError("native Qwen3.8 continuation oracle differs")
        published_state = transaction.publish_commit(prepared)
        if published_state != expected_state:
            raise FlatSegmentError("native Qwen3.8 published state differs")
        published = True
    finally:
        if transaction is not None and not transaction.closed and not published:
            transaction.abort()
    identity = dict(expected_identity)
    target_receipt = target.receipt()
    return Qwen38HybridNativeFlatProbeResult(
        receipt={
            "schema": PROBE_SCHEMA,
            "route": "qwen38-hybrid-native-flat-direct-probe",
            "execution_scope": NATIVE_EXECUTION_SCOPE,
            "checkpoint": identity["artifact"],
            "source_revision": identity["source_revision"],
            "build": identity["build"],
            "device": identity["device"],
            "device_attestation": admission.device_attestation,
            "admission_manifest_sha256": admission.receipt()["manifest_sha256"],
            "model_manifest_sha256": target._expected_model_manifest_sha256,
            "quantized_manifest_sha256": target._quantized_manifest_sha256,
            "geometry_sha256": descriptor.geometry_sha256,
            "lane_count": descriptor.lane_count,
            "logical_rows": descriptor.logical_rows,
            "physical_rows": descriptor.logical_rows,
            "padding_rows": 0,
            "accepted_lengths": list(accepted),
            "bound_prefix_rows_per_lane_hard_limit": MAX_BOUND_PREFIX_ROWS,
            "initial_boundary_positions": initial_positions,
            "initial_boundary_generations": initial_generations,
            "target_invocations": 1,
            "outer_layer_traversals": len(target.model.language_model.model.layers),
            "ordinary_serial_oracle": {
                "full_logits": "bit_exact",
                "capture_taps": "bit_exact",
                "kv_and_gdn_state": "bit_exact_prepublication",
                "continuation_logits_and_taps": "bit_exact_prepublication",
            },
            "logical_layout": LAYOUT_CLASS,
            "physical_layout": PHYSICAL_LAYOUT,
            "row_arithmetic": "ordinary_b1s1_direct",
            "depthwise_shrinking_loop": False,
            "padded_rectangle_constructed": False,
            "embedded_mtp_present": target._mtp_present,
            "embedded_mtp_layers": target._mtp_layers,
            "embedded_mtp_engaged": False,
            "implemented": True,
            "qualified": False,
            "selected": False,
            "default": False,
            "observed_used_cpu_fixture": target_receipt["observed_used_cpu_fixture"],
            "observed_used_real_artifact": target_receipt[
                "observed_used_real_artifact"
            ],
            "serving_wired": False,
            "apcv2_wired": False,
            "performance_claim": False,
        }
    )


__all__ = [
    "PHYSICAL_LAYOUT",
    "PROBE_SCHEMA",
    "SOURCE_PATHS",
    "Qwen38HybridNativeFlatProbeResult",
    "Qwen38HybridNativeFlatTarget",
    "install_qwen38_native_flat_physical_target_provider",
    "make_reviewed_qwen38_27b_native_flat_target",
    "run_qwen38_hybrid_native_flat_probe",
]
