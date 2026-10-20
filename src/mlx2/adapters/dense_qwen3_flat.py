"""Adapter-owned exact flat-segment tensor math for dense standard Qwen3.

This direct probe keeps the ordinary model route unchanged.  It flattens all
complete K+1 paths once, but deliberately uses the model's proven B1/S1
projection arithmetic instead of treating a wider packed matmul as exact.
The CPU and default-off Metal-probe protocols are distinct. Serving selection,
performance policy, and GPU qualification are out of scope.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path

import mlx.core as mx
from mlx import nn

from ..runtime.flat_segmented_ast import (
    ABORT_LAW,
    COMMIT_LAW,
    DESCRIPTOR_SCHEMA,
    EXECUTION_LAW,
    FIDELITY_EXACT,
    LAYOUT_CLASS,
    DEVICE_ATTESTATION_SCHEMA,
    METAL_EXECUTION_SCOPE,
    METAL_TARGET_PROTOCOL,
    REAL_EXECUTION_SCOPE,
    REAL_TARGET_PROTOCOL,
    STATE_WRITE_LAW,
    TARGET_CAPABILITY_SCHEMA,
    TERMINAL_LAW,
    FlatSegmentDescriptor,
    FlatSegmentExecution,
    validate_real_flat_segment_admission,
    validate_metal_flat_segment_admission,
)
from ..runtime.flat_segmented_kv import (
    METAL_PROBE_HARD_LIMITS,
    REAL_FLAT_HARD_LIMITS,
    FlatSegmentedKVRows,
    FlatSegmentedKVTransaction,
    MetalFlatSegmentedKVRows,
)
from ..runtime.models import standard_decoder as standard

_PACKAGE = Path(__file__).resolve().parents[1]
_PACKAGE_PREFIX = "src/mlx2/"
_SOURCE_PATHS = (
    "src/mlx2/adapters/dense_qwen3_flat.py",
    "src/mlx2/runtime/models/standard_decoder.py",
    "src/mlx2/runtime/models/base.py",
    "src/mlx2/runtime/models/activations.py",
    "src/mlx2/runtime/models/rope_utils.py",
    "src/mlx2/runtime/flat_segmented_ast.py",
    "src/mlx2/runtime/flat_segmented_kv.py",
)
_METAL_SOURCE_PATHS = _SOURCE_PATHS + (
    "src/mlx2/adapters/dense_qwen3_metal_probe.py",
)


def _source_hashes(paths=_SOURCE_PATHS) -> dict[str, str]:
    result = {}
    for name in paths:
        if not name.startswith(_PACKAGE_PREFIX):
            raise ValueError("dense Qwen3 flat source path must be package-relative")
        path = _PACKAGE / name[len(_PACKAGE_PREFIX) :]
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _identity(identity: Mapping[str, str]) -> dict[str, str]:
    fields = {"source_revision", "build", "device", "artifact"}
    if not isinstance(identity, Mapping) or set(identity) != fields or any(
        not isinstance(identity[name], str) or not identity[name] for name in fields
    ):
        raise ValueError("dense Qwen3 flat identity fields differ")
    return {name: identity[name] for name in sorted(fields)}


def dense_qwen3_metal_device_attestation() -> dict[str, object]:
    """Return the live, normalized MLX Metal identity without doing tensor work."""

    ready = mx.metal.is_available() and mx.default_device() == mx.gpu
    info = mx.device_info() if ready else {}
    architecture = info.get("architecture")
    name = info.get("device_name")
    identity_ready = all(
        isinstance(value, str) and value for value in (architecture, name)
    )
    return {
        "schema": DEVICE_ATTESTATION_SCHEMA,
        "backend": "metal",
        "device": (
            f"mlx-metal:{architecture}:{name}"
            if ready and identity_ready
            else "mlx-metal:unavailable"
        ),
        "build": f"mlx-{mx.__version__}",
        "ready": ready and identity_ready,
    }


def _rowwise(call, value):
    """Run row-independent math with the ordinary B1/S1 launch geometry."""

    return mx.concatenate(
        [call(value[:, row : row + 1]) for row in range(value.shape[1])],
        axis=1,
    )


def _rowwise_add(left, right):
    return mx.concatenate(
        [
            left[:, row : row + 1] + right[:, row : row + 1]
            for row in range(left.shape[1])
        ],
        axis=1,
    )


def _linear_contract(module, *, input_features, output_features, bias):
    if type(module) is not nn.Linear or module.weight.shape != (
        output_features,
        input_features,
    ):
        return False
    actual_bias = getattr(module, "bias", None)
    return (
        actual_bias is None
        if not bias
        else actual_bias is not None and actual_bias.shape == (output_features,)
    )


def _norm_contract(module, *, dimensions, epsilon):
    return (
        type(module) is nn.RMSNorm
        and module.eps == epsilon
        and module.weight.shape == (dimensions,)
    )


class DenseQwen3FlatSegmentTarget:
    """One adapter invocation over complete dense-Qwen3 segment paths."""

    execution_scope = REAL_EXECUTION_SCOPE
    target_protocol = REAL_TARGET_PROTOCOL
    owner_type = FlatSegmentedKVRows
    admission_validator = staticmethod(validate_real_flat_segment_admission)
    hard_limits = REAL_FLAT_HARD_LIMITS

    def __init__(self, model, *, capture_layers, identity: Mapping[str, str]):
        self._model = model
        self._capture_layers = tuple(capture_layers)
        self.identity = _identity(identity)
        self.invocations = 0
        self.executed_rows = 0
        self._validate_model()

    @property
    def model(self):
        return self._model

    @property
    def capture_layers(self) -> tuple[int, ...]:
        return self._capture_layers

    def ast_flat_segment_execution_identity(self) -> tuple[int, tuple[int, ...]]:
        """Private ticket binding for the exact model object and tap schema."""

        return id(self._model), self._capture_layers

    def ast_flat_segment_device_attestation(self) -> dict[str, object]:
        """Bind the CPU-labelled contract to the live MLX CPU backend."""

        return {
            "schema": DEVICE_ATTESTATION_SCHEMA,
            "backend": "cpu",
            "device": "mlx-cpu",
            "build": f"mlx-{mx.__version__}",
            "ready": mx.default_device() == mx.cpu,
        }

    def _validate_model(self) -> None:
        model = self.model
        if type(model) is not standard.Model:
            raise TypeError("real flat executor requires the standard decoder model")
        args = model.args
        if (
            args.model_type != "qwen3"
            or args.num_experts
            or args.rope_scaling is not None
        ):
            raise ValueError("real flat executor requires dense full-attention Qwen3")
        if os.environ.get("MLX2_FP_DECODE_KERNEL", "0") == "1":
            raise ValueError("real flat executor excludes the FP decode kernel")
        if (
            not self.capture_layers
            or self.capture_layers != tuple(sorted(set(self.capture_layers)))
            or self.capture_layers[0] < 0
            or self.capture_layers[-1] >= len(model.layers)
        ):
            raise ValueError("real flat executor needs valid target capture layers")
        for layer in model.layers:
            if (
                type(layer) is not standard.Layer
                or type(layer.self_attn) is not standard.Attention
                or type(layer.mlp) is not standard.MLP
            ):
                raise ValueError("real flat executor requires the native dense topology")
            attention = layer.self_attn
            dim = args.hidden_size
            head_dim = args.head_dim
            if not all(
                (
                    _linear_contract(
                        attention.q_proj,
                        input_features=dim,
                        output_features=args.num_attention_heads * head_dim,
                        bias=args.attention_bias,
                    ),
                    _linear_contract(
                        attention.k_proj,
                        input_features=dim,
                        output_features=args.num_key_value_heads * head_dim,
                        bias=args.attention_bias,
                    ),
                    _linear_contract(
                        attention.v_proj,
                        input_features=dim,
                        output_features=args.num_key_value_heads * head_dim,
                        bias=args.attention_bias,
                    ),
                    _linear_contract(
                        attention.o_proj,
                        input_features=args.num_attention_heads * head_dim,
                        output_features=dim,
                        bias=False,
                    ),
                    _linear_contract(
                        layer.mlp.gate_proj,
                        input_features=dim,
                        output_features=args.intermediate_size,
                        bias=False,
                    ),
                    _linear_contract(
                        layer.mlp.up_proj,
                        input_features=dim,
                        output_features=args.intermediate_size,
                        bias=False,
                    ),
                    _linear_contract(
                        layer.mlp.down_proj,
                        input_features=args.intermediate_size,
                        output_features=dim,
                        bias=False,
                    ),
                )
            ):
                raise ValueError("real flat executor projector contract differs")
            if (
                not _norm_contract(
                    layer.input_layernorm,
                    dimensions=args.hidden_size,
                    epsilon=args.rms_norm_eps,
                )
                or not _norm_contract(
                    layer.post_attention_layernorm,
                    dimensions=args.hidden_size,
                    epsilon=args.rms_norm_eps,
                )
                or not _norm_contract(
                    attention.q_norm,
                    dimensions=args.head_dim,
                    epsilon=args.rms_norm_eps,
                )
                or not _norm_contract(
                    attention.k_norm,
                    dimensions=args.head_dim,
                    epsilon=args.rms_norm_eps,
                )
                or type(attention.rope) is not nn.RoPE
                or attention.n_heads != args.num_attention_heads
                or attention.n_kv_heads != args.num_key_value_heads
                or attention.head_dim != args.head_dim
                or attention.scale != args.head_dim**-0.5
                or attention.rope.dims != args.head_dim
                or attention.rope.base != args.rope_theta
                or attention.rope.traditional != args.rope_traditional
                or attention.rope.scale != 1.0
            ):
                raise ValueError("real flat executor model math contract differs")
        if not _norm_contract(
            model.model.norm,
            dimensions=args.hidden_size,
            epsilon=args.rms_norm_eps,
        ):
            raise ValueError("real flat executor final norm contract differs")
        if (
            type(model.model.embed_tokens) is not nn.Embedding
            or model.model.embed_tokens.weight.shape
            != (args.vocab_size, args.hidden_size)
            or (
            not args.tie_word_embeddings and type(model.lm_head) is not nn.Linear
            )
            or (
                not args.tie_word_embeddings
                and not _linear_contract(
                    model.lm_head,
                    input_features=args.hidden_size,
                    output_features=args.vocab_size,
                    bias=False,
                )
            )
        ):
            raise ValueError("real flat executor requires an unquantized output head")
        self.backbone_dtype = model._homogeneous_backbone_dtype()

    def validate_flat_owner(
        self,
        owner: FlatSegmentedKVRows,
        descriptor: FlatSegmentDescriptor,
    ) -> None:
        """Validate dynamic model/cache compatibility before ``begin_flat``."""

        self._validate_model()
        if type(owner) is not self.owner_type:
            raise TypeError("real flat target requires its exact KV owner")
        if not isinstance(descriptor, FlatSegmentDescriptor):
            raise TypeError("real flat target requires a validated descriptor")
        if descriptor.lane_count != len(owner.rows):
            raise ValueError("real flat target lane ownership differs")
        if any(len(row) != len(self.model.layers) for row in owner.rows):
            raise ValueError("real flat KV owner does not cover every target layer")
        args = self.model.args
        for lane, row in enumerate(owner.rows):
            if row[0].offset + descriptor.query_lengths[lane] > (
                args.max_position_embeddings
            ):
                raise ValueError("real flat segment exceeds target position capacity")
            for layer, cache in zip(self.model.layers, row, strict=True):
                if cache.keys is None:
                    continue
                attention = layer.self_attn
                if (
                    cache.values is None
                    or cache.keys.shape[0] != 1
                    or cache.values.shape[0] != 1
                    or cache.keys.shape[1] != attention.n_kv_heads
                    or cache.values.shape[1] != attention.n_kv_heads
                    or cache.keys.shape[-1] != attention.head_dim
                    or cache.values.shape[-1] != attention.head_dim
                    or cache.keys.dtype != self.backbone_dtype
                    or cache.values.dtype != self.backbone_dtype
                ):
                    raise ValueError("real flat KV target geometry or dtype differs")

    @property
    def source_hashes(self) -> dict[str, str]:
        return _source_hashes()

    def ast_flat_segment_target_contract(self) -> dict:
        # Revalidate module ownership each time capability is queried: serving
        # policies may replace projections after adapter construction.
        self._validate_model()
        return {
            "schema": TARGET_CAPABILITY_SCHEMA,
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
            "hard_limits": {
                **self.hard_limits,
                "max_segment_rows": min(
                    self.hard_limits["max_segment_rows"],
                    self.model.args.max_position_embeddings,
                ),
            },
            "source_hashes": self.source_hashes,
            "identity": dict(self.identity),
        }

    def execute_flat_segments(
        self,
        flat_inputs,
        descriptor: FlatSegmentDescriptor,
        transaction: FlatSegmentedKVTransaction,
    ) -> FlatSegmentExecution:
        self._validate_model()
        if not isinstance(descriptor, FlatSegmentDescriptor):
            raise TypeError("real flat target requires a validated descriptor")
        if type(transaction) is not FlatSegmentedKVTransaction:
            raise TypeError("real flat target requires its exact KV transaction")
        if transaction.descriptor != descriptor:
            raise ValueError("real flat target descriptor differs from its transaction")
        self.admission_validator(
            transaction.admission,
            transaction.owner,
            self,
            descriptor,
        )
        tokens = tuple(flat_inputs)
        if len(tokens) != descriptor.logical_rows or any(
            type(token) is not int or token < 0 for token in tokens
        ):
            raise ValueError("real flat target inputs must cover every logical row")
        self.validate_flat_owner(transaction.owner, descriptor)

        hidden = self.model.model.embed_tokens(
            mx.array([tokens], dtype=mx.int32)
        )
        taps = []
        offsets = transaction.lane_offsets
        for index, layer in enumerate(self.model.layers):
            attention = layer.self_attn
            normalized = _rowwise(layer.input_layernorm, hidden)
            rows = descriptor.logical_rows
            queries = standard._project(
                attention.q_proj, normalized, row_exact=True
            ).reshape(1, rows, attention.n_heads, attention.head_dim)
            keys = standard._project(
                attention.k_proj, normalized, row_exact=True
            ).reshape(1, rows, attention.n_kv_heads, attention.head_dim)
            values = standard._project(
                attention.v_proj, normalized, row_exact=True
            ).reshape(1, rows, attention.n_kv_heads, attention.head_dim)
            queries = _rowwise(attention.q_norm, queries)
            keys = _rowwise(attention.k_norm, keys)
            q_parts = []
            k_parts = []
            for lane, offset in enumerate(offsets):
                span = descriptor.lane_slice(lane)
                for local, position in enumerate(range(span.start, span.stop)):
                    query = queries[:, position : position + 1].transpose(0, 2, 1, 3)
                    key = keys[:, position : position + 1].transpose(0, 2, 1, 3)
                    q_parts.append(
                        attention.rope(query, offset=offset + local).transpose(
                            0, 2, 1, 3
                        )
                    )
                    k_parts.append(
                        attention.rope(key, offset=offset + local).transpose(
                            0, 2, 1, 3
                        )
                    )
            queries = mx.concatenate(q_parts, axis=1)
            keys = mx.concatenate(k_parts, axis=1)
            attended = transaction.attend_flat_layer(
                index,
                queries,
                keys,
                values,
                scale=attention.scale,
            )
            projected = standard._project(
                attention.o_proj,
                attended.reshape(1, rows, -1),
                row_exact=True,
            )
            residual = _rowwise_add(hidden, projected)
            post_attention = _rowwise(layer.post_attention_layernorm, residual)
            hidden = _rowwise_add(
                residual,
                _rowwise(layer.mlp, post_attention),
            )
            if index in self.capture_layers:
                taps.append(hidden)
        features = mx.concatenate(taps, axis=-1)
        normalized = _rowwise(self.model.model.norm, hidden)
        logits = self.model.logits(normalized, row_exact=True)
        mx.eval(logits, features)
        self.invocations += 1
        self.executed_rows += descriptor.logical_rows
        outputs = tuple(
            (logits[0, row], features[0, row])
            for row in range(descriptor.logical_rows)
        )
        return FlatSegmentExecution(
            outputs=outputs,
            geometry_sha256=descriptor.geometry_sha256,
            invocation_count=1,
        )

    def receipt(self) -> dict[str, object]:
        return {
            "route": "dense-qwen3-flat-segment-cpu-direct",
            "execution_scope": self.execution_scope,
            "implemented": True,
            "qualified": False,
            "selected": False,
            "observed_used": self.invocations > 0,
            "target_invocations": self.invocations,
            "logical_rows": self.executed_rows,
            "packed_matmul_authority": False,
            "projection_fidelity": "ordinary_b1s1_row_exact",
            "serving_wired": False,
        }


class DenseQwen3MetalFlatSegmentTarget(DenseQwen3FlatSegmentTarget):
    """Default-off exact-artifact Metal probe target; never a serving route."""

    execution_scope = METAL_EXECUTION_SCOPE
    target_protocol = METAL_TARGET_PROTOCOL
    owner_type = MetalFlatSegmentedKVRows
    admission_validator = staticmethod(validate_metal_flat_segment_admission)
    hard_limits = METAL_PROBE_HARD_LIMITS

    @property
    def source_hashes(self) -> dict[str, str]:
        return _source_hashes(_METAL_SOURCE_PATHS)

    def ast_flat_segment_device_attestation(self) -> dict[str, object]:
        return dense_qwen3_metal_device_attestation()

    def receipt(self) -> dict[str, object]:
        receipt = super().receipt()
        attestation = self.ast_flat_segment_device_attestation()
        receipt.update(
            {
                "route": "dense-qwen3-flat-segment-metal-direct-probe",
                "device_backend": "metal",
                "device_attestation": attestation,
                "device_admitted": attestation["ready"],
                "direct_probe_only": True,
                "performance_claim": False,
                "default": False,
            }
        )
        return receipt


__all__ = [
    "DenseQwen3FlatSegmentTarget",
    "DenseQwen3MetalFlatSegmentTarget",
    "dense_qwen3_metal_device_attestation",
]
