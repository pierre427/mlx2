"""Default-off exact-artifact Qwen3.8 hybrid Metal AST probe.

The public input/output contract is logical ``flat_segmented`` geometry.  The
adapter deliberately implements it with one padded B4 call to the existing
Qwen hybrid row-exact executor and compacts only valid result rows.  It does
not claim a native varlen kernel: the physical layout is reported as
``padded_rectangular_valid_row_compaction``.  KV and GDN state live on private
copies until every
ordinary serial output/state/continuation oracle passes.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from mlx import nn

from ..runtime.flat_segmented_ast import (
    ABORT_LAW,
    COMMIT_LAW,
    DESCRIPTOR_SCHEMA,
    DEVICE_ATTESTATION_SCHEMA,
    EXECUTION_LAW,
    FIDELITY_EXACT,
    LAYOUT_CLASS,
    METAL_EXECUTION_SCOPE,
    METAL_TARGET_PROTOCOL,
    STATE_WRITE_LAW,
    TARGET_CAPABILITY_SCHEMA,
    TERMINAL_LAW,
    FlatSegmentDescriptor,
    FlatSegmentError,
    FlatSegmentExecution,
    FlatSegmentUnsupported,
    query_metal_flat_segment_admission,
    validate_metal_flat_segment_admission,
)
from ..runtime.flat_segmented_hybrid import (
    FlatSegmentedHybridTransaction,
    HybridLiveStateBoundary,
    MetalFlatSegmentedHybridRows,
    successor_bound_hybrid_state,
)
from ..runtime.flat_segmented_kv import METAL_PROBE_HARD_LIMITS
from ..runtime.models import qwen38_27b as qwen38
from ..runtime.models.cache import ArraysCache, KVCache
from .dense_qwen3_flat import dense_qwen3_metal_device_attestation

PROBE_SCHEMA = "mlx2.ast-qwen38-hybrid-metal-exact-artifact-probe.v1"
MODEL_MANIFEST_SCHEMA = "mlx2.qwen38-hybrid-flat-model-manifest.v1"
PHYSICAL_LAYOUT = "padded_rectangular_valid_row_compaction"
MAX_BOUND_PREFIX_ROWS = 8
QWEN38_27B_OQ4E_MTP_CHECKPOINT = (
    "Qwen3.8-27B-oQ4e-mtp@sha256:"
    "fb827b658dc3d26ac569efacefd0466dbf41cc1a5cb5e3a40fc1c6175298810f"
)
QWEN38_27B_OQ4E_MTP_REVIEWED_MANIFEST = {
    "config_sha256": "7a063b70e128d238784dce2c594cb7e0f714f44c4387975c99acc8fcf6511eb5",
    "index_sha256": "7f3e6f1baba9bec91b20f25a9a505a6e268377012d038f36febc3af6c478f625",
    "model_manifest_sha256": (
        "12c4d0eecbbcd7e3228dc86f051980fdcea3f1ea783f311acc807e6f00a3fc8a"
    ),
    "quantized_module_count": 498,
    "tensor_count": 2209,
}
# Exact executable-model manifest for the same reviewed checkpoint when the
# source-owned external-draft loader omits the unused one-layer MTP subtree.
# The artifact/config/header identity remains the MTP checkpoint above; only
# ``mtp_num_hidden_layers``, ``mtp_present`` and ``mtp_layers`` are zeroed.
QWEN38_27B_OQ4E_NO_MTP_MODEL_MANIFEST_SHA256 = (
    "40715500c0299eb8c04c2e198cd1bbb5a9a07c4e4ccff2e4919c958ff376d76d"
)
_PACKAGE = Path(__file__).resolve().parents[1]
_PACKAGE_PREFIX = "src/mlx2/"
_MISSING = object()
_SAFETENSORS_DTYPES = {
    "U32": "uint32",
    "BF16": "bfloat16",
    "F16": "float16",
    "F32": "float32",
}
_CANONICAL_MLX_DTYPES = (
    (mx.uint32, "uint32"),
    (mx.bfloat16, "bfloat16"),
    (mx.float16, "float16"),
    (mx.float32, "float32"),
)
_SOURCE_PATHS = (
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


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _source_hashes() -> dict[str, str]:
    return {
        name: hashlib.sha256(
            (_PACKAGE / name[len(_PACKAGE_PREFIX) :]).read_bytes()
        ).hexdigest()
        for name in _SOURCE_PATHS
    }


def _identity(identity: Mapping[str, str]) -> dict[str, str]:
    fields = {"source_revision", "build", "device", "artifact"}
    if not isinstance(identity, Mapping) or set(identity) != fields or any(
        not isinstance(identity[name], str) or not identity[name] for name in fields
    ):
        raise ValueError("Qwen3.8 hybrid probe identity fields differ")
    return {name: identity[name] for name in sorted(fields)}


def _sha256(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _array_exact(actual, expected) -> bool:
    return (
        isinstance(actual, mx.array)
        and isinstance(expected, mx.array)
        and actual.shape == expected.shape
        and actual.dtype == expected.dtype
        and bool(mx.array_equal(actual, expected).item())
    )


def _outputs_exact(actual, expected) -> bool:
    return len(actual) == len(expected) and all(
        _array_exact(a_logits, e_logits) and _array_exact(a_taps, e_taps)
        for (a_logits, a_taps), (e_logits, e_taps) in zip(
            actual, expected, strict=True
        )
    )


def _canonical_mlx_dtype(dtype) -> str:
    """Map live MLX dtypes to the safetensors manifest vocabulary.

    ``str(mx.uint32)`` is ``"mlx.core.uint32"`` while safetensors records
    ``"U32"``.  The model and header manifest builders must therefore share
    an explicit vocabulary rather than depending on either representation.
    Unknown dtypes fail closed instead of entering the reviewed digest.
    """

    for candidate, name in _CANONICAL_MLX_DTYPES:
        if dtype == candidate:
            return name
    raise ValueError(f"unsupported Qwen3.8 target dtype: {dtype!s}")


def _ordinary_forward_with_taps(model, inputs, cache, capture_layers):
    """Bypass the instance-installed target hook for the reference arm."""

    return type(model).forward_with_taps(model, inputs, cache, capture_layers)


def _quantized_manifest(model) -> tuple[tuple[object, ...], ...]:
    rows = []
    seen = set()
    for name, module in model.named_modules():
        if "mtp" in name.split("."):
            continue
        if id(module) in seen:
            continue
        seen.add(id(module))
        if isinstance(module, (nn.Linear, nn.Embedding)) and not isinstance(
            module, (nn.QuantizedLinear, nn.QuantizedEmbedding)
        ):
            raise ValueError(f"unquantized target tensor surface: {name}")
        if not isinstance(module, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
            continue
        fields = []
        for field in ("weight", "scales", "biases"):
            value = getattr(module, field, None)
            if not isinstance(value, mx.array):
                raise ValueError(f"quantized target field unavailable: {name}.{field}")
            fields.append(
                (field, tuple(value.shape), _canonical_mlx_dtype(value.dtype))
            )
        rows.append(
            (
                name,
                "quantized_linear"
                if isinstance(module, nn.QuantizedLinear)
                else "quantized_embedding",
                int(module.group_size),
                int(module.bits),
                getattr(module, "mode", None),
                tuple(fields),
            )
        )
    if not rows:
        raise ValueError("Qwen3.8 hybrid Metal probe requires a quantized artifact")
    return tuple(sorted(rows))


def qwen38_hybrid_model_manifest(model) -> dict[str, object]:
    """Describe the reviewed non-weight execution topology and quantized ABI.

    This helper discovers a candidate digest; it is not authority by itself.
    The target requires its caller to supply an independently reviewed digest
    and compares the live model against it before admission, begin, and execute.
    Exact tensor payload identity remains the separately supplied artifact hash.
    """

    if type(model) is not qwen38.Model:
        raise TypeError("Qwen3.8 manifest requires the exact model wrapper")
    args = model.language_model.args
    trunk = model.language_model.model
    return {
        "schema": MODEL_MANIFEST_SCHEMA,
        "wrapper": f"{type(model).__module__}.{type(model).__qualname__}",
        "text_args": dataclasses.asdict(args),
        "layer_topology": [
            {
                "index": index,
                "kind": "gdn" if layer.is_linear else "full_attention",
                "type": f"{type(layer).__module__}.{type(layer).__qualname__}",
            }
            for index, layer in enumerate(trunk.layers)
        ],
        "mtp_present": model.mtp is not None,
        "mtp_layers": 0 if model.mtp is None else len(model.mtp.layers),
        "quantized_modules": _quantized_manifest(model),
    }


def qwen38_hybrid_model_manifest_sha256(model) -> str:
    """Return the canonical digest of the live reviewed model surface."""

    return hashlib.sha256(_canonical(qwen38_hybrid_model_manifest(model))).hexdigest()


def _reviewed_qwen38_27b_artifact_manifest(
    model_path,
) -> tuple[dict[str, object], dict[str, object]]:
    """Build reviewed metadata and its canonical manifest from file headers."""

    root = Path(model_path).expanduser().resolve()
    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    try:
        config_bytes = config_path.read_bytes()
        index_bytes = index_path.read_bytes()
        config = json.loads(config_bytes)
        index = json.loads(index_bytes)
    except (OSError, ValueError, TypeError) as error:
        raise ValueError("Qwen3.8 reviewed artifact metadata is unavailable") from error
    headers = {}
    shard_names = sorted(set(index.get("weight_map", {}).values()))
    if not shard_names:
        raise ValueError("Qwen3.8 reviewed artifact has no indexed tensors")
    for shard_name in shard_names:
        shard = root / shard_name
        try:
            with shard.open("rb") as handle:
                header_length = struct.unpack("<Q", handle.read(8))[0]
                if header_length <= 0 or header_length > 64 * 1024 * 1024:
                    raise ValueError("safetensors header length differs")
                header = json.loads(handle.read(header_length))
        except (OSError, ValueError, TypeError, struct.error) as error:
            raise ValueError("Qwen3.8 reviewed tensor header is unavailable") from error
        for name, value in header.items():
            if name == "__metadata__":
                continue
            if name in headers or not isinstance(value, dict):
                raise ValueError("Qwen3.8 reviewed tensor headers differ")
            headers[name] = value
    if set(headers) != set(index["weight_map"]):
        raise ValueError("Qwen3.8 reviewed tensor index differs from headers")
    quantization = config.get("quantization")
    if not isinstance(quantization, dict):
        raise ValueError("Qwen3.8 reviewed quantization metadata is unavailable")
    modules = {}
    for name, value in headers.items():
        if name.endswith((".weight", ".scales", ".biases")):
            module_name, field = name.rsplit(".", 1)
            modules.setdefault(module_name, {})[field] = value
    quantized_modules = []
    for name, fields in modules.items():
        if "mtp" in name.split(".") or set(fields) != {
            "weight",
            "scales",
            "biases",
        }:
            continue
        override = quantization.get(name, {})
        if not isinstance(override, dict):
            raise ValueError("Qwen3.8 reviewed quantization override differs")
        settings = {
            key: override.get(key, quantization.get(key))
            for key in ("group_size", "bits", "mode")
        }
        if (
            type(settings["group_size"]) is not int
            or type(settings["bits"]) is not int
            or not isinstance(settings["mode"], str)
        ):
            raise ValueError("Qwen3.8 reviewed quantization settings differ")
        field_rows = []
        for field in ("weight", "scales", "biases"):
            value = fields[field]
            dtype = _SAFETENSORS_DTYPES.get(value.get("dtype"))
            shape = value.get("shape")
            if dtype is None or not isinstance(shape, list) or any(
                type(size) is not int or size < 0 for size in shape
            ):
                raise ValueError("Qwen3.8 reviewed tensor ABI differs")
            field_rows.append((field, tuple(shape), dtype))
        quantized_modules.append(
            (
                name,
                "quantized_embedding"
                if name.endswith("embed_tokens")
                else "quantized_linear",
                settings["group_size"],
                settings["bits"],
                settings["mode"],
                tuple(field_rows),
            )
        )
    text_args = qwen38.TextModelArgs.from_dict(config["text_config"])
    interval = int(text_args.full_attention_interval)
    manifest = {
        "schema": MODEL_MANIFEST_SCHEMA,
        "wrapper": "mlx2.runtime.models.qwen38_27b.Model",
        "text_args": dataclasses.asdict(text_args),
        "layer_topology": [
            {
                "index": index,
                "kind": "gdn" if (index + 1) % interval else "full_attention",
                "type": "mlx2.runtime.models.qwen38_27b.DecoderLayer",
            }
            for index in range(int(text_args.num_hidden_layers))
        ],
        "mtp_present": int(text_args.mtp_num_hidden_layers) > 0,
        "mtp_layers": int(text_args.mtp_num_hidden_layers),
        "quantized_modules": tuple(sorted(quantized_modules)),
    }
    result = {
        "checkpoint": QWEN38_27B_OQ4E_MTP_CHECKPOINT,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "model_manifest_sha256": hashlib.sha256(_canonical(manifest)).hexdigest(),
        "quantized_module_count": len(quantized_modules),
        "tensor_count": len(headers),
    }
    expected = {
        "checkpoint": QWEN38_27B_OQ4E_MTP_CHECKPOINT,
        **QWEN38_27B_OQ4E_MTP_REVIEWED_MANIFEST,
    }
    if result != expected:
        raise ValueError("Qwen3.8 artifact metadata differs from reviewed record")
    return result, manifest


def reviewed_qwen38_27b_artifact_metadata(model_path) -> dict[str, object]:
    """Validate the reviewed 27B config and tensor headers without loading tensors."""

    result, _ = _reviewed_qwen38_27b_artifact_manifest(model_path)
    return result


def _manifest_component_sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def diagnose_qwen38_hybrid_model_manifest(
    model,
    model_path,
    *,
    max_items: int = 16,
) -> dict[str, object]:
    """Compare a live model with reviewed headers using bounded metadata only.

    The report never grants authority and never includes tensor payloads.  It
    is intended for a failed-closed target-binding diagnosis: every component
    carries a digest, and bounded field/module lists identify representation
    drift without another speculative AST launch.
    """

    if type(max_items) is not int or not 1 <= max_items <= 64:
        raise ValueError("manifest diagnostic max_items must be in 1..64")
    metadata, reviewed = _reviewed_qwen38_27b_artifact_manifest(model_path)
    live = qwen38_hybrid_model_manifest(model)
    fields = (
        "schema",
        "wrapper",
        "text_args",
        "layer_topology",
        "mtp_present",
        "mtp_layers",
        "quantized_modules",
    )
    component_digests = {
        name: {
            "match": live[name] == reviewed[name],
            "live_sha256": _manifest_component_sha256(live[name]),
            "reviewed_sha256": _manifest_component_sha256(reviewed[name]),
        }
        for name in fields
    }
    live_args = live["text_args"]
    reviewed_args = reviewed["text_args"]
    differing_args = [
        name
        for name in sorted(set(live_args) | set(reviewed_args))
        if live_args.get(name, _MISSING) != reviewed_args.get(name, _MISSING)
    ]
    live_modules = {row[0]: row[1:] for row in live["quantized_modules"]}
    reviewed_modules = {
        row[0]: row[1:] for row in reviewed["quantized_modules"]
    }
    shared = sorted(set(live_modules) & set(reviewed_modules))
    differing_modules = [
        name for name in shared if live_modules[name] != reviewed_modules[name]
    ]
    missing_modules = sorted(set(reviewed_modules) - set(live_modules))
    extra_modules = sorted(set(live_modules) - set(reviewed_modules))

    def bounded(names: Sequence[str]) -> dict[str, object]:
        return {"count": len(names), "first": list(names[:max_items])}

    return {
        "schema": "mlx2.qwen38-hybrid-model-manifest-diagnostic.v1",
        "authoritative": False,
        "checkpoint": metadata["checkpoint"],
        "live_manifest_sha256": _manifest_component_sha256(live),
        "reviewed_manifest_sha256": metadata["model_manifest_sha256"],
        "match": live == reviewed,
        "component_digests": component_digests,
        "text_arg_differences": bounded(differing_args),
        "quantized_module_differences": {
            "missing": bounded(missing_modules),
            "extra": bounded(extra_modules),
            "changed": bounded(differing_modules),
        },
    }


class Qwen38HybridMetalFlatTarget:
    """One logical flat invocation over the installed exact hybrid executor."""

    execution_scope = METAL_EXECUTION_SCOPE
    target_protocol = METAL_TARGET_PROTOCOL
    hard_limits = METAL_PROBE_HARD_LIMITS

    def __init__(
        self,
        model,
        *,
        capture_layers,
        identity: Mapping[str, str],
        expected_model_manifest_sha256: str,
    ):
        if type(model) is not qwen38.Model:
            raise TypeError("hybrid Metal target requires exact Qwen3.8 wrapper")
        self._model = model
        self._capture_layers = tuple(capture_layers)
        self.identity = _identity(identity)
        self.invocations = 0
        self.executed_rows = 0
        self.physical_rows = 0
        self._quantized_manifest = _quantized_manifest(model)
        self._quantized_manifest_sha256 = hashlib.sha256(
            _canonical(self._quantized_manifest)
        ).hexdigest()
        self._mtp_present = model.mtp is not None
        self._mtp_layers = 0 if model.mtp is None else len(model.mtp.layers)
        self._expected_model_manifest_sha256 = _sha256(
            expected_model_manifest_sha256,
            "expected Qwen3.8 model manifest",
        )
        self._model_manifest_sha256 = qwen38_hybrid_model_manifest_sha256(model)
        if self._model_manifest_sha256 != self._expected_model_manifest_sha256:
            raise ValueError(
                "Qwen3.8 model differs from reviewed manifest: "
                f"live={self._model_manifest_sha256} "
                f"reviewed={self._expected_model_manifest_sha256}"
            )
        self._validate_topology()
        existing_route = model.__dict__.get("_mlx2_qwen_hybrid_row_exact")
        route_binding_names = (
            "progressive_external_target_rows",
            "progressive_external_verify_protocol",
            "progressive_external_exact_tile_widths",
            "progressive_external_exact_terminal_input_widths",
            "progressive_external_exact_max_lanes",
            "progressive_external_compact_exact_rows",
            "progressive_external_target_modules",
            "forward_with_taps",
        )
        prior_bindings = {
            name: model.__dict__.get(name, _MISSING)
            for name in route_binding_names
        }
        prior_attention_window = (
            None
            if existing_route is None
            else existing_route.attention_window_row_exact
        )
        installed_here = existing_route is None
        try:
            self._route = model.configure_target_verify_row_exact(True)
            self._route.set_attention_window_row_exact(False)
            self._validate_model()
        except BaseException:
            if installed_here:
                model.configure_target_verify_row_exact(False)
                for name, value in prior_bindings.items():
                    if value is _MISSING:
                        model.__dict__.pop(name, None)
                    else:
                        object.__setattr__(model, name, value)
            elif existing_route is not None:
                existing_route.set_attention_window_row_exact(prior_attention_window)
            raise

    @property
    def model(self):
        return self._model

    @property
    def capture_layers(self):
        return self._capture_layers

    def _validate_topology(self):
        model = self.model
        args = model.language_model.args
        trunk = model.language_model.model
        if (
            type(model) is not qwen38.Model
            or model.args.model_type != "qwen3_5"
            or args.num_experts != 0
            or trunk.pipeline_size != 1
            or not trunk.layers
            or not any(layer.is_linear for layer in trunk.layers)
            or not any(not layer.is_linear for layer in trunk.layers)
            or not self.capture_layers
            or self.capture_layers != tuple(sorted(set(self.capture_layers)))
            or self.capture_layers[0] < 0
            or self.capture_layers[-1] >= len(trunk.layers)
        ):
            raise ValueError("Qwen3.8 hybrid Metal topology differs")

    def _validate_model(self):
        self._validate_topology()
        model = self.model
        trunk = model.language_model.model
        if (
            model.__dict__.get("_mlx2_qwen_hybrid_row_exact") is not self._route
            or self._route.model is not model
            or self._route.static_refusals
            or self._route.attention_window_row_exact
            or any(
                type(module) is not swapped
                for module, _original, swapped in self._route._swapped
            )
            or len(self._route._gdn)
            != sum(layer.is_linear for layer in trunk.layers)
            or len(self._route._attention)
            != sum(not layer.is_linear for layer in trunk.layers)
            or _quantized_manifest(model) != self._quantized_manifest
            or (model.mtp is not None) != self._mtp_present
            or (0 if model.mtp is None else len(model.mtp.layers))
            != self._mtp_layers
            or qwen38_hybrid_model_manifest_sha256(model)
            != self._expected_model_manifest_sha256
        ):
            raise ValueError("Qwen3.8 hybrid row-exact or quantized binding drifted")

    def ast_flat_segment_execution_identity(self):
        self._validate_model()
        manifest_word = int(self._expected_model_manifest_sha256, 16)
        return (
            id(self.model),
            self.capture_layers
            + (manifest_word, int(self._mtp_present), self._mtp_layers),
        )

    def ast_flat_segment_device_attestation(self):
        return dense_qwen3_metal_device_attestation()

    @property
    def source_hashes(self):
        return _source_hashes()

    def make_bound_rows(self, prefixes: Sequence[Sequence[int]]):
        """Create artifact-bound hybrid rows from short ordinary B1 prefixes.

        This is the only supported owner construction path.  The resulting
        live boundaries are issued by this exact target and bind every KV and
        recurrent plane to its artifact revision, lane, and token position.
        """

        self._validate_model()
        prefixes = tuple(tuple(prefix) for prefix in prefixes)
        if (
            not prefixes
            or len(prefixes) > self.hard_limits["max_lanes"]
            or any(
                not prefix
                or len(prefix) > MAX_BOUND_PREFIX_ROWS
                or any(type(token) is not int or token < 0 for token in prefix)
                for prefix in prefixes
            )
        ):
            raise ValueError("Qwen3.8 bound-row prefixes exceed probe limits")
        rows = []
        boundaries = []
        for lane, prefix in enumerate(prefixes):
            caches = self.model.make_cache()
            mx.eval(
                self.model(
                    mx.array([prefix], dtype=mx.int32),
                    cache=caches,
                )
            )
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
        return MetalFlatSegmentedHybridRows(
            rows,
            identity=self.identity,
            boundary_checkpoints=boundaries,
            target_executor=self,
        )

    def validate_flat_owner(self, owner, descriptor):
        self._validate_model()
        if type(owner) is not MetalFlatSegmentedHybridRows:
            raise TypeError("Qwen3.8 hybrid target requires its exact hybrid owner")
        if not isinstance(descriptor, FlatSegmentDescriptor):
            raise TypeError("Qwen3.8 hybrid target requires a validated descriptor")
        owner._validate_rows(
            owner.rows,
            owner.boundaries,
            artifact=self.identity["artifact"],
            issuer_id=id(self),
        )
        if owner._target_executor_id != id(self) or owner.identity != self.identity:
            raise ValueError("Qwen3.8 hybrid owner target/artifact binding differs")
        layers = self.model.language_model.model.layers
        if descriptor.lane_count != len(owner.rows) or any(
            len(row) != len(layers) for row in owner.rows
        ):
            raise ValueError("Qwen3.8 hybrid owner layer geometry differs")
        args = self.model.language_model.args
        for lane, row in enumerate(owner.rows):
            kv_offsets = []
            for layer, cache in zip(layers, row, strict=True):
                if layer.is_linear:
                    if type(cache) is not ArraysCache:
                        raise TypeError("Qwen3.8 GDN layer lacks ArraysCache")
                    gdn = layer.linear_attn
                    conv, state = cache.cache
                    if (
                        conv.shape != (1, gdn.conv_kernel_size - 1, gdn.conv_dim)
                        or state.shape
                        != (
                            1,
                            gdn.num_v_heads,
                            gdn.head_v_dim,
                            gdn.head_k_dim,
                        )
                    ):
                        raise ValueError("Qwen3.8 GDN state geometry differs")
                else:
                    if type(cache) is not KVCache or cache.keys is None:
                        raise TypeError("Qwen3.8 attention layer lacks initialized KV")
                    attention = layer.self_attn
                    if (
                        cache.keys.shape != cache.values.shape
                        or cache.keys.shape[0] != 1
                        or cache.keys.shape[1] != attention.num_key_value_heads
                        or cache.keys.shape[-1] != attention.head_dim
                    ):
                        raise ValueError("Qwen3.8 attention state geometry differs")
                    kv_offsets.append(cache.offset)
            if len(set(kv_offsets)) != 1:
                raise ValueError("Qwen3.8 attention offsets differ within a lane")
            if kv_offsets[0] + descriptor.query_lengths[lane] > (
                args.max_position_embeddings
            ):
                raise ValueError("Qwen3.8 hybrid segment exceeds context capacity")

    def ast_flat_segment_target_contract(self):
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
            "hard_limits": dict(self.hard_limits),
            "source_hashes": self.source_hashes,
            "identity": dict(self.identity),
        }

    def execute_flat_segments(self, flat_inputs, descriptor, transaction):
        self._validate_model()
        if type(transaction) is not FlatSegmentedHybridTransaction:
            raise TypeError("Qwen3.8 hybrid target requires its exact transaction")
        if transaction.descriptor != descriptor:
            raise ValueError("Qwen3.8 hybrid target descriptor differs")
        validate_metal_flat_segment_admission(
            transaction.admission, transaction.owner, self, descriptor
        )
        tokens = tuple(flat_inputs)
        if len(tokens) != descriptor.logical_rows or any(
            type(token) is not int or token < 0 for token in tokens
        ):
            raise ValueError("Qwen3.8 flat inputs must cover every logical row")
        self.validate_flat_owner(transaction.owner, descriptor)
        width = descriptor.max_segment_rows
        padded = []
        for lane in range(descriptor.lane_count):
            values = list(tokens[descriptor.lane_slice(lane)])
            values.extend([0] * (width - len(values)))
            padded.append(values)
        from ..runtime import row_exact_verify as REV

        with REV.valid_rows(descriptor.query_lengths, width):
            result = self._route.execute(
                mx.array(padded, dtype=mx.int32),
                transaction.caches,
                self.capture_layers,
            )
        if not result.row_exact:
            raise FlatSegmentUnsupported(
                "Qwen3.8 hybrid row-exact executor declined: "
                + str(result.detail.get("reason"))
            )
        mx.eval(result.logits, result.features)
        outputs = []
        for lane, length in enumerate(descriptor.query_lengths):
            for position in range(length):
                outputs.append(
                    (
                        result.logits[lane, position],
                        result.features[lane, position],
                    )
                )
        self.invocations += 1
        self.executed_rows += descriptor.logical_rows
        self.physical_rows += descriptor.lane_count * width
        return FlatSegmentExecution(
            outputs=tuple(outputs),
            geometry_sha256=descriptor.geometry_sha256,
            invocation_count=1,
        )

    def receipt(self):
        attestation = self.ast_flat_segment_device_attestation()
        return {
            "route": "qwen38-hybrid-flat-segment-metal-direct-probe",
            "execution_scope": self.execution_scope,
            "logical_layout": LAYOUT_CLASS,
            "physical_layout": PHYSICAL_LAYOUT,
            "implemented": True,
            "qualified": False,
            "selected": False,
            "default": False,
            "observed_used": self.invocations > 0,
            "target_invocations": self.invocations,
            "logical_rows": self.executed_rows,
            "physical_rows": self.physical_rows,
            "padding_rows": self.physical_rows - self.executed_rows,
            "target_row_exact_installed": True,
            "attention_window_row_exact": False,
            "embedded_mtp_present": self._mtp_present,
            "embedded_mtp_layers": self._mtp_layers,
            "embedded_mtp_engaged": False,
            "embedded_mtp_tensor_authority": (
                "presence_and_layer_count_bound_target_tensor_subtree_excluded"
            ),
            "quantized_manifest_sha256": self._quantized_manifest_sha256,
            "model_manifest_sha256": self._expected_model_manifest_sha256,
            "bound_prefix_rows_per_lane_hard_limit": MAX_BOUND_PREFIX_ROWS,
            "state_boundary_authority": (
                "target_issued_artifact_revision_and_recurrent_state_digest"
            ),
            "device_attestation": attestation,
            "device_admitted": attestation["ready"],
            "serving_wired": False,
            "direct_probe_only": True,
            "performance_claim": False,
        }


def make_reviewed_qwen38_27b_metal_target(
    model,
    *,
    capture_layers,
    identity: Mapping[str, str],
    artifact_metadata: Mapping[str, object],
):
    """Construct the real target from the independently reviewed metadata record.

    A direct runner first calls :func:`reviewed_qwen38_27b_artifact_metadata`
    on the checkpoint directory (header reads only), loads the model once with
    the ordinary adapter, then passes that record and the exact checkpoint
    identity here.  It must never substitute ``qwen38_hybrid_model_manifest_sha256``
    of that live model as the expected authority.
    """

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
        raise ValueError("Qwen3.8 target lacks its reviewed artifact metadata")
    return Qwen38HybridMetalFlatTarget(
        model,
        capture_layers=capture_layers,
        identity=checked_identity,
        expected_model_manifest_sha256=QWEN38_27B_OQ4E_MTP_REVIEWED_MANIFEST[
            "model_manifest_sha256"
        ],
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
class Qwen38HybridMetalProbeResult:
    receipt: dict[str, object]


def run_qwen38_hybrid_metal_flat_probe(
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
    """Run one bounded correctness unit; publication is the final step."""

    if type(owner) is not MetalFlatSegmentedHybridRows:
        raise TypeError("Qwen3.8 hybrid probe requires its exact owner")
    if type(target) is not Qwen38HybridMetalFlatTarget:
        raise TypeError("Qwen3.8 hybrid probe requires its exact target")
    if target.invocations != 0:
        raise FlatSegmentUnsupported("Qwen3.8 hybrid probe target must be fresh")
    tokens = tuple(flat_tokens)
    accepted = tuple(accepted_lengths)
    continuation = tuple(continuation_tokens)
    if (
        not isinstance(descriptor, FlatSegmentDescriptor)
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
        raise FlatSegmentError("Qwen3.8 hybrid probe input geometry differs")
    admission = query_metal_flat_segment_admission(
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
    )
    if not admission.available:
        raise FlatSegmentUnsupported(
            "Qwen3.8 hybrid Metal admission declined: "
            + ",".join(admission.decline_reasons)
        )
    initial_boundary_positions = [
        boundary.position for boundary in owner.boundaries
    ]
    initial_boundary_generations = [
        boundary.generation for boundary in owner.boundaries
    ]
    expected_outputs, expected_state, expected_continuation = (
        _ordinary_serial_oracles(
            target, owner, descriptor, tokens, accepted, continuation
        )
    )
    transaction = None
    published = False
    try:
        transaction = owner.begin_flat(
            descriptor, target_executor=target, admission=admission
        )
        execution = target.execute_flat_segments(tokens, descriptor, transaction)
        if (
            type(execution.invocation_count) is not int
            or execution.invocation_count != 1
            or execution.geometry_sha256 != descriptor.geometry_sha256
            or target.invocations != 1
        ):
            raise FlatSegmentError("Qwen3.8 hybrid invocation contract differs")
        if not _outputs_exact(execution.outputs, expected_outputs):
            raise FlatSegmentError("Qwen3.8 hybrid full-logit/tap oracle differs")
        prepared = transaction.prepare_commit(accepted)
        if prepared.state != expected_state:
            raise FlatSegmentError("Qwen3.8 hybrid prepared state differs")
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
            raise FlatSegmentError("Qwen3.8 hybrid continuation oracle differs")
        transaction.publish_commit(prepared)
        published = True
    finally:
        if transaction is not None and not transaction.closed and not published:
            transaction.abort()
    identity = dict(expected_identity)
    return Qwen38HybridMetalProbeResult(
        receipt={
            "schema": PROBE_SCHEMA,
            "route": "qwen38-hybrid-flat-segment-metal-direct-probe",
            "execution_scope": METAL_EXECUTION_SCOPE,
            "checkpoint": identity["artifact"],
            "source_revision": identity["source_revision"],
            "build": identity["build"],
            "device": identity["device"],
            "device_attestation": admission.device_attestation,
            "geometry_sha256": descriptor.geometry_sha256,
            "lane_count": descriptor.lane_count,
            "logical_rows": descriptor.logical_rows,
            "physical_rows": (
                descriptor.lane_count * descriptor.max_segment_rows
            ),
            "padding_rows": (
                descriptor.lane_count * descriptor.max_segment_rows
                - descriptor.logical_rows
            ),
            "accepted_lengths": list(accepted),
            "initial_boundary_positions": initial_boundary_positions,
            "initial_boundary_generations": initial_boundary_generations,
            "bound_prefix_rows_per_lane_hard_limit": MAX_BOUND_PREFIX_ROWS,
            "state_boundary_authority": (
                "target_issued_artifact_revision_and_recurrent_state_digest"
            ),
            "target_invocations": 1,
            "logical_layout": LAYOUT_CLASS,
            "physical_layout": PHYSICAL_LAYOUT,
            "ordinary_serial_oracle": {
                "full_logits": "bit_exact",
                "capture_taps": "bit_exact",
                "kv_and_gdn_state": "bit_exact_prepublication",
                "continuation_logits_and_taps": "bit_exact",
            },
            "attention_window_row_exact": False,
            "embedded_mtp_present": target._mtp_present,
            "embedded_mtp_layers": target._mtp_layers,
            "embedded_mtp_engaged": False,
            "embedded_mtp_tensor_authority": (
                "presence_and_layer_count_bound_target_tensor_subtree_excluded"
            ),
            "model_manifest_sha256": target._expected_model_manifest_sha256,
            "packed_matmul_authority": False,
            "projection_fidelity": "ordinary_b1s1_row_exact",
            "direct_probe_only": True,
            "performance_claim": False,
            "serving_wired": False,
            "implemented": True,
            "qualified": False,
            "selected": False,
            "default": False,
            "observed_used": True,
            "admission_manifest_sha256": admission.receipt()["manifest_sha256"],
        }
    )


__all__ = [
    "QWEN38_27B_OQ4E_MTP_CHECKPOINT",
    "QWEN38_27B_OQ4E_MTP_REVIEWED_MANIFEST",
    "QWEN38_27B_OQ4E_NO_MTP_MODEL_MANIFEST_SHA256",
    "PHYSICAL_LAYOUT",
    "Qwen38HybridMetalFlatTarget",
    "Qwen38HybridMetalProbeResult",
    "diagnose_qwen38_hybrid_model_manifest",
    "make_reviewed_qwen38_27b_metal_target",
    "qwen38_hybrid_model_manifest",
    "reviewed_qwen38_27b_artifact_metadata",
    "run_qwen38_hybrid_metal_flat_probe",
    "qwen38_hybrid_model_manifest_sha256",
]
