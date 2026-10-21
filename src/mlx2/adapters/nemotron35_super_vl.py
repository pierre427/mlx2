"""Pinned Nemotron 3.5 Super VL text backbone, without remote-code execution.

The Omni source forwards text embeddings unchanged to Nemotron-H. Reuse that
runtime and omit media/MTP tensors before evaluation. Inspection is host-only.
See provenance/nemotron35-super-vl-text.json.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

from ..contracts import Capability, ModelDescriptor, StatePlane
from ..runtime.io_ownership import call_io, storage_owner
from ..sampling_defaults import GENERATION_CONFIG, SamplingDefaults, VendorSampling
from .artifact_paths import shard_within_artifact
from .dflash2 import _decode_unique_json, _read_safetensors_header, _source_binding
from .external_draft_policy import ExternalDraftAdapterMixin
from .nemotron3_super import Nemotron3SuperAdapter, configure_environment
from .nemotron3_super_memory import NemotronCacheBudget
from .nemotron35_super import (
    NEMOTRON_INT8_PREFILL_SCOPES,
    _moe_nax_gather_report,
    nemotron_int8_prefill_select,
    _split_moe_nax_gather,
    _split_super_execution_policy,
    _super_expert_gathers,
    _super_mamba_mixers,
    _super_moe_mixers,
)

SOURCE_REVISION = "533d66230b6e75518bf32d2d7a7b95a24c3dd517"
CONFIG_SHA256 = "d04bef96be42705e90a60034800b23cb9b22bfdae27cca7c49d6dafd53454901"
INDEX_SHA256 = "be78ca60affeba9bc90e6ba22cc387926e8a4f63fffc8672f0781bad3a91037e"
TOKENIZER_SHA256 = {
    "tokenizer.json": "d0432ad8b1c2984a2e2dca82e1933e3fd5628df6e95c51fba4c3ee9950dce299",
    "tokenizer_config.json": "89fa70d31734c0c75ffdcee4114449f8cf8abd64bc233d420b58e25763f91aba",
    "chat_template.jinja": "1369e90b19976f99ce2070eb21ade8fcb4af640f008e14e82af5c5e89a9cc816",
    "generation_config.json": "015bf4f765008e744f4d325f9320cfc2de35539f9cb972a505979471695a0feb",
}
TEXT_PREFIX = "language_model."
CACHE_LAYOUT = "nemotron35-super-vl-text-mamba-kv-affine4-protected8-v1"
DESCRIPTOR = ModelDescriptor(
    model_type="nemotron_h_omni",
    family="nemotron35-super-vl-120b-a12b",
    variant="text-affine4-protected8",
    cache_layout=CACHE_LAYOUT,
    capabilities=frozenset(
        {
            Capability.TEXT,
            Capability.STREAMING,
            Capability.TOOLS,
            Capability.REASONING,
            Capability.CONTINUOUS_BATCH,
            Capability.PREFIX_REUSE,
            Capability.APC_V2,
            Capability.LAYERED_CACHE,
            Capability.GRAMMAR,
            # Row-exact B1 verify (nemotron_h.Model.pld_verify_forward).
            Capability.PROMPT_LOOKUP,
        }
    ),
    state_planes=frozenset(
        {
            StatePlane.ATTENTION_KV,
            StatePlane.RECURRENT,
            StatePlane.RNG,
            StatePlane.TRANSCRIPT,
        }
    ),
    metadata={
        "execution": "mlx2.adapters.nemotron35_super_vl.Nemotron35SuperVLAdapter",
        "qualification": "pending",
        "scope": "text-only",
        "ordinary_reference": "implemented",
        "media": "unsupported",
        "native_mtp": "unsupported",
        "external_draft": "unsupported-unbound",
    },
)
SAMPLING = VendorSampling.single(
    SamplingDefaults(temperature=1.0, top_p=0.95, source=GENERATION_CONFIG),
    model="NVIDIA-Nemotron-3.5-Super-VL-120B-A12B",
)


@dataclass(frozen=True)
class SuperVLCacheBudget(NemotronCacheBudget):
    # Geometry is exact; this forward allowance remains unmeasured.
    transient_gib_per_lane: float = 4.0

    def as_dict(self):
        return {
            **super().as_dict(),
            "schema": "nemotron35-super-vl-text-cache-v1",
            "transient_bound": "conservative-unmeasured-allowance",
        }


def runtime_config(config):
    """Normalize only the source's tagged infinity and nested text namespace."""
    result = dict(config["llm_config"])
    result["num_hidden_layers"] = len(result["layers_block_type"])
    limit = result.get("time_step_limit") or [0.0, float("inf")]
    values = [
        float("inf") if x == {"__float__": "Infinity"} else float(x) for x in limit
    ]
    if (
        len(values) != 2
        or not 0 <= values[0] <= values[1]
        or not math.isfinite(values[0])
    ):
        raise ValueError("Invalid Nemotron time-step limit")
    result["time_step_limit"] = tuple(values)
    result["num_nextn_predict_layers"] = 0
    result["mtp_layers_block_type"] = None
    return result


def expected_text_tensors(config):
    """Raw converted tensor schema, before the runtime's Conv1d transpose."""
    c = runtime_config(config)
    h, v = c["hidden_size"], c["vocab_size"]
    shapes = {
        "backbone.embeddings.weight": [v, h],
        "backbone.norm_f.weight": [h],
        "lm_head.weight": [v, h],
    }
    for i, kind in enumerate(c["layers_block_type"]):
        p = f"backbone.layers.{i}."
        shapes[p + "norm.weight"] = [h]
        p += "mixer."
        if kind == "mamba":
            heads = c["mamba_num_heads"]
            inner = heads * c["mamba_head_dim"]
            conv = inner + 2 * c["n_groups"] * c["ssm_state_size"]
            shapes.update(
                {
                    p + "A_log": [heads],
                    p + "D": [heads],
                    p + "dt_bias": [heads],
                    p + "conv1d.weight": [conv, 1, c["conv_kernel"]],
                    p + "conv1d.bias": [conv],
                    p + "norm.weight": [inner],
                    p + "in_proj.weight": [inner + conv + heads, h],
                    p + "out_proj.weight": [h, inner],
                }
            )
        elif kind == "attention":
            q = c["num_attention_heads"] * c["head_dim"]
            kv = c["num_key_value_heads"] * c["head_dim"]
            shapes.update(
                {
                    p + "q_proj.weight": [q, h],
                    p + "k_proj.weight": [kv, h],
                    p + "v_proj.weight": [kv, h],
                    p + "o_proj.weight": [h, q],
                }
            )
        elif kind == "moe":
            n, latent = c["n_routed_experts"], c["moe_latent_size"]
            inter, shared = (
                c["moe_intermediate_size"],
                c["moe_shared_expert_intermediate_size"],
            )
            shapes.update(
                {
                    p + "gate.weight": [n, h],
                    p + "gate.e_score_correction_bias": [n],
                    p + "fc1_latent_proj.weight": [latent, h],
                    p + "fc2_latent_proj.weight": [h, latent],
                    p + "shared_experts.up_proj.weight": [shared, h],
                    p + "shared_experts.down_proj.weight": [h, shared],
                    p + "switch_mlp.fc1.weight": [n, inter, latent],
                    p + "switch_mlp.fc2.weight": [n, latent, inter],
                }
            )
        else:
            raise ValueError(f"Unsupported Nemotron layer kind: {kind}")
    result = {}
    quant = config["quantization"]
    used = set()
    for name, shape in shapes.items():
        name = TEXT_PREFIX + name
        settings = (
            quant.get(name.removesuffix(".weight"))
            if name.endswith(".weight")
            else None
        )
        if settings is None:
            result[name] = (
                "F32" if name.endswith("e_score_correction_bias") else "BF16",
                shape,
            )
            continue
        bits, group = settings.get("bits"), settings.get("group_size")
        if (
            set(settings) != {"bits", "group_size", "mode"}
            or bits not in (4, 8)
            or group != 64
            or settings["mode"] != "affine"
            or len(shape) < 2
            or shape[-1] % group
        ):
            raise ValueError(f"Unsupported affine module recipe: {name}")
        used.add(name.removesuffix(".weight"))
        result[name] = ("U32", [*shape[:-1], shape[-1] * bits // 32])
        for suffix in ("scales", "biases"):
            result[name.removesuffix("weight") + suffix] = (
                "BF16",
                [*shape[:-1], shape[-1] // group],
            )
    ordinary_quant = {
        key
        for key in quant
        if key.startswith(TEXT_PREFIX) and not key.startswith(TEXT_PREFIX + "mtp.")
    }
    if used != ordinary_quant:
        raise ValueError("Quantization recipe names do not match ordinary text modules")
    return result


def _metadata_bytes(path, limit=4 << 20):
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(f"Oversized Nemotron metadata: {path.name}")
    return raw


def _inspect(path):
    path = Path(path).expanduser().resolve()
    raw_config = _metadata_bytes(path / "config.json")
    raw_index = _metadata_bytes(path / "model.safetensors.index.json")
    if (
        hashlib.sha256(raw_config).hexdigest() != CONFIG_SHA256
        or hashlib.sha256(raw_index).hexdigest() != INDEX_SHA256
    ):
        raise ValueError(
            "Nemotron 3.5 Super VL metadata differs from the pinned native export"
        )
    config = _decode_unique_json(raw_config, "config")
    weight_map = _decode_unique_json(raw_index, "index")["weight_map"]
    expected = expected_text_tensors(config)
    selected = {
        key
        for key in weight_map
        if key.startswith(TEXT_PREFIX) and not key.startswith(TEXT_PREFIX + "mtp.")
    }
    if selected != expected.keys():
        raise ValueError("Nemotron ordinary text tensor inventory differs")
    receipt = _decode_unique_json(
        _metadata_bytes(path / "mlx2-conversion.json"), "conversion"
    )
    if (
        receipt.get("schema") != "mlx2.nemotron35-super-vl-conversion.v1"
        or receipt.get("recipe") != "nemotron35-super-vl-affine-experts4-protected8-v1"
        or receipt.get("source_revision") != SOURCE_REVISION
        or receipt.get("output_config_sha256") != CONFIG_SHA256
        or receipt.get("output_index_sha256") != INDEX_SHA256
    ):
        raise ValueError("Nemotron conversion identity mismatch")
    digest = hashlib.sha256(raw_config + raw_index)
    for name, expected_hash in TOKENIZER_SHA256.items():
        raw = _metadata_bytes(path / name, 64 << 20)
        if (
            hashlib.sha256(raw).hexdigest() != expected_hash
            or receipt["metadata_sha256"].get(name) != expected_hash
        ):
            raise ValueError(f"Converted tokenizer metadata mismatch: {name}")
        digest.update(raw)
    outputs = {record["file"]: record["bytes"] for record in receipt["output_shards"]}
    if set(outputs) != set(weight_map.values()):
        raise ValueError("Conversion shard inventory mismatch")
    records, text_bytes, excluded_bytes = [], 0, 0
    dtype_bytes = {"BF16": 2, "F32": 4, "U32": 4, "I64": 8}
    for name in sorted(outputs):
        file = (path / name).resolve()
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or not shard_within_artifact(path, file)
        ):
            raise ValueError("Unsafe Nemotron shard path")
        binding = _source_binding(file)
        raw, header, size = _read_safetensors_header(file)
        tensors = {key: value for key, value in header.items() if key != "__metadata__"}
        if tensors.keys() != {
            key for key, shard in weight_map.items() if shard == name
        }:
            raise ValueError(f"Shard index/header mismatch: {name}")
        ranges = []
        for key, value in tensors.items():
            shape, dtype, span = value["shape"], value["dtype"], value["data_offsets"]
            if (
                not isinstance(shape, list)
                or any(type(x) is not int or x < 0 for x in shape)
                or dtype not in dtype_bytes
                or len(span) != 2
                or any(type(x) is not int for x in span)
                or not 0 <= span[0] <= span[1] <= size
                or math.prod(shape) * dtype_bytes[dtype] != span[1] - span[0]
            ):
                raise ValueError(f"Invalid tensor extent: {key}")
            if key in expected and (dtype, shape) != expected[key]:
                raise ValueError(f"Unexpected tensor dtype/shape: {key}")
            ranges.append(span)
            if key in selected:
                text_bytes += span[1] - span[0]
            else:
                excluded_bytes += span[1] - span[0]
        end = 0
        for start, stop in sorted(ranges):
            if start != end:
                raise ValueError(f"Overlapping or incomplete tensor payload: {name}")
            end = stop
        if (
            end != size
            or binding != _source_binding(file)
            or file.stat().st_size != outputs[name]
        ):
            raise ValueError(f"Incomplete or changed shard: {name}")
        records.append((name, binding))
        digest.update(name.encode())
        digest.update(raw)
        digest.update(json.dumps(binding).encode())
    if text_bytes + excluded_bytes != receipt["tensor_bytes"]:
        raise ValueError("Conversion tensor byte accounting differs")
    return {
        "config": config,
        "conversion": receipt,
        "text_config": runtime_config(config),
        "weight_map": weight_map,
        "has_mtp": False,
        "identity": {
            "path": str(path),
            "fingerprint": digest.hexdigest(),
            "files": records,
        },
        "load_plan": {
            "scope": "ordinary-text-only",
            "text_tensor_count": len(selected),
            "excluded_tensor_count": len(weight_map) - len(selected),
            "resident_weight_bytes": text_bytes,
            "excluded_weight_bytes": excluded_bytes,
            "largest_source_shard_bytes": max(outputs.values()),
            "identity_validation": "metadata-header-and-file-binding; payload-not-rehashed",
            "qualified": False,
            "selected": False,
            "observed_used": False,
        },
    }


def inspect_artifact(model_path):
    return call_io("nemotron35-super-vl-inspect", _inspect, str(model_path),
                   owner=storage_owner("weights", model_path))


def text_weights(weights):
    """Loader callback runs before evaluation, so ignored media stays lazy."""
    return {
        key[len(TEXT_PREFIX) :]: value
        for key, value in weights.items()
        if key.startswith(TEXT_PREFIX) and not key.startswith(TEXT_PREFIX + "mtp.")
    }


def load_text_model(artifact):
    """Build on the model thread; no MLX work enters inventory workers."""
    import mlx.core as mx
    from mlx import nn

    from ..runtime.models.nemotron_h import Model, ModelArgs
    from ..runtime.ubc_evict import load_shards_evicting

    path = Path(artifact["identity"]["path"])
    model = Model(ModelArgs.from_dict(runtime_config(artifact["config"])))
    quant = artifact["config"]["quantization"]
    nn.quantize(
        model,
        class_predicate=lambda name, module: (
            dict(quant[TEXT_PREFIX + name])
            if TEXT_PREFIX + name in quant and hasattr(module, "to_quantized")
            else False
        ),
    )
    files = [
        path / name
        for name in sorted(
            {
                shard
                for key, shard in artifact["weight_map"].items()
                if key.startswith(TEXT_PREFIX)
                and not key.startswith(TEXT_PREFIX + "mtp.")
            }
        )
    ]
    weights = model.sanitize(load_shards_evicting(files, sanitize=text_weights))
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    weights.clear()
    mx.clear_cache()
    return model


class Nemotron35SuperVLAdapter(ExternalDraftAdapterMixin, Nemotron3SuperAdapter):
    descriptor = DESCRIPTOR
    sampling_defaults = SAMPLING
    artifact_inspector = staticmethod(inspect_artifact)
    default_route = "ordinary"
    default_mtp_ordinary_handoff_max_width = None
    think_close_separator = ""
    EXTERNAL_DEFAULT_NUM_DRAFT = 7
    EXTERNAL_ROUTE_TAG = "external-nemotron35-lilicorr-bf16-v1"
    EXTERNAL_PROFILE = "nemotron35-super-vl-apcv2-lilicorr-bf16"
    # int8 (W8A8) prefill: declared, off unless explicitly or record-selected.
    int8_prefill_supported = staticmethod(lambda: NEMOTRON_INT8_PREFILL_SCOPES)
    int8_prefill_select = staticmethod(nemotron_int8_prefill_select)

    def __init__(self, model_path, *, execution_policy=None):
        from .process_globals import guarded_construction

        guarded_construction(
            self, lambda: self._initialize(model_path, execution_policy)
        )

    def _initialize(self, model_path, execution_policy):
        execution_policy, self.moe_nax_gather = _split_moe_nax_gather(execution_policy)
        (
            expert_sort,
            mamba_conv,
            mamba_norm,
            mamba_ssm,
            mamba_projection,
            mamba_segment,
            relu2_down,
            external_policy,
        ) = _split_super_execution_policy(execution_policy)
        external = self._parse_external_policy(
            external_policy, family="Nemotron 3.5 Super VL"
        )
        self.expert_gather_sort = expert_sort
        self.mamba_conv = mamba_conv
        self.mamba_norm = mamba_norm
        self.mamba_ssm = mamba_ssm
        self.mamba_projection = mamba_projection
        self.mamba_segment = mamba_segment
        self.moe_relu2_down = relu2_down
        artifact = inspect_artifact(model_path)
        draft_record = None
        if external:
            from .lilicorr_nemotron import inspect_drafter

            draft_record = inspect_drafter(
                self.external_policy["draft_model"], target_artifact=artifact
            )
            self._check_num_draft(draft_record)
        self.identity, self.load_plan = artifact["identity"], artifact["load_plan"]
        self.layout, self._num_draft, self._has_mtp = CACHE_LAYOUT, 0, False
        self.environment = configure_environment()
        from .process_globals import claim_moe, claim_stock_moe

        if self.moe_nax_gather == "off":
            claim_stock_moe(self, "the Nemotron 3.5 Super VL text adapter")
        else:
            claim_moe(self, "the Nemotron 3.5 Super VL text adapter", self.moe_nax_gather)
        self.model = load_text_model(artifact)
        self._super_expert_gathers = _super_expert_gathers(self.model)
        self._super_mamba_mixers = _super_mamba_mixers(self.model)
        self._super_moe_mixers = _super_moe_mixers(self.model)
        if self.expert_gather_sort != "auto":
            for switch in self._super_expert_gathers:
                switch.set_sort_policy(self.expert_gather_sort)
            self.layout += ":expert-gather-unsorted-decode-v1"
        if self.mamba_conv != "auto":
            for mixer in self._super_mamba_mixers:
                mixer.set_conv_policy(self.mamba_conv)
            self.layout += (
                ":mamba-conv4-silu-short-checkpoints-v1"
                if self.mamba_conv == "fused_short"
                else (
                    ":mamba-conv4-silu-b1-compact-state-v2"
                    if self.mamba_conv == "fused_b1_compact"
                    else ":mamba-conv4-silu-b1-v1"
                )
            )
        if self.mamba_norm != "auto":
            for mixer in self._super_mamba_mixers:
                mixer.norm.set_policy(self.mamba_norm)
            self.layout += (
                ":mamba-gate-group-rms-short-v1"
                if self.mamba_norm == "fused_short"
                else ":mamba-gate-group-rms-b1-v1"
            )
        if self.mamba_ssm != "auto":
            for mixer in self._super_mamba_mixers:
                mixer.set_ssm_policy(self.mamba_ssm)
            self.layout += (
                ":mamba-ssm-short-dt-gate-v1"
                if self.mamba_ssm == "fused_short_gate"
                else ":mamba-ssm-short-dt-v1"
            )
        if self.mamba_projection != "auto":
            for mixer in self._super_mamba_mixers:
                mixer.set_projection_policy(self.mamba_projection)
            projection_layout = {
                "compact_padded": ":mamba-varlen-q8-projection-v1",
                "compact_row_exact": ":mamba-varlen-row-exact-q8-projection-v1",
                "adaptive_row_exact": ":mamba-varlen-adaptive-row-exact-q8-projection-v1",
                "short_row_exact": ":mamba-short-row-exact-q8-projection-v1",
                "native_qmm_residual_b1": ":mamba-native-qmm-residual-b1-v1",
            }
            self.layout += projection_layout[self.mamba_projection]
        if self.mamba_segment != "auto":
            self.model.set_short_segment_policy(self.mamba_segment)
            self.layout += (
                ":mamba-layerwise-short-exact-routed-batch-v1"
                if self.mamba_segment == "layerwise_short_exact_routed_batch"
                else ":mamba-layerwise-short-exact-v1"
            )
        if self.moe_relu2_down != "auto":
            for mixer in self._super_moe_mixers:
                mixer.set_relu2_down_policy(self.moe_relu2_down)
            self.layout += ":relu2-down-fp32-top22-b1-v1"
        # Reject changed files before making the adapter reachable.
        if inspect_artifact(model_path)["identity"] != self.identity:
            raise ValueError("Nemotron artifact changed during model load")
        from transformers import AutoTokenizer

        from ..runtime.tokenizer_integrity import repair_loaded_tokenizer
        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper

        path = Path(self.identity["path"])
        tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        self.pretokenizer_receipt = repair_loaded_tokenizer(tokenizer, path)
        self.tokenizer = TokenizerWrapper(
            tokenizer, detokenizer_class=BPEStreamingDetokenizer, eos_token_ids=[2, 11]
        )
        self.max_context = artifact["text_config"]["max_position_embeddings"]
        if draft_record is not None:
            from .lilicorr_nemotron import load_drafter

            contract = self.nemotron_lilicorr_target_contract(draft_record)

            def loader(record, target_model):
                return load_drafter(record, target_model, target_contract=contract)

            self._bind_external_drafter(draft_record, loader, DESCRIPTOR)
            self.descriptor = replace(
                self.descriptor,
                metadata={
                    **(self.descriptor.metadata or {}),
                    "external_draft": "implemented-unqualified-pinned-bf16",
                },
            )

    def nemotron_lilicorr_target_contract(self, draft_record):
        """Bind one inspected companion to this adapter's exact target API."""
        taps = tuple(draft_record["args"].target_layer_ids)
        text = draft_record["target"]["text_config"]
        if draft_record["target_fingerprint"] != self.identity["fingerprint"]:
            raise ValueError("Nemotron LiLiCorr target revision changed")
        return {
            "schema": "mlx2.nemotron-lilicorr-target.v1",
            "target_model_type": "nemotron_h_omni",
            "target_fingerprint": self.identity["fingerprint"],
            "target_layer_ids": taps,
            "target_feature_width": len(taps) * int(text["hidden_size"]),
            "capture": "post-layer-pre-final-norm",
            "ordinary_reference_preserved": True,
            "speculative_rollback": "exact",
            "owned_lm_head": True,
            "owned_final_norm": True,
            "qualified": False,
        }

    def profile_name(self, mtp):
        if mtp:
            raise ValueError("Nemotron 3.5 Super VL native MTP is not implemented")
        return "nemotron35-super-vl-affine4-protected8-apcv2-ordinary"

    def execution_config(self, *, max_lanes, prefill_step):
        if getattr(self, "draft_model", None) is not None:
            return self._external_execution_config(
                max_lanes=max_lanes, prefill_step=prefill_step
            )
        return super().execution_config(max_lanes=max_lanes, prefill_step=prefill_step)

    def exact_prefix_cascade_contract(self):
        # The inherited older-family primitive carries a different source
        # binding; do not advertise it for this artifact without that binding.
        return {
            "implemented": False,
            "qualified": False,
            "selected": False,
            "observed_used": False,
            "reason": "no artifact-bound cascade route",
        }

    @staticmethod
    def planning_cache_budget(config, *, mtp):
        if mtp:
            raise ValueError("Nemotron 3.5 Super VL native MTP is not implemented")
        c = runtime_config(config)
        c["hybrid_override_pattern"] = [
            {"mamba": "M", "attention": "*", "moe": "E"}[x]
            for x in c["layers_block_type"]
        ]
        return SuperVLCacheBudget.from_config(SimpleNamespace(**c), mtp=False)

    def cache_budget(self, *, mtp):
        if mtp:
            raise ValueError("Nemotron 3.5 Super VL native MTP is not implemented")
        return SuperVLCacheBudget.from_config(self.model.args)

    def route_capability_contract(self):
        external = getattr(self, "draft_model", None) is not None
        return {
            "schema": "mlx2.nemotron35-super-vl-text-route.v1",
            "scope": "text-only",
            "ordinary_reference_preserved": True,
            "qualification": "unqualified",
            "native_mtp": {"implemented": False},
            "media": {"implemented": False},
            "external_draft": {
                "implemented": True,
                "selected": external,
                "observed_used": False,
                "qualified": False,
                "scope": "pinned-bf16-owned-head-sliding8192"
                if external
                else "available-with-explicit-pinned-bf16-companion",
                "quantized_exports": "refused",
            },
            "apcv2": {"implemented": True, "exact_only": True, "qualified": False},
            "forward_memory_allowance": "4-GiB-per-lane-unmeasured",
        }

    def diagnostics(self):
        base = {
            "architecture": "nemotron-h-mamba-moe-gqa",
            "layout": self.layout,
            "qualification": "unqualified",
            "load_plan": self.load_plan,
            "route_capability_contract": self.route_capability_contract(),
        }
        gather_counts = self._sum_candidate_counts(
            self._super_expert_gathers, "sort_status"
        )
        conv_counts = self._sum_candidate_counts(
            self._super_mamba_mixers, "conv_status"
        )
        norm_counts = self._sum_candidate_counts(
            (mixer.norm for mixer in self._super_mamba_mixers), "status"
        )
        ssm_counts = self._sum_candidate_counts(self._super_mamba_mixers, "ssm_status")
        projection_counts = self._sum_candidate_counts(
            self._super_mamba_mixers, "projection_status"
        )
        relu2_counts = self._sum_candidate_counts(
            self._super_moe_mixers, "relu2_down_status"
        )
        return {
            **base,
            "moe_nax_gather": _moe_nax_gather_report(
                getattr(self, "moe_nax_gather", "off")
            ),
            "row_exact": (
                self.model.row_exact_status()
                if hasattr(getattr(self, "model", None), "row_exact_status")
                else None
            ),
            "expert_gather_sort": self._candidate_status(
                self.expert_gather_sort,
                gather_counts,
                ("unsorted_decode",),
                len(self._super_expert_gathers),
            ),
            "mamba_conv": self._candidate_status(
                self.mamba_conv,
                conv_counts,
                ("fused_b1", "fused_b1_compact", "fused_short"),
                len(self._super_mamba_mixers),
            ),
            "mamba_norm": self._candidate_status(
                self.mamba_norm,
                norm_counts,
                ("fused_b1", "fused_short"),
                len(self._super_mamba_mixers),
            ),
            "mamba_ssm": self._candidate_status(
                self.mamba_ssm,
                ssm_counts,
                ("fused_short", "fused_short_gate"),
                len(self._super_mamba_mixers),
            ),
            "mamba_projection": self._candidate_status(
                self.mamba_projection,
                projection_counts,
                (
                    "in_proj_compaction_calls",
                    "out_proj_compaction_calls",
                    "in_proj_short_kernel",
                    "out_proj_short_kernel",
                    "native_qmm_residual_b1",
                ),
                len(self._super_mamba_mixers),
            ),
            "mamba_segment": {
                **self._candidate_status(
                    self.mamba_segment,
                    self.model.short_segment_status()["counts"],
                    (
                        "layerwise_short_exact",
                        "layerwise_short_exact_routed_batch",
                    ),
                    1,
                ),
                "schema": "mlx2.nemotron35-super-mamba-segment.v1",
                "refusal_reasons": self.model.short_segment_status()["refusal_reasons"],
            },
            "moe_relu2_down": self._candidate_status(
                self.moe_relu2_down,
                relu2_counts,
                ("fused_b1",),
                len(self._super_moe_mixers),
            ),
        }

    @staticmethod
    def _sum_candidate_counts(modules, status_name):
        totals = {}
        for module in modules:
            status = getattr(module, status_name)()
            for name, count in status.get("counts", {}).items():
                totals[name] = totals.get(name, 0) + int(count)
        return totals

    @staticmethod
    def _candidate_status(policy, counts, observed_names, modules):
        return {
            "implemented": True,
            "qualified": False,
            "selected": policy != "auto",
            "observed_used": any(counts.get(name, 0) > 0 for name in observed_names),
            "policy": policy,
            "modules": modules,
            "counts": counts,
        }

    def close(self):
        self.draft_model = None
        super().close()
