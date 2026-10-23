"""Strict artifact gate and ordinary serving adapter for Recursive Flint."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from ..sampling_defaults import GENERATION_CONFIG, SamplingDefaults, VendorSampling
from .artifact_paths import shard_within_artifact
from .external_draft_policy import ExternalDraftAdapterMixin

SOURCE_REPOSITORY = "RecursiveSI/flint"
SOURCE_REVISION = "2a247be0c0aec895ccb0ac7ca92229ec6df37288"
SOURCE_INDEX_SHA256 = "e5e9c492685f84c8d2d8369f36c74674b03410715f23f6179313f6f1feb90311"
SGLANG_WHEEL_SHA256 = "f9d1ba4ec8ddfb1d09d386e7995f56732431066587a73e0fb458d80de6e605cb"
CACHE_LAYOUT = "flint-full-swa-layer-segments-v1"


DESCRIPTOR = ModelDescriptor(
    model_type="flint",
    family="flint",
    variant="40b-a2.5b-ordinary",
    state_planes=frozenset(
        {
            StatePlane.ATTENTION_KV,
            StatePlane.RNG,
            StatePlane.TRANSCRIPT,
            StatePlane.GRAMMAR,
        }
    ),
    capabilities=frozenset(
        {
            Capability.TEXT,
            Capability.STREAMING,
            Capability.CONTINUOUS_BATCH,
            Capability.PREFIX_REUSE,
            Capability.APC_V2,
            Capability.LAYERED_CACHE,
            Capability.PROMPT_LOOKUP,
            Capability.TOOLS,
            Capability.REASONING,
        }
    ),
    cache_layout=CACHE_LAYOUT,
    metadata={
        "execution": "mlx2.adapters.flint.FlintAdapter",
        "qualification": "pending",
        "scope": "ordinary text decode; target-verified prompt lookup; prompt-lookup "
        "AST tree tiles on the generic attention target; model-selected optional "
        "reasoning and tools",
    },
)

SAMPLING = VendorSampling.single(
    SamplingDefaults(
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        source=GENERATION_CONFIG,
    ),
    model="RecursiveSI/flint",
)

_EXPECTED = {
    "architectures": ["FlintForCausalLM"],
    "model_type": "flint",
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "num_hidden_layers": 49,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "max_position_embeddings": 262144,
    "sliding_window": 513,
    "moe_intermediate_size": 512,
    "shared_expert_intermediate_size": 1024,
    "num_experts": 256,
    "num_experts_per_tok": 6,
    "vocab_size": 150144,
    "mlp_only_layers": [0, 1],
    "tie_word_embeddings": False,
    "attention_bias": False,
    "hidden_act": "silu",
    "norm_topk_prob": True,
    "router_score_function": "sigmoid",
    "softmax_type": "learnable",
    "rope_theta": 10000.0,
}
_FULL_LAYERS = (0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 48)
_LAYER_TYPES = [
    "full_attention" if index in _FULL_LAYERS else "sliding_attention"
    for index in range(49)
]


def _json(path: Path) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r} in {path.name}")
            result[key] = value
        return result

    value = json.loads(path.read_text(), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must be an object")
    return value


def _quantization(config: dict, weights: dict) -> dict | None:
    quant = config.get("quantization")
    if quant is None:
        if any(name.endswith((".scales", ".biases")) for name in weights):
            raise ValueError("Flint packed weights require quantization metadata")
        return None
    if not isinstance(quant, dict):
        raise TypeError("Flint quantization must be an object")
    base = {name: quant.get(name) for name in ("group_size", "bits", "mode")}
    if (
        base["group_size"] not in (32, 64, 128)
        or base["bits"] not in (2, 3, 4, 5, 6, 8)
        or base["mode"] != "affine"
    ):
        raise ValueError("unsupported Flint affine quantization default")
    for name, value in quant.items():
        if name in base:
            continue
        if not isinstance(value, dict):
            raise TypeError(f"Flint quantization override is not an object: {name}")
        merged = {**base, **value}
        if (
            merged["group_size"] not in (32, 64, 128)
            or merged["bits"] not in (2, 3, 4, 5, 6, 8)
            or merged["mode"] != "affine"
        ):
            raise ValueError(f"unsupported Flint quantization override: {name}")
    return quant


UPSTREAM_INT4_MANIFEST = "flint-int4.json"
UPSTREAM_INT4_REPOSITORY = "RecursiveSI/flint-int4"
UPSTREAM_INT4_REVISION = "3b93f9053b530488405ae9fca51d4900d6ce21f0"
UPSTREAM_INT4_FORMAT = "mlx-affine-int4"
_SAFETENSORS_BYTES = {"U8": 1, "U32": 4}


def _shard_headers(path: Path, names) -> dict:
    """{tensor: (dtype, shape, shard)} from safetensors headers only."""
    import struct

    tensors = {}
    for name in names:
        with open(path / name, "rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(size))
        metadata = header.pop("__metadata__", None) or {}
        if metadata.get("format") != "mlx":
            raise ValueError(f"Flint upstream INT4 shard {name} is not an MLX export")
        for tensor, entry in header.items():
            if tensor in tensors:
                raise ValueError(f"Flint tensor {tensor} appears in two shards")
            tensors[tensor] = (entry["dtype"], tuple(entry["shape"]), name)
    return tensors


def _upstream_int4_layout(path: Path, config: dict) -> tuple[dict, dict]:
    """Weight map and quantization law for ``RecursiveSI/flint-int4``.

    The upstream MLX export (``RecursiveSI/flint-int4@3b93f905``) ships no
    safetensors index, stores packed weights as ``uint8`` rows (the same
    little-endian nibble order as MLX's ``uint32`` packing; viewed back at
    load) and names its quantization without ``mode`` and without the INT8
    embedding override, which ``config.mlx_export`` records instead.  Accept
    exactly that layout, fail closed otherwise: the manifest must name the
    format and every shard, config.json must match its manifest digest, and
    every quantized tensor's packed width must imply the declared bits.
    """
    manifest = _json(path / UPSTREAM_INT4_MANIFEST)
    export = config.get("mlx_export")
    if manifest.get("weight_format") != UPSTREAM_INT4_FORMAT or not isinstance(export, dict):
        raise ValueError("Flint artifact has no safetensors index and is not the upstream INT4 export")
    declared = manifest.get("checkpoint_files") or {}
    shards = sorted(name for name in declared if name.endswith(".safetensors"))
    present = sorted(item.name for item in path.glob("*.safetensors"))
    if not shards or shards != present:
        raise ValueError("Flint upstream INT4 shards differ from its manifest")
    if hashlib.sha256((path / "config.json").read_bytes()).hexdigest() != declared.get("config.json"):
        raise ValueError("Flint upstream INT4 config.json differs from its manifest digest")
    quant = config.get("quantization") or {}
    group, bits = quant.get("group_size"), quant.get("bits")
    if quant.get("mode", "affine") != "affine" or set(quant) - {"group_size", "bits", "mode"}:
        raise ValueError("Flint upstream INT4 quantization is not the uniform affine export")
    if bits != export.get("bits") or bits != export.get("routed_expert_bits"):
        raise ValueError("Flint upstream INT4 bits disagree with mlx_export")
    tensors = _shard_headers(path, shards)
    law = {"group_size": group, "bits": bits, "mode": "affine"}
    for name, (dtype, shape, _) in tensors.items():
        if not name.endswith(".scales"):
            continue
        module = name[: -len(".scales")]
        weight = tensors.get(module + ".weight")
        if weight is None or weight[0] not in _SAFETENSORS_BYTES or module + ".biases" not in tensors:
            raise ValueError(f"Flint upstream INT4 tensor {module} is not affine packed")
        packed_bits = weight[1][-1] * _SAFETENSORS_BYTES[weight[0]] * 8
        implied = packed_bits // (shape[-1] * group)
        if implied * shape[-1] * group != packed_bits:
            raise ValueError(f"Flint upstream INT4 tensor {module} has a ragged packing")
        expected = export.get("embedding_bits") if module == "model.embed_tokens" else bits
        if implied != expected:
            raise ValueError(
                f"Flint upstream INT4 tensor {module} packs {implied}-bit values, declared {expected}"
            )
        if implied != bits:
            law[module] = {"group_size": group, "bits": implied, "mode": "affine"}
    return {name: entry[2] for name, entry in tensors.items()}, law


def inspect_artifact(model_path: str | Path) -> dict:
    path = Path(model_path).expanduser().resolve()
    config = _json(path / "config.json")
    if any(config.get(key) != value for key, value in _EXPECTED.items()):
        raise ValueError("Flint topology differs from the pinned 40B-A2.5B release")
    if config.get("layer_types") != _LAYER_TYPES:
        raise ValueError("Flint full/sliding attention order differs")
    if config.get("no_rope_layer_indices") != list(_FULL_LAYERS):
        raise ValueError("Flint RoPE-free layer order differs")
    if config.get("use_sliding_window") is not True:
        raise ValueError("Flint sliding attention must be enabled")
    if config.get("attention_dropout") not in (0, 0.0):
        raise ValueError("Flint attention dropout must be zero at inference")
    if config.get("rms_norm_eps") != 1e-6:
        raise ValueError("Flint RMSNorm epsilon differs")
    index_path = path / "model.safetensors.index.json"
    upstream_law = None
    if not index_path.is_file() and (path / UPSTREAM_INT4_MANIFEST).is_file():
        weights, upstream_law = _upstream_int4_layout(path, config)
    else:
        index = _json(index_path)
        weights = index.get("weight_map")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("Flint weight index is empty")
    files = []
    for name in sorted(set(weights.values())):
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not name.endswith(".safetensors")
        ):
            raise ValueError("Flint shard path is unsafe")
        shard = path / name
        if not shard.is_file() or not shard_within_artifact(path, shard.resolve()):
            raise ValueError(f"Flint shard is missing or escaped: {name}")
        stat = shard.stat()
        files.append((name, stat.st_size, stat.st_mtime_ns))

    required = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    for layer in range(49):
        prefix = f"model.layers.{layer}."
        required.update(
            prefix + suffix
            for suffix in (
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
                "self_attn.o_proj.weight",
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
                "self_attn.softmax_offset",
            )
        )
        if layer < 2:
            required.update(
                prefix + f"mlp.{projection}_proj.weight"
                for projection in ("gate", "up", "down")
            )
        else:
            required.update(
                {
                    prefix + "mlp.gate.weight",
                    prefix + "mlp.shared_expert.gate_proj.weight",
                    prefix + "mlp.shared_expert.up_proj.weight",
                    prefix + "mlp.shared_expert.down_proj.weight",
                }
            )
            source = prefix + "mlp.experts.0.gate_proj.weight"
            packed = prefix + "mlp.switch_mlp.gate_proj.weight"
            if source in weights:
                for expert in range(256):
                    required.update(
                        prefix + f"mlp.experts.{expert}.{projection}_proj.weight"
                        for projection in ("gate", "up", "down")
                    )
            elif packed in weights:
                required.update(
                    prefix + f"mlp.switch_mlp.{projection}_proj.weight"
                    for projection in ("gate", "up", "down")
                )
            else:
                raise ValueError(
                    f"Flint layer {layer} has no complete expert representation"
                )
    missing = required - weights.keys()
    if missing:
        raise ValueError(
            "Flint indexed tensor topology is incomplete: "
            + ", ".join(sorted(missing)[:5])
        )
    quant = _quantization(
        {**config, "quantization": upstream_law} if upstream_law else config, weights
    )
    digest = hashlib.sha256()
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        UPSTREAM_INT4_MANIFEST,
    ):
        item = path / name
        if item.is_file():
            digest.update(name.encode())
            digest.update(item.read_bytes())
    for record in files:
        digest.update(json.dumps(record).encode())
    return {
        "config": config,
        "weight_map": weights,
        "quantization": quant,
        "packed_uint8": upstream_law is not None,
        "source_layout": "packed"
        if any(".switch_mlp." in key for key in weights)
        else "hf-per-expert",
        "identity": {
            "path": str(path),
            "fingerprint": digest.hexdigest(),
            "files": files,
        },
        "source_repository": UPSTREAM_INT4_REPOSITORY if upstream_law else SOURCE_REPOSITORY,
        "source_revision": UPSTREAM_INT4_REVISION if upstream_law
        else config.get("mlx2_conversion", {}).get("source_revision", SOURCE_REVISION),
        "qualification": "pending",
    }


def _validate_execution_policy(execution_policy) -> dict:
    """Reject unsupported Flint routes before artifact or tensor access.

    The only external policy is the AST tree-tile route with the model-free
    prompt-lookup drafter (``--ast-drafter pld`` writes it); its knobs are
    validated here and its tile defaults filled.
    """
    if execution_policy in (None, {}):
        return {}
    if not isinstance(execution_policy, dict):
        raise TypeError("Flint execution policy must be an object")
    if "fused_decode" in execution_policy:
        from ..runtime.models.flint_fused import MODE
        if set(execution_policy) != {"fused_decode"}:
            raise ValueError("Flint fused decode cannot combine with another execution policy")
        if execution_policy["fused_decode"] not in ("off", MODE):
            raise ValueError(f"Flint fused_decode must be off or {MODE}")
        return {}
    from .qwen38_27b import PROMPT_LOOKUP_DRAFT_NAMES, _native_mtp_tree_policy

    if execution_policy.get("draft_model") in PROMPT_LOOKUP_DRAFT_NAMES:
        if execution_policy.get("ast_graft_prompt_lookup"):
            raise ValueError("Flint prompt-lookup tiles have no base drafter to graft onto")
        return _native_mtp_tree_policy(execution_policy)
    if execution_policy.get("draft_model"):
        raise ValueError(
            "Flint external drafting is implemented for the prompt-lookup "
            "tree-tile drafter only (draft_model \"prompt_lookup\")"
        )
    raise ValueError(
        "Flint supports ordinary decode, request-scoped prompt lookup and "
        "prompt-lookup AST tiles; it has no other execution-policy overrides"
    )


NAX_GATHER_ENV = "MLX2_MOE_NAX_GATHER"
ROUTED_DECODE_ENV = "MLX2_FLINT_ROUTED_DECODE"
# Adapter defaults for the two expert-kernel selections.  Both are
# bit-identical to the stock gathers (the NAX sorted gather by its per-kernel
# canary; the routed decode by its first-call bitwise probe), and both fall
# back to the stock op when not admitted.
#
# Measured on flint-2a247be0-oq4e (M5 Max, qualification/runs/
# flint-e2e-20261021/evidence/oq4e-kernels-ab-r2): the NAX gather prefilled a
# cold 4,096-token prompt at 2,631 vs 2,558 tok/s median (3/3 interleaved
# reps no slower, final logits bitwise equal), so it is the default; hosts
# without NAX fall back to the stock kernel.  Routed decode was bitwise equal
# on 64/64 rows in that direct loop (host-synchronized per token, where it
# showed no gain); served at B1 (oq4e-w2-r2: oq4e-rd-r2 between two ordinary
# controls) gate_up_down decoded 135.1/133.8 tok/s (code/chat) against
# 121.7/120.7 and 121.6/121.3, +11%, 144,384 routed calls, no fallback, and
# greedy outputs identical to both controls.  It serves one-token rows only
# (B>1 rows keep the stock gathers) and refuses itself for the process unless
# its first call proves bitwise, so it is the default.  Unqualified.
DEFAULT_NAX_GATHER = "gather"
DEFAULT_ROUTED_DECODE = "gate_up_down"


def kernel_defaults(environ=None) -> tuple[str, str]:
    """(NAX gather mode, routed-decode mode): operator env, else defaults."""
    from ..runtime.models import flint_routed_decode, moe_nax_gather

    environ = os.environ if environ is None else environ
    nax = environ.get(NAX_GATHER_ENV, "").strip().lower() or DEFAULT_NAX_GATHER
    if nax not in ("off", "gather"):
        raise ValueError(f"Flint supports {NAX_GATHER_ENV}=off or gather, not {nax!r}")
    routed = environ.get(ROUTED_DECODE_ENV, "").strip().lower() or DEFAULT_ROUTED_DECODE
    if routed not in flint_routed_decode.MODES:
        raise ValueError(
            f"{ROUTED_DECODE_ENV} must be one of {flint_routed_decode.MODES}, not {routed!r}"
        )
    assert nax in moe_nax_gather.MODES
    return nax, routed


class FlintAdapter(ExternalDraftAdapterMixin):
    default_route = "ordinary"
    # Best-route rule (Pierre 2026-10-21; geomean >= 1.05 over ordinary, no
    # cell < 0.85x).  flint-e2e-20261021 on flint-2a247be0-oq4e: latched
    # adaptive PLD 1.02-1.15x at B1 but 0.30-0.42x at B4 (latched lanes decode
    # one target row at a time; pld_batched_max_width=1), geomean 0.61; AST
    # prompt-lookup tiles 0.51.  Ordinary is the measured-best no-flag route;
    # --prompt-lookup and --ast stay explicit.  Unqualified.
    prompt_lookup_default = False
    # AST tree tiles on the attention-only target (no recurrent state):
    # ``--ast-drafter pld`` runs the external route with the prompt-lookup
    # tree drafter, verified by the generic AttentionTreeExecutor.
    ast_tree_tiles = True
    attention_tree_tiles = True
    EXTERNAL_DEFAULT_NUM_DRAFT = 15
    EXTERNAL_ROUTE_TAG = "external-flint-ast-tiles-v1"
    EXTERNAL_PROFILE = "flint-apcv2-ast-prompt-lookup"
    descriptor = DESCRIPTOR
    sampling_defaults = SAMPLING
    artifact_inspector = staticmethod(inspect_artifact)
    profile = "flint-apcv2-ordinary"
    tool_call_open_marker = "<tool_call>"

    @staticmethod
    def spomin_backend(model, prompt_cache):
        return None

    def approximate_kv_operations(self):
        return {}

    def profile_name(self, mtp):
        if mtp:
            raise ValueError("Flint has no native MTP route")
        return self.profile

    def execution_config(self, *, max_lanes, prefill_step):
        if getattr(self, "fused_decode", "off") != "off" and max_lanes != 1:
            raise ValueError("Flint fused decode candidate requires max_lanes=1")
        if getattr(self, "draft_model", None) is not None:
            return self._external_execution_config(
                max_lanes=max_lanes, prefill_step=prefill_step
            )
        return {
            "persistent": True,
            "num_draft": 0,
            "rate_gate": False,
            "prefill_step_size": prefill_step,
            "segment_aware_live_tip": False,
            "segment_aware_cohort_size": max_lanes,
        }

    def prefill_step_default(self):
        return 512

    def cache_budget(self, *, mtp):
        """Bound the actual full/sliding K/V, including APC restore windows."""
        if mtp:
            raise ValueError("Flint has no native MTP route")
        from ..runtime.memory_policy import SelfMTPLaneAdmissionController
        from .mlx_vlm_memory import (
            SlidingKVCacheBudget,
            _kv_item_bytes,
            _mlx2_rotating_checkpoint_policy,
        )

        config = self.config
        layers = config["layer_types"]
        copies, stride = _mlx2_rotating_checkpoint_policy()
        return SlidingKVCacheBudget(
            global_layers=layers.count("full_attention"),
            global_kv_heads=int(config["num_key_value_heads"]),
            global_head_dim=int(config["head_dim"]),
            sliding_layers=layers.count("sliding_attention"),
            sliding_kv_heads=int(config["num_key_value_heads"]),
            sliding_head_dim=int(config["head_dim"]),
            sliding_window=int(config["sliding_window"]),
            prefill_step=self.prefill_step_default(),
            checkpoint_copies=copies,
            checkpoint_stride=stride,
            item_bytes=_kv_item_bytes(config),
            # Preserve the existing generic workspace allowance. Only replace
            # its unrelated dense-model K/V growth estimate with our geometry.
            transient_gib_per_lane=SelfMTPLaneAdmissionController.K2_TRANSIENT_GIB_PER_LANE,
            transient_basis="provisional generic workspace; Flint not measured",
            schema="flint-full-sliding-cache-geometry-v1",
        )._validated()

    def __init__(self, model_path: str, *, execution_policy=None):
        from .process_globals import guarded_construction

        guarded_construction(
            self,
            lambda: self._init_flint(model_path, execution_policy=execution_policy),
        )

    def _init_flint(self, model_path: str, *, execution_policy=None):
        external_policy = _validate_execution_policy(execution_policy)
        self.fused_decode = (execution_policy or {}).get("fused_decode", "off")
        self.external_policy = {}
        self.draft_model = None
        artifact = inspect_artifact(model_path)
        self.identity = artifact["identity"]
        self.config = artifact["config"]
        if self.fused_decode != "off":
            from ..runtime.models.flint_fused import validate_config, validate_device
            validate_config(self.config)
            validate_device()
        self.layout = CACHE_LAYOUT
        from ..process_env import (
            PROCESS_NUMERICS,
            assert_process_profile_applied,
            clear_inherited_profile,
            require_process_numerics,
        )

        require_process_numerics("the Flint profile")
        clear_inherited_profile(())
        self.environment = {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            **PROCESS_NUMERICS,
        }
        os.environ.update(self.environment)
        assert_process_profile_applied("the Flint profile")
        from .process_globals import claim_moe

        # Expert-kernel selections (see ``kernel_defaults``); an explicit
        # operator value wins.  Both are bit-identical to the stock gathers.
        self.moe_nax_gather, self.routed_decode = kernel_defaults()
        claim_moe(self, "the Flint adapter", self.moe_nax_gather)

        import mlx.core as mx
        from mlx import nn
        from transformers import AutoTokenizer

        from ..runtime.models.flint import Model, ModelArgs
        from ..runtime.tokenizer_integrity import repair_loaded_tokenizer
        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper
        from ..runtime.ubc_evict import load_shards_evicting

        path = Path(self.identity["path"])
        self.model = Model(ModelArgs.from_dict(self.config))
        files = [path / name for name, _, _ in self.identity["files"]]
        weights = load_shards_evicting(files)
        if artifact.get("packed_uint8"):
            # Upstream INT4 rows are uint8 bytes in MLX's little-endian
            # nibble order: the same bits as the uint32 words MLX expects.
            weights = {
                name: value.view(mx.uint32)
                if value.dtype == mx.uint8 and name.endswith(".weight")
                and name[: -len(".weight")] + ".scales" in weights
                else value
                for name, value in weights.items()
            }
        weights = self.model.sanitize(weights, eager=True)
        quant = artifact["quantization"]
        if quant:
            keep = self.model.quant_predicate

            def predicate(name, module):
                override = quant.get(name)
                if isinstance(override, dict):
                    return override
                return (
                    hasattr(module, "to_quantized")
                    and keep(name, module)
                    and f"{name}.scales" in weights
                )

            nn.quantize(
                self.model,
                group_size=quant["group_size"],
                bits=quant["bits"],
                mode=quant["mode"],
                class_predicate=predicate,
            )
        self.model.load_weights(list(weights.items()), strict=True)
        self.model.eval()
        mx.eval(self.model.parameters())
        weights.clear()
        mx.clear_cache()
        for layer in self.model.layers:
            if hasattr(layer.mlp, "routed_decode"):
                layer.mlp.routed_decode = self.routed_decode
        if self.fused_decode != "off":
            from ..runtime.models.flint_fused import install
            install(self.model.model)
        tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        self.pretokenizer_receipt = repair_loaded_tokenizer(tokenizer, path)
        eos = self.config.get("eos_token_id", tokenizer.eos_token_id)
        eos_ids = eos if isinstance(eos, list) else [eos]
        self.tokenizer = TokenizerWrapper(
            tokenizer,
            detokenizer_class=BPEStreamingDetokenizer,
            eos_token_ids=eos_ids,
        )
        self.max_context = int(self.config["max_position_embeddings"])
        if external_policy:
            from ..runtime.drafters.prompt_lookup_draft import (
                load_prompt_lookup_drafter,
            )
            from .qwen38_27b import _prompt_lookup_record

            self.external_policy = external_policy
            record = _prompt_lookup_record(self.external_policy, self.identity["fingerprint"])
            self.identity = {
                **self.identity,
                "draft_revision": record["draft_revision"],
                "target_revision": record["target_revision"],
            }
            self._bind_external_drafter(record, load_prompt_lookup_drafter, self.descriptor)

    def prompt_tokens(self, request: dict) -> list[int]:
        if "messages" not in request:
            return self.tokenizer.encode(request["prompt"], add_special_tokens=False)
        messages = copy.deepcopy(request["messages"])
        for message in messages:
            for call in message.get("tool_calls", []):
                function = call["function"]
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(arguments, dict):
                    raise TypeError("Flint tool arguments must be a JSON object")
                function["arguments"] = arguments
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            tools=request.get("tools")
            if request.get("tool_choice") != "none"
            else None,
        )

    def render_prompt(self, request: dict) -> str:
        if "messages" not in request:
            return request["prompt"]
        messages = copy.deepcopy(request["messages"])
        for message in messages:
            for call in message.get("tool_calls", []):
                function = call["function"]
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                function["arguments"] = arguments
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            tools=request.get("tools")
            if request.get("tool_choice") != "none"
            else None,
        )

    @staticmethod
    def thinking_enabled(request: dict) -> bool:
        # Flint chooses per turn; never tell serving that the generation starts
        # inside a think block.  OutputParser still recognizes a leading one.
        return False

    def thinking_close_token_ids(self):
        ids = self.tokenizer.encode("</think>", add_special_tokens=False)
        return (int(ids[0]),) if len(ids) == 1 else None

    def output_parser(self, request):
        from ..output import OutputParser, constrained_tool_choice
        from ..runtime.tool_parsers.laguna import parse_tool_call

        return OutputParser(
            chat="messages" in request,
            thinking=False,
            tools=request.get("tools")
            if request.get("tool_choice") != "none"
            else None,
            parse_tool=parse_tool_call,
            stops=request.get("stop", ()),
            constrained_tools=constrained_tool_choice(request),
            parallel_tool_calls=request.get("parallel_tool_calls", True),
            tolerant_tool_markers=request.get("_tolerant_tool_markers", False),
        )

    def diagnostics(self):
        return {
            "architecture": "flint-sparse-moe-gqa",
            "cache_layout": self.layout,
            "attention": {
                "full_layers": 14,
                "sliding_layers": 35,
                "sliding_window": 513,
            },
            "experts": {"total": 256, "active": 6, "shared": 1, "packed": True},
            "ordinary_reference": True,
            "qualification": "pending",
            "adapter_implemented": True,
            "adapter_default_route": "ordinary",
            "target_forward_observed": getattr(self.model, "completed_forwards", 0) > 0,
            "capability_receipts": self.capability_receipts(),
            "moe_kernels": self.moe_kernel_status(),
            "fused_decode": self.fused_decode_status(),
        }

    def execution_numerics_contract(self):
        if getattr(self, "fused_decode", "off") == "off":
            return None
        from ..runtime.models.flint_fused import SCHEMA, shader_identity
        return {"operation": SCHEMA, "mode": self.fused_decode,
                "shader_sha256": shader_identity(), "scope": "B1 one-token decode",
                "prefill": "ordinary_reference", "qualified": False}

    def fused_decode_status(self):
        blocks = getattr(getattr(getattr(self, "model", None), "model", None), "fused_blocks", ())
        calls = sum(block.calls for block in blocks)
        return {"implemented": True, "qualified": False,
                "selected": getattr(self, "fused_decode", "off"),
                "observed_used": calls > 0, "layer_calls": calls,
                "scope": "M3 Pro; B1; uniform affine Q4g64 layers; BF16 KV",
                "numerics": self.execution_numerics_contract()}

    def moe_kernel_status(self):
        """Selected and observed expert-kernel use (counters are cumulative)."""
        from ..runtime.models import flint_routed_decode, moe_nax_gather

        blocks = [layer.mlp for layer in getattr(self.model, "layers", ()) or ()
                  if hasattr(layer.mlp, "routed_decode")]
        nax = moe_nax_gather.status()
        return {
            "nax_gather": {
                "selected": getattr(self, "moe_nax_gather", "off"),
                "mode": nax.get("mode"),
                "nax_host": nax.get("nax_host"),
                "calls": nax.get("calls"),
                "calls_by_format": nax.get("calls_by_format"),
                "fallbacks": nax.get("fallbacks"),
                "not_candidates": nax.get("not_candidates"),
            },
            "routed_decode": {
                "selected": getattr(self, "routed_decode", "off"),
                "calls": sum(block.routed_decode_calls for block in blocks),
                "fallbacks": sum(block.routed_decode_fallbacks for block in blocks),
                "last_fallback": next((block.routed_decode_last_fallback for block in blocks
                                       if block.routed_decode_last_fallback), None),
                "probe": flint_routed_decode.status(),
            },
        }

    def capability_receipts(self):
        """Adapter-local feature states; runtime receipts own observations."""
        return {
            "ordinary_reference": {
                "implemented": True, "qualified": False, "selected": None,
                "observed_used": False, "default": False,
                "selection_authority": "serving_route_receipt", "profile": self.profile,
            },
            "apcv2": {
                "implemented": True, "qualified": False, "selected": None,
                "observed_used": False, "selection_authority": "serving_route_receipt",
                "observation_authority": "serving_route_receipt", "cache_layout": self.layout,
            },
            "prompt_lookup": {
                "implemented": True, "qualified": False, "selected": None,
                "observed_used": False, "selection": "explicit_flag_only",
                "selection_authority": "serving_route_receipt",
                "observation_authority": "speculative_route_receipt",
            },
            "external_draft": {
                "implemented": True, "qualified": False,
                "selected": getattr(self, "draft_model", None) is not None,
                "observed_used": False, "drafters": ["prompt_lookup_tree_tiles"],
                "tree_target": "attention_tree_v1",
            },
            "approximate_kv": {
                "implemented": False, "qualified": False, "selected": False,
                "observed_used": False,
                "refusal": "quantized SDPA has no learned-sink path",
            },
        }

    def close(self):
        from .process_globals import release

        release(self)
        self.model = None
        self.tokenizer = None
