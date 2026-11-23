"""Strict artifact gate and serving adapter for Maxwell-V1-Flash (``maxwell_moe``).

Scope: the bf16 HF checkpoint, text with reasoning and qwen3-coder XML tool
calls, contexts up to ``max_position_embeddings`` (Block Sparse GQA evaluated
exactly past its 4,096-token dense regime), APCv2 prefix reuse over plain KV
planes (trunk GQA + per-layer indexer key), target-verified prompt lookup and
the checkpoint's native MTP layer.  The ordinary route stays the default; MTP
is selected with ``--native-mtp``.  Every route is unqualified.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from .artifact_paths import shard_within_artifact
from .mtp_depth_cap import validate_self_mtp_num_draft

SOURCE_REPOSITORY = "Maxwell-V1-Flash (local download; model card names no hub revision)"
# Must equal mlx2.runtime.models.maxwell_moe.CACHE_LAYOUT (checked at load).
CACHE_LAYOUT = "maxwell-mhc-bsgqa-kv-indexer-planes-v1"


def descriptor_for(*, has_mtp: bool) -> ModelDescriptor:
    capabilities = {
        Capability.TEXT, Capability.STREAMING, Capability.REASONING, Capability.TOOLS,
        Capability.GRAMMAR, Capability.CONTINUOUS_BATCH, Capability.PREFIX_REUSE,
        Capability.APC_V2, Capability.LAYERED_CACHE,
        # Every cache plane is a plain KVCache, so target-verified prompt
        # lookup rolls back exactly by trimming.
        Capability.PROMPT_LOOKUP,
    }
    planes = {StatePlane.ATTENTION_KV, StatePlane.RNG, StatePlane.TRANSCRIPT, StatePlane.GRAMMAR}
    if has_mtp:
        capabilities.add(Capability.MTP)
        planes.add(StatePlane.DRAFT)
    return ModelDescriptor(
        model_type="maxwell_moe",
        family="maxwell",
        variant="v1-flash-native-mtp" if has_mtp else "v1-flash-ordinary",
        state_planes=frozenset(planes),
        capabilities=frozenset(capabilities),
        cache_layout=CACHE_LAYOUT,
        metadata={
            "execution": "mlx2.adapters.maxwell_moe.MaxwellMoeAdapter",
            "qualification": "unqualified",
            "scope": "text, reasoning, qwen3-coder XML tools; Block Sparse GQA to 512K positions",
            "residual": "mhc-4-stream-sinkhorn",
            "ordinary_reference": "implemented",
            "native_mtp": "implemented" if has_mtp else "absent",
        },
    )


DESCRIPTOR = descriptor_for(has_mtp=False)

_EXPECTED = {
    "architectures": ["MaxwellMoeForCausalLM"],
    "model_type": "maxwell_moe",
    "attention_variant": "block_sparse_gqa",
    "hidden_size": 2048,
    "intermediate_size": 6144,
    "num_hidden_layers": 45,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "attention_head_size": 128,
    "moe_intermediate_size": 512,
    "shared_expert_intermediate_size": 1024,
    "num_experts": 256,
    "num_experts_per_tok": 12,
    "vocab_size": 200704,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-06,
    "rope_theta": 20000000,
    "tie_word_embeddings": False,
    "qk_layernorm": True,
    "enable_hyper_connections": True,
    "num_residual_streams": 4,
    "mhc_sinkhorn_iterations": 20,
    "moe_router_enable_expert_bias": True,
    "router_score_function": "sigmoid",
    "block_sparse_gqa_indexer_n_heads": 4,
    "block_sparse_gqa_indexer_head_dim": 128,
    "block_sparse_gqa_block_size": 128,
    "block_sparse_gqa_prefix_window_size": 1024,
    "block_sparse_gqa_local_window_size": 1024,
    "block_sparse_gqa_num_remote_blocks": 16,
}
_IDENTITY_FILES = ("config.json", "model.safetensors.index.json", "tokenizer.json",
                   "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
                   "special_tokens_map.json")


def _json(path: Path) -> dict:
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"duplicate key {key!r} in {path.name}")
            out[key] = value
        return out

    value = json.loads(path.read_text(), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must be an object")
    return value


def required_tensors(config: dict) -> set[str]:
    """Every trunk tensor (the native MTP layer is checked separately by ``mtp_tensors``)."""
    names = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight",
             "model.hc_head.hc_head_fn", "model.hc_head.hc_head_base", "model.hc_head.hc_head_scale"}
    dense = set(config["mlp_only_layers"])
    for layer in range(config["num_hidden_layers"]):
        p = f"model.layers.{layer}."
        names.update(p + s for s in (
            "input_layernorm.weight", "post_attention_layernorm.weight",
            "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
            "self_attn.o_proj.weight", "self_attn.q_layernorm.weight", "self_attn.k_layernorm.weight",
            "self_attn.indexer.linear_q_proj.weight", "self_attn.indexer.linear_k_proj.weight",
            "self_attn.indexer.q_layernorm.weight", "self_attn.indexer.k_layernorm.weight"))
        for hc in ("self_attn_hyper_connection", "mlp_hyper_connection"):
            names.update(f"{p}{hc}.{s}" for s in ("alpha_pre", "alpha_post", "alpha_res", "bias",
                                                  "mapping_proj.weight"))
        moe = p + "block_sparse_moe."
        if layer in dense:
            names.update(f"{moe}{w}.weight" for w in ("w1", "w2", "w3"))
        else:
            names.update({moe + "gate.weight", moe + "expert_bias"})
            names.update(f"{moe}shared_expert.{w}.weight" for w in ("w1", "w2", "w3"))
            names.update(f"{moe}experts.{e}.{w}.weight"
                         for e in range(config["num_experts"]) for w in ("w1", "w2", "w3"))
    return names


def mtp_tensors(config: dict) -> set[str]:
    """The one native MTP layer's tensors (``mtp_layers.0``)."""
    p = "mtp_layers.0."
    names = {p + s for s in ("enorm.weight", "hnorm.weight", "e_proj.weight", "h_proj.weight",
                             "final_layernorm.weight", "hc_head.hc_head_fn", "hc_head.hc_head_base",
                             "hc_head.hc_head_scale")}
    d = p + "decoder_layer."
    names.update(d + s for s in (
        "input_layernorm.weight", "post_attention_layernorm.weight",
        "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
        "self_attn.o_proj.weight", "self_attn.q_layernorm.weight", "self_attn.k_layernorm.weight"))
    for hc in ("self_attn_hyper_connection", "mlp_hyper_connection"):
        names.update(f"{d}{hc}.{s}" for s in ("alpha_pre", "alpha_post", "alpha_res", "bias",
                                              "mapping_proj.weight"))
    moe = d + "block_sparse_moe."
    names.update({moe + "gate.weight", moe + "expert_bias"})
    names.update(f"{moe}shared_expert.{w}.weight" for w in ("w1", "w2", "w3"))
    names.update(f"{moe}experts.{e}.{w}.weight" for e in range(config["num_experts"]) for w in ("w1", "w2", "w3"))
    return names


def inspect_artifact(model_path: str | Path) -> dict:
    """Topology, index closure and shard containment; identity from the indexed files only.

    Only shards the index names are read or fingerprinted, so exFAT AppleDouble
    ``._*.safetensors`` siblings are never touched; an index naming one is refused.
    """
    path = Path(model_path).expanduser().resolve()
    config = _json(path / "config.json")
    if any(config.get(k) != v for k, v in _EXPECTED.items()):
        bad = sorted(k for k, v in _EXPECTED.items() if config.get(k) != v)
        raise ValueError(f"Maxwell topology differs from the pinned V1-Flash release: {bad}")
    if config.get("block_sparse_gqa_layer_freq") not in (None, [1] * 45):
        raise ValueError("Maxwell requires Block Sparse GQA on every layer")
    if config.get("mlp_only_layers") != [0, 1, 2]:
        raise ValueError("Maxwell dense-layer order differs")
    if config.get("attention_dropout", 0.0) not in (0, 0.0) or config.get("rope_scaling") is not None:
        raise ValueError("Maxwell requires zero attention dropout and unscaled RoPE")
    if config.get("use_shared_expert_gate", False):
        raise ValueError("Maxwell shared expert must be ungated")
    quantization = _quantization(config)
    weights = _json(path / "model.safetensors.index.json").get("weight_map")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("Maxwell weight index is empty")
    files = []
    for name in sorted(set(weights.values())):
        if (not isinstance(name, str) or Path(name).name != name or name.startswith("._")
                or not name.endswith(".safetensors")):
            raise ValueError(f"Maxwell shard path is unsafe: {name!r}")
        shard = path / name
        if not shard.is_file() or not shard_within_artifact(path, shard.resolve()):
            raise ValueError(f"Maxwell shard is missing or escaped: {name}")
        st = shard.stat()
        files.append((name, st.st_size, st.st_mtime_ns))
    missing = required_tensors(config) - weights.keys()
    if missing:
        raise ValueError("Maxwell indexed tensor topology is incomplete: " + ", ".join(sorted(missing)[:5]))
    quantized_names = sorted(k.removesuffix(".scales") for k in weights if k.endswith(".scales"))
    if quantized_names and quantization is None:
        raise ValueError("Maxwell artifact stores quantized tensors without a quantization config")
    if quantization is not None:
        if not quantized_names:
            raise ValueError("Maxwell quantization config names no quantized tensors")
        stray = [n for n in quantized_names if f"{n}.weight" not in weights or f"{n}.biases" not in weights]
        if stray:
            raise ValueError(f"Maxwell quantized tensor lacks weight/biases: {stray[0]}")
        _require_fp32(path, weights, [k for k in weights if k.endswith(".expert_bias")])
    mtp_names = {k.removesuffix(".scales").removesuffix(".biases") + (".weight" if k.endswith((".scales", ".biases")) else "")
                 for k in weights if k.startswith("mtp_layers.")}
    declared = config.get("mtp_num_layers", 0)
    if declared not in (0, 1):
        raise ValueError("Maxwell declares an unsupported MTP layer count")
    if declared == 1 and mtp_names:
        expected = mtp_tensors(config)
        if mtp_names != expected:
            gap = sorted((expected - mtp_names) or (mtp_names - expected))
            raise ValueError(f"Maxwell MTP tensor set differs from one layer: {gap[0]}")
    elif mtp_names:
        raise ValueError("Maxwell MTP tensors present but mtp_num_layers is not 1")
    has_mtp = declared == 1 and bool(mtp_names)
    digest = hashlib.sha256()
    for name in _IDENTITY_FILES:
        item = path / name
        if item.is_file():
            digest.update(name.encode())
            digest.update(item.read_bytes())
    for record in files:
        digest.update(json.dumps(record).encode())
    return {
        "config": config,
        "weight_map": weights,
        "identity": {"path": str(path), "fingerprint": digest.hexdigest(), "files": files},
        "source_repository": SOURCE_REPOSITORY,
        "qualification": "unqualified",
        "has_mtp": has_mtp,
        "storage_format": "bf16" if quantization is None else "affine-quantized",
        "quantization": quantization,
        "quantized_names": quantized_names,
    }


def _quantization(config: dict):
    """The artifact's affine quantization block (oQ / mlx-lm layout), or None for bf16."""
    blocks = [config.get(k) for k in ("quantization", "quantization_config") if config.get(k) is not None]
    if not blocks:
        return None
    quant = blocks[0]
    if any(b != quant for b in blocks[1:]) or not isinstance(quant, dict):
        raise ValueError("Maxwell quantization and quantization_config disagree")
    if quant.get("mode", "affine") != "affine" or quant.get("bits") not in (2, 3, 4, 5, 6, 8) \
            or quant.get("group_size") not in (32, 64, 128):
        raise ValueError(f"Maxwell admits affine 2-8 bit quantization only, not {quant}")
    for key, value in quant.items():
        if isinstance(value, dict) and (
            value.get("mode", quant.get("mode", "affine")) != "affine"
            or value.get("bits", quant["bits"]) not in (2, 3, 4, 5, 6, 8)
            or value.get("group_size", quant["group_size"]) not in (32, 64, 128)
        ):
            raise ValueError(f"Maxwell module {key} has an unsupported quantization {value}")
    return quant


def _require_fp32(path: Path, weights: dict, names: list[str]):
    """Fail closed unless ``names`` are stored F32 (read from the shard headers only)."""
    import struct

    by_shard = {}
    for name in names:
        by_shard.setdefault(weights[name], []).append(name)
    for shard, wanted in by_shard.items():
        with (path / shard).open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(size))
        for name in wanted:
            if header.get(name, {}).get("dtype") != "F32":
                raise ValueError(f"Maxwell {name} must be stored F32 (its spread is below one bf16 step)")


def _messages(request):
    """Tool-call arguments as mappings (the template iterates ``arguments|items``)."""
    messages = copy.deepcopy(request["messages"])
    for message in messages:
        for call in message.get("tool_calls") or ():
            function = call.get("function", call)
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments) if arguments.strip() else {}
            if not isinstance(arguments, dict):
                raise TypeError("Maxwell tool arguments must be a JSON object")
            function["arguments"] = arguments
    return messages


class MaxwellMoeAdapter:
    default_route = "ordinary"
    prompt_lookup_default = False
    descriptor = DESCRIPTOR
    artifact_inspector = staticmethod(inspect_artifact)
    profile = "maxwell-apcv2-ordinary"
    tool_call_open_marker = "<tool_call>"
    # One trained MTP layer; deeper drafts chain its own multi-stream output.
    DEFAULT_NUM_DRAFT = 1

    def __init__(self, model_path: str, *, execution_policy=None, require_mtp: bool = False):
        from .process_globals import guarded_construction

        guarded_construction(self, lambda: self._init(model_path, execution_policy, require_mtp))

    def _init(self, model_path, execution_policy, require_mtp):
        if execution_policy is not None and not isinstance(execution_policy, dict):
            raise ValueError("Maxwell execution policy must be a JSON object")
        policy = dict(execution_policy or {})
        if set(policy) - {"num_draft"}:
            raise ValueError("Maxwell execution policy supports only num_draft")
        self._num_draft = validate_self_mtp_num_draft(policy.get("num_draft", self.DEFAULT_NUM_DRAFT))
        artifact = inspect_artifact(model_path)
        has_mtp = bool(artifact.get("has_mtp", False))
        if require_mtp and not has_mtp:
            raise ValueError("requested MTP requires the checkpoint's mtp_layers.0 tensors")
        self.identity, self.config, self.layout = artifact["identity"], artifact["config"], CACHE_LAYOUT
        self.descriptor = descriptor_for(has_mtp=has_mtp)
        from ..process_env import (
            PROCESS_NUMERICS,
            assert_process_profile_applied,
            clear_inherited_profile,
            require_process_numerics,
        )

        require_process_numerics("the Maxwell profile")
        clear_inherited_profile(())
        self.environment = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", **PROCESS_NUMERICS}
        os.environ.update(self.environment)
        assert_process_profile_applied("the Maxwell profile")
        from .process_globals import claim_stock_moe

        claim_stock_moe(self, "the Maxwell adapter")

        import mlx.core as mx
        from transformers import AutoTokenizer

        from ..runtime.models.maxwell_moe import CACHE_LAYOUT as MODEL_LAYOUT, FUSED_QKV, Model, ModelArgs
        from ..runtime.tokenizer_integrity import repair_loaded_tokenizer
        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper
        from ..runtime.ubc_evict import load_shards_evicting

        if MODEL_LAYOUT != CACHE_LAYOUT:
            raise ValueError("Maxwell model and adapter cache layouts disagree")
        path = Path(self.identity["path"])
        config = dict(self.config)
        config["mtp_num_layers"] = 1 if has_mtp else 0
        self.model = Model(ModelArgs.from_dict(config))
        self.storage_format = artifact["storage_format"]
        if artifact["quantization"] is not None:
            self.model.apply_quantization(artifact["quantization"], artifact["quantized_names"])
        weights = load_shards_evicting([path / name for name, _, _ in self.identity["files"]])
        weights = self.model.sanitize(weights, eager=True)
        self.model.load_weights(list(weights.items()), strict=True)
        weights.clear()
        if FUSED_QKV:
            self.model.fuse_projections()
        self.model.eval()
        mx.eval(self.model.parameters())
        mx.clear_cache()
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        self.pretokenizer_receipt = repair_loaded_tokenizer(tokenizer, path)
        generation = path / "generation_config.json"
        eos = (_json(generation).get("eos_token_id") if generation.is_file() else None) \
            or self.config.get("eos_token_id", tokenizer.eos_token_id)
        self.tokenizer = TokenizerWrapper(tokenizer, detokenizer_class=BPEStreamingDetokenizer,
                                          eos_token_ids=eos if isinstance(eos, list) else [eos])
        self.max_context = int(self.model.args.max_position_embeddings)

    # ------------------------------------------------------------ routes

    def profile_name(self, mtp):
        if mtp:
            if Capability.MTP not in self.descriptor.capabilities:
                raise ValueError("Maxwell native MTP requires the checkpoint's mtp_layers.0 tensors")
            return f"maxwell-apcv2-mtp{self._num_draft}"
        return self.profile

    def execution_config(self, *, max_lanes, prefill_step):
        return {"persistent": True,
                "num_draft": self._num_draft if Capability.MTP in self.descriptor.capabilities else 0,
                "backend": "ordinary", "rate_gate": False,
                "prefill_step_size": prefill_step, "segment_aware_live_tip": False,
                "segment_aware_cohort_size": max_lanes}

    def cache_budget(self, *, mtp):
        from .maxwell_memory import MaxwellCacheBudget

        return MaxwellCacheBudget.from_config(self.config, mtp=mtp)

    def prefill_step_default(self):
        return 512

    def approximate_kv_operations(self):
        return {}

    @staticmethod
    def spomin_backend(model, prompt_cache):
        return None

    # ------------------------------------------------------------ prompts

    @staticmethod
    def _template_kwargs(request):
        kwargs = dict(request.get("chat_template_kwargs") or {})
        if set(kwargs) - {"enable_thinking"}:
            raise ValueError("Maxwell chat template accepts only enable_thinking")
        # The server folds every public thinking control (enable_thinking, think,
        # chat_template_kwargs.enable_thinking, reasoning_effort "none") into the
        # top-level enable_thinking and removes chat_template_kwargs before the
        # adapter sees the request.
        flag = request.get("enable_thinking")
        if type(flag) is bool:
            kwargs["enable_thinking"] = flag
        return kwargs

    def _template(self, request, tokenize):
        from ..output import callable_tools

        return self.tokenizer.apply_chat_template(
            _messages(request), add_generation_prompt=True, tokenize=tokenize,
            tools=callable_tools(request), **self._template_kwargs(request))

    def prompt_tokens(self, request: dict) -> list[int]:
        if "messages" not in request:
            return self.tokenizer.encode(request["prompt"], add_special_tokens=False)
        return self._template(request, True)

    def render_prompt(self, request: dict) -> str:
        return request["prompt"] if "messages" not in request else self._template(request, False)

    @classmethod
    def thinking_enabled(cls, request: dict) -> bool:
        # The template opens <think> at the generation prompt unless enable_thinking is false.
        return "messages" in request and cls._template_kwargs(request).get("enable_thinking") is not False

    def thinking_close_token_ids(self):
        try:
            ids = list(self.tokenizer.encode("</think>", add_special_tokens=False))
        except (AttributeError, TypeError, ValueError):
            return None
        return (int(ids[0]),) if len(ids) == 1 else None

    def output_parser(self, request):
        from ..output import OutputParser, callable_tools, constrained_tool_choice
        from ..runtime.tool_parsers.qwen3_coder import parse_tool_call

        return OutputParser(
            chat="messages" in request, thinking=self.thinking_enabled(request),
            tools=callable_tools(request), parse_tool=parse_tool_call,
            stops=request.get("stop", ()), constrained_tools=constrained_tool_choice(request),
            parallel_tool_calls=request.get("parallel_tool_calls", True),
            tolerant_tool_markers=request.get("_tolerant_tool_markers", False),
            think_close_separator="\n")

    def tool_constraint(self, request):
        from ..output import constrained_tool_choice
        from ..runtime.tool_parsers.qwen3_coder import constrained_tool_grammar

        if not constrained_tool_choice(request):
            return None
        return constrained_tool_grammar(request["tools"], request["tool_choice"],
                                        parallel_tool_calls=request.get("parallel_tool_calls", True))

    # ------------------------------------------------------------ receipts

    def route_capability_contract(self):
        from ..runtime.models import maxwell_moe as M

        has_mtp = Capability.MTP in self.descriptor.capabilities
        unq = {"qualified": False, "selected": None, "selection_authority": "serving_route_receipt"}
        return {
            "schema": "mlx2.maxwell-route-capability-contract.v1",
            "family": self.descriptor.family,
            "variant": self.descriptor.variant,
            "default_route": "ordinary",
            "ordinary_reference_preserved": True,
            "qualification": "unqualified",
            "block_sparse_gqa": {"implemented": True, "dense_regime_tokens": self.model.args.dense_context_limit,
                                 "decode_gather_rows": M.SPARSE_GATHER_ROWS,
                                 "long_chunk_route": M.SPARSE_PREFILL,
                                 "routes": (f"dense-sdpa<={self.model.args.dense_context_limit}",
                                            f"gather<={M.SPARSE_GATHER_ROWS}-rows", M.SPARSE_PREFILL),
                                 "batch": "per-row (left and right padding)",
                                 "max_positions": self.max_context, **unq},
            "apcv2": {"implemented": True, "exact_only": True, "cache_layout": self.descriptor.cache_layout,
                      "state_planes": tuple(sorted(p.value for p in self.descriptor.state_planes)), **unq},
            "prompt_lookup": {"implemented": True, "target_verified": True,
                              "rollback": "exact-kv-and-indexer-plane-trim", **unq},
            "native_mtp": {"implemented": has_mtp, "artifact_bound": has_mtp, "num_draft": self._num_draft,
                           "trained_depth": 1, "deeper_depths": "chained extrapolation, verified exact",
                           "default_preferred": False, **unq},
            "tools": {"implemented": True, "format": "qwen3-coder-xml", **unq},
            "external_draft": {"implemented": False, "reason": "no revision-bound draft artifact"},
        }

    def capability_receipts(self):
        contract = self.route_capability_contract()
        return {name: {k: v for k, v in contract[name].items()}
                for name in ("block_sparse_gqa", "apcv2", "prompt_lookup", "native_mtp", "tools")} | {
            "ordinary_reference": {"implemented": True, "qualified": False, "selected": None,
                                   "observed_used": False, "selection_authority": "serving_route_receipt",
                                   "profile": self.profile}}

    def diagnostics(self):
        from ..runtime.models.maxwell_moe import maxwell_stats

        return {
            "architecture": "maxwell-mhc-sparse-moe-bsgqa",
            "storage_format": getattr(self, "storage_format", None),
            "cache_layout": self.layout,
            "experts": {"total": 256, "active": 12, "shared": 1, "packed": True},
            "ordinary_reference": True,
            "qualification": "unqualified",
            "max_context": getattr(self, "max_context", None),
            "mtp_head_present": Capability.MTP in self.descriptor.capabilities,
            "num_draft": getattr(self, "_num_draft", None),
            "maxwell_attention": maxwell_stats(),
            "target_forward_observed": getattr(getattr(self, "model", None), "completed_forwards", 0) > 0,
            "capability_receipts": self.capability_receipts(),
        }

    def close(self):
        from .process_globals import release

        release(self)
        self.model = None
        self.tokenizer = None
