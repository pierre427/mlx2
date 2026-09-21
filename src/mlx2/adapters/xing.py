"""CPU-safe Xing4.0-29B-A4B artifact inspection and serving adapter.

Xing4.0 is DeepSeek-V3-style MLA + noaux_tc MoE with a 4-stream mHC residual
and one embedded MTP layer.  Only the MLA latent is cross-token state (plain
``KVCache`` rows), so APCv2 prefix reuse, COW branches, rollback and the
segmented continuous-batching MTP path use the shared runtime unchanged; the
mHC streams exist only inside a forward pass.  Artifacts are produced by
``scripts/convert_xing4_0.py`` (sanitized module tree) and carry a fast
tokenizer built and parity-stamped by ``scripts/xing4_0_tokenizer.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from .mtp_depth_cap import validate_self_mtp_num_draft
from ..sampling_defaults import XING4_SAMPLING

CACHE_LAYOUT = "xing4-0-mla-latent-layer-segments-v1"
CONVERSION_LAYOUT = "xing4_0-sanitized-v1"
# Default off until the threshold-4 route is requalified after a7a8372 fixed
# required + parallel_tool_calls:false output. Operators may still select the
# handoff explicitly through the execution policy.
DEFAULT_MTP_ORDINARY_HANDOFF_MAX_WIDTH = None
THINK_END_ID = 10
_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"}

_TOPOLOGY = {
    "model_type": "xing4_0",
    "hidden_size": 3584,
    "num_hidden_layers": 40,
    "num_attention_heads": 32,
    "num_key_value_heads": 32,
    "q_lora_rank": 768,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "vocab_size": 131072,
    "n_routed_experts": 64,
    "n_shared_experts": 1,
    "num_experts_per_tok": 4,
    "moe_intermediate_size": 1024,
    "intermediate_size": 9216,
    "first_k_dense_replace": 2,
    "topk_method": "noaux_tc",
    "scoring_func": "sigmoid",
    "n_group": 1,
    "topk_group": 1,
    "hc_mult": 4,
    "tie_word_embeddings": False,
}


def descriptor_for(*, has_mtp: bool) -> ModelDescriptor:
    capabilities = {
        Capability.TEXT,
        Capability.STREAMING,
        Capability.TOOLS,
        Capability.REASONING,
        Capability.CONTINUOUS_BATCH,
        Capability.PREFIX_REUSE,
        Capability.APC_V2,
        Capability.LAYERED_CACHE,
        # Target-verified prompt lookup rolls plain KVCache rows back exactly.
        Capability.PROMPT_LOOKUP,
        # Grammar defers past the atomic ``</think>`` token (id 10).
        Capability.GRAMMAR,
    }
    planes = {
        StatePlane.ATTENTION_KV,
        StatePlane.RNG,
        StatePlane.TRANSCRIPT,
        StatePlane.GRAMMAR,
    }
    if has_mtp:
        capabilities.update({Capability.MTP, Capability.SEGMENTED_MTP})
        planes.add(StatePlane.DRAFT)
    return ModelDescriptor(
        model_type="xing4_0",
        family="xing4.0-29b-a4b",
        variant="native-mtp" if has_mtp else "ordinary",
        state_planes=frozenset(planes),
        capabilities=frozenset(capabilities),
        cache_layout=CACHE_LAYOUT,
        metadata={
            "execution": "mlx2.adapters.xing.XingAdapter",
            "qualification": "pending",
            "scope": "text-only",
            "residual": "mhc-4-stream-sinkhorn",
        },
    )


XING4_0 = descriptor_for(has_mtp=False)


def _load_json(path: Path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r} in {path.name}")
            value[key] = item
        return value

    return json.loads(path.read_text(), object_pairs_hook=unique)


def _header_sha256(path: Path) -> str:
    """Digest of a shard's safetensors header (tensor names, dtypes, offsets)."""
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError(f"truncated safetensors file: {path.name}")
        length = struct.unpack("<Q", raw)[0]
        if not 0 < length <= min(64 << 20, path.stat().st_size - 8):
            raise ValueError(f"invalid safetensors header length: {path.name}")
        return hashlib.sha256(stream.read(length)).hexdigest()


def verify_shard_digests(model_path: str | Path) -> None:
    """Full-content check of every shard against the converter's sha256."""
    path = Path(model_path).expanduser().resolve()
    config = _load_json(path / "config.json")
    for name, expected in (config.get("mlx2_conversion") or {}).get("shard_sha256", {}).items():
        digest = hashlib.sha256()
        with (path / name).open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 24), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise ValueError(f"shard {name} does not match its recorded sha256")


def inspect_artifact(model_path: str | Path) -> dict:
    """Validate a converted Xing4.0 artifact without importing MLX."""
    path = Path(model_path).expanduser().resolve()
    config = _load_json(path / "config.json")
    if any(config.get(key) != value for key, value in _TOPOLOGY.items()):
        raise ValueError("artifact topology does not match Xing4.0-29B-A4B")
    conversion = config.get("mlx2_conversion")
    if not isinstance(conversion, dict) or conversion.get("layout") != CONVERSION_LAYOUT:
        raise ValueError(
            "Xing4.0 artifacts must be converted with scripts/convert_xing4_0.py"
        )
    if int(config.get("num_nextn_predict_layers", 0)) not in (0, 1):
        raise ValueError("only the single-layer Xing4.0 MTP head is implemented")
    index = _load_json(path / "model.safetensors.index.json").get("weight_map")
    if not isinstance(index, dict) or not index:
        raise ValueError("Xing4.0 artifact must have a nonempty weight index")
    records = []
    for name in sorted(set(index.values())):
        if not isinstance(name, str):
            raise TypeError("weight shard path must be a string")
        item = (path / name).resolve()
        if not item.is_relative_to(path) or item.suffix != ".safetensors" or not item.is_file():
            raise ValueError("weight index must reference local safetensors files")
        stat = item.stat()
        records.append((name, stat.st_size, stat.st_mtime_ns))
    if any(key.startswith("model.layers.40.") for key in index):
        raise ValueError("unconverted Xing4.0 MTP tensors; rerun the converter")
    mtp_keys = [key for key in index if key.startswith("mtp.layers.0.")]
    has_mtp = bool(mtp_keys)
    if has_mtp:
        required = {
            "mtp.layers.0.enorm.weight",
            "mtp.layers.0.hnorm.weight",
            "mtp.layers.0.shared_head.norm.weight",
            "mtp.layers.0.self_attn.kv_a_proj_with_mqa.weight",
            "mtp.layers.0.mlp.gate.weight",
            "mtp.layers.0.mlp.switch_mlp.down_proj.weight",
        }
        if not required <= set(index) or not any(k.startswith("mtp.layers.0.eh_proj.") for k in index):
            raise ValueError("embedded Xing4.0 MTP head is incomplete")
        if int(config.get("num_nextn_predict_layers", 0)) != 1:
            raise ValueError("MTP tensors and configured head count disagree")
        # Absence means "shared with the trunk" only when the converter proved
        # equality and recorded it; otherwise a missing tensor is a defect.
        for flag, key in (
            ("mtp_embedding_shared", "mtp.layers.0.embed_tokens.weight"),
            ("mtp_head_shared", "mtp.layers.0.shared_head.head.weight"),
        ):
            shared = conversion.get(flag)
            if type(shared) is not bool:
                raise ValueError(f"Xing4.0 conversion metadata lacks {flag}")
            if shared == (key in index):
                raise ValueError(f"MTP tensor {key} disagrees with {flag}={shared}")
    for name in ("tokenizer.json", "xing4_0_tokenizer_parity.json", "chat_template.jinja"):
        if not (path / name).is_file():
            raise ValueError(
                f"Xing4.0 artifact is missing {name}; build it with scripts/xing4_0_tokenizer.py"
            )
    digest = hashlib.sha256()
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "xing4_0_tokenizer_parity.json",
        "chat_template.jinja",
        "generation_config.json",
    ):
        item = path / name
        if item.is_file():
            digest.update(name.encode())
            digest.update(item.read_bytes())
    shard_sha256 = conversion.get("shard_sha256")
    if not isinstance(shard_sha256, dict) or set(shard_sha256) != {r[0] for r in records}:
        raise ValueError("Xing4.0 conversion metadata must list every shard's sha256")
    for record in records:
        digest.update(json.dumps(record).encode())
        digest.update(shard_sha256[record[0]].encode())
        digest.update(_header_sha256(path / record[0]).encode())
    return {
        "config": config,
        "weight_map": index,
        "has_mtp": has_mtp,
        "mtp_tensor_count": len(mtp_keys),
        "identity": {"path": str(path), "fingerprint": digest.hexdigest(), "files": records},
        "qualification": "pending",
    }


def configure_environment() -> dict[str, str]:
    profile = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "MLX_ENABLE_TF32": "0",
        "MLX_LM_COMPILED_DECODE": "0",
        "MLX_LM_SEGMENTED_SELF_MTP": "1",
        "MLX_LM_TRUE_BATCHED_SEGMENTED_MTP": "1",
        "MLX_LM_SHARED_QSA_SUFFIX": "0",
        # Committed MTP-boundary COW snapshots, pinned so the receipt records it.
        "MLX_LM_MTP_BOUNDARY_COW": "1",
    }
    for name in tuple(os.environ):
        if name.startswith(("MLX_QWEN", "MLX_LM_", "MLXUAG_", "MLX_GDN_")):
            del os.environ[name]
    os.environ.update(profile)
    return profile


def reasoning_policy(request: dict) -> tuple[str, bool]:
    """Resolve Xing thinking once for prompting, parsing, and receipts.

    Thinking is on unless the client turns it off, as in the vendor chat
    template (``enable_thinking`` defaults true).  Unlike North, constrained
    requests keep thinking on: ``</think>`` is one atomic token, so the
    grammar is deferred until reasoning closes.
    """
    effort = request.get("reasoning_effort")
    if effort is None:
        effort = "none" if request.get("enable_thinking") is False else "high"
    if effort not in _REASONING_EFFORTS:
        raise ValueError("Unsupported Xing4.0 reasoning_effort")
    thinking = bool(request.get("enable_thinking", effort != "none"))
    return effort, thinking


def _tools(request):
    return request.get("tools") if request.get("tool_choice") != "none" else None


class XingToolCallProcessor:
    """Force Xing's tool-call branch after its optional thinking block.

    The mask is derived only from committed token history, so speculative and
    prompt-lookup probes can replay it without request-local mutable state.
    """

    history_pure = True  # P5

    def __init__(self, *, prompt_length, tool_open, thinking_close=None):
        self.prompt_length = int(prompt_length)
        self.tool_open = tuple(int(token) for token in tool_open)
        self.thinking_close = (
            tuple(int(token) for token in thinking_close)
            if thinking_close is not None
            else None
        )
        if self.prompt_length < 0 or not self.tool_open or self.thinking_close == ():
            raise ValueError("Xing tool constraints require nonempty markers")

    def __call__(self, tokens, logits):
        import mlx.core as mx

        generated = tokens[self.prompt_length :]
        length = generated.shape[0]
        vocabulary = mx.arange(logits.shape[-1])
        if self.thinking_close is None:
            compared = min(length, len(self.tool_open))
            matches = (
                mx.array(True)
                if compared == 0
                else mx.all(
                    generated[:compared]
                    == mx.array(self.tool_open[:compared], dtype=tokens.dtype)
                )
            )
            if length >= len(self.tool_open):
                allowed = matches
            else:
                allowed = mx.logical_and(
                    matches, vocabulary == self.tool_open[length]
                )
        else:
            active = mx.array(False)
            constrained = mx.zeros((logits.shape[-1],), dtype=mx.bool_)
            for offset in range(len(self.tool_open)):
                suffix = (*self.thinking_close, *self.tool_open[:offset])
                if length < len(suffix):
                    continue
                matches = mx.all(
                    generated[-len(suffix) :]
                    == mx.array(suffix, dtype=tokens.dtype)
                )
                active = mx.logical_or(active, matches)
                constrained = mx.logical_or(
                    constrained,
                    mx.logical_and(
                        matches, vocabulary == self.tool_open[offset]
                    ),
                )
            allowed = mx.logical_or(constrained, ~active)
        return mx.where(
            allowed, logits, mx.array(float("-inf"), dtype=logits.dtype)
        )

    def probe(self, tokens, logits):
        """Speculative probes use the same pure history-derived mask."""
        return self(tokens, logits)


class XingAdapter:
    descriptor = XING4_0
    default_route = "native_mtp"
    default_mtp_ordinary_handoff_max_width = (
        DEFAULT_MTP_ORDINARY_HANDOFF_MAX_WIDTH
    )
    sampling_defaults = XING4_SAMPLING
    reasoning_effort_semantics = "thinking_toggle"
    # Xing reasons at length but not pathologically: with 8192 tokens and no
    # guard every probe closed </think> and answered; open-ended prompts took
    # ~1.2K reasoning + ~1.1K answer tokens (qualification/runs/
    # xing4-0-gpu-20260918/thinking-length).  Harnesses size budgets from this.
    thinking_allowance_tokens = 4096

    @staticmethod
    def thinking_enabled(request: dict) -> bool:
        return reasoning_policy(request)[1]

    @staticmethod
    def thinking_close_token_ids():
        return (THINK_END_ID,)

    @staticmethod
    def int8_prefill_supported():
        # Dense MLP, shared expert, and (with "all") MLA projections.  Routed
        # experts, the router, mHC operands, heads and the MTP layer stay stock.
        return ("mlp", "all")

    def __init__(self, model_path: str, *, require_mtp: bool = False, execution_policy=None):
        if execution_policy is not None and not isinstance(execution_policy, dict):
            raise ValueError("execution policy must be a JSON object")
        policy = {} if execution_policy is None else dict(execution_policy)
        if set(policy) - {"num_draft", "tokenizer_reference_fallback"}:
            raise ValueError(
                "Xing4.0 execution policy supports only num_draft and tokenizer_reference_fallback"
            )
        # One trained MTP layer; vLLM's recommended depth is 1.
        self._num_draft = validate_self_mtp_num_draft(policy.get("num_draft", 1))
        fallback = policy.get("tokenizer_reference_fallback", False)
        if type(fallback) is not bool:
            raise ValueError("tokenizer_reference_fallback must be boolean")
        artifact = inspect_artifact(model_path)
        if require_mtp and not artifact["has_mtp"]:
            raise ValueError("requested MTP requires embedded head weights")
        self.identity = artifact["identity"]
        self.descriptor = descriptor_for(has_mtp=artifact["has_mtp"])
        self.config = dict(artifact["config"])
        self.environment = configure_environment()
        self.layout = CACHE_LAYOUT
        path = Path(self.identity["path"])
        config = dict(artifact["config"])
        if not artifact["has_mtp"]:
            config["num_nextn_predict_layers"] = 0

        import mlx.core as mx
        import mlx.nn as nn

        from ..runtime.models.xing4_0 import Model, ModelArgs
        from ..runtime.ubc_evict import load_shards_evicting
        from .xing_tokenizer import load_tokenizer, make_tokenizer_wrapper

        self.model = None
        self.tokenizer = None
        try:
            self.model = Model(ModelArgs.from_dict(config))
            if self.model.apc_v2_layout != CACHE_LAYOUT:
                raise ValueError("Xing4.0 model and adapter cache layouts disagree")
            files = [path / name for name in sorted(set(artifact["weight_map"].values()))]
            weights = self.model.sanitize(load_shards_evicting(files))
            quant = config.get("quantization", config.get("quantization_config"))
            if quant:
                def predicate(name, module):
                    override = quant.get(name)
                    if isinstance(override, dict):
                        return override
                    return hasattr(module, "to_quantized") and f"{name}.scales" in weights

                nn.quantize(
                    self.model,
                    group_size=quant["group_size"],
                    bits=quant["bits"],
                    mode=quant.get("mode", "affine"),
                    class_predicate=predicate,
                )
            self.model.load_weights(list(weights.items()), strict=True)
            self.model.eval()
            mx.eval(self.model.parameters())
            weights.clear()
            mx.clear_cache()
            tokenizer, self.tokenizer_receipt = load_tokenizer(
                path, allow_reference_fallback=fallback
            )
            eos = config.get("eos_token_id", 2)
            self.tokenizer = make_tokenizer_wrapper(
                tokenizer, eos_token_ids=[eos] if isinstance(eos, int) else list(eos)
            )
        except BaseException:
            self.close()
            raise
        self.max_context = int(config["max_position_embeddings"])

    def prompt_tokens(self, request: dict) -> list[int]:
        if "messages" not in request:
            return self.tokenizer.encode(request["prompt"], add_special_tokens=False)
        from .xing_tokenizer import render_prompt

        _, thinking = reasoning_policy(request)
        return render_prompt(
            self.tokenizer,
            request["messages"],
            tools=_tools(request),
            enable_thinking=thinking,
        )

    def render_prompt(self, request: dict) -> str:
        """Prompt text whose special-token-free encoding is ``prompt_tokens``."""
        if "messages" not in request:
            return request["prompt"]
        from .xing_tokenizer import render_prompt_text

        _, thinking = reasoning_policy(request)
        return render_prompt_text(
            self.tokenizer,
            request["messages"],
            tools=_tools(request),
            enable_thinking=thinking,
        )

    def output_parser(self, request):
        from .xing_output import XingOutputParser

        _, thinking = reasoning_policy(request)
        return XingOutputParser(
            chat="messages" in request,
            thinking=thinking,
            tools=_tools(request),
            stops=request.get("stop", ()),
            parallel_tool_calls=request.get("parallel_tool_calls", True),
        )

    def request_logits_processors(self, request, *, prompt_length):
        """Constrain required/named tool requests to Xing's tool-call branch."""
        if "messages" not in request:
            return ()
        choice = request.get("tool_choice", "auto")
        if choice != "required" and not isinstance(choice, dict):
            return ()
        if not request.get("tools"):
            return ()
        from .xing_output import THINK_CLOSE, TOOL_OPEN

        _, thinking = reasoning_policy(request)
        tool_open = self.tokenizer.encode(TOOL_OPEN, add_special_tokens=False)
        thinking_close = (
            self.tokenizer.encode(THINK_CLOSE, add_special_tokens=False)
            if thinking
            else None
        )
        return (
            XingToolCallProcessor(
                prompt_length=prompt_length,
                tool_open=tool_open,
                thinking_close=thinking_close,
            ),
        )

    def profile_name(self, mtp):
        if mtp and Capability.MTP not in self.descriptor.capabilities:
            raise ValueError("requested MTP requires embedded head weights")
        return f"xing4-0-apcv2-{'mtp' + str(self._num_draft) if mtp else 'ordinary'}"

    def execution_config(self, *, max_lanes, prefill_step):
        return {
            "persistent": True,
            "num_draft": self._num_draft
            if Capability.MTP in self.descriptor.capabilities
            else 0,
            "rate_gate": False,
            "prefill_step_size": prefill_step,
            "segment_aware_live_tip": True,
            "segment_aware_cohort_size": max_lanes,
        }

    def cache_budget(self, *, mtp):
        from .xing_memory import XingCacheBudget

        return XingCacheBudget.from_config(self.config, mtp=mtp)

    def diagnostics(self):
        from ..runtime.models.xing4_0 import mhc_stats
        from ..runtime.segmented_self_mtp import segmented_self_mtp_stats

        return {
            "architecture": "mla-noaux-moe-mhc4",
            "cache_layout": self.layout,
            "mtp_head_present": Capability.MTP in self.descriptor.capabilities,
            "num_draft": self._num_draft,
            "tokenizer": dict(getattr(self, "tokenizer_receipt", {})),
            "mhc": mhc_stats(),
            "segmented_mtp": segmented_self_mtp_stats(),
        }

    def close(self):
        """Release model ownership before the shared worker shuts down."""
        had_resources = any(
            getattr(self, name, None) is not None for name in ("model", "tokenizer")
        )
        self.model = None
        self.tokenizer = None
        if had_resources:
            import mlx.core as mx

            mx.clear_cache()
