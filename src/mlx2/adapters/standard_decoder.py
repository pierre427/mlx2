"""CPU-safe artifact gate and ordinary serving adapters for standard decoders."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from .external_draft_policy import ExternalDraftAdapterMixin

FAMILIES = frozenset({"qwen3", "qwen3_moe", "qwen2", "llama"})
CACHE_LAYOUT = "standard-full-kv-layer-segments-v1"


def descriptor_for(family: str) -> ModelDescriptor:
    if family not in FAMILIES:
        raise ValueError(f"unsupported standard decoder family: {family}")
    return ModelDescriptor(
        model_type=family,
        family=family.replace("_", "-"),
        variant="ordinary",
        state_planes=frozenset(
            {StatePlane.ATTENTION_KV, StatePlane.RNG, StatePlane.TRANSCRIPT}
        ),
        capabilities=frozenset(
            {
                Capability.TEXT,
                Capability.STREAMING,
                Capability.CONTINUOUS_BATCH,
                Capability.PREFIX_REUSE,
                Capability.APC_V2,
                Capability.LAYERED_CACHE,
            }
        ),
        cache_layout=CACHE_LAYOUT,
        metadata={
            "execution": "mlx2.adapters.standard_decoder.StandardDecoderAdapter",
            "qualification": "pending",
            "scope": "text-only ordinary decode",
        },
    )


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
        raise ValueError(f"{path.name} must be an object")  # noqa: TRY004
    return value


def _positive(config: dict, name: str) -> int:
    value = config.get(name)
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def inspect_artifact(model_path: str | Path, *, expected: str | None = None) -> dict:
    """Inspect topology, index closure and identity with no tensor imports."""
    path = Path(model_path).expanduser().resolve()
    config = _json(path / "config.json")
    family = config.get("model_type")
    if family not in FAMILIES or (expected is not None and family != expected):
        raise ValueError(f"unsupported artifact family: {family!r}")
    for key in (
        "hidden_size",
        "num_hidden_layers",
        "intermediate_size",
        "num_attention_heads",
        "vocab_size",
        "max_position_embeddings",
    ):
        _positive(config, key)
    heads = _positive(config, "num_attention_heads")
    kv_heads = config.get("num_key_value_heads", heads)
    if type(kv_heads) is not int or kv_heads < 1 or heads % kv_heads:
        raise ValueError("invalid grouped-query head geometry")
    if (
        type(config.get("rms_norm_eps")) not in {float, int}
        or config["rms_norm_eps"] <= 0
    ):
        raise ValueError("invalid RMS normalization epsilon")
    if type(config.get("rope_theta")) not in {float, int} or config["rope_theta"] <= 0:
        raise ValueError("invalid rotary base")
    if type(config.get("tie_word_embeddings")) is not bool:
        raise ValueError("tie_word_embeddings must be declared")
    if family in {"qwen3", "qwen3_moe"} or config.get("head_dim") is not None:
        _positive(config, "head_dim")
    elif config["hidden_size"] % heads:
        raise ValueError("head dimension is not integral")
    if family == "qwen3_moe":
        experts = _positive(config, "num_experts")
        top_k = _positive(config, "num_experts_per_tok")
        _positive(config, "moe_intermediate_size")
        _positive(config, "decoder_sparse_step")
        if top_k > experts or not isinstance(config.get("mlp_only_layers"), list):
            raise ValueError("invalid MoE routing geometry")
        if any(
            type(i) is not int or i < 0 or i >= config["num_hidden_layers"]
            for i in config["mlp_only_layers"]
        ):
            raise ValueError("invalid dense-only layer list")
        if type(config.get("norm_topk_prob")) is not bool:
            raise ValueError("norm_topk_prob must be declared")
    if family == "llama" and config.get("layer_types") not in (
        None,
        ["full_attention"] * config["num_hidden_layers"],
    ):
        raise ValueError("sliding Llama requires a separate cache layout")
    if config.get("sliding_window") and family == "llama":
        raise ValueError("sliding Llama requires a separate cache layout")
    quant = config.get("quantization", config.get("quantization_config"))
    if quant is not None and (
        not isinstance(quant, dict)
        or type(quant.get("group_size")) is not int
        or type(quant.get("bits")) is not int
        or quant["group_size"] <= 0
        or quant["bits"] not in {4, 6, 8}
    ):
        raise ValueError("unsupported quantization configuration")
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        weights = _json(index_path).get("weight_map")
        if not isinstance(weights, dict) or not weights:
            raise ValueError("empty safetensors index")
    else:
        only = path / "model.safetensors"
        if not only.is_file():
            raise ValueError("missing model weights")
        # Indexless BF16 artifacts are accepted; strict tensor names are
        # checked by load_weights(strict=True) at the model-load boundary.
        weights = None
    names = (
        sorted(set(weights.values())) if weights is not None else ["model.safetensors"]
    )
    records = []
    for name in names:
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or not name.endswith(".safetensors")
        ):
            raise ValueError("weight index contains an unsafe shard path")
        # Hugging Face snapshots link shards into their sibling blobs tree.
        # The basename gate above prevents traversal while preserving those
        # legitimate local links.
        item = path / name
        if not item.is_file() or item.stat().st_size < 8:
            raise ValueError(f"missing or empty weight shard: {name}")
        stat = item.stat()
        records.append((name, stat.st_size, stat.st_mtime_ns))
    if weights is not None:
        if any(k.startswith(("mtp.", "draft.")) for k in weights):
            raise ValueError("speculative head is not an ordinary target artifact")
        required = {
            "model.embed_tokens.weight",
            "model.norm.weight",
            "model.layers.0.self_attn.q_proj.weight",
        }
        if not required.issubset(weights):
            raise ValueError("index lacks required decoder tensors")
        if family == "qwen3_moe" and "model.layers.0.mlp.gate.weight" not in weights:
            raise ValueError("MoE index lacks router tensors")
    digest = hashlib.sha256()
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
    ):
        item = path / name
        if item.is_file():
            digest.update(name.encode())
            digest.update(item.read_bytes())
    for record in records:
        digest.update(json.dumps(record).encode())
    normalized = dict(config)
    normalized["num_key_value_heads"] = kv_heads
    normalized.setdefault("head_dim", config["hidden_size"] // heads)
    normalized.setdefault("rope_traditional", False)
    normalized.setdefault("attention_bias", False)
    normalized.setdefault("mlp_bias", False)
    return {
        "config": normalized,
        "weight_map": weights,
        "identity": {
            "path": str(path),
            "fingerprint": digest.hexdigest(),
            "files": records,
        },
        "qualification": "pending",
        "has_mtp": False,
    }


class StandardDecoderAdapter(ExternalDraftAdapterMixin):
    default_route = "ordinary"
    descriptor = descriptor_for("qwen3")
    EXTERNAL_DEFAULT_NUM_DRAFT = 15
    EXTERNAL_ROUTE_TAG = "external-xpress-qwen3-v1"
    EXTERNAL_PROFILE = "qwen3-apcv2-xpress"

    def profile_name(self, mtp):
        if mtp:
            raise ValueError("standard decoder has no native MTP route")
        return f"{self.descriptor.model_type}-apcv2-ordinary"

    def execution_config(self, *, max_lanes, prefill_step):
        if getattr(self, "draft_model", None) is not None:
            return self._external_execution_config(
                max_lanes=max_lanes, prefill_step=prefill_step
            )
        return {
            "persistent": True,
            "num_draft": 0,
            "backend": "ordinary",
            "rate_gate": False,
            "prefill_step_size": prefill_step,
            "segment_aware_live_tip": False,
            "segment_aware_cohort_size": max_lanes,
        }

    def close(self):
        self._close_external_feedback()

    def __init__(self, model_path: str, *, execution_policy=None):
        self.external_policy = dict(execution_policy or {})
        self.draft_model = None
        allowed = {
            "draft_model",
            "num_draft",
            "xpress_num_passes",
            "draft_quantization",
            "adaptive_verification",
            "draft_revision",
            "target_fingerprint",
            "draft_attention_windows",
            "proposal_composition",
            "continuation_pool",
            "lilicorr_feedback",
            "target_verify_row_exact",
        }
        row_exact = self.external_policy.get("target_verify_row_exact", False)
        if type(row_exact) is not bool:
            raise ValueError("target_verify_row_exact must be a boolean")
        if not row_exact:
            self.external_policy.pop("target_verify_row_exact", None)
        if set(self.external_policy) - allowed or (
            self.external_policy and not self.external_policy.get("draft_model")
        ):
            raise ValueError(
                "standard decoder execution policy requires a supported draft_model"
            )
        artifact = inspect_artifact(model_path)
        if row_exact:
            config = artifact["config"]
            if (
                config["model_type"] != "qwen3"
                or config.get("num_experts", 0)
                or config.get("rope_scaling") is not None
                or config.get("quantization")
                or config.get("quantization_config")
                or config.get("sliding_window")
                or config.get("use_sliding_window")
                or config.get("layer_types")
                not in (None, ["full_attention"] * config["num_hidden_layers"])
            ):
                raise ValueError(
                    "target_verify_row_exact requires an unquantized dense full-attention Qwen3 target"
                )
            if os.environ.get("MLX2_FP_DECODE_KERNEL", "0") == "1":
                raise ValueError(
                    "target_verify_row_exact does not support MLX2_FP_DECODE_KERNEL"
                )
        draft_record = None
        if self.external_policy:
            if artifact["config"]["model_type"] != "qwen3":
                raise ValueError("external drafting currently requires a Qwen3 target")
            draft_path = (
                Path(self.external_policy["draft_model"]).expanduser().resolve()
            )
            architecture = _json(draft_path / "config.json").get("architectures")
            if architecture == ["Qwen3XPressModel"]:
                from .xpress import content_revision, inspect_drafter, load_drafter

                draft_kind = "xpress"
            elif architecture == ["LiLiCorrDraftModel"]:
                from .lilicorr import content_revision, inspect_drafter, load_drafter

                draft_kind = "lilicorr"
                if "xpress_num_passes" in self.external_policy:
                    raise ValueError("xpress_num_passes requires an XPress drafter")
            else:
                raise ValueError(
                    f"unsupported standard decoder draft architecture: {architecture!r}"
                )
            self.EXTERNAL_ROUTE_TAG = f"external-{draft_kind}-qwen3-v1"
            self.EXTERNAL_PROFILE = f"qwen3-apcv2-{draft_kind}"
            draft_record = inspect_drafter(draft_path, model_path)
            from ..runtime.drafters.attention_windows import validate_attention_windows

            validate_attention_windows(
                self.external_policy.get("draft_attention_windows"),
                draft_record["args"].num_hidden_layers,
            )
            if "draft_revision" in self.external_policy and self.external_policy[
                "draft_revision"
            ] != content_revision(draft_record):
                raise ValueError("external drafter source revision mismatch")
            if (
                "target_fingerprint" in self.external_policy
                and self.external_policy["target_fingerprint"]
                != artifact["identity"]["fingerprint"]
            ):
                raise ValueError("external target artifact fingerprint mismatch")
            self._check_num_draft(draft_record)
            if draft_kind == "xpress":
                passes = self.external_policy.get(
                    "xpress_num_passes", draft_record["args"].xpress_num_passes
                )
                if (
                    type(passes) is not int
                    or not 1 <= passes <= draft_record["args"].block_size
                ):
                    raise ValueError(
                        "xpress_num_passes must be a positive integer within the trained block"
                    )
            if self.external_policy.get("draft_quantization") is not None:
                raise ValueError(f"{draft_kind} runtime quantization is not supported")
            adaptive = self.external_policy.get("adaptive_verification")
            if adaptive is not None:
                from ..runtime.acceptance_estimator import AdaptiveVerificationPolicy

                AdaptiveVerificationPolicy.from_value(
                    adaptive, self._external_num_draft()
                )
        self.config = artifact["config"]
        self.identity = artifact["identity"]
        self.descriptor = descriptor_for(self.config["model_type"])
        self.layout = CACHE_LAYOUT
        self.environment = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        path = Path(self.identity["path"])
        import mlx.core as mx
        from mlx import nn
        from transformers import AutoTokenizer

        from ..runtime.models.standard_decoder import Model, ModelArgs
        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper
        from ..runtime.ubc_evict import load_shards_evicting

        self.model = Model(ModelArgs.from_dict(self.config))
        files = [path / name for name, _, _ in self.identity["files"]]
        weights = self.model.sanitize(load_shards_evicting(files))
        quant = self.config.get("quantization", self.config.get("quantization_config"))
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
        if row_exact:
            from ..runtime.models.standard_decoder import (
                TARGET_VERIFY_ROW_EXACT_VERSION,
            )

            self.model.configure_target_verify_row_exact(True)
            # Pin the changed target arithmetic before source/session/teacher
            # identities are created by the external drafter binding.
            execution_settings = {
                "algorithm": TARGET_VERIFY_ROW_EXACT_VERSION,
                "draft_attention_windows": self.external_policy.get(
                    "draft_attention_windows"
                ),
            }
            if draft_kind == "xpress":
                execution_settings["xpress_num_passes"] = passes
            digest = hashlib.sha256(
                (
                    self.identity["fingerprint"]
                    + json.dumps(execution_settings, sort_keys=True)
                ).encode()
            ).hexdigest()
            self.identity = {
                **self.identity,
                "artifact_fingerprint": self.identity["fingerprint"],
                "fingerprint": digest,
            }
            self.layout += ":target-verify-row-exact:" + digest
        self.model.eval()
        mx.eval(self.model.parameters())
        weights.clear()
        mx.clear_cache()
        tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        # Some tokenizer classes rebuild the pre-tokenizer instead of reading tokenizer.json.
        from ..runtime.tokenizer_integrity import repair_loaded_tokenizer

        self.pretokenizer_receipt = repair_loaded_tokenizer(tokenizer, path)
        from .eos import artifact_eos_token_ids

        self.tokenizer = TokenizerWrapper(
            tokenizer,
            detokenizer_class=BPEStreamingDetokenizer,
            eos_token_ids=artifact_eos_token_ids(path, self.config, tokenizer),
        )
        self.max_context = self.config["max_position_embeddings"]
        if draft_record is not None:

            def loader(record, target):
                options = {
                    "runtime_quantization": self.external_policy.get(
                        "draft_quantization"
                    ),
                    "draft_attention_windows": self.external_policy.get(
                        "draft_attention_windows"
                    ),
                }
                if draft_kind == "xpress":
                    options["num_passes"] = self.external_policy.get(
                        "xpress_num_passes"
                    )
                return load_drafter(record, target, **options)

            self._bind_external_drafter(draft_record, loader, self.descriptor)
            # Runtime quantization changes stored draft context. Bind the
            # execution settings before any paired sidecar is published.
            settings = json.dumps(self.draft_model.receipt_settings, sort_keys=True)
            settings += json.dumps(
                self.external_policy.get("draft_quantization"), sort_keys=True
            )
            fingerprint = hashlib.sha256(
                (self.identity["fingerprint"] + settings).encode()
            ).hexdigest()
            self.identity = {**self.identity, "fingerprint": fingerprint}
            self.layout += ":" + fingerprint
            self.descriptor = replace(
                self.descriptor,
                variant=f"external-{draft_kind}",
                cache_layout=self.layout,
                metadata={
                    **self.descriptor.metadata,
                    "qualification": "unqualified",
                    "scope": f"text-only {draft_kind} draft/verify",
                    "implemented": True,
                },
            )
            if row_exact:
                self.descriptor = replace(
                    self.descriptor,
                    metadata={
                        **self.descriptor.metadata,
                        "target_verify_row_exact": {
                            "algorithm": TARGET_VERIFY_ROW_EXACT_VERSION,
                            "implemented": True,
                            "qualified": False,
                            "selected": True,
                            "performance_claim": False,
                        },
                    },
                )

    def external_profile_name(self, mtp):
        if mtp:
            raise ValueError("External draft is not native MTP")
        return self.EXTERNAL_PROFILE

    def create_external_batch(self, **kwargs):
        adaptive = self.external_policy.get("adaptive_verification")
        if adaptive is not None:
            kwargs["adaptive_verification"] = adaptive
        return super().create_external_batch(**kwargs)

    # The descriptor declares no reasoning (an explicit enable_thinking: true
    # is refused), so thinking defaults off.  Defaulting to the tokenizer's
    # has_thinking rendered Qwen3 prompts in thinking mode and parsed a
    # reasoning channel that serving's thinking_enabled() said was closed.
    @staticmethod
    def _thinking(request: dict) -> bool:
        return bool(request.get("enable_thinking", False))

    def prompt_tokens(self, request: dict) -> list[int]:
        if "messages" in request:
            return self.tokenizer.apply_chat_template(
                request["messages"],
                add_generation_prompt=True,
                tokenize=True,
                enable_thinking=self._thinking(request),
            )
        return self.tokenizer.encode(request["prompt"], add_special_tokens=False)

    def render_prompt(self, request: dict) -> str:
        if "messages" in request:
            return self.tokenizer.apply_chat_template(
                request["messages"],
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=self._thinking(request),
            )
        return request["prompt"]

    def output_parser(self, request):
        from ..output import OutputParser

        if request.get("tools") or request.get("tool_choice") not in (
            None,
            "none",
            "auto",
        ):
            raise ValueError("tool calling is not qualified for this adapter")
        return OutputParser(
            chat="messages" in request,
            thinking=self.config["model_type"] in {"qwen3", "qwen3_moe"}
            and self._thinking(request),
            stops=request.get("stop", ()),
        )

    def diagnostics(self):
        external = getattr(self, "draft_model", None) is not None
        result = {
            "architecture": self.config["model_type"],
            "cache_layout": self.layout,
            "qualification": "unqualified" if external else "pending",
            "route": "external_draft" if external else "ordinary",
        }
        if external:
            result["speculation"] = {
                "kind": self.draft_model.receipt_kind,
                "implemented": True,
                "qualified": False,
                "settings": self.draft_model.receipt_settings,
                "adaptive_verification": self.external_policy.get(
                    "adaptive_verification"
                ),
            }
        if self.external_policy.get("target_verify_row_exact", False):
            result["target_protocol"] = self.model.external_execution_receipt
        return result
