"""Muse Glimmer text adapter. Import and inspect without importing MLX.

An instantiated adapter is an execution candidate, not a qualified route.
Only the serving qualification gate may publish a selectable profile.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from .muse_glimmer_config import ModelArgs
from ..sampling_defaults import SamplingDefaults, VendorSampling


MUSE_GLIMMER = ModelDescriptor(
    model_type="muse_glimmer",
    family="muse-glimmer",
    variant="text",
    state_planes=frozenset(
        {StatePlane.ATTENTION_KV, StatePlane.RNG, StatePlane.TRANSCRIPT}
    ),
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
            # Target-verified prompt lookup is model-neutral; it rolls stock
            # KVCache and RotatingKVCache planes back exactly.
            Capability.PROMPT_LOOKUP,
            # Constrained decoding works on token text and the tokenizer's EOS
            # ids only; nothing in it is specific to a model family.
            Capability.GRAMMAR,
        }
    ),
    cache_layout="muse-glimmer-layer-segments-v1",
    metadata={
        "execution": "mlx2.adapters.muse_glimmer.MuseGlimmerAdapter",
        "qualification": "pending",
        "speculation": "external-dflash2-candidate",
    },
)


def inspect_artifact(model_path: str | Path) -> dict:
    """Metadata-only inspection; never opens weight payloads or imports MLX."""
    path = Path(model_path).expanduser().resolve()
    config = json.loads((path / "config.json").read_text())
    if "dflash_config" in config or any(
        "Draft" in name for name in config.get("architectures", [])
    ):
        raise ValueError("Expected a Muse target artifact, not a speculative drafter")
    if config.get("model_type") not in {"muse_glimmer", "muse_glimmer_text"}:
        raise ValueError("Expected a Muse Glimmer target artifact, not a drafter")
    args = ModelArgs.from_dict(config)
    index_path = path / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())["weight_map"]
        if not isinstance(index, dict) or not index:
            raise ValueError("Artifact must have a nonempty weight index")
        names = sorted(set(index.values()))
    else:
        names = ["model.safetensors"]
    digest = hashlib.sha256()
    for name in (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ):
        item = path / name
        if item.exists():
            digest.update(name.encode())
            digest.update(item.read_bytes())
    records = []
    for name in names:
        item = (path / name).resolve()
        if not item.is_relative_to(path) or item.suffix != ".safetensors":
            raise ValueError("Weight index must reference local safetensors files")
        stat = item.stat()
        record = (name, stat.st_size, stat.st_mtime_ns)
        records.append(record)
        digest.update(json.dumps(record).encode())
    return {
        "path": str(path),
        "fingerprint": digest.hexdigest(),
        "files": records,
        "model_type": "muse_glimmer",
        "cache_layout": args.cache_layout,
        "max_context": args.max_position_embeddings,
        "layers": args.num_hidden_layers,
        "sliding_layers": args.layer_types.count("sliding_attention"),
        "global_layers": args.layer_types.count("full_attention"),
        "sliding_window": args.sliding_window,
        "qualification": "pending",
        "supports_self_mtp": False,
    }


def configure_environment() -> dict[str, str]:
    profile = {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "MLX_ENABLE_TF32": "0",
        "MLX_LM_COMPILED_DECODE": "0",
        "MLX_LM_SEGMENTED_SELF_MTP": "0",
        "MLX_LM_TRUE_BATCHED_SEGMENTED_MTP": "0",
    }
    for name in tuple(os.environ):
        if name.startswith(("MLX_QWEN", "MLX_LM_", "MLXUAG_")):
            del os.environ[name]
    os.environ.update(profile)
    return profile


def normalize_messages(messages: list[dict]) -> list[dict]:
    """The local ATEM template requires object-valued tool arguments."""
    messages = copy.deepcopy(messages)
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            if any(part.get("type") != "text" for part in content):
                raise ValueError("Muse mlx2 port currently accepts text only")
            message["content"] = "".join(part["text"] for part in content)
        for call in message.get("tool_calls", []):
            function = call["function"]
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be a JSON object")
            function["arguments"] = arguments
    return messages


# Vendor sampling defaults.  Muse-Glimmer-30B model card ("To achieve best
# performance ... Sampling Parameters", ~/mlx-models/
# Muse-Glimmer-30B/README.md): temperature 1.0, top_p 0.95, top_k 64, for all
# reasoning strengths.  The artifact's generation_config.json says only
# ``do_sample: false`` (a transformers default, no sampled values); the card is
# the vendor recommendation and wins.  The DFlash2 drafter card benchmarks
# with the same "officially recommended" values.
MUSE_GLIMMER_SAMPLING = VendorSampling.single(
    SamplingDefaults(
        temperature=1.0, top_p=0.95, top_k=64,
        source="model card (Muse-Glimmer-30B, Sampling Parameters)",
    ),
    model="Muse-Glimmer-30B",
)


class MuseGlimmerAdapter:
    descriptor = MUSE_GLIMMER
    sampling_defaults = MUSE_GLIMMER_SAMPLING
    reasoning_effort_semantics = "reasoning_strength"

    @staticmethod
    def spomin_backend(model, prompt_cache):
        """Adapter-owned approximate KV surgery for full + sliding attention."""
        from ..runtime.spomin_standard_surgery import StandardAttentionSpominBackend

        return StandardAttentionSpominBackend(model, prompt_cache)

    @staticmethod
    def profile_name(mtp):
        if mtp:
            raise ValueError(
                "Muse has no native self-MTP; DFlash2 is not integrated/qualified"
            )
        return "muse-glimmer-apcv2-ordinary"

    def execution_config(self, *, max_lanes, prefill_step):
        return {
            "persistent": True,
            "num_draft": self.external_policy.get("num_draft", 4) if getattr(self,"draft_model",None) is not None else 0,
            "backend": "external_draft" if getattr(self,"draft_model",None) is not None else "ordinary",
            "rate_gate": False,
            "prefill_step_size": prefill_step,
            "segment_aware_live_tip": False,
            "segment_aware_cohort_size": max_lanes,
        }

    def __init__(self, model_path: str, *, execution_policy=None):
        self.external_policy = dict(execution_policy or {})
        if set(self.external_policy) - {"draft_model", "num_draft"}:
            raise ValueError("Unsupported Muse execution policy")
        if self.external_policy and not self.external_policy.get("draft_model"):
            raise ValueError("Muse policy overrides require draft_model")
        self.draft_model = None
        draft_record = None
        if self.external_policy:
            from .dflash2 import inspect_drafter
            draft_record = inspect_drafter(self.external_policy["draft_model"], model_path)
            count = self.external_policy.get("num_draft", 4)
            if type(count) is not int or not 1 <= count < draft_record["args"].block_size:
                raise ValueError("num_draft must be a positive integer below draft block size")
        path = Path(model_path).expanduser().resolve()
        self.identity = inspect_artifact(path)
        self.environment = configure_environment()
        self.layout = self.identity["cache_layout"]
        self.max_context = self.identity["max_context"]
        config = json.loads((path / "config.json").read_text())
        import mlx.core as mx
        import mlx.nn as nn
        from transformers import AutoTokenizer
        from ..runtime.models.muse_glimmer import Model
        from ..runtime.tokenizer_utils import TokenizerWrapper, BPEStreamingDetokenizer
        from ..runtime.ubc_evict import ubc_evict_paths

        self.model = Model(ModelArgs.from_dict(config))
        weights = {}
        files = [path / record[0] for record in self.identity["files"]]
        for file in files:
            weights.update(self.model.sanitize(mx.load(str(file))))
        quant = config.get("quantization", config.get("quantization_config"))
        if quant:

            def predicate(name, module):
                override = quant.get(name, quant.get("language_model." + name))
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
        ubc_evict_paths([str(file) for file in files])
        tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        eos = config.get(
            "eos_token_id", config.get("text_config", {}).get("eos_token_id")
        )
        eos = [eos] if isinstance(eos, int) else list(eos or [])
        # End-of-message is a channel boundary, not a turn boundary.
        eot = tokenizer.convert_tokens_to_ids("<|eot|>")
        if isinstance(eot, int) and tokenizer.convert_ids_to_tokens(eot) == "<|eot|>":
            eos.append(eot)
        self.tokenizer = TokenizerWrapper(
            tokenizer, detokenizer_class=BPEStreamingDetokenizer, eos_token_ids=set(eos)
        )

        if draft_record is not None:
            from .dflash2 import load_drafter
            from dataclasses import replace
            self.draft_model = load_drafter(draft_record, self.model)
            self.profile_name = self.external_profile_name
            digest = hashlib.sha256((self.identity["fingerprint"] + draft_record["fingerprint"] + "external-dflash2-v1").encode()).hexdigest()
            self.identity = {**self.identity, "target_fingerprint": self.identity["fingerprint"], "draft_fingerprint": draft_record["fingerprint"], "fingerprint": digest}
            self.layout += ":external-dflash2-v1:" + digest
            self.descriptor = replace(MUSE_GLIMMER, capabilities=MUSE_GLIMMER.capabilities | {Capability.EXTERNAL_DRAFT}, state_planes=MUSE_GLIMMER.state_planes | {StatePlane.DRAFT}, cache_layout=self.layout)

    @staticmethod
    def external_profile_name(mtp):
        if mtp: raise ValueError("External draft is not native MTP")
        return "muse-glimmer-apcv2-dflash2"

    def create_external_batch(self, **kwargs):
        if self.draft_model is None:
            raise ValueError("No external draft model bound")
        from ..runtime.external_speculative import ExternalDraftBatchGenerator
        return ExternalDraftBatchGenerator(self.model, draft_model=self.draft_model, binding=self.identity["fingerprint"], num_draft=self.external_policy.get("num_draft",4), **kwargs)

    def prompt_tokens(self, request: dict) -> list[int]:
        if "messages" not in request:
            return self.tokenizer.encode(request["prompt"], add_special_tokens=False)
        effort = request.get(
            "reasoning_effort",
            "high" if request.get("enable_thinking", False) else "none",
        )
        strengths = {
            "none": "low",
            "minimal": "low",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "high",
            "max": "high",
            "ultra": "high",
        }
        if effort not in strengths:
            raise ValueError("Unsupported Muse reasoning_effort")
        prompt = self.tokenizer.apply_chat_template(
            normalize_messages(request["messages"]),
            add_generation_prompt=True,
            tokenize=False,
            reasoning_strength=strengths[effort],
            tools=request.get("tools")
            if request.get("tool_choice") != "none"
            else None,
        )
        if request.get("enable_thinking", effort != "none") is False:
            # Muse's template ends at assistant; finish its user recipient header
            # to select a direct answer. This is a prompt policy, not a token budget.
            prompt += " to=user<|message|>"
        return self.tokenizer.encode(prompt, add_special_tokens=False)

    def output_parser(self, request):
        from .muse_glimmer_output import MuseOutputParser

        return MuseOutputParser(
            chat="messages" in request,
            tools=request.get("tools")
            if request.get("tool_choice") != "none"
            else None,
            stops=request.get("stop", ()),
            parallel_tool_calls=request.get("parallel_tool_calls", True),
        )

    def tool_constraint(self, request):
        from ..output import constrained_tool_choice
        from .muse_glimmer_output import constrained_tool_grammar

        if not constrained_tool_choice(request):
            return None
        return constrained_tool_grammar(
            request["tools"],
            request["tool_choice"],
            parallel_tool_calls=request.get("parallel_tool_calls", True),
        )

    def diagnostics(self):
        return {
            "architecture": "muse_glimmer",
            "cache_layout": self.layout,
            "sliding_layers": self.identity["sliding_layers"],
            "global_layers": self.identity["global_layers"],
            "speculation": "external-dflash2-implemented-unqualified" if self.draft_model is not None else "ordinary",
        }

    def close(self):
        pass
