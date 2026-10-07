"""CPU-safe GPT-OSS/Puzzle inspection and ordinary text execution.

Registration is deliberately separate.  Import and inspection never import MLX;
only constructing an adapter opens tensors or allocates model state.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from ..sampling_defaults import GENERATION_CONFIG, SamplingDefaults, VendorSampling
from ..process_env import PROCESS_NUMERICS, require_process_numerics


_COMMON = frozenset({
    Capability.TEXT, Capability.STREAMING, Capability.CONTINUOUS_BATCH,
    Capability.PREFIX_REUSE, Capability.APC_V2, Capability.LAYERED_CACHE,
    Capability.REASONING,
})
# The template's ``Reasoning:`` levels for the server's reasoning_effort
# values; "none" means thinking off (see ``thinking_enabled``).
_FINAL_SWITCH = "<|end|><|start|>assistant<|channel|>final<|message|>"
_NUDGE_TEXT = ("\n\nI've reasoned enough about this \u2014 let me stop here and give "
               "the final answer.\n\n")
_EFFORTS = {
    "none": "low", "minimal": "low", "low": "low", "medium": "medium",
    "high": "high", "xhigh": "high", "max": "high", "ultra": "high",
}


def descriptor_for(model_type: str) -> ModelDescriptor:
    if model_type not in {"gpt_oss", "gpt_oss_puzzle"}:
        raise ValueError("unsupported GPT-OSS model type")
    puzzle = model_type == "gpt_oss_puzzle"
    return ModelDescriptor(
        model_type=model_type,
        family="gpt-oss-puzzle" if puzzle else "gpt-oss",
        variant="88b-ordinary" if puzzle else "20b-ordinary",
        state_planes=frozenset({StatePlane.ATTENTION_KV, StatePlane.RNG, StatePlane.TRANSCRIPT}),
        capabilities=_COMMON,
        cache_layout="gpt-oss-puzzle-per-layer-kv-v1" if puzzle else "gpt-oss-alternating-kv-v1",
        metadata={
            "execution": "mlx2.adapters.gpt_oss.GptOssPuzzleAdapter" if puzzle
            else "mlx2.adapters.gpt_oss.GptOssAdapter",
            "qualification": "pending",
            "scope": "ordinary text decode with harmony analysis/final channels; "
                     "no tool or speculative route",
        },
    )


GPT_OSS = descriptor_for("gpt_oss")
GPT_OSS_PUZZLE = descriptor_for("gpt_oss_puzzle")
# Both local generation_config.json files set do_sample=true and specify no
# temperature or truncation controls.  Explicit neutral sampling fills those
# omitted fields without inheriting mlx2's historical 0.7/0.8/20 profile.
GPT_OSS_SAMPLING = VendorSampling.single(
    SamplingDefaults(temperature=1.0, source=GENERATION_CONFIG,
                     note="do_sample=true; temperature omitted"),
    model="GPT-OSS 20B / GPT-OSS Puzzle 88B local artifacts",
)


def _json(path: Path) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate {key!r} in {path.name}")
            result[key] = value
        return result
    value = json.loads(path.read_text(), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _topology(config: dict, model_type: str) -> tuple[int, list[int | None]]:
    puzzle = model_type == "gpt_oss_puzzle"
    expected = {
        "model_type": model_type,
        "architectures": ["GptOssPuzzleForCausalLM"] if puzzle else ["GptOssForCausalLM"],
        "hidden_size": 2880, "intermediate_size": 2880, "vocab_size": 201088,
        "num_attention_heads": 64, "num_key_value_heads": 8, "head_dim": 64,
        "num_experts_per_tok": 4, "num_hidden_layers": 36 if puzzle else 24,
        "tie_word_embeddings": False,
    }
    if not puzzle:
        expected.update(num_local_experts=32, sliding_window=128)
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("GPT-OSS artifact topology does not match the local target")
    n = expected["num_hidden_layers"]
    if puzzle:
        blocks = config.get("block_configs")
        if not isinstance(blocks, list) or len(blocks) != n:
            raise ValueError("Puzzle requires one block config per layer")
        if any(
            not isinstance(block, dict)
            or block.get("num_local_experts") not in (64, 128)
            or block.get("sliding_window", object()) not in (128, 8192, None)
            for block in blocks
        ):
            raise ValueError("Puzzle per-layer expert/window topology is invalid")
        windows = [block["sliding_window"] for block in blocks]
    else:
        windows = [128 if i % 2 == 0 else None for i in range(n)]
    layer_types = ["full_attention" if window is None else "sliding_attention" for window in windows]
    if config.get("layer_types") != layer_types:
        raise ValueError("GPT-OSS layer order disagrees with window topology")
    return n, windows


def inspect_artifact(model_path: str | Path, *, expected: str | None = None) -> dict:
    path = Path(model_path).expanduser().resolve()
    config = _json(path / "config.json")
    model_type = config.get("model_type")
    if model_type not in {"gpt_oss", "gpt_oss_puzzle"} or (expected and model_type != expected):
        raise ValueError("artifact is not the requested GPT-OSS target")
    n, windows = _topology(config, model_type)
    weight_map = _json(path / "model.safetensors.index.json").get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("GPT-OSS requires a nonempty weight index")
    required = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    for layer in range(n):
        prefix = f"model.layers.{layer}."
        required.update(prefix + name for name in (
            "input_layernorm.weight", "post_attention_layernorm.weight",
            "self_attn.sinks", "mlp.router.weight", "mlp.router.bias",
            "self_attn.q_proj.weight", "self_attn.q_proj.bias",
            "self_attn.k_proj.weight", "self_attn.k_proj.bias",
            "self_attn.v_proj.weight", "self_attn.v_proj.bias",
            "self_attn.o_proj.weight", "self_attn.o_proj.bias",
        ))
        if model_type == "gpt_oss_puzzle":
            required.update(prefix + "mlp.experts." + name for name in (
                "gate_proj.weight", "gate_proj.scales", "gate_proj.bias",
                "up_proj.weight", "up_proj.scales", "up_proj.bias",
                "down_proj.weight", "down_proj.scales", "down_proj.bias",
            ))
        else:
            required.update(prefix + "mlp.experts." + name for name in (
                "gate_up_proj_blocks", "gate_up_proj_scales",
                "gate_up_proj_bias", "down_proj_blocks",
                "down_proj_scales", "down_proj_bias",
            ))
    if not required <= weight_map.keys():
        missing = sorted(required - weight_map.keys())
        raise ValueError(f"GPT-OSS indexed tensor topology is incomplete: {missing[0]}")
    if any(marker in key.lower() for key in weight_map for marker in ("mtp.", "draft.")):
        raise ValueError("GPT-OSS target index contains speculative tensors")
    names = sorted(set(weight_map.values()))
    records = []
    for name in names:
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts or Path(name).suffix != ".safetensors":
            raise ValueError("GPT-OSS index contains an unsafe shard path")
        item = path / name
        # Hub snapshots use links into their own blob store. Allow those links
        # within the same model repository, but reject a link to another tree.
        repository_root = path.parent.parent if path.parent.name == "snapshots" else path
        if not item.is_file() or not item.resolve().is_relative_to(repository_root):
            raise ValueError(f"missing or foreign GPT-OSS shard: {name}")
        stat = item.stat()
        records.append((name, stat.st_size, stat.st_mtime_ns))
    digest = hashlib.sha256()
    for name in ("config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
        item = path / name
        if item.is_file():
            digest.update(name.encode())
            digest.update(item.read_bytes())
    for record in records:
        digest.update(json.dumps(record).encode())
    return {
        "config": config, "weight_map": weight_map, "windows": windows,
        "identity": {"path": str(path), "fingerprint": digest.hexdigest(), "files": records},
        "has_mtp": False,
    }


def configure_environment() -> dict[str, str]:
    # Refuse an explicit TF32 value before the profile overwrites it
    # (sweep 1002 review item 6).
    require_process_numerics("the gpt-oss profile")
    profile = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", **PROCESS_NUMERICS}
    os.environ.update(profile)
    return profile


class _GptOssOrdinaryAdapter:
    default_route = "ordinary"
    expected_type: str
    # Whether the model answers well when the prompt opens the final channel
    # directly, skipping analysis.  Stock GPT-OSS does; the NAS-pruned Puzzle
    # model degenerates (NBSP/ellipsis runs) and must reason first.
    direct_final: bool = True
    reasoning_effort_semantics = "reasoning_strength"

    def __init__(self, model_path: str, *, execution_policy=None):
        from .process_globals import guarded_construction

        guarded_construction(
            self, lambda: self._init_gpt_oss(model_path, execution_policy=execution_policy)
        )

    def _init_gpt_oss(self, model_path: str, *, execution_policy=None):
        from .process_globals import claim_stock_moe

        if execution_policy not in (None, {}):
            raise ValueError("GPT-OSS supports only ordinary execution")
        artifact = inspect_artifact(model_path, expected=self.expected_type)
        self.identity = artifact["identity"]
        self.descriptor = descriptor_for(self.expected_type)
        self.layout = self.descriptor.cache_layout
        self.environment = configure_environment()
        # The biased experts skip the NAX gather, but every sorted gather
        # reads the rhs pad policy.
        claim_stock_moe(self, f"the {self.expected_type} adapter")
        config = artifact["config"]
        path = Path(self.identity["path"])

        import mlx.core as mx
        from mlx import nn
        from transformers import AutoTokenizer
        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper
        from ..runtime.ubc_evict import load_shards_evicting
        if self.expected_type == "gpt_oss_puzzle":
            from ..runtime.models.gpt_oss_puzzle import Model, ModelArgs
        else:
            from ..runtime.models.gpt_oss import Model, ModelArgs

        self.model = Model(ModelArgs.from_dict(config))
        files = [path / name for name in sorted(set(artifact["weight_map"].values()))]
        weights = self.model.sanitize(load_shards_evicting(files))
        quant = config.get("quantization")
        if quant is None:
            method = config.get("quantization_config", {}).get("quant_method")
            if method != "mxfp4":
                raise ValueError("GPT-OSS 20B requires MXFP4 quantization metadata")
            quant = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
        if any(key not in {"group_size", "bits", "mode"} and not isinstance(value, dict) for key, value in quant.items()):
            raise ValueError("invalid per-module GPT-OSS quantization metadata")
        def predicate(name, module):
            override = quant.get(name)
            if isinstance(override, dict):
                return override
            return hasattr(module, "to_quantized") and f"{name}.scales" in weights
        nn.quantize(self.model, group_size=quant["group_size"], bits=quant["bits"], mode=quant.get("mode", "affine"), class_predicate=predicate)
        self.model.load_weights(list(weights.items()), strict=True)
        self.model.eval()
        mx.eval(self.model.parameters())
        weights.clear()
        mx.clear_cache()
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        # Some tokenizer classes rebuild the pre-tokenizer instead of reading tokenizer.json.
        from ..runtime.tokenizer_integrity import repair_loaded_tokenizer

        self.pretokenizer_receipt = repair_loaded_tokenizer(tokenizer, path)
        generation = _json(path / "generation_config.json")
        configured_eos = generation.get("eos_token_id", config["eos_token_id"])
        eos_ids = configured_eos if isinstance(configured_eos, list) else [configured_eos]
        if not eos_ids or any(type(value) is not int or value < 0 for value in eos_ids):
            raise ValueError("GPT-OSS generation config has invalid EOS tokens")
        self.tokenizer = TokenizerWrapper(tokenizer, detokenizer_class=BPEStreamingDetokenizer,
                                          eos_token_ids=eos_ids)
        self.max_context = int(config["max_position_embeddings"])

    def thinking_enabled(self, request: dict) -> bool:
        """Whether this chat request shows the analysis channel.

        An explicit ``enable_thinking`` wins; otherwise a ``reasoning_effort``
        shows it, and the model's own default applies: a direct answer where
        the model supports one, reasoning where it does not.
        """
        if "messages" not in request:
            return False
        if request.get("enable_thinking") is not None:
            return bool(request["enable_thinking"])
        effort = request.get("reasoning_effort")
        if effort is not None:
            return effort != "none"
        return not self.direct_final

    def thinking_close_token_ids(self):
        """The harmony switch from analysis to the answer, or None.

        ``<|end|><|start|>assistant<|channel|>final<|message|>``: the thinking
        guard forces it token by token when a budget runs out, and grammars
        would defer to it.  Declared only when it encodes to one id per
        marker piece and decodes back exactly.
        """
        try:
            ids = [int(token) for token in self.tokenizer.encode(_FINAL_SWITCH, add_special_tokens=False)]
            if len(ids) != 6 or self.tokenizer.decode(ids) != _FINAL_SWITCH:
                return None
        except Exception:  # noqa: BLE001 - undeclared marker, not a load failure
            return None
        return tuple(ids)

    def thinking_nudge_token_ids(self):
        """The JUICE soft-landing nudge the thinking guard writes at its soft budget.

        The lab's Puzzle server injects this text once, at 80% of the budget,
        so the model wraps up its analysis before the forced final switch
        (mlx-uag puzzle_openai_server.py NUDGE_TEXT, commit 1c72610).
        """
        try:
            return tuple(int(token) for token in self.tokenizer.encode(_NUDGE_TEXT, add_special_tokens=False))
        except Exception:  # noqa: BLE001 - no nudge, not a load failure
            return ()

    def hidden_thinking_budget(self, request: dict) -> int:
        """Reasoning-token bound for a thinking-off request, or 0 for none.

        A model that cannot skip analysis still reasons with thinking off
        (hidden, ``Reasoning: low``).  Two thirds of what ``max_tokens`` leaves
        after the six-token final-channel switch go to it (at most 512), so a
        short-budget request still has room to answer: the guard switches to
        the final channel when the bound is reached.  Two thirds beat one half
        on Puzzle-88B, 137/150 vs 130/150 at max_tokens 32-128 (8 fixes, 1
        break; qualification/experiments/gpt-oss-harmony-20260930).
        """
        if self.direct_final or "messages" not in request or self.thinking_enabled(request):
            return 0
        limit = request.get("max_tokens")
        if not isinstance(limit, int) or limit <= 0:
            return 512
        switch = len(self.thinking_close_token_ids() or ()) or 6
        return max(1, min(512, (limit - switch) * 2 // 3))

    def _answers_directly(self, request: dict) -> bool:
        return self.direct_final and not self.thinking_enabled(request)

    def prompt_tokens(self, request: dict) -> list[int]:
        if request.get("tools"):
            raise ValueError("GPT-OSS tool calling is not implemented")
        if "messages" not in request:
            return self.tokenizer.encode(request["prompt"], add_special_tokens=False)
        effort = request.get("reasoning_effort")
        if effort is not None and effort not in _EFFORTS:
            raise ValueError("unsupported GPT-OSS reasoning_effort")
        level = None if effort is None else _EFFORTS[effort]
        if not self.thinking_enabled(request) and not self.direct_final:
            # Thinking off on a model that cannot skip analysis: reason
            # briefly and hide it (the parser drops the analysis channel).
            level = "low"
        kwargs = {} if level is None else {"reasoning_effort": level}
        tokens = list(self.tokenizer.apply_chat_template(
            request["messages"], add_generation_prompt=True, tokenize=True, **kwargs
        ))
        # The artifact template ends in an open ``<|start|>assistant`` header.
        if self._answers_directly(request):
            tokens += self.tokenizer.encode("<|channel|>final<|message|>", add_special_tokens=False)
        return tokens

    def output_parser(self, request):
        if request.get("tools"):
            raise ValueError("GPT-OSS tool calling is not implemented")
        from .gpt_oss_output import HarmonyOutputParser
        return HarmonyOutputParser(
            chat="messages" in request,
            show_reasoning=self.thinking_enabled(request),
            start_in_final=self._answers_directly(request),
            stops=request.get("stop", ()),
        )

    def profile_name(self, mtp):
        if mtp:
            raise ValueError("GPT-OSS MTP is not implemented")
        return f"{self.expected_type}-apcv2-ordinary"

    def execution_config(self, *, max_lanes, prefill_step):
        return {"persistent": True, "num_draft": 0, "rate_gate": False,
                "prefill_step_size": prefill_step, "segment_aware_live_tip": False,
                "segment_aware_cohort_size": max_lanes}

    def diagnostics(self):
        return {"architecture": self.expected_type, "cache_layout": self.layout,
                "qualification": "pending", "route": "ordinary"}

    def close(self):
        from .process_globals import release

        release(self)
        self.model = None
        self.tokenizer = None


class GptOssAdapter(_GptOssOrdinaryAdapter):
    expected_type = "gpt_oss"
    descriptor = GPT_OSS
    sampling_defaults = GPT_OSS_SAMPLING


class GptOssPuzzleAdapter(_GptOssOrdinaryAdapter):
    expected_type = "gpt_oss_puzzle"
    direct_final = False
    descriptor = GPT_OSS_PUZZLE
    sampling_defaults = GPT_OSS_SAMPLING
