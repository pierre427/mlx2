"""Artifact inspection and ordinary text serving for SenseNova Loop."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from typing import ClassVar

from ..contracts import Capability, ModelDescriptor, StatePlane
from ..sampling_defaults import GENERATION_CONFIG, SamplingDefaults, VendorSampling
from .ordinary_artifact import inspect_indexed_artifact
from .ordinary_text import OrdinaryTextAdapter

CACHE_LAYOUT = "sensenova-loop-visit-segments-v1"
DESCRIPTOR = ModelDescriptor(
    model_type="sensenova_loop",
    family="sensenova-loop",
    variant="9b-bf16-text",
    state_planes=frozenset(
        {
            StatePlane.ATTENTION_KV,
            StatePlane.RECURRENT,
            StatePlane.RNG,
            StatePlane.TRANSCRIPT,
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
        }
    ),
    cache_layout=CACHE_LAYOUT,
    metadata={
        "execution": "mlx2.adapters.sensenova_loop.SenseNovaLoopAdapter",
        "qualification": "pending",
        "scope": "text-only",
    },
)


def execution_plan(config):
    text = config.get("text_config")
    loop = config.get("loop_settings")
    mhc = config.get("mhc")
    if not all(isinstance(part, dict) for part in (text, loop, mhc)):
        raise ValueError(
            "SenseNova requires explicit text, loop, and mHC configurations"
        )
    layers = text.get("num_hidden_layers")
    interval = text.get("full_attention_interval")
    start, end, count = (loop.get(key) for key in ("start", "end", "eval_count"))
    if (
        any(type(value) is not int for value in (layers, interval, start, end, count))
        or not 0 <= start < end <= layers
        or not 1 < interval <= layers
        or count < 1
        or loop.get("counts") != [count]
    ):
        raise ValueError(
            "SenseNova requires a fixed-count loop with a valid layer interval"
        )
    kinds = [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(layers)
    ]
    if text.get("layer_types") != kinds:
        raise ValueError("SenseNova requires the Qwen3.5 hybrid layer ordering")
    left, right, iterations = (mhc.get(key) for key in ("start", "end", "iterations"))
    scale = mhc.get("pre_scale", 1.0)
    phases = (0, start, end, layers)
    if (
        any(type(value) is not int for value in (left, right, iterations))
        or not any(a <= left < right <= b for a, b in pairwise(phases))
        or mhc.get("streams") != 2
        or mhc.get("fixed_identity") is not False
        or iterations < 1
        or type(scale) not in (int, float)
        or not math.isfinite(scale)
        or scale <= 0
        or set(mhc)
        - {"start", "end", "streams", "iterations", "fixed_identity", "pre_scale"}
    ):
        raise ValueError("unsupported SenseNova two-stream mHC configuration")
    return (*range(start), *(tuple(range(start, end)) * count), *range(end, layers))


def inspect_artifact(model_path: str | Path) -> dict:
    artifact = inspect_indexed_artifact(model_path)
    config = artifact["config"]
    if (
        config.get("model_type") != "sensenova_loop"
        or config.get("quantization") is not None
        or config.get("quantization_config") is not None
    ):
        raise ValueError("SenseNova ordinary text requires the original dense release")
    visits = execution_plan(config)
    text = config["text_config"]
    expected = {
        "model_type": "qwen3_5_text",
        "hidden_act": "silu",
        "attn_output_gate": True,
        "attention_bias": False,
        "tie_word_embeddings": False,
        "dtype": "bfloat16",
        "mamba_ssm_dtype": "float32",
    }
    rope = text.get("rope_parameters")
    if (
        any(text.get(key) != value for key, value in expected.items())
        or text.get("num_experts", 0)
        or text.get("mlp_only_layers")
        or not isinstance(rope, dict)
        or rope.get("rope_type") != "yarn"
    ):
        raise ValueError("unsupported SenseNova Qwen3.5 backbone")
    mhc = config["mhc"]
    weights = artifact["weight_map"]
    required = {
        "lm_head.weight",
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
    }
    required.update(
        f"model.language_model.layers.{i}.mhc_{branch}.{part}"
        for i in range(mhc["start"], mhc["end"])
        for branch in ("attn", "ffn")
        for part in ("weight", "bias")
    )
    if not required <= weights.keys():
        raise ValueError("SenseNova release is missing text or mHC weights")
    return {**artifact, "visits": visits, "has_mtp": False}


class SenseNovaLoopAdapter(OrdinaryTextAdapter):
    descriptor = DESCRIPTOR
    adapter_owned_default_policy_keys = frozenset({"compiled_mhc"})
    default_route_execution_policy: ClassVar = {"ordinary": {"compiled_mhc": True}}
    artifact_inspector = staticmethod(inspect_artifact)
    profile = "sensenova-loop-bf16-apcv2-ordinary"
    sampling_defaults = VendorSampling.single(
        SamplingDefaults(temperature=1.0, top_p=0.95, source=GENERATION_CONFIG),
        model="SenseNova-Loop-9B",
    )

    def __init__(self, model_path: str | Path, *, execution_policy=None):
        from .process_globals import guarded_construction

        guarded_construction(self, lambda: self._load(model_path, execution_policy))

    def _load(self, model_path, policy):
        from .qwen35_9b import configure_environment

        policy = {} if policy is None else policy
        if not isinstance(policy, dict) or set(policy) - {"compiled_mhc"}:
            raise ValueError(
                "SenseNova ordinary text accepts only the compiled_mhc policy"
            )
        self.compiled_mhc = policy.get(
            "compiled_mhc",
            self.default_route_execution_policy["ordinary"]["compiled_mhc"],
        )
        if type(self.compiled_mhc) is not bool:
            raise ValueError("compiled_mhc must be boolean")
        artifact = inspect_artifact(model_path)
        self.config = artifact["config"]
        self.identity = artifact["identity"]
        self.layout = CACHE_LAYOUT + (":compiled-mhc-v1" if self.compiled_mhc else "")
        self.descriptor = replace(DESCRIPTOR, cache_layout=self.layout)
        self.environment = configure_environment()

        import mlx.core as mx
        from transformers import AutoTokenizer

        from ..runtime.models.import_env import assert_profile_applied
        from ..runtime.models.sensenova_loop import Model
        from ..runtime.tokenizer_integrity import repair_loaded_tokenizer
        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper
        from ..runtime.ubc_evict import load_shards_evicting

        assert_profile_applied("the SenseNova Loop adapter")
        path = Path(self.identity["path"])
        tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=False
        )
        self.pretokenizer_receipt = repair_loaded_tokenizer(tokenizer, path)
        eos = json.loads((path / "generation_config.json").read_text())["eos_token_id"]
        self.tokenizer = TokenizerWrapper(
            tokenizer,
            detokenizer_class=BPEStreamingDetokenizer,
            eos_token_ids=[eos] if isinstance(eos, int) else eos,
        )
        self.max_context = int(self.config["text_config"]["max_position_embeddings"])

        self.model = Model(
            self.config, artifact["visits"], self.layout, compiled_mhc=self.compiled_mhc
        )
        files = {
            path / shard
            for name, shard in artifact["weight_map"].items()
            if name.startswith("model.language_model.") or name == "lm_head.weight"
        }
        weights = load_shards_evicting(sorted(files), sanitize=self.model.sanitize)
        self.model.load_weights(list(weights.items()), strict=True)
        self.model.eval()
        mx.eval(self.model.parameters())
        weights.clear()
        mx.clear_cache()

    @staticmethod
    def planning_cache_budget(config, *, mtp):
        from .qwen38_memory import Qwen38CacheBudget

        if mtp:
            raise ValueError("SenseNova ordinary text has no MTP cache")
        text = config["text_config"]
        budget = Qwen38CacheBudget.from_config(text, mtp=False)
        visits = execution_plan(config)
        attention = sum(text["layer_types"][i] == "full_attention" for i in visits)
        return replace(
            budget, attention_layers=attention, recurrent_layers=len(visits) - attention
        )

    def cache_budget(self, *, mtp):
        return self.planning_cache_budget(self.config, mtp=mtp)

    def diagnostics(self):
        return {
            **super().diagnostics(),
            "compiled_mhc": self.compiled_mhc,
            "physical_layers": len(self.model.layers),
            "layer_visits": len(self.model.visits),
            "attention_cache_slots": self.model.linear_visits.count(False),
            "recurrent_cache_slots": self.model.linear_visits.count(True),
        }
