"""Nemotron 3 Super 120B-A12B 5-bit hybrid/MTP decode adapter.

Inspection is CPU-only and reads safetensors headers, never tensor payloads.
Model construction is separate; production route selection needs a qualification receipt.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from ..process_env import (
    PROCESS_NUMERICS,
    clear_inherited_profile,
    require_process_numerics,
)
from ..sampling_defaults import GENERATION_CONFIG, SamplingDefaults, VendorSampling
from .artifact_paths import shard_within_artifact
from .flash_next import FlashNextAdapter

CACHE_LAYOUT = "nemotron3-super-hybrid-mamba-kv-v1"
EXACT_PREFIX_SOURCE_REVISION = "f6415871606005626a2ac36c0c5630d5b5a8568e"
SAMPLING = VendorSampling.single(
    SamplingDefaults(temperature=1.0, top_p=0.95, source=GENERATION_CONFIG),
    model="NVIDIA-Nemotron-3-Super-120B-A12B-5bit-MTP",
)
DESCRIPTOR = ModelDescriptor(
    model_type="nemotron_h",
    family="nemotron3-super-120b-a12b",
    variant="5bit-mtp",
    state_planes=frozenset({StatePlane.ATTENTION_KV, StatePlane.RECURRENT, StatePlane.RNG, StatePlane.TRANSCRIPT, StatePlane.DRAFT}),
    capabilities=frozenset({Capability.TEXT, Capability.STREAMING, Capability.TOOLS, Capability.REASONING, Capability.CONTINUOUS_BATCH, Capability.PREFIX_REUSE, Capability.APC_V2, Capability.LAYERED_CACHE, Capability.GRAMMAR, Capability.MTP}),
    cache_layout=CACHE_LAYOUT,
    metadata={"execution": "mlx2.adapters.nemotron3_super.Nemotron3SuperAdapter", "qualification": "pending", "scope": "text-only", "embedded_mtp": "implemented-candidate"},
)


def _shard_header(path: Path) -> dict:
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"truncated safetensors shard: {path.name}")
        header_size = struct.unpack("<Q", raw)[0]
        if header_size > 64 * 1024 * 1024 or header_size > path.stat().st_size - 8:
            raise ValueError(f"invalid safetensors header: {path.name}")
        header = json.loads(f.read(header_size))
    if not isinstance(header, dict):
        raise TypeError(f"invalid safetensors header: {path.name}")
    payload_size = path.stat().st_size - 8 - header_size
    spans = []
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start <= end <= payload_size):
            raise ValueError(f"incomplete safetensors payload: {path.name}:{key}")
        spans.append((start, end))
    if not spans or min(spans)[0] != 0 or max(spans)[1] != payload_size:
        raise ValueError(f"incomplete safetensors payload: {path.name}")
    if any(left[1] != right[0] for left, right in zip(sorted(spans), sorted(spans)[1:])):
        raise ValueError(f"safetensors payload has gaps or overlaps: {path.name}")
    return header


def inspect_artifact(model_path: str | Path) -> dict:
    path = Path(model_path).expanduser().resolve()
    config = json.loads((path / "config.json").read_text())
    expected = {
        "model_type": "nemotron_h", "num_hidden_layers": 88,
        "hidden_size": 4096, "vocab_size": 131072,
        "num_attention_heads": 32, "num_key_value_heads": 2, "head_dim": 128,
        "mamba_num_heads": 128, "mamba_head_dim": 64,
        "ssm_state_size": 128, "conv_kernel": 4, "n_groups": 8,
        "n_routed_experts": 512, "num_experts_per_tok": 22,
        "moe_latent_size": 1024, "moe_intermediate_size": 2688,
        "num_nextn_predict_layers": 1, "mtp_hybrid_override_pattern": "*E",
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("artifact topology does not match Nemotron 3 Super 120B-A12B")
    pattern = config.get("hybrid_override_pattern")
    if not isinstance(pattern, str) or len(pattern) != 88 or any(pattern.count(k) != n for k, n in (("M", 40), ("E", 40), ("*", 8))):
        raise ValueError("artifact hybrid layer order is invalid")
    if config.get("quantization") != {"group_size": 64, "bits": 5, "mode": "affine"}:
        raise ValueError("artifact is not affine 5-bit")
    if config.get("eos_token_id") != [2, 11]:
        raise ValueError("Nemotron 3 Super EOS token contract changed")
    tokenizer_config = json.loads((path / "tokenizer_config.json").read_text())
    if tokenizer_config.get("tool_parser_type") != "qwen3_coder":
        raise ValueError("Nemotron 3 Super tool parser contract changed")
    template = (path / "chat_template.jinja").read_text()
    if not all(marker in template for marker in ("<tool_call>", "<function=", "<parameter=", "enable_thinking", "</think>")):
        raise ValueError("Nemotron 3 Super chat template contract changed")
    index = json.loads((path / "model.safetensors.index.json").read_text())
    weights = index.get("weight_map")
    if not isinstance(weights, dict) or len(weights) != 1511:
        raise ValueError("Nemotron 3 Super weight index is incomplete")
    mtp_tensor_count = sum(key.startswith("mtp.") for key in weights)
    if mtp_tensor_count != 40:
        raise ValueError("Nemotron 3 Super embedded MTP index is incomplete")
    names = sorted(set(weights.values()))
    if len(names) != 17:
        raise ValueError("Nemotron 3 Super requires 17 complete shards")
    required = {
        "backbone.embeddings.weight", "backbone.layers.0.mixer.conv1d.weight",
        "backbone.layers.87.norm.weight", "lm_head.weight",
        "mtp.layers.0.eh_proj.weight", "mtp.layers.1.mixer.switch_mlp.fc1.weight",
    }
    if not required <= weights.keys():
        raise ValueError("required backbone or MTP tensors are absent")
    for i, block in enumerate(pattern):
        prefix = f"backbone.layers.{i}.mixer."
        key = {"M": "conv1d.weight", "E": "switch_mlp.fc1.weight", "*": "q_proj.weight"}[block]
        if prefix + key not in weights:
            raise ValueError(f"layer {i} tensor layout does not match hybrid pattern")
    digest = hashlib.sha256()
    for filename in ("config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json"):
        item = path / filename
        if not item.is_file():
            raise ValueError(f"missing artifact metadata: {filename}")
        digest.update(filename.encode()); digest.update(item.read_bytes())
    records = []
    observed = set()
    for name in names:
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("weight shard paths must stay within the artifact")
        item = (path / name).resolve()
        if not shard_within_artifact(path, item) or not item.is_file():
            raise ValueError(f"missing or escaped weight shard: {name}")
        header = _shard_header(item)
        tensor_names = set(header) - {"__metadata__"}
        indexed = {key for key, shard in weights.items() if shard == name}
        if tensor_names != indexed or observed & tensor_names:
            raise ValueError(f"safetensors index/header mismatch: {name}")
        observed.update(tensor_names)
        stat = item.stat()
        record = (name, stat.st_size, stat.st_mtime_ns)
        records.append(record); digest.update(json.dumps(record).encode())
    return {"config": config, "weight_map": weights, "has_mtp": True,
            "mtp_tensor_count": mtp_tensor_count,
            "identity": {"path": str(path), "fingerprint": digest.hexdigest(), "files": records}}


def configure_environment() -> dict[str, str]:
    """Pin the serving profile and drop inherited lab switches.

    Same hygiene as every other adapter: an ``MLX_QWEN*``/``MLX_LM_*``/
    ``MLXUAG_*``/``MLX_GDN_*`` variable left in the shell by an experiment
    must not silently change this model's serving path.  The profile itself
    (and so the qualification identity) is unchanged.
    """
    # Refuse an explicit TF32 value before the profile overwrites it
    # (sweep 1002 review item 6).
    require_process_numerics("the Nemotron-3 Super profile")
    profile = {
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        **PROCESS_NUMERICS}
    clear_inherited_profile(("MLX_QWEN", "MLX_LM_", "MLXUAG_", "MLX_GDN_"))
    os.environ.update(profile)
    return profile

class Nemotron3SuperAdapter(FlashNextAdapter):
    descriptor = DESCRIPTOR
    sampling_defaults = SAMPLING
    artifact_inspector = staticmethod(inspect_artifact)
    # This adapter is not well tuned yet. Its exact B1 MTP verifier is slower
    # than ordinary decode at the tested 8K and 32K contexts; batched MTP,
    # cached-prefix parity, and longer contexts still need qualification.
    default_route = "ordinary"
    # Preserve the source-bound Nemotron serving profile.  The generic
    # prompt-length schedule is used only by adapters that return ``None``.
    default_prefill_step = 2048
    # The template renders a turn as ``<think>\n...\n</think>\n`` + content
    # (content trimmed), and the model writes ``</think>\n\n`` before its
    # answer.  Those newlines are template structure, not the answer.
    think_close_separator = "\r\n"

    @staticmethod
    def spomin_backend(model, prompt_cache):
        return None

    def __init__(self, model_path: str, *, execution_policy=None):
        # Overrides FlashNextAdapter.__init__, so it owns its own claim.
        from .process_globals import guarded_construction

        guarded_construction(
            self, lambda: self._init_nemotron(model_path, execution_policy=execution_policy)
        )

    def _init_nemotron(self, model_path: str, *, execution_policy=None):
        from .process_globals import claim_stock_moe

        if execution_policy is not None and (
            not isinstance(execution_policy, dict)
            or set(execution_policy) - {"num_draft"}
        ):
            raise ValueError("Nemotron 3 Super execution policy supports only num_draft")
        from .mtp_depth_cap import validate_self_mtp_num_draft

        self._num_draft = validate_self_mtp_num_draft(
            (execution_policy or {}).get("num_draft", 3)
        )
        if self._num_draft > 3:
            raise ValueError("Nemotron 3 Super MTP depth above 3 is not qualified")
        artifact = inspect_artifact(model_path)
        self.identity = artifact["identity"]
        self.layout = CACHE_LAYOUT
        self.environment = configure_environment()
        claim_stock_moe(self, "the Nemotron 3 Super adapter")
        path = Path(self.identity["path"])
        import mlx.core as mx
        from mlx import nn
        from transformers import AutoTokenizer

        from ..runtime.models.nemotron_h import Model, ModelArgs
        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper
        from ..runtime.ubc_evict import load_shards_evicting

        config = dict(artifact["config"])
        self.model = Model(ModelArgs.from_dict(config))
        files = [path / name for name in sorted(set(artifact["weight_map"].values()))]
        weights = self.model.sanitize(load_shards_evicting(files))
        quant = config["quantization"]
        nn.quantize(self.model, group_size=quant["group_size"], bits=quant["bits"],
                    mode=quant["mode"], class_predicate=lambda name, module: hasattr(module, "to_quantized") and f"{name}.scales" in weights)
        self.model.load_weights(list(weights.items()), strict=True)
        self.model.eval(); mx.eval(self.model.parameters())
        weights.clear(); mx.clear_cache()
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        # Some tokenizer classes rebuild the pre-tokenizer instead of reading tokenizer.json.
        from ..runtime.tokenizer_integrity import repair_loaded_tokenizer

        self.pretokenizer_receipt = repair_loaded_tokenizer(tokenizer, path)
        self.tokenizer = TokenizerWrapper(tokenizer, detokenizer_class=BPEStreamingDetokenizer,
                                          eos_token_ids=[2, 11])
        self.max_context = config["max_position_embeddings"]

    # The inherited Flash-Next renderer defaults thinking off; this family
    # defaults it on, so the incremental tokenizer cache renders through the
    # same default as render_prompt / prompt_tokens (sweep 2026-10-06 G2-01).
    incremental_tokenizer_renderer_revision = "nemotron-h-renderer-v1"

    @staticmethod
    def _thinking_request(request):
        return {**request, "enable_thinking": bool(request.get("enable_thinking", True))}

    def thinking_enabled(self, request):
        return self._thinking_request(request)["enable_thinking"]

    @staticmethod
    def render_incremental_prompt(tokenizer, request: dict) -> str:
        """Render against the cache's immutable tokenizer/template snapshot."""
        if "messages" in request:
            from .flash_next import chat_template
            return chat_template(
                tokenizer, Nemotron3SuperAdapter._thinking_request(request), tokenize=False
            )
        return request["prompt"]

    def prompt_tokens(self, request):
        if "messages" in request:
            from .flash_next import chat_template
            return chat_template(self.tokenizer, {**request, "enable_thinking": self.thinking_enabled(request)}, tokenize=True)
        return self.tokenizer.encode(request["prompt"], add_special_tokens=False)

    def render_prompt(self, request):
        if "messages" in request:
            from .flash_next import chat_template
            return chat_template(self.tokenizer, {**request, "enable_thinking": self.thinking_enabled(request)}, tokenize=False)
        return request["prompt"]

    def output_parser(self, request):
        return super().output_parser({**request, "enable_thinking": self.thinking_enabled(request)})

    def profile_name(self, mtp):
        return (
            f"nemotron3-super-5bit-apcv2-mtp{self._num_draft}"
            if mtp else "nemotron3-super-5bit-apcv2-ordinary"
        )

    def prefill_step_default(self):
        """Family-owned conservative chunk; an explicit operator value wins."""

        return int(type(self).default_prefill_step)

    def exact_prefix_cascade_contract(self):
        """Declare the narrow exact-state boundary for Nemotron-H cascades."""

        return {
            "schema": "mlx2.exact-prefix-cascade-contract.v1",
            "verification_order": "longest_first",
            "invalid_sibling_pruning": True,
            "accepted_prefix_state": "b1_tokenwise_hybrid_transaction",
            "shared_prefix_reuse": "committed_target_cache_clone",
            "common_tokens_recomputed": False,
            "request_private_only": True,
            "apcv2_publication": False,
            "mtp_cache_reuse": False,
            "twotower_reuse": False,
            "implemented": True,
            "implementation_scope": "planner_and_adapter_state_primitive",
            "serving_route_implemented": False,
            "http_request_selection_implemented": False,
            "state_binding": (
                "artifact_source_layout_request_owner_checkpoint_planes_b1"
            ),
            "target_observation": (
                "unimplemented_sampler_processors_rng_history"
            ),
            "qualified": False,
            "selected": False,
            "observed_used": False,
        }

    def plan_exact_prefix_cascade(self, paths, accepted_prefix=(), *, attempted=()):
        from ..runtime.exact_prefix_cascade import next_cascade_stage

        return next_cascade_stage(paths, accepted_prefix, attempted=attempted)

    def verify_exact_prefix_path(self, cache, tokens, *, capture_layers=()):
        """Run the adapter-owned B1 primitive for a direct or serving caller."""

        from ..runtime.nemotron_prefix_reuse import verify_longest_prefix

        return verify_longest_prefix(
            self.model, cache, tokens, capture_layers=capture_layers
        )

    def exact_prefix_reuse_geometry(self, cache, *, state_binding, request_id):
        """Validate authoritative ownership and live exact B1 geometry."""

        from ..runtime.exact_prefix_state_binding import (
            validate_exact_prefix_state_binding,
        )
        from ..runtime.nemotron_prefix_reuse import exact_prefix_reuse_geometry

        expected_fingerprint = getattr(self, "identity", {}).get("fingerprint")
        expected_planes = frozenset({"attention_kv", "recurrent"})
        common = {
            "expected_artifact_fingerprint": expected_fingerprint,
            "expected_source_revision": EXACT_PREFIX_SOURCE_REVISION,
            "expected_cache_layout": self.layout,
            "expected_request_id": request_id,
            "expected_state_planes": expected_planes,
            "expected_execution_domain": "ordinary_target_b1",
        }
        validate_exact_prefix_state_binding(state_binding, **common)
        geometry = exact_prefix_reuse_geometry(
            self.model,
            cache,
            state_revision=state_binding["artifact_fingerprint"],
            cache_layout=state_binding["cache_layout"],
        )
        validate_exact_prefix_state_binding(
            state_binding,
            live_checkpoint_position=geometry.receipt()["position"],
            **common,
        )
        return geometry

    def execution_config(self, *, max_lanes, prefill_step):
        return {"persistent": True, "num_draft": self._num_draft, "rate_gate": False,
                "prefill_step_size": prefill_step, "segment_aware_live_tip": True,
                "segment_aware_cohort_size": max_lanes}

    def cache_budget(self, *, mtp):
        from .nemotron3_super_memory import NemotronCacheBudget
        return NemotronCacheBudget.from_config(self.model.args, mtp=mtp)

    def diagnostics(self):
        return {"architecture": "nemotron-h-mamba-moe-gqa", "layout": self.layout,
                "mtp_head_present": True, "mtp_candidate": True}

    def close(self):
        from .process_globals import release

        release(self)
        self.model = None
