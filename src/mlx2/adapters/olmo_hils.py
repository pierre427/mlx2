"""HiLS Attention 7B candidate; its landmark cache is not APCv2 qualified."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

from ..contracts import Capability, ModelDescriptor, StatePlane
from ..process_env import PROCESS_NUMERICS, require_process_numerics
from .artifact_paths import shard_within_artifact
from .ordinary_text import OrdinaryTextAdapter

LIVE_CACHE_LAYOUT = "hils-landmark-custom-cache-v1"


DESCRIPTOR = ModelDescriptor(
    model_type="olmo_hils", family="hils-attention", variant="7b-ordinary-b1",
    state_planes=frozenset({StatePlane.ATTENTION_KV, StatePlane.RNG, StatePlane.TRANSCRIPT}),
    capabilities=frozenset({Capability.TEXT, Capability.STREAMING}),
    cache_layout=None,
    metadata={"execution": "mlx2.adapters.olmo_hils.OlmoHiLSAdapter",
              "qualification": "pending", "scope": "plain text batch width one; no APCv2 publication"},
)


def inspect_artifact(model_path: str | Path) -> dict:
    path = Path(model_path).expanduser().resolve()
    config = json.loads((path / "config.json").read_text())
    if not isinstance(config, dict):
        raise TypeError("HiLS config must be an object")
    expected = {
        "model_type": "olmo_hils", "architectures": ["HiLSForCausalLM"],
        "hidden_size": 4096, "intermediate_size": 11008,
        "num_hidden_layers": 32, "num_attention_heads": 32,
        "num_key_value_heads": 32, "vocab_size": 100278,
        "max_position_embeddings": 131072, "sliding_window": 512,
        "hils_sliding_window": 512, "chunk_size": 64, "hils_topk": 32,
        "full_attn_interleave": 4, "layerwise_qk_norm": True,
        "layerwise_lmkq_norm": True, "apply_hils_rope": True,
        "enable_inrange_rope": True, "rope_context_length": 8192,
        "rope_period_multiplier": 2.0, "tie_word_embeddings": False,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("HiLS 7B landmark topology does not match")
    if config.get("num_swa_layers") not in (None, 0):
        raise ValueError("unsupported HiLS layer schedule")
    quant = config.get("quantization")
    if quant is not None and quant != {"group_size": 64, "bits": 6, "mode": "affine"}:
        raise ValueError("unsupported HiLS quantization")
    index = json.loads((path / "model.safetensors.index.json").read_text())
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("HiLS requires an indexed checkpoint")
    required = {"model.embed_tokens.weight", "model.lmk_embed", "model.norm.weight", "lm_head.weight"}
    for layer in range(32):
        prefix = f"model.layers.{layer}."
        required.update(prefix + suffix for suffix in (
            "self_attn.q_proj.weight", "self_attn.k_proj.weight",
            "self_attn.v_proj.weight", "self_attn.o_proj.weight",
            "self_attn.q_norm.weight", "self_attn.k_norm.weight",
            "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
        ))
        if layer % 4 == 3:
            required.update({prefix + "self_attn.lmk_q_proj.0.weight",
                             prefix + "self_attn.lmk_q_proj.1.weight",
                             prefix + "self_attn.lmk_q_norm.weight"})
    if not required <= weight_map.keys():
        raise ValueError("HiLS indexed landmark tensor topology is incomplete")
    names = sorted(set(weight_map.values()))
    records = []
    for name in names:
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts or Path(name).suffix != ".safetensors":
            raise ValueError("unsafe HiLS shard path")
        item = path / name
        if not item.is_file() or not shard_within_artifact(path, item.resolve()):
            raise ValueError(f"missing HiLS shard: {name}")
        stat = item.stat(); records.append((name, stat.st_size, stat.st_mtime_ns))
    digest = hashlib.sha256()
    for name in ("config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json"):
        item = path / name
        if item.is_file():
            digest.update(name.encode()); digest.update(item.read_bytes())
    for record in records:
        digest.update(json.dumps(record).encode())
    return {"config": config, "weight_map": weight_map, "quantized": quant is not None,
            "identity": {"path": str(path), "fingerprint": digest.hexdigest(), "files": records},
            "hils_layers": 8, "swa_layers": 24, "apcv2_qualified": False}


class OlmoHiLSAdapter(OrdinaryTextAdapter):
    descriptor = DESCRIPTOR
    profile = "olmo-hils-7b-b1-ordinary"

    def __init__(self, model_path: str, *, execution_policy=None):
        if execution_policy not in (None, {}):
            raise ValueError("HiLS supports ordinary B1 execution only")
        artifact = inspect_artifact(model_path)
        self.identity = artifact["identity"]
        self.config = artifact["config"]
        self.layout = LIVE_CACHE_LAYOUT
        # Refuse an explicit TF32 value before the profile overwrites it.
        require_process_numerics("the OLMo HiLS profile")
        self.environment = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", **PROCESS_NUMERICS}
        os.environ.update(self.environment)
        path = Path(self.identity["path"])
        import mlx.core as mx
        from mlx import nn
        from transformers import AutoTokenizer

        from ..runtime.models.olmo_hils import Model, ModelArgs
        from ..runtime.tokenizer_utils import BPEStreamingDetokenizer, TokenizerWrapper
        from ..runtime.ubc_evict import load_shards_evicting
        self.model = Model(ModelArgs.from_dict(self.config))
        files = [path / name for name in sorted(set(artifact["weight_map"].values()))]
        weights = load_shards_evicting(files)
        quant = self.config.get("quantization")
        if quant:
            nn.quantize(self.model, group_size=quant["group_size"], bits=quant["bits"],
                        mode=quant["mode"], class_predicate=lambda name, module:
                        hasattr(module, "to_quantized") and f"{name}.scales" in weights)
        self.model.load_weights(list(weights.items()), strict=True)
        self.model.eval(); mx.eval(self.model.parameters())
        weights.clear(); mx.clear_cache()
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        # Some tokenizer classes rebuild the pre-tokenizer instead of reading tokenizer.json.
        from ..runtime.tokenizer_integrity import repair_loaded_tokenizer

        self.pretokenizer_receipt = repair_loaded_tokenizer(tokenizer, path)
        self.tokenizer = TokenizerWrapper(tokenizer, detokenizer_class=BPEStreamingDetokenizer,
                                          eos_token_ids=[int(self.config["eos_token_id"])])
        self.max_context = int(self.config["max_position_embeddings"])

    @staticmethod
    def lane_projection_groups():
        """Offered to the lane installer; stacked only under ``declared_groups``."""
        from ..runtime.models.olmo_hils import lane_projection_groups

        return lane_projection_groups()

    def prompt_tokens(self, request):
        if "messages" in request or request.get("tools"):
            raise ValueError("HiLS candidate accepts plain text prompts only")
        return self.tokenizer.encode(request["prompt"], add_special_tokens=False)

    def execution_config(self, *, max_lanes, prefill_step):
        if max_lanes != 1:
            raise ValueError("HiLS landmark bookkeeping supports one lane")
        return super().execution_config(max_lanes=max_lanes, prefill_step=prefill_step)

    def prefill_step_default(self):
        """Fill an integral eight-chunk/512-inserted-row landmark window.

        A 64-row HiLS chunk contains 63 real tokens and one inserted landmark.
        The released 512-row window therefore spans 8 * 63 = 504 real tokens.
        This is a family geometry choice, not a performance qualification.
        """
        chunk = int(self.config["chunk_size"])
        window = int(self.config["hils_sliding_window"])
        if chunk < 2 or window < chunk or window % chunk:
            raise ValueError("HiLS prefill geometry is not whole landmark chunks")
        return (chunk - 1) * (window // chunk)

    def exact_prefix_cascade_contract(self):
        """Declare the request-private ordinary-S1 continuation boundary."""

        return {
            "schema": "mlx2.exact-prefix-cascade-contract.v1",
            "family": "hils-attention",
            "verification_order": "longest_first",
            "invalid_sibling_pruning": True,
            "accepted_prefix_state": "authoritative_ordinary_s1_live",
            "shared_prefix_reuse": "suffix_only_after_hils_live_cache_gate",
            "cache_layout": LIVE_CACHE_LAYOUT,
            "state_planes": ("attention_kv", "rng", "transcript"),
            "hils_layers": 8,
            "sliding_attention_layers": 24,
            "inserted_chunk_size": 64,
            "sliding_window": 512,
            "request_private_only": True,
            "apcv2_publication": False,
            "transactional_multirow_state_reuse": False,
            "ordinary_reference_preserved": True,
            "implemented": True,
            "implementation_scope": "planner_adapter_contract_and_live_s1_gate",
            "qualified": False,
            "selected": False,
            "observed_used": False,
        }

    def exact_shared_prefix_geometry(self, caches, checkpoint_position):
        """Validate one live authoritative HiLS cache boundary.

        This gate describes state already produced by ordinary single-token
        execution.  It does not authorize rollback, multirow verification, or
        serialization into APCv2.
        """

        if type(checkpoint_position) is not int or checkpoint_position < 1:
            raise ValueError("checkpoint_position must be a positive integer")
        caches = tuple(caches)
        layers = int(self.config["num_hidden_layers"])
        chunk = int(self.config["chunk_size"])
        window = int(self.config["hils_sliding_window"])
        heads = int(self.config["num_attention_heads"])
        head_dim = int(self.config["hidden_size"]) // heads
        if len(caches) != layers:
            return {
                "eligible": False,
                "reason": "cache_layer_count_mismatch",
                "expected_layers": layers,
                "actual_layers": len(caches),
            }
        expected_inserted = checkpoint_position + checkpoint_position // (chunk - 1)
        expected_pooled = expected_inserted // chunk
        for index, cache in enumerate(caches):
            hils = index % int(self.config["full_attn_interleave"]) == 3
            wanted = "HiLSCache" if hils else "SWABandCache"
            actual_type = type(cache)
            actual = f"{actual_type.__module__}.{actual_type.__qualname__}"
            if actual != f"mlx2.runtime.models.olmo_hils.{wanted}":
                return {
                    "eligible": False,
                    "reason": "cache_plane_type_mismatch",
                    "layer": index,
                    "expected": wanted,
                    "actual": actual_type.__name__,
                }
            if (
                int(cache.offset) != checkpoint_position
                or int(cache.ins_offset) != expected_inserted
                or int(cache.chunk_size) != chunk
            ):
                return {
                    "eligible": False,
                    "reason": "inserted_coordinate_mismatch",
                    "layer": index,
                }
            start = int(cache.start_pos)
            rows = expected_inserted - start
            key_shape = tuple(getattr(getattr(cache, "keys", None), "shape", ()))
            value_shape = tuple(getattr(getattr(cache, "values", None), "shape", ()))
            expected_prefix = (1, heads)
            if (
                start < 0
                or rows < 1
                or len(key_shape) != 4
                or len(value_shape) != 4
                or key_shape[:2] != expected_prefix
                or value_shape[:2] != expected_prefix
                or key_shape[2] < rows
                or value_shape[2] < rows
                or key_shape[3] != head_dim
                or value_shape[3] != head_dim
            ):
                return {
                    "eligible": False,
                    "reason": "cache_tensor_geometry_mismatch",
                    "layer": index,
                }
            if hils:
                landmark_shape = tuple(
                    getattr(getattr(cache, "lmk_k", None), "shape", ())
                )
                prior_shape = tuple(
                    getattr(getattr(cache, "prior_b", None), "shape", ())
                )
                if (
                    start != 0
                    or int(cache.num_pooled_chunks) != expected_pooled
                    or (
                        expected_pooled
                        and (
                            landmark_shape != (1, expected_pooled, heads, head_dim)
                            or prior_shape != (1, expected_pooled, heads)
                        )
                    )
                ):
                    return {
                        "eligible": False,
                        "reason": "landmark_pool_geometry_mismatch",
                        "layer": index,
                    }
            elif int(cache.window) != window:
                return {
                    "eligible": False,
                    "reason": "sliding_window_geometry_mismatch",
                    "layer": index,
                }
        return {
            "eligible": True,
            "authority": "live_authoritative_ordinary_s1",
            "cache_layout": LIVE_CACHE_LAYOUT,
            "checkpoint_position": checkpoint_position,
            "inserted_position": expected_inserted,
            "recompute_common_tokens": False,
            "suffix_only": True,
            "rollback_authorized": False,
            "multirow_verify": False,
            "publishable": False,
        }

    def plan_exact_prefix_cascade(
        self, paths, accepted_prefix=(), *, attempted=(), state_binding=None
    ):
        """Plan one suffix only after binding it to exact live S1 state."""

        from ..runtime.exact_prefix_cascade import next_cascade_stage

        stage = next_cascade_stage(paths, accepted_prefix, attempted=attempted)
        if stage is None or not stage.accepted_prefix:
            return stage
        if not isinstance(state_binding, Mapping):
            raise TypeError("HiLS shared-prefix reuse requires a state binding")
        checkpoint = state_binding.get("checkpoint_position")
        expected_revision = getattr(self, "identity", {}).get("fingerprint")
        failures = []
        if state_binding.get("execution_domain") != "ordinary_s1":
            failures.append("execution domain")
        if state_binding.get("cache_layout") != LIVE_CACHE_LAYOUT:
            failures.append("cache layout")
        if state_binding.get("state_revision") != expected_revision:
            failures.append("state revision")
        if state_binding.get("checkpoint_kind") != "live_authoritative_exact":
            failures.append("checkpoint kind")
        if type(checkpoint) is not int or checkpoint < len(stage.accepted_prefix):
            failures.append("checkpoint position")
        if tuple(state_binding.get("transcript_tail", ())) != stage.accepted_prefix:
            failures.append("transcript tail")
        if state_binding.get("rng_state") != "authoritative_after_prefix":
            failures.append("RNG state")
        if state_binding.get("batch_size") != 1:
            failures.append("batch size")
        if failures:
            raise ValueError(
                "HiLS exact-prefix state binding differs: " + ", ".join(failures)
            )
        geometry = self.exact_shared_prefix_geometry(
            state_binding.get("caches", ()), checkpoint
        )
        if geometry.get("eligible") is not True:
            raise ValueError(
                "HiLS exact-prefix cache differs: " + str(geometry.get("reason"))
            )
        return stage

    def diagnostics(self):
        return {
            "route": "ordinary",
            "qualification": "pending",
            "prefix_candidate_verification": self.exact_prefix_cascade_contract(),
        }

    def close(self):
        self.model = None
        self.tokenizer = None
