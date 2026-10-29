"""Opt-in Nanbeige tuning contracts; parsing is independent of device state.

Original glue over shared mlx2 mechanisms; provenance/nanbeige43-tuning.json.
No option here grants qualification or changes ordinary defaults.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def source_identity():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def split_tuning_policy(value):
    if value is not None and not isinstance(value, dict):
        raise TypeError("Nanbeige execution policy must be an object")
    policy = dict(value or {})
    tuning = policy.pop("nanbeige", {})
    if not isinstance(tuning, dict):
        raise TypeError("nanbeige tuning policy must be an object")
    unknown = set(tuning) - {
        "kernels", "tree", "lean_greedy", "int8_prefill", "prefill_plan",
        "kv_reserve_tokens", "adaptive_pld",
        "draft_block_size", "draft_quantization", "draft_kernel", "draft_vocab", "q8_ast",
    }
    if unknown:
        raise ValueError(f"unknown Nanbeige tuning keys: {sorted(unknown)}")
    result = dict(tuning)
    for key in ("lean_greedy", "int8_prefill", "prefill_plan", "adaptive_pld", "q8_ast"):
        if key in result and type(result[key]) is not bool:
            raise ValueError(f"nanbeige.{key} must be boolean")
    if result.get("tree", "ordinary") not in ("ordinary", "packed", "implicit", "segmented", "shared"):
        raise ValueError("unsupported Nanbeige tree implementation")
    if result.get("q8_ast") and (result.get("tree", "ordinary") != "packed" or result.get("kv_reserve_tokens")):
        raise ValueError("q8_ast requires packed tree and no exact KV reservation")
    reserve = result.get("kv_reserve_tokens", 0)
    if type(reserve) is not int or not 0 <= reserve <= 8192:
        raise ValueError("nanbeige.kv_reserve_tokens must be 0..8192")
    if result.get("kernels") and policy.get("nanbeige_fused") is not True:
        raise ValueError("Nanbeige kernel candidates require nanbeige_fused=true")
    if result.get("tree", "ordinary") != "ordinary" and not policy.get("draft_model"):
        raise ValueError("Nanbeige packed tree requires an explicit drafter")
    if result.get("adaptive_pld") and policy.get("draft_model") not in ("pld", "prompt_lookup"):
        raise ValueError("Nanbeige adaptive PLD requires a prompt-lookup drafter")
    draft_options = set(result) & {"draft_block_size", "draft_quantization", "draft_kernel", "draft_vocab"}
    if draft_options:
        if policy.get("draft_model") in (None, "pld", "prompt_lookup"):
            raise ValueError("learned draft tuning requires a bound DFlash2 checkpoint")
        block = result.get("draft_block_size", 5)
        if type(block) is not int or not 3 <= block <= 8:
            raise ValueError("Nanbeige DFlash2 runtime block must be 3..8")
        from .dflash2 import validate_draft_kernel, validate_runtime_quantization

        validate_runtime_quantization(result.get("draft_quantization"))
        validate_draft_kernel(result.get("draft_kernel"))
        from .nanbeige43_draft_vocab import validate

        if result.get("draft_vocab") is not None:
            result["draft_vocab"] = validate(result["draft_vocab"])
    return result, policy


class PrefillPlan:
    """Bounded context/load policy; never enlarges the operator's maximum.

    Decode contention keeps slices small. A warm suffix is bounded again by
    the engine's remaining/checkpoint limits. This is a candidate scheduling
    profile, not an arithmetic or performance equivalence assertion.
    """
    def __init__(self):
        self.calls = 0
        self.last = None

    def __call__(self, *, total_tokens, active_decode_lanes, maximum):
        if any(type(v) is not int for v in (total_tokens, active_decode_lanes, maximum)):
            raise TypeError("prefill geometry must contain integers")
        if total_tokens < 1 or active_decode_lanes < 0 or maximum < 1:
            raise ValueError("invalid prefill geometry")
        if active_decode_lanes:
            step = 128 if active_decode_lanes == 1 else 64
        else:
            step = 128 if total_tokens <= 512 else 256 if total_tokens <= 2048 else 512
        step = min(step, maximum)
        self.calls += 1
        self.last = {"context": total_tokens, "decode_lanes": active_decode_lanes, "step": step}
        return step

    def receipt(self):
        return {"schema": "nanbeige-prefill-context-load-v1", "qualified": False,
                "calls": self.calls, "last": self.last, "maximum_candidate_rows": 512}


def inspect_learned_draft(path, target):
    """Validate the trained target geometry before loading any draft tensors."""
    from .dflash2 import inspect_drafter

    record = inspect_drafter(path, target)
    args = record["args"]
    expected = {"hidden_size": 7168, "vocab_size": 166912, "num_target_layers": 52,
                "block_size": 8, "num_hidden_layers": 3, "head_dim": 128,
                "num_attention_heads": 28, "num_key_value_heads": 4,
                "rope_theta": 120000000, "rms_norm_eps": 1e-5}
    if any(getattr(args, key) != value for key, value in expected.items()):
        raise ValueError("Nanbeige DFlash2 trained geometry mismatch")
    if tuple(args.target_layer_ids) != (1, 25, 49):
        raise ValueError("Nanbeige DFlash2 requires post-layer taps 1,25,49")
    config = record["config"]
    if config.get("dflash_config", {}).get("tap_shift", 0) != 0:
        raise ValueError("Nanbeige DFlash2 tap_shift must be zero")
    # Geometry alone does not identify a model trained for this target. Only
    # the inspected published payload is bound by this initial adapter route.
    if record["weight_sha256"] != [PUBLISHED_DRAFT_WEIGHT_SHA256]:
        raise ValueError("Nanbeige DFlash2 payload is not the bound trained checkpoint")
    return record


PUBLISHED_DRAFT_WEIGHT_SHA256 = "b031ede26614f8800110e062f81fbaf3d1974dfc43bb32f346e6de941dc0efff"


def load_learned_draft(record, target, *, block_size=None, quantization=None, kernel=None, draft_vocab=None):
    from dataclasses import replace

    from .dflash2 import load_drafter

    if block_size is not None:
        record = {**record, "args": replace(record["args"], runtime_block_size=block_size)}
    draft = load_drafter(record, target, runtime_quantization=quantization, draft_kernel=kernel)
    # Shared DFlash2 tensor implementation is also used by family adapters
    # whose diagnostics supply the label themselves. Standard decoder asks
    # its bound provider for this role explicitly.
    object.__setattr__(draft, "receipt_kind", "external_dflash2")
    if draft_vocab is not None:
        from .nanbeige43_draft_vocab import install

        install(draft, draft_vocab, lane=kernel in ("lane", "fused"))
    return draft
