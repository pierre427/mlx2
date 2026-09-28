"""Tunable lane-matmul policy: detected defaults, family defaults, overrides.

Resolution order (later wins, per key):

1. built-in defaults, chosen by what is detected on the loaded model:
   whether it is a mixture of experts, and each projection's weight format;
2. per-family defaults (``FAMILY_DEFAULTS``, keyed by the adapter
   descriptor's ``family``);
3. operator overrides (``--lane-policy`` JSON).

Every projection gets its own ``min_rows`` from its format class, so a
mixed-precision artifact uses the right crossover on each layer.  The built-in
thresholds come from the 2026-09-28 sweep of 34 artifacts (see
docs/experiments/LANE-MATMUL-2026-09-28.md): at 8 rows the lane matmul beat
stock on dense 2-8-bit weights, bf16 weights needed 16 rows, and MoE models
were neutral because their expert layers (gather_qmm) are not covered.
"""

from __future__ import annotations

import copy
import fnmatch
import json
from collections import Counter
from pathlib import Path

from mlx import nn

MODES = ("auto", "off", "crossover", "exact")
FORMAT_CLASSES = ("q2", "q3", "q4", "q5", "q6", "q8", "bf16", "fp16")

BUILTIN = {
    "mode": "crossover",
    "min_rows": {"q2": 8, "q3": 8, "q4": 8, "q5": 8, "q6": 8, "q8": 8,
                 "bf16": 16, "fp16": 16},
    "max_rows": 32,
    "grouping": True,
    "skip": [],
    # Applied on top when the model is a mixture of experts.
    "moe": {"mode": "off"},
}

# Family-specific adjustments (descriptor.family -> partial policy).  Empty
# until a family measures differently from its format defaults.
FAMILY_DEFAULTS: dict[str, dict] = {}

_KEYS = {"mode", "min_rows", "max_rows", "grouping", "skip", "moe"}


def format_class(module) -> str | None:
    """Format class of one projection, or None when it is not a linear layer."""
    import mlx.core as mx

    if isinstance(module, nn.QuantizedLinear):
        return f"q{int(module.bits)}"
    if isinstance(module, nn.Linear):
        dtype = module["weight"].dtype
        return {mx.bfloat16: "bf16", mx.float16: "fp16"}.get(dtype)
    return None


_EXPERT_KEYS = ("num_experts", "n_routed_experts", "num_local_experts", "moe_num_experts")


def config_moe(config: dict | None) -> bool:
    """Whether an artifact config declares routed experts (top level or text_config)."""
    if not isinstance(config, dict):
        return False
    for scope in (config, config.get("text_config"), config.get("llm_config")):
        if isinstance(scope, dict) and any(
            isinstance(scope.get(key), int) and not isinstance(scope.get(key), bool)
            and scope[key] > 0 for key in _EXPERT_KEYS
        ):
            return True
    return False


def detect(model, config: dict | None = None) -> dict:
    """What the policy keys on: MoE-ness and the projection formats present.

    MoE is detected from either the artifact config's expert count or an
    expert module in the model (``*Switch*``, ``*MoE``, ``*SparseMoeBlock``,
    ``*Experts``).
    """
    moe = config_moe(config)
    formats: Counter = Counter()
    for _name, module in model.named_modules():
        kind = type(module).__name__
        if "Switch" in kind or kind.endswith(("MoE", "SparseMoeBlock", "Experts")):
            moe = True
        fmt = format_class(module)
        if fmt is not None:
            formats[fmt] += 1
    return {"moe": moe, "formats": dict(formats)}


def load_overrides(value) -> dict:
    """``--lane-policy``: a JSON object, inline or as a file path."""
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return copy.deepcopy(value)
    text = str(value)
    path = Path(text)
    raw = path.read_text() if not text.lstrip().startswith("{") and path.is_file() else text
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise TypeError("lane policy must be a JSON object")
    return parsed


def _validate(partial: dict, where: str) -> None:
    unknown = set(partial) - _KEYS
    if unknown:
        raise ValueError(f"{where}: unknown lane policy keys {sorted(unknown)}")
    if "mode" in partial and partial["mode"] not in MODES:
        raise ValueError(f"{where}: mode must be one of {MODES}")
    rows = partial.get("min_rows", {})
    if not isinstance(rows, dict) or set(rows) - set(FORMAT_CLASSES):
        raise ValueError(f"{where}: min_rows keys must be among {FORMAT_CLASSES}")
    for fmt, value in rows.items():
        if type(value) is not int or value < 1:
            raise ValueError(f"{where}: min_rows[{fmt}] must be a positive integer")
    if "max_rows" in partial and (type(partial["max_rows"]) is not int
                                  or not 1 <= partial["max_rows"] <= 128):
        raise ValueError(f"{where}: max_rows must be 1-128")
    if "grouping" in partial and type(partial["grouping"]) is not bool:
        raise ValueError(f"{where}: grouping must be boolean")
    if "skip" in partial and (not isinstance(partial["skip"], list)
                              or not all(isinstance(p, str) for p in partial["skip"])):
        raise ValueError(f"{where}: skip must be a list of name patterns")
    if "moe" in partial:
        if not isinstance(partial["moe"], dict):
            raise ValueError(f"{where}: moe must be an object")
        _validate({k: v for k, v in partial["moe"].items() if k != "moe"}, f"{where}.moe")


def _merge(base: dict, partial: dict, source: str, sources: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in partial.items():
        if key == "min_rows":
            for fmt, rows in value.items():
                out["min_rows"][fmt] = rows
                sources[f"min_rows.{fmt}"] = source
        elif key == "moe":
            out["moe"] = {**out.get("moe", {}), **copy.deepcopy(value)}
            sources["moe"] = source
        else:
            out[key] = copy.deepcopy(value)
            sources[key] = source
    return out


def resolve(detected: dict, *, family: str | None = None, overrides=None,
            mode: str | None = None) -> dict:
    """Resolve the effective policy; ``mode`` (the CLI switch) wins last.

    Returns the policy plus ``sources`` naming where every tuned value came
    from (builtin, builtin.moe, family:<name>, override, override.moe, cli).
    """
    sources = {key: "builtin" for key in ("mode", "max_rows", "grouping", "skip")}
    sources.update({f"min_rows.{fmt}": "builtin" for fmt in FORMAT_CLASSES})
    policy = copy.deepcopy(BUILTIN)
    user = load_overrides(overrides)
    _validate(user, "--lane-policy")
    fam = FAMILY_DEFAULTS.get(family or "", {})
    _validate(fam, f"family {family}")
    policy = _merge(policy, fam, f"family:{family}", sources)
    policy = _merge(policy, user, "override", sources)
    if detected.get("moe"):
        # MoE adjustments apply after defaults, but explicit top-level
        # overrides of the same keys still win.
        moe = {k: v for k, v in policy.get("moe", {}).items()}
        moe_source = sources.get("moe", "builtin")
        for key, value in moe.items():
            if key == "min_rows":
                for fmt, rows in value.items():
                    if sources.get(f"min_rows.{fmt}") != "override":
                        policy["min_rows"][fmt] = rows
                        sources[f"min_rows.{fmt}"] = f"{moe_source}.moe"
            elif sources.get(key) != "override":
                policy[key] = value
                sources[key] = f"{moe_source}.moe"
    if mode is not None and mode != "auto":
        policy["mode"] = mode
        sources["mode"] = "cli"
    if policy["mode"] == "auto":
        policy["mode"] = "crossover"
    policy.pop("moe", None)
    return {**policy, "detected": detected, "family": family, "sources": sources}


def min_rows_for(policy: dict, fmt: str | None) -> int | None:
    """Per-projection threshold; None means keep the projection on stock."""
    if policy["mode"] == "off" or fmt is None:
        return None
    if policy["mode"] == "exact":
        return 1
    return policy["min_rows"].get(fmt)


def skipped(policy: dict, name: str) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in policy.get("skip", ()))
