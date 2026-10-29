"""Flash-Next ``lane_dense``: the lane matmul on the dense projections only.

``--lane-matmul auto`` resolves a mixture-of-experts model to ``off``
(``runtime/lane/policy.py``: the routed experts run ``gather_qmm`` and are
not lane classes, and MoE models measured neutral).  Flash-Next's dense
projections still run several multi-row calls per verify -- attention q/k/v/o,
the QSA indexer projection, the GDN input/output projections, the MTP fusion
projections and lm_head -- and the self-MTP draft loop is only selected when
the lane matmul covers the model (``ServingEngine._select_draft_loop``).

The Flash-Next policy field ``lane_dense`` (default off, implemented,
unqualified) opts in.  The adapter then

* declares lane defaults (``lane_policy_defaults``): the crossover mode for
  this MoE model plus a skip list of every projection another Flash-Next
  kernel reads directly or that must stay on its current kernels
  (``EXCLUSIONS``);
* refuses a resolved lane policy that would put an excluded projection on
  the lane law, or whose row window reaches a fused path that reads the same
  weights with its own stock product (``FUSED_CONSUMERS``) -- one projection
  would then run two numerical laws at the same row count
  (``validate_lane_policy``);
* after the serving install, re-checks the live model and fails closed when
  nothing was covered, an excluded projection was installed or a fused path
  collides, and records which projections are covered and excluded
  (``bind_lane_receipt``).

The lane arithmetic is a numerical law distinct from stock MLX: the serving
install binds its ``law_id`` (skip digest included) into the route's lane
receipt and the APCv2 namespace (``installer.apc_lane_fingerprint``).

Original mlx2 code.
"""

from __future__ import annotations

import fnmatch
import re
from collections import Counter

SCHEMA = "mlx2.flash-next-lane-dense.v1"

# Projections that keep their current kernels, by module-name pattern
# (``fnmatch``; ``*`` crosses dots), with the reason.  Each becomes a lane
# policy ``skip`` pattern, so the set is part of the lane law.
EXCLUSIONS = (
    (
        "*.mlp.*",
        (
            "mixture-of-experts block: the router gate, the shared expert and its "
            "gate stay on the routed kernels (moe_routed_decode folds the shared "
            "expert and gate into its launches, moe_window and varlen sparse MoE "
            "read their weights directly); routed experts are SwitchLinear, never "
            "lane classes"
        ),
    ),
    (
        "*hyper_connection*",
        (
            "hyper-connection projections: hc_decode_kernels (1-8 rows), "
            "hc_prefill_kernels (9+ rows) and the compiled HC glue read their "
            "weights directly"
        ),
    ),
    (
        "*.ple.*",
        (
            "PLE projections run inside the PLE compiled chain (mx.compile): the "
            "lane dispatch and its counters would be fixed at trace time"
        ),
    ),
)

# Fused paths that read a lane-eligible projection's weights with their own
# stock product.  ``owner`` is an attribute the owning module class carries;
# ``members`` are child paths; ``max_rows`` is the widest call (rows =
# batch x length) the fused path serves, None for any width.  A member whose
# lane window starts at or below ``max_rows`` while the path is live would
# run the stock law for some calls of a width the lane law claims.
FUSED_CONSUMERS = {
    "attn_fused_rows": {
        "owner": "_fused_projection_table",
        "members": ("q_proj", "k_proj", "v_proj", "indexer.index_qk_proj"),
        "max_rows": 7,
    },
    "qsa_fused_projection": {
        "owner": "_fused_projection_table",
        "members": ("q_proj", "k_proj", "v_proj", "indexer.index_qk_proj"),
        "max_rows": None,
    },
    "gdn_fused_inproj": {
        "owner": "gdn_fused_inproj",
        "members": ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"),
        "max_rows": 8,
    },
}

# Bundled draft-loop acceptance priors a Flash-Next route may name, with the
# model family each was measured on (none was measured on Flash-Next).
DRAFT_LOOP_PRIORS = {
    "qwen38_27b": {
        "resource": "draft_loop_prior_qwen38_27b.json",
        "measured_on": "qwen3.8-27b",
    },
}


def prior_base(name: str) -> int:
    from ..runtime.verify_topology import load_prior_grid

    return int(load_prior_grid(DRAFT_LOOP_PRIORS[name]["resource"])["base"])


def prior_resource(policy) -> str | None:
    name = getattr(policy, "draft_loop_prior", None)
    return None if name is None else DRAFT_LOOP_PRIORS[name]["resource"]


def enabled(policy) -> bool:
    return bool(getattr(policy, "lane_dense", False))


def lane_policy_defaults(policy) -> dict | None:
    """Adapter lane defaults: crossover on this MoE model, exclusions skipped."""
    if not enabled(policy):
        return None
    return {
        "mode": "crossover",
        # The builtin MoE adjustment (mode off) applies after adapter
        # defaults; the adapter's own MoE value is what it resolves to.
        "moe": {"mode": "crossover"},
        "skip": [pattern for pattern, _reason in EXCLUSIONS],
    }


def _excluded(name: str):
    for pattern, reason in EXCLUSIONS:
        if fnmatch.fnmatch(name, pattern):
            return pattern, reason
    return None


def _member(module, path: str):
    for part in path.split("."):
        module = getattr(module, part, None)
        if module is None:
            return None
    return module


def live_consumers(policy) -> dict:
    """The fused paths live in this process, with their row bound."""
    from ..runtime.models import qwen4_attn_rows, qwen4_exp

    live = {}
    if bool(getattr(policy, "attn_fused_rows", False)) or qwen4_attn_rows.enabled():
        live["attn_fused_rows"] = FUSED_CONSUMERS["attn_fused_rows"]
    if qwen4_exp._QSA_FUSED_PROJ:
        live["qsa_fused_projection"] = FUSED_CONSUMERS["qsa_fused_projection"]
    # Per layer (``set_gdn_fused_inproj`` toggles it live); checked per owner.
    live["gdn_fused_inproj"] = FUSED_CONSUMERS["gdn_fused_inproj"]
    return live


def _collisions(model, policy, threshold) -> list[str]:
    """``consumer: member`` pairs whose lane window meets a live fused path.

    ``threshold(name, module)`` is the member's lane ``min_rows`` (None when
    the lane does not cover it).
    """
    names = {id(module): name for name, module in model.named_modules()}
    found = []
    for consumer, spec in live_consumers(policy).items():
        for _name, owner in model.named_modules():
            marker = getattr(owner, spec["owner"], None)
            if marker is None or marker is False:
                continue
            for path in spec["members"]:
                member = _member(owner, path)
                if member is None:
                    continue
                name = names.get(id(member), path)
                rows = threshold(name, member)
                if rows is None:
                    continue
                if spec["max_rows"] is None or rows <= spec["max_rows"]:
                    found.append(f"{consumer}: {name} (lane from {rows} rows)")
    return found


def _policy_threshold(lane_policy):
    from ..runtime.lane.policy import format_class, skipped

    def threshold(name, module):
        fmt = format_class(module)
        if fmt is None or fmt not in lane_policy["min_rows"] or skipped(lane_policy, name):
            return None
        if lane_policy["mode"] == "exact":
            return 1
        rows = int(lane_policy["min_rows"][fmt])
        return None if rows > int(lane_policy["max_rows"]) else rows

    return threshold


def validate_lane_policy(policy, model, lane_policy: dict) -> None:
    """Refuse a resolved lane policy ``lane_dense`` cannot serve (before install)."""
    if not enabled(policy):
        return
    if lane_policy.get("mode") == "off":
        raise ValueError("lane_dense requires the lane matmul, but the lane policy resolved to off")
    threshold = _policy_threshold(lane_policy)
    covered_excluded = sorted(
        name for name, module in model.named_modules()
        if _excluded(name) is not None and threshold(name, module) is not None
    )
    if covered_excluded:
        raise ValueError(
            "lane_dense: the lane policy would cover projections Flash-Next keeps on "
            f"their kernels ({len(covered_excluded)}, first {covered_excluded[0]!r}); "
            "keep the adapter's skip patterns "
            f"{[pattern for pattern, _reason in EXCLUSIONS]}"
        )
    collisions = _collisions(model, policy, threshold)
    if collisions:
        raise ValueError(
            "lane_dense: the lane window reaches rows a fused Flash-Next path serves "
            f"from the same weights ({len(collisions)}, first {collisions[0]}); use "
            "crossover thresholds above that path's rows or turn the path off"
        )


def _pattern(name: str) -> str:
    return re.sub(r"\.\d+\.", ".*.", name)


def bind_lane_receipt(policy, model, receipt) -> dict | None:
    """Check the installed model and return the ``lane_dense`` diagnostics.

    Fails closed (``ValueError``, after restoring stock classes) when the
    install covered nothing, ran another install path, covered an excluded
    projection, or meets a live fused path.
    """
    if not enabled(policy):
        return None
    from ..runtime.lane import installer

    receipt = receipt or {}
    lane_policy = receipt.get("policy") or {}
    if lane_policy.get("mode") not in ("crossover", "exact"):
        raise ValueError(
            "lane_dense requires the mlx2 lane policy install (crossover or exact); "
            f"got mode {lane_policy.get('mode')!r} (receipt law {receipt.get('law_id')!r})"
        )
    if receipt.get("available") is False or not receipt.get("covered"):
        raise ValueError(
            "lane_dense requested but the lane matmul covered no projection "
            f"(available={receipt.get('available')!r}, refused={receipt.get('refused')!r})"
        )

    def installed_threshold(_name, module):
        if not installer.installed(module):
            return None
        return int(getattr(module, "_lane_min_rows", 1))

    covered: Counter = Counter()
    excluded: dict = {}
    wrong = []
    for name, module in model.named_modules():
        hit = _excluded(name)
        lane = installer.installed(module)
        if hit is not None:
            if lane:
                wrong.append(name)
            elif hasattr(module, "weight") and type(module).__name__.endswith("Linear"):
                entry = excluded.setdefault(_pattern(name), {"reason": hit[1], "count": 0})
                entry["count"] += 1
        elif lane:
            covered[_pattern(name)] += 1
    problems = [f"excluded projection installed: {name}" for name in wrong]
    problems += _collisions(model, policy, installed_threshold)
    if problems:
        installer.uninstall(model)
        raise ValueError(
            f"lane_dense refused after install ({len(problems)}, first {problems[0]})"
        )
    prior = getattr(policy, "draft_loop_prior", None)
    return {
        "schema": SCHEMA,
        "state": "implemented-unqualified",
        "law_id": receipt.get("law_id"),
        "mode": lane_policy.get("mode"),
        "min_rows": lane_policy.get("min_rows"),
        "covered_by_format": dict(receipt.get("covered") or {}),
        "covered": dict(sorted(covered.items())),
        "excluded": dict(sorted(excluded.items())),
        "refused": dict(receipt.get("refused") or {}),
        "routed_experts": "SwitchLinear tables: not lane classes, kernels unchanged",
        "draft_loop_prior": None if prior is None else {
            "name": prior,
            **DRAFT_LOOP_PRIORS[prior],
            "borrowed": True,
        },
    }
