"""Bounded observability for the ``--ast`` route.

Two surfaces read this module:

* ``/v1/status`` gets an ``ast_route`` object: the startup route selection
  (``server.apply_ast_invocation``) reduced to closed enums, plus the loaded
  drafter's identity prefix.  It is configuration, not evidence that the
  route engaged or is qualified.
* ``/metrics`` gets ``mlx2_ast_route_info`` and the AST controller series.
  Every scheduler receipt the AST route publishes (tile-size controller,
  tree calibration, row sharing, planner outcomes) is walked by
  :func:`ast_scheduler_series` through declarative field tables, so a new
  receipt key is exported by adding one table row.

Labels are closed enums or bounded integers.  Filesystem paths, refusal
text and per-request identifiers never become label values; refusal text is
status-only.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Iterator, Mapping, NamedTuple

DRAFTER_KINDS = frozenset(
    {"dflash2", "eagle4", "dspark", "mtp", "pld", "external", "other"}
)
ARTIFACT_DRAFTER_KINDS = frozenset({"dflash2", "eagle4", "dspark", "external", "other"})
TREE_TILE_STATES = frozenset({"on", "off", "refused"})
TILE_ROWS_MODES = frozenset({"adaptive", "fixed", "not_applicable"})
DRAFTER_KERNELS = frozenset({"lane", "stock", "default", "other"})
TREE_CALIBRATION_MODES = frozenset({"off", "order", "order_size", "other"})
FINGERPRINT_PREFIX = 12

# Lane widths 1..8 are exact (the tree route's cohort limit is small); wider
# cohorts collapse into two buckets.
WIDTH_BUCKETS = tuple(str(width) for width in range(1, 9)) + ("9-16", "17+")
# Verified proposal nodes per lane: exact up to 32 (a shared 16-row tile of
# two lanes reaches 31), one overflow bucket beyond.
NODE_LABELS = tuple(str(nodes) for nodes in range(0, 33)) + ("33+",)
DEPTH_LABELS = tuple(str(depth) for depth in range(1, 17))
CALIBRATION_TABLES = frozenset({"greedy", "sampled", "other"})
PLANNER_BACKEND_ROLES = frozenset({"primary", "alternative"})
PLANNER_ROW_KINDS = frozenset({"logical", "physical", "padding"})
# SourceBudget.reason values from ast_planner.py, the tree tuner
# (ast_tree_tuning.py) and the fixed tile policies in external_speculative.
PLANNER_SOURCE_REASONS = frozenset(
    {
        "terminal_primary",
        "bounded_reprobe",
        "learned_branch_uplift",
        "hardware_cohort_cost_prior",
        "cohort_batched_source",
        "fixed_tree_source_budget",
        "fixed_packed_proposal_tile",
        "fixed_b1_horizon15_four_path_tile",
        "explicit_fixed",
        "terminal_anchor",
        "startup_measurement",
        "adjacent_exploration",
        "adjacent_recheck",
        "measured_throughput_hysteresis",
        "other",
    }
)

_TOPOLOGY_RE = re.compile(r"^(chain|drafter|[0-9]{1,2}(x[0-9]{1,2}|(/[0-9]{1,2}){0,15}))$")
_HEX_RE = re.compile(r"^[0-9a-f]+$")


# --------------------------------------------------------------------------
# Route status (startup) and its info labels.


def _drafter_kind(drafter: Any) -> str:
    if drafter in ("mtp", "pld"):
        return str(drafter)
    if not isinstance(drafter, str) or not drafter:
        return "other"
    try:
        config = json.loads((Path(drafter).expanduser() / "config.json").read_text())
    except (OSError, ValueError, TypeError):
        return "external"
    # The format table's factory names, without calling a factory: a factory
    # imports its adapter module, which must not load before the engine's
    # adapter is constructed.
    from .adapters.external_drafter_formats import _FORMATS

    architectures = config.get("architectures") if isinstance(config, dict) else None
    name = "dflash2"
    if isinstance(architectures, list) and len(architectures) == 1:
        factory = _FORMATS.get(architectures[0])
        if factory is not None:
            name = getattr(factory, "__name__", "").lstrip("_")
    return name if name in DRAFTER_KINDS else "other"


def _quantization_label(value: Any, present: bool) -> str:
    if not present:
        return "default"
    if value in (None, False):
        return "none"
    if isinstance(value, Mapping):
        bits, group = value.get("bits"), value.get("group_size")
        if (
            type(bits) is int and 2 <= bits <= 8
            and type(group) is int and group in (32, 64, 128)
        ):
            return f"q{bits}g{group}"
    return "other"


def _tile_rows_label(value: Any) -> str:
    if type(value) is int and 1 <= value <= 16:
        return str(value)
    return "default" if value is None else "other"


def route_status(route: Mapping[str, Any] | None, policy: Mapping[str, Any] | None = None,
                 adapter_type: Any = None) -> dict[str, Any] | None:
    """Reduce ``args.ast_route`` to the bounded ``/v1/status`` object.

    ``policy`` is the resolved execution policy (operator keys and adapter
    defaults); it carries the effective drafter kernel and quantization.
    Returns None when ``--ast`` was not selected.
    """

    if not isinstance(route, Mapping):
        return None
    policy = policy if isinstance(policy, Mapping) else {}
    refused = route.get("tree_tiles_refused")
    tiles = bool(route.get("tree_tiles"))
    tree_tiles = "on" if tiles else "refused" if refused else "off"
    # The policy already includes explicit CLI overrides and adapter defaults.
    # Invocation records retain None/False placeholders for omitted flags;
    # those placeholders must not conceal an effective policy selection.
    topology = policy.get("ast_topology", route.get("topology"))
    if topology is None:
        topology_label = "default"
    elif isinstance(topology, str) and _TOPOLOGY_RE.fullmatch(topology):
        topology_label = topology
    else:
        topology_label = "other"
    explicit_rows = policy.get("ast_tile_rows", route.get("tile_rows"))
    rows_authority = "operator" if explicit_rows is not None else "default"
    if explicit_rows is None and tiles:
        adapter_rows = getattr(adapter_type, "default_ast_tile_rows", None)
        if type(adapter_rows) is int:
            explicit_rows, rows_authority = adapter_rows, "adapter_default"
    if not tiles:
        rows_mode = "not_applicable"
    elif policy.get("ast_adaptive_rows") is False:
        rows_mode = "fixed"
    else:
        rows_mode = "adaptive"
    kernel = policy.get("draft_kernel") if "draft_kernel" in policy else None
    kernel_label = (
        "default" if kernel is None else kernel if kernel in DRAFTER_KERNELS else "other"
    )
    calibration = policy.get("ast_tree_calibration", route.get("tree_calibration"))
    if calibration in (None, False):
        calibration_label = "off"
    elif calibration is True:
        calibration_label = "order"
    else:
        calibration_label = (
            calibration if calibration in TREE_CALIBRATION_MODES else "other"
        )
    defaults = route.get("drafter_policy_defaults") or {}
    return {
        "selected": True,
        "receipt_kind": "configuration",
        "drafter_kind": _drafter_kind(route.get("drafter")),
        "tree_tiles": tree_tiles,
        "tree_tiles_refused_reason": None if refused is None else str(refused),
        "topology": topology_label,
        "tiles": (policy.get("ast_tiles", route.get("tiles"))
                  if type(policy.get("ast_tiles", route.get("tiles"))) is int else None),
        "tile_rows": _tile_rows_label(explicit_rows),
        "tile_rows_authority": rows_authority,
        "tile_rows_mode": rows_mode,
        "graft_pld": bool(policy.get("ast_graft_prompt_lookup", route.get("graft_pld"))),
        "drafter_kernel": kernel_label,
        "drafter_quantization": _quantization_label(
            policy.get("draft_quantization"), "draft_quantization" in policy
        ),
        "tree_calibration": calibration_label,
        "batch_row_sharing": bool(
            policy.get("ast_batch_row_sharing", route.get("batch_row_sharing", False))
        ),
        "drafter_policy_defaults": sorted(
            str(key) for key in defaults if isinstance(key, str)
        ),
    }


def route_view(engine: Any) -> dict[str, Any] | None:
    """The status object with the loaded drafter's identity prefix."""

    static = getattr(engine, "ast_route", None)
    if not isinstance(static, Mapping):
        return None
    view = dict(static)
    if view.get("drafter_kind") not in ARTIFACT_DRAFTER_KINDS:
        view["drafter_fingerprint"] = None
        view["drafter_identity"] = "not_applicable"
        return view
    identity = getattr(getattr(engine, "adapter", None), "identity", None)
    fingerprint = identity.get("draft_fingerprint") if isinstance(identity, Mapping) else None
    if isinstance(fingerprint, str) and _HEX_RE.fullmatch(fingerprint.lower()):
        view["drafter_fingerprint"] = fingerprint.lower()[:FINGERPRINT_PREFIX]
        view["drafter_identity"] = "loaded"
    else:
        view["drafter_fingerprint"] = None
        view["drafter_identity"] = "unavailable"
    return view


def route_info_labels(view: Mapping[str, Any]) -> dict[str, str]:
    """Closed-enum labels of ``mlx2_ast_route_info``."""

    def closed(value, allowed, fallback="other"):
        return value if value in allowed else fallback

    fingerprint = view.get("drafter_fingerprint")
    if not (
        isinstance(fingerprint, str)
        and len(fingerprint) == FINGERPRINT_PREFIX
        and _HEX_RE.fullmatch(fingerprint)
    ):
        fingerprint = (
            "none" if view.get("drafter_identity") == "not_applicable" else "unknown"
        )
    topology = view.get("topology")
    tile_rows = view.get("tile_rows")
    quantization = view.get("drafter_quantization")
    return {
        "drafter": closed(view.get("drafter_kind"), DRAFTER_KINDS),
        "drafter_fingerprint": fingerprint,
        "topology": (
            topology
            if topology == "default"
            or (isinstance(topology, str) and _TOPOLOGY_RE.fullmatch(topology))
            else "other"
        ),
        "tree_tiles": closed(view.get("tree_tiles"), TREE_TILE_STATES),
        "tile_rows": (
            tile_rows
            if tile_rows in ("default", "other")
            or (isinstance(tile_rows, str) and tile_rows.isdigit() and 1 <= int(tile_rows) <= 16)
            else "other"
        ),
        "tile_rows_mode": closed(view.get("tile_rows_mode"), TILE_ROWS_MODES),
        "graft_pld": "true" if view.get("graft_pld") else "false",
        "drafter_kernel": closed(view.get("drafter_kernel"), DRAFTER_KERNELS),
        "drafter_quantization": (
            quantization
            if quantization in ("default", "none", "other")
            or (isinstance(quantization, str) and re.fullmatch(r"q[2-8]g(32|64|128)", quantization))
            else "other"
        ),
        "tree_calibration": closed(view.get("tree_calibration"), TREE_CALIBRATION_MODES),
        "batch_row_sharing": "true" if view.get("batch_row_sharing") else "false",
    }


# --------------------------------------------------------------------------
# Scheduler receipt walker.


class Series(NamedTuple):
    name: str
    kind: str  # "counter" or "gauge"
    help: str
    value: float
    labels: tuple[tuple[str, str], ...]


def width_bucket(width: Any) -> str | None:
    try:
        width = int(width)
    except (TypeError, ValueError):
        return None
    if width < 1:
        return None
    if width <= 8:
        return str(width)
    return "9-16" if width <= 16 else "17+"


def node_label(nodes: Any, *, exact_only: bool = False) -> str | None:
    try:
        nodes = int(nodes)
    except (TypeError, ValueError):
        return None
    if nodes < 0:
        return None
    if nodes <= 32:
        return str(nodes)
    return None if exact_only else "33+"


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value


# One row per receipt field of a lane width in ``external_tree_tile_budget``
# (``TileBudgetController.receipt()``).  shape: "scalar" for one value per
# width, "nodes" for a ``{node count: value}`` map.  scale converts the
# receipt unit to the Prometheus base unit.  Unknown fields are ignored, so a
# controller key is exported only once it is added here.
TILE_WIDTH_FIELDS: dict[str, tuple[str, str, str, str, float]] = {
    "rounds": (
        "mlx2_ast_tile_rounds_total", "counter", "scalar", 1.0,
        "Lane-rounds the AST tile-size controller observed, by lane-width bucket.",
    ),
    "chosen": (
        "mlx2_ast_tile_chosen_nodes", "gauge", "scalar", 1.0,
        "Proposal nodes per lane the AST tile-size controller currently serves, by lane-width bucket.",
    ),
    "cost_ms": (
        "mlx2_ast_tile_cost_seconds", "gauge", "nodes", 1e-3,
        "EWMA measured AST tile round seconds by lane-width bucket and verified node count.",
    ),
    "tokens_per_round": (
        "mlx2_ast_tile_tokens_per_round", "gauge", "nodes", 1.0,
        "Counterfactual committed tokens per lane-round had the tile verified this many nodes.",
    ),
    # Optional: the fitted a + b * rows cost line, when the controller
    # publishes it (``{"intercept_ms": a, "slope_ms_per_row": b}``).
    "fit": (
        "mlx2_ast_tile_cost_fit_seconds", "gauge", "fit", 1e-3,
        "Fitted AST tile round cost line a + b*rows (term=intercept seconds, slope seconds per row).",
    ),
}
_FIT_TERMS = {"intercept_ms": "intercept", "slope_ms_per_row": "slope"}

# ``external_tree_calibration`` (per-depth drafter calibration receipt).
CALIBRATION_FIELDS: dict[str, tuple[str, str, Mapping[str, str], str]] = {
    "trees": ("mlx2_ast_tree_calibration_trees_total", "counter", {},
              "Verified trees fed to the AST per-depth drafter calibration."),
    "fits": ("mlx2_ast_tree_calibration_fits_total", "counter", {},
             "AST per-depth drafter calibration refits."),
    "truncated_rounds": ("mlx2_ast_tree_calibration_truncations_total", "counter",
                         {"unit": "rounds"},
                         "Rounds and nodes cut from AST trees by the calibrated live-row rule."),
    "truncated_nodes": ("mlx2_ast_tree_calibration_truncations_total", "counter",
                        {"unit": "nodes"},
                        "Rounds and nodes cut from AST trees by the calibrated live-row rule."),
    "predicted_tokens_per_tree": ("mlx2_ast_tree_calibration_tokens_per_tree", "gauge",
                                  {"kind": "predicted"},
                                  "Mean predicted and realized committed tokens per calibrated tree."),
    "realized_tokens_per_tree": ("mlx2_ast_tree_calibration_tokens_per_tree", "gauge",
                                 {"kind": "realized"},
                                 "Mean predicted and realized committed tokens per calibrated tree."),
    "rate_tokens_per_second": ("mlx2_ast_tree_calibration_rate_tokens_per_second", "gauge", {},
                               "Committed tokens per second the calibrated live-row rule prices rows at."),
}
# Per-table, per-depth maps inside ``tables``.
CALIBRATION_DEPTH_FIELDS: dict[str, tuple[str, str, Mapping[str, str], str]] = {
    "alpha": ("mlx2_ast_tree_calibration_parameter", "gauge", {"parameter": "alpha"},
              "Per-depth AST drafter calibration parameters sigmoid(alpha+beta*logit(q))."),
    "beta": ("mlx2_ast_tree_calibration_parameter", "gauge", {"parameter": "beta"},
             "Per-depth AST drafter calibration parameters sigmoid(alpha+beta*logit(q))."),
    "trials": ("mlx2_ast_tree_calibration_window_trials", "gauge", {},
               "Per-depth AST drafter calibration trials in the current fit window."),
    "hit_rate": ("mlx2_ast_tree_calibration_hit_ratio", "gauge", {},
                 "Per-depth AST drafter observed acceptance in the calibration window."),
}

_DECISIONS = (
    "mlx2_ast_tile_decisions_total",
    "AST tree proposal-node budgets requested per lane, by chosen node count.",
)
_ROW_SHARING_ROUNDS = (
    "mlx2_ast_row_sharing_rounds_total",
    "Batched AST rounds whose tile rows were shared among lanes by calibrated survival.",
)
_ROW_SHARING_NODES = (
    "mlx2_ast_row_sharing_lane_nodes_total",
    "Lanes of row-shared AST rounds by the proposal nodes each lane kept.",
)
_VERIFIED_NODES = (
    "mlx2_ast_tree_verified_nodes_total",
    "AST lane-rounds by tree nodes sent to the target verify (after calibration and row sharing).",
)
_PLANNER_SOURCE = (
    "mlx2_ast_planner_source_budgets_total",
    "AST planner source-budget outcomes per lane-round, by bounded planner reason.",
)
_PLANNER_LAUNCHES = (
    "mlx2_ast_planner_launches_total",
    "AST planner physical launches by primary or alternative execution backend.",
)
_PLANNER_ROWS = (
    "mlx2_ast_planner_launch_rows_total",
    "AST planner launch rows: logical (useful), physical (launched) and padding.",
)


def _labels(**labels: str) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(labels.items()))


def _walk_tile_budget(receipt: Mapping[str, Any]) -> Iterator[Series]:
    counters: dict[tuple[str, tuple], float] = {}
    gauges: dict[tuple[str, tuple], float] = {}
    helps: dict[str, tuple[str, str]] = {}
    widths = []
    for width, state in receipt.items():
        bucket = width_bucket(width)
        if bucket is None or not isinstance(state, Mapping):
            continue
        rounds = _number(state.get("rounds")) or 0
        widths.append((rounds, str(width), bucket, state))
    # Gauges of widths sharing a bucket keep the most-observed width's value
    # (processed last); counters sum.
    for _rounds, _width, bucket, state in sorted(widths, key=lambda item: item[:2]):
        for field, (name, kind, shape, scale, help_text) in TILE_WIDTH_FIELDS.items():
            raw = state.get(field)
            if raw is None:
                continue
            helps[name] = (kind, help_text)
            if shape == "scalar":
                points = [((), raw)]
            elif shape == "nodes":
                if not isinstance(raw, Mapping):
                    continue
                points = []
                for nodes, value in raw.items():
                    label = node_label(nodes, exact_only=True)
                    if label is not None:
                        points.append(((("nodes", label),), value))
            else:  # fit
                if not isinstance(raw, Mapping):
                    continue
                points = [
                    ((("term", term),), raw.get(key))
                    for key, term in _FIT_TERMS.items()
                ]
            for extra, value in points:
                value = _number(value)
                if value is None:
                    continue
                key = (name, tuple(sorted((("width_bucket", bucket),) + extra)))
                if kind == "counter":
                    counters[key] = counters.get(key, 0) + value * scale
                else:
                    gauges[key] = value * scale
    for store in (counters, gauges):
        for (name, labels), value in store.items():
            kind, help_text = helps[name]
            yield Series(name, kind, help_text, value, labels)


def _walk_calibration(receipt: Mapping[str, Any]) -> Iterator[Series]:
    mode = receipt.get("mode")
    yield Series(
        "mlx2_ast_tree_calibration_enabled", "gauge",
        "Whether AST per-depth drafter calibration is active, by mode.",
        1, _labels(mode=mode if mode in TREE_CALIBRATION_MODES - {"off"} else "other"),
    )
    for field, (name, kind, extra, help_text) in CALIBRATION_FIELDS.items():
        value = _number(receipt.get(field))
        if value is not None:
            yield Series(name, kind, help_text, value, _labels(**extra))
    tables = receipt.get("tables")
    if not isinstance(tables, Mapping):
        return
    folded: dict[tuple[str, tuple], tuple[str, str, float]] = {}
    for table, state in tables.items():
        table_label = table if table in CALIBRATION_TABLES else "other"
        if not isinstance(state, Mapping):
            continue
        for field, (name, kind, extra, help_text) in CALIBRATION_DEPTH_FIELDS.items():
            depths = state.get(field)
            if not isinstance(depths, Mapping):
                continue
            for depth, value in depths.items():
                value = _number(value)
                if value is None or str(depth) not in DEPTH_LABELS:
                    continue
                key = (name, _labels(table=table_label, depth=str(depth), **extra))
                if kind == "counter" and key in folded:
                    value += folded[key][2]
                folded[key] = (kind, help_text, value)
    for (name, labels), (kind, help_text, value) in folded.items():
        yield Series(name, kind, help_text, value, labels)


def _walk_counter_map(raw: Mapping[str, Any], metric: tuple[str, str], label: str,
                      bound) -> Iterator[Series]:
    totals: dict[str, float] = {}
    for key, value in dict(raw).items():
        bounded = bound(key)
        value = _number(value)
        if bounded is None or value is None:
            continue
        totals[bounded] = totals.get(bounded, 0) + value
    for bounded, value in totals.items():
        yield Series(metric[0], "counter", metric[1], value, _labels(**{label: bounded}))


def _closed(allowed):
    return lambda key: key if key in allowed else "other"


# scheduler_stats key -> walker.  The one place an AST scheduler receipt is
# added to the exporter.
SCHEDULER_RECEIPTS = {
    "external_tree_tile_budget": _walk_tile_budget,
    "external_tree_calibration": _walk_calibration,
    "external_tree_node_budget_histogram": lambda raw: _walk_counter_map(
        raw, _DECISIONS, "nodes", node_label
    ),
    "external_tree_row_sharing_lane_nodes": lambda raw: _walk_counter_map(
        raw, _ROW_SHARING_NODES, "nodes", node_label
    ),
    "external_tree_verified_nodes": lambda raw: _walk_counter_map(
        raw, _VERIFIED_NODES, "nodes", node_label
    ),
    "ast_planner_source_budgets": lambda raw: _walk_counter_map(
        raw, _PLANNER_SOURCE, "reason", _closed(PLANNER_SOURCE_REASONS)
    ),
    "ast_planner_launches": lambda raw: _walk_counter_map(
        raw, _PLANNER_LAUNCHES, "backend_role",
        lambda key: key if key in PLANNER_BACKEND_ROLES else None,
    ),
    "ast_planner_launch_rows": lambda raw: _walk_counter_map(
        raw, _PLANNER_ROWS, "kind",
        lambda key: key if key in PLANNER_ROW_KINDS else None,
    ),
}
SCHEDULER_SCALARS = {
    "external_tree_row_sharing_rounds": _ROW_SHARING_ROUNDS,
}


def ast_scheduler_series(scheduler: Mapping[str, Any]) -> list[Series]:
    """Every AST series from one scheduler snapshot (absent keys emit none)."""

    series: list[Series] = []
    for key, walker in SCHEDULER_RECEIPTS.items():
        raw = scheduler.get(key)
        if isinstance(raw, Mapping) and raw:
            # The generator may grow an in-place histogram while a scrape
            # runs; dict() copies it in one step under the GIL.
            series.extend(walker(dict(raw)))
    for key, (name, help_text) in SCHEDULER_SCALARS.items():
        value = _number(scheduler.get(key))
        if value is not None:
            series.append(Series(name, "counter", help_text, value, ()))
    return series


__all__ = [
    "SCHEDULER_RECEIPTS",
    "SCHEDULER_SCALARS",
    "TILE_WIDTH_FIELDS",
    "Series",
    "ast_scheduler_series",
    "route_info_labels",
    "route_status",
    "route_view",
]
