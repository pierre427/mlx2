"""Per-artifact best-route defaults: the measured-best route with no route flag.

Decision (Pierre, 2026-10-21): build per-artifact best-route defaults now,
auto-pair verified on-disk drafters, use ``--ast`` where it wins, label them
unqualified in receipts, and queue qualification afterwards.  This is the
one exception to "a default changes only after qualification"; every
selection here is ``selected_unqualified``.

The table (``best_route_defaults.json``, beside this module) is keyed by the
artifact fingerprint the adapter registry computes, with optional
family-level entries keyed by adapter class.  It is data: no scheduler,
cache or request-lifecycle code branches on a model name.  A table entry is
translated into the exact CLI intent an operator would type (``--ast
--ast-drafter <path>``, ``--native-mtp`` ...) before route resolution, so an
auto-selected route goes through the same validation as an explicit one.

Selection rule (:func:`rule_verdict`): a route becomes the default when its
speedup over plain decode, as a geometric mean over the measured cells (code,
chat, ~7K long context and copy-heavy prompts, each at B1 and B4), is at
least 1.05 and no cell falls below 0.85x plain decode.  Among qualifying
routes the highest geometric mean wins; otherwise the current default stays.

External drafters are auto-paired: on-disk drafters are discovered under the
model roots, filtered by declared architecture and evidence identity, and
verified by the target adapter's own inspector (geometry, headers and the
payload-bound content revision) exactly as for an explicit ``--ast-drafter``.
The verified revision is then pinned into the execution policy so the
adapter re-checks it at load.  A refused or missing drafter fails over to the
entry's next route, and finally to today's adapter default; an unverified
drafter is never used.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

SCHEMA = "mlx2.best-route-defaults.v1"
RECORD_SCHEMA = "mlx2.best-route-default-selection.v1"
TABLE_PATH = Path(__file__).with_name("best_route_defaults.json")

# The selection rule.  Written here once; docs/SERVING.md quotes it.
RULE = {
    "baseline": "ordinary",
    "min_geomean": 1.05,
    "min_cell": 0.85,
    "prompts": ("code", "chat", "long", "copy"),
    "widths": ("B1", "B4"),
    "metric": "B1 median decode tok/s; B4 per-rep sum of lane decode tok/s, median over reps",
}
STATUSES = frozenset({"selected_unqualified", "kept_current_default", "pending"})
ROUTE_KINDS = frozenset({"ast", "native_mtp", "prompt_lookup", "ordinary", "adapter_default"})
ROUTE_FLAGS = ("ordinary", "native_mtp", "external_draft", "prompt_lookup", "ast")
# Options whose meaning depends on the route the operator chose; any of them
# keeps the operator on today's resolution rather than an auto route.
ROUTE_CONSTRAINING_OPTIONS = (
    "draft_model", "execution_policy", "adaptive_mtp_depth", "mtp_acceptance_log",
    "approximate_kv", "ast_drafter", "ast_topology", "ast_graft_pld", "ast_tiles",
    "ast_tile_rows", "ast_tree_calibration", "ast_batch_row_sharing", "mtp_head",
)


# ------------------------------------------------------------------ rule --

def rule_cells() -> tuple[str, ...]:
    return tuple(f"{prompt}/{width}" for prompt in RULE["prompts"] for width in RULE["widths"])


def route_score(baseline: dict, candidate: dict) -> dict:
    """Speedup ratios, geometric mean and minimum cell of one route."""
    cells = rule_cells()
    missing = [cell for cell in cells if not baseline.get(cell) or not candidate.get(cell)]
    if missing:
        return {"qualifies": False, "missing": missing, "ratios": {}, "geomean": None, "min_cell": None}
    ratios = {cell: candidate[cell] / baseline[cell] for cell in cells}
    if any(not math.isfinite(value) or value <= 0 for value in ratios.values()):
        raise ValueError("speedup ratios must be finite and positive")
    geomean = math.exp(sum(math.log(value) for value in ratios.values()) / len(ratios))
    worst = min(ratios, key=ratios.get)
    qualifies = geomean >= RULE["min_geomean"] and ratios[worst] >= RULE["min_cell"]
    return {
        "qualifies": qualifies,
        "missing": [],
        "ratios": {cell: round(value, 4) for cell, value in ratios.items()},
        "geomean": round(geomean, 4),
        "min_cell": round(ratios[worst], 4),
        "min_cell_name": worst,
    }


def rule_verdict(baseline: dict, candidates: dict, current_default: str) -> dict:
    """Apply the rule: ``candidates`` maps route name to ``{cell: tok/s}``.

    ``baseline`` is plain decode on the same cells.  Returns every route's
    score and the ``winner``: the qualifying route with the highest
    geometric mean, else ``current_default``.
    """
    scores = {name: route_score(baseline, cells) for name, cells in candidates.items()}
    qualifying = [name for name, score in scores.items() if score["qualifies"]]
    if qualifying:
        winner = max(qualifying, key=lambda name: scores[name]["geomean"])
        decision = "selected"
    else:
        winner = current_default
        decision = "kept_current_default"
    return {"rule": dict(RULE, prompts=list(RULE["prompts"]), widths=list(RULE["widths"])),
            "scores": scores, "winner": winner, "decision": decision}


# ----------------------------------------------------------------- table --

def load_table(path: Path | None = None) -> dict:
    path = TABLE_PATH if path is None else Path(path)
    raw = path.read_bytes()
    table = json.loads(raw)
    if not isinstance(table, dict) or table.get("schema") != SCHEMA:
        raise ValueError(f"{path}: not a {SCHEMA} table")
    entries = table.get("entries")
    if not isinstance(entries, list):
        raise TypeError(f"{path}: entries must be a list")
    seen = set()
    for entry in entries:
        validate_entry(entry)
        key = entry.get("fingerprint") or ("family", entry["family"])
        if key in seen:
            raise ValueError(f"{path}: duplicate entry {key}")
        seen.add(key)
    table["_sha256"] = hashlib.sha256(raw).hexdigest()
    table["_path"] = str(path)
    return table


def validate_entry(entry: dict) -> None:
    if not isinstance(entry, dict):
        raise TypeError("best-route entry must be an object")
    if ("fingerprint" in entry) == ("family" in entry):
        raise ValueError("best-route entry needs exactly one of fingerprint or family")
    if "fingerprint" in entry:
        if not isinstance(entry["fingerprint"], str) or len(entry["fingerprint"]) != 64:
            raise ValueError("best-route fingerprint must be a sha256 hex digest")
        if not isinstance(entry.get("checkpoint"), str) or "@sha256:" not in entry["checkpoint"]:
            raise ValueError("best-route checkpoint must read <name>@sha256:<fingerprint>")
        if not entry["checkpoint"].endswith(entry["fingerprint"]):
            raise ValueError("best-route checkpoint and fingerprint disagree")
    if not isinstance(entry.get("adapter") or entry.get("family"), str):
        raise TypeError("best-route entry must name its adapter class")
    if entry.get("status") not in STATUSES:
        raise ValueError(f"best-route status must be one of {sorted(STATUSES)}")
    evidence = entry.get("evidence")
    if not isinstance(evidence, dict) or not evidence.get("files") or not evidence.get("source_revision"):
        raise ValueError("best-route entry must cite evidence files and a source revision")
    routes = [entry.get("default"), *entry.get("fallback", ())]
    for spec in routes:
        if not isinstance(spec, dict) or spec.get("kind") not in ROUTE_KINDS:
            raise ValueError(f"best-route route kind must be one of {sorted(ROUTE_KINDS)}")
        drafter = spec.get("drafter")
        if isinstance(drafter, dict):
            if spec["kind"] != "ast":
                raise ValueError("an external drafter is selected through the ast route")
            revisions = drafter.get("identities")
            if not drafter.get("architectures") or not revisions or any(
                not isinstance(value, str) or len(value) != 64 for value in revisions
            ):
                raise ValueError("an auto-paired drafter needs architectures and sha256 identities")
        elif drafter is not None and drafter not in ("mtp", "pld", "auto"):
            raise ValueError("ast drafter must be mtp, pld, auto or an auto-pairing spec")


def find_entry(table: dict, adapter_name: str, fingerprint: str | None) -> dict | None:
    for entry in table["entries"]:
        if fingerprint and entry.get("fingerprint") == fingerprint:
            if entry.get("adapter") != adapter_name:
                raise ValueError(
                    f"best-route entry {entry['checkpoint']} names {entry.get('adapter')}, "
                    f"artifact resolves to {adapter_name}"
                )
            return entry
    for entry in table["entries"]:
        if entry.get("family") == adapter_name:
            return entry
    return None


# ----------------------------------------------------------- drafters --

def default_drafter_roots() -> tuple[list[Path], bool]:
    from .adapters.drafter_pairing import DEFAULT_ROOTS

    return list(DEFAULT_ROOTS), True


def _config_sha256(path: str) -> str | None:
    try:
        return hashlib.sha256((Path(path) / "config.json").read_bytes()).hexdigest()
    except OSError:
        return None


def pair_auto_drafter(target: dict, spec: dict, drafters: list[dict]) -> tuple[dict | None, list[dict]]:
    """First on-disk drafter that verifies against ``spec``; and every refusal.

    Candidates are filtered by architecture, then (cheaply) by the evidence
    drafter's ``config.json`` digest when the spec names one, and finally
    verified by the target adapter's own inspector.  Only a drafter whose
    verified identity equals one of the spec's evidence identities is used.
    """
    from .adapters.drafter_pairing import pair_drafter

    architectures = list(spec["architectures"])
    identities = set(spec["identities"])
    configs = set(spec.get("config_sha256") or ())
    refusals = []
    for row in drafters:
        if (row.get("architectures") or [None])[0] not in architectures:
            continue
        if configs and _config_sha256(row["path"]) not in configs:
            refusals.append({"path": row["path"], "reason": "config.json differs from the evidence drafter"})
            continue
        try:
            pairing = pair_drafter(target, row, target_revision=False)
        except Exception as error:  # noqa: BLE001 - any inspector refusal is a refusal
            refusals.append({"path": row["path"], "reason": f"{type(error).__name__}: {str(error)[:300]}"})
            continue
        if pairing.get("identity") not in identities:
            refusals.append({
                "path": row["path"],
                "reason": f"identity {str(pairing.get('identity'))[:16]} is not an evidence drafter",
            })
            continue
        return pairing, refusals
    return None, refusals


# ------------------------------------------------------------ resolver --

def explicit_route_options(args) -> list[str]:
    named = [name for name in ROUTE_FLAGS if getattr(args, name, False)]
    for name in ROUTE_CONSTRAINING_OPTIONS:
        value = getattr(args, name, None)
        if value not in (None, False, (), []):
            named.append(name)
    return named


def _record(table: dict | None, status: str, **fields) -> dict:
    record = {"schema": RECORD_SCHEMA, "status": status, "applied": False,
              "qualification": "unqualified"}
    if table is not None:
        record["table"] = {"path": "src/mlx2/best_route_defaults.json", "sha256": table["_sha256"]}
    record.update(fields)
    return record


def _apply_spec(args, spec: dict, adapter_resolution, drafters_cache: dict, roots, include_hf) -> dict:
    """Apply one route spec to ``args``; raise ``LookupError`` to fail over."""
    from .contracts import Capability

    kind = spec["kind"]
    capabilities = adapter_resolution.descriptor.capabilities
    applied: dict = {"name": spec.get("name", kind), "kind": kind}
    if kind == "adapter_default":
        return applied
    if kind == "native_mtp":
        if Capability.MTP not in capabilities:
            raise LookupError("artifact has no native MTP route")
        args.native_mtp = True
        return applied
    if kind == "prompt_lookup":
        if Capability.PROMPT_LOOKUP not in capabilities:
            raise LookupError("adapter does not declare prompt lookup")
        args.prompt_lookup = True
        return applied
    if kind == "ordinary":
        args.ordinary = True
        return applied
    drafter = spec.get("drafter")
    if isinstance(drafter, dict):
        if "drafters" not in drafters_cache:
            from .adapters.drafter_pairing import discover_drafters

            drafters_cache["drafters"] = discover_drafters(roots, include_hf=include_hf)
        target = {"adapter": adapter_resolution.adapter_type.__name__, "path": str(args.model)}
        pairing, refusals = pair_auto_drafter(target, drafter, drafters_cache["drafters"])
        applied["refused_drafters"] = refusals
        if pairing is None:
            error = LookupError(
                f"no verified {'/'.join(drafter['architectures'])} drafter found "
                f"({len(refusals)} refused)"
            )
            error.refusals = refusals
            raise error
        applied["drafter"] = {
            "path": pairing["draft_path"],
            "identity": pairing["draft"],
            "kind": pairing["draft_kind"],
            "pins": dict(pairing["pins"]),
        }
        args.ast_drafter = pairing["draft_path"]
    elif drafter == "mtp":
        artifact = getattr(adapter_resolution, "artifact", None) or {}
        if not (artifact.get("has_mtp") or Capability.MTP in capabilities):
            raise LookupError("artifact has no native MTP head")
        args.ast_drafter = "mtp"
    elif drafter in ("pld",):
        args.ast_drafter = drafter
    options = spec.get("options") or {}
    for key, attribute in (("topology", "ast_topology"), ("tiles", "ast_tiles"),
                           ("tile_rows", "ast_tile_rows"), ("graft_pld", "ast_graft_pld")):
        if key in options:
            setattr(args, attribute, options[key])
    args.ast = True
    return applied


def resolve_best_route_default(
    args,
    adapter_resolution,
    *,
    table: dict | None = None,
    roots=None,
    include_hf: bool | None = None,
    streaming_selected: bool = False,
) -> dict:
    """Select the artifact's best-route default by setting the CLI intent.

    Mutates ``args`` (``ast``/``ast_drafter``/``native_mtp`` ...) only when a
    table entry selects a route and no explicit route option was given, and
    stores the selection record on ``args.route_default_selection``.  The
    record is always produced, also when nothing was applied, so status can
    say why the route is what it is.
    """
    if getattr(args, "no_auto_route", False):
        record = _record(None, "opted_out", reason="--no-auto-route")
        args.route_default_selection = record
        return record
    explicit = explicit_route_options(args)
    if explicit:
        record = _record(None, "explicit_route", explicit_options=explicit)
        args.route_default_selection = record
        return record
    execution_kind = getattr(getattr(adapter_resolution, "descriptor", None), "metadata", {}).get(
        "execution_kind")
    if execution_kind in {"embeddings", "rerank"}:
        record = _record(None, "not_applicable", reason=f"{execution_kind} adapter")
        args.route_default_selection = record
        return record
    if streaming_selected:
        # Every measured route was measured resident; automatic expert
        # streaming (single lane, resident load does not fit) is another
        # serving shape, so today's resolution stands.
        record = _record(None, "not_applicable",
                         reason="automatic expert streaming selected; best routes were measured resident")
        args.route_default_selection = record
        return record
    table = load_table() if table is None else table
    adapter_name = getattr(getattr(adapter_resolution, "adapter_type", None), "__name__", None)
    try:
        fingerprint = adapter_resolution.artifact_fingerprint
    except (AttributeError, ValueError):
        fingerprint = None
    entry = None if adapter_name is None else find_entry(table, adapter_name, fingerprint)
    if entry is None:
        record = _record(table, "no_entry", adapter=adapter_name, fingerprint=fingerprint,
                         reason="no measured best-route entry for this artifact; today's adapter default")
        args.route_default_selection = record
        return record
    common = {
        "adapter": adapter_name,
        "match": "checkpoint" if "fingerprint" in entry else "family",
        "checkpoint": entry.get("checkpoint"),
        "fingerprint": fingerprint,
        "reason": entry.get("reason"),
        "evidence": entry["evidence"],
    }
    if entry["status"] != "selected_unqualified":
        record = _record(table, entry["status"], **common)
        args.route_default_selection = record
        return record
    if roots is None:
        roots, default_hf = default_drafter_roots()
        include_hf = default_hf if include_hf is None else include_hf
    include_hf = True if include_hf is None else include_hf
    drafters_cache: dict = {}
    failover = []
    for spec in (entry["default"], *entry.get("fallback", ())):
        before = {name: getattr(args, name, None) for name in (
            "ast", "ast_drafter", "ast_topology", "ast_tiles", "ast_tile_rows", "ast_graft_pld",
            "native_mtp", "prompt_lookup", "ordinary")}
        try:
            applied = _apply_spec(args, spec, adapter_resolution, drafters_cache, roots, include_hf)
        except LookupError as error:
            for name, value in before.items():
                setattr(args, name, value)
            item = {"route": spec.get("name", spec["kind"]), "reason": str(error)}
            if getattr(error, "refusals", None):
                item["refused_drafters"] = error.refusals
            failover.append(item)
            continue
        primary = spec is entry["default"]
        record = _record(
            table,
            "selected_unqualified" if primary else "failed_over_unqualified",
            applied=True,
            default=applied["name"],
            route=applied,
            drafter=applied.get("drafter"),
            failover=failover,
            **common,
        )
        args.route_default_selection = record
        return record
    record = _record(table, "failed_over_to_adapter_default", failover=failover, **common)
    args.route_default_selection = record
    return record


def finish_best_route_default(args, policy):
    """Pin the verified drafter identity into the resolved policy.

    Called after ``apply_ast_invocation``.  The adapter re-verifies the pin at
    load, so a drafter that changed after pairing fails closed.
    """
    record = getattr(args, "route_default_selection", None)
    if not record or not record.get("applied"):
        return policy
    if getattr(args, "ast_route", None) is not None:
        record["route"]["ast_route"] = dict(args.ast_route)
    drafter = record.get("drafter")
    if not drafter or not drafter.get("pins"):
        return policy
    resolved = {} if policy is None else dict(policy)
    if resolved.get("draft_model") != drafter["path"]:
        return policy
    for key, value in drafter["pins"].items():
        if key in resolved and resolved[key] != value:
            raise ValueError(f"auto-paired drafter pin {key} conflicts with the policy")
        resolved[key] = value
    return resolved


def receipt_view(record: dict | None) -> dict | None:
    """The compact selection receipt carried by every request receipt."""
    if not record:
        return None
    view = {key: record[key] for key in ("status", "applied", "qualification") if key in record}
    for key in ("default", "checkpoint", "match", "reason"):
        if record.get(key) is not None:
            view[key] = record[key]
    evidence = record.get("evidence")
    if evidence:
        view["evidence"] = {"files": list(evidence.get("files", ()))[:3],
                            "source_revision": evidence.get("source_revision")}
    drafter = record.get("drafter")
    if drafter:
        view["drafter"] = {"identity": drafter.get("identity"), "path": drafter.get("path")}
    if record.get("failover"):
        view["failover"] = [item["route"] for item in record["failover"]]
    return view
