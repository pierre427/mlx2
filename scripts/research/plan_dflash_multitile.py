#!/usr/bin/env python3
"""CPU-only planner for long, serial, progressive, and eager DFlash work.

The probe keeps four mechanisms separate:

* native monolithic: one trained DFlash call and one full target verify;
* exact serial staging: tile i+1 is drafted only after tile i verifies and
  commits, so all draft and verify work remains reach-probability weighted;
* progressive verification: one native long proposal is target-verified in
  small chunks, with no second DFlash call;
* eager frontier precompute: future DFlash work starts at uncommitted leaves.

Acceptance comes from empirical accepted-count histograms and optional
conditional full-tile rates.  No independent per-token acceptance law is
invented.  This file never imports MLX, loads weights, executes a GPU, or
authorizes a serving route.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "mlx2.dflash-multitile-plan.v2"
K_VALUES = (4, 8, 12, 15)
TILE_COUNTS = (1, 2, 3)
FRONTIER_WIDTHS = (1, 2, 4)
PROGRESSIVE_VERIFY_TILES = (2, 3, 4, 5, 8, 15)
EAGER_APIS = frozenset(
    {
        "snapshot_speculative_frontier",
        "branch_speculative_frontier",
        "promote_speculative_frontier",
    }
)
PROGRESSIVE_APIS = frozenset(
    {"prepare_progressive_verification", "advance_progressive_verification"}
)


@dataclass(frozen=True)
class EmpiricalAcceptance:
    """Accepted-count observations and tile-level transition measurements."""

    accepted_count_histograms: Mapping[int, Mapping[int, int]]
    full_tile_rates_by_k: Mapping[int, Sequence[float]] | None = None

    def validate(self) -> None:
        if set(self.accepted_count_histograms) != set(K_VALUES):
            raise ValueError(f"accepted-count histograms must cover exactly {K_VALUES}")
        for k, histogram in self.accepted_count_histograms.items():
            if not histogram:
                raise ValueError(f"K={k} accepted-count histogram is empty")
            if any(
                type(accepted) is not int
                or not 0 <= accepted <= k
                or type(count) is not int
                or count <= 0
                for accepted, count in histogram.items()
            ):
                raise ValueError(f"K={k} histogram has an invalid bin")
        if self.full_tile_rates_by_k is None:
            return
        if set(self.full_tile_rates_by_k) != set(K_VALUES):
            raise ValueError(f"full-tile rates must cover exactly {K_VALUES}")
        for k, rates in self.full_tile_rates_by_k.items():
            if not 1 <= len(rates) <= 2:
                raise ValueError(f"K={k} needs one or two full-tile rates")
            if any(
                type(rate) not in (int, float)
                or not math.isfinite(rate)
                or not 0 <= rate <= 1
                for rate in rates
            ):
                raise ValueError(f"K={k} full-tile rates must be within [0, 1]")

    def mean_accepted(self, k: int) -> float:
        self.validate()
        histogram = self.accepted_count_histograms[k]
        total = sum(histogram.values())
        return sum(accepted * count for accepted, count in histogram.items()) / total

    def empirical_full_fraction(self, k: int) -> float:
        self.validate()
        histogram = self.accepted_count_histograms[k]
        return histogram.get(k, 0) / sum(histogram.values())

    def transition_rates(self, k: int, tiles: int) -> tuple[tuple[float, ...], str]:
        self.validate()
        if tiles not in TILE_COUNTS:
            raise ValueError(f"tiles must be one of {TILE_COUNTS}")
        needed = tiles - 1
        if needed == 0:
            return (), "not_applicable"
        if self.full_tile_rates_by_k is None:
            rate = self.empirical_full_fraction(k)
            return (rate,) * needed, "histogram_full_fraction_stationary_reuse"
        observed = tuple(float(value) for value in self.full_tile_rates_by_k[k])
        if len(observed) < needed:
            raise ValueError(f"K={k} lacks {needed} conditional transition rates")
        return observed[:needed], "empirical_conditional_full_tile_rates"


@dataclass(frozen=True)
class CostInputs:
    """Measured costs for one reached tile and a scheduler round boundary."""

    draft_ms_by_k: Mapping[int, float]
    verify_ms_by_k: Mapping[int, float]
    scheduler_boundary_ms: float

    def validate(self) -> None:
        if set(self.draft_ms_by_k) != set(K_VALUES):
            raise ValueError(f"draft costs must cover exactly {K_VALUES}")
        if set(self.verify_ms_by_k) != set(K_VALUES):
            raise ValueError(f"verify costs must cover exactly {K_VALUES}")
        values = (
            *self.draft_ms_by_k.values(),
            *self.verify_ms_by_k.values(),
            self.scheduler_boundary_ms,
        )
        if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
            raise ValueError("cost inputs must be finite numbers")
        if any(float(value) <= 0 for value in self.draft_ms_by_k.values()):
            raise ValueError("draft costs must be positive")
        if any(float(value) <= 0 for value in self.verify_ms_by_k.values()):
            raise ValueError("verify costs must be positive")
        if self.scheduler_boundary_ms < 0:
            raise ValueError("scheduler boundary cost must be nonnegative")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _class_methods(path: Path, class_name: str) -> set[str]:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    return set()


def _assigned_methods(path: Path, class_name: str) -> set[str]:
    found: set[str] = set()
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == class_name
            ):
                found.add(target.attr)
    return found


def _source_revision(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        snapshot = root / "SNAPSHOT_REVISION"
        if snapshot.is_file() and snapshot.read_text().strip():
            return snapshot.read_text().strip()
        return "unavailable"


def audit_source(root: Path = ROOT) -> dict[str, Any]:
    """Audit native, committed-serial, progressive, and eager seams."""

    dflash = root / "src/mlx2/runtime/drafters/dflash2.py"
    executor = root / "src/mlx2/runtime/external_speculative.py"
    config = root / "src/mlx2/runtime/drafters/dflash2_config.py"
    required = (dflash, executor, config)
    missing = [str(path.relative_to(root)) for path in required if not path.is_file()]
    if missing:
        return {
            "native_monolithic_ready": False,
            "serial_committed_state_ready": False,
            "progressive_same_proposal_api_ready": False,
            "eager_frontier_exact_state_ready": False,
            "blockers": ["required_source_files_missing"],
            "missing_files": missing,
            "source_files": {},
        }
    dflash_methods = _class_methods(dflash, "DFlash2DraftModel") | _assigned_methods(
        dflash, "DFlash2DraftModel"
    )
    executor_methods = _class_methods(executor, "ExternalDraftBatchGenerator")
    executor_text = executor.read_text()
    config_text = config.read_text()
    native_contracts = {"draft_distributions", "append_context", "batch_caches"}
    count_bound = "1 <= self.num_draft < draft_model.config.block_size" in executor_text
    trained_block_bound = (
        "runtime_block_size must be between 2 and block_size" in config_text
        and 'flat["runtime_block_size"] = min(5, int(flat["block_size"]))' in config_text
    )
    native_ready = native_contracts <= dflash_methods and count_bound and trained_block_bound
    serial_ready = native_ready and {"_propose", "_commit"} <= executor_methods
    progressive_ready = native_ready and PROGRESSIVE_APIS <= executor_methods
    eager_ready = native_ready and EAGER_APIS <= executor_methods
    blockers = []
    if not native_ready:
        blockers.append("native_monolithic_contract_missing")
    if not serial_ready:
        blockers.append("committed_serial_reentry_contract_missing")
    if not progressive_ready:
        blockers.append("progressive_same_proposal_api_missing")
    if not eager_ready:
        blockers.append("exact_uncommitted_frontier_api_missing")
    return {
        "native_monolithic_ready": native_ready,
        "serial_committed_state_ready": serial_ready,
        "progressive_same_proposal_api_ready": progressive_ready,
        "eager_frontier_exact_state_ready": eager_ready,
        "blockers": blockers,
        "observed_native_contracts": sorted(native_contracts & dflash_methods),
        "observed_serial_contracts": sorted({"_propose", "_commit"} & executor_methods),
        "observed_progressive_apis": sorted(PROGRESSIVE_APIS & executor_methods),
        "observed_eager_apis": sorted(EAGER_APIS & executor_methods),
        "required_progressive_apis": sorted(PROGRESSIVE_APIS),
        "required_eager_apis": sorted(EAGER_APIS),
        "trained_block_count_bound": count_bound,
        "runtime_block_bound": trained_block_bound,
        "source_files": {
            str(path.relative_to(root)): _sha256(path) for path in required
        },
    }


def _candidate_state(*, implemented: bool, exact_state: bool) -> dict[str, bool]:
    return {
        "implemented": implemented,
        "exact_state_available": exact_state,
        "qualified": False,
        "selected": False,
        "observed_used": False,
        "performance_claim": False,
    }


def tile_reach_probabilities(k: int, tiles: int, evidence: EmpiricalAcceptance):
    if k not in K_VALUES:
        raise ValueError(f"k must be one of {K_VALUES}")
    rates, source = evidence.transition_rates(k, tiles)
    reach = [1.0]
    for rate in rates:
        reach.append(reach[-1] * rate)
    return tuple(reach), source


def expected_serial_commits(k: int, tiles: int, evidence: EmpiricalAcceptance):
    reach, source = tile_reach_probabilities(k, tiles, evidence)
    commits_per_reached_tile = 1.0 + evidence.mean_accepted(k)
    return commits_per_reached_tile * sum(reach), reach, source


def progressive_geometry(
    k: int, verify_tile: int, histogram: Mapping[int, int]
) -> dict[str, float]:
    """Expected target rows/launches for one proposal verified incrementally."""

    if k not in K_VALUES:
        raise ValueError(f"k must be one of {K_VALUES}")
    if type(verify_tile) is not int or not 1 <= verify_tile <= k + 1:
        raise ValueError("verify tile must be within [1, K+1]")
    if not histogram or any(
        type(accepted) is not int
        or not 0 <= accepted <= k
        or type(count) is not int
        or count <= 0
        for accepted, count in histogram.items()
    ):
        raise ValueError("accepted-count histogram has an invalid bin")
    total = sum(histogram.values())
    if verify_tile == k:
        return {
            "fixed_target_rows": float(k + 1),
            "fixed_target_launches": 1.0,
            "expected_target_rows": float(k + 1),
            "expected_target_launches": 1.0,
            "target_row_reduction_fraction": 0.0,
        }
    rows = 0.0
    launches = 0.0
    for accepted, count in histogram.items():
        needed = accepted + 1  # rejection position or all-accepted bonus
        launch_count = math.ceil(needed / verify_tile)
        evaluated = min(k + 1, launch_count * verify_tile)
        rows += evaluated * count
        launches += launch_count * count
    expected_rows = rows / total
    expected_launches = launches / total
    return {
        "fixed_target_rows": float(k + 1),
        "fixed_target_launches": 1.0,
        "expected_target_rows": expected_rows,
        "expected_target_launches": expected_launches,
        "target_row_reduction_fraction": 1.0 - expected_rows / (k + 1),
    }


def _artifact_supports(k: int, artifact_block_size: int | None) -> bool | None:
    if artifact_block_size is None:
        return None
    if type(artifact_block_size) is not int or artifact_block_size < 2:
        raise ValueError("artifact block size must be an integer of at least 2")
    return k < artifact_block_size


def native_cell(k, capability, artifact_block_size=None, evidence=None, costs=None):
    source_ready = bool(capability["native_monolithic_ready"])
    artifact_ready = _artifact_supports(k, artifact_block_size)
    ready = source_ready and artifact_ready is True
    if not source_ready:
        decision = "refused_native_contract_missing"
    elif artifact_ready is None:
        decision = "requires_observed_artifact_block_size"
    elif not artifact_ready:
        decision = "refused_artifact_trained_block_too_short"
    else:
        decision = "native_probe_candidate"
    row: dict[str, Any] = {
        "mechanism": "native_monolithic",
        "k": k,
        "tiles": 1,
        "proposal_horizon": k,
        "candidate_state": _candidate_state(implemented=ready, exact_state=ready),
        "artifact_block_size": artifact_block_size,
        "decision": decision,
    }
    if evidence is not None:
        committed, reach, source = expected_serial_commits(k, 1, evidence)
        row.update(
            expected_committed_tokens=committed,
            tile_reach_probabilities=list(reach),
            acceptance_evidence=source,
        )
    if evidence is not None and costs is not None:
        total = costs.draft_ms_by_k[k] + costs.verify_ms_by_k[k] + costs.scheduler_boundary_ms
        row.update(expected_ms=total, ms_per_expected_commit=total / committed)
    return row


def serial_cell(
    k, tiles, capability, artifact_block_size=None, evidence=None, costs=None
):
    source_ready = bool(capability["serial_committed_state_ready"])
    artifact_ready = _artifact_supports(k, artifact_block_size)
    ready = source_ready and artifact_ready is True
    if not source_ready:
        decision = "refused_committed_serial_reentry_contract_missing"
    elif artifact_ready is None:
        decision = "requires_observed_artifact_block_size"
    elif not artifact_ready:
        decision = "refused_artifact_trained_block_too_short"
    else:
        decision = "serial_boundary_coalescing_probe_candidate"
    row: dict[str, Any] = {
        "mechanism": "exact_staged_serial",
        "k_per_tile": k,
        "tiles": tiles,
        "proposal_horizon": k * tiles,
        "artifact_block_size": artifact_block_size,
        "launch_rule": "next_tile_after_prior_full_verification_and_commit",
        "candidate_state": _candidate_state(implemented=False, exact_state=ready),
        "current_reference_available": ready,
        "decision": decision,
    }
    if evidence is None:
        return row
    committed, reach, source = expected_serial_commits(k, tiles, evidence)
    row.update(
        expected_committed_tokens=committed,
        tile_reach_probabilities=list(reach),
        acceptance_evidence=source,
    )
    if costs is None:
        return row
    reached = sum(reach)
    work = reached * (costs.draft_ms_by_k[k] + costs.verify_ms_by_k[k])
    separate = work + reached * costs.scheduler_boundary_ms
    staged = work + costs.scheduler_boundary_ms
    row.update(
        expected_draft_plus_verify_ms=work,
        separate_rounds_expected_ms=separate,
        staged_serial_expected_ms=staged,
        scheduler_boundary_savings_ms=costs.scheduler_boundary_ms * sum(reach[1:]),
        staged_ms_per_expected_commit=staged / committed,
        savings_source="scheduler_boundary_only",
    )
    return row


def progressive_cell(
    verify_tile, capability, artifact_block_size=None, evidence=None
):
    k = 15
    artifact_ready = _artifact_supports(k, artifact_block_size)
    fixed_control = verify_tile == k
    implemented = bool(
        (
            capability["native_monolithic_ready"]
            if fixed_control
            else capability["progressive_same_proposal_api_ready"]
        )
        and artifact_ready is True
    )
    if artifact_ready is None:
        decision = "requires_artifact_block_size_and_progressive_api"
    elif not artifact_ready:
        decision = "refused_artifact_trained_block_too_short"
    elif fixed_control and implemented:
        decision = "fixed_native_control"
    elif implemented:
        decision = "probe_candidate_shape_parity_required"
    else:
        decision = "requires_progressive_api_and_shape_parity_probe"
    row: dict[str, Any] = {
        "mechanism": "progressive_verify_same_native_proposal",
        "k": k,
        "verify_tile": verify_tile,
        "fixed_control": fixed_control,
        "artifact_block_size": artifact_block_size,
        "dflash_calls": 1,
        "redrafts_after_partial_verify": 0,
        "candidate_state": _candidate_state(
            implemented=implemented, exact_state=implemented
        ),
        "shape_sensitive_parity_gate": {
            "required": not fixed_control,
            "state": "not_applicable" if fixed_control else "unverified",
            "reason": (
                "native fixed-shape control"
                if fixed_control
                else "target forward row shape changes from fixed16 to progressive chunks"
            ),
        },
        "decision": decision,
    }
    if evidence is not None:
        row.update(
            progressive_geometry(
                k, verify_tile, evidence.accepted_count_histograms[k]
            )
        )
        row["acceptance_evidence"] = "empirical_accepted_count_histogram"
    return row


def eager_cell(k, tiles, width, capability, artifact_block_size=None):
    artifact_ready = _artifact_supports(k, artifact_block_size)
    ready = bool(
        capability["eager_frontier_exact_state_ready"]
        and artifact_ready is True
    )
    if artifact_ready is None:
        decision = "requires_observed_artifact_block_size"
    elif not artifact_ready:
        decision = "refused_artifact_trained_block_too_short"
    elif ready:
        decision = "frontier_probe_candidate_requires_empirical_costs"
    else:
        decision = "refused_exact_uncommitted_frontier_api_missing"
    return {
        "mechanism": "eager_top_k_frontier_precompute",
        "k_per_tile": k,
        "tiles": tiles,
        "frontier_leaves": width,
        "artifact_block_size": artifact_block_size,
        "launch_rule": "future_tile_before_parent_verification",
        "may_waste_work": True,
        "economics": "requires_empirical_leaf_coverage_and_batched_frontier_cost",
        "candidate_state": _candidate_state(implemented=False, exact_state=ready),
        "decision": decision,
    }


def build_plan(*, root=ROOT, artifact_block_size=None, evidence=None, costs=None):
    if costs is not None and evidence is None:
        raise ValueError("cost modeling requires empirical acceptance evidence")
    if evidence is not None:
        evidence.validate()
    if costs is not None:
        costs.validate()
    if artifact_block_size is not None:
        _artifact_supports(1, artifact_block_size)
    capability = audit_source(Path(root))
    return {
        "schema": SCHEMA,
        "source_revision": _source_revision(Path(root)),
        "probe_state": {
            "implemented": True,
            "cpu_validated": True,
            "gpu_executed": False,
            "qualified": False,
            "selected": False,
            "observed_used": False,
            "performance_claim": False,
        },
        "scope": {
            "native_k": list(K_VALUES),
            "serial_tile_counts": list(TILE_COUNTS),
            "progressive_verify_tiles": list(PROGRESSIVE_VERIFY_TILES),
            "frontier_widths": list(FRONTIER_WIDTHS),
            "tree15_is_separate_from_chain_k15": True,
            "ordinary_reference_preserved": True,
            "authorizes_serving": False,
        },
        "capability": capability,
        "artifact_observation": {
            "block_size": artifact_block_size,
            "k15_legal": _artifact_supports(15, artifact_block_size),
        },
        "acceptance_evidence": None if evidence is None else asdict(evidence),
        "cost_inputs": None if costs is None else asdict(costs),
        "native_monolithic": [
            native_cell(k, capability, artifact_block_size, evidence, costs)
            for k in K_VALUES
        ],
        "exact_staged_serial": [
            serial_cell(
                k, tiles, capability, artifact_block_size, evidence, costs
            )
            for k in K_VALUES
            for tiles in TILE_COUNTS
        ],
        "progressive_same_proposal": [
            progressive_cell(tile, capability, artifact_block_size, evidence)
            for tile in PROGRESSIVE_VERIFY_TILES
        ],
        "eager_frontier_precompute": [
            eager_cell(k, tiles, width, capability, artifact_block_size)
            for k in K_VALUES
            for tiles in TILE_COUNTS[1:]
            for width in FRONTIER_WIDTHS
        ],
        "formulae": {
            "commits_per_reached_serial_tile": "1 + empirical_mean(accepted_count)",
            "serial_tile_reach": "cumulative_product(empirical_conditional_full_tile_rates)",
            "serial_work": "sum(tile_reach)*(draft_ms+verify_ms)",
            "serial_candidate_savings": "scheduler_boundary_ms*sum(tile_reach[1:])",
            "progressive_target_rows": "mean(min(K+1, tile*ceil((accepted_count+1)/tile)))",
            "progressive_target_launches": "mean(ceil((accepted_count+1)/tile))",
            "per_token_independence_assumed": False,
        },
    }


def _parse_k_map(values, label):
    result = {}
    for value in values:
        try:
            key_text, number_text = value.split("=", 1)
            key, number = int(key_text), float(number_text)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"{label} values must use K=VALUE") from error
        if key in result:
            raise argparse.ArgumentTypeError(f"duplicate {label} K={key}")
        result[key] = number
    return result


def _parse_histograms(values):
    result = {}
    for value in values:
        try:
            key_text, bins_text = value.split("=", 1)
            key = int(key_text)
            bins = {}
            for item in bins_text.split(","):
                accepted_text, count_text = item.split(":", 1)
                bins[int(accepted_text)] = int(count_text)
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                "histogram values must use K=ACCEPTED:COUNT,..."
            ) from error
        if key in result:
            raise argparse.ArgumentTypeError(f"duplicate histogram K={key}")
        result[key] = bins
    return result


def _parse_rates(values):
    result = {}
    for value in values:
        try:
            key_text, rates_text = value.split("=", 1)
            key = int(key_text)
            rates = tuple(float(rate) for rate in rates_text.split(","))
        except ValueError as error:
            raise argparse.ArgumentTypeError(
                "full-tile rates must use K=RATE[,RATE]"
            ) from error
        if key in result:
            raise argparse.ArgumentTypeError(f"duplicate full-tile rates K={key}")
        result[key] = rates
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histogram", action="append", default=[], metavar="K=A:C,...")
    parser.add_argument("--full-tile-rates", action="append", default=[], metavar="K=R[,R]")
    parser.add_argument("--draft-ms", action="append", default=[], metavar="K=MS")
    parser.add_argument("--verify-ms", action="append", default=[], metavar="K=MS")
    parser.add_argument("--scheduler-boundary-ms", type=float, default=0.0)
    parser.add_argument("--artifact-block-size", type=int)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--require-eager-frontier", action="store_true")
    args = parser.parse_args(argv)
    evidence = None
    costs = None
    try:
        if args.histogram:
            evidence = EmpiricalAcceptance(
                accepted_count_histograms=_parse_histograms(args.histogram),
                full_tile_rates_by_k=(
                    _parse_rates(args.full_tile_rates) if args.full_tile_rates else None
                ),
            )
            evidence.validate()
        elif args.full_tile_rates:
            parser.error("--full-tile-rates requires --histogram evidence")
        if args.draft_ms or args.verify_ms:
            if evidence is None or not args.draft_ms or not args.verify_ms:
                parser.error("cost modeling requires histograms, draft costs, and verify costs")
            costs = CostInputs(
                draft_ms_by_k=_parse_k_map(args.draft_ms, "draft-ms"),
                verify_ms_by_k=_parse_k_map(args.verify_ms, "verify-ms"),
                scheduler_boundary_ms=args.scheduler_boundary_ms,
            )
            costs.validate()
    except (ValueError, argparse.ArgumentTypeError) as error:
        parser.error(str(error))
    result = build_plan(
        artifact_block_size=args.artifact_block_size,
        evidence=evidence,
        costs=costs,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        print(rendered, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
    if args.require_eager_frontier and not result["capability"][
        "eager_frontier_exact_state_ready"
    ]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
