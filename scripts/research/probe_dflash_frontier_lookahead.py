#!/usr/bin/env python3
"""Source-bound gate and economics probe for DFlash frontier lookahead.

The proposed mechanism drafts one additional tile for the most likely one,
two, or four leaves of the current DFlash proposal.  Reusing such work is
correct only when every extension is made from the exact target and draft
state of its parent leaf.  A token tail alone is not sufficient: DFlash2's
next proposal consumes target hidden features through ``append_context``.

This research probe intentionally does not patch the serving route.  It audits
the checked-out DFlash implementation, emits a dry-run plan by default, and
fails closed before importing MLX when the exact speculative-leaf state seam
is absent.  That makes a negative M3 result useful evidence without turning a
token-only continuation into an approximate state operation.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DFLASH = ROOT / "src/mlx2/runtime/drafters/dflash2.py"
EXECUTOR = ROOT / "src/mlx2/runtime/external_speculative.py"
SCHEMA = "mlx2.dflash-frontier-lookahead-probe.v1"
ARMS = ("on_demand", "top_1", "top_2", "top_4")


class ExactFrontierUnavailable(RuntimeError):
    """The source cannot construct an exact state for an uncommitted leaf."""


@dataclass(frozen=True)
class FrontierIdentity:
    """All revisions required to promote one private extension exactly."""

    request_id: str
    generator_revision: str
    context_revision: int
    round_id: int
    parent_path_digest: str
    draft_source_revision: str
    sampler_revision: str

    def __post_init__(self) -> None:
        text_fields = (
            self.request_id,
            self.generator_revision,
            self.parent_path_digest,
            self.draft_source_revision,
            self.sampler_revision,
        )
        if any(not isinstance(value, str) or not value for value in text_fields):
            raise ValueError("frontier identity strings must be nonempty")
        if self.context_revision < 0 or self.round_id < 0:
            raise ValueError("frontier revisions must be nonnegative")


@dataclass
class PrivateExtension:
    """Research-only private state; never an APCv2 or target-state entry."""

    identity: FrontierIdentity
    tokens: tuple[int, ...]
    state: Any

    def __post_init__(self) -> None:
        if not self.tokens or any(type(token) is not int or token < 0 for token in self.tokens):
            raise ValueError("an extension needs nonnegative integer tokens")


class PrivateFrontierStore:
    """Single-use exact-parent store used by the CPU contract tests.

    A mismatch discards all private work for the request.  It never returns a
    nearest revision and never publishes state to APCv2.
    """

    def __init__(self) -> None:
        self._entries: dict[FrontierIdentity, PrivateExtension] = {}

    def put(self, extension: PrivateExtension) -> None:
        if extension.identity in self._entries:
            raise ValueError("duplicate frontier identity")
        self._entries[extension.identity] = extension

    def claim(self, identity: FrontierIdentity) -> PrivateExtension | None:
        exact = self._entries.pop(identity, None)
        if exact is not None:
            return exact
        self.discard_request(identity.request_id)
        return None

    def discard_request(self, request_id: str) -> int:
        doomed = [key for key in self._entries if key.request_id == request_id]
        for key in doomed:
            del self._entries[key]
        return len(doomed)

    def __len__(self) -> int:
        return len(self._entries)


def path_digest(tokens: tuple[int, ...] | list[int]) -> str:
    if not tokens or any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("parent path needs nonnegative integer tokens")
    payload = json.dumps(list(tokens), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def source_revision(root: Path = ROOT) -> str:
    snapshot = root / "SNAPSHOT_REVISION"
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        if snapshot.is_file():
            value = snapshot.read_text().strip()
            if value:
                return value
        raise RuntimeError("source revision is unavailable") from None


def _method_names(path: Path, class_name: str) -> set[str]:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return {
                child.name
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
    return set()


def audit_exact_frontier_capability() -> dict[str, Any]:
    """Return a source-bound capability result without importing MLX."""

    dflash_text = DFLASH.read_text()
    executor_text = EXECUTOR.read_text()
    generator_methods = _method_names(EXECUTOR, "ExternalDraftBatchGenerator")
    required_dflash = {
        "DFlash2DraftModel.draft_distributions",
        "DFlash2DraftModel.append_context",
        "DFlash2DraftModel.batch_caches",
    }
    observed_dflash = {
        name for name in required_dflash if name in dflash_text
    }
    exact_api_names = {
        "snapshot_speculative_frontier",
        "branch_speculative_frontier",
        "promote_speculative_frontier",
    }
    observed_exact_apis = sorted(exact_api_names & generator_methods)
    append_uses_target_hidden = (
        "def _append_context(self, hidden, cache):" in dflash_text
        and "self.fc(hidden)" in dflash_text
    )
    commit_precedes_prelaunch = (
        "self._commit(\n                cohort," in executor_text
        and "self._prelaunch_tree(cohort[0], decisions[0])" in executor_text
        and executor_text.index("self._commit(\n                cohort,")
        < executor_text.index("self._prelaunch_tree(cohort[0], decisions[0])")
    )
    supports_exact = bool(
        observed_dflash == required_dflash
        and append_uses_target_hidden
        and exact_api_names <= generator_methods
    )
    blockers = []
    if observed_dflash != required_dflash:
        blockers.append("required_dflash_contract_missing")
    if not append_uses_target_hidden:
        blockers.append("dflash_target_hidden_dependency_not_proven")
    if exact_api_names - generator_methods:
        blockers.append("no_exact_uncommitted_leaf_state_api")
    if not commit_precedes_prelaunch:
        blockers.append("committed_boundary_pipeline_order_not_proven")
    return {
        "supports_exact_frontier_lookahead": supports_exact,
        "blockers": blockers,
        "observed_dflash_contracts": sorted(observed_dflash),
        "observed_exact_frontier_apis": observed_exact_apis,
        "dflash_next_tile_requires_target_hidden": append_uses_target_hidden,
        "current_pipeline_begins_after_commit": commit_precedes_prelaunch,
        "source_files": {
            str(DFLASH.relative_to(ROOT)): sha256_file(DFLASH),
            str(EXECUTOR.relative_to(ROOT)): sha256_file(EXECUTOR),
        },
    }


def evaluate_economics(
    *,
    width: int,
    hit_rate: float,
    single_tile_ms: float,
    batched_tile_ms: float,
    overlap_fraction: float,
    promotion_ms: float = 0.0,
) -> dict[str, Any]:
    """Evaluate one arm without claiming a serving performance result."""

    if width not in (0, 1, 2, 4):
        raise ValueError("width must be 0, 1, 2, or 4")
    finite = (hit_rate, single_tile_ms, batched_tile_ms, overlap_fraction, promotion_ms)
    if any(not math.isfinite(value) for value in finite):
        raise ValueError("economics inputs must be finite")
    if not 0 <= hit_rate <= 1 or not 0 <= overlap_fraction <= 1:
        raise ValueError("rates must be within [0, 1]")
    if single_tile_ms <= 0 or batched_tile_ms < 0 or promotion_ms < 0:
        raise ValueError("timings must be nonnegative and single-tile time positive")
    if width == 0:
        return {
            "arm": "on_demand",
            "width": 0,
            "expected_delta_ms": 0.0,
            "break_even": True,
            "performance_claim": False,
        }
    exposed = (1.0 - overlap_fraction) * batched_tile_ms
    expected_delta = hit_rate * single_tile_ms - exposed - promotion_ms
    return {
        "arm": f"top_{width}",
        "width": width,
        "expected_saved_on_hit_ms": hit_rate * single_tile_ms,
        "exposed_precompute_ms": exposed,
        "promotion_ms": promotion_ms,
        "expected_delta_ms": expected_delta,
        "break_even": expected_delta >= 0,
        "performance_claim": False,
    }


def build_plan(
    revision: str,
    capability: dict[str, Any],
    *,
    single_tile_ms: float | None = None,
    batched_tile_ms: dict[int, float] | None = None,
    hit_rates: dict[int, float] | None = None,
    overlap_fraction: float = 0.0,
) -> dict[str, Any]:
    economics = []
    if single_tile_ms is not None and batched_tile_ms is not None and hit_rates is not None:
        economics.append(
            evaluate_economics(
                width=0,
                hit_rate=1.0,
                single_tile_ms=single_tile_ms,
                batched_tile_ms=0.0,
                overlap_fraction=0.0,
            )
        )
        for width in (1, 2, 4):
            economics.append(
                evaluate_economics(
                    width=width,
                    hit_rate=hit_rates[width],
                    single_tile_ms=single_tile_ms,
                    batched_tile_ms=batched_tile_ms[width],
                    overlap_fraction=overlap_fraction,
                )
            )
    return {
        "schema": SCHEMA,
        "attempt_id": uuid.uuid4().hex,
        "created_at": datetime.now(UTC).isoformat(),
        "source_revision": revision,
        "mode": "dry_run",
        "status": "planned",
        "metal_executed": False,
        "arms": list(ARMS),
        "reference_arm": "on_demand",
        "candidate_contract": {
            "extra_tiles": 1,
            "state_visibility": "private_until_exact_parent_commits",
            "parent_match": "request_generator_context_round_path_and_source_revisions",
            "on_mismatch": "discard_without_publication",
            "apcv2_publication": False,
            "target_state_publication_before_commit": False,
        },
        "required_observations": {
            "primary_metric": "end_to_end_ms_through_next_committed_tile",
            "per_arm": [
                "lookahead_attempts",
                "exact_parent_hits",
                "exact_parent_misses",
                "useful_extension_tokens",
                "discarded_extension_tokens",
                "private_state_peak_bytes",
                "verifier_delay_ms",
                "draft_extension_ms",
                "promotion_ms",
                "net_time_saved_ms",
            ],
            "correctness": [
                "committed_token_parity_with_on_demand",
                "committed_target_state_digest_parity",
                "committed_draft_state_digest_parity",
                "request_generator_context_round_parent_source_sampler_revision_match",
            ],
        },
        "decision_gate": {
            "requires_exact_state": True,
            "requires_token_and_state_parity": True,
            "requires_positive_end_to_end_delta": True,
            "raw_draft_throughput_is_not_sufficient": True,
        },
        "capability": capability,
        "economics": economics,
        "probe_state": {
            "implemented": True,
            "cpu_validated": True,
        },
        "candidate_state": {
            "implemented": False,
            "qualified": False,
            "selected": False,
            "observed_used": False,
            "performance_claim": False,
        },
    }


def write_immutable(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _triplet(value: str) -> dict[int, float]:
    items = value.split(",")
    if len(items) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated values for top-1,2,4")
    try:
        parsed = [float(item) for item in items]
    except ValueError as error:
        raise argparse.ArgumentTypeError("values must be numbers") from error
    return dict(zip((1, 2, 4), parsed))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="run the exact-state capability gate")
    parser.add_argument("--i-own-the-gpu", action="store_true")
    parser.add_argument("--expect-source-commit")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--single-tile-ms", type=float)
    parser.add_argument("--batched-tile-ms", type=_triplet)
    parser.add_argument("--hit-rates", type=_triplet)
    parser.add_argument("--overlap-fraction", type=float, default=0.0)
    args = parser.parse_args(argv)

    revision = source_revision()
    if args.expect_source_commit and revision != args.expect_source_commit:
        parser.error(
            f"source revision {revision!r} does not match {args.expect_source_commit!r}"
        )
    timing_args = (args.single_tile_ms, args.batched_tile_ms, args.hit_rates)
    if any(value is not None for value in timing_args) and not all(
        value is not None for value in timing_args
    ):
        parser.error("timing economics require --single-tile-ms, --batched-tile-ms and --hit-rates")
    capability = audit_exact_frontier_capability()
    receipt = build_plan(
        revision,
        capability,
        single_tile_ms=args.single_tile_ms,
        batched_tile_ms=args.batched_tile_ms,
        hit_rates=args.hit_rates,
        overlap_fraction=args.overlap_fraction,
    )
    exit_code = 0
    if args.execute:
        if not args.i_own_the_gpu:
            parser.error("--execute requires --i-own-the-gpu")
        receipt["mode"] = "execute"
        if not capability["supports_exact_frontier_lookahead"]:
            receipt["status"] = "refused_exact_state_unavailable"
            receipt["refusal_stage"] = "pre_mlx_import"
            receipt["metal_executed"] = False
            exit_code = 2
        else:
            # The capability names above are the serving seam a future
            # implementation must provide.  Never silently substitute a
            # token-only extension when the source starts advertising them.
            receipt["status"] = "refused_research_executor_not_implemented"
            receipt["refusal_stage"] = "pre_mlx_import"
            receipt["metal_executed"] = False
            exit_code = 2
    if args.output:
        write_immutable(args.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
