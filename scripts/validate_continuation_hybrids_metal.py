#!/usr/bin/env python3
# ruff: noqa: EXE001
"""Owned M3 diagnostics for 15 COMPLETE paths on real tiny hybrid runtimes.

Synthetic random targets and draft heads exercise serving/cache contracts. They
are not compatible trained artifacts, qualification, or a performance benchmark.
The operator acquires GPU ownership externally; this script never takes locks.
"""

from __future__ import annotations

import argparse
import copy
import faulthandler
import hashlib
import importlib.util
import json
import signal
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HELPER_PATH = ROOT / "scripts/validate_parallel_hybrids_metal.py"
_spec = importlib.util.spec_from_file_location(
    "mlx2_hybrid_validation_helpers", HELPER_PATH
)
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
FAMILIES = _helpers.FAMILIES
KINDS = _helpers.KINDS
SCHEMA = "mlx2.complete-continuation-hybrids-metal.v1"


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--out", type=Path, required=True)
    result.add_argument("--deadline-seconds", type=int, default=900)
    result.add_argument("--expected-chip", choices=("M3",), default="M3")
    result.add_argument("--i-own-the-gpu", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def cell_plan():
    return [
        f"{family}_{kind}_{operation}"
        for family in FAMILIES
        for kind in KINDS
        for operation in ("pool15_prefix", "mixed_sampled", "late_failure_retry")
    ]


def preflight(args):
    if not 1 <= args.deadline_seconds <= 900:
        raise ValueError("deadline-seconds must be in [1,900]")
    if not args.dry_run and not args.i_own_the_gpu:
        raise ValueError(
            "Metal validation requires --i-own-the-gpu under operator ownership"
        )
    return {
        "schema": SCHEMA,
        "cells_planned": cell_plan(),
        "expected_chip": args.expected_chip,
        "deadline_seconds": args.deadline_seconds,
        "will_execute": not args.dry_run,
        "synthetic_weights": True,
        "trained_checkpoint_qualification": False,
        "qualified": False,
        "performance_claim": False,
        "gpu_training": False,
        "selected": False,
        "max_sequences": 15,
        "unit": "complete_continuation_sequences",
        "reference_modules": list(_helpers.REFERENCES),
    }


def source_identity():
    identity = _helpers.source_identity()
    files = dict(identity["files_sha256"])
    files[str(Path(__file__).resolve().relative_to(ROOT))] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()
    files[str(HELPER_PATH.relative_to(ROOT))] = hashlib.sha256(
        HELPER_PATH.read_bytes()
    ).hexdigest()
    return {
        "root": str(ROOT),
        "files_sha256": files,
        "sha256": hashlib.sha256(
            json.dumps(files, sort_keys=True).encode()
        ).hexdigest(),
    }


class Validation(_helpers.Validation):
    def make_batch(self, family, kind):
        from mlx2.adapters.proposal_path_sources import build_continuation_drafter
        from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
        from mlx2.runtime.proposal_pool import ProposalRankingRegistry

        model, geometry = self.target(family)
        base = self.draft(model, geometry, kind, windows=[4], lilicorr_topk=4)
        binding = hashlib.sha256(
            f"synthetic-pool15-{family}-{kind}".encode()
        ).hexdigest()
        draft = build_continuation_drafter(
            model,
            base,
            {"sources": ["external"], "limit": 15},
            target_revision=binding,
            draft_revision="b" * 64,
            tokenizer_revision="c" * 64,
            session_revision=binding,
        )
        # Each diagnostic cell uses an explicitly isolated registry. Separate
        # threaded host tests cover shared global/model/session ranking.
        draft.proposal_pool.ranking_registry = ProposalRankingRegistry()
        batch = ExternalDraftBatchGenerator(
            model,
            draft_model=draft,
            binding=binding,
            num_draft=2,
            prefill_step_size=4,
            completion_batch_size=2,
            ready_drain="all",
        )
        return model, base, draft, batch

    def prefill(self, batch, prompts, sampling=None):
        from mlx2.runtime.sample_utils import LaneRNG

        ids = batch.insert(
            prompts,
            max_tokens=[8] * len(prompts),
            lane_rngs=[LaneRNG(907 + i) for i in range(len(prompts))],
            sampling_configs=sampling,
        )
        lanes = [batch.lanes[uid] for uid in ids]
        for lane in lanes:
            while lane.anchor is None:
                batch._prefill(lane)
        return lanes

    def prefix_state(self, model, lane, label):
        cache = model.make_cache()
        self.mx.eval(model(self.mx.array([lane.history]), cache=cache))
        self.state(lane.cache, cache, label)
        actual = model(self.mx.array([[lane.anchor]]), cache=copy.deepcopy(lane.cache))[
            0, -1
        ]
        reference = model(self.mx.array([[lane.anchor]]), cache=cache)[0, -1]
        self.compare(label + ".next_logits", actual, reference)

    def tracked(self, model):
        original, calls = model.forward_with_taps, []

        def forward(tokens, *args, **kwargs):
            calls.append(tuple(int(v) for v in tokens.shape))
            return original(tokens, *args, **kwargs)

        model.forward_with_taps = forward
        return original, calls

    def pool15_prefix(self, family, kind):
        model, _base, draft, batch = self.make_batch(family, kind)
        prompt = [1, 2, 3]
        lane = self.prefill(batch, [prompt])[0]
        original, calls = self.tracked(model)
        try:
            batch._round([lane])
            actual = [row.token for row in lane.ready]
            expected = self.ordinary(model, prompt, len(actual))
            self.current.update(
                actual_token_ids=actual,
                expected_token_ids=expected,
                physical_verification_shapes=calls,
                receipt=lane.ready[-1].speculative_receipt,
                stats=dict(batch.scheduler_stats),
            )
            if (
                calls != [(15, 3)]
                or len(draft.last_continuation_selections[0].paths) != 15
            ):
                raise AssertionError(
                    "exactly 15 complete two-token paths were not physically verified"
                )
            if actual != expected:
                raise AssertionError("pool greedy tokens differ from ordinary target")
            if batch.scheduler_stats.get("external_continuation_target_rows") != 45:
                raise AssertionError("actual 15x3 target rows disagree with receipt")
            self.prefix_state(model, lane, "selected_committed_prefix")
            batch._sidecar(lane).validate(batch.binding, len(lane.history))
            if (
                draft.proposal_pool.feedback_revision != 1
                or draft.proposal_pool._pending
            ):
                raise AssertionError(
                    "successful round did not settle exactly one feedback ticket"
                )
        finally:
            model.forward_with_taps = original
            batch.close()

    def mixed_sampled(self, family, kind):
        from mlx2.runtime.speculative_sampling import softmax

        model, _base, draft, batch = self.make_batch(family, kind)
        prompts = [[1, 2, 3], [2, 3, 4, 5]]
        lanes = self.prefill(
            batch, prompts, [{"sampling_temp": 0}, {"sampling_temp": 0.8}]
        )
        original, calls = self.tracked(model)
        original_law, observations = batch._target_law, []

        def law(lane, logits, history, *args):
            result = original_law(lane, logits, history, *args)
            oracle = model(self.mx.array([history]), cache=model.make_cache())[0, -1]
            if lane.sampling["sampling_temp"]:
                self.compare(
                    f"sampled.{lane.uid}.prefix{len(history)}.target_law",
                    result,
                    softmax(self.host(oracle), 0.8),
                    atol=5e-4,
                    rtol=5e-4,
                )
            elif int(self.np.argmax(result)) != int(self.mx.argmax(oracle).item()):
                raise AssertionError(
                    "greedy actual-prefix target law differs from ordinary"
                )
            observations.append(
                {
                    "uid": lane.uid,
                    "history": list(history),
                    "sampling_temp": lane.sampling["sampling_temp"],
                }
            )
            return result

        batch._target_law = law
        try:
            batch._round(lanes)
            if calls != [(15, 3), (15, 3)]:
                raise AssertionError(
                    "mixed requests did not execute two independent 15-path forwards"
                )
            for index, lane in enumerate(lanes):
                emitted = [row.token for row in lane.ready]
                if lane.rng.draws != len(emitted):
                    raise AssertionError(
                        "continuation walk did not draw once per emitted target prefix"
                    )
                if sum(row["uid"] == lane.uid for row in observations) != len(emitted):
                    raise AssertionError(
                        "target law was evaluated outside reached actual prefixes"
                    )
                self.prefix_state(model, lane, f"mixed_request{index}")
                batch._sidecar(lane).validate(batch.binding, len(lane.history))
            self.current.update(
                physical_verification_shapes=calls,
                target_law_observations=observations,
                token_ids={
                    lane.uid: [row.token for row in lane.ready] for lane in lanes
                },
                receipts={
                    lane.uid: lane.ready[-1].speculative_receipt for lane in lanes
                },
                stats=dict(batch.scheduler_stats),
                ranking=draft.proposal_pool.receipt(),
            )
        finally:
            batch._target_law = original_law
            model.forward_with_taps = original
            batch.close()

    def late_failure_retry(self, family, kind):
        from mlx2.runtime.lilicorr_feedback import LiLiCorrFeedbackManager

        model, base, draft, batch = self.make_batch(family, kind)
        with tempfile.TemporaryDirectory(
            prefix="mlx2-continuation-feedback-"
        ) as directory:
            manager = None
            if kind == "lilicorr":
                manager = LiLiCorrFeedbackManager(
                    base,
                    {
                        "directory": directory,
                        "min_examples": 8,
                        "train_every": 16,
                        "steps": 1,
                    },
                    target_revision=batch.binding,
                    draft_revision="b" * 64,
                    binding=batch.binding,
                )
                base.feedback_manager = manager
            lanes = self.prefill(
                batch,
                [[1, 2, 3], [2, 3, 4, 5]],
                [{"sampling_temp": 0}, {"sampling_temp": 0.8}],
            )
            before = [
                (
                    list(lane.history),
                    lane.anchor,
                    lane.rng.snapshot(),
                    copy.deepcopy(lane.cache),
                    copy.deepcopy(lane.draft_cache),
                    self.host(lane.tail).copy(),
                )
                for lane in lanes
            ]
            original, calls = self.tracked(model)

            def fail(tokens, *args, **kwargs):
                result = model_forward(tokens, *args, **kwargs)
                if len(calls) == 2:
                    raise RuntimeError(
                        "injected after later request private branch write"
                    )
                return result

            model_forward = model.forward_with_taps
            model.forward_with_taps = fail
            try:
                try:
                    batch._round(lanes)
                except RuntimeError as error:
                    if "injected after" not in str(error):
                        raise
                else:
                    raise AssertionError("late private-branch failure was not raised")
                if (
                    draft.proposal_pool.feedback_revision
                    or draft.proposal_pool._pending
                ):
                    raise AssertionError(
                        "failed outer round published ranking labels or retained tickets"
                    )
                if manager is not None and manager.stats["committed"]:
                    raise AssertionError(
                        "failed outer round submitted LiLiCoRR teacher labels"
                    )
                for index, (lane, checkpoint) in enumerate(zip(lanes, before)):
                    history, anchor, rng, cache, draft_cache, tail = checkpoint
                    if (
                        lane.history != history
                        or lane.anchor != anchor
                        or lane.rng.snapshot() != rng
                    ):
                        raise AssertionError(
                            "later-request failure changed committed history or RNG"
                        )
                    if (
                        lane.ready
                        or lane.generated
                        or getattr(lane, "continuation_coverage", {})
                    ):
                        raise AssertionError(
                            "failed round retained outputs or adaptive labels"
                        )
                    self.state(lane.cache, cache, f"rollback.request{index}.target")
                    for layer, (actual, expected) in enumerate(
                        zip(lane.draft_cache, draft_cache)
                    ):
                        if getattr(actual, "keys", None) is None:
                            if (
                                getattr(expected, "keys", None) is not None
                                or actual.offset != expected.offset
                            ):
                                raise AssertionError(
                                    "empty draft cache changed after rollback"
                                )
                        else:
                            self.state(
                                [actual],
                                [expected],
                                f"rollback.request{index}.draft{layer}",
                            )
                    self.compare(
                        f"rollback.request{index}.tail", lane.tail, tail, atol=0, rtol=0
                    )
                failed_shapes = list(calls)
                model.forward_with_taps = model_forward
                calls.clear()
                batch._round(lanes)
                if (
                    calls != [(15, 3), (15, 3)]
                    or draft.proposal_pool.feedback_revision != 2
                ):
                    raise AssertionError(
                        "retry did not settle two real15-path target verifications"
                    )
                if manager is not None and manager.stats["committed"] != 2:
                    raise AssertionError(
                        "successful retry did not submit exactly two verified teacher payloads"
                    )
                for index, lane in enumerate(lanes):
                    self.prefix_state(model, lane, f"retry.request{index}")
                self.current.update(
                    failed_physical_shapes=failed_shapes,
                    retry_physical_shapes=list(calls),
                    ranking=draft.proposal_pool.receipt(),
                    feedback=None if manager is None else manager.receipt(),
                    stats=dict(batch.scheduler_stats),
                    feedback_training_executed=False,
                )
            finally:
                model.forward_with_taps = original
                batch.close()
                if manager is not None:
                    manager.close()

    def run(self):
        for family in FAMILIES:
            for kind in KINDS:
                for operation in (
                    "pool15_prefix",
                    "mixed_sampled",
                    "late_failure_retry",
                ):
                    self.cell(
                        f"{family}_{kind}_{operation}",
                        lambda family=family, kind=kind, operation=operation: getattr(
                            self, operation
                        )(family, kind),
                    )


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    try:
        report = preflight(args)
    except ValueError as error:
        arguments.error(str(error))
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report.update(
        progress_path=str(args.out),
        passed=False,
        started_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    faulthandler.enable()

    def deadline(_signal, _frame):
        raise TimeoutError(f"validation exceeded {args.deadline_seconds} seconds")

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(args.deadline_seconds)
    try:
        report["source_identity"] = source_identity()
        Validation(report).run()
        report["passed"] = bool(report["cells"]) and all(
            cell["passed"] for cell in report["cells"]
        )
        report["source_unchanged"] = (
            source_identity()["sha256"] == report["source_identity"]["sha256"]
        )
        if not report["source_unchanged"]:
            report.update(passed=False, error="source changed during validation")
    except Exception as error:  # noqa: BLE001 - preserve diagnostic failures
        report.update(
            passed=False,
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
    finally:
        signal.alarm(0)
    report["finished_at_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "out": str(args.out),
                "error": report.get("error"),
            }
        ),
        flush=True,
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
