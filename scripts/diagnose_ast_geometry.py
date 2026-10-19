#!/usr/bin/env python3
"""Source-bound AST geometry probe. Diagnostic overhead invalidates timings.

The server-only wrappers observe existing host decisions and array shapes. They
never read tensor values, add MLX evaluation, change proposals, or select routes.
Raw prompts, tokens and traces stay outside Git. Run under CPG and paired locks.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, is_dataclass
import functools
import hashlib
import importlib.machinery
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def load_execution_policy(default_path, override_path=None, required_enabled=()):
    """Load one explicit policy and fail closed on required feature gates."""
    path = Path(default_path if override_path is None else override_path).resolve()
    policy = json.loads(path.read_text())
    if not isinstance(policy, dict):
        raise ValueError("execution policy must be a JSON object")
    for key in required_enabled:
        if policy.get(key) is not True:
            raise ValueError(f"execution policy must explicitly enable {key}")
    encoded = json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
    return policy, path, hashlib.sha256(encoded).hexdigest()


def capture_options(content, prompt, position, options):
    """Capture only the previously localized one-based divergence position."""
    result = dict(options or {})
    if position is not None and content == prompt:
        if type(position) is not int or position < 1:
            raise ValueError("capture position must be a positive one-based index")
        result.update(logprobs=True, top_logprobs=20,
                      logprobs_start=position, logprobs_end=position)
    return result


def host_value(value):
    """Do not stringify/evaluate tensors or arbitrary runtime objects."""
    if value is None or type(value) in (str, bool, int, float):
        return value
    if is_dataclass(value):
        return host_value(asdict(value))
    if isinstance(value, dict):
        return {str(k): host_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [host_value(v) for v in value]
    raise TypeError(f"non-host diagnostic value: {type(value).__name__}")


def topology(paths):
    """Token-free trie: node IDs preserve exact edge sharing within this slate."""
    nodes, edges, branches = [], {}, []
    for path in paths:
        parent, branch = 0, []
        for depth, token in enumerate(path, 1):
            edge = (parent, int(token))
            if edge not in edges:
                edges[edge] = len(nodes) + 1
                nodes.append({"id": edges[edge], "parent": parent, "depth": depth})
            parent = edges[edge]
            branch.append(parent)
        branches.append(branch)
    return {"nodes": nodes, "paths": branches,
            "path_depths": list(map(len, branches)), "unique_nodes": len(nodes)}


class Recorder:
    def __init__(self, path, *, max_rounds=2048, max_events=512):
        self.output = path.open("x")
        self.current = ContextVar("ast_geometry_round", default=None)
        self.target_windows = ContextVar("ast_geometry_target_windows", default=None)
        self.rounds = 0
        self.max_rounds, self.max_events = max_rounds, max_events

    def event(self, kind, **fields):
        row = self.current.get()
        if row is None:
            return
        if len(row["events"]) >= self.max_events:
            raise RuntimeError("diagnostic event bound exceeded; trace is incomplete")
        row["events"].append({"kind": kind, **host_value(fields)})

    def install_lazily(self):
        """Let the adapter pin import-time model flags before touching runtime."""
        target = "mlx2.runtime.external_speculative"
        if target in sys.modules:
            self.install()
            return
        recorder = self

        class Hook:
            def find_spec(self, fullname, path=None, target_module=None):
                if fullname != target:
                    return None
                spec = importlib.machinery.PathFinder.find_spec(fullname, path)
                original = spec.loader

                class Loader:
                    def create_module(self, module_spec):
                        return original.create_module(module_spec)

                    def exec_module(self, module):
                        original.exec_module(module)
                        sys.meta_path.remove(hook)
                        recorder.install()

                spec.loader = Loader()
                return spec

        hook = Hook()
        sys.meta_path.insert(0, hook)

    def install(self):
        from mlx2.runtime import external_speculative as ext
        from mlx2.runtime import row_exact_verify as rev
        from mlx2.runtime.proposal_providers import ContinuationDraftModel

        cls = ext.ExternalDraftBatchGenerator
        original_round = cls._run_round

        @functools.wraps(original_round)
        def run_round(engine, cohort, *args, **kwargs):
            if self.rounds >= self.max_rounds:
                raise RuntimeError("diagnostic round bound exceeded")
            self.rounds += 1
            row = {"schema": "mlx2.ast-round-diagnostic.v1", "round": self.rounds,
                   "performance_evidence": False,
                   "sessions": [{"uid": int(lane.uid), "request_id": lane.proposal_request_id,
                                 "generated": lane.generated, "remaining": lane.maximum - lane.generated,
                                 "tail_shape": list(lane.tail.shape)} for lane in cohort],
                   "target_execution": engine.target_execution,
                   "configured_tile": engine.progressive_verification_tile,
                   "tile_policy": engine.progressive_verification_tile_policy,
                   "configured_branches": engine.progressive_alternative_branches,
                   "pack_rows": engine.progressive_cross_session_pack_rows,
                   "pack_mode": engine.progressive_cross_session_pack_mode,
                   "events": [], "completed": False}
            token = self.current.set(row)
            try:
                result = original_round(engine, cohort, *args, **kwargs)
                row["completed"] = True
                return result
            except BaseException as error:
                row["error_type"] = type(error).__name__
                raise
            finally:
                self.current.reset(token)
                self.output.write(json.dumps(row, allow_nan=False) + "\n")
                self.output.flush()

        cls._run_round = run_round
        original_draft = ContinuationDraftModel.draft_distributions

        @functools.wraps(original_draft)
        def draft(model, *args, **kwargs):
            result = original_draft(model, *args, **kwargs)
            for identity, selection, admission, diagnostics in zip(
                kwargs["proposal_contexts"], model.last_continuation_selections,
                model.last_continuation_admissions, model.last_continuation_source_diagnostics,
                strict=True,
            ):
                paths = [] if selection is None else selection.paths
                self.event("source_slate", request_id=identity["request_id"],
                           proposal_round=identity["round_id"],
                           topology=topology([path.tokens for path in paths]),
                           sources=[path.representative.source_id for path in paths],
                           ranking=[{"score": path.score, "authority": path.score_authority,
                                     "expected_accepted_tokens": path.expected_accepted_tokens}
                                    for path in paths],
                           admission=admission, source_geometry=diagnostics)
            return result

        ContinuationDraftModel.draft_distributions = draft
        original_propose = cls._propose

        @functools.wraps(original_propose)
        def propose(engine, cohort, *args, **kwargs):
            result = original_propose(engine, cohort, *args, **kwargs)
            for lane, primary in zip(cohort, result, strict=True):
                paths = engine._last_alternative_blocks.get(lane.uid, ())
                self.event("admitted_topology", uid=int(lane.uid),
                           primary_depth=0 if primary is None else len(primary.tokens),
                           topology=topology([block.tokens for block in paths]),
                           prefix=engine._last_alternative_prefix_tokens.get(lane.uid),
                           latch=engine._last_alternative_latch.get(lane.uid),
                           topology_kind=engine._last_alternative_topology.get(lane.uid),
                           downshift=engine._last_alternative_downshift.get(lane.uid),
                           source_budget=getattr(lane, "ast_source_budget", None),
                           execution_planner=(engine.ast_execution_planner.receipt()
                                              if getattr(engine, "ast_execution_planner", None) is not None else None),
                           prefix_search=engine._last_alternative_prefix_diagnostics.get(lane.uid))
            return result

        cls._propose = propose
        original_window = rev.window

        @contextmanager
        def window(record):
            records = self.target_windows.get()
            if records is not None and record is not None and all(record is not r for r in records):
                records.append(record)
            with original_window(record):
                yield record

        rev.window = window
        original_execute = ext.execute_progressive_target_rows

        @functools.wraps(original_execute)
        def execute(model, tokens, caches, capture_layers, **kwargs):
            # This diagnostic is intentionally source-bound: the caller owns
            # the authoritative tile-to-session map, including dynamic refill.
            caller = sys._getframe(1)
            state = caller.f_locals
            if caller.f_code is cls._progressive_verify_many.__wrapped__.__code__:
                active = state["active"]
                metadata = state["row_metadata"]
                mapping = [{"uid": int(state["states"][i]["lane"].uid),
                            "offset": int(state["states"][i]["offset"]),
                            "input_rows": len(state["inputs"][physical]),
                            "metadata": None if metadata is None else metadata[i]}
                           for physical, i in enumerate(active)]
                kind, layout = "target_tile", state["last_layout_receipt"]
            elif caller.f_code is cls._cohort_target_forward.__wrapped__.__code__:
                mapping = [{"uid": int(lane.uid)} for lane in state["cohort"]]
                kind, layout = "target_standard_exact", {"site": state["site"]}
            else:
                raise RuntimeError("target executor called outside the bound planners")
            windows = []
            token = self.target_windows.set(windows)
            try:
                result = original_execute(model, tokens, caches, capture_layers, **kwargs)
                self.event(kind, mapping=mapping,
                           shape=list(tokens.shape), layout=layout,
                           executor=result[2],
                           windows=[{"rows": w.rows, "stages": w.stages,
                                     "failures": w.failures} for w in windows])
                return result
            finally:
                self.target_windows.reset(token)

        ext.execute_progressive_target_rows = execute
        original_forward = cls._cohort_target_forward

        @functools.wraps(original_forward)
        def forward(engine, inputs, transaction, cohort, **kwargs):
            windows = []
            token = self.target_windows.set(windows)
            try:
                result = original_forward(engine, inputs, transaction, cohort, **kwargs)
                self.event("cohort_forward", uids=[int(l.uid) for l in cohort],
                           site=kwargs["site"], shape=list(inputs.shape),
                           windows=[{"rows": w.rows, "stages": w.stages,
                                     "failures": w.failures} for w in windows])
                return result
            finally:
                self.target_windows.reset(token)

        cls._cohort_target_forward = forward
        original_many = cls._progressive_verify_many

        @functools.wraps(original_many)
        def many(engine, cohort, blocks, **kwargs):
            result = original_many(engine, cohort, blocks, **kwargs)
            self.event("tile_group_result", uids=[int(l.uid) for l in cohort],
                       proposal_depths=[len(b.tokens) for b in blocks],
                       terminal=kwargs.get("terminal", True),
                       decisions=[{"accepted": d.accepted, "emitted": len(d.emitted),
                                   "commit_count": d.commit_count} for d in result[0]],
                       per_row=result[4], physical=result[5])
            return result

        cls._progressive_verify_many = many
        original_geometry = cls._record_ast_geometry

        @functools.wraps(original_geometry)
        def geometry(engine, lane, blocks, accepts, counts, committed, winner, details):
            result = original_geometry(engine, lane, blocks, accepts, counts, committed, winner, details)
            self.event("verified_selection", uid=int(lane.uid), accepts=accepts,
                       proposal_counts=counts, committed=committed, winner=winner, geometry=details)
            return result

        cls._record_ast_geometry = geometry
        original_commit = cls._commit

        @functools.wraps(original_commit)
        def commit(engine, cohort, decisions, *args, **kwargs):
            result = original_commit(engine, cohort, decisions, *args, **kwargs)
            self.event("committed", sessions=[{"uid": int(lane.uid), "generated": lane.generated,
                                               "accepted": decision.accepted,
                                               "emitted": len(decision.emitted)}
                                              for lane, decision in zip(cohort, decisions, strict=True)])
            return result

        cls._commit = commit


def serve(arguments):
    recorder = Recorder(Path(arguments[0]))
    recorder.install_lazily()
    from mlx2 import server
    sys.argv = ["mlx2.server", *arguments[2:]]
    try:
        server.main()
    finally:
        recorder.output.close()


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--serve-trace":
        return serve(sys.argv[2:])
    import benchmark_ast_repair as bench

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", choices=bench.CONFIGS, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--cpg-owner", type=Path, required=True)
    ap.add_argument("--start-max-c", type=float, default=65)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--ast-planner", action="store_true")
    ap.add_argument("--auto-tile", action="store_true")
    ap.add_argument(
        "--policy-path",
        type=Path,
        help="explicit execution policy; defaults to the historical config policy",
    )
    ap.add_argument(
        "--require-policy-enabled",
        action="append",
        default=[],
        metavar="KEY",
        help="fail unless the resolved policy explicitly sets KEY to true",
    )
    ap.add_argument("--capture-position", type=int, help="one-based, already localized divergence position")
    ap.add_argument("--capture-prompt", default="code_edit", choices=("short_math", "short_explain", "code_edit"))
    ap.add_argument("--capture-at", action="append", default=[], metavar="PROMPT=POSITION",
                    help="additional previously localized one-based position for a named prompt")
    args = ap.parse_args()
    if args.capture_position is not None and args.capture_position < 1:
        ap.error("capture-position must be positive")
    source = bench.source(ROOT)
    owned = bench.ownership(args.cpg_owner)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    smoke = bench.load(ROOT / "qualification/runs/ast-smokes-20261018/ast_smoke.py", "geometry_smoke")
    captures = {} if args.capture_position is None else {args.capture_prompt: args.capture_position}
    for capture in args.capture_at:
        label, separator, position = capture.partition("=")
        if not separator or label not in smoke.PROMPTS or not position.isdigit() or int(position) < 1:
            ap.error("capture-at must name an existing prompt and a positive one-based position")
        if label in captures and captures[label] != int(position):
            ap.error("capture-at must not override a different capture position")
        captures[label] = int(position)
    guards = bench.load(ROOT / "scripts/run_with_gpu_locks.py", "geometry_locks")
    if foreign := guards.foreign_model_processes():
        raise RuntimeError(f"foreign model processes: {foreign}")
    name, policy_path, lane, lane_policy = bench.CONFIGS[args.config]
    policy, resolved_policy_path, policy_sha256 = load_execution_policy(
        policy_path, args.policy_path, args.require_policy_enabled
    )
    if args.ast_planner:
        policy["continuation_pool"]["ast_planner"] = True
    if args.auto_tile:
        if not args.ast_planner:
            ap.error("auto-tile diagnostic requires ast-planner")
        policy["progressive_verification_tile"] = "auto"
    bench.atomic(args.out_dir / "policy.json", policy)
    identities = bench.inventory()
    meta = {"schema": "mlx2.ast-geometry-probe.v1", "config": args.config,
            "source_revision": source, "ownership": owned, "policy": policy,
            "policy_source": str(resolved_policy_path),
            "policy_sha256": policy_sha256,
            "checkpoint": identities["targets"][name]["checkpoint"], "inventory": identities,
            "performance_evidence": False, "qualification": False,
            "bounded_logprob_capture": {label: {"position_one_based": position, "top_logprobs": 20}
                                        for label, position in captures.items()},
            "shape": {"request_widths": [1, 4], "max_tokens": args.max_tokens,
                      "max_context": 32768, "max_lanes": 4, "max_inflight": 8, "cache_gib": 8},
            "status": "running"}
    bench.atomic(args.out_dir / "meta.json", meta)
    from mlx2.apple_telemetry import TemperatureSampler
    temps = TemperatureSampler()
    raw_stream, raw_concurrent = smoke.stream, smoke.concurrent

    def thermal_gate():
        deadline = time.monotonic() + 180
        while True:
            reading = temps.die_summary()
            if reading["die_max_c"] is None:
                raise RuntimeError("die temperature unavailable")
            if reading["die_max_c"] <= args.start_max_c:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("diagnostic thermal gate timed out")
            time.sleep(5)
        if any("VirtualMachine" in p["executable"] and p["cpu_percent"] > 100 for p in bench.host_load()):
            raise RuntimeError("busy VM in diagnostic window")
        return {**reading, "swap": smoke.swap_state(), "pmset": bench.command("pmset", "-g", "therm")}

    def checked_stream(*positional, **keywords):
        # Concurrent wrapper gates the whole cohort; no per-thread delay that
        # would change intended arrival overlap.
        for label, position in captures.items():
            keywords["request_options"] = capture_options(
                positional[2], smoke.PROMPTS[label], position, keywords.get("request_options"))
        row = raw_stream(*positional, **keywords)
        if (row.get("error") or not row.get("done") or not row.get("prompt_tokens")
                or not row.get("completion_tokens") or not row.get("route") or not row.get("token_ids")):
            row["error"] = row.get("error") or "incomplete stream/trace/route"
        return row

    def single(*positional, **keywords):
        before = thermal_gate()
        row = checked_stream(*positional, **keywords)
        row["thermal_before"], row["thermal_after"] = before, temps.die_summary()
        return row

    def concurrent(*positional, **keywords):
        before = thermal_gate()
        # raw_concurrent resolves smoke.stream globally in worker threads.
        smoke.stream = checked_stream
        try:
            result = raw_concurrent(*positional, **keywords)
        finally:
            smoke.stream = single
        result["thermal_before"], result["thermal_after"] = before, temps.die_summary()
        return result

    smoke.stream, smoke.concurrent = single, concurrent

    class ProcessProxy:
        def __getattr__(self, key):
            return getattr(subprocess, key)

        def Popen(self, argv, **keywords):
            if "--external-draft" in argv:
                assert argv[1:4] == ["-u", "-m", "mlx2.server"]
                argv = [argv[0], "-u", str(Path(__file__).resolve()), "--serve-trace",
                        str(args.out_dir / "rounds.jsonl"), "--", *argv[4:]]
            return subprocess.Popen(argv, **keywords)

    smoke.subprocess = ProcessProxy()
    serving = argparse.Namespace(python=sys.executable, model=str(bench.MODELS / name), max_context=32768,
                                cache_gib=8, lane_matmul=lane, lane_policy=lane_policy,
                                server_arg=["--int8-prefill", "off"], load_timeout=1200,
                                skip_writing_prefix_cache=False, max_tokens=args.max_tokens)
    # Exactly four distinct, short prompts allow real overlap; heterogeneous
    # completion lengths exercise retirement without a long-prefill confound.
    prompts = {**smoke.PROMPTS, "early_stop": "Reply with exactly this one word: READY"}
    assert len(prompts) == 4
    bench.atomic(args.out_dir / "prompts.json", prompts)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, interrupted)
    reports = {}
    try:
        for arm in ("ordinary", "instrumented"):
            bench.ownership(args.cpg_owner)
            if bench.source(ROOT) != source:
                raise RuntimeError("source changed during diagnostic")
            reports[arm] = smoke.run_arm(
                serving, arm, bench.arm_arguments(arm, args.out_dir / "policy.json"), args.out_dir,
                {"b1": prompts, "b4_cold": True, "b4_warm": True},
            )
            if foreign := guards.foreign_model_processes():
                raise RuntimeError(f"server survived cleanup: {foreign}")
        bench.atomic(args.out_dir / "verdicts.json", smoke.verdicts(reports["ordinary"], reports["instrumented"]))
        meta["status"] = "complete"
    except BaseException as error:
        meta["status"], meta["error_type"] = "failed", type(error).__name__
        raise
    finally:
        bench.atomic(args.out_dir / "meta.json", meta)


if __name__ == "__main__":
    main()
