#!/usr/bin/env python3
"""Offline oracle for confidence-gated self-MTP draft looping (DLoop, untrained).

DLoop (arXiv 2610.07659) keeps drafting another stage of ``k`` tokens while the
drafter's summed log-probability over the stage it just produced stays at or
above a threshold ``g``, then verifies every staged token in one target
forward.  For an autoregressive head such as Qwen3.8's MTP layer, the extra
stages need no retraining: the head already feeds its own hidden state back.

This script answers "would it pay on this host?" from two measured inputs and
no serving change:

``--log``
    ``mlx2.mtp_acceptance_log.v1`` rows from ``mtp_confidence_gpu.py collect``
    with greedy lookahead.  Rows must carry ``emitted``/``position``/``request``
    (added 2026-10-09), so each request's committed stream can be rebuilt and
    every drafted position, verified or lookahead, labelled exactly: under
    greedy decoding draft ``j`` of a cycle would be accepted iff drafts
    ``1..j`` equal the committed tokens at those offsets.
``--cost``
    ``mlx2.mtp_verify_cost.v2`` from ``mtp_confidence_gpu.py profile``: the
    measured seconds of a whole self-MTP cycle at each fixed depth.

Each logged cycle is treated as one sample of (match length, per-position
confidence).  A policy maps a sample to a depth ``d``; it commits
``min(match, d) + 1`` tokens and costs ``cycle(d)``.  The estimate is
``E[cycle] / E[tokens]`` per policy, the same truncation argument as
``best_constant_depth.py``.  It ignores that a different depth changes the
next cycle's anchor; the serving A/B is the check on that.

Policies reported: fixed depths, the DLoop gate at each ``g``, and the oracle
gate that extends only when the stage it just drafted will be fully accepted
(the paper's "Oracle" row).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

LOG_SCHEMA = "mlx2.mtp_acceptance_log.v1"
COST_SCHEMA = "mlx2.mtp_verify_cost.v2"


def load_rows(paths):
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if data.get("schema") != LOG_SCHEMA:
                    raise SystemExit(f"{path}: unsupported schema {data.get('schema')!r}")
                rows.append(data)
    return rows


def split_requests(rows):
    """Give every row a request key that is unique across the whole log.

    A generator may reuse a uid for later requests (the server's B1 lane is
    always uid 0), so a uid's rows are split into epochs: a row whose
    position does not move past the previous row of the same uid starts a
    new request.  Rows must be in file (commit) order.
    """
    last = {}
    epoch = defaultdict(int)
    for row in rows:
        uid = row.get("request", -1)
        if uid in last and row.get("position", -1) <= last[uid]:
            epoch[uid] += 1
        last[uid] = row.get("position", -1)
        row["request_key"] = f"{uid}.{epoch[uid]}"
    return rows


def build_streams(rows):
    """Per request: {generated index: token}, from each cycle's delivered tokens.

    Returns (streams, ends, problems).  ``ends[request]`` is one past the last
    known index.  A cycle whose delivered tokens disagree with an earlier one
    at the same index, or whose accepted drafts disagree with what it
    delivered, is a problem: the log is not a single greedy trajectory.
    """
    streams = defaultdict(dict)
    problems = []
    for row in rows:
        if row.get("position", -1) < 0 or row.get("request", -1) < 0:
            raise SystemExit(
                "log rows lack position/request: collect with a tree that records "
                "emitted tokens (mtp_confidence 2026-10-09 or later)"
            )
        request, start, emitted = row["request_key"], row["position"], row["emitted"]
        accepted = min(row["accepted"], len(emitted))
        if list(row["tokens"][:accepted]) != list(emitted[:accepted]):
            problems.append(("accepted_mismatch", request, start))
        stream = streams[request]
        for offset, token in enumerate(emitted):
            seen = stream.setdefault(start + offset, token)
            if seen != token:
                problems.append(("stream_conflict", request, start + offset))
    ends = {}
    for request, stream in streams.items():
        # The prefill-sampled first token is never a cycle's output, so a
        # stream usually starts at position 1, not 0.
        end = min(stream)
        while end in stream:
            end += 1
        ends[request] = end
    return streams, ends, problems


def label_cycles(rows, streams, ends, depth):
    """Samples with exact match lengths over the first ``depth`` drafts.

    A cycle is kept when the committed stream is known at every position its
    drafts could reach before the first mismatch.  Positions are unknown past
    the end of a request, and where an unlogged cycle (a copy-draft span, for
    one) delivered the tokens.  Returns (samples, dropped).
    """
    samples = []
    dropped = 0
    for row in rows:
        tokens = row["tokens"]
        if len(tokens) < depth or len(row["features"]) < depth:
            dropped += 1
            continue
        request, start = row["request_key"], row["position"]
        stream = streams[request]
        match = 0
        while match < depth and start + match in stream and stream[start + match] == tokens[match]:
            match += 1
        if match < depth and start + match not in stream:
            dropped += 1
            continue
        top1 = [max(float(f[0]), 1e-12) for f in row["features"][:depth]]
        samples.append({"match": match, "logq": [math.log(p) for p in top1]})
    return samples, dropped


def load_cost(path, lanes):
    data = json.loads(Path(path).read_text())
    if data.get("schema") != COST_SCHEMA:
        raise SystemExit(f"{path}: need schema {COST_SCHEMA}, got {data.get('schema')!r}")
    seconds = defaultdict(list)
    for item in data["raw"]:
        if int(item["lanes"]) == lanes:
            seconds[int(item["depth"])].append(float(item["seconds_per_cycle"]))
    if not seconds:
        raise SystemExit(f"{path}: no raw cycles for lanes={lanes}")
    return {depth: statistics.median(values) for depth, values in sorted(seconds.items())}


def cycle_cost(table, depth):
    """Measured cycle seconds at ``depth``; linear between measured depths only."""
    if depth in table:
        return table[depth]
    lower = max((d for d in table if d < depth), default=None)
    upper = min((d for d in table if d > depth), default=None)
    if lower is None or upper is None:
        raise SystemExit(f"cost table does not bracket depth {depth}: {sorted(table)}")
    weight = (depth - lower) / (upper - lower)
    return table[lower] * (1 - weight) + table[upper] * weight


def _boundaries(stage, max_stages, boundaries=None):
    if boundaries:
        return list(boundaries)
    return [stage * (i + 1) for i in range(max_stages)]


def gated_depth(sample, stage, max_stages, threshold, boundaries=None):
    """Depth reached when each stage must score >= threshold to continue.

    ``boundaries`` (increasing depths) overrides uniform stages, so a host
    can stop exactly where its verify cost steps (e.g. 3 then 7 drafts for
    an 8-row tile).  The score of a stage is the sum over its own drafts.
    """
    ends = _boundaries(stage, max_stages, boundaries)
    reached = 0
    for previous, end in zip([0] + ends[:-1], ends):
        reached = end
        if end == ends[-1] or sum(sample["logq"][previous:end]) < threshold:
            break
    return reached


def oracle_depth(sample, stage, max_stages, boundaries=None):
    ends = _boundaries(stage, max_stages, boundaries)
    reached = ends[0]
    for end in ends[1:]:
        if sample["match"] < reached:
            break
        reached = end
    return reached


def evaluate(samples, depth_of, table):
    tokens = rows = cost = 0.0
    histogram = defaultdict(int)
    for sample in samples:
        depth = depth_of(sample)
        histogram[depth] += 1
        tokens += min(sample["match"], depth) + 1
        rows += depth + 1
        if table is not None:
            cost += cycle_cost(table, depth)
    n = len(samples)
    out = {
        "tau": tokens / n,
        "mean_verify_rows": rows / n,
        "depths": dict(sorted(histogram.items())),
    }
    if table is not None:
        out["ms_per_cycle"] = 1e3 * cost / n
        out["ms_per_token"] = 1e3 * cost / tokens
    return out


def run(args):
    rows = split_requests(load_rows(args.log))
    bounds = _boundaries(args.stage, args.max_stages, args.boundaries)
    if bounds != sorted(set(bounds)) or bounds[0] < 1:
        raise SystemExit(f"boundaries must be increasing positive depths: {bounds}")
    depth = bounds[-1]
    max_decisions = len(bounds) - 1
    streams, ends, problems = build_streams(rows)
    samples, dropped = label_cycles(rows, streams, ends, depth)
    if not samples:
        raise SystemExit(f"no cycle has {depth} labelled drafted positions")
    table = load_cost(args.cost, args.lanes) if args.cost else None

    report = {
        "cycles_logged": len(rows),
        "cycles_used": len(samples),
        "cycles_dropped_short_or_censored": dropped,
        "stream_problems": len(problems),
        "boundaries": bounds,
        "lanes": args.lanes,
        "cost_table_ms": None if table is None else {d: 1e3 * s for d, s in table.items()},
        "gate_sync_ms": args.gate_sync_ms,
        "full_stage_accept_ratio": [
            sum(s["match"] >= end for s in samples)
            / max(1, sum(s["match"] >= previous for s in samples))
            for previous, end in zip([0] + bounds[:-1], bounds)
        ],
        "policies": {},
    }
    if problems:
        report["first_problems"] = problems[:10]

    def add(name, depth_of, stages_of=None):
        result = evaluate(samples, depth_of, table)
        if table is not None and stages_of is not None:
            # One host readback per gate decision: after every drafted stage
            # except the last allowed one, whether it extends or stops.
            syncs = sum(min(stages_of(s), max_decisions) for s in samples) / len(samples)
            per_cycle = result["ms_per_cycle"] + args.gate_sync_ms * syncs
            result["ms_per_cycle"] = per_cycle
            result["ms_per_token"] = per_cycle / result["tau"]
        report["policies"][name] = result

    for fixed in range(1, depth + 1):
        if table is None or fixed in table or (min(table) < fixed < max(table)):
            add(f"fixed_{fixed}", lambda s, d=fixed: d)
    def stages(depth_value):
        return bounds.index(depth_value) + 1

    for threshold in args.thresholds:
        add(
            f"gate_{threshold:g}",
            lambda s, g=threshold: gated_depth(s, 0, 0, g, bounds),
            lambda s, g=threshold: stages(gated_depth(s, 0, 0, g, bounds)),
        )
    add(
        "oracle_gate",
        lambda s: oracle_depth(s, 0, 0, bounds),
        lambda s: stages(oracle_depth(s, 0, 0, bounds)),
    )
    if table is not None:
        base = report["policies"][f"fixed_{bounds[0]}"]["ms_per_token"]
        for result in report["policies"].values():
            result["speedup_vs_first_boundary"] = base / result["ms_per_token"]
    return report


def acceptance_prior(samples, base, threshold, depth, source):
    """Two-stage prior for load-time topology selection.

    For a lane that drafts ``base`` tokens and extends to ``end`` when the
    first stage scores at least ``threshold``: the extension rate and the
    mean committed tokens per verify with and without the extension, for
    every ``end`` the log can label.  Costs are left to the host's probe.
    """
    extended = [s for s in samples if sum(s["logq"][:base]) >= threshold]
    stopped = [s for s in samples if sum(s["logq"][:base]) < threshold]
    mean = lambda items, d: sum(min(s["match"], d) + 1 for s in items) / max(1, len(items))
    return {
        "schema": "mlx2.draft_loop_prior.v1",
        "source": source,
        "cycles": len(samples),
        "base": base,
        "threshold": threshold,
        "extend_rate": len(extended) / len(samples),
        "tokens_if_stopped": mean(stopped, base),
        "tokens_at_base": mean(samples, base),
        "tokens_if_extended": {str(end): mean(extended, end) for end in range(base + 1, depth + 1)},
        # A lane that failed its own gate but drafts on because another lane
        # in its padded cohort passed (cohort rule "any").
        "tokens_if_stopped_extended": {
            str(end): mean(stopped, end) for end in range(base + 1, depth + 1)
        },
    }


def acceptance_prior_grid(samples, base, thresholds, depth, source):
    """``acceptance_prior`` at each threshold, for operator-chosen gates."""
    rows = []
    for threshold in thresholds:
        prior = acceptance_prior(samples, base, threshold, depth, source)
        rows.append({k: prior[k] for k in (
            "threshold", "extend_rate", "tokens_if_stopped", "tokens_if_extended",
            "tokens_if_stopped_extended")})
    return {
        "schema": "mlx2.draft_loop_prior_grid.v1",
        "source": source,
        "cycles": len(samples),
        "base": base,
        "tokens_at_base": sum(min(s["match"], base) + 1 for s in samples) / len(samples),
        "grid": rows,
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--log", nargs="+", required=True, type=Path)
    parser.add_argument("--cost", type=Path)
    parser.add_argument("--lanes", type=int, default=1)
    parser.add_argument("--stage", type=int, default=3, help="drafts per stage (k)")
    parser.add_argument("--max-stages", type=int, default=3)
    parser.add_argument("--boundaries", type=int, nargs="+",
                        help="explicit increasing stage-end depths, e.g. 3 7; overrides --stage/--max-stages")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[-0.25, -0.5, -0.75, -1.0])
    parser.add_argument("--gate-sync-ms", type=float, default=0.3,
                        help="host readback charged per extra stage decision")
    parser.add_argument("--emit-prior", type=float, metavar="THRESHOLD",
                        help="write the two-stage acceptance prior at this threshold instead of a report")
    parser.add_argument("--emit-prior-grid", type=float, nargs=3, metavar=("START", "STOP", "STEP"),
                        help="write priors at thresholds START, START-STEP, ... STOP (e.g. -0.05 -2.0 0.05)")
    parser.add_argument("--out", type=Path)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.emit_prior_grid is not None:
        start, stop, step = args.emit_prior_grid
        count = int(round((start - stop) / step)) + 1
        thresholds = [round(start - i * step, 6) for i in range(count)]
        rows = split_requests(load_rows(args.log))
        streams, ends, _ = build_streams(rows)
        depth = max(_boundaries(args.stage, args.max_stages, args.boundaries))
        samples, _ = label_cycles(rows, streams, ends, depth)
        grid = acceptance_prior_grid(
            samples, args.stage, thresholds, depth, [str(p) for p in args.log]
        )
        text = json.dumps(grid, indent=1)
        if args.out:
            args.out.write_text(text)
        print(text[:400])
        return
    if args.emit_prior is not None:
        rows = split_requests(load_rows(args.log))
        streams, ends, _ = build_streams(rows)
        depth = max(_boundaries(args.stage, args.max_stages, args.boundaries))
        samples, _ = label_cycles(rows, streams, ends, depth)
        prior = acceptance_prior(
            samples, args.stage, args.emit_prior, depth, [str(p) for p in args.log]
        )
        text = json.dumps(prior, indent=1)
        if args.out:
            args.out.write_text(text)
        print(text)
        return
    report = run(args)
    text = json.dumps(report, indent=1)
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    sys.exit(main())
