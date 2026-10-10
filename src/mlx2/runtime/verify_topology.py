"""Load-time verify-topology probe and draft-loop selection.

The draft loop (``draft_loop.py``) only pays where extra verify rows are
cheap, and where they are cheap depends on the host: on an M3 Pro with the
SIMD lane matmul an 8-row verify fills one tile and costs about what 5 rows
cost on stock kernels, while a 9th row starts a second tile; on stock
kernels every extra row costs a near-constant step and the loop loses.  This
module measures that at load and picks the topology:

1. ``probe_row_costs``: one forward of ``R`` tokens on a fresh cache for each
   ``R`` up to ``max_rows``, the medians giving the host's row-cost staircase.
   A ``tile edge`` is the last ``R`` before cost steps up.
2. ``probe_cycle_costs``: real self-MTP cycles (draft and verify) at the base
   depth and at each candidate end depth, ``edge - 1`` drafts for every edge.
3. ``select_topology``: combine those cycle costs with an acceptance prior
   at the operator's gate threshold (a grid measured on Qwen3.8-27B; an
   adapter may supply its own) and keep the end depth with the best
   predicted ms per token, or no loop when no end beats fixed depth by
   ``MIN_PREDICTED_GAIN``.

The result is cached per device, MLX build, model identity, lane law and
base depth, so only the first load of a configuration pays for the probe.
Original mlx2 code; see ``provenance/dloop-untrained-gate.json``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

SCHEMA = "mlx2.verify_topology.v2"
DEFAULT_THRESHOLD = -0.4
# Selection margin: an end depth is chosen only when its predicted ms/token
# beats fixed depth by this factor.  The prior is a different model's
# acceptance on another prompt mix, so a small predicted edge is not
# evidence; the M3 measurement landed within 4% of its prediction.
MIN_PREDICTED_GAIN = 1.05
# A row count is a tile edge when the next row costs this much more.
EDGE_STEP = 0.08
# A probe whose base cycle moves by more than this between its first and
# last measurement ran under contention (another GPU job, a thermal step):
# its costs are not trusted, cached or selected from.
MAX_PROBE_DRIFT = 0.15
# Host readback per gate decision, charged in the prediction.
GATE_SYNC_SECONDS = 0.001
# Deepest end depth considered: the acceptance prior labels drafts to 9.
MAX_END = 9
# Cohort widths the load-time probe measures.  Wider cohorts already fill a
# lane tile at the base depth on the hosts measured (M3 Pro 8-row SIMD tile,
# M5 16-row MPP fragment), and no rule paid at 4 lanes on either
# (qualification/runs/dloop-width-20261009/); they keep fixed depth.
PROBE_WIDTHS = (1, 2)
# Widest cohort an operator may ask the probe to consider.
MAX_PROBE_WIDTH = 4
# Long-form text for the cycle probe, so no lane finishes inside the timed
# window; the probe's acceptance does not matter, only the cycle's cost.
PROBE_TEXT = (
    "Explain, in numbered sections with examples, how a compiler turns source "
    "code into machine code: lexing, parsing, semantic analysis, intermediate "
    "representations, optimisation passes, register allocation and code emission."
)

# Two-stage acceptance prior for base depth 3 at gate thresholds -0.05 to
# -2.0 in steps of 0.05: Qwen3.8-27B 4-bit + MTP, 917 exactly labelled greedy
# cycles over the mixed prompt set (scripts/dloop_oracle.py --emit-prior-grid
# on qualification/runs/dloop-untrained-20261009/phase0/accept-27b-la6.jsonl).
PRIOR_RESOURCE = "draft_loop_prior_qwen38_27b.json"


def load_prior_grid(name: str = PRIOR_RESOURCE) -> dict:
    from importlib.resources import files

    return json.loads(files("mlx2.runtime").joinpath("data", name).read_text())


def prior_at(grid: Mapping[str, Any], threshold: float) -> Optional[dict]:
    """The two-stage prior at ``threshold``, linear between grid points.

    None outside the grid: a threshold the log never sampled has no
    evidence behind its predicted gain.
    """
    rows = sorted(grid["grid"], key=lambda row: row["threshold"])
    threshold = float(threshold)
    if not rows or not rows[0]["threshold"] - 1e-9 <= threshold <= rows[-1]["threshold"] + 1e-9:
        return None
    for low, high in zip(rows, rows[1:] + rows[-1:]):
        if low["threshold"] - 1e-9 <= threshold <= high["threshold"] + 1e-9:
            span = high["threshold"] - low["threshold"]
            w = 0.0 if span == 0 else (threshold - low["threshold"]) / span
            mix = lambda a, b: (1 - w) * a + w * b
            return {
                "base": grid["base"],
                "threshold": threshold,
                "tokens_at_base": grid["tokens_at_base"],
                "extend_rate": mix(low["extend_rate"], high["extend_rate"]),
                "tokens_if_stopped": mix(low["tokens_if_stopped"], high["tokens_if_stopped"]),
                "tokens_if_extended": {
                    end: mix(low["tokens_if_extended"][end], high["tokens_if_extended"][end])
                    for end in low["tokens_if_extended"]
                },
                "tokens_if_stopped_extended": {
                    end: mix(low["tokens_if_stopped_extended"][end],
                             high["tokens_if_stopped_extended"][end])
                    for end in low.get("tokens_if_stopped_extended", {})
                },
            }
    return None


def threshold_range(grid: Optional[Mapping[str, Any]] = None) -> tuple[float, float]:
    values = [row["threshold"] for row in (grid or load_prior_grid())["grid"]]
    return min(values), max(values)


def tile_edges(row_costs: Mapping[int, float], *, step: float = EDGE_STEP) -> list[int]:
    """Row counts after which the next row costs at least ``step`` more."""
    rows = sorted(int(r) for r in row_costs)
    edges = []
    for r, nxt in zip(rows, rows[1:]):
        if nxt == r + 1 and row_costs[nxt] > row_costs[r] * (1.0 + step):
            edges.append(r)
    return edges


def candidate_ends(base: int, row_costs: Mapping[int, float], max_end: int, width: int = 1) -> list[int]:
    """End depths worth timing at ``width`` lanes.

    A padded cohort verifies ``width * (depth + 1)`` rows, so for every tile
    edge the candidate is the deepest depth whose rows still fit under it,
    plus the cap.
    """
    width = int(width)
    ends = {edge // width - 1 for edge in tile_edges(row_costs)}
    ends.add(max_end)
    return sorted(e for e in ends if base < e <= max_end)


def predict(base: int, end: int, cycle_seconds: Mapping[int, float], prior: Mapping[str, Any],
            width: int = 1):
    """Predicted (ms/token at fixed base, ms/token with the loop) at ``width``.

    Lanes are independent draws from the prior.  The cohort rule is
    ``any``: the cohort extends when at least one lane's first stage passes,
    and every lane then drafts to ``end`` -- a lane that failed its own gate
    yields ``tokens_if_stopped_extended``.  At one lane this is the per-lane
    gate.
    """
    width = int(width)
    p = float(prior["extend_rate"])
    passed = float(prior["tokens_if_extended"][str(end)])
    stopped = float(prior["tokens_if_stopped"])
    carried = (
        float(prior["tokens_if_stopped_extended"][str(end)])
        if width > 1 else stopped
    )
    tokens = 0.0
    for k in range(width + 1):
        weight = math.comb(width, k) * p ** k * (1 - p) ** (width - k)
        tokens += weight * (width * stopped if k == 0 else k * passed + (width - k) * carried)
    none = (1 - p) ** width
    seconds = none * cycle_seconds[base] + (1 - none) * cycle_seconds[end] + GATE_SYNC_SECONDS
    fixed = cycle_seconds[base] / (width * float(prior["tokens_at_base"]))
    return 1e3 * fixed, 1e3 * seconds / tokens


def select_width(base, cycle_seconds, prior, width, min_gain):
    scored = []
    for end in sorted(cycle_seconds):
        if end <= base or str(end) not in prior["tokens_if_extended"]:
            continue
        if width > 1 and str(end) not in prior.get("tokens_if_stopped_extended", {}):
            continue
        fixed, looped = predict(base, end, cycle_seconds, prior, width)
        scored.append({"end": end, "fixed_ms_per_token": fixed,
                       "loop_ms_per_token": looped, "predicted_gain": fixed / looped})
    best = max(scored, key=lambda item: item["predicted_gain"], default=None)
    chosen = best["end"] if best is not None and best["predicted_gain"] >= min_gain else None
    return chosen, scored


def select_topology(
    base: int,
    cycles_by_width: Mapping[int, Mapping[int, float]],
    grid: Mapping[str, Any],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    min_gain: float = MIN_PREDICTED_GAIN,
) -> dict:
    """Per-width end depths; ``selected`` is a ``by_width`` draft loop or None."""
    prior = prior_at(grid, threshold) if int(grid["base"]) == base else None
    if prior is None:
        return {"selected": None, "reason": "prior does not cover this base depth and threshold"}
    by_width, candidates = {}, {}
    for width in sorted(cycles_by_width):
        chosen, scored = select_width(base, cycles_by_width[width], prior, width, min_gain)
        candidates[str(width)] = scored
        if chosen is not None:
            by_width[str(width)] = [base, chosen]
    if not by_width:
        return {"selected": None, "candidates": candidates,
                "reason": f"no width has an end depth predicted >= {min_gain}x over fixed depth {base}"}
    return {
        "selected": {"by_width": by_width, "threshold": float(threshold), "cohort": "any"},
        "candidates": candidates,
        "reason": "best predicted gain per width",
    }


def probe_row_costs(model, vocab_size: int, max_rows: int, *, reps: int = 3) -> dict[int, float]:
    """Median seconds of one ``R``-token forward on a fresh cache, ``R = 1..max_rows``."""
    import mlx.core as mx

    from .models.cache import make_prompt_cache

    def once(rows: int) -> float:
        cache = make_prompt_cache(model)
        tokens = mx.random.randint(0, vocab_size, (1, rows), dtype=mx.uint32)
        mx.eval(tokens)
        mx.synchronize()
        start = time.perf_counter()
        mx.eval(model(tokens, cache=cache))
        mx.synchronize()
        return time.perf_counter() - start

    once(max_rows)  # warm-up: kernels compile and the allocator settles
    samples: dict[int, list[float]] = {r: [] for r in range(1, max_rows + 1)}
    for _ in range(reps):
        # Interleave row counts so drift does not masquerade as a tile edge.
        for rows in range(1, max_rows + 1):
            samples[rows].append(once(rows))
    mx.clear_cache()
    return {r: statistics.median(v) for r, v in samples.items()}


def probe_cycle_costs(
    model, prompt: Sequence[int], depths: Sequence[int], *, cycles: int = 12, warmup: int = 4,
    self_mtp: Optional[Mapping[str, Any]] = None, width: int = 1,
) -> dict[int, float]:
    """Median seconds of one real self-MTP cycle of ``width`` lanes at each depth."""
    import mlx.core as mx

    from .generate import BatchGenerator
    from .sample_utils import LaneRNG

    result = {}
    for depth in depths:
        config = {"persistent": True, "segment_aware_live_tip": True,
                  "segment_aware_cohort_size": width, **(self_mtp or {}), "num_draft": int(depth)}
        config.pop("draft_loop", None)
        gen = BatchGenerator(model, completion_batch_size=width, prefill_batch_size=width,
                             prefill_step_size=2048, self_mtp=config)
        gen.insert([list(prompt)] * width,
                   max_tokens=[(warmup + cycles) * (int(depth) + 2) + 64] * width,
                   lane_rngs=[LaneRNG(i) for i in range(width)],
                   self_mtp_configs=[{"sampling_temp": 0.0}] * width)
        times = []
        seen = 0
        last = None
        while len(times) < cycles:
            _, responses = gen.next()
            if not responses:
                continue
            mx.synchronize()
            now = time.perf_counter()
            # Each call that returns responses is one committed cycle.
            if last is not None and seen > warmup:
                times.append(now - last)
            seen += 1
            last = now
            if any(r.finish_reason for r in responses):
                break
        gen.close()
        mx.clear_cache()
        if not times:
            raise RuntimeError(f"cycle probe at depth {depth} produced no timed cycles")
        result[int(depth)] = statistics.median(times)
    return result


def self_mtp_digest(self_mtp: Optional[Mapping[str, Any]]) -> str:
    """Stable digest of a self-MTP execution policy without its draft_loop.

    ``probe_cycle_costs`` times cycles under the policy minus ``draft_loop``;
    everything else in it (num_draft aside, which is the base) can change a
    cycle's cost, so it keys the cached selection.  Non-JSON values are
    digested by their repr."""
    policy = {k: v for k, v in dict(self_mtp or {}).items() if k != "draft_loop"}
    encoded = json.dumps(policy, sort_keys=True, default=repr).encode()
    return hashlib.sha256(encoded).hexdigest()


def cache_key(parts: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:32]


def cache_dir() -> Path:
    root = os.environ.get("MLX2_VERIFY_TOPOLOGY_CACHE")
    return Path(root) if root else Path.home() / ".cache" / "mlx2" / "verify-topology"


def resolve_topology(
    model,
    *,
    identity: Mapping[str, Any],
    base: int,
    max_end: int,
    vocab_size: int,
    prompt: Sequence[int],
    prior: Optional[Mapping[str, Any]] = None,
    threshold: float = DEFAULT_THRESHOLD,
    self_mtp: Optional[Mapping[str, Any]] = None,
    use_cache: bool = True,
    widths: Sequence[int] = (1,),
) -> dict:
    """Probe (or load) this host's verify topology and select a draft loop.

    ``identity`` must pin everything that changes costs: device, MLX build,
    model fingerprint and lane law.  Returns a receipt; ``selected`` is a
    ``draft_loop`` policy mapping or None.
    """
    grid = dict(prior or load_prior_grid())
    # The probe's measurements do not depend on the threshold, but the
    # selection does; both are cached together under one key.  The cycle
    # probe runs real self-MTP cycles under ``self_mtp``, so the policy
    # (minus the draft_loop the probe strips) is part of what was timed.
    widths = sorted({int(w) for w in widths if int(w) >= 1})
    key_parts = {"schema": SCHEMA, "identity": dict(identity), "base": base,
                 "max_end": max_end, "threshold": float(threshold), "widths": widths,
                 "prior": hashlib.sha256(json.dumps(grid, sort_keys=True).encode()).hexdigest(),
                 "self_mtp": self_mtp_digest(self_mtp)}
    path = cache_dir() / f"{cache_key(key_parts)}.json"
    if use_cache and path.is_file():
        try:
            cached = json.loads(path.read_text())
            if cached.get("schema") == SCHEMA and cached.get("key") == key_parts:
                cached["source"] = "cache"
                return cached
        except (OSError, ValueError):
            pass
    started = time.perf_counter()
    row_costs = probe_row_costs(model, vocab_size, max(widths) * (max_end + 1))
    cycles_by_width = {}
    for width in widths:
        ends = candidate_ends(base, row_costs, max_end, width)
        cycles_by_width[width] = probe_cycle_costs(
            model, prompt, [base, *ends], self_mtp=self_mtp, width=width
        )
    # Contention check: time the one-lane base cycle again after everything
    # else and compare it with the first measurement.
    first = cycles_by_width[widths[0]][base]
    again = probe_cycle_costs(model, prompt, [base], self_mtp=self_mtp, width=widths[0])[base]
    drift = abs(again - first) / first
    stable = drift <= MAX_PROBE_DRIFT
    if stable:
        choice = select_topology(base, cycles_by_width, grid, threshold=threshold)
    else:
        choice = {"selected": None,
                  "reason": f"probe unstable: base cycle moved {drift:.0%} during the probe"}
    receipt = {
        "schema": SCHEMA,
        "key": key_parts,
        "source": "probe",
        "probe_seconds": time.perf_counter() - started,
        "stable": stable,
        "base_cycle_drift": drift,
        "row_cost_ms": {str(r): 1e3 * s for r, s in row_costs.items()},
        "tile_edges": tile_edges(row_costs),
        "cycle_ms": {
            str(w): {str(d): 1e3 * sec for d, sec in cycles.items()}
            for w, cycles in cycles_by_width.items()
        },
        **choice,
    }
    if use_cache and stable:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(receipt, indent=1))
        except OSError:
            pass
    return receipt


__all__ = [
    "DEFAULT_THRESHOLD", "PRIOR_RESOURCE", "load_prior_grid", "prior_at", "threshold_range", "MIN_PREDICTED_GAIN", "SCHEMA",
    "candidate_ends", "predict", "probe_cycle_costs", "probe_row_costs",
    "resolve_topology", "select_topology", "self_mtp_digest", "tile_edges",
]
