"""Inkling 2 DSpark cost-scheduled draft depth (``inkling2_dspark_cost_depth``).

Original mlx2 code; the mechanism comes from g13-engine's DSpark scheduler
(``_select_verify_k`` and its solo cooldown, g13-engine c91c0180, MIT; see
docs/PROVENANCE.md).  Nothing new runs in the scheduler: the option builds
an ``adaptive_verification`` cost model for the existing cohort-mode
``AdaptiveVerificationPolicy``, which already picks the depth ``K`` that
maximizes ``E[commit | K] / (draft_cost + verify_cost[K])`` from its online
per-position acceptance estimator.  This module supplies:

- the verify cost table, in ms per verify of ``K + 1`` rows, either from the
  policy (``verify_ms``) or timed once at load on the bound target and drafter
  (``measure_costs``; the clock is injectable so tests are deterministic);
- the policy settings g13 adds: a reject cooldown of solo (anchor-only, no
  draft) rounds, and, opt-in, depth 0 competing in the cost model;
- ``greedy_only``: a cohort with a sampled lane keeps the fixed depth, so a
  sampled request draws exactly the RNG values the fixed-depth route draws.

Under exact verification the depth never changes greedy output: each round
commits the target's own argmax prefix, whatever the proposal count.  The
cost table binds one cohort width (1); wider cohorts keep the fixed depth.

Why this is not the 27B no-go again (runtime/mtp_confidence.py): there, a
confidence scheduler lost 3 to 14% against fixed depth because the 27B verify
cost barely grows with rows and the adapter cap was already the best constant
depth.  This option is only worth having if the measured Inkling table is
steep (g13 measured 42 ms at 1 row and 133 ms at 8 on an M1 Max).  The A/B
therefore compares against fixed depth 7 and against the best constant depth,
on the same serving shape.
"""

from __future__ import annotations

import statistics
import time

OPTION = "inkling2_dspark_cost_depth"
SETTINGS = {"verify_ms", "draft_ms", "reject_cooldown", "allow_solo", "min_gain",
            "min_observations", "full_depth_interval", "refit_interval",
            "measure_context", "measure_repeats"}
# refit_interval: the generic estimator refits every 100 rounds; a request
# of a few hundred tokens would then schedule from the cold-start estimate.
# allow_solo is off by default: the depth is chosen before drafting from
# lagged estimates (g13 chooses after drafting, from the current draft's
# confidences), and a solo round observes nothing, so a cost-chosen solo
# streak only ends at the next full_depth_interval probe. Depth 1 keeps
# observing; the cooldown supplies bounded solo stretches.
DEFAULTS = {"reject_cooldown": 8, "allow_solo": False, "min_gain": 0.05,
            "min_observations": 32, "full_depth_interval": 32, "refit_interval": 16,
            "measure_context": 256, "measure_repeats": 3}


def _positive(value, name, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
        raise TypeError(f"Inkling 2 {OPTION} {name} must be a number")
    if value < 0 or (value == 0 and not zero) or value == float("inf"):
        raise ValueError(f"Inkling 2 {OPTION} {name} must be positive and finite")
    return float(value)


def parse(value):
    """Validate the policy value; returns None (off) or a settings dict.

    ``true`` measures the table at load; a dict may give ``verify_ms`` (rows
    1, 2, ...) and ``draft_ms`` to use instead, plus the scheduler settings.
    """
    if value is False:
        return None
    if value is True:
        value = {}
    if not isinstance(value, dict):
        raise TypeError(f"Inkling 2 {OPTION} must be a boolean or an object")
    unknown = set(value) - SETTINGS
    if unknown:
        raise ValueError(f"Inkling 2 {OPTION} has unknown settings {sorted(unknown)}")
    if ("verify_ms" in value) != ("draft_ms" in value):
        raise ValueError(f"Inkling 2 {OPTION} verify_ms and draft_ms go together")
    settings = {**DEFAULTS, **value}
    if "verify_ms" in value:
        table = value["verify_ms"]
        if not isinstance(table, (list, tuple)) or len(table) < 2:
            raise ValueError(f"Inkling 2 {OPTION} verify_ms must list rows 1, 2, ...")
        settings["verify_ms"] = [_positive(v, "verify_ms") for v in table]
        settings["draft_ms"] = _positive(value["draft_ms"], "draft_ms", zero=True)
    for name in ("reject_cooldown", "min_observations", "full_depth_interval",
                 "refit_interval", "measure_context", "measure_repeats"):
        number = settings[name]
        if type(number) is not int or number < (0 if name == "reject_cooldown" else 1):
            raise TypeError(f"Inkling 2 {OPTION} {name} must be an integer")
    if settings["reject_cooldown"] > 64:
        raise ValueError(f"Inkling 2 {OPTION} reject_cooldown must be at most 64")
    if type(settings["allow_solo"]) is not bool:
        raise TypeError(f"Inkling 2 {OPTION} allow_solo must be a boolean")
    gain = settings["min_gain"]
    if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not 0 <= gain < 1:
        raise ValueError(f"Inkling 2 {OPTION} min_gain must be in [0, 1)")
    return settings


def adaptive_verification(settings, num_draft, *, verify_ms, draft_ms, source):
    """The cohort-mode ``adaptive_verification`` value for these costs."""
    if len(verify_ms) < num_draft + 1:
        raise ValueError(
            f"Inkling 2 {OPTION} verify_ms needs rows 1..{num_draft + 1} for num_draft {num_draft}"
        )
    return {
        "verification_costs": [float(v) for v in verify_ms[: num_draft + 1]],
        "draft_cost": float(draft_ms),
        "cohort_size": 1,
        "min_observations": settings["min_observations"],
        "full_depth_interval": settings["full_depth_interval"],
        "refit_interval": settings["refit_interval"],
        "min_gain": float(settings["min_gain"]),
        "reject_cooldown": settings["reject_cooldown"],
        "allow_solo": settings["allow_solo"],
        "greedy_only": True,
        "cost_source": source,
    }


def _filler(count, vocab, offset=0):
    # Arbitrary in-vocabulary tokens; costs do not depend on their values.
    return [(offset + 7919 * i) % (vocab - 1) + 1 for i in range(count)]


def measure_costs(target, drafter, rows, *, context=256, repeats=3, clock=time.perf_counter):
    """Median ms of one B1 verify forward of ``1..rows`` rows after a
    ``context``-token prefill, and of one full DSpark draft block.

    The target call is the verify call (``forward_with_taps`` with the
    drafter's taps) with the kernels the adapter configured.  The first call
    of each width is a warm-up.  The target's forward counter is restored, so
    a measurement does not read as served traffic.
    """
    import mlx.core as mx

    vocab = int(drafter.config.vocab_size)
    taps = list(drafter.config.target_layer_ids)
    forwards = getattr(target, "completed_forwards", None)
    stats = [dict(layer.self_attn.kv_headroom_stats) for layer in drafter.layers]
    verify = []
    try:
        for width in range(1, rows + 1):
            cache = target.make_cache()
            logits, _ = target.forward_with_taps(mx.array([_filler(context, vocab)]), cache, taps)
            mx.eval(logits)
            samples = []
            for repeat in range(repeats + 1):
                tokens = mx.array([_filler(width, vocab, offset=repeat + 1)])
                start = clock()
                logits, features = target.forward_with_taps(tokens, cache, taps)
                mx.eval(logits, features)
                elapsed = (clock() - start) * 1000.0
                if repeat:
                    samples.append(elapsed)
            verify.append(statistics.median(samples))
        cache = drafter.make_cache()
        features = mx.zeros((1, context, drafter.config.feature_width), mx.bfloat16)
        drafter.commit_context(features, cache)
        samples = []
        for repeat in range(repeats + 1):
            start = clock()
            drafter.draft_cached(1 + repeat, features[:, -1:], cache, drafter.config.block_size,
                                 lambda logits: mx.argmax(logits, -1)).tolist()
            elapsed = (clock() - start) * 1000.0
            if repeat:
                samples.append(elapsed)
        draft = statistics.median(samples)
    finally:
        if forwards is not None:
            target.completed_forwards = forwards
        for layer, saved in zip(drafter.layers, stats):
            layer.self_attn.kv_headroom_stats.update(saved)
    return verify, draft


__all__ = ["DEFAULTS", "OPTION", "adaptive_verification", "measure_costs", "parse"]
