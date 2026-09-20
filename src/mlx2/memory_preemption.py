"""Preempt-and-replay under memory pressure (default off).

Today a lane that the memory controller cannot grow for 60 s fails with HTTP
429.  With ``memory_preemption`` enabled the worker instead removes the
youngest *preemptible* lane, returns it to the head of the deferred queue with
its warm APCv2 prefix leased, pauses ordinary admission, and later replays it
through the normal admission path: prompt = original prompt + the tokens the
client has already received, ``max_tokens`` reduced by the same count.  The
detokenizer, output parser and stop state never leave the job, so the client
stream simply continues.

A lane is preemptible only when that replay is an exact continuation:

* prefill-phase lanes (no token delivered yet) restart from scratch, which is
  exactly what a fresh admission of the same request with the same seed does;
* decode-phase lanes must be greedy (or ordinary-route sampled, whose
  ``LaneRNG`` draws exactly one key per delivered token and is rebuilt from the
  seed), and every logits processor must declare ``history_pure`` (its state is
  a function of token history, contract P5).

Everything else keeps today's behaviour.  The replayed continuation is the
model's output for the extended prompt; it equals the uninterrupted stream up
to prefill-vs-decode kernel numerics, the same caveat any exact prefix-cache
restore carries.

Design reference: Splash ``Engine::suspendForGrowth`` / ``admitQueued``
(rev f58d36dd, Apache-2.0); see ``provenance/splash-08-suspend-replay.json``.
"""

from __future__ import annotations

import math

MAX_REPLAYS_PER_JOB = 2

_DEFAULTS = {"enabled": False, "stall_seconds": 60.0, "on_pressure": True}


def memory_preemption_policy(value) -> dict:
    """Validate the server-owned ``memory_preemption`` execution policy."""
    if value is None:
        return dict(_DEFAULTS)
    if not isinstance(value, dict):
        raise ValueError("memory_preemption must be an object")
    unknown = set(value) - set(_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown memory_preemption settings: {sorted(unknown)}")
    policy = {**_DEFAULTS, **value}
    for name in ("enabled", "on_pressure"):
        if type(policy[name]) is not bool:
            raise ValueError(f"memory_preemption {name} must be boolean")
    stall = policy["stall_seconds"]
    if (
        isinstance(stall, bool)
        or not isinstance(stall, (int, float))
        or not math.isfinite(stall)
        or not 0 < stall <= 3600
    ):
        raise ValueError(
            "memory_preemption stall_seconds must be positive and at most 3600"
        )
    policy["stall_seconds"] = float(stall)
    return policy


def processors_history_pure(processors) -> bool:
    """Contract P5: every processor's state is rebuildable from token history.

    Absent the attribute a processor is assumed stateful, so a lane carrying it
    is not replayable once decode has started.
    """
    return all(getattr(processor, "history_pure", False) for processor in processors)


DECODE_REPLAY_BLOCK = "decode_state_not_reconstructible"


def decode_replay_block() -> str:
    """Why no decode-phase lane may be preempted, on any route.

    Replay rebuilds a preempted lane by looking the prompt plus its delivered
    tokens up in APCv2 and *prefilling* whatever the lookup does not cover.
    The original run produced those tokens by *decode*, one at a time.  Those
    are different numerical paths and they do not agree bit-for-bit: measured
    on CPU, comparing the next-token logits after eight tokens produced by
    decode against the same eight replayed as one prefill, max |delta| is
    1.6e-06 on the dense tiny model and 7.7e-07 on the hybrid GDN/MTP one.
    Fused Metal prefill kernels are further from decode than that, not closer.

    Greedy decoding hides the difference until the perturbation exceeds the
    top-2 margin at the resume position, and then the argmax flips and every
    later token diverges.  That is what the GPU requeue saw: the ordinary arm
    diverged from the first replayed token while the self-MTP arm matched --
    same mechanism, different margin.  The self-MTP pass was luck, so blocking
    only the ordinary route would leave a route that is exact by coincidence.

    Prefill-phase preemption is unaffected and stays exact by construction:
    nothing has been decoded, so there is no decode-produced state to rebuild.

    This subsumes the earlier per-cause checks (sampled speculative route,
    steering, int8 prefill, and a processor that is not ``history_pure``),
    which were narrower statements of the same problem.  Re-enabling decode
    replay needs a prefill path measured bit-equal to decode -- not an
    argument that the difference is small.
    """
    return DECODE_REPLAY_BLOCK


def preemption_block(job) -> str | None:
    """Why ``job`` must not be preempted now, or None when it may be."""
    request = job.request
    if job.preemption_prompt is None:
        return "untracked"
    if job.preemptions >= MAX_REPLAYS_PER_JOB:
        return "replay_cap"
    if request.get("batch_cohort") is not None:
        return "batch_cohort"
    if job.fanout_group or job.parallel_sample:
        return "parallel_samples"
    if job.cache_capsule is not None:
        return "cache_capsule"
    if request.get("_mlx2_prefill_inputs") is not None or request.get(
        "_mlx2_media_fingerprint"
    ):
        return "multimodal"
    if job.approximate_kv_applied or (job.spomin_receipt or {}).get(
        "status"
    ) == "applied":
        return "approximate"
    if job.completion_tokens:
        # Decode has started, so replay would have to rebuild decode-produced
        # state by prefilling.  Refuse rather than replay inexactly.
        return job.decode_replay_block or DECODE_REPLAY_BLOCK
    return None


def choose_preemption_victim(active, stalled=()):
    """The youngest preemptible lane uid, or None.

    Age is the request's first admission time, so a replayed request keeps its
    place instead of always looking youngest.  For a stall, only lanes no
    older than the oldest stalled lane qualify: older work is never parked to
    make room for younger work.
    """
    floor = min((active[uid].started for uid in stalled), default=None)
    victim = None
    for uid, job in active.items():
        if preemption_block(job) is not None:
            continue
        if floor is not None and job.started < floor:
            continue
        if victim is None or job.started > active[victim].started:
            victim = uid
    return victim


def replay_lane_rng(lane_rng_type, seed, draws):
    """An ordinary-route ``LaneRNG`` positioned after ``draws`` delivered tokens.

    The ordinary sampler takes exactly one key per sampled token, and the
    pipelined next token that was discarded at removal is the draw the replay's
    first sample repeats.
    """
    rng = lane_rng_type(seed)
    for _ in range(int(draws)):
        rng.next_key()
    return rng
