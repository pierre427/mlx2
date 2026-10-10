"""Shared decode publication and bounded prefill policy for all executors.

The executor yields only committed responses, outside verification transactions.
A suspended phase owns no request admission: executors must recheck membership
before prefilling a saved candidate. See provenance/decode-first-publish.json.
"""

from .adaptive_policy import DecodeFirstPublish
from .prefill_plan import shared_prefill_budget_width


def drive_round(round_):
    """Run a scheduler round generator to completion; return its result."""
    try:
        while True:
            next(round_)
    except StopIteration as stop:
        return stop.value


def initialize(owner, policy):
    owner.decode_first = DecodeFirstPublish.from_value(policy)
    owner._decode_first_pending = None
    owner._decode_first_round_mode = "off"


def sync_stats(owner):
    for key, value in owner.decode_first.counters.items():
        owner.scheduler_stats[f"decode_first_{key}"] = int(value)


def close_pending(owner):
    pending = getattr(owner, "_decode_first_pending", None)
    if pending is not None:
        owner._decode_first_pending = None
        pending[0].close()


def prefill_width(owner, base, rows=1):
    """Apply one padded-token budget to a physical prefill slice."""
    enabled = getattr(owner, "_decode_first_round_mode", "off") == "all"
    policy = owner.decode_first
    width = shared_prefill_budget_width(
        base,
        rows,
        enabled=enabled,
        token_budget=policy.prefill_token_budget if enabled else None,
    )
    if width < base:
        policy.bump("budget_split_rounds")
        sync_stats(owner)
    return width


def publish_round(owner, mode: str, round_factory):
    """Return a round's decode responses before its prefill phase runs.

    A call first finishes the previous round's pending prefill phase,
    then runs the next round's decode phase and returns. Executors define
    their enabled round as decode k, prefill k, decode k+1, so the serving
    loop can deliver decode k's tokens before prefill k runs. Legacy off
    ordering stays executor-owned. A round with no
    decode output, a fused mixed round, or a round that ends before its
    phase boundary (a deferred prefill or no admission) runs whole.
    """
    policy = owner.decode_first
    prompts, generations = [], []
    # A resumed prompt boundary has not reached serving yet. Executors retiring
    # a lane in the next decode phase must retain that boundary for this return.
    owner._decode_first_resumed_prompt_uids = set()
    resumed = owner._decode_first_pending is not None
    if resumed:
        (pending, published) = owner._decode_first_pending
        owner._decode_first_pending = None
        (rest_prompts, rest_generations) = drive_round(pending)
        prompts.extend(rest_prompts)
        owner._decode_first_resumed_prompt_uids.update(
            prompt.uid for prompt in rest_prompts if prompt.end_of_prompt
        )
        generations.extend(rest_generations[published:])
        policy.bump("prefill_phases_resumed")
    if mode == "off":
        # Kill switch flipped with a phase pending: finish whole rounds.
        (more_prompts, more_generations) = drive_round(round_factory())
        prompts.extend(more_prompts)
        generations.extend(more_generations)
        sync_stats(owner)
        return (prompts, generations)
    owner._round_fused = False
    round_ = round_factory()
    try:
        decoded = next(round_)
    except StopIteration as stop:
        (more_prompts, more_generations) = stop.value
        prompts.extend(more_prompts)
        generations.extend(more_generations)
        policy.bump("fused_rounds" if owner._round_fused else "unsplit_rounds")
        sync_stats(owner)
        return (prompts, generations)
    if decoded:
        generations.extend(decoded)
        # ``decoded`` is the round's own response list: the suspended
        # frame would keep every published response, and a finished
        # lane's cache with it, alive until the next call -- which never
        # comes once the last request is done (Codex port review
        # 2026-10-02 item 5).  Drop the references in place; the length
        # and truthiness the prefill phase reads are kept, and the
        # resumed phase's output is taken from index ``published`` on.
        decoded[:] = [None] * len(decoded)
        owner._decode_first_pending = (round_, len(decoded))
        policy.bump("published_rounds")
        policy.bump("published_tokens", len(decoded))
    elif prompts or generations:
        # Nothing decoded, but the resumed phase produced output: keep
        # one prefill phase per call, as ``_next`` does.
        owner._decode_first_pending = (round_, 0)
        policy.bump("no_decode_rounds")
    else:
        (more_prompts, more_generations) = drive_round(round_)
        prompts.extend(more_prompts)
        generations.extend(more_generations)
        policy.bump("no_decode_rounds")
    sync_stats(owner)
    return (prompts, generations)
