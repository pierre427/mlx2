# SPDX-License-Identifier: MIT
# Adapted from mlx-lm-unified; see docs/PROVENANCE.md and provenance/flashnext.json.
import copy
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import mlx.core as mx
import mlx.nn as nn
from .committed_recovery import CommittedRecoverySlot
from .cow_cache import (
    mtp_boundary_cow_enabled,
    restore_recovery_descriptors,
    snapshot_prompt_cache_descriptors,
    snapshot_recovery_descriptors,
)
import numpy as np
from .generate import generation_stream
from .models.cache import (
    can_trim_prompt_cache,
    make_prompt_cache,
    trim_ragged_prompt_cache,
)
from .prompt_lookup import HybridStats as _PromptLookupStatsBase
from . import round_levers
from .sample_utils import LaneRNG, draw_key, make_transformed_logprobs
from .verify_sync import record_verify_sync, verify_sync_round


def _start_speculation_or_cleanup(caches, required_caches, error_message):
    """Enable rollback atomically and fail without leaking cache state."""
    try:
        for c in caches:
            c.start_speculation()
        if not can_trim_prompt_cache(required_caches):
            raise ValueError(error_message)
    except Exception:
        for c in caches:
            try:
                c.stop_speculation()
            except Exception:
                pass
        raise


def _stop_all_speculation(caches):
    """Run ``stop_speculation`` on every cache even when one raises.

    A failing cleanup hook must not leave later caches speculating (holding
    rollback stashes and reporting trimmable state). Every hook runs; the
    first error is re-raised only after all caches had their turn.
    """
    first_error = None
    for c in caches:
        try:
            c.stop_speculation()
        except BaseException as e:
            if first_error is None:
                first_error = e
    if first_error is not None:
        raise first_error


@dataclass
class HybridStats(_PromptLookupStatsBase):
    """Per-source accounting for one hybrid generation run.

    Extends prompt_lookup.HybridStats (the F7' unification: shared retrieval/
    plain/span fields are defined ONCE, there) with the draft-chain and
    external-cache accounting the MTP paths need. isinstance-compatible with
    the base, so a hybrid stats object can be passed anywhere the
    prompt-lookup one is expected; the reverse (base into an MTP path) is
    rejected early with a clear error instead of an attribute crash mid-run.
    """

    draft_cycles: int = 0
    draft_proposed: int = 0
    draft_accepted: int = 0
    external_cache_reconciled: bool = False
    external_cache_trimmed_tokens: int = 0
    rate_gate_probed: bool = False
    rate_gate_delatched: bool = False
    rate_gate_spec_ms_per_tok: float = 0.0
    rate_gate_plain_ms_per_tok: float = 0.0
    router_plain_cycles: int = 0
    router_reengagements: int = 0
    router_last_num_draft: int = 0
    router_accept_prob: float = 0.0

    @property
    def total_emitted(self) -> int:
        return (
            self.retrieval_accepted
            + self.draft_accepted
            + self.bonus_tokens
            + self.plain_tokens
        )

    @property
    def mean_retrieval_span_proposed(self) -> float:
        return self.retrieval_proposed / max(self.retrieval_cycles, 1)

    @property
    def mean_retrieval_span_accepted(self) -> float:
        return self.retrieval_accepted / max(self.retrieval_cycles, 1)

    @property
    def mean_draft_span_proposed(self) -> float:
        return self.draft_proposed / max(self.draft_cycles, 1)

    @property
    def mean_draft_span_accepted(self) -> float:
        return self.draft_accepted / max(self.draft_cycles, 1)

    def summary(self) -> str:
        tot = max(self.total_emitted, 1)
        lines = [
            f"cycles: {self.cycles} (retrieval {self.retrieval_cycles}, draft {self.draft_cycles}, plain {self.plain_cycles})",
            f"tokens: {self.total_emitted} = retrieval {self.retrieval_accepted} ({self.retrieval_accepted / tot:.1%}) + draft {self.draft_accepted} ({self.draft_accepted / tot:.1%}) + bonus {self.bonus_tokens} ({self.bonus_tokens / tot:.1%}) + plain {self.plain_tokens} ({self.plain_tokens / tot:.1%})",
        ]
        if self.retrieval_cycles:
            lines.append(
                f"retrieval span: proposed {self.mean_retrieval_span_proposed:.2f} / accepted {self.mean_retrieval_span_accepted:.2f} (acceptance {self.retrieval_accepted / max(self.retrieval_proposed, 1):.1%})"
            )
        if self.draft_cycles:
            lines.append(
                f"draft span:     proposed {self.mean_draft_span_proposed:.2f} / accepted {self.mean_draft_span_accepted:.2f} (acceptance {self.draft_accepted / max(self.draft_proposed, 1):.1%})"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class MTPToken:
    """One token authorized by a self-MTP target verification row."""

    token: int
    logprobs: mx.array
    from_draft: bool


class ZeroDepthFastUnavailable(RuntimeError):
    """The cache topology lacks the direct batched K=0 target ABI."""


@dataclass
class SelfMTPCachePair:
    target: List[Any]
    draft: List[Any]


@dataclass
class SelfMTPLane:
    uid: int
    cur: int
    seed_h: mx.array
    pending_hs: Optional[mx.array]
    pending_ts: List[int]
    token_prefix: mx.array
    rng: Optional[LaneRNG]
    ntoks: int
    max_tokens: int
    num_draft: int
    sampling_temp: float
    accept_rule: str
    logprob_transform: Optional[Callable]
    logits_processors: List[Callable]
    stats: HybridStats
    share_qsa_indices: bool = False
    fly_verification: Optional[Any] = None
    relaxed_accepts: int = 0
    # Default-off copy-draft state (runtime/copy_draft.py).  None keeps the
    # historical MTP-head-only proposal.
    copy_draft: Optional[Any] = None
    # Optional mtp_confidence.DraftConfidenceProbe: per-position draft
    # features (and greedy lookahead) for confidence-scheduled depth.
    confidence_probe: Optional[Any] = None


@dataclass
class DetachedSelfMTPLane:
    lane: SelfMTPLane
    caches: SelfMTPCachePair
    segment_transaction: Optional[Any] = field(default=None, repr=False, compare=False)
    shared_qsa_prefix_id: Optional[str] = field(default=None, repr=False, compare=False)


@dataclass
class BatchedSelfMTPState:
    lanes: List[SelfMTPLane]
    caches: SelfMTPCachePair
    membership_epoch: int
    proposal_open: bool = False
    poisoned: bool = False
    poison_reason: Optional[str] = None
    _open_proposal: Optional["SelfMTPCycleResult"] = field(
        default=None, repr=False, compare=False
    )


@dataclass
class SegmentedSelfMTPState:
    """Independent concrete B1 caches under one scheduler cohort.

    Persistent storage remains B1.  An optional transient batched compute view
    streams the model once while each row carries a generation-safe aligned-
    plane transaction used only at commit.
    """

    lanes: List[SelfMTPLane]
    row_caches: List[SelfMTPCachePair]
    transactions: List[Any]
    membership_epoch: int
    shared_qsa_prefix_id: Optional[str] = field(default=None, repr=False, compare=False)
    proposal_open: bool = False
    poisoned: bool = False
    poison_reason: Optional[str] = None
    _open_proposal: Optional["SelfMTPCycleResult"] = field(
        default=None, repr=False, compare=False
    )
    _row_states: List[BatchedSelfMTPState] = field(
        default_factory=list, repr=False, compare=False
    )
    _row_proposals: List["SelfMTPCycleResult"] = field(
        default_factory=list, repr=False, compare=False
    )
    _batched_state: Optional[BatchedSelfMTPState] = field(
        default=None, repr=False, compare=False
    )
    _segmented_caches: Optional[SelfMTPCachePair] = field(
        default=None, repr=False, compare=False
    )
    _transaction_branches: List[Any] = field(
        default_factory=list, repr=False, compare=False
    )
    _recovery_checkpoints: List[Any] = field(
        default_factory=list, repr=False, compare=False
    )


@dataclass(frozen=True)
class SelfMTPCycleResult:
    membership_epoch: int
    lane_uids: Tuple[int, ...]
    draft_depths: Tuple[int, ...]
    accepted_lengths: Tuple[int, ...]
    target_drops: Tuple[int, ...]
    head_drops: Tuple[int, ...]
    outputs: Tuple[Tuple[MTPToken, ...], ...]
    _old_curs: Tuple[int, ...] = field(default=(), repr=False, compare=False)
    _old_seed_hs: Tuple[mx.array, ...] = field(default=(), repr=False, compare=False)
    _drafts: Tuple[Tuple[int, ...], ...] = field(default=(), repr=False, compare=False)
    _vhidden: Tuple[mx.array, ...] = field(default=(), repr=False, compare=False)
    _logprobs: Tuple[mx.array, ...] = field(default=(), repr=False, compare=False)
    _bonuses: Tuple[int, ...] = field(default=(), repr=False, compare=False)
    zero_fast_path: bool = False
    true_batched: bool = False
    relaxed_accepts: Tuple[int, ...] = ()
    # Per lane: copied-span length verified this round (0 = MTP-head lane)
    # and the copy gate's decision ("off" when the lane has no copy state).
    copy_spans: Tuple[int, ...] = ()
    copy_decisions: Tuple[str, ...] = ()
    # Host copies of per-draft-position confidence features, one tuple per
    # lane (verified positions first, then unverified lookahead positions),
    # and the matching drafted token ids.  Empty unless a probe was set.
    draft_features: Tuple[Tuple[Tuple[float, ...], ...], ...] = ()
    draft_feature_tokens: Tuple[Tuple[int, ...], ...] = ()


def _mtp_backbone(model, tokens, cache):
    """Return (LM-head hidden, MTP seed hidden) for one trunk forward.

    Conventional MTP models use the same post-norm hidden for both. Qwen4
    scheme A keeps its pre-final-mixer HC multi-stream tensor for drafting.
    """
    if hasattr(model, "mtp_backbone"):
        return model.mtp_backbone(tokens, cache=cache)
    hidden = model.model(tokens, cache=cache)
    return (hidden, hidden)


def _restore_mtp_state(cache, mtp_state):
    """Validate and unpack an MTP sidecar paired with ``cache``.

    A persistent MTP cache contains one fewer teacher-forced pair than the
    target cache contains tokens.  The sidecar also needs the trunk hidden of
    the final cached token so the first uncached token can form the boundary
    pair.  Validate this relationship before mutating either cache.
    """
    (mtp_cache, restored_seed_h) = mtp_state
    prefix_len = max((getattr(c, "offset", 0) for c in cache), default=0)
    mtp_offset = max((getattr(c, "offset", 0) for c in mtp_cache), default=0)
    if prefix_len > 0:
        if restored_seed_h is None:
            raise ValueError(
                "mtp_state with a non-empty prompt_cache prefix requires prev_tail_hidden (the trunk hidden of the last cached token); without it the boundary pair is skipped and the MTP cache drafts one position behind."
            )
        if mtp_offset != prefix_len - 1:
            raise ValueError(
                f"mtp_state offset mismatch: MTP cache covers {mtp_offset} pairs but the prompt_cache prefix has {prefix_len} tokens (expected {prefix_len - 1} pairs). The prefix snapshot and its draft sidecar were not captured together."
            )
    elif restored_seed_h is not None or mtp_offset != 0:
        raise ValueError(
            f"mtp_state carries a restored draft context ({mtp_offset} pairs, prev_tail_hidden {('set' if restored_seed_h is not None else 'unset')}) but the prompt_cache prefix is empty; the sidecar must cover exactly the cached tokens."
        )
    return (mtp_cache, restored_seed_h)


def _temperature_logprobs(logits, sampling_temp: float = 0.0):
    logits = logits.astype(mx.float32)
    if sampling_temp and sampling_temp > 0:
        logits = logits / float(sampling_temp)
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def _apply_logits_processors(logits_processors, y, logits):
    """Apply processors with the same rank convention as ``generate_step``."""
    if not logits_processors:
        return logits
    batched = logits[None] if logits.ndim == 1 else logits
    for processor in logits_processors:
        batched = processor(y, batched)
    return batched[0] if logits.ndim == 1 else batched


def _probe_logits_processors(logits_processors, y, logits):
    """Apply processors to provisional draft state through isolated copies."""
    from .processor_probe import probe_logits_processors

    if not logits_processors:
        return logits
    batched = logits[None] if logits.ndim == 1 else logits
    batched = probe_logits_processors(logits_processors, y, batched)
    return batched[0] if logits.ndim == 1 else batched


def _device_draft_token(logprobs, sampling_temp: float, *, rng=None) -> mx.array:
    """The draw ``_sample_from_logprobs`` makes, left on device (uint32 scalar).

    Same op, same key: ``draw_key`` advances the lane key (or the global key
    sequence) when the op is built, so the token is identical; only the
    per-draft host sync is gone.
    """
    if sampling_temp and sampling_temp > 0:
        return mx.random.categorical(logprobs, key=draw_key(rng)).astype(mx.uint32)
    return mx.argmax(logprobs).astype(mx.uint32)


def _sample_from_logprobs(logprobs, sampling_temp: float = 0.0, *, rng=None) -> int:
    if sampling_temp and sampling_temp > 0:
        record_verify_sync("hybrid.sample.categorical_item")
        return int(mx.random.categorical(logprobs, key=draw_key(rng)).item())
    record_verify_sync("hybrid.sample.argmax_item")
    return int(mx.argmax(logprobs).item())


def _residual_sample(
    target_logprobs,
    draft_logprobs,
    sampling_temp: float,
    scale: float = 1.0,
    *,
    rng=None,
) -> int:
    residual = mx.maximum(scale * mx.exp(target_logprobs) - mx.exp(draft_logprobs), 0.0)
    total = mx.sum(residual)
    record_verify_sync("hybrid.residual.total_eval")
    mx.eval(total)
    record_verify_sync("hybrid.residual.total_item")
    if float(total.item()) <= 0.0:
        return _sample_from_logprobs(target_logprobs, sampling_temp, rng=rng)
    residual_logprobs = mx.log(residual / total)
    record_verify_sync("hybrid.residual.categorical_item")
    return int(mx.random.categorical(residual_logprobs, key=draw_key(rng)).item())


def _make_sampling_transform(
    sampling_temp: float, top_p: float = 1.0, top_k: int = 0, min_p: float = 0.0
) -> Optional[Callable[[mx.array], mx.array]]:
    """Return a shared draft/target logprob transform, or ``None``.

    ``None`` means no filter is active (or ``temp == 0``, where the filters
    cannot change the argmax) and the caller keeps the incumbent
    temperature-only path bit-exactly. Otherwise the returned callable maps
    raw logits to the log-probabilities ``make_sampler`` samples from, with
    filtered tokens exactly ``-inf`` — applied identically to draft and
    target so the residual acceptance ratio is well-defined.
    """
    filtered = 0.0 < top_p < 1.0 or top_k > 0 or min_p > 0.0
    if not filtered or not sampling_temp or sampling_temp <= 0:
        return None
    return make_transformed_logprobs(
        sampling_temp, top_p=top_p, top_k=top_k, min_p=min_p
    )


def _batched_residual_verify(
    logprobs, draft_logprobs, drafts, sampling_temp: float, *, rng=None
):
    """Residual acceptance over all k positions with one GPU sync.

    Same rule as ``_accept_sampled_draft`` scanned per position, but every
    uniform and log-ratio is computed in one graph and drained with a single
    ``mx.eval`` (vs one per accepted position). A draft the target transform
    filtered has ratio exactly 0 and is always rejected — ``u <= 0`` cannot
    rescue it. Returns ``(n_accept, bonus)``.
    """
    k = len(drafts)
    d = mx.array(drafts)[:, None]
    target_at = mx.take_along_axis(logprobs[:k], d, axis=-1)[:, 0]
    draft_at = mx.take_along_axis(mx.stack(draft_logprobs), d, axis=-1)[:, 0]
    ratios = mx.exp(mx.minimum(target_at - draft_at, 0.0))
    us = _draw_mtp_acceptance_uniforms(k, rng=rng)
    record_verify_sync("hybrid.residual_verify.eval")
    mx.eval(ratios, us)
    record_verify_sync("hybrid.residual_verify.ratios_tolist")
    record_verify_sync("hybrid.residual_verify.uniforms_tolist")
    (ratios, us) = (ratios.tolist(), us.tolist())
    n_accept = 0
    while (
        n_accept < k and ratios[n_accept] > 0.0 and (us[n_accept] <= ratios[n_accept])
    ):
        n_accept += 1
    if n_accept < k:
        bonus = _residual_sample(
            logprobs[n_accept], draft_logprobs[n_accept], sampling_temp, rng=rng
        )
    else:
        bonus = _sample_from_logprobs(logprobs[n_accept], sampling_temp, rng=rng)
    return (n_accept, bonus)


def _draw_mtp_acceptance_uniforms(k: int, *, rng=None) -> mx.array:
    """Draw one lane's native acceptance vector, never a padded batch shape."""
    if k < 0:
        raise ValueError("acceptance width must be non-negative")
    return mx.random.uniform(shape=(k,), key=draw_key(rng))


def _accept_sampled_draft(
    target_logprobs, draft_logprobs, token: int, *, rng=None
) -> bool:
    log_ratio = mx.minimum(target_logprobs[token] - draft_logprobs[token], 0.0)
    ratio = mx.exp(log_ratio)
    u = mx.random.uniform(shape=(), key=draw_key(rng))
    record_verify_sync("hybrid.accept_sampled.eval")
    mx.eval(ratio, u)
    record_verify_sync("hybrid.accept_sampled.uniform_item")
    record_verify_sync("hybrid.accept_sampled.ratio_item")
    return float(u.item()) <= float(ratio.item())


def _block_verify(logprobs, draft_logprobs, drafts, sampling_temp: float, *, rng=None):
    """Block verification (Sun et al., arXiv 2403.10444): accept a draft
    PREFIX by cumulative joint likelihood ratio instead of independent
    per-token coin flips.

    ``p_cum_i = min(p_cum_{i-1} * p_i(x_i)/q_i(x_i), 1)`` tracks the joint
    target/draft ratio of the drafted prefix ``x_1..x_i``. Each prefix length
    ``i`` is checked against a threshold ``h_i``: the full block uses
    ``h_k = p_cum_k``; shorter prefixes use the residual-mass form
    ``h_i = S_i / (S_i + (1 - p_cum_i))`` with
    ``S_i = sum(relu(p_cum_i * p_{i+1} - q_{i+1}))`` over the vocabulary.
    The accepted length ``tau`` is the LARGEST ``i`` whose check
    ``eta_i <= h_i`` passes — a later position can rescue an earlier
    failure, which is why block verification provably accepts at least as
    many tokens in expectation as per-token rejection sampling while
    preserving the target distribution exactly. On ``tau < k`` the
    correction token comes from the scaled residual
    ``relu(p_cum_tau * p - q)``; on ``tau == k`` it is a plain target
    sample (the bonus).

    ``logprobs`` has ``k+1`` target rows (row ``i`` conditions on
    ``x_1..x_i``), ``draft_logprobs`` the ``k`` draft rows that produced
    ``drafts``. Returns ``(n_accept, bonus)``.
    """
    k = len(drafts)
    etas = _draw_mtp_acceptance_uniforms(k, rng=rng)
    record_verify_sync("hybrid.block_verify.uniforms_eval")
    mx.eval(etas)
    p_cums = [1.0]
    p_cum = 1.0
    tau = 0
    for i in range(k):
        d = drafts[i]
        record_verify_sync("hybrid.block_verify.log_ratio_item")
        log_ratio = float((logprobs[i][d] - draft_logprobs[i][d]).item())
        p_cum = min(p_cum * math.exp(log_ratio), 1.0)
        p_cums.append(p_cum)
        if i == k - 1:
            h = p_cum
        else:
            record_verify_sync("hybrid.block_verify.residual_item")
            s = float(
                mx.sum(
                    mx.maximum(
                        p_cum * mx.exp(logprobs[i + 1]) - mx.exp(draft_logprobs[i + 1]),
                        0.0,
                    )
                ).item()
            )
            denom = s + (1.0 - p_cum)
            h = 1.0 if denom <= 0.0 else s / denom
        record_verify_sync("hybrid.block_verify.uniform_item")
        if float(etas[i].item()) <= h:
            tau = i + 1
    if tau == k:
        bonus = _sample_from_logprobs(logprobs[k], sampling_temp, rng=rng)
    else:
        bonus = _residual_sample(
            logprobs[tau],
            draft_logprobs[tau],
            sampling_temp,
            scale=p_cums[tau],
            rng=rng,
        )
    return (tau, bonus)


def _self_mtp_cache_offset(cache) -> int:
    offset = getattr(cache, "offset", 0)
    if isinstance(offset, mx.array):
        if int(offset.size) != 1:
            raise ValueError("detached self-MTP caches must contain exactly one row")
        return int(offset.item())
    return int(offset)


def _self_mtp_group_offset(caches: Sequence[Any]) -> int:
    return max((_self_mtp_cache_offset(c) for c in caches), default=0)


def _reject_unsupported_self_mtp_caches(caches: Sequence[Any]) -> None:
    unsupported = [
        type(cache).__name__
        for cache in caches
        if "SinkWindow" in type(cache).__name__ or "Rotating" in type(cache).__name__
    ]
    if unsupported:
        raise ValueError(
            "batched self-MTP excludes windowed caches; got " + ", ".join(unsupported)
        )


def _validate_detached_self_mtp(detached: DetachedSelfMTPLane) -> None:
    lane = detached.lane
    if lane.pending_hs is not None or lane.pending_ts:
        raise ValueError("a detached self-MTP lane must have no pending pairs")
    if lane.seed_h is None or lane.seed_h.ndim != 3 or lane.seed_h.shape[:2] != (1, 1):
        raise ValueError("a detached lane seed_h must have shape [1, 1, H]")
    if not detached.caches.target or not detached.caches.draft:
        raise ValueError("a detached self-MTP lane requires target and draft caches")
    _reject_unsupported_self_mtp_caches(detached.caches.target)
    _reject_unsupported_self_mtp_caches(detached.caches.draft)
    covered = _self_mtp_group_offset(detached.caches.target)
    draft_offset = _self_mtp_group_offset(detached.caches.draft)
    if covered <= 0 or draft_offset != covered - 1:
        raise ValueError(
            f"detached self-MTP cache mismatch: target covers {covered} tokens but draft covers {draft_offset} pairs"
        )


def _merge_self_mtp_cache_groups(groups: Sequence[Sequence[Any]]) -> List[Any]:
    if not groups:
        return []
    width = len(groups[0])
    if width == 0 or any((len(group) != width for group in groups)):
        raise ValueError("self-MTP cache groups must have the same non-zero width")
    merged = []
    for rows in zip(*groups):
        merge = getattr(type(rows[0]), "merge", None)
        if merge is None:
            raise ValueError(f"{type(rows[0]).__name__} cannot merge cache rows")
        merged.append(merge(list(rows)))
    return merged


def _extract_self_mtp_cache_pair(
    caches: SelfMTPCachePair, indices: Sequence[int], *, batched: bool = False
) -> SelfMTPCachePair:
    """Copy-build a cache pair for ``indices`` without mutating the live pair."""
    indices = [int(index) for index in indices]
    if not indices:
        return SelfMTPCachePair(target=[], draft=[])
    if len(indices) == 1 and (not batched):
        index = indices[0]
        return SelfMTPCachePair(
            target=[cache.extract(index) for cache in caches.target],
            draft=[cache.extract(index) for cache in caches.draft],
        )
    rows = [
        SelfMTPCachePair(
            target=[cache.extract(index) for cache in caches.target],
            draft=[cache.extract(index) for cache in caches.draft],
        )
        for index in indices
    ]
    return SelfMTPCachePair(
        target=_merge_self_mtp_cache_groups([row.target for row in rows]),
        draft=_merge_self_mtp_cache_groups([row.draft for row in rows]),
    )


def _copy_build_self_mtp_cache_pair(
    current: Optional[SelfMTPCachePair],
    current_rows: int,
    joining: Sequence[SelfMTPCachePair],
) -> SelfMTPCachePair:
    """Build a complete replacement pair before publishing membership.

    ``extend`` and ``filter`` mutate layer objects one at a time. A late layer
    failure can therefore leave target and draft groups with different row
    membership. Extracting canonical rows and merging replacements keeps the
    old pair untouched until every layer has succeeded.
    """
    rows = []
    if current is not None:
        rows.extend(
            (
                SelfMTPCachePair(
                    target=[cache.extract(index) for cache in current.target],
                    draft=[cache.extract(index) for cache in current.draft],
                )
                for index in range(current_rows)
            )
        )
    rows.extend(joining)
    if not rows:
        return SelfMTPCachePair(target=[], draft=[])
    return SelfMTPCachePair(
        target=_merge_self_mtp_cache_groups([row.target for row in rows]),
        draft=_merge_self_mtp_cache_groups([row.draft for row in rows]),
    )


def _poison_self_mtp_batch(batch: BatchedSelfMTPState, reason: str) -> None:
    batch.poisoned = True
    batch.poison_reason = str(reason)


def _require_healthy_self_mtp_batch(batch: BatchedSelfMTPState) -> None:
    if batch.poisoned:
        reason = batch.poison_reason or "unproved transaction rollback"
        raise RuntimeError(f"self-MTP batch is poisoned: {reason}")


def _restart_live_self_mtp_or_poison(
    batch: BatchedSelfMTPState, cause: BaseException
) -> None:
    """Restore rollback recording after a failed copy-build operation."""
    try:
        _start_speculation_or_cleanup(
            batch.caches.target,
            batch.caches.target,
            "batched self-MTP requires ragged-trimmable target caches",
        )
    except BaseException as restart_error:
        _poison_self_mtp_batch(
            batch,
            f"membership rebuild failed ({cause}); rollback restart failed: {restart_error}",
        )
        raise RuntimeError(batch.poison_reason) from restart_error


def _prepare_self_mtp_cache_group(caches, lengths, right_padding) -> None:
    for cache in caches:
        prepare = getattr(cache, "prepare_self_mtp_step", None)
        if prepare is None:
            prepare = getattr(cache, "prepare", None)
        if prepare is None:
            if len(lengths) != 1 or any((int(value) for value in right_padding)):
                raise TypeError(
                    f"{type(cache).__name__} has no batched prepare contract"
                )
            continue
        prepare(lengths=lengths, right_padding=right_padding)


def _finalize_self_mtp_cache_group(caches) -> None:
    first_error = None
    for cache in caches:
        try:
            finalize = getattr(cache, "finalize_self_mtp_step", None)
            if finalize is None:
                finalize = getattr(cache, "finalize", None)
            if finalize is not None:
                finalize()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error


def _trim_self_mtp_cache_group(caches, counts, *, validate: bool) -> List[int]:
    """Trim a merged ragged cache or one standalone B1 cache exactly."""
    counts = [int(value) for value in counts]
    supports_ragged = bool(caches) and all(
        (getattr(cache, "supports_ragged_trim", lambda: False)() for cache in caches)
    )
    if not caches or supports_ragged or len(counts) != 1:
        return trim_ragged_prompt_cache(caches, counts, validate=validate)
    count = counts[0]
    if count < 0:
        raise ValueError("self-MTP trim count must be non-negative")
    if count == 0:
        return [0]
    if validate:
        position = _self_mtp_group_offset(caches)
        if count > position:
            raise ValueError("self-MTP B1 trim exceeds its cache position")
    applied = [int(cache.trim(count)) for cache in caches]
    if any((value != count for value in applied)):
        raise RuntimeError(
            f"standalone B1 trim diverged: expected {count}, got {applied}"
        )
    return [count]


def _eval_self_mtp_lane_state(detached: DetachedSelfMTPLane) -> None:
    values = [c.state for c in detached.caches.target]
    values.extend((c.state for c in detached.caches.draft))
    values.append(detached.lane.seed_h)
    if detached.lane.rng is not None:
        values.append(detached.lane.rng.key)
    mx.eval(*values)


def _lane_mtp_logprobs(lane: SelfMTPLane, logits: mx.array) -> mx.array:
    if lane.logprob_transform is not None:
        return lane.logprob_transform(logits)
    return _temperature_logprobs(logits, lane.sampling_temp)


def _lane_mtp_draft_logprobs(
    lane: SelfMTPLane, logits: mx.array, drafted: Sequence[mx.array]
) -> mx.array:
    """Apply the lane's output constraint before sampling an MTP draft token.

    Draft tokens are part of the generated prefix even though they remain
    provisional until target verification. Applying processors only during
    verification lets the draft head create an invalid intermediate prefix;
    a fail-closed grammar then has no legal continuation at the next position.
    Use the same prefix convention as target verification so both proposal
    distributions are conditioned on an identical constrained history.
    """
    if lane.logits_processors:
        processor_tokens = mx.concatenate(
            [lane.token_prefix, mx.array([lane.cur], mx.uint32)]
            + [mx.reshape(token, (1,)) for token in drafted]
        )
        logits = _probe_logits_processors(
            lane.logits_processors, processor_tokens, logits
        )
    return _lane_mtp_logprobs(lane, logits)


def prepare_self_mtp_lane(
    prompt: mx.array,
    model: nn.Module,
    *,
    uid: int,
    max_tokens: int,
    prompt_cache: Optional[List[Any]],
    mtp_state: Optional[Tuple[List[Any], mx.array]],
    lane_rng: Optional[LaneRNG],
    num_draft: int,
    sampling_temp: float,
    sampling_top_p: float,
    sampling_top_k: int,
    sampling_min_p: float,
    accept_rule: str,
    logits_processors: List[Callable],
    prefill_step_size: int,
    share_qsa_indices: bool,
    record_prefix_fanout: bool = False,
    diagnostic_stages: Optional[Dict[str, float]] = None,
    fused_gdn_catchup: bool = False,
    prompt_boundary_out: Optional[dict] = None,
    fly_verification=None,
) -> Tuple[DetachedSelfMTPLane, MTPToken]:
    """Prefill one canonical persistent self-MTP lane without attaching it.

    When ``prompt_boundary_out`` is supplied, capture an independent exact
    checkpoint immediately before the final prompt token is consumed.  That
    is the APC-compatible boundary: the target covers ``P - 1`` tokens, the
    draft covers ``P - 2``, and ``seed_h`` is the hidden state for token
    ``P - 2``.  No speculative proposal has been opened at this point, so a
    rejected draft can never leak into this snapshot.
    """
    if getattr(model, "mtp", None) is None:
        raise ValueError("model has no MTP head")
    if max_tokens <= 0:
        raise ValueError("prepare_self_mtp_lane requires max_tokens > 0")
    if num_draft < 1:
        raise ValueError("prepare_self_mtp_lane requires num_draft >= 1")
    if prefill_step_size <= 0:
        raise ValueError("prefill_step_size must be positive")
    if accept_rule not in ("exact", "residual", "block"):
        raise ValueError(
            f"accept_rule must be 'exact', 'residual', or 'block'; got {accept_rule!r}"
        )
    if lane_rng is not None and (not isinstance(lane_rng, LaneRNG)):
        raise TypeError("lane_rng must be a sample_utils.LaneRNG")
    if prompt.ndim != 1 or int(prompt.size) == 0:
        raise ValueError("prompt must be a non-empty rank-1 token array")
    transform = _make_sampling_transform(
        sampling_temp, sampling_top_p, sampling_top_k, sampling_min_p
    )
    if transform is not None and accept_rule != "residual":
        raise ValueError("transformed sampling supports only accept_rule='residual'")
    target_cache = (
        prompt_cache if prompt_cache is not None else make_prompt_cache(model)
    )
    _reject_unsupported_self_mtp_caches(target_cache)
    if mtp_state is None:
        draft_cache = model.make_mtp_cache()
        restored_seed_h = None
    else:
        (draft_cache, restored_seed_h) = _restore_mtp_state(target_cache, mtp_state)
    _reject_unsupported_self_mtp_caches(draft_cache)
    diagnostic_started_ns = time.perf_counter_ns()

    def finish_diagnostic_stage(name: str) -> None:
        nonlocal diagnostic_started_ns
        if diagnostic_stages is None:
            return
        mx.synchronize(generation_stream)
        now_ns = time.perf_counter_ns()
        diagnostic_stages[name] = (
            diagnostic_stages.get(name, 0.0)
            + (now_ns - diagnostic_started_ns) / 1000000.0
        )
        diagnostic_started_ns = now_ns

    finish_diagnostic_stage("cache_restore_setup_ms")
    processor_prompt = prompt.astype(mx.uint32)
    y = processor_prompt
    prev_h = restored_seed_h
    with mx.stream(generation_stream):
        while y.size > 1:
            n = min(prefill_step_size, int(y.size) - 1)
            scope = getattr(model, "gdn_catchup_scope", None)
            context = scope(fused_gdn_catchup) if callable(scope) else nullcontext()
            with context:
                (_, h_chunk) = _mtp_backbone(model, y[:n][None], target_cache)
            if diagnostic_stages is not None:
                mx.eval(h_chunk, [c.state for c in target_cache])
            finish_diagnostic_stage("target_catchup_ms")
            if prev_h is None:
                (hs, ts) = (h_chunk[:, :-1], y[1:n][None])
            else:
                hs = mx.concatenate([prev_h, h_chunk[:, :-1]], axis=1)
                ts = y[:n][None]
            if ts.size > 0:
                model.mtp_step(hs, ts, draft_cache)
                if diagnostic_stages is not None:
                    mx.eval([c.state for c in draft_cache])
            finish_diagnostic_stage("mtp_teacher_force_ms")
            prev_h = h_chunk[:, -1:, :]
            mx.eval([c.state for c in target_cache], [c.state for c in draft_cache])
            finish_diagnostic_stage("cache_eval_ms")
            y = y[n:]
            mx.clear_cache()
            finish_diagnostic_stage("cache_clear_ms")
        if prompt_boundary_out is not None and prev_h is not None:
            checkpoint = capture_self_mtp_checkpoint(
                target_cache,
                (draft_cache, prev_h),
                rng_key=None if lane_rng is None else lane_rng.key,
                rng_draws=0 if lane_rng is None else int(lane_rng.draws),
            )
            if checkpoint is not None:
                prompt_boundary_out.update(checkpoint)
        if prev_h is not None:
            model.mtp_step(prev_h, y[None], draft_cache)
            if diagnostic_stages is not None:
                mx.eval([c.state for c in draft_cache])
        finish_diagnostic_stage("final_mtp_boundary_ms")
        if record_prefix_fanout:
            _start_speculation_or_cleanup(
                target_cache,
                target_cache,
                "GDN prefix fan-out requires exact target-cache rollback",
            )
        try:
            (logit_hidden, hidden) = _mtp_backbone(model, y[None], target_cache)
            if diagnostic_stages is not None:
                mx.eval(logit_hidden, hidden, [c.state for c in target_cache])
            finish_diagnostic_stage("final_target_m1_ms")
            seed_h = hidden[:, -1:, :]
            logits = model.logits(logit_hidden[:, -1:, :])[0, -1]
            logits = _apply_logits_processors(
                logits_processors, processor_prompt, logits
            )
            first_lp = (
                transform(logits)
                if transform is not None
                else _temperature_logprobs(logits, sampling_temp)
            )
            cur = _sample_from_logprobs(first_lp, sampling_temp, rng=lane_rng)
            if diagnostic_stages is not None:
                mx.eval(first_lp, cur)
            finish_diagnostic_stage("lm_head_and_sample_ms")
        except BaseException:
            if record_prefix_fanout:
                _stop_all_speculation(target_cache)
            raise
    stats = HybridStats()
    stats.plain_tokens = 1
    from .speculative_sampling import FLyVerificationPolicy

    lane = SelfMTPLane(
        uid=int(uid),
        cur=cur,
        seed_h=seed_h,
        pending_hs=None,
        pending_ts=[],
        token_prefix=processor_prompt,
        rng=lane_rng,
        ntoks=1,
        max_tokens=int(max_tokens),
        num_draft=int(num_draft),
        sampling_temp=float(sampling_temp),
        accept_rule=accept_rule,
        logprob_transform=transform,
        logits_processors=list(logits_processors or []),
        stats=stats,
        share_qsa_indices=bool(share_qsa_indices),
        fly_verification=FLyVerificationPolicy.from_value(fly_verification),
    )
    detached = DetachedSelfMTPLane(
        lane=lane, caches=SelfMTPCachePair(target=target_cache, draft=draft_cache)
    )
    _eval_self_mtp_lane_state(detached)
    _validate_detached_self_mtp(detached)
    finish_diagnostic_stage("lane_finalize_ms")
    return (detached, MTPToken(cur, first_lp, False))


def advance_self_mtp_prefill(
    prompt: mx.array,
    model: nn.Module,
    *,
    prompt_cache: Optional[List[Any]],
    mtp_state: Optional[Tuple[List[Any], mx.array]],
    max_tokens: int,
    fused_gdn_catchup: bool = False,
) -> Tuple[mx.array, List[Any], Tuple[List[Any], mx.array], int]:
    """Advance one exact teacher-forced self-MTP prefill slice.

    The final prompt token is deliberately retained. Once only that token
    remains, :func:`prepare_self_mtp_lane` can consume it and enter generation
    through the existing snapshot, sampling, and transaction path. Returning
    both target and draft state lets the serving scheduler interleave these
    slices with an active self-MTP batch without opening a speculative proposal.
    """
    if prompt.ndim != 1 or int(prompt.size) == 0:
        raise ValueError("prompt must be a non-empty rank-1 token array")
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    target_cache = (
        prompt_cache if prompt_cache is not None else make_prompt_cache(model)
    )
    _reject_unsupported_self_mtp_caches(target_cache)
    if mtp_state is None:
        draft_cache = model.make_mtp_cache()
        prev_h = None
    else:
        (draft_cache, prev_h) = _restore_mtp_state(target_cache, mtp_state)
    _reject_unsupported_self_mtp_caches(draft_cache)
    n = min(int(max_tokens), int(prompt.size) - 1)
    if n <= 0:
        return (prompt, target_cache, (draft_cache, prev_h), 0)
    with mx.stream(generation_stream):
        scope = getattr(model, "gdn_catchup_scope", None)
        context = scope(fused_gdn_catchup) if callable(scope) else nullcontext()
        with context:
            (_, h_chunk) = _mtp_backbone(model, prompt[:n][None], target_cache)
        if prev_h is None:
            (hs, ts) = (h_chunk[:, :-1], prompt[1:n][None])
        else:
            hs = mx.concatenate([prev_h, h_chunk[:, :-1]], axis=1)
            ts = prompt[:n][None]
        if ts.size > 0:
            model.mtp_step(hs, ts, draft_cache)
        prev_h = h_chunk[:, -1:, :]
        mx.eval([c.state for c in target_cache], [c.state for c in draft_cache])
        mx.clear_cache()
    return (prompt[n:], target_cache, (draft_cache, prev_h), n)


def capture_self_mtp_checkpoint(
    target_cache,
    mtp_state,
    *,
    rng_key=None,
    rng_draws: int = 0,
):
    """Capture one exact committed target/draft checkpoint.

    The target covers ``p`` tokens, the draft covers ``p - 1`` teacher-forced
    pairs, and the saved seed is the trunk hidden for token ``p - 1``.  This is
    the same invariant :func:`_restore_mtp_state` validates.  Descriptor COW is
    preferred; the existing deep-copy fallback preserves the historical prompt
    boundary contract when descriptor capture is unavailable.
    """
    (draft_cache, seed_h) = mtp_state
    covered_tokens = _self_mtp_group_offset(target_cache)
    draft_covered = _self_mtp_group_offset(draft_cache)
    if covered_tokens <= 0 or seed_h is None or draft_covered != covered_tokens - 1:
        return None
    values = [c.state for c in target_cache]
    values.extend((c.state for c in draft_cache))
    values.append(seed_h)
    if rng_key is not None:
        values.append(rng_key)
    mx.eval(*values)
    snapshot_started_ns = time.perf_counter_ns()
    snapshot_mode = "deepcopy_fallback"
    try:
        if not mtp_boundary_cow_enabled():
            raise RuntimeError("MTP boundary descriptor COW disabled")
        (saved_target, saved_sidecar, snapshot_receipt) = (
            snapshot_prompt_cache_descriptors(
                target_cache,
                (draft_cache, seed_h, rng_key),
            )
        )
        (saved_draft, saved_seed, saved_rng_key) = saved_sidecar
        snapshot_mode = "descriptor_cow"
    except Exception:
        saved_target = copy.deepcopy(target_cache)
        saved_draft = copy.deepcopy(draft_cache)
        saved_seed = copy.deepcopy(seed_h)
        saved_rng_key = None if rng_key is None else copy.deepcopy(rng_key)
        snapshot_receipt = {
            "snapshot_ns": int(time.perf_counter_ns() - snapshot_started_ns),
            "snapshot_bytes": int(
                sum((int(getattr(c, "nbytes", 0)) for c in saved_target))
                + sum((int(getattr(c, "nbytes", 0)) for c in saved_draft))
                + int(getattr(saved_seed, "nbytes", 0))
                + int(getattr(saved_rng_key, "nbytes", 0))
            ),
        }
    saved_values = [c.state for c in saved_target]
    saved_values.extend((c.state for c in saved_draft))
    saved_values.append(saved_seed)
    if saved_rng_key is not None:
        saved_values.append(saved_rng_key)
    mx.eval(*saved_values)
    return {
        "target_cache": saved_target,
        "mtp_state": (saved_draft, saved_seed),
        "covered_tokens": covered_tokens,
        "rng_key": saved_rng_key,
        "rng_draws": int(rng_draws),
        "committed_only": True,
        "snapshot_mode": snapshot_mode,
        **snapshot_receipt,
    }


def attach_self_mtp_lanes(
    model: nn.Module,
    batch: Optional[BatchedSelfMTPState],
    joining: Sequence[DetachedSelfMTPLane],
) -> BatchedSelfMTPState:
    """Attach canonical rows at a transaction boundary and restart rollback."""
    joining = list(joining)
    if batch is not None:
        _require_healthy_self_mtp_batch(batch)
        if batch.proposal_open:
            raise RuntimeError("cannot attach self-MTP lanes while a proposal is open")
    if not joining:
        if batch is None:
            raise ValueError("cannot create an empty self-MTP batch")
        return batch
    for detached in joining:
        _validate_detached_self_mtp(detached)
    old_lanes = [] if batch is None else batch.lanes
    uids = [lane.uid for lane in old_lanes] + [item.lane.uid for item in joining]
    if len(set(uids)) != len(uids):
        raise ValueError("self-MTP lane uid values must be unique")
    configured = {lane.num_draft for lane in old_lanes}
    configured.update((item.lane.num_draft for item in joining))
    if len(configured) != 1:
        raise ValueError("adaptive per-lane self-MTP depth is excluded")
    share_modes = {lane.share_qsa_indices for lane in old_lanes}
    share_modes.update((item.lane.share_qsa_indices for item in joining))
    if len(share_modes) != 1:
        raise ValueError("mixed shared-QSA modes cannot enter one self-MTP batch")
    if batch is None or not batch.lanes:
        incoming = _copy_build_self_mtp_cache_pair(
            None, 0, [item.caches for item in joining]
        )
        epoch = 1 if batch is None else batch.membership_epoch + 1
        result = BatchedSelfMTPState(
            lanes=[item.lane for item in joining],
            caches=incoming,
            membership_epoch=epoch,
        )
    else:
        try:
            _stop_all_speculation(batch.caches.target)
        except BaseException as error:
            _poison_self_mtp_batch(batch, f"rollback stop failed: {error}")
            raise
        try:
            replacement = _copy_build_self_mtp_cache_pair(
                batch.caches, len(batch.lanes), [item.caches for item in joining]
            )
            _start_speculation_or_cleanup(
                replacement.target,
                replacement.target,
                "batched self-MTP requires ragged-trimmable target caches",
            )
        except BaseException as error:
            _restart_live_self_mtp_or_poison(batch, error)
            raise
        batch.caches = replacement
        batch.lanes = [*batch.lanes, *(item.lane for item in joining)]
        batch.membership_epoch += 1
        result = batch
    if batch is None or not old_lanes:
        _start_speculation_or_cleanup(
            result.caches.target,
            result.caches.target,
            "batched self-MTP requires ragged-trimmable target caches",
        )
    return result


def attach_prebatched_self_mtp_lanes(
    model: nn.Module,
    detached_lanes: Sequence[DetachedSelfMTPLane],
    caches: SelfMTPCachePair,
) -> BatchedSelfMTPState:
    """Publish a fully built cache batch and start its rollback transaction.

    Prefix fan-out builds every target and draft layer before this call.  This
    admission boundary performs the same invariants as ``attach_self_mtp_lanes``
    without extracting and re-merging those already batched rows.
    """
    detached_lanes = list(detached_lanes)
    if not detached_lanes:
        raise ValueError("a prebatched self-MTP cache needs at least one lane")
    for detached in detached_lanes:
        _validate_detached_self_mtp(detached)
    lanes = [detached.lane for detached in detached_lanes]
    if len({lane.uid for lane in lanes}) != len(lanes):
        raise ValueError("self-MTP lane uid values must be unique")
    if len({lane.num_draft for lane in lanes}) != 1:
        raise ValueError("adaptive per-lane self-MTP depth is excluded")
    if len({lane.share_qsa_indices for lane in lanes}) != 1:
        raise ValueError("mixed shared-QSA modes cannot enter one self-MTP batch")
    if not caches.target or not caches.draft:
        raise ValueError("prebatched self-MTP needs target and draft cache groups")
    _reject_unsupported_self_mtp_caches(caches.target)
    _reject_unsupported_self_mtp_caches(caches.draft)
    rows = len(lanes)
    for cache in [*caches.target, *caches.draft]:
        batch_size = getattr(cache, "batch_size", None)
        if batch_size is None:
            offset = getattr(cache, "offset", None)
            if isinstance(offset, mx.array) and offset.ndim == 1:
                batch_size = int(offset.size)
            elif isinstance(offset, (list, tuple)):
                batch_size = len(offset)
            else:
                raise ValueError(
                    f"{type(cache).__name__} does not expose batched row count"
                )
        if batch_size != rows:
            raise ValueError(
                f"{type(cache).__name__} has {batch_size} rows, expected {rows}"
            )

    def batched_group_offset(group):
        offsets = []
        for cache in group:
            offset = getattr(cache, "offset", None)
            if isinstance(offset, mx.array):
                values = [int(value) for value in offset.tolist()]
                if len(set(values)) != 1:
                    raise ValueError(
                        "a freshly attached prebatched cache must have uniform offsets"
                    )
                offsets.append(values[0])
            elif isinstance(offset, int):
                offsets.append(offset)
            elif callable(getattr(cache, "size", None)):
                offsets.append(int(cache.size()))
            else:
                raise ValueError(
                    f"{type(cache).__name__} does not expose a cache offset"
                )
        return max(offsets, default=0)

    covered = batched_group_offset(caches.target)
    draft_offset = batched_group_offset(caches.draft)
    if covered <= 0 or draft_offset != covered - 1:
        raise ValueError(
            f"prebatched self-MTP cache mismatch: target covers {covered} tokens but draft covers {draft_offset} pairs"
        )
    result = BatchedSelfMTPState(lanes, caches, 1)
    _start_speculation_or_cleanup(
        result.caches.target,
        result.caches.target,
        "prebatched self-MTP requires ragged-trimmable target caches",
    )
    return result


def attach_segmented_self_mtp_lanes(
    model: nn.Module,
    batch: Optional[SegmentedSelfMTPState],
    joining: Sequence[DetachedSelfMTPLane],
) -> SegmentedSelfMTPState:
    """Attach request-private B1 rows without constructing a merged cache."""
    from .segmented_self_mtp import SegmentedLaneTransaction, note_segmented_self_mtp

    joining = list(joining)
    if batch is not None:
        if not isinstance(batch, SegmentedSelfMTPState):
            raise TypeError("cannot mix segmented and physically batched MTP state")
        _require_healthy_self_mtp_batch(batch)
        if batch.proposal_open:
            raise RuntimeError("cannot attach self-MTP lanes during a proposal")
    if not joining:
        if batch is None:
            return SegmentedSelfMTPState([], [], [], 0)
        return batch
    for detached in joining:
        _validate_detached_self_mtp(detached)
    prefix_ids = [item.shared_qsa_prefix_id for item in joining]
    initial_shared_qsa_prefix_id = (
        prefix_ids[0]
        if batch is None
        and len(prefix_ids) >= 2
        and (prefix_ids[0] is not None)
        and all((value == prefix_ids[0] for value in prefix_ids[1:]))
        else None
    )
    if initial_shared_qsa_prefix_id is not None:
        from .models.qwen4_exp import QSAKVCache
        from .qsa_shared_suffix import split_attested_qsa_rows
        from .segmented_self_mtp import (
            qsa_private_delta_enabled,
            shared_qsa_suffix_admission,
        )

        target_groups = [item.caches.target for item in joining]
        qsa_contexts = [
            int(layer_rows[0].offset)
            for layer_rows in zip(*target_groups)
            if all((type(row) is QSAKVCache for row in layer_rows))
        ]
        base_tokens = min(qsa_contexts, default=0)
        remaining_tokens = max(
            (item.lane.max_tokens - item.lane.ntoks for item in joining)
        )
        (admitted, _reason, cutoff) = shared_qsa_suffix_admission(
            base_tokens=base_tokens, remaining_tokens=remaining_tokens
        )
        note_segmented_self_mtp("shared_qsa_policy_checks")
        note_segmented_self_mtp(
            "shared_qsa_policy_admitted" if admitted else "shared_qsa_policy_declined"
        )
        note_segmented_self_mtp(
            "shared_qsa_policy_context_tokens_cumulative", base_tokens
        )
        note_segmented_self_mtp(
            "shared_qsa_policy_remaining_tokens_cumulative", remaining_tokens
        )
        note_segmented_self_mtp("shared_qsa_policy_cutoff_tokens_cumulative", cutoff)
        if admitted and qsa_private_delta_enabled():
            updates = []
            for layer_index, layer_rows in enumerate(zip(*target_groups)):
                if all((type(row) is QSAKVCache for row in layer_rows)):
                    updates.append(
                        (
                            layer_index,
                            split_attested_qsa_rows(
                                layer_rows,
                                layout_id=f"qwen4-shared-qsa-v1:{layer_index}",
                                note=note_segmented_self_mtp,
                            ),
                        )
                    )
            # Scheduler transfers already own a revision-bound transaction.
            # Validate it before replacing the physical layout, then bind a
            # successor to the exact shared representation at the same position.
            predecessors = []
            for item in joining:
                old = item.segment_transaction
                position = _self_mtp_group_offset(item.caches.target)
                if old is not None:
                    old.validate(item.caches, item.lane, position)
                predecessors.append((item, old, position))
            for layer_index, rows in updates:
                for item, row in zip(joining, rows):
                    item.caches.target[layer_index] = row
            for item, old, position in predecessors:
                if old is not None and updates:
                    successor = SegmentedLaneTransaction(item.caches, item.lane, position)
                    successor.predecessor_lineage_id = old.lineage.lineage_id
                    item.segment_transaction = successor
                    old.close()
                    note_segmented_self_mtp("transaction_canonicalizations")
    if batch is not None and any(
        (
            getattr(cache, "supports_shared_qsa_suffix", False)
            for pair in batch.row_caches
            for cache in pair.target
        )
    ):
        replacements = []
        try:
            for pair, lane, transaction in zip(
                batch.row_caches, batch.lanes, batch.transactions
            ):
                position = transaction.position
                transaction.validate(pair, lane, position)
                target = list(pair.target)
                changed = False
                for layer_index, cache in enumerate(target):
                    if getattr(cache, "supports_shared_qsa_suffix", False):
                        (target[layer_index], _) = cache.materialize_to_qsa()
                        changed = True
                if changed:
                    _stop_all_speculation(pair.target)
                    pair.target = target
                    _start_speculation_or_cleanup(
                        pair.target,
                        pair.target,
                        "materialized shared-QSA rows must remain trimmable",
                    )
                    successor = SegmentedLaneTransaction(pair, lane, position)
                    successor.predecessor_lineage_id = transaction.lineage.lineage_id
                    replacements.append((transaction, successor))
                else:
                    replacements.append((transaction, transaction))
        except BaseException as error:
            _poison_self_mtp_batch(
                batch, f"shared-QSA membership materialization failed: {error}"
            )
            note_segmented_self_mtp("failures")
            raise
        batch.transactions = [successor for (_, successor) in replacements]
        for previous, successor in replacements:
            if previous is not successor:
                previous.close()
                note_segmented_self_mtp("transaction_canonicalizations")
    existing_lanes = [] if batch is None else batch.lanes
    uids = [lane.uid for lane in existing_lanes]
    uids.extend((item.lane.uid for item in joining))
    if len(set(uids)) != len(uids):
        raise ValueError("self-MTP lane uid values must be unique")
    depths = {lane.num_draft for lane in existing_lanes}
    depths.update((item.lane.num_draft for item in joining))
    if len(depths) != 1:
        raise ValueError("adaptive per-lane self-MTP depth is excluded")
    share_modes = {lane.share_qsa_indices for lane in existing_lanes}
    share_modes.update((item.lane.share_qsa_indices for item in joining))
    if len(share_modes) != 1:
        raise ValueError("mixed shared-QSA modes cannot enter one MTP cohort")
    existing_objects = {
        id(cache)
        for pair in ([] if batch is None else batch.row_caches)
        for cache in [*pair.target, *pair.draft]
    }
    joining_objects = []
    for item in joining:
        objects = {id(cache) for cache in [*item.caches.target, *item.caches.draft]}
        if existing_objects.intersection(objects):
            raise ValueError("segmented self-MTP lanes must own distinct cache objects")
        if any((objects.intersection(previous) for previous in joining_objects)):
            raise ValueError("segmented self-MTP lanes alias one cache object")
        joining_objects.append(objects)
    started = []
    transactions = []
    created_transactions = []
    try:
        for item in joining:
            _start_speculation_or_cleanup(
                item.caches.target,
                item.caches.target,
                "segmented self-MTP requires trimmable B1 target caches",
            )
            started.append(item.caches.target)
            position = _self_mtp_group_offset(item.caches.target)
            transaction = item.segment_transaction
            if transaction is None:
                transaction = SegmentedLaneTransaction(item.caches, item.lane, position)
                created_transactions.append((item, transaction))
            else:
                transaction.validate(item.caches, item.lane, position)
                if transaction.position != position:
                    raise ValueError(
                        "segmented cache transaction position disagrees with target cache"
                    )
            item.segment_transaction = transaction
            transactions.append(transaction)
    except BaseException:
        for caches in started:
            try:
                _stop_all_speculation(caches)
            except BaseException:
                pass
        for item, transaction in created_transactions:
            try:
                transaction.close()
            except BaseException:
                pass
            if item.segment_transaction is transaction:
                item.segment_transaction = None
        note_segmented_self_mtp("failures")
        raise
    if batch is None:
        result = SegmentedSelfMTPState(
            lanes=[item.lane for item in joining],
            row_caches=[item.caches for item in joining],
            transactions=transactions,
            membership_epoch=1,
            shared_qsa_prefix_id=initial_shared_qsa_prefix_id,
        )
        note_segmented_self_mtp("requests")
        note_segmented_self_mtp("engaged")
    else:
        batch._segmented_caches = None
        batch.shared_qsa_prefix_id = None
        batch.lanes.extend((item.lane for item in joining))
        batch.row_caches.extend((item.caches for item in joining))
        batch.transactions.extend(transactions)
        batch.membership_epoch += 1
        result = batch
    return result


def close_segmented_self_mtp_state(batch: SegmentedSelfMTPState) -> None:
    """Stop rollback and release every transaction lineage exactly once."""
    first_error = None
    for pair in batch.row_caches:
        try:
            _stop_all_speculation(pair.target)
        except BaseException as error:
            if first_error is None:
                first_error = error
        close_target = getattr(pair.target, "close", None)
        if callable(close_target):
            try:
                close_target()
            except BaseException as error:
                if first_error is None:
                    first_error = error
    for transaction in batch.transactions:
        try:
            transaction.close()
        except BaseException as error:
            if first_error is None:
                first_error = error
    batch.lanes.clear()
    batch.row_caches.clear()
    batch.transactions.clear()
    batch._row_states.clear()
    batch._row_proposals.clear()
    batch._batched_state = None
    batch._segmented_caches = None
    batch._transaction_branches.clear()
    batch._recovery_checkpoints.clear()
    batch.proposal_open = False
    batch._open_proposal = None
    if first_error is not None:
        raise first_error


_MTP_LANE_ARRAY_FIELDS = frozenset({"seed_h", "pending_hs", "token_prefix"})


def _snapshot_segmented_recovery_row(value, memo=None):
    """Freeze one committed MTP row without rewinding its random stream."""
    lane, pair = value
    # Append-only KV planes keep only their fill level: an alias of a buffer
    # the cycle then appends to would copy that whole buffer every append.
    target, draft, borrowed = snapshot_recovery_descriptors(
        pair.target, pair.draft, memo=memo
    )
    from .processor_probe import copy_sharing, rollback_shared_memo

    shared = rollback_shared_memo(getattr(lane, "logits_processors", ()))
    lane_fields = {}
    for name, current in vars(lane).items():
        if name == "rng":
            continue
        lane_fields[name] = (
            current
            if name in _MTP_LANE_ARRAY_FIELDS
            else copy_sharing(current, shared)
        )
    return lane_fields, target, draft, borrowed


def _frozen_segmented_recovery_row(snapshot):
    """Hand back the frozen row; the cohort restore clones every row at once."""
    return snapshot


def _restore_segmented_recovery_row(snapshot, memo=None):
    lane_fields, target, draft, borrowed = snapshot
    restored_target, restored_draft = restore_recovery_descriptors(
        target, draft, borrowed, memo=memo
    )
    restored_fields = {}
    for name, current in lane_fields.items():
        restored_fields[name] = (
            current if name in _MTP_LANE_ARRAY_FIELDS else copy.deepcopy(current)
        )
    return restored_fields, SelfMTPCachePair(restored_target, restored_draft)


def _capture_segmented_recovery(batch: SegmentedSelfMTPState) -> None:
    from .segmented_self_mtp import note_segmented_self_mtp

    checkpoints = []
    # One clone memo for the whole cohort keeps what rows share, such as one
    # immutable QSA base, shared in the checkpoint: the segmented view refuses
    # rows whose bases are separate copies. The memo is keyed by live object
    # ids, so no row restarts speculation, which may replace objects, until
    # every row has been captured.
    memo = {}
    stopped = []
    try:
        try:
            for lane, pair, transaction in zip(
                batch.lanes, batch.row_caches, batch.transactions
            ):
                boundary = int(transaction.position)
                transaction.validate(pair, lane, boundary)
                revision = (
                    f"{transaction.lineage.lineage_id}:"
                    f"{batch.membership_epoch}:{lane.uid}"
                )
                slot = CommittedRecoverySlot()
                _stop_all_speculation(pair.target)
                stopped.append(pair)
                slot.capture(
                    route="self_mtp",
                    revision=revision,
                    boundary=boundary,
                    value=(lane, pair),
                    snapshot=lambda value: _snapshot_segmented_recovery_row(
                        value, memo=memo
                    ),
                    restore=_frozen_segmented_recovery_row,
                )
                checkpoints.append((slot, revision, boundary))
        finally:
            for pair in stopped:
                _start_speculation_or_cleanup(
                    pair.target,
                    pair.target,
                    "segmented MTP recovery capture must restart rollback",
                )
        for lane, pair, transaction, (_slot, _revision, boundary) in zip(
            batch.lanes, batch.row_caches, batch.transactions, checkpoints
        ):
            transaction.validate(pair, lane, boundary)
    except BaseException:
        batch._recovery_checkpoints.clear()
        note_segmented_self_mtp("recovery_checkpoint_failures")
        raise
    batch._recovery_checkpoints = checkpoints
    note_segmented_self_mtp("recovery_checkpoint_captures", len(checkpoints))


def _restore_segmented_recovery(batch: SegmentedSelfMTPState) -> None:
    """Restore canonical rows and restart their generation-bound lineages."""
    from .segmented_self_mtp import SegmentedLaneTransaction, note_segmented_self_mtp

    checkpoints = list(batch._recovery_checkpoints)
    if len(checkpoints) != len(batch.lanes):
        note_segmented_self_mtp("recovery_checkpoint_failures")
        raise RuntimeError("segmented MTP recovery checkpoints are misaligned")
    for pair in batch.row_caches:
        _stop_all_speculation(pair.target)
    for transaction in batch.transactions:
        transaction.close()
    restored_pairs = []
    restored_fields = []
    # Restore the cohort through one memo, as it was captured, so rows that
    # shared an immutable base share its restored clone.
    memo = {}
    for slot, revision, boundary in checkpoints:
        fields, pair = _restore_segmented_recovery_row(
            slot.restore(route="self_mtp", revision=revision, boundary=boundary),
            memo=memo,
        )
        restored_fields.append(fields)
        restored_pairs.append(pair)
    for lane, fields in zip(batch.lanes, restored_fields):
        rng = lane.rng
        for name, current in fields.items():
            setattr(lane, name, current)
        lane.rng = rng
    for pair in restored_pairs:
        _start_speculation_or_cleanup(
            pair.target,
            pair.target,
            "segmented MTP recovery restore must restart rollback",
        )
    batch.row_caches = restored_pairs
    batch.transactions = [
        SegmentedLaneTransaction(pair, lane, boundary)
        for lane, pair, (_slot, _revision, boundary) in zip(
            batch.lanes, restored_pairs, checkpoints
        )
    ]
    batch._recovery_checkpoints.clear()
    batch._segmented_caches = None
    note_segmented_self_mtp("recovery_checkpoint_restores", len(restored_pairs))


def _propose_segmented_self_mtp(
    model: nn.Module, batch: SegmentedSelfMTPState
) -> SelfMTPCycleResult:
    """Run one batched cycle over B1 lineages, or the exact serial fallback."""
    from .segmented_self_mtp import (
        note_segmented_self_mtp,
        segmented_self_mtp_timing_enabled,
        true_batched_segmented_self_mtp_enabled,
    )

    _require_healthy_self_mtp_batch(batch)
    if batch.proposal_open:
        raise RuntimeError("a self-MTP proposal is already open")
    if not batch.lanes:
        raise ValueError("cannot propose on an empty self-MTP batch")
    if not len(batch.lanes) == len(batch.row_caches) == len(batch.transactions):
        raise RuntimeError("segmented self-MTP row ownership is misaligned")
    _capture_segmented_recovery(batch)
    timing = segmented_self_mtp_timing_enabled()
    started_ns = time.perf_counter_ns() if timing else 0
    row_states = []
    row_proposals = []
    branches = []
    try:
        for lane, pair, transaction in zip(
            batch.lanes, batch.row_caches, batch.transactions
        ):
            transaction.validate(pair, lane, transaction.position)
            branches.append(
                transaction.fork(f"mtp:{batch.membership_epoch}:{lane.uid}")
            )
        true_batched = true_batched_segmented_self_mtp_enabled()
        if true_batched:
            from .segmented_batch_cache import (
                SegmentedBatchUnsupported,
                build_segmented_batch_cache_pair,
            )

            note_segmented_self_mtp("true_batched_requests")
            compute_caches = batch._segmented_caches
            if compute_caches is None:
                try:
                    compute_caches = build_segmented_batch_cache_pair(
                        batch.row_caches,
                        note=note_segmented_self_mtp,
                        shared_qsa_prefix=batch.shared_qsa_prefix_id is not None,
                    )
                except SegmentedBatchUnsupported:
                    note_segmented_self_mtp("true_batched_declined")
            if compute_caches is not None:
                batch._segmented_caches = compute_caches
                batched_state = BatchedSelfMTPState(
                    lanes=batch.lanes,
                    caches=compute_caches,
                    membership_epoch=batch.membership_epoch,
                )
                proposal = _propose_batched_self_mtp_impl(model, batched_state)
                batch._batched_state = batched_state
                batch._row_states = []
                batch._row_proposals = []
                batch._transaction_branches = branches
                batch.proposal_open = True
                batch._open_proposal = proposal
                note_segmented_self_mtp("true_batched_engaged")
                note_segmented_self_mtp("batched_target_forwards")
                note_segmented_self_mtp(
                    "batched_draft_forwards", max(proposal.draft_depths, default=0)
                )
                if timing:
                    note_segmented_self_mtp(
                        "proposal_ns", time.perf_counter_ns() - started_ns
                    )
                return proposal
        else:
            batch._segmented_caches = None
        batch.shared_qsa_prefix_id = None
        for lane, pair, transaction in zip(
            batch.lanes, batch.row_caches, batch.transactions
        ):
            row_state = BatchedSelfMTPState(
                lanes=[lane], caches=pair, membership_epoch=batch.membership_epoch
            )
            row_states.append(row_state)
            proposal = _propose_batched_self_mtp_impl(model, row_state)
            row_proposals.append(proposal)
            note_segmented_self_mtp("b1_target_forwards")
            note_segmented_self_mtp("b1_draft_forwards", proposal.draft_depths[0])
    except BaseException as error:
        for branch in branches:
            try:
                branch.close()
            except BaseException:
                pass
        batch.proposal_open = False
        batch._open_proposal = None
        batch._batched_state = None
        batch._segmented_caches = None
        try:
            _restore_segmented_recovery(batch)
        except BaseException as recovery_error:
            _poison_self_mtp_batch(
                batch,
                f"segmented proposal failed: {error}; recovery failed: {recovery_error}",
            )
        note_segmented_self_mtp("failures")
        raise
    aggregate = SelfMTPCycleResult(
        membership_epoch=batch.membership_epoch,
        lane_uids=tuple((item.lane_uids[0] for item in row_proposals)),
        draft_depths=tuple((item.draft_depths[0] for item in row_proposals)),
        accepted_lengths=tuple((item.accepted_lengths[0] for item in row_proposals)),
        target_drops=tuple((item.target_drops[0] for item in row_proposals)),
        head_drops=tuple((item.head_drops[0] for item in row_proposals)),
        outputs=tuple((item.outputs[0] for item in row_proposals)),
        _old_curs=tuple((item._old_curs[0] for item in row_proposals)),
        _old_seed_hs=tuple((item._old_seed_hs[0] for item in row_proposals)),
        _drafts=tuple((item._drafts[0] for item in row_proposals)),
        _vhidden=tuple((item._vhidden[0] for item in row_proposals)),
        _logprobs=tuple((item._logprobs[0] for item in row_proposals)),
        _bonuses=tuple((item._bonuses[0] for item in row_proposals)),
        relaxed_accepts=tuple(
            item.relaxed_accepts[0] if item.relaxed_accepts else 0
            for item in row_proposals
        ),
        copy_spans=tuple(
            item.copy_spans[0] if item.copy_spans else 0 for item in row_proposals
        ),
        copy_decisions=tuple(
            item.copy_decisions[0] if item.copy_decisions else "off"
            for item in row_proposals
        ),
        draft_features=(
            tuple(
                item.draft_features[0] if item.draft_features else ()
                for item in row_proposals
            )
            if any(item.draft_features for item in row_proposals)
            else ()
        ),
        draft_feature_tokens=(
            tuple(
                item.draft_feature_tokens[0] if item.draft_feature_tokens else ()
                for item in row_proposals
            )
            if any(item.draft_feature_tokens for item in row_proposals)
            else ()
        ),
    )
    batch._row_states = row_states
    batch._row_proposals = row_proposals
    batch._batched_state = None
    batch._transaction_branches = branches
    batch.proposal_open = True
    batch._open_proposal = aggregate
    if timing:
        note_segmented_self_mtp("proposal_ns", time.perf_counter_ns() - started_ns)
    return aggregate


def _plan_copy_drafts(
    lanes: Sequence[SelfMTPLane], head_depths: Sequence[int]
) -> Tuple[Tuple[Tuple[int, ...], ...], Tuple[str, ...]]:
    """Choose, per lane, a copied span or the MTP head for this round.

    Host-only (dict lookups over committed tokens); never touches caches.
    Lanes without copy state keep the historical head proposal.
    """
    if not any(lane.copy_draft is not None for lane in lanes):
        return (((),) * len(lanes), ("off",) * len(lanes))
    from .copy_draft import cohort_copy_cap

    rows = []
    decisions = []
    for lane, head_depth in zip(lanes, head_depths):
        state = lane.copy_draft
        if state is None:
            rows.append(())
            decisions.append("off")
            continue
        remaining = max(lane.max_tokens - lane.ntoks - 1, 0)
        cap = min(
            remaining,
            cohort_copy_cap(state.policy, lanes=len(lanes), head_depths=head_depths),
        )
        (span, decision) = state.plan(head_depth=head_depth, cap=cap)
        rows.append(tuple(int(token) for token in span))
        decisions.append(decision)
    return (tuple(rows), tuple(decisions))


def copy_draft_candidate_pending(
    batch: Union[BatchedSelfMTPState, SegmentedSelfMTPState],
) -> bool:
    """Whether any lane could propose a copy now (host-side peek)."""
    return any(
        lane.copy_draft is not None
        and max(lane.max_tokens - lane.ntoks - 1, 0) > 0
        and lane.copy_draft.has_candidate()
        for lane in batch.lanes
    )


def _propose_batched_self_mtp_impl(
    model: nn.Module, batch: BatchedSelfMTPState
) -> SelfMTPCycleResult:
    """Open one batched draft/verify transaction over the current membership."""
    with verify_sync_round():
        return _propose_batched_self_mtp_round(model, batch)


def advance_batched_self_mtp_zero(
    model: nn.Module, batch: Union[BatchedSelfMTPState, SegmentedSelfMTPState]
) -> SelfMTPCycleResult:
    """Advance an exact K=0 target round without proposal/rollback machinery.

    No draft head runs at depth zero.  The target hidden/token pair is retained
    in ``pending_hs``/``pending_ts`` so the next K>0 round teacher-forces the
    draft cache before proposing.  Segmented rows still publish one exact
    target delta through their live lineage; that ownership operation is not a
    speculative rollback transaction and cannot be dropped safely.
    """
    _require_healthy_self_mtp_batch(batch)
    if batch.proposal_open:
        raise RuntimeError("a self-MTP proposal is already open")
    if not batch.lanes:
        raise ValueError("cannot advance an empty self-MTP batch")
    depths = tuple(
        min(lane.num_draft, max(lane.max_tokens - lane.ntoks - 1, 0))
        for lane in batch.lanes
    )
    if any(depths):
        raise ValueError("self-MTP zero fast path requires depth zero for every lane")

    segmented = isinstance(batch, SegmentedSelfMTPState)
    branches = []
    true_batched = False
    if segmented:
        from .segmented_batch_cache import (
            SegmentedBatchUnsupported,
            build_segmented_batch_cache_pair,
        )
        from .segmented_self_mtp import (
            note_segmented_self_mtp,
            true_batched_segmented_self_mtp_enabled,
        )

        if not true_batched_segmented_self_mtp_enabled():
            # Reuse the existing transactional B1 proposal/commit path.  It
            # performs the same exact K=0 target step without locking or
            # reporting a physical width greater than one.
            raise ZeroDepthFastUnavailable(
                "true-batched segmented self-MTP is disabled"
            )

        if not len(batch.lanes) == len(batch.row_caches) == len(batch.transactions):
            raise RuntimeError("segmented self-MTP row ownership is misaligned")
        for lane, pair, transaction in zip(
            batch.lanes, batch.row_caches, batch.transactions
        ):
            transaction.validate(pair, lane, transaction.position)
        compute_caches = batch._segmented_caches
        if compute_caches is None:
            try:
                compute_caches = build_segmented_batch_cache_pair(
                    batch.row_caches,
                    note=note_segmented_self_mtp,
                    shared_qsa_prefix=batch.shared_qsa_prefix_id is not None,
                )
            except SegmentedBatchUnsupported as error:
                raise ZeroDepthFastUnavailable(
                    "segmented K=0 fast path requires a batched cache view"
                ) from error
        batch._segmented_caches = compute_caches
        target_caches = compute_caches.target
        for lane, transaction in zip(batch.lanes, batch.transactions):
            branches.append(
                transaction.fork(f"mtp-zero:{batch.membership_epoch}:{lane.uid}")
            )
        true_batched = True
        note_segmented_self_mtp("true_batched_requests")
    else:
        target_caches = batch.caches.target

    old_curs = tuple(lane.cur for lane in batch.lanes)
    old_seed_hs = tuple(lane.seed_h for lane in batch.lanes)
    inputs = mx.array([[token] for token in old_curs], mx.uint32)
    lengths = (1,) * len(batch.lanes)
    try:
        _prepare_self_mtp_cache_group(target_caches, lengths, (0,) * len(lengths))
        try:
            (logit_hidden, target_hidden) = _mtp_backbone(
                model, inputs, target_caches
            )
            batched_logits = model.logits(logit_hidden)
        finally:
            _finalize_self_mtp_cache_group(target_caches)

        outputs = []
        for row, lane in enumerate(batch.lanes):
            logits = batched_logits[row, 0]
            if lane.logits_processors:
                processor_tokens = mx.concatenate(
                    [lane.token_prefix, mx.array([lane.cur], mx.uint32)]
                )
                logits = _apply_logits_processors(
                    lane.logits_processors, processor_tokens, logits
                )
            logprobs = _lane_mtp_logprobs(lane, logits)
            bonus = _sample_from_logprobs(
                logprobs, lane.sampling_temp, rng=lane.rng
            )
            new_hidden = old_seed_hs[row]
            new_tokens = [old_curs[row]]
            if lane.pending_ts:
                new_hidden = mx.concatenate([lane.pending_hs, new_hidden], axis=1)
                new_tokens = lane.pending_ts + new_tokens
            lane.pending_hs = new_hidden
            lane.pending_ts = new_tokens
            lane.seed_h = target_hidden[row : row + 1, :1, :]
            lane.cur = bonus
            lane.token_prefix = mx.concatenate(
                [lane.token_prefix, mx.array([old_curs[row]], mx.uint32)]
            )
            lane.ntoks += 1
            lane.stats.cycles += 1
            lane.stats.draft_cycles += 1
            lane.stats.bonus_tokens += 1
            if lane.copy_draft is not None:
                lane.copy_draft.record(
                    copy_span=0,
                    head_depth=0,
                    accepted=0,
                    emitted=1,
                    committed=[int(bonus)],
                )
            outputs.append((MTPToken(bonus, logprobs, False),))

        if segmented:
            for row, (branch, transaction, pair, lane) in enumerate(
                zip(branches, batch.transactions, batch.row_caches, batch.lanes)
            ):
                batch.transactions[row] = transaction.publish(
                    branch,
                    pair,
                    lane,
                    1,
                    proposed=0,
                    accepted=0,
                )
            note_segmented_self_mtp("true_batched_engaged")
            note_segmented_self_mtp("batched_target_forwards")
            note_segmented_self_mtp("committed_cycles")
            note_segmented_self_mtp("zero_depth_fast_rounds")
    except BaseException as error:
        for branch in branches:
            try:
                branch.close()
            except BaseException:
                pass
        _poison_self_mtp_batch(batch, f"zero-depth target advance failed: {error}")
        if segmented:
            note_segmented_self_mtp("failures")
        raise

    return SelfMTPCycleResult(
        membership_epoch=batch.membership_epoch,
        lane_uids=tuple(lane.uid for lane in batch.lanes),
        draft_depths=depths,
        accepted_lengths=(0,) * len(batch.lanes),
        target_drops=(0,) * len(batch.lanes),
        head_drops=(0,) * len(batch.lanes),
        outputs=tuple(outputs),
        zero_fast_path=True,
        true_batched=true_batched,
    )


def _draft_confidence_features(probe, logprobs, hidden):
    from .mtp_confidence import draft_position_features

    return draft_position_features(logprobs, hidden, probe.projection)


def _stack_confidence_payload(draft_features, draft_tokens, probes):
    """Device arrays ``(features[rows, D, F], tokens[rows, D])`` for one eval."""
    widths = {
        int(probe.width) for probe in probes if probe is not None
    }
    if len(widths) != 1:
        raise ValueError("confidence probes in one cycle must share a feature width")
    width = widths.pop()
    depth = max((len(row) for row in draft_features), default=0)
    if depth == 0:
        return None
    feature_rows = []
    token_rows = []
    for features, tokens in zip(draft_features, draft_tokens):
        pad = depth - len(features)
        if features:
            stacked = mx.stack(features)
        else:
            stacked = mx.zeros((0, width), mx.float32)
        feature_rows.append(mx.pad(stacked, [(0, pad), (0, 0)]))
        ids = (
            mx.stack(tokens[: len(features)]).astype(mx.uint32)
            if features
            else mx.zeros((0,), mx.uint32)
        )
        token_rows.append(mx.pad(ids, [(0, pad)]))
    return (mx.stack(feature_rows), mx.stack(token_rows))


def _host_confidence_payload(payload, d_vector, probes, *, evaluated):
    if payload is None:
        return ((), ())
    if not evaluated:
        # Sampled cycles already synchronise per position; this is the one
        # extra read, and it happens only when a probe is attached.
        record_verify_sync("hybrid.confidence.features")
        mx.eval(*payload)
    (features, tokens) = (payload[0].tolist(), payload[1].tolist())
    host_features = []
    host_tokens = []
    for row, (depth, probe) in enumerate(zip(d_vector, probes)):
        if probe is None or depth == 0:
            host_features.append(())
            host_tokens.append(())
            continue
        host_features.append(
            tuple(tuple(float(v) for v in features[row][j]) for j in range(depth))
        )
        host_tokens.append(tuple(int(t) for t in tokens[row][:depth]))
    return (tuple(host_features), tuple(host_tokens))


def _propose_batched_self_mtp_round(
    model: nn.Module, batch: BatchedSelfMTPState
) -> SelfMTPCycleResult:
    if batch.proposal_open:
        raise RuntimeError("a self-MTP proposal is already open")
    if not batch.lanes:
        raise ValueError("cannot propose on an empty self-MTP batch")
    n_lanes = len(batch.lanes)
    lane_uids = tuple((lane.uid for lane in batch.lanes))
    if len(set(lane_uids)) != n_lanes:
        raise ValueError("self-MTP batch contains duplicate lane uid values")
    head_k_vector = tuple(
        (
            min(lane.num_draft, max(lane.max_tokens - lane.ntoks - 1, 0))
            for lane in batch.lanes
        )
    )
    copy_rows, copy_decisions = _plan_copy_drafts(batch.lanes, head_k_vector)
    if any(copy_rows):
        head_k_vector = tuple(
            0 if copy else k for (copy, k) in zip(copy_rows, head_k_vector)
        )
    k_vector = head_k_vector
    active_share_modes = {
        lane.share_qsa_indices for (lane, k) in zip(batch.lanes, k_vector) if k > 1
    }
    if len(active_share_modes) > 1:
        raise ValueError("mixed shared-QSA modes cannot share a draft cycle")
    drafts: List[List[int]] = [[] for _ in batch.lanes]
    draft_tokens: List[List[mx.array]] = [[] for _ in batch.lanes]
    draft_logprobs: List[List[mx.array]] = [[] for _ in batch.lanes]
    draft_h = [lane.seed_h for lane in batch.lanes]
    draft_steps = [0] * n_lanes
    greedy_cycle = all((lane.sampling_temp <= 0 for lane in batch.lanes))
    # Sampled (or mixed) cycles draft on device too unless a lane's logits
    # processors need host tokens: each ``_sample_from_logprobs(...).item()``
    # was a blocking sync per lane per depth (2026-09-23 audit). The hosted
    # ints are read once, below, before verification.
    device_rows = [
        (not greedy_cycle) and (not lane.logits_processors) for lane in batch.lanes
    ]
    probes = [getattr(lane, "confidence_probe", None) for lane in batch.lanes]
    probe_active = any(probe is not None for probe in probes)
    # Unverified lookahead drafts exist only to observe confidences past the
    # verify depth.  They are greedy-only (no RNG draws) and the draft cache
    # is trimmed of every draft step below, so outputs are unchanged.
    d_vector = tuple(
        k
        + (
            int(probe.lookahead)
            if probe is not None and greedy_cycle and k > 0
            else 0
        )
        for (k, probe) in zip(k_vector, probes)
    )
    draft_features: List[List[mx.array]] = [[] for _ in batch.lanes]
    max_k = max(d_vector)
    if max_k > 0:
        start_cycle = getattr(model, "mtp_start_cycle", None)
        if start_cycle is not None:
            share_qsa_this_cycle = bool(
                active_share_modes
                and next(iter(active_share_modes))
                and (len(set(d_vector)) == 1)
            )
            start_cycle(batch.caches.draft, share_qsa_this_cycle)
        try:
            first_lengths = [
                len(lane.pending_ts) + 1 if k > 0 else 0
                for (lane, k) in zip(batch.lanes, d_vector)
            ]
            width = max(first_lengths)
            hidden_rows = []
            token_rows = []
            for lane, valid in zip(batch.lanes, first_lengths):
                if valid:
                    if lane.pending_hs is None:
                        if lane.pending_ts:
                            raise RuntimeError(
                                "pending token list has no hidden tensor"
                            )
                        hs = lane.seed_h
                    else:
                        if lane.pending_hs.shape[1] != len(lane.pending_ts):
                            raise RuntimeError("pending hidden/token lengths disagree")
                        hs = mx.concatenate([lane.pending_hs, lane.seed_h], axis=1)
                    ts = mx.array([lane.pending_ts + [lane.cur]], mx.uint32)
                else:
                    hs = mx.zeros_like(lane.seed_h)
                    ts = mx.zeros((1, 1), mx.uint32)
                pad = width - valid if valid else width - 1
                hidden_rows.append(mx.pad(hs, [(0, 0), (0, pad), (0, 0)]))
                token_rows.append(mx.pad(ts, [(0, 0), (0, pad)]))
            right_padding = [width - valid for valid in first_lengths]
            _prepare_self_mtp_cache_group(
                batch.caches.draft, first_lengths, right_padding
            )
            try:
                (d_logits, post) = model.mtp_step(
                    mx.concatenate(hidden_rows),
                    mx.concatenate(token_rows),
                    batch.caches.draft,
                )
            finally:
                _finalize_self_mtp_cache_group(batch.caches.draft)
            for row, (lane, k, valid) in enumerate(
                zip(batch.lanes, d_vector, first_lengths)
            ):
                if k == 0:
                    continue
                pos = valid - 1
                draft_h[row] = post[row : row + 1, pos : pos + 1, :]
                lp = _lane_mtp_draft_logprobs(
                    lane, d_logits[row, pos], draft_tokens[row]
                )
                if probes[row] is not None:
                    draft_features[row].append(
                        _draft_confidence_features(probes[row], lp, draft_h[row])
                    )
                if greedy_cycle:
                    token = mx.argmax(lp).astype(mx.uint32)
                elif device_rows[row]:
                    token = _device_draft_token(lp, lane.sampling_temp, rng=lane.rng)
                    round_levers.bump("device_sampled_drafts")
                else:
                    hosted_token = _sample_from_logprobs(
                        lp, lane.sampling_temp, rng=lane.rng
                    )
                    token = mx.array(hosted_token, mx.uint32)
                    drafts[row].append(hosted_token)
                draft_tokens[row].append(token)
                draft_logprobs[row].append(lp)
                draft_steps[row] += 1
                lane.pending_hs = None
                lane.pending_ts = []
            if greedy_cycle or any(device_rows):
                mx.async_eval(
                    *(
                        row_tokens[-1]
                        for (row, row_tokens) in enumerate(draft_tokens)
                        if row_tokens and (greedy_cycle or device_rows[row])
                    ),
                    *(draft_h[row] for (row, k) in enumerate(k_vector) if k),
                )
            for depth in range(1, max_k):
                lengths = [1 if depth < k else 0 for k in d_vector]
                right_padding = [1 - length for length in lengths]
                hidden = mx.concatenate(draft_h)
                tokens = mx.concatenate(
                    [
                        mx.reshape(draft_tokens[row][-1], (1, 1))
                        if lengths[row]
                        else mx.zeros((1, 1), mx.uint32)
                        for row in range(n_lanes)
                    ],
                    axis=0,
                )
                _prepare_self_mtp_cache_group(
                    batch.caches.draft, lengths, right_padding
                )
                try:
                    (d_logits, post) = model.mtp_step(
                        hidden, tokens, batch.caches.draft
                    )
                finally:
                    _finalize_self_mtp_cache_group(batch.caches.draft)
                for row, (lane, active) in enumerate(zip(batch.lanes, lengths)):
                    if not active:
                        continue
                    draft_h[row] = post[row : row + 1, -1:, :]
                    lp = _lane_mtp_draft_logprobs(
                        lane, d_logits[row, -1], draft_tokens[row]
                    )
                    if probes[row] is not None:
                        draft_features[row].append(
                            _draft_confidence_features(probes[row], lp, draft_h[row])
                        )
                    if greedy_cycle:
                        token = mx.argmax(lp).astype(mx.uint32)
                    elif device_rows[row]:
                        token = _device_draft_token(
                            lp, lane.sampling_temp, rng=lane.rng
                        )
                        round_levers.bump("device_sampled_drafts")
                    else:
                        hosted_token = _sample_from_logprobs(
                            lp, lane.sampling_temp, rng=lane.rng
                        )
                        token = mx.array(hosted_token, mx.uint32)
                        drafts[row].append(hosted_token)
                    draft_tokens[row].append(token)
                    draft_logprobs[row].append(lp)
                    draft_steps[row] += 1
                if greedy_cycle or any(device_rows):
                    mx.async_eval(
                        *(
                            draft_tokens[row][-1]
                            for (row, active) in enumerate(lengths)
                            if active and (greedy_cycle or device_rows[row])
                        ),
                        *(
                            draft_h[row]
                            for (row, active) in enumerate(lengths)
                            if active
                        ),
                    )
        finally:
            if any(draft_steps):
                _trim_self_mtp_cache_group(
                    batch.caches.draft, draft_steps, validate=False
                )
            end_cycle = getattr(model, "mtp_end_cycle", None)
            if end_cycle is not None:
                end_cycle(batch.caches.draft)
        if tuple(draft_steps) != d_vector:
            raise RuntimeError(
                f"draft head advanced {tuple(draft_steps)}, expected {d_vector}"
            )
    device_drafted = [
        row for row in range(n_lanes) if device_rows[row] and draft_tokens[row]
    ]
    if device_drafted:
        stacked = [mx.stack(draft_tokens[row]) for row in device_drafted]
        record_verify_sync("hybrid.sampled.draft_boundary")
        mx.eval(stacked)
        for row, values in zip(device_drafted, stacked):
            drafts[row] = [int(value) for value in values.tolist()]
    feature_payload = None
    if probe_active:
        feature_payload = _stack_confidence_payload(
            draft_features, draft_tokens, probes
        )
    if d_vector != k_vector:
        # Drop lookahead drafts before verification; the verify transaction
        # sees exactly the depth the scheduler selected.
        draft_tokens = [row[:k] for (row, k) in zip(draft_tokens, k_vector)]
        draft_logprobs = [row[:k] for (row, k) in zip(draft_logprobs, k_vector)]
        drafts = [row[:k] for (row, k) in zip(drafts, k_vector)]
    if any(copy_rows):
        # Copied spans are point-mass proposals: host tokens, no draft law.
        # They join the same ragged verify rows as the head's drafts.
        for row, copy in enumerate(copy_rows):
            if copy:
                drafts[row] = list(copy)
                draft_tokens[row] = [mx.array(token, mx.uint32) for token in copy]
                draft_logprobs[row] = []
        k_vector = tuple(
            len(copy) if copy else k for (copy, k) in zip(copy_rows, head_k_vector)
        )
    valid_lengths = tuple((k + 1 for k in k_vector))
    width = max(valid_lengths)
    right_padding = tuple((width - valid for valid in valid_lengths))
    verify_rows = []
    for lane, row in zip(batch.lanes, draft_tokens):
        verify_rows.append(
            mx.concatenate(
                [mx.array([[lane.cur]], mx.uint32)]
                + [mx.reshape(token, (1, 1)) for token in row]
                + [mx.zeros((1, width - len(row) - 1), mx.uint32)],
                axis=1,
            )
        )
    verify_ids = mx.concatenate(verify_rows, axis=0)
    _prepare_self_mtp_cache_group(batch.caches.target, valid_lengths, right_padding)
    try:
        (vlogit_hidden, batched_hidden) = _mtp_backbone(
            model, verify_ids, batch.caches.target
        )
        batched_logits = model.logits(vlogit_hidden)
    finally:
        _finalize_self_mtp_cache_group(batch.caches.target)
    old_curs = tuple((lane.cur for lane in batch.lanes))
    old_seed_hs = tuple((lane.seed_h for lane in batch.lanes))
    lane_logprobs: List[mx.array] = []
    lane_hiddens: List[mx.array] = []
    # Copy rows under logits processors: each row's reachability, evaluated
    # at the acceptance boundary, and the inputs the real processors are
    # then run on through the reachable prefix.
    copy_guards: Dict[int, Tuple[mx.array, List[mx.array], List[mx.array]]] = {}
    for row, (lane, k, valid) in enumerate(zip(batch.lanes, k_vector, valid_lengths)):
        if lane.logits_processors and copy_rows[row]:
            # Head drafts were drawn from the processed law, but copied spans
            # are host tokens no processor has seen.  A row that follows a
            # copied token the processors forbid is never used by
            # verification, and its history is already outside a structured
            # output grammar, so asking the real (latching) processors about
            # it would fail a healthy lane.  Deciding that row by row took a
            # host sync per copied token, so every row is scored through
            # isolated probes instead, its legality stays on the device, and
            # rows past the first forbidden copied token keep the raw logits,
            # as on the external route.
            copied = copy_rows[row]
            histories = [
                mx.concatenate(
                    [lane.token_prefix, mx.array([lane.cur], mx.uint32)]
                    + [mx.reshape(token, (1,)) for token in draft_tokens[row][:pos]]
                )
                for pos in range(valid)
            ]
            raw = [batched_logits[row, pos] for pos in range(valid)]
            probed = [
                _probe_logits_processors(lane.logits_processors, history, value)
                for (history, value) in zip(histories, raw)
            ]
            legal = mx.stack(
                [
                    mx.logical_not(mx.isneginf(probed[pos][int(copied[pos])]))
                    for pos in range(k)
                ]
            )
            reach = mx.concatenate(
                [mx.array([True]), mx.cumprod(legal.astype(mx.int32)) > 0]
            )
            logits = mx.where(reach[:, None], mx.stack(probed), mx.stack(raw))
            copy_guards[row] = (reach, histories, raw)
        elif lane.logits_processors:
            processed = []
            for pos in range(valid):
                processor_tokens = mx.concatenate(
                    [lane.token_prefix, mx.array([lane.cur], mx.uint32)]
                    + [mx.reshape(token, (1,)) for token in draft_tokens[row][:pos]]
                )
                processed.append(
                    _apply_logits_processors(
                        lane.logits_processors,
                        processor_tokens,
                        batched_logits[row, pos],
                    )
                )
            logits = mx.stack(processed)
        else:
            logits = batched_logits[row, :valid]
        logprobs = _lane_mtp_logprobs(lane, logits)
        hidden = batched_hidden[row : row + 1, :valid, :]
        lane_logprobs.append(logprobs)
        lane_hiddens.append(hidden)
    greedy_targets = None
    if greedy_cycle:
        target_rows = []
        drafted_rows = []
        for row, (k, valid) in enumerate(zip(k_vector, valid_lengths)):
            target = mx.argmax(lane_logprobs[row], axis=-1).astype(mx.uint32)
            target_rows.append(mx.pad(target, [(0, width - valid)]))
            drafted = mx.stack(draft_tokens[row]) if k else mx.zeros((0,), mx.uint32)
            drafted_rows.append(mx.pad(drafted, [(0, width - k)]))
        accept_payload = mx.stack([mx.stack(target_rows), mx.stack(drafted_rows)])
        record_verify_sync("hybrid.greedy.accept_boundary")
        # Confidence features and copy-row reachability ride on the existing
        # accept boundary.
        mx.eval(
            accept_payload,
            *(() if feature_payload is None else feature_payload),
            *(guard[0] for guard in copy_guards.values()),
        )
        (greedy_targets, hosted_drafts) = accept_payload.tolist()
        drafts = [row[:k] for (row, k) in zip(hosted_drafts, k_vector)]
    accepted: List[int] = []
    relaxed_accepts: List[int] = []
    bonuses: List[int] = []
    output_rows: List[Tuple[MTPToken, ...]] = []
    for row, (lane, k) in enumerate(zip(batch.lanes, k_vector)):
        logprobs = lane_logprobs[row]
        relaxed = 0
        if k == 0:
            if greedy_cycle:
                n_accept = 0
                bonus = int(greedy_targets[row][0])
            else:
                n_accept = 0
                bonus = _sample_from_logprobs(
                    logprobs[0], lane.sampling_temp, rng=lane.rng
                )
        elif copy_rows[row] and lane.sampling_temp > 0:
            # Point-mass proposal: sample every verify row from the fully
            # transformed target law and accept while it equals the copy.
            # This is the exact speculative-sampling law for q = delta_d.
            from .copy_draft import verify_point_mass_by_sampling

            sampled = mx.random.categorical(logprobs, key=draw_key(lane.rng))
            record_verify_sync("hybrid.copy.sampled_eval")
            mx.eval(sampled, *(copy_guards[row][:1] if row in copy_guards else ()))
            record_verify_sync("hybrid.copy.sampled_tolist")
            (n_accept, bonus) = verify_point_mass_by_sampling(
                drafts[row], sampled.tolist()
            )
        elif lane.sampling_temp > 0:
            if lane.logprob_transform is not None:
                (n_accept, bonus) = _batched_residual_verify(
                    logprobs,
                    draft_logprobs[row],
                    drafts[row],
                    lane.sampling_temp,
                    rng=lane.rng,
                )
            elif lane.accept_rule == "block":
                (n_accept, bonus) = _block_verify(
                    logprobs,
                    draft_logprobs[row],
                    drafts[row],
                    lane.sampling_temp,
                    rng=lane.rng,
                )
            elif lane.accept_rule == "exact":
                sampled = mx.random.categorical(logprobs, key=draw_key(lane.rng))
                record_verify_sync("hybrid.exact.sampled_eval")
                mx.eval(sampled)
                record_verify_sync("hybrid.exact.sampled_tolist")
                sampled = sampled.tolist()
                n_accept = 0
                while n_accept < k and sampled[n_accept] == drafts[row][n_accept]:
                    n_accept += 1
                bonus = int(sampled[n_accept])
            else:
                n_accept = 0
                while n_accept < k and _accept_sampled_draft(
                    logprobs[n_accept],
                    draft_logprobs[row][n_accept],
                    drafts[row][n_accept],
                    rng=lane.rng,
                ):
                    n_accept += 1
                if n_accept < k:
                    bonus = _residual_sample(
                        logprobs[n_accept],
                        draft_logprobs[row][n_accept],
                        lane.sampling_temp,
                        rng=lane.rng,
                    )
                else:
                    bonus = _sample_from_logprobs(
                        logprobs[n_accept], lane.sampling_temp, rng=lane.rng
                    )
        else:
            if greedy_cycle:
                targets = greedy_targets[row]
            else:
                record_verify_sync("hybrid.greedy.targets_tolist")
                targets = mx.argmax(logprobs, axis=-1)
                mx.eval(targets, *(copy_guards[row][:1] if row in copy_guards else ()))
                targets = targets.tolist()
            fly = lane.fly_verification
            if (
                fly is not None
                and fly.enabled
                and not lane.logits_processors
                and k > 0
                and not copy_rows[row]
            ):
                from .speculative_sampling import apply_fly_relaxation

                target_laws = np.asarray(
                    mx.exp(logprobs[:k]).astype(mx.float32)
                )
                n_accept, relaxed = apply_fly_relaxation(
                    drafts[row],
                    target_laws,
                    [
                        targets[index] == drafts[row][index]
                        for index in range(k)
                    ],
                    fly,
                )
            else:
                n_accept = 0
                while n_accept < k and targets[n_accept] == drafts[row][n_accept]:
                    n_accept += 1
            bonus = int(targets[n_accept])
        accepted.append(n_accept)
        relaxed_accepts.append(relaxed)
        bonuses.append(bonus)
        output_rows.append(
            tuple(
                [
                    MTPToken(drafts[row][pos], logprobs[pos], True)
                    for pos in range(n_accept)
                ]
                + [MTPToken(bonus, logprobs[n_accept], False)]
            )
        )
    # The real processors see the copy rows they would have seen scored one
    # at a time: the reachable prefix, known now without another sync.
    for row, (reach, histories, raw) in copy_guards.items():
        lane = batch.lanes[row]
        for pos in range(sum(reach.tolist())):
            _apply_logits_processors(lane.logits_processors, histories[pos], raw[pos])
    target_drops = tuple((k - a for (k, a) in zip(k_vector, accepted)))
    _trim_self_mtp_cache_group(batch.caches.target, target_drops, validate=False)
    (host_features, host_feature_tokens) = _host_confidence_payload(
        feature_payload, d_vector, probes, evaluated=greedy_cycle
    )
    proposal = SelfMTPCycleResult(
        membership_epoch=batch.membership_epoch,
        lane_uids=lane_uids,
        draft_depths=k_vector,
        accepted_lengths=tuple(accepted),
        target_drops=target_drops,
        head_drops=head_k_vector,
        outputs=tuple(output_rows),
        _old_curs=old_curs,
        _old_seed_hs=old_seed_hs,
        _drafts=tuple((tuple(row) for row in drafts)),
        _vhidden=tuple(lane_hiddens),
        _logprobs=tuple(lane_logprobs),
        _bonuses=tuple(bonuses),
        relaxed_accepts=tuple(relaxed_accepts),
        copy_spans=tuple(len(copy) for copy in copy_rows),
        copy_decisions=copy_decisions,
        draft_features=host_features,
        draft_feature_tokens=host_feature_tokens,
    )
    batch.proposal_open = True
    batch._open_proposal = proposal
    return proposal


def propose_batched_self_mtp(
    model: nn.Module, batch: Union[BatchedSelfMTPState, SegmentedSelfMTPState]
) -> SelfMTPCycleResult:
    """Open a proposal, poisoning state when rollback cannot be proved."""
    if isinstance(batch, SegmentedSelfMTPState):
        return _propose_segmented_self_mtp(model, batch)
    _require_healthy_self_mtp_batch(batch)
    if batch.proposal_open:
        raise RuntimeError("a self-MTP proposal is already open")
    if not batch.lanes:
        raise ValueError("cannot propose on an empty self-MTP batch")
    try:
        return _propose_batched_self_mtp_impl(model, batch)
    except BaseException as error:
        batch.proposal_open = False
        batch._open_proposal = None
        _poison_self_mtp_batch(batch, f"proposal rollback unproved: {error}")
        raise


def _commit_segmented_self_mtp(
    batch: SegmentedSelfMTPState,
    proposal: SelfMTPCycleResult,
    *,
    emitted_counts: Sequence[int],
    terminal: Sequence[bool],
) -> None:
    from .segmented_self_mtp import (
        note_segmented_self_mtp,
        segmented_self_mtp_timing_enabled,
    )

    _require_healthy_self_mtp_batch(batch)
    if not batch.proposal_open or batch._open_proposal is not proposal:
        raise RuntimeError("commit requires the open segmented MTP proposal")
    if proposal.membership_epoch != batch.membership_epoch:
        raise RuntimeError("segmented MTP membership changed during a proposal")
    if proposal.lane_uids != tuple((lane.uid for lane in batch.lanes)):
        raise RuntimeError("segmented MTP lane order changed during a proposal")
    true_batched = batch._batched_state is not None
    if true_batched:
        if len(batch._transaction_branches) != len(batch.lanes):
            raise RuntimeError("segmented transaction ownership is misaligned")
    elif (
        not len(batch.lanes)
        == len(batch._row_states)
        == len(batch._row_proposals)
        == len(batch._transaction_branches)
    ):
        raise RuntimeError("segmented proposal ownership is misaligned")
    if len(emitted_counts) != len(batch.lanes) or len(terminal) != len(batch.lanes):
        raise ValueError("commit vectors must have one entry per segmented lane")
    timing = segmented_self_mtp_timing_enabled()
    started_ns = time.perf_counter_ns() if timing else 0
    try:
        if true_batched:
            commit_batched_self_mtp(
                batch._batched_state,
                proposal,
                emitted_counts=emitted_counts,
                terminal=terminal,
            )
            rows = zip(
                batch._transaction_branches, batch.transactions, batch.row_caches
            )
        else:
            rows = zip(
                batch._row_states,
                batch._row_proposals,
                batch._transaction_branches,
                batch.transactions,
                batch.row_caches,
            )
        for row, values in enumerate(rows):
            if true_batched:
                (branch, transaction, pair) = values
                row_lane = batch.lanes[row]
                proposed = proposal.draft_depths[row]
                accepted = proposal.accepted_lengths[row]
            else:
                (row_state, row_proposal, branch, transaction, pair) = values
                count = int(emitted_counts[row])
                is_terminal = bool(terminal[row])
                commit_batched_self_mtp(
                    row_state,
                    row_proposal,
                    emitted_counts=[count],
                    terminal=[is_terminal],
                )
                row_lane = row_state.lanes[0]
                proposed = row_proposal.draft_depths[0]
                accepted = row_proposal.accepted_lengths[0]
            count = int(emitted_counts[row])
            is_terminal = bool(terminal[row])
            batch.transactions[row] = transaction.publish(
                branch,
                pair,
                row_lane,
                count,
                proposed=proposed,
                accepted=accepted,
                zero_rollback_attested=count == 0 and is_terminal,
            )
    except BaseException as error:
        for branch in batch._transaction_branches:
            try:
                branch.close()
            except BaseException:
                pass
        batch.proposal_open = False
        batch._open_proposal = None
        batch._row_states.clear()
        batch._row_proposals.clear()
        batch._batched_state = None
        batch._segmented_caches = None
        batch._transaction_branches.clear()
        try:
            _restore_segmented_recovery(batch)
        except BaseException as recovery_error:
            _poison_self_mtp_batch(
                batch,
                f"segmented commit failed: {error}; recovery failed: {recovery_error}",
            )
        note_segmented_self_mtp("failures")
        raise
    batch.proposal_open = False
    batch._open_proposal = None
    batch._row_states.clear()
    batch._row_proposals.clear()
    batch._batched_state = None
    batch._transaction_branches.clear()
    batch._recovery_checkpoints.clear()
    note_segmented_self_mtp("committed_cycles")
    if timing:
        note_segmented_self_mtp("commit_ns", time.perf_counter_ns() - started_ns)


def commit_batched_self_mtp(
    batch: Union[BatchedSelfMTPState, SegmentedSelfMTPState],
    proposal: SelfMTPCycleResult,
    *,
    emitted_counts: Sequence[int],
    terminal: Sequence[bool],
) -> None:
    """Commit exactly the delivered prefix of one open proposal."""
    if isinstance(batch, SegmentedSelfMTPState):
        return _commit_segmented_self_mtp(
            batch, proposal, emitted_counts=emitted_counts, terminal=terminal
        )
    _require_healthy_self_mtp_batch(batch)
    if not batch.proposal_open or batch._open_proposal is not proposal:
        raise RuntimeError("commit requires the currently open self-MTP proposal")
    if proposal.membership_epoch != batch.membership_epoch:
        raise RuntimeError("self-MTP membership changed during an open proposal")
    if proposal.lane_uids != tuple((lane.uid for lane in batch.lanes)):
        raise RuntimeError("self-MTP lane order changed during an open proposal")
    n_lanes = len(batch.lanes)
    if len(emitted_counts) != n_lanes or len(terminal) != n_lanes:
        raise ValueError("commit vectors must have one entry per lane")
    emitted = tuple((int(value) for value in emitted_counts))
    terminal = tuple((bool(value) for value in terminal))
    delivery_drops = []
    for row, (count, is_terminal, outputs, accepted) in enumerate(
        zip(emitted, terminal, proposal.outputs, proposal.accepted_lengths)
    ):
        if count < 0 or count > len(outputs):
            raise ValueError(f"lane {row} emitted_count {count} is out of range")
        if not is_terminal and count != len(outputs):
            raise ValueError("a nonterminal lane must consume its entire proposal")
        if is_terminal and count < len(outputs) and (count > accepted):
            raise ValueError("a terminal prefix cannot skip part of the bonus token")
        delivery_drops.append(
            accepted - count + 1 if is_terminal and count <= accepted else 0
        )
    try:
        if any(delivery_drops):
            _trim_self_mtp_cache_group(
                batch.caches.target, delivery_drops, validate=False
            )
        for row, lane in enumerate(batch.lanes):
            accepted = proposal.accepted_lengths[row]
            count = emitted[row]
            old_cur = proposal._old_curs[row]
            old_seed_h = proposal._old_seed_hs[row]
            drafts = list(proposal._drafts[row])
            hidden = proposal._vhidden[row]
            consumed_accepted = min(count, accepted)
            if terminal[row] and count <= accepted:
                if count > 0:
                    # Copy-draft rounds run the head at depth 0 and leave
                    # their pairs pending, so pending can span several rounds.
                    # Extend it exactly as the nonterminal branch does; the
                    # detach replay must cover every undrafted position.
                    new_hs = mx.concatenate(
                        [old_seed_h, hidden[:, : count - 1, :]], axis=1
                    )
                    new_ts = [old_cur] + drafts[: count - 1]
                    if lane.pending_ts:
                        new_hs = mx.concatenate([lane.pending_hs, new_hs], axis=1)
                    lane.pending_hs = new_hs
                    lane.pending_ts = lane.pending_ts + new_ts
                    lane.seed_h = hidden[:, count - 1 : count, :]
                    lane.cur = proposal.outputs[row][count - 1].token
                    lane.token_prefix = mx.concatenate(
                        [
                            lane.token_prefix,
                            mx.array([old_cur] + drafts[: count - 1], mx.uint32),
                        ]
                    )
            else:
                new_hs = mx.concatenate([old_seed_h, hidden[:, :accepted, :]], axis=1)
                new_ts = [old_cur] + drafts[:accepted]
                if lane.pending_ts:
                    new_hs = mx.concatenate([lane.pending_hs, new_hs], axis=1)
                    new_ts = lane.pending_ts + new_ts
                lane.pending_hs = new_hs
                lane.pending_ts = new_ts
                lane.seed_h = hidden[:, accepted : accepted + 1, :]
                lane.cur = proposal._bonuses[row]
                lane.token_prefix = mx.concatenate(
                    [
                        lane.token_prefix,
                        mx.array([old_cur] + drafts[:accepted], mx.uint32),
                    ]
                )
            lane.ntoks += count
            lane.stats.cycles += 1
            copy_span = proposal.copy_spans[row] if proposal.copy_spans else 0
            if copy_span:
                lane.stats.retrieval_cycles += 1
                lane.stats.retrieval_proposed += copy_span
                lane.stats.retrieval_accepted += consumed_accepted
            else:
                lane.stats.draft_cycles += 1
                lane.stats.draft_proposed += proposal.draft_depths[row]
                lane.stats.draft_accepted += consumed_accepted
            # Per-round distributions, recorded for every verify round the
            # lane commits (copy rounds included). Both inputs are host ints
            # already in hand -- ``draft_depths`` and ``accepted_lengths`` are
            # plain tuples closed with the proposal -- so this is two dict
            # bumps: no device work, no eval, no allocation per round beyond
            # the at-most-(K+1) integer keys each histogram ever holds.
            span = int(proposal.draft_depths[row]) + 1
            lane.stats.verify_span_hist[span] = (
                lane.stats.verify_span_hist.get(span, 0) + 1
            )
            lane.stats.verify_accept_hist[consumed_accepted] = (
                lane.stats.verify_accept_hist.get(consumed_accepted, 0) + 1
            )
            if lane.copy_draft is not None:
                # Index exactly what this round committed after the previous
                # ``cur`` (already indexed): accepted proposal tokens, then
                # the bonus unless a terminal lane stopped inside them.
                committed = list(drafts[: min(count, accepted)])
                if count > accepted:
                    committed.append(proposal._bonuses[row])
                lane.copy_draft.record(
                    copy_span=copy_span,
                    head_depth=0 if copy_span else proposal.draft_depths[row],
                    accepted=accepted,
                    emitted=count,
                    committed=committed,
                )
            lane.relaxed_accepts += (
                proposal.relaxed_accepts[row] if proposal.relaxed_accepts else 0
            )
            if count > accepted:
                lane.stats.bonus_tokens += 1
    except BaseException as error:
        batch.proposal_open = False
        batch._open_proposal = None
        _poison_self_mtp_batch(batch, f"commit rollback unproved: {error}")
        raise
    batch.proposal_open = False
    batch._open_proposal = None


def abort_batched_self_mtp(
    batch: Union[BatchedSelfMTPState, SegmentedSelfMTPState],
    proposal: SelfMTPCycleResult,
    *,
    cause: Optional[BaseException] = None,
) -> None:
    """Close an interrupted proposal at its last committed boundary.

    Segmented MTP restores its request-private committed row checkpoints while
    retaining each lane's advanced random stream, so consumed random keys are
    never reused.  The older physically batched path has no independent row
    checkpoint and therefore retains the fail-closed poison contract.
    """
    _require_healthy_self_mtp_batch(batch)
    if not batch.proposal_open or batch._open_proposal is not proposal:
        raise RuntimeError("abort requires the currently open self-MTP proposal")
    if isinstance(batch, SegmentedSelfMTPState):
        from .segmented_self_mtp import note_segmented_self_mtp

        for branch in batch._transaction_branches:
            try:
                rejected = branch.close()
            except BaseException:
                rejected = None
            if rejected is not None:
                note_segmented_self_mtp("transaction_rejections")
        for row_state in batch._row_states:
            row_state.proposal_open = False
            row_state._open_proposal = None
        batch._row_states.clear()
        batch._row_proposals.clear()
        if batch._batched_state is not None:
            batch._batched_state.proposal_open = False
            batch._batched_state._open_proposal = None
        batch._batched_state = None
        batch._segmented_caches = None
        batch._transaction_branches.clear()
        batch.proposal_open = False
        batch._open_proposal = None
        try:
            _restore_segmented_recovery(batch)
        except BaseException as error:
            _poison_self_mtp_batch(
                batch, f"segmented proposal abort recovery failed: {error}"
            )
        return
    batch.proposal_open = False
    batch._open_proposal = None
    detail = "explicit proposal abort"
    if cause is not None:
        detail = f"proposal delivery aborted: {cause}"
    _poison_self_mtp_batch(batch, detail)


def detach_self_mtp_lanes(
    model: nn.Module,
    batch: Union[BatchedSelfMTPState, SegmentedSelfMTPState],
    indices: Sequence[int],
) -> Tuple[
    Union[BatchedSelfMTPState, SegmentedSelfMTPState], List[DetachedSelfMTPLane]
]:
    """Extract canonical rows before filtering the old batch membership."""
    _require_healthy_self_mtp_batch(batch)
    if batch.proposal_open:
        raise RuntimeError("cannot detach self-MTP lanes while a proposal is open")
    requested = [int(index) for index in indices]
    if len(set(requested)) != len(requested):
        raise ValueError("detach indices must be unique")
    if any((index < 0 or index >= len(batch.lanes) for index in requested)):
        raise IndexError("detach index is outside the self-MTP batch")
    if not requested:
        return (batch, [])
    if isinstance(batch, SegmentedSelfMTPState):
        from .segmented_self_mtp import (
            SegmentedLaneTransaction,
            note_segmented_self_mtp,
        )

        batch._segmented_caches = None
        batch.shared_qsa_prefix_id = None
        leaving = set(requested)
        keep = [index for index in range(len(batch.lanes)) if index not in leaving]
        detached = []
        successor_transactions = []
        try:
            for index in requested:
                lane = batch.lanes[index]
                pair = batch.row_caches[index]
                transaction = batch.transactions[index]
                position = transaction.position
                transaction.validate(pair, lane, position)
                _stop_all_speculation(pair.target)
                if lane.pending_hs is not None and lane.pending_ts:
                    model.mtp_step(
                        lane.pending_hs,
                        mx.array([lane.pending_ts], mx.uint32),
                        pair.draft,
                    )
                lane.pending_hs = None
                lane.pending_ts = []
                shared_materialized = False
                target = list(pair.target)
                for layer_index, cache in enumerate(target):
                    if getattr(cache, "supports_shared_qsa_suffix", False):
                        (target[layer_index], _) = cache.materialize_to_qsa()
                        shared_materialized = True
                if shared_materialized:
                    pair.target = target
                    successor = SegmentedLaneTransaction(pair, lane, position)
                    successor.predecessor_lineage_id = transaction.lineage.lineage_id
                    transaction.close()
                    note_segmented_self_mtp("transaction_canonicalizations")
                    transaction = successor
                else:
                    transaction = transaction.canonicalize_live_tip(
                        pair, lane, position
                    )
                successor_transactions.append(transaction)
                item = DetachedSelfMTPLane(
                    lane=lane, caches=pair, segment_transaction=transaction
                )
                _eval_self_mtp_lane_state(item)
                _validate_detached_self_mtp(item)
                detached.append(item)
        except BaseException as error:
            for transaction in successor_transactions:
                try:
                    transaction.close()
                except BaseException:
                    pass
            _poison_self_mtp_batch(batch, f"segmented detach failed: {error}")
            raise
        batch.lanes = [batch.lanes[index] for index in keep]
        batch.row_caches = [batch.row_caches[index] for index in keep]
        batch.transactions = [batch.transactions[index] for index in keep]
        batch.membership_epoch += 1
        return (batch, detached)
    leaving = set(requested)
    keep = [index for index in range(len(batch.lanes)) if index not in leaving]
    try:
        _stop_all_speculation(batch.caches.target)
    except BaseException as error:
        _poison_self_mtp_batch(batch, f"rollback stop failed: {error}")
        raise
    try:
        detached: List[DetachedSelfMTPLane] = []
        for index in requested:
            lane = copy.copy(batch.lanes[index])
            caches = _extract_self_mtp_cache_pair(batch.caches, [index])
            if lane.pending_hs is not None and lane.pending_ts:
                model.mtp_step(
                    lane.pending_hs,
                    mx.array([lane.pending_ts], mx.uint32),
                    caches.draft,
                )
            lane.pending_hs = None
            lane.pending_ts = []
            item = DetachedSelfMTPLane(lane=lane, caches=caches)
            _eval_self_mtp_lane_state(item)
            _validate_detached_self_mtp(item)
            detached.append(item)
        replacement = _extract_self_mtp_cache_pair(batch.caches, keep, batched=True)
        if keep:
            _start_speculation_or_cleanup(
                replacement.target,
                replacement.target,
                "batched self-MTP requires ragged-trimmable target caches",
            )
    except BaseException as error:
        _restart_live_self_mtp_or_poison(batch, error)
        raise
    batch.lanes = [batch.lanes[index] for index in keep]
    batch.caches = replacement
    batch.membership_epoch += 1
    return (batch, detached)
