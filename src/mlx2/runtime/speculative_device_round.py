"""Speculative round decided on the device, with one host read (opt-in).

Mechanisms adapted from MTPLX (https://github.com/youssofal/MTPLX,
Apache-2.0, revision 3890be556160af48887d5bfbf5304c4ad5b9417f:
``mtplx/models/inkling/fast/round.py`` decide/read_packet,
``mtplx/models/inkling/fast/round_program.py`` lazy draft->verify,
``mtplx/models/inkling/fast/nucleus.py`` kept set on the device); the code
here is a re-implementation over mlx2's generic external draft/verify engine
(``ExternalDraftBatchGenerator``).  See provenance/mtplx-device-round.json.

Every switch is default off; with ``device_round`` absent the engine never
calls into this module.  Switches (``device_round`` policy object):

``lazy_draft``
    Draft and verify in one queued graph: the drafter's tokens stay lazy on
    the device (``draft_lazy``), the verify input is built from them, and
    the host reads the tokens once, together with the verify result.  Same
    ops as ``draft_distributions`` and the chain verify, so greedy output is
    bitwise unchanged.
``decide``
    The accept/reject decision on the device, one small int32 packet read
    per lane (accepted count, finite flag, last emitted token, drafts).
    Greedy: the same per-row float32 argmax the host verifier compares
    against, so tokens are identical and the lane RNG advances exactly as
    ``verify_greedy_proposals`` advances it.  Sampled point-mass proposals:
    every target row drawn on the device (``mx.random.categorical`` under a
    key seeded by one lane-RNG draw per round), accepted while each draw
    matches its proposal: the same law as ``verify_point_mass_proposals``,
    different tokens for a given seed.
``sampled_drafts`` (needs ``lazy_draft`` and ``decide``)
    At temperature > 0, the drafter samples its tokens from its own tempered
    law q instead of taking the argmax, and the round accepts each with
    ``min(1, p/q)`` and resamples a rejection from ``normalized(p - q)+``
    (standard speculative sampling; the bonus from p).  Exact output law.
``nucleus`` (needs ``decide``)
    Requests with top-p / top-k / min-p keep the device decision: the
    filtered law is the same transform ordinary sampling uses
    (``make_transformed_logprobs``), applied on the device; a top-p-only
    request takes the certified top-512 window (``apply_top_p_topk512``,
    partition instead of a vocabulary sort, bitwise the dense kept set where
    its fit flag holds) with the fit flag in the packet and a dense redo on
    a miss.  Without this switch such requests keep the host verifier
    (counted decline).

``processors`` (needs ``lazy_draft`` or ``decide``; device_round v2 slice
    1a-2, design section 4.7)
    Lanes whose logits processors all declare the stateless branch contract
    (``speculative_topology.processors_branch_safe``: history-pure, no
    mutable state; e.g. the presence penalty that vendor profiles such as
    Raon's set by default) keep the device round.  The lane's own processor
    functions run on token arrays that hold the lazy drafts, on every verify
    row (``_target_law``'s per-row history) and, for a drafter that declares
    ``draft_lazy_processors``, at every drafting position, so the processed
    laws are those of the host path with no host read.  Without this switch
    (or for any other processor kind) such lanes keep the host path
    (counted ``logits_processors`` decline).

``prompt_copy_round`` (separate policy object, ``prompt_copy_round.py``)
    replaces a round's drafter proposals with a long copy from the prompt.

Anything a round cannot serve falls back to the existing path and is
counted under ``external_device_round_decline_reasons``.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

VERSION = "speculative-device-round-v1"
POLICY_KEY = "device_round"
_SWITCHES = ("lazy_draft", "decide", "sampled_drafts", "nucleus", "processors")
_SEED_SCALE = float(1 << 53)

# Present in ``scheduler_stats`` only when the option is configured.
COUNTERS = (
    "external_device_round_rounds",
    "external_device_round_lazy_draft_lanes",
    "external_device_round_host_reads",
    "external_device_round_decide_rounds",
    "external_device_round_decide_greedy_lanes",
    "external_device_round_decide_point_mass_lanes",
    "external_device_round_decide_ratio_lanes",
    "external_device_round_nucleus_lanes",
    "external_device_round_nucleus_window_misses",
    "external_device_round_declines",
)
DECLINE_REASONS = "external_device_round_decline_reasons"


@dataclass(frozen=True)
class DeviceRoundPolicy:
    lazy_draft: bool = False
    decide: bool = False
    sampled_drafts: bool = False
    nucleus: bool = False
    processors: bool = False

    @classmethod
    def from_value(cls, value):
        """``None``/``False``/``{}`` -> None (off); otherwise a validated policy."""
        if value is None or value is False:
            return None
        if not isinstance(value, dict):
            raise ValueError(f"{POLICY_KEY} must be an object of boolean switches")  # noqa: TRY004
        unknown = set(value) - set(_SWITCHES)
        if unknown:
            raise ValueError(f"unknown {POLICY_KEY} switches: {sorted(unknown)}")
        for key, flag in value.items():
            if type(flag) is not bool:
                raise ValueError(f"{POLICY_KEY}.{key} must be a boolean")
        policy = cls(**value)
        if policy.sampled_drafts and not (policy.lazy_draft and policy.decide):
            raise ValueError(f"{POLICY_KEY}.sampled_drafts requires lazy_draft and decide")
        if policy.nucleus and not policy.decide:
            raise ValueError(f"{POLICY_KEY}.nucleus requires decide")
        if policy.processors and not (policy.lazy_draft or policy.decide):
            raise ValueError(f"{POLICY_KEY}.processors requires lazy_draft or decide")
        if not any(getattr(policy, key) for key in _SWITCHES):
            return None
        return policy

    def as_dict(self):
        return {key: getattr(self, key) for key in _SWITCHES}


class HostRedo(Exception):  # noqa: N818 - control flow, not an error
    """A lazy draft asks for this round on the host path (restored first)."""


def _host_redo(tokens):
    from .drafters.dflash2 import DRAFT_LAZY_HOST_REDO

    return any(int(token) == DRAFT_LAZY_HOST_REDO for token in tokens)


def _bump(stats, key, amount=1):
    stats[key] = stats.get(key, 0) + amount


def _decline(owner, reason, lanes=1):
    _bump(owner.scheduler_stats, "external_device_round_declines", lanes)
    reasons = owner.scheduler_stats.setdefault(DECLINE_REASONS, {})
    reasons[reason] = reasons.get(reason, 0) + lanes
    return False


def _filters(lane):
    """Whether the lane's sampler restricts its law (top-p, top-k, min-p)."""
    top_p = float(lane.sampling.get("top_p", 0) or 0)
    top_k = int(lane.sampling.get("top_k", 0) or 0)
    min_p = float(lane.sampling.get("min_p", 0) or 0)
    return (0.0 < top_p < 1.0) or top_k > 0 or min_p != 0.0


def _temperature(lane):
    return float(lane.sampling.get("sampling_temp", 0))


class DeviceRound:
    """The engine-attached runtime of the ``device_round`` and
    ``prompt_copy_round`` options (``install``)."""

    def __init__(self, owner, policy, copy_policy):
        self.policy = policy
        self.copy = None
        if copy_policy is not None:
            from .prompt_copy_round import PromptCopyRuntime

            self.copy = PromptCopyRuntime(copy_policy)
        stats = owner.scheduler_stats
        # Installed whenever the runtime exists: a copy-only engine's runner
        # declines are counted here too.
        stats.update(dict.fromkeys(COUNTERS, 0))
        stats[DECLINE_REASONS] = {}
        if self.copy is not None:
            self.copy.install_counters(stats)

    # -- receipts ---------------------------------------------------------------
    def settings(self):
        return {
            "version": VERSION,
            "implemented": True,
            "qualified": False,
            "policy": None if self.policy is None else self.policy.as_dict(),
            "prompt_copy_round": None if self.copy is None else self.copy.policy.as_dict(),
        }

    def lane_receipt(self, lane):
        counts = getattr(lane, "device_round_counts", None)
        receipt = {**self.settings(), "lane": dict(counts or {})}
        if self.copy is not None:
            receipt["prompt_copy"] = self.copy.lane_receipt(lane)
        return {"device_round": receipt}

    @staticmethod
    def _lane_count(lane, key, amount=1):
        counts = getattr(lane, "device_round_counts", None)
        if counts is None:
            counts = lane.device_round_counts = {}
        counts[key] = counts.get(key, 0) + amount

    # -- eligibility --------------------------------------------------------------
    @staticmethod
    def _engine_reason(owner):
        """Engine-level reason the runner cannot serve a chain round, or None."""
        import os

        if owner.draft_topology == "tree15":
            return "tree_topology"
        if owner.continuation_policy is not None:
            return "continuation_pool"
        if owner.adaptive_policy is not None:
            return "adaptive_verification"
        if getattr(owner, "progressive_verification_tile", None) is not None:
            return "progressive_tiles"
        if (getattr(owner, "ast_execution_planner", None) is not None
                or getattr(owner, "ast_tree_tile_rows", 0)
                or getattr(owner, "ast_b1_tile_policy", None) is not None
                or getattr(owner, "ast_packed_pattern", None)):
            return "ast"
        if getattr(owner, "progressive_alternative_branches", 1) != 1:
            return "alternative_branches"
        if getattr(owner, "target_execution", "reference") != "reference":
            return "target_tree_provider"
        if getattr(owner, "dynamic_singleton_tree", False):
            return "dynamic_tree"
        if owner.fly_verification.enabled:
            return "fly_verification"
        if owner.pair_context_tokens:
            return "context_pairing"
        if getattr(owner, "predraft", False):
            return "predraft"
        if getattr(owner, "multilane_draft_cap", None) is not None:
            return "multilane_draft_cap"
        if getattr(owner, "draft_tail_equalize", False):
            return "draft_tail_equalize"
        if getattr(owner, "exact_verification", "token") != "token":
            return "block_verification"
        if getattr(owner, "pairwise_selection", "host") == "batched":
            return "pairwise_selection"
        if getattr(owner.draft, "requires_proposal_contexts", False):
            return "proposal_contexts"
        if getattr(owner.draft, "requires_target_caches", False):
            return "target_caches"
        if (getattr(owner.draft, "requires_proposal_score_states", False)
                and not getattr(owner.draft, "draft_lazy_composed", False)):
            # Source arbitration and its lagged score feedback live in the
            # host proposal path unless the wrapper drafts composed rows
            # lazily (ComposedDraftModel.draft_lazy, slice 1a-1b).
            return "proposal_composition"
        if getattr(getattr(owner.model, "model", None), "residual_taps", None) is not None:
            return "residual_steer"
        if os.environ.get("MLX2_EXTERNAL_VERIFY_PREBUILD", "0") == "1":
            return "verify_prebuild"
        return None

    def _processors_ok(self, lane):
        """``processors`` switch: every processor is stateless and history-pure."""
        from .speculative_topology import processors_branch_safe

        return bool(
            self.policy is not None and self.policy.processors
            and processors_branch_safe(lane.processors)
            # A verify ledger snapshots per-row state through VerifyWindow.
            and not any(callable(getattr(processor, "verify_ledger", None))
                        for processor in lane.processors)
        )

    def _lane_reason(self, lane):
        from .external_speculative import _speculation_parked

        if lane.cancelled:
            return "cancelled_lane"
        if lane.ordinary or _speculation_parked(lane):
            return "ordinary_lane"
        if lane.processors and not self._processors_ok(lane):
            return "logits_processors"
        if lane.anchor is None:
            return "no_anchor"
        return None

    # -- the runner ---------------------------------------------------------------
    def run(self, owner, cohort, forced_depth):
        """Serve one chain round here; False leaves it to ``_run_round``."""
        lazy = self.policy is not None and self.policy.lazy_draft
        if not lazy and self.copy is None:
            return False
        if not cohort:
            return False
        if forced_depth is not None:
            return _decline(owner, "forced_depth", len(cohort))
        reason = self._engine_reason(owner)
        if reason is None and getattr(owner.draft, "requires_proposal_score_states", False):
            if not lazy or self.copy is not None:
                # Prompt-copy rounds (and decide-only rounds) would bypass
                # composition's source arbitration; keep its host path.
                reason = "proposal_composition"
            elif int(getattr(owner.draft, "maximum_proposal_length", 0)) > int(
                    getattr(owner.draft, "backend_proposal_length", 0)):
                # ``prompt_lookup_max_span``: the host plans copies for the
                # engine depth, beyond the backend's; keep that path exact.
                reason = "composition_long_copies"
        if reason is None:
            for lane in cohort:
                reason = self._lane_reason(lane)
                if reason is not None:
                    break
        if reason is None and lazy and not callable(getattr(owner.draft, "draft_lazy", None)):
            reason = "drafter_without_lazy_draft"
        if reason is None and lazy:
            # Drafter-declared limits of its lazy draft (DFlash2: greedy, one lane).
            if getattr(owner.draft, "draft_lazy_greedy_only", False) and any(
                    _temperature(lane) > 0 for lane in cohort):
                reason = "lazy_draft_sampled_lane"
            elif getattr(owner.draft, "draft_lazy_width_one", False) and len(cohort) > 1:
                reason = "lazy_draft_width"
            elif any(lane.processors for lane in cohort) and not getattr(
                    owner.draft, "draft_lazy_processors", False):
                reason = "lazy_draft_processors"
        if reason is not None:
            return _decline(owner, reason, len(cohort))
        base = min(owner.num_draft, min(l.maximum - l.generated - 1 for l in cohort))
        if base < 1:
            return False  # a final-budget round: the zero-count chain body serves it
        copies = [None] * len(cohort)
        if self.copy is not None:
            cap = self._lane_exact_copy_cap(owner, cohort)
            outcomes = [self.copy.propose(owner, lane, cap=cap) for lane in cohort]
            copies = [outcome.tokens for outcome in outcomes]
            if not lazy and not all(copy is not None for copy in copies):
                if any(copy is not None for copy in copies):
                    # Discarded copies are not booked as probes or limits.
                    self.copy.count(owner, "mixed_cohort_without_lazy_draft",
                                    sum(copy is not None for copy in copies))
                return False
            for lane, outcome in zip(cohort, outcomes, strict=True):
                self.copy.book(owner, lane, outcome)
        draft_depth = min(base, int(getattr(owner.draft, "num_draft", base)))
        try:
            self._run(owner, cohort, copies, draft_depth)
        except HostRedo:
            # ``_run`` restored the lanes and statistics before re-raising.
            return _decline(owner, "host_redo", len(cohort))
        except Exception as error:
            from .proposal_composition import LazyDecline

            if not isinstance(error, LazyDecline):
                raise
            # A composed row selected another source (prompt-lookup copy,
            # native MTP, trusted PLD): restored, the host path proposes it.
            return _decline(owner, error.reason, len(cohort))
        return True

    @staticmethod
    def _composition_row(draft):
        """The composed row's attributes, as ``_propose`` sets them on its block."""
        sources = getattr(draft, "last_proposal_sources", None)
        if not sources or len(sources) != 1 or sources[0] != "external":
            raise RuntimeError("composed lazy draft published no external source row")
        return {
            "proposal_source": sources[0],
            "proposal_score_key": draft.last_proposal_score_keys[0],
            "proposal_score": float(draft.last_proposal_scores[0]),
            "proposal_trust_key": draft.last_proposal_trust_keys[0],
            "proposal_trusted": bool(draft.last_proposal_trusted[0]),
            "proposal_audit": bool(draft.last_proposal_audits[0]),
            "proposal_periodic_audit": bool(draft.last_proposal_periodic_audits[0]),
        }

    @staticmethod
    def _lane_exact_copy_cap(owner, cohort):
        """Widest copy that keeps the lane-exact cohort executor, or None.

        At B>1 with exact progressive fidelity and a row-exact target
        executor, ``_cohort_target_forward_rows`` runs lane-exact only up to
        16 verify rows; a wider copy would quietly fall back to the batched
        forward.  Copies are capped to 15 tokens there instead (counted as
        ``limit_lane_exact_width``)."""
        from .external_speculative import PROGRESSIVE_FIDELITY_EXACT

        if (len(cohort) >= 2
                and getattr(owner.model, "progressive_external_target_rows", None) is not None
                and getattr(owner, "progressive_verification_fidelity", None)
                == PROGRESSIVE_FIDELITY_EXACT):
            return 15
        return None

    def _run(self, owner, cohort, copies, draft_depth):
        from .external_speculative import HostDraftRow
        from .proposal_providers import PointMassLaw

        mx = owner.mx
        recovery = owner._snapshot_round(cohort)
        owner.scheduler_stats["recovery_checkpoint_captures"] += len(recovery)
        stats_snapshot = owner._snapshot_scheduler_stats()
        # A composed lazy draft books ``composition_stats`` before the read; a
        # restored round (redo, decline, failure) must not count it twice.
        composition_snapshot = (
            dict(owner.draft.composition_stats)
            if getattr(owner.draft, "draft_lazy_composed", False) else None
        )
        owner._open = True
        transaction = None
        # Whether this cohort's decision runs on the device (``_plan`` keeps
        # filtered sampled lanes on the host without ``nucleus``).  Sampled
        # drafts are only ever verified by the device ratio rule.
        device_decide = bool(
            self.policy is not None and self.policy.decide and not any(
                _temperature(lane) > 0 and _filters(lane) and not self.policy.nucleus
                for lane in cohort
            )
        )
        try:
            rows = []
            for lane, copy in zip(cohort, copies, strict=True):
                temp = _temperature(lane)
                if copy is not None:
                    # A copy round runs no drafter: commit the pending draft
                    # context now, as a zero-count round does.
                    if lane.tail.shape[1]:
                        owner._append_context(lane, lane.anchor)
                    rows.append(SimpleNamespace(
                        lane=lane, count=len(copy), host=list(copy),
                        drafts=mx.array(copy, dtype=mx.int32), q=None, source="prompt_copy",
                        seed=None, ratio=False,
                    ))
                    continue
                ratio = bool(device_decide and self.policy.sampled_drafts and temp > 0)
                seed = recorded = sampler = None
                if ratio:
                    seed = int(lane.rng.uniform() * _SEED_SCALE)
                    keys = mx.random.split(mx.random.key(seed), 2 * draft_depth + 2)
                    recorded = []

                    def sampler(logits, keys=keys, recorded=recorded, temp=temp):
                        scaled = logits.astype(mx.float32) * (1.0 / temp)
                        recorded.append(scaled)
                        return mx.random.categorical(
                            scaled, key=keys[len(recorded) - 1]
                        ).astype(mx.int32)

                try:
                    extra = ({"rng": lane.rng}
                             if getattr(owner.draft, "draft_lazy_consumes_rng", False) else {})
                    if lane.processors:
                        # Admitted by ``_lane_reason``/``run`` (``processors``).
                        extra.update(logits_processors=lane.processors,
                                     processor_history=list(lane.history))
                    composed = getattr(owner.draft, "draft_lazy_composed", False)
                    if composed:
                        # The executor-owned lane states ``_propose`` passes.
                        extra.update(processor_history=list(lane.history),
                                     proposal_score_state=lane.proposal_score_counts,
                                     proposal_trust_state=lane.proposal_trust_states,
                                     cohort_width=len(cohort))
                    drafts = owner.draft.draft_lazy(
                        lane.anchor, lane.tail, lane.draft_cache, draft_depth, sampler, **extra
                    )
                    composition = self._composition_row(owner.draft) if composed else None
                except Exception as error:
                    from .external_speculative import DraftUnavailable

                    if isinstance(error, DraftUnavailable):
                        error.failed_uids = (lane.uid,)
                    raise
                drafts = drafts.reshape(-1).astype(mx.int32)
                q = None
                if ratio:
                    q = mx.concatenate(recorded, axis=0)
                rows.append(SimpleNamespace(
                    # A composed backend may draft fewer than the round depth.
                    lane=lane, count=int(drafts.shape[0]), host=None, drafts=drafts, q=q,
                    composition=composition,
                    source="device_round", seed=seed, ratio=ratio, keys=None,
                ))
                _bump(owner.scheduler_stats, "external_device_round_lazy_draft_lanes")
                self._lane_count(lane, "lazy_draft_rounds")
            drafted = sum(row.source != "prompt_copy" for row in rows)
            owner.scheduler_stats["draft_max_width"] = max(
                owner.scheduler_stats["draft_max_width"], drafted
            )
            counts = [row.count for row in rows]
            width = max(counts) + 1
            inputs = mx.stack([
                mx.concatenate([
                    mx.array([int(row.lane.anchor)], dtype=mx.int32),
                    row.drafts,
                    mx.zeros((width - row.count - 1,), dtype=mx.int32),
                ])
                for row in rows
            ])
            target = owner._target_owner([lane.cache for lane in cohort])
            transaction = target.begin(lengths=[count + 1 for count in counts])
            logits, features = owner._cohort_target_forward(
                inputs, transaction, cohort, site="standard_verify"
            )
            plans = self._plan(owner, cohort, rows, logits) if device_decide else None
            if self.policy is not None and self.policy.decide and not device_decide:
                _decline(owner, "sampling_filters", len(cohort))
            if plans is not None:
                mx.eval(logits, features, *[plan.packet for plan in plans])
                _bump(owner.scheduler_stats, "external_device_round_host_reads")
                decisions = self._finish(owner, plans, logits)
                hosts = [plan.drafts_host for plan in plans]
            else:
                mx.eval(logits, features, *[row.drafts for row in rows])
                _bump(owner.scheduler_stats, "external_device_round_host_reads")
                hosts = [
                    row.host if row.host is not None
                    else [int(t) for t in row.drafts.tolist()]
                    for row in rows
                ]
                if any(_host_redo(tokens) for tokens in hosts):
                    raise HostRedo()
                vocab = int(logits.shape[-1])
                for tokens in hosts:
                    if any(not 0 <= token < vocab for token in tokens):
                        raise ValueError("device-round draft token outside the vocabulary")
                blocks = [
                    HostDraftRow(tokens, [PointMassLaw(t) for t in tokens], len(cohort))
                    for tokens in hosts
                ]
                decisions = owner._verify(cohort, blocks, logits)
            blocks = []
            for row, tokens in zip(rows, hosts, strict=True):
                laws = [None] * len(tokens) if row.ratio else [PointMassLaw(t) for t in tokens]
                block = HostDraftRow(tokens, laws, len(cohort))
                if row.source == "prompt_copy":
                    block.proposal_source = row.source
                for key, value in (getattr(row, "composition", None) or {}).items():
                    # ``_propose``'s composed-row attributes: ``_commit`` learns
                    # the lagged score and trust feedback from them.
                    setattr(block, key, value)
                blocks.append(block)
            # Book this round before ``_commit`` builds the round receipts, so
            # they include it (a failing commit restores lanes and stats).
            if self.copy is not None:
                for row, decision in zip(rows, decisions, strict=True):
                    if row.source == "prompt_copy":
                        self.copy.observe(
                            owner, row.lane, row.count, decision.accepted,
                            emitted=int(row.lane.generated) + len(decision.emitted),
                        )
            _bump(owner.scheduler_stats, "external_device_round_rounds")
            for lane in cohort:
                self._lane_count(lane, "rounds")
            owner._commit(
                cohort, decisions, features, blocks=blocks, transaction=transaction,
                deferred_context=False, deferred_context_uids=set(),
            )
        except BaseException:
            if transaction is not None and not transaction.closed:
                try:
                    transaction.abort()
                except BaseException:  # noqa: BLE001, S110 - snapshots restore below
                    pass
            owner._restore_round(cohort, recovery)
            owner.scheduler_stats = stats_snapshot
            if composition_snapshot is not None:
                owner.draft.composition_stats.clear()
                owner.draft.composition_stats.update(composition_snapshot)
            owner.scheduler_stats["recovery_checkpoint_restores"] += len(recovery)
            raise
        finally:
            owner._open = False

    # -- the decision -------------------------------------------------------------
    def verify(self, owner, cohort, blocks, logits):
        """``owner._verify`` replacement for the chain round (``decide``)."""
        from .external_speculative import HostDraftRow
        from .proposal_providers import PointMassLaw

        if self.policy is None or not self.policy.decide:
            return owner._verify(cohort, blocks, logits)
        if owner.fly_verification.enabled:
            _decline(owner, "fly_verification", len(cohort))
            return owner._verify(cohort, blocks, logits)
        if getattr(owner, "exact_verification", "token") != "token":
            _decline(owner, "block_verification", len(cohort))
            return owner._verify(cohort, blocks, logits)
        mx = owner.mx
        rows = []
        for lane, block in zip(cohort, blocks, strict=True):
            if lane.processors and not self._processors_ok(lane):
                _decline(owner, "logits_processors", len(cohort))
                return owner._verify(cohort, blocks, logits)
            if block is None:
                tokens = []
            elif isinstance(block, HostDraftRow) and all(
                isinstance(law, PointMassLaw) and law.token == int(token)
                for law, token in zip(block.laws, block.tokens, strict=True)
            ):
                tokens = [int(t) for t in block.tokens]
            else:
                _decline(owner, "non_point_mass_proposals", len(cohort))
                return owner._verify(cohort, blocks, logits)
            rows.append(SimpleNamespace(
                lane=lane, count=len(tokens), host=tokens,
                drafts=mx.array(tokens, dtype=mx.int32), q=None, ratio=False, seed=None,
            ))
        vocab = int(logits.shape[-1])
        for row in rows:
            if any(not 0 <= token < vocab for token in row.host):
                raise ValueError("invalid sparse point-mass proposal laws")
        plans = self._plan(owner, cohort, rows, logits)
        if plans is None:
            return owner._verify(cohort, blocks, logits)
        # One read for the cohort's packets (``host_reads`` counts the
        # runner's rounds only; this path has already read the verify rows).
        mx.eval(*[plan.packet for plan in plans])
        return self._finish(owner, plans, logits)

    def _plan(self, owner, cohort, rows, logits):
        """Lazy per-lane decisions, or None (counted) when a lane needs the host."""
        for lane in cohort:
            if _temperature(lane) > 0 and _filters(lane) and not self.policy.nucleus:
                _decline(owner, "sampling_filters", len(cohort))
                return None
        plans = []
        for index, row in enumerate(rows):
            plans.append(self._lane_plan(owner, row, logits[index, : row.count + 1]))
        _bump(owner.scheduler_stats, "external_device_round_decide_rounds")
        return plans

    def _transform(self, lane):
        from . import sample_utils

        temp = _temperature(lane)
        top_p = float(lane.sampling.get("top_p", 0) or 0)
        top_k = int(lane.sampling.get("top_k", 0) or 0)
        min_p = float(lane.sampling.get("min_p", 0) or 0)
        if (self.policy.nucleus and 0.0 < top_p < 1.0 and top_k == 0 and min_p == 0.0
                and not sample_utils.sampling_topk_cutoff_enabled()):
            import mlx.core as mx

            # The certified top-512 window (no vocabulary sort), forced on
            # for this request: bitwise the dense kept set where it fits.
            return sample_utils._make_transformed_logprobs(
                temp, top_p=top_p, min_p=0.0, top_k=0, min_tokens_to_keep=1,
                fused=False, fast_top_p=str(mx.default_device()),
            )
        return sample_utils.make_transformed_logprobs(
            temp, top_p=top_p, top_k=top_k, min_p=min_p
        )

    @staticmethod
    def _row_value(owner, row, rows_logits, j):
        """Verify row ``j`` as ``_target_law`` builds it: float32, then the
        lane's processors on ``history + [anchor] + drafts[:j]``, with the
        drafts still lazy on the device (``processors`` switch)."""
        mx = owner.mx
        value = rows_logits[j][None].astype(mx.float32)
        lane = row.lane
        if getattr(lane, "processors", None):
            prefix = getattr(row, "processor_prefix", None)
            if prefix is None:  # built once per lane-round, not per verify row
                prefix = row.processor_prefix = mx.array(
                    [*lane.history, int(lane.anchor)], dtype=mx.int32)
            tokens = mx.concatenate([prefix, row.drafts.reshape(-1)[:j].astype(mx.int32)])
            for processor in lane.processors:
                value = processor(tokens, value)
        return value

    def _lane_plan(self, owner, row, rows_logits, *, redo=False):
        mx = owner.mx
        lane, count = row.lane, row.count
        emit = lane.sampling.get("emit_logprobs", True)
        temp = _temperature(lane)
        int32 = mx.int32
        plan = SimpleNamespace(row=row, kind=None, response=None, fits=None, deferred=None)
        if temp == 0:
            plan.kind = "greedy"
            selected, norms, response = [], [], []
            for j in range(count + 1):
                value = self._row_value(owner, row, rows_logits, j)[0].astype(mx.float32)
                norm = mx.logsumexp(value)
                selected.append(mx.argmax(value).astype(int32))
                norms.append(norm)
                if emit:
                    response.append(value - norm)
            selected = mx.stack(selected)
            finite = mx.all(mx.isfinite(mx.stack(norms))).astype(int32)
            plan.response = response if emit else None
            accepted = self._leading(mx, row.drafts == selected[:count], count)
            last = mx.take(selected, accepted.reshape(1))
        else:
            plan.kind = "ratio" if row.ratio else "point_mass"
            transform = self._transform(lane)
            if _filters(lane):
                plan.nucleus = True
            from .sample_utils import transform_top_p_deferred

            logp, deferred = [], []
            for j in range(count + 1):
                value = self._row_value(owner, row, rows_logits, j)
                pending = transform_top_p_deferred(transform, value)
                deferred.append(pending)
                logp.append(pending.redo()[0] if redo and pending.fits is not None
                            else pending.value[0])
            fits = [p.fits.reshape(-1) for p in deferred if p.fits is not None]
            plan.deferred = deferred
            plan.fits = mx.concatenate(fits).astype(int32) if fits and not redo else None
            logp = mx.stack(logp)
            finite = mx.all(mx.isfinite(mx.logsumexp(logp, axis=-1))).astype(int32)
            plan.response = [logp[j] for j in range(count + 1)] if emit else None
            if row.seed is None:
                row.seed = int(lane.rng.uniform() * _SEED_SCALE)
            if plan.kind == "point_mass":
                keys = mx.random.split(mx.random.key(row.seed), count + 1)
                draws = mx.stack([
                    mx.random.categorical(logp[j], key=keys[j]).astype(int32)
                    for j in range(count + 1)
                ])
                accepted = self._leading(mx, row.drafts == draws[:count], count)
                last = mx.take(draws, accepted.reshape(1))
            else:
                # Keys [0, count) drew the drafts; the rest are this decision's.
                keys = mx.random.split(mx.random.key(row.seed), 2 * count + 2)
                logq = row.q
                width = int(logp.shape[-1])
                if int(logq.shape[-1]) > width:
                    raise ValueError("drafter law is wider than the target law")
                logq = logq - mx.logsumexp(logq, axis=-1, keepdims=True)
                if int(logq.shape[-1]) < width:
                    pad = mx.full((count, width - int(logq.shape[-1])), -mx.inf, mx.float32)
                    logq = mx.concatenate([logq, pad], axis=-1)
                idx = row.drafts.reshape(count, 1)
                lp = mx.take_along_axis(logp[:count], idx, axis=-1).reshape(-1)
                lq = mx.take_along_axis(logq, idx, axis=-1).reshape(-1)
                uniform = mx.random.uniform(shape=(count,), key=keys[count])
                accept = mx.log(uniform) < (lp - lq)
                accepted = self._leading(mx, accept, count)
                p = mx.exp(logp[:count])
                residual = mx.maximum(p - mx.exp(logq), 0.0)
                mass = mx.sum(residual, axis=-1, keepdims=True)
                log_residual = mx.where(mass > 0, mx.log(residual), logp[:count])
                corrections = [
                    mx.random.categorical(log_residual[j], key=keys[count + 1 + j]).astype(int32)
                    for j in range(count)
                ]
                bonus = mx.random.categorical(logp[count], key=keys[2 * count + 1]).astype(int32)
                outcomes = mx.stack([*corrections, bonus])
                last = mx.take(outcomes, accepted.reshape(1))
        parts = [accepted.reshape(1).astype(int32), finite.reshape(1), last.reshape(1),
                 row.drafts.reshape(-1).astype(int32)]
        if plan.fits is not None:
            parts.append(plan.fits)
        plan.packet = mx.concatenate(parts)
        return plan

    @staticmethod
    def _leading(mx, matches, count):
        if count == 0:
            return mx.array(0, dtype=mx.int32)
        return mx.sum(mx.cumprod(matches.astype(mx.int32))).astype(mx.int32)

    def _finish(self, owner, plans, logits):
        """Host walk of the packets: RNG schedule, stops, ``RoundDecision``s."""
        from .external_speculative import LaneFailure, RoundDecision
        from .processor_probe import VerifyWindow
        from .sample_utils import settle_top_p_deferred

        mx = owner.mx
        decisions = []
        for index, plan in enumerate(plans):
            row, lane = plan.row, plan.row.lane
            packet = [int(v) for v in np.asarray(plan.packet).tolist()]
            if plan.fits is not None:
                fits = packet[3 + row.count:]
                if not settle_top_p_deferred(fits):
                    _bump(owner.scheduler_stats, "external_device_round_nucleus_window_misses")
                    plan = plans[index] = self._lane_plan(
                        owner, row, logits[index, : row.count + 1], redo=True
                    )
                    mx.eval(plan.packet)
                    _bump(owner.scheduler_stats, "external_device_round_host_reads")
                    packet = [int(v) for v in np.asarray(plan.packet).tolist()]
            accepted, finite, last = packet[0], packet[1], packet[2]
            drafts = packet[3: 3 + row.count]
            if _host_redo(drafts):
                raise HostRedo()
            vocab = int(logits.shape[-1])
            # Drafts first: the host path refuses an invalid draft row (a
            # DFlash2 lazy draft's -1 tokens) before it ever verifies.
            if any(not 0 <= token < vocab for token in drafts):
                raise ValueError("device-round token outside the vocabulary")
            if not finite:
                raise LaneFailure(
                    lane.uid, "external draft target law is not a probability distribution"
                )
            if not 0 <= last < vocab:
                raise ValueError("device-round token outside the vocabulary")
            if row.host is not None and drafts != row.host:
                raise RuntimeError("device-round packet drafts differ from the host proposals")
            plan.drafts_host = drafts
            if plan.kind == "greedy":
                # ``verify_greedy_proposals``' RNG schedule, draw for draw.
                for _ in range(accepted + 2 if accepted < row.count else row.count + 1):
                    lane.rng.uniform()
                _bump(owner.scheduler_stats, "external_device_round_decide_greedy_lanes")
                self._lane_count(lane, "decide_greedy_rounds")
            elif plan.kind == "point_mass":
                _bump(owner.scheduler_stats, "external_device_round_decide_point_mass_lanes")
                self._lane_count(lane, "decide_point_mass_rounds")
            else:
                _bump(owner.scheduler_stats, "external_device_round_decide_ratio_lanes")
                self._lane_count(lane, "decide_ratio_rounds")
            if getattr(plan, "nucleus", False):
                _bump(owner.scheduler_stats, "external_device_round_nucleus_lanes")
                self._lane_count(lane, "nucleus_rounds")
            emitted = list(drafts[:accepted]) + [last]
            for j, token in enumerate(emitted):
                if token in owner.stops:
                    emitted = emitted[: j + 1]
                    break
            window = VerifyWindow(lane.processors)
            for _ in range(row.count + 1):
                window.mark()
            decisions.append(RoundDecision(
                accepted, emitted, None, 0, plan.response, window,
            ))
        return decisions


def install(owner, device_round, prompt_copy_round):
    """Attach the runtime to ``owner`` when either option is configured."""
    from .prompt_copy_round import PromptCopyPolicy

    policy = DeviceRoundPolicy.from_value(device_round)
    copy_policy = PromptCopyPolicy.from_value(prompt_copy_round)
    if policy is None and copy_policy is None:
        owner.device_round = None
        return None
    owner.device_round = DeviceRound(owner, policy, copy_policy)
    return owner.device_round


__all__ = ["COUNTERS", "DECLINE_REASONS", "VERSION", "DeviceRound", "DeviceRoundPolicy", "install"]
