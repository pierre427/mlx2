"""External draft/verify execution composed with mlx2 admission and APCv2.

Original implementation. Target verification uses per-lane segmented cache
transactions; proposal q is supplied by the bound external draft model.
No capability is qualified by importing or constructing this executor.
"""
from __future__ import annotations

import copy
import json
from collections import deque
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np

from .committed_recovery import CommittedRecoverySlot
from .cow_cache import snapshot_prompt_cache_descriptors
from .speculative_sampling import (
    FLyVerificationPolicy,
    RequestRNG,
    probability,
    verify_proposals,
)

_COUNTER_MAX = (1 << 63) - 1


def _bump(stats, key, amount=1):
    stats[key] = min(_COUNTER_MAX, int(stats.get(key, 0)) + int(amount))


class DraftUnavailable(RuntimeError):
    """A recoverable drafter-only failure; target ordinary path remains valid."""


@dataclass
class ExternalDraftState:
    state: object
    covered_tokens: int
    rng_key: object = None
    rng_draws: int = 0
    kind: str = "external_draft_v1"
    binding: str = ""

    @property
    def nbytes(self):
        return sum(int(getattr(c, "nbytes", 0)) for c in self.state[0]) + int(getattr(self.state[1], "nbytes", 0)) + int(getattr(self.rng_key, "nbytes", 0))

    def validate(self, binding, covered):
        if self.kind != "external_draft_v1" or self.binding != binding or self.covered_tokens != covered:
            raise ValueError("External draft sidecar revision/boundary mismatch")
        length = 0 if self.state[1] is None else int(self.state[1].shape[1])
        if any(int(c.offset) + length != covered for c in self.state[0]):
            raise ValueError("External draft context is not paired to target boundary")


@dataclass
class Lane:
    uid: int
    history: list
    remaining: deque
    cache: list
    draft_cache: list
    tail: object
    rng: RequestRNG
    maximum: int
    processors: list
    sampling: dict
    generated: int = 0
    anchor: int | None = None
    ready: deque = field(default_factory=deque)
    cancelled: bool = False
    ordinary: bool = False
    external_rounds: int = 0
    proposed: int = 0
    accepted: int = 0
    target_max_width: int = 0
    draft_max_width: int = 0
    relaxed_accepts: int = 0


class ExternalDraftBatchGenerator:
    """Same request-lifecycle seam as BatchGenerator, distinct external protocol.

    The target trunk is evaluated at cohort width B. Draft trunk forwards
    group lanes by pending-context length; attention remains per-row SDPA.
    Receipts report actual target and draft projection widths. No dense cache repacking or shared global random generator.
    """
    def __init__(self, model, *, draft_model, binding, completion_batch_size=4,
                 prefill_step_size=2048, num_draft=4, stop_tokens=(), memory_headroom=None,
                 reclaim_memory=None, evict_checkpoint=None, fly_verification=None,
                 **kwargs):
        import mlx.core as mx
        self.mx = mx; self.model = model; self.draft = draft_model
        self.memory_headroom = memory_headroom
        self.reclaim_memory = reclaim_memory; self.evict_checkpoint = evict_checkpoint
        self._schedule_cursor = 0; self._reclaims_left = 0; self._allocator_reclaimed = False
        self.binding = binding; self.capacity = completion_batch_size
        self.fly_verification = FLyVerificationPolicy.from_value(fly_verification)
        self.prefill_step = prefill_step_size; self.num_draft = int(num_draft)
        if not 1 <= self.num_draft < draft_model.config.block_size:
            raise ValueError("External draft count must fit trained block")
        if any(len(t) != 1 for t in stop_tokens):
            raise ValueError("External executor currently requires single-token stops")
        self.stops = {int(t[0]) for t in stop_tokens}
        self.layers = tuple(draft_model.config.target_layer_ids)
        self.lanes = {}; self.next_uid = 0; self.boundaries = {}
        self.scheduler_stats = {"external_rounds": 0, "accepted_proposals": 0, "proposed_tokens": 0, "ordinary_rounds": 0, "cancelled": 0, "target_max_width": 0, "draft_max_width": 1, "prefill_rounds": 0, "paired_cache_resumes": 0, "segmented_transactions": 0, "segmented_rollbacks": 0, "draft_fallbacks": 0, "recovery_checkpoint_captures": 0, "recovery_checkpoint_restores": 0, "external_draft_masked_positions": 0, "external_ordinary_fast_path_rounds": 0, "external_ordinary_fast_path_lanes": 0, "external_draft_context_skipped": 0, "external_taps_skipped": 0, "external_transactions_skipped": 0, "fly_relaxed_accepts": 0}
        self._open = False

    def _empty_hidden(self):
        return self.mx.zeros((1, 0, len(self.layers)*self.model.args.hidden_size))

    @staticmethod
    def _snapshot_draft_state(draft_cache, tail):
        draft_cache, tail, _ = snapshot_prompt_cache_descriptors(
            draft_cache, tail
        )
        return draft_cache, tail

    def insert(self, prompts, *, max_tokens, caches=None, all_tokens=None,
               cache_states=None, lane_rngs=None, sampling_configs=None,
               logits_processors=None, samplers=None, resume_rng=False,
               apc_interior_positions=None, prefill_inputs=None):
        if self._open:
            raise RuntimeError("Membership change during external transaction")
        apc_interior_positions = (
            [()] * len(prompts)
            if apc_interior_positions is None
            else apc_interior_positions
        )
        prefill_inputs = (
            [None] * len(prompts) if prefill_inputs is None else prefill_inputs
        )
        if len(apc_interior_positions) != len(prompts):
            raise ValueError(
                "apc_interior_positions must have one entry per external request"
            )
        if len(prefill_inputs) != len(prompts):
            raise ValueError("prefill_inputs must have one entry per external request")
        if any(positions for positions in apc_interior_positions):
            raise ValueError(
                "APCv2 interior checkpoints are unavailable on external draft routes"
            )
        if any(value is not None for value in prefill_inputs):
            raise ValueError("multimodal prefill is unavailable on external draft routes")
        uids = []
        for i, prompt in enumerate(prompts):
            prefix = list((all_tokens or [[]]*len(prompts))[i]); todo = list(prompt)
            if not todo or max_tokens[i] <= 0:
                raise ValueError("External request needs prompt tokens and positive budget")
            state = (cache_states or [None]*len(prompts))[i]
            target = (caches or [None]*len(prompts))[i]
            seed_key = (lane_rngs or [None]*len(prompts))[i]
            seed = np.asarray(seed_key.key).tolist() if seed_key is not None else self.next_uid
            rng = RequestRNG(seed)
            draft_cache = self.draft.make_cache(); tail = self._empty_hidden()
            if state is not None:
                state.validate(self.binding, len(prefix))
                if target is None or any(int(c.offset) != len(prefix) for c in target):
                    raise ValueError("External target cache boundary mismatch")
                target, state, _ = snapshot_prompt_cache_descriptors(
                    target, state
                )
                state.validate(self.binding, len(prefix))
                draft_cache, tail = state.state
                self.scheduler_stats["paired_cache_resumes"] += 1
                if resume_rng and state.rng_key is not None:
                    rng = RequestRNG(state=json.loads(bytes(np.asarray(state.rng_key, dtype=np.uint8)).decode()))
            elif prefix:
                # An ordinary APC hit cannot fabricate draft context. Rebuild
                # both planes from transcript instead of pairing stale caches.
                todo = prefix + todo; prefix = []; target = None
            uid = self.next_uid; self.next_uid += 1
            lane = Lane(uid, prefix, deque(todo), list(target) if target is not None else self.model.make_cache(), draft_cache, tail, rng, int(max_tokens[i]), (logits_processors or [[]]*len(prompts))[i] or [], dict((sampling_configs or [{}]*len(prompts))[i]))
            self.lanes[uid] = lane; uids.append(uid)
        return uids

    def _sidecar(self, lane):
        draft_cache, tail = self._snapshot_draft_state(
            lane.draft_cache, lane.tail
        )
        state = ExternalDraftState((draft_cache, tail), len(lane.history), self.mx.array(list(json.dumps(lane.rng.snapshot(), sort_keys=True).encode()), dtype=self.mx.uint8), lane.rng.draws, binding=self.binding)
        state.validate(self.binding, len(lane.history)); return state

    def _prefill(self, lane):
        if len(lane.remaining) > 1:
            n = min(self.prefill_step, len(lane.remaining)-1)
            inputs = [lane.remaining.popleft() for _ in range(n)]
            if lane.tail.shape[1]:
                self.draft.append_context(lane.tail, lane.draft_cache)
            lane.tail = self.model.prefill_body(self.mx.array([inputs]), lane.cache, self.layers)
            lane.history.extend(inputs)
            self.mx.eval(lane.tail, [c.state for c in lane.cache], [c.state for c in lane.draft_cache if c.offset])
            self.scheduler_stats["prefill_rounds"] += 1
        done = len(lane.remaining) == 1
        if done:
            lane.anchor = lane.remaining.popleft()
            if lane.history:
                if lane.tail.shape[1]:
                    self.draft.append_context(lane.tail, lane.draft_cache)
                    lane.tail = lane.tail[:, :0]
                self.boundaries[lane.uid] = {"committed_only": True, "tokens": list(lane.history), "target_cache": copy.deepcopy(lane.cache), "covered_tokens": len(lane.history), "cache_sidecar": self._sidecar(lane)}
        return SimpleNamespace(uid=lane.uid, end_of_prompt=done)

    def _target_law(self, lane, logits, history, reachable=True):
        from .sample_utils import make_transformed_logprobs
        value = logits[None]
        tokens = self.mx.array(history, dtype=self.mx.int32)
        # A row that follows a draft token the target gives zero probability is
        # never used by verification.  Its history can already be outside a
        # structured-output grammar, and asking the processor about it would
        # latch a dead-end failure on a healthy lane (GPU: every structured
        # request on the DFlash2 route returned 502).
        for processor in (lane.processors if reachable else ()):
            value = processor(tokens, value)
        temp = float(lane.sampling.get("sampling_temp", 0))
        if temp == 0:
            p = np.zeros(value.shape[-1]); p[int(self.mx.argmax(value).item())] = 1; return p
        transform = make_transformed_logprobs(temp, top_p=lane.sampling.get("top_p", 0), top_k=lane.sampling.get("top_k", 0), min_p=lane.sampling.get("min_p", 0))
        return probability(np.asarray(self.mx.exp(transform(value)[0])))

    def _verification_receipt(self, lane):
        if self.fly_verification.enabled and not lane.processors:
            return {
                "verification": "fly",
                "parameters": self.fly_verification.as_dict(),
                "relaxed_accepts": lane.relaxed_accepts,
            }
        result = {
            "verification": "exact",
            "relaxed_accepts": lane.relaxed_accepts,
        }
        if self.fly_verification.enabled and lane.processors:
            result["fly_disabled"] = "logits_processors"
        return result

    def _round(self, cohort):
        from .segmented_rotating_kv import SegmentedKVRows
        if cohort and all(lane.ordinary for lane in cohort):
            return self._ordinary_round(cohort)
        recovery = []
        for lane in cohort:
            slot = CommittedRecoverySlot()
            slot.capture(
                route="external_dflash2",
                revision=self.binding,
                boundary=len(lane.history),
                value=lane.__dict__,
                snapshot=copy.deepcopy,
                restore=copy.deepcopy,
            )
            recovery.append((slot, len(lane.history)))
        self.scheduler_stats["recovery_checkpoint_captures"] += len(recovery)
        stats_snapshot = dict(self.scheduler_stats)
        self._open = True
        transaction = None
        try:
            requested_count = min(self.num_draft, min(l.maximum-l.generated-1 for l in cohort))
            if any(l.ordinary for l in cohort): requested_count = 0
            drafts, laws = [None]*len(cohort), [None]*len(cohort)
            draft_widths = [1]*len(cohort)
            if requested_count:
                groups = {}
                for row,lane in enumerate(cohort):
                    groups.setdefault((int(lane.tail.shape[1]),str(lane.tail.dtype)),[]).append(row)
                for indices in groups.values():
                    lanes = [cohort[i] for i in indices]
                    processors = [lane.processors for lane in lanes]
                    arguments = (
                        [l.anchor for l in lanes],
                        self.mx.concatenate([l.tail for l in lanes],axis=0),
                        self.draft.batch_caches([l.draft_cache for l in lanes]),
                        requested_count,
                        [l.rng for l in lanes],
                        [float(l.sampling.get("sampling_temp",0)) for l in lanes],
                    )
                    try:
                        import inspect

                        supports_processors = bool(
                            getattr(
                                self.draft,
                                "supports_logits_processors",
                                False,
                            )
                        )
                        try:
                            parameters = inspect.signature(
                                self.draft.draft_distributions
                            ).parameters
                            supports_processors = supports_processors or (
                                "logits_processors" in parameters
                                or any(
                                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                                    for parameter in parameters.values()
                                )
                            )
                        except (TypeError, ValueError):
                            pass
                        if any(processors) and supports_processors:
                            tokens,q = self.draft.draft_distributions(
                                *arguments,
                                logits_processors=processors,
                                processor_histories=[list(l.history) for l in lanes],
                            )
                        else:
                            # Preserve the unstructured fast path byte-for-byte:
                            # no processor lists, prefixes or keyword handling.
                            tokens,q = self.draft.draft_distributions(*arguments)
                    except DraftUnavailable as error:
                        failed_rows = getattr(error, "failed_rows", None)
                        if failed_rows is None:
                            error.failed_uids = tuple(lane.uid for lane in lanes)
                        else:
                            error.failed_uids = tuple(
                                lanes[int(local_row)].uid for local_row in failed_rows
                            )
                        raise
                    _bump(
                        self.scheduler_stats,
                        "external_draft_masked_positions",
                        requested_count * sum(bool(value) for value in processors),
                    )
                    self.scheduler_stats["draft_max_width"] = max(self.scheduler_stats["draft_max_width"],len(lanes))
                    for j,row in enumerate(indices):
                        drafts[row],laws[row] = tokens[j],q[j]
                        draft_widths[row] = len(lanes)
            else:
                for row,lane in enumerate(cohort):
                    if lane.tail.shape[1]: self.draft.append_context(lane.tail,lane.draft_cache)
                    drafts[row],laws[row] = [],[]
            proposal_counts = [len(row) for row in drafts]
            verify_width = max(proposal_counts, default=0) + 1
            owner = SegmentedKVRows([l.cache for l in cohort])
            transaction = owner.begin(lengths=[count+1 for count in proposal_counts])
            inputs = [
                [lane.anchor]+draft+[0]*(verify_width-len(draft)-1)
                for lane,draft in zip(cohort,drafts)
            ]
            logits, features = self.model.forward_with_taps(self.mx.array(inputs), transaction.caches, self.layers)
            self.mx.eval(logits, features)
            results = []
            for row, lane in enumerate(cohort):
                targets, reachable = [], True
                count = proposal_counts[row]
                for j in range(count+1):
                    targets.append(self._target_law(lane, logits[row,j], lane.history + inputs[row][:j+1], reachable))
                    if reachable and j < count and lane.processors and targets[-1][int(drafts[row][j])] <= 0:
                        reachable = False
                if self.fly_verification.enabled and not lane.processors:
                    result = verify_proposals(
                        drafts[row],
                        laws[row],
                        targets,
                        lane.rng,
                        fly_verification=self.fly_verification,
                    )
                else:
                    # Preserve exact/default-off verification, including its
                    # RNG draw schedule and existing call seam.
                    result = verify_proposals(
                        drafts[row], laws[row], targets, lane.rng
                    )
                # Stop/length truncation is part of the same transaction.
                emitted = list(result.emitted)
                for j,t in enumerate(emitted):
                    if t in self.stops:
                        emitted = emitted[:j+1]; break
                consumed = min(result.accepted+1, len(emitted))
                results.append((result, emitted, consumed))
            rows = transaction.commit(accepted_lengths=[r[2] for r in results]); transaction = None
            self.scheduler_stats["segmented_transactions"] += len(results)
            self.scheduler_stats["segmented_rollbacks"] += sum(
                int(consumed < count + 1)
                for count,(_, _, consumed) in zip(proposal_counts,results)
            )
            for row, (lane, (result, emitted, consumed)) in enumerate(zip(cohort,results)):
                count = proposal_counts[row]
                lane.cache = rows[row]
                lane.tail = features[row:row+1,:consumed]
                lane.history.extend(inputs[row][:consumed])
                lane.anchor = emitted[-1]
                round_accepted = min(result.accepted, len(emitted)-1)
                self.scheduler_stats["accepted_proposals"] += round_accepted
                self.scheduler_stats["proposed_tokens"] += count
                lane.external_rounds += int(count > 0)
                lane.proposed += count
                lane.accepted += round_accepted
                lane.relaxed_accepts += result.relaxed_accepts
                _bump(
                    self.scheduler_stats,
                    "fly_relaxed_accepts",
                    result.relaxed_accepts,
                )
                lane.target_max_width = max(lane.target_max_width, len(cohort))
                if count:
                    lane.draft_max_width = max(lane.draft_max_width, draft_widths[row])
                for j,token in enumerate(emitted):
                    lane.generated += 1
                    finish = "stop" if token in self.stops else "length" if lane.generated >= lane.maximum else None
                    final = j == len(emitted)-1
                    logp = self.mx.log(self.mx.array(result.target_probabilities[j].astype(np.float32)))
                    lane.ready.append(SimpleNamespace(uid=lane.uid, token=token, logprobs=logp, finish_reason=finish, execution_width=len(cohort), all_tokens=list(lane.history) if final else None, prompt_cache=copy.deepcopy(lane.cache) if finish else None, cache_sidecar=self._sidecar(lane) if finish else None, mtp_state=None, mtp_receipt=None, speculative_receipt={"kind":"external_dflash2", "execution":"external_draft_verify" if lane.external_rounds else "ordinary_target", "current_execution":"ordinary_target" if count == 0 else "external_draft_verify", "ordinary_fallback":lane.ordinary, "external_rounds":lane.external_rounds, "accepted":lane.accepted,"proposed":lane.proposed,"round_accepted":round_accepted,"round_proposed":count,"target_width":lane.target_max_width,"draft_width":lane.draft_max_width,"qualification_authority":"serving_route", **self._verification_receipt(lane)}))
            self.scheduler_stats[
                "external_rounds" if any(proposal_counts) else "ordinary_rounds"
            ] += 1
            self.scheduler_stats["target_max_width"] = max(self.scheduler_stats["target_max_width"],len(cohort))
        except BaseException:
            if transaction is not None and not transaction.closed:
                try: transaction.abort()
                except BaseException:  # noqa: BLE001, S110 - authoritative snapshots restore below
                    pass
            for lane, (slot, boundary) in zip(cohort, recovery):
                state = slot.restore(
                    route="external_dflash2",
                    revision=self.binding,
                    boundary=boundary,
                )
                lane.__dict__.clear(); lane.__dict__.update(state)
            self.scheduler_stats = stats_snapshot
            self.scheduler_stats["recovery_checkpoint_restores"] += len(recovery)
            raise
        finally: self._open = False

    def _ordinary_round(self, cohort):
        """Advance permanently ordinary external lanes on target state only.

        ``Lane.ordinary`` is irreversible in this executor.  Unlike a final
        transient depth-zero round, these rows can never need DFlash context
        again, so avoid draft append, target taps and speculative rollback.
        """
        snapshots = [copy.deepcopy(lane.__dict__) for lane in cohort]
        stats_snapshot = dict(self.scheduler_stats)
        self._open = True
        try:
            inputs = self.mx.array(
                [[lane.anchor] for lane in cohort], dtype=self.mx.int32
            )
            if len(cohort) == 1:
                # Preserve the authoritative row's allocation/capacity. A
                # merge/extract round-trip compacts it and perturbs admission.
                batched_cache = cohort[0].cache
                logits = self.model(inputs, cache=batched_cache)
                self.mx.eval(logits, [cache.state for cache in batched_cache])
            else:
                batched_cache = []
                for layer in range(len(cohort[0].cache)):
                    rows = [lane.cache[layer] for lane in cohort]
                    merge = getattr(rows[0], "merge", None)
                    if not callable(merge):
                        raise TypeError(
                            f"{type(rows[0]).__name__} cannot batch ordinary external lanes"
                        )
                    batched_cache.append(merge(rows))
                logits = self.model(inputs, cache=batched_cache)
                self.mx.eval(logits, [cache.state for cache in batched_cache])
            for row,lane in enumerate(cohort):
                if len(cohort) > 1:
                    lane.cache = [cache.extract(row) for cache in batched_cache]
                targets = [
                    self._target_law(
                        lane, logits[row, 0], lane.history + [lane.anchor], True
                    )
                ]
                result = verify_proposals([], [], targets, lane.rng)
                token = int(result.emitted[0])
                lane.history.append(lane.anchor)
                lane.anchor = token
                lane.generated += 1
                lane.target_max_width = max(lane.target_max_width, len(cohort))
                finish = (
                    "stop"
                    if token in self.stops
                    else "length"
                    if lane.generated >= lane.maximum
                    else None
                )
                logp = self.mx.log(
                    self.mx.array(result.target_probabilities[0].astype(np.float32))
                )
                lane.ready.append(
                    SimpleNamespace(
                        uid=lane.uid,
                        token=token,
                        logprobs=logp,
                        finish_reason=finish,
                        execution_width=len(cohort),
                        # One token is this round's final response, even when
                        # the request continues. Match the multi-token round
                        # contract consumed by serving and downstream adapters.
                        all_tokens=list(lane.history),
                        prompt_cache=copy.deepcopy(lane.cache) if finish else None,
                        # The draft plane stopped at fallback and is stale by
                        # construction.  Publish this completion as target-only.
                        cache_sidecar=None,
                        mtp_state=None,
                        mtp_receipt=None,
                        speculative_receipt={
                            "kind":"external_dflash2",
                            "execution":"external_draft_verify" if lane.external_rounds else "ordinary_target",
                            "current_execution":"ordinary_target",
                            "ordinary_fallback":True,
                            "external_rounds":lane.external_rounds,
                            "accepted":lane.accepted,
                            "proposed":lane.proposed,
                            "round_accepted":0,
                            "round_proposed":0,
                            "target_width":lane.target_max_width,
                            "draft_width":lane.draft_max_width,
                            "qualification_authority":"serving_route",
                            **self._verification_receipt(lane),
                        },
                    )
                )
            count = len(cohort)
            self.scheduler_stats["ordinary_rounds"] += 1
            _bump(self.scheduler_stats, "external_ordinary_fast_path_rounds")
            _bump(self.scheduler_stats, "external_ordinary_fast_path_lanes", count)
            _bump(self.scheduler_stats, "external_draft_context_skipped", count)
            _bump(self.scheduler_stats, "external_taps_skipped", count)
            _bump(self.scheduler_stats, "external_transactions_skipped", count)
            self.scheduler_stats["target_max_width"] = max(
                self.scheduler_stats["target_max_width"], len(cohort)
            )
        except BaseException:
            for lane,state in zip(cohort,snapshots):
                lane.__dict__.clear(); lane.__dict__.update(state)
            self.scheduler_stats = stats_snapshot
            raise
        finally:
            self._open = False

    def _admit(self, lanes, append, *, prefill=False):
        """Explicit reservation estimate; measured peak qualification still required.

        Global KV grows; sliding KV is window bounded but a prefill chunk can
        retain the whole appended span until attention completes. Four bytes
        per KV element conservatively covers the current FP32/BF16 paths.
        The server callback already excludes its mandatory process reserve.
        """
        if self.memory_headroom is None: return True
        target = self.model.args; draft = self.draft.config
        target_per_token = 2*target.num_key_value_heads*target.head_dim*4
        draft_per_token = 2*draft.num_key_value_heads*draft.head_dim*4
        required = 0
        for lane in lanes:
            target_bytes = sum(int(getattr(c,"nbytes",0)) for c in lane.cache)
            draft_bytes = sum(int(getattr(c,"nbytes",0)) for c in lane.draft_cache)
            required += 4*target_bytes + 3*draft_bytes
            required += append*(target.num_hidden_layers*target_per_token + draft.num_hidden_layers*draft_per_token)
            required += append*target.hidden_size*len(self.layers)*8
            required += append*(target.num_hidden_layers*(target.intermediate_size+target.hidden_size)*16 + draft.num_hidden_layers*(draft.intermediate_size+draft.hidden_size)*16)
            if not prefill:
                # Four coexisting float64 q/p originals+normalized copies,
                # native logits/exp/output buffers, selector and safety slack.
                required += append*target.vocab_size*128
        self.scheduler_stats["reservation_bytes"] = required
        if required > self.memory_headroom():
            self.scheduler_stats["memory_deferred"] = self.scheduler_stats.get("memory_deferred",0)+1
            return False
        return True

    def _reclaim_for_admission(self):
        """Pressure-only, bounded reclaim; never reclaim for a lane-count cap."""
        if not self._allocator_reclaimed and self.reclaim_memory is not None:
            self._allocator_reclaimed = True
            self.reclaim_memory()
            return True
        if self._reclaims_left and self.evict_checkpoint is not None:
            self._reclaims_left -= 1
            if self.evict_checkpoint():
                self.scheduler_stats["memory_pressure_evictions"] = self.scheduler_stats.get("memory_pressure_evictions", 0) + 1
                if self.reclaim_memory is not None:
                    self.reclaim_memory()
                return True
        return False

    def _fit_cohort(self, candidates, append):
        # A failed B4 reservation must not block B1/B2 work. Retain each lane's
        # own proposal count; changing width never changes its random schedule.
        while True:
            selected = []
            for lane in candidates:
                if self._admit(selected + [lane], append):
                    selected.append(lane)
            if selected or not self._reclaim_for_admission():
                return selected

    def disable_speculation(self, uid):
        if self._open: raise RuntimeError("Cannot change route inside a transaction")
        self.lanes[uid].ordinary = True

    def next(self):
        prompts, responses = [], []
        self._reclaims_left = 2; self._allocator_reclaimed = False
        ordered = list(self.lanes.values())
        if ordered:
            start = self._schedule_cursor % len(ordered)
            ordered = ordered[start:] + ordered[:start]
            self._schedule_cursor += 1
        # At most one bounded prefill slice; active decode progresses every poll.
        active_count = sum(l.anchor is not None for l in self.lanes.values())
        for lane in ordered:
            if lane.anchor is not None or active_count >= self.capacity:
                continue
            append = min(self.prefill_step, max(1, len(lane.remaining)-1))
            admitted = self._admit([lane], append, prefill=True)
            while not admitted and self._reclaim_for_admission():
                admitted = self._admit([lane], append, prefill=True)
            if admitted:
                prompts.append(self._prefill(lane)); break
        ready = [l for l in ordered if l.anchor is not None and not l.ready and not l.cancelled][:self.capacity]
        groups = {}
        for lane in ready:
            # Another lane's token budget must not change this lane's proposal
            # count and random draw schedule. Cohort only compatible counts.
            count = 0 if lane.ordinary else min(self.num_draft,lane.maximum-lane.generated-1)
            # Permanent ordinary lanes must not share a zero-depth round with
            # transient final-budget rows, whose draft context remains exact.
            groups.setdefault((count, lane.ordinary),[]).append(lane)
        for (count, _ordinary), candidates in groups.items():
            cohort = self._fit_cohort(candidates, count+1)
            if not cohort: continue
            pending = [cohort]
            while pending:
                group = pending.pop()
                try:
                    self._round(group)
                except DraftUnavailable as error:
                    failed_uids = set(
                        getattr(error, "failed_uids", ())
                        or (lane.uid for lane in group)
                    )
                    failed = [lane for lane in group if lane.uid in failed_uids]
                    retry = [lane for lane in group if lane.uid not in failed_uids]
                    if not failed:
                        raise
                    for lane in failed:
                        lane.ordinary = True
                    self.scheduler_stats["draft_fallbacks"] += len(failed)
                    self._round(failed)
                    if retry:
                        pending.append(retry)
        for lane in list(self.lanes.values()):
            if lane.ready:
                response = lane.ready.popleft(); responses.append(response)
                if response.finish_reason: self.remove([lane.uid], cancelled=False)
        return prompts, responses

    def pop_prompt_boundary(self, uid): return self.boundaries.pop(uid, None)

    def remove(self, uids, return_prompt_caches=False, *, cancelled=True):
        if self._open: raise RuntimeError("Cannot remove during external transaction")
        result = {}
        for uid in uids:
            lane = self.lanes.pop(uid,None); self.boundaries.pop(uid,None)
            if lane is not None:
                lane.cancelled = bool(cancelled); self.scheduler_stats["cancelled"] += int(cancelled)
                if return_prompt_caches: result[uid] = copy.deepcopy(lane.cache)
                lane.ready.clear()
        return result

    def close(self):
        self.remove(list(self.lanes)); self.boundaries.clear()
