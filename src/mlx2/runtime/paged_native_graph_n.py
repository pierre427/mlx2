"""Ready N1..20 atomic continuation and online ragged verification."""
from contextlib import ExitStack

from .hybrid_packed_prefill_n import ROUTE, bootstrap_attribution
from .paged_native_atomic_owner import publish_native_cohort
from .paged_request_transaction import CandidateRequest
from .ragged_verify_layout import RaggedVerifyCapabilities, RaggedVerifyLayout


def validate_group(lanes):
    if (type(lanes) is not tuple or not 1 <= len(lanes) <= 20 or
            len({id(lane) for lane in lanes}) != len(lanes)):
        raise ValueError("distinct N1..20 continuations required")
    candidate = lanes[0].candidate
    if (getattr(candidate, "_serving_n20", False) is not True or
            any(lane.candidate is not candidate or lane.closed or
                lane.research_only or lane.owner.supported_planes != ("kv", "gdn")
                for lane in lanes)):
        raise ValueError("one shared admitted live N candidate required")
    return candidate


def decorate(response, candidate, width, proof=None, layout_receipt=None, *,
             speculative=False, publication="full_query_row_only",
             proposal=None):
    response.execution_width = width
    response.mtp_receipt.update(
        route=ROUTE, selected=True, observed_used=True, qualified=False,
        price_usable=False, actual_active_lanes=width, packed_lanes=width,
        ordinary_forward_calls=0, **bootstrap_attribution(candidate))
    if proof is not None:
        response.mtp_receipt["native_n20_graph_proof"] = dict(proof)
    if layout_receipt is not None:
        response.mtp_receipt["ragged_verify_layout"] = dict(layout_receipt)
        response.mtp_receipt["speculative_verification"] = speculative
        response.mtp_receipt["state_publication"] = publication
    if proposal is not None:
        response.mtp_receipt.update(proposal)
    return response


def _prompt_lookup_token_rows(lanes, candidate):
    """Return one exact deterministic proposal path per lane, if enabled."""
    config = getattr(candidate, "_native_ragged_prompt_lookup", None)
    if config is None:
        return tuple((lane._pending_token,) for lane in lanes), tuple(
            {"source": "ordinary", "proposed": 0} for _ in lanes)
    if (type(config) is not tuple or len(config) != 2 or
            type(config[1]) is not int or not 1 <= config[1] <= 15):
        raise RuntimeError("native N20 ragged prompt-lookup policy is invalid")
    from .proposal_providers import (
        ContinuationContext, ContinuationPoolPolicy,
        PromptLookupContinuationSource)

    policy = (config[0] if isinstance(config[0], ContinuationPoolPolicy)
              else ContinuationPoolPolicy.from_value(config[0]))
    source = PromptLookupContinuationSource(policy)
    rows, receipts = [], []
    for lane in lanes:
        anchor = lane._pending_token
        remaining = max(0, lane.maximum - lane.count)
        depth = min(config[1], max(0, remaining - 1))
        paths = source(ContinuationContext(
            tuple(lane.tokens), anchor, depth, None, None,
            tuple(lane.processors)), 1) if depth else ()
        path = tuple(paths[0].tokens[:depth]) if paths else ()
        rows.append((anchor, *path))
        receipts.append({
            "source": source.mechanism, "proposed": len(path),
            "proposal_distribution": "deterministic_point_mass",
            "ranking_convention": (paths[0].ranking_convention if paths else None),
            "ngram_min": policy.ngram_min,
            "ngram_max": policy.ngram_max,
        })
    return tuple(rows), tuple(receipts)


def run_native_graph_n(lanes):
    candidate = validate_group(lanes)
    priming = [lane._first_logits is not None for lane in lanes]
    if any(priming):
        if not all(priming) or any(lane._pending_token is not None for lane in lanes):
            raise RuntimeError("N cold priming state must be coherent")
        return tuple(decorate(lane.next(), candidate, len(lanes)) for lane in lanes)
    if any(type(lane._pending_token) is not int for lane in lanes):
        raise RuntimeError("all active N lanes must have pending sampled tokens")
    token_rows, proposal_receipts = _prompt_lookup_token_rows(lanes, candidate)
    return run_native_ragged_verify(
        lanes, token_rows, proposal_receipts=proposal_receipts)


def _ordinary_q1(lanes, token_rows, candidate, layout_receipt, metadata,
                 branches, prepared, reads, terminals, spans):
    """Preserve the proven one-row continuation path byte-for-byte."""
    backend = candidate.backend
    depth = candidate.native_layer_count
    for lane, row in zip(lanes, token_rows):
        branches.append(lane.owner.begin(CandidateRequest(
            lane.uid, lane.revision, len(row), ("kv", "gdn"))))
    packed = tuple(candidate.packed_lane(row, branch)
                   for row, branch in zip(token_rows, branches))
    logits, proof = candidate.forward_staged(
        packed, tuple(branches), permit_candidate=True,
        reserve_scratch=candidate.reserve_serving_scratch)
    if (len(logits) != len(lanes) or proof.get("packed_lanes") != len(lanes) or
            backend.read_submissions - reads != depth or
            backend.terminal_successes - terminals != depth or
            tuple(backend.staged_read_spans[spans:]) != (len(lanes),) * depth):
        raise RuntimeError("N exact read/span/terminal proof differs")
    for branch in branches:
        prepared.append(branch.prepare(1))
    publish_native_cohort(tuple(prepared))
    for lane, row in zip(lanes, token_rows):
        lane.tokens.extend(row)
        lane._pending_token = None
        lane.native_read_calls = backend.read_submissions
        lane.terminal_successes = backend.terminal_successes
    responses = []
    for index, (lane, origin) in enumerate(zip(lanes, metadata)):
        response = lane._next_with_reader(
            origin[0], origin[1], logits_override=logits[index])
        responses.append(decorate(
            response, candidate, len(lanes), proof, layout_receipt,
            speculative=False, publication="full_query_row_only",
            proposal={"native_n20_ragged_observed_used": False,
                      "draft_proposed": 0, "draft_accepted": 0,
                      "verifier_executed_rows": 1}))
    return tuple(responses)


def run_native_ragged_verify(lanes, token_rows, *, proposal_receipts=None):
    """Execute one exact online K+1 verify and atomically publish each prefix."""
    candidate = validate_group(lanes)
    if (type(token_rows) is not tuple or len(token_rows) != len(lanes) or
            any(type(row) is not tuple or not row or
                any(type(token) is not int for token in row)
                for row in token_rows)):
        raise ValueError("one nonempty integer token row is required per live lane")
    if proposal_receipts is None:
        proposal_receipts = tuple(
            {"source": "caller", "proposed": len(row) - 1}
            for row in token_rows)
    if (type(proposal_receipts) is not tuple or
            len(proposal_receipts) != len(lanes)):
        raise ValueError("one proposal receipt is required per live lane")
    layout = RaggedVerifyLayout.from_draft_depths(
        (lane.uid for lane in lanes), (len(row) - 1 for row in token_rows))
    plan = layout.select_backend(RaggedVerifyCapabilities(
        flattened=True, max_lanes=20, max_query_len=16,
        max_total_rows=20 * 16))
    layout_receipt = layout.receipt(plan)
    speculative = layout.max_query_len > 1
    layout_receipt.update(
        execution=("online_shrinking_cohort_verify" if speculative
                   else "ordinary_continuation_query"),
        recurrent_checkpoint=("selected_prefix_successor" if speculative
                              else "single_successor_only"))
    branches, prepared = [], []
    backend = candidate.backend
    depth = candidate.native_layer_count
    published = [False]
    with ExitStack() as leases:
        def reap_published_generations():
            if published[0]:
                for current_lane in lanes:
                    current_lane.owner.reap_retired()

        # Registered before reader leases so non-tail-sharing readers close
        # first. Tail-sharing lanes intentionally close their public readers
        # before mutation, then register a fresh post-publication reap.
        leases.callback(reap_published_generations)
        public = tuple(leases.enter_context(lane.owner.snapshot()) for lane in lanes)
        if any(view.revision != lane.revision or
               view.offset != len(lane.tokens) or
               len(view.layer_owners) != depth
               for lane, view in zip(lanes, public)):
            raise RuntimeError("N public revision/offset state drifted")
        metadata = tuple((view.generation, view.offset) for view in public)
        if all(getattr(lane.owner, "_reuse_private_tail", False) for lane in lanes):
            leases.close()
            leases.callback(reap_published_generations)
        reads = backend.read_submissions
        terminals = backend.terminal_successes
        spans = len(backend.staged_read_spans)
        try:
            if not speculative:
                result = _ordinary_q1(
                    lanes, token_rows, candidate, layout_receipt, metadata,
                    branches, prepared, reads, terminals, spans)
                published[0] = True
                return result

            for lane, row in zip(lanes, token_rows):
                branches.append(lane.owner.begin(CandidateRequest(
                    lane.uid, lane.revision, len(row), ("kv", "gdn"))))
            packed = tuple(candidate.packed_lane(row, branch)
                           for row, branch in zip(token_rows, branches))
            emitted = [[] for _ in lanes]
            events = []
            preview_states = [lane.matcher_state for lane in lanes]
            preview_counts = [lane.count for lane in lanes]
            accepted = [0 for _ in lanes]

            def decide_next(indices, step, logits):
                import mlx.core as mx
                from .generate import StopSequenceMatcher, _invalid_output_reason

                decisions = []
                width = len(indices)
                for row_index, lane_index in enumerate(indices):
                    lane = lanes[lane_index]
                    context = [*lane.tokens, *token_rows[lane_index][:step + 1]]
                    sampled, logprobs = lane._stage_sample_for_context(
                        logits[row_index], context)
                    mx.eval(sampled, logprobs)
                    if sampled.shape != (1,):
                        raise RuntimeError("native sampler must return one token")
                    token = int(sampled[0].item())
                    reason = _invalid_output_reason(token, logprobs[0])
                    if reason is not None:
                        raise RuntimeError(f"native sampled output invalid: {reason}")
                    state, matched = StopSequenceMatcher.match(
                        preview_states[lane_index], lane.matcher._trie, token)
                    preview_states[lane_index] = state
                    preview_counts[lane_index] += 1
                    finished = matched or preview_counts[lane_index] >= lane.maximum
                    from_draft = (step + 1 < len(token_rows[lane_index]) and
                                  token == token_rows[lane_index][step + 1])
                    if from_draft:
                        accepted[lane_index] += 1
                    item = (sampled, logprobs, from_draft, width)
                    emitted[lane_index].append(item)
                    events.append((lane_index, item))
                    decisions.append(from_draft and not finished)
                return tuple(decisions)

            _logits, proof = candidate.forward_staged_ragged(
                layout, packed, tuple(branches), decide_next=decide_next,
                permit_candidate=True,
                reserve_scratch=candidate.reserve_serving_scratch)
            executed = tuple(proof.get("executed_query_lengths", ()))
            round_widths = tuple(proof.get("round_widths", ()))
            if (len(executed) != len(lanes) or
                    any(value != len(items)
                        for value, items in zip(executed, emitted)) or
                    backend.read_submissions - reads != depth * len(round_widths) or
                    backend.terminal_successes - terminals != depth * len(round_widths) or
                    tuple(backend.staged_read_spans[spans:]) != tuple(
                        width for width in round_widths for _ in range(depth))):
                raise RuntimeError("N ragged read/span/terminal proof differs")
            for branch, rows in zip(branches, executed):
                prepared.append(branch.prepare(rows))
            publish_native_cohort(tuple(prepared))
            published[0] = True
            for lane, row, rows in zip(lanes, token_rows, executed):
                lane.tokens.extend(row[:rows])
                lane._pending_token = None
                lane.native_read_calls = backend.read_submissions
                lane.terminal_successes = backend.terminal_successes
            responses = []
            for lane_index, item in events:
                lane = lanes[lane_index]
                sampled, logprobs, from_draft, width = item
                response = lane._response_from_sample(
                    metadata[lane_index][0], metadata[lane_index][1],
                    sampled, logprobs)
                response.from_draft = from_draft
                proposal = {
                    "native_n20_ragged_observed_used": (
                        len(token_rows[lane_index]) > 1),
                    "draft_source": proposal_receipts[lane_index].get("source"),
                    "draft_distribution": proposal_receipts[lane_index].get(
                        "proposal_distribution", "deterministic_point_mass"),
                    "draft_proposed": len(token_rows[lane_index]) - 1,
                    "draft_accepted": accepted[lane_index],
                    "draft_configured_max_depth": (
                        candidate._native_ragged_prompt_lookup[1]
                        if isinstance(getattr(candidate,
                            "_native_ragged_prompt_lookup", None), tuple) else 0),
                    "verifier_executed_rows": executed[lane_index],
                    "verifier_round_widths": list(round_widths),
                    "recurrent_transition_peak": "one_selected_successor_per_lane",
                }
                responses.append(decorate(
                    response, candidate, width, proof, layout_receipt,
                    speculative=True,
                    publication="atomic_selected_executed_prefix",
                    proposal=proposal))
            return tuple(responses)
        except BaseException:
            for state in prepared:
                try:
                    state.rollback()
                except BaseException:
                    pass
            for branch in branches:
                try:
                    branch.rollback()
                except BaseException:
                    pass
            raise
__all__ = ["run_native_graph_n", "run_native_ragged_verify", "validate_group"]
