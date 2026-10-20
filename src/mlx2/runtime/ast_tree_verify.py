"""Private speculative tree tiles through a capability-bound target provider.

The scheduler owns only token topology and target-draw prefix matching. Tensor
math and per-path KV/recurrent commit remain inside the adapter transaction.
"""

from __future__ import annotations

import copy

from .ast_private_state import PrivateHistory
from .ast_tree_tiles import ASTProposalTrie, plan_tree_tile
from .speculative_sampling import RequestRNG


def project_tree_slate(generator, lanes, primary_tokens, widths):
    """Keep ranked complete paths without requiring one common fork depth."""
    from .proposal_pool import VERIFICATION_CONTRACT

    rows, provenance_rows = [], []
    selections = generator.draft.last_continuation_selections
    if len(selections) != len(lanes):
        raise ValueError("AST tree slate does not match its request cohort")
    for lane, primary, selection in zip(lanes, primary_tokens, selections, strict=True):
        selection.require_verification_contract(VERIFICATION_CONTRACT)
        ranked = list(enumerate(selection.paths))
        if not ranked or tuple(primary) != tuple(ranked[0][1].tokens):
            raise ValueError("AST tree primary differs from its selected slate")
        count = min(4, int(widths[lane.uid]))
        selected = ranked[:count]
        rows.append([list(path.tokens) for _, path in selected])
        provenance_rows.append(
            tuple(
                {
                    "rank": rank,
                    "path_id": path.path_id,
                    "contributors": tuple(
                        (item.source_id, item.source_revision, item.candidate_id)
                        for item in path.contributors
                    ),
                }
                for rank, path in selected
            )
        )
        generator._last_alternative_topology[lane.uid] = "unique_prefix_tree_tiles"
    generator.draft.last_ordered_alternative_provenance = tuple(provenance_rows)
    generator.draft.last_ordered_alternative_prefix_tokens = (0,) * len(rows)
    generator.draft.last_ordered_alternative_search_diagnostics = (None,) * len(rows)
    return rows, (0,) * len(rows)


def _slate_priors(blocks):
    """Use only matching proposal-pool probabilities, never ordinal scores."""
    probabilities, authorities = [], set()
    for block in blocks:
        selection = getattr(block, "continuation_selection", None)
        path = next(
            (
                path
                for path in getattr(selection, "paths", ())
                if tuple(block.tokens) == tuple(path.tokens[: len(block.tokens)])
            ),
            None,
        )
        if path is None:
            probabilities.append(None)
        else:
            probabilities.append(
                tuple(path.conditional_acceptance[: len(block.tokens)])
            )
            authorities.add(str(path.score_authority))
    return tuple(probabilities), "+".join(sorted(authorities)) or "unknown"


def _prepare_request(generator, lane, blocks):
    from types import SimpleNamespace

    from .external_speculative import _target_executor_stats

    if (
        not 1 <= len(blocks) <= 4
        or any(block is None or not block.tokens for block in blocks)
        or float(lane.sampling.get("sampling_temp", 0)) != 0.0
        or not generator._batched_law_contract(lane)
    ):
        raise ValueError("AST tree tiles require bounded greedy branch-safe paths")
    probabilities, authority = _slate_priors(blocks)
    trie = ASTProposalTrie.from_paths(
        int(lane.anchor),
        [tuple(int(token) for token in block.tokens) for block in blocks],
        conditional_acceptance=probabilities,
        score_authority=authority,
    )
    private = copy.copy(lane)
    private.cache = generator._copy_ast_private_cache(lane.cache)
    private.rng = RequestRNG(state=lane.rng.snapshot())
    private.history = PrivateHistory(lane.history)
    source_plan = getattr(lane, "ast_tree_source_plan", None)
    return SimpleNamespace(
        generator=generator,
        lane=lane,
        blocks=blocks,
        trie=trie,
        private=private,
        root=0,
        emitted=[],
        committed_inputs=[],
        feature_parts=[],
        laws=[],
        response_rows=[],
        keep_laws=True,
        keep_response=True,
        tiles=[],
        executed_nodes=set(),
        checkpoints=[],
        preplans={},
        plan_rows=None,
        prepared_plans=0,
        consumed_plans=0,
        async_participations=0,
        target_executor=_target_executor_stats(),
        source_plan=source_plan,
        tile_rows=(
            source_plan.max_tile_rows
            if source_plan is not None
            else generator.ast_tree_tile_rows
        ),
        max_tiles=source_plan.max_tiles if source_plan is not None else None,
        reason=None,
        physical_launches=0,
        physical_rows=0,
        max_cohort_width=1,
        packed=0,
        aligned={width: 0 for width in range(1, 5)},
        continued={width: 0 for width in range(1, 5)},
    )


def _plan_request(work, quota):
    from .external_speculative import TreeDraftRow

    # A measured source plan already includes the actual physical cohort
    # quota. Ready pressure is not a second division of that admitted budget.
    rows = (
        work.tile_rows if work.source_plan is not None else min(work.tile_rows, quota)
    )
    key = (id(work.trie), work.root, rows)
    tile = work.preplans.pop(key, None)
    work.preplans.clear()
    if tile is None:
        tile = plan_tree_tile(work.trie, root=work.root, max_rows=rows)
    else:
        work.consumed_plans += 1
    work.plan_rows = rows
    if work.executed_nodes.intersection(tile.global_indices):
        raise RuntimeError("AST tree scheduled an already verified node")
    work.private.anchor = tile.tokens[0]
    block = TreeDraftRow(
        list(tile.tokens[1:]), [parent - 1 for parent in tile.parents[1:]]
    )
    return tile, block


def _preplan_successors(trie, tile, *, max_rows):
    """Prepare existing host-only frontier metadata, never a fresh draft.

    A parent-closed tile cuts each of the at most four source paths once.
    No target choices, tensors, request RNG or private state are consulted.
    """
    selected = set(tile.global_indices)
    roots = tuple(
        child
        for node in tile.global_indices
        for child in trie.children[node]
        if child not in selected
    )
    if len(roots) > 4:
        raise RuntimeError("AST successor frontier exceeded its source-path bound")
    return {
        (id(trie), root, max_rows): plan_tree_tile(trie, root=root, max_rows=max_rows)
        for root in roots
    }


def _advance_request(work, tile, decision, rows, features, cache, cohort_width):
    generator, private, trie = work.generator, work.private, work.trie
    private.cache = cache
    work.feature_parts.append(features)
    inputs = [tile.tokens[index] for index in rows]
    work.checkpoints.append(len(work.committed_inputs))
    work.committed_inputs.extend(inputs)
    private.history.extend(inputs)
    work.emitted.extend(decision.emitted)
    if decision.target_laws is None:
        work.keep_laws = False
    else:
        work.laws.extend(decision.target_laws)
    if decision.response_logprobs is None:
        work.keep_response = False
    else:
        work.response_rows.extend(decision.response_logprobs)
    work.executed_nodes.update(tile.global_indices)
    work.tiles.append({**tile.receipt(), "cohort_width": cohort_width})
    work.max_cohort_width = max(work.max_cohort_width, cohort_width)
    work.packed += int(cohort_width > 1)
    last = tile.global_indices[rows[-1]]
    next_token = int(decision.emitted[-1])
    child = trie.matching_child(last, next_token)
    if next_token in generator.stops:
        work.reason, work.root = "stop", None
    elif len(work.emitted) == work.lane.maximum - work.lane.generated:
        work.reason, work.root = "limit", None
    elif child is None:
        work.reason = "complete_proposal" if not trie.children[last] else "correction"
        work.root = None
    elif work.max_tiles is not None and len(work.tiles) >= work.max_tiles:
        work.reason, work.root = "tile_budget", None
    else:
        if child in work.executed_nodes:
            raise RuntimeError("AST tree walker stopped before a verified child")
        work.reason, work.root = "reachable_continuation", child
    # The target's committed path alone selects a prepared continuation. A
    # stopped/corrected request retains none of the speculative host metadata.
    work.preplans = {
        key: plan for key, plan in work.preplans.items() if key[1] == work.root
    }


def _execute_group(generator, group):
    """Evaluate every owner before committing the aggregate private state."""
    from .external_speculative import _observe_target_executor

    owners = [work.private for work, _tile, _block in group]
    width, token_width = len(group), len(group[0][1].tokens)
    transaction = None
    pending_graph = None
    scopes = tuple(work.lane.session_scope_hash for work, _tile, _block in group)
    continued = any(work.tiles for work, _tile, _block in group)
    try:
        if width == 1:
            work, tile, _block = group[0]
            logits, features, transaction = generator._ast_target_tree_forward(
                work.private,
                work.private.cache,
                list(tile.tokens),
                parents=list(tile.parents),
            )
        else:
            logits, features, transaction = generator._ast_target_tree_forward_many(
                owners,
                [owner.cache for owner in owners],
                [list(tile.tokens) for _work, tile, _block in group],
                parents=[list(tile.parents) for _work, tile, _block in group],
            )
        if tuple(logits.shape[:2]) != (width, token_width) or tuple(
            features.shape[:2]
        ) != (width, token_width):
            raise RuntimeError("AST cohort target returned a foreign output geometry")
        launched, fences = [], []
        for index, (work, _tile, block) in enumerate(group):
            fence, state = generator._launch_tree_laws(
                work.private, block, logits[index : index + 1]
            )
            launched.append(state)
            fences.extend(fence)
        lookahead = [
            (work, tile)
            for work, tile, _block in group
            if (work.max_tiles is None or len(work.tiles) + 1 < work.max_tiles)
            and work.lane.maximum - work.lane.generated - len(work.emitted) > 1
            and any(
                child not in tile.global_indices
                for node in tile.global_indices
                for child in work.trie.children[node]
            )
        ]
        if lookahead:
            pending_graph = (features, *fences)
            generator.mx.async_eval(*pending_graph)
            for work, tile in lookahead:
                work.async_participations += 1
                work.preplans = _preplan_successors(
                    work.trie, tile, max_rows=work.plan_rows
                )
                work.prepared_plans += len(work.preplans)
        # Keep the original completion fence before any target-law read or
        # private-state commit; asynchronous submission hides host work only.
        pending_graph = None
        generator.mx.eval(features, *fences)
        pending = []
        for index, ((work, _tile, block), state) in enumerate(
            zip(group, launched, strict=True)
        ):
            decision = generator._verify_tree_batched(
                work.private, block, state, logprobs_on_request=True
            )
            rows = list(decision.commit_rows)
            if (
                not rows
                or len(work.emitted) + len(decision.emitted)
                > work.lane.maximum - work.lane.generated
            ):
                raise RuntimeError("AST tree exceeded the remaining response budget")
            selected = generator.mx.take(
                features[index : index + 1], generator.mx.array(rows), axis=1
            )
            pending.append((decision, rows, selected))
        caches = transaction.commit_paths(
            [rows for _decision, rows, _features in pending]
        )
        if not isinstance(caches, (tuple, list)) or len(caches) != width:
            raise RuntimeError("AST cohort transaction returned foreign private owners")
        if generator.ast_memo is not None:
            generator.ast_memo.advance(scopes, "transaction_commit")
    except BaseException as error:
        if pending_graph is not None:
            try:
                generator.mx.eval(*pending_graph)
            except BaseException as fence_error:  # noqa: BLE001 - retain preparation failure
                error.add_note(
                    f"AST pending graph fence also failed: {type(fence_error).__name__}"
                )
        if transaction is not None and not transaction.closed:
            try:
                transaction.abort()
            except BaseException as abort_error:  # noqa: BLE001 - retain the initiating failure
                error.add_note(
                    f"AST private transaction abort also failed: {type(abort_error).__name__}"
                )
        if generator.ast_memo is not None:
            generator.ast_memo.advance(scopes, "transaction_abort")
        raise
    # Original lanes remain untouched even if a later private tile fails.
    for (work, tile, _block), (decision, rows, selected), cache in zip(
        group, pending, caches, strict=True
    ):
        _advance_request(work, tile, decision, rows, selected, cache, width)
    charged = group[0][0]
    charged.physical_launches += 1
    charged.physical_rows += width * token_width
    charged.aligned[width] += 1
    charged.continued[width] += int(continued)
    _observe_target_executor(
        charged.target_executor,
        {
            "execution": "adapter_owned",
            "row_exact": True,
            "physical_width": width,
            "token_width": token_width,
            "adapter_detail": {
                "route": "ast_unique_tree_tiles",
                "projection_law": "admitted_same_law_target_tree",
                "provider_protocol": generator.ast_target_tree_executor_provider.protocol,
            },
        },
    )


def verify_tree_tiles_many(generator, lanes, blocks_by_lane):
    """Return existing per-request ABIs with exactly-once physical accounting.

    Equal-length tiles share one invocation, with no padding. Physical counts
    are attributed to its first owner; request receipts retain all participation.
    A capability miss selects B1 before dispatch, never after a failed forward.
    """
    lanes, blocks_by_lane = tuple(lanes), tuple(blocks_by_lane)
    if (
        not 1 <= len(lanes) <= 4
        or len(blocks_by_lane) != len(lanes)
        or len({id(lane) for lane in lanes}) != len(lanes)
        or len({lane.uid for lane in lanes}) != len(lanes)
    ):
        raise ValueError("AST tree cohort needs one to four distinct request owners")
    works = [
        _prepare_request(generator, lane, blocks)
        for lane, blocks in zip(lanes, blocks_by_lane, strict=True)
    ]
    provider = generator.ast_target_tree_executor_provider
    maximum = getattr(provider, "ast_tree_max_sessions", 1)
    widths = getattr(provider, "ast_tree_cohort_input_widths", None)
    if not callable(
        getattr(generator, "_ast_target_tree_forward_many", None)
    ) or not callable(widths):
        maximum = 1
    if type(maximum) is not int or not 1 <= maximum <= 4:
        raise ValueError("AST tree provider declared an invalid owner capacity")
    ready = works
    while ready:
        quota = generator.ast_tree_tile_rows // min(len(ready), maximum)
        groups = {}
        for work in ready:
            tile, block = _plan_request(work, quota)
            groups.setdefault(len(tile.tokens), []).append((work, tile, block))
        for token_width, compatible in groups.items():
            while compatible:
                count = min(
                    len(compatible),
                    maximum,
                    generator.ast_tree_tile_rows // token_width,
                )
                while count > 1 and token_width not in widths(count):
                    count -= 1
                group, compatible = compatible[:count], compatible[count:]
                _execute_group(generator, group)
        ready = [work for work in works if work.root is not None]
    finished = [_finish_request(work) for work in works]
    for work, (_result, receipt) in zip(works, finished, strict=True):
        work.lane.ast_tree_tiles_receipt = receipt
    return [result for result, _receipt in finished]


def verify_tree_tiles(generator, lane, blocks):
    """Original single-owner API, sharing the same private workspace driver."""
    return verify_tree_tiles_many(generator, [lane], [blocks])[0]


def _finish_request(work):
    from .external_speculative import RoundDecision

    generator, lane, blocks = work.generator, work.lane, work.blocks
    private, trie = work.private, work.trie
    emitted, committed_inputs = work.emitted, work.committed_inputs
    feature_parts, laws, response_rows = (
        work.feature_parts,
        work.laws,
        work.response_rows,
    )
    keep_laws, keep_response = work.keep_laws, work.keep_response
    tiles, executed_nodes, checkpoints = (
        work.tiles,
        work.executed_nodes,
        work.checkpoints,
    )
    target_executor, source_plan, reason = (
        work.target_executor,
        work.source_plan,
        work.reason,
    )

    def matched(tokens):
        count = 0
        for expected, actual in zip(tokens, emitted):
            if int(expected) != int(actual):
                break
            count += 1
        return count

    accepts = [matched(block.tokens) for block in blocks]
    decisions = []
    for accepted in accepts:
        count = min(accepted + 1, len(emitted))
        decisions.append(
            RoundDecision(
                accepted=accepted,
                emitted=emitted[:count],
                target_laws=laws[:count] if keep_laws else None,
                response_logprobs=response_rows[:count] if keep_response else None,
                commit_count=count,
                committed_inputs=committed_inputs[:count],
            )
        )
    from .speculative_topology import ordered_alternative_winner

    winner, _full = ordered_alternative_winner(
        decisions,
        tuple(len(block.tokens) for block in blocks),
        allow_partial_uplift=True,
    )
    chosen = decisions[winner]
    if chosen.commit_count != len(committed_inputs):
        raise RuntimeError("AST tree winner does not own the completed state boundary")
    # Loser metadata is an exact prefix observation of the same target walk,
    # not a fabricated zero-utility branch. Only the selected winner owns the
    # final private state; an unexpected caller cannot publish a loser cache.
    feature_row = generator.mx.concatenate(feature_parts, axis=1)
    feature_rows = generator.mx.concatenate(
        [
            feature_row if index == winner else generator.mx.zeros_like(feature_row)
            for index in range(len(blocks))
        ],
        axis=0,
    )
    rows = sum(tile["physical_rows"] for tile in tiles)
    stat = {
        "launches": len(tiles),
        "target_rows": rows,
        "rejected": int(reason == "correction"),
        "full_tiles": sum(
            tile["physical_rows"] == generator.ast_tree_tile_rows for tile in tiles
        ),
        "tile_checkpoints": checkpoints,
        "tile_input_rows": [tile["physical_rows"] for tile in tiles],
        "rollback_base_tile": len(tiles) - 1,
        "cohort_width": work.max_cohort_width,
    }
    zero = {
        **stat,
        "launches": 0,
        "target_rows": 0,
        "rejected": 0,
        "full_tiles": 0,
        "tile_checkpoints": [],
        "tile_input_rows": [],
    }
    record = {
        "tiles": tiles,
        "chosen_branch": winner + 1,
        "accepted_depth": chosen.accepted,
        "termination_reason": reason,
        "pending_nodes_skipped": len(trie.nodes) - len(executed_nodes),
        "unique_proposal_nodes": len(trie.nodes) - 1,
        "shared_rows_saved": sum(tile["shared_rows_saved"] for tile in tiles),
    }
    previous = getattr(lane, "ast_tree_tiles_receipt", {})
    receipt = {
        "implemented": True,
        "observed_used": True,
        "default": False,
        "packing": "measured_round_conditional_reach"
        if source_plan is not None
        else "fixed_budget_conditional_reach",
        "source_plan": source_plan.receipt() if source_plan is not None else None,
        "tuning": getattr(lane, "ast_tree_tuning", None),
        "launches": previous.get("launches", 0) + len(tiles),
        "target_rows": previous.get("target_rows", 0) + rows,
        "max_tile_rows": max(
            previous.get("max_tile_rows", 0), *stat["tile_input_rows"]
        ),
        "max_physical_cohort_width": max(
            previous.get("max_physical_cohort_width", 1), work.max_cohort_width
        ),
        "packed_launch_participations": previous.get("packed_launch_participations", 0)
        + work.packed,
        "physical_packed_observed": bool(
            previous.get("physical_packed_observed", False) or work.packed
        ),
        "prepared_successor_plans": previous.get("prepared_successor_plans", 0)
        + work.prepared_plans,
        "consumed_successor_plans": previous.get("consumed_successor_plans", 0)
        + work.consumed_plans,
        "async_lookahead_participations": previous.get(
            "async_lookahead_participations", 0
        )
        + work.async_participations,
        "pending_nodes_skipped": previous.get("pending_nodes_skipped", 0)
        + record["pending_nodes_skipped"],
        "last": record,
    }
    physical = {
        "physical_launches": work.physical_launches,
        "physical_target_rows": work.physical_rows,
        "unique_target_rows": work.physical_rows,
        "padding_rows": 0,
        "ragged_launches": 0,
        "max_cohort_width": work.max_cohort_width,
        "target_row_executor": target_executor,
        "aligned_tile_launches": dict(work.aligned),
        "aligned_continued_launches": dict(work.continued),
        "request_tile_launches": len(tiles),
        "owned_target_rows": rows,
        "first_tile_pairwise_disjoint": False,
        "shared_prefix_tokens": 0,
        "short_circuit_after_branch": winner + 1
        if accepts[winner] == len(blocks[winner].tokens)
        else None,
    }
    result = (
        decisions,
        feature_rows,
        [private.cache if index == winner else None for index in range(len(blocks))],
        [private.rng if index == winner else None for index in range(len(blocks))],
        [dict(stat) if index == winner else dict(zero) for index in range(len(blocks))],
        physical,
    )
    return result, receipt
