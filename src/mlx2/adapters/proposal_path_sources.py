"""Private model proposals for a pool of complete continuation sequences.

Tensor math stays in existing model/drafter protocols. Beam scores are private
source ranking proxies; they are never supplied as target verification q.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class SourceContinuation:
    tokens: tuple[int, ...]
    confidence_features: tuple[float | None, ...]
    ranking_score: float
    ranking_convention: str


def _log_probs(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("continuation source produced invalid logits")
    finite = np.isfinite(values)
    if not finite.any():
        return np.full(values.shape, -np.inf)
    exponent = np.exp(values - values[finite].max())
    return values - (values[finite].max() + math.log(float(exponent.sum())))


def _confidence(log_probability):
    probability = min(1 - 1e-12, max(1e-12, math.exp(log_probability)))
    return math.log(probability) - math.log1p(-probability)


def _top(log_probs, limit):
    ids = np.flatnonzero(np.isfinite(log_probs))
    if limit <= 0:
        return []
    if len(ids) > limit:
        values = log_probs[ids]
        # Partition identifies the cutoff in O(V). Resolve every cutoff tie
        # using the same ascending token-ID rule as the full stable ranking.
        threshold = np.partition(values, len(ids) - limit)[len(ids) - limit]
        better = ids[values > threshold]
        ties = ids[values == threshold]
        ids = np.concatenate([better, ties[: limit - len(better)]])
    return sorted(ids.tolist(), key=lambda token: (-float(log_probs[token]), token))[
        :limit
    ]


def _processed(processors, history, anchor, path, values):
    if not processors:
        return values
    import mlx.core as mx

    from ..runtime.processor_probe import probe_logits_processors

    prefix = mx.array([*history, anchor, *path], mx.uint32)
    return np.asarray(
        probe_logits_processors(processors, prefix, mx.array(values)[None])[0].astype(
            mx.float32
        )
    )


class ExternalContinuationSource:
    """Loaded external head as a request-private complete-path provider."""

    def __init__(self, draft):
        distribution = getattr(draft, "proposal_distribution", None)
        if distribution != "deterministic_point_mass" and not callable(
            getattr(draft, "propose_tree", None)
        ):
            raise ValueError(
                "complete external continuations require deterministic or tree head"
            )
        if getattr(draft, "continuation_protocol", None) == "dspark" and callable(
            getattr(draft, "propose_complete_paths", None)
        ):
            self.mechanism = "dspark"
        elif getattr(
            draft, "continuation_protocol", None
        ) == "native_mtp_head" and callable(
            getattr(draft, "propose_complete_paths", None)
        ):
            # The target's own MTP head: its paths come from logits the
            # round's greedy chain already computed (drafters/native_mtp.py).
            self.mechanism = "native_mtp"
        elif hasattr(draft, "xpress_head"):
            self.mechanism = "xpress"
        elif hasattr(draft, "lilicorr"):
            self.mechanism = "lilicorr"
        elif callable(getattr(draft, "propose_tree", None)):
            self.mechanism = (
                "dflash2_fold"
                if getattr(draft, "fold_policy", None) is not None
                else "dflash2"
            )
        else:
            raise ValueError(
                "loaded external head has no supported continuation protocol"
            )
        self.draft = draft
        self.last_diagnostics = None

    def __call__(self, context, limit):
        draft, config = self.draft, self.draft.config
        if not 1 <= context.depth < config.block_size:
            raise ValueError("continuation depth exceeds the loaded trained block")
        if self.mechanism == "native_mtp":
            result = draft.propose_complete_paths(context, limit)
            return [
                candidate
                for candidate in result[:limit]
                if isinstance(candidate, SourceContinuation)
                and 1 <= len(candidate.tokens) <= context.depth
            ]
        cache = (None if getattr(context, "draft_lattice", None) is not None
                 else copy.deepcopy(context.draft_cache))
        pending = context.pending_features
        if self.mechanism in {"dflash2", "dflash2_fold"}:
            from ..runtime.drafters.dflash_tree import tree_paths

            nodes = getattr(context, "node_budget", None) or 15
            options = {}
            codebook_cache_hit = None
            cached_codebooks = getattr(draft, "tree_codebooks", None)
            if callable(cached_codebooks):
                predecessor, successor, codebook_cache_hit = cached_codebooks()
                options["codebooks"] = (predecessor, successor)
            lattice = getattr(context, "draft_lattice", None)
            if lattice is not None:
                if "codebooks" not in options:
                    raise ValueError("reused DFlash lattice requires weight-bound codebooks")
                if lattice["anchors"] != [context.anchor] or lattice["depth"] != context.depth:
                    raise ValueError("continuation lattice anchor/depth mismatch")
                tokens, parents = draft.finish_tree({
                    **lattice, "max_nodes": nodes, "codebooks": options.get("codebooks"),
                })[0]
            else:
                tokens, parents = draft.propose_tree(
                    [context.anchor], pending, cache, nodes,
                    lattice_positions=context.depth + 1, **options,
                )[0]
            row_paths = tree_paths(parents)
            depth_hist = {}
            leaf_depth_hist = {}
            parent_rows = {int(parent) for parent in parents if int(parent) >= 0}
            for row, path_rows in enumerate(row_paths):
                key = str(len(path_rows))
                depth_hist[key] = depth_hist.get(key, 0) + 1
                if row not in parent_rows:
                    leaf_depth_hist[key] = leaf_depth_hist.get(key, 0) + 1
            self.last_diagnostics = {
                "mechanism": self.mechanism,
                "node_budget": int(nodes),
                "nodes": len(row_paths),
                "roots": sum(int(parent) < 0 for parent in parents),
                "leaves": len(row_paths) - len(parent_rows),
                "requested_depth": int(context.depth),
                "depth_hist": depth_hist,
                "leaf_depth_hist": leaf_depth_hist,
                "full_depth_nodes": sum(
                    len(path_rows) == context.depth for path_rows in row_paths
                ),
                "full_depth_leaves": sum(
                    row not in parent_rows and len(path_rows) == context.depth
                    for row, path_rows in enumerate(row_paths)
                ),
            }
            if codebook_cache_hit is not None:
                self.last_diagnostics["codebook_cache_hit"] = bool(codebook_cache_hit)
            if lattice is not None:
                self.last_diagnostics.update(backbone_reused=True, private_head_calls=0)
            result = []
            seen = set()
            for ordinal, rows in enumerate(row_paths):
                if not 1 <= len(rows) <= context.depth:
                    continue
                path = tuple(int(tokens[row]) for row in rows)
                if path in seen:
                    continue
                seen.add(path)
                result.append(
                    SourceContinuation(
                        path,
                        tuple(None for _ in path),
                        float(-ordinal),
                        "dflash2_best_first_node_ordinal_proxy",
                    )
                )
            return result[:limit]
        if self.mechanism == "dspark":
            private_context = replace(context, draft_cache=cache)
            result = draft.propose_complete_paths(private_context, limit)
            if not isinstance(result, (list, tuple)):
                raise ValueError("DSpark complete-path provider returned invalid data")
            normalized = []
            for ordinal, item in enumerate(result[:limit]):
                if isinstance(item, SourceContinuation):
                    candidate = item
                elif isinstance(item, dict):
                    tokens = tuple(item.get("tokens") or ())
                    candidate = SourceContinuation(
                        tokens,
                        tuple(item.get("confidence_features") or (None,) * len(tokens)),
                        float(item.get("ranking_score", -ordinal)),
                        str(
                            item.get(
                                "ranking_convention",
                                "dspark_complete_path_private_proxy",
                            )
                        ),
                    )
                else:
                    raise TypeError(
                        "DSpark complete-path provider returned invalid candidate"
                    )
                if len(candidate.tokens) == context.depth:
                    normalized.append(candidate)
            return normalized
        import mlx.core as mx

        anchor = mx.array([context.anchor], mx.int32)
        inputs = mx.concatenate(
            [
                anchor[:, None],
                mx.full((1, config.block_size - 1), config.mask_token_id, mx.int32),
            ],
            axis=1,
        )
        if self.mechanism == "lilicorr":
            if pending.shape[1] > 0:
                anchor_hidden = draft.hidden_norm(draft.fc(pending[:, -1]))
                anchor_valid = mx.ones((1,), mx.bool_)
            else:
                if any(item.offset > 0 for item in cache):
                    raise ValueError(
                        "LiLiCoRR continuation requires pending anchor features"
                    )
                anchor_hidden = mx.zeros((1, config.hidden_size))
                anchor_valid = mx.zeros((1,), mx.bool_)
            features = draft._hidden(inputs, pending, cache)[:, 1:]
            logits = draft._logits(features)
            candidates = mx.argsort(-logits, axis=-1)[
                ..., : config.lilicorr_candidate_topk
            ]
            log_probs = logits.astype(mx.float32) - mx.logsumexp(
                logits.astype(mx.float32), axis=-1, keepdims=True
            )
            candidate_probs = mx.take_along_axis(log_probs, candidates, -1)
            scores = draft.lilicorr(
                draft.embed_tokens(candidates),
                candidate_probs,
                features,
                anchor_hidden,
                anchor_valid,
            )
            lattice = np.asarray(scores.astype(mx.float32))[0]
            candidate_ids = np.asarray(candidates)[0]
            unary = np.asarray(logits.astype(mx.float32))[0]
        else:
            features = draft._hidden(inputs, pending, cache)
            _, refined = draft.xpress_head.jacobi_refine_greedy(
                draft._logits(features),
                features,
                anchor,
                mx.array(
                    [context.history[-1] if context.history else context.anchor],
                    mx.int32,
                ),
                config.xpress_num_passes,
            )
            unary = np.asarray(refined.astype(mx.float32))[0]
        # (score, token path, selected candidate column, confidence path)
        beams = [(0.0, (), 0, ())]
        for position in range(context.depth):
            next_beams = []
            fixed_options = None
            if self.mechanism == "xpress" and not context.processors:
                # Jacobi's final block is fixed before beam search. Without
                # prefix-dependent processors, every beam sees this same row.
                probs = _log_probs(unary[position])
                fixed_options = [
                    (token, float(probs[token]), _confidence(float(probs[token])))
                    for token in _top(probs, limit)
                ]
            for score, path, predecessor, confidence in beams:
                if fixed_options is not None:
                    for token, value, feature in fixed_options:
                        next_beams.append(
                            (
                                score + value,
                                (*path, token),
                                token,
                                (*confidence, feature),
                            )
                        )
                    continue
                if self.mechanism == "lilicorr":
                    values = lattice[position, predecessor].copy()
                    allowed = _processed(
                        context.processors,
                        context.history,
                        context.anchor,
                        path,
                        unary[position],
                    )
                    values[~np.isfinite(allowed[candidate_ids[position]])] = -np.inf
                    probs = _log_probs(values)
                    columns = _top(probs, limit)
                    options = [
                        (column, int(candidate_ids[position, column]))
                        for column in columns
                    ]
                else:
                    probs = _log_probs(
                        _processed(
                            context.processors,
                            context.history,
                            context.anchor,
                            path,
                            unary[position],
                        )
                    )
                    options = [(token, token) for token in _top(probs, limit)]
                for column, token in options:
                    value = float(probs[column])
                    next_beams.append(
                        (
                            score + value,
                            (*path, token),
                            column,
                            (*confidence, _confidence(value)),
                        )
                    )
            next_beams.sort(key=lambda item: (-item[0], item[1]))
            beams = next_beams[:limit]
            if not beams:
                break
        convention = (
            "refined_block_log_softmax_proxy"
            if self.mechanism == "xpress"
            else "correlation_transition_log_softmax_proxy"
        )
        return [
            SourceContinuation(path, confidence, score, convention)
            for score, path, _, confidence in beams
            if len(path) == context.depth
        ]


class NativeMTPContinuationSource:
    """A beam over a loaded target's native head, using private cache branches."""

    mechanism = "native_mtp"

    def __init__(self, model, *, max_history=4096):
        from .proposal_sources import native_mtp_source

        native_mtp_source(model)  # Validate the actual resident head protocol.
        if bool(getattr(model, "mtp_draft_vocab_enabled", False)) and not callable(
            getattr(model, "mtp_step_full_vocab", None)
        ):
            raise ValueError(
                "native MTP continuation requires a full-vocabulary head protocol"
            )
        if type(max_history) is not int or max_history < 1:
            raise ValueError("native MTP continuation history bound must be positive")
        self.model, self.max_history = model, max_history

    def open_sidecar_tree(
        self,
        checkpoint,
        candidates,
        materialize,
        *,
        sampling_temperature=0.0,
        proposal_distribution="deterministic_point_mass",
    ):
        """Bind ranked paths to request-private native-MTP cache sidecars.

        Proposal ranking remains in this adapter-owned source.  Target
        verification chooses the canonical branch; the returned transaction
        then publishes only that branch's complete target/draft/pending state.
        """

        from ..runtime.native_mtp_continuation_tree import (
            NativeMTPContinuationTreeTransaction,
        )

        paths = tuple(
            tuple(candidate.tokens)
            if isinstance(candidate, SourceContinuation)
            else tuple(candidate)
            for candidate in candidates
        )
        return NativeMTPContinuationTreeTransaction(
            checkpoint,
            paths,
            materialize,
            sampling_temperature=sampling_temperature,
            proposal_distribution=proposal_distribution,
        )

    def __call__(self, context, limit):
        import mlx.core as mx

        from ..runtime.hybrid_speculative import _mtp_backbone
        from ..runtime.models.cache import make_prompt_cache

        if not 0 < len(context.history) <= self.max_history:
            return []
        model = self.model
        target_cache, head_cache = make_prompt_cache(model), model.make_mtp_cache()
        step = getattr(model, "mtp_step_full_vocab", model.mtp_step)
        tokens = mx.array([context.history], mx.uint32)
        _, hidden = _mtp_backbone(model, tokens, target_cache)
        if len(context.history) > 1:
            step(hidden[:, :-1], tokens[:, 1:], head_cache)
        start = getattr(model, "mtp_start_cycle", None)
        end = getattr(model, "mtp_end_cycle", None)
        if callable(start):
            # As native_mtp_source: the drafted steps form one head cycle.
            start(head_cache, share_index_state=False)
        hidden = hidden[:, -1:]
        beams = [(0.0, (), hidden, head_cache, ())]
        # Every branch holds its own copy of the cycle ledger, so pruned
        # branches are ended too, not only the surviving beams.
        branches = [head_cache]
        try:
            for _ in range(context.depth):
                next_beams = []
                for score, path, hidden, cache, confidence in beams:
                    private_cache = copy.deepcopy(cache)
                    branches.append(private_cache)
                    logits, next_hidden = step(
                        hidden,
                        mx.array([[path[-1] if path else context.anchor]], mx.uint32),
                        private_cache,
                    )
                    values = np.asarray(logits[0, -1].astype(mx.float32))
                    probs = _log_probs(
                        _processed(
                            context.processors,
                            context.history,
                            context.anchor,
                            path,
                            values,
                        )
                    )
                    for token in _top(probs, limit):
                        value = float(probs[token])
                        next_beams.append(
                            (
                                score + value,
                                (*path, token),
                                next_hidden,
                                private_cache,
                                (*confidence, _confidence(value)),
                            )
                        )
                next_beams.sort(key=lambda item: (-item[0], item[1]))
                beams = next_beams[:limit]
                if not beams:
                    break
            return [
                SourceContinuation(
                    path, confidence, score, "native_head_beam_log_softmax_proxy"
                )
                for score, path, _, _, confidence in beams
                if len(path) == context.depth
            ]
        finally:
            if callable(end):
                for cache in branches:
                    end(cache)


def build_continuation_drafter(
    target,
    draft,
    policy,
    *,
    target_revision,
    draft_revision,
    tokenizer_revision,
    session_revision,
):
    """Admit only actual resident providers into a revision-bound session."""
    import hashlib
    from pathlib import Path

    from ..runtime.drafter_portfolio import (
        DrafterPortfolio,
        PortfolioContinuationSource,
    )
    from ..runtime.proposal_pool import ProposalSession, ProposalSource
    from ..runtime.proposal_providers import (
        ContinuationDraftModel,
        ContinuationPoolPolicy,
        PromptLookupContinuationSource,
    )

    policy = dict(policy)
    if "sources" not in policy:
        policy["sources"] = (
            [item.provider_id for item in draft.providers]
            if isinstance(draft, DrafterPortfolio)
            else ["external"]
        )
        policy["sources"].append("prompt_lookup")
        if getattr(target, "mtp", None) is not None:
            policy["sources"].append("native_mtp")
    parsed = ContinuationPoolPolicy.from_value(policy)
    session = ProposalSession(
        session_revision, target_revision, tokenizer_revision, draft.config.vocab_size
    )
    providers, records = {}, {}
    portfolio_records = (
        {item.provider_id: item for item in draft.providers}
        if isinstance(draft, DrafterPortfolio)
        else {}
    )
    for name in parsed.sources:
        if name in portfolio_records:
            resident = portfolio_records[name]
            provider = PortfolioContinuationSource(
                draft, name, ExternalContinuationSource(resident.draft)
            )
            revision = resident.revision
            lane_cost = resident.lane_cost
            tile_rows = resident.tile_rows
            fallback = resident.fallback
            max_depth = resident.draft.config.block_size - 1
        elif name == "external":
            provider = ExternalContinuationSource(draft)
            revision = draft_revision
            lane_cost = tile_rows = 1
            fallback = False
            max_depth = draft.config.block_size - 1
        elif name == "native_mtp":
            provider = NativeMTPContinuationSource(
                target, max_history=parsed.mtp_max_history
            )
            revision = target_revision
            lane_cost = tile_rows = 1
            fallback = False
            max_depth = draft.config.block_size - 1
        else:
            if name != "prompt_lookup":
                raise ValueError(
                    f"continuation source {name!r} is unavailable because it is "
                    "not resident in this route"
                )
            provider = PromptLookupContinuationSource(parsed)
            revision = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            lane_cost = tile_rows = 1
            fallback = True
            max_depth = draft.config.block_size - 1
        providers[name] = provider
        records[name] = ProposalSource(
            name,
            provider.mechanism,
            revision,
            session_revision,
            target_revision,
            tokenizer_revision,
            draft.config.vocab_size,
            loaded=True,
            token_path_support=True,
            max_depth=max_depth,
            lane_cost=lane_cost,
            tile_rows=tile_rows,
            fallback=fallback,
        )
    return ContinuationDraftModel(
        draft, policy, session=session, source_records=records, providers=providers
    )
