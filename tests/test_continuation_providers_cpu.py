"""Complete sequence providers use actual loaded tensor protocols on CPU."""

import copy
import hashlib
import itertools

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from test_parallel_draft_apcv2_batching_cpu import collect, pair, serving_engine
from test_proposal_composition_cpu import _native_qsa_pair
from test_standard_xpress_serving_cpu import drain, generator, reference

from mlx2.adapters.proposal_path_sources import (
    ExternalContinuationSource,
    NativeMTPContinuationSource,
    build_continuation_drafter,
)
from mlx2.runtime.proposal_providers import (
    ContinuationContext,
    ContinuationPoolPolicy,
    PromptLookupContinuationSource,
    continuation_context_revision,
)

PIN = "a" * 64


def context(model, draft, history=(1, 2), anchor=3, depth=2):
    pending = model.prefill_body(
        mx.array([history]), model.make_cache(), draft.config.target_layer_ids
    )
    return ContinuationContext(history, anchor, depth, draft.make_cache(), pending)


@pytest.mark.parametrize("kind", ["xpress", "lilicorr"])
def test_actual_external_head_beam_returns_complete_unique_paths_private_cache(kind):
    model, draft = pair(kind)
    ctx = context(model, draft)
    source = ExternalContinuationSource(draft)
    before = copy.deepcopy(ctx.draft_cache)
    paths = source(ctx, 15)
    expected_count = (
        15 if kind == "xpress" else draft.config.lilicorr_candidate_topk**ctx.depth
    )
    assert len(paths) == expected_count
    assert len({path.tokens for path in paths}) == len(paths)
    assert all(
        len(path.tokens) == 2 and len(path.confidence_features) == 2 for path in paths
    )
    assert all(np.isfinite(path.ranking_score) for path in paths)
    assert [path.ranking_score for path in paths] == sorted(
        [path.ranking_score for path in paths], reverse=True
    )
    assert all(
        now.offset == old.offset == 0 for now, old in zip(ctx.draft_cache, before)
    )
    again = source(ctx, 15)
    assert paths == again


def test_xpress_beam_matches_bruteforce_refined_block_proxy(monkeypatch):
    model, draft = pair("xpress")
    ctx = context(model, draft, depth=3)
    captured = []
    original = draft.xpress_head.jacobi_refine_greedy

    def capture(*args, **kwargs):
        tokens, logits = original(*args, **kwargs)
        captured.append(np.asarray(logits.astype(mx.float32))[0].astype(np.float64))
        return tokens, logits

    monkeypatch.setattr(draft.xpress_head, "jacobi_refine_greedy", capture)
    paths = ExternalContinuationSource(draft)(ctx, 15)
    logits = captured[0][: ctx.depth]
    exponent = np.exp(logits - logits.max(axis=-1, keepdims=True))
    log_probs = np.log(exponent / exponent.sum(axis=-1, keepdims=True))
    all_paths = [
        (sum(log_probs[position, token] for position, token in enumerate(path)), path)
        for path in itertools.product(range(draft.config.vocab_size), repeat=ctx.depth)
    ]
    expected = sorted(all_paths, key=lambda item: (-item[0], item[1]))[:15]
    assert [path.tokens for path in paths] == [path for _, path in expected]
    np.testing.assert_allclose(
        [path.ranking_score for path in paths],
        [score for score, _ in expected],
        atol=1e-10,
    )


@pytest.mark.parametrize("limit", [1, 7, 15, 65536])
def test_partitioned_top_matches_full_vocab_sort_with_cutoff_ties(limit):
    from mlx2.adapters.proposal_path_sources import _top

    rng = np.random.default_rng(58)
    values = rng.normal(size=65536)
    values[[0, 8, 100, 65535]] = 12
    values[[1, 9, 101, 60000, 64000, 65000, 65534]] = 11
    values[[7, 23, 36000]] = -np.inf
    values[[14, 65530]] = np.nan
    finite = np.flatnonzero(np.isfinite(values)).tolist()
    expected = sorted(finite, key=lambda token: (-float(values[token]), token))[:limit]
    assert _top(values, limit) == expected


def test_fixed_xpress_rows_normalize_once_per_position_without_processors(monkeypatch):
    from mlx2.adapters import proposal_path_sources

    model, draft = pair("xpress")
    ctx = context(model, draft, depth=3)
    original = proposal_path_sources._log_probs
    normalized = []

    def capture(values):
        normalized.append(np.array(values, copy=True))
        return original(values)

    monkeypatch.setattr(proposal_path_sources, "_log_probs", capture)
    paths = ExternalContinuationSource(draft)(ctx, 15)
    assert len(paths) == 15 and all(len(path.tokens) == 3 for path in paths)
    assert len(normalized) == ctx.depth


def test_native_mtp_beam_uses_actual_anchor_loaded_head_and_private_branches():
    model, draft = _native_qsa_pair("xpress")
    ctx = context(model, draft, history=(1, 2, 3), anchor=17, depth=2)
    paths = NativeMTPContinuationSource(model)(ctx, 15)
    assert len(paths) == 15 and len({path.tokens for path in paths}) == 15
    assert all(len(path.tokens) == 2 for path in paths)
    assert paths == NativeMTPContinuationSource(model)(ctx, 15)
    assert all(cache.offset == 0 for cache in ctx.draft_cache)
    assert NativeMTPContinuationSource(model, max_history=2)(ctx, 15) == []


def test_pld_enumerates_multiple_committed_matching_continuations():
    policy = ContinuationPoolPolicy.from_value({"ngram_min": 1, "ngram_max": 1})
    ctx = ContinuationContext((1, 2, 3, 1, 4, 5, 1, 6, 7), 1, 2, None, None)
    paths = PromptLookupContinuationSource(policy)(ctx, 15)
    assert {path.tokens for path in paths} == {(2, 3), (4, 5), (6, 7)}
    assert ctx.history == (1, 2, 3, 1, 4, 5, 1, 6, 7)


def wrapped(model, draft, policy=None):
    return build_continuation_drafter(
        model,
        draft,
        policy or {},
        target_revision=PIN,
        draft_revision="b" * 64,
        tokenizer_revision=PIN,
        session_revision="c" * 64,
    )


@pytest.mark.parametrize("kind", ["xpress", "lilicorr"])
def test_bound_wrapper_selects_fifteen_complete_paths_and_advances_context_once(kind):
    model, draft = pair(kind)
    wrapper = wrapped(model, draft, {"ngram_min": 1, "ngram_max": 1})
    histories, anchors = [[1, 2], [1, 2]], [3, 4]
    hidden = model.prefill_body(mx.array(histories), model.make_cache(), [0, 2])
    caches = [draft.make_cache(), draft.make_cache()]
    identities = [
        {
            "request_id": f"request-{row}",
            "round_id": 0,
            "context_revision": continuation_context_revision(
                wrapper.session.session_revision, history, anchor
            ),
        }
        for row, (history, anchor) in enumerate(zip(histories, anchors))
    ]
    tokens, laws = wrapper.draft_distributions(
        anchors,
        hidden,
        draft.batch_caches(caches),
        2,
        [None, None],
        [0, 0],
        processor_histories=histories,
        proposal_contexts=identities,
    )
    assert draft.stats["backbone_blocks"] == 2
    assert all(cache[0].offset == 2 for cache in caches)
    for row, selection in enumerate(wrapper.last_continuation_selections):
        assert 1 <= len(selection.paths) <= 15
        assert all(len(path.tokens) == 2 for path in selection.paths)
        assert tokens[row] == list(selection.paths[0].tokens)
        assert all(
            q[token] == 1 and np.count_nonzero(q) == 1
            for token, q in zip(tokens[row], laws[row])
        )
        wrapper.proposal_pool.discard(selection)


def test_inventory_auto_admits_actual_resident_mtp_and_refuses_missing_requested_source():
    model, draft = _native_qsa_pair("xpress")
    wrapper = wrapped(model, draft)
    assert set(wrapper.providers) == {"external", "prompt_lookup", "native_mtp"}
    model, draft = pair("xpress")
    assert set(wrapped(model, draft).providers) == {"external", "prompt_lookup"}
    with pytest.raises(ValueError, match="native MTP"):
        wrapped(model, draft, {"sources": ["native_mtp"]})
    for missing in ("dpara", "dflash2", "unknown"):
        with pytest.raises(ValueError, match="unavailable"):
            wrapped(model, draft, {"sources": [missing]})


def test_wrong_request_context_refused_before_backbone_cache_advance():
    model, draft = pair("xpress")
    wrapper = wrapped(model, draft)
    ctx = context(model, draft)
    with pytest.raises(ValueError, match="revision mismatch"):
        wrapper.draft_distributions(
            [ctx.anchor],
            ctx.pending_features,
            ctx.draft_cache,
            2,
            [None],
            [0],
            processor_histories=[ctx.history],
            proposal_contexts=[
                {
                    "request_id": "r",
                    "round_id": 0,
                    "context_revision": hashlib.sha256(b"wrong").hexdigest(),
                }
            ],
        )
    assert draft.stats["backbone_blocks"] == 0 and ctx.draft_cache[0].offset == 0


def test_shared_adapter_pool_feedback_uses_final_binding_and_closes(tmp_path):
    from mlx2.adapters.external_draft_policy import ExternalDraftAdapterMixin
    from mlx2.contracts import Capability, ModelDescriptor

    model, draft = pair("lilicorr")
    adapter = ExternalDraftAdapterMixin()
    adapter.model, adapter.layout = model, "target-layout"
    adapter.identity = {"fingerprint": PIN}
    adapter.external_policy = {
        "draft_model": "synthetic",
        "num_draft": 2,
        "continuation_pool": {"ngram_min": 1, "ngram_max": 1},
        "lilicorr_feedback": {
            "directory": str(tmp_path),
            "max_examples": 2,
            "min_examples": 2,
            "train_every": 100,
        },
    }
    record = {"fingerprint": "b" * 64, "args": draft.config}
    adapter._check_num_draft(record)
    descriptor = ModelDescriptor(
        model_type="qwen3",
        family="qwen3",
        variant="tiny",
        cache_layout="tiny",
        capabilities=frozenset({Capability.TEXT}),
        state_planes=frozenset(),
    )
    adapter._bind_external_drafter(record, lambda record, target: draft, descriptor)
    assert draft.feedback_manager is None
    final = hashlib.sha256(
        (adapter.identity["fingerprint"] + "final-settings").encode()
    ).hexdigest()
    adapter.identity["fingerprint"] = final
    adapter.create_external_batch()
    manager = draft.feedback_manager
    assert manager.binding == final and not manager.closed
    assert adapter.draft_model.feedback_manager is manager
    adapter.close()
    assert manager.closed and manager.child is None


def test_header_policy_refuses_incompatible_feedback_or_continuation_before_load(
    tmp_path,
):
    from types import SimpleNamespace

    from mlx2.adapters.external_draft_policy import ExternalDraftAdapterMixin

    adapter = ExternalDraftAdapterMixin()
    adapter.external_policy = {
        "num_draft": 2,
        "lilicorr_feedback": {"directory": str(tmp_path)},
    }
    with pytest.raises(ValueError, match="LiLiCoRR artifact"):
        adapter._check_num_draft({"args": SimpleNamespace(block_size=4)})
    adapter.external_policy = {"num_draft": 2, "continuation_pool": {}}
    with pytest.raises(ValueError, match="complete-path external head"):
        adapter._check_num_draft(
            {"args": SimpleNamespace(block_size=4), "config": {}}
        )


@pytest.mark.parametrize("kind", ["xpress", "lilicorr"])
def test_actual_pool_serving_cold_warm_apcv2_window_and_complete_path_counter(
    kind, monkeypatch
):
    model, draft = pair(kind, [2])
    wrapper = wrapped(model, draft, {"ngram_min": 1, "ngram_max": 1})
    engine, batches = serving_engine(monkeypatch, kind, model, wrapper)
    prompt = [1, 2, 3, 1, 2, 3, 1, 2]
    try:
        cold = collect(
            engine.submit(
                {"tokens": prompt, "max_tokens": 8, "temperature": 0, "top_k": 0}
            )
        )
        assert "error" not in cold
        assert cold["tokens"] == reference(model, prompt, 8)
        assert batches[0].scheduler_stats["external_continuation_sequences"] > 0
        assert batches[0].scheduler_stats["target_max_width"] > 1
        warm_prompt = prompt + cold["tokens"]
        warm = collect(
            engine.submit(
                {"tokens": warm_prompt, "max_tokens": 5, "temperature": 0, "top_k": 0}
            )
        )
        assert "error" not in warm
        assert warm["receipt"]["cached_tokens"] == len(warm_prompt) - 1
        assert warm["tokens"] == reference(model, warm_prompt, 5)
    finally:
        engine.close()


def test_constrained_target_outside_lili_candidates_falls_back_ordinary_without_labels():
    model, draft = pair("lilicorr")
    wrapper = wrapped(model, draft, {"sources": ["external"]})
    prompt = [1, 2, 3]

    class ForceAfterPrompt:
        history_pure = True
        token = None

        def __call__(self, tokens, logits):
            if int(tokens.size) <= len(prompt) or self.token is None:
                return logits
            return mx.where(mx.arange(logits.shape[-1]) == self.token, logits, -mx.inf)

    processor = ForceAfterPrompt()
    batch = generator(model, wrapper)
    batch.insert([prompt], max_tokens=[6], logits_processors=[[processor]])
    lane = batch.lanes[0]
    while lane.remaining:
        batch._prefill(lane)
    current = ContinuationContext(
        tuple(lane.history), lane.anchor, 2, lane.draft_cache, lane.tail
    )
    possible = ExternalContinuationSource(draft)(current, 15)
    first_ids = {path.tokens[0] for path in possible}
    processor.token = next(
        token for token in range(draft.config.vocab_size) if token not in first_ids
    )
    revision = wrapper.proposal_pool.feedback_revision
    output, final = drain(batch)
    assert output[0] == [reference(model, prompt, 1)[0]] + [processor.token] * 5
    assert batch.scheduler_stats["draft_fallbacks"] == 1
    assert final[0].speculative_receipt["ordinary_fallback"]
    assert wrapper.proposal_pool.feedback_revision == revision
