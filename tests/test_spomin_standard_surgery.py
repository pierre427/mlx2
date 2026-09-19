"""Standard-attention Spomin backend and its serving lifecycle boundary."""
from types import SimpleNamespace as NS

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx2 import memory, serving
from mlx2.runtime import apc_v2, generate, os_memory
from mlx2.runtime.cache_planes import TranscriptLedgerPlane, TranscriptLedgerSegment
from mlx2.runtime.models.cache import KVCache, RotatingKVCache
from mlx2.runtime.spomin_layer import (
    SpominCapabilityError,
    SpominConfig,
    SpominLayer,
    SpominTargetState,
)
from mlx2.runtime.spomin_live_surgery import (
    ServingSpominPolicy,
    SpominLiveSurgeryManager,
)
from mlx2.runtime.spomin_standard_surgery import (
    StandardAttentionSpominBackend,
    shift_rope,
)
from mlx2.server import collect_nonstream_job


class Attention(nn.Module):
    def __init__(self, *, traditional, use_rope=True):
        super().__init__()
        self.q = nn.Linear(16, 16, bias=False)
        self.k = nn.Linear(16, 8, bias=False)
        self.v = nn.Linear(16, 8, bias=False)
        self.use_rope = use_rope
        self.rope = nn.RoPE(4, traditional=traditional, base=10_000.0)

    def __call__(self, x, cache):
        B, T, _ = x.shape
        q = self.q(x).reshape(B, T, 4, 4).transpose(0, 2, 1, 3)
        k = self.k(x).reshape(B, T, 2, 4).transpose(0, 2, 1, 3)
        v = self.v(x).reshape(B, T, 2, 4).transpose(0, 2, 1, 3)
        if self.use_rope:
            q, k = self.rope(q, offset=cache.offset), self.rope(k, offset=cache.offset)
        k, v = cache.update_and_fetch(k, v)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=0.5)
        return out.transpose(0, 2, 1, 3).reshape(B, T, 16)


class Toy(nn.Module):
    """One attention layer: surgery is exact because K/V depend on one token."""

    def __init__(self, *, traditional, use_rope=True):
        super().__init__()
        self.embed = nn.Embedding(32, 16)
        self.layers = [NS(self_attn=Attention(traditional=traditional, use_rope=use_rope))]

    def step(self, tokens, cache):
        out = None
        for token in tokens:  # single-token steps need no mask for either cache
            x = self.embed(mx.array([[token]]))
            out = x + self.layers[0].self_attn(x, cache[0])
        return out


def transcript():
    return TranscriptLedgerPlane(
        tokenizer_identity="t", tokenizer_version="1", revision="r1",
        segments=(
            TranscriptLedgerSegment("s1", 0, 3, (1, 2, 3)),
            TranscriptLedgerSegment("s2", 3, 7, (4, 5, 6, 7)),
            TranscriptLedgerSegment("s3", 7, 9, (8, 9)),
            TranscriptLedgerSegment("s4", 9, 12, (10, 11, 12)),
        ),
    )


def planned(strategy="largest_first", protected=()):
    ledger = transcript()
    state = SpominTargetState(
        revision="target-r1", target_tokens=12, transcript=ledger,
        visible_segment_ids=tuple(s.segment_id for s in ledger.segments),
    )
    layer = SpominLayer(SpominConfig(
        capacity_tokens=16, pressure_ratio=0.70, target_ratio=0.50,
        protect_recent_segments=0, strategy=strategy,
    ))
    return state, layer, layer.plan(state, protected_segment_ids=protected)


@pytest.mark.parametrize("traditional", [False, True])
def test_shift_rope_composes_with_stock_rope(traditional):
    rope = nn.RoPE(4, traditional=traditional, base=10_000.0)
    x = mx.random.normal((1, 2, 5, 6))
    at_nine = rope(x, offset=9)
    moved = shift_rope(
        at_nine, mx.full((1, 1, 5), -6.0), dims=4, base=10_000.0,
        traditional=traditional,
    )
    np.testing.assert_allclose(
        np.asarray(moved), np.asarray(rope(x, offset=3)), rtol=1e-4, atol=1e-5
    )


@pytest.mark.parametrize("traditional,use_rope", [(False, True), (True, True), (False, False)])
def test_full_attention_surgery_matches_compacted_rebuild(traditional, use_rope):
    mx.random.seed(3)
    model = Toy(traditional=traditional, use_rope=use_rope)
    state, layer, plan = planned()  # removes s2 (positions 3..6)
    assert plan.selection.segment_ids == ("s2",)
    cache = [KVCache()]
    model.step(range(1, 13), cache)
    updated = layer.apply(state, plan, StandardAttentionSpominBackend(model, cache))
    assert updated.target_tokens == 8 and cache[0].offset == 8
    rebuilt = [KVCache()]
    model.step([1, 2, 3, 8, 9, 10, 11, 12], rebuilt)
    np.testing.assert_allclose(
        np.asarray(model.step([13], cache)), np.asarray(model.step([13], rebuilt)),
        rtol=2e-5, atol=2e-5,
    )


def test_wrapped_ring_is_uniformly_rephased_and_stays_a_legal_ring():
    mx.random.seed(4)
    model = Toy(traditional=False)
    state, layer, plan = planned()
    cache = [RotatingKVCache(max_size=4)]
    model.step(range(1, 13), cache)  # ring holds positions 8..11, wrapped
    layer.apply(state, plan, StandardAttentionSpominBackend(model, cache))
    assert cache[0].offset == 8 and cache[0].keys.shape[2] == 4
    rebuilt = [RotatingKVCache(max_size=4)]
    model.step([1, 2, 3, 8, 9, 10, 11, 12], rebuilt)
    for token in (13, 14, 15, 16, 17):  # past another full wrap
        np.testing.assert_allclose(
            np.asarray(model.step([token], cache)),
            np.asarray(model.step([token], rebuilt)),
            rtol=2e-5, atol=2e-5,
        )


def test_ring_still_holding_removed_tokens_is_refused_before_mutation():
    model = Toy(traditional=False)
    state, layer, plan = planned()
    cache = [RotatingKVCache(max_size=8)]  # holds 4..11, overlaps s2
    model.step(range(1, 13), cache)
    before = np.asarray(cache[0].keys)
    with pytest.raises(SpominCapabilityError, match="sliding window still holds"):
        layer.apply(state, plan, StandardAttentionSpominBackend(model, cache))
    assert cache[0].offset == 12
    np.testing.assert_array_equal(np.asarray(cache[0].keys), before)


def test_unknown_rotary_or_cache_type_is_refused():
    model = Toy(traditional=False)
    state, layer, plan = planned()
    cache = [KVCache()]
    model.step(range(1, 13), cache)
    model.layers[0].self_attn.rope = NS(dims=4, base=1e4, traditional=False, scale=1.0)
    with pytest.raises(SpominCapabilityError, match="only default RoPE"):
        layer.apply(state, plan, StandardAttentionSpominBackend(model, cache))
    with pytest.raises(SpominCapabilityError, match="layer count"):
        layer.apply(state, plan, StandardAttentionSpominBackend(model, cache + cache))


def test_manager_protects_prefix_and_declines_without_adapter_backend():
    policy = ServingSpominPolicy(enabled=True, capacity_tokens=16, segment_tokens=3)
    assert policy.protect_prefix_segments == 1
    tokens = tuple(range(1, 13))
    ledger = policy.transcript(tokens, tokenizer_identity="t", revision="r")
    manager = SpominLiveSurgeryManager(enabled=True, backend_factory=lambda m, c: None)
    transaction = manager.prepare(
        request_id="a", prompt_token_ids=tokens, transcript=ledger,
        capacity_tokens=16, strategy="oldest_contiguous",
        has_mtp_state=False, has_recurrent_state=False,
        cache_is_request_private=True,
        protected_segment_ids=(ledger.segments[0].segment_id,),
    )
    assert ledger.segments[0].segment_id not in transaction.plan.selection.segment_ids
    receipt = transaction.apply(
        None, [], request_quiescent=True, device_work_drained=True
    )
    assert receipt["status"] == "declined" and receipt["reason"] == "backend_unavailable"
    assert receipt["selected"] is False


def test_adapters_declare_their_surgical_backend():
    from mlx2.adapters.flash_next import FlashNextAdapter
    from mlx2.adapters.muse_glimmer import MuseGlimmerAdapter
    from mlx2.adapters.north_mini_code import NorthMiniCodeAdapter
    from mlx2.runtime.spomin_qwen4_surgery import Qwen4SpominSurgeryBackend

    assert isinstance(MuseGlimmerAdapter.spomin_backend(None, []), StandardAttentionSpominBackend)
    assert isinstance(NorthMiniCodeAdapter.spomin_backend(None, []), StandardAttentionSpominBackend)
    assert isinstance(FlashNextAdapter.spomin_backend(None, []), Qwen4SpominSurgeryBackend)


def _engine(monkeypatch, *, receipt_status, exact_boundary=False):
    stores = []

    class APC:
        def __init__(self, **_kw):
            self.apc_stats = {}

        def key(self, *_a, **_kw):
            return "key"

        def lookup(self, _key, tokens, **_kw):
            return NS(cache=None, cached_tokens=0, remaining_tokens=list(tokens),
                      miss_reason="cold", sidecar=None, hit=False)

        def store(self, _key, tokens, *_a, **kw):
            stores.append((list(tokens), kw.get("retention_role")))
            return set()

        def spill_idle_entries(self):
            pass

        def evict_oldest_unleased(self):
            return False

        def clear(self):
            pass

    class Batch:
        scheduler_stats = {}

        def __init__(self, *_a, **kw):
            self.transform = kw.get("post_prefill_transform")
            self.step = 0

        def insert(self, *_a, **_kw):
            return [0]

        def pop_post_prefill_receipt(self, _uid):
            return {"schema": "mlx2.spomin-live-surgery.v1", "status": receipt_status,
                    "reason": "committed" if receipt_status == "applied" else "below_pressure"}

        def pop_prompt_boundary(self, _uid):
            return {
                "tokens": [1, 2, 3, 4] if exact_boundary else [1, 4],
                "target_cache": [object()],
                "committed_only": True,
                **({"pre_transform_exact": True} if exact_boundary else {}),
            }

        def pop_interior_checkpoints(self, _uid):
            return [{
                "tokens": [1],
                "target_cache": [object()],
                "covered_tokens": 1,
                "committed_only": True,
            }]

        def next(self):
            self.step += 1
            if self.step == 1:
                return [NS(uid=0, end_of_prompt=True, progress=(4, 4))], []
            if self.step == 2:
                return [], [NS(uid=0, execution_width=1, finish_reason="length", token=3,
                               mtp_state=None, all_tokens=[1, 4, 3], prompt_cache=[],
                               mtp_receipt=None)]
            return [], []

        def remove(self, _uids):
            pass

        def close(self):
            pass

    class Detok:
        last_segment = "x"

        def reset(self):
            pass

        def add_token(self, _t):
            pass

        def finalize(self):
            pass

    class Parser:
        stopped = False
        tool_count = 0

        def push(self, text, final=False):
            return [{"content": text}]

    class Adapter:
        max_context = 100000
        identity = {"fingerprint": "fake"}
        environment = {}
        layout = "fake"
        model = None
        tokenizer = NS(vocab_size=100, eos_token_ids=[], detokenizer=Detok())

        def __init__(self, _path):
            pass

        def profile_name(self, _mtp):
            return "fake"

        def execution_config(self, **_kw):
            return {"num_draft": 0}

        def prompt_tokens(self, _request):
            return [1, 2, 3, 4]

        def output_parser(self, _request):
            return Parser()

        def diagnostics(self):
            return {}

        def close(self):
            pass

    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "fake"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)
    monkeypatch.setattr(apc_v2, "APCv2", APC)
    monkeypatch.setattr(generate, "BatchGenerator", Batch)
    engine = serving.ServingEngine(
        "fake", adapter_factory=Adapter, qualification_mode=True, mtp=False,
        max_lanes=1, max_inflight=2,
        spomin_live_surgery={"enabled": True, "capacity_tokens": 4, "segment_tokens": 1},
    )
    return engine, stores


def test_compacted_state_is_never_published_to_the_exact_prefix_cache(monkeypatch):
    engine, stores = _engine(monkeypatch, receipt_status="applied")
    try:
        assert engine.ready.wait(5)
        body = {"prompt": "hi", "max_tokens": 4}
        _c, _u, receipt = collect_nonstream_job(engine.submit(body), body, chat=False)
        status = engine.status()
    finally:
        engine.close()
    assert stores == []
    assert engine.counts["apcv2_store_skipped_approximate"] == 2
    assert engine.counts["apc_interior_checkpoints_skipped_approximate"] == 1
    assert receipt["spomin_live_surgery"]["status"] == "applied"
    assert status["spomin_live_surgery"]["enabled"] is True


def test_exact_pre_surgery_boundary_is_published_for_repeat_prompt_reuse(monkeypatch):
    engine, stores = _engine(
        monkeypatch, receipt_status="applied", exact_boundary=True
    )
    try:
        assert engine.ready.wait(5)
        body = {"prompt": "hi", "max_tokens": 4}
        _c, _u, receipt = collect_nonstream_job(engine.submit(body), body, chat=False)
    finally:
        engine.close()
    assert stores == [([1, 2, 3, 4], "committed_prompt_boundary")]
    assert engine.counts["spomin_exact_boundary_stores"] == 1
    # The compacted decode tail is still approximate and is never published.
    assert engine.counts["apcv2_store_skipped_approximate"] == 1
    assert receipt["spomin_live_surgery"]["status"] == "applied"


def test_declined_surgery_keeps_exact_publication(monkeypatch):
    engine, stores = _engine(monkeypatch, receipt_status="declined")
    try:
        assert engine.ready.wait(5)
        body = {"prompt": "hi", "max_tokens": 4}
        collect_nonstream_job(engine.submit(body), body, chat=False)
    finally:
        engine.close()
    assert [role for _t, role in stores] == [
        "interior_checkpoint",
        "committed_prompt_boundary",
        None,
    ]
    # Always-on sanity telemetry is derivable from /v1/status alone.
    assert engine.counts["finish_length"] == 1 and engine.counts["completed"] == 1
    assert engine.counts["prompt_tokens"] == 4 and engine.counts["peak_observed_width"] >= 1
    assert engine.counts["apcv2_store_skipped_approximate"] == 0


def test_parallel_samples_are_refused_with_live_surgery(monkeypatch):
    engine, _stores = _engine(monkeypatch, receipt_status="declined")
    try:
        assert engine.ready.wait(5)
        with pytest.raises(ValueError, match="parallel samples are unavailable"):
            engine.submit_many([{"prompt": "hi"}, {"prompt": "hi"}])
    finally:
        engine.close()


def _north(seed=7):
    from mlx2.runtime.models.cohere2_moe import Model, ModelArgs

    mx.random.seed(seed)
    model = Model(ModelArgs(
        hidden_size=16, head_dim=4, num_hidden_layers=2, intermediate_size=8,
        prefix_dense_intermediate_size=24, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=32, num_experts=4,
        num_experts_per_tok=2, first_k_dense_replace=1, sliding_window=4,
        layer_types=["full_attention", "sliding_attention"],
    ))
    model.eval()
    return model


def _generate(model, prompts, lanes):
    from mlx2.runtime.generate import BatchGenerator

    policy = ServingSpominPolicy(enabled=True, capacity_tokens=16, segment_tokens=3)
    manager = SpominLiveSurgeryManager(
        enabled=True, backend_factory=StandardAttentionSpominBackend
    )

    def transform(*, uid, model, prompt_cache, cached_token_ids):
        ledger = policy.transcript(cached_token_ids, tokenizer_identity="t", revision=f"r{uid}")
        transaction = manager.prepare(
            request_id=str(uid), prompt_token_ids=cached_token_ids, transcript=ledger,
            capacity_tokens=16, strategy=policy.strategy, has_mtp_state=False,
            has_recurrent_state=False, cache_is_request_private=True,
            protected_segment_ids=(ledger.segments[0].segment_id,),
        )
        if transaction is None:
            return {"receipt": manager.snapshot()["recent"][-1]}
        mx.synchronize()
        receipt = transaction.apply(
            model, prompt_cache, request_quiescent=True, device_work_drained=True
        )
        if receipt["status"] != "applied":
            return {"receipt": receipt}
        return {"receipt": receipt, "retained_token_ids": transaction.retained_token_ids,
                "prompt_cache": prompt_cache}

    batch = BatchGenerator(
        model, completion_batch_size=lanes, prefill_batch_size=1,
        prefill_step_size=32, post_prefill_transform=transform,
    )
    uids = batch.insert(prompts, max_tokens=[8] * len(prompts))
    tokens, receipts = {uid: [] for uid in uids}, {}
    for _ in range(200):
        prompt_responses, responses = batch.next()
        for response in prompt_responses:
            if response.end_of_prompt:
                receipts[response.uid] = batch.pop_post_prefill_receipt(response.uid)
        for response in responses:
            tokens[response.uid].append(response.token)
        if all(len(row) >= 8 for row in tokens.values()):
            break
    return [tokens[uid] for uid in uids], [receipts[uid] for uid in uids]


def test_compacted_lane_decodes_and_batches_with_an_exact_lane():
    """Full + sliding layers after a chunked prefill; decode past the boundary."""
    model = _north()
    long_prompt, short_prompt = list(range(1, 14)), [5, 6, 7]
    (alone,), (receipt,) = _generate(model, [long_prompt], 1)
    assert receipt["status"] == "applied", receipt
    assert receipt["source_tokens"] == 12 and receipt["retained_tokens"] == 9
    (short_alone,), (short_receipt,) = _generate(model, [short_prompt], 1)
    assert short_receipt["reason"] == "below_pressure"
    together, receipts = _generate(model, [long_prompt, short_prompt], 2)
    assert [r["status"] for r in receipts] == ["applied", "declined"]
    assert together == [alone, short_alone]
    assert len(alone) == 8


def test_serving_reuses_exact_pre_surgery_boundary_on_repeated_prompt(monkeypatch):
    """The exact shadow is reusable; compacted state never becomes the APC key."""
    class Detok:
        def __init__(self):
            self.last_segment = ""

        def reset(self):
            self.last_segment = ""

        def add_token(self, token):
            self.last_segment = f"{int(token)} "

        def finalize(self):
            pass

    class Parser:
        stopped = False
        tool_count = 0

        def push(self, text, final=False):
            return [{"content": text}] if text else []

    class Tokenizer:
        vocab_size = 32
        eos_token_ids = []

        @property
        def detokenizer(self):
            return Detok()

    class Adapter:
        max_context = 256
        identity = {"fingerprint": "tiny-north-spomin"}
        environment = {}
        layout = "tiny-north-layout"
        tokenizer = Tokenizer()

        def __init__(self, _path):
            # MLX CPU streams are thread-local; construct the tiny model on the
            # serving worker that owns and executes it.
            self.model = _north(seed=11)

        def profile_name(self, _mtp):
            return "tiny-north-ordinary"

        def execution_config(self, **_kwargs):
            return {"num_draft": 0}

        def prompt_tokens(self, request):
            return list(request["tokens"])

        def output_parser(self, _request):
            return Parser()

        def spomin_backend(self, model, prompt_cache):
            return StandardAttentionSpominBackend(model, prompt_cache)

        def diagnostics(self):
            return {}

        def close(self):
            pass

    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "src"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)
    engine = serving.ServingEngine(
        "tiny",
        adapter_factory=Adapter,
        qualification_mode=True,
        mtp=False,
        max_lanes=2,
        spomin_live_surgery={
            "enabled": True,
            "capacity_tokens": 16,
            "segment_tokens": 3,
        },
    )
    try:
        assert engine.ready.wait(5)
        body = {"tokens": list(range(1, 14)), "max_tokens": 2, "temperature": 0}
        _c1, _u1, first = collect_nonstream_job(
            engine.submit(body), body, chat=False
        )
        _c2, _u2, second = collect_nonstream_job(
            engine.submit(body), body, chat=False
        )
    finally:
        engine.close()
    assert first["spomin_live_surgery"]["status"] == "applied"
    assert second["spomin_live_surgery"]["status"] == "applied"
    assert first["cached_tokens"] == 0
    assert second["cached_tokens"] == len(body["tokens"]) - 1
    assert engine.counts["spomin_exact_boundary_stores"] == 2
    assert engine.counts["apcv2_store_skipped_approximate"] == 2
