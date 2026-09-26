"""Qwen3.8 27B + DFlash2 external draft on tiny CPU models; no artifact loads.

The hybrid target (gated-delta recurrent layers + full attention) verifies
through ``HybridVerifyRows``.  Exactness is checked against ordinary greedy
decode of the same target, with a real random drafter and with an oracle
drafter that injects a wrong token at every position of the block in turn,
so every accept length 0..K and every rollback depth is exercised.
"""
import numpy as np
import pytest

import mlx.core as mx

mx.set_default_device(mx.cpu)

from mlx2.runtime.drafters.dflash2 import DFlash2DraftModel
from mlx2.runtime.drafters.dflash2_config import DFlash2Config
from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

VOCAB = 128


def tiny_target(seed=7):
    from mlx2.runtime.models.qwen38_27b import Model, ModelArgs

    text = dict(
        model_type="qwen3_5", hidden_size=32, intermediate_size=64,
        num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, vocab_size=VOCAB, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=4, full_attention_interval=4,
        mtp_num_hidden_layers=0, partial_rotary_factor=0.5,
        rope_parameters=None, max_position_embeddings=512,
    )
    mx.random.seed(seed)
    model = Model(ModelArgs(model_type="qwen3_5", text_config=text))
    model.eval()
    mx.eval(model.parameters())
    return model


def tiny_draft_args(**overrides):
    values = dict(
        hidden_size=32, intermediate_size=48, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        vocab_size=VOCAB, num_target_layers=8, target_layer_ids=[1, 6],
        conv_kernel_size=2, conv_group_size=4, selector_rank=8, selector_top_k=6,
        block_size=8, mask_token_id=VOCAB - 1, max_position_embeddings=512,
        sliding_window=6, layer_types=["sliding_attention"] * 2,
    )
    values.update(overrides)
    return DFlash2Config(**values)


def tiny_pair(seed=7):
    target = tiny_target(seed)
    mx.random.seed(seed + 1)
    draft = DFlash2DraftModel(tiny_draft_args()).bind(target)
    mx.eval(draft.parameters())
    return target, draft


def generator(target, draft, num_draft=3, **kwargs):
    return ExternalDraftBatchGenerator(
        target, draft_model=draft, binding="test", num_draft=num_draft,
        prefill_step_size=4, **kwargs,
    )


def drain(batch):
    output, final = {}, {}
    for _ in range(400):
        _, responses = batch.next()
        for response in responses:
            output.setdefault(response.uid, []).append(response.token)
            if response.finish_reason:
                final[response.uid] = response
        if not batch.lanes:
            return output, final
    raise AssertionError("scheduler stalled")


def ordinary_greedy(target, prompt, count):
    cache = target.make_cache()
    tokens, out = list(prompt), []
    for step in range(count):
        logits = target(mx.array([tokens if step == 0 else [tokens[-1]]]), cache=cache)
        token = int(mx.argmax(logits[0, -1]).item())
        tokens.append(token)
        out.append(token)
    return out


PROMPTS = [[3, 9, 27, 81, 15, 44, 2], [5, 6], [70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]]


def test_target_taps_are_post_block_residuals_and_logits_unchanged():
    target = tiny_target()
    x = mx.array([[1, 2, 3, 4, 5]])
    logits, taps = target.forward_with_taps(x, target.make_cache(), [1, 6])
    ordinary = target(x, cache=target.make_cache())
    np.testing.assert_allclose(np.asarray(logits), np.asarray(ordinary), atol=1e-5)
    assert taps.shape == (1, 5, 64)
    # Tap 1 is the residual stream after decoder layer 1: recompute by hand.
    inner = target.language_model.model
    h = inner.embed_tokens(x)
    for index, layer in enumerate(inner.layers[:2]):
        h = layer(h, mask=None, cache=None)
    np.testing.assert_allclose(np.asarray(taps[..., :32]), np.asarray(h), atol=1e-5)
    body = target.prefill_body(x, target.make_cache(), [1, 6])
    np.testing.assert_allclose(np.asarray(body), np.asarray(taps), atol=1e-6)
    with pytest.raises(ValueError, match="capture layers"):
        target.forward_with_taps(x, target.make_cache(), [6, 1])
    with pytest.raises(ValueError, match="capture layers"):
        target.forward_with_taps(x, target.make_cache(), [8])


def test_split_prefill_taps_match_one_shot():
    target = tiny_target()
    x = mx.array([[1, 2, 3, 4, 5, 6, 7]])
    whole = target.prefill_body(x, target.make_cache(), [1, 6])
    cache = target.make_cache()
    left = target.prefill_body(x[:, :3], cache, [1, 6])
    right = target.prefill_body(x[:, 3:], cache, [1, 6])
    np.testing.assert_allclose(
        np.asarray(mx.concatenate([left, right], axis=1)), np.asarray(whole), atol=1e-5
    )


@pytest.mark.parametrize("num_draft", [1, 3, 7])
def test_random_drafter_greedy_equals_ordinary_b1(num_draft):
    target, draft = tiny_pair()
    batch = generator(target, draft, num_draft=num_draft)
    uid = batch.insert([PROMPTS[0]], max_tokens=[12], sampling_configs=[{"sampling_temp": 0}])[0]
    got, final = drain(batch)
    assert got[uid] == ordinary_greedy(target, PROMPTS[0], 12)
    assert batch.scheduler_stats["external_hybrid_transactions"] > 0
    final[uid].cache_sidecar.validate("test", len(final[uid].all_tokens))


def _install_oracle(monkeypatch, batch, target, prompts, budget):
    """Propose the true greedy continuation with one injected error per round.

    The error position cycles 0..K so accept lengths 0..K-1 and K (no error)
    all occur; the laws are one-hot, as a greedy drafter's are.
    """
    references = {}
    rounds = {"n": 0, "accepts": set()}

    def reference(history):
        key = tuple(history)
        for prefix, full in references.items():
            if key[: len(prefix)] == prefix and full[: len(key)] == list(key):
                return full
        return None

    for prompt in prompts:
        references[tuple(prompt)] = list(prompt) + ordinary_greedy(target, prompt, budget + 8)

    def oracle(anchors, hidden, cache, count, rngs, temperatures, **kwargs):
        # The real drafter still consumes context so its plane stays paired.
        draft.draft_distributions_real(anchors, hidden, cache, count, rngs, temperatures)
        tokens, laws = [], []
        for anchor in anchors:
            lane = next(l for l in batch.lanes.values() if l.anchor == int(anchor))
            full = reference(lane.history + [lane.anchor])
            start = len(lane.history) + 1
            proposal = list(full[start:start + count])
            wrong = rounds["n"] % (count + 1)
            if wrong < count:
                proposal[wrong] = (proposal[wrong] + 1) % (VOCAB - 1)
            rounds["n"] += 1
            rows = []
            for token in proposal:
                law = np.zeros(VOCAB)
                law[token] = 1.0
                rows.append(law)
            tokens.append(proposal)
            laws.append(rows)
        return tokens, laws

    draft = batch.draft
    draft.draft_distributions_real = draft.draft_distributions
    monkeypatch.setattr(draft, "draft_distributions", oracle)


@pytest.mark.parametrize("lanes", [1, 2, 3])
def test_oracle_drafter_every_accept_length_equals_ordinary(monkeypatch, lanes):
    target, draft = tiny_pair()
    batch = generator(target, draft, num_draft=4)
    prompts = PROMPTS[:lanes]
    _install_oracle(monkeypatch, batch, target, prompts, 20)
    uids = batch.insert(prompts, max_tokens=[20] * lanes, sampling_configs=[{"sampling_temp": 0}] * lanes)
    got, final = drain(batch)
    for uid, prompt in zip(uids, prompts):
        assert got[uid] == ordinary_greedy(target, prompt, 20)
        receipt = final[uid].speculative_receipt
        # Every accept length 0..4 was produced by the injected schedule.
        assert set(receipt["verify_accept_hist"]) >= {0, 1, 2, 3} or lanes > 1
        assert receipt["accepted"] > 0
    stats = batch.scheduler_stats
    assert stats["segmented_rollbacks"] > 0 and stats["accepted_proposals"] > 0
    # Ready-queue drains desynchronize lanes, so a cohort need not reach
    # every lane at once; it must still batch hybrid rows.
    if lanes == 2:
        assert stats["target_max_width"] == 2


def test_hybrid_rows_leave_no_rollback_records_after_commit():
    from mlx2.runtime.hybrid_verify_rows import HybridVerifyRows
    from mlx2.runtime.models.cache import ArraysCache

    target = tiny_target()
    rows = []
    for prompt in PROMPTS[:2]:
        cache = target.make_cache()
        target(mx.array([prompt]), cache=cache)
        rows.append(cache)
    transaction = HybridVerifyRows(rows).begin([4, 2])
    inputs = mx.array([[1, 2, 3, 4], [5, 6, 0, 0]])
    logits, _ = target.forward_with_taps(inputs, transaction.caches, [1, 6])
    mx.eval(logits)
    committed = transaction.commit([2, 1])
    for row, prompt, kept in zip(committed, PROMPTS[:2], [2, 1]):
        for cache in row:
            if isinstance(cache, ArraysCache):
                assert not cache.speculating and not cache._rollbacks
            else:
                assert cache.offset == len(prompt) + kept
    # The committed recurrent rows equal a fresh forward over the kept prefix.
    for row, prompt, extra in zip(committed, PROMPTS[:2], [[1, 2], [5]]):
        fresh = target.make_cache()
        target(mx.array([prompt + extra]), cache=fresh)
        for mine, ref in zip(row, fresh):
            if isinstance(mine, ArraysCache):
                for a, b in zip(mine.cache, ref.cache):
                    np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-5)


def test_hybrid_rows_refuse_unsupported_topologies():
    from mlx2.runtime.hybrid_verify_rows import HybridVerifyRows, is_hybrid_rows
    from mlx2.runtime.models.cache import RotatingKVCache

    target = tiny_target()
    cache = target.make_cache()
    assert is_hybrid_rows([cache])
    assert not is_hybrid_rows([[RotatingKVCache(max_size=4)]])
    mixed = list(cache)
    mixed[3] = RotatingKVCache(max_size=4)
    assert not is_hybrid_rows([mixed])
    with pytest.raises(ValueError, match="independent"):
        HybridVerifyRows([cache, cache])
    transaction = HybridVerifyRows([cache]).begin([2])
    with pytest.raises(RuntimeError, match="leased"):
        transaction.owner.begin([2])
    with pytest.raises(ValueError, match="exceeds"):
        transaction.commit([3])


def test_target_without_rollback_declaration_fails_closed(monkeypatch):
    from mlx2.runtime.models.qwen38_27b import Model

    target, draft = tiny_pair()
    monkeypatch.setattr(Model, "supports_speculative_rollback", False)
    batch = generator(target, draft)
    batch.insert([PROMPTS[1]], max_tokens=[4], sampling_configs=[{"sampling_temp": 0}])
    with pytest.raises(ValueError, match="speculative rollback"):
        drain(batch)


def test_sampled_route_runs_and_is_seed_deterministic():
    outputs = []
    for _ in range(2):
        target, draft = tiny_pair()
        batch = generator(target, draft, num_draft=3, pairwise_selection="batched")
        uid = batch.insert(
            [PROMPTS[0]], max_tokens=[10],
            sampling_configs=[{"sampling_temp": 0.7, "top_p": 0.95, "top_k": 20}],
        )[0]
        got, _ = drain(batch)
        outputs.append(got[uid])
    assert outputs[0] == outputs[1] and len(outputs[0]) == 10


def test_batched_pairwise_greedy_equals_ordinary_b2():
    target, draft = tiny_pair()
    batch = generator(target, draft, num_draft=5, pairwise_selection="batched")
    uids = batch.insert(PROMPTS[:2], max_tokens=[9, 9], sampling_configs=[{"sampling_temp": 0}] * 2)
    got, _ = drain(batch)
    for uid, prompt in zip(uids, PROMPTS[:2]):
        assert got[uid] == ordinary_greedy(target, prompt, 9)


@pytest.mark.parametrize("mode", ["one", "all"])
def test_ready_drain_modes_are_exact_and_all_keeps_lanes_in_lockstep(monkeypatch, mode):
    target, draft = tiny_pair()
    batch = generator(target, draft, num_draft=4, ready_drain=mode)
    prompts = [PROMPTS[0], [11, 12, 13, 14, 15, 16, 17]]
    _install_oracle(monkeypatch, batch, target, prompts, 24)
    widths = []
    real_round = batch._round

    def record(cohort):
        widths.append(len(cohort))
        return real_round(cohort)

    monkeypatch.setattr(batch, "_round", record)
    uids = batch.insert(prompts, max_tokens=[24, 24], sampling_configs=[{"sampling_temp": 0}] * 2)
    got, _ = drain(batch)
    for uid, prompt in zip(uids, prompts):
        assert got[uid] == ordinary_greedy(target, prompt, 24)
    start = widths.index(2)
    lockstep = all(a >= b for a, b in zip(widths[start:], widths[start + 1:]))
    if mode == "all":
        # Once both lanes decode, they share every round until one finishes.
        assert lockstep, widths
    else:
        # A lane draining a longer round sits out the other's next round.
        assert not lockstep, widths


def test_ready_drain_rejects_unknown_mode():
    target, draft = tiny_pair()
    with pytest.raises(ValueError, match="ready_drain"):
        generator(target, draft, ready_drain="some")


# --- served route on the real ServingEngine (CPU, tiny models) -------------


def _serving_adapter(target, draft, *, external, vocab=VOCAB):
    from mlx2.contracts import Capability

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
        vocab_size = vocab
        eos_token_ids = []

        @property
        def detokenizer(self):
            return Detok()

    class Adapter:
        max_context = 512
        identity = {"fingerprint": "tiny-qwen38-dflash2" if external else "tiny-qwen38"}
        environment = {}
        layout = "tiny-qwen38-layout"
        tokenizer = Tokenizer()

        def __init__(self, _path):
            self.model = target
            self.draft_model = draft if external else None

        def profile_name(self, mtp):
            return "tiny-dflash2" if external else "tiny-ordinary"

        def execution_config(self, *, max_lanes, prefill_step):
            config = {"persistent": True, "num_draft": 0, "rate_gate": False,
                      "prefill_step_size": prefill_step}
            if external:
                config.update(num_draft=4, backend="external_draft")
            return config

        def create_external_batch(self, **kwargs):
            return ExternalDraftBatchGenerator(
                self.model, draft_model=self.draft_model,
                binding=self.identity["fingerprint"], num_draft=4,
                pairwise_selection="batched", ready_drain="all", **kwargs,
            )

        def prompt_tokens(self, request):
            return list(request["tokens"])

        def output_parser(self, _request):
            return Parser()

        def diagnostics(self):
            return {}

        def close(self):
            pass

    return Adapter


def _serve(adapter, requests, **kw):
    import threading

    from mlx2.serving import ServingEngine

    engine = ServingEngine("tiny", adapter_factory=adapter, qualification_mode=True,
                           mtp=False, max_lanes=4, prefill_step=8, **kw)
    assert engine.ready.wait(60), engine.error
    results = [None] * len(requests)

    def one(index, tokens):
        job = engine.submit({"tokens": list(tokens), "max_tokens": 16, "temperature": 0})
        text = ""
        while True:
            event = job.events.get(timeout=300)
            assert "error" not in event, event
            if "delta" in event:
                text += event["delta"].get("content", "")
            if "finish_reason" in event:
                results[index] = ([int(t) for t in text.split()], event.get("receipt") or {}, job)
                return

    try:
        threads = [threading.Thread(target=one, args=pair) for pair in enumerate(requests)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        status = engine.status()
    finally:
        engine.close()
    return results, status


@pytest.fixture
def host(monkeypatch):
    from mlx2 import memory, serving
    from mlx2.runtime import os_memory

    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "src"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)


def test_served_external_route_b1_b3_equals_ordinary_and_reports_dflash2(host):
    target, draft = tiny_pair()
    prompts = [[(7 * i + 3) % 120 + 1 for i in range(20)], PROMPTS[0], PROMPTS[2]]
    ordinary, _ = _serve(_serving_adapter(target, None, external=False), prompts[:1])
    served, _ = _serve(_serving_adapter(target, draft, external=True), prompts[:1])
    assert served[0][0] == ordinary[0][0] == ordinary_greedy(target, prompts[0], 16)
    receipt = served[0][1]
    assert receipt["speculation"]["kind"] == "external_dflash2"
    assert receipt["speculation"]["external_rounds"] > 0
    batched, status = _serve(_serving_adapter(target, draft, external=True), prompts)
    for (tokens, receipt, _), prompt in zip(batched, prompts):
        assert tokens == ordinary_greedy(target, prompt, 16)
        assert receipt["speculation"]["kind"] == "external_dflash2"


def test_served_external_warm_prefix_hit_equals_cold(host):
    target, draft = tiny_pair()
    prompt = [(5 * i + 2) % 120 + 1 for i in range(40)]
    extended = prompt + [9, 8, 7, 6, 5]
    adapter = _serving_adapter(target, draft, external=True)
    from mlx2.serving import ServingEngine

    engine = ServingEngine("tiny", adapter_factory=adapter, qualification_mode=True,
                           mtp=False, max_lanes=1, prefill_step=8)
    assert engine.ready.wait(60), engine.error
    outputs = []
    try:
        for tokens in (prompt, extended):
            job = engine.submit({"tokens": tokens, "max_tokens": 12, "temperature": 0})
            text = ""
            while True:
                event = job.events.get(timeout=300)
                assert "error" not in event, event
                if "delta" in event:
                    text += event["delta"].get("content", "")
                if "finish_reason" in event:
                    outputs.append(([int(t) for t in text.split()], job))
                    break
    finally:
        engine.close()
    assert outputs[1][0] == ordinary_greedy(target, extended, 12)
    assert outputs[0][0] == ordinary_greedy(target, prompt, 12)
    # The second request resumed the paired target + draft boundary.
    assert int(outputs[1][1].cached_tokens or 0) > 0
