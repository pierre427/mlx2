"""Opt-in: native-MTP drafting skips a thinking guard that cannot act yet.

omlx#4926 measured 11-14% decode lost to a thinking budget that never fires.
On mlx2's native-MTP lane an idle guard made every drafted position probe it
(one host read of the lazily drafted tokens per position) and forced the
full-vocab draft head.  ``MLX2_MTP_DRAFT_SKIP_PASSIVE_PROCESSORS=1`` lets
drafting skip processors reporting ``draft_passive()``; verification still
applies them, so committed tokens and guard receipts must equal the policy-off
engine.  CPU tiny-checkpoint tests only; the decode-time effect is unmeasured.
"""

import mlx.core as mx

from mlx2.runtime import hybrid_speculative
from mlx2.thinking_guard import ThinkingGuard

CLOSE, VOCAB = 7, 16


def _feed(guard, generated, prompt=(1, 2)):
    guard(mx.array(list(prompt) + list(generated), dtype=mx.uint32), mx.zeros((1, VOCAB)))


def test_draft_passive_tracks_when_the_guard_can_change_a_row():
    guard = ThinkingGuard(2, (CLOSE,), budget=200, soft_ratio=0.5)  # soft at 100
    _feed(guard, list(range(8, 16)))
    assert guard.draft_passive(margin=32)
    _feed(guard, [8 + (i * 5) % 8 + (i % 3) for i in range(70)])
    assert not guard.draft_passive(margin=32)  # the soft budget is within reach
    no_budget = ThinkingGuard(2, (CLOSE,), budget=None)
    _feed(no_budget, list(range(8, 16)))
    assert no_budget.draft_passive()
    tripped = ThinkingGuard(2, (CLOSE,), budget=4000, tau=2.5, ngram=6)
    _feed(tripped, [8, 9, 10, 11, 12, 13, 14, 15] * 8)
    assert tripped.receipt()["tripped"] == "run_on" and not tripped.draft_passive()
    _feed(tripped, [8, 9, 10, CLOSE, 3, 4])
    assert tripped.draft_passive()  # reasoning closed: the guard is out of the way


def test_only_passive_processors_are_skipped(monkeypatch):
    guard = ThinkingGuard(2, (CLOSE,), budget=None)
    _feed(guard, [8, 9])
    other = lambda tokens, logits: logits  # noqa: E731 - no draft_passive
    monkeypatch.delenv(hybrid_speculative.DRAFT_PASSIVE_PROCESSORS_ENV, raising=False)
    assert hybrid_speculative._draft_constrained([guard])
    monkeypatch.setenv(hybrid_speculative.DRAFT_PASSIVE_PROCESSORS_ENV, "1")
    assert not hybrid_speculative._draft_constrained([guard])
    assert hybrid_speculative._draft_constrained([guard, other])
    assert not hybrid_speculative._draft_constrained([])


def test_native_mtp_with_an_idle_guard_commits_the_same_stream(monkeypatch):
    from route_harness import make_engine, patch_host, run, tiny_qwen38_mtp

    patch_host(monkeypatch)
    model, vocab = tiny_qwen38_mtp()
    prompt = [(7 * i + 3) % (vocab - 2) + 1 for i in range(40)]
    requests = [
        {"messages": [{"role": "user", "content": "x"}], "tokens": prompt,
         "max_tokens": 30, "temperature": temperature, "seed": 3,
         "thinking_budget": budget}
        # 4000 never fires; 8 trips inside the run.
        for budget in (4000, 8) for temperature in (0, 0.8)
    ]
    outputs = {}
    for arm in ("off", "on"):
        if arm == "on":
            monkeypatch.setenv(hybrid_speculative.DRAFT_PASSIVE_PROCESSORS_ENV, "1")
        else:
            monkeypatch.delenv(hybrid_speculative.DRAFT_PASSIVE_PROCESSORS_ENV, raising=False)
        before = dict(hybrid_speculative.DRAFT_PASSIVE_STATS)
        engine = make_engine(model, vocab, eos=(24,), close_id=99, mtp=True)
        try:
            outputs[arm] = [run(engine, dict(request)) for request in requests]
        finally:
            engine.close()
        outputs[arm + "_skips"] = (
            hybrid_speculative.DRAFT_PASSIVE_STATS["skipped_draft_rows"]
            - before["skipped_draft_rows"]
        )
    assert outputs["off_skips"] == 0 and outputs["on_skips"] > 0
    lane_skips = lambda result: result["receipt"]["mtp"]["stats"]["draft_passive_processor_skips"]  # noqa: E731
    assert all(lane_skips(result) == 0 for result in outputs["off"])
    assert any(lane_skips(result) > 0 for result in outputs["on"])
    controls = "request_controls"
    for request, off, on in zip(requests, outputs["off"], outputs["on"]):
        expected = None if request["thinking_budget"] == 4000 else "budget_soft"
        assert off["receipt"][controls]["thinking_guard"]["tripped"] == expected
        if request["temperature"] == 0:
            assert on["tokens"] == off["tokens"], request
            assert on["receipt"][controls]["thinking_guard"] == off["receipt"][controls]["thinking_guard"]
