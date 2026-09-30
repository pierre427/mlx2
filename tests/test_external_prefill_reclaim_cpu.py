"""CPU tests for the default-off external-draft prefill allocator reclaim.

``ExternalDraftBatchGenerator(prefill_allocator_reclaim=True)`` clears the MLX
pool after each non-empty prefill chunk, once that chunk's target and draft
state are materialized (the ordinary and prompt-lookup prefill contract). It
is a direct-model candidate: no serving policy selects it, and the default
keeps the receipt key set and the decode reclaim cadence unchanged.
"""

import mlx.core as mx
import pytest

from scripts.paired_direct_ab import state_digest
from test_external_dflash2_cpu import generator, tiny


def _spy_clears(monkeypatch, events=None):
    calls = {"n": 0}
    real = mx.clear_cache

    def clear():
        calls["n"] += 1
        if events is not None:
            events.append("clear")
        real()

    monkeypatch.setattr(mx, "clear_cache", clear)
    return calls


def _prefill_all(gen, prompt, **insert):
    uid = gen.insert([prompt], max_tokens=[4], **insert)[0]
    lane = gen.lanes[uid]
    while lane.anchor is None:
        gen._prefill(lane)
    return uid, lane


def test_default_is_off_and_keeps_the_receipt_key_set(monkeypatch):
    model, draft = tiny()
    off = generator(model, draft)
    on = generator(model, draft, prefill_allocator_reclaim=True)
    try:
        assert off.prefill_allocator_reclaim is False
        assert "external_prefill_allocator_reclaims" not in off.scheduler_stats
        assert set(on.scheduler_stats) - set(off.scheduler_stats) == {
            "external_prefill_allocator_reclaims"}
        calls = _spy_clears(monkeypatch)
        _prefill_all(off, list(range(1, 11)))
        assert calls["n"] == 0
    finally:
        off.close()
        on.close()
    with pytest.raises(ValueError, match="boolean"):
        generator(model, draft, prefill_allocator_reclaim=1)


@pytest.mark.parametrize("length,chunks", [(1, 0), (2, 1), (4, 1), (5, 2), (10, 3)])
def test_one_clear_per_nonempty_chunk(monkeypatch, length, chunks):
    """prefill_step_size=3; the final prompt token is left for decode."""
    model, draft = tiny()
    gen = generator(model, draft, prefill_allocator_reclaim=True)
    try:
        calls = _spy_clears(monkeypatch)
        _uid, lane = _prefill_all(gen, list(range(1, length + 1)))
        gen._prefill(lane)  # a done lane: no chunk, no clear
        assert gen.scheduler_stats["prefill_rounds"] == chunks
        assert gen.scheduler_stats["external_prefill_allocator_reclaims"] == chunks
        assert calls["n"] == chunks
        assert gen.scheduler_stats["external_allocator_reclaims"] == 0
    finally:
        gen.close()


def test_clear_follows_the_chunk_materialization(monkeypatch):
    model, draft = tiny()
    gen = generator(model, draft, prefill_allocator_reclaim=True)
    events = []
    real_eval = mx.eval

    def eval_(*args):
        events.append("eval")
        return real_eval(*args)

    try:
        monkeypatch.setattr(mx, "eval", eval_)
        _spy_clears(monkeypatch, events)
        _prefill_all(gen, list(range(1, 11)))
        clears = [i for i, e in enumerate(events) if e == "clear"]
        assert len(clears) == 3
        assert all(events[i - 1] == "eval" for i in clears)
    finally:
        gen.close()


def _run(reclaim, prompt, *, max_tokens=6, factory=None, **insert):
    model, draft = (factory or tiny)()
    make = generator
    if factory is not None:
        from test_cohere_eagle_cpu import generator as make
    gen = make(model, draft, prefill_allocator_reclaim=reclaim)
    try:
        uid = gen.insert([prompt], max_tokens=[max_tokens],
                         sampling_configs=[{"sampling_temp": 0}], **insert)[0]
        tokens, final, boundary = [], None, None
        for _ in range(400):
            _, responses = gen.next()
            if boundary is None and uid in gen.boundaries:
                boundary = gen.boundaries[uid]
                boundary = {
                    "tokens": list(boundary["tokens"]),
                    "covered": boundary["covered_tokens"],
                    "target": state_digest(boundary["target_cache"]),
                    "sidecar": state_digest(boundary["cache_sidecar"]),
                }
            for response in responses:
                tokens.append(int(response.token))
                if response.finish_reason:
                    final = response
            if not gen.lanes:
                break
        stats = dict(gen.scheduler_stats)
    finally:
        gen.close()
    return {
        "tokens": tokens,
        "boundary": boundary,
        "target": state_digest(final.prompt_cache),
        "sidecar": state_digest(final.cache_sidecar),
        "stats": stats,
    }


def _assert_parity(on, off):
    assert on["tokens"] == off["tokens"] and on["tokens"]
    assert on["boundary"] == off["boundary"] and on["boundary"] is not None
    assert on["boundary"]["target"]["status"] == "complete"
    assert on["boundary"]["sidecar"]["status"] == "complete"
    assert on["target"] == off["target"] and on["target"]["status"] == "complete"
    assert on["sidecar"] == off["sidecar"] and on["sidecar"]["status"] == "complete"
    assert on["stats"]["external_allocator_reclaims"] == off["stats"]["external_allocator_reclaims"]
    for key in ("prefill_rounds", "external_rounds", "proposed_tokens", "accepted_proposals"):
        assert on["stats"][key] == off["stats"][key], key


def test_real_clear_keeps_boundary_tokens_and_state_exact():
    prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    on, off = _run(True, prompt), _run(False, prompt)
    _assert_parity(on, off)
    assert on["stats"]["external_prefill_allocator_reclaims"] == on["stats"]["prefill_rounds"] == 4


def test_warm_resume_with_a_pending_tail_is_exact():
    """A paired APCv2-style resume prefills its suffix in chunks; the
    carried draft tail is appended at the first chunk, before the clear."""

    def resumed(reclaim):
        # A fresh, deterministic source run per arm: independent resume state.
        model, draft = tiny()
        source = generator(model, draft)
        try:
            uid = source.insert([[1, 2, 3, 4, 5]], max_tokens=[4],
                                sampling_configs=[{"sampling_temp": 0}])[0]
            end = None
            while source.lanes:
                for response in source.next()[1]:
                    if response.finish_reason:
                        end = response
        finally:
            source.close()
        assert end.cache_sidecar.state[1] is not None
        return _run(reclaim, [end.token, 7, 8, 9, 10, 11, 12], caches=[end.prompt_cache],
                    all_tokens=[end.all_tokens], cache_states=[end.cache_sidecar])

    on, off = resumed(True), resumed(False)
    _assert_parity(on, off)
    assert on["stats"]["paired_cache_resumes"] == off["stats"]["paired_cache_resumes"] == 1
    assert on["stats"]["external_prefill_allocator_reclaims"] == on["stats"]["prefill_rounds"] >= 2


def test_pairing_drafter_keeps_its_last_chunk_pending():
    from test_cohere_eagle_cpu import tiny as eagle_tiny

    prompt = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    on = _run(True, prompt, factory=eagle_tiny)
    off = _run(False, prompt, factory=eagle_tiny)
    _assert_parity(on, off)
    assert on["stats"]["external_context_token_pairings"] == off["stats"]["external_context_token_pairings"] > 0
    assert on["stats"]["external_prefill_allocator_reclaims"] == on["stats"]["prefill_rounds"] > 0


def test_no_serving_or_adapter_code_selects_the_candidate():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "mlx2"
    users = sorted(
        str(path.relative_to(root)) for path in root.rglob("*.py")
        if "prefill_allocator_reclaim" in path.read_text()
    )
    assert users == ["runtime/external_speculative.py"]
