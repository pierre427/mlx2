"""Prompt-lookup admission seats a lane at the widest verify span that fits.

Series 2026-09-24, Qwen3.8-27B-MLX-4bit --prompt-lookup on a 36 GiB M3 Pro:
every request -- the first on an idle server included -- waited out its
60 s admission deadline and answered 429 (39 of 43 feature checks), while
the ordinary route on the same host served them.  No grant was ever held
(``memory_admission_deferred_behind_grants`` never appeared); the full
num_draft=8 verify term alone was 3.1 * 8 / 3 = 8.27 GiB, so a 111-token
chat lane needed 13.66 GiB against the 11-13 GiB the idle host measured.
"""

import time

import mlx.core as mx
import pytest

from mlx2 import memory, serving
from mlx2.runtime.memory_policy import SelfMTPLaneAdmissionController as C
from mlx2.runtime.pld import PromptLookupBatchGenerator
from route_harness import collect, make_engine, patch_host, tiny_qwen38_mtp

GIB = 1 << 30
M3 = dict(host_memory_gib=36.0, advisory_gib=28.08)


def _qwen38_m3():
    from mlx2.adapters.qwen38_memory import Qwen38CacheBudget

    budget = Qwen38CacheBudget(
        attention_layers=16, recurrent_layers=48, mtp_layers=0, kv_heads=4,
        head_dim=256, recurrent_heads=48, recurrent_key_heads=16,
        recurrent_key_dim=128, recurrent_value_dim=128, conv_kernel=4,
    )
    return C(**M3, cache_estimator=budget.project,
             transient_gib_per_lane=budget.transient_gib_per_lane,
             saturation_lane_cap=4, verification_row_cap=36)


def _admit(controller, headroom_gib):
    return serving.admit_prompt_lookup_lane(
        controller, context_tokens=111, cache_gib=0.0, num_draft=8,
        headroom=lambda: int(headroom_gib * GIB), reclaim=lambda: False,
        evict=lambda: False, evictable=lambda: 0,
    )


def test_m3_chat_lane_is_seated_below_the_full_span():
    controller = _qwen38_m3()
    full = serving.lane_admission_required_gib(
        controller, context_tokens=111, draft_depth=0, cache_gib=0.0,
        prompt_lookup_num_draft=8,
    )
    assert full == pytest.approx(13.66, abs=0.01)
    # Idle M3: min(psutil available, advisory - footprint) was 11-13 GiB.
    admitted, span, required = _admit(controller, 12.0)
    assert admitted and span == 6
    assert required == pytest.approx(5.39 + 6 * 3.1 / 3, abs=0.01)
    admitted, span, required = _admit(controller, 6.0)
    assert admitted and span == 0 and required == pytest.approx(5.39, abs=0.01)
    admitted, span, _required = _admit(controller, 14.0)
    assert admitted and span == 8
    admitted, _span, _required = _admit(controller, 5.0)
    assert not admitted


def _north(seed=5):
    from mlx2.runtime.models.cohere2_moe import Model, ModelArgs

    mx.random.seed(seed)
    model = Model(ModelArgs(
        hidden_size=16, head_dim=4, num_hidden_layers=4, intermediate_size=8,
        prefix_dense_intermediate_size=24, num_attention_heads=4,
        num_key_value_heads=2, vocab_size=32, num_experts=4,
        num_experts_per_tok=2, first_k_dense_replace=1, sliding_window=6,
        layer_types=["full_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
    ))
    model.eval()
    return model


@pytest.mark.parametrize("cap", [0, 2])
def test_memory_cap_bounds_every_verify_forward_and_stays_exact(cap):
    model = _north()
    prompt = [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1, 2, 3]

    def run(configs):
        generator = PromptLookupBatchGenerator(
            model, completion_batch_size=1, prefill_step_size=5,
            prompt_lookup={"num_draft": 4, "ngram_min": 2, "ngram_max": 3,
                           "adaptive": False, "deferred_admission": False},
        )
        (uid,) = generator.insert([prompt], max_tokens=[24], prompt_lookup_configs=configs)
        tokens, final = [], None
        for _ in range(500):
            _prompts, responses = generator.next()
            for response in responses:
                tokens.append(response.token)
                final = response if response.finish_reason else final
            if final is not None:
                return tokens, final
        raise AssertionError("lane did not finish")

    free, uncapped = run(None)
    assert max(map(int, uncapped.speculative_receipt["verify_span_hist"])) > cap + 1
    capped, final = run([{"memory_max_draft": cap}])
    assert capped == free
    receipt = final.speculative_receipt
    assert receipt["memory_max_draft"] == cap
    assert max(map(int, receipt["verify_span_hist"])) <= cap + 1
    if cap == 0:
        assert receipt["proposed"] == 0


def test_engine_serves_prompt_lookup_when_only_the_floor_fits(monkeypatch):
    """End to end on the prompt-lookup route: no 429 on an idle server."""
    patch_host(monkeypatch)
    monkeypatch.setattr(memory, "host_memory_gib", lambda: M3["host_memory_gib"])
    monkeypatch.setattr(memory, "metal_advisory_gib", lambda: M3["advisory_gib"])
    monkeypatch.setattr(serving.ServingEngine, "MEMORY_ADMISSION_TIMEOUT", 2.0)
    reserve = sum(C.host_scaled_reserves(**M3))
    # Room for the width-one lane (no adapter budget: 1.76/3 GiB transient
    # plus a tiny envelope) but not for any verify row (0.59 GiB each).
    monkeypatch.setattr(
        memory, "execution_headroom", lambda **_kw: int((reserve + 0.9) * GIB)
    )
    model, vocab = tiny_qwen38_mtp()
    engine = make_engine(model, vocab, mtp=False, prompt_lookup=True, max_lanes=2)
    try:
        prompt = [(3 * i + 1) % (vocab - 2) + 1 for i in range(30)]
        result = collect(engine.submit({"tokens": prompt, "max_tokens": 6,
                                        "temperature": 0}))
        assert "error" not in result, result
        speculation = result["receipt"]["speculation"]
        assert speculation["memory_max_draft"] == 0
        assert engine.counts["prompt_lookup_admission_span_capped"] >= 1
    finally:
        engine.close()


def test_no_terminal_path_leaves_a_grant(monkeypatch):
    """Cancelled, refused and completed jobs all end with no grant held."""
    patch_host(monkeypatch)
    monkeypatch.setattr(memory, "host_memory_gib", lambda: M3["host_memory_gib"])
    monkeypatch.setattr(memory, "metal_advisory_gib", lambda: M3["advisory_gib"])
    reserve = sum(C.host_scaled_reserves(**M3))
    lane = 3.0
    monkeypatch.setattr(
        serving, "lane_admission_required_gib",
        lambda controller, **_kw: controller.hard_reserve_gib + lane,
    )
    monkeypatch.setattr(
        memory, "execution_headroom", lambda **_kw: int((reserve + 1.5 * lane) * GIB)
    )
    model, vocab = tiny_qwen38_mtp()
    engine = make_engine(model, vocab, mtp=False, max_lanes=4)
    try:
        prompt = [(5 * i + 3) % (vocab - 2) + 1 for i in range(40)]
        body = {"tokens": prompt, "max_tokens": 4, "temperature": 0}
        cancelled = engine.submit(dict(body, max_tokens=64))
        cancelled.cancelled.set()
        collect(cancelled)
        cohort = {"id": "grant-leak", "size": 2}
        refused = [engine.submit(dict(body, batch_cohort=dict(cohort))) for _ in range(2)]
        assert all(collect(job).get("status") == 429 for job in refused)
        done = engine.submit(dict(body))
        assert "error" not in collect(done)
        for job in (cancelled, *refused, done):
            assert job.admission_reserved_gib == 0.0, job
        # Admission is not starved afterwards: requests keep being served.
        start = time.monotonic()
        for _ in range(3):
            assert "error" not in collect(engine.submit(dict(body)))
        assert time.monotonic() - start < 30
    finally:
        engine.close()
