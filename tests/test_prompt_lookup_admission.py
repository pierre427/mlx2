"""Prompt-lookup admission seats a lane at the widest verify span that fits.

Series 2026-09-24, Qwen3.8-27B-MLX-4bit --prompt-lookup on a 36 GiB M3 Pro:
every request -- the first on an idle server included -- waited out its
60 s admission deadline and answered 429 (39 of 43 feature checks), while
the ordinary route on the same host served them.  No grant was ever held
(``memory_admission_deferred_behind_grants`` never appeared); the full
num_draft=8 verify term alone was 3.1 * 8 / 3 = 8.27 GiB, so a 111-token
chat lane needed 13.66 GiB against the 11-13 GiB the idle host measured.
"""

import json
import time
from pathlib import Path

import mlx.core as mx
import pytest

from mlx2 import memory, serving
from mlx2.runtime.memory_policy import SelfMTPLaneAdmissionController as C
from mlx2.runtime.models.cache import KVCache
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


# The selected policy (num_draft 8, cliff_aware_span) moves a nominal 9-row
# verify, inside the 9..15 cliff, to its far side: 15 proposals, a 16-row
# verify forward.  Admission charged num_draft (8) rows and set no
# memory_max_draft on a lane seated at that span, and int8 prefill's
# decode/verify row bound assumed num_draft + 2 rows per lane, so the bound
# that keeps verify forwards off the approximate int8 kernels was understated.
SELECTED_PLD = json.loads(
    (Path(__file__).resolve().parents[1] / "qualification/policies/prompt-lookup.json")
    .read_text()
)["prompt_lookup"]


class _Successor:
    """Greedy target that predicts token + 1; records every forward's shape."""

    def __init__(self):
        self.shapes = []

    def __call__(self, tokens, *, cache):
        self.shapes.append(tuple(tokens.shape))
        for entry in cache:
            values = tokens.astype(mx.float32)[:, None, :, None]
            entry.update_and_fetch(values, values)
        predicted = (tokens.astype(mx.int32) + 1) % 64
        return mx.where(mx.arange(64)[None, None, :] == predicted[..., None], 20.0, -20.0)


def _verify_rows(policy, lanes, configs=None):
    """Widest decode/verify forward (rows) a copy-positive run presents."""
    model = _Successor()
    generator = PromptLookupBatchGenerator(
        model, completion_batch_size=lanes, prefill_step_size=256, prompt_lookup=policy
    )
    prompt = list(range(1, 61)) + [7, 8, 9, 10, 11, 12]
    generator.insert(
        [prompt] * lanes, max_tokens=[40] * lanes,
        caches=[[KVCache()] for _ in range(lanes)],
        prompt_lookup_configs=configs,
    )
    for _ in range(400):
        generator.next()
        if not generator.lanes:
            break
    assert not generator.lanes
    return max(b * l for b, l in model.shapes if l < len(prompt) - 1)


@pytest.mark.parametrize(
    "policy, lanes",
    [
        (SELECTED_PLD, 1),  # the selected policy: per-lane verify
        ({k: v for k, v in SELECTED_PLD.items() if k != "batched_verify"}, 2),
    ],
)
def test_int8_decode_row_bound_covers_cliff_aware_verify(policy, lanes):
    from mlx2.runtime.int8_prefill import max_decode_rows

    widest = _verify_rows(policy, lanes)
    assert widest > lanes * (policy["num_draft"] + 1)  # the cliff extension engaged
    bound = max_decode_rows(
        max_lanes=lanes, config={}, speculation="prompt_lookup",
        prompt_lookup_policy=policy,
    )
    assert bound >= widest


def test_memory_capped_cliff_aware_lane_stays_below_the_cliff():
    # A lane seated below the far-side span must not verify inside the cliff
    # the policy exists to avoid (9..15 rows), nor above its cap.
    widest = _verify_rows(SELECTED_PLD, 1, configs=[{"memory_max_draft": 10}])
    assert widest <= SELECTED_PLD.get("verify_cliff_start", 9) - 1


def test_full_span_lane_never_verifies_wider_than_admission_charged(monkeypatch):
    from mlx2.runtime.prompt_lookup import plan_proposal_around_verify_cliff

    patch_host(monkeypatch)
    admitted_spans = []
    real_admit = serving.admit_prompt_lookup_lane

    def admit_spy(controller, **kwargs):
        result = real_admit(controller, **kwargs)
        admitted_spans.append(result)
        return result

    monkeypatch.setattr(serving, "admit_prompt_lookup_lane", admit_spy)
    inserted = []
    real_insert = PromptLookupBatchGenerator.insert

    def insert_spy(self, *args, **kwargs):
        inserted.append((self, kwargs.get("prompt_lookup_configs")))
        return real_insert(self, *args, **kwargs)

    monkeypatch.setattr(PromptLookupBatchGenerator, "insert", insert_spy)
    model, vocab = tiny_qwen38_mtp()
    engine = make_engine(
        model, vocab, mtp=False, prompt_lookup=True, max_lanes=1,
        execution_policy={"prompt_lookup": dict(SELECTED_PLD)},
    )
    try:
        prompt = [(3 * i + 1) % (vocab - 2) + 1 for i in range(30)]
        result = collect(engine.submit({"tokens": prompt, "max_tokens": 6, "temperature": 0}))
        assert "error" not in result, result
    finally:
        engine.close()
    (admitted, span, _required) = admitted_spans[-1]
    assert admitted
    generator, configs = inserted[-1]
    lane = {**generator.config, **((configs or [{}])[0])}
    assert lane.get("cliff_aware_span") is True
    widest = plan_proposal_around_verify_cliff(
        generator.num_draft, 1 << 20, 1,
        cliff_start=int(lane.get("verify_cliff_start", 9)),
        cliff_end=int(lane.get("verify_cliff_end", 15)),
    )
    if lane.get("memory_max_draft") is not None:
        widest = min(widest, lane["memory_max_draft"])
    # Admission charged ``span`` proposal rows for this lane.
    assert widest <= span, (widest, span, lane.get("memory_max_draft"))


# Admission and the int8 row bound charge the generator's widest round
# (``max_proposal_span`` of its policy), but ``insert`` let a lane override
# the cliff keys: a generator charged for 15 proposals accepted
# {"verify_cliff_end": 31} and verified 31.
_NO_CLIFF = {k: v for k, v in SELECTED_PLD.items() if k != "cliff_aware_span"}
_LATE_CLIFF = {**SELECTED_PLD, "verify_cliff_start": 10}


@pytest.mark.parametrize(
    "policy, override",
    [
        (SELECTED_PLD, {"verify_cliff_end": 31}),
        (_LATE_CLIFF, {"verify_cliff_start": 9}),
        (_NO_CLIFF, {"cliff_aware_span": True}),
    ],
    ids=["cliff_end", "cliff_start", "cliff_aware"],
)
def test_lane_override_cannot_widen_the_charged_span(policy, override):
    with pytest.raises(ValueError, match="exceeds the generator's charged span"):
        _verify_rows(policy, 1, configs=[override])


def test_widening_lane_override_refuses_the_whole_insert():
    generator = PromptLookupBatchGenerator(
        _Successor(), completion_batch_size=2, prefill_step_size=256,
        prompt_lookup=SELECTED_PLD,
    )
    prompt = list(range(1, 61)) + [7, 8, 9, 10, 11, 12]
    with pytest.raises(ValueError, match="exceeds the generator's charged span"):
        generator.insert(
            [prompt] * 2, max_tokens=[40] * 2,
            caches=[[KVCache()] for _ in range(2)],
            prompt_lookup_configs=[{}, {"verify_cliff_end": 31}],
        )
    assert generator.lanes == {} and generator.next_uid == 0


@pytest.mark.parametrize(
    "override",
    [
        {"verify_cliff_end": 12},
        {"verify_cliff_start": 10},
        {"cliff_aware_span": False},
        {"memory_max_draft": 3},
    ],
    ids=["cliff_end", "cliff_start", "cliff_off", "memory_cap"],
)
def test_narrowing_lane_override_is_seated_within_the_charged_span(override):
    from mlx2.runtime.prompt_lookup import max_proposal_span

    widest = _verify_rows(SELECTED_PLD, 1, configs=[override])
    assert widest <= max_proposal_span(SELECTED_PLD) + 1


# Rounds always plan from the generator's num_draft (``_round_steps``), but
# ``insert`` accepted and recorded a narrower lane num_draft: a lane asked for
# {"num_draft": 2} on an 8-draft generator verified 16 rows, and the ragged
# PLD qualifier reported it as a K2 lane.  A wider one was likewise ignored.
@pytest.mark.parametrize(
    "policy, override",
    [
        (SELECTED_PLD, {"num_draft": 2}),
        (_NO_CLIFF, {"num_draft": 2}),
        (SELECTED_PLD, {"num_draft": 4, "verify_cliff_end": 20}),
        (_NO_CLIFF, {"num_draft": 16}),
    ],
    ids=["narrower", "narrower_no_cliff", "narrower_wide_cliff", "wider"],
)
def test_lane_num_draft_differing_from_the_generator_is_refused(policy, override):
    generator = PromptLookupBatchGenerator(
        _Successor(), completion_batch_size=2, prefill_step_size=256,
        prompt_lookup=policy,
    )
    prompt = list(range(1, 61)) + [7, 8, 9, 10, 11, 12]
    with pytest.raises(
        ValueError,
        match=f"lane num_draft {override['num_draft']} must equal the "
        f"generator's num_draft {policy['num_draft']}",
    ):
        generator.insert(
            [prompt] * 2, max_tokens=[40] * 2,
            caches=[[KVCache()] for _ in range(2)],
            prompt_lookup_configs=[{}, override],
        )
    assert generator.lanes == {} and generator.next_uid == 0


@pytest.mark.parametrize(
    "policy, override",
    [
        (SELECTED_PLD, {"num_draft": SELECTED_PLD["num_draft"]}),
        # The generator's default depth, named by the lane only.
        ({k: v for k, v in _NO_CLIFF.items() if k != "num_draft"}, {"num_draft": 8}),
    ],
    ids=["selected", "default"],
)
def test_lane_num_draft_equal_to_the_generator_is_seated(policy, override):
    assert _verify_rows(policy, 1, configs=[override]) == _verify_rows(policy, 1)


# A refused insert left ``lanes`` and ``next_uid`` alone, but each lane was
# validated and armed in turn: with [{"cost_aware_admission": True},
# {"num_draft": 2}] lane 0 stamped the PLD B=1 mask onto its caller-owned
# cache before lane 1 was refused, and the caller kept that cache.
def _filled_cache():
    entry = KVCache()
    values = mx.arange(16, dtype=mx.float32).reshape(1, 1, 4, 4)
    entry.update_and_fetch(values, values)
    return [entry]


def _cache_state(caches):
    return [
        (
            sorted(vars(entry)),
            entry.offset,
            entry.keys[..., : entry.offset, :].tolist(),
            entry.values[..., : entry.offset, :].tolist(),
        )
        for entry in caches
    ]


@pytest.mark.parametrize(
    "override, maximum, cache, error, match",
    [
        ({"num_draft": 2}, 40, None, ValueError, "must equal the generator's num_draft"),
        ({"verify_cliff_end": 31}, 40, None, ValueError, "exceeds the generator's charged span"),
        ({"ngram_min": 0}, 40, None, ValueError, "ngram_min"),
        ({"recent_committed_segments": True}, 40, None, ValueError, "source scope"),
        ({}, 0, None, ValueError, "output budget"),
        ({}, 40, [object()], NotImplementedError, "rollback-capable"),
    ],
    ids=["num_draft", "widening", "invalid_key", "source_scope", "budget", "cache"],
)
def test_later_lane_refusal_leaves_every_supplied_cache_untouched(
    override, maximum, cache, error, match
):
    generator = PromptLookupBatchGenerator(
        _Successor(), completion_batch_size=2, prefill_step_size=256,
        prompt_lookup=SELECTED_PLD,
    )
    caches = [_filled_cache(), cache or _filled_cache()]
    before = [_cache_state(lane) for lane in caches if lane is not cache]
    prompt = list(range(5, 61)) + [7, 8, 9, 10, 11, 12]
    with pytest.raises(error, match=match):
        generator.insert(
            [prompt] * 2, max_tokens=[40, maximum], caches=caches,
            all_tokens=[[1, 2, 3, 4]] * 2,
            prompt_lookup_configs=[{"cost_aware_admission": True}, override],
        )
    assert generator.lanes == {} and generator.next_uid == 0
    assert [_cache_state(lane) for lane in caches if lane is not cache] == before
    # The same lane 0 is armed once the batch is admissible.
    generator.insert(
        [prompt] * 2, max_tokens=[40, 40], caches=[caches[0], _filled_cache()],
        all_tokens=[[1, 2, 3, 4]] * 2,
        prompt_lookup_configs=[{"cost_aware_admission": True}, {}],
    )
    assert caches[0][0]._pld_ordinary_mask_padding is not None


def test_serving_refuses_a_widening_lane_override_as_a_request_error(monkeypatch):
    patch_host(monkeypatch)
    override = {"verify_cliff_end": 31}
    real_insert = PromptLookupBatchGenerator.insert

    def insert_with_override(self, *args, **kwargs):
        # Stands in for a lane override reaching ``insert`` from serving.
        if override:
            kwargs["prompt_lookup_configs"] = [
                {**config, **override} for config in kwargs["prompt_lookup_configs"]
            ]
        return real_insert(self, *args, **kwargs)

    monkeypatch.setattr(PromptLookupBatchGenerator, "insert", insert_with_override)
    model, vocab = tiny_qwen38_mtp()
    engine = make_engine(
        model, vocab, mtp=False, prompt_lookup=True, max_lanes=1,
        execution_policy={"prompt_lookup": dict(SELECTED_PLD)},
    )
    try:
        prompt = [(3 * i + 1) % (vocab - 2) + 1 for i in range(30)]
        request = {"tokens": prompt, "max_tokens": 6, "temperature": 0}
        refused = collect(engine.submit(dict(request)))
        assert refused.get("status") == 400, refused
        assert "exceeds the generator's charged span" in refused["error"]
        override.clear()
        served = collect(engine.submit(dict(request)))
        assert "error" not in served, served
        assert served["finish"]
    finally:
        engine.close()
