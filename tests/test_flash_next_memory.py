from dataclasses import asdict
import mlx.core as mx
import pytest
from mlx2.adapters.flash_next_memory import FlashNextCacheBudget
from mlx2.runtime.memory_policy import SelfMTPLaneAdmissionController
from test_batched_mtp import _tiny_qwen4_model, _prepare_lane


@pytest.mark.parametrize("length", [17, 255, 257, 600])
def test_geometry_bound_covers_real_target_and_draft_cache_bytes(length):
    mx.random.seed(412)
    model = _tiny_qwen4_model()
    budget = FlashNextCacheBudget.from_config(model.args.text_config, mtp=True)
    lane = _prepare_lane(model, 0, [(i % 60) + 1 for i in range(length)])
    observed = sum(cache.nbytes for cache in [*lane.caches.target, *lane.caches.draft])
    assert 0 < observed <= budget.project(length)
    assert budget.project(length + 1024) >= budget.project(length)


def test_actual_hybrid_geometry_admits_long_context_without_spending_reserve():
    config = dict(num_hidden_layers=48, layer_types=["linear_attention"] * 36 + ["full_attention"] * 12,
        mtp_num_hidden_layers=1, num_key_value_heads=2, head_dim=256,
        indexer_head_dim=128, indexer_compress_ratio=4, linear_num_value_heads=48,
        linear_key_head_dim=128, linear_value_head_dim=128, linear_num_key_heads=16,
        linear_conv_kernel_dim=4, ple_layer_ids=[2], ple_embed_dim=2560, ple_conv_kernel_size=4)
    geometry = FlashNextCacheBudget.from_config(config, mtp=True)
    controller = SelfMTPLaneAdmissionController(cache_estimator=geometry.project)
    assert 7 < geometry.project(131072) / 2**30 < 9
    assert 14 < geometry.project(262144) / 2**30 < 17
    assert controller.hard_reserve_gib == 20
    decision = controller.decide([131072], 46.9)
    assert decision.modes == ("self_mtp",)
    assert decision.estimated_gib < decision.usable_gib == 26.9
    denied = controller.decide([131072], 20)
    assert denied.modes == ("queue",)
    # Actual retained state remains a floor even if it exceeds the model bound.
    assert controller.lane_gib(131072, 2, cache_gib=30) > 30


def test_configured_512_output_default_reproduces_old_admission_width():
    config = dict(
        num_hidden_layers=48,
        layer_types=["linear_attention"] * 36 + ["full_attention"] * 12,
        mtp_num_hidden_layers=1,
        num_key_value_heads=2,
        head_dim=256,
        indexer_head_dim=128,
        indexer_compress_ratio=4,
        linear_num_value_heads=48,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=16,
        linear_conv_kernel_dim=4,
        ple_layer_ids=[2],
        ple_embed_dim=2560,
        ple_conv_kernel_size=4,
    )
    geometry = FlashNextCacheBudget.from_config(config, mtp=True)
    controller = SelfMTPLaneAdmissionController(cache_estimator=geometry.project)
    free_gib = 128.0 - controller.BASE_RESIDENT_GIB

    def admitted(prompt_tokens, default_max_tokens):
        context_tokens = prompt_tokens + min(
            default_max_tokens, 262_144 - prompt_tokens
        )
        decision = controller.decide([context_tokens] * 64, free_gib, max_draft=2)
        return decision.modes.count("self_mtp")

    prompts = (2_048, 16_384, 65_536)
    assert [admitted(prompt, 512) for prompt in prompts] == [16, 11, 6]
    assert [admitted(prompt, 65_536) for prompt in prompts] == [6, 5, 3]


def test_invalid_estimator_fails_closed():
    controller = SelfMTPLaneAdmissionController(cache_estimator=lambda _: float("nan"))
    with pytest.raises(ValueError):
        controller.lane_gib(1000, 2)


def test_reclaim_allocator_scratch_before_decode_route_demotion():
    from mlx2.runtime.memory_policy import _make_self_mtp_admission_callback
    free = [21.0]
    reclaimed = []
    def reclaim():
        reclaimed.append(1)
        free[0] = 30.0
    callback = _make_self_mtp_admission_callback(
        free_memory=lambda: free[0], reclaim_memory=reclaim)
    assert callback([(1, 1024, 2, True, 1.0)]) == {1: 2}
    assert reclaimed == [1]
    # An already admissible cycle does not churn the scratch allocator.
    assert callback([(1, 1024, 2, True, 1.0)]) == {1: 2}
    assert reclaimed == [1]
    # A real lack of memory is still denied after an unsuccessful reclaim.
    blocked = _make_self_mtp_admission_callback(
        free_memory=lambda: 20.0, reclaim_memory=lambda: None)
    assert blocked([(1, 1024, 2, True, 1.0)]) == {1: "queue"}


def test_atomic_cohort_lowers_uniform_depth_before_dropping_a_lane():
    from mlx2.runtime.memory_policy import _make_self_mtp_admission_callback

    controller = SelfMTPLaneAdmissionController(
        transient_gib_per_lane=1.0,
        saturation_lane_cap=20,
        verification_row_cap=60,
    )
    # At zero context each k2 lane costs 1 GiB and each k1 lane costs 0.8 GiB.
    # 19/20 k2 fits in 19.5 usable GiB; the ordinary policy therefore chooses
    # fewer lanes.  The declared atomic policy instead fits all 20 at k1.
    observed = []
    callback = _make_self_mtp_admission_callback(
        controller,
        free_memory=lambda: controller.hard_reserve_gib + 19.5,
        observer=observed.append,
    )
    rows = [(uid, 0, 2, False, 0.0) for uid in range(20)]

    ordinary = callback(rows)
    assert list(ordinary.values()).count(2) == 19
    assert list(ordinary.values()).count("queue") == 1
    assert observed[-1].stage == "fewer_lanes"
    assert observed[-1].speculative_rows == 38

    atomic = callback.atomic(rows)
    assert atomic == {uid: 1 for uid in range(20)}
    decision = observed[-1]
    assert decision.stage == "lower_k"
    assert decision.draft_depths == (1,) * 20
    assert decision.primary_rows == 20
    assert decision.speculative_rows == 20
    assert decision.estimated_gib == pytest.approx(16.0)
    assert decision.usable_gib == pytest.approx(19.5)


def test_warm_copy_uses_observed_dtype_capacity_but_still_charges_full_copy():
    controller = SelfMTPLaneAdmissionController(cache_estimator=lambda n: n * 65536)
    cold = controller.lane_gib(131072, 2)
    warm = controller.lane_gib(131072, 2, cache_gib=4.0)
    resident = controller.lane_gib(131072, 2, cache_gib=4.0, resident_cache=True)
    assert cold > 9
    assert 5.76 < warm < 6.0
    assert warm == resident + 4.0
    # A bf16 warm copy fits without spending the unchanged20GiB reserve.
    assert controller.decide([131072], 27, cache_gib=[4.0]).modes == ("self_mtp",)
    assert controller.decide([131072], 24, cache_gib=[4.0]).modes == ("queue",)


def test_unused_checkpoint_reclaimed_before_splitting_warm_cohort():
    from mlx2.runtime.memory_policy import _make_self_mtp_admission_callback
    controller = SelfMTPLaneAdmissionController(cache_estimator=lambda n: n * 65536)
    rows = [(1, 131072, 2, False, 4.0), (2, 131072, 2, False, 4.0)]
    free, pending = [30.0], [0.0]
    events = []
    entries = [2.0, 4.0]  # Unused checkpoint first; the warm owner is leased.
    def reclaim():
        events.append("reclaim")
        free[0] += pending[0]
        pending[0] = 0.0
    def evict():
        events.append("evict")
        if not entries:
            return False
        pending[0] = entries.pop(0)
        return True
    baseline = _make_self_mtp_admission_callback(controller, free_memory=lambda: free[0])
    assert baseline(rows) == {1: 2, 2: "queue"}
    callback = _make_self_mtp_admission_callback(controller, free_memory=lambda: free[0],
        reclaim_memory=reclaim, evict_unused_cache=evict)
    assert callback(rows) == {1: 2, 2: 2}
    assert events == ["reclaim", "evict", "reclaim"]
    assert entries == [4.0]  # Stop as soon as actual measured capacity fits.
    assert controller.hard_reserve_gib == 20.0
    assert callback(rows) == {1: 2, 2: 2}
    assert events == ["reclaim", "evict", "reclaim"]


def test_checkpoint_eviction_respects_capacity_limits_and_exhaustion():
    from mlx2.runtime.memory_policy import _make_self_mtp_admission_callback
    events = []
    def evict():
        events.append("evict")
        return False
    rows = [(1, 1024, 2, True, 1.0), (2, 1024, 2, True, 1.0)]
    for controller in (
        SelfMTPLaneAdmissionController(saturation_lane_cap=1),
        SelfMTPLaneAdmissionController(verification_row_cap=4),
    ):
        callback = _make_self_mtp_admission_callback(controller,
            free_memory=lambda: 40.0, evict_unused_cache=evict)
        assert callback(rows) == {1: 2, 2: "queue"}
        assert events == []
    callback = _make_self_mtp_admission_callback(free_memory=lambda: 20.0,
        reclaim_memory=lambda: events.append("reclaim"), evict_unused_cache=evict)
    assert callback(rows) == {1: "queue", 2: "queue"}
    assert events == ["reclaim", "evict"]


def test_checkpoint_reclaim_is_bounded_when_callback_cannot_release_memory():
    from mlx2.runtime.memory_policy import _make_self_mtp_admission_callback
    calls = []
    callback = _make_self_mtp_admission_callback(free_memory=lambda: 20.0,
        evict_unused_cache=lambda: calls.append(1) or True)
    assert callback([(1, 1024, 2, True, 1.0)]) == {1: "queue"}
    assert len(calls) == 32
