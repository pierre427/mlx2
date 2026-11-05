"""CPU fixtures only: source-bound masks, verified byte IO and bounded residency."""

import dataclasses
import hashlib
import threading

import numpy as np
import pytest

from mlx2.experimental.weight_space.experts import (
    ExpertBudgetError,
    ExpertKey,
    ExpertMaskPlan,
    ExpertNotReady,
    ExpertRecord,
    ExpertResidencyController,
    ExpertSourceChanged,
    LayerExpertMask,
    plan_residency_budget,
    route_experts_numpy,
    select_expert_mask,
)
from mlx2.runtime.weight_space_metrics import WeightSpaceMetrics

CHECKPOINT = "expert-byte-fixture@sha256:" + "a" * 64
SOURCE_REVISION = "e448f070b84642bfc5d11dc04719e40336326a68"
SOURCE = "b" * 64
CALIBRATION = "c" * 64


def plan(*, enabled=(0, 1, 2), top_k=2, overlay="none"):
    return ExpertMaskPlan(CHECKPOINT, SOURCE_REVISION, SOURCE,
                          (LayerExpertMask(23, 3, top_k, enabled, score_scale=1.5),),
                          "native_fp8", "ordinary_b1", CALIBRATION, overlay)


def fixture_controller(*, budget=20, chunk=4, overlay=0, rate=None, clock=lambda: 0.0,
                       reader_override=None, validator=None, selected_plan=None):
    data = {ExpertKey(23, index): bytes([index + 1]) * 8 for index in range(3)}
    records = [ExpertRecord(key, "fixture-shard", index * 8, len(value),
                            hashlib.sha256(value).hexdigest(), SOURCE, "fp8")
               for index, (key, value) in enumerate(data.items())]
    reads = []

    def reader(record, offset, length):
        reads.append((record.key, offset, length))
        if reader_override:
            return reader_override(record, offset, length)
        return data[record.key][offset:offset + length]

    controller = ExpertResidencyController(
        plan=selected_plan or plan(overlay="d" * 64 if overlay else "none"), records=records,
        budget_bytes=budget, chunk_bytes=chunk, overlay_bytes=overlay, reader=reader,
        source_validator=validator or (lambda: SOURCE), bytes_per_second=rate, clock=clock,
    )
    return controller, data, reads


def load(controller, key):
    controller.request(key)
    result = controller.step(max_chunks=16)
    assert result["blocked"] is None
    return result


def test_score_selection_keeps_original_ids_and_topk_coverage():
    selected = select_expert_mask(
        importance={23: [0.5, 2.0, 1.0]}, keep_counts={23: 2},
        geometries={23: {"num_experts": 3, "top_k": 2, "score_scale": 1.5}},
        checkpoint=CHECKPOINT, source_revision=SOURCE_REVISION, source_fingerprint=SOURCE,
        calibration_digest=CALIBRATION, quantization_profile="native_fp8", execution_profile="b1",
    )
    assert selected.layer(23).enabled == (1, 2)
    assert selected.layer(23).num_experts == 3
    assert selected.approximate is True
    assert selected.selection_method == "supplied_calibration_importance_descending"
    assert selected.to_dict()["status"] == {"implemented": True, "qualified": False, "selected": False, "observed_used": False}
    assert ExpertMaskPlan.from_dict(selected.to_dict()) == selected
    with pytest.raises(ValueError, match="top_k"):
        dataclasses.replace(selected.layer(23), enabled=(1,))


def test_pruning_and_residency_identities_are_separate_and_content_bound():
    base = plan()
    variants = [dataclasses.replace(base, source_revision="other-exact-revision"),
                dataclasses.replace(base, checkpoint="other@sha256:" + "a" * 64),
                dataclasses.replace(base, source_fingerprint="e" * 64),
                dataclasses.replace(base, quantization_profile="fp4_overlay"),
                dataclasses.replace(base, execution_profile="wide2"),
                dataclasses.replace(base, overlay_fingerprint="d" * 64),
                dataclasses.replace(base, layers=(LayerExpertMask(23, 3, 2, (1, 2)),))]
    assert len({base.plan_id, *(variant.plan_id for variant in variants)}) == 8
    a = fixture_controller(budget=20)[0]
    b = fixture_controller(budget=28)[0]
    assert a.plan.plan_id == b.plan.plan_id
    assert a.controller_id != b.controller_id
    payload = base.to_dict()
    payload["status"]["qualified"] = True
    with pytest.raises(ValueError, match="identity"):
        ExpertMaskPlan.from_dict(payload)


@pytest.mark.parametrize("kwargs", [{"enabled": (0, 0)}, {"enabled": ()}, {"top_k": 0},
                                    {"enabled": (0, 3)}, {"normalize": 1}, {"score_scale": float("nan")}])
def test_invalid_mask_refuses_without_shape_remapping(kwargs):
    values = {"layer": 23, "num_experts": 3, "top_k": 2, "enabled": (0, 1, 2)}
    values.update(kwargs)
    with pytest.raises(ValueError):
        LayerExpertMask(**values)


def test_mask_routes_with_original_scores_and_renormalizes():
    selected = plan(enabled=(0, 2))
    ids, weights = route_experts_numpy(selected, 23, [1, 9, 3], ranking_scores=[8, 10, 1])
    np.testing.assert_array_equal(ids, [0, 2])
    np.testing.assert_allclose(weights, [0.375, 1.125], rtol=0, atol=0)
    assert weights.sum() == 1.5


def test_all_enabled_control_uses_stable_ids_and_preserves_native_score_law():
    ids, weights = route_experts_numpy(plan(), 23, [[1, 4, 2], [2, 1, 3]],
                                       ranking_scores=[[9, 1, 8], [0, 4, 4]])
    np.testing.assert_array_equal(ids, [[0, 2], [1, 2]])
    np.testing.assert_allclose(weights, [[0.5, 1], [0.375, 1.125]], rtol=0, atol=0)
    plain = dataclasses.replace(plan(), layers=(LayerExpertMask(23, 3, 2, (0, 1, 2), False, 2),))
    _, unnormalized = route_experts_numpy(plain, 23, [1, 4, 2])
    np.testing.assert_array_equal(unnormalized, [8, 4])


@pytest.mark.parametrize("scores", [[0, 0, 0], [1, -1, 2], [1, float("inf"), 2], [1, 2]])
def test_invalid_router_scores_refuse(scores):
    with pytest.raises(ValueError):
        route_experts_numpy(plan(), 23, scores)


def test_chunks_are_not_exposed_until_complete_digest_is_verified():
    controller, data, reads = fixture_controller()
    key = next(iter(data))
    assert controller.request(key) == "queued"
    first = controller.step()
    assert first["bytes"] == 4 and not first["completed"]
    with pytest.raises(ExpertNotReady):
        controller.acquire(key)
    controller.step()
    with controller.acquire(key) as lease:
        assert lease.read(0, 4) + lease.read(4, 4) == data[key]
        with pytest.raises(ValueError, match="length"):
            lease.read(0, 8)
    with pytest.raises(RuntimeError, match="released"):
        lease.read()
    assert reads == [(key, 0, 4), (key, 4, 4)]
    assert controller.metrics_snapshot()["streamed_bytes"] == 8
    assert controller.receipt()["native_arithmetic_or_gpu_qualified"] is False


def test_pins_block_eviction_and_release_allows_demand_swap():
    controller, data, _ = fixture_controller(budget=12)
    a, b, _ = data
    load(controller, a)
    lease = controller.acquire(a)
    controller.request(b)
    assert controller.step()["blocked"] == "pinned_budget"
    with pytest.raises(ExpertBudgetError, match="pinned"):
        controller.evict(a)
    assert lease.read() == data[a][:4]
    lease.release()
    lease.release()
    controller.step(max_chunks=2)
    with controller.acquire(b) as new_lease:
        assert new_lease.read() == data[b][:4]
    with pytest.raises(ExpertNotReady):
        controller.acquire(a)
    state = controller.metrics_snapshot()
    assert state["evictions"] == 1 and state["reserved_bytes"] == 12


def test_multiple_leases_account_unique_pinned_bytes():
    controller, data, _ = fixture_controller()
    a = next(iter(data))
    load(controller, a)
    first, second = controller.acquire(a), controller.acquire(a)
    state = controller.metrics_snapshot()
    assert state["leases"] == 2 and state["pinned_bytes"] == 8
    first.release()
    with pytest.raises(ExpertBudgetError):
        controller.close()
    assert second.read() == data[a][:4]
    second.release()
    controller.close()
    with pytest.raises(RuntimeError, match="closed"):
        controller.request(a)


def test_prefetch_never_evicts_demand_residents_and_demand_preempts_partial_prefetch():
    controller, data, _ = fixture_controller(budget=12)
    a, b, c = data
    load(controller, a)
    controller.prefetch(b)
    result = controller.step()
    assert result["blocked"] == "prefetch_budget"
    with controller.acquire(a) as lease:
        assert lease.read() == data[a][:4]
    controller.evict(a)
    controller.prefetch(b)
    controller.step()
    controller.request(c)
    assert controller.metrics_snapshot()["cancelled_bytes"] == 4
    controller.step(max_chunks=2)
    with pytest.raises(ExpertNotReady):
        controller.acquire(b)
    with controller.acquire(c) as lease:
        assert lease.read() == data[c][:4]


def test_pacing_is_cooperative_bounded_and_does_not_sleep():
    now = [0.0]
    controller, data, reads = fixture_controller(rate=4, clock=lambda: now[0])
    key = next(iter(data))
    controller.request(key)
    result = controller.step(max_chunks=1024)
    assert result["chunks"] == 1 and result["next_ready_time"] == 1.0
    assert controller.step()["blocked"] == "pacing"
    assert len(reads) == 1
    now[0] = 1.0
    assert controller.step()["completed"]
    assert controller.metrics_snapshot()["pacing_waits"] == 2


def test_overlay_charge_reduces_payload_capacity_and_requires_content_binding():
    with pytest.raises(ExpertBudgetError, match="overlays"):
        fixture_controller(budget=15, overlay=4)
    with pytest.raises(ValueError, match="fingerprint"):
        fixture_controller(overlay=4, selected_plan=plan())
    controller, data, _ = fixture_controller(budget=16, overlay=4)
    load(controller, next(iter(data)))
    state = controller.metrics_snapshot()
    assert state["overlay_bytes"] == 4 and state["reserved_bytes"] == 16
    assert controller.plan.overlay_fingerprint == "d" * 64


@pytest.mark.parametrize("kind", ["truncated", "wrong_content", "mutable", "exception"])
def test_io_failure_never_publishes_zero_or_corrupt_weights(kind):
    def reader(record, offset, length):
        if kind == "exception":
            raise OSError("simulated SSD error")
        if kind == "truncated":
            return b"x" * (length - 1)
        if kind == "mutable":
            return bytearray(length)
        return b"x" * length

    controller, data, _ = fixture_controller(reader_override=reader)
    key = next(iter(data))
    controller.request(key)
    with pytest.raises(OSError):
        controller.step(max_chunks=4)
    with pytest.raises(ExpertNotReady):
        controller.acquire(key)
    state = controller.metrics_snapshot()
    assert state["resident_bytes"] == state["loading_bytes"] == 0
    assert state["faults"] == 1


def test_source_change_during_io_poison_refuses_mixed_revision():
    source = [SOURCE]

    def reader(record, offset, length):
        source[0] = "f" * 64
        return b"x" * length

    controller, data, _ = fixture_controller(reader_override=reader, validator=lambda: source[0])
    key = next(iter(data))
    controller.request(key)
    with pytest.raises(ExpertSourceChanged):
        controller.step()
    source[0] = SOURCE
    with pytest.raises(ExpertSourceChanged, match="stale"):
        controller.request(key)
    assert controller.receipt()["source_stale"] is True
    assert controller.metrics_snapshot()["resident_bytes"] == 0


def test_source_change_invalidates_live_lease_without_freeing_pinned_bytes_early():
    source = [SOURCE]
    controller, data, _ = fixture_controller(validator=lambda: source[0])
    key = next(iter(data))
    load(controller, key)
    lease = controller.acquire(key)
    source[0] = "f" * 64
    with pytest.raises(ExpertSourceChanged):
        lease.read()
    assert controller.metrics_snapshot()["pinned_bytes"] == 8
    lease.release()
    assert controller.metrics_snapshot()["resident_bytes"] == 0


def test_disabled_expert_and_stale_index_cannot_be_requested():
    controller, _data, _ = fixture_controller(selected_plan=plan(enabled=(0, 2)))
    with pytest.raises(ValueError, match="disabled"):
        controller.request(ExpertKey(23, 1))
    record = next(iter(controller.records.values()))
    with pytest.raises(ValueError, match="source"):
        ExpertResidencyController(plan=plan(top_k=1, enabled=(0,)),
                                  records=[dataclasses.replace(record, source_fingerprint="e" * 64)],
                                  budget_bytes=12, chunk_bytes=4,
                                  reader=lambda *_: b"", source_validator=lambda: SOURCE)


def test_cancel_discards_partial_load_and_retry_starts_at_zero():
    controller, data, reads = fixture_controller()
    key = next(iter(data))
    controller.request(key)
    controller.step()
    assert controller.cancel(key)
    assert controller.metrics_snapshot()["loading_bytes"] == 0
    assert not controller.cancel(key)
    load(controller, key)
    assert [row[1] for row in reads] == [0, 0, 4]


def test_atomic_metrics_are_low_cardinality_and_counter_totals_survive_close():
    collector = WeightSpaceMetrics()
    controller, data, _ = fixture_controller()
    collector.register_experts(controller)
    a, b, c = data
    errors = []

    def sample():
        for _ in range(100):
            snapshot = controller.metrics_snapshot()
            if snapshot["reserved_bytes"] > snapshot["budget_bytes"]:
                errors.append(snapshot)
            assert all(isinstance(value, (int, float)) for value in snapshot.values())

    thread = threading.Thread(target=sample)
    thread.start()
    for key in (a, b, c):
        load(controller, key)
    thread.join()
    collector.sample_experts(controller)
    assert not errors
    state = collector.snapshot()
    assert state["expert_counters"]["streamed_bytes"] == 24
    assert state["expert_counters"]["evictions"] == 1
    controller.close()
    collector.sample_experts(controller)
    assert collector.snapshot()["expert_counters"]["streamed_bytes"] == 24


def test_budget_plan_counts_nonexpert_and_overlay_memory_separately():
    controller, _, _ = fixture_controller(budget=24, overlay=4)
    budget = plan_residency_budget(plan=controller.plan, records=controller.records.values(),
                                   memory_budget_bytes=100, nonexpert_reserved_bytes=76,
                                   chunk_bytes=4, overlay_bytes=4)
    assert budget["payload_budget_bytes"] == 16
    assert budget["controller_budget_bytes"] == 24
    assert budget["maximum_single_token_layer_union_bytes"] == 16
    assert budget["single_token_layer_union_fits"] is True
    assert budget["all_enabled_payload_fits"] is False
    assert budget["physical_ram_or_os_swap_measured"] is False
    limited = plan_residency_budget(plan=controller.plan, records=controller.records.values(),
                                    memory_budget_bytes=100, nonexpert_reserved_bytes=76,
                                    chunk_bytes=4, overlay_bytes=4, maximum_resident_bytes=8)
    assert limited["single_token_layer_union_fits"] is False
    assert limited["budget_plan_id"] != budget["budget_plan_id"]
    with pytest.raises(ExpertBudgetError):
        plan_residency_budget(plan=controller.plan, records=controller.records.values(),
                              memory_budget_bytes=10, nonexpert_reserved_bytes=0,
                              chunk_bytes=4, overlay_bytes=4)


def test_multi_projection_budget_uses_whole_indexed_expert_bundle():
    controller, _, _ = fixture_controller()
    records = [dataclasses.replace(record, key=dataclasses.replace(record.key, projection=projection))
               for record in controller.records.values() for projection in ("up", "down")]
    budget = plan_residency_budget(plan=controller.plan, records=records,
                                   memory_budget_bytes=36, nonexpert_reserved_bytes=0, chunk_bytes=4)
    assert budget["largest_indexed_expert_bundle_bytes"] == 16
    assert budget["maximum_single_token_layer_union_bytes"] == 32
    assert budget["indexed_enabled_payload_bytes"] == 48
    assert budget["source_index_completeness_verified"] is False


def test_native_tie_policy_has_distinct_identity_and_is_not_emulated_by_cpu_oracle():
    native = dataclasses.replace(plan(), tie_policy="native_topk_after_mask")
    assert native.plan_id != plan().plan_id
    assert ExpertMaskPlan.from_dict(native.to_dict()) == native
    with pytest.raises(ValueError, match="cannot emulate"):
        route_experts_numpy(native, 23, [1, 1, 1])


def test_close_counts_discarded_partial_io_as_cancelled_bytes():
    controller, data, _ = fixture_controller()
    controller.prefetch(next(iter(data)))
    controller.step()
    controller.close()
    assert controller.metrics_snapshot()["cancelled_bytes"] == 4
