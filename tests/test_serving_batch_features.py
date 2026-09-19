"""Exercise new serving controls through the actual tiny-model batch lifecycle."""
from types import SimpleNamespace
import mlx.core as mx
import pytest
from mlx2.runtime.apc_v2 import APCv2, MTPAPCSidecar
from mlx2.runtime.generate import BatchGenerator, _share_qsa_indices_for_config
from mlx2.runtime.sample_utils import LaneRNG, make_logits_processors
from mlx2.runtime.segmented_self_mtp import segmented_self_mtp_stats
from mlx2.serving import shared_prefix_attestation
from test_batched_mtp import _tiny_qwen4_model


@pytest.mark.parametrize(
    "config, expected",
    [
        ({}, True),
        ({"share_qsa_indices": True}, True),
        ({"share_qsa_indices": False}, False),
    ],
)
def test_parent_mtp_route_defaults_shared_qsa_on_and_preserves_opt_out(
    config, expected
):
    assert _share_qsa_indices_for_config(config) is expected


@pytest.mark.parametrize("async_promotion,shared_suffix", [(False, True), (True, True), (True, False)])
def test_shared_warm_checkpoint_attestation_survives_scheduler_transfer(monkeypatch, async_promotion, shared_suffix):
    monkeypatch.setenv("MLX_LM_SEGMENTED_SELF_MTP", "1")
    monkeypatch.setenv("MLX_LM_TRUE_BATCHED_SEGMENTED_MTP", "1")
    monkeypatch.setenv("MLX_LM_SHARED_QSA_SUFFIX", "on" if shared_suffix else "off")
    # This integration oracle owns CPU streams even when production requests Metal.
    new_stream = mx.new_stream
    monkeypatch.setattr(mx, "new_stream", lambda device: new_stream(mx.cpu))
    mx.random.seed(99)
    from mlx2.runtime.models import qwen4_exp
    monkeypatch.setattr(qwen4_exp, "_QSA_POOLED_KEY_CACHE", True)
    model = _tiny_qwen4_model()
    indexer = model.language_model.model.layers[1].self_attn.indexer
    indexer.compress_ratio = 4
    indexer.summary_identity["block_size"] = 4
    indexer.summary_identity["compress_ratio"] = 4
    config = {"persistent": True, "num_draft": 2, "segment_aware_live_tip": True,
              "segment_aware_cohort_size": 2,
              "segment_aware_async_qsa_promotion": async_promotion}
    apc = APCv2(layout_name=model.apc_v2_layout)
    key = apc.key("tiny", revision="cohort")
    tokens = list(range(1, 13))
    def create():
        return BatchGenerator(model, completion_batch_size=2, prefill_step_size=4,
                              self_mtp=config)
    def drain(batch):
        emitted = {}
        for _ in range(40):
            prompts, responses = batch.next()
            for response in prompts:
                boundary = batch.pop_prompt_boundary(response.uid)
                if boundary:
                    apc.store(key, boundary["tokens"], boundary["target_cache"],
                              sidecar=MTPAPCSidecar(boundary["mtp_state"], boundary["covered_tokens"]))
            for response in responses:
                emitted.setdefault(response.uid, []).append(response.token)
                if response.finish_reason:
                    emitted[response.uid].append("done")
            if emitted and all(row[-1] == "done" for row in emitted.values()):
                return emitted
        raise AssertionError("batch failed to complete")
    first = create()
    first.insert([tokens], max_tokens=[6], lane_rngs=[LaneRNG(77)])
    try:
        reference = drain(first)[0][:-1]
    finally:
        first.close()
    hits = [apc.lookup(key, tokens), apc.lookup(key, tokens)]
    assert hits[0].cached_tokens == len(tokens) - 1
    identities = [shared_prefix_attestation(hit) for hit in hits]
    assert identities[0] and identities[0] == identities[1]
    batch = create()
    before = segmented_self_mtp_stats()
    batch.insert([hit.remaining_tokens for hit in hits], max_tokens=[32, 32],
                 caches=[hit.cache for hit in hits], all_tokens=[tokens[:-1]] * 2,
                 mtp_states=[hit.sidecar.state for hit in hits],
                 lane_rngs=[LaneRNG(77), LaneRNG(78)],
                 self_mtp_configs=[{"shared_prefix_attestation": value} for value in identities])
    try:
        result = drain(batch)
        after = segmented_self_mtp_stats()
        assert len(result) == 2
        assert all(row[:len(reference)] == reference for row in result.values())
        if shared_suffix:
            assert after["shared_qsa_policy_admitted"] > before["shared_qsa_policy_admitted"]
            assert after["shared_qsa_rows"] > before["shared_qsa_rows"]
        assert after["batched_target_forwards"] > before["batched_target_forwards"]
        assert after["async_qsa_promotion_failures"] == before["async_qsa_promotion_failures"]
        if async_promotion and shared_suffix:
            assert after["async_qsa_promotion_engaged"] == before["async_qsa_promotion_engaged"]
            assert after["async_qsa_promotion_declined_shared_suffix"] > before["async_qsa_promotion_declined_shared_suffix"]
        elif async_promotion:
            assert after["async_qsa_promotion_engaged"] > before["async_qsa_promotion_engaged"]
    finally:
        batch.close()
        for hit in hits:
            hit.cache.close()
        apc.clear()


def test_actual_warm_heterogeneous_prefixes_form_true_b4_under_hybrid_budget(monkeypatch):
    from mlx2.adapters.flash_next_memory import FlashNextCacheBudget
    from mlx2.runtime.memory_policy import SelfMTPLaneAdmissionController, _make_self_mtp_admission_callback
    monkeypatch.setenv("MLX_LM_SEGMENTED_SELF_MTP", "1")
    monkeypatch.setenv("MLX_LM_TRUE_BATCHED_SEGMENTED_MTP", "1")
    monkeypatch.setenv("MLX_LM_SHARED_QSA_SUFFIX", "off")
    mx.random.seed(309)
    model = _tiny_qwen4_model()
    config = {"persistent": True, "num_draft": 2, "segment_aware_live_tip": True,
              "segment_aware_cohort_size": 4, "segment_aware_async_qsa_promotion": False}
    apc = APCv2(layout_name=model.apc_v2_layout)
    key = apc.key("tiny", revision="heterogeneous")
    prompts = [list(range(i + 1, i + 13 + i)) for i in range(4)]
    def drain(batch):
        outputs, receipts = {}, {}
        for _ in range(80):
            processed, generated = batch.next()
            for item in processed:
                boundary = batch.pop_prompt_boundary(item.uid)
                if boundary:
                    apc.store(key, boundary["tokens"], boundary["target_cache"],
                              sidecar=MTPAPCSidecar(boundary["mtp_state"], boundary["covered_tokens"]))
            for item in generated:
                outputs.setdefault(item.uid, []).append(item.token)
                if item.finish_reason:
                    receipts[item.uid] = item.mtp_receipt
            if len(receipts) == len(outputs) and receipts:
                return outputs, receipts
        raise AssertionError("warm cohort failed to complete")
    references = []
    for tokens in prompts:
        prime = BatchGenerator(model, completion_batch_size=4, self_mtp=config)
        prime.insert([tokens], max_tokens=[6])
        try: references.append(drain(prime)[0][0])
        finally: prime.close()
    hits = [apc.lookup(key, tokens) for tokens in prompts]
    assert len({hit.cached_tokens for hit in hits}) == 4
    assert all(hit.sidecar is not None and len(hit.remaining_tokens) == 1 for hit in hits)
    geometry = FlashNextCacheBudget.from_config(model.args.text_config, mtp=True)
    controller = SelfMTPLaneAdmissionController(cache_estimator=geometry.project,
                                               transient_gib_per_lane=1e-6)
    # Small CPU model/temporary scale, but production20GiB reserve remains.
    # The former fixed_state/context extrapolation queues every warm row here.
    decisions = []
    callback = _make_self_mtp_admission_callback(controller, free_memory=lambda: 20.003,
                                                 observer=decisions.append)
    batch = BatchGenerator(model, completion_batch_size=4, self_mtp=config,
                           mtp_admission=callback)
    batch.insert([hit.remaining_tokens for hit in hits], max_tokens=[8] * 4,
                 caches=[hit.cache for hit in hits],
                 all_tokens=[tokens[:-1] for tokens in prompts],
                 mtp_states=[hit.sidecar.state for hit in hits])
    try:
        outputs, receipts = drain(batch)
        assert len(receipts) == 4
        assert all(4 in receipt["observed_compute_widths"] for receipt in receipts.values())
        assert any(len(d.modes) == 4 and d.stage == "full" for d in decisions)
        assert all(outputs[i][:6] == references[i] for i in range(4))
    finally:
        batch.close()
        for hit in hits: hit.cache.close()
        apc.clear()


def test_atomic_b4_receipt_reports_uniform_lower_k(monkeypatch):
    from mlx2.runtime.memory_policy import (
        SelfMTPLaneAdmissionController,
        _make_self_mtp_admission_callback,
    )

    monkeypatch.setenv("MLX_LM_SEGMENTED_SELF_MTP", "1")
    monkeypatch.setenv("MLX_LM_TRUE_BATCHED_SEGMENTED_MTP", "1")
    mx.random.seed(310)
    model = _tiny_qwen4_model()
    config = {
        "persistent": True,
        "num_draft": 2,
        "segment_aware_live_tip": True,
        "segment_aware_cohort_size": 4,
        "segment_aware_async_qsa_promotion": False,
    }
    controller = SelfMTPLaneAdmissionController(
        transient_gib_per_lane=1.0,
        saturation_lane_cap=4,
        verification_row_cap=12,
        cache_estimator=lambda _tokens: 0,
    )
    decisions = []
    callback = _make_self_mtp_admission_callback(
        controller,
        free_memory=lambda: controller.hard_reserve_gib + 3.5,
        observer=decisions.append,
    )
    batch = BatchGenerator(
        model,
        completion_batch_size=4,
        self_mtp=config,
        mtp_admission=callback,
    )
    cohort = {"tenant_id": "tenant-a", "id": "cpu-b4", "size": 4}
    batch.insert(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
        max_tokens=[4] * 4,
        self_mtp_configs=[{"batch_cohort": dict(cohort)} for _ in range(4)],
    )
    receipts = {}
    try:
        for _ in range(40):
            _, generated = batch.next()
            for item in generated:
                if item.finish_reason:
                    receipts[item.uid] = item.mtp_receipt
            if len(receipts) == 4:
                break
        assert len(receipts) == 4
        assert all(
            4 in receipt["observed_compute_widths"]
            and receipt["num_draft"] == 1
            and receipt["requested_num_draft"] == 2
            and receipt["admission_stage"] == "lower_k"
            for receipt in receipts.values()
        )
        assert any(
            decision.stage == "lower_k"
            and decision.draft_depths == (1,) * 4
            for decision in decisions
        )
        assert not any(decision.stage == "fewer_lanes" for decision in decisions)
        assert decisions[-1].stage == "lower_k"
    finally:
        batch.close()


def test_atomic_cohort_memory_loss_fails_before_plain_fallback(monkeypatch):
    from mlx2.runtime.memory_policy import (
        SelfMTPLaneAdmissionController,
        _make_self_mtp_admission_callback,
    )

    monkeypatch.setenv("MLX_LM_SEGMENTED_SELF_MTP", "1")
    monkeypatch.setenv("MLX_LM_TRUE_BATCHED_SEGMENTED_MTP", "1")
    mx.random.seed(311)
    model = _tiny_qwen4_model()
    config = {
        "persistent": True,
        "num_draft": 2,
        "segment_aware_live_tip": True,
        "segment_aware_cohort_size": 4,
        "segment_aware_async_qsa_promotion": False,
    }
    controller = SelfMTPLaneAdmissionController(
        transient_gib_per_lane=1.0,
        saturation_lane_cap=4,
        verification_row_cap=12,
        cache_estimator=lambda _tokens: 0,
    )
    free = [controller.hard_reserve_gib + 3.5]
    callback = _make_self_mtp_admission_callback(
        controller, free_memory=lambda: free[0]
    )
    batch = BatchGenerator(
        model,
        completion_batch_size=4,
        self_mtp=config,
        mtp_admission=callback,
    )
    cohort = {"tenant_id": "tenant-a", "id": "fail-b4", "size": 4}
    uids = batch.insert(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
        max_tokens=[16] * 4,
        self_mtp_configs=[{"batch_cohort": dict(cohort)} for _ in range(4)],
    )
    try:
        # Prepare and merge the uniformly lowered cohort while it fits.
        batch.next()
        free[0] = controller.hard_reserve_gib
        failure = None
        generated = []
        for _ in range(12):
            _, step = batch.next()
            generated.extend(step)
            failures = batch.take_atomic_cohort_failures()
            if failures:
                failure = failures[0]
                break
        assert failure is not None
        assert failure["uids"] == tuple(uids)
        assert failure["cohort"] == cohort
        assert "remain wholly admitted" in failure["reason"]
        assert len(batch._plain_fallback_batch) == 0
        assert batch.scheduler_stats.get("starved_mtp_plain_fallbacks", 0) == 0
        assert not any(
            item.finish_reason and item.mtp_receipt is None for item in generated
        )
    finally:
        batch.remove(uids)
        batch.close()
