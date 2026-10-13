"""L2-04 (sweep 2026-10-12): a shared-prefix attestation must not make the
Qwen3.8 segmented view refuse true-batched execution. The provider declares
no shared-prefix plan, so the cohort carries no shared-state identity."""

from route_harness import tiny_qwen38_mtp


def _qwen38_true_batched_stats(attestation):
    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.sample_utils import LaneRNG
    from mlx2.runtime.segmented_self_mtp import segmented_self_mtp_stats

    model, _vocab = tiny_qwen38_mtp()
    segmented_self_mtp_stats(reset=True)
    config = {"persistent": True, "segment_aware_live_tip": True,
              "segment_aware_cohort_size": 2, "num_draft": 2}
    gen = BatchGenerator(model, completion_batch_size=2, prefill_batch_size=2,
                         prefill_step_size=64, self_mtp=config)
    lane_cfg = {"sampling_temp": 0.0}
    if attestation:
        lane_cfg["shared_prefix_attestation"] = "same-apc-generation"
    prompt = [3, 5, 7, 11, 13, 17, 19, 23]
    gen.insert([prompt, prompt], max_tokens=[24, 24],
               lane_rngs=[LaneRNG(0), LaneRNG(1)],
               self_mtp_configs=[dict(lane_cfg), dict(lane_cfg)])
    tokens = {}
    for _ in range(40):
        for response in gen.next()[1]:
            tokens.setdefault(response.uid, []).append(response.token)
    stats = segmented_self_mtp_stats()
    return tokens, stats


def test_l2_04_shared_prefix_attestation_keeps_qwen38_true_batched():
    base_tokens, base = _qwen38_true_batched_stats(False)
    attested_tokens, attested = _qwen38_true_batched_stats(True)
    assert base["true_batched_requests"] > 0
    assert base["true_batched_declined"] == 0
    assert attested["true_batched_requests"] == base["true_batched_requests"]
    assert attested["true_batched_engaged"] == base["true_batched_engaged"]
    assert attested["true_batched_declined"] == 0
    assert attested_tokens == base_tokens
