"""Per-request speculation opt-out (``"speculation": "off"``)."""

import mlx.core as mx
import pytest

from mlx2.runtime.sample_utils import LaneRNG
from mlx2.server import validate_request
from mlx2.serving import (
    REQUEST_PLAIN_DECODE_REASON,
    speculation_request_policy,
    speculation_request_receipt,
)

from test_batched_mtp import _tiny_qwen4_model

CHAT = {"messages": [{"role": "user", "content": "hi"}]}


def test_validate_request_accepts_auto_and_off_only():
    assert validate_request({**CHAT, "speculation": "off"})["speculation"] == "off"
    assert validate_request({**CHAT, "speculation": "auto"})["speculation"] == "auto"
    for value in ("on", "OFF", False, 0, ["off"]):
        with pytest.raises(ValueError, match="speculation must be"):
            validate_request({**CHAT, "speculation": value})


def test_policy_per_route():
    for route in ("native_mtp", "prompt_lookup"):
        policy = speculation_request_policy(route)
        assert policy["status"] == "plain_decode"
        assert policy["reason"] == REQUEST_PLAIN_DECODE_REASON
    assert speculation_request_policy("ordinary")["status"] == "not_applicable"
    refused = speculation_request_policy("external_draft")
    assert refused["status"] == "refused"
    assert "external_draft" in refused["reason"]


def test_receipt_reports_what_the_lane_ran():
    policy = speculation_request_policy("native_mtp")
    fallback = {
        "route": "ordinary_after_mtp_fallback",
        "mtp_plain_fallback": {
            "reason": REQUEST_PLAIN_DECODE_REASON,
            "stats_before_fallback": {"draft_proposed": 0},
        },
    }
    receipt = speculation_request_receipt(policy, fallback, None)
    assert receipt["schema"] == "mlx2.speculation-request.v1"
    assert receipt["decode"] == "ordinary"
    # A lane that speculated, or a fallback for another reason, is not the
    # opt-out: report it truthfully.
    assert speculation_request_receipt(
        policy, {"route": "segmented_self_mtp"}, None
    )["decode"] == "speculative"
    other = {
        "route": "ordinary_after_mtp_fallback",
        "mtp_plain_fallback": {"reason": "admission_plain"},
    }
    assert speculation_request_receipt(policy, other, None)["decode"] == "speculative"
    assert speculation_request_receipt(policy, None, None)["decode"] == "unobserved"
    pld = speculation_request_receipt(
        speculation_request_policy("prompt_lookup"),
        None,
        {"speculation": "disabled", "decode": "ordinary"},
    )
    assert (pld["speculation"], pld["decode"]) == ("disabled", "ordinary")
    ordinary = speculation_request_receipt(
        speculation_request_policy("ordinary"), None, None
    )
    assert ordinary["decode"] == "ordinary"


@pytest.mark.parametrize("segmented", [False, True])
def test_opted_out_lane_decodes_ordinary_beside_a_speculating_lane(segmented):
    from mlx2.runtime.generate import BatchGenerator

    previous_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    ordinary = mixed = None
    try:
        mx.random.seed(924)
        model = _tiny_qwen4_model()
        prompts = [[1, 7, 3, 9, 2], [4, 5, 6, 2, 8]]
        ordinary = BatchGenerator(
            model, completion_batch_size=2, prefill_batch_size=2, prefill_step_size=32
        )
        self_mtp = {"num_draft": 2, "persistent": True}
        if segmented:
            self_mtp.update(
                {"segment_aware_live_tip": True, "segment_aware_cohort_size": 2}
            )
        mixed = BatchGenerator(
            model,
            completion_batch_size=2,
            prefill_batch_size=2,
            prefill_step_size=32,
            self_mtp=self_mtp,
        )
        ordinary_uids = ordinary.insert(prompts, max_tokens=[8, 8])
        mixed_uids = mixed.insert(
            prompts,
            max_tokens=[8, 8],
            lane_rngs=[LaneRNG(31), LaneRNG(32)],
            self_mtp_configs=[
                {
                    "sampling_temp": 0.0,
                    "plain_decode_reason": REQUEST_PLAIN_DECODE_REASON,
                },
                {"sampling_temp": 0.0},
            ],
        )

        def finish(generator, uids):
            output = {uid: [] for uid in uids}
            terminal = {}
            for _ in range(100):
                _, responses = generator.next()
                for response in responses:
                    output[response.uid].append(response.token)
                    if response.finish_reason:
                        terminal[response.uid] = response
                if len(terminal) == len(uids):
                    return output, terminal
            raise AssertionError("generator did not terminate")

        ordinary_output, _ = finish(ordinary, ordinary_uids)
        mixed_output, terminal = finish(mixed, mixed_uids)
        # Greedy: both lanes match ordinary decode token for token.
        assert [ordinary_output[uid] for uid in ordinary_uids] == [
            mixed_output[uid] for uid in mixed_uids
        ]
        opted, speculating = (terminal[uid].mtp_receipt for uid in mixed_uids)
        assert opted["route"] == "ordinary_after_mtp_fallback"
        transition = opted["mtp_plain_fallback"]
        assert transition["reason"] == REQUEST_PLAIN_DECODE_REASON
        assert transition["one_way"] and not transition["selected"]
        assert transition["stats_before_fallback"]["draft_proposed"] == 0
        assert not transition["mtp_observed_compute_widths_before_fallback"]
        assert speculating["route"] != "ordinary_after_mtp_fallback"
        assert speculating["stats"]["draft_proposed"] > 0
        stats = mixed.scheduler_stats
        assert stats["mtp_plain_fallback_request_opt_out"] == 1
        assert stats["mtp_plain_fallback_lanes"] == 1
        receipt = speculation_request_receipt(
            speculation_request_policy("native_mtp"), opted, None
        )
        assert receipt["decode"] == "ordinary"
    finally:
        if ordinary is not None:
            ordinary.close()
        if mixed is not None:
            mixed.close()
        mx.set_default_device(previous_device)


def test_serving_opt_out_equals_ordinary_and_receipts_it(monkeypatch):
    from route_harness import (
        make_engine, make_external_engine, patch_host, run, tiny_muse_dflash, tiny_qwen38_mtp,
    )

    from mlx2.api_resources import CapabilityUnavailable

    patch_host(monkeypatch)
    model, vocab = tiny_qwen38_mtp()
    request = {"tokens": [3, 9, 4, 17, 5, 22, 6], "max_tokens": 12, "temperature": 0}
    ordinary = make_engine(model, vocab, mtp=False)
    native = make_engine(model, vocab, mtp=True, max_lanes=2)
    try:
        reference = run(ordinary, request)
        speculative = run(native, request)
        opted = run(native, {**request, "speculation": "off"})
        on_ordinary = run(ordinary, {**request, "speculation": "off"})
        counts = dict(native.counts)
    finally:
        ordinary.close()
        native.close()
    assert opted["tokens"] == reference["tokens"] == speculative["tokens"]
    assert "speculation_request" not in speculative["receipt"]
    receipt = opted["receipt"]["speculation_request"]
    assert (receipt["route"], receipt["status"], receipt["decode"]) == (
        "native_mtp", "plain_decode", "ordinary",
    )
    assert receipt["observed_route"] == "ordinary_after_mtp_fallback"
    assert counts["speculation_opt_out_requests"] == 1
    assert on_ordinary["tokens"] == reference["tokens"]
    assert on_ordinary["receipt"]["speculation_request"]["status"] == "not_applicable"

    muse, draft, muse_vocab = tiny_muse_dflash()
    external = make_external_engine(muse, draft, muse_vocab)
    try:
        with pytest.raises(CapabilityUnavailable, match="no per-request speculation opt-out"):
            external.submit({**request, "speculation": "off"})
    finally:
        external.close()
