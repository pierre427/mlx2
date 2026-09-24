"""Sampled batched MTP drafts are drawn on device, bit-identically (2026-09-23).

A lane with logits processors keeps the host path (``.item()`` per draft).
A no-op processor therefore gives a host-path oracle with the same
distribution and the same RNG draws: the device path must produce exactly
the same tokens, and must actually engage (counter).
"""
import mlx.core as mx
import pytest

from mlx2.runtime import round_levers
from mlx2.runtime.hybrid_speculative import (
    attach_self_mtp_lanes,
    commit_batched_self_mtp,
    prepare_self_mtp_lane,
    propose_batched_self_mtp,
)
from mlx2.runtime.sample_utils import LaneRNG
from test_batched_mtp import _tiny_qwen4_model


def _noop(tokens, logits):
    return logits


def _generate(model, processors, *, cycles=4, temps=(0.8, 0.8)):
    prompts = [[3, 7, 11, 5, 2, 9], [4, 1, 8, 6, 10]]
    lanes = [
        prepare_self_mtp_lane(
            mx.array(prompt, mx.uint32), model, uid=uid, max_tokens=64, prompt_cache=None,
            mtp_state=None, lane_rng=LaneRNG(900 + uid), num_draft=2, sampling_temp=temp,
            sampling_top_p=0.95, sampling_top_k=8, sampling_min_p=0.0, accept_rule="residual",
            logits_processors=list(processors), prefill_step_size=4, share_qsa_indices=False,
        )[0]
        for uid, (prompt, temp) in enumerate(zip(prompts, temps))
    ]
    batch = attach_self_mtp_lanes(model, None, lanes)
    emitted = [[] for _ in prompts]
    for _ in range(cycles):
        proposal = propose_batched_self_mtp(model, batch)
        for row, outs in enumerate(proposal.outputs):
            emitted[row].extend(int(tok.token) for tok in outs)
        commit_batched_self_mtp(batch, proposal,
                                emitted_counts=[len(r) for r in proposal.outputs],
                                terminal=[False] * len(prompts))
    return emitted


@pytest.mark.parametrize("temps", [(0.8, 0.8), (0.8, 0.0)], ids=["sampled", "mixed"])
def test_device_drafts_match_the_host_path_exactly(temps):
    model = _tiny_qwen4_model()
    round_levers.reset_counters()
    device = _generate(model, [], temps=temps)
    engaged = round_levers.counters()["device_sampled_drafts"]
    round_levers.reset_counters()
    host = _generate(model, [_noop], temps=temps)
    assert engaged > 0, "the device-draft path never engaged"
    assert round_levers.counters()["device_sampled_drafts"] == 0
    assert device == host
    assert all(len(row) >= 4 for row in device)


def test_greedy_cycles_do_not_touch_the_sampled_path():
    model = _tiny_qwen4_model()
    round_levers.reset_counters()
    _generate(model, [], temps=(0.0, 0.0))
    assert round_levers.counters()["device_sampled_drafts"] == 0
