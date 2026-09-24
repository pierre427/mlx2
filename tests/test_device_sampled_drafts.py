"""Sampled batched MTP drafts are drawn on device, bit-identically (2026-09-23).

``DEVICE_SAMPLED_DRAFTS = False`` restores the host path (``.item()`` per
draft), which is the oracle: same distribution, same RNG draws. The device
path must produce exactly the same tokens and must actually engage (counter),
including with logits processors -- Qwen's non-thinking profile always
carries presence_penalty 1.5, and an earlier processor gate silently kept
every such request on the host path.
"""
import mlx.core as mx
import pytest

from mlx2.runtime import hybrid_speculative, round_levers
from mlx2.runtime.hybrid_speculative import (
    attach_self_mtp_lanes,
    commit_batched_self_mtp,
    prepare_self_mtp_lane,
    propose_batched_self_mtp,
)
from mlx2.runtime.sample_utils import LaneRNG, make_presence_penalty
from test_batched_mtp import _tiny_qwen4_model


def _processors(kind):
    if kind == "presence":
        return [make_presence_penalty(1.5, 0)]
    return []


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


@pytest.mark.parametrize("kind", ["none", "presence"])
@pytest.mark.parametrize("temps", [(0.8, 0.8), (0.8, 0.0)], ids=["sampled", "mixed"])
def test_device_drafts_match_the_host_path_exactly(temps, kind, monkeypatch):
    model = _tiny_qwen4_model()
    round_levers.reset_counters()
    device = _generate(model, _processors(kind), temps=temps)
    engaged = round_levers.counters()["device_sampled_drafts"]
    round_levers.reset_counters()
    monkeypatch.setattr(hybrid_speculative, "DEVICE_SAMPLED_DRAFTS", False)
    host = _generate(model, _processors(kind), temps=temps)
    assert engaged > 0, "the device-draft path never engaged"
    assert round_levers.counters()["device_sampled_drafts"] == 0
    assert device == host
    assert all(len(row) >= 4 for row in device)


def test_greedy_cycles_do_not_touch_the_sampled_path():
    model = _tiny_qwen4_model()
    round_levers.reset_counters()
    _generate(model, [], temps=(0.0, 0.0))
    assert round_levers.counters()["device_sampled_drafts"] == 0
