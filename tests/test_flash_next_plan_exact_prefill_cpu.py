"""Flash-Next self-MTP prefill keeps the idle plan's slices beside decode.

GPU 2026-10-28 (Flash-Next default route, 4 lanes): decode-fairness caps
shortened the slices of lanes prefilled beside live decode, the prompt state
left the planned grid, and APCv2 refused it (apcv2_store_skipped_slice_schedule
28), so warm requests missed 0/4 at 4K.  ``prefill_plan_exact`` makes load
defer a slice instead of reshaping it; a live memory cap still wins.  CPU
only, on the tiny hybrid model the slice-schedule tests use.
"""

import mlx.core as mx
import pytest

from mlx2.adapters.flash_next_policy import FlashNextPolicy
from mlx2.runtime.generate import BatchGenerator
from mlx2.runtime.sample_utils import LaneRNG
from test_batched_mtp import _tiny_qwen4_model

STEP = 32


@pytest.fixture(scope="module")
def model():
    mx.random.seed(41)
    return _tiny_qwen4_model()


def _run(model, prompt, monkeypatch, *, plan_exact, fair_cap=None, live_cap=None):
    if fair_cap is not None:
        original = BatchGenerator._advance_mtp_prefill

        def fair(self, index, max_tokens):
            # The chunk a contended round hands in after its fairness cap.
            return original(self, index, min(max_tokens, fair_cap))

        monkeypatch.setattr(BatchGenerator, "_advance_mtp_prefill", fair)
    if live_cap is not None:
        original_step = BatchGenerator._prompt_prefill_step
        monkeypatch.setattr(
            BatchGenerator, "_prompt_prefill_step",
            lambda self, total: min(original_step(self, total), live_cap),
        )
    self_mtp = {"num_draft": 2, "persistent": True}
    if plan_exact:
        self_mtp["prefill_plan_exact"] = True
    gen = BatchGenerator(
        model, prefill_step_size=STEP, completion_batch_size=1,
        prefill_batch_size=1, adaptive_prefill=True, self_mtp=self_mtp,
    )
    try:
        uid = gen.insert(
            [prompt], max_tokens=[3], lane_rngs=[LaneRNG(5)], mtp_states=[None],
            self_mtp_configs=[{"sampling_temp": 0.0, "num_draft": 2}],
        )[0]
        schedule = None
        for _ in range(1000):
            prompts, responses = gen.next()
            if schedule is None and any(
                getattr(p, "end_of_prompt", False) and p.uid == uid for p in prompts
            ):
                schedule = gen.pop_prefill_slice_schedule(uid)
            if any(r.finish_reason for r in responses):
                break
        else:
            raise AssertionError("request did not finish")
        if schedule is None:
            schedule = gen.pop_prefill_slice_schedule(uid)
        return schedule, dict(gen.scheduler_stats)
    finally:
        gen.close()


PROMPT = [(5 * i + 3) % 60 + 2 for i in range(150)]


def test_fairness_cap_takes_the_schedule_off_plan_without_the_law(model, monkeypatch):
    schedule, _ = _run(model, PROMPT, monkeypatch, plan_exact=False, fair_cap=8)
    assert not schedule["planned"] and schedule["deviation_reason"] == "width"


def test_plan_exact_law_keeps_planned_slices_under_a_fairness_cap(model, monkeypatch):
    schedule, stats = _run(model, PROMPT, monkeypatch, plan_exact=True, fair_cap=8)
    assert schedule["planned"], schedule
    assert schedule["boundaries"][0] == STEP
    assert schedule["covered"] == len(PROMPT) - 1
    assert stats["mtp_prefill_plan_exact_reshaped"] >= 1


def test_plan_exact_law_is_inert_when_idle(model, monkeypatch):
    schedule, stats = _run(model, PROMPT, monkeypatch, plan_exact=True)
    assert schedule["planned"], schedule
    assert stats["mtp_prefill_plan_exact_reshaped"] == 0


def test_live_memory_cap_still_wins_over_the_law(model, monkeypatch):
    schedule, stats = _run(model, PROMPT, monkeypatch, plan_exact=True, live_cap=8)
    assert max(b - a for a, b in zip([0] + schedule["boundaries"], schedule["boundaries"])) <= 8
    assert stats["mtp_prefill_plan_exact_reshaped"] == 0


def test_flash_next_policy_declares_the_law_and_the_8192_cell():
    policy = FlashNextPolicy()
    assert policy.prefill_step == 8192
    assert policy.batch_config(max_lanes=4, prefill_step=8192)["prefill_plan_exact"] is True
    off = FlashNextPolicy(prefill_plan_exact_slices=False)
    assert "prefill_plan_exact" not in off.batch_config(max_lanes=4, prefill_step=8192)


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None, [], {}])
def test_plan_exact_policy_refuses_nonboolean_values(value):
    # In particular, a JSON string "false" is truthy and used to silently
    # select the exact-slice law despite the operator's apparent opt-out.
    with pytest.raises(ValueError, match="prefill_plan_exact_slices must be boolean"):
        FlashNextPolicy.from_mapping({"prefill_plan_exact_slices": value})
