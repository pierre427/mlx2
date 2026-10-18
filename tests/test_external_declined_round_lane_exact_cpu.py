"""A progressive-declined external round at B>1 keeps each lane's B1 bits.

When progressive verify declines a cohort round (a stop token in a proposal,
too few proposals), the standard verify used to run the whole cohort as one
batched forward over segmented views: multi-row SDPA and batched recurrent,
MoE and projection kernels whose bits depend on which lanes share the round.
That is how the Nemotron 3.5 Super VL external-AST width-4 ladder kept
cold/warm differences while width 1 was exact.  The round now launches the
adapter-owned row-exact executor progressive verify already uses, so every
row gets its own lane's one-token decode arithmetic; the receipt names the
path.  ``test_external_declined_round_lane_exact_metal.py`` pins the bits.
"""

import copy

import mlx.core as mx
import numpy as np
import pytest

from mlx2.adapters.xpress import XPressConfig
from mlx2.runtime.drafters.xpress import XPressDraftModel
from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
from mlx2.runtime.models.nemotron35_super import Model as SuperModel
from mlx2.runtime.models.nemotron_h import Model as LanguageModel
from mlx2.runtime.sample_utils import LaneRNG
from tests.test_nemotron35_progressive_cpu import close_caches
from tests.test_nemotron_external_taps_cpu import tiny_model

PROPOSALS = 7


def _selected_target(device=None):
    mx.set_default_device(mx.cpu if device is None else device)
    mx.random.seed(81)
    args = copy.copy(tiny_model().args)
    # A 32-wide state lets Metal's sequential SSM kernels (serial-exact for
    # 1..8 rows) run; narrower states always take the matrix form.
    args.ssm_state_size = 32
    model = SuperModel.__new__(SuperModel)
    LanguageModel.__init__(model, args)
    model.set_short_segment_policy("layerwise_short_exact_routed_batch")
    for layer in model.layers:
        if layer.block_type == "M":
            layer.mixer.set_conv_policy("fused_short")
            layer.mixer.set_ssm_policy("fused_short")
            layer.mixer.set_projection_policy("short_row_exact")
    return model


def _greedy_rows(model, prompts):
    rows = []
    for prompt in prompts:
        cache = model.make_cache()
        model(mx.array([prompt[:-1]]), cache)
        token, row = prompt[-1], []
        for _ in range(PROPOSALS):
            token = int(mx.argmax(model(mx.array([[token]]), cache)[0, -1]).item())
            row.append(token)
        rows.append(row)
    return rows


def _run(prompts, *, stop, device=None, tile=4):
    model = _selected_target(device)
    exact_rows = dict(zip(map(tuple, prompts), _greedy_rows(model, prompts)))
    args = model.args
    draft = XPressDraftModel(XPressConfig(
        hidden_size=8, intermediate_size=12, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4,
        vocab_size=16, mask_token_id=15, num_target_layers=5,
        target_layer_ids=[0, 1, 4], block_size=PROPOSALS + 1,
        layer_types=["full_attention"], xpress_rank=3,
        xpress_mlp_hidden=5, xpress_num_passes=3,
    )).bind(model)
    anchors = [prompt[-1] for prompt in prompts]
    rows = [exact_rows[tuple(prompt)] for prompt in prompts]
    original = draft.draft_distributions

    def forced(round_anchors, hidden, caches, count, rngs, temps, **kwargs):
        generated = original(round_anchors, hidden, caches, count, rngs, temps, **kwargs)
        if list(round_anchors) != anchors:
            return generated
        draft.adaptive_confidence_features = [[2.0] * PROPOSALS for _ in rows]
        laws = []
        for row in rows:
            row_laws = []
            for token in row:
                law = np.zeros(args.vocab_size, np.float64)
                law[token] = 1.0
                row_laws.append(law)
            laws.append(row_laws)
        return [list(row) for row in rows], laws

    draft.draft_distributions = forced
    batch = ExternalDraftBatchGenerator(
        model,
        draft_model=draft,
        binding="declined-round-lane-exact-cpu",
        num_draft=PROPOSALS,
        completion_batch_size=len(prompts),
        prefill_step_size=8,
        progressive_verification_tile=tile,
        stop_tokens=[[stop]],
    )
    uids = batch.insert(
        prompts,
        max_tokens=[16] * len(prompts),
        lane_rngs=[LaneRNG(1200 + row) for row in range(len(prompts))],
        sampling_configs=[{"sampling_temp": 0}] * len(prompts),
    )
    lanes = [batch.lanes[uid] for uid in uids]
    for lane in lanes:
        while lane.anchor is None:
            batch._prefill(lane)
    batch._round(lanes)
    return batch, lanes, rows


PROMPTS = [[1, 2, 3, 4], [3, 5, 7, 9]]


def test_stop_declined_b2_round_runs_lane_exact_and_matches_b1():
    probe = _selected_target()
    rows = _greedy_rows(probe, PROMPTS)
    stop = rows[0][5]  # inside lane 0's proposal: progressive declines
    batch, lanes, _ = _run(PROMPTS, stop=stop)
    references = []
    try:
        stats = batch.scheduler_stats
        assert stats.get("external_cohort_target_lane_exact_standard_verify_rounds") == 1
        assert not stats.get("external_cohort_target_batched_standard_verify_rounds")
        for index, (prompt, lane) in enumerate(zip(PROMPTS, lanes)):
            receipt = lane.ready[-1].speculative_receipt["progressive_verification"]
            assert receipt["cohort_target_execution"] == {"standard_verify:lane_exact": 1}
            ref_batch, (ref_lane,), _ = _run([prompt], stop=stop)
            references.append(ref_batch)
            assert lane.history == ref_lane.history
            assert lane.anchor == ref_lane.anchor
            assert [item.token for item in lane.ready] == [
                item.token for item in ref_lane.ready
            ]
            # CPU refuses the fused short kernels, so tile widths still move
            # the last ulp here (as in the progressive CPU parity tests); the
            # Metal test pins exact bits.
            close_caches(lane.cache, ref_lane.cache)
    finally:
        batch.close()
        for ref in references:
            ref.close()


def test_unselected_target_keeps_the_batched_forward(monkeypatch):
    """Without the adapter-owned executor the counters say so truthfully."""
    monkeypatch.setattr(
        SuperModel, "progressive_external_target_rows", property(lambda self: None)
    )
    probe = _selected_target()
    stop = _greedy_rows(probe, PROMPTS)[0][5]
    batch, lanes, _ = _run(PROMPTS, stop=stop)
    try:
        stats = batch.scheduler_stats
        assert stats.get("external_cohort_target_batched_standard_verify_rounds") == 1
        assert not stats.get("external_cohort_target_lane_exact_standard_verify_rounds")
        receipt = lanes[0].ready[-1].speculative_receipt["progressive_verification"]
        assert receipt["cohort_target_execution"] == {"standard_verify:batched": 1}
    finally:
        batch.close()


def _ordinary_steps(prompts, *, device=None, steps=3, tile=4, guard=None):
    batch, lanes, _ = _run(prompts, stop=15, device=device, tile=tile)
    if guard is not None:
        guard(batch)
    for lane in lanes:
        lane.ordinary = True
    for _ in range(steps):
        batch._ordinary_round(lanes)
    return batch, lanes


def test_ordinary_b2_round_runs_lane_exact():
    batch, lanes = _ordinary_steps(PROMPTS)
    try:
        stats = batch.scheduler_stats
        assert stats.get("external_cohort_target_lane_exact_ordinary_round_rounds") == 3
        assert not stats.get("external_cohort_target_batched_ordinary_round_rounds")
        for lane in lanes:
            assert lane.cohort_target_execution["ordinary_round:lane_exact"] == 3
    finally:
        batch.close()


def test_ordinary_round_without_exact_progressive_keeps_logits_only_call():
    """Codex round 11: with no lane-exact executor selected, a B2 hybrid
    ordinary round must keep the parent's logits-only model call and build
    no feature-capture graph."""

    def forbid_taps(batch):
        def raising(*_args, **_kwargs):
            raise AssertionError("ordinary round built a feature-capture forward")

        batch.model.forward_with_taps = raising

    batch, lanes = _ordinary_steps(PROMPTS, tile=None, guard=forbid_taps)
    try:
        stats = batch.scheduler_stats
        assert stats.get("external_cohort_target_batched_ordinary_round_rounds") == 3
        assert not stats.get("external_cohort_target_lane_exact_ordinary_round_rounds")
        for lane in lanes:
            assert lane.cohort_target_execution["ordinary_round:batched"] == 3
    finally:
        batch.close()


class _KVTransactionStub:
    """A SegmentedKVTransaction-like B2 transaction: no ``batched`` attribute."""

    caches = ()


def test_kv_cohort_transaction_is_classified_by_lane_count():
    """Codex round 11: a B2+ KV transaction (no ``batched`` attribute) ran
    the batched forward but was recorded as ``single_lane``."""
    from types import SimpleNamespace

    calls = []

    class _Model:
        def forward_with_taps(self, inputs, caches, layers):
            calls.append(int(inputs.shape[0]))
            return mx.zeros((inputs.shape[0], inputs.shape[1], 4)), mx.zeros((1,))

    batch = ExternalDraftBatchGenerator.__new__(ExternalDraftBatchGenerator)
    batch.model = _Model()
    batch.layers = (0,)
    batch.scheduler_stats = {}
    batch.progressive_verification_fidelity = None
    lanes = [SimpleNamespace(), SimpleNamespace()]
    batch._cohort_target_forward(
        mx.zeros((2, 3), dtype=mx.int32), _KVTransactionStub(), lanes,
        site="standard_verify",
    )
    single = [SimpleNamespace()]
    batch._cohort_target_forward(
        mx.zeros((1, 3), dtype=mx.int32), _KVTransactionStub(), single,
        site="standard_verify",
    )
    assert calls == [2, 1]
    assert batch.scheduler_stats == {
        "external_cohort_target_batched_standard_verify_rounds": 1,
        "external_cohort_target_single_lane_standard_verify_rounds": 1,
    }
    assert [lane.cohort_target_execution for lane in lanes] == [
        {"standard_verify:batched": 1}
    ] * 2
    assert single[0].cohort_target_execution == {"standard_verify:single_lane": 1}
