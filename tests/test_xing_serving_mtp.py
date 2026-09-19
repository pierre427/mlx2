"""Xing4.0 through the real batch scheduler: continuous self-MTP and APCv2.

Greedy self-MTP must emit exactly the ordinary greedy tokens, for lanes that
start together, for a lane that joins a running MTP cohort, and for lanes
resumed from an APCv2 prefix hit (with its MTP sidecar) on a COW branch.
"""

import json
from pathlib import Path

import mlx.core as mx
import pytest

from mlx2.runtime.apc_v2 import APCv2, MTPAPCSidecar
from mlx2.runtime.generate import BatchGenerator
from mlx2.runtime.models.xing4_0 import Model, ModelArgs
from mlx2.runtime.sample_utils import LaneRNG

FIXTURE = Path(__file__).parent / "fixtures" / "xing4_0_tiny"
PROMPTS = [[3, 17, 5, 9, 22, 41, 7], [8, 8, 30, 2, 11], [19, 4, 4, 27, 13, 6, 25, 1, 33]]
MAX_TOKENS = 10


@pytest.fixture(scope="module")
def model():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    config = json.loads((FIXTURE / "config.json").read_text())
    model = Model(ModelArgs.from_dict(config))
    weights = model.sanitize(mx.load(str(FIXTURE / "weights.safetensors")))
    model.load_weights(list(weights.items()), strict=True)
    model.eval()
    mx.eval(model.parameters())
    assert model.mtp is not None
    yield model
    mx.set_default_device(previous)


@pytest.fixture(autouse=True)
def _cpu_streams(monkeypatch):
    new_stream = mx.new_stream
    monkeypatch.setattr(mx, "new_stream", lambda device: new_stream(mx.cpu))


def _drain(batch, *, on_step=None, on_prompt=None, expected=1, limit=400):
    emitted = {}
    for step in range(limit):
        if on_step is not None:
            on_step(step, emitted)
        prompts, responses = batch.next()
        for response in prompts:
            if on_prompt is not None:
                on_prompt(response)
        for response in responses:
            row = emitted.setdefault(response.uid, [])
            row.append(response.token)
            if response.finish_reason:
                row.append("done")
        if len(emitted) >= expected and all(row[-1] == "done" for row in emitted.values()):
            return {uid: row[:-1] for uid, row in emitted.items()}
    raise AssertionError("batch failed to complete")


def _ordinary(model, prompt):
    batch = BatchGenerator(model, completion_batch_size=1, prefill_step_size=4)
    try:
        batch.insert([prompt], max_tokens=[MAX_TOKENS])
        return next(iter(_drain(batch).values()))
    finally:
        batch.close()


@pytest.fixture(scope="module")
def reference(model):
    return [_ordinary(model, prompt) for prompt in PROMPTS]


def _mtp_config(segmented):
    config = {"persistent": True, "num_draft": 2}
    if segmented:
        config.update(segment_aware_live_tip=True, segment_aware_cohort_size=3)
    return config


@pytest.mark.parametrize("segmented", [False, True])
def test_self_mtp_cohort_and_late_join_match_ordinary_greedy(model, reference, monkeypatch, segmented):
    monkeypatch.setenv("MLX_LM_SEGMENTED_SELF_MTP", "1" if segmented else "0")
    monkeypatch.setenv("MLX_LM_TRUE_BATCHED_SEGMENTED_MTP", "1" if segmented else "0")
    batch = BatchGenerator(
        model, completion_batch_size=3, prefill_step_size=4, self_mtp=_mtp_config(segmented)
    )
    uids = {}

    def join_late(step, emitted):
        if step == 3 and 2 not in uids.values():
            (uid,) = batch.insert(
                [PROMPTS[2]], max_tokens=[MAX_TOKENS], lane_rngs=[LaneRNG(5)],
                self_mtp_configs=[{"sampling_temp": 0.0}],
            )
            uids[uid] = 2
            assert emitted, "the first cohort must already be decoding"

    try:
        first = batch.insert(
            PROMPTS[:2], max_tokens=[MAX_TOKENS] * 2, lane_rngs=[LaneRNG(3), LaneRNG(4)],
            self_mtp_configs=[{"sampling_temp": 0.0}] * 2,
        )
        uids.update({uid: index for index, uid in enumerate(first)})
        result = _drain(batch, on_step=join_late, expected=3)
    finally:
        batch.close()
    assert sorted(uids.values()) == [0, 1, 2]
    for uid, index in uids.items():
        assert result[uid] == reference[index], (segmented, index)


def test_apcv2_prefix_hit_with_mtp_sidecar_resumes_exactly(model, reference, monkeypatch):
    monkeypatch.setenv("MLX_LM_SEGMENTED_SELF_MTP", "0")
    apc = APCv2(layout_name=model.apc_v2_layout)
    key = apc.key("xing-tiny", revision="apc")
    prompt = PROMPTS[0]

    def store(response):
        boundary = batch.pop_prompt_boundary(response.uid)
        if boundary:
            apc.store(
                key, boundary["tokens"], boundary["target_cache"],
                sidecar=MTPAPCSidecar(boundary["mtp_state"], boundary["covered_tokens"]),
            )

    batch = BatchGenerator(model, completion_batch_size=2, prefill_step_size=4, self_mtp=_mtp_config(False))
    try:
        batch.insert([prompt], max_tokens=[MAX_TOKENS], lane_rngs=[LaneRNG(3)],
                     self_mtp_configs=[{"sampling_temp": 0.0}])
        cold = next(iter(_drain(batch, on_prompt=store).values()))
    finally:
        batch.close()
    assert cold == reference[0]

    # Two concurrent hits branch the same stored prefix (COW) and must not
    # disturb each other or the stored entry.
    hits = [apc.lookup(key, prompt), apc.lookup(key, prompt)]
    assert all(hit.cached_tokens == len(prompt) - 1 for hit in hits)
    batch = BatchGenerator(model, completion_batch_size=2, prefill_step_size=4, self_mtp=_mtp_config(False))
    try:
        batch.insert(
            [hit.remaining_tokens for hit in hits], max_tokens=[MAX_TOKENS] * 2,
            caches=[hit.cache for hit in hits], all_tokens=[prompt[:-1]] * 2,
            mtp_states=[hit.sidecar.state for hit in hits],
            lane_rngs=[LaneRNG(3), LaneRNG(4)],
            self_mtp_configs=[{"sampling_temp": 0.0}] * 2,
        )
        warm = _drain(batch, expected=2)
    finally:
        batch.close()
        for hit in hits:
            hit.cache.close()
    assert list(warm.values()) == [reference[0], reference[0]]
    again = apc.lookup(key, prompt)
    try:
        assert again.cached_tokens == len(prompt) - 1
    finally:
        again.cache.close()
        apc.clear()


def test_batched_prompt_lookup_verify_matches_per_lane_forward(model):
    """Batched PLD verify (SegmentedKVView rows) equals each lane alone.

    GPU 2026-09-18: the batched verify path handed MLA a segmented view whose
    append returns no arrays, and Xing refused it as a quantized cache.
    """
    import copy

    from mlx2.runtime.segmented_rotating_kv import SegmentedKVRows

    histories = [[3, 17, 5, 9, 22, 41, 7], [8, 8, 30, 2, 11]]
    blocks = [[50, 51, 52], [60, 61]]
    lanes = []
    for history in histories:
        cache = model.make_cache()
        model(mx.array([history]), cache=cache)
        mx.eval([c.state for c in cache])
        lanes.append(cache)
    want = []
    for cache, block in zip(lanes, blocks):
        alone = copy.deepcopy(cache)
        want.append(model(mx.array([block]), cache=alone)[0])
    width = max(len(block) for block in blocks)
    transaction = SegmentedKVRows(lanes).begin(lengths=[len(block) for block in blocks])
    try:
        padded = [block + [0] * (width - len(block)) for block in blocks]
        logits = model(mx.array(padded), cache=transaction.caches)
        mx.eval(logits)
        for row, (block, expected) in enumerate(zip(blocks, want)):
            assert mx.allclose(logits[row, : len(block)], expected, rtol=1e-4, atol=1e-4).item()
    finally:
        transaction.abort()
