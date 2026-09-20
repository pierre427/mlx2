"""Item 3: DFlash2 blocks at the trained width. CPU-only; no artifact loads."""
import json

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)
from mlx2.adapters.muse_glimmer_config import ModelArgs
from mlx2.runtime.drafters.dflash2 import DFlash2DraftModel
from mlx2.runtime.drafters.dflash2_config import DFlash2Config
from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
from mlx2.runtime.models.muse_glimmer import Model
from mlx2.runtime.speculative_sampling import RequestRNG

BLOCK = 8
WINDOW = 5  # draft rotating ring keeps WINDOW - 1 = 4 context rows


def _draft_config(**overrides):
    values = dict(
        hidden_size=8, intermediate_size=16, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4,
        vocab_size=32, num_target_layers=4, target_layer_ids=[0, 3],
        conv_kernel_size=2, conv_group_size=2, selector_rank=4,
        selector_top_k=4, block_size=BLOCK, mask_token_id=31,
        max_position_embeddings=256, sliding_window=WINDOW,
        layer_types=["sliding_attention", "full_attention"],
    )
    values.update(overrides)
    return DFlash2Config(**values)


def tiny():
    mx.random.seed(8)
    m = Model(ModelArgs(
        hidden_size=8, intermediate_size=16, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4,
        vocab_size=32, sliding_window=3, max_position_embeddings=256,
    ))
    return m, DFlash2DraftModel(_draft_config()).bind(m)


def generator(m, d, num_draft, **kwargs):
    return ExternalDraftBatchGenerator(
        m, draft_model=d, binding="test", num_draft=num_draft,
        prefill_step_size=3, **kwargs,
    )


def drain(b, limit=400):
    output, receipts = {}, {}
    for _ in range(limit):
        _, responses = b.next()
        for r in responses:
            output.setdefault(r.uid, []).append(r.token)
            receipts.setdefault(r.uid, []).append(r.speculative_receipt)
        if not b.lanes:
            return output, receipts
    raise AssertionError("scheduler stalled")


def greedy_reference(m, prompt, count):
    cache = m.make_cache(); tokens = list(prompt); out = []
    for i in range(count):
        logits = m(mx.array([tokens if i == 0 else [tokens[-1]]]), cache=cache)
        token = int(mx.argmax(logits[0, -1]).item()); tokens.append(token); out.append(token)
    return out


def _force_accept_all(monkeypatch, d, lane_of, full):
    """Propose the target's own greedy continuation so every row is accepted.

    The tiny random drafter almost never agrees with the tiny target; this
    keeps full K+1-row commits (the widest draft-context append) exercised
    while the real drafter still runs and appends its context.
    """
    real = d.draft_distributions

    def proposals(anchors, hidden, cache, count, rngs, temperatures, **kwargs):
        real(anchors, hidden, cache, count, rngs, temperatures, **kwargs)
        start = len(lane_of().history) + 1
        tokens = [list(full[start:start + count])]
        return tokens, [[np.eye(32)[t] for t in tokens[0]]]

    monkeypatch.setattr(d, "draft_distributions", proposals)


@pytest.mark.parametrize("prompts", [[[1, 2, 3, 4, 5]], [[1, 2, 3, 4, 5], [3, 1], [7, 7, 2], [9]]])
def test_trained_width_greedy_is_exact_and_receipts_show_full_block(prompts):
    m, d = tiny()
    b = generator(m, d, BLOCK - 1)
    count = 19
    ids = b.insert(prompts, max_tokens=[count] * len(prompts),
                   sampling_configs=[{"sampling_temp": 0}] * len(prompts))
    got, receipts = drain(b)
    for uid, prompt in zip(ids, prompts):
        assert got[uid] == greedy_reference(m, prompt, count)
        proposed = [r["round_proposed"] for r in receipts[uid]]
        assert max(proposed) == BLOCK - 1
        # Budget clamp: a round never proposes past the lane's remaining budget.
        assert all(p <= count - 1 for p in proposed)
    assert b.scheduler_stats["proposed_tokens"] >= (BLOCK - 1) * len(prompts)
    if len(prompts) > 1:
        assert b.scheduler_stats["target_max_width"] == len(prompts)


@pytest.mark.parametrize("num_draft", [0, BLOCK, BLOCK + 1])
def test_executor_rejects_count_outside_trained_block(num_draft):
    m, d = tiny()
    with pytest.raises(ValueError, match="trained block"):
        generator(m, d, num_draft)


def _header_only_pair(tmp_path, block_size):
    from mlx2.adapters.dflash2 import _expected_weight_shapes
    from test_external_dflash2_cpu import _write_safetensors_headers

    target = tmp_path / "target"; draft = tmp_path / "draft"
    target.mkdir(); draft.mkdir()
    (target / "config.json").write_text(json.dumps({
        "model_type": "muse_glimmer", "hidden_size": 8, "vocab_size": 32,
        "num_hidden_layers": 4,
    }))
    config = {
        "architectures": ["DFlash2DraftModel"], "model_type": "qwen3",
        "dtype": "bfloat16", "hidden_size": 8, "intermediate_size": 16,
        "num_hidden_layers": 2, "num_attention_heads": 2,
        "num_key_value_heads": 1, "head_dim": 4, "vocab_size": 32,
        "max_position_embeddings": 128, "sliding_window": 3,
        "layer_types": ["sliding_attention", "sliding_attention"],
        "num_target_layers": 4,
        "dflash_config": {
            "block_size": block_size, "mask_token_id": 31,
            "target_layer_ids": [0, 3], "conv_kernel_size": 2,
            "conv_group_size": 2, "selector_rank": 4, "selector_top_k": 4,
        },
    }
    (draft / "config.json").write_text(json.dumps(config))
    _write_safetensors_headers(
        draft / "model.safetensors",
        _expected_weight_shapes(DFlash2Config.from_dict(config)),
    )
    return target, draft


@pytest.mark.parametrize("block_size", [8, 16])
def test_muse_policy_bounds_num_draft_by_artifact_block_size(tmp_path, block_size):
    from mlx2.adapters.muse_glimmer import MuseGlimmerAdapter

    target, draft = _header_only_pair(tmp_path, block_size)
    for bad in (0, block_size, block_size + 1, True, 7.0):
        with pytest.raises(ValueError, match="num_draft"):
            MuseGlimmerAdapter(str(target), execution_policy={
                "draft_model": str(draft), "num_draft": bad,
            })
    # The widest legal count passes policy validation and only then reaches
    # target artifact inspection (the header-only target has no weights).
    with pytest.raises(Exception) as error:
        MuseGlimmerAdapter(str(target), execution_policy={
            "draft_model": str(draft), "num_draft": block_size - 1,
        })
    assert "num_draft" not in str(error.value)


def test_round_admission_charges_full_verify_width(monkeypatch):
    m, d = tiny()
    appends = []
    b = generator(m, d, BLOCK - 1, memory_headroom=lambda: 10**12)
    real = b._admit

    def spy(lanes, append, *, prefill=False):
        if not prefill:
            appends.append(append)
        return real(lanes, append, prefill=prefill)

    monkeypatch.setattr(b, "_admit", spy)
    b.insert([[1, 2, 3], [4, 5, 6]], max_tokens=[30, 30],
             sampling_configs=[{"sampling_temp": 0}] * 2)
    drain(b)
    assert max(appends) == BLOCK  # anchor + num_draft rows per lane
    # Reservation is monotone in verify width: the budget that fits a
    # num_draft=4 round defers a num_draft=7 round without mutating the lane.
    m, d = tiny()
    probe = generator(m, d, BLOCK - 1, memory_headroom=lambda: 10**12)
    uid = probe.insert([[1, 2, 3]], max_tokens=[30])[0]
    lane = probe.lanes[uid]
    while lane.anchor is None:
        probe._prefill(lane)
    probe._admit([lane], 5); narrow = probe.scheduler_stats["reservation_bytes"]
    probe._admit([lane], BLOCK); wide = probe.scheduler_stats["reservation_bytes"]
    assert wide > narrow
    probe.memory_headroom = lambda: narrow
    before = (lane.rng.snapshot(), list(lane.history), lane.generated)
    assert probe.next() == ([], [])
    assert (lane.rng.snapshot(), list(lane.history), lane.generated) == before
    assert probe.scheduler_stats["memory_deferred"] >= 1
    probe.close()


def _context_keys(entry):
    if hasattr(entry, "_temporal_order"):
        return np.asarray(entry._temporal_order(entry.keys))
    return np.asarray(entry.keys[..., : entry.offset, :])


@pytest.mark.parametrize("accept_all", [False, True])
def test_rotating_draft_window_holds_last_rows_after_wide_commits(monkeypatch, accept_all):
    """K+1-row commits must leave exactly the last window-1 context rows.

    Reference: a fresh draft ring fed the committed transcript's target taps
    in one append. Keys are RoPE'd at absolute positions, so equality of the
    ring (in temporal order) proves both the trim and the positions.
    """
    m, d = tiny()
    prompt = [1, 2, 3, 4, 5]
    count = 40
    full = prompt + greedy_reference(m, prompt, count)
    b = generator(m, d, BLOCK - 1)
    uid = b.insert([prompt], max_tokens=[count], sampling_configs=[{"sampling_temp": 0}])[0]
    lane = b.lanes[uid]
    if accept_all:
        _force_accept_all(monkeypatch, d, lambda: lane, full)
    accepted = []
    while uid in b.lanes and len(lane.history) <= WINDOW + 2 * BLOCK:
        _, responses = b.next()
        accepted.extend(r.speculative_receipt["round_accepted"] for r in responses)
    assert uid in b.lanes
    committed = list(lane.history)[: len(lane.history) - int(lane.tail.shape[1])]
    assert len(committed) > 2 * WINDOW  # the ring has wrapped
    assert lane.history + [lane.anchor] == full[: len(lane.history) + 1]
    if accept_all:
        assert max(accepted) == BLOCK - 1
    fresh = d.make_cache()
    taps = m.prefill_body(
        mx.array([committed]), m.make_cache(), list(d.config.target_layer_ids)
    )
    d.append_context(taps, fresh)
    for layer, (live, ref) in enumerate(zip(lane.draft_cache, fresh)):
        assert live.offset == ref.offset == len(committed)
        live_keys, ref_keys = _context_keys(live), _context_keys(ref)
        if d.config.layer_types[layer] == "sliding_attention":
            assert live_keys.shape[2] == WINDOW - 1
        else:
            assert live_keys.shape[2] == len(committed)
        np.testing.assert_allclose(live_keys, ref_keys, atol=1e-4)
    # The next full-width block's law is the one the reference ring gives.
    args = ([lane.anchor], lane.tail)
    got = DFlash2DraftModel.draft_distributions(
        d, *args, d.batch_caches([lane.draft_cache]), BLOCK - 1, [RequestRNG(1)], [0.8]
    )
    want = DFlash2DraftModel.draft_distributions(
        d, *args, d.batch_caches([fresh]), BLOCK - 1, [RequestRNG(1)], [0.8]
    )
    assert got[0] == want[0]
    np.testing.assert_allclose(np.array(got[1][0]), np.array(want[1][0]), atol=1e-5)
    b.close()
