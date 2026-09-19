"""Impact checks for three peer hybrid-speculation bugs (2026-09-18 survey).

- vllm#52244: under MTP, a replayed hybrid prompt never reaches the deepest
  reusable prefix because the recurrent state is published at a position a
  replay cannot land on.
- sglang#35694: a speculative verify commits a whole accepted chunk, so the
  cache key keeps tokens past the point where the request actually stopped.
- sglang#40001: accepted recurrent state is not committed (or is committed
  to the wrong row) after verify.  mlx2 has no pipeline parallelism; the
  transferable claim is "after a verify, every row's committed recurrent,
  conv, PLE n-gram and attention state equals a fresh teacher-forced prefill
  of exactly the tokens that row delivered".

Each test drives the real tiny Qwen4 hybrid (GDN + QSA + PLE) on CPU.
"""

from itertools import product
from unittest.mock import patch

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.apc_v2 import APCKey, APCv2, MTPAPCSidecar
from mlx2.runtime.generate import (
    BatchGenerator,
    MTPGenerationBatch,
    interior_checkpoint_positions,
)
from mlx2.runtime.hybrid_speculative import (
    attach_self_mtp_lanes,
    commit_batched_self_mtp,
    detach_self_mtp_lanes,
    prepare_self_mtp_lane,
    propose_batched_self_mtp,
)
from mlx2.runtime.models.qwen4_exp import QSAKVCache
from mlx2.runtime.sample_utils import LaneRNG

from test_batched_mtp import _tiny_qwen4_model, _tree_arrays


@pytest.fixture(scope="module")
def model():
    mx.random.seed(41)
    return _tiny_qwen4_model()


def _prepare(model, uid, prompt):
    return prepare_self_mtp_lane(
        mx.array(prompt, mx.uint32),
        model,
        uid=uid,
        max_tokens=16,
        prompt_cache=None,
        mtp_state=None,
        lane_rng=LaneRNG(900 + uid),
        num_draft=2,
        sampling_temp=0.8,
        sampling_top_p=1.0,
        sampling_top_k=8,
        sampling_min_p=0.0,
        accept_rule="residual",
        logits_processors=[],
        prefill_step_size=4,
        share_qsa_indices=False,
    )


def _assert_state_equal(actual, expected):
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        assert type(got) is type(want)
        assert getattr(got, "offset", None) == getattr(want, "offset", None)
        got_arrays = list(_tree_arrays(got.state))
        want_arrays = list(_tree_arrays(want.state))
        assert len(got_arrays) == len(want_arrays)
        for left, right in zip(got_arrays, want_arrays):
            if left.shape != right.shape:
                # A fresh KV cache keeps its step-sized allocation; an
                # extracted row is trimmed.  Compare the valid positions.
                n = int(got.offset)
                left, right = left[..., :n, :], right[..., :n, :]
                assert left.shape == right.shape
            np.testing.assert_allclose(
                np.asarray(left.astype(mx.float32)),
                np.asarray(right.astype(mx.float32)),
                rtol=2e-4,
                atol=2e-5,
            )
        if isinstance(got, QSAKVCache):
            np.testing.assert_allclose(
                np.asarray(got.index_keys.astype(mx.float32)),
                np.asarray(want.index_keys.astype(mx.float32)),
                rtol=2e-4,
                atol=2e-5,
            )


def test_terminal_mid_proposal_commits_exactly_the_delivered_prefix(model):
    """sglang#35694 + the sglang#40001 bug class, two rows, every pairing.

    Both rows accept both drafts (three outputs each); each row then stops
    at output 1, 2 or 3.  The committed key must be prompt + delivered
    tokens minus the terminal token (which is never fed), and the committed
    target *and* draft state must equal a fresh prefill of that key.
    """
    prompts = ([1, 2, 3, 4, 5], [7, 8, 9, 10, 11, 12])
    for counts in product((1, 2, 3), repeat=2):
        prepared = [_prepare(model, uid, p) for uid, p in enumerate(prompts)]
        firsts = [first.token for _lane, first in prepared]
        batch = attach_self_mtp_lanes(model, None, [lane for lane, _ in prepared])

        def accept_all(logprobs, _draft_lps, drafts, _temperature, *, rng=None):
            return len(drafts), int(mx.argmax(logprobs[len(drafts)]).item())

        with patch(
            "mlx2.runtime.hybrid_speculative._batched_residual_verify",
            side_effect=accept_all,
        ):
            proposal = propose_batched_self_mtp(model, batch)
        assert all(len(row) == 3 for row in proposal.outputs)
        assert tuple(proposal.accepted_lengths) == (2, 2)
        commit_batched_self_mtp(
            batch, proposal, emitted_counts=list(counts), terminal=[True, True]
        )
        batch, detached = detach_self_mtp_lanes(model, batch, [0, 1])
        for row, (prompt, count) in enumerate(zip(prompts, counts)):
            delivered = [firsts[row]] + [
                output.token for output in proposal.outputs[row][:count]
            ]
            expected_key = list(prompt) + delivered[:-1]
            item = detached[row]
            # The overshoot (outputs past ``count``) is out of the key...
            assert MTPGenerationBatch._prefix_tokens(item.lane) == expected_key
            # ...and out of the state: exact against a fresh teacher-forced
            # prefill of the key, for GDN/conv/PLE/QSA target and the draft.
            oracle, _first = _prepare(model, 50 + row, expected_key)
            _assert_state_equal(item.caches.target, oracle.caches.target)
            _assert_state_equal(item.caches.draft, oracle.caches.draft)
            np.testing.assert_allclose(
                np.asarray(item.lane.seed_h.astype(mx.float32)),
                np.asarray(oracle.lane.seed_h.astype(mx.float32)),
                rtol=2e-4,
                atol=2e-5,
            )


def _run_to_finish(gen, uid):
    boundary = finish = None
    tokens = []
    for _ in range(200):
        prompts, responses = gen.next()
        for response in prompts:
            if response.uid == uid and response.end_of_prompt:
                boundary = gen.pop_prompt_boundary(uid)
        for response in responses:
            if response.uid != uid:
                continue
            tokens.append(response.token)
            if response.finish_reason:
                finish = response
        if finish is not None:
            return boundary, finish, tokens
    raise AssertionError("request did not finish")


def _run_with_interiors(gen, uid):
    boundary, finish, tokens = _run_to_finish(gen, uid)
    return boundary, gen.pop_interior_checkpoints(uid), finish, tokens


@pytest.mark.parametrize("prompt_len", [3, 4, 5, 8, 9])
def test_mtp_replay_reaches_the_deepest_reusable_prefix(model, prompt_len):
    """vllm#52244: hit depth under MTP, including chunk-aligned prompt ends.

    With ``prefill_step_size=4`` the lengths 4 and 8 end exactly on a chunk
    boundary, which is where vLLM's page/hash-unit alignment collapsed the
    hit to zero.  mlx2 publishes the prompt boundary at P-1 (the deepest
    position a replay can land on, since the last token must be recomputed
    for logits) and the finish checkpoint at the committed key, so:

    - an exact replay hits P-1 tokens;
    - the next turn hits every committed token of the previous turn;
    - warm decoding reproduces cold decoding token for token.
    """
    prompt = [(3 * i + 1) % 60 + 2 for i in range(prompt_len)]
    config = {"sampling_temp": 0.0, "num_draft": 2}
    apc = APCv2(max_size=8, layout_name="peer-pr-impact-v1")
    key = APCKey("tiny-qwen4", revision="v1")

    def generator():
        return BatchGenerator(
            model, self_mtp={"num_draft": 2, "persistent": True}, prefill_step_size=4
        )

    gen = generator()
    try:
        uid = gen.insert(
            [prompt], max_tokens=[5], mtp_states=[None], lane_rngs=[LaneRNG(3)],
            self_mtp_configs=[dict(config)],
        )[0]
        boundary, finish, cold_tokens = _run_to_finish(gen, uid)
    finally:
        gen.close()

    assert boundary is not None and boundary["committed_only"]
    assert boundary["covered_tokens"] == prompt_len - 1
    apc.store(
        key, boundary["tokens"], boundary["target_cache"],
        sidecar=MTPAPCSidecar(boundary["mtp_state"], boundary["covered_tokens"]),
        retention_role="committed_prompt_boundary",
    )
    committed = list(finish.all_tokens)
    assert committed == prompt + cold_tokens[:-1]
    apc.store(
        key, committed, finish.prompt_cache,
        sidecar=MTPAPCSidecar(finish.mtp_state, len(committed)),
    )

    replay = apc.lookup(key, prompt)
    assert replay.hit and replay.hit_kind == "mtp_sidecar"
    assert replay.cached_tokens == prompt_len - 1

    gen = generator()
    try:
        uid = gen.insert(
            [replay.remaining_tokens], max_tokens=[5],
            caches=[replay.cache], all_tokens=[prompt[: replay.cached_tokens]],
            mtp_states=[replay.sidecar.state], lane_rngs=[LaneRNG(3)],
            self_mtp_configs=[dict(config)],
        )[0]
        _b, _f, warm_tokens = _run_to_finish(gen, uid)
    finally:
        gen.close()
    assert warm_tokens == cold_tokens

    next_turn = committed + [cold_tokens[-1], 11, 12]
    follow = apc.lookup(key, next_turn)
    assert follow.hit and follow.hit_kind == "mtp_sidecar"
    assert follow.cached_tokens == len(committed)
    follow.cache.close()
    apc.clear(release_memory=False)


@pytest.mark.parametrize("disk_action", ["park", "suspend"])
def test_session_disk_resume_prefetch_keeps_tiny_hybrid_decode_exact(
    model, tmp_path, disk_action
):
    """A real hybrid Qwen4 checkpoint leaves memory and returns before decode."""
    prompt = [4, 7, 10, 13, 16, 19, 22, 25, 28]
    config = {"sampling_temp": 0.0, "num_draft": 2}
    key = APCKey("tiny-qwen4", revision="session-v1")
    tag = ("tenant-a", "tool-run")

    def generator():
        return BatchGenerator(
            model,
            self_mtp={"num_draft": 2, "persistent": True},
            prefill_step_size=4,
        )

    cold = generator()
    try:
        uid = cold.insert(
            [prompt], max_tokens=[5], mtp_states=[None],
            lane_rngs=[LaneRNG(123)], self_mtp_configs=[dict(config)],
        )[0]
        boundary, _finish, cold_tokens = _run_to_finish(cold, uid)
    finally:
        cold.close()

    apc = APCv2(
        max_size=8,
        max_bytes=1 << 30,
        layout_name="peer-pr-session-v1",
        idle_disk_seconds=180,
        idle_disk_dir=str(tmp_path),
    )
    apc.store(
        key,
        boundary["tokens"],
        boundary["target_cache"],
        sidecar=MTPAPCSidecar(
            boundary["mtp_state"], boundary["covered_tokens"]
        ),
        retention_role="committed_prompt_boundary",
        session_tag=tag,
    )
    assert apc.apc_stats["idle_disk"]["resident_bytes"] > 0
    if disk_action == "park":
        assert apc.park_session(*tag, ttl_seconds=60)["state"] == "disk"
    else:
        suspended = apc.suspend_resident()
        assert suspended["entries"] == 1
        assert suspended["resident_entries_after"] == 0
    assert apc.apc_stats["idle_disk"]["resident_bytes"] == 0
    apc.resume_session(*tag, ttl_seconds=30)
    # Serving executes queued restores on its model worker's idle boundary so
    # MLX graph evaluation cannot race decode on a separate prefetch thread.
    apc.service_pending_prefetch()
    deadline = __import__("time").monotonic() + 5
    while apc.session_state(*tag)["state"] != "resident":
        assert __import__("time").monotonic() < deadline
        __import__("time").sleep(0.01)
    replay = apc.lookup(key, prompt, session_tag=tag)
    assert replay.hit and replay.cached_tokens == len(prompt) - 1

    warm = generator()
    try:
        uid = warm.insert(
            [replay.remaining_tokens], max_tokens=[5], caches=[replay.cache],
            all_tokens=[prompt[: replay.cached_tokens]],
            mtp_states=[replay.sidecar.state], lane_rngs=[LaneRNG(123)],
            self_mtp_configs=[dict(config)],
        )[0]
        _boundary, _finish, warm_tokens = _run_to_finish(warm, uid)
    finally:
        warm.close()
    assert warm_tokens == cold_tokens
    assert apc.apc_stats["idle_disk"]["prefetch_hits"] == 1
    apc.close()


def test_budgeted_interior_checkpoint_reuses_edited_hybrid_prompt(model):
    """omlx#3456/Rapid-MLX#3435: edited turns resume exact hybrid state."""
    prompt_a = [(5 * i + 3) % 60 + 2 for i in range(40)]
    prompt_b = prompt_a[:23] + [((token + 17) % 60) + 2 for token in prompt_a[23:]]
    policy = {"count": 3, "min_stride": 4}
    assert interior_checkpoint_positions(len(prompt_a), **policy) == (8, 16, 32)
    assert interior_checkpoint_positions(len(prompt_b), **policy) == (8, 16, 32)
    key = APCKey("tiny-qwen4", revision="interior-v1")
    apc = APCv2(max_size=16, layout_name="peer-pr-interior-v1")

    def generator(checkpoints=policy):
        return BatchGenerator(
            model,
            self_mtp={"num_draft": 2, "persistent": True},
            prefill_step_size=7,
            apc_interior_checkpoints=checkpoints,
        )

    gen = generator()
    try:
        uid = gen.insert(
            [prompt_a], max_tokens=[4], mtp_states=[None],
            lane_rngs=[LaneRNG(71)],
            self_mtp_configs=[{"sampling_temp": 0.0, "num_draft": 2}],
        )[0]
        _boundary, interiors, _finish, _tokens = _run_with_interiors(gen, uid)
    finally:
        gen.close()
    assert [item["covered_tokens"] for item in interiors] == [8, 16, 32]
    for item in interiors:
        assert max(cache.offset for cache in item["mtp_state"][0]) == item[
            "covered_tokens"
        ] - 1
        apc.store(
            key,
            item["tokens"],
            item["target_cache"],
            sidecar=MTPAPCSidecar(
                item["mtp_state"],
                item["covered_tokens"],
                rng_key=item.get("rng_key"),
                rng_draws=item.get("rng_draws", 0),
            ),
            retention_role="interior_checkpoint",
        )

    warm = apc.lookup(key, prompt_b)
    assert warm.hit and warm.cached_tokens == 16
    assert warm.retention_role == "interior_checkpoint"
    assert apc.apc_stats["lifetime"]["interior_hits"] == 1

    gen = generator()
    try:
        uid = gen.insert(
            [warm.remaining_tokens], max_tokens=[5], caches=[warm.cache],
            all_tokens=[prompt_b[: warm.cached_tokens]],
            mtp_states=[warm.sidecar.state], lane_rngs=[LaneRNG(91)],
            self_mtp_configs=[{"sampling_temp": 0.0, "num_draft": 2}],
        )[0]
        _b, warm_interiors, _f, warm_tokens = _run_with_interiors(gen, uid)
    finally:
        gen.close()
    assert [item["covered_tokens"] for item in warm_interiors] == [32]

    cold = generator({"count": 0, "min_stride": 4})
    try:
        uid = cold.insert(
            [prompt_b], max_tokens=[5], mtp_states=[None],
            lane_rngs=[LaneRNG(91)],
            self_mtp_configs=[{"sampling_temp": 0.0, "num_draft": 2}],
        )[0]
        _b, cold_interiors, _f, cold_tokens = _run_with_interiors(cold, uid)
    finally:
        cold.close()
    assert cold_interiors == []
    assert warm_tokens == cold_tokens
    apc.clear(release_memory=False)
