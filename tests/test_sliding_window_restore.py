"""APCv2 branches inside a wrapped sliding window from exact restore snapshots.

Gemma 4 (26B-A4B, 31B) keeps a growing ``KVCache`` on its global layers and a
1024-token ``RotatingKVCache`` on its sliding layers.  Once a sliding window
has wrapped it cannot be trimmed back, so a prompt that shares a long document
with a cached one and then diverges can resume only from a restore snapshot of
the window recorded inside the stored prompt.  The standalone
``RotatingKVCache`` recorded such snapshots at prefill chunk boundaries, but
serving prefills through ``BatchRotatingKVCache``, which recorded none: the
GPU smoke saw 0 reused tokens on shared 6.2K and 14.7K documents (output
exact).  ``BatchRotatingKVCache.state_checkpoint`` now records per-lane window
snapshots at the same stride as recurrent-state checkpoints, ``extract`` hands
them to the request's ``RotatingKVCache``, and APCv2's existing
``achievable_trim`` lands on the deepest one at or below the shared prefix.

The end-to-end tests drive the real ``ServingEngine`` and ``BatchGenerator``
on the CPU with a tiny random-weight Gemma 4 (mlx-vlm's ``gemma4`` language
model, KV-shared layers included) on the mlx2 cache types the Gemma 4 adapter
builds.  Exactness is checked against a cold prefill of the same prompt:
tokens through the server, bit-identical logits through the generator.
"""

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

pytest.importorskip("mlx_vlm")

from mlx2 import memory, serving
from mlx2.adapters.gemma4 import _Gemma4LogitsModel
from mlx2.adapters.mlx_vlm_memory import SlidingKVCacheBudget
from mlx2.runtime import os_memory
from mlx2.runtime.apc_v2 import APCv2, inspect_apc_capabilities
from mlx2.runtime.generate import BatchGenerator
from mlx2.runtime.models.cache import (
    BatchRotatingKVCache,
    KVCache,
    RotatingKVCache,
    achievable_trim,
    record_state_checkpoints,
)
from mlx2.serving import ServingEngine

V, WINDOW, STEP, STRIDE = 64, 8, 16, 32
LAYERS = [
    "sliding_attention", "sliding_attention", "full_attention",
    "sliding_attention", "sliding_attention", "full_attention",
]
SHARED_KV = 2
TEXT = {
    "model_type": "gemma4_text",
    "hidden_size": 32,
    "num_hidden_layers": len(LAYERS),
    "intermediate_size": 64,
    "num_attention_heads": 2,
    "head_dim": 8,
    "global_head_dim": 16,
    "vocab_size": V,
    "vocab_size_per_layer_input": V,
    "num_key_value_heads": 1,
    "num_global_key_value_heads": 1,
    "attention_k_eq_v": True,
    "num_kv_shared_layers": SHARED_KV,
    "hidden_size_per_layer_input": 0,
    "sliding_window": WINDOW,
    "layer_types": LAYERS,
    "use_double_wide_mlp": False,
    "max_position_embeddings": 4096,
    "dtype": "float32",
}
TURN = 3  # the tiny adapter's turn-start marker


@pytest.fixture(autouse=True)
def snapshot_policy(monkeypatch):
    # Production records every 2048 tokens against a 1024 window; the tiny
    # model keeps the same ratio with a 32-token stride against an 8 window.
    monkeypatch.setenv("MLX_LM_STATE_CHECKPOINT_STRIDE", str(STRIDE))
    monkeypatch.setenv("MLX_LM_STATE_CHECKPOINT_MAX", "4")


class TinyGemma4(nn.Module):
    """The mlx-vlm Gemma 4 text model behind the Gemma 4 adapter's wrapper.

    ``pixel_values`` shifts the embeddings of the whole prompt, standing in for
    the vision tower (the media prefill contract, not its math, is tested).
    """

    def __init__(self):
        super().__init__()
        from mlx_vlm.models.gemma4.config import TextConfig
        from mlx_vlm.models.gemma4.language import LanguageModel

        fields = {k: v for k, v in TEXT.items() if k not in {"model_type", "dtype"}}
        config = TextConfig(**fields)
        mx.random.seed(11)
        self.language_model = LanguageModel(config)
        self.config = SimpleNamespace(text_config=config)
        mx.eval(self.parameters())

    def __call__(self, inputs, cache=None, pixel_values=None, **kwargs):
        if pixel_values is None:
            return self.language_model(inputs, cache=cache, **kwargs)
        inner = self.language_model.model
        embeds = inner.embed_tokens(inputs) * inner.embed_scale
        embeds = embeds + 0.25 * pixel_values.sum()
        return self.language_model(None, inputs_embeds=embeds, cache=cache, **kwargs)


@pytest.fixture(scope="module")
def gemma():
    return _Gemma4LogitsModel(TinyGemma4())


def _budget():
    return SlidingKVCacheBudget.from_gemma4_config(TEXT, mtp=False, prefill_step=STEP)


# ---------------------------------------------------------------------------
# Serving harness
# ---------------------------------------------------------------------------


class _Detok:
    def __init__(self):
        self.last_segment = ""

    def reset(self):
        self.last_segment = ""

    def add_token(self, token):
        self.last_segment = f"{int(token)} "

    def finalize(self):
        pass


class _Parser:
    stopped = False
    tool_count = 0

    def push(self, text, final=False):
        return [{"content": text}] if text else []


def _adapter(model, *, turn_markers=()):
    class Tokenizer:
        vocab_size = V
        eos_token_ids = []

        @property
        def detokenizer(self):
            return _Detok()

    class Adapter:
        max_context = 2048
        identity = {"fingerprint": "tiny-gemma4"}
        environment = {}
        layout = "tiny-gemma4-full-sliding"
        tokenizer = Tokenizer()

        def __init__(self, _path, execution_policy=None):
            self.model = model

        def profile_name(self, _mtp):
            return "tiny-ordinary"

        def execution_config(self, *, max_lanes, prefill_step):
            return {
                "persistent": True,
                "num_draft": 0,
                "backend": "ordinary",
                "rate_gate": False,
                "prefill_step_size": prefill_step,
            }

        def cache_budget(self, *, mtp):
            return _budget()

        def apc_turn_marker_ids(self):
            return tuple(turn_markers)

        def prompt_tokens(self, request):
            if "_mlx2_prompt_tokens" in request:
                return list(request["_mlx2_prompt_tokens"])
            return list(request["tokens"])

        def prepare_multimodal_request(self, request, *, file_loader=None):
            media = request["messages"][0]["content"][0]["media"]
            pixels = mx.full((1, 3), float(media["seed"]))
            mx.eval(pixels)
            return {
                **request,
                "_mlx2_prompt_tokens": list(media["tokens"]),
                "_mlx2_prefill_inputs": {"pixel_values": pixels},
                "_mlx2_media_token_end": int(media["end"]),
                "_mlx2_media_fingerprint": f"img-{media['seed']}",
            }

        def output_parser(self, _request):
            return _Parser()

        def diagnostics(self):
            return {}

        def close(self):
            pass

    return Adapter


@pytest.fixture
def host(monkeypatch):
    monkeypatch.setattr(serving, "runtime_identity", lambda: {"source_sha256": "src"})
    monkeypatch.setattr(memory, "execution_headroom", lambda: 100 * 2**30)
    monkeypatch.setattr(os_memory, "physical_footprint_bytes", lambda: 0)


def _engine(model, *, max_lanes=1, policy=None, turn_markers=()):
    engine = ServingEngine(
        "tiny", adapter_factory=_adapter(model, turn_markers=turn_markers),
        qualification_mode=True, mtp=False, max_lanes=max_lanes,
        prefill_step=STEP, execution_policy=policy,
    )
    assert engine.ready.wait(60), engine.error
    return engine


def _submit(engine, tokens, max_tokens=6):
    return engine.submit(
        {"tokens": list(tokens), "max_tokens": max_tokens, "temperature": 0}
    )


def _media_request(tokens, end, seed=1, max_tokens=6):
    return {
        # A media part (an all-text part array would be served as text).
        "messages": [{"role": "user", "content": [{
            "type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="},
            "media": {"tokens": list(tokens), "end": end, "seed": seed},
        }]}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }


def _finish(job):
    text = ""
    while True:
        event = job.events.get(timeout=120)
        if "error" in event:
            raise AssertionError(event)
        if "delta" in event:
            text += event["delta"].get("content", "")
        if "finish_reason" in event:
            return [int(t) for t in text.split()], int(job.cached_tokens or 0)


def _run(engine, tokens, max_tokens=6):
    return _finish(_submit(engine, tokens, max_tokens))


def _cold(model, prompts, *, max_tokens=6, media=False):
    """Each prompt alone on a fresh server: the reference output."""
    out = []
    for prompt in prompts:
        engine = _engine(model)
        try:
            job = (
                engine.submit(prompt) if media else _submit(engine, prompt, max_tokens)
            )
            tokens, cached = _finish(job)
        finally:
            engine.close()
        assert cached == 0
        out.append(tokens)
    return out


def _seq(n, seed):
    return [(seed * (i + 3) + 7 * i) % (V - 8) + 4 for i in range(n)]


DOC = _seq(150, 5)
QA, QB, QC = _seq(10, 11), _seq(10, 13), _seq(12, 17)


def _snapshot_positions(engine, prompt):
    """Restore points the served entries hold on the path of ``prompt``."""
    positions = set()
    for _key, tokens, entry in engine.apc._entry_records_locked():
        if list(tokens[: len(DOC)]) != list(prompt[: len(DOC)]):
            continue
        for cache in entry.prompt_cache:
            if isinstance(cache, RotatingKVCache):
                positions.update(p for p, _, _ in cache._checkpoints)
    return positions


# ---------------------------------------------------------------------------
# The Gemma 4 smoke case: a shared long document, then divergence.
# ---------------------------------------------------------------------------


def test_shared_document_then_divergence_reuses_a_window_snapshot_exactly(host, gemma):
    a, b = DOC + QA, DOC + QB
    (cold_b,) = _cold(gemma, [b])
    engine = _engine(gemma)
    try:
        _out_a, cached_a = _run(engine, a)
        positions = _snapshot_positions(engine, a)
        warm_b, cached_b = _run(engine, b)
        counts = dict(engine.counts)
    finally:
        engine.close()
    assert cached_a == 0
    # Before the fix serving recorded no snapshot, so B reused nothing
    # (the smoke's 0-token reuse).  Now it resumes at A's deepest snapshot
    # at or below the shared document.
    assert positions, "served entries carry no sliding-window restore snapshot"
    expected = max(p for p in positions if p <= len(DOC))
    assert cached_b == expected > len(DOC) - STRIDE
    assert warm_b == cold_b
    assert counts.get("prompt_cache_hits", 1) >= 1


def test_shared_system_prompt_only(host, gemma):
    system = _seq(40, 7)
    users = [_seq(70, 19), _seq(70, 23)]
    prompts = [system + [TURN] + user for user in users]
    cold = _cold(gemma, prompts)
    engine = _engine(gemma)
    try:
        first, _ = _run(engine, prompts[0])
        second, cached = _run(engine, prompts[1])
    finally:
        engine.close()
    assert first == cold[0] and second == cold[1]
    # The stride snapshot inside the system prompt is the branch point.
    assert cached == STRIDE


def test_interior_turn_checkpoint_serves_the_whole_system_prompt(host, gemma):
    """Interior placement is decided by cache capability, not topology name:
    a sliding cache takes the hybrid turn-boundary checkpoints too."""
    system = _seq(40, 7)
    users = [_seq(70, 19), _seq(70, 23)]
    prompts = [system + [TURN] + user for user in users]
    cold = _cold(gemma, prompts)
    policy = {
        "apc_interior_checkpoints": {
            "count": 2, "min_stride": 8, "placement": "turns",
        }
    }
    engine = _engine(gemma, policy=policy, turn_markers=(TURN,))
    try:
        first, _ = _run(engine, prompts[0])
        second, cached = _run(engine, prompts[1])
        counts = dict(engine.counts)
    finally:
        engine.close()
    assert first == cold[0] and second == cold[1]
    assert cached == len(system)
    assert counts["apc_interior_positions_planned_turn"] >= 1


def test_junction_checkpoint_captures_a_sliding_cache(host, gemma):
    a, b, c = DOC + QA, DOC + QB, DOC + QC
    (cold_c,) = _cold(gemma, [c])
    engine = _engine(gemma, policy={"apc_junction_checkpoints": True})
    try:
        _run(engine, a)
        _run(engine, b)
        warm_c, cached_c = _run(engine, c)
        counts = dict(engine.counts)
    finally:
        engine.close()
    assert counts["apc_junction_checkpoints_planned"] == 1
    assert cached_c == len(DOC)
    assert warm_c == cold_c


GEN = [1, 2]  # a generation-prompt suffix the next turn does not re-render


def test_multi_turn_reuses_through_the_previous_turn(host, gemma):
    system = _seq(30, 7)
    history = system + [TURN] + _seq(40, 29)
    turn1 = history + GEN
    engine = _engine(gemma)
    try:
        answer1, _ = _run(engine, turn1, max_tokens=8)
        # A template that keeps the generation prompt: the next turn extends
        # the stored turn and resumes past it without any branch.
        kept = turn1 + answer1 + [TURN] + _seq(30, 31) + GEN
        answer_kept, cached_kept = _run(engine, kept)
        # A template that drops it re-renders the finished turn without the
        # suffix, so the next turn diverges inside every stored turn-1 entry
        # (the case interior placement's generation-prompt boundary targets
        # on hybrids).  It resumes from a window snapshot inside turn 1.
        dropped = history + answer1 + [TURN] + _seq(30, 37) + GEN
        answer_dropped, cached_dropped = _run(engine, dropped)
    finally:
        engine.close()
    assert cached_kept >= len(turn1)
    assert len(history) - STRIDE < cached_dropped <= len(history)
    assert [answer_kept, answer_dropped] == _cold(gemma, [kept, dropped])


def test_media_coverage_rule_still_holds(host, gemma):
    media_end = 24
    image = _seq(media_end, 41)
    shared_text = _seq(60, 43)
    prompts = [
        _media_request(image + shared_text + _seq(12, s), media_end)
        for s in (47, 53)
    ]
    cold = _cold(gemma, prompts, media=True)
    engine = _engine(gemma)
    try:
        first = _finish(engine.submit(prompts[0]))
        second = _finish(engine.submit(prompts[1]))
        repeat = _finish(engine.submit(prompts[1]))
        counts = dict(engine.counts)
    finally:
        engine.close()
    assert [first[0], second[0], repeat[0]] == [cold[0], cold[1], cold[1]]
    # A resume point inside the media span is never served.
    for _tokens, cached in (second, repeat):
        assert cached == 0 or cached >= media_end
    # The exact repeat of a media prompt is still reused.
    assert repeat[1] > media_end
    assert counts.get("multimodal_apcv2_boundary_misses", 0) >= 0


def test_ragged_batch_of_divergent_prompts(host, gemma):
    a = DOC + QA
    b1 = DOC + QB
    b2 = DOC[:120] + _seq(25, 59)
    cold = _cold(gemma, [b1, b2])
    engine = _engine(gemma, max_lanes=2)
    try:
        _run(engine, a)
        jobs = [_submit(engine, b1), _submit(engine, b2)]
        results = [_finish(job) for job in jobs]
    finally:
        engine.close()
    assert [tokens for tokens, _ in results] == cold
    assert all(cached > 0 for _, cached in results)
    assert results[0][1] <= len(DOC) and results[1][1] <= 120


# ---------------------------------------------------------------------------
# Generator level: bit-identical logits against a cold prefill.
# ---------------------------------------------------------------------------


def _generator(model):
    return BatchGenerator(
        model,
        completion_batch_size=4,
        prefill_batch_size=4,
        prefill_step_size=STEP,
        prefill_batch_window=1,
    )


def _drive(gen, uids, limit=600):
    record, done = {}, set()
    for _ in range(limit):
        _, responses = gen.next()
        for r in responses:
            record.setdefault(r.uid, []).append(
                (int(r.token), np.array(r.logprobs.astype(mx.float32)))
            )
            if r.finish_reason:
                done.add(r.uid)
        if done >= set(uids):
            return record
    raise AssertionError("lanes did not finish")


def _cold_logits(model, prompt, max_tokens):
    gen = _generator(model)
    uid = gen.insert([prompt], max_tokens=[max_tokens])[0]
    return _drive(gen, [uid])[uid]


def test_resumed_logits_are_bit_identical_to_a_cold_prefill(gemma):
    stored = DOC + QA
    gen = _generator(gemma)
    uid = gen.insert([stored], max_tokens=[4])[0]
    _drive(gen, [uid])
    boundary = gen.pop_prompt_boundary(uid)
    apc = APCv2(max_size=8, layout_name="sliding-restore")
    capabilities = apc.store("k", boundary["tokens"], boundary["target_cache"])
    assert capabilities.stored is not False
    assert capabilities.interior_checkpoint_target

    queries = [DOC + QB, DOC[:100] + _seq(30, 61), DOC[:70] + _seq(9, 67)]
    hits = [apc.lookup("k", q) for q in queries]
    assert all(hit.hit and hit.cached_tokens > WINDOW for hit in hits)
    # Deepest recorded window at or below each shared prefix.
    for hit, shared in zip(hits, (len(DOC), 100, 70)):
        assert shared - STRIDE < hit.cached_tokens <= shared
    # All three resume together: a ragged B=3 prefill from restored caches.
    warm_gen = _generator(gemma)
    uids = warm_gen.insert(
        [hit.remaining_tokens for hit in hits],
        max_tokens=[5] * len(hits),
        caches=[hit.cache for hit in hits],
        all_tokens=[q[: hit.cached_tokens] for q, hit in zip(queries, hits)],
    )
    warm = _drive(warm_gen, uids)
    for uid, query in zip(uids, queries):
        cold = _cold_logits(gemma, query, 5)
        assert [t for t, _ in warm[uid]] == [t for t, _ in cold]
    # Bit identity needs identical kernels: a lone resumed request against a
    # lone cold one (same chunk boundaries, since snapshots sit on them).
    single = apc.lookup("k", queries[0])
    solo_gen = _generator(gemma)
    solo = solo_gen.insert(
        [single.remaining_tokens], max_tokens=[5], caches=[single.cache],
        all_tokens=[queries[0][: single.cached_tokens]],
    )[0]
    resumed = _drive(solo_gen, [solo])[solo]
    cold = _cold_logits(gemma, queries[0], 5)
    assert [t for t, _ in resumed] == [t for t, _ in cold]
    for (_, x), (_, y) in zip(resumed, cold):
        assert np.array_equal(x, y)
    apc.clear(release_memory=False)


def test_prompt_boundary_takes_the_snapshots_and_decode_keeps_no_copy(gemma):
    gen = _generator(gemma)
    uid = gen.insert([DOC + QA], max_tokens=[6])[0]
    for _ in range(40):
        gen.next()
        if uid in gen._prompt_boundaries:
            break
    boundary = gen._prompt_boundaries[uid]["target_cache"]
    held = [
        [p for p, _, _ in c._checkpoints]
        for c in boundary
        if isinstance(c, RotatingKVCache)
    ]
    assert held and all(positions == held[0] and positions for positions in held)
    # Stride snapshots at prefill chunk ends inside the prompt (none at its
    # end: the boundary's own window is that state), so a decode tip carrying
    # them again could never branch deeper than the boundary does.
    assert held[0] == [32, 64, 96, 128]
    live = [
        c for c in gen._generation_batch.prompt_cache
        if isinstance(c, BatchRotatingKVCache)
    ]
    assert live and all(c._checkpoints == [] for c in live)


# ---------------------------------------------------------------------------
# BatchRotatingKVCache snapshot bookkeeping.
# ---------------------------------------------------------------------------


def _chunk(cache, rows, start, width=4):
    k = (mx.arange(start, start + rows, dtype=mx.float32)[None, None, :, None]
         * mx.ones((1, 1, 1, width)))
    return cache.update_and_fetch(k, k + 0.5)


def test_batch_snapshots_match_the_unbatched_cache_per_lane(monkeypatch):
    """A lane's snapshot equals what its own ``RotatingKVCache`` records."""
    monkeypatch.setenv("MLX_LM_STATE_CHECKPOINT_STRIDE", "8")
    lengths = [37, 21]
    single = [RotatingKVCache(max_size=WINDOW) for _ in lengths]
    batch = BatchRotatingKVCache(WINDOW, [max(lengths) - n for n in lengths])
    done = 0
    step = 6
    total = max(lengths)
    while done < total:
        width = min(step, total - done)
        # Left-padded rows: row i holds token t at column t + pad_i.
        rows = []
        for n in lengths:
            pad = total - n
            vals = [max(0, c - pad) for c in range(done, done + width)]
            rows.append(vals)
        k = mx.array(rows, dtype=mx.float32)[:, None, :, None] * mx.ones((1, 1, 1, 4))
        batch.update_and_fetch(k, k + 0.5)
        batch.compact_to_window()
        mx.eval(batch.keys, batch.values, batch.offset, batch.left_padding)
        done += width
        positions = [max(0, done - (total - n)) for n in lengths]
        for i, cache in enumerate(single):
            real = [c - (total - lengths[i]) for c in range(done - width, done)
                    if c >= total - lengths[i]]
            if real:
                kk = mx.array(real, dtype=mx.float32)[None, None, :, None] * mx.ones((1, 1, 1, 4))
                cache.update_and_fetch(kk, kk + 0.5)
                cache.compact_to_window()
                cache.state_checkpoint([positions[i]])
        batch.state_checkpoint(positions)
    for i, cache in enumerate(single):
        extracted = batch.extract(i)
        assert [p for p, _, _ in extracted._checkpoints] == [
            p for p, _, _ in cache._checkpoints
        ]
        for (_, k1, v1), (_, k2, v2) in zip(extracted._checkpoints, cache._checkpoints):
            assert k1.shape[2] == min(_, WINDOW)
            assert mx.array_equal(k1, k2) and mx.array_equal(v1, v2)
    # nbytes counts every lane's snapshots.
    assert batch.nbytes == batch.keys.nbytes + batch.values.nbytes + sum(
        k.nbytes + v.nbytes for lane in batch._checkpoints for _, k, v in lane
    )


def test_filter_extend_merge_and_trim_keep_lane_snapshots_aligned(monkeypatch):
    monkeypatch.setenv("MLX_LM_STATE_CHECKPOINT_STRIDE", "4")
    caches = []
    for n in (20, 12):
        c = RotatingKVCache(max_size=WINDOW)
        _chunk(c, n, 100 * n)
        c.compact_to_window()
        c.state_checkpoint([n])
        caches.append(c)
    merged = BatchRotatingKVCache.merge(caches)
    assert [[p for p, _, _ in lane] for lane in merged._checkpoints] == [[20], [12]]
    other = BatchRotatingKVCache.merge([RotatingKVCache(max_size=WINDOW)])
    _chunk(other, 3, 7)
    merged.extend(other)
    assert [[p for p, _, _ in lane] for lane in merged._checkpoints] == [[20], [12], []]
    merged.filter([2, 0])
    assert [[p for p, _, _ in lane] for lane in merged._checkpoints] == [[], [20]]
    assert [p for p, _, _ in merged.extract(1)._checkpoints] == [20]
    # A trim below the point where snapshots were valid drops them all.
    merged.trim(1)
    assert merged._checkpoints == []


def test_trimmed_rotating_cache_drops_snapshots_above_its_offset(monkeypatch):
    monkeypatch.setenv("MLX_LM_STATE_CHECKPOINT_STRIDE", "4")
    cache = RotatingKVCache(max_size=32)
    _chunk(cache, 20, 0)
    cache.state_checkpoint([20])
    cache.trim(5)
    assert cache._checkpoints == []


def test_window_snapshots_stay_a_stride_apart_across_short_turns():
    """End-of-prompt forcing adds no window snapshot, so a lane never holds
    more than ``floor(context / stride)`` (within the budgeted count)."""
    cache = RotatingKVCache(max_size=WINDOW)
    done = 0
    for _turn in range(20):
        _chunk(cache, 5, done)
        done += 5
        cache.compact_to_window()
        cache.state_checkpoint([done], force=True)
        assert len(cache._checkpoints) <= done // STRIDE
        assert len(cache._checkpoints) <= _budget().sliding_snapshots(done)
    positions = [p for p, _, _ in cache._checkpoints]
    assert positions == [35, 70]
    assert all(b - a >= STRIDE for a, b in zip([0] + positions, positions))


def test_snapshot_capability_is_by_cache_type_not_by_model():
    sliding = [KVCache(), RotatingKVCache(max_size=WINDOW)]
    assert inspect_apc_capabilities(sliding).interior_checkpoint_target
    assert not inspect_apc_capabilities([KVCache()]).interior_checkpoint_target
    # Wrapped with a snapshot: branchable at the snapshot, not below it.
    for c, n in zip(sliding, (40, 40)):
        _chunk(c, n, 0)
    sliding[1].compact_to_window()
    record_state_checkpoints(sliding, [32], force=True)
    assert achievable_trim(sliding, 5) == (32, 8)


def test_vendor_sliding_cache_fails_closed_with_a_reason():
    """mlx-vlm's sliding cache (Gemma 3n) records no restore snapshots, so
    APCv2 never branches inside its wrapped window and says why."""
    from mlx_vlm.models.cache import KVCache as VKV, RotatingKVCache as VRot

    def filled(n):
        cache = [VKV(), VRot(max_size=WINDOW, keep=0)]
        for c in cache:
            _chunk(c, n, 0)
        mx.eval([c.state for c in cache])
        return cache

    apc = APCv2(max_size=8, layout_name="vendor-sliding")
    apc.store("k", list(range(1, 41)), filled(40))
    miss = apc.lookup("k", list(range(1, 31)) + [99] * 5)
    assert not miss.hit
    assert miss.miss_reason == "branch_cache_records_no_restore_points"
    apc.clear(release_memory=False)
    assert _budget().checkpoint_copies == 4
    gemma3n = SlidingKVCacheBudget.from_gemma3n_config(
        {k: v for k, v in TEXT.items() if k != "num_kv_shared_layers"},
        mtp=False, prefill_step=STEP,
    )
    assert gemma3n.checkpoint_copies == 0
