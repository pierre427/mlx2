"""CPU contract tests for the North Cohere EAGLE chain drafter; tiny random weights."""
import json
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from mlx2.runtime.drafters.cohere_eagle import (
    CohereEagleConfig,
    CohereEagleDraftModel,
    EagleWindowKVCache,
    expected_weight_shapes,
)
from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator
from mlx2.runtime.models.cohere2_moe import Model, ModelArgs
from mlx2.runtime.speculative_sampling import RequestRNG

EAGLE_SNAPSHOT = Path(
    "~/.cache/huggingface/hub/models--CohereLabs--North-Mini-Code-1.0-eagle/"
    "snapshots/8c7fcb575f107e9968b61cc93a756e6fc2c86713"
)


def tiny(window=4, seed=3):
    mx.random.seed(seed)
    target = Model(ModelArgs(
        hidden_size=16, head_dim=4, num_hidden_layers=4, intermediate_size=8,
        prefix_dense_intermediate_size=16, num_attention_heads=4, num_key_value_heads=2,
        vocab_size=40, sliding_window=window, max_position_embeddings=256,
        num_experts=4, num_experts_per_tok=2,
    ))
    draft = CohereEagleDraftModel(CohereEagleConfig(
        hidden_size=16, intermediate_size=8, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=4, vocab_size=40, sliding_window=window,
        num_target_layers=4, block_size=5,
    ))
    mx.eval(target.parameters(), draft.parameters())
    return target, draft.bind(target)


def generator(target, draft, **kwargs):
    kwargs.setdefault("num_draft", 3)
    return ExternalDraftBatchGenerator(
        target, draft_model=draft, binding="test", prefill_step_size=3, **kwargs
    )


def drain(batch):
    output, final = {}, {}
    for _ in range(200):
        _, responses = batch.next()
        for response in responses:
            output.setdefault(response.uid, []).append(response.token)
            if response.finish_reason:
                final[response.uid] = response
        if not batch.lanes:
            return output, final
    raise AssertionError("scheduler stalled")


def greedy_reference(target, prompt, count):
    cache = target.make_cache()
    tokens = list(prompt)
    out = []
    for step in range(count):
        logits = target(mx.array([tokens if step == 0 else [tokens[-1]]]), cache=cache)
        token = int(mx.argmax(logits[0, -1]).item())
        tokens.append(token)
        out.append(token)
    return out


def test_final_norm_tap_matches_ordinary_forward_and_logits():
    target, _ = tiny()
    x = mx.array([[1, 2, 3, 4, 5]])
    logits, features = target.forward_with_taps(x, target.make_cache(), [4])
    ordinary = target(x, cache=target.make_cache())
    np.testing.assert_allclose(np.asarray(logits), np.asarray(ordinary), atol=1e-5)
    hidden = target.model(x, target.make_cache())
    np.testing.assert_allclose(np.asarray(features), np.asarray(hidden), atol=1e-6)
    _, mixed = target.forward_with_taps(x, target.make_cache(), [1, 4])
    assert mixed.shape == (1, 5, 32)
    assert target.model.residual_taps.capture is None
    with pytest.raises(ValueError):
        target.forward_with_taps(x, target.make_cache(), [5])


def test_committed_block_equals_stepwise_commits():
    target, draft = tiny(window=3)
    features = target.prefill_body(mx.array([[1, 2, 3, 4, 5, 6]]), target.make_cache(), [4])
    nxt = [2, 3, 4, 5, 6, 7]
    whole = draft.make_cache()
    out_whole = draft.append_context(features, whole, context_tokens=[nxt])
    step = draft.make_cache()
    outs = [
        draft.append_context(features[:, i:i + 1], step, context_tokens=[[nxt[i]]])
        for i in range(6)
    ]
    np.testing.assert_allclose(
        np.asarray(out_whole), np.asarray(mx.concatenate(outs, axis=1)), atol=1e-5
    )
    for a, b in zip(whole, step):
        assert a.offset == b.offset == 6 and a.keys.shape[2] == 3  # window-trimmed
        np.testing.assert_allclose(np.asarray(a.keys), np.asarray(b.keys), atol=1e-5)


def test_chain_laws_are_exact_and_round_local():
    target, draft = tiny()
    features = target.prefill_body(mx.array([[1, 2, 3]]), target.make_cache(), [4])
    cache = draft.make_cache()
    tokens, laws = draft.draft_distributions(
        [9], features, cache, 3, [RequestRNG(5)], [0.8], context_tokens=[[2, 3, 9]]
    )
    assert len(tokens[0]) == 3
    for token, law in zip(tokens[0], laws[0]):
        assert law[token] > 0 and abs(law.sum() - 1) < 1e-9
    # Only the three target-backed positions are committed.
    assert all(entry.offset == 3 for entry in cache)
    assert draft.stats["chain_steps"] == 2 and draft.stats["committed_positions"] == 3


def test_external_eagle_greedy_matches_ordinary_and_pairs_sidecar():
    target, draft = tiny()
    batch = generator(target, draft)
    prompts = [[1, 2, 3, 4, 5, 6, 7], [3, 1]]
    ids = batch.insert(prompts, max_tokens=[9, 9], sampling_configs=[{"sampling_temp": 0}] * 2)
    got, final = drain(batch)
    for uid, prompt in zip(ids, prompts):
        assert got[uid] == greedy_reference(target, prompt, 9)
        state = final[uid].cache_sidecar
        state.validate("test", len(final[uid].all_tokens))
        assert final[uid].speculative_receipt["kind"] == "external_cohere_eagle"
    stats = batch.scheduler_stats
    # Mechanism assertions: the EAGLE route actually proposed and paired.
    assert stats["external_rounds"] > 0 and stats["proposed_tokens"] > 0
    assert stats["external_context_token_pairings"] > 0
    assert draft.stats["chain_steps"] > 0


def test_sampled_route_is_seed_deterministic():
    runs = []
    for _ in range(2):
        target, draft = tiny()
        batch = generator(target, draft)
        batch.insert([[1, 2, 3]], max_tokens=[8], sampling_configs=[{"sampling_temp": 0.9}])
        runs.append(drain(batch)[0])
    assert runs[0] == runs[1]


def test_eagle_window_cache_serializes_through_prompt_cache_files(tmp_path):
    from mlx2.runtime.models.cache import load_prompt_cache, save_prompt_cache

    target, draft = tiny(window=3)
    cache = draft.make_cache()
    features = target.prefill_body(mx.array([[1, 2, 3, 4]]), target.make_cache(), [4])
    draft.append_context(features, cache, context_tokens=[[2, 3, 4, 5]])
    path = tmp_path / "draft.safetensors"
    save_prompt_cache(str(path), cache)
    restored = load_prompt_cache(str(path))
    for a, b in zip(cache, restored):
        assert isinstance(b, EagleWindowKVCache)
        assert (a.offset, a.window) == (b.offset, b.window)
        np.testing.assert_allclose(np.asarray(a.keys), np.asarray(b.keys))
    empty = tmp_path / "empty.safetensors"
    save_prompt_cache(str(empty), draft.make_cache())
    assert all(entry.offset == 0 for entry in load_prompt_cache(str(empty)))


class _Steerer:
    """Minimal thinking-guard stand-in: asks for steering, never edits logits."""

    close_ids = (39,)

    def __init__(self, vector):
        self.vector = vector
        self.asked = 0

    def residual_steer(self, token):
        self.asked += 1
        return 1, self.vector

    def __call__(self, tokens, logits):
        return logits


def test_external_route_applies_residual_steering():
    target, draft = tiny()
    mx.random.seed(11)
    steer = _Steerer(mx.random.normal((16,)) * 8.0)
    batch = generator(target, draft)
    batch.insert([[1, 2, 3]], max_tokens=[6], sampling_configs=[{"sampling_temp": 0}],
                 logits_processors=[[steer]])
    steered, _ = drain(batch)
    assert batch.scheduler_stats["external_verify_steer_rounds"] > 0
    assert steer.asked > 0
    assert target.model.residual_taps.steer is None
    plain = generator(target, draft)
    plain.insert([[1, 2, 3]], max_tokens=[6], sampling_configs=[{"sampling_temp": 0}])
    unsteered, _ = drain(plain)
    assert plain.scheduler_stats["external_verify_steer_rounds"] == 0
    assert list(steered.values()) != list(unsteered.values())
    # Exactness under steering: equals ordinary greedy decode where every
    # decode step (from the last prompt token on) carries the same vector.
    taps = target.model.residual_taps
    cache = target.make_cache()
    target(mx.array([[1, 2]]), cache=cache)
    token, reference = 3, []
    for _ in range(6):
        taps.steer = (1, steer.vector[None, None, :])
        try:
            logits = target(mx.array([[token]]), cache=cache)
        finally:
            taps.steer = None
        token = int(mx.argmax(logits[0, -1]).item())
        reference.append(token)
    assert list(steered.values())[0] == reference


@pytest.mark.parametrize("width", [1, 2])
def test_ordinary_fallback_binds_and_clears_residual_steering(width):
    target, draft = tiny()
    steer = _Steerer(mx.ones((16,)))
    batch = generator(target, draft)
    uids = batch.insert(
        [[1, 2, 3]] * width, max_tokens=[3] * width,
        logits_processors=[[steer] for _ in range(width)],
    )
    cohort = [batch.lanes[uid] for uid in uids]
    for lane in cohort:
        while lane.anchor is None:
            batch._prefill(lane)
        lane.ordinary = True

    class ObservingTarget:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.model = wrapped.model
            self.observed = []

        def __call__(self, inputs, *, cache):
            self.observed.append(self.model.residual_taps.steer)
            return self.wrapped(inputs, cache=cache)

    observer = ObservingTarget(target)
    batch.model = observer
    batch._ordinary_round(cohort)
    assert len(observer.observed) == 1 and observer.observed[0] is not None
    assert observer.observed[0][1].shape[0] == width
    assert target.model.residual_taps.steer is None


def test_dflash_family_never_receives_context_tokens():
    target, draft = tiny()
    draft.requires_context_tokens = False
    batch = generator(target, draft)
    assert batch.pair_context_tokens is False
    batch.insert([[1, 2, 3, 4]], max_tokens=[3], sampling_configs=[{"sampling_temp": 0}])
    with pytest.raises(ValueError, match="needs the token"):
        drain(batch)


def test_config_rejects_non_parallel_or_eagle3_layouts():
    base = json.loads((EAGLE_SNAPSHOT / "config.json").read_text()) if EAGLE_SNAPSHOT.exists() else None
    if base is None:
        pytest.skip("official North EAGLE artifact not in HF cache")
    config = CohereEagleConfig.from_hf(base, num_target_layers=49)
    assert config.target_layer_ids == [49] and config.sliding_window == 4096
    for key, value in (("transformer_block_type", "sequential"), ("use_qk_norm", True),
                       ("architectures", ["Eagle3LlamaForCausalLM"])):
        with pytest.raises(ValueError):
            CohereEagleConfig.from_hf({**base, key: value}, num_target_layers=49)
    with pytest.raises(ValueError):
        CohereEagleConfig(num_target_layers=49, target_layer_ids=[2, 24, 46])


def _header(path):
    with path.open("rb") as stream:
        size = struct.unpack("<Q", stream.read(8))[0]
        return json.loads(stream.read(size))


def test_real_checkpoint_header_matches_schema_and_loads_strictly():
    if not EAGLE_SNAPSHOT.exists():
        pytest.skip("official North EAGLE artifact not in HF cache")
    from mlx2.adapters.cohere_eagle import inspect_drafter, load_drafter

    config = CohereEagleConfig.from_hf(
        json.loads((EAGLE_SNAPSHOT / "config.json").read_text()), num_target_layers=49
    )
    header = _header(EAGLE_SNAPSHOT / "worker-000-000.safetensors")
    observed = {k: v["shape"] for k, v in header.items() if k != "__metadata__"}
    assert observed == expected_weight_shapes(config)
    record = inspect_drafter(EAGLE_SNAPSHOT, target_config={
        "hidden_size": 2048, "vocab_size": 262144, "num_hidden_layers": 49})
    assert record["args"].num_hidden_layers == 3
    assert len(record["header_sha256"]) == 1
    # Strict CPU load of the real payload (158 MB) into the module tree.
    model = load_drafter(record, target_model=None)
    assert model.fc.weight.shape == (2048, 4096)
    assert model.fc.bias.dtype == mx.bfloat16


def test_north_adapter_external_policy_fails_closed_before_loading(tmp_path):
    from mlx2.adapters.north_mini_code import NorthMiniCodeAdapter

    with pytest.raises(ValueError, match="no qualified overrides"):
        NorthMiniCodeAdapter("/missing", execution_policy={"num_draft": 2})
    with pytest.raises(ValueError, match="no qualified overrides"):
        NorthMiniCodeAdapter("/missing", execution_policy={"draft_model": "x", "fly": 1})
    adapter = object.__new__(NorthMiniCodeAdapter)
    assert "backend" not in adapter.execution_config(max_lanes=4, prefill_step=64)
    adapter.draft_model, adapter.external_policy = object(), {"draft_model": "x"}
    config = adapter.execution_config(max_lanes=4, prefill_step=64)
    assert config["backend"] == "external_draft" and config["num_draft"] == 3
    assert NorthMiniCodeAdapter.external_profile_name(False) == "north-mini-code-apcv2-cohere-eagle"


def test_north_adapter_rejects_num_draft_outside_block_before_target_load(monkeypatch):
    if not EAGLE_SNAPSHOT.exists():
        pytest.skip("official North EAGLE artifact not in HF cache")
    import mlx2.adapters.north_mini_code as north

    config = {"hidden_size": 2048, "vocab_size": 262144, "num_hidden_layers": 49}
    monkeypatch.setattr(north, "inspect_artifact", lambda path: {"config": config})
    for count in (0, 8, 2.0):
        with pytest.raises(ValueError, match="num_draft"):
            north.NorthMiniCodeAdapter(
                "/unused", execution_policy={"draft_model": str(EAGLE_SNAPSHOT), "num_draft": count}
            )


def test_accepted_multi_token_tails_pair_and_stay_exact():
    """Force acceptance so multi-token tails (consumed > 1) exercise pairing.

    The drafter's head is biased toward the target's most frequent greedy
    token; every accepted run commits an L>1 tail through ``append_context``
    with ``(history + [anchor])[-L:]`` pairing.  Output must still equal the
    ordinary greedy reference (exactness does not depend on the drafter)."""
    from collections import Counter

    target, draft = tiny()
    prompt = [1, 2, 3, 4, 5]
    reference = greedy_reference(target, prompt, 16)
    favourite = Counter(reference).most_common(1)[0][0]
    base = draft._logits

    def biased(hidden):
        bonus = mx.where(mx.arange(40) == favourite, 1e4, 0.0)
        return base(hidden) + bonus

    draft._logits = biased
    batch = generator(target, draft)
    uid = batch.insert([prompt], max_tokens=[16], sampling_configs=[{"sampling_temp": 0}])[0]
    got, final = drain(batch)
    assert got[uid] == reference
    assert batch.scheduler_stats["accepted_proposals"] > 0
    final[uid].cache_sidecar.validate("test", len(final[uid].all_tokens))
    # Every committed draft position is backed by a target feature.
    assert all(c.offset + final[uid].cache_sidecar.state[1].shape[1]
               == len(final[uid].all_tokens) for c in final[uid].cache_sidecar.state[0])
