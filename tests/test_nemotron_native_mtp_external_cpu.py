"""Nemotron-H native MTP head on the external draft/verify engine, CPU only.

Tiny random Nemotron-H targets with an MTP head (no real weights).  Pins:
(a) the adapter-selected ``mtp_seed`` external feature is bitwise the seed the
self-MTP route hands ``mtp_step`` (``mtp_backbone`` / ``mtp_verify_backbone``);
(b) the generic native-MTP drafter on the external engine decodes ordinary
greedy output; (c) ``NativeMTPDraftModel.draft_lazy`` returns the host chain's
tokens and ``device_round {lazy_draft, decide}`` decodes what the option-off
engine decodes; (d) ``device_commit`` commits on the host with a counted
reason (no Mamba device rollback); (e) the Lightning adapter's policy split,
refusals and route contract.
"""

import copy

import mlx.core as mx
import numpy as np
import pytest

mx.set_default_device(mx.cpu)

from mlx2.runtime.drafters.native_mtp import NativeMTPDraftModel
from mlx2.runtime.models.nemotron_h import Model, ModelArgs
from mlx2.runtime.speculative_device_round import (
    DECLINE_REASONS,
    DEVICE_COMMIT_SKIPS,
)

PROMPT = [3, 9, 14, 2, 27, 5, 11, 20, 8, 17, 6, 25]
BOTH = {"lazy_draft": True, "decide": True}


def tiny(seed=28, mtp_pattern=("*", "-")):
    mx.random.seed(seed)
    args = ModelArgs(
        model_type="nemotron_h", vocab_size=32, hidden_size=16, intermediate_size=24,
        num_hidden_layers=5, max_position_embeddings=256, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, attention_bias=False, mamba_num_heads=2,
        mamba_head_dim=8, mamba_proj_bias=False, ssm_state_size=4, conv_kernel=3,
        n_groups=1, mlp_bias=False, layer_norm_epsilon=1e-5, use_bias=False,
        use_conv_bias=True, hybrid_override_pattern=["-", "M", "*", "-", "M"],
        num_nextn_predict_layers=1, mtp_hybrid_override_pattern=list(mtp_pattern),
    )
    model = Model(args)
    model.eval()
    mx.eval(model.parameters())
    return model


def aligned(seed=28, mixing=0.1):
    """A tiny target whose random MTP head mostly agrees with it.

    Every residual branch is damped (the target's next token is mostly a
    function of the current one, yet Mamba and attention still move it) and
    the head's fusion passes the next token's embedding through, so drafts
    are accepted often enough to exercise multi-row commits and rejections.
    """
    from mlx.utils import tree_flatten

    model = tiny(seed)
    updates = []
    for name, value in tree_flatten(model.parameters()):
        if name.endswith(("out_proj.weight", "o_proj.weight", "down_proj.weight")):
            updates.append((name, value * mixing))
    width = model.args.hidden_size
    eh = model.mtp.layers[0].eh_proj.weight
    updates.append(("mtp.layers.0.eh_proj.weight", mx.concatenate(
        [mx.eye(width, dtype=eh.dtype), eh[:, width:] * 0.05], axis=1)))
    model.load_weights(updates, strict=False)
    mx.eval(model.parameters())
    return model


def seeded(build=None):
    model = (build or aligned)()
    model.select_external_feature_source("mtp_seed")
    return model


def ordinary_greedy(model, prompt, count):
    cache = model.make_cache()
    tokens, out = list(prompt), []
    for step in range(count):
        logits = model(mx.array([tokens if step == 0 else [tokens[-1]]]), cache=cache)
        token = int(mx.argmax(logits[0, -1]).item())
        tokens.append(token)
        out.append(token)
    return out


def engine(model, num_draft=2, **options):
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    draft = NativeMTPDraftModel(model, num_draft=num_draft, revision="a" * 64)
    return ExternalDraftBatchGenerator(model, draft_model=draft, binding="test",
                                       num_draft=num_draft, prefill_step_size=4, **options)


def decode(steps=14, prompts=(PROMPT,), sampling=None, num_draft=2, build=None, **options):
    batch = engine(seeded(build), num_draft=num_draft, **options)
    try:
        uids = batch.insert([list(p) for p in prompts], max_tokens=[steps] * len(prompts),
                            sampling_configs=[sampling or {"sampling_temp": 0}] * len(prompts))
        got = {uid: [] for uid in uids}
        for _ in range(600):
            _, responses = batch.next()
            for response in responses:
                got[response.uid].append(response.token)
            if not batch.lanes:
                break
        else:
            raise AssertionError("scheduler stalled")
        return [got[uid] for uid in uids], dict(batch.scheduler_stats)
    finally:
        batch.close()


# -- (a) the mtp_seed feature is the self-MTP seed ----------------------------


def test_feature_source_selection_is_validated():
    model = tiny()
    assert model.external_feature_source is None and model.external_feature_width is None
    with pytest.raises(ValueError, match="unsupported"):
        model.select_external_feature_source("hc_mean")
    model.select_external_feature_source("mtp_seed")
    assert model.external_feature_width == model.args.hidden_size
    last = len(model.layers) - 1
    with pytest.raises(ValueError, match="post-norm hidden"):
        model.forward_with_taps(mx.array([[1, 2]]), model.make_cache(), (1, last))
    model.select_external_feature_source(None)
    assert model.external_feature_source is None
    headless = tiny()
    del headless.mtp
    with pytest.raises(ValueError, match="loaded MTP head"):
        headless.select_external_feature_source("mtp_seed")


def test_prefill_seed_equals_mtp_backbone_bitwise():
    model = seeded()
    last = len(model.layers) - 1
    tokens = mx.array([PROMPT[:7]])
    _, seed_ref = model.mtp_backbone(tokens, cache=model.make_cache())
    logits, seed = model.forward_with_taps(tokens, model.make_cache(), (last,))
    np.testing.assert_array_equal(np.asarray(seed), np.asarray(seed_ref))
    np.testing.assert_array_equal(np.asarray(logits), np.asarray(model.logits(seed_ref)))
    body = model.prefill_body(tokens, model.make_cache(), (last,))
    np.testing.assert_array_equal(np.asarray(body), np.asarray(seed_ref))
    # The ordinary (unselected) taps are pre-final-norm, so they differ.
    model.select_external_feature_source(None)
    _, taps = model.forward_with_taps(tokens, model.make_cache(), (last,))
    assert not np.array_equal(np.asarray(taps), np.asarray(seed_ref))


def test_verify_seed_equals_mtp_verify_backbone_bitwise():
    model = seeded()
    last = len(model.layers) - 1
    base = model.make_cache()
    model(mx.array([PROMPT[:5]]), base)
    reference, candidate = copy.deepcopy(base), copy.deepcopy(base)
    for cache in (*reference, *candidate):
        cache.start_speculation()
    draft = mx.array([[4, 5, 6]])
    # The self-MTP B1 verifier (one ordinary token step per row).
    _, seed_ref = model.mtp_verify_backbone(draft, reference)
    _, seed = model.forward_with_taps(draft, candidate, (last,))
    np.testing.assert_array_equal(np.asarray(seed), np.asarray(seed_ref))


def test_short_exact_policy_seed_is_the_row_exact_final_norm():
    model = seeded()
    try:
        model.set_short_segment_policy("layerwise_short_exact")
    except ValueError as error:  # pragma: no cover - policy refused on this config
        pytest.skip(f"short exact policy unavailable: {error}")
    last = len(model.layers) - 1
    base = model.make_cache()
    model(mx.array([PROMPT[:5]]), base)
    reference, candidate = copy.deepcopy(base), copy.deepcopy(base)
    for cache in (*reference, *candidate):
        cache.start_speculation()
    draft = mx.array([[4, 5, 6]])
    refusal = model._short_segment_refusal(draft, candidate)
    if refusal is not None:
        # The recipe also needs the fused Mamba conv/SSM/projection policies
        # (set by the adapter on real checkpoints), refused here as "mamba conv auto".
        pytest.skip(f"short exact recipe refused on the tiny CPU config: {refusal}")
    _, seed_ref = model.mtp_verify_backbone(draft, reference)
    _, seed = model.forward_with_taps(draft, candidate, (last,))
    np.testing.assert_array_equal(np.asarray(seed), np.asarray(seed_ref))


# -- (b) external engine greedy equals ordinary greedy -------------------------


@pytest.mark.parametrize("build", [aligned, tiny], ids=["aligned_head", "random_head"])
@pytest.mark.parametrize("num_draft", [1, 2, 3])
def test_external_native_mtp_greedy_equals_ordinary(num_draft, build):
    expected = ordinary_greedy(build(), PROMPT, 24)
    got, stats = decode(steps=24, num_draft=num_draft, build=build)
    assert got == [expected]
    assert stats["proposed_tokens"] > 0
    assert stats["external_context_token_pairings"] > 0
    if build is aligned:
        # Accepted multi-row commits and rejections are both exercised.
        assert 0 < stats["accepted_proposals"] < stats["proposed_tokens"]


def test_external_native_mtp_two_lanes_greedy_equals_ordinary():
    prompts = (PROMPT, list(reversed(PROMPT)))
    model = aligned()
    got, _ = decode(prompts=prompts)
    assert got == [ordinary_greedy(model, p, 14) for p in prompts]


# -- (c) draft_lazy and the device round ----------------------------------------


def _committed_tail(model, draft, prompt, anchor=7):
    """A pending seed tail and its pairing as the engine would hold them."""
    last = len(model.layers) - 1
    seeds = model.prefill_body(mx.array([prompt]), model.make_cache(), (last,))
    return seeds, list(prompt[1:]) + [anchor]  # seed t pairs with token t + 1


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_draft_lazy_equals_host_draft_distributions(depth):
    model = seeded()
    draft = NativeMTPDraftModel(model, num_draft=3, revision="a" * 64)
    seeds, following = _committed_tail(model, draft, PROMPT[:6])
    anchor = following[-1]
    cache_a, cache_b = draft.make_cache(), draft.make_cache()
    tokens, _laws = draft.draft_distributions(
        [anchor], seeds, draft.batch_caches([cache_a]), depth, [None], [0.0],
        context_tokens=[following])
    lazy = draft.draft_lazy(anchor, seeds, cache_b, depth, context_tokens=[following])
    assert lazy.dtype == mx.int32 and lazy.shape == (depth,)
    assert [int(t) for t in lazy.tolist()] == tokens[0]
    # The committed head caches are identical too (pending seeds appended).
    for a, b in zip(cache_a, cache_b, strict=True):
        assert a.offset == b.offset
        for x, y in zip(a.state, b.state, strict=True):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_draft_lazy_with_processors_equals_host_path():
    from mlx2.runtime.sample_utils import make_logits_processors

    model = seeded()
    draft = NativeMTPDraftModel(model, num_draft=3, revision="a" * 64)
    seeds, following = _committed_tail(model, draft, PROMPT[:6])
    anchor = following[-1]
    processors = make_logits_processors(presence_penalty=1.5)
    history = list(PROMPT[:6])
    tokens, _ = draft.draft_distributions(
        [anchor], seeds, draft.batch_caches([draft.make_cache()]), 3, [None], [0.0],
        context_tokens=[following], logits_processors=[processors],
        processor_histories=[history])
    lazy = draft.draft_lazy(anchor, seeds, draft.make_cache(), 3, context_tokens=[following],
                            logits_processors=processors, processor_history=history)
    assert [int(t) for t in lazy.tolist()] == tokens[0]


def test_draft_lazy_refusals():
    from mlx2.runtime.proposal_composition import LazyDecline

    model = seeded()
    draft = NativeMTPDraftModel(model, num_draft=2, revision="a" * 64)
    seeds, following = _committed_tail(model, draft, PROMPT[:4])
    with pytest.raises(ValueError, match="end with the anchor"):
        draft.draft_lazy(1, seeds, draft.make_cache(), 2, context_tokens=[following])
    with pytest.raises(ValueError, match="pairing"):
        draft.draft_lazy(following[-1], seeds, draft.make_cache(), 2)
    with pytest.raises(LazyDecline):
        draft.draft_lazy(4, seeds[:, :0], draft.make_cache(), 2, context_tokens=[[]])
    with pytest.raises(ValueError, match="host anchor"):
        draft.draft_lazy(mx.array([4]), seeds, draft.make_cache(), 2)
    with pytest.raises(ValueError, match="one lane"):
        draft.draft_lazy(4, mx.concatenate([seeds, seeds]), draft.make_cache(), 2)


def test_draft_lazy_dead_position_keeps_histories_in_vocabulary():
    from mlx2.runtime.drafters.dflash2 import DRAFT_LAZY_HOST_REDO

    model = seeded()
    draft = NativeMTPDraftModel(model, num_draft=3, revision="a" * 64)
    seeds, following = _committed_tail(model, draft, PROMPT[:6])
    anchor = following[-1]
    history = list(PROMPT[:6])
    vocab = model.args.vocab_size
    seen = []

    def dead_after_first(tokens, logits):
        # Records every history it is handed; kills every logit at position 1.
        seen.append(tokens)
        if tokens.shape[0] > len(history) + 1:
            return mx.full(logits.shape, -float("inf"), dtype=logits.dtype)
        return logits

    # Drafting calls a processor's ``probe`` (no isolating copy of the closure).
    dead_after_first.probe = dead_after_first
    lazy = draft.draft_lazy(anchor, seeds, draft.make_cache(), 3, context_tokens=[following],
                            logits_processors=[dead_after_first], processor_history=history)
    assert [int(t) for t in lazy.tolist()] == [DRAFT_LAZY_HOST_REDO] * 3
    assert len(seen) == 3
    for tokens in seen:
        values = [int(t) for t in tokens.tolist()]
        assert all(0 <= value < vocab for value in values), values


def test_restored_device_round_does_not_double_count_drafter_stats(monkeypatch):
    from mlx2.runtime.proposal_composition import LazyDecline

    real = NativeMTPDraftModel.draft_lazy
    calls = {"n": 0}

    def decline_first(self, *args, **kwargs):
        calls["n"] += 1
        tokens = real(self, *args, **kwargs)  # mutates head cache and stats
        if calls["n"] == 1:
            raise LazyDecline("test_decline")
        return tokens

    monkeypatch.setattr(NativeMTPDraftModel, "draft_lazy", decline_first)
    batch = engine(seeded(), device_round=BOTH)
    off_batch = engine(seeded())
    try:
        outputs = []
        for b in (batch, off_batch):
            uid = b.insert([PROMPT], max_tokens=[14], sampling_configs=[{"sampling_temp": 0}])[0]
            got = []
            for _ in range(600):
                _, responses = b.next()
                got += [r.token for r in responses if r.uid == uid]
                if not b.lanes:
                    break
            outputs.append(got)
        assert outputs[0] == outputs[1]
        assert batch.scheduler_stats[DECLINE_REASONS] == {"test_decline": 1}
        for key in ("draft_calls", "committed_positions", "chain_steps"):
            assert batch.draft.stats[key] == off_batch.draft.stats[key], key
    finally:
        batch.close()
        off_batch.close()


@pytest.mark.parametrize(("options", "sampling"), [
    (BOTH, {"sampling_temp": 0}),
    ({"lazy_draft": True}, {"sampling_temp": 0}),
    ({**BOTH, "processors": True}, {"sampling_temp": 0, "presence_penalty": 1.5}),
])
def test_cascaded_inkling_head_device_round_equals_the_host_round(options, sampling):
    from test_inkling_mtp_cpu import attached

    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    def run(**options):
        model, draft = attached()
        batch = ExternalDraftBatchGenerator(model, draft_model=draft, binding="test",
                                            num_draft=3, prefill_step_size=64, **options)
        try:
            prompt = [5, 17, 2, 33, 8, 21, 4, 11, 26, 9, 14, 3]
            uid = batch.insert([prompt], max_tokens=[24], sampling_configs=[sampling])[0]
            got = []
            for _ in range(400):
                _, responses = batch.next()
                got += [r.token for r in responses if r.uid == uid]
                if not batch.lanes:
                    break
            return got, dict(batch.scheduler_stats), dict(draft.stats)
        finally:
            batch.close()

    off, off_stats, off_draft = run()
    on, stats, on_draft = run(device_round=options)
    assert on == off
    rounds = stats["external_device_round_rounds"]
    assert rounds > 0
    assert stats["external_device_round_lazy_draft_lanes"] == rounds
    assert stats["external_device_round_declines"] == 0 and stats[DECLINE_REASONS] == {}
    # The same drafts were proposed, with real accepts and rejects.
    for key in ("proposed_tokens", "accepted_proposals"):
        assert stats[key] == off_stats[key], key
    assert 0 < stats["accepted_proposals"] < stats["proposed_tokens"]
    # The drafter's whole counter dict, ``lazy_chain_calls`` included.
    assert on_draft == off_draft


def test_device_round_greedy_output_unchanged_one_read_per_round():
    off, off_stats = decode()
    on, stats = decode(device_round=BOTH)
    assert on == off
    rounds = stats["external_device_round_rounds"]
    assert rounds > 0
    assert stats["external_device_round_lazy_draft_lanes"] == rounds
    assert stats["external_device_round_host_reads"] == rounds
    assert stats["external_device_round_declines"] == 0
    assert stats[DECLINE_REASONS] == {}
    # The same tokens were proposed and accepted as on the host round.
    assert stats["accepted_proposals"] > 0
    for key in ("proposed_tokens", "accepted_proposals"):
        assert stats[key] == off_stats[key], key


def test_device_round_lazy_draft_alone_and_two_lanes():
    off, _ = decode()
    on, stats = decode(device_round={"lazy_draft": True})
    assert on == off and stats["external_device_round_rounds"] > 0
    prompts = (PROMPT, list(reversed(PROMPT)))
    off, _ = decode(prompts=prompts)
    on, stats = decode(prompts=prompts, device_round=BOTH)
    assert on == off and stats["external_device_round_rounds"] > 0


def test_device_round_presence_penalty_lane_matches_host():
    sampling = {"sampling_temp": 0, "presence_penalty": 1.5}
    off, _ = decode(sampling=sampling)
    on, stats = decode(sampling=sampling, device_round={**BOTH, "processors": True})
    assert on == off
    if stats["external_device_round_rounds"] == 0:
        pytest.skip("no processor lane reached the device round on this engine")


def test_device_round_fixed_shape_greedy_is_exact_on_cpu():
    off, _ = decode()
    on, stats = decode(device_round={**BOTH, "fixed_shape": True})
    assert on == off
    assert stats["external_device_round_fixed_shape_rounds"] > 0


# -- (d) device_commit declines on a recurrent target ---------------------------


def test_device_commit_commits_on_the_host_with_a_counted_reason():
    off, _ = decode()
    on, stats = decode(device_round={**BOTH, "device_commit": True})
    assert on == off
    skips = stats[DEVICE_COMMIT_SKIPS]
    assert sum(skips.values()) > 0
    # NemotronMambaCache is an ArraysCache subclass the device form does not
    # take (no device rollback): every round commits on the host.
    assert skips == {"target_cache": stats["external_device_round_rounds"]}
    assert stats["external_device_round_device_commit_rounds"] == 0
    assert stats["external_device_round_device_commit_enters"] == 0
    # The rounds themselves still ran through the device round (host commit).
    assert stats["external_device_round_rounds"] > 0


# -- (e) the Lightning adapter's opt-in route ----------------------------------


def test_lightning_policy_split_keeps_self_mtp_policy_unchanged():
    from mlx2.adapters.nemotron35_lightning import split_external_native_mtp_policy

    assert split_external_native_mtp_policy(None) == ({}, {})
    assert split_external_native_mtp_policy({"num_draft": 3}) == ({"num_draft": 3}, {})
    native, external = split_external_native_mtp_policy({
        "draft_model": "/m", "num_draft": 2, "device_round": BOTH,
        "mamba_segment": "layerwise_short_exact",
    })
    assert native == {"mamba_segment": "layerwise_short_exact"}
    assert external == {"draft_model": "/m", "num_draft": 2, "device_round": BOTH}
    with pytest.raises(ValueError, match="require draft_model"):
        split_external_native_mtp_policy({"device_round": BOTH})


def _adapter_shell(tmp_path, monkeypatch, *, has_mtp=True):
    from mlx2.adapters import nemotron35_lightning as L

    artifact = {"identity": {"path": str(tmp_path), "fingerprint": "f" * 64, "files": []},
                "has_mtp": has_mtp, "config": {}, "weight_map": {}, "mtp_path": None}
    monkeypatch.setattr(L, "inspect_artifact", lambda _path: artifact)

    class Loaded(Exception):
        pass

    def stop(*_args, **_kwargs):
        raise Loaded

    # Stop at the first step that would touch weights.
    monkeypatch.setattr(L, "configure_environment", stop)
    return L, Loaded


def test_lightning_external_policy_refusals_before_any_load(tmp_path, monkeypatch):
    L, Loaded = _adapter_shell(tmp_path, monkeypatch)
    adapter = object.__new__(L.Nemotron35LightningAdapter)
    with pytest.raises(ValueError, match="target artifact itself"):
        adapter._init_lightning(str(tmp_path), execution_policy={"draft_model": "/elsewhere"})
    with pytest.raises(ValueError, match="above three"):
        adapter._init_lightning(str(tmp_path),
                                execution_policy={"draft_model": str(tmp_path), "num_draft": 4})
    with pytest.raises(ValueError, match="unknown device_round"):
        adapter._init_lightning(str(tmp_path), execution_policy={
            "draft_model": str(tmp_path), "device_round": {"bogus": True}})
    with pytest.raises(ValueError, match="supports num_draft"):
        adapter._init_lightning(str(tmp_path), execution_policy={
            "draft_model": str(tmp_path), "continuation_pool": {}})
    # A valid policy passes validation and reaches the load step.
    with pytest.raises(Loaded):
        adapter._init_lightning(str(tmp_path), execution_policy={
            "draft_model": str(tmp_path), "num_draft": 3, "device_round": BOTH})
    assert adapter.external_policy["num_draft"] == 3
    # Without a verified head the external route fails closed.
    L, _ = _adapter_shell(tmp_path, monkeypatch, has_mtp=False)
    with pytest.raises(ValueError, match="verified MTP head"):
        object.__new__(L.Nemotron35LightningAdapter)._init_lightning(
            str(tmp_path), execution_policy={"draft_model": str(tmp_path)})


def test_lightning_bound_route_contract_and_engine(tmp_path):
    from mlx2.adapters import nemotron35_lightning as L
    from mlx2.runtime.drafters.native_mtp import inspect_native_mtp_policy

    adapter = object.__new__(L.Nemotron35LightningAdapter)
    adapter._has_mtp, adapter._num_draft = True, 2
    adapter.layout = L.CACHE_LAYOUT
    adapter.identity = {"path": str(tmp_path), "fingerprint": "f" * 64, "files": []}
    adapter.descriptor = L.descriptor_for(has_mtp=True)
    adapter._parse_external_policy(
        {"draft_model": str(tmp_path), "num_draft": 2, "device_round": BOTH},
        family="Nemotron 3.5 Lightning")
    record = inspect_native_mtp_policy(adapter.external_policy, str(tmp_path), "f" * 64)
    adapter._check_num_draft(record)
    adapter.model = aligned()
    adapter._bind_external_native_mtp(record)

    assert adapter.model.external_feature_source == "mtp_seed"
    assert adapter.draft_model.receipt_kind == "external_native_mtp"
    assert adapter.draft_model.receipt_settings["qualified"] is False
    from mlx2.contracts import Capability

    assert Capability.EXTERNAL_DRAFT in adapter.descriptor.capabilities
    assert adapter.descriptor.metadata["qualification"] == "pending"
    assert adapter.identity["fingerprint"] != "f" * 64
    config = adapter.execution_config(max_lanes=1, prefill_step=64)
    assert config["backend"] == "external_draft" and config["num_draft"] == 2
    # Exact only at width one: wider serving shapes are refused.
    with pytest.raises(ValueError, match="exact at width one"):
        adapter.execution_config(max_lanes=2, prefill_step=64)
    with pytest.raises(ValueError, match="exact at width one"):
        adapter.create_external_batch(completion_batch_size=2, prefill_step_size=4)
    assert adapter.profile_name(False) == L.Nemotron35LightningAdapter.EXTERNAL_PROFILE
    with pytest.raises(ValueError, match="not native MTP"):
        adapter.profile_name(True)
    contract = adapter.route_capability_contract()
    external = contract["external_draft"]
    assert external["implemented"] is True and external["selected"] is True
    assert external["qualified"] is False and external["observed_used"] is False
    assert external["verification"] == "exact at width one" and external["max_lanes"] == 1
    assert external["device_round"]["selected"] is True
    assert external["device_round"]["policy"]["lazy_draft"] is True
    assert external["device_round"]["device_commit"].startswith("declined")
    # An all-false policy is inert, not selected.
    adapter.external_policy["device_round"] = {"lazy_draft": False}
    assert adapter._device_round_contract()["selected"] is False
    assert adapter._device_round_contract()["policy"] is None
    adapter.external_policy["device_round"] = BOTH
    assert contract["native_mtp"]["default"] is False
    assert contract["default_route"] == "ordinary"
    # The adapter's own batch is the external engine with the device round on.
    batch = adapter.create_external_batch(completion_batch_size=1, prefill_step_size=4)
    try:
        uid = batch.insert([PROMPT], max_tokens=[10], sampling_configs=[{"sampling_temp": 0}])[0]
        got = []
        for _ in range(400):
            _, responses = batch.next()
            got += [r.token for r in responses if r.uid == uid]
            if not batch.lanes:
                break
        assert got == ordinary_greedy(aligned(), PROMPT, 10)
        assert batch.scheduler_stats["external_device_round_rounds"] > 0
    finally:
        batch.close()


def test_lightning_contract_without_head_declares_external_unimplemented():
    from mlx2.adapters import nemotron35_lightning as L

    adapter = object.__new__(L.Nemotron35LightningAdapter)
    adapter._has_mtp, adapter._num_draft = False, 2
    adapter.layout = L.CACHE_LAYOUT
    adapter.descriptor = L.descriptor_for(has_mtp=False)
    contract = adapter.route_capability_contract()
    assert contract["external_draft"]["implemented"] is False
    assert L.descriptor_for(has_mtp=False).metadata["external_draft"] == "unsupported-no-mtp-head"
