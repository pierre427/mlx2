"""Qwen3.8-27B adapter on GPU-resident speculative rounds (``device_round``), CPU only.

Gap 1 (adapter policy): ``device_round`` and ``prompt_copy_round`` are
accepted and validated on both external routes of ``Qwen3827BAdapter`` (the
DFlash2 drafter and the native-MTP / prompt-lookup tree tiles), forwarded by
its own ``create_external_batch`` and reported in the execution config
(route receipt).  The Qwen3.6-27B subclass, which reuses this load path,
refuses them.

Gap 2 (tree provider): ``Qwen38TargetTreeExecutor`` declares
``device_tree_inputs``; ``qwen38_tensorfold.forward`` embeds a device token
row in place of the vendored forward's host-id embedding.  The TensorFold
kernels are Metal-only, so the end-to-end checks run the provider with a CPU
oracle in place of the vendored ``lane_tree`` (every root-to-node path
replayed on a private cache copy from the forward's own embedded rows, the
same ``tree_forward``/``commit_tree`` contract).  A tiny random Qwen3.8-
shaped target with a native MTP head drafts ``tree15`` rounds at B1; with
``tree_lazy`` (and ``tree_decide``) the rounds run on the device (not
declined) and every round's tokens, decision, RNG draws and cache planes
equal the host tree round's.  The expected declines (width above one,
processors, a DFlash2 drafter without ``draft_tree_lazy``) still fire with
their reasons and serve the host round.
"""

from __future__ import annotations

import copy
import dataclasses

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.speculative_device_tree import COUNTERS, DECLINES

PROMPT = [3, 9, 14, 2, 27, 5, 11, 20, 8, 17, 6, 25]
LAZY = {"tree_lazy": True}
DECIDE = {"tree_lazy": True, "tree_decide": True}


@pytest.fixture(autouse=True)
def cpu(monkeypatch):
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    monkeypatch.setenv("MLX2_DFLASH_TOPOLOGY", "tree15")
    monkeypatch.delenv("MLX2_TARGET_TREE_EXECUTION", raising=False)
    monkeypatch.delenv("MLX2_QWEN_TARGET_EXECUTION", raising=False)
    monkeypatch.setenv("MLX2_AST_CONTROLLER_MODE", "deterministic")
    monkeypatch.setenv("MLX2_AST_CONTROLLER_COST",
                       '{"a_ms": 5.0, "b_ms": 0.5, "ordinary_ms": 4.0}')
    yield
    mx.set_default_device(previous)


# -- a tiny Qwen3.8-27B-shaped target with a native MTP head --------------------
def tiny(seed=11, mixing=0.2, vocab=48):
    """Dense hybrid (3 GDN + 1 attention layer) with an MTP head that mostly
    agrees with the target: residual branches damped, and the head's fusion
    passes the next token's embedding through, so drafts are accepted often
    enough to exercise multi-row commits, branch accepts and root rejects."""
    from mlx.utils import tree_flatten

    from mlx2.runtime.models.qwen38_27b import Model, ModelArgs
    from mlx2.seeded_fixtures import legacy_conv_init

    text = dict(
        model_type="qwen3_5", hidden_size=32, intermediate_size=48,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, vocab_size=vocab, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=3, full_attention_interval=4,
        mtp_num_hidden_layers=1, partial_rotary_factor=0.5,
        rope_parameters=None, max_position_embeddings=512,
        tie_word_embeddings=False, rms_norm_eps=1e-6,
    )
    mx.random.seed(seed)
    with legacy_conv_init():
        model = Model(ModelArgs(model_type="qwen3_5", text_config=text))
    model.eval()
    updates = []
    for name, value in tree_flatten(model.parameters()):
        if name.endswith(("out_proj.weight", "o_proj.weight", "down_proj.weight")):
            updates.append((name, value * mixing))
    width = text["hidden_size"]
    fc = model.language_model.mtp.fc.weight
    updates.append(("language_model.mtp.fc.weight", mx.concatenate(
        [mx.eye(width, dtype=fc.dtype) * 4.0, fc[:, width:] * 0.05], axis=1)))
    model.load_weights(updates, strict=False)
    mx.eval(model.parameters())
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


# -- a CPU oracle for the vendored TensorFold lane_tree --------------------------
def _paths(parents):
    paths = []
    for row, parent in enumerate(parents):
        paths.append([row] if parent < 0 else [*paths[parent], row])
    return paths


def _private(cache):
    cloned = copy.deepcopy(cache)
    for item in cloned:
        if getattr(item, "keys", None) is not None and isinstance(getattr(item, "offset", None), int):
            item.keys = item.keys[..., : item.offset, :]
            item.values = item.values[..., : item.offset, :]
    return cloned


class OracleLaneTree:
    """``lane_tree``'s ``tree_forward``/``commit_tree`` contract on the CPU.

    Embeds the host ids once through ``core.embed_tokens`` exactly as the
    vendored ``lane_multi`` does, then replays every root-to-node path of
    those embedded rows on a private cache copy; the commit replays the
    accepted path's rows on the lane cache."""

    HIDDEN_SINK = None

    def __init__(self):
        self.embedded = []  # the embedding input of every forward (host or device ids)

    def tree_forward(self, core, head, tokens, parents, cache, start):
        assert isinstance(tokens, list) and all(type(t) is int for t in tokens)
        embedding = core.embed_tokens
        hidden = embedding(mx.array([tokens], dtype=mx.uint32))
        self.embedded.append(type(embedding).__name__)
        rows = []
        for path in _paths(parents):
            out = core(None, _private(cache), input_embeddings=hidden[:, mx.array(path)])
            rows.append(out[:, -1:])
        post = mx.concatenate(rows, axis=1)
        if self.HIDDEN_SINK is not None:
            self.HIDDEN_SINK.append(post)
        mx.eval(post)
        return head(post), {"core": core, "hidden": hidden}

    def commit_tree(self, cache, record, path, window, start):
        if path:
            out = record["core"](None, cache, input_embeddings=record["hidden"][:, mx.array(path)])
            mx.eval(out, [c.state for c in cache])


def provider(monkeypatch, oracle):
    from mlx2.adapters.qwen38_target_tree_executor import Qwen38TargetTreeExecutor
    from mlx2.runtime import qwen38_tensorfold

    # A private token table: the real vendored module is validated and bound
    # first (as in serving), then its token resolves to the CPU oracle.
    monkeypatch.setattr(qwen38_tensorfold, "_VALIDATED_EXECUTORS", {})
    executor = Qwen38TargetTreeExecutor.create()
    qwen38_tensorfold._VALIDATED_EXECUTORS[executor.source_revision_token] = oracle
    forward = qwen38_tensorfold.forward

    def forward_many(model, token_rows, parent_rows, caches, layers, *, cached=False,
                     validated_token=None, ragged=False):
        """Cohorts (host lists only) as one oracle forward per lane."""
        results = [forward(model, row, parents, cache, layers, cached=cached,
                           validated_token=validated_token)
                   for row, parents, cache in zip(token_rows, parent_rows, caches, strict=True)]
        width = max(int(result[0].shape[1]) for result in results)

        def padded(value):
            return mx.pad(value, [(0, 0), (0, width - int(value.shape[1])), (0, 0)])

        return (mx.concatenate([padded(result[0]) for result in results]),
                mx.concatenate([padded(result[1]) for result in results]),
                qwen38_tensorfold.TensorfoldCohortTransaction(
                    [result[2] for result in results]))

    monkeypatch.setattr(qwen38_tensorfold, "forward_many", forward_many)
    return executor


# -- the external engine on the tiny target -----------------------------------
def _planes(cache):
    from mlx.utils import tree_flatten

    planes = []
    for entry in cache:
        try:
            offset = getattr(entry, "offset", None)
        except RuntimeError:
            # A device-form ring (device_commit) has no host offset; its state
            # (keys, values, slot_pos, pos) is recorded whole below.
            offset = None
        if isinstance(offset, int):
            planes.append(np.array([offset]))
        for _key, value in tree_flatten(entry.state):
            if value is None:
                continue
            if isinstance(offset, int) and getattr(entry, "keys", None) is not None \
                    and value.ndim == 4:
                value = value[..., :offset, :]
            planes.append(np.asarray(value))
    return planes


def _decision(decision):
    return {"accepted": decision.accepted, "accepted_rows": decision.accepted_rows,
            "emitted": list(decision.emitted), "commit_rows": decision.commit_rows}


def decode(monkeypatch, *, prompts=(PROMPT,), steps=24, sampling=None, processors=None,
           draft=None, model=None, oracle=None, tree=True, **options):
    from mlx2.runtime.drafters.native_mtp import NativeMTPDraftModel
    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator

    model = model or tiny()
    oracle = oracle or OracleLaneTree()
    draft = draft or NativeMTPDraftModel(model, num_draft=4, revision="a" * 64)
    width = len(prompts)
    tree_width = 1 if width == 1 else 4
    # The served tree15_b1_b4_chain_b5plus_v1 shape; ``tree=False`` is the
    # chain route (``"batch_size_route": null``).
    route = dict(
        dynamic_singleton_tree=True, dynamic_tree_max_width=tree_width,
        dynamic_tree_overflow="chain",
        tree_node_budget_by_lanes={lanes: budget for lanes, budget in
                                   ((1, 15), (2, 7), (3, 4), (4, 3)) if lanes <= tree_width},
        target_tree_executor_provider=provider(monkeypatch, oracle),
    ) if tree else {}
    if not tree:
        monkeypatch.delenv("MLX2_DFLASH_TOPOLOGY", raising=False)
    batch = ExternalDraftBatchGenerator(
        model, draft_model=draft, binding="qwen38-27b-device-round-cpu",
        num_draft=int(draft.config.block_size) - 1, prefill_step_size=4,
        completion_batch_size=width, cost_per_commit_handoff=False, ready_drain="all",
        **route, **options,
    )
    rounds = []
    commit = batch._commit

    def recorded(cohort, decisions, features, *, blocks, **kwargs):
        result = commit(cohort, decisions, features, blocks=blocks, **kwargs)
        for lane, block, decision in zip(cohort, blocks, decisions, strict=True):
            rounds.append({
                "tokens": None if block is None else [int(t) for t in block.tokens],
                "parents": list(getattr(block, "parents", None) or []),
                "decision": _decision(decision),
                "draws": lane.rng.draws,
                "cache": _planes(lane.cache),
                "draft": _planes(lane.draft_cache),
            })
        return result

    batch._commit = recorded
    try:
        uids = batch.insert(
            [list(prompt) for prompt in prompts], max_tokens=[steps] * width,
            sampling_configs=[sampling or {"sampling_temp": 0}] * width,
            logits_processors=None if processors is None else [processors() for _ in prompts])
        out = {uid: [] for uid in uids}
        for _ in range(600):
            _, responses = batch.next()
            for response in responses:
                out[response.uid].append(response.token)
            if not batch.lanes:
                break
        else:
            raise AssertionError("scheduler stalled")
        return [out[uid] for uid in uids], rounds, dict(batch.scheduler_stats), oracle
    finally:
        batch.close()


def _assert_rounds_equal(host, device):
    assert len(host) == len(device)
    for left, right in zip(host, device, strict=True):
        assert left["tokens"] == right["tokens"]
        assert left["parents"] == right["parents"]
        assert left["decision"] == right["decision"]
        assert left["draws"] == right["draws"]
        for name in ("cache", "draft"):
            assert len(left[name]) == len(right[name])
            for a, b in zip(left[name], right[name], strict=True):
                assert a.dtype == b.dtype and a.shape == b.shape
                assert np.array_equal(a.view(np.uint8), b.view(np.uint8))


_TREE_KEYS = ("external_tree_rounds", "external_tree_nodes", "external_tree_accepted_edges",
              "external_tree_verified_nodes")


# -- gap 2: device token rows into the TensorFold forward ----------------------
def test_provider_declares_device_tree_inputs():
    from mlx2.adapters.qwen38_target_tree_executor import Qwen38TargetTreeExecutor

    executor = Qwen38TargetTreeExecutor.create()
    assert executor.device_tree_inputs is True
    assert executor.receipt()["device_tree_inputs"] == "b1_forward_device_row_embedding_v1"
    # A class constant, not a dataclass field (identity and equality unchanged).
    assert "device_tree_inputs" not in {field.name for field in dataclasses.fields(executor)}


def test_device_row_forward_equals_the_host_list_forward(monkeypatch):
    model = tiny()
    oracle = OracleLaneTree()
    executor = provider(monkeypatch, oracle)
    layers = (len(model.layers) - 1,)
    tokens, parents = [17, 4, 9, 30, 4], [-1, 0, 1, 0, 3]

    def run(inputs):
        cache = model.make_cache()
        mx.eval(model(mx.array([PROMPT]), cache=cache))
        logits, features, transaction = executor.forward(
            model, inputs, parents, cache, layers, cached=True)
        mx.eval(logits, features)
        transaction.commit_paths([[0, 3, 4]])
        return np.asarray(logits), np.asarray(features), _planes(cache)

    host = run(tokens)
    device = run(mx.array(tokens, dtype=mx.int32))
    assert oracle.embedded == ["Embedding", "_DeviceRowEmbedding"]
    for a, b in zip(host[:2], device[:2], strict=True):
        assert np.array_equal(a, b)
    for a, b in zip(host[2], device[2], strict=True):
        assert np.array_equal(a, b)
    # The shadow is gone after the forward: the trunk embeds host ids again.
    assert type(model.model.embed_tokens).__name__ == "Embedding"
    assert "embed_tokens" not in vars(model.model)


def test_device_row_must_match_the_tree():
    from mlx2.runtime import qwen38_tensorfold

    model = tiny()
    with pytest.raises(ValueError, match="device token row and parents differ"):
        qwen38_tensorfold._forward(OracleLaneTree(), model, mx.array([1, 2], dtype=mx.int32),
                                   [-1, 0, 1], model.make_cache(), (3,), cached=False)

    class Skips(OracleLaneTree):
        def tree_forward(self, core, head, tokens, parents, cache, start):
            return mx.zeros((1, len(tokens), 4)), {"core": core, "hidden": None}

    cache = model.make_cache()
    mx.eval(model(mx.array([PROMPT]), cache=cache))
    with pytest.raises(RuntimeError, match="did not embed the device token row"):
        qwen38_tensorfold._forward(Skips(), model, mx.array([1, 2], dtype=mx.int32),
                                   [-1, 0], cache, (3,), cached=False)
    assert "embed_tokens" not in vars(model.model)

    class Twice(OracleLaneTree):
        def tree_forward(self, core, head, tokens, parents, cache, start):
            core.embed_tokens(mx.array([tokens + [0]], dtype=mx.uint32))
            raise AssertionError("unreachable")

    with pytest.raises(RuntimeError, match="not the device token row's shape"):
        qwen38_tensorfold._forward(Twice(), model, mx.array([1, 2], dtype=mx.int32),
                                   [-1, 0], cache, (3,), cached=False)
    assert "embed_tokens" not in vars(model.model)


# -- the tree round engages with the native MTP drafter at B1 --------------------
@pytest.mark.parametrize("policy", [LAZY, DECIDE], ids=["tree_lazy", "tree_decide"])
@pytest.mark.parametrize("mixing", [0.2, 0.5])
def test_mtp_tree_rounds_run_on_the_device_and_equal_the_host_round(monkeypatch, policy, mixing):
    def build():
        return tiny(mixing=mixing)

    host = decode(monkeypatch, model=build())
    device = decode(monkeypatch, model=build(), device_round=policy)
    assert host[0] == [ordinary_greedy(build(), PROMPT, 24)]
    assert device[0] == host[0]
    _assert_rounds_equal(host[1], device[1])
    for key in _TREE_KEYS:
        assert device[2].get(key) == host[2].get(key), key
    stats = device[2]
    rounds = host[2]["external_tree_rounds"]
    assert rounds > 0
    assert stats["external_device_tree_rounds"] == rounds
    assert stats["external_device_tree_lazy_lanes"] == rounds
    assert stats["external_device_tree_host_reads"] == rounds
    assert stats["external_device_tree_declines"] == 0 and stats[DECLINES] == {}
    if policy is DECIDE:
        assert stats["external_device_tree_decide_lanes"] == rounds
        assert stats["external_device_tree_host_decide_lanes"] == 0
    # Every device round embedded its device token row in the provider.
    assert device[3].embedded.count("_DeviceRowEmbedding") == rounds
    assert host[3].embedded.count("_DeviceRowEmbedding") == 0
    # Multi-row commits and root rejects both occurred.
    lengths = {len(item["decision"]["commit_rows"] or [0]) for item in host[1] if item["parents"]}
    assert 1 in lengths and max(lengths) >= 3


# -- declines stay fail-closed with reasons ----------------------------------------
def _presence():
    from mlx2.runtime.sample_utils import make_logits_processors

    return make_logits_processors(presence_penalty=1.5, presence_context_size=20,
                                  penalty_generation_start=len(PROMPT))


def test_processor_lanes_decline_and_serve_the_host_round(monkeypatch):
    host = decode(monkeypatch, processors=_presence)
    device = decode(monkeypatch, processors=_presence, device_round=LAZY)
    assert device[0] == host[0]
    stats = device[2]
    assert host[2]["external_tree_rounds"] > 0
    assert stats["external_device_tree_rounds"] == 0
    assert stats[DECLINES] == {"processors": host[2]["external_tree_rounds"]}


def test_width_above_one_declines_and_serves_the_host_round(monkeypatch):
    prompts = (PROMPT, [7, 1, 30, 4, 4, 22, 13])
    host = decode(monkeypatch, prompts=prompts, steps=12)
    device = decode(monkeypatch, prompts=prompts, steps=12, device_round=LAZY)
    assert device[0] == host[0]
    stats = device[2]
    # Two-lane tree rounds decline (lane-rounds counted); rounds the lanes
    # happened to run alone (staggered prefill, one lane finished) engage.
    assert set(stats[DECLINES]) == {"width"} and stats[DECLINES]["width"] > 0
    assert stats["external_device_tree_declines"] == stats[DECLINES]["width"]


def _dflash2(model):
    from mlx2.runtime.drafters.dflash2 import DFlash2DraftModel
    from mlx2.runtime.drafters.dflash2_config import DFlash2Config

    model.select_external_feature_source(None)
    mx.random.seed(5)
    return DFlash2DraftModel(DFlash2Config(
        hidden_size=32, intermediate_size=24, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, vocab_size=48, num_target_layers=4,
        target_layer_ids=[0, 3], conv_kernel_size=2, conv_group_size=2, selector_rank=4,
        selector_top_k=4, block_size=5, mask_token_id=47, max_position_embeddings=512,
        sliding_window=8, layer_types=["sliding_attention"] * 2)).bind(model)


CHAIN = {"lazy_draft": True, "decide": True}


@pytest.mark.parametrize("drafter", ["mtp", "dflash2"])
def test_chain_device_rounds_equal_the_host_chain_at_b1_greedy(monkeypatch, drafter):
    """``"batch_size_route": null``: the linear chain, drafted lazily and decided on the device."""

    def build():
        model = tiny()
        return model, (_dflash2(model) if drafter == "dflash2" else None)

    model, draft = build()
    host = decode(monkeypatch, model=model, draft=draft, tree=False)
    model, draft = build()
    device = decode(monkeypatch, model=model, draft=draft, tree=False, device_round=CHAIN)
    assert device[0] == host[0]
    _assert_rounds_equal(host[1], device[1])
    stats = device[2]
    assert host[2].get("external_tree_rounds", 0) == 0
    assert stats["external_device_round_rounds"] > 0
    assert stats["external_device_round_lazy_draft_lanes"] == stats["external_device_round_rounds"]
    # Every round, the drafted ones and any served by a host proposal, is
    # decided on the device.
    assert stats["external_device_round_decide_greedy_lanes"] >= stats["external_device_round_rounds"]
    assert stats["external_device_round_declines"] == 0
    if drafter == "mtp":
        assert device[0] == [ordinary_greedy(tiny(), PROMPT, 24)]
        assert max(item["decision"]["accepted"] for item in device[1]) >= 2


def test_sampled_dflash2_lazy_drafts_decline_with_their_reason(monkeypatch):
    """A sampled lane keeps the host draft (``lazy_draft_sampled_lane``).

    Tokens are not compared: a round the device still decides draws its
    point-mass rows on the device (the same law as the host verifier, other
    tokens for a seed; speculative_device_round ``decide``)."""
    sampling = {"sampling_temp": 0.8, "seed": 3}
    model = tiny()
    device = decode(monkeypatch, model=model, draft=_dflash2(model), tree=False, steps=16,
                    sampling=sampling, device_round={**CHAIN, "sampled_drafts": True})
    assert len(device[0][0]) == 16
    stats = device[2]
    assert stats["external_device_round_rounds"] == 0
    assert stats["external_device_round_decline_reasons"].get("lazy_draft_sampled_lane", 0) > 0


def _dflash2_taps_oracle():
    class Taps(OracleLaneTree):
        """Post-block taps of layers 0 and 3 (the DFlash2 feature)."""

        def tree_forward(self, core, head, tokens, parents, cache, start):
            hidden = core.embed_tokens(mx.array([tokens], dtype=mx.uint32))
            logits, taps = [], []
            for path in _paths(parents):
                sink = []
                out = core(None, _private(cache), input_embeddings=hidden[:, mx.array(path)],
                           capture_layers=frozenset((0, 3)), hidden_sink=sink)
                logits.append(head(out[:, -1:]))
                taps.append(mx.concatenate(sink, axis=-1)[:, -1:])
            joined = mx.concatenate(taps, axis=1)
            for layer in core.layers:
                storage = getattr(layer, "_storage", None)
                if storage is not None:
                    storage[layer._idx] = joined[..., : joined.shape[-1] // 2] \
                        if layer._idx == 0 else joined[..., joined.shape[-1] // 2:]
            return mx.concatenate(logits, axis=1), {"core": core, "hidden": hidden}

    return Taps()


@pytest.mark.parametrize("policy", [{"tree_lazy": True}, {"tree_lazy": True, "tree_decide": True}])
def test_dflash2_tree_runs_on_the_device_tree_round(monkeypatch, policy):
    """The 27B best route (DFlash2 tree15 at B1): with gap 3's draft_tree_lazy and
    gap 2's device token rows, the device tree round engages and keeps the host tokens."""
    model = tiny()
    host = decode(monkeypatch, model=model, draft=_dflash2(model), steps=12,
                  oracle=_dflash2_taps_oracle())
    model = tiny()
    device = decode(monkeypatch, model=model, draft=_dflash2(model), steps=12,
                    oracle=_dflash2_taps_oracle(), device_round=policy)
    assert device[0] == host[0]
    rounds = host[2]["external_tree_rounds"]
    assert rounds > 0
    assert device[2]["external_device_tree_rounds"] > 0
    assert not device[2][DECLINES]


# -- gap 1: the adapter accepts, forwards and receipts the options -----------------
def _bare_adapter(monkeypatch, policy, adapter_type=None):
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter

    monkeypatch.delenv("MLX2_DFLASH_TOPOLOGY", raising=False)
    adapter_type = adapter_type or Qwen3827BAdapter
    adapter = object.__new__(adapter_type)
    adapter.model = object()
    adapter.draft_model = object()
    adapter.identity = {"fingerprint": "test"}
    adapter.external_policy = dict(policy)
    monkeypatch.setattr(adapter_type, "_initialize_external_feedback", lambda self: None)
    monkeypatch.setattr(adapter_type, "_external_num_draft", lambda self: 7)
    calls = []
    monkeypatch.setattr(
        "mlx2.runtime.external_speculative.ExternalDraftBatchGenerator",
        lambda *args, **kwargs: calls.append(kwargs) or object(),
    )
    return adapter, calls


def test_create_external_batch_forwards_device_round_and_receipts_it(monkeypatch):
    chain = {"lazy_draft": True, "decide": True}
    adapter, calls = _bare_adapter(monkeypatch, {
        "device_round": chain, "prompt_copy_round": {"max_tokens": 12},
    })
    adapter.create_external_batch()
    assert calls[-1]["device_round"] == chain
    assert calls[-1]["prompt_copy_round"] == {"max_tokens": 12}
    config = adapter.execution_config(max_lanes=1, prefill_step=16)
    receipt = config["device_round"]
    assert receipt["policy"]["lazy_draft"] and receipt["policy"]["decide"]
    assert not receipt["policy"]["tree_lazy"]
    assert receipt["prompt_copy_round"]["max_tokens"] == 12
    assert receipt["qualified"] is False and receipt["selected"] is True
    # device_commit was not asked for.
    assert "device_commit" not in receipt

    tree = dict(DECIDE)
    adapter, calls = _bare_adapter(monkeypatch, {
        "device_round": tree, "batch_size_route": "tree15_b1_b4_chain_b5plus_v1",
    })
    monkeypatch.setattr("mlx2.adapters.tensorfold_dtype_gate.model_tree_tile_dtype_issue",
                        lambda *args, **kwargs: None)
    adapter.create_external_batch()
    assert calls[-1]["device_round"] == tree
    assert calls[-1]["target_tree_executor_provider"].device_tree_inputs is True
    from mlx2.adapters.qwen38_27b import _device_round_receipt

    receipt = _device_round_receipt(adapter.external_policy)
    assert receipt["policy"]["tree_decide"] is True
    committed = _device_round_receipt({"device_round": {**CHAIN, "device_commit": True}})
    assert committed["device_commit"] == "device-gdn-rollback-serial"


def test_device_commit_rolls_gdn_state_back_on_the_device(monkeypatch):
    """Device commit (DFlash2-only, speculative_device_commit.drafter) engages on
    Qwen3.8 now that its GDN layers stage a device rollback, and keeps the host tokens."""
    model = tiny()
    host = decode(monkeypatch, model=model, draft=_dflash2(model), tree=False)
    model = tiny()
    device = decode(monkeypatch, model=model, draft=_dflash2(model), tree=False,
                    device_round={**CHAIN, "device_commit": True})
    assert device[0] == host[0]
    stats = device[2]
    assert stats["external_device_round_device_commit_rounds"] > 0
    assert "recurrent_rollback" not in stats["external_device_round_device_commit_skips"]


def test_device_commit_with_the_mtp_drafter_commits_on_the_host(monkeypatch):
    """Device commit serves the DFlash2 drafter only; the MTP chain keeps host commits."""
    device = decode(monkeypatch, tree=False, device_round={**CHAIN, "device_commit": True})
    assert device[0] == decode(monkeypatch, tree=False)[0]
    stats = device[2]
    assert stats["external_device_round_device_commit_rounds"] == 0
    assert stats["external_device_round_device_commit_skips"].get("drafter", 0) > 0


def test_inert_or_absent_options_are_not_forwarded_or_receipted(monkeypatch):
    for policy in ({}, {"device_round": False}, {"device_round": {"tree_lazy": False}}):
        adapter, calls = _bare_adapter(monkeypatch, policy)
        adapter.create_external_batch()
        if policy.get("device_round") is False or not policy:
            assert "device_round" not in calls[-1]
        assert "device_round" not in adapter.execution_config(max_lanes=1, prefill_step=16)


def test_split_validates_with_the_shared_parser_and_gates_the_descriptor():
    from mlx2.adapters.qwen36_27b import Qwen3627BAdapter
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter, _split_device_round_policy

    adapter = object.__new__(Qwen3827BAdapter)
    policy = {"draft_model": "/m/draft", "device_round": LAZY, "num_draft": 5}
    assert _split_device_round_policy(adapter, policy) == {"device_round": LAZY}
    assert policy == {"draft_model": "/m/draft", "num_draft": 5}
    assert _split_device_round_policy(adapter, {"draft_model": "x"}) == {}
    with pytest.raises(ValueError, match="unknown device_round switches"):
        _split_device_round_policy(adapter, {"draft_model": "x", "device_round": {"bogus": True}})
    with pytest.raises(ValueError, match="tree_decide requires tree_lazy"):
        _split_device_round_policy(adapter, {"draft_model": "x", "device_round": {"tree_decide": True}})
    with pytest.raises(ValueError, match="must be a boolean"):
        _split_device_round_policy(adapter, {"draft_model": "x", "device_round": {"tree_lazy": 1}})
    with pytest.raises(ValueError, match="prompt_copy_round"):
        _split_device_round_policy(adapter, {"draft_model": "x", "prompt_copy_round": {"bogus": 1}})
    with pytest.raises(ValueError, match="device_commit is refused beside prompt_copy_round"):
        _split_device_round_policy(adapter, {
            "draft_model": "x", "prompt_copy_round": {"max_tokens": 12},
            "device_round": {"lazy_draft": True, "decide": True, "device_commit": True}})
    with pytest.raises(ValueError, match="require an external draft_model"):
        _split_device_round_policy(adapter, {"device_round": LAZY})
    # The Qwen3.6-27B subclass reuses the load path but not the declaration.
    with pytest.raises(ValueError, match="Qwen3627BAdapter has no checked device_round route"):
        _split_device_round_policy(object.__new__(Qwen3627BAdapter),
                                   {"draft_model": "x", "device_round": LAZY})


class _Stop(Exception):
    pass


def _load_until_artifact(monkeypatch, path, policy, adapter_type=None):
    """Run the adapter's load path up to the artifact read (no tensors)."""
    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter

    monkeypatch.delenv("MLX2_DFLASH_TOPOLOGY", raising=False)
    adapter_type = adapter_type or Qwen3827BAdapter

    def stop(_path):
        raise _Stop

    monkeypatch.setattr(adapter_type, "artifact_inspector", staticmethod(stop))
    adapter = object.__new__(adapter_type)
    with pytest.raises(_Stop):
        adapter._init_qwen38(str(path), execution_policy=policy)
    return adapter


def test_native_mtp_tree_route_accepts_and_binds_the_options(monkeypatch, tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    for drafter in (str(target), "prompt_lookup"):
        adapter = _load_until_artifact(monkeypatch, target, {
            "draft_model": drafter, "device_round": DECIDE,
            "prompt_copy_round": {"max_tokens": 12},
        })
        assert adapter.external_policy["device_round"] == DECIDE
        assert adapter.external_policy["prompt_copy_round"] == {"max_tokens": 12}
        assert adapter.external_policy["batch_size_route"] == "tree15_b1_b4_chain_b5plus_v1"


def test_dflash2_route_accepts_and_binds_the_options(monkeypatch, tmp_path):
    from test_qwen38_dflash2_adapter import _pins, _write_artifacts

    target, draft = _write_artifacts(tmp_path)
    chain = {"lazy_draft": True, "decide": True, "processors": True}
    adapter = _load_until_artifact(monkeypatch, target, {
        **_pins(target, draft), "num_draft": 5, "device_round": chain,
    })
    assert adapter.external_policy["device_round"] == chain
    # Before this change the route's own allowlist refused the key.
    from mlx2.adapters.qwen38_27b import EXTERNAL_POLICY_KEYS, NATIVE_MTP_TREE_POLICY_KEYS

    assert "device_round" not in EXTERNAL_POLICY_KEYS | NATIVE_MTP_TREE_POLICY_KEYS


def test_qwen36_27b_refuses_the_options_before_any_artifact_read(monkeypatch, tmp_path):
    from mlx2.adapters.qwen36_27b import Qwen3627BAdapter

    adapter = object.__new__(Qwen3627BAdapter)
    with pytest.raises(ValueError, match="no checked device_round route"):
        adapter._init_qwen38(str(tmp_path), execution_policy={
            "draft_model": str(tmp_path), "device_round": LAZY})


def test_counters_installed_by_the_forwarded_policy(monkeypatch):
    """The forwarded policy reaches the engine: tree counters are installed."""
    _out, _rounds, stats, _oracle = decode(monkeypatch, steps=2, device_round=LAZY)
    assert all(key in stats for key in COUNTERS)


# -- DC-5 through the adapter's own binding (Model.device_commit_kv_device_position) ---------
DC4_LAGGED = {"lazy_draft": True, "decide": True, "device_commit": True, "lagged_width": True}
DC5 = {**DC4_LAGGED, "pipeline": True}
DC5_COMPOSED = {**DC5, "lagged_composition": True}
DC5_PROMPT = [3, 14, 15, 9, 26, 5, 35, 8]


def _adapter_chain(monkeypatch, device_round=None, **extra):
    """Bind the tiny DFlash2 pair the way the 27B adapter does (chain route,
    ``batch_size_route: null``, the route's default proposal composition) and
    decode 40 greedy tokens through its own ``create_external_batch``."""
    from test_qwen38_dflash2_cpu import tiny_pair

    from mlx2.adapters.qwen38_27b import Qwen3827BAdapter

    monkeypatch.delenv("MLX2_DFLASH_TOPOLOGY", raising=False)
    target, draft = tiny_pair()
    adapter = object.__new__(Qwen3827BAdapter)
    adapter.model, adapter.layout = target, "qwen38-27b-dc5-cpu"
    adapter.identity = {"fingerprint": "f" * 64}
    adapter.external_policy = {"draft_model": "/m/dflash2", "num_draft": 3, "batch_size_route": None,
                               **extra, **({"device_round": device_round} if device_round is not None else {})}
    monkeypatch.setattr(Qwen3827BAdapter, "_initialize_external_feedback", lambda self: None)
    adapter._bind_external_drafter({"fingerprint": "d" * 64}, lambda _record, _model: draft,
                                   adapter.descriptor)
    engine = adapter.create_external_batch(prefill_step_size=4, completion_batch_size=1)
    # The adapter hands the engine the runtime Model itself: the declaration is read there.
    assert engine.model is target and target.device_commit_kv_device_position is True
    try:
        uid = engine.insert([list(DC5_PROMPT)], max_tokens=[40],
                            sampling_configs=[{"sampling_temp": 0.0}])[0]
        out = []
        for _ in range(400):
            _, responses = engine.next()
            out.extend(r.token for r in responses if r.uid == uid)
            if not engine.lanes:
                break
        return out, dict(engine.scheduler_stats), adapter
    finally:
        engine.close()


@pytest.mark.parametrize(("policy", "extra"), [
    (DC5, {"proposal_composition": False}),
    (DC5_COMPOSED, {}),
], ids=["dc5-plain-dflash2", "dc5-lagged-composition"])
def test_dc5_pipeline_engages_through_the_adapter_and_equals_dc4(monkeypatch, policy, extra):
    from test_qwen38_dflash2_cpu import ordinary_greedy, tiny_target

    host, _, _ = _adapter_chain(monkeypatch, False, **extra)
    dc4, dc4_stats, _ = _adapter_chain(monkeypatch, DC4_LAGGED, **extra)
    dc5, stats, adapter = _adapter_chain(monkeypatch, policy, **extra)
    assert dc5 == dc4 == host == ordinary_greedy(tiny_target(), DC5_PROMPT, 40)
    assert dc4_stats["external_device_round_device_commit_rounds"] > 0
    assert stats["external_device_round_device_commit_rounds"] > 0
    assert stats["external_device_round_pipeline_rounds"] > 0
    assert stats["external_device_round_pipeline_followers"] > 0
    skips = stats["external_device_round_pipeline_skips"]
    assert "position_encoding" not in skips and "composition" not in skips
    assert "recurrent_rollback" not in stats["external_device_round_device_commit_skips"]
    receipt = adapter.execution_config(max_lanes=1, prefill_step=4)["device_round"]
    assert receipt["pipeline"].startswith("device-rope-position")


def test_dc5_under_the_default_composition_needs_lagged_composition(monkeypatch):
    """The chain route composes proposals by default: a plain pipeline keeps the
    serial DC-4 round there, counted under ``composition`` (fail closed)."""
    dc4, _, _ = _adapter_chain(monkeypatch, DC4_LAGGED)
    dc5, stats, adapter = _adapter_chain(monkeypatch, DC5)
    assert type(adapter.draft_model).__name__ == "ComposedDraftModel"
    assert dc5 == dc4
    assert stats["external_device_round_pipeline_rounds"] == 0
    assert stats["external_device_round_pipeline_skips"].get("composition", 0) > 0


# -- tree rounds with logits processors (``processors`` on tree_lazy / tree_decide) ----------
def _penalties(kind, context=20):
    """A vendor-profile penalty (presence 1.5 counts generated tokens only)."""
    from mlx2.runtime.sample_utils import make_logits_processors

    def build():
        options = {"presence": dict(presence_penalty=1.5, presence_context_size=context),
                   "frequency": dict(frequency_penalty=0.7, frequency_context_size=context),
                   "repetition": dict(repetition_penalty=1.3, repetition_context_size=context),
                   "presence+repetition": dict(presence_penalty=1.5, presence_context_size=context,
                                               repetition_penalty=1.1,
                                               repetition_context_size=context)}[kind]
        return make_logits_processors(**options, penalty_generation_start=len(PROMPT))

    return build


def ordinary_processed(model, prompt, count, processors):
    """Plain greedy decode with the same processors (``_target_law``'s order)."""
    cache = model.make_cache()
    tokens, out = list(prompt), []
    for step in range(count):
        logits = model(mx.array([tokens if step == 0 else [tokens[-1]]]), cache=cache)
        value = logits[0, -1][None].astype(mx.float32)
        for processor in processors:
            value = processor(mx.array(tokens, dtype=mx.int32), value)
        token = int(mx.argmax(value[0]).item())
        tokens.append(token)
        out.append(token)
    return out


LAZY_P = {"tree_lazy": True, "processors": True}
DECIDE_P = {"tree_lazy": True, "tree_decide": True, "processors": True}


@pytest.mark.parametrize("policy", [LAZY_P, DECIDE_P], ids=["tree_lazy", "tree_decide"])
@pytest.mark.parametrize(("kind", "context"), [
    ("presence", 20), ("presence", 0), ("presence", 3),
    ("frequency", 20), ("repetition", 20), ("presence+repetition", 20),
])
def test_mtp_tree_rounds_with_penalties_engage_and_equal_host_and_ordinary(
        monkeypatch, policy, kind, context):
    processors = _penalties(kind, context)
    host = decode(monkeypatch, processors=processors)
    device = decode(monkeypatch, processors=processors, device_round=policy)
    assert host[0] == [ordinary_processed(tiny(), PROMPT, 24, processors())]
    assert device[0] == host[0]
    _assert_rounds_equal(host[1], device[1])
    stats = device[2]
    rounds = host[2]["external_tree_rounds"]
    assert rounds > 0
    assert stats["external_device_tree_rounds"] == rounds
    assert stats["external_device_tree_declines"] == 0 and stats[DECLINES] == {}
    if policy is DECIDE_P:
        assert stats["external_device_tree_decide_lanes"] == rounds
    # The penalty moved the decode (a no-op penalty would prove nothing); the
    # 3-token presence window happens not to on this prompt.
    if (kind, context) != ("presence", 3):
        assert host[0] != decode(monkeypatch)[0]


@pytest.mark.parametrize("policy", [LAZY_P, DECIDE_P], ids=["tree_lazy", "tree_decide"])
def test_dflash2_tree_rounds_with_presence_engage_and_equal_the_host(monkeypatch, policy):
    processors = _penalties("presence")
    model = tiny()
    host = decode(monkeypatch, model=model, draft=_dflash2(model), steps=12,
                  oracle=_dflash2_taps_oracle(), processors=processors)
    model = tiny()
    device = decode(monkeypatch, model=model, draft=_dflash2(model), steps=12,
                    oracle=_dflash2_taps_oracle(), processors=processors, device_round=policy)
    assert device[0] == host[0]
    reference = tiny()
    reference.select_external_feature_source(None)
    assert host[0] == [ordinary_processed(reference, PROMPT, 12, processors())]
    assert host[2]["external_tree_rounds"] > 0
    assert device[2]["external_device_tree_rounds"] > 0
    assert "processors" not in device[2][DECLINES]


def test_tree_processors_need_the_switch_and_a_stateless_processor(monkeypatch):
    # Without the switch: the host tree round, counted (as before).
    host = decode(monkeypatch, processors=_penalties("presence"))
    off = decode(monkeypatch, processors=_penalties("presence"), device_round=LAZY)
    assert off[0] == host[0]
    assert off[2][DECLINES] == {"processors": host[2]["external_tree_rounds"]}

    # A processor that does not declare itself stateless keeps declining.
    def opaque():
        def processor(tokens, logits):
            return logits
        return [processor]

    gated = decode(monkeypatch, processors=opaque, device_round=DECIDE_P)
    assert gated[2]["external_device_tree_rounds"] == 0
    assert set(gated[2][DECLINES]) == {"processors"}
    with pytest.raises(ValueError, match="processors requires lazy_draft, decide or tree_lazy"):
        from mlx2.runtime.speculative_device_round import DeviceRoundPolicy

        DeviceRoundPolicy.from_value({"processors": True})


@pytest.mark.parametrize("context", [0, 1, 3, 20])
@pytest.mark.parametrize("start", [None, 0, 5, 12, 14])
def test_batched_presence_rows_equal_the_host_launch_bitwise(context, start):
    """``_batched_processed_rows`` (path tokens on the device) equals
    ``_launch_tree_laws``' presence and forbidden-length masks on random trees."""
    from types import SimpleNamespace

    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator, TreeDraftRow
    from mlx2.runtime.sample_utils import make_presence_penalty
    from mlx2.runtime.speculative_device_tree import _batched_processed_rows, geometry
    from mlx2.serving import minimum_tokens_processor

    rng = np.random.default_rng(context * 31 + (start or 0))
    owner = object.__new__(ExternalDraftBatchGenerator)
    owner.mx = mx
    owner.tree_gates = {"logprobs_on_request": False}
    vocab = 11
    for trial in range(40):
        nodes = int(rng.integers(1, 15))
        parents = [-1] + [int(rng.integers(-1, node)) for node in range(1, nodes)]
        tokens = [int(rng.integers(0, vocab)) for _ in range(nodes)]
        history = [int(rng.integers(0, vocab)) for _ in range(int(rng.integers(0, 18)))]
        processors = [make_presence_penalty(1.5, context, generation_start=start)]
        if trial % 3 == 0:
            # EOS forbidden until three generated tokens past the history.
            processors.insert(trial % 2, minimum_tokens_processor(
                mx, [int(rng.integers(0, vocab))], len(history), 3))
        lane = SimpleNamespace(history=history, anchor=int(rng.integers(0, vocab)),
                               processors=processors, sampling={"sampling_temp": 0})
        block = TreeDraftRow(list(tokens), list(parents), 1)
        geo = geometry(tuple(parents))
        logits = mx.array(rng.normal(size=(1, geo.rows, vocab)).astype(np.float32))
        _fence, state = owner._launch_tree_laws(lane, block, logits, logprobs_on_request=False)
        inputs = mx.array([lane.anchor, *tokens], dtype=mx.int32)
        device = _batched_processed_rows(mx, lane, geo, inputs,
                                         logits[0, :geo.rows].astype(mx.float32)).astype(mx.float32)
        assert np.array_equal(np.asarray(device).view(np.uint32),
                              np.asarray(state["values"]).view(np.uint32)), (trial, parents)


@pytest.mark.parametrize("kind", ["presence", "frequency", "repetition", "presence+repetition"])
@pytest.mark.parametrize("context", [0, 2, 20])
def test_per_row_processed_values_equal_the_host_target_law_bitwise(kind, context):
    """``tree_decide``'s per-row form (``_target_law``'s processor calls on each
    row's own root-to-node history, drafts on the device) equals the host
    ``_target_law(..., history_suffix=path inputs)`` row for row, bitwise."""
    from types import SimpleNamespace

    from mlx2.runtime.external_speculative import ExternalDraftBatchGenerator, TreeDraftRow
    from mlx2.runtime.speculative_device_tree import DeviceTree, geometry

    rng = np.random.default_rng(context * 7 + len(kind))
    owner = object.__new__(ExternalDraftBatchGenerator)
    owner.mx = mx
    # Per-row form: the batched-law gate off, as the reference route serves it.
    owner.tree_gates = {"batched_laws": False, "logprobs_on_request": False}
    runner = DeviceTree(SimpleNamespace(policy=None))
    vocab = 13
    for trial in range(30):
        nodes = int(rng.integers(1, 15))
        parents = [-1] + [int(rng.integers(-1, node)) for node in range(1, nodes)]
        tokens = [int(rng.integers(0, vocab)) for _ in range(nodes)]
        history = [int(rng.integers(0, vocab)) for _ in range(int(rng.integers(1, 18)))]
        processors = _penalties(kind, context)()
        lane = SimpleNamespace(uid=1, history=history, anchor=int(rng.integers(0, vocab)),
                               processors=processors,
                               sampling={"sampling_temp": 0, "emit_logprobs": True})
        geo = geometry(tuple(parents))
        logits = mx.array(rng.normal(size=(1, geo.rows, vocab)).astype(np.float32))
        plan = runner._plan(owner, lane, TreeDraftRow(mx.array(tokens, dtype=mx.int32),
                                                      list(parents), 1), logits)
        assert not plan.batched
        inputs = [lane.anchor, *tokens]
        for row, path in enumerate(geo.paths):
            response = []
            owner._target_law(lane, logits[0, row], history, True, response,
                              history_suffix=[inputs[index] for index in path])
            device = plan.values[row] - plan.norms[row]
            assert np.array_equal(np.asarray(device).view(np.uint32),
                                  np.asarray(response[0]).view(np.uint32)), (trial, row, path)


@pytest.mark.parametrize("decide", [False, True], ids=["tree_lazy", "tree_decide"])
def test_a_declined_device_tree_round_is_the_switch_off_round(monkeypatch, decide):
    """r1 anomaly guard: a round the device tree declines (processors without
    the switch) is the switch-off round itself: same tree shapes (tile budget
    rows and depth), decisions, RNG, caches and tree counters under the
    deterministic controller.  r1's MAL/tok/s gap between all-declined arms
    and host came from the measured (wall-clock EWMA) tile-budget chooser
    picking 3- or 5-deep trees per regime, not from the decline path; tokens
    were identical on all 30 questions."""
    processors = _penalties("presence")
    host = decode(monkeypatch, processors=processors)
    declined = decode(monkeypatch, processors=processors,
                      device_round=DECIDE if decide else LAZY)
    assert declined[2][DECLINES] == {"processors": host[2]["external_tree_rounds"]}
    assert declined[0] == host[0]
    _assert_rounds_equal(host[1], declined[1])
    for key in (*_TREE_KEYS, "external_tree_tile_budget"):
        assert declined[2].get(key) == host[2].get(key), key


# -- the DFlash2 device_round default (Pierre, 2026-11-27) ---------------------------------
def test_dflash2_routes_get_the_device_round_default_through_the_shared_binding(monkeypatch):
    from mlx2.adapters.external_draft_policy import DFLASH2_DEVICE_ROUND_DEFAULT
    from test_qwen38_dflash2_cpu import ordinary_greedy, tiny_target

    monkeypatch.delenv("MLX2_DFLASH2_DEVICE_ROUND_DEFAULT", raising=False)
    default, stats, adapter = _adapter_chain(monkeypatch)
    host, host_stats, _ = _adapter_chain(monkeypatch, False)
    assert default == host == ordinary_greedy(tiny_target(), DC5_PROMPT, 40)
    # Bound at the head (DFlash2), before the default composition wrapper.
    assert adapter._external_dflash2_drafter is True
    assert type(adapter.draft_model).__name__ == "ComposedDraftModel"
    assert "device_round" not in adapter.external_policy  # the policy stays the operator's
    assert stats["external_device_round_rounds"] > 0
    assert "external_device_round_rounds" not in host_stats
    for key in ("device_commit_rounds", "pipeline_rounds", "fixed_shape_rounds"):
        assert f"external_device_round_{key}" not in stats
    receipt = adapter.execution_config(max_lanes=1, prefill_step=4)["device_round"]
    assert receipt["selected"] is True and receipt["selection"] == "dflash2_default"
    assert receipt["qualified"] is False
    assert {k for k, v in receipt["policy"].items() if v} == set(DFLASH2_DEVICE_ROUND_DEFAULT)


def test_the_dflash2_default_yields_to_policy_kill_switch_and_route_refusals(monkeypatch):
    from types import SimpleNamespace

    from mlx2.adapters.external_draft_policy import (
        DFLASH2_DEVICE_ROUND_DEFAULT,
        device_round_receipt,
        effective_device_round,
    )

    monkeypatch.delenv("MLX2_DFLASH2_DEVICE_ROUND_DEFAULT", raising=False)
    dflash2 = SimpleNamespace(external_policy={}, _external_dflash2_drafter=True)
    assert effective_device_round(dflash2) == (DFLASH2_DEVICE_ROUND_DEFAULT, "dflash2_default")
    # An explicit policy always wins, ``false`` included (the host rounds).
    off = SimpleNamespace(external_policy={"device_round": False}, _external_dflash2_drafter=True)
    assert effective_device_round(off) == (False, "execution_policy")
    # Off emits nothing: the route keeps its earlier settings and identity.
    assert device_round_receipt(off) is None
    chosen = {"lazy_draft": True, "decide": True, "device_commit": True}
    explicit = SimpleNamespace(external_policy={"device_round": chosen}, _external_dflash2_drafter=True)
    assert effective_device_round(explicit) == (chosen, "execution_policy")
    # Other drafters get nothing.
    assert effective_device_round(SimpleNamespace(external_policy={})) == (None, None)
    assert device_round_receipt(SimpleNamespace(external_policy={})) is None
    # A route the adapter keeps on the host (``_device_round_default_refusal``).
    refused = SimpleNamespace(external_policy={}, _external_dflash2_drafter=True,
                              _device_round_default_refusal=lambda: "example_tree_tiles")
    assert effective_device_round(refused) == (None, None)
    assert device_round_receipt(refused) is None
    monkeypatch.setenv("MLX2_DFLASH2_DEVICE_ROUND_DEFAULT", "0")
    assert effective_device_round(dflash2) == (None, None)
    assert device_round_receipt(dflash2) is None


def test_the_27b_split_admits_the_opt_out_on_every_class():
    from mlx2.adapters.qwen36_27b import Qwen3627BAdapter
    from mlx2.adapters.qwen38_27b import _split_device_round_policy

    for value in (False, {}, {"tree_lazy": False}):
        policy = {"draft_model": "x", "device_round": value}
        assert _split_device_round_policy(object.__new__(Qwen3627BAdapter), policy) == {
            "device_round": value}
    with pytest.raises(ValueError, match="no checked device_round route"):
        _split_device_round_policy(object.__new__(Qwen3627BAdapter),
                                   {"draft_model": "x", "device_round": LAZY})


@pytest.mark.parametrize("value", [False, {}, {"lazy_draft": False}])
def test_the_serving_engine_takes_the_opt_out_before_any_adapter_allowlist(value):
    """``"device_round": false`` reaches adapters whose own key lists name no
    device_round: removed before construction, restored on the bound adapter."""
    from types import SimpleNamespace

    from mlx2.adapters.external_draft_policy import (
        effective_device_round,
        restore_device_round_opt_out,
        split_device_round_opt_out,
    )

    policy, opted_out = split_device_round_opt_out({"draft_model": "/m/d", "device_round": value})
    assert policy == {"draft_model": "/m/d"} and opted_out is True
    adapter = SimpleNamespace(draft_model=object(), external_policy={"draft_model": "/m/d"},
                              _external_dflash2_drafter=True)
    restore_device_round_opt_out(adapter)
    assert effective_device_round(adapter) == (False, "execution_policy")
    # A non-inert or invalid value stays for the adapter to accept or refuse.
    for kept in ({"tree_lazy": True}, {"bogus": True}, "yes"):
        assert split_device_round_opt_out({"device_round": kept}) == ({"device_round": kept}, False)
    assert split_device_round_opt_out(None) == (None, False)
    # An ordinary route (no bound drafter) is left untouched.
    ordinary = SimpleNamespace(draft_model=None, external_policy={})
    restore_device_round_opt_out(ordinary)
    assert ordinary.external_policy == {}
