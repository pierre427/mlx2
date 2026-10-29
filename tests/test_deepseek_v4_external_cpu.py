"""Exact target rollback probes, CPU only; no model-path qualification."""

from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx2.adapters.deepseek_v4_external import (
    DeepSeekV4DraftCache,
    DeepSeekV4DSparkDraftModel,
    DeepSeekV4VerifyRows,
)
from mlx2.runtime.drafters.deepseek_v4_dspark import (
    SIDECAR_SHA256,
    TARGET_SHA256,
    DeepSeekV4DSparkBackbone,
    DSparkContractError,
)


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


class Cache:
    pending = False

    def __init__(self):
        self.values = [3, 5]
        self.offset = 2
        self.carry = 8

    def snapshot(self):
        return tuple(self.values), self.offset, self.carry

    def restore(self, state):
        values, self.offset, self.carry = state
        self.values = list(values)


class Model:
    identity = "cpu-target"

    def __init__(self):
        self.calls = []
        self.fail = None

    def forward_with_taps(self, tokens, caches, layers):
        token = int(tokens.item())
        self.calls.append(token)
        for c in caches:
            c.values.append(token)
            c.offset += 1
            c.carry = (c.carry * 7 + token) % 19
        if token == self.fail:
            raise RuntimeError("injected mutation failure")
        return mx.array([[[float(caches[0].carry)]]]), mx.array([[[float(token)]]])


def test_rejection_restores_every_plane_then_replays_exact_prefix():
    model, c = Model(), Cache()
    tx = DeepSeekV4VerifyRows(model, [[c]], [1]).begin([3])
    logits, hidden = tx.forward(mx.array([[9, 11, 13]]))
    assert logits.shape == hidden.shape == (1, 3, 1)
    tx.commit([1])
    expected = Cache()
    Model().forward_with_taps(mx.array([[9]]), [expected], [1])
    assert c.snapshot() == expected.snapshot()
    assert model.calls == [9, 11, 13, 9]


def test_full_accept_avoids_replay_and_abort_restores():
    model, c = Model(), Cache()
    owner = DeepSeekV4VerifyRows(model, [[c]], [1])
    tx = owner.begin([2])
    tx.forward(mx.array([[9, 11]]))
    tx.commit([2])
    assert model.calls == [9, 11]
    saved = c.snapshot()
    tx = owner.begin([1])
    tx.forward(mx.array([[13]]))
    tx.abort()
    assert c.snapshot() == saved
    assert owner._active is None


def test_mutating_forward_failure_restores_and_closes():
    model, c = Model(), Cache()
    model.fail = 11
    saved = c.snapshot()
    tx = DeepSeekV4VerifyRows(model, [[c]], [1]).begin([2])
    with pytest.raises(RuntimeError, match="injected"):
        tx.forward(mx.array([[9, 11]]))
    assert c.snapshot() == saved and tx.closed


def test_mutating_replay_failure_restores_original_boundary():
    model, c = Model(), Cache()
    saved = c.snapshot()
    tx = DeepSeekV4VerifyRows(model, [[c]], [1]).begin([2])
    tx.forward(mx.array([[9, 11]]))
    model.fail = 9
    with pytest.raises(RuntimeError):
        tx.commit([1])
    assert c.snapshot() == saved and tx.closed


def test_wrong_geometry_and_b2_fail_closed():
    with pytest.raises(DSparkContractError):
        DeepSeekV4VerifyRows(Model(), [[Cache()], [Cache()]], [1])
    tx = DeepSeekV4VerifyRows(Model(), [[Cache()]], [1]).begin([2])
    with pytest.raises(DSparkContractError):
        tx.forward(mx.array([[1]]))
    tx.abort()


def test_support_cache_roundtrip_owns_values():
    c = DeepSeekV4DraftCache("verified-binding", 3)
    x = mx.array([[[1.0, 2.0]]])
    c.history.append(x, (x, x, x), 0)
    restored = DeepSeekV4DraftCache.from_state(c.state, c.meta_state)
    assert restored.offset == 1
    c.history.append(x * 4, (x, x, x), 1)
    assert restored.offset == 1
    assert restored.state[0].tolist() == [[[1.0, 2.0]]]


class Backbone(DeepSeekV4DSparkBackbone):
    identity = f"{TARGET_SHA256}:{SIDECAR_SHA256}:" + "a" * 64
    target_weights = object()
    target_config = SimpleNamespace(
        hidden_size=2,
        num_key_value_heads=1,
        num_hidden_layers=1,
        head_dim=2,
        intermediate_size=2,
        vocab_size=16,
    )

    def __init__(self):
        pass

    config = SimpleNamespace(
        target_layers=(1, 3),
        hidden_size=2,
        vocab_size=8,
        block_size=4,
        stages=3,
        window=128,
        head_dim=2,
        expert_hidden=2,
    )

    def record_features(self, captures, position, *, history):
        x = captures[1]
        history.append(x, (x, x, x), position)

    def propose(self, seed, **kwargs):
        assert kwargs["history"].next_position == kwargs["position"]
        return (seed + 1, seed + 2)[: kwargs["max_proposals"]], (0.1, 0.2)[
            : kwargs["max_proposals"]
        ]


def test_wrapper_request_isolation_and_pointmass_laws():
    wrapper = DeepSeekV4DSparkDraftModel(Backbone()).bind(
        SimpleNamespace(
            weights=Backbone.target_weights,
            args=Backbone.target_config,
            identity="cpu-target",
        )
    )
    a, b = wrapper.make_cache(), wrapper.make_cache()
    tokens, laws = wrapper.draft_distributions(
        [2], mx.ones((1, 1, 4)), a, 2, [None], [0.0]
    )
    assert tokens == [[3, 4]] and [x.token for x in laws[0]] == [3, 4]
    assert a[0].offset == 1 and b[0].offset == 0
    with pytest.raises(DSparkContractError, match="greedy"):
        wrapper.draft_distributions([2], mx.ones((1, 1, 4)), b, 2, [None], [1.0])
    assert b[0].offset == 0


def test_support_cache_actual_safetensors_roundtrip(tmp_path):
    from mlx2.runtime.models.cache import load_prompt_cache, save_prompt_cache

    c = DeepSeekV4DraftCache("full-binding", 3)
    x = mx.array([[[1.0, 2.0]]])
    c.history.append(x, (x, x, x), 0)
    path = str(tmp_path / "support.safetensors")
    save_prompt_cache(path, [c])
    restored = load_prompt_cache(path)[0]
    assert restored.meta_state == c.meta_state
    assert restored.state[0].tolist() == c.state[0].tolist()


def test_zero_proposal_appends_context_without_running_backbone():
    wrapper = DeepSeekV4DSparkDraftModel(Backbone()).bind(
        SimpleNamespace(
            weights=Backbone.target_weights,
            args=Backbone.target_config,
            identity="cpu-target",
        )
    )
    cache = wrapper.make_cache()
    tokens, laws = wrapper.draft_distributions(
        [2], mx.ones((1, 1, 4)), cache, 0, [None], [0.0]
    )
    assert tokens == laws == [[]] and cache[0].offset == 1


def test_generator_binds_owned_transaction_and_rejects_packed_ast():
    from mlx2.adapters.deepseek_v4_external import DeepSeekV4ExternalBatchGenerator

    model = Model()
    model.args = Backbone.target_config
    model.weights = Backbone.target_weights
    wrapper = DeepSeekV4DSparkDraftModel(Backbone()).bind(
        SimpleNamespace(
            weights=Backbone.target_weights,
            args=Backbone.target_config,
            identity="cpu-target",
        )
    )
    generator = DeepSeekV4ExternalBatchGenerator(
        model, draft_model=wrapper, num_draft=4
    )
    tx = generator._target_owner([[Cache()]]).begin([2])
    out, taps = generator._cohort_target_forward(
        mx.array([[9, 11]]), tx, [], site="test"
    )
    assert out.shape == taps.shape == (1, 2, 1)
    tx.abort()
    with pytest.raises(DSparkContractError, match="packed AST"):
        DeepSeekV4ExternalBatchGenerator(
            model, draft_model=wrapper, binding="test", ast_tree_tile_rows=16
        )


@pytest.mark.parametrize(
    "prompt,bad_draft", [([1, 2, 3], False), ([1], True), ([1, 2], "no-proposals")]
)
def test_generic_external_loop_matches_ordinary_tokens_and_sidecar(prompt, bad_draft):
    from mlx2.adapters.deepseek_v4_external import DeepSeekV4ExternalBatchGenerator
    from mlx2.runtime.models.deepseek_v4_cache import DeepSeekV4LayerCache

    class LoopBackbone(Backbone):
        config = SimpleNamespace(
            target_layers=(0,),
            hidden_size=2,
            vocab_size=16,
            block_size=4,
            stages=3,
            window=128,
            head_dim=2,
            expert_hidden=2,
        )

        def record_features(self, captures, position, *, history):
            x = captures[0]
            history.append(x, (x, x, x), position)

        def propose(self, seed, **kwargs):
            if bad_draft == "no-proposals":
                assert kwargs["confidence_threshold"] == 0.6
                return (), (-10.0,)
            shift = 2 if bad_draft else 1
            count = kwargs["max_proposals"]
            return tuple((seed + i + shift) % 16 for i in range(count)), (0.0,) * count

    class LoopModel:
        identity = "cpu-target"
        args, weights = Backbone.target_config, Backbone.target_weights

        def make_cache(self):
            return [
                DeepSeekV4LayerCache(
                    xp=mx, head_dim=2, index_head_dim=2, identity="cpu-fixture"
                )
            ]

        def forward_with_taps(self, inputs, cache, layers):
            cache[0].update_local(
                mx.broadcast_to(
                    inputs[:, None, :, None], (1, 1, inputs.shape[1], 2)
                ).astype(mx.float32)
            )
            logits = mx.full((1, inputs.shape[1], 16), -10.0)
            logits = mx.put_along_axis(
                logits,
                ((inputs + 1) % 16)[:, :, None],
                mx.full((1, inputs.shape[1], 1), 10.0),
                axis=-1,
            )
            return logits, mx.broadcast_to(
                inputs[:, :, None], (1, inputs.shape[1], 2)
            ).astype(mx.float32)

        def prefill_body(self, inputs, cache, layers):
            return self.forward_with_taps(inputs, cache, layers)[1]

        def __call__(self, inputs, cache):
            return self.forward_with_taps(inputs, cache, (0,))[0]

    generator = DeepSeekV4ExternalBatchGenerator(
        LoopModel(),
        draft_model=DeepSeekV4DSparkDraftModel(LoopBackbone()),
        num_draft=4,
        memory_headroom=lambda: 1 << 30,
    )
    generator.insert([prompt], max_tokens=[8])
    emitted, last = [], None
    for _ in range(20):
        _, rows = generator.next()
        for row in rows:
            emitted.append(row.token)
            last = row
        if not generator.lanes:
            break
    assert emitted == [(prompt[-1] + i + 1) % 16 for i in range(8)]
    assert last.finish_reason == "length"
    last.cache_sidecar.validate(generator.binding, len(prompt) + 7)
    assert last.prompt_cache[0].offset == len(prompt) + 7
    assert last.speculative_receipt["draft_settings"]["packed_ast"] is False


def test_support_cache_rejects_different_runtime_policy_binding():
    target = SimpleNamespace(
        weights=Backbone.target_weights,
        args=Backbone.target_config,
        identity="attention=reference",
    )
    first = DeepSeekV4DSparkDraftModel(Backbone()).bind(target)
    old_cache = first.make_cache()
    target.identity = "attention=visible_rows_fp32"
    second = DeepSeekV4DSparkDraftModel(Backbone()).bind(target)
    assert first.execution_binding != second.execution_binding
    with pytest.raises(DSparkContractError, match="binding"):
        second.append_context(mx.ones((1, 1, 4)), old_cache)
    with pytest.raises(DSparkContractError, match="change target policy"):
        first.bind(target)


def test_failed_restore_closes_transaction_owner():
    class BrokenCache(Cache):
        def restore(self, state):
            raise RuntimeError("restore failed")

    c = BrokenCache()
    owner = DeepSeekV4VerifyRows(Model(), [[c]], [1])
    tx = owner.begin([1])
    with pytest.raises(RuntimeError, match="restore failed"):
        tx.abort()
    assert tx.closed and owner._active is None
    with pytest.raises(DSparkContractError, match="poisoned"):
        owner.begin([1])


def test_confidence_identity_and_cache_binding_are_distinct():
    target = SimpleNamespace(
        weights=Backbone.target_weights,
        args=Backbone.target_config,
        identity="cpu-target",
    )
    zero = DeepSeekV4DSparkDraftModel(Backbone(), confidence_threshold=0).bind(target)
    default = DeepSeekV4DSparkDraftModel(Backbone()).bind(target)
    assert zero.execution_binding != default.execution_binding
    assert zero.make_cache()[0].meta_state != default.make_cache()[0].meta_state
    assert default.receipt_settings["confidence_threshold"] == 0.6
    assert zero.receipt_settings["confidence_threshold"] == 0.0
    # Earlier threshold-zero target identity is preserved exactly.
    import json

    assert json.loads(zero.target_identity) == ["cpu-target", None, None, "sequential"]


@pytest.mark.parametrize(
    "threshold", [True, None, "0.6", float("nan"), float("inf"), -1, 1.01]
)
def test_wrapper_confidence_rejects_invalid(threshold):
    with pytest.raises(DSparkContractError, match="confidence"):
        DeepSeekV4DSparkDraftModel(Backbone(), confidence_threshold=threshold)
