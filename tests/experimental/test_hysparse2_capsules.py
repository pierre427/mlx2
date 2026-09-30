"""Capsule-content coupling, persistence and exact revision contracts."""

import hashlib

import pytest

mx = pytest.importorskip("mlx.core")
from mlx import nn, optimizers
from mlx.utils import tree_flatten

from mlx2.experimental.hysparse2.capsule_memory import CapsuleMemory
from mlx2.experimental.hysparse2.config import Config
from mlx2.experimental.hysparse2.model import Model
from mlx2.experimental.hysparse2.train import load_checkpoint, loss, save_checkpoint
from mlx2.runtime.semantic_capsules import CapsuleStore
from mlx2.runtime.semantic_memory import SEMANTIC_SCHEMA


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.random.seed(42)
    yield
    mx.set_default_device(previous)


def snapshot(store, vocab_size, *, text="valid arguments", parent=None):
    evidence = hashlib.sha256(text.encode()).hexdigest()
    graph = {
        "schema": SEMANTIC_SCHEMA,
        "concepts": {"a": {"label": "tool call"}, "b": {"label": text}},
        "edges": [
            {
                "subject": "a",
                "relation": "has_property",
                "object": "b",
                "authority": "committed",
                "evidence_digest": evidence,
            }
        ],
        "proposals": [],
    }
    capsule = store.put(
        kind="semantic_delta" if parent else "semantic_base",
        data=graph,
        parents=(parent,) if parent else (),
        model_binding="checkpoint-fixture",
        tokenizer_binding="character-fixture",
        runtime_binding="hysparse2-capsule-v1",
        provenance={"source": "original synthetic contract fixture"},
    )
    return CapsuleMemory.from_store(
        store,
        capsule.digest,
        model_binding="checkpoint-fixture",
        tokenizer_binding="character-fixture",
        runtime_binding="hysparse2-capsule-v1",
        encode=lambda s: [ord(c) % vocab_size for c in s],
        vocab_size=vocab_size,
    )


def test_capsule_content_parity_and_revision(tmp_path):
    c = Config.smoke()
    model = Model(c)
    model.eval()
    store = CapsuleStore(tmp_path / "capsules")
    memory = snapshot(store, c.vocab_size)
    tokens = mx.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]])
    baseline = model(tokens)[0]
    mx.eval(baseline)
    model.attach_semantic_capsules(memory)
    full = model(tokens)[0]
    mx.eval(full)
    assert float(mx.max(mx.abs(full - baseline)).item()) > 0
    got, cache = model.prefill(tokens[:, :4])
    assert cache.apcv2_identity["semantic_fingerprint"][1] == memory.capsule_digest
    assert cache.apcv2_identity["capsule_read_fingerprint"] == memory.fingerprint
    assert float(mx.max(mx.abs(got - full[:, 3:4])).item()) < 1e-3
    got = model.decode(tokens[:, 4:5], cache)
    assert float(mx.max(mx.abs(got - full[:, 4:5])).item()) < 1e-3
    with pytest.raises(ValueError, match="requested capsule"):
        model.new_cache(semantic_capsule_digest="a" * 64)
    revised = snapshot(
        store, c.vocab_size, text="new evidence", parent=memory.capsule_digest
    )
    model.attach_semantic_capsules(revised)
    with pytest.raises(ValueError, match="another model"):
        model.decode(tokens[:, 5:6], cache)
    changed = model(tokens)[0]
    assert float(mx.max(mx.abs(changed - full)).item()) > 0
    model.attach_semantic_capsules(None)
    assert float(mx.max(mx.abs(model(tokens)[0] - baseline)).item()) == 0


def test_memory_gradient_and_exact_checkpoint_binding(tmp_path):
    c = Config.smoke()
    model = Model(c)
    memory = snapshot(CapsuleStore(tmp_path / "capsules"), c.vocab_size)
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    mx.random.seed(7)
    _, without = nn.value_and_grad(model, loss)(model, tokens)
    model.attach_semantic_capsules(memory)
    mx.random.seed(7)
    _, with_memory = nn.value_and_grad(model, loss)(model, tokens)
    a, b = dict(tree_flatten(without)), dict(tree_flatten(with_memory))
    for key in [
        "semantic_ple.embedding.weight",
        "semantic_ple.key.weight",
        "semantic_ple.value.weight",
        "self_decoder.0.attention.q.weight",
    ]:
        assert float(mx.max(mx.abs(a[key] - b[key])).item()) > 0
    optimizer = optimizers.Adam(1e-4)
    path = save_checkpoint(
        tmp_path / "checkpoint", model, optimizer, 0, {"fixture": True}
    )
    model.eval()
    expected = model(tokens)[0]
    fresh = Model(c)
    with pytest.raises(ValueError, match="capsule read binding"):
        load_checkpoint(path, fresh, optimizers.Adam(1e-4), {"fixture": True})
    fresh.attach_semantic_capsules(memory)
    load_checkpoint(path, fresh, optimizers.Adam(1e-4), {"fixture": True})
    fresh.eval()
    assert float(mx.max(mx.abs(fresh(tokens)[0] - expected)).item()) == 0


def test_wrong_bindings_and_token_bounds_fail_closed(tmp_path):
    c = Config.smoke()
    store = CapsuleStore(tmp_path)
    memory = snapshot(store, c.vocab_size)
    with pytest.raises(ValueError, match="bindings"):
        CapsuleMemory.from_store(
            store,
            memory.capsule_digest,
            model_binding="wrong",
            tokenizer_binding="character-fixture",
            runtime_binding="hysparse2-capsule-v1",
            encode=lambda _: [1],
            vocab_size=c.vocab_size,
        )

    with pytest.raises(ValueError, match="tokenization"):
        CapsuleMemory.from_store(
            store,
            memory.capsule_digest,
            model_binding="checkpoint-fixture",
            tokenizer_binding="character-fixture",
            runtime_binding="hysparse2-capsule-v1",
            encode=lambda _: [c.vocab_size],
            vocab_size=c.vocab_size,
        )


def test_memory_batch_apc_diffusion_and_adapter_composition(tmp_path):
    from dataclasses import replace

    from mlx2.experimental.hysparse2.apc import EndpointAPC
    from mlx2.experimental.hysparse2.batching import ResearchBatcher
    from mlx2.experimental.hysparse2.lora import LoRAEpisode
    from mlx2.runtime.apc_v2 import APCv2

    c = replace(
        Config.smoke(),
        diffusion_conditioning="prefix",
        diffusion_position_encoding="sinusoidal",
    )
    model = Model(c)
    model.eval()
    store = CapsuleStore(tmp_path / "capsules")
    a = snapshot(store, c.vocab_size)
    b = snapshot(store, c.vocab_size, text="revised arguments", parent=a.capsule_digest)
    model.attach_semantic_capsules(a)
    batcher = ResearchBatcher(model)
    prompts = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10]]
    _, caches, _ = batcher.prefill(prompts)
    engine = APCv2(max_size=4, layout_name="hysparse2-endpoint-v1")
    episode = None
    leases = []
    try:
        bridge = EndpointAPC(
            model,
            engine,
            checkpoint_revision="fixture",
            tokenizer_fingerprint="character-fixture",
        )
        bridge.publish(prompts[0], caches[0])
        expected = model.decode(mx.array([[11]]), caches[0])
        model.attach_semantic_capsules(b)
        with pytest.raises(ValueError, match="owner"):
            batcher.decode([[11], [12]], caches)
        assert bridge.restore(prompts[0] + [11])[0] is None
        model.attach_semantic_capsules(a)
        restored, hit = bridge.restore(prompts[0] + [11])
        leases.append(hit.cache)
        assert hit.hit
        model.diffusion_propose(restored, count=3, steps=2)
        assert restored.length == 6
        got, _, _ = batcher.decode([[11]], [restored])
        assert float(mx.max(mx.abs(got[0] - expected)).item()) == 0
        episode = LoRAEpisode(model, ["semantic_ple.value"], base_revision="fixture")
        assert bridge.restore(prompts[0] + [11])[0] is None
        model.eval()
        with pytest.raises(ValueError, match="owner"):
            batcher.decode([[11]], [restored])
        episode.close()
        restored, hit = bridge.restore(prompts[0] + [11])
        leases.append(hit.cache)
        assert hit.hit
        assert (
            float(
                mx.max(
                    mx.abs(model.decode(mx.array([[11]]), restored) - expected)
                ).item()
            )
            == 0
        )
    finally:
        if episode is not None:
            episode.close()
        for lease in leases:
            if hasattr(lease, "close"):
                lease.close()
        engine.close()
