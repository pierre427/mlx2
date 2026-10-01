"""CPU-only reference checks. No serving or 2M quality claim."""

import json
from dataclasses import replace

import pytest

np = pytest.importorskip("numpy")
mx = pytest.importorskip("mlx.core")
from mlx import nn, optimizers
from mlx.utils import tree_flatten

from mlx2.experimental.hysparse2.attention import attention, sparse_attention
from mlx2.experimental.hysparse2.config import Config
from mlx2.experimental.hysparse2.model import DiffusionLayer, Model
from mlx2.experimental.hysparse2.train import (
    initialize_from_checkpoint,
    load_checkpoint,
    loss,
    save_checkpoint,
)


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.random.seed(9)
    yield
    mx.set_default_device(previous)


def close(a, b, tol=3e-5):
    np.testing.assert_allclose(np.array(a), np.array(b), atol=tol, rtol=tol)


def test_diffusion_fused_bidirectional_attention_matches_dense_reference():
    c = Config.smoke()
    layer = DiffusionLayer(c)
    x = mx.random.normal((2, 7, c.hidden_size))
    h = layer.norm1(x)
    shape = (*h.shape[:-1], layer.heads, layer.head_dim)
    q = layer.q(h).reshape(shape).transpose(0, 2, 1, 3)
    k = layer.k(h).reshape(shape).transpose(0, 2, 1, 3)
    v = layer.v(h).reshape(shape).transpose(0, 2, 1, 3)
    probabilities = mx.softmax(
        (q.astype(mx.float32) @ k.astype(mx.float32).swapaxes(-1, -2))
        * layer.head_dim**-0.5,
        axis=-1,
    ).astype(v.dtype)
    dense = (probabilities @ v).transpose(0, 2, 1, 3).reshape(x.shape)
    fused = (
        mx.fast.scaled_dot_product_attention(q, k, v, scale=layer.head_dim**-0.5)
        .transpose(0, 2, 1, 3)
        .reshape(x.shape)
    )
    close(fused, dense, 1e-4)


def test_diffusion_feedback_connects_trunk_and_ple_without_changing_forward():
    config = Config.smoke()
    detached = Model(config)
    coupled = Model(replace(config, diffusion_trunk_gradient_scale=1.0))
    coupled.load_weights(tree_flatten(detached.parameters()), strict=True)
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    a, ga = nn.value_and_grad(detached, loss)(detached, tokens)
    b, gb = nn.value_and_grad(coupled, loss)(coupled, tokens)
    close(a, b, 0)
    da, db = dict(tree_flatten(ga)), dict(tree_flatten(gb))
    for name in (
        "self_decoder.0.attention.q.weight",
        "cross_decoder.0.attention.q.weight",
        "semantic_ple.embedding.weight",
    ):
        assert float(mx.max(mx.abs(da[name] - db[name])).item()) > 1e-7, name
    detached.eval()
    coupled.eval()
    close(detached(tokens)[0], coupled(tokens)[0], 0)


@pytest.mark.parametrize("scale", [-1, 1.01, float("nan"), float("inf")])
def test_diffusion_feedback_rejects_invalid_scale(scale):
    with pytest.raises(ValueError, match="gradient scale"):
        replace(Config.smoke(), diffusion_trunk_gradient_scale=scale)


@pytest.mark.parametrize("window", [None, 3])
@pytest.mark.parametrize("sink", [False, True])
def test_tiled_attention_dense_reference(window, sink):
    q = mx.random.normal((2, 3, 11, 4))
    k = mx.random.normal((2, 1, 11, 4))
    v = mx.random.normal((2, 1, 11, 4))
    sinks = mx.array([-0.5, 0.0, 0.8]) if sink else None
    result, _ = attention(
        q,
        [(k[:, :, :5], v[:, :, :5], 0), (k[:, :, 5:], v[:, :, 5:], 5)],
        offset=0,
        query_tile=3,
        key_tile=4,
        window=window,
        sinks=sinks,
    )
    positions = mx.arange(11)
    mask = positions[None, :] <= positions[:, None]
    if window:
        mask = mask & (positions[None, :] > positions[:, None] - window)
    logits = mx.where(mask, q @ k.swapaxes(-1, -2) / 2, -1e30)
    if sink:
        logits = mx.concatenate(
            (logits, mx.broadcast_to(sinks[None, :, None, None], (2, 3, 11, 1))),
            axis=-1,
        )
    weights = mx.softmax(logits, axis=-1)[..., :11]
    close(result, weights @ v)


def test_oracle_forced_local_global_no_future_and_tile_ties():
    # All oracle scores equal: earliest global positions must break ties.
    q = mx.zeros((1, 2, 9, 4))
    k = mx.zeros((1, 1, 9, 4))
    v = mx.random.normal((1, 1, 9, 4))
    selections = []
    for tile in (2, 4, 9):
        _, selected = attention(
            q, [(k, v, 0)], offset=0, query_tile=3, key_tile=tile, select=(3, 2)
        )
        positions = np.array(selected[2])[0]
        for t in range(9):
            actual = set(positions[t][positions[t] <= t].tolist())
            assert actual == set(range(max(0, t - 2), t + 1)) | set(
                range(min(2, max(0, t - 2)))
            )
        selections.append(selected)
    sq = mx.random.normal(q.shape)
    for s in selections[1:]:
        close(
            sparse_attention(sq, s, offset=0, sinks=mx.zeros(2)),
            sparse_attention(sq, selections[0], offset=0, sinks=mx.zeros(2)),
        )


def test_block_selector_masks_unused_support_slots():
    q = mx.zeros((1, 1, 1, 1))
    k = mx.zeros((1, 1, 12, 1))
    v = mx.arange(12, dtype=mx.float32).reshape(1, 1, 12, 1)
    _, selected = attention(
        q,
        [(k, v, 0)],
        offset=11,
        query_tile=1,
        key_tile=4,
        select=(2, 4),
        block_select=(2, 1),
    )
    positions = np.array(selected[2])[0, 0]
    assert set(positions[positions <= 11].tolist()) == {0, 1, 10, 11}
    got = sparse_attention(q, selected, offset=11, sinks=mx.zeros(1))
    # Four valid tokens plus a zero-value sink with equal logits.
    close(got, mx.array([[[[22.0 / 5]]]]))


@pytest.mark.parametrize("tied", [False, True])
def test_block_selector_grouping_preserves_support_and_gradients(tied):
    q = mx.zeros((2, 3, 13, 4)) if tied else mx.random.normal((2, 3, 13, 4))
    k = mx.random.normal((2, 1, 13, 4))
    v = mx.random.normal(k.shape)
    blocks = [
        (k[:, :, :3], v[:, :, :3], 0),
        (k[:, :, 3:9], v[:, :, 3:9], 3),
        (k[:, :, 9:], v[:, :, 9:], 9),
    ]

    def execute(q, v, tile):
        split = [
            (block[0], v[:, :, start : start + block[0].shape[2]], start)
            for block, start in zip(blocks, (0, 3, 9))
        ]
        dense, selected = attention(
            q,
            split,
            offset=0,
            query_tile=4,
            key_tile=tile,
            select=(2, 4),
            block_select=(2, 2),
        )
        sparse = sparse_attention(q, selected, offset=0, sinks=mx.zeros(3))
        return dense, selected, sparse

    reference = execute(q, v, 2)
    grouped = execute(q, v, 8)
    close(reference[0], grouped[0])
    close(reference[1][2], grouped[1][2], 0)
    close(reference[2], grouped[2])
    for tile in (2, 8):
        gradient = mx.grad(lambda v, tile=tile: mx.sum(execute(q, v, tile)[2]))(v)
        assert bool(mx.all(mx.isfinite(gradient)).item())
        if tile == 2:
            expected_gradient = gradient
        else:
            close(expected_gradient, gradient)


@pytest.mark.parametrize("chunk", [1, 4, 16])
def test_prefill_decode_and_cache_bytes(chunk):
    c = replace(Config.smoke(), prefill_chunk=chunk)
    m = Model(c)
    m.eval()
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8, 9], [2, 2, 3, 1, 4, 2, 5, 3, 6]])
    full = m(tokens)[0]
    logits, cache = m.prefill(tokens[:, :7])
    close(logits, full[:, 6:7])
    assert cache.cross_layer_calls == len(m.cross_decoder)
    for i in range(7, 9):
        close(m.decode(tokens[:, i : i + 1], cache), full[:, i : i + 1])
    capacity = c.capacity(batch=2, context=9, bytes_per_element=4)
    assert (
        cache.resident_bytes()
        == capacity["full_kv_bytes"] + capacity["rolling_kv_bytes"]
    )
    assert all(
        not hasattr(layer.attention, "k")
        for layer in m.cross_decoder
        if layer.kind == "sparse"
    )


def test_cache_only_and_identity_limits():
    c = replace(Config.smoke(), max_context=8)
    m = Model(c)
    m.eval()
    logits, cache = m.prefill(mx.array([[1, 2, 3, 4]]), return_logits=False)
    assert logits is None and cache.cross_layer_calls == 0
    m.prefill(mx.array([[5, 6, 7, 8]]), cache, return_logits=False)
    with pytest.raises(ValueError):
        m.decode(mx.array([[1]]), cache)
    assert cache.length == 8
    other = Model(c)
    other.eval()
    with pytest.raises(ValueError):
        other.decode(mx.array([[1]]), cache)


def test_cache_rejects_changed_ple_revision_before_mutating_state():
    model = Model(Config.smoke())
    model.eval()
    model.ple_sidecar_digest = "a" * 64
    _, cache = model.prefill(mx.array([[1, 2, 3]]))
    model.ple_sidecar_digest = "b" * 64
    with pytest.raises(ValueError, match="revision"):
        model.decode(mx.array([[4]]), cache)
    assert cache.length == 3


@pytest.mark.parametrize(
    "change", [{"rope_base": 20000.0}, {"rope_dims": 0}, {"norm_eps": 1e-5}]
)
def test_cache_identity_binds_attention_math(change):
    config = Config.smoke()
    assert config.apcv2_identity() != replace(config, **change).apcv2_identity()


def test_old_checkpoint_load_rebuilds_math_bound_cache_identity(tmp_path):
    config = Config.smoke()
    model = Model(config)
    path = save_checkpoint(
        tmp_path, model, optimizers.AdamW(learning_rate=1e-4), 0, {}, mode="model"
    )
    state_path = path / "state.json"
    state = json.loads(state_path.read_text())
    identity = state["permanent_sidecar"]["apcv2_identity"]
    identity["cache_layout_fingerprint"] = identity["cache_layout_fingerprint"].rsplit(
        ":", 1
    )[0]
    state_path.write_text(json.dumps(state))
    restored = Model(config)
    initialize_from_checkpoint(path, restored)
    assert (
        restored.new_cache().apcv2_identity["cache_layout_fingerprint"]
        == config.apcv2_identity()["cache_layout_fingerprint"]
    )


def test_high_absolute_positions_without_full_allocation():
    offset = 2097152 - 3
    q = mx.random.normal((1, 2, 3, 8))
    k = mx.random.normal((1, 1, 5, 8))
    v = mx.random.normal(k.shape)
    high, hs = attention(
        q, [(k, v, offset - 2)], offset=offset, query_tile=2, key_tile=2, select=(2, 1)
    )
    low, ls = attention(
        q, [(k, v, 0)], offset=2, query_tile=2, key_tile=2, select=(2, 1)
    )
    close(high, low)
    close(hs[2] - offset + 2, ls[2], 0)
    cap = Config().capacity()
    assert cap["layers"] == 49 and cap["full_kv_bytes"] == 10 * 1024**3


def test_parameters_gradients_checkpoint_and_resume(tmp_path):
    c = Config.smoke()
    m = Model(c)
    assert (
        sum(x.size for _, x in tree_flatten(m.parameters()))
        == c.capacity()["parameters"]
    )
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    fn = nn.value_and_grad(m, loss)
    a, ga = fn(m, tokens)
    m.checkpoint_layers = True
    b, gb = fn(m, tokens)
    close(a, b)
    for (name, x), (_, y) in zip(tree_flatten(ga), tree_flatten(gb)):
        close(x, y, 1e-4)
        assert bool(mx.all(mx.isfinite(y)).item()), name
    gradients = dict(tree_flatten(gb))
    for name in (
        "cross_decoder.0.attention.k.weight",
        "mtp_head.projection.weight",
        "self_decoder.0.moe.router.weight",
        "semantic_ple.embedding.weight",
        "diffusion_student.layers.0.mlp.up.weight",
    ):
        assert float(mx.sum(mx.abs(gradients[name])).item()) > 0, name
    optimizer = optimizers.AdamW(learning_rate=1e-4)
    optimizer.update(m, gb)
    mx.eval(m.parameters(), optimizer.state)
    checkpoint = save_checkpoint(tmp_path, m, optimizer, 1, {"seed": 9})
    state = json.loads((checkpoint / "state.json").read_text())
    assert state["permanent_sidecar"]["schema"] == "mlx2.hysparse2-semantic-ple.v1"
    assert (
        state["permanent_sidecar"]["apcv2_identity"]["semantic_fingerprint"][2]
        == state["permanent_sidecar"]["sha256"]
    )
    assert (checkpoint / "semantic-ple.safetensors").is_file()
    restored = Model(c)
    restored.checkpoint_layers = True
    opt2 = optimizers.AdamW(learning_rate=1e-4)
    assert load_checkpoint(checkpoint, restored, opt2, {"seed": 9}) == 1
    for model, opt in ((m, optimizer), (restored, opt2)):
        _, grad = nn.value_and_grad(model, loss)(model, tokens)
        opt.update(model, grad)
        mx.eval(model.parameters(), opt.state)
    for (_, x), (_, y) in zip(
        tree_flatten(m.parameters()), tree_flatten(restored.parameters())
    ):
        close(x, y, 1e-6)
    with pytest.raises(ValueError):
        load_checkpoint(checkpoint, restored, opt2, {"seed": 10})
    # The first full-size validation predates PLE-SHA cache identity binding.
    # Its separately verified sidecar remains an exact training resume input.
    legacy_identity = c.apcv2_identity()
    fingerprint = legacy_identity["semantic_fingerprint"]
    legacy_identity["semantic_fingerprint"] = fingerprint[:2] + fingerprint[3:]
    state["permanent_sidecar"]["apcv2_identity"] = legacy_identity
    (checkpoint / "state.json").write_text(json.dumps(state))
    legacy = Model(c)
    assert (
        load_checkpoint(
            checkpoint,
            legacy,
            optimizers.AdamW(learning_rate=1e-4),
            {"seed": 9},
        )
        == 1
    )
    assert legacy.ple_sidecar_digest == state["permanent_sidecar"]["sha256"]


@pytest.mark.parametrize("failure", ["optimizer", "metadata", "rename"])
def test_failed_checkpoint_preserves_live_identity_and_cleans_own_staging(tmp_path, monkeypatch, failure):
    from pathlib import Path
    from mlx import optimizers

    model = Model(Config.smoke())
    model.eval()
    tokens = mx.array([[1, 2, 3, 4]])
    _, cache = model.prefill(tokens)
    owner, digest = model._cache_owner, model.ple_sidecar_digest
    unrelated = tmp_path / ".writing-unrelated"
    unrelated.mkdir()
    (unrelated / "keep").write_text("preserve")
    def fail(*args, **kwargs):
        raise OSError("injected checkpoint failure")
    if failure == "optimizer":
        save = mx.save_safetensors
        def injected(path, *args, **kwargs):
            return fail() if Path(path).name == "optimizer.safetensors" else save(path, *args, **kwargs)
        monkeypatch.setattr(mx, "save_safetensors", injected)
    elif failure == "metadata":
        write = Path.write_text
        def injected(path, *args, **kwargs):
            return fail() if path.name == "state.json" else write(path, *args, **kwargs)
        monkeypatch.setattr(Path, "write_text", injected)
    else:
        monkeypatch.setattr(Path, "rename", fail)
    with pytest.raises(OSError, match="injected checkpoint failure"):
        save_checkpoint(tmp_path, model, optimizers.Adam(1e-3), 0, {})
    assert model._cache_owner is owner and model.ple_sidecar_digest == digest
    assert sorted(p.name for p in tmp_path.iterdir()) == [".writing-unrelated"]
    assert (unrelated / "keep").read_text() == "preserve"
    model.decode(mx.array([[5]]), cache)


@pytest.mark.parametrize("failure", ["model_file", "partial_update", "optimizer_file"])
def test_failed_checkpoint_load_preserves_live_state(tmp_path, monkeypatch, failure):
    source = Model(Config.smoke())
    checkpoint = save_checkpoint(tmp_path, source, optimizers.Adam(1e-3), 0, {})
    model = Model(source.config)
    model.eval()
    tokens = mx.array([[1, 2, 3, 4]])
    _, cache = model.prefill(tokens)
    before = dict(tree_flatten(model.parameters()))
    owner, epoch, digest = model._cache_owner, model._parameter_epoch, model.ple_sidecar_digest
    optimizer = optimizers.Adam(1e-3)
    state = optimizer.state
    if failure == "model_file":
        (checkpoint / "model.safetensors").write_bytes(b"corrupt")
    elif failure == "optimizer_file":
        (checkpoint / "optimizer.safetensors").write_bytes(b"corrupt")
    else:
        def partial(*a, **kw):
            model.update({"embedding": {"weight": model.embedding.weight + 0.1}})
            raise RuntimeError("partial weight load")
        monkeypatch.setattr(model, "load_weights", partial)
    with pytest.raises((ValueError, RuntimeError)):
        load_checkpoint(checkpoint, model, optimizer, {})
    after = dict(tree_flatten(model.parameters()))
    assert after.keys() == before.keys()
    assert all(float(mx.max(mx.abs(after[k] - v)).item()) == 0 for k, v in before.items())
    assert model._cache_owner is owner and model._parameter_epoch is epoch
    assert model.ple_sidecar_digest == digest and optimizer.state is state
    model.decode(mx.array([[5]]), cache)


@pytest.mark.parametrize("mismatch", ["values", "coverage", "shape", "dtype"])
def test_checkpoint_ple_sidecar_must_match_restored_tensors(tmp_path, mismatch):
    from mlx2.experimental.hysparse2.train import file_hash
    source = Model(Config.smoke())
    checkpoint = save_checkpoint(tmp_path, source, optimizers.Adam(1e-3), 0, {}, mode="model")
    model = Model(source.config)
    model.eval()
    before = dict(tree_flatten(model.parameters()))
    owner, epoch, digest = model._cache_owner, model._parameter_epoch, model.ple_sidecar_digest
    if mismatch == "values":
        path = checkpoint / "model.safetensors"
        values = mx.load(str(path))
        values["semantic_ple.value.weight"] = values["semantic_ple.value.weight"] + 0.1
        mx.eval(values)
        mx.save_safetensors(str(path), values)
    else:
        path = checkpoint / "semantic-ple.safetensors"
        values = mx.load(str(path))
        key = "semantic_ple.value.weight"
        if mismatch == "coverage":
            del values[key]
        elif mismatch == "shape":
            values[key] = values[key][:1]
        else:
            values[key] = values[key].astype(mx.float16)
        mx.eval(values)
        mx.save_safetensors(str(path), values)
        metadata = json.loads((checkpoint / "state.json").read_text())
        sidecar = metadata["permanent_sidecar"]
        sidecar["sha256"] = file_hash(path)
        sidecar["apcv2_identity"] = source.config.apcv2_identity(ple_sidecar_digest=sidecar["sha256"])
        (checkpoint / "state.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="PLE sidecar tensors"):
        initialize_from_checkpoint(checkpoint, model)
    assert model._cache_owner is owner and model._parameter_epoch is epoch
    assert model.ple_sidecar_digest == digest
    assert all(float(mx.max(mx.abs(dict(tree_flatten(model.parameters()))[k] - v)).item()) == 0 for k, v in before.items())


@pytest.mark.parametrize("damage", ["missing_moment", "shape", "nonfinite", "step"])
def test_invalid_optimizer_resume_rejects_before_live_mutation(tmp_path, damage):
    source = Model(Config.smoke())
    optimizer = optimizers.AdamW(1e-3)
    tokens = mx.array([[1, 2, 3, 4, 5]])
    _, gradients = nn.value_and_grad(source, loss)(source, tokens)
    optimizer.update(source, gradients)
    checkpoint = save_checkpoint(tmp_path, source, optimizer, 1, {})
    path = checkpoint / "optimizer.safetensors"
    state = mx.load(str(path))
    moment = next(k for k in state if k.endswith(".m"))
    if damage == "missing_moment":
        del state[moment]
    elif damage == "shape":
        state[moment] = state[moment].reshape(-1)[:1]
    elif damage == "nonfinite":
        state[moment] = mx.full(state[moment].shape, float("nan"))
    else:
        state["step"] = state["step"] + 1
    mx.eval(state)
    mx.save_safetensors(str(path), state)
    model = Model(source.config)
    before = dict(tree_flatten(model.parameters()))
    owner = model._cache_owner
    fresh_optimizer = optimizers.AdamW(1e-3)
    previous = fresh_optimizer.state
    with pytest.raises(ValueError, match="optimizer state"):
        load_checkpoint(checkpoint, model, fresh_optimizer, {})
    assert model._cache_owner is owner and fresh_optimizer.state is previous
    assert all(dict(tree_flatten(model.parameters()))[k] is v for k, v in before.items())


def test_checkpoint_optimizer_class_must_match(tmp_path):
    source = Model(Config.smoke())
    checkpoint = save_checkpoint(tmp_path, source, optimizers.AdamW(1e-3), 0, {})
    model = Model(source.config)
    owner = model._cache_owner
    with pytest.raises(ValueError, match="matching supported Adam optimizer"):
        load_checkpoint(checkpoint, model, optimizers.Adam(1e-3), {})
    assert model._cache_owner is owner


def test_model_only_checkpoint_is_explicitly_not_an_exact_resume(tmp_path):
    c = Config.smoke()
    model = Model(c)
    optimizer = optimizers.AdamW(learning_rate=1e-4)
    tokens = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    _, gradients = nn.value_and_grad(model, loss)(model, tokens)
    optimizer.update(model, gradients)
    mx.eval(model.parameters(), optimizer.state)

    checkpoint = save_checkpoint(
        tmp_path, model, optimizer, 7, {"seed": 9}, mode="model"
    )
    state = json.loads((checkpoint / "state.json").read_text())
    assert state["schema"] == "mlx2.hysparse2-model-checkpoint.v1"
    assert state["optimizer_state_saved"] is False
    assert state["exact_training_resume"] is False
    assert (checkpoint / "model.safetensors").is_file()
    assert (checkpoint / "semantic-ple.safetensors").is_file()
    assert not (checkpoint / "optimizer.safetensors").exists()

    restored = Model(c)
    with pytest.raises(ValueError, match="exact optimizer resume state"):
        load_checkpoint(
            checkpoint,
            restored,
            optimizers.AdamW(learning_rate=1e-4),
            {"seed": 9},
        )

    initialized = Model(c)
    receipt = initialize_from_checkpoint(checkpoint, initialized)
    assert receipt["schema"] == "mlx2.hysparse2-model-checkpoint.v1"
    assert receipt["step"] == 7
    assert receipt["optimizer_state_restored"] is False
    assert receipt["exact_training_resume"] is False
    assert receipt["semantic_ple_sha256"] == state["permanent_sidecar"]["sha256"]
    assert len(receipt["model_sha256"]) == len(receipt["state_sha256"]) == 64
    for (_, expected), (_, actual) in zip(
        tree_flatten(model.parameters()), tree_flatten(initialized.parameters())
    ):
        close(expected, actual, 0)

    (checkpoint / "semantic-ple.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="missing or corrupt"):
        initialize_from_checkpoint(checkpoint, Model(c))


def test_apcv2_identity_binds_block_selector_and_semantic_capsule():
    c = Config.smoke()
    digest = "a" * 64
    ple_digest = "b" * 64
    a = c.apcv2_identity(digest, ple_digest)
    b = replace(c, candidate_blocks=c.candidate_blocks + 1).apcv2_identity(
        digest, ple_digest
    )
    assert a["cache_layout_fingerprint"] != b["cache_layout_fingerprint"]
    assert a["semantic_fingerprint"][1] == digest
    assert a["semantic_fingerprint"][2] == ple_digest
    model = Model(c)
    model.ple_sidecar_digest = ple_digest
    cache = model.new_cache(semantic_capsule_digest=digest)
    assert cache.apcv2_identity == a
    assert c.apcv2_identity(digest, "c" * 64) != a
    assert c.apcv2_identity(digest)["semantic_fingerprint"][2] == "unversioned-ple"
    with pytest.raises(ValueError):
        c.apcv2_identity("stale")
    with pytest.raises(ValueError):
        replace(c, semantic_ple_rows=0, semantic_ple_dim=0).apcv2_identity(
            ple_sidecar_digest=ple_digest
        )


def test_oracle_matches_dense_normalized_multihead_ranking():
    q = mx.random.normal((2, 3, 9, 4))
    k = mx.random.normal((2, 1, 9, 4))
    v = mx.random.normal(k.shape)
    _, selected = attention(
        q, [(k, v, 0)], offset=0, query_tile=4, key_tile=3, select=(2, 3)
    )
    scores = np.array(q @ k.swapaxes(-1, -2) / 2)
    for t in range(9):
        raw = scores[:, :, t, : t + 1]
        prob = np.exp(raw - raw.max(axis=-1, keepdims=True))
        prob /= prob.sum(axis=-1, keepdims=True)
        importance = prob.mean(axis=1)
        for b in range(2):
            local = set(range(max(0, t - 1), t + 1))
            older = list(range(max(0, t - 1)))
            globals_ = sorted(older, key=lambda p: (-importance[b, p], p))[:3]
            positions = np.array(selected[2])[b, t]
            assert set(positions[positions <= t].tolist()) == local | set(globals_)


def test_full_layout_cache_matches_full_forward():
    c = replace(
        Config.smoke(),
        self_layers=25,
        self_full_layer=12,
        cross_blocks=4,
        sparse_per_block=5,
    )
    m = Model(c)
    m.eval()
    tokens = mx.array([[1, 2, 3, 4, 5, 6]])
    full = m(tokens)[0]
    last, cache = m.prefill(tokens[:, :5])
    close(last, full[:, 4:5], 1e-4)
    close(m.decode(tokens[:, 5:], cache), full[:, 5:], 1e-4)
    assert len(m.self_decoder) + len(m.cross_decoder) == 49
