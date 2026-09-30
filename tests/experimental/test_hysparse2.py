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
from mlx2.experimental.hysparse2.model import Model
from mlx2.experimental.hysparse2.train import load_checkpoint, loss, save_checkpoint


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.random.seed(9)
    yield
    mx.set_default_device(previous)


def close(a, b, tol=3e-5):
    np.testing.assert_allclose(np.array(a), np.array(b), atol=tol, rtol=tol)


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


def test_apcv2_identity_binds_block_selector_and_semantic_capsule():
    c = Config.smoke()
    digest = "a" * 64
    a = c.apcv2_identity(digest)
    b = replace(c, candidate_blocks=c.candidate_blocks + 1).apcv2_identity(digest)
    assert a["cache_layout_fingerprint"] != b["cache_layout_fingerprint"]
    assert a["semantic_fingerprint"][1] == digest
    cache = Model(c).new_cache(semantic_capsule_digest=digest)
    assert cache.apcv2_identity == a
    with pytest.raises(ValueError):
        c.apcv2_identity("stale")


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
