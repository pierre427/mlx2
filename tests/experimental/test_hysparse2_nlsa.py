"""CPU checks for the HySparse2 N-LSA selection arm (arXiv 2610.42147).

Mechanism only: clustering law, causality, chunk invariance, forced window,
best-first bucket expansion and model prefill/decode parity. No quality claim.
"""

from dataclasses import replace

import pytest

np = pytest.importorskip("numpy")
mx = pytest.importorskip("mlx.core")
from mlx import nn

from mlx2.experimental.hysparse2.attention import (
    _nlsa_index,
    attention,
    nlsa_cluster,
    sparse_attention,
)
from mlx2.experimental.hysparse2.config import Config
from mlx2.experimental.hysparse2.model import Model

LOCAL, GLOBAL = 4, 12
NLSA = {"block": 8, "pool": 2, "buckets": 2, "bucket_max": 8, "sinks": 2}


@pytest.fixture(autouse=True)
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    mx.random.seed(11)
    yield
    mx.set_default_device(previous)


def reference_cluster(keys, present, pool, buckets, bucket_max):
    """Plain-loop restatement of LSA Algorithm 1 for one block."""
    block, d = keys.shape
    unit = keys / np.maximum(np.linalg.norm(keys, axis=-1, keepdims=True), 1e-12)
    unit = np.where(present[:, None], unit, 0.0)
    pools = block // pool
    sums = unit.reshape(pools, pool, d).sum(1)
    size = present.reshape(pools, pool).sum(1)
    active = [bool(size[i] > 0 and np.linalg.norm(sums[i]) > 0) for i in range(pools)]
    owner = list(range(pools))
    while sum(active) > buckets:
        norm = np.linalg.norm(sums, axis=-1)
        cost = np.full((pools, pools), np.inf)
        for i in range(pools):
            for j in range(pools):
                if i != j and active[i] and active[j] and size[i] + size[j] <= bucket_max:
                    cost[i, j] = norm[i] + norm[j] - np.linalg.norm(sums[i] + sums[j])
        nearest = cost.argmin(1)
        pairs = [
            (cost[i, nearest[i]], i)
            for i in range(pools)
            if np.isfinite(cost[i]).any() and nearest[nearest[i]] == i and i < nearest[i]
        ]
        if not pairs:
            break
        pairs.sort()
        for _, i in pairs[: sum(active) - buckets]:
            j = nearest[i]
            sums[i] += sums[j]
            size[i] += size[j]
            sums[j] = 0
            size[j] = 0
            active[j] = False
            owner = [i if o == j else o for o in owner]
    return np.repeat(owner, pool), np.array(active)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_clustering_matches_plain_reference(seed):
    rng = np.random.default_rng(seed)
    keys = rng.normal(size=(3, 32, 6)).astype(np.float32)
    present = np.ones((3, 32), dtype=bool)
    present[2, 5:9] = False  # a gap inside a closed block
    centroids, active, labels = nlsa_cluster(
        mx.array(keys), mx.array(present), pool=4, buckets=4, bucket_max=8
    )
    for n in range(3):
        expected_labels, expected_active = reference_cluster(
            keys[n].astype(np.float64), present[n], 4, 4, 8
        )
        np.testing.assert_array_equal(np.array(labels[n]), expected_labels)
        np.testing.assert_array_equal(np.array(active[n]), expected_active)
    norms = np.linalg.norm(np.array(centroids), axis=-1)
    np.testing.assert_allclose(norms[np.array(active)], 1.0, atol=1e-5)


def test_bucket_cap_is_never_exceeded():
    keys = mx.random.normal((4, 64, 8))
    present = mx.ones((4, 64), dtype=mx.bool_)
    _, active, labels = nlsa_cluster(keys, present, pool=4, buckets=8, bucket_max=12)
    labels = np.array(labels)
    for n in range(4):
        counts = np.bincount(labels[n], minlength=16)
        assert counts.max() <= 12
        assert int(np.array(active[n]).sum()) >= 8


def _inputs(t=64, h=2, d=8):
    return (
        mx.random.normal((1, h, t, d)),
        mx.random.normal((1, 1, t, d)),
        mx.random.normal((1, 1, t, d)),
    )


def _select(q, segments, offset=0, query_tile=5, key_tile=16):
    return attention(
        q,
        segments,
        offset=offset,
        query_tile=query_tile,
        key_tile=key_tile,
        select=(LOCAL, GLOBAL),
        nlsa=NLSA,
    )[1]


def _support(selected):
    positions = np.array(selected[2])[0]
    return [
        set(row[row < 2147483647].tolist()) for row in positions
    ]


def test_chunked_history_matches_one_call():
    q, k, v = _inputs()
    whole = _support(_select(q, [(k, v, 0)]))
    segments, pieces = [], []
    for start in range(0, 64, 5):
        end = min(start + 5, 64)
        segments.append((k[:, :, start:end], v[:, :, start:end], start))
        pieces.append(_support(_select(q[:, :, start:end], list(segments), offset=start)))
    chunked = [row for piece in pieces for row in piece]
    assert chunked == whole


def test_future_keys_do_not_change_earlier_selection():
    q, k, v = _inputs()
    base = _support(_select(q, [(k, v, 0)]))
    for cut in (17, 32, 40):
        altered = mx.concatenate(
            (k[:, :, :cut], mx.random.normal(k[:, :, cut:].shape)), axis=2
        )
        changed = _support(_select(q, [(altered, v, 0)]))
        assert changed[:cut] == base[:cut]


def test_window_sinks_and_budget():
    q, k, v = _inputs()
    support = _support(_select(q, [(k, v, 0)]))
    block = NLSA["block"]
    for t, chosen in enumerate(support):
        assert all(p <= t for p in chosen)
        assert len(chosen) == min(t + 1, LOCAL + GLOBAL)
        forced = set(range(max(0, t - LOCAL + 1), t + 1))
        # A block becomes searchable once full, including for its last row.
        forced |= set(range(((t + 1) // block) * block, t + 1))
        forced |= set(range(min(NLSA["sinks"], t + 1)))
        assert forced <= chosen


def test_clustered_tokens_expand_best_bucket_first():
    q, k, v = _inputs()
    selected = _select(q, [(k, v, 0)])
    support = _support(selected)
    index = _nlsa_index([(k, v, 0)], NLSA["block"], 63, NLSA)
    centroids, active, labels = (np.array(a) for a in index[:3])
    qhat = np.array(q)[0] / np.linalg.norm(np.array(q)[0], axis=-1, keepdims=True)
    block = NLSA["block"]
    for t in range(40, 64):
        searchable = (t + 1) // block
        forced = set(range(t - LOCAL + 1, t + 1)) | set(range(searchable * block, t + 1))
        forced |= set(range(NLSA["sinks"]))
        room = LOCAL + GLOBAL - len(forced)
        buckets = []
        for n in range(searchable):
            for p in range(centroids.shape[2]):
                if active[0, n, p]:
                    score = max(float(qhat[hh, t] @ centroids[0, n, p]) for hh in range(qhat.shape[0]))
                    members = [n * block + i for i in range(block) if labels[0, n, i] == p]
                    buckets.append((-score, members))
        buckets.sort(key=lambda item: (item[0], item[1][0]))
        expected = set()
        for _, members in buckets:
            for position in members:
                if position not in forced and len(expected) < room:
                    expected.add(position)
        assert support[t] - forced == expected


def _nlsa_config(**changes):
    base = replace(
        Config.smoke(),
        candidate_block_size=0,
        candidate_blocks=0,
        sparse_selection="nlsa",
        local_window=3,
        global_tokens=6,
        nlsa_cluster_block=4,
        nlsa_pool=1,
        nlsa_bucket_tokens=2,
        nlsa_bucket_max=4,
        nlsa_sink_tokens=1,
        max_context=64,
    )
    return replace(base, **changes)


@pytest.mark.parametrize("chunk", [1, 4, 7])
def test_model_prefill_decode_match_full_forward(chunk):
    m = Model(_nlsa_config(prefill_chunk=chunk))
    m.eval()
    tokens = mx.array([[(i * 7 + 3) % 40 + 1 for i in range(24)]])
    full = m(tokens)[0]
    logits, cache = m.prefill(tokens[:, :20])
    np.testing.assert_allclose(np.array(logits), np.array(full[:, 19:20]), atol=3e-5)
    for i in range(20, 24):
        out = m.decode(tokens[:, i : i + 1], cache)
        np.testing.assert_allclose(np.array(out), np.array(full[:, i : i + 1]), atol=3e-5)


def test_model_gradients_are_finite():
    m = Model(_nlsa_config())
    tokens = mx.array([[(i * 5 + 1) % 40 + 1 for i in range(24)]])

    def objective(model):
        logits = model(tokens[:, :-1])[0]
        return nn.losses.cross_entropy(logits, tokens[:, 1:], reduction="mean")

    value, grads = nn.value_and_grad(m, objective)(m)
    flat = [g for _, g in __import__("mlx.utils").utils.tree_flatten(grads)]
    assert np.isfinite(float(value))
    assert all(bool(mx.all(mx.isfinite(g)).item()) for g in flat)


def test_config_validation_and_identity():
    oracle = Config.smoke()
    # nlsa_* settings do not touch oracle fingerprints.
    assert oracle.apcv2_identity() == replace(oracle, nlsa_pool=2).apcv2_identity()
    nlsa = _nlsa_config()
    assert nlsa.apcv2_identity() != replace(nlsa, sparse_selection="oracle").apcv2_identity()
    assert nlsa.apcv2_identity() != replace(nlsa, nlsa_bucket_max=2).apcv2_identity()
    with pytest.raises(ValueError):
        replace(nlsa, candidate_block_size=2, candidate_blocks=2)
    with pytest.raises(ValueError):
        replace(nlsa, nlsa_cluster_block=6, nlsa_bucket_tokens=4)
    with pytest.raises(ValueError):
        replace(nlsa, nlsa_cluster_block=10)  # window + sinks exceed budget
    with pytest.raises(ValueError):
        # Review case: a wide local window plus sinks that exceed global.
        replace(nlsa, local_window=8, global_tokens=2, nlsa_sink_tokens=2)
    with pytest.raises(ValueError):
        replace(oracle, sparse_selection="lsa")


@pytest.mark.parametrize("dtype", [mx.float32, mx.bfloat16])
def test_batch_gaps_and_dtype(dtype):
    t = 64
    q = mx.random.normal((2, 2, t, 8)).astype(dtype)
    k = mx.random.normal((2, 1, t, 8)).astype(dtype)
    v = mx.random.normal((2, 1, t, 8)).astype(dtype)
    # Positions 20..27 are absent (a cache fragment gap); history starts at 3.
    segments = [(k[:, :, 3:20], v[:, :, 3:20], 3), (k[:, :, 28:], v[:, :, 28:], 28)]
    q_rows = q[:, :, 28:]
    selected = attention(
        q_rows, segments, offset=28, query_tile=5, key_tile=16,
        select=(LOCAL, GLOBAL), nlsa=NLSA,
    )[1]
    positions = np.array(selected[2])
    assert positions.shape[-1] == LOCAL + GLOBAL
    for row in range(2):
        alone = attention(
            q_rows[row : row + 1],
            [(a[row : row + 1], b[row : row + 1], s) for a, b, s in segments],
            offset=28, query_tile=5, key_tile=16, select=(LOCAL, GLOBAL), nlsa=NLSA,
        )[1]
        np.testing.assert_array_equal(np.array(alone[2])[0], positions[row])
        for i, chosen in enumerate(positions[row]):
            real = chosen[chosen < 2147483647]
            assert not np.any((real >= 20) & (real < 28)) and np.all(real >= 3)
            assert np.all(real <= 28 + i)


def test_sparse_attention_runs_on_nlsa_support():
    q, k, v = _inputs()
    selected = _select(q, [(k, v, 0)])
    out = sparse_attention(q, selected, offset=0, sinks=mx.zeros(2))
    assert out.shape == q.shape and bool(mx.all(mx.isfinite(out)).item())
