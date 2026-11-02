"""GLM-5.3 decode glue levers (glm53_router, glm53_kda_small): exactness vs the reference law.

CPU tests prove the selection law (rank = strictly-before count with index
tie-break) equals the stable ``argsort(-choice)[..., :8]``, that the fused
policies run the reference chain off Metal, and that the policy keys are
validated and enter the APCv2 profile only off the reference. The Metal tests
(MLX2_RUN_GPU_TESTS=1) compare each kernel with the reference op chain bitwise.
"""

import os

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models import glm53_glue_metal as glue
from mlx2.runtime.models.glm53_gguf import GLUE_COUNTERS, Glm53GGUFModel, Glm53KDACache

GPU = os.environ.get("MLX2_RUN_GPU_TESTS") == "1"
EXPERTS, TOPK, HEADS, DIM = 144, 8, 64, 128


def bits(a):
    return np.array(a).view(np.uint32)


def reference_route(logits, bias):
    """Today's Glm53GGUFModel._mlp routing chain, verbatim."""
    scores = mx.sigmoid(logits)
    choice = scores + bias
    ids = mx.argsort(-choice, axis=-1)[..., :8].astype(mx.uint32)
    selected = mx.take_along_axis(scores, ids, axis=-1)
    selected = selected / selected.sum(-1, keepdims=True) * 2.5
    return ids, selected


def reference_kda_gate(a, dt_bias, ssm_a, beta, gate):
    """Today's Glm53GGUFModel._kda gate chain, verbatim ([..., heads, dim] views)."""
    a = a + dt_bias.reshape(HEADS, DIM)
    decay = mx.exp(-5 * mx.sigmoid(a * ssm_a.reshape(HEADS, 1)))
    return decay, mx.sigmoid(beta), mx.sigmoid(gate)


def rank_law(choice):
    """The fused router's selection law in numpy: rank each expert, keep ranks < TOPK."""
    v = -np.asarray(choice, np.float32)
    rows = v.reshape(-1, v.shape[-1])
    out = np.empty((rows.shape[0], TOPK), np.uint32)
    for r, row in enumerate(rows):
        nan = np.isnan(row)
        for i in range(row.size):
            less_ji = np.where(nan | nan[i], (~nan) & nan[i], row < row[i])
            less_ij = np.where(nan | nan[i], (~nan[i]) & nan, row[i] < row)
            before = less_ji | (~less_ij & (np.arange(row.size) < i))
            rank = int(before.sum())
            if rank < TOPK:
                out[r, rank] = i
    return out.reshape(*v.shape[:-1], TOPK)


def tie_cases(rng):
    """(name, logits [rows, E], bias [E]) covering random, near-tie and exact-tie rows."""
    bias = rng.normal(0, 0.05, EXPERTS).astype(np.float32)
    random = rng.normal(0, 2, (5, EXPERTS)).astype(np.float32)
    # Exact ties: equal logits with equal bias; ties straddle the rank-8 cut.
    exact = np.zeros((3, EXPERTS), np.float32)
    exact[0, ::7] = 1.0
    exact[1, 3:40] = 0.5
    exact[2, :] = -0.25
    flat = np.zeros(EXPERTS, np.float32)
    # Near ties: neighbours one ulp apart across the cut, plus +-0.0.
    near = np.tile(rng.normal(0, 0.01, EXPERTS).astype(np.float32), (3, 1))
    near[0, 10:20] = np.float32(0.3)
    near[0, 15] = np.nextafter(np.float32(0.3), np.float32(1))
    near[0, 17] = np.nextafter(np.float32(0.3), np.float32(-1))
    near[1, :] = 0.0
    near[1, 1::2] = -0.0
    near[2, 100:] = 6.0  # many saturated sigmoid(x) == 1 - tiny scores
    return [
        ("random", random, bias),
        ("exact", exact, flat),
        ("exact_biased", exact, bias),
        ("near", near, flat),
    ]


# ---- CPU ----------------------------------------------------------------------


@pytest.fixture
def cpu():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def test_rank_law_equals_stable_argsort_top8_including_ties(cpu):
    for name, logits, bias in tie_cases(np.random.default_rng(531)):
        ids, _ = reference_route(mx.array(logits), mx.array(bias))
        choice = mx.sigmoid(mx.array(logits)) + mx.array(bias)
        np.testing.assert_array_equal(rank_law(np.array(choice)), np.array(ids), err_msg=name)


class GlueWeights:
    def __init__(self, seed=532):
        rng = np.random.default_rng(seed)
        self.values = {
            "blk.3.exp_probs_b.bias": rng.normal(0, 0.05, EXPERTS).astype(np.float32),
            "blk.0.ssm_dt.bias": rng.normal(-2, 0.2, HEADS * DIM).astype(np.float32),
            "blk.0.ssm_a": rng.uniform(1, 8, HEADS).astype(np.float32),
            "blk.0.ssm_norm.weight": np.ones(DIM, np.float32),
            "blk.0.ssm_conv1d_q.weight": rng.normal(0, 0.2, (HEADS * DIM, 1, 4)).astype(np.float32),
            "blk.0.ssm_conv1d_k.weight": rng.normal(0, 0.2, (HEADS * DIM, 1, 4)).astype(np.float32),
            "blk.0.ssm_conv1d_v.weight": rng.normal(0, 0.2, (HEADS * DIM, 1, 4)).astype(np.float32),
        }
        self.rng = rng
        self.mats = {}

    def array(self, name):
        return mx.array(self.values[name])

    def linear(self, name):
        def call(x):
            if name not in self.mats:
                out = {"ssm_f_a": 128, "ssm_g_a": 128, "ssm_beta": 64, "attn_output": 4}.get(
                    name.split(".")[2], HEADS * DIM
                )
                self.mats[name] = mx.array(
                    self.rng.normal(0, 0.05, (x.shape[-1], out)).astype(np.float32)
                )
            return x @ self.mats[name]

        return call


def model(weights, **policy):
    return Glm53GGUFModel(
        {"layer_types": [], "rms_norm_eps": 1e-5}, weights, identity="synthetic", **policy
    )


def test_fused_policies_run_the_reference_chain_off_metal_and_count_it(cpu):
    weights = GlueWeights()
    reference, fused = model(weights), model(weights, router="fused_v1", kda_small="fused_v1")
    logits = mx.array(np.random.default_rng(533).normal(0, 2, (1, 3, EXPERTS)).astype(np.float32))
    for got, want in zip(fused._route(logits, "blk.3"), reference._route(logits, "blk.3")):
        assert np.array_equal(bits(got), bits(want))
    x = mx.array(np.random.default_rng(534).normal(size=(1, 3, 4)).astype(np.float32))
    one, two = Glm53KDACache("synthetic"), Glm53KDACache("synthetic")
    want, got = reference._kda(x, one, "blk.0"), fused._kda(x, two, "blk.0")
    assert np.array_equal(bits(got), bits(want))
    for a, b in zip(one.state, two.state):
        assert np.array_equal(bits(a), bits(b))
    assert fused.calls["router_fused_declined"] == fused.calls["kda_small_fused_declined"] == 1
    assert fused.calls["router_fused"] == fused.calls["kda_small_fused"] == 0
    assert all(reference.calls[name] == 0 for name in GLUE_COUNTERS)


def test_glue_policy_values_are_validated_by_the_model(cpu):
    weights = GlueWeights()
    assert model(weights).router == model(weights).kda_small == "reference"
    for bad in ({"router": "fused"}, {"kda_small": "kblock"}, {"router": None}):
        with pytest.raises(ValueError, match="glue policy"):
            model(weights, **bad)


def _adapter(policy):
    from mlx2.adapters.glm53_gguf import Glm53GGUFAdapter

    adapter = object.__new__(Glm53GGUFAdapter)
    with pytest.raises(FileNotFoundError):  # validation and profile precede the artifact
        adapter._load("/nonexistent/glm53-decode-levers", policy)
    return adapter


def test_adapter_keys_extend_the_profile_only_off_the_reference():
    from mlx2.adapters.glm53_gguf import Glm53GGUFAdapter

    # Both are adapter defaults since 2026-10-31 (Pierre); "reference" opts out.
    default = _adapter({})
    explicit = _adapter({"glm53_router": "fused_v1", "glm53_kda_small": "fused_v1"})
    assert default.profile == explicit.profile
    assert default.kernel_policy["glm53_router"] == default.kernel_policy["glm53_kda_small"] == "fused_v1"
    reference = _adapter({"glm53_router": "reference", "glm53_kda_small": "reference"})
    assert "router" not in reference.profile and "kda_small" not in reference.profile
    serving = {"glm53_row_kernel": "kblock_rows", "glm53_kda_group": "kblock"}
    both = _adapter(serving)
    assert ":kda_group=kblock:router=fused_v1:kda_small=fused_v1:" in both.profile + ":"
    router = _adapter({"glm53_kda_small": "reference"}).profile
    small = _adapter({"glm53_router": "reference"}).profile
    assert len({default.profile, router, small, reference.profile, both.profile}) == 5
    for bad in ({"glm53_router": "on"}, {"glm53_kda_small": "kblock"}, {"glm53_router": "fused_v3"}):
        with pytest.raises(ValueError, match="unsupported GLM-5.3 packed execution policy"):
            object.__new__(Glm53GGUFAdapter)._load("/nonexistent", bad)


# ---- Metal (MLX2_RUN_GPU_TESTS=1 only) ------------------------------------------

metal = pytest.mark.skipif(not (GPU and mx.metal.is_available()),
                           reason="Metal kernel test; set MLX2_RUN_GPU_TESTS=1")


@pytest.fixture
def gpu():
    previous = mx.default_device()
    mx.set_default_device(mx.gpu)
    yield
    mx.set_default_device(previous)


@metal
def test_gpu_router_kernel_is_bitwise_the_reference_chain(gpu):
    rng = np.random.default_rng(535)
    cases = tie_cases(rng)
    extreme = rng.normal(0, 30, (4, EXPERTS)).astype(np.float32)
    extreme[0, 5] = np.inf
    extreme[1, 9] = -np.inf
    extreme[2, :] = 88.0  # every score saturates to exactly 1.0; the bias alone orders them
    cases.append(("extreme", extreme, cases[0][2]))
    for name, logits, bias in cases:
        for shape in ((1, 1, EXPERTS), (1, logits.shape[0], EXPERTS)):  # decode and prefill rows
            rows = mx.array(logits[: shape[1]].reshape(shape))
            b = mx.array(bias)
            assert glue.router_admits(rows, b, TOPK)
            got_ids, got_w = glue.fused_router(rows, b)
            want_ids, want_w = reference_route(rows, b)
            mx.eval(got_ids, got_w, want_ids, want_w)
            assert got_ids.dtype == mx.uint32 and got_ids.shape == want_ids.shape
            np.testing.assert_array_equal(np.array(got_ids), np.array(want_ids), err_msg=name)
            assert np.array_equal(bits(got_w), bits(want_w)), name
    # NaN logits sort last in index order: a row with three NaNs, a row with
    # fewer than eight finite choices (NaNs selected, NaN weights) and an
    # all-NaN row. Weights compare by bits, NaN payloads included.
    nan = rng.normal(0, 1, (3, EXPERTS)).astype(np.float32)
    nan[0, [0, 50, 143]] = np.nan
    nan[1, 5:] = np.nan
    nan[2, :] = np.nan
    for rows in (mx.array(nan[None]), mx.array(nan[1:2][None])):
        b = mx.array(cases[0][2])
        (got_ids, got_w), (want_ids, want_w) = glue.fused_router(rows, b), reference_route(rows, b)
        mx.eval(got_ids, got_w, want_ids, want_w)
        np.testing.assert_array_equal(np.array(got_ids), np.array(want_ids))
        assert np.array_equal(bits(got_w), bits(want_w))


@metal
def test_gpu_router_through_the_model_counts_and_matches(gpu):
    weights = GlueWeights()
    reference, fused = model(weights), model(weights, router="fused_v1")
    logits = mx.array(np.random.default_rng(536).normal(0, 2, (1, 6, EXPERTS)).astype(np.float32))
    for got, want in zip(fused._route(logits, "blk.3"), reference._route(logits, "blk.3")):
        assert np.array_equal(bits(got), bits(want))
    assert fused.calls["router_fused"] == 1 and fused.calls["router_fused_declined"] == 0


@metal
def test_gpu_kda_gate_kernel_is_bitwise_the_reference_chain(gpu):
    rng = np.random.default_rng(537)
    dt_bias = mx.array(rng.normal(-2, 0.2, HEADS * DIM).astype(np.float32))
    ssm_a = mx.array(rng.uniform(1, 8, HEADS).astype(np.float32))
    for length in (1, 5):
        a = rng.normal(0, 3, (1, length, HEADS * DIM)).astype(np.float32)
        a[0, 0, :8] = [0.0, -0.0, np.inf, -np.inf, 2.0, -2.0, 40.0, -40.0]
        a = mx.array(a)
        beta = mx.array(rng.normal(0, 4, (1, length, HEADS)).astype(np.float32))
        gate = mx.array(rng.normal(0, 4, (1, length, HEADS * DIM)).astype(np.float32))
        assert glue.kda_gate_admits(a, dt_bias, ssm_a, beta, gate, heads=HEADS, dim=DIM)
        got = glue.fused_kda_gate(a, dt_bias, ssm_a, beta, gate, heads=HEADS, dim=DIM)
        want = reference_kda_gate(
            a.reshape(1, length, HEADS, DIM), dt_bias, ssm_a, beta,
            gate.reshape(1, length, HEADS, DIM),
        )
        for g, w in zip(got, want):
            mx.eval(g, w)
            assert np.array_equal(bits(g).reshape(-1), bits(w).reshape(-1))


@metal
def test_gpu_kda_with_the_fused_gate_keeps_output_and_state_bitwise(gpu):
    weights = GlueWeights()
    reference, fused = model(weights), model(weights, kda_small="fused_v1")
    x = mx.array(np.random.default_rng(538).normal(size=(1, 4, 4)).astype(np.float32))
    one, two = Glm53KDACache("synthetic"), Glm53KDACache("synthetic")
    want = mx.concatenate([reference._kda(x[:, :3], one, "blk.0"), reference._kda(x[:, 3:], one, "blk.0")], 1)
    got = mx.concatenate([fused._kda(x[:, :3], two, "blk.0"), fused._kda(x[:, 3:], two, "blk.0")], 1)
    mx.eval(want, got, one.state, two.state)
    assert np.array_equal(bits(got), bits(want))
    for a, b in zip(one.state, two.state):
        assert np.array_equal(bits(a), bits(b))
    assert fused.calls["kda_small_fused"] == 2 and fused.calls["kda_small_fused_declined"] == 0
