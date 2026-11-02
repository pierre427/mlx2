"""glm53_kblock_element=fma_v1: the VEC=4 K-block chunk decoder (kblock_dot32_v5).

Q3_K and Q6_K element values come from one exact fused multiply-add instead of
an offset then a multiply; every launch must equal the VEC=1 (kblock_dot32_v2)
launch bit for bit. CPU tests cover the policy wiring; the Metal tests
(MLX2_RUN_GPU_TESTS=1) compare dense, grouped-free multi, routed and
shared-fold MoE launches on random packed blocks whose scales include negative,
tiny (fp16 subnormal) and large values, with zero activations so zero-valued
elements and zero partial sums are exercised.
"""

import os

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.gguf import kblock_metal as km
from mlx2.runtime.gguf.k_blocks import SIZES
from mlx2.runtime.gguf.packed_metal import PackedMatrix
from mlx2.runtime.gguf.reader import GGUFTensor

GPU = os.environ.get("MLX2_RUN_GPU_TESTS") == "1"
Q4_0, Q3_K, Q4_K, Q5_K, Q6_K = 2, 11, 12, 13, 14
# Byte offsets of the fp16 super-block scales inside one block.
D_FIELDS = {Q4_0: (0,), Q3_K: (108,), Q4_K: (0, 2), Q5_K: (0, 2), Q6_K: (208,)}


def _blocks(rng, ggml_type, rows, k, nonfinite=False):
    block, nbytes = SIZES[ggml_type]
    count = rows * (k // block)
    raw = rng.integers(0, 256, size=(count, nbytes), dtype=np.uint8)
    for offset in D_FIELDS[ggml_type]:
        d = rng.choice([1e-3, 2e-2, 0.5, 7.0], size=count) * rng.choice([-1.0, 1.0], size=count)
        d[::17] = 6e-8  # fp16 subnormal
        d[5::23] = 6.0e4
        d[3::29] = 0.0
        if nonfinite:
            d[7::31] = np.inf
            d[11::37] = -np.inf
        raw[:, offset:offset + 2] = d.astype(np.float16).view(np.uint8).reshape(count, 2)
    return raw.reshape(-1)


def _packed(rng, ggml_type, shape, nonfinite=False):
    """PackedMatrix of numpy shape ``shape`` ([experts,] out, in)."""
    rows = int(np.prod(shape[:-1]))
    data = _blocks(rng, ggml_type, rows, shape[-1], nonfinite)
    tensor = GGUFTensor(name="synthetic", shape=tuple(shape[::-1]), ggml_type=ggml_type, offset=0,
                        nbytes=data.size)
    return PackedMatrix(mx.array(data), tensor)


def _x(rng, b, k):
    x = rng.normal(size=(b, k)).astype(np.float32)
    x[:, 32:64] = 0.0  # one all-zero chunk per row
    x[:, 100:300:3] = -0.0
    return mx.array(x)


def bits(a):
    return np.array(a).view(np.uint32)


# ---- CPU ----------------------------------------------------------------------


def _adapter(policy):
    from mlx2.adapters.glm53_gguf import Glm53GGUFAdapter

    adapter = object.__new__(Glm53GGUFAdapter)
    with pytest.raises(FileNotFoundError):  # validation and profile precede the artifact
        adapter._load("/nonexistent/glm53-kblock-element", policy)
    return adapter


def test_adapter_key_validates_needs_kblock_rows_and_marks_the_profile():
    from mlx2.adapters.glm53_gguf import Glm53GGUFAdapter

    rows = {"glm53_row_kernel": "kblock_rows"}
    off = _adapter(rows)
    on = _adapter({**rows, "glm53_kblock_element": "fma_v1"})
    assert off.kernel_policy["glm53_kblock_element"] == "reference" and "kblock_element" not in off.profile
    assert on.kernel_policy["glm53_kblock_element"] == "fma_v1" and ":kblock_element=fma_v1" in on.profile
    for bad in ({"glm53_kblock_element": "fma_v1"},  # reference rows: nothing would engage
                {**rows, "glm53_kblock_element": "fma_v2"}):
        with pytest.raises(ValueError, match="unsupported GLM-5.3 packed execution policy"):
            object.__new__(Glm53GGUFAdapter)._load("/nonexistent", bad)


def test_every_one_row_chunk_law_launch_forwards_the_decoder_choice():
    """weights.py passes vec=self._kblock_vec to each one-row chunk-law launch
    (dense, grouped, routed MoE, multi, shared-fold MoE) and never to the
    token-shared dense path, which needs VEC=1."""
    import inspect
    import re

    from mlx2.runtime.gguf import weights

    source = inspect.getsource(weights)
    for call in ("kblock_dense(flat, weight, rows=4, exact", "kblock_grouped(", "partial(kblock_moe,",
                 "kblock_multi(flats", "kblock_moe_shared(x"):
        line = next(line for line in source.splitlines() if call in line)
        assert "vec=self._kblock_vec" in line, line
    assert re.search(r"kblock_dense\(flat, weight, rows=4, tok=tok\)", source)


def test_v5_decoder_is_selected_only_by_vec_4():
    assert "VEC == 4 ? kblock_dot32_v5" in km.KBLOCK_HEADER
    assert "inline float kblock_dot32_v5" in km.KBLOCK_HEADER


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
@pytest.mark.parametrize("ggml_type", [Q3_K, Q6_K, Q4_K, Q5_K, Q4_0])
@pytest.mark.parametrize("rows", [1, 4, 8])
def test_gpu_dense_vec4_is_bitwise_vec1(gpu, ggml_type, rows):
    rng = np.random.default_rng(5400 + ggml_type + rows)
    for n, k in ((64, 512), (96, 4096)):
        w = _packed(rng, ggml_type, (n, k))
        for b in (1, 3):
            x = _x(rng, b, k)
            want = km.kblock_dense(x, w, rows=rows, vec=1)
            got = km.kblock_dense(x, w, rows=rows, vec=4)
            mx.eval(want, got)
            assert np.isfinite(np.array(want)).any()
            assert np.array_equal(bits(got), bits(want)), (ggml_type, rows, n, k, b)


@metal
def test_gpu_multi_vec4_is_bitwise_vec1(gpu):
    rng = np.random.default_rng(5410)
    k = 2048
    ws = [_packed(rng, Q3_K, (128, k)), _packed(rng, Q6_K, (64, k)), _packed(rng, Q4_K, (32, k))]
    xs = [_x(rng, 1, k) for _ in ws]
    want = km.kblock_multi(xs, ws, vec=1)
    got = km.kblock_multi(xs, ws, vec=4)
    mx.eval(want, got)
    assert np.array_equal(bits(got), bits(want))


@metal
@pytest.mark.parametrize("shared", [False, True], ids=["routed", "shared_fold"])
def test_gpu_moe_vec4_is_bitwise_vec1(gpu, shared):
    rng = np.random.default_rng(5420 + shared)
    experts, m, k = 16, 256, 1024
    gate, up = _packed(rng, Q3_K, (experts, m, k)), _packed(rng, Q3_K, (experts, m, k))
    down = _packed(rng, Q6_K, (experts, k, m))
    x = _x(rng, 2, k)
    ids = mx.array(np.stack([rng.permutation(experts)[:8] for _ in range(2)]).astype(np.uint32))
    scores = mx.array(rng.uniform(0.1, 1.0, (2, 8)).astype(np.float32))
    if shared:
        sh = [_packed(rng, Q3_K, (m, k)), _packed(rng, Q3_K, (m, k)), _packed(rng, Q6_K, (k, m))]
        want = km.kblock_moe_shared(x, ids, scores, gate, up, down, *sh, vec=1)
        got = km.kblock_moe_shared(x, ids, scores, gate, up, down, *sh, vec=4)
    else:
        want = km.kblock_moe(x, ids, scores, gate, up, down, vec=1)
        got = km.kblock_moe(x, ids, scores, gate, up, down, vec=4)
    mx.eval(want, got)
    assert np.array_equal(bits(got), bits(want))


@metal
@pytest.mark.parametrize("ggml_type", [Q3_K, Q6_K])
def test_gpu_nonfinite_scales_keep_v2_values(gpu, ggml_type):
    rng = np.random.default_rng(5430 + ggml_type)
    w = _packed(rng, ggml_type, (64, 2048), nonfinite=True)
    for x in (_x(rng, 2, 2048), mx.ones((1, 2048), mx.float32)):
        want = km.kblock_dense(x, w, vec=1)
        got = km.kblock_dense(x, w, vec=4)
        mx.eval(want, got)
        a, b = np.array(got), np.array(want)
        assert np.isinf(b).any() or np.isnan(b).any()
        # Same bits, and NaN exactly where v2 has NaN.
        assert np.array_equal(np.isnan(a), np.isnan(b))
        assert np.array_equal(bits(got)[~np.isnan(b)], bits(want)[~np.isnan(b)])


@metal
def test_gpu_grouped_and_mixed_q4_0_multi_vec4_is_bitwise_vec1(gpu):
    rng = np.random.default_rng(5440)
    k = 1024
    ws = [_packed(rng, Q4_0, (128, k)), _packed(rng, Q3_K, (64, k)), _packed(rng, Q6_K, (32, k))]
    xs = [_x(rng, 1, k) for _ in ws]
    want, got = km.kblock_multi(xs, ws, vec=1), km.kblock_multi(xs, ws, vec=4)
    mx.eval(want, got)
    assert np.array_equal(bits(got), bits(want))
    grouped = _packed(rng, Q3_K, (4, 64, k))
    x = mx.array(rng.normal(size=(2, 4, k)).astype(np.float32))
    want, got = km.kblock_grouped(x, grouped, vec=1), km.kblock_grouped(x, grouped, vec=4)
    mx.eval(want, got)
    assert np.array_equal(bits(got), bits(want))


def test_prompt_eval_policy_validates_and_defaults_off():
    from mlx2.adapters.glm53_gguf import Glm53GGUFAdapter

    assert _adapter({}).kernel_policy["glm53_prompt_eval"] == "off"
    assert _adapter({"glm53_prompt_eval": "layer_v1"}).kernel_policy["glm53_prompt_eval"] == "layer_v1"
    with pytest.raises(ValueError, match="unsupported GLM-5.3 packed execution policy"):
        object.__new__(Glm53GGUFAdapter)._load("/nonexistent", {"glm53_prompt_eval": "chunk"})


def test_command_buffer_policy_requires_prompt_layer_eval_on_glm(monkeypatch):
    from mlx2.adapters.glm53_gguf import Glm53GGUFAdapter

    monkeypatch.setenv("MLX_MAX_MB_PER_BUFFER", "1000000")  # however it was raised
    with pytest.raises(ValueError, match="requires glm53_prompt_eval=layer_v1"):
        object.__new__(Glm53GGUFAdapter)._load("/nonexistent", {})
    assert _adapter({"glm53_prompt_eval": "layer_v1"}).kernel_policy["glm53_prompt_eval"] == "layer_v1"
    monkeypatch.delenv("MLX_MAX_MB_PER_BUFFER")
    with pytest.raises(ValueError, match="external-draft"):
        object.__new__(Glm53GGUFAdapter)._load("/nonexistent", {"glm53_prompt_eval": "layer_v1",
                                                                 "draft_model": "mtp", "num_draft": 2})


def test_command_buffer_switches_are_recorded_by_serving():
    from mlx2.runtime.env_switches import serving_env_switches

    env = {"MLX2_COMMAND_BUFFERS": "ops100_v1", "MLX_MAX_OPS_PER_BUFFER": "100", "MLX_MAX_MB_PER_BUFFER": "1000000"}
    assert serving_env_switches(env) == env
    assert serving_env_switches({}) == {}
