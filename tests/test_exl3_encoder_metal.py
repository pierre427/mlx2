"""Real-Metal gate for the EXL3 encoder's Metal Viterbi (scripts/research/ds41_exl3).

Run with ``MLX2_RUN_METAL_TESTS=1`` in an owned GPU window. Proves the v2
search (register-resident costs, threadgroup exchange, fused k=2 steps) is
bit-identical to the C search, to the v1 Metal search and to the exllamav3
MCG fixtures, and that the async/pipelined encoders reproduce encode_many on
the CPU. Synthetic tiles and small matrices only: no speed claim.
"""

import glob
import os
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/research/ds41_exl3"))
import exl3_encode as E  # noqa: E402
import viterbi_metal as V  # noqa: E402
from mlx2.runtime.models.exl3_reference import decode_inner  # noqa: E402

pytestmark = pytest.mark.skipif(
    os.environ.get("MLX2_RUN_METAL_TESTS") != "1" or not mx.metal.is_available(),
    reason="set MLX2_RUN_METAL_TESTS=1 (owned GPU window) for the real-Metal gate",
)

MCG16 = sorted(p for p in glob.glob(str(ROOT / "scripts/sushi_probe/fixtures/exl3_*.safetensors"))
               if "mcg" in p and "w12" not in p)


@pytest.fixture(autouse=True)
def v2():
    prev = V.KERNEL_VERSION
    V.KERNEL_VERSION = "v2"
    yield
    V.KERNEL_VERSION = prev


def _tiles(seed, n, scale=1.0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, 256)) * E.codebook_rms() * scale).astype(np.float32)


def test_v2_codebook_matches_table_exhaustively():
    src = "uint S = thread_position_in_grid.x; out[S] = cbv(S * MCG);"
    k = mx.fast.metal_kernel(name="exl3_cbv_check", input_names=["dummy"], output_names=["out"],
                             source=src, header=V._HEADER_V2)
    (o,) = k(inputs=[mx.zeros(1)], grid=(65536, 1, 1), threadgroup=(256, 1, 1),
             output_shapes=[(65536,)], output_dtypes=[mx.float32], stream=mx.gpu)
    assert np.array_equal(np.array(o), E.codebook_values())


@pytest.mark.parametrize("words", [32, 36, 40, 44, 48, 56, 64])
@pytest.mark.parametrize("probe", [2, 8, 32, 128])
def test_v2_matches_c_search(words, probe):
    assert V.v2_supported(words)
    tg = _tiles(words * 1000 + probe, 24)
    assert np.array_equal(V.viterbi_metal(tg, words, probe=probe), E.viterbi(tg, words, probe=probe))


@pytest.mark.parametrize("words", [32, 40, 48, 64])
def test_v2_matches_v1_and_c_on_a_large_batch(words):
    tg = _tiles(words, 700, scale=1.7)  # outliers past the codebook range too
    c = E.viterbi(tg, words)
    assert np.array_equal(V.viterbi_metal(tg, words, batch=256), c)  # several dispatches
    V.KERNEL_VERSION = "v1"
    assert np.array_equal(V.viterbi_metal(tg, words), c)


@pytest.mark.parametrize("words", [32, 48, 64])
def test_v2_tie_heavy_targets_match_c(words):
    # Zero tiles, constant tiles and exact codebook points make many equal
    # path costs: the first-index / lowest-state tie rules must agree with C.
    cbv = E.codebook_values()
    rng = np.random.default_rng(7)
    tg = np.stack([np.zeros(256), np.full(256, cbv[12345]), cbv[rng.integers(0, 65536, 256)],
                   np.round(_tiles(1, 1)[0] * 4) / 4]).astype(np.float32)
    assert np.array_equal(V.viterbi_metal(tg, words), E.viterbi(tg, words))


@pytest.mark.parametrize("path", MCG16, ids=lambda p: Path(p).stem)
def test_v2_reencodes_mcg_fixture_bit_exact(path):
    from safetensors.numpy import load_file

    t = load_file(path)["trellis"]
    words = t.shape[-1]
    assert V.v2_supported(words)
    inner = decode_inner(t, "mcg", 16).astype(np.float32)
    st = V.viterbi_metal(E.tiles_from_matrix(inner).reshape(-1, 256), words)
    assert np.array_equal(E.pack_states(st.reshape(t.shape[0], t.shape[1], 256), words), t)


def test_mcg_fixtures_present():
    assert len(MCG16) == 2


def _problem(seed, I=256, O=128):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((I, O)) * 0.02
    x = rng.standard_normal((2048, I)) @ (np.eye(I) + 0.3 * rng.standard_normal((I, I)) / np.sqrt(I))
    return w, x.T @ x / len(x)


def _same(a, b):
    for p, q in zip(a, b):
        assert np.array_equal(p["trellis"], q["trellis"])
        assert np.array_equal(p["suh"], q["suh"]) and np.array_equal(p["svh"], q["svh"])
        assert np.array_equal(p["weight"], q["weight"])


@pytest.mark.parametrize("words", [32, 40])
def test_pipelined_metal_encode_equals_cpu_encode(words):
    items = [_problem(1), _problem(2, 128, 256), _problem(3, 384, 128)]
    cpu = E.encode_many(items, words, seed=11, backend="cpu")
    _same(E.encode_many(items, words, seed=11, backend="metal"), cpu)  # one group per matrix
    _same(E.encode_many(items, words, seed=11, backend="metal", groups=[0, 0, 0]), cpu)


@pytest.mark.parametrize("group", ["unit", "job"])
def test_encode_experts_stream_equals_encode_many(group):
    units = [([_problem(4), _problem(5)], 100), ([_problem(6, 128, 256)], 200), ([_problem(7)], 300)]
    ref = [E.encode_many(items, 32, seed=seed, backend="cpu") for items, seed in units]
    got = list(E.encode_experts(((lambda u=u: u) for u in units), 32, inflight=2, group=group))
    assert len(got) == len(ref)
    for g, r in zip(got, ref):
        _same(g, r)


def test_encode_experts_per_unit_bit_rate_and_guards():
    units = [([_problem(8)], 10, 48), ([_problem(9)], 20), ([_problem(10, 128, 256)], 30, 40)]
    ref = [E.encode_many(u[0], u[2] if len(u) > 2 else 32, seed=u[1], backend="cpu") for u in units]
    got = list(E.encode_experts(iter(units), 32, inflight=1))
    for g, r in zip(got, ref):
        _same(g, r)
    with pytest.raises(ValueError):
        list(E.encode_experts(iter(units), 32, inflight=0))
    with pytest.raises(ValueError):
        list(E.encode_experts(iter([([], 1)]), 32))
