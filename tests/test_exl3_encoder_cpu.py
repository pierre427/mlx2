"""EXL3 encoder (scripts/research/ds41_exl3): layout, round trip, oracles. CPU only."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/research/ds41_exl3"))
import exl3_encode as E  # noqa: E402
from mlx2.runtime.models.exl3_reference import decode_inner, effective_weight  # noqa: E402

@pytest.fixture(autouse=True, scope="module")
def _viterbi_builds_outside_the_tree(tmp_path_factory):
    """The C Viterbi compiles on first use into EXL3_ENC_BUILD (default: the
    gitignored scripts/research/ds41_exl3/.build). The preflight fingerprints
    scripts/ by glob, ignored files included, so a build landing there during
    the suite made a fresh checkout's receipt report the source as changed."""
    previous = os.environ.get("EXL3_ENC_BUILD")
    os.environ["EXL3_ENC_BUILD"] = str(tmp_path_factory.mktemp("exl3_enc_build"))
    yield
    if previous is None:
        os.environ.pop("EXL3_ENC_BUILD", None)
    else:
        os.environ["EXL3_ENC_BUILD"] = previous


FIXTURES = sorted(str(p) for p in (ROOT / "scripts/sushi_probe/fixtures").glob("exl3_*.safetensors"))


def _fixture(path):
    from safetensors.numpy import load_file

    f = load_file(path)
    cb = "mcg" if "mcg" in path else "mul1"
    window = 12 if "w12" in path else 16
    return f, cb, window


def test_fixtures_present():
    if not (ROOT / "scripts/sushi_probe").is_dir():
        pytest.skip("the Sushi probe fixtures are not part of this checkout")
    assert len(FIXTURES) == 6


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: Path(p).stem)
def test_bitstream_pack_round_trips_exllamav3_fixtures(path):
    f, cb, window = _fixture(path)
    t = f["trellis"]
    cw = E.unpack_codewords(t)
    assert np.array_equal(E.pack_states(cw, t.shape[-1]), t)
    mine = E.matrix_from_tiles(E.decode_states(cw, cb, window))
    assert np.array_equal(mine.astype(np.float16), decode_inner(t, cb, window))


@pytest.mark.parametrize("path", FIXTURES, ids=lambda p: Path(p).stem)
def test_viterbi_reencodes_fixture_bit_exact(path):
    # Targets that are exact codebook points of a valid stream: the encoder
    # must find a zero-error path, and here it reproduces exllamav3's bits.
    f, cb, window = _fixture(path)
    t = f["trellis"]
    inner = decode_inner(t, cb, window).astype(np.float32)
    st = E.viterbi(E.tiles_from_matrix(inner).reshape(-1, 256), t.shape[-1], cb, window)
    tr = E.pack_states(st.reshape(t.shape[0], t.shape[1], 256), t.shape[-1])
    assert np.array_equal(decode_inner(tr, cb, window), inner.astype(np.float16))
    assert np.array_equal(tr, t)


def test_tile_layout_inverse():
    m = np.arange(32 * 48, dtype=np.float64).reshape(32, 48)
    t = E.tiles_from_matrix(m)
    assert t.shape == (2, 3, 256)
    assert np.array_equal(E.matrix_from_tiles(t), m)


@pytest.mark.parametrize("words", [32, 40, 48, 64])
def test_c_viterbi_matches_numpy_oracle(words):
    rng = np.random.default_rng(words)
    tg = (rng.standard_normal((2, 256)) * E.codebook_rms()).astype(np.float32)
    st = E.viterbi(tg, words, threads=1)
    ref = np.stack([E.viterbi_reference(x, words) for x in tg])
    assert np.array_equal(st, ref)


def test_viterbi_streams_are_valid_and_near_qtip_mse():
    # Gaussian source: QTIP reports ~0.069 (K=2) / ~0.0175 (K=3) MSE for L=16 codes.
    rng = np.random.default_rng(0)
    tg = (rng.standard_normal((48, 256)) * E.codebook_rms()).astype(np.float32)
    for words, ceiling in ((32, 0.075), (48, 0.019)):
        st = E.viterbi(tg, words)
        tr = E.pack_states(st[None], words)
        # Re-reading the packed stream yields the same codewords (tail-biting valid).
        assert np.array_equal(E.unpack_codewords(tr)[0], st)
        mse = np.mean((E.decode_states(st) - tg) ** 2) / np.var(tg)
        assert mse < ceiling


def test_encode_linear_decodes_through_mlx2_reference():
    rng = np.random.default_rng(3)
    I, O = 256, 128
    w = rng.standard_normal((I, O)) * 0.02
    x = rng.standard_normal((2048, I)) @ (np.eye(I) + 0.3 * rng.standard_normal((I, I)) / np.sqrt(I))
    hx = x.T @ x / len(x)
    out = E.encode_linear(w, hx, 48)
    assert out["trellis"].shape == (I // 16, O // 16, 48) and out["trellis"].dtype == np.uint16
    assert out["suh"].dtype == np.float16 and out["suh"].shape == (I,)
    assert out["svh"].dtype == np.float16 and out["svh"].shape == (O,)
    ref = effective_weight(out["trellis"], out["suh"], out["svh"], "mcg")
    assert np.allclose(ref, out["weight"], rtol=0, atol=1e-9 * np.abs(w).max())
    assert out["stats"]["weight_rel_err"] < 0.2


def test_ldlq_lowers_hessian_proxy_error():
    rng = np.random.default_rng(4)
    I, O = 256, 128
    w = rng.standard_normal((I, O)) * 0.02
    mix = rng.standard_normal((I, I)) / np.sqrt(I)
    x = rng.standard_normal((4096, I)) @ (np.eye(I) * 0.2 + mix)
    hx = x.T @ x / len(x)
    plain = E.encode_linear(w, hx, 32, ldlq=False)["stats"]["proxy_rel_err"]
    ldlq = E.encode_linear(w, hx, 32, ldlq=True)["stats"]["proxy_rel_err"]
    assert ldlq < plain * 0.95


@pytest.mark.parametrize("shape", [(128, 128), (256, 384), (640, 2304), (2304, 640)])
def test_threaded_hadamard_is_bit_identical_to_reference(shape):
    from mlx2.runtime.models.exl3_reference import hadamard128

    a = np.random.default_rng(shape[0]).standard_normal(shape)
    assert np.array_equal(E.hadamard_cols(a), hadamard128(a))
    assert np.array_equal(E.hadamard_rows(a), hadamard128(a.T).T)


def test_lockstep_and_grouped_cpu_encodes_agree():
    rng = np.random.default_rng(9)
    items = []
    for I, O in ((256, 128), (128, 256)):
        w = rng.standard_normal((I, O)) * 0.02
        x = rng.standard_normal((1024, I))
        items.append((w, x.T @ x / len(x)))
    a = E.encode_many(items, 32, seed=5)
    b = E.encode_many(items, 32, seed=5, groups=[0, 1])
    for p, q in zip(a, b):
        assert np.array_equal(p["trellis"], q["trellis"]) and np.array_equal(p["svh"], q["svh"])


def test_threaded_hadamard_leaves_input_untouched():
    a = np.random.default_rng(1).standard_normal((256, 128))
    keep = a.copy()
    E.hadamard_rows(a)
    E.hadamard_cols(a)
    assert np.array_equal(a, keep)
