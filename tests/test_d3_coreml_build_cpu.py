"""Pure CPU contracts of the D3 Core ML feasibility builder (no Core ML, no MLX)."""

import json
import subprocess
import sys

import numpy as np
import pytest

from mlx2.decisions.candidates import d3_coreml_build as B


def _gdn_inputs(rng, T, H, D, *, decay_scale=0.3):
    k = rng.standard_normal((T, H, D))
    k /= np.linalg.norm(k, axis=-1, keepdims=True)
    q = rng.standard_normal((T, H, D))
    q /= np.linalg.norm(q, axis=-1, keepdims=True) * np.sqrt(D)
    v = rng.standard_normal((T, H, D))
    log_g = -np.abs(rng.standard_normal((T, H))) * decay_scale
    beta = 1.0 / (1.0 + np.exp(-rng.standard_normal((T, H))))
    return q, k, v, log_g, beta


@pytest.mark.parametrize("T,chunk,block", [(200, 64, 16), (64, 64, 16), (37, 16, 4), (130, 32, 8)])
def test_chunked_gdn_matches_sequential_scan(T, chunk, block):
    rng = np.random.default_rng(T)
    q, k, v, log_g, beta = _gdn_inputs(rng, T, 3, 16)
    expected, expected_state = B.gdn_sequential_numpy(q, k, v, log_g, beta)
    actual, state = B.gdn_chunked_numpy(q, k, v, log_g, beta, chunk=chunk, block=block)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
    np.testing.assert_allclose(state, expected_state, rtol=0, atol=1e-10)


def test_chunked_gdn_carries_initial_state_and_strong_decay():
    rng = np.random.default_rng(5)
    q, k, v, log_g, beta = _gdn_inputs(rng, 150, 2, 8, decay_scale=6.0)
    initial = rng.standard_normal((2, 8, 8))
    expected, expected_state = B.gdn_sequential_numpy(q, k, v, log_g, beta, state=initial)
    actual, state = B.gdn_chunked_numpy(q, k, v, log_g, beta, state=initial)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
    np.testing.assert_allclose(state, expected_state, rtol=0, atol=1e-10)


def test_right_padding_never_reaches_real_positions():
    rng = np.random.default_rng(9)
    q, k, v, log_g, beta = _gdn_inputs(rng, 100, 2, 8)
    real, _ = B.gdn_sequential_numpy(q[:70], k[:70], v[:70], log_g[:70], beta[:70])
    padded, _ = B.gdn_chunked_numpy(q, k, v, log_g, beta)
    np.testing.assert_allclose(padded[:70], real, rtol=0, atol=1e-10)


@pytest.mark.parametrize("chunk,block", [(64, 16), (64, 64), (16, 4), (32, 32)])
def test_blocked_unit_lower_inverse_is_exact(chunk, block):
    rng = np.random.default_rng(chunk + block)
    strict = np.tril(rng.uniform(-0.8, 0.8, (4, chunk, chunk)), -1)
    expected = np.linalg.inv(np.eye(chunk) + strict)
    np.testing.assert_allclose(B.unit_lower_inverse(strict, block=block), expected, atol=1e-8)


def test_unit_lower_inverse_rejects_non_power_of_two_blocks():
    with pytest.raises(ValueError):
        B.unit_lower_inverse(np.zeros((48, 48)), block=16)


def test_decay_sum_matrix_gives_cancellation_free_differences():
    chunk = 8
    log_g = -np.arange(1, chunk + 1, dtype=np.float64)
    gamma = np.cumsum(log_g)
    differences = (log_g @ B.decay_sum_matrix(chunk).T).reshape(chunk, chunk)
    for i in range(chunk):
        for j in range(chunk):
            expected = gamma[i] - gamma[j] if i >= j else 0.0
            assert differences[i, j] == pytest.approx(expected)


def test_bucket_selection_and_refusal():
    assert B.bucket_for(1) == 128
    assert B.bucket_for(128) == 128
    assert B.bucket_for(129) == 256
    assert B.bucket_for(1024) == 1024
    with pytest.raises(ValueError, match="no truncation"):
        B.bucket_for(1025)
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError):
            B.bucket_for(bad)
    assert B.validate_buckets([64, 128, 1024]) == (64, 128, 1024)
    for bad in ([], [128, 128], [256, 128], [100], [2048], list(range(64, 640, 64))):
        with pytest.raises(ValueError):
            B.validate_buckets(bad)
    with pytest.raises(TypeError):
        B.validate_buckets("128")


def test_length_distribution_reports_buckets_and_padding():
    summary = B.length_distribution([103, 127, 300, 870, 1500])
    assert summary["per_bucket"] == {"128": 2, "512": 1, "1024": 1, "over": 1}
    assert summary["min"] == 103 and summary["max"] == 1500
    assert 0.0 < summary["padding_fraction_mean"] < 1.0


def test_readout_select_is_one_hot_at_last_real_token():
    mask = np.array([[1, 1, 1, 0, 0]])
    np.testing.assert_array_equal(B.readout_select(mask), [[0, 0, 1, 0, 0]])
    full = np.ones((1, 4), int)
    np.testing.assert_array_equal(B.readout_select(full), [[0, 0, 0, 1]])
    assert B.check_right_padded(mask) == 3
    for bad in ([[0, 1, 1]], [[1, 0, 1]], [[0, 0, 0]]):
        with pytest.raises(ValueError):
            B.check_right_padded(np.array(bad))


def test_full_head_rope_tables_match_partial_rope():
    rng = np.random.default_rng(3)
    length = 11
    x = rng.standard_normal((2, length, B.HEAD_DIM)).astype(np.float32)
    cosine, sine = B.rope_tables(length)
    expected = B._rope(x, cosine, sine)
    full_cos, full_sin, permutation = B.rope_full_tables(length)
    actual = x * full_cos + (x @ permutation) * full_sin
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(actual[..., B.ROPE_DIMS:], x[..., B.ROPE_DIMS:])


def test_inverse_op_names_and_variants():
    assert B.is_inverse_op("gdncore3inv_12_matmul")
    assert B.is_inverse_op("gdncore10inv_1_sub_cast_fp16")
    assert not B.is_inverse_op("gdncore3_12_matmul")
    assert not B.is_inverse_op("linear_5")
    assert not B.is_inverse_op("gdncoreinv_1_mul")
    assert B.PRECISION_VARIANTS == ("fp16", "fp32_gdn", "fp32_inverse")


def test_decision_probabilities_follow_served_readout_law():
    rng = np.random.default_rng(1)
    hidden = rng.standard_normal(B.HIDDEN).astype(np.float32)
    readout = rng.standard_normal((B.READOUT_ROWS, B.HIDDEN)).astype(np.float32)
    probabilities = B.decision_probabilities(hidden, readout, 3, temperature=1.0)
    logits = readout[:3] @ hidden
    expected = np.exp(logits - logits.max())
    np.testing.assert_allclose(probabilities, expected / expected.sum(), rtol=1e-5)
    assert probabilities.shape == (3,)


def test_parity_cases_are_varied_realistic_requests():
    cases = B.PARITY_CASES
    assert len(cases) >= 30
    assert {question["type"] for _, question in cases} == {"choice", "noul", "score"}
    keys = [json.dumps([state, question], sort_keys=True) for state, question in cases]
    assert len(set(keys)) == len(keys)
    for state, _ in cases:
        text = state if isinstance(state, str) else json.dumps(state)
        if not text:
            continue
        sentences = [part.strip() for part in text.replace("\n", ". ").split(". ") if part.strip()]
        # No repeated-string filler: no sentence repeats within a state.
        assert len(sentences) == len(set(sentences))


def test_ane_failure_marker_detection():
    assert B.ane_compile_failed(
        "E5RT encountered an STL exception. msg = MILCompilerForANE error: failed to "
        "compile ANE model using ANEF. Error=_ANECompiler : ANECCompile() FAILED."
    )
    assert not B.ane_compile_failed("")
    assert not B.ane_compile_failed(None)


def test_gpu_lock_state_is_read_only_and_blocks_benchmarks(tmp_path):
    assert B.gpu_lock_state(tmp_path / "absent.json")["latency_allowed"] is True
    owner = tmp_path / "owner.json"
    owner.write_text(json.dumps({"label": "ordinary-b1-ladder"}))
    assert B.gpu_lock_state(owner) == {"held": True, "label": "ordinary-b1-ladder",
                                       "latency_allowed": False}
    owner.write_text(json.dumps({"label": "q8-component"}))
    assert B.gpu_lock_state(owner)["latency_allowed"] is True


def test_module_import_pulls_in_neither_coremltools_nor_mlx():
    code = (
        "import sys\n"
        "import mlx2.decisions.candidates.d3_coreml_build as module\n"
        "loaded = sorted(name for name in ('coremltools', 'mlx', 'mlx.core') if name in sys.modules)\n"
        "print(loaded)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            timeout=120, check=True)
    assert result.stdout.strip().splitlines()[-1] == "[]"
