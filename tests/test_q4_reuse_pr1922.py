"""CPU/static checks for the research-only PR #1922 comparator."""

import importlib.util
from pathlib import Path

import numpy as np


PATH = Path(__file__).resolve().parents[1] / "scripts/research/bench_q4_reuse_pr1922.py"
SPEC = importlib.util.spec_from_file_location("bench_q4_reuse_pr1922", PATH)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)
OMLX_PATH = PATH.with_name("omlx3958_ksplit.py")
OMLX_SPEC = importlib.util.spec_from_file_location("omlx3958_ksplit", OMLX_PATH)
OMLX = importlib.util.module_from_spec(OMLX_SPEC)
OMLX_SPEC.loader.exec_module(OMLX)


def test_packed_math_reuses_same_weight_for_each_batch_row():
    rng = np.random.default_rng(1922)
    m, n, k, group = 8, 12, 128, 64
    x = rng.normal(size=(m, k))
    packed = rng.integers(0, 2**32, size=(n, k // 8), dtype=np.uint32)
    scales = rng.uniform(0.001, 0.03, size=(n, k // group))
    biases = rng.uniform(-0.01, 0.01, size=(n, k // group))
    got = BENCH.packed_reference(x, packed, scales, biases, group)
    q = np.empty((n, k), dtype=np.float64)
    for j in range(k):
        value = (packed[:, j // 8] >> np.uint32(4 * (j % 8))) & 15
        q[:, j] = value * scales[:, j // group] + biases[:, j // group]
    np.testing.assert_allclose(got, np.einsum("mk,nk->mn", x, q), atol=1e-12, rtol=1e-12)


def test_admission_and_expected_stock_dispatch():
    assert BENCH.eligible(4, 1024, 5120, 4, 64, "bfloat16")
    assert BENCH.eligible(8, 5120, 17408, 4, 64, "float16")
    for args in ((3, 1024, 5120, 4, 64, "bfloat16"),
                 (9, 1024, 5120, 4, 64, "bfloat16"),
                 (4, 1023, 5120, 4, 64, "bfloat16"),
                 (4, 1024, 5120, 5, 64, "bfloat16"),
                 (4, 1024, 5120, 4, 64, "float32")):
        assert not BENCH.eligible(*args)
    assert BENCH.stock_path(8, 1024, 5120) == "qmv_wide"
    assert BENCH.stock_path(8, 17408, 5120) == "qmv_nax"
    assert BENCH.comparison_scope(8, 1024, 5120) == "non_nax_primary"
    assert BENCH.comparison_scope(8, 17408, 5120) == "nax_contaminated_exploratory"
    assert BENCH.plan(["attn_kv_b4"], [4, 6, 8])[-1]["sp_qmm_policy_expected"]
    assert OMLX.eligible(4, 17408, 5120)
    assert OMLX.eligible(6, 17408, 5120)
    assert not OMLX.eligible(8, 17408, 5120)
    assert not OMLX.eligible(4, 5120, 17408)
