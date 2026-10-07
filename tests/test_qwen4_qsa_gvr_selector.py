"""Exactness of the GVR-style self-sampling QSA selector against the selector law.

CPU cases exercise the NumPy mirror of the kernel algorithm
(``qwen4_qsa_gvr_reference``) on adversarial inputs and check the policy wiring.
The Metal cases run only with ``MLX2_RUN_GPU_TESTS=1`` (while holding the GPU
lease): the kernel must equal direct8 and the selector law, and its per-row
diagnostics must equal the NumPy mirror's, row for row.
"""

from __future__ import annotations

import inspect
import os
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from mlx2.runtime.models import qwen4_qsa_gvr_reference as gvr
from mlx2.runtime.models import qwen4_qsa_selector as selector
from mlx2.runtime.models import qwen4_qsa_stage1 as stage1

RATIO = 4
GPU = os.environ.get("MLX2_RUN_GPU_TESTS") == "1"


def _full(rows: int, blocks: int) -> np.ndarray:
    return np.full(rows, blocks * RATIO - 1, dtype=np.int32)


def _ordered_float(value: float) -> int:
    bits = struct.unpack("I", struct.pack("f", value))[0]
    return ~bits & 0xFFFFFFFF if bits & 0x80000000 else bits ^ 0x80000000


def _brute_force_law(values, qpos, topk):
    """Independent pure-Python statement of the selector law for one row."""
    blocks = len(values)
    complete = int((qpos + 1) / RATIO) if qpos + 1 >= 0 else 0
    valid = min(blocks, max(0, complete))
    order = sorted(
        range(valid),
        key=lambda index: (_ordered_float(values[index]), index),
        reverse=True,
    )
    chosen = sorted(order[: min(topk, valid)])
    invalid = topk - len(chosen)
    return chosen + [blocks - invalid + slot for slot in range(invalid)]


def _cases():
    rng = np.random.default_rng(20261006)
    big = 4 * 2048 + 3  # beyond capacity so the sampled path runs
    k = 544

    def scores(values):
        return np.asarray(values, dtype=np.float32)

    random = rng.random((3, big), dtype=np.float32)
    all_equal = np.full((2, big), 0.25, dtype=np.float32)
    zeros = np.zeros((2, big), dtype=np.float32)

    # Dense near-tie band: a few hundred keys strictly above, then thousands of
    # copies of the k-th value and its float neighbours around the threshold.
    band = rng.random((2, big), dtype=np.float32) * 0.5
    band[:, rng.choice(big, 400, replace=False)] = 0.9
    tie_value = np.float32(0.75)
    tied = rng.choice(big, 3000, replace=False)
    band[:, tied] = tie_value
    band[0, tied[:50]] = np.nextafter(tie_value, np.float32(1.0))
    band[1, tied[:50]] = np.nextafter(tie_value, np.float32(0.0))

    # The same band shape small enough to stay on the sampled path: 300 keys
    # above and 700 exact ties straddling the k-th key, so 244 of the ties
    # are chosen by the largest-block-ID rule.
    band_fit = rng.random((2, big), dtype=np.float32) * 0.5
    above = rng.choice(big, 1000, replace=False)
    band_fit[:, above[:300]] = 0.9
    band_fit[:, above[300:]] = tie_value
    band_fit[1, above[300:340]] = np.nextafter(tie_value, np.float32(0.0))

    # Ties that fit the candidate capacity: the sampled path must fill ties
    # from the largest block IDs.
    fit = np.floor(rng.random((3, big), dtype=np.float32) * 64.0) / 64.0

    outliers = rng.standard_normal((2, big)).astype(np.float32)
    outliers[0, :7] = np.float32(3.0e38)
    outliers[0, 7:20] = np.float32(np.inf)
    outliers[0, 20:30] = np.float32(-np.inf)
    outliers[1, ::97] = np.float32(1.0e-42)  # denormal
    outliers[1, 1::97] = np.float32(-0.0)
    outliers[1, 2::97] = np.float32(0.0)

    sentinel = rng.random((3, big), dtype=np.float32)
    sentinel[0, big // 3 :] = -np.inf
    sentinel[1, 600:] = -np.inf
    sentinel[2, :] = -np.inf

    nan = rng.random((2, big), dtype=np.float32)
    nan_bits = nan.view(np.uint32)
    nan_bits[0, ::31] = np.uint32(0x7FC00000)  # +NaN orders above +inf
    nan_bits[1, ::29] = np.uint32(0xFFC00000)  # -NaN orders below -inf
    nan_bits[1, 5] = np.uint32(0xFFFFFFFF)  # order key 0

    heavy_relu = np.maximum(rng.standard_normal((2, big)), 0).astype(np.float32)

    cases = [
        ("random", random, _full(3, big), k),
        ("random_k512", random, _full(3, big), 512),
        ("all_equal", all_equal, _full(2, big), k),
        ("zeros", zeros, _full(2, big), k),
        ("near_tie_band", band, _full(2, big), k),
        ("near_tie_band_fits", band_fit, _full(2, big), k),
        ("fit_ties", fit, _full(3, big), k),
        ("outliers", outliers, _full(2, big), k),
        ("sentinel_padding", sentinel, _full(3, big), k),
        ("nan_bits", nan, _full(2, big), k),
        ("relu_zero_mass", heavy_relu, _full(2, big), k),
    ]
    for blocks in (k - 1, k, k + 1, 2047, 2048, 2049, 4097):
        values = rng.random((2, blocks), dtype=np.float32)
        values[1] = np.floor(values[1] * 8.0) / 8.0
        cases.append((f"blocks_{blocks}", values, _full(2, blocks), k))
    small = rng.random((7, big), dtype=np.float32)
    small_positions = np.array(
        [-5, 0, RATIO - 2, RATIO - 1, (k - 1) * RATIO - 1, k * RATIO, 2100 * RATIO],
        dtype=np.int32,
    )
    cases.append(("small_valid_count", small, small_positions, k))
    for index in range(4):
        rows = rng.integers(1, 4)
        blocks = int(rng.integers(600, 12000))
        values = rng.standard_normal((rows, blocks)).astype(np.float32)
        positions = rng.integers(0, blocks * RATIO, size=rows).astype(np.int32)
        cases.append((f"random_partial_{index}", values, positions, k))
    return cases


CASES = _cases()


def test_selector_law_matches_independent_brute_force():
    rng = np.random.default_rng(3)
    values = np.floor(rng.random((3, 300), dtype=np.float32) * 5) / 5
    values[0, 10] = np.float32(-0.0)
    values[0, 11] = np.float32(0.0)
    values[2, ::3] = np.float32(np.nan)
    positions = np.array([300 * RATIO - 1, 70 * RATIO - 1, 2], dtype=np.int32)
    law = gvr.selector_law(values, positions, topk=64, compress_ratio=RATIO)
    for row in range(3):
        expected = _brute_force_law(values[row].tolist(), int(positions[row]), 64)
        assert law[row].tolist() == expected


@pytest.mark.parametrize("name,values,positions,topk", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("capacity", [1024, 2048])
def test_reference_algorithm_is_exact_on_adversarial_inputs(
    name, values, positions, topk, capacity
):
    del name
    ids, diag = gvr.gvr_reference_select(
        values, positions, topk=topk, compress_ratio=RATIO, capacity=capacity
    )
    law = gvr.selector_law(values, positions, topk=topk, compress_ratio=RATIO)
    np.testing.assert_array_equal(ids, law)
    assert set(diag[:, 0].tolist()) <= set(gvr.PATH_NAMES)
    sampled = diag[:, 0] == gvr.PATH_SAMPLED
    assert np.all(diag[sampled, 1] <= capacity)
    assert np.all(diag[sampled, 1] >= np.minimum(topk, values.shape[1]))


def test_paths_cover_dense_sampled_fallback_and_empty():
    by_name = {case[0]: case for case in CASES}

    def paths(name):
        _, values, positions, topk = by_name[name]
        _, diag = gvr.gvr_reference_select(
            values, positions, topk=topk, compress_ratio=RATIO
        )
        return diag[:, 0].tolist()

    assert set(paths("random")) == {gvr.PATH_SAMPLED}
    assert set(paths("all_equal")) == {gvr.PATH_FALLBACK}
    assert set(paths("zeros")) == {gvr.PATH_FALLBACK}
    assert set(paths("blocks_2048")) == {gvr.PATH_DENSE}
    assert set(paths("fit_ties")) == {gvr.PATH_SAMPLED}
    assert set(paths("near_tie_band")) == {gvr.PATH_FALLBACK}
    assert set(paths("near_tie_band_fits")) == {gvr.PATH_SAMPLED}
    assert paths("small_valid_count")[:3] == [gvr.PATH_EMPTY] * 3
    assert paths("small_valid_count")[3] == gvr.PATH_DENSE


def test_threshold_ranks_and_fractions_are_monotone():
    config = gvr.GvrConfig(544, RATIO, 1024, 2048)
    fractions = config.fractions256
    assert list(fractions) == sorted(fractions)
    assert fractions[0] >= gvr.FRACTION_SCALE
    assert fractions[-1] * 544 <= 2048 * gvr.FRACTION_SCALE
    ranks = gvr.threshold_ranks(544, 32768, config)
    assert ranks == sorted(ranks)
    assert 0 <= ranks[0] and ranks[-1] < config.samples
    positions = gvr.sample_positions(32768, config)
    assert positions[0] == 0 and positions[-1] < 32768
    assert np.all(np.diff(positions) == 32)


def test_continuous_rows_rarely_fall_back():
    rng = np.random.default_rng(11)
    values = rng.random((32, 32768), dtype=np.float32)
    _, diag = gvr.gvr_reference_select(
        values, _full(32, 32768), topk=544, compress_ratio=RATIO
    )
    assert np.count_nonzero(diag[:, 0] == gvr.PATH_FALLBACK) == 0
    assert np.median(diag[:, 1]) < 1024


@pytest.mark.parametrize(
    "kwargs",
    [
        {"topk": 0, "ratio": 4},
        {"topk": 1025, "ratio": 4},
        {"topk": 512, "ratio": 0},
        {"topk": 512, "ratio": 4, "width": 384},
        {"topk": 512, "ratio": 4, "capacity": 1536},
        {"topk": 512, "ratio": 4, "width": 256, "capacity": 2048},
    ],
)
def test_config_rejects_unsupported_geometry(kwargs):
    with pytest.raises(ValueError):
        gvr.GvrConfig(**kwargs)


def test_gvr_selector_is_default_off_and_opt_in(monkeypatch):
    monkeypatch.setattr(stage1, "_ONEPASS_TOPK", False)
    monkeypatch.setattr(stage1, "_DIRECT_SELECTOR", "off")
    assert (
        stage1.qsa_stage1_selector_producer(blocks=32768, block_topk=544)
        == "radix_exact"
    )
    monkeypatch.setattr(stage1, "_DIRECT_SELECTOR", "gvr")
    monkeypatch.setattr(stage1, "_DIRECT_SELECTOR_MIN_BLOCKS", 1)
    assert (
        stage1.qsa_stage1_selector_producer(blocks=32768, block_topk=544) == "gvr_exact"
    )
    assert (
        stage1.qsa_stage1_selector_producer(blocks=544, block_topk=512) == "gvr_exact"
    )
    assert (
        stage1.qsa_stage1_selector_producer(blocks=65536, block_topk=1025)
        == "radix_exact"
    )


def test_gvr_env_value_is_accepted_at_import():
    assert stage1._DIRECT_SELECTOR_MODES == ("off", "direct8", "direct4", "gvr")
    source = inspect.getsource(stage1)
    assert "MLX_QWEN4_QSA_STAGE1_GVR_COUNT_PATHS" in source


def test_gvr_dispatch_and_path_counters(monkeypatch):
    scores = SimpleNamespace(shape=(4, 8192))
    positions = object()
    sentinel = object()
    calls = []

    def fake_gvr(scores_arg, positions_arg, *, topk, compress_ratio, **kwargs):
        calls.append((topk, compress_ratio, kwargs))
        if kwargs.get("return_diagnostics"):
            diag = stage1.mx.array(
                [[2, 600, 0], [2, 700, 1], [3, 0, 4], [1, 544, 0]],
                dtype=stage1.mx.uint32,
            )
            return sentinel, diag
        return sentinel

    monkeypatch.setattr(stage1, "_DIRECT_SELECTOR", "gvr")
    monkeypatch.setattr(stage1, "_DIRECT_SELECTOR_MIN_BLOCKS", 1)
    monkeypatch.setattr(stage1, "select_scores_gvr", fake_gvr)
    monkeypatch.setattr(stage1, "_GVR_COUNT_PATHS", False)
    stage1.qsa_stage1_candidate_status(reset=True)
    assert (
        stage1._select_scores(scores, positions, topk=544, compress_ratio=4) is sentinel
    )
    assert calls[-1] == (544, 4, {})
    assert stage1.qsa_stage1_candidate_status()["runtime_counts"] == {
        "gvr_topk_dispatches": 1
    }

    monkeypatch.setattr(stage1, "_GVR_COUNT_PATHS", True)
    assert (
        stage1._select_scores(scores, positions, topk=544, compress_ratio=4) is sentinel
    )
    assert calls[-1] == (544, 4, {"return_diagnostics": True})
    assert stage1.qsa_stage1_candidate_status(reset=True)["runtime_counts"] == {
        "gvr_topk_dispatches": 2,
        "gvr_sampled_rows": 2,
        "gvr_fallback_rows": 1,
        "gvr_dense_rows": 1,
    }


def test_gvr_status_is_reported_unqualified():
    status = stage1.qsa_stage1_candidate_status()
    assert status["gvr_qualification"] == "unqualified_research_candidate"
    assert status["direct_selector_selected"] is False


def test_gvr_kernel_source_preserves_law_and_fallback():
    source = selector._GVR_SOURCE
    assert "valid_count <= CAP" in source
    assert "largest block IDs" in source
    assert "for (uint pass = 0u; pass < 8u; ++pass)" in source
    assert "qsa_direct_composite_key" in source
    assert "out[slot] = blocks - invalid_count + (slot - selected_valid)" in source
    assert "row_diag[0] = path" in source
    assert "simd_shuffle_xor" in selector._GVR_HEADER


metal = pytest.mark.skipif(
    not GPU, reason="Metal kernel test; set MLX2_RUN_GPU_TESTS=1 under the lease"
)


@pytest.fixture
def gpu_device():
    mx = stage1.mx
    mx.set_cache_limit(4 << 30)
    mx.set_default_device(mx.gpu)
    try:
        yield mx
    finally:
        mx.set_default_device(mx.cpu)


@metal
@pytest.mark.parametrize("name,values,positions,topk", CASES, ids=[c[0] for c in CASES])
@pytest.mark.parametrize("width,capacity", [(1024, 2048), (1024, 1024), (512, 2048)])
def test_metal_gvr_matches_direct8_law_and_reference_paths(
    gpu_device, name, values, positions, topk, width, capacity
):
    del name
    mx = gpu_device
    scores = mx.array(values)
    q_positions = mx.array(positions)
    ids, diag = selector.select_scores_gvr(
        scores,
        q_positions,
        topk=topk,
        compress_ratio=RATIO,
        width=width,
        capacity=capacity,
        return_diagnostics=True,
    )
    direct8 = selector.select_scores_direct8(
        scores, q_positions, topk=topk, compress_ratio=RATIO
    )
    mx.eval(ids, diag, direct8)
    law = gvr.selector_law(values, positions, topk=topk, compress_ratio=RATIO)
    _, expected_diag = gvr.gvr_reference_select(
        values,
        positions,
        topk=topk,
        compress_ratio=RATIO,
        width=width,
        capacity=capacity,
    )
    np.testing.assert_array_equal(np.asarray(direct8), law)
    np.testing.assert_array_equal(np.asarray(ids), law)
    np.testing.assert_array_equal(np.asarray(diag), expected_diag)
