"""Vector-SDPA Metal sources and launches: byte-identical to the legacy copies.

``tests/fixtures/window_attention_legacy_kernels.json`` was recorded (on CPU,
``tests/window_attention_capture.py``) from the modules as they were at mlx2
d46e7a3ea, before their Metal text moved into
``mlx2.runtime.technologies.window_attention.vector_sources``:
``qwen4_attn_rows``, ``qwen4_attn_window``, ``muse_attn_window`` and
``sdpa_vector_d512``.  Every kernel's ``mx.fast.metal_kernel`` arguments
(name, input/output names, header, source, flags) and every launch's
template, grid, threadgroup, inputs and outputs must match exactly.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from window_attention_capture import capture  # noqa: E402

from mlx2.runtime.technologies.window_attention import vector_launch as VL  # noqa: E402
from mlx2.runtime.technologies.window_attention import vector_sources as VS  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "window_attention_legacy_kernels.json"


@pytest.fixture(scope="module")
def legacy():
    return json.loads(FIXTURE.read_text())


@pytest.fixture
def current(monkeypatch):
    # Round-trip through JSON so tuples/lists compare like the fixture.
    return json.loads(json.dumps(capture(monkeypatch)))


def test_every_kernel_is_byte_identical(legacy, current):
    old = {k["name"]: k for k in legacy["kernels"]}
    new = {k["name"]: k for k in current["kernels"]}
    assert sorted(new) == sorted(old)
    for name in old:
        for key in sorted(set(old[name]) | set(new[name])):
            assert new[name].get(key) == old[name].get(key), (name, key)


def test_every_launch_is_identical(legacy, current):
    assert sorted(current["launches"]) == sorted(legacy["launches"])
    for label, record in legacy["launches"].items():
        assert current["launches"][label] == record, label


def test_fixture_covers_every_family_and_plan(legacy):
    names = {k["name"] for k in legacy["kernels"]}
    assert len(names) == 20
    kernels = {launch["kernel"] for rec in legacy["launches"].values() for launch in rec["launches"]}
    # One- and two-pass, masked and unmasked, gated and plain, per family.
    for name in names:
        assert name in kernels, name


def _all_generated():
    out = {}
    families = {
        VS.MUSE_D128_WINDOW: [(s, m, False) for s in VS.STAGES for m in (False, True)
                              if not (m and "combine" in s)],
        VS.QWEN4_ROWS: [(s, m, g) for s in VS.STAGES for m in (False, True) for g in (False, True)
                        if not (m and "combine" in s) and not (g and "chains" in s)],
        VS.QWEN4_WINDOW: [(s, m, False) for s in ("one_chains", "two_chains") for m in (False, True)],
    }
    for family, variants in families.items():
        for stage, masked, gated in variants:
            kw = VS.kernel_kwargs(family, stage, masked=masked, gated=gated)
            out[kw["name"]] = kw
    for stage in ("pass1", "pass2"):
        kw = VS.strided_tk_kwargs(stage)
        out[kw["name"]] = kw
    return out


def test_generator_alone_reproduces_the_fixture(legacy):
    """Independent of the model modules' wiring."""
    old = {k["name"]: k for k in legacy["kernels"]}
    generated = json.loads(json.dumps(_all_generated()))
    assert sorted(generated) == sorted(old)
    for name, kwargs in generated.items():
        assert kwargs == old[name], name


def test_generated_sources_have_no_unresolved_markers():
    for name, kwargs in _all_generated().items():
        assert "@" not in kwargs["source"], name
        assert "@" not in kwargs.get("header", ""), name


def test_generator_rejects_unknown_parameters():
    with pytest.raises(ValueError):
        VS.one_pass_chains_source(key_count="bogus", mask="none", style=VS.ROWS_STYLE)
    with pytest.raises(ValueError):
        VS.two_pass_chains_source(key_count="cache_view", two_pass_rows="x", mask="none",
                                  style=VS.ROWS_STYLE)
    with pytest.raises(ValueError):
        VS.two_pass_combine_source(store="gate", output="single_row", style=VS.ROWS_STYLE)
    with pytest.raises(ValueError):
        VS.kernel_kwargs(VS.QWEN4_ROWS, "one_chains", gated=True)
    with pytest.raises(ValueError):
        VS.kernel_kwargs(VS.QWEN4_ROWS, "one_combine", masked=True)


# The block choice the three copies carried, verbatim (legacy).
def _legacy_qwen4_select_sdpa_blocks(devc, n_keys, n_simds):
    if devc == "s":
        blocks = 64
        if n_keys > 1024 and n_simds > 4:
            if n_keys <= 8192:
                blocks = 128
            elif n_keys <= 32768:
                blocks = 256
            elif n_keys <= 65536:
                blocks = 512
            else:
                blocks = 1024
        return blocks
    if devc == "d":
        blocks = 128
        if n_simds <= 2 and n_keys > 8192:
            blocks = 256
        elif n_simds >= 6:
            if 16384 <= n_keys < 65536:
                blocks = 512
            elif n_keys >= 65536:
                blocks = 1024
        return blocks
    return None


def _legacy_d512_stock_blocks_core(n, gqa, devc):
    n_simds = gqa
    if devc == "s":
        blocks = 64
        if n > 1024 and n_simds > 4:
            blocks = 128 if n <= 8192 else 256 if n <= 32768 else 512 if n <= 65536 else 1024
    elif devc == "d":
        blocks = 128
        if n_simds <= 2 and n > 8192:
            blocks = 256
        elif n_simds >= 6:
            if 16384 <= n < 65536:
                blocks = 512
            elif n >= 65536:
                blocks = 1024
    else:
        blocks = 64 if n_simds >= 4 else 32
    return blocks


_NS = sorted({1, 2, 1023, 1024, 1025, 4095, 4096, 8191, 8192, 8193, 16383, 16384, 16385,
              32767, 32768, 32769, 65535, 65536, 65537, 131072})


def test_select_sdpa_blocks_matches_every_legacy_copy(monkeypatch):
    from mlx2.runtime.models import muse_attn_window as MW
    from mlx2.runtime.models import qwen4_attn_rows as AR
    from mlx2.runtime.models import sdpa_vector_d512 as D5

    monkeypatch.delenv("MLX_SDPA_BLOCKS", raising=False)
    for devc, n, simds in itertools.product(("s", "d", "g", ""), _NS, range(1, 33)):
        want = _legacy_qwen4_select_sdpa_blocks(devc, n, simds)
        assert VL.select_sdpa_blocks(devc, n, simds) == want
        assert AR.select_sdpa_blocks(devc, n, simds) == want
        assert MW._select_blocks(devc, n, simds) == want
        assert D5.stock_blocks(n, simds, devc) == _legacy_d512_stock_blocks_core(n, simds, devc)


def test_plan_runs_matches_legacy():
    def legacy(plans):
        runs = []
        start = 0
        for j in range(1, len(plans) + 1):
            if j == len(plans) or plans[j] != plans[start]:
                runs.append((start, j, plans[start]))
                start = j
        return runs

    cases = [[], [1], [1, 1, 2, 2, 2, 3], [(1, None), (2, 64), (2, 64), (2, 128)], [None, None, 1]]
    for plans in cases:
        assert VL.plan_runs(plans) == legacy(plans)
