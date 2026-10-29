"""Probe fixture preparation and workspace bounds, no GPU/model payload."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def probe(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "scripts/research"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "packed_visible_probe_test", directory / "probe_deepseek_v4_packed_visible.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "ratio,offset,topk", [(0, 0, 0), (4, 125, 0), (128, 125, 0), (4, 4093, 512)]
)
def test_fixture_uses_exact_causal_pool_count_and_top512(probe, ratio, offset, topk):
    plan = {"ratio": ratio, "offset": offset, "length": 3}
    if topk:
        plan["topk"] = topk
    q, local, packed, full, indices, _sinks = probe.fixture(plan, heads=2)
    assert q.shape == (1, 2, 3, 512) and local.shape[2] == min(offset, 127) + 3
    if ratio:
        assert packed.codes.shape == (1, (offset + 3) // ratio, 448)
        assert full.shape == (1, (offset + 3) // ratio, 512)
        expected = probe.unpack_fp8_kv(packed, xp=np)
        np.testing.assert_array_equal(full.view(np.uint32), expected.view(np.uint32))
    else:
        assert packed is None and full is None
    if topk:
        assert indices.shape == (1, 3, 512)
        for row in range(3):
            assert len(np.unique(indices[0, row])) == 512
            assert indices[0, row].max() < (offset + row + 1) // ratio


def test_default_long_fixture_remains_bounded_and_candidate_has_no_pool_scratch(probe):
    from mlx2.runtime.models.deepseek_v4_packed_visible_attention import workspace_bytes

    assert max(probe.working_bound(plan) for plan in probe.PLANS) < 2 << 30
    assert (
        workspace_bytes(heads=64, length=8, topk=512)
        == 64 * 8 * 512 * 4 + 8 * 512 * 4 + 28
    )
