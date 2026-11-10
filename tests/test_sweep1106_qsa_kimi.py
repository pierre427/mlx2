"""Bug sweep 2026-11-06, lane qsa-kimi: CPU regressions."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx2.runtime.models import qsa_sidecar as qs
from mlx2.runtime.models.qwen4_qsa_nax import compact_token_validity

ROOT = Path(__file__).resolve().parents[1]


def _compact_visible(cb):
    """Token sets the NAX kernel admits, per query row (its validity law)."""
    (*_, physical, valid) = compact_token_validity(cb)
    (p, v) = (np.array(physical)[0], np.array(valid)[0])
    return [sorted(set(p[row][v[row]].tolist())) for row in range(p.shape[0])]


def _mask_visible(ids, pv, q_pos, n_blocks, total):
    vis = np.array(qs._dense_mask(ids, pv, q_pos, n_blocks, total))[0, 0]
    return [np.nonzero(r)[0].tolist() for r in vis]


def _reference_visible(ids, pv, q_pos):
    """research/qsa/sparse_attention.build_sparse_mask: selected blocks plus
    the query's own block up to itself, causal."""
    out = []
    for row, p in enumerate(np.array(q_pos).tolist()):
        seen = set(range((p // 4) * 4, p + 1))
        for b, ok in zip(np.array(ids[row]).tolist(), np.array(pv[row]).tolist()):
            if ok:
                seen.update(range(b * 4, b * 4 + 4))
        out.append(sorted(t for t in seen if t <= p))
    return out


def test_nax_compact_keeps_unselected_own_block_at_block_end():
    # q_pos 15 ends block 3, which the selection skipped: the reference (and
    # the mask/gather paths) still show tokens 12..15.
    ids = mx.array([[0, 1]], dtype=mx.int32)
    pv = mx.array([[True, True]])
    q_pos = mx.array([15], dtype=mx.int32)
    got = _compact_visible(qs._compact(ids, pv, q_pos, 16))
    assert got == _reference_visible(ids, pv, q_pos)
    assert got == _mask_visible(ids, pv, q_pos, 4, 16)


@pytest.mark.parametrize("fused", [False, True])
def test_nax_compact_matches_reference_on_every_residue(fused):
    rng = np.random.default_rng(1106)
    (offset, length, topk) = (40, 13, 4)
    total = offset + length
    n_blocks = total // 4
    q_pos = mx.arange(offset, total, dtype=mx.int32)
    valid = (np.arange(n_blocks)[None] * 4 + 3) <= np.array(q_pos)[:, None]
    rows_ids = []
    for row in range(length):
        pool = np.nonzero(valid[row])[0]
        if row % 2 == 0:  # half the rows also select their newest valid block
            rest = rng.choice(pool[:-1], size=topk - 1, replace=False)
            pick = np.concatenate([rest, pool[-1:]])
        else:
            pick = rng.choice(pool[:-1], size=topk, replace=False)
        rows_ids.append(np.sort(pick))
    rows_pv = [np.ones(topk, bool)] * length
    counts = [topk] * length
    ids = mx.array(np.array(rows_ids, np.int32))
    pv = mx.array(np.array(rows_pv))
    cnt = mx.array(np.array(counts, np.int32)) if fused else None
    got = _compact_visible(qs._compact(ids, pv, q_pos, total, cnt))
    assert got == _reference_visible(ids, pv, q_pos)


def test_dense_mask_keeps_block_zero_under_fused_padding():
    # Fused selection for a row with fewer valid blocks than topk: ids
    # [0, 1, 0, 0], counts 2.  The zero pads must not clear block 0.
    ids = mx.array([[0, 1, 0, 0]], dtype=mx.int32)
    pv = mx.arange(4)[None] < mx.array([2])[:, None]
    q_pos = mx.array([10], dtype=mx.int32)
    assert _mask_visible(ids, pv, q_pos, 5, 21) == _reference_visible(ids, pv, q_pos)


def test_kernel_bench_baseline_arms_pin_law_v1(monkeypatch):
    from mlx2.runtime.models import qwen4_qsa_nax as nax

    seen = []
    monkeypatch.setattr(nax, "nax_qsa_attention", lambda *a, **kw: seen.append(kw.get("law")))
    spec = importlib.util.spec_from_file_location(
        "bench_qsa_nax_kernel_ab_sweep", ROOT / "scripts/bench_qsa_nax_kernel_ab.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    case = SimpleNamespace(q=None, k=None, v=None, kvh=2)
    for arm in ("A0k", "A1k"):
        bench.run_arm(arm, case, (None, None, None, 8, None, None, 16))
    assert seen == ["v1", "v1"]


def _sidecar_probe():
    spec = importlib.util.spec_from_file_location(
        "probe_qsa_sidecar_sweep1106", ROOT / "scripts" / "probe_qsa_sidecar.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(tokens, windows):
    return {"tokens": tokens, "top5": windows}


def test_continuation_near_tie_needs_a_functional_oracle():
    """Reported QSAK item: numerics alone never earn near_tie_equivalent."""
    probe = _sidecar_probe()
    a = _run([1, 2], [[(1, -0.1), (9, -3.0)], [(2, -0.10), (3, -0.30), (7, -4.0)]])
    b = _run([1, 3], [[(1, -0.1), (9, -3.0)], [(3, -0.12), (2, -0.25), (7, -4.0)]])
    assert probe.classify_divergence(a, b)["verdict"] == "near_tie_no_oracle"
    assert probe.classify_divergence(a, b, oracle=(True, True))["verdict"] == "near_tie_equivalent"
    assert probe.classify_divergence(a, b, oracle=(True, False))["verdict"] == "near_tie_oracle_failed"
    assert probe.classify_divergence(a, a) == {"diverged": False}


def test_continuation_missing_token_fails_when_the_window_proves_it_far():
    probe = _sidecar_probe()
    a = _run([2], [[(2, -0.1), (5, -0.4), (6, -0.7), (7, -1.0), (8, -2.0)]])
    far = _run([3], [[(3, -0.1), (2, -0.3), (6, -0.7), (7, -1.0), (8, -2.0)]])
    # 3 is absent from a's window, whose floor sits 1.9 nat below its top.
    assert probe.classify_divergence(a, far)["verdict"] == "high_margin"
    close = {"tokens": [2], "top5": [[(2, -0.1), (5, -0.2), (6, -0.3), (7, -0.4), (8, -0.5)]]}
    assert probe.classify_divergence(close, far)["verdict"] == "unclassified"
