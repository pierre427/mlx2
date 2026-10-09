"""2026-10-06 sweep (kernels lane): QSA stage-one exact band and selector
status.

KR-01: the exact-band refine marked invalid candidates with a -inf score, but
the default radix selector relu-clamps its input, so those slots tied with
valid blocks whose exact score is 0 and won by the larger-id rule.  The
refine now excludes them by index.  The Metal selector cannot run on CPU, so
it is replaced by a numpy model of its contract (relu'd score, composite key
(score, id) larger first, valid blocks ascending then fill ids).
"""

from __future__ import annotations

import importlib

import mlx.core as mx
import numpy as np
import pytest

from mlx2.runtime.models import qwen4_qsa_stage1 as stage1


def _radix_model(scores, q_positions, *, topk, compress_ratio, role="primary"):
    s = np.maximum(np.array(scores, dtype=np.float32), np.float32(0.0))
    rows, blocks = s.shape
    out = []
    for row, pos in enumerate(np.array(q_positions).reshape(-1).tolist()):
        complete = max(0, int((pos + 1) / compress_ratio))  # C truncation
        valid = min(blocks, complete)
        keep = min(topk, valid)
        order = sorted(range(valid), key=lambda b: (s[row, b], b), reverse=True)
        out.append(sorted(order[:keep]) + list(range(blocks - (topk - keep), blocks)))
    return mx.array(np.array(out, dtype=np.uint32))


def _setup(monkeypatch):
    monkeypatch.setattr(stage1, "qsa_stage1_supported", lambda *a, **k: True)
    monkeypatch.setattr(
        stage1, "qsa_stage1_route", lambda *a, **k: {"score_producer": "mpp_exact_band"}
    )

    def mpp_scores(q, pooled):
        dots = mx.einsum("blhd,bnd->blnh", q.astype(mx.float32), pooled.astype(mx.float32))
        return (mx.sum(mx.maximum(dots, 0), axis=-1) / np.sqrt(q.shape[-1])).reshape(
            -1, pooled.shape[1]
        )

    monkeypatch.setattr(stage1, "_mpp_scores", mpp_scores)
    monkeypatch.setattr(stage1, "_select_scores", _radix_model)


@pytest.mark.parametrize("valid", [0, 1, 500, 530, 544, 600])
def test_exact_band_keeps_every_valid_zero_score_block(monkeypatch, valid):
    _setup(monkeypatch)
    rng = np.random.default_rng(3)
    topk, ratio, blocks, dim = 512, 4, 1024, 8
    q = mx.ones((1, 1, 4, dim), dtype=mx.float32)
    pooled = rng.random((1, blocks, dim)).astype(np.float32) + 0.01
    zero_ids = rng.choice(max(valid, 1), min(40, valid), replace=False)
    pooled[0, zero_ids] = -pooled[0, zero_ids]  # every head dot <= 0: score 0
    positions = mx.array([[valid * ratio + ratio - 2 if valid else 0]], dtype=mx.int32)
    got = np.array(
        stage1.qsa_stage1_select(
            q, mx.array(pooled), positions, block_topk=topk, compress_ratio=ratio
        )
    ).reshape(-1).tolist()
    # The single-pass reference: exact scores straight into the selector.
    exact = stage1._mpp_scores(q, mx.array(pooled))
    want = np.array(_radix_model(exact, positions, topk=topk, compress_ratio=ratio))
    want = want.reshape(-1).tolist()
    if valid <= topk:
        assert set(range(valid)) <= set(got)
        assert got == list(range(valid)) + list(range(blocks - (topk - valid), blocks))
    assert got == want


def test_direct_selector_status_reports_selection_and_use(monkeypatch):
    monkeypatch.setattr(stage1, "_DIRECT_SELECTOR", "off")
    stage1.qsa_stage1_candidate_status(reset=True)
    report = stage1.qsa_stage1_candidate_status()
    assert report["direct_selector_selected"] is False
    assert report["direct_selector_observed_used"] is False

    monkeypatch.setattr(stage1, "_DIRECT_SELECTOR", "direct4")
    assert stage1.qsa_stage1_selector_producer(blocks=8192, block_topk=544) == "direct4_exact"
    report = stage1.qsa_stage1_candidate_status()
    assert report["direct_selector_configured"] == "direct4"
    assert report["direct_selector_selected"] is True
    assert report["direct_selector_observed_used"] is False
    stage1._record_candidate_dispatch("direct4_topk_dispatches")
    assert stage1.qsa_stage1_candidate_status(reset=True)["direct_selector_observed_used"] is True


def test_unknown_direct_selector_value_fails_closed(monkeypatch):
    monkeypatch.setenv("MLX_QWEN4_QSA_STAGE1_DIRECT_SELECTOR", "direct-4")
    try:
        with pytest.raises(ValueError, match="DIRECT_SELECTOR"):
            importlib.reload(stage1)
    finally:
        monkeypatch.delenv("MLX_QWEN4_QSA_STAGE1_DIRECT_SELECTOR")
        importlib.reload(stage1)
    for value in ("", "off", "direct8", " Direct4 "):
        assert stage1._direct_selector_mode(value) in {"off", "direct8", "direct4"}
