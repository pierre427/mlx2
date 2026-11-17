"""CPU tests of the carve driver's clean stop (--stop-at) and resume (FN Night A, 2026-11-15)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from research.amoeba.carve.carving import CLEAN_STOP_RC, CleanStop, train_mask
from research.amoeba.carve.host import Host
from test_amoeba_carve_cpu import Tok, toy


@pytest.fixture(autouse=True)
def _cpu():
    mx.set_default_device(mx.cpu)


def _problem():
    rng = np.random.default_rng(0)
    tgt = mx.array(rng.standard_normal((4, 16)).astype(np.float32))

    def grad(m, micro):
        d = m - tgt * micro
        return (d * d).mean(), 2 * d / d.size
    return mx.zeros((4, 16)), mx.array((rng.random((4, 16)) > 0.8).astype(np.float32)), grad, \
        (lambda step: [float(1 + step % 3)])


def test_resume_from_checkpoint_equals_uninterrupted(tmp_path):
    Z0, fixed, grad, batches = _problem()
    full = train_mask(Z0 + 3, fixed, 0.5, 7, 0.1, grad, batches, lambda *_: None, "t", every=2)
    ck = tmp_path / "ck.npz"
    with pytest.raises(CleanStop) as e:
        train_mask(Z0 + 3, fixed, 0.5, 7, 0.1, grad, batches, lambda *_: None, "t", every=2, ckpt=ck, deadline=0)
    assert e.value.code == CLEAN_STOP_RC and ck.exists() and int(np.load(ck)["next_step"]) == 1
    res = train_mask(Z0 + 3, fixed, 0.5, 7, 0.1, grad, batches, lambda *_: None, "t", every=2, ckpt=ck)
    assert np.allclose(np.array(full), np.array(res), atol=1e-6)


def test_driver_clean_stop_then_resume(tmp_path):
    from research.amoeba.carve import __main__ as D
    fac = lambda log: Host(model=toy(moe=True), tokenizer=Tok(), fingerprint="toy:abc", log=log)  # noqa: E731
    base = ["--model", "toy", "--smoke", "--cpu", "--seq-len", "16", "--ablations", "zero", "--steps", "2",
            "--stages", "inspect,mean,rank", "--out", str(tmp_path / "r")]
    with pytest.raises(SystemExit) as e:
        D.main(base + ["--stop-at", "0"], host_factory=fac)
    assert e.value.code == CLEAN_STOP_RC
    rep = json.loads((tmp_path / "r" / "carve.json").read_text())
    assert "deadline" in rep["clean_stop"]["why"] and rep["stages_done"] == []
    D.main(base + ["--resume"], host_factory=fac)
    rep = json.loads((tmp_path / "r" / "carve.json").read_text())
    assert {"inspect", "mean", "rank"} <= set(rep["stages_done"]) and "clean_stop" not in rep
    assert (tmp_path / "r" / "ranking_zero.npz").exists()
    # the checkpoint is not a ranking (bisect globs ranking_*.npz)
    assert sorted(p.name for p in (tmp_path / "r").glob("ranking_*.npz")) == ["ranking_zero.npz"]
