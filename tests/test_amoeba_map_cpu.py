"""CPU tests of the aMoEba map path: ragged carve captures, two-level clusters, labels with no core."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def _ragged_capture(tmp, widths=(64, 32, 64, 32), per=20, seed=0, padded=False):
    """Dense layers 0, 2 (64 units) and expert layers 1, 3 (32 units), zero-padded to 64.
    Units 0-11 of layers 0 and 2 and experts 0-5 of layer 1 fire on prog.python texts."""
    rng = np.random.default_rng(seed)
    topics = ["prog.python"] * (2 * per) + ["prog.c"] * (2 * per) + ["prog.rust"] * (2 * per) + \
             ["lang.fra"] * (2 * per) + ["lang.deu"] * (2 * per)
    S, L, N = len(topics), len(widths), max(widths)
    items = [{"id": f"t{i}", "topics": [t], "split": "fit" if i % 2 == 0 else "held", "n_tokens": 64}
             for i, t in enumerate(topics)]
    py = np.array([t == "prog.python" for t in topics])
    shared = rng.gamma(4.0, 0.25, size=S)
    for l, w in enumerate(widths):
        R = np.zeros((S, N), np.float32)
        R[:, :w] = rng.gamma(4.0, 0.25, size=(S, w))
        planted = np.arange(12) if w == 64 else (np.arange(6) if l == 1 else np.array([], int))
        if len(planted):
            R[np.ix_(py, planted)] *= 3.0
            R[:, planted] *= (1 + 0.3 * shared)[:, None]
        np.save(tmp / f"resp_L{l:02d}.npy", (R if padded else R[:, :w]).astype(np.float16))
    (tmp / "items.json").write_text(json.dumps(items))
    (tmp / "meta.json").write_text(json.dumps({"checkpoint": "ragged@00000000", "layers": L, "neurons": N,
                                               "widths": list(widths), "sequences": S,
                                               "mlx2_source_revision": "test", "corpus_sha256": "x"}))
    return tmp


@pytest.mark.parametrize("padded", [False, True], ids=["own-width", "padded"])
def test_map_cluster_and_no_core_labels(tmp_path, padded):
    (tmp_path / "cap").mkdir()
    cap = _ragged_capture(tmp_path / "cap", padded=padded)
    r = subprocess.run([sys.executable, str(ROOT / "scripts/amoeba_map_cluster.py"), "--capture", str(cap),
                        "--out", str(tmp_path / "cl"), "--k", "4", "--ve-per-layer", "8"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    z = np.load(tmp_path / "cl/clusters.npz")
    pool, lab = z["pool"], z["labels"]
    # padding never becomes a unit
    assert len(pool) == 64 + 32 + 64 + 32
    assert not ((pool[:, 0] % 2 == 1) & (pool[:, 1] >= 32)).any()
    ve = np.load(tmp_path / "cl/virtual_experts.npz")["ve"]
    assert (ve[1, 32:] == -1).all() and (ve[1, :32] >= 0).all()
    # the planted python units (dense and experts) share one global cluster
    planted = ((pool[:, 0] % 2 == 0) & (pool[:, 1] < 12)) | ((pool[:, 0] == 1) & (pool[:, 1] < 6))
    vals, cnt = np.unique(lab[planted], return_counts=True)
    assert cnt.max() / planted.sum() >= 0.9
    py_cluster = int(vals[cnt.argmax()])
    # pure-noise units carry no structure, so their virtual experts fall into any global cluster;
    # only the planted group's cohesion is asserted here

    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("")
    r = subprocess.run([sys.executable, str(ROOT / "scripts/amoeba_topics_label.py"), "--capture", str(cap),
                        "--corpus", str(corpus), "--out", str(tmp_path / "lab"), "--arrays", str(tmp_path / "arr"),
                        "--clusters", str(tmp_path / "cl/clusters.npz"), "--no-core",
                        "--twins", "8", "--perm", "300"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    rep = json.loads((tmp_path / "lab/labels.json").read_text())
    assert rep["core_fraction"] == 0.0 and rep["padding_units_excluded"] == 2 * 32
    by = {c["id"]: c for c in rep["clusters"]}
    assert "prog.python" in [x["topic"] for x in by[f"pool:C{py_cluster}"]["labels"]]


def test_label_without_core_or_flag_is_refused(tmp_path):
    (tmp_path / "cap").mkdir()
    cap = _ragged_capture(tmp_path / "cap")
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("")
    r = subprocess.run([sys.executable, str(ROOT / "scripts/amoeba_topics_label.py"), "--capture", str(cap),
                        "--corpus", str(corpus), "--out", str(tmp_path / "lab"), "--arrays", str(tmp_path / "arr"),
                        "--form-clusters", "3"], capture_output=True, text=True)
    assert r.returncode != 0 and "--no-core" in (r.stdout + r.stderr)


def test_topics_backend_ragged_mean_and_stats():
    mx = pytest.importorskip("mlx.core")
    mx.set_default_device(mx.cpu)
    import test_amoeba_carve_cpu as C
    from research.amoeba.carve.host import Host, TopicsBackend
    b = TopicsBackend(None, host=Host(model=C.toy(), tokenizer=C.Tok(), fingerprint="toy:abc", log=lambda *_: None))
    assert b.N is None and b.widths == [C.H, C.E, C.H, C.E]
    rng = np.random.default_rng(0)
    seqs = [[int(t) for t in rng.integers(0, C.V, 12)] for _ in range(5)]
    mu = b.mean_act(seqs, bs=2)
    assert mu.shape == (4, C.H)
    assert np.abs(mu[0]).sum() > 0 and (mu[1] == 0).all()          # experts are not mean-centred
    st = b.stats(seqs[:2], mu)
    assert [s.shape for s in st] == [(2, C.H), (2, C.E), (2, C.H), (2, C.E)]
    assert all(np.isfinite(s).all() for s in st)
