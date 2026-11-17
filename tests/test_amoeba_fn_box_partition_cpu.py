"""CPU test of scripts/amoeba_fn_box_partition.py: picking, transitive merges, folds, exact partition."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _box(root, name, mask, t2, held=0.02, t2h=0.3, rnd=0.2, ok=True):
    d = root / f"box-{name}"
    d.mkdir(parents=True)
    np.savez(d / "export_core.npz", core=(t2 | mask).astype(np.uint8))
    (d / "carve.json").write_text(json.dumps({
        "inspect": {"paths": {"layerwise_grad": {"ok": ok}}}, "export_core": {"held": {"mean_d_ce": held}},
        "fixed_only_held": {"mean_d_ce": t2h},
        "bisect": {"zero": {"0.05": {"random_same_counts_held": {"mean_d_ce": rnd}}}}}))


def test_partition(tmp_path):
    L, E = 4, 64
    t2 = np.zeros((L, E), bool)
    t2[:, :32] = True
    def m(*cols):
        x = np.zeros((L, E), bool); x[:, list(cols)] = True; return x
    root = tmp_path / "boxes"
    _box(root, "a", m(32, 33, 34, 35), t2)
    _box(root, "b", m(34, 35, 36, 37), t2)                 # J(a,b) = 8/24 < 0.5: shared tile {a,b}
    _box(root, "c", m(40, 41, 42), t2)
    _box(root, "c2", m(40, 41, 42, 43), t2)                # J(c,c2) = 0.75: merged
    _box(root, "d", m(50), t2, ok=False)                    # gate-refused: dropped
    _box(root, "e", m(51, 52), t2, held=0.25)               # does not close the T2 gap: dropped
    out = tmp_path / "out"
    np.savez(tmp_path / "t2.npz", core=t2.astype(np.uint8))
    r = subprocess.run([sys.executable, str(ROOT / "scripts/amoeba_fn_box_partition.py"), "--t2", str(tmp_path / "t2.npz"),
                        "--boxes-root", str(root), "--out", str(out), "--min-tile", "2"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    p = json.loads((out / "partition.json").read_text())
    assert p["coverage"]["partition"]
    assert sorted(p["passing"]) == ["a", "b", "c", "c2"] and not p["verdicts"]["d"]["pass"] and not p["verdicts"]["e"]["pass"]
    assert p["merges"] == {"c+c2": ["c", "c2"]} and p["distinct_boxes"] == 3
    kinds = {t["tile"]: t for t in p["tiles"]}
    assert kinds["shared.a&b"]["experts"] == 2 * L and set(kinds["shared.a&b"]["opens_on"]) == {"a", "b"}
    assert kinds["box.c+c2"]["experts"] == 4 * L and kinds["pool.unclaimed"]["kind"] == "remainder"
    z = np.load(out / "partition.npz")
    tot = z["core"].astype(int) + sum(z[k].astype(int) for k in z.files if k.endswith("__mask"))
    assert (tot == 1).all()
