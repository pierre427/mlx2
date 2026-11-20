"""The box battery's sparse tile store (scripts/amoeba_box_battery.py Tiles, 2026-11-19 memory fix) builds exactly
the masks the dense np.load reads built: per tile, every served composition, and the joint arm's opened-tile
union. On a toy partition always; on a sample of the real FN catboxes partitions when the volume is mounted."""
import json
import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import amoeba_box_battery as B  # noqa: E402

REAL = Path.home() / "amoeba-runs/fn-catboxes-064cfd4f3"


def _write_toy(path, rng, n_tiles=40, shape=(3, 64)):
    lab = rng.integers(0, n_tiles, size=shape)
    names = ["core"] + [f"box:b{i}" for i in range(n_tiles - 10)] + [f"joint:b{i}+b{i + 1}" for i in range(9)]
    arr = {k: lab == i for i, k in enumerate(names)}
    np.savez_compressed(path, layers=np.array([1, 4, 7]), **arr)
    return arr


def test_toy_masks_identical(tmp_path):
    rng = np.random.default_rng(0)
    arr = _write_toy(tmp_path / "tiles.npz", rng)
    t = B.Tiles(tmp_path / "tiles.npz")
    z = np.load(tmp_path / "tiles.npz")
    assert t.layers == [1, 4, 7] and t.files == z.files and t.keys == [k for k in z.files if k != "layers"]
    for k in t.keys:
        m = t[k]
        assert m.dtype == bool and np.array_equal(m, z[k]) and np.array_equal(m, arr[k]) and t.units(k) == m.sum()
        m[:] = True                                            # a fresh array: mutating it changes no later read
        assert np.array_equal(t[k], arr[k])
    ks = ["core", "box:b3", "joint:b1+b2"]
    assert np.array_equal(t.mask(*ks), z["core"] | z["box:b3"] | z["joint:b1+b2"])
    assert not t.mask().any() and t.mask().shape == (3, 64)


def _compare_partition(pd, jd, n_boxes, n_joint, seed=0):
    prep = json.loads((pd / "partition.json").read_text())
    old = np.load(pd / "tiles.npz")
    new = B.Tiles(pd / "tiles.npz")
    keys = [k for k in old.files if k != "layers"]
    assert new.keys == keys and new.layers == [int(x) for x in old["layers"]]
    rnd = random.Random(seed)
    for k in rnd.sample(keys, min(40, len(keys))):
        assert np.array_equal(new[k], old[k]), k
    boxes = sorted(k[4:] for k in keys if k.startswith("box:"))
    for b in rnd.sample(boxes, min(n_boxes, len(boxes))):
        a = B.served_keep(lambda k: old[k], keys, prep, b, old["core"])
        c = B.served_keep(lambda k: new[k], keys, prep, b, new["core"])
        assert np.array_equal(a, c), b
        B._ETILES.clear()
    if jd is None:
        return
    jprep = json.loads((jd / "partition.json").read_text())
    jold = np.load(jd / "tiles.npz")
    jnew = B.Tiles(jd / "tiles.npz")
    jk = list(jnew.keys)
    joint = [k for k in jk if k.startswith("joint:")]
    folded = jold["core"].copy()
    for k in joint:                                            # the battery's former check, dense
        folded |= jold[k]
    assert np.array_equal(jnew.mask("core", *joint), folded)
    assert np.array_equal(folded, old["core"])
    tiles = jprep["joint"]["tiles"]
    top = sorted(tiles, key=lambda k: -tiles[k]["units"])[:n_joint]
    for name in top:
        for m in tiles[name]["members"][:2]:
            lin = {m} | set(B.ancestors_of(jprep, m)) | set(B.descendants_of(jprep, m))
            en_, closed_ = B.english_box(jprep)
            if en_ and m not in closed_:
                lin.add(en_)
            opened = [k for k in joint if lin & set(k[6:].split("+"))]
            a = B.served_keep(lambda k: jold[k], jk, jprep, m, jold["core"])
            for k in opened:                                   # the former per-tile loop
                a = a | jold[k]
            c = B.served_keep(lambda k: jnew[k], jk, jprep, m, jnew["core"])
            if opened:
                c = c | jnew.mask(*opened)
            assert np.array_equal(a, c), (name, m)
            B._ETILES.clear()


@pytest.mark.skipif(not (REAL / "partition-joint" / "tiles.npz").exists(), reason="FN catboxes run directory absent")
@pytest.mark.parametrize("main,joint", [("partition", "partition-joint"),
                                        ("partition-enshared", "partition-enshared-joint")])
def test_real_partitions_sample(main, joint):
    _compare_partition(REAL / main, REAL / joint, n_boxes=8, n_joint=4)
