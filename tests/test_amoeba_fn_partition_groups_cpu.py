"""CPU test of scripts/amoeba_fn_partition.py: the default output is unchanged (golden digest made with the script
before family tiles), and the opt-in family (group) tiles: assignment, laminar refusal, the master rule, cascades,
auto proposals, --emit-unassigned and --check-against."""
import hashlib
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/amoeba_fn_partition.py"
LAYERS = [1, 3, 4]
E, F = 4, 8
N = E * F
# Column plan (every layer the same, plus a per-layer twist on col 31). Boxes a..i.
CLAIMS = {
    9: "abc", 10: "abc", 11: "abc", 12: "abc",      # closed inside {a,b,c}
    13: "abcd", 14: "abcd",                          # inside big = {a,b,c,d,e}
    15: "adf",                                       # crosses groups -> core
    16: "a", 17: "a", 18: "b", 19: "c", 20: "d", 21: "e",
    22: "ab", 23: "de",
    26: "f", 27: "gh", 28: "fgh",                    # {f,g,h}: one box + one pair -> dropped; parent fghi takes 28
    29: "i", 30: "fgi",
    7: "abc",                                        # inside T2 (expert 0): stays core
    8: "abc",                                        # inside the English map: stays core
}
BOXES = "abcdefghi"


def _npz_fixed(path, **arrays):
    """An .npz with fixed zip timestamps, so its bytes (and the partition's T2 digest) are reproducible."""
    with zipfile.ZipFile(path, "w") as z:
        for k, v in arrays.items():
            buf = io.BytesIO()
            np.save(buf, v)
            z.writestr(zipfile.ZipInfo(f"{k}.npy", date_time=(2026, 1, 1, 0, 0, 0)), buf.getvalue())


def _map(d, target, sel, fp="f" * 64, every=None):
    d.mkdir(parents=True, exist_ok=every is not None)
    S = len(LAYERS)
    arr = {}
    for s in range(S):
        if every is not None:
            arr[f"every_source_{s}"] = every[s]
        arr[f"target_{s}"] = target[s]
        arr[f"mean_target_{s}"] = np.where(target[s], 1.0, 0.0).astype(np.float32)
        arr[f"mean_baseline_{s}"] = np.where(sel[s], 0.1, 0.9).astype(np.float32)
    np.savez(d / "map.npz", **arr)
    (d / "report.json").write_text(json.dumps({
        "per_site": [{"class": "moe", "units": N, "layer": l} for l in LAYERS], "fingerprint": fp,
        "checkpoint": "toy@ffffffff", "baseline": {"corpus": "base.jsonl"}, "target": {"corpus": f"{d.name}.jsonl"},
        "capture_rev": "toyrev"}))


def _toy(root):
    S = len(LAYERS)
    for b in BOXES:
        sel = np.zeros((S, N), bool)
        for c, who in CLAIMS.items():
            if b in who:
                sel[:, c] = True
        sel[1, 31] = b in "ab"                       # layer twist: a 2-claim on one layer only
        target = sel.copy()
        target[:, 24] = True                          # targeted but not exclusive (baseline too high): unselected
        _map(root / "maps" / b, target, sel)
    eng = np.zeros((S, N), bool)
    eng[:, 8] = True
    _map(root / "maps" / "eng", eng, eng)
    t2 = np.zeros((6, E), np.uint8)
    t2[:, 0] = 1                                      # expert 0 = cols 0..7
    _npz_fixed(root / "t2.npz", core=t2)


def _run(root, out, *extra):
    cmd = [sys.executable, str(SCRIPT)] + [x for b in BOXES for x in ("--box", f"{b}=maps/{b}")] + [
        "--core-map", "maps/eng", "--t2", "t2.npz", "--out", out, *extra]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=root)


def _digest(out):
    z = np.load(out / "tiles.npz")
    h = hashlib.sha256((out / "partition.json").read_bytes())
    for k in z.files:
        v = z[k]
        h.update(f"{k}|{v.dtype}|{v.shape}|".encode())
        h.update(np.ascontiguousarray(v).tobytes())
    return h.hexdigest()


# Made with scripts/amoeba_fn_partition.py at 1b2b51ae5 (before family tiles) on this toy.
GOLDEN_DEFAULT = "c2fcbaa8165234a2df8fc64bb37c9313e42cb45b6f696e5b8b9ed02fb1842526"


def test_default_output_unchanged(tmp_path):
    _toy(tmp_path)
    r = _run(tmp_path, "out")
    assert r.returncode == 0, r.stderr
    assert not (tmp_path / "out" / "unassigned.npz").exists()
    p = json.loads((tmp_path / "out" / "partition.json").read_text())
    assert "groups" not in p and p["schema"] == "mlx2.amoeba-fn-partition.v1"
    assert _digest(tmp_path / "out") == GOLDEN_DEFAULT


def _groups(tmp_path, g):
    f = tmp_path / "groups.json"
    f.write_text(json.dumps({"_note": "toy", "groups": g}))
    return str(f)


def test_family_tiles(tmp_path):
    _toy(tmp_path)
    assert _run(tmp_path, "base", "--emit-unassigned").returncode == 0
    g = _groups(tmp_path, {"abc": ["a", "b", "c"], "big": ["a", "b", "c", "d", "e"], "ab": ["a", "b"],
                           "fgh": ["f", "g", "h"], "fghi": ["f", "g", "h", "i"]})
    r = _run(tmp_path, "fam", "--groups", g, "--emit-unassigned", "--check-against", "base")
    assert r.returncode == 0, r.stderr
    p = json.loads((tmp_path / "fam" / "partition.json").read_text())
    z = np.load(tmp_path / "fam" / "tiles.npz")
    base = np.load(tmp_path / "base" / "tiles.npz")
    G = p["groups"]
    assert set(G["kept"]) == {"abc", "big", "fghi"}
    assert "fewer than 3 members" in G["dropped"]["ab"]["reason"]
    assert G["dropped"]["fgh"]["reason"].startswith("only one box and one pair")
    cols = lambda m: sorted(set(np.flatnonzero(m.any(0))))           # noqa: E731
    assert cols(z["group:abc"]) == [9, 10, 11, 12] and z["group:abc"][:, 9:13].all()
    assert cols(z["group:big"]) == [13, 14]
    assert cols(z["group:fghi"]) == [28, 30]                          # the dropped subgroup's units pass up
    assert G["kept"]["abc"]["parent"] == "big" and G["kept"]["big"]["subgroups"] == ["abc"]
    assert G["kept"]["big"]["boxes"] == ["d", "e"] and G["kept"]["fghi"]["boxes"] == ["f", "i"]
    core = z["core"]
    assert core[:, 7].all() and core[:, 8].all() and core[:, 15].all()   # T2, English map, cross-group 3+
    assert not core[:, 9:15].any()
    # box and pair tiles unchanged; core | groups = the default core; exact cover
    for k in base.files:
        if k not in ("core", "layers"):
            assert np.array_equal(base[k], z[k]), k
    folded = core | z["group:abc"] | z["group:big"] | z["group:fghi"]
    assert np.array_equal(folded, base["core"])
    cover = sum(z[k].astype(int) for k in z.files if k != "layers")
    assert (cover == 1).all()
    assert p["check_against"]["equal_with_groups_folded"]
    assert p["core"]["to_group_tiles"] == int(z["group:abc"].sum() + z["group:big"].sum() + z["group:fghi"].sum())
    assert p["core"]["share"] < G["core_share_without_groups"]
    u = np.load(tmp_path / "fam" / "unassigned.npz")["unassigned"]
    assert np.array_equal(u, np.load(tmp_path / "base" / "unassigned.npz")["unassigned"])
    assert (u <= core).all() and u[:, 25].all() and int(u.sum()) == p["core"]["unassigned"]


def test_laminar_refusal_and_unknown(tmp_path):
    _toy(tmp_path)
    r = _run(tmp_path, "x", "--groups", _groups(tmp_path, {"p": ["a", "b", "c"], "q": ["c", "d", "e"]}))
    assert r.returncode != 0 and "laminar" in r.stderr
    r = _run(tmp_path, "y", "--groups", _groups(tmp_path, {"p": ["a", "b", "zz"]}))
    assert r.returncode != 0 and "unknown boxes" in r.stderr


def test_check_against_refuses_a_different_partition(tmp_path):
    _toy(tmp_path)
    assert _run(tmp_path, "base", "--ratio", "0.5").returncode == 0
    r = _run(tmp_path, "fam", "--groups", _groups(tmp_path, {"abc": ["a", "b", "c"]}), "--check-against", "base")
    assert r.returncode != 0 and "check-against: ratio" in r.stderr
    assert _run(tmp_path, "base2").returncode == 0
    z = dict(np.load(tmp_path / "base2" / "tiles.npz"))
    z["box:a"] = z["box:a"].copy()
    z["box:a"][0, 16] = False                                         # a tampered reference tile
    z["core"] = z["core"].copy()
    z["core"][0, 16] = True
    np.savez(tmp_path / "base2" / "tiles.npz", **z)
    r = _run(tmp_path, "fam2", "--groups", _groups(tmp_path, {"abc": ["a", "b", "c"]}), "--check-against", "base2")
    assert r.returncode != 0 and "tile box:a differs" in r.stderr
    assert _run(tmp_path, "fam3", "--groups", _groups(tmp_path, {"abc": ["a", "b", "c"]})).returncode == 0
    r = _run(tmp_path, "fam4", "--groups", _groups(tmp_path, {"abc": ["a", "b", "c"]}), "--check-against", "fam3")
    assert r.returncode != 0 and "reference has group tiles" in r.stderr


def test_auto_groups(tmp_path):
    _toy(tmp_path)
    r = _run(tmp_path, "auto", "--groups-auto", "--auto-min-units", "6", "--no-tiles")
    assert r.returncode == 0, r.stderr
    assert not (tmp_path / "auto" / "tiles.npz").exists()
    G = json.loads((tmp_path / "auto" / "partition.json").read_text())["groups"]
    # exact sets with >= 6 units (3 layers x cols): abc 12, abcd 6, adf 3, fgh 3, fgi 3 -> candidates abc, abcd
    names = [c["group"] for c in G["auto_candidates"]]
    assert names == ["a+b+c+d", "a+b+c"]
    # a+b+c+d: direct children = subgroup a+b+c + box d -> kept; a+b+c kept below it
    assert set(G["kept"]) == {"a+b+c", "a+b+c+d"} and G["kept"]["a+b+c"]["parent"] == "a+b+c+d"
    assert ">= 6 units are candidates" in G["criterion"]["auto"]


def test_auto_skips_non_laminar_and_declared_duplicates(tmp_path):
    _toy(tmp_path)
    r = _run(tmp_path, "auto", "--groups-auto", "--auto-min-units", "3", "--no-tiles",
             "--groups", _groups(tmp_path, {"abc": ["a", "b", "c"]}))
    assert r.returncode == 0, r.stderr
    G = json.loads((tmp_path / "auto" / "partition.json").read_text())["groups"]
    log = {c["group"]: c for c in G["auto_candidates"]}
    assert "same members as declared" in log["a+b+c"]["dropped"]
    assert "with group abc " in log["a+d+f"]["dropped"] and "with group f+g+h" in log["f+g+i"]["dropped"]
    assert log["f+g+h"]["proposed"] and G["dropped"]["f+g+h"]["reason"].startswith("only one box and one pair")
    assert G["kept"]["abc"]["source"] == "declared"


def test_auto_min_members_guard(tmp_path):
    _toy(tmp_path)
    r = _run(tmp_path, "x", "--groups-auto", "--auto-min-members", "2")
    assert r.returncode != 0


@pytest.mark.parametrize("flag", ["--emit-unassigned"])
def test_unassigned_alone_keeps_tiles_identical(tmp_path, flag):
    _toy(tmp_path)
    assert _run(tmp_path, "a").returncode == 0
    assert _run(tmp_path, "b", flag).returncode == 0
    za, zb = np.load(tmp_path / "a" / "tiles.npz"), np.load(tmp_path / "b" / "tiles.npz")
    assert za.files == zb.files and all(np.array_equal(za[k], zb[k]) for k in za.files)


def _with_every_source(root, drop=None):
    """Rewrite each box map with every_source arrays (= target), except False on drop[box] columns."""
    S = len(LAYERS)
    for b in BOXES:
        z = np.load(root / "maps" / b / "map.npz")
        target = np.stack([z[f"target_{s}"] for s in range(S)])
        sel = np.stack([z[f"mean_baseline_{s}"] < 0.5 for s in range(S)])
        every = target.copy()
        for c in (drop or {}).get(b, ()):
            every[:, c] = False
        _map(root / "maps" / b, target, sel, every=every)


def test_every_source_all_true_keeps_tiles(tmp_path):
    _toy(tmp_path)
    assert _run(tmp_path, "base").returncode == 0
    _with_every_source(tmp_path)
    r = _run(tmp_path, "ev", "--require-every-source")
    assert r.returncode == 0, r.stderr
    a, b = np.load(tmp_path / "base" / "tiles.npz"), np.load(tmp_path / "ev" / "tiles.npz")
    assert set(a.files) == set(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files)
    p = json.loads((tmp_path / "ev" / "partition.json").read_text())
    assert p["require_every_source"] is True and "every_source" in p["rule"]


def test_every_source_drops_single_source_claims(tmp_path):
    _toy(tmp_path)
    _with_every_source(tmp_path, drop={"a": (9, 10, 11, 12), "b": (9, 10, 11, 12)})
    r = _run(tmp_path, "ev", "--require-every-source")
    assert r.returncode == 0, r.stderr
    z = np.load(tmp_path / "ev" / "tiles.npz")
    for c in (9, 10, 11, 12):                        # was {a,b,c} -> core; now only c claims it
        assert z["box:c"][:, c].all() and not z["core"][:, c].any()
    p = json.loads((tmp_path / "ev" / "partition.json").read_text())
    assert p["boxes"]["a"]["selected_before_every_source"] - p["boxes"]["a"]["selected"] == 4 * len(LAYERS)


def test_every_source_missing_arrays_refused(tmp_path):
    _toy(tmp_path)
    r = _run(tmp_path, "ev", "--require-every-source")
    assert r.returncode != 0 and "every_source" in r.stderr


def test_check_against_refuses_every_source_mismatch(tmp_path):
    _toy(tmp_path)
    assert _run(tmp_path, "base").returncode == 0
    _with_every_source(tmp_path)
    r = _run(tmp_path, "ev", "--require-every-source", "--check-against", "base")
    assert r.returncode != 0 and "every_source" in r.stderr
