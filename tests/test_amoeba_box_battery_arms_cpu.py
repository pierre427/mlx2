"""CPU test of scripts/amoeba_box_battery.py with a fake host and fake evals (no model, no GPU): the default box
battery is unchanged (golden digest made with the script before the opt-in arms), and the opt-in unclaimed and
family-tile arms score the masks they claim to."""
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import test_amoeba_fn_partition_groups_cpu as P  # noqa: E402

HOST_L = 6
CALLS = []


class FakeHost:
    def __init__(self, model, log=None, moe_units=None):
        self.L = HOST_L
        self.widths = [P.N] * HOST_L
        self.fingerprint = "f" * 64
        self.checkpoint = "toy@ffffffff"

    def encode(self, text):
        return [ord(c) for c in text]


def _weights(name):
    return np.random.default_rng(sum(map(ord, name))).random((HOST_L, P.N))


def _english_cost(host, held, clean, cfg, pooled=False):
    CALLS.append((sorted(held), cfg.copy()))
    out = {k: float(((1.0 - cfg) * _weights(k)).sum() / 100.0) for k in held}
    out["mean_d_ce"] = float(np.mean(list(out.values())))
    return out


def _install_fakes(monkeypatch):
    pkg = {}
    for name in ("research", "research.amoeba", "research.amoeba.carve"):
        pkg[name] = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, pkg[name])
    D = types.ModuleType("research.amoeba.carve.__main__")
    D.from_np = lambda m, host: m
    E = types.ModuleType("research.amoeba.carve.evals")
    E.configure = lambda host, m, _x, L: m
    E.english_cost = _english_cost
    O = types.ModuleType("research.amoeba.carve.ops")
    O.footprint_watchdog = lambda *a, **k: None
    O.setup_mlx = lambda *a, **k: None
    O.git_rev = lambda: "toyrev"
    O.peak_gb = lambda: 0.0
    H = types.ModuleType("research.amoeba.carve.host")
    H.Host = FakeHost
    for m, mod in (("__main__", D), ("evals", E), ("ops", O), ("host", H)):
        setattr(pkg["research.amoeba.carve"], m, mod)
        monkeypatch.setitem(sys.modules, f"research.amoeba.carve.{m}", mod)


def _load(script):
    spec = importlib.util.spec_from_file_location("battery_under_test", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _corpora(root):
    (root / "held").mkdir()
    for b in list(P.BOXES) + ["eng"]:
        rows = [{"split": "held", "text": f"{b} held text number {i} with words"} for i in range(6)]
        rows.append({"split": "fit", "text": "fit row never read"})
        (root / "held" / f"{b}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    g = [{"question": f"What is {i} plus {i}?", "answer": f"{i} + {i} = {2 * i}\n#### {2 * i}"} for i in range(5)]
    (root / "held" / "gsm8k.jsonl").write_text("\n".join(json.dumps(r) for r in g) + "\n")


def _battery(monkeypatch, root, script, out, *extra):
    _install_fakes(monkeypatch)
    mod = _load(script)
    held = [x for b in list(P.BOXES) + ["eng"] for x in ("--held", f"{b}=held/{b}.jsonl@8")]
    monkeypatch.chdir(root)
    monkeypatch.setattr(sys, "argv", ["battery", "--model", "m", "--partition", extra[0], "--out", out, *held,
                                      "--docs", "4", *extra[1:]])
    CALLS.clear()
    mod.main()


def _digest(out):
    h = hashlib.sha256((out / "battery.md").read_bytes())
    for x in (out / "battery.jsonl").read_text().splitlines():
        r = json.loads(x)
        r.pop("wall_s")
        h.update(json.dumps(r, sort_keys=True).encode())
    h.update((out / "path_offset.json").read_bytes())
    return h.hexdigest()


# Made with scripts/amoeba_box_battery.py at 1b2b51ae5 (before the opt-in arms) on this toy.
GOLDEN_DEFAULT = "ccced09a9d32be5256f0e1c7623e084f0c7499a54bfb997eb9b68bf4e0c04696"


@pytest.fixture
def toy(tmp_path):
    P._toy(tmp_path)
    _corpora(tmp_path)
    assert P._run(tmp_path, "base", "--emit-unassigned").returncode == 0
    g = P._groups(tmp_path, {"abc": ["a", "b", "c"], "big": ["a", "b", "c", "d", "e"], "fghi": ["f", "g", "h", "i"]})
    r = P._run(tmp_path, "fam", "--groups", g, "--emit-unassigned", "--check-against", "base")
    assert r.returncode == 0, r.stderr
    return tmp_path


def test_default_battery_unchanged(toy, monkeypatch):
    _battery(monkeypatch, toy, ROOT / "scripts/amoeba_box_battery.py", "bat", "base")
    assert not (toy / "bat" / "unclaimed.jsonl").exists() and not (toy / "bat" / "groups.jsonl").exists()
    meta = json.loads((toy / "bat" / "battery.meta.json").read_text())
    assert "arms" not in meta
    assert _digest(toy / "bat") == GOLDEN_DEFAULT


def _mask_for(sets):
    return [m for s, m in CALLS if s == sorted(sets)]


def test_unclaimed_arm(toy, monkeypatch):
    _battery(monkeypatch, toy, ROOT / "scripts/amoeba_box_battery.py", "u", "fam", "--arms", "unclaimed",
             "--held", "gsm8k=gsm8k:held/gsm8k.jsonl@8", "--global-held", "gsm8k",
             "--held-note", "a=toy note for a")
    rows = {json.loads(x)["key"]: json.loads(x) for x in (toy / "u" / "unclaimed.jsonl").read_text().splitlines()}
    assert not (toy / "u" / "battery.jsonl").exists()
    assert set(rows) == {"global:full_minus_unclaimed", "global:core", "global:core_minus_unclaimed"} | \
        {f"box:{b}" for b in P.BOXES}
    base = np.load(toy / "base" / "tiles.npz")
    fam = np.load(toy / "fam" / "tiles.npz")
    U = np.load(toy / "fam" / "unassigned.npz")["unassigned"]
    # box a: (default core - unclaimed) | box a | pairs with a, in the layers of the partition
    keep = (base["core"] & ~U) | base["box:a"] | base["pair:a__b"]
    m = _mask_for(["a", "eng"])
    assert any(np.array_equal(x[P.LAYERS] > 0, keep) for x in m)
    assert all((x[[0, 2, 5]] == 1).all() for x in m)                 # non-MoE-partition layers untouched
    assert rows["global:core"]["kept_units"] == int(base["core"].sum())
    assert rows["global:core_minus_unclaimed"]["kept_units"] == int((base["core"] & ~U).sum())
    assert rows["global:core"]["sets"] == ["eng", "gsm8k"]
    assert rows["box:a"]["held_notes"] == {"a": "toy note for a"}
    assert int(fam["core"].sum()) < rows["global:core"]["kept_units"]
    md = (toy / "u" / "battery.md").read_text()
    assert "Unclaimed arm" in md and "gsm8k: GSM8K test rows" in md and "| box | units |" not in md
    assert "a\u2020 " in md and "gsm8k\u2020 " in md


def test_groups_arm(toy, monkeypatch):
    _battery(monkeypatch, toy, ROOT / "scripts/amoeba_box_battery.py", "g", "fam", "--arms", "groups,unclaimed")
    rows = {json.loads(x)["key"]: json.loads(x) for x in (toy / "g" / "groups.jsonl").read_text().splitlines()}
    assert {"group:abc", "group:big", "group:fghi"} <= set(rows)
    assert {f"served:{b}" for b in "abcdefghi"} <= set(rows)
    fam = np.load(toy / "fam" / "tiles.npz")
    # served a = core | group abc | group big | box a | pair a__b
    keep = fam["core"] | fam["group:abc"] | fam["group:big"] | fam["box:a"] | fam["pair:a__b"]
    assert any(np.array_equal(x[P.LAYERS] > 0, keep) for x in _mask_for(["a", "eng"]))
    assert rows["served:a"]["groups"] == ["group:abc", "group:big"]
    assert rows["served:f"]["groups"] == ["group:fghi"]
    # removal of group abc: exactly that tile off, on a, b, c and English
    off = [x for x in _mask_for(["a", "b", "c", "eng"]) if not np.array_equal(x[P.LAYERS] > 0, ~fam["group:abc"])]
    assert len(off) == 1                                              # the other one is the random control
    rnd = ~(off[0][P.LAYERS] > 0)
    assert (rnd.sum(1) == fam["group:abc"].sum(1)).all() and not (rnd & fam["group:abc"]).any()
    v = rows["group:abc"]["verdict"]
    assert set(v) >= {"removal_beats_random", "served_+0.05", "members_without_held"}
    assert rows["group:abc"]["parent"] == "big"
    md = (toy / "g" / "battery.md").read_text()
    assert "Family tiles" in md and "| abc |" in md


def test_reference_is_net_and_resume_skips_rows(toy, monkeypatch):
    _battery(monkeypatch, toy, ROOT / "scripts/amoeba_box_battery.py", "bat", "base")
    ref = [json.loads(x) for x in (toy / "bat" / "battery.jsonl").read_text().splitlines()]
    _battery(monkeypatch, toy, ROOT / "scripts/amoeba_box_battery.py", "u", "fam", "--arms", "groups,unclaimed",
             "--reference-battery", "bat/battery.jsonl")
    md = (toy / "u" / "battery.md").read_text()
    r = next(x for x in ref if x["box"] == "a")
    assert f"a {r['core_plus']['a'] - r['path_offset']['a']:+.3f}, eng" in md
    n = len(CALLS)
    _battery(monkeypatch, toy, ROOT / "scripts/amoeba_box_battery.py", "u", "fam", "--arms", "unclaimed,groups")
    assert n > 0 and len(CALLS) == 0                                  # every arm row reused, path offset kept


def test_gsm8k_filter_refused(toy, monkeypatch):
    with pytest.raises(SystemExit, match="no \\?field=value"):
        _battery(monkeypatch, toy, ROOT / "scripts/amoeba_box_battery.py", "x", "fam", "--arms", "unclaimed",
                 "--held", "gsm8k=gsm8k:held/gsm8k.jsonl?x=1@8")


def test_groups_arm_refuses_plain_partition(toy, monkeypatch):
    with pytest.raises(SystemExit, match="family tiles"):
        _battery(monkeypatch, toy, ROOT / "scripts/amoeba_box_battery.py", "x", "base", "--arms", "groups")


def test_unclaimed_arm_refuses_foreign_mask(toy, monkeypatch):
    U = np.load(toy / "base" / "unassigned.npz")
    bad = U["unassigned"].copy()
    bad[:, 16] = True                                                   # box a's units: not in the core
    np.savez(toy / "bad.npz", layers=U["layers"], unassigned=bad)
    with pytest.raises(SystemExit, match="outside the core"):
        _battery(monkeypatch, toy, ROOT / "scripts/amoeba_box_battery.py", "x", "base", "--arms", "unclaimed",
                 "--unclaimed", "bad.npz")


def test_meta_written_at_start_survives_a_crash(toy, monkeypatch):
    """2026-11-19: a battery killed mid-run left battery/ empty; battery.meta.json (held specs, notes, skips) is now
    written before the model loads and again once the held sets load, so scripts/amoeba_fn_interactions.py can read
    it after a crash. The end-of-run write still replaces it."""
    def boom(*a, **k):
        raise RuntimeError("killed mid-run")
    _install_fakes(monkeypatch)
    monkeypatch.setattr(sys.modules["research.amoeba.carve.evals"], "english_cost", boom)
    mod = _load(ROOT / "scripts/amoeba_box_battery.py")
    held = [x for b in list(P.BOXES) + ["eng"] for x in ("--held", f"{b}=held/{b}.jsonl@8")]
    monkeypatch.chdir(toy)
    monkeypatch.setattr(sys, "argv", ["battery", "--model", "m", "--partition", "base", "--out", "crash", *held,
                                      "--held", "gsm8k=gsm8k:held/gsm8k.jsonl@8", "--held-note", "a=toy note",
                                      "--docs", "4"])
    with pytest.raises(RuntimeError):
        mod.main()
    meta = json.loads((toy / "crash" / "battery.meta.json").read_text())
    assert meta["status"] == "scoring" and meta["fingerprint"] == "f" * 64 and meta["docs"] == 4
    assert set(meta["held"]) == set(P.BOXES) | {"eng", "gsm8k"}
    assert meta["held"]["a"] == {"spec": "a=held/a.jsonl@8", "docs_loaded": 4}
    assert meta["held_notes"]["a"] == "toy note" and meta["held_notes"]["gsm8k"].startswith("GSM8K test rows")
    assert meta["skipped"] == {} and meta["partition"] == "base"
    _battery(monkeypatch, toy, ROOT / "scripts/amoeba_box_battery.py", "crash", "base")    # resumed to the end
    meta = json.loads((toy / "crash" / "battery.meta.json").read_text())
    assert meta["status"] == "finished" and meta["checkpoint"] == "toy@ffffffff"
