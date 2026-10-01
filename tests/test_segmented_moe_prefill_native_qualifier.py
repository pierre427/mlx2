"""Host-only tests of the segmented MoE native qualifier harness.

These exercise the harness's own orchestration, evaluation and source/build
admission with a CPU fake backend marked SUBSTITUTED. They never import MLX
(blocked before collection-time module import and around every test, with
cached MLX modules hidden), never query a device, never construct
NativeBackend and never build or run a Metal kernel. They are NOT native
numerical qualification of the candidate.
"""

import contextlib
import importlib.abc
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/qualify_segmented_moe_prefill_research.py"
PROVENANCE = ROOT / "provenance/segmented-moe-prefill-native-qualifier.json"
MODULE_NAME = "mlx2_segmented_moe_native_qualifier"


class _BlockMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if _is_mlx(name):
            raise ImportError(f"host-only test: MLX import blocked ({name})")
        return None


def _is_mlx(name):
    return name == "mlx" or name.startswith("mlx.")


@contextlib.contextmanager
def _mlx_blocked():
    hidden = {k: sys.modules.pop(k) for k in list(sys.modules) if _is_mlx(k)}
    blocker = _BlockMLX()
    sys.meta_path.insert(0, blocker)
    try:
        yield
    finally:
        sys.meta_path.remove(blocker)
        leaked = [k for k in sys.modules if _is_mlx(k)]
        for k in leaked:
            del sys.modules[k]
        sys.modules.update(hidden)
    assert not leaked, leaked


def _load():
    """The harness module (a mutation runner may pre-install an in-memory variant under MODULE_NAME)."""
    mod = sys.modules.get(MODULE_NAME)
    if mod is None:
        spec = importlib.util.spec_from_file_location(MODULE_NAME, SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[MODULE_NAME] = mod
        spec.loader.exec_module(mod)
    return mod


with _mlx_blocked():                       # collection-time: blocked before the harness/candidate import
    q = _load()
    cand = q._candidate()


def _no_native(*_a, **_k):
    raise AssertionError("NativeBackend must never be constructed in host tests")


@pytest.fixture(autouse=True)
def no_mlx(monkeypatch):
    monkeypatch.setattr(q.NativeBackend, "__init__", _no_native)
    with _mlx_blocked():
        yield


SMALL = ["d35-t128-uniform", "d35-t1-decode", "d35-t4-mtp-verify", "d35-t127-below"]


# ------------------------------------------------------------------ fake backend (substituted)

def _pattern(shape, seed):
    """Finite, nonzero bf16 bit patterns of magnitude [1, 2) (uint16 only: the seam cells stay cheap)."""
    return np.random.default_rng(seed).integers(0x3F80, 0x4000, size=shape, dtype=np.uint16)


def _eng(kind, prefix="substituted", **override):
    eng = dict.fromkeys([k for k in cand.ENGAGEMENT.snapshot() if k != "epoch"], 0)
    eng.update({"calls": 1, f"{prefix}.successful_chains.{kind}": 1})
    eng.update(override)
    return eng


def tiny_tables(geometry):
    one = {"weight": np.zeros(1, np.uint32), "scales": np.zeros(1, np.uint16), "biases": np.zeros(1, np.uint16)}
    return {"gate_up": dict(one), "down": dict(one)}


class FakeBackend:
    """CPU stand-in for the protocol; outputs are substituted, never native. Ties are broken in
    REVERSE index order, so the harness must not assume the stable host tie order."""

    def __init__(self, tamper=None, raise_at=None, prefix="substituted", native=False, dispatch_claim=None):
        self.tamper, self.raise_at, self.prefix, self.native = tamper or {}, raise_at, prefix, native
        self.dispatch_claim = dispatch_claim or {}
        self.events, self.dispatched = [], 0

    def identity(self):
        return {"backend": "FakeBackend", "substituted": True}

    def load_tables(self, geometry, host):
        self.events.append(("load", geometry))

    def drop_tables(self):
        self.events.append(("drop",))

    def release(self):
        self.events.append(("release",))

    def _dispatch(self, guard):
        guard()
        self.dispatched += 1
        if self.raise_at == self.dispatched:
            raise RuntimeError("fake backend failure")

    def _emit(self, emit, stage, payload):
        if stage in self.dispatch_claim:
            payload["dispatches"] = self.dispatch_claim[stage]
        fn = self.tamper.get(stage)
        emit(stage, fn(payload) if fn else payload)

    def _proj(self, kind, cand_bits, ref_bits, pp):
        return {"candidate": cand_bits, "reference": ref_bits, "dispatches": 1, "kind": kind, "native": self.native,
                "engagement": _eng(kind, self.prefix), "plan": q.plan_record(pp)}

    def run_cell(self, cell, inputs, emit, guard):
        self.events.append(("cell", cell["id"]))
        T, k, E, D, I = (cell[x] for x in ("tokens", "top_k", "experts", "hidden", "intermediate"))
        n, M = cell["assignments"], cell["rows"]
        flat = inputs["ids"].reshape(-1).astype(np.int64)
        order = np.lexsort((-np.arange(n), flat))
        inv = np.argsort(order)
        sid = np.concatenate([flat[order], np.repeat(flat[order][-1:], M - n)]).astype(np.uint32)
        row_map = np.concatenate([order // k, np.repeat(order[-1:] // k, M - n)])
        self._dispatch(guard)
        self._dispatch(guard)
        self._emit(emit, "routes", {"sorted_ids": sid, "inv_order": inv.astype(np.uint32),
                                    "x_sorted_bits": inputs["x_bits"][row_map].reshape(M, 1, D), "dispatches": 2})
        hidden, down = _pattern((M, 1, I), 1), _pattern((M, 1, D), 2)
        restored = down.reshape(M, D)[inv].reshape(T, k, 1, D)
        scans = {}
        for v in q.variants(cell):
            pinned, lab = (None if v is None else cand.Plan(*v)), q.variant_label(v)
            gu = cand.projection_plan(M, E, D, 2 * I, paired=True, pinned=pinned)
            dn = cand.projection_plan(M, E, I, D, paired=False, pinned=pinned)
            self._dispatch(guard)
            self._emit(emit, f"gate_up:{lab}", self._proj(q.KINDS[0], hidden.copy(), hidden, gu))
            self._dispatch(guard)
            self._emit(emit, f"down_isolated:{lab}", self._proj(q.KINDS[1], down.copy(), down, dn))
            if v is None:
                self._dispatch(guard)
                self._dispatch(guard)
                payload = self._proj(q.KINDS[1], restored.copy(), restored, dn)
                payload.update(candidate_sorted=down.copy(), dispatches=2)
                self._emit(emit, "chain_restored", payload)
            for pp in (gu, dn):
                scans.setdefault(pp.plan.bm, pp.max_tiles)
        for bm, max_tiles in sorted(scans.items()):
            tiles = q.tile_law(sid, E, bm)
            self._dispatch(guard)
            self._emit(emit, f"scan:bm{bm}", {"tiles": tiles, "count": int(len(tiles)), "bm": bm,
                                              "max_tiles": max_tiles, "dispatches": 1})


class _ExtraStage(FakeBackend):
    def run_cell(self, cell, inputs, emit, guard):
        super().run_cell(cell, inputs, emit, guard)
        emit("gate_up:planner", {})


def _run(ids=SMALL, backend=None, guard=None):
    cells, full = q.select_cells(ids)
    backend = backend or FakeBackend()
    report = q.run_gate(backend, cells, guard=guard, tables=tiny_tables)
    return report, cells, full, backend


def _evaluate(**kw):
    report, cells, full, _ = _run(**kw)
    return q.evaluate(report, cells, full_requested=full)


def _failed_with(evaluation, text):
    assert evaluation["verdict"] == "failed", evaluation["verdict"]
    assert any(text in r for r in evaluation["refusals"]), evaluation["refusals"][:5]


def _stage(stage, fn):
    return FakeBackend(tamper={stage: fn})


# ------------------------------------------------------------------ host-only CLI (fresh processes)

_PRELUDE = """
import importlib.abc, runpy, sys
class B(importlib.abc.MetaPathFinder):
    def find_spec(self, n, p=None, t=None):
        if n == "mlx" or n.startswith("mlx."):
            raise ImportError("blocked " + n)
sys.meta_path.insert(0, B())
script, argv = sys.argv[1], sys.argv[2:]
code = 0
if argv == ["--import-only"]:
    import importlib.util
    spec = importlib.util.spec_from_file_location("q", script)
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
else:
    sys.argv = [script] + argv
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as e:
        code = e.code or 0
assert not [m for m in sys.modules if m == "mlx" or m.startswith("mlx.")]
print("HOST-ONLY-EXIT", code)
"""


def _fresh(*argv):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    env.pop("MLX2_INTAKE_SOURCE_COMMIT", None)
    return subprocess.run([sys.executable, "-c", _PRELUDE, str(SCRIPT), *argv], capture_output=True, text=True,
                          cwd=ROOT, env=env, timeout=120)


@pytest.mark.parametrize("argv,code", [(["--import-only"], 0), (["--help"], 0), (["--catalogue"], 0),
                                       (["--timing", "--run-native", "--i-own-the-gpu"], 1),
                                       (["--run-native", "--source-root", str(ROOT)], 1)])
def test_fresh_process_paths_are_host_only(argv, code):
    proc = _fresh(*argv)
    assert f"HOST-ONLY-EXIT {code}" in proc.stdout, proc.stderr[-2000:]
    if argv == ["--catalogue"]:
        listing = json.loads(proc.stdout.rsplit("HOST-ONLY-EXIT", 1)[0])
        assert set(listing["geometries"]) == {"default35b", "flash_next"}
    if argv[0] == "--timing":
        assert "timing is not implemented" in proc.stdout
    if argv[0] == "--run-native":
        assert "acknowledgement missing" in proc.stdout


def test_timing_refused_before_admission_imports_or_backend(monkeypatch, capsys):
    for name in ("native_admission", "_candidate", "_run_native", "select_cells"):
        monkeypatch.setattr(q, name, _no_native)
    assert q.main(["--timing", "--run-native", "--i-own-the-gpu", "--source-root", str(ROOT)]) == 1
    assert "timing is not implemented" in capsys.readouterr().out


def test_admission_itself_refuses_timing():
    with pytest.raises(q.Refused, match="timing"):
        q.native_admission(q.build_parser().parse_args(["--timing", "--run-native"]), {}, root=ROOT)


def test_harness_has_no_module_scope_mlx_import_or_copyright_header():
    top = [line for line in SCRIPT.read_text().splitlines() if line.startswith(("import ", "from "))]
    assert not [line for line in top if line.split()[1].split(".")[0] == "mlx"]
    for path in (SCRIPT, Path(__file__)):
        assert not any("Copyright" in line for line in path.read_text().splitlines()[:5])


# ------------------------------------------------------------------ catalogue law

def test_exactly_two_named_geometries():
    assert q.GEOMETRIES == {"default35b": {"experts": 256, "top_k": 8, "hidden": 2048, "intermediate": 512},
                            "flash_next": {"experts": 512, "top_k": 10, "hidden": 2560, "intermediate": 640}}
    assert q.GEOMETRIES == {k: dict(v) for k, v in cand.NAMED_GEOMETRIES.items()}
    assert {c["geometry"] for c in q.CATALOGUE} == set(q.GEOMETRIES)
    for c in q.CATALOGUE:
        assert {x: c[x] for x in ("experts", "top_k", "hidden", "intermediate")} == q.GEOMETRIES[c["geometry"]]
    cells, full = q.select_cells()
    assert full is True and [c["id"] for c in cells] == list(q.MANDATORY)


def _admits(c):
    try:
        cand.admit(cand.MoEPrefillRequest(tokens=c["tokens"], top_k=c["top_k"], experts=c["experts"],
                                          hidden=c["hidden"], intermediate=c["intermediate"]))
        return None
    except cand.SegmentedMoERefused as exc:
        return exc.reason


def test_boundaries_decode_and_verify_cells():
    table = q.catalogue_by_id()
    for cid, reason in (("d35-t128-uniform", None), ("d35-t128-subset", None), ("d35-t127-below", "too_small"),
                        ("d35-t1-decode", "too_small"), ("d35-t4-mtp-verify", "too_small"),
                        ("fn-t205-uniform", None), ("fn-t205-subset", None), ("fn-t204-below", "too_small"),
                        ("fn-t128-below", "too_small"), ("fn-t1-decode", "too_small"),
                        ("fn-t4-mtp-verify", "too_small")):
        assert _admits(table[cid]) == reason, cid
        assert table[cid]["expect"] == ("refuse" if reason else "pass")
    assert cand.min_admitted_tokens(8, 256) == 128 and cand.min_admitted_tokens(10, 512) == 205
    assert all(_admits(c) is None for c in q.CATALOGUE if c["expect"] == "pass")


def test_one_seam_pad_cell_per_geometry():
    seams = [c for c in q.CATALOGUE if c["assignments"] > q.SEAM_THRESHOLD]
    assert sorted(c["geometry"] for c in seams) == ["default35b", "flash_next"]
    for c in seams:
        assert c["pad"] == cand.seam_pad(c["assignments"]) > 0 and c["rows"] % 64 == 0 and c["expect"] == "pass"
    assert all(c["pad"] == 0 for c in q.CATALOGUE if c not in seams)


def _counts(cid):
    c = q.catalogue_by_id()[cid]
    return c, np.bincount(q.cell_law(c)["flat"], minlength=c["experts"])


def test_router_laws_unique_exact_topk_and_distributions():
    for c in q.CATALOGUE:
        ids = q.router_ids(c)
        assert ids.shape == (c["tokens"], c["top_k"]) and ids.dtype == np.uint32
        assert all(len(set(row)) == c["top_k"] for row in ids.tolist()) and int(ids.max()) < c["experts"]
    _, cnt = _counts("d35-t128-subset")
    assert (cnt == 0).sum() == 224 and cnt[0] == 0 and cnt[-1] == 0 and set(np.flatnonzero(cnt) % 8) == {3}
    for cid in ("d35-t512-zipf", "fn-t410-zipf"):
        c, cnt = _counts(cid)
        assert (cnt == 0).sum() >= 50 and cnt.max() >= 0.98 * c["tokens"] and cnt.max() > 10 * cnt[cnt > 0].mean()
    for cid in ("d35-t300-hot-edges", "fn-t300-hot-edges"):
        c, cnt = _counts(cid)
        assert cnt[0] == cnt[-1] == c["tokens"]
    for cid in ("d35-t205-uniform", "fn-t333-uniform"):
        _, cnt = _counts(cid)
        assert ((cnt % 64) != 0).sum() > 0 and cnt.max() < 64


def test_law_is_seed_bound_and_cell_specific():
    table = q.catalogue_by_id()
    a, b = q.cell_law(table["d35-t128-uniform"]), q.cell_law(table["d35-t128-uniform"])
    assert a["hashes"] == b["hashes"]
    other = q.cell_law(table["d35-t128-subset"])["hashes"]
    assert other["router_ids"] != a["hashes"]["router_ids"] and other["x_bits"] != a["hashes"]["x_bits"]


def test_plan_coverage_and_sweep():
    for g in q.GEOMETRIES:
        seen = {"gate_up": set(), "down": set()}
        for c in q.CATALOGUE:
            if c["geometry"] != g or c["expect"] != "pass":
                continue
            for name, e in q.expectations(c, cand).items():
                if e["kind"] in q.KINDS and name != "chain_restored":
                    seen[name.split(":")[0].replace("_isolated", "")].add(e["plan"]["effective"][0])
        assert seen == {"gate_up": {0, 1}, "down": {0, 1}}, (g, seen)
    for p in q.PLAN_SWEEP:
        cand.validate_plan(cand.Plan(*p))
    sweep = q.expectations(q.catalogue_by_id()["fn-t615-plan-sweep"], cand)
    assert {k for k in sweep if k.startswith("scan:")} == {"scan:bm64", "scan:bm96", "scan:bm128"}
    assert all(e["candidate_mirror_agrees"] for e in sweep.values() if e["kind"] == "scan")


def test_tables_are_packed_seed_bound_and_tiny_here():
    dims = {"experts": 3, "hidden": 128, "intermediate": 64}
    a, b = q.host_tables("default35b", dims=dims), q.host_tables("default35b", dims=dims)
    assert q.table_hashes(a) == q.table_hashes(b)
    assert q.table_hashes(a) != q.table_hashes(q.host_tables("flash_next", dims=dims))
    gu, dn = a["gate_up"], a["down"]
    assert gu["weight"].shape == (3, 128, 16) and gu["scales"].shape == (3, 128, 2)
    assert dn["weight"].shape == (3, 128, 8) and dn["scales"].shape == (3, 128, 1)
    s, bias = q.bf16_bits_to_f32(gu["scales"]), q.bf16_bits_to_f32(gu["biases"])
    assert np.isfinite(s).all() and (s > 0).all() and (bias < 0).all() and np.isfinite(bias).all()
    nibbles = (gu["weight"][..., None] >> (4 * np.arange(8, dtype=np.uint32))) & 0xF
    assert set(np.unique(nibbles).tolist()) == set(range(16))


def test_bf16_bits_round_to_nearest_even():
    vals = np.array([1.0, -2.0, 1.0 + 2**-8, 1.0 + 3 * 2**-8, 1.0 + 2**-8 + 2**-20], np.float32)
    assert q.f32_to_bf16_bits(vals).tolist() == [0x3F80, 0xC000, 0x3F80, 0x3F82, 0x3F81]
    with pytest.raises(q.Refused):
        q.f32_to_bf16_bits(np.array([np.nan], np.float32))
    assert q.nonfinite_count(np.array([0x7F80, 0xFF80, 0x7FC0, 0x7F7F], np.uint16)) == 3


def test_chunked_bit_summary_counts_across_chunk_boundaries(monkeypatch):
    monkeypatch.setattr(q, "_chunks", lambda a, size=3: (a.reshape(-1)[i:i + size] for i in range(0, a.size, size)))
    ref = np.full((4, 1, 5), 0x3F80, np.uint16)
    cand_bits = ref.copy()
    cand_bits[1, 0, 4] ^= 0x10
    cand_bits[3, 0, 0] ^= 0x1
    ref[2, 0, 2] = cand_bits[2, 0, 2] = 0x7FC0
    s = q.summarize_bits([4, 1, 5], cand_bits, ref)
    assert (s["bit_mismatches"], s["first_mismatch"], s["max_bit_delta"]) == (2, [1, 0, 4], 0x10)
    assert (s["candidate_nonfinite"], s["reference_nonfinite"], s["reference_nonzero_fraction"]) == (1, 1, 1.0)


def test_real_installed_header_closure_is_read_as_files_only():
    k = "mlx/backend/metal/kernels/"
    assert q.header_closure(q.mlx_base() / "include", q.MLX_GATE_UP_HEADERS) == [
        k + "steel/gemm/nax.h", k + "steel/defines.h", k + "steel/utils/integral_constant.h",
        k + "steel/utils/type_traits.h", k + "unary_ops.h", k + "cexpf.h", k + "erf.h", k + "expm1f.h", k + "fp8.h",
        k + "binary_ops.h"]


# ------------------------------------------------------------------ orchestration with the substituted fake

def test_fake_partial_run_is_evidence_only():
    report, cells, full, backend = _run()
    ev = q.evaluate(report, cells, full_requested=full)
    json.dumps({"report": report, "evaluation": ev})       # a receipt needs no lossy default= conversion
    assert ev["verdict"] == "partial_bit_identity_evidence", ev["refusals"][:5]
    assert ev["chains"] == {"native": 0, "substituted": 3} and ev["expected_chains"] == 3
    assert ev["native_synthetic_gate"] is False and ev["qualified"] is False and ev["model_gain"] is False
    assert [e for e in backend.events if e[0] == "cell"] == [("cell", "d35-t128-uniform")]
    assert backend.events[-2:] == [("release",), ("drop",)]
    assert report["cells"][0]["evidence"]["stages"]["routes"]["tie_order_equals_stable_host"] is False
    verdict = q.native_verdict(report)
    assert verdict["native_synthetic_gate"] is False
    assert any("no live native orchestration" in r for r in verdict["reasons"])


def test_forged_native_report_and_live_lookalikes_never_stamp():
    report, cells, _, backend = _run(backend=FakeBackend(prefix="native", native=True))
    report["producer"] = "NativeBackend"
    report["identity"] = {"backend": "NativeBackend", "architecture": "applegpu_g17s"}
    ev = q.evaluate(report, cells, full_requested=False)
    assert ev["chains"]["native"] == 3 and ev["native_synthetic_gate"] is False
    rt = json.loads(json.dumps(report, default=str))
    for obj in (report, rt, {"report": rt, "cells": cells, "_witness": None}):
        assert q.native_verdict(obj)["native_synthetic_gate"] is False
    live = q._LiveNativeRun(q._WITNESS, backend, {}, report, [], cells, False, {})
    assert "backend is not the native backend" in q.native_verdict(live)["reasons"]
    wrong = q._LiveNativeRun(object(), backend, {}, report, [], cells, False, {})
    assert any("no live native" in r for r in q.native_verdict(wrong)["reasons"])
    sub = type("Sub", (q.NativeBackend,), {})
    live = q._LiveNativeRun(q._WITNESS, object.__new__(sub), {}, report, [], cells, False, {})
    assert "backend is not the native backend" in q.native_verdict(live)["reasons"]
    look = type("_LiveNativeRun", (), {})()
    look.report, look.cells, look.full_requested, look._witness = report, cells, True, q._WITNESS
    assert q.native_verdict(look)["native_synthetic_gate"] is False


def test_native_verdict_requires_matching_fresh_native_counters(monkeypatch):
    """HOST ASSOCIATION CHECK, not native execution: the actual full catalogue through the substituted
    FakeBackend, run_gate and the captured evaluator (no evaluator injection); an UNINITIALIZED original
    NativeBackend instance only reaches the counter and post-run checks."""
    report, cells, full, _ = _run(ids=None, backend=FakeBackend(prefix="native", native=True))
    ev = q.evaluate(report, cells, full_requested=full)
    per_kind = {k: 0 for k in q.KINDS}
    for c in cells:
        if c["expect"] == "pass":
            for e in q.expectations(c, cand).values():
                if e["kind"] in q.KINDS:
                    per_kind[e["kind"]] += 1
    assert full and len(cells) == len(q.CATALOGUE) == 21
    assert per_kind == {"gate_up_mapped_swiglu": 26, "down_segmented": 40}
    assert ev["verdict"] == "bit_identity_evidence_pass", ev["refusals"][:5]
    assert ev["chains"] == {"native": 66, "substituted": 0} and ev["expected_chains"] == 66
    native = object.__new__(q.NativeBackend)
    down = "native.successful_chains.down_segmented"
    good = {"calls": 66, **{f"native.successful_chains.{k}": v for k, v in per_kind.items()}}

    def stamp(fresh, post=()):
        live = q._LiveNativeRun(q._WITNESS, native, {}, report, list(post), cells, True, fresh)
        return q.native_verdict(live)
    v = stamp(good)
    assert v["native_synthetic_gate"] is True and v["reasons"] == []
    assert ({k: v["evaluation"][k] for k in ("verdict", "chains", "expected_chains", "refusals")}
            == {k: ev[k] for k in ("verdict", "chains", "expected_chains", "refusals")})
    for fresh in ({}, dict(good, **{down: 39}), dict(good, backend_raised=1),
                  dict(good, **{"substituted.successful_chains.down_segmented": 1}), dict(good, calls=67),
                  dict(good, calls=65, **{down: 39})):
        assert stamp(fresh)["native_synthetic_gate"] is False, fresh
    v = stamp(good, ["git HEAD differs from the admitted commit"])
    assert v["native_synthetic_gate"] is False and v["reasons"] == ["git HEAD differs from the admitted commit"]


def test_rebinding_module_names_cannot_make_a_fake_native(monkeypatch):
    report, cells, _, fake = _run(ids=["d35-t128-uniform"], backend=FakeBackend(prefix="native", native=True))
    original_live, witness = q._LiveNativeRun, q._WITNESS
    monkeypatch.setattr(q, "NativeBackend", FakeBackend)
    monkeypatch.setattr(q, "_LiveNativeRun", type("_LiveNativeRun", (), {}))
    monkeypatch.setattr(q, "evaluate", lambda r, c, full_requested: {
        "verdict": "bit_identity_evidence_pass", "chains": {"native": 3, "substituted": 0}, "expected_chains": 3})
    fresh = {"calls": 3, "native.successful_chains.gate_up_mapped_swiglu": 1,
             "native.successful_chains.down_segmented": 2}
    v = q.native_verdict(original_live(witness, fake, {}, report, [], cells, True, fresh))
    assert v["native_synthetic_gate"] is False and "backend is not the native backend" in v["reasons"]
    assert any("did not pass the full mandatory catalogue" in r for r in v["reasons"])
    with pytest.raises(AssertionError, match="never be constructed"):   # the captured class, not the rebound fake
        q._run_native({}, cells, False)


def test_full_catalogue_fake_with_native_counters_and_rebound_class_stays_non_native(monkeypatch):
    """Root falsifier 4a7d10cc: a full-catalogue fake report that passes evaluation, counters matching
    per stage, the real witness/run class and a rebound NativeBackend must not stamp native."""
    report, cells, full, fake = _run(ids=None, backend=FakeBackend(prefix="native", native=True))
    ev = q.evaluate(report, cells, full_requested=full)
    assert full and ev["verdict"] == "bit_identity_evidence_pass" and ev["chains"]["native"] == ev["expected_chains"]
    per_kind = {k: 0 for k in q.KINDS}
    for c in cells:
        if c["expect"] == "pass":
            for e in q.expectations(c, cand).values():
                if e["kind"] in q.KINDS:
                    per_kind[e["kind"]] += 1
    fresh = {"calls": sum(per_kind.values()), **{f"native.successful_chains.{k}": v for k, v in per_kind.items()}}
    monkeypatch.setattr(q, "NativeBackend", FakeBackend)
    v = q.native_verdict(q._LiveNativeRun(q._WITNESS, fake, {}, report, [], cells, True, fresh))
    assert v["native_synthetic_gate"] is False and v["reasons"] == ["backend is not the native backend"]


def _flip(stage, index=0):
    def fn(p):
        a = p["candidate"].copy()
        a.reshape(-1)[index] ^= 1
        p["candidate"] = a
        return p
    return _stage(stage, fn)


def test_single_bit_difference_fails_without_tolerance():
    _failed_with(_evaluate(backend=_flip("gate_up:planner")), "bit mismatch: 1 elements")
    _failed_with(_evaluate(backend=_flip("down_isolated:planner", index=777)), "bit mismatch: 1 elements")
    _failed_with(_evaluate(backend=_flip("chain_restored", index=3)), "bit mismatch: 1 elements")


@pytest.mark.parametrize("bits", [0x7FC0, 0x7F80, 0xFF80])
def test_identical_nonfinite_outputs_fail(bits):
    def both(p):
        for key in ("candidate", "reference"):
            a = p[key].copy()
            a.reshape(-1)[5] = bits
            p[key] = a
        return p
    _failed_with(_evaluate(backend=_stage("gate_up:planner", both)), "non-finite bits")


def test_vacuous_zero_outputs_fail():
    def zeros(p):
        p["candidate"], p["reference"] = np.zeros_like(p["candidate"]), np.zeros_like(p["reference"])
        return p
    _failed_with(_evaluate(backend=_stage("down_isolated:planner", zeros)), "vacuous reference")


@pytest.mark.parametrize("fn,text", [
    (lambda p: dict(p, candidate=p["candidate"][:-1]), "candidate: shape"),
    (lambda p: dict(p, candidate=p["candidate"].view(np.float16)), "not a bfloat16 bit"),
    (lambda p: dict(p, candidate=q.bf16_bits_to_f32(p["candidate"])), "not a bfloat16 bit"),
    (lambda p: dict(p, reference=None), "reference: not a bfloat16 bit"),
])
def test_shape_and_dtype_failures(fn, text):
    _failed_with(_evaluate(backend=_stage("gate_up:planner", fn)), text)


def _swap01(a):
    a = a.copy()
    a[[0, 1]] = a[[1, 0]]
    return a


@pytest.mark.parametrize("fn,check", [
    (lambda p: dict(p, sorted_ids=np.sort(p["sorted_ids"])[::-1].copy()), "nondecreasing"),
    (lambda p: dict(p, sorted_ids=np.where(p["sorted_ids"] == 7, 8, p["sorted_ids"]).astype(np.uint32)),
     "sorted_matches_router"),
    (lambda p: dict(p, x_sorted_bits=_swap01(p["x_sorted_bits"])), "x_sorted_is_row_map_copy"),
    (lambda p: dict(p, sorted_ids=np.concatenate([p["sorted_ids"], p["sorted_ids"][-1:]])), "equals_host_sorted"),
])
def test_route_tampering_fails(fn, check):
    _failed_with(_evaluate(backend=_stage("routes", fn)), check)


def test_supplied_route_summary_must_match_the_router_law():
    report, cells, full, _ = _run()
    report["cells"][0]["evidence"]["stages"]["routes"]["sorted_sha256"] = "0" * 64   # checks still all True
    _failed_with(q.evaluate(report, cells, full_requested=full), "sorted expert ids differ from the router law")


@pytest.mark.parametrize("fn", [lambda p: dict(p, inv_order=np.zeros_like(p["inv_order"])),
                                lambda p: dict(p, sorted_ids=p["sorted_ids"].astype(np.int64)),
                                lambda p: dict(p, inv_order=p["inv_order"][:-1].copy())])
def test_unusable_ordinary_routes_stop_the_run(fn):
    backend = _stage("routes", fn)
    with pytest.raises(q.Refused, match="ordinary routes unusable"):
        _run(backend=backend)
    assert backend.events[-2:] == [("release",), ("drop",)]


def test_restoration_tampering_fails():
    def permute_both(p):
        p["candidate"], p["reference"] = _swap01(p["candidate"]), _swap01(p["reference"])
        return p
    ev = _evaluate(backend=_stage("chain_restored", permute_both))
    _failed_with(ev, "candidate_restore_is_inverse")
    _failed_with(ev, "reference_restore_matches_host_inverse")
    _failed_with(_evaluate(backend=_stage("chain_restored", lambda p: dict(p, candidate_sorted=None))),
                 "candidate_restore_is_inverse")


K0 = "gate_up_mapped_swiglu"


@pytest.mark.parametrize("eng,flag,text", [
    ({}, False, "engagement missing"),
    ({"calls": 1, "backend_raised": 0}, False, "not one fresh successful"),
    (_eng(K0, **{f"substituted.successful_chains.{K0}": 2}), False, "not one fresh successful"),
    (_eng(K0, backend_raised=1), False, "not one fresh successful"),
    (_eng(K0, **{"refused.too_small": 1}), False, "unexpected engagement deltas"),
    (_eng(K0, **{f"native.successful_chains.{K0}": 1}), False, "not one fresh successful"),
    (_eng(K0), True, "native flag disagrees"),
])
def test_engagement_failures(eng, flag, text):
    fn = (lambda p: dict(p, engagement=eng, native=True if flag else p["native"]))
    _failed_with(_evaluate(backend=_stage("gate_up:planner", fn)), text)


@pytest.mark.parametrize("fn,text", [
    (lambda p: dict(p, tiles=p["tiles"][:-1].copy(), count=p["count"] - 1), "rows_covered_once"),
    (lambda p: dict(p, tiles=np.concatenate([p["tiles"], p["tiles"][-1:]]), count=p["count"] + 1),
     "rows_covered_once"),
    (lambda p: dict(p, tiles=np.where(np.arange(4) == 1, p["tiles"] + 1, p["tiles"]).astype(np.uint32)),
     "single_expert_tiles"),
    (lambda p: dict(p, count=p["count"] + 0.0), "not a [count, 4]"),
    (lambda p: dict(p, tiles=p["tiles"][::-1].copy()), "equals_tile_law"),
])
def test_scan_probe_tampering_fails(fn, text):
    _failed_with(_evaluate(backend=_stage("scan:bm64", fn)), text)


def test_plan_and_kind_mismatch_fail():
    def plan(p):
        p["plan"] = dict(p["plan"], describe="seg 128x64 bk128 gx32 pad8192")
        return p
    _failed_with(_evaluate(backend=_stage("down_isolated:planner", plan)), "differ from admission")
    _failed_with(_evaluate(backend=_stage("gate_up:planner", lambda p: dict(p, kind=q.KINDS[1]))),
                 "differ from admission")


def test_dispatches_must_each_follow_a_guard():
    _failed_with(_evaluate(backend=FakeBackend(dispatch_claim={"gate_up:planner": 3})), "not each preceded")


def test_refuse_cells_never_reach_the_backend_and_carry_no_outputs():
    report, cells, full, backend = _run()
    assert {e[1] for e in backend.events if e[0] == "cell"} == {"d35-t128-uniform"}
    for record in report["cells"][1:]:
        assert record["evidence"]["refused"] == "too_small" and "stages" not in record["evidence"]
        assert record["evidence"]["route"] == cand.ORDINARY_REFERENCE
    report["cells"][1]["evidence"]["stages"] = {}
    _failed_with(q.evaluate(report, cells, full_requested=full), "refused before any backend")


def test_full_gate_requires_the_complete_catalogue_and_rejects_verdicts_timing_and_drift():
    report, cells, _, _ = _run()
    _failed_with(q.evaluate(report, cells, full_requested=True), "incomplete")
    _failed_with(q.evaluate({"cells": report["cells"][:1]}, cells, full_requested=False), "missing cell")
    forged = json.loads(json.dumps(report, default=str))
    forged["cells"][0]["evidence"]["verdict"] = "bit_identity_evidence_pass"
    _failed_with(q.evaluate(forged, cells, full_requested=False), "verdict fields")
    timed = dict(report, cells=[dict(report["cells"][0], timing={"ms": 1.0})] + report["cells"][1:])
    _failed_with(q.evaluate(timed, cells, full_requested=False), "timing evidence")
    drift = json.loads(json.dumps(report, default=str))
    drift["cells"][0]["evidence"]["input_hashes"]["x_bits"] = "0" * 64
    _failed_with(q.evaluate(drift, cells, full_requested=False), "input identity")


def test_table_identity_must_agree_within_a_geometry():
    report, cells, _, _ = _run(ids=["d35-t128-uniform", "d35-t205-uniform"])
    assert q.evaluate(report, cells, full_requested=False)["verdict"] == "partial_bit_identity_evidence"
    tables = json.loads(json.dumps(report["cells"][1]["evidence"]["tables"]))
    tables["down"]["weight"]["sha256"] = "1" * 64
    report["cells"][1]["evidence"]["tables"] = tables
    _failed_with(q.evaluate(report, cells, full_requested=False), "table identity differs")


def test_recorder_rejects_unexpected_or_duplicate_stages():
    with pytest.raises(q.Refused, match="duplicate or unexpected stage"):
        _run(backend=_ExtraStage())


def test_guard_failure_stops_before_the_next_dispatch_and_cleans_up():
    calls = {"n": 0}

    def guard():
        calls["n"] += 1
        if calls["n"] == 4:
            raise q.Refused("bound source files changed since admission")
    backend = FakeBackend()
    with pytest.raises(q.Refused, match="changed since admission"):
        _run(backend=backend, guard=guard)
    assert backend.dispatched == 1                       # the dispatch behind the failing guard never ran
    assert backend.events[-2:] == [("release",), ("drop",)]


def test_partial_backend_failure_propagates_and_cleans_up():
    backend = FakeBackend(raise_at=4)
    with pytest.raises(RuntimeError, match="fake backend failure"):
        _run(ids=["d35-t128-uniform", "d35-t205-uniform"], backend=backend)
    assert [e for e in backend.events if e[0] == "cell"] == [("cell", "d35-t128-uniform")]
    assert backend.events[-2:] == [("release",), ("drop",)]


def test_main_never_writes_a_receipt_for_refused_or_failed_runs(tmp_path, monkeypatch, capsys):
    out = tmp_path / "receipt.json"
    assert q.main(["--run-native", "--source-root", str(ROOT), "--out", str(out)], environ={}) == 1
    assert not out.exists() and "acknowledgement" in capsys.readouterr().out
    monkeypatch.setattr(q, "native_admission", lambda args, environ: {"commit": "x"})

    def died(*_a):
        raise RuntimeError("[metal] backend died")
    monkeypatch.setattr(q, "_run_native", died)
    assert q.main(["--run-native", "--i-own-the-gpu", "--out", str(out)], environ={}) == 1
    assert not out.exists() and "backend died" in capsys.readouterr().out
    out.write_text("previous")
    with pytest.raises(FileExistsError):
        q.write_receipt(out, {"new": True})
    assert out.read_text() == "previous"
    with pytest.raises(TypeError):
        q.write_receipt(tmp_path / "unserializable.json", {"bad": object()})
    assert not (tmp_path / "unserializable.json").exists()


def test_receipt_path_appearing_during_the_run_is_never_overwritten(tmp_path, monkeypatch, capsys):
    out = tmp_path / "receipt.json"
    monkeypatch.setattr(q, "native_admission", lambda args, environ: {"commit": "x"})

    def racing_run(*_a):
        out.write_text("someone else's receipt")
        return types.SimpleNamespace(report={}, cells=(), admission={}, fresh={})
    monkeypatch.setattr(q, "_run_native", racing_run)
    assert q.main(["--run-native", "--i-own-the-gpu", "--out", str(out)], environ={}) == 1
    assert "receipt not written" in capsys.readouterr().out and out.read_text() == "someone else's receipt"


@pytest.mark.parametrize("make,text", [(lambda d: None, "required"), (lambda d: d / "r.json", "exists"),
                                       (lambda d: d / "missing" / "r.json", "missing or not writable")])
def test_output_preflight(tmp_path, make, text):
    (tmp_path / "r.json").write_text("{}")
    path = make(tmp_path)
    assert text in q.output_refusal(None if path is None else str(path))
    assert q.output_refusal(str(tmp_path / "fresh.json")) is None


# ------------------------------------------------------------------ source / build admission (tmp git + fake MLX)

_K = "mlx/backend/metal/kernels/"
_INC = '#include "mlx/backend/metal/kernels/{}"'
_HEADERS = {
    _K + "steel/gemm/nax.h": ["#include <metal_stdlib>", _INC.format("steel/defines.h"),
                              _INC.format("steel/utils/integral_constant.h")],
    _K + "unary_ops.h": [_INC.format(h) for h in ("cexpf.h", "erf.h", "expm1f.h", "fp8.h")],
    _K + "utils.h": [_INC.format("bf16.h")],
}
_LEAVES = ("steel/defines.h", "steel/utils/integral_constant.h", "cexpf.h", "erf.h", "expm1f.h", "fp8.h",
           "binary_ops.h", "bf16.h", "bf16_math.h", "complex.h", "defines.h", "logging.h")


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t", "-c",
                    "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false", *args],
                   check=True, capture_output=True)


def _head(repo):
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()


@pytest.fixture
def tree(tmp_path, monkeypatch):
    repo = (tmp_path / "mlx2").resolve()
    for rel in q.BOUND_FILES + ("src/mlx2/other.py",):
        dst = repo / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((ROOT / rel).read_bytes() if (ROOT / rel).exists() else b"x = 1" + bytes([10]))
    (repo / ".gitignore").write_text("*.so" + chr(10) + "__pycache__/" + chr(10))
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "fixture")
    base = (tmp_path / "site" / "mlx").resolve()
    for rel, body in [("core.cpython-312-darwin.so", "core"), ("lib/libmlx.dylib", "lib"),
                      ("lib/mlx.metallib", "metallib"), ("nn/__init__.py", "")]:
        (base / rel).parent.mkdir(parents=True, exist_ok=True)
        (base / rel).write_text(body)
    for rel in [_K + h for h in _LEAVES] + list(_HEADERS):
        p = base / "include" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(chr(10).join(_HEADERS.get(rel, ["// leaf"])) + chr(10))
    monkeypatch.setattr(q, "mlx_base", lambda: base)
    monkeypatch.setattr(q, "mlx_version", lambda: "0.0-test")
    monkeypatch.setattr(sys, "path", [str(repo / "src")] + sys.path)
    monkeypatch.delitem(sys.modules, "mlx2", raising=False)
    ns = types.SimpleNamespace(repo=repo, base=base, tmp=tmp_path)
    ns.args = lambda: q.build_parser().parse_args(
        ["--run-native", "--i-own-the-gpu", "--source-root", str(repo), "--source-commit", _head(repo),
         "--out", str(tmp_path / "receipt.json")])
    ns.env = lambda: {"MLX2_INTAKE_SOURCE_COMMIT": _head(repo)}
    ns.admit = lambda args=None, env=None: q.native_admission(args or ns.args(), env or ns.env(), root=repo)
    return ns


def test_admission_binds_head_blobs_bound_files_and_used_headers(tree):
    adm = tree.admit()
    assert adm["commit"] == _head(tree.repo) and set(adm["bound"]) == set(q.BOUND_FILES)
    assert all(adm["bound"][p] == sha for p, sha in q.FROZEN.items())
    got = {str(Path(f).relative_to(tree.base)) for f in adm["mlx_files"]}
    want = {"core.cpython-312-darwin.so", "lib/libmlx.dylib", "lib/mlx.metallib", "nn/__init__.py"}
    want |= {"include/" + h for h in q.MLX_PREAMBLE_HEADERS + q.MLX_GATE_UP_HEADERS}
    want |= {"include/" + _K + h for h in ("steel/defines.h", "steel/utils/integral_constant.h", "cexpf.h",
                                           "erf.h", "expm1f.h", "fp8.h")}
    assert got == want
    assert adm["mlx_core"].endswith("core.cpython-312-darwin.so")
    assert adm["tracked_files"] == len(q.BOUND_FILES) + 1
    q.source_guard(adm)                                   # unchanged tree passes


def _edit(path, text="# drift"):
    path.write_text(path.read_text() + chr(10) + text + chr(10))


REFUSALS = {
    "no-run-native": "no --run-native",
    "no-ack": "acknowledgement missing",
    "relative-root": "absolute path of this checkout",
    "symlink-alias-root": "absolute path of this checkout",
    "env-root": "MLX2_INTAKE_SOURCE_ROOT",
    "short-commit": "full sha",
    "env-commit": "full sha",
    "head-moved": "HEAD does not equal",
    "tracked-edit": "not clean",
    "untracked": "not clean",
    "ignored-binary": "not clean",
    "assume-unchanged": "differ from HEAD blobs",
    "frozen-drift": "frozen candidate files",
    "witness-drift": "expected served path",
    "sister-pythonpath": "another mlx2 checkout",
    "out-exists": "out exists",
    "out-missing": "out is required",
    "bound-uncommitted": "not committed at HEAD",
    "mlx-header-missing": "header",
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_admission_refusals(tree, case):
    repo, args, env = tree.repo, tree.args(), tree.env()
    if case == "no-run-native":
        args.run_native = False
    elif case == "no-ack":
        args.i_own_the_gpu = False
    elif case == "relative-root":
        args.source_root = os.path.relpath(repo)
    elif case == "symlink-alias-root":
        alias = tree.tmp / "alias"
        alias.symlink_to(repo)
        args.source_root = str(alias)
    elif case == "env-root":
        env["MLX2_INTAKE_SOURCE_ROOT"] = str(ROOT)
    elif case == "short-commit":
        args.source_commit = env["MLX2_INTAKE_SOURCE_COMMIT"] = _head(repo)[:12]
    elif case == "env-commit":
        env["MLX2_INTAKE_SOURCE_COMMIT"] = "0" * 40
    elif case == "head-moved":
        _git(repo, "commit", "-q", "--allow-empty", "-m", "moved")
    elif case == "tracked-edit":
        _edit(repo / "src/mlx2/other.py")
    elif case == "untracked":
        (repo / "tests/stray.py").write_text("")
    elif case == "ignored-binary":
        (repo / "src/mlx2/shadow.so").write_text("")
    elif case == "assume-unchanged":
        _git(repo, "update-index", "--assume-unchanged", "src/mlx2/other.py")
        _edit(repo / "src/mlx2/other.py")
    elif case in ("frozen-drift", "witness-drift"):
        path = repo / (q.CANDIDATE_FILE if case == "frozen-drift" else q.REFERENCE_FILES[2])
        text = path.read_text()
        if case == "witness-drift":
            text = text.replace("gate_up[..., half:], gate_up[..., :half]", "gate_up[..., :half], gate_up[..., half:]")
        path.write_text(text + chr(10))
        _git(repo, "commit", "-q", "-am", "drift")
        args.source_commit = env["MLX2_INTAKE_SOURCE_COMMIT"] = _head(repo)
    elif case == "sister-pythonpath":
        env["PYTHONPATH"] = str(ROOT / "src")
    elif case == "out-exists":
        (tree.tmp / "receipt.json").write_text("{}")
    elif case == "out-missing":
        args.out = None
    elif case == "bound-uncommitted":
        _git(repo, "rm", "-q", "--cached", q.HARNESS_FILES[2])
        _git(repo, "commit", "-q", "-m", "untrack")
        (repo / q.HARNESS_FILES[2]).unlink()
        args.source_commit = env["MLX2_INTAKE_SOURCE_COMMIT"] = _head(repo)
    elif case == "mlx-header-missing":
        (tree.base / "include" / _K / "fp8.h").unlink()
    with pytest.raises(q.Refused) as info:
        tree.admit(args, env)
    assert REFUSALS[case] in str(info.value), str(info.value)


@pytest.mark.parametrize("change", ["bound", "unbound-tracked", "untracked", "commit", "transitive-header",
                                    "metallib", "extra-mlx-py"])
def test_source_guard_stops_on_any_change_after_admission(tree, change):
    adm = tree.admit()
    if change == "bound":
        _edit(tree.repo / q.REFERENCE_FILES[0])
    elif change == "unbound-tracked":
        _git(tree.repo, "update-index", "--assume-unchanged", "src/mlx2/other.py")
        _edit(tree.repo / "src/mlx2/other.py")
    elif change == "untracked":
        (tree.repo / "scripts/stray.py").write_text("")
    elif change == "commit":
        _git(tree.repo, "commit", "-q", "--allow-empty", "-m", "moved")
    elif change == "transitive-header":
        _edit(tree.base / "include" / _K / "steel/utils/integral_constant.h", "// drift")
    elif change == "metallib":
        (tree.base / "lib/mlx.metallib").write_text("other build")
    elif change == "extra-mlx-py":
        (tree.base / "nn" / "extra.py").write_text("")
    with pytest.raises(q.Refused):
        q.source_guard(adm)


def test_import_identity_must_match_before_device_queries(tree):
    adm = tree.admit()
    core = adm["mlx_core"]
    assert q.import_identity_refusal(adm, core, "0.0-test") is None
    assert "not the admitted file" in q.import_identity_refusal(adm, str(tree.base / "lib/libmlx.dylib"), "0.0-test")
    assert "version" in q.import_identity_refusal(adm, core, "9.9")
    Path(core).write_text("swapped")
    assert "bytes differ" in q.import_identity_refusal(adm, core, "0.0-test")


def test_module_paths_outside_the_checkout_or_admitted_mlx_are_refused(tree):
    adm = tree.admit()
    probe = types.ModuleType("mlx2._sister_probe")
    probe.__file__ = str(ROOT.parent / "elsewhere" / "mlx2" / "x.py")
    sys.modules["mlx2._sister_probe"] = probe
    try:
        refusals = q.module_path_refusals(adm, adm["mlx_core"])
    finally:
        del sys.modules["mlx2._sister_probe"]
    assert any("mlx2._sister_probe" in r for r in refusals)
    assert any("not the admitted extension" in r for r in q.module_path_refusals(adm, str(tree.base / "x.so")))


def test_candidate_identity_controls():
    assert q.candidate_refusal(cand) is None
    base = {k: getattr(cand, k) for k in dir(cand) if not k.startswith("__")}
    base["__file__"] = cand.__file__
    for change, text in [({"__file__": str(SCRIPT)}, "not imported from this checkout"),
                         ({"STATE": dict(cand.STATE, qualified=True)}, "state flags"),
                         ({"NAMED_GEOMETRIES": {"default35b": cand.NAMED_GEOMETRIES["default35b"]}}, "geometries"),
                         ({"_MLX_OPS_HEADERS": cand._MLX_OPS_HEADERS[:1]}, "other MLX headers"),
                         ({"_is_native": lambda backend, _cls=object: True}, "native class capture"),
                         ({"SEAM_PAD_THRESHOLD": 65536}, "seam/schedule")]:
        assert text in q.candidate_refusal(types.SimpleNamespace(**dict(base, **change))), change


def test_provenance_scopes_the_harness_and_keeps_states_false():
    prov = json.loads(PROVENANCE.read_text())
    assert prov["destination_paths"] == list(q.HARNESS_FILES)
    assert prov["subject"]["sha256"] == q.FROZEN[q.CANDIDATE_FILE]
    assert prov["subject"]["committed_in"] == "03e75c4fdc9c85b27c2eeb9dd6bb20ab0e7b17e3"
    assert all(prov[k] is False for k in ("qualified", "selected", "observed_used", "model_gain", "native_run"))
    assert "Apache-2.0" in prov["license"] and "no third-party" in prov["license"]
