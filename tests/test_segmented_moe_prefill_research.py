"""Host-only tests of the segmented sorted MoE prefill research candidate.

These never import MLX (an import blocker is installed BEFORE the candidate
module is imported at collection, and around every test), never query a
device, and never build or run a Metal kernel. They do NOT establish Metal
numerical identity; native qualification is external and pending.
"""

import contextlib
import hashlib
import importlib.abc
import json
import random
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROVENANCE = json.loads((ROOT / "provenance/segmented-moe-prefill-research.json").read_text())
MODULE_PATH = ROOT / "src/mlx2/adapters/segmented_moe_prefill_research.py"


class _BlockMLX(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if _is_mlx(name):
            raise ImportError(f"host-only test: MLX import blocked ({name})")
        return None


def _is_mlx(name: str) -> bool:
    return name == "mlx" or name.startswith("mlx.")


@contextlib.contextmanager
def _mlx_blocked():
    """Block MLX imports; MLX modules some other test may have loaded are hidden
    (so a cached module cannot bypass the blocker) and restored afterwards."""
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


@pytest.fixture(autouse=True)
def no_mlx():
    with _mlx_blocked():
        yield


with _mlx_blocked():                       # collection-time: blocked before the module import
    import mlx2.adapters.segmented_moe_prefill_research as seg  # noqa: E402

D35 = dict(experts=256, top_k=8, hidden=2048, intermediate=512)
FN = dict(experts=512, top_k=10, hidden=2560, intermediate=640)


def req(tokens, **kw):
    base = dict(D35)
    base.update(kw)
    return seg.MoEPrefillRequest(tokens=tokens, **base)


class FakeArray:
    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype


class FakeBackend:
    native = False

    def __init__(self, *, capable=True, error=None):
        self.capable, self.error, self.calls = capable, error, []

    def require_capability(self):
        if not self.capable:
            raise seg.SegmentedMoEUnavailable("no NAX here")

    def upload_u32(self, values):
        return ("u32", tuple(values))

    def mapped_gate_up_swiglu(self, x, row_map, idx, w, s, b, *, plan, rows, experts):
        self.calls.append(("gate_up", row_map, idx, plan, rows, experts))
        if self.error:
            raise self.error
        return FakeArray((rows, 1, plan.out_cols), "bfloat16")

    def segmented_down(self, x, idx, w, s, b, *, plan, rows, experts):
        self.calls.append(("down", idx, plan, rows, experts))
        if self.error:
            raise self.error
        return FakeArray((rows, 1, plan.out_cols), "bfloat16")


class ExplodingBackend:
    def __getattr__(self, name):
        raise AssertionError(f"backend touched after a refusal: {name}")


def balanced_ids(tokens, top_k, experts, seed=0):
    rng = random.Random(seed)
    return [e for _ in range(tokens) for e in rng.sample(range(experts), top_k)]


def tables(request, M=None, *, gate_up=True):
    E, D, I = request.experts, request.hidden, request.intermediate
    N, K = (2 * I, D) if gate_up else (D, I)
    return (FakeArray((E, N, K * 4 // 32), "mlx.core.uint32"),
            FakeArray((E, N, K // 64), "mlx.core.bfloat16"),
            FakeArray((E, N, K // 64), "mlx.core.bfloat16"))


def small_case(tokens=16, top_k=2, experts=8, hidden=128, intermediate=64, seed=1, pad_policy="none"):
    r = seg.MoEPrefillRequest(tokens=tokens, top_k=top_k, experts=experts, hidden=hidden,
                              intermediate=intermediate)
    a = seg.admit(r)
    routes = seg.sort_routes_host(balanced_ids(tokens, top_k, experts, seed), top_k=top_k,
                                  experts=experts, pad_policy=pad_policy)
    return r, a, routes


# ------------------------------------------------------------- isolation/state

@pytest.mark.parametrize("poison", [False, True])
def test_module_import_is_host_only_in_a_fresh_interpreter(poison):
    """Fresh interpreter; with poison, sys.modules entries of None make any MLX
    import attempt during module import raise."""
    code = ("import sys\n"
            + ("for k in ('mlx', 'mlx.core', 'mlx.nn'): sys.modules[k] = None\n" if poison else "")
            + "import mlx2.adapters.segmented_moe_prefill_research as m\n"
            "bad = [k for k, v in sys.modules.items() if (k == 'mlx' or k.startswith('mlx.')) and v is not None]\n"
            "assert not bad, bad\nassert not m._KERNELS\nprint('ok')\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={"PYTHONPATH": f"{ROOT / 'src'}:{ROOT}", "PATH": "/usr/bin:/bin"})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"


def test_state_is_immutable_research_only():
    assert seg.STATE["default"] == "off"
    assert seg.STATE["qualified"] is False and seg.STATE["selected"] is False
    assert seg.STATE["observed_used"] is False and seg.STATE["route"] is None
    with pytest.raises(TypeError):
        seg.STATE["qualified"] = True
    with pytest.raises(TypeError):
        seg.SOURCE["revision"] = "x"


def test_nothing_in_the_tree_imports_the_candidate():
    hits = []
    for path in (ROOT / "src").rglob("*.py"):
        if path == MODULE_PATH:
            continue
        if "segmented_moe_prefill_research" in path.read_text(errors="ignore"):
            hits.append(str(path.relative_to(ROOT)))
    assert hits == []


def test_ordinary_reference_untouched_by_module():
    text = MODULE_PATH.read_text()
    for forbidden in ("switch_layers", "qwen3_next", "scheduler", "apc", "_gather_sort ="):
        assert f"import {forbidden}" not in text and f"from .{forbidden}" not in text


# ------------------------------------------------------------------- admission

@pytest.mark.parametrize("geometry,first", [(D35, 128), (FN, 205)])
def test_named_geometry_admission_boundary(geometry, first):
    def request(tokens):
        return seg.MoEPrefillRequest(tokens=tokens, **geometry)
    assert seg.min_admitted_tokens(geometry["top_k"], geometry["experts"]) == first
    a = seg.admit(request(first))
    assert a.rows == first * geometry["top_k"] and a.rows // geometry["experts"] >= 4
    assert a.min_tokens == first and a.route == seg.ORDINARY_REFERENCE
    assert a.named_geometry in ("default35b", "flash_next")
    with pytest.raises(seg.SegmentedMoERefused) as exc:
        seg.admit(request(first - 1))
    assert exc.value.reason == "too_small"


@pytest.mark.parametrize("tokens", [1, 2, 4, 8, 16, 64])
def test_decode_and_verify_windows_refused(tokens):
    for geometry in (D35, FN):
        with pytest.raises(seg.SegmentedMoERefused) as exc:
            seg.admit(seg.MoEPrefillRequest(tokens=tokens, **geometry))
        assert exc.value.reason == "too_small"


def test_rows_floor_of_sixteen_overrides_omlx_m8():
    small = dict(experts=2, top_k=1, hidden=64, intermediate=64)
    with pytest.raises(seg.SegmentedMoERefused) as exc:
        seg.admit(seg.MoEPrefillRequest(tokens=8, **small))      # oMLX would take M >= 8
    assert exc.value.reason == "too_small"
    with pytest.raises(seg.SegmentedMoERefused):
        seg.admit(seg.MoEPrefillRequest(tokens=15, **small))
    assert seg.admit(seg.MoEPrefillRequest(tokens=16, **small)).rows == 16


@pytest.mark.parametrize("kw,reason", [
    (dict(training=True), "training"),
    (dict(has_bias=True), "bias"),
    (dict(shared_folded=True), "shared_fold"),
    (dict(gate_up_layout="split"), "layout"),
    (dict(dtype="float16"), "dtype"),
    (dict(bits=8), "quantization"),
    (dict(group_size=32), "quantization"),
    (dict(mode="mxfp4"), "quantization"),
    (dict(hidden=2048 + 32), "shape"),
    (dict(intermediate=512 + 32), "shape"),
    (dict(experts=4096), "shape"),
    (dict(top_k=300), "shape"),
    (dict(top_k=0), "shape"),
    (dict(training=1), "type"),
])
def test_unsupported_conditions_refused(kw, reason):
    with pytest.raises(seg.SegmentedMoERefused) as exc:
        seg.admit(req(4096, **kw))
    assert exc.value.reason == reason


@pytest.mark.parametrize("tokens", [True, 128.0, "128", -128, 0])
def test_token_type_and_sign_refused(tokens):
    with pytest.raises(seg.SegmentedMoERefused) as exc:
        seg.admit(req(tokens))
    assert exc.value.reason == "shape"


def test_overflow_bounds():
    with pytest.raises(seg.SegmentedMoERefused) as exc:
        seg.admit(req(2**32 // 2048))                          # T * D == 2**32
    assert exc.value.reason == "overflow"
    assert seg.admit(req(2**32 // 2048 - 1)).rows == (2**32 // 2048 - 1) * 8
    with pytest.raises(seg.SegmentedMoERefused) as exc:
        seg.admit(seg.MoEPrefillRequest(tokens=2**25, top_k=64, experts=256, hidden=64,
                                        intermediate=64))       # rows 2**31, T * D 2**31
    assert exc.value.reason == "overflow" and "sorted rows" in str(exc.value)


def test_scan_loop_and_padded_rows_bound():
    assert seg.MAX_ROWS == 2**31 - 1 - 4096 and seg.SCAN_STRIDE == 4096
    last = (seg.MAX_ROWS - 63) // 64                         # rows + 63 == MAX_ROWS - 63 + ... <= MAX_ROWS
    geom = dict(top_k=64, experts=64, hidden=64, intermediate=64)
    a = seg.admit(seg.MoEPrefillRequest(tokens=last, **geom))
    assert a.rows + 63 <= seg.MAX_ROWS
    with pytest.raises(seg.SegmentedMoERefused) as exc:      # one more token: padded rows oversize
        seg.admit(seg.MoEPrefillRequest(tokens=last + 1, **geom))
    assert exc.value.reason == "overflow" and "scan loop" in str(exc.value)
    assert (last + 1) * 64 + 63 <= 2**31 - 1                 # still inside plain int32: only the scan bound refuses
    window = seg.MoEPrefillRequest(tokens=67108735, top_k=32, experts=32, hidden=64, intermediate=64)
    rows = window.tokens * window.top_k                      # rows fit the scan bound, rows + seam pad do not
    assert rows <= seg.MAX_ROWS < rows + 63 and window.tokens * window.hidden < 2**32
    with pytest.raises(seg.SegmentedMoERefused) as exc:
        seg.admit(window)
    assert exc.value.reason == "overflow" and "seam pad" in str(exc.value)
    seg.projection_plan(seg.MAX_ROWS, 64, 64, 64, paired=False)
    with pytest.raises(seg.SegmentedMoERefused) as exc:      # entry-time rows (seam padding included)
        seg.projection_plan(seg.MAX_ROWS + 1, 64, 64, 64, paired=False)
    assert exc.value.reason == "overflow" and "scan loop" in str(exc.value)


def test_admission_requires_request_type():
    with pytest.raises(seg.SegmentedMoERefused) as exc:
        seg.admit(dict(D35, tokens=4096))
    assert exc.value.reason == "type"


def test_refusal_reasons_are_closed():
    with pytest.raises(ValueError):
        seg.SegmentedMoERefused("made_up", "x")


# --------------------------------------------------------------------- planner

@pytest.mark.parametrize("rows,experts,K,N,want", [
    (81920, 512, 2560, 640, seg.Plan(0, 128, 128, 32, 8192)),   # upstream test cases
    (81920, 512, 640, 2560, seg.Plan(0, 96, 128, 32, 0)),
    (33010, 512, 2560, 640, seg.Plan(1, 96, 64, 32, 0)),
    (129, 8, 96, 64, seg.Plan(0, 64, 64, 0, 0)),
    (35 * 8, 8, 2048, 1024, seg.Plan(1, 64, 64, 0, 0)),         # per expert < 36
    (36 * 8, 8, 2048, 1024, seg.Plan(1, 64, 64, 32, 0)),
    (47 * 8, 8, 2048, 1024, seg.Plan(1, 64, 64, 32, 0)),
    (48 * 8, 8, 2048, 1024, seg.Plan(1, 96, 64, 32, 0)),
    (95 * 8, 8, 2048, 1024, seg.Plan(1, 96, 64, 32, 0)),
    (96 * 8, 8, 2048, 1024, seg.Plan(0, 128, 128, 32, 8192)),
    (119 * 8, 8, 512, 2048, seg.Plan(1, 64, 64, 0, 0)),         # short K below 120
    (120 * 8, 8, 512, 2048, seg.Plan(0, 96, 128, 32, 0)),
    (4096, 8, 2048, 96, seg.Plan(0, 64, 64, 0, 0)),             # ragged N
    (4096, 8, 2080, 1024, seg.Plan(0, 64, 64, 0, 0)),           # ragged K
])
def test_planner_port_boundaries(rows, experts, K, N, want):
    assert seg.plan_kernel(rows, experts, K, N) == want


def test_planner_has_no_environment_override(monkeypatch):
    monkeypatch.setenv("OMLX_M5_GATHER_QMM_NAX_PLAN", "seg,128,128,32,8192")
    assert seg.plan_kernel(129, 8, 2048, 1024) == seg.Plan(1, 64, 64, 0, 0)
    text = MODULE_PATH.read_text()
    assert "os.environ" not in text and "getenv" not in text and "import os" not in text


def test_effective_plan_fallbacks():
    db = seg.Plan(seg.SCHED_DB, 64, 64, 0, 0)
    assert seg.effective_plan(db, 2048, 1024, paired=True) == db
    assert seg.effective_plan(db, 2080, 1024, paired=True).sched == seg.SCHED_SEG
    assert seg.effective_plan(db, 2048, 96, paired=False).sched == seg.SCHED_SEG
    assert seg.effective_plan(db._replace(bk=128), 2048, 1024, paired=True).sched == seg.SCHED_SEG
    seg_plan = seg.Plan(seg.SCHED_SEG, 128, 128, 32, 8192)
    assert seg.effective_plan(seg_plan, 2080, 96, paired=False) == seg_plan


@pytest.mark.parametrize("plan", [
    seg.Plan(2, 64, 64, 0, 0), seg.Plan(0, 32, 64, 0, 0), seg.Plan(0, 64, 96, 0, 0),
    seg.Plan(0, 64, 64, -1, 0), seg.Plan(0, 64, 64, 0, 8193), seg.Plan(1, 128, 128, 0, 8192),
    (0, 64, 64, 0, 0), seg.Plan(0, 64.0, 64, 0, 0), seg.Plan(0, 64, 64, 64, 0),
    seg.Plan(0, 64, 64, 10**12, 0),
])
def test_pinned_plan_validation(plan):
    with pytest.raises(seg.SegmentedMoERefused) as exc:
        seg.validate_plan(plan)
    assert exc.value.reason == "plan"


def test_all_planner_outputs_fit_threadgroup_memory():
    for rows in (1024, 2048, 4096, 8192, 32768, 81920):
        for E in (8, 256, 512):
            for K, N in ((2048, 1024), (512, 2048), (2560, 1280), (640, 2560)):
                for paired in (True, False):
                    p = seg.projection_plan(rows, E, K, N, paired=paired).plan
                    assert seg.threadgroup_memory_bytes(p) <= seg.MAX_THREADGROUP_MEMORY


def test_named_geometry_plans_are_recorded_not_measured():
    a = seg.admit(seg.MoEPrefillRequest(tokens=8192, **FN))
    assert a.gate_up.plan == seg.Plan(0, 128, 128, 32, 8192) and a.gate_up.out_cols == 640
    assert a.down.plan == seg.Plan(0, 96, 128, 32, 0) and a.down.out_cols == 2560
    assert a.gate_up.align_k and a.down.align_k and a.down.align_n
    b = seg.admit(req(128))
    assert b.gate_up.plan == seg.Plan(1, 64, 64, 0, 0) and b.down.plan == seg.Plan(1, 64, 64, 0, 0)
    assert "NOT measured" in a.plan_provenance


def test_dispatch_geometry_follows_upstream_launch():
    a = seg.admit(seg.MoEPrefillRequest(tokens=8192, **FN))
    gu, dn = a.gate_up, a.down
    assert gu.max_tiles == -(-81920 // 128) + 512
    assert gu.grid == (32 * 32, -(-gu.max_tiles // 32) * (1280 // 64) * 2, 4)
    assert gu.threadgroup == (32, 2, 4)
    assert dn.grid == (32 * 32, -(-dn.max_tiles // 32) * (2560 // 64) * 2, 3)
    b = seg.admit(req(128))
    assert b.gate_up.grid == (1024 // 64 * 32, b.gate_up.max_tiles * 2, 2)


def test_int32_uniforms_and_grid_bounded():
    with pytest.raises(seg.SegmentedMoERefused) as exc:      # N = 2I = 2**31
        seg.admit(seg.MoEPrefillRequest(tokens=16, top_k=1, experts=4, hidden=64, intermediate=2**30))
    assert exc.value.reason == "overflow"
    with pytest.raises(seg.SegmentedMoERefused) as exc:      # K = N = 2**31 - 64 direct
        seg.projection_plan(64, 4, 64, 2**31 - 64, paired=False)
    assert exc.value.reason == "overflow"
    with pytest.raises(seg.SegmentedMoERefused) as exc:      # grid y past int32
        seg.projection_plan(2**30, 1, 64, 2**24, paired=False)
    assert exc.value.reason == "overflow"


def small_geom(D, I, tokens=16, top_k=4, experts=4):
    return seg.MoEPrefillRequest(tokens=tokens, top_k=top_k, experts=experts, hidden=D, intermediate=I)


def test_dimension_floor_int32_max_over_32():
    assert seg.MAX_DIM == (2**31 - 1) // 32
    top = seg.MAX_DIM // 64 * 64                             # largest aligned accepted dim
    a = seg.admit(small_geom(top, 64))
    assert a.gate_up.K == top and a.down.N == top
    b = seg.admit(small_geom(64, top, tokens=4))             # db down: 31 * K + 28 fits
    assert b.down.plan.sched == seg.SCHED_DB and 31 * top + 28 <= 2**31 - 1
    for D, I, tokens in ((top + 64, 64, 16), (64, top + 64, 4)):
        with pytest.raises(seg.SegmentedMoERefused) as exc:
            seg.admit(small_geom(D, I, tokens=tokens))
        assert exc.value.reason == "overflow" and "INT32_MAX // 32" in str(exc.value)


@pytest.mark.parametrize("D,I,tokens,product", [
    (2**29, 64, 4, "k_times_bits"),                          # root 2942: mapped K_w intermediate K * 4 = 2**31
    (64, 2**28, 16, "db_row_offset_x"),                      # root 2942: plain down db 15 * K = 4,026,531,840
])
def test_root_examples_refused_before_backend(D, I, tokens, product):
    with pytest.raises(seg.SegmentedMoERefused) as exc:
        seg.admit(small_geom(D, I, tokens=tokens))
    assert exc.value.reason == "overflow"
    K, N, paired = (D, 2 * I, True) if product == "k_times_bits" else (I, D, False)
    plan = seg.Plan(seg.SCHED_DB, 64, 64, 0, 0)
    assert seg.signed_int_intermediates(K, N, plan, paired=paired)[product] > 2**31 - 1


def test_seg_row_offsets_need_per_plan_bounds_below_the_floor():
    """Below the dim floor, seg tiles still form tm * K with tm up to bm - 32 = 96."""
    I = 2**25                                                 # <= MAX_DIM
    assert I <= seg.MAX_DIM
    with pytest.raises(seg.SegmentedMoERefused) as exc:      # 96 rows/expert -> seg 128 down
        seg.admit(small_geom(64, I, tokens=96))
    assert exc.value.reason == "overflow" and "seg_row_offset_x" in str(exc.value)
    ok = seg.projection_plan(384, 4, I, 64, paired=False, pinned=seg.Plan(seg.SCHED_SEG, 64, 64, 0, 0))
    assert seg.signed_int_intermediates(I, 64, ok.plan, paired=False)["seg_row_offset_x"] == 32 * I
    with pytest.raises(seg.SegmentedMoERefused) as exc:      # pinned plans are checked too
        seg.projection_plan(384, 4, I, 64, paired=False, pinned=seg.Plan(seg.SCHED_SEG, 96, 128, 32, 0))
    assert exc.value.reason == "overflow"
    with pytest.raises(seg.SegmentedMoERefused) as exc:      # paired seg output tm * (N / 2)
        seg.projection_plan(384, 4, 64, 2 * I, paired=True, pinned=seg.Plan(seg.SCHED_SEG, 128, 128, 32, 8192))
    assert "seg_row_offset_out" in str(exc.value)


def test_named_geometry_intermediates_are_small():
    for geometry in (D35, FN):
        for tokens in (seg.min_admitted_tokens(geometry["top_k"], geometry["experts"]), 8192):
            a = seg.admit(seg.MoEPrefillRequest(tokens=tokens, **geometry))
            for pp in (a.gate_up, a.down):
                values = seg.signed_int_intermediates(pp.K, pp.N, pp.plan, paired=pp.paired)
                assert max(values.values()) < 2**24


# -------------------------------------------------------------- route decision

def test_route_decision_never_promotes():
    for evidence in (None, {"qualified": True, "smoke": "pass"}, "host report: 100% pass"):
        d = seg.decide_route(req(4096), evidence=evidence)
        assert d.route == seg.ORDINARY_REFERENCE
        assert d.research_candidate_admissible is True
        assert d.qualified is False and d.selected is False
    d = seg.decide_route(req(16))
    assert d.route == seg.ORDINARY_REFERENCE and d.research_candidate_admissible is False
    assert "too_small" in d.reason
    assert seg.STATE["qualified"] is False and seg.STATE["selected"] is False


# --------------------------------------------------------------- host routes

def test_sort_routes_mirror_gather_sort():
    ids = [3, 1, 1, 0, 3, 2, 0, 1]                 # T=4, k=2, expert 4..5 empty
    r = seg.sort_routes_host(ids, top_k=2, experts=6)
    assert r.sorted_experts == (0, 0, 1, 1, 1, 2, 3, 3)
    assert r.row_map == tuple(a // 2 for a in r.order)
    assert [ids[a] for a in r.order] == list(r.sorted_experts)
    assert r.runs == ((0, 0, 2), (1, 2, 5), (2, 5, 6), (3, 6, 8))
    assert r.pad == 0 and r.rows == r.assignments == 8
    assert seg.restore_token_order_host(list(r.sorted_experts), r) == ids


def test_repeated_token_maps_and_empty_experts():
    ids = balanced_ids(64, 4, 32, seed=7)
    r = seg.sort_routes_host(ids, top_k=4, experts=32)
    assert sorted(r.row_map) == sorted(t for t in range(64) for _ in range(4))
    present = {e for e, _, _ in r.runs}
    ids2 = [e % 5 for e in ids]                    # experts 5..31 empty
    r2 = seg.sort_routes_host(ids2, top_k=4, experts=32)
    assert {e for e, _, _ in r2.runs} == set(range(5)) and present


@pytest.mark.parametrize("bm", seg.TILE_ROWS)
def test_tile_mirror_covers_ragged_runs_once(bm):
    counts = [70, 0, 5, 33, 64, 17, 140, 11, 0, 129]   # upstream canary profile + ragged tail
    se = [e for e, n in enumerate(counts) for _ in range(n)]
    tiles = seg.tile_table_host(se, experts=len(counts), bm=bm)
    covered = [0] * len(se)
    for start, e, rows in tiles:
        assert 0 < rows <= bm
        for i in range(start, start + rows):
            assert se[i] == e
            covered[i] += 1
    assert covered == [1] * len(se)
    assert len(tiles) <= seg.max_tiles(len(se), len(counts), bm)
    for e, n in enumerate(counts):                     # partial tiles only at a run's end
        rows = [t[2] for t in tiles if t[1] == e]
        assert sum(rows) == n and all(x == bm for x in rows[:-1])


def test_split_run_leaves_rows_uncovered_and_is_refused():
    se = [0, 0, 1, 1, 0, 0]
    tiles = seg.tile_table_host(se, experts=2, bm=64)
    covered = {i for s, _, n in tiles for i in range(s, s + n)}
    assert covered != set(range(len(se)))
    with pytest.raises(seg.SegmentedMoERefused) as exc:
        seg.validate_sorted_routes(se, [0, 1, 0, 1, 2, 2], tokens=3, top_k=2, experts=2)
    assert exc.value.reason == "indices"


def test_descending_single_runs_are_valid():
    se = [2, 2, 0, 0, 1, 1]
    r = seg.validate_sorted_routes(se, [0, 1, 0, 2, 1, 2], tokens=3, top_k=2, experts=3)
    tiles = seg.tile_table_host(r.sorted_experts, experts=3, bm=64)
    assert [t[1] for t in tiles] == [0, 1, 2]           # expert-major, like the pre-pass


def test_seam_padding_and_inverse_restoration():
    assert seg.seam_pad(32768) == 0 and seg.seam_pad(32769) == 63
    assert seg.seam_pad(32832) == 0 and seg.seam_pad(100) == 0
    tokens, k, E = 4097, 8, 256
    ids = balanced_ids(tokens, k, E, seed=3)
    r = seg.sort_routes_host(ids, top_k=k, experts=E, pad_policy="seam")
    assert r.assignments == 32776 and r.pad == 56 and r.rows == 32832
    assert set(r.sorted_experts[-57:]) == {r.sorted_experts[-57]}
    assert set(r.row_map[-57:]) == {r.row_map[-57]}
    restored = seg.restore_token_order_host(list(r.sorted_experts), r)
    assert restored == ids                                    # padding dropped, order restored
    plain = seg.sort_routes_host(ids, top_k=k, experts=E)
    assert plain.pad == 0 and plain.rows == 32776


@pytest.mark.parametrize("ids,reason", [
    ([0, 1, 2, 8], "indices"), ([0, -1, 2, 3], "indices"), ([0, True, 2, 3], "indices"),
    ([0, 1.0, 2, 3], "indices"), ([0, 1, 2], "indices"), ([], "indices"), ("0123", "indices"),
])
def test_invalid_expert_ids(ids, reason):
    with pytest.raises(seg.SegmentedMoERefused) as exc:
        seg.sort_routes_host(ids, top_k=2, experts=8)
    assert exc.value.reason == reason


def test_invalid_maps_and_orders():
    good = seg.sort_routes_host([0, 1, 1, 2], top_k=2, experts=3)
    se, rm = good.sorted_experts, good.row_map
    cases = [
        (dict(row_map=(0, 0, 1, 2)), "row_map"),                    # out of range
        (dict(row_map=(0, 1, 0, 1)), "row_map"),                    # counts fine, not order // k
        (dict(row_map=(0, 0, 1)), "routes"),                        # length
        (dict(inv_order=(1, 0, 2, 3)), "routes"),                   # not the inverse
        (dict(order=(0, 0, 2, 3)), "routes"),                       # not a permutation
        (dict(inv_order=None), "routes"),                           # half an order
    ]
    for override, reason in cases:
        kw = dict(sorted_experts=se, row_map=rm, tokens=2, top_k=2, experts=3,
                  order=good.order, inv_order=good.inv_order)
        kw.update(override)
        with pytest.raises(seg.SegmentedMoERefused) as exc:
            seg.validate_sorted_routes(**kw)
        assert exc.value.reason == reason, override
    with pytest.raises(seg.SegmentedMoERefused) as exc:              # out of range, no order
        seg.validate_sorted_routes(se, (0, 0, 1, 2), tokens=2, top_k=2, experts=3)
    assert exc.value.reason == "row_map"
    with pytest.raises(seg.SegmentedMoERefused) as exc:              # token mapped 3x / 1x
        seg.validate_sorted_routes(se, (0, 0, 0, 1), tokens=2, top_k=2, experts=3)
    assert exc.value.reason == "row_map"
    with pytest.raises(seg.SegmentedMoERefused) as exc:              # bad padding
        seg.validate_sorted_routes(se + (se[-1],), rm + (0,), tokens=2, top_k=2, experts=3, pad=1)
    assert exc.value.reason == "routes"


def test_restore_requires_sealed_routes():
    with pytest.raises(seg.SegmentedMoERefused) as exc:
        seg.restore_token_order_host([0], object())
    assert exc.value.reason == "unsealed"


# ------------------------------------------------------ research entry points

def gate_up_args(r, routes):
    return (FakeArray((r.tokens, 1, r.hidden), "mlx.core.bfloat16"), routes) + tables(r)


def down_args(r, routes):
    return (FakeArray((routes.rows, 1, r.intermediate), "mlx.core.bfloat16"), routes) + tables(r, gate_up=False)


@pytest.mark.parametrize("opt_in", [False, None, 1, "yes", "True"])
def test_missing_opt_in_is_ordinary_reference_only(opt_in):
    r, a, routes = small_case()
    before = seg.ENGAGEMENT.snapshot()
    for fn, args in ((seg.research_mapped_gate_up_swiglu, gate_up_args(r, routes)),
                     (seg.research_segmented_down, down_args(r, routes))):
        with pytest.raises(seg.SegmentedMoERefused) as exc:
            fn(*args, admission=a, research_opt_in=opt_in, backend=ExplodingBackend())
        assert exc.value.reason == "opt_in_missing" and "ordinary reference only" in str(exc.value)
    fresh = seg.ENGAGEMENT.fresh_since(before)
    assert fresh["refused.opt_in_missing"] == 2 and fresh["calls"] == 2
    assert not any(v for k, v in fresh.items() if k.startswith(("native.", "substituted.")))
    assert not seg._KERNELS


def test_every_refusal_precedes_the_default_backend():
    """backend=None: a refusal must win before the lazy MLX import is attempted."""
    r, a, routes = small_case()
    other = seg.sort_routes_host(balanced_ids(32, 2, 8), top_k=2, experts=8)
    x, rt, w, s, b = gate_up_args(r, routes)
    cases = [
        (dict(research_opt_in=False), (x, rt, w, s, b), "opt_in_missing"),
        (dict(research_opt_in=True), gate_up_args(r, other), "routes"),
        (dict(research_opt_in=True), (x, rt, w, s, None), "quantization"),
        (dict(research_opt_in=True, plan=seg.Plan(0, 48, 64, 0, 0)), (x, rt, w, s, b), "plan"),
    ]
    for kw, args, reason in cases:
        with pytest.raises(seg.SegmentedMoERefused) as exc:
            seg.research_mapped_gate_up_swiglu(*args, admission=a, **kw)
        assert exc.value.reason == reason
    with pytest.raises(seg.SegmentedMoERefused) as exc:
        seg.research_segmented_down(*down_args(r, routes), admission=a)
    assert exc.value.reason == "opt_in_missing"
    assert not seg._KERNELS


def test_unsealed_admission_and_routes_refused():
    r, a, routes = small_case()
    forged = seg.SegmentedMoEAdmission(request=r, rows=a.rows, rows_per_expert=a.rows_per_expert,
                                       min_tokens=a.min_tokens, gate_up=a.gate_up, down=a.down,
                                       named_geometry=None)
    with pytest.raises(seg.SegmentedMoERefused) as exc:
        seg.research_mapped_gate_up_swiglu(*gate_up_args(r, routes), admission=forged,
                                           research_opt_in=True, backend=ExplodingBackend())
    assert exc.value.reason == "unsealed"
    fake_routes = seg.SortedRoutes(tokens=routes.tokens, top_k=routes.top_k, experts=routes.experts,
                                   sorted_experts=routes.sorted_experts, row_map=routes.row_map,
                                   order=routes.order, inv_order=routes.inv_order, pad=0, runs=routes.runs)
    with pytest.raises(seg.SegmentedMoERefused) as exc:
        seg.research_mapped_gate_up_swiglu(*gate_up_args(r, fake_routes), admission=a,
                                           research_opt_in=True, backend=ExplodingBackend())
    assert exc.value.reason == "unsealed"


def test_replaced_copies_keep_the_seal_but_not_the_gate():
    r, a, routes = small_case()
    x, rt, w, s, b = gate_up_args(r, routes)
    forged = [
        ((replace(a, request=replace(r, training=True)), routes), "training"),
        ((replace(a, request=replace(r, has_bias=True)), routes), "bias"),
        ((replace(a, rows=a.rows + 1), routes), "unsealed"),
        ((replace(a, gate_up=replace(a.gate_up, plan=seg.Plan(0, 128, 128, 32, 8192))), routes), "unsealed"),
        ((replace(a, named_geometry="flash_next"), routes), "unsealed"),
        ((a, replace(routes, row_map=(routes.tokens,) * routes.rows)), "row_map"),
        ((a, replace(routes, sorted_experts=(0,) * routes.rows)), "indices"),
        ((a, replace(routes, sorted_experts=(0,) * routes.rows, runs=((0, 0, routes.rows),))), "indices"),
        ((a, replace(routes, expert_ids=None)), "routes"),
        ((a, replace(routes, order=None, inv_order=None)), "routes"),
        ((a, replace(routes, runs=())), "unsealed"),
        ((a, replace(routes, pad=1)), "routes"),
    ]
    for (adm, rts), reason in forged:
        assert adm._seal is a._seal and rts._seal is routes._seal
        with pytest.raises(seg.SegmentedMoERefused) as exc:
            seg.research_mapped_gate_up_swiglu(x, rts, w, s, b, admission=adm, research_opt_in=True,
                                               backend=ExplodingBackend())
        assert exc.value.reason == reason, (adm, reason)
    assert not seg._KERNELS


def test_routes_must_match_admission():
    r, a, _ = small_case(tokens=16)
    other = seg.sort_routes_host(balanced_ids(32, 2, 8), top_k=2, experts=8)
    with pytest.raises(seg.SegmentedMoERefused) as exc:
        seg.research_mapped_gate_up_swiglu(*gate_up_args(r, other), admission=a,
                                           research_opt_in=True, backend=ExplodingBackend())
    assert exc.value.reason == "routes"


def test_array_metadata_refusals_before_backend():
    r, a, routes = small_case()
    x, rt, w, s, b = gate_up_args(r, routes)
    bad = [
        ((FakeArray((r.tokens, r.hidden), "bfloat16"), rt, w, s, b), "shape"),
        ((FakeArray((r.tokens, 1, r.hidden), "float16"), rt, w, s, b), "dtype"),
        ((x, rt, FakeArray((r.experts + 1,) + w.shape[1:], "uint32"), s, b), "shape"),   # shared fold
        ((x, rt, FakeArray((r.experts, r.intermediate) + w.shape[2:], "uint32"), s, b), "shape"),  # split table
        ((x, rt, FakeArray(w.shape, "uint8"), s, b), "quantization"),
        ((x, rt, w, FakeArray(s.shape[:2] + (r.hidden // 32,), "bfloat16"), b), "shape"),  # group 32
        ((x, rt, w, s, None), "quantization"),
        ((x, rt, w, s, FakeArray(b.shape, "float16")), "dtype"),
        ((object(), rt, w, s, b), "type"),
    ]
    for args, reason in bad:
        with pytest.raises(seg.SegmentedMoERefused) as exc:
            seg.research_mapped_gate_up_swiglu(*args, admission=a, research_opt_in=True,
                                               backend=ExplodingBackend())
        assert exc.value.reason == reason, args
    h, rt, w, s, b = down_args(r, routes)
    with pytest.raises(seg.SegmentedMoERefused) as exc:
        seg.research_segmented_down(FakeArray((routes.rows - 1, 1, r.intermediate), "bfloat16"),
                                    rt, w, s, b, admission=a, research_opt_in=True,
                                    backend=ExplodingBackend())
    assert exc.value.reason == "shape"


def test_invalid_pinned_plan_refused_before_backend():
    r, a, routes = small_case()
    with pytest.raises(seg.SegmentedMoERefused) as exc:
        seg.research_segmented_down(*down_args(r, routes), admission=a, research_opt_in=True,
                                    plan=seg.Plan(0, 48, 64, 0, 0), backend=ExplodingBackend())
    assert exc.value.reason == "plan"


def test_default_backend_imports_mlx_only_after_every_refusal():
    r, a, routes = small_case()
    before = seg.ENGAGEMENT.snapshot()
    with pytest.raises(ImportError, match="MLX import blocked"):
        seg.research_mapped_gate_up_swiglu(*gate_up_args(r, routes), admission=a, research_opt_in=True)
    fresh = seg.ENGAGEMENT.fresh_since(before)
    assert fresh["calls"] == 1 and fresh["backend_raised"] == 1
    assert not any(v for k, v in fresh.items() if k not in ("calls", "backend_raised"))
    assert not seg._KERNELS


class _LookalikeNative(FakeBackend):
    pass


def test_rebinding_the_native_class_cannot_promote_a_fake(monkeypatch):
    monkeypatch.setattr(seg, "_MLXBackend", _LookalikeNative)
    r, a, routes = small_case()
    before = seg.ENGAGEMENT.snapshot()
    with pytest.raises(ImportError, match="MLX import blocked"):   # the class captured at import is used
        seg.research_segmented_down(*down_args(r, routes), admission=a, research_opt_in=True)
    fresh = seg.ENGAGEMENT.fresh_since(before)
    assert fresh["backend_raised"] == 1
    assert not any(v for k, v in fresh.items() if ".successful_chains." in k)


class _NativeSubclassFake(FakeBackend, seg._MLXBackend):
    pass


@pytest.mark.parametrize("fake", [_LookalikeNative, _NativeSubclassFake])
def test_substituted_factory_never_counts_native(monkeypatch, fake):
    monkeypatch.setattr(seg, "_native_backend", lambda: fake())
    r, a, routes = small_case()
    before = seg.ENGAGEMENT.snapshot()
    out = seg.research_segmented_down(*down_args(r, routes), admission=a, research_opt_in=True)
    fresh = seg.ENGAGEMENT.fresh_since(before)
    assert out.native is False
    assert fresh["substituted.successful_chains.down_segmented"] == 1
    assert all(fresh[f"native.successful_chains.{k}"] == 0 for k in seg.DISPATCH_KINDS)


class _PartialBackend(FakeBackend):
    """Issues the tile scan, then the matmul raises."""

    def mapped_gate_up_swiglu(self, *args, **kw):
        self.calls.append(("scan_issued",))
        raise self.error


def test_partial_chain_failure_counts_no_successful_chain():
    r, a, routes = small_case()
    boom = RuntimeError("matmul pipeline failed after the scan")
    be = _PartialBackend(error=boom)
    before = seg.ENGAGEMENT.snapshot()
    with pytest.raises(RuntimeError) as exc:
        seg.research_mapped_gate_up_swiglu(*gate_up_args(r, routes), admission=a,
                                           research_opt_in=True, backend=be)
    assert exc.value is boom and be.calls == [("scan_issued",)]
    fresh = seg.ENGAGEMENT.fresh_since(before)
    assert fresh["backend_raised"] == 1
    assert not any(v for k, v in fresh.items() if ".successful_chains." in k)


def test_missing_capability_propagates_without_dispatch():
    r, a, routes = small_case()
    be = FakeBackend(capable=False)
    before = seg.ENGAGEMENT.snapshot()
    with pytest.raises(seg.SegmentedMoEUnavailable):
        seg.research_mapped_gate_up_swiglu(*gate_up_args(r, routes), admission=a,
                                           research_opt_in=True, backend=be)
    assert be.calls == []
    fresh = seg.ENGAGEMENT.fresh_since(before)
    assert not any(v for k, v in fresh.items() if k.startswith(("native.", "substituted.")))


def test_backend_errors_propagate_unchanged():
    r, a, routes = small_case()
    boom = RuntimeError("metal pipeline failed")
    be = FakeBackend(error=boom)
    before = seg.ENGAGEMENT.snapshot()
    with pytest.raises(RuntimeError) as exc:
        seg.research_segmented_down(*down_args(r, routes), admission=a, research_opt_in=True, backend=be)
    assert exc.value is boom
    fresh = seg.ENGAGEMENT.fresh_since(before)
    assert not any(v for k, v in fresh.items() if k.startswith(("native.", "substituted.")))


def test_substituted_dispatch_never_counts_as_native():
    r, a, routes = small_case(tokens=64, top_k=2, experts=8)
    be = FakeBackend()
    before = seg.ENGAGEMENT.snapshot()
    gu = seg.research_mapped_gate_up_swiglu(*gate_up_args(r, routes), admission=a,
                                            research_opt_in=True, backend=be)
    dn = seg.research_segmented_down(*down_args(r, routes), admission=a, research_opt_in=True, backend=be)
    assert gu.native is False and dn.native is False
    assert gu.route == dn.route == "research-explicit-unserved"
    assert gu.state["qualified"] is False and gu.state["selected"] is False
    assert gu.output.shape == (routes.rows, 1, r.intermediate)
    assert dn.output.shape == (routes.rows, 1, r.hidden)
    kind, row_map, idx, plan, rows, experts = be.calls[0]
    assert row_map == ("u32", routes.row_map) and idx == ("u32", routes.sorted_experts)
    assert plan == seg.projection_plan(routes.rows, 8, r.hidden, 2 * r.intermediate, paired=True)
    assert (rows, experts) == (routes.rows, 8)
    fresh = seg.ENGAGEMENT.fresh_since(before)
    assert fresh["substituted.successful_chains.gate_up_mapped_swiglu"] == 1
    assert fresh["substituted.successful_chains.down_segmented"] == 1
    assert all(fresh[f"native.successful_chains.{k}"] == 0 for k in seg.DISPATCH_KINDS)
    assert fresh["backend_raised"] == 0


def test_provided_mlx_backend_subclass_never_counts_native():
    class Sub(FakeBackend, seg._MLXBackend):
        pass

    r, a, routes = small_case()
    be = Sub()
    assert isinstance(be, seg._MLXBackend)
    before = seg.ENGAGEMENT.snapshot()
    out = seg.research_segmented_down(*down_args(r, routes), admission=a, research_opt_in=True, backend=be)
    fresh = seg.ENGAGEMENT.fresh_since(before)
    assert out.native is False
    assert fresh["substituted.successful_chains.down_segmented"] == 1
    assert all(fresh[f"native.successful_chains.{k}"] == 0 for k in seg.DISPATCH_KINDS)


def test_padded_seam_routes_are_planned_on_actual_rows():
    tokens, k, E = 4097, 8, 256
    r = seg.MoEPrefillRequest(tokens=tokens, top_k=k, experts=E, hidden=128, intermediate=64)
    a = seg.admit(r)
    routes = seg.sort_routes_host(balanced_ids(tokens, k, E, seed=5), top_k=k, experts=E, pad_policy="seam")
    be = FakeBackend()
    out = seg.research_segmented_down(*down_args(r, routes), admission=a, research_opt_in=True, backend=be)
    assert out.rows == routes.rows == 32832 and be.calls[0][3] == 32832
    assert out.plan.max_tiles == seg.max_tiles(32832, E, out.plan.plan.bm)


def test_pinned_plan_is_explicit_and_recorded():
    r, a, routes = small_case(tokens=64)
    be = FakeBackend()
    pinned = seg.Plan(seg.SCHED_SEG, 96, 128, 32, 0)
    out = seg.research_mapped_gate_up_swiglu(*gate_up_args(r, routes), admission=a, research_opt_in=True,
                                             plan=pinned, backend=be)
    assert out.plan.planned == pinned == out.plan.plan == be.calls[0][3].plan
    assert out.plan != seg.projection_plan(routes.rows, 8, r.hidden, 2 * r.intermediate, paired=True)


# ----------------------------------------------------------------- counters

def test_engagement_counters_are_bounded_and_fixed_keyed():
    stats = seg.EngagementStats()
    keys = set(stats.snapshot())
    key = "native.successful_chains.down_segmented"
    assert keys == ({"calls", "backend_raised", "epoch"} | {f"refused.{r}" for r in seg.REFUSAL_REASONS}
                    | {f"{p}.successful_chains.{k}" for p in ("native", "substituted") for k in seg.DISPATCH_KINDS})
    stats._counts[key] = seg.COUNTER_CAP - 1
    stats._epoch = seg.COUNTER_CAP
    for _ in range(3):
        stats._bump(key)
    snap = stats.snapshot()
    assert snap[key] == seg.COUNTER_CAP and snap["epoch"] == seg.COUNTER_CAP
    assert set(snap) == keys
    with pytest.raises(KeyError):
        stats._bump("native.scan")                           # retired key: scans are not counted alone
    with pytest.raises(TypeError):
        snap[key] = 0


# ---------------------------------------------------------- source binding

def _literal_sha(name):
    return hashlib.sha256(getattr(seg, name).encode()).hexdigest()


def test_kernel_text_bound_to_provenance():
    retained = PROVENANCE["retained_kernel_text"]
    assert set(retained) == {"_SCAN_HEADER", "_SCAN_SOURCE", "_MM_HEADER", "_AFFINE_SOURCE",
                             "_ACT_HEADER", "_AFFINE_ACT_MAP_SOURCE"}
    for name, entry in retained.items():
        assert _literal_sha(name) == entry["retained_sha256"], name
        assert (entry["retained_sha256"] == entry["upstream_sha256"]) == entry["verbatim"]
    src = PROVENANCE["source"]
    for key in ("repository", "path", "revision", "blob", "sha256"):
        assert seg.SOURCE[key] == src[key]
    assert PROVENANCE["state"] == dict(seg.STATE)


def test_retained_mechanisms_and_removals():
    assert "pair_row" in seg._MM_HEADER and "map_rows" in seg._MM_HEADER
    assert "rmap[row_start + r] * uint(K)" in seg._MM_HEADER          # 32-bit row-map offsets
    assert "scale * q + bias" in seg._MM_HEADER                        # affine dequant comment
    assert "static_cast<WT>(p.s * float(q) + p.b)" in seg._MM_HEADER
    assert "Multiply()(Multiply()(g, Sigmoid()(g)), u)" in seg._ACT_HEADER
    assert "static_assert(EPI == 1" in seg._ACT_HEADER
    assert "Minimum()" not in seg._ACT_HEADER
    for text in (seg._MM_HEADER, seg._ACT_HEADER, seg._AFFINE_SOURCE, seg._AFFINE_ACT_MAP_SOURCE):
        assert "Mxfp4Q" not in text and "fp4_e2m1" not in text
    assert "gather_seg<T, Q, G, true, ALIGN_K, EPI, true>" in seg._AFFINE_ACT_MAP_SOURCE
    assert "x, rmap," in seg._AFFINE_ACT_MAP_SOURCE
    assert "gather_seg<T, Q, G, ALIGN_N, ALIGN_K>" in seg._AFFINE_SOURCE


def test_no_apple_copyright_header_in_project_files():
    for path in (MODULE_PATH, Path(__file__)):
        head = path.read_text().splitlines()[:5]
        assert not any("Apple" in line for line in head)
    assert MODULE_PATH.read_text().splitlines()[0] == "# SPDX-License-Identifier: Apache-2.0"


# ------------------------------------------------------------ header reader

def test_read_mlx_headers_flattens_installed_layout(tmp_path):
    k = tmp_path / "mlx/backend/metal/kernels"
    (k / "steel/gemm").mkdir(parents=True)
    (k / "utils.h").write_text("UTILS\n")
    (k / "steel/defines.h").write_text("#pragma once\nDEFINES\n")
    (k / "steel/gemm/nax.h").write_text(
        '#pragma once\n#include <metal_stdlib>\n#include "mlx/backend/metal/kernels/steel/defines.h"\n'
        '#include "mlx/backend/metal/kernels/utils.h"\n#include "mlx/backend/metal/kernels/steel/defines.h"\nNAX\n')
    text = seg.read_mlx_headers(tmp_path, seg._MLX_MM_HEADERS)
    assert text.count("DEFINES") == 1 and "NAX" in text and "UTILS" not in text
    assert "#pragma once" not in text and "#include <metal_stdlib>" in text
    with pytest.raises(seg.SegmentedMoEUnavailable):
        seg.read_mlx_headers(tmp_path / "missing", seg._MLX_MM_HEADERS)
    with pytest.raises(seg.SegmentedMoEUnavailable):
        seg.read_mlx_headers(tmp_path, seg._MLX_OPS_HEADERS)
