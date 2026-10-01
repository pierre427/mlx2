"""Host-only checks for the mechanism-intake merge into main.

Main parent 709bedf8 (split routed decode, served down, shared fold, MoE row
windows, top-k launch/fold) and intake parent b6d54c2e (default-off omlx #4113
routed candidate, ``down_calls``/``degraded`` counters, block score plumbing)
both edited the routed-decode files. These tests execute AST-extracted merged
functions against fakes and compare merged symbols with the exact parent
sources. Nothing native is imported: mlx, mlx_lm, mlx_vlm and mlx2 are
poisoned for the whole module, and no kernel is constructed.
"""

from __future__ import annotations

import __future__
import ast
import importlib.abc
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_NATIVE = {"mlx", "mlx_lm", "mlx_vlm", "mlx2", "mlx_metal"}


class _NativeImportPoison(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in _NATIVE:
            raise ImportError(f"native import poisoned in host-only test: {name}")
        return None


sys.meta_path.insert(0, _NativeImportPoison())

ROOT = Path(__file__).resolve().parents[1]
MAIN = "709bedf8dabf671b6d60a326bae26fe49b92ae7c"
INTAKE = "b6d54c2e19b6ce4b6012f05332749d39bd9e7192"
BASE = "d5886a98925dc9e731dcef3f45552d60e8c88612"
# The merge as committed. Later edits to these files (e.g. the served-exp
# gates, recon-20261001) carry their own tests; this audit checks the merge.
MERGE = "e311587a3e4092c27a4e87d6baad94dec2b6decf"
QN = "src/mlx2/runtime/models/qwen3_next.py"
RD = "src/mlx2/runtime/models/qwen4_routed_decode.py"
WIN = "src/mlx2/runtime/models/qwen4_moe_window.py"
GDN = "src/mlx2/runtime/models/qwen4_fused_gdn.py"
FN = "src/mlx2/adapters/flash_next.py"
TESTS = "tests/test_omlx_fn_ports.py"
PROV = "docs/PROVENANCE.md"
H = 16
TOP_K = 10


def _git_show(rev: str, path: str) -> str:
    out = subprocess.run(
        ["git", "show", f"{rev}:{path}"], cwd=ROOT, capture_output=True, text=True
    )
    if out.returncode != 0:
        pytest.skip(f"parent object {rev[:8]}:{path} unavailable: {out.stderr.strip()}")
    return out.stdout


def _read(path: str) -> str:
    return _git_show(MERGE, path)


def _symbols(src: str) -> dict:
    """Top-level functions/classes/assignments and ``Class.method`` -> node."""
    found = {}
    for node in ast.parse(src).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            found[node.name] = node
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        found[f"{node.name}.{item.name}"] = item
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    found[target.id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            found[node.target.id] = node
    return found


def _dumps(src: str) -> dict:
    return {name: ast.dump(node) for name, node in _symbols(src).items()}


def _literal(src: str, name: str):
    return ast.literal_eval(_symbols(src)[name].value)


def _load(names, namespace, src=None):
    """Exec the named merged qwen3_next symbols into ``namespace``."""
    nodes = _symbols(src if src is not None else _read(QN))
    module = ast.Module(body=[nodes[name] for name in names], type_ignores=[])
    code = compile(
        module, f"<merged {QN}>", "exec",
        flags=__future__.annotations.compiler_flag, dont_inherit=True,
    )
    exec(code, namespace)
    return namespace


class A:
    """Array stand-in: shape, size, dtype and a tag that survives reshape."""

    def __init__(self, shape, dtype="bf16", tag=None):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.tag = tag
        self.size = 1
        for extent in self.shape:
            self.size *= int(extent)

    @property
    def ndim(self):
        return len(self.shape)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        return A(shape, self.dtype, self.tag)

    def __add__(self, other):
        return A(self.shape, self.dtype, ("add", self.tag))

    __radd__ = __add__

    def __mul__(self, other):
        return A(self.shape, self.dtype, ("mul", self.tag))

    __rmul__ = __mul__


class FakeModule:
    def __init__(self, *args, **kwargs):
        self.training = False

    def __contains__(self, name):
        return name in self.__dict__

    def get(self, name, default=None):
        return self.__dict__.get(name, default)


def _routed_constants():
    src = _read(RD)
    return {
        "MODES": _literal(src, "MODES"),
        "CANDIDATE_MODES": _literal(src, "CANDIDATE_MODES"),
    }


def _fake_routed(log, **overrides):
    accept = lambda *a, **k: SimpleNamespace(accepted=True, reason="eligible")
    routed = SimpleNamespace(
        **_routed_constants(),
        RoutedDecodeAdmission=lambda accepted, reason: SimpleNamespace(
            accepted=accepted, reason=reason
        ),
        runtime_supported=lambda: True,
        admit_routed_decode=lambda *a: (log.append("admit_fused"), accept())[1],
        admit_split_routed_decode=lambda *a: (log.append("admit_split"), accept())[1],
        gate_up_swiglu=lambda x, i, gu: (log.append("gate_up_swiglu"), A((TOP_K, 8), tag="hidden"))[1],
        split_gate_up_swiglu=lambda x, i, g, u: (
            log.append("split_gate_up_swiglu"), A((TOP_K, 8), tag="hidden"))[1],
        served_down=lambda h, i, s, d: (log.append("served_down"), A((H,), tag="served"))[1],
        down_combine=lambda h, i, s, d: (log.append("down_combine"), A((H,), tag="combine"))[1],
        admit_routed_candidate=lambda *a: (log.append("admit_candidate"), accept())[1],
        candidate_runtime_refusal=lambda: None,
        candidate_gate_up_swiglu=lambda x, i, g, u: (
            log.append("candidate_gate_up"), A((TOP_K, 8), tag="cand_hidden"))[1],
        candidate_down_combine=lambda h, i, s, d: (
            log.append("candidate_down"), A((H,), tag="candidate"))[1],
        shared_fold_decode=lambda *a: (log.append("shared_fold_decode"), A((H,), tag="fold"))[1],
    )
    for key, value in overrides.items():
        setattr(routed, key, value)
    return routed


def _switch(split=True):
    sw = FakeModule()
    if split:
        sw.gate_proj, sw.up_proj = "gate_table", "up_table"
    else:
        sw.gate_up_proj = "gate_up_table"
    sw.down_proj = "down_table"
    return sw


def _counters_ns(log, **routed_overrides):
    ns = {"_routed": _fake_routed(log, **routed_overrides)}
    return _load(
        ["_enable_routed_decode", "_try_routed_decode", "_CANDIDATE_REASON_SLOTS",
         "_enable_routed_candidate", "_try_routed_candidate"],
        ns,
    )


def _one_token():
    return A((H,)), A((1, TOP_K), "uint32"), A((1, TOP_K))


# ---------------------------------------------------------------- structure


def test_native_imports_are_poisoned():
    for name in ("mlx", "mlx.core", "mlx_lm", "mlx_vlm", "mlx2"):
        with pytest.raises(ImportError, match="poisoned"):
            __import__(name)
    assert not any(name.split(".")[0] in _NATIVE for name in sys.modules)


@pytest.mark.parametrize("path", [PROV, QN, RD, TESTS])
def test_resolved_files_have_no_conflict_markers(path):
    text = _read(path)
    for marker in ("<<<<<<< ", "\n=======\n", ">>>>>>> ", "||||||| "):
        assert marker not in text, (path, marker)
    if path.endswith(".py"):
        ast.parse(text)


# Whole sections added after the merge.  Removing exactly these must give the
# merge's text back byte for byte, so every original byte is still proven.
_PROV_POST_MERGE_SECTIONS = (
    "## 2026-10-01 — memory-budgeted disk weight streaming\n",
)


def _without_post_merge_sections(text):
    for header in _PROV_POST_MERGE_SECTIONS:
        assert text.count(header) == 1, header
        start = text.index(header)
        end = text.find("\n## ", start + len(header))
        text = text[:start] + (text[end + 1:] if end >= 0 else "")
    return text


def test_provenance_keeps_every_main_and_intake_section_verbatim():
    base_lines = _git_show(BASE, PROV).splitlines(True)
    intake_tail = "".join(_git_show(INTAKE, PROV).splitlines(True)[len(base_lines):])
    assert _without_post_merge_sections(_read(PROV)) == _git_show(MAIN, PROV) + intake_tail


def test_routed_decode_module_is_main_plus_the_intake_candidate_block():
    base = _git_show(BASE, RD)
    intake = _git_show(INTAKE, RD)
    assert intake.startswith(base)  # intake only appended
    assert _read(RD) == _git_show(MAIN, RD) + intake[len(base):]


_CANDIDATE_OWN = {
    "CANDIDATE_MODES", "CANDIDATE_TOP_K", "qmv_fast_layout", "_candidate_table_ok",
    "admit_routed_candidate", "CANDIDATE_GATE_UP_SOURCE", "CANDIDATE_DOWN_SOURCE",
    "_CANDIDATE_KERNELS", "_candidate_kernel", "candidate_gate_up_swiglu",
    "candidate_down_combine", "candidate_runtime_refusal",
}


def test_candidate_dependencies_are_unchanged_by_main():
    """Every module-level name the intake candidate reads is either its own
    (equal to intake) or identical in base, main, intake and merge, so main's
    edits cannot reach the candidate's math or shapes."""
    base, main, intake, merged = (
        _dumps(_git_show(BASE, RD)), _dumps(_git_show(MAIN, RD)),
        _dumps(_git_show(INTAKE, RD)), _dumps(_read(RD)),
    )
    assert set(intake) - set(base) == _CANDIDATE_OWN
    for name in _CANDIDATE_OWN:
        assert merged[name] == intake[name], name
        assert name not in main, f"main also defines {name}"
    nodes = _symbols(_read(RD))
    reads = set()
    for name in _CANDIDATE_OWN | {"_header"}:
        for node in ast.walk(nodes[name]):
            if isinstance(node, ast.Name):
                reads.add(node.id)
    shared = (reads & set(merged)) - _CANDIDATE_OWN
    assert {"QMV_HEADER", "SIGMOID", "_header", "_quantized_ok", "BITS", "GROUP_SIZE",
            "GATE_UP_ROWS", "GATE_UP_SIMDGROUPS", "DOWN_ROWS", "RoutedDecodeAdmission",
            "runtime_supported"} <= shared
    for name in shared:
        assert base[name] == main[name] == intake[name] == merged[name], name
    # The served-SiLU probe the candidate gates on is intake's, unchanged.
    gdn_intake = _dumps(_git_show(INTAKE, GDN))["served_silu_refusal"]
    assert _dumps(_read(GDN))["served_silu_refusal"] == gdn_intake


def test_main_routed_helpers_are_main_verbatim():
    main, merged = _dumps(_git_show(MAIN, RD)), _dumps(_read(RD))
    for name in ("MODES", "_kernels", "gate_up_swiglu", "down_combine",
                 "admit_split_routed_decode", "split_gate_up_swiglu", "served_down",
                 "QMV_ROWS", "_format_header", "SHARED_COMMON", "admit_shared_fold",
                 "shared_fold_decode", "SERVED_DOWN_SOURCE", "SPLIT_GATE_UP_SOURCE"):
        assert merged[name] == main[name], name


_QN_MERGED = {
    "FusedDownSwitchGLU.__call__", "Qwen3NextSparseMoeBlock.__call__",
    "Qwen3NextSparseMoeBlock.__init__", "_enable_routed_decode", "_try_routed_decode",
    "_try_shared_fold", "_try_topk_fold_decode",
}
# Edited after the merge, each with its own equivalence test below:
# _concat_parts gained the disk-weight-streaming concat-ledger record.
_QN_POST_MERGE = {"_concat_parts"}


def test_qwen3_next_symbols_come_from_their_parent():
    main, intake, merged = (
        _dumps(_git_show(MAIN, QN)), _dumps(_git_show(INTAKE, QN)), _dumps(_read(QN))
    )
    assert set(main) | set(intake) <= set(merged)
    changed = {
        name for name, dump in merged.items()
        if dump != main.get(name) and dump != intake.get(name)
    }
    # The classes differ because their methods do.
    assert (
        changed - {"FusedDownSwitchGLU", "Qwen3NextSparseMoeBlock"} - _QN_POST_MERGE
        == _QN_MERGED
    )
    for name in ("_CANDIDATE_REASON_SLOTS", "_enable_routed_candidate",
                 "_try_routed_candidate", "routed_candidate_stats",
                 "Qwen3NextSparseMoeBlock.set_moe_routed_candidate_mode",
                 "Qwen3NextSparseMoeBlock.set_fused_expert_kernel_mode"):
        assert merged[name] == intake[name], name
    for name in ("_served_down_refusal", "_shared_fold_refusal", "_ShapeOnly",
                 "_enable_moe_window", "_moe_window_consumer", "_moe_rows_refusal",
                 "_shared_fold_ok", "_stock_routing", "_row_projections",
                 "_run_moe_window", "_try_moe_window", "_topk_refusal", "_try_topk_launch",
                 "FusedGateUpSwitchGLU.__call__",
                 "Qwen3NextSparseMoeBlock.set_moe_routed_decode_mode",
                 "Qwen3NextSparseMoeBlock.set_moe_window_consumers",
                 "Qwen3NextSparseMoeBlock.set_moe_topk_mode", "_MOE_WINDOW_CONSUMERS",
                 "_MOE_TOPK_MODE", "_MOE_ROUTED_DECODE"):
        assert merged[name] == main[name], name


def _concat_parts_from(src, namespace):
    """Exec ``_concat_parts`` from ``src`` with its in-function imports removed."""

    class StripImports(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            return None

        def visit_Import(self, node):
            return None

    node = StripImports().visit(ast.parse(ast.unparse(_symbols(src)["_concat_parts"])))
    code = compile(
        node, "<_concat_parts>", "exec",
        flags=__future__.annotations.compiler_flag, dont_inherit=True,
    )
    exec(code, namespace)
    return namespace["_concat_parts"]


def test_concat_parts_post_merge_hook_is_output_equivalent_with_an_inactive_ledger():
    """The ledger hook only records; with no ledger active (every ordinary
    load) the merged function returns exactly what main's does, and with one
    active it records each fused suffix once."""
    records = []
    fake_mx = SimpleNamespace(
        concatenate=lambda arrays, axis: ("cat", tuple(a.tag for a in arrays), axis)
    )
    main = _concat_parts_from(
        _git_show(MAIN, QN), {"mx": fake_mx}
    )
    inactive = _concat_parts_from(
        _read(QN), {"mx": fake_mx, "record_concat": lambda *_a: None}
    )
    active = _concat_parts_from(
        _read(QN),
        {"mx": fake_mx, "record_concat": lambda result, parts, axis: records.append(
            (result, tuple(p.tag for p in parts), axis))},
    )
    cases = [
        [{"weight": A((4, 6, 8), tag="g"), "scales": A((4, 6, 1), tag="gs")},
         {"weight": A((4, 6, 8), tag="u"), "scales": A((4, 6, 1), tag="us")}],
        [{"weight": A((4, 6, 8), tag="g")}, {"weight": A((4, 3, 8), tag="u")}],
        [{"weight": A((4, 6, 8), tag="g")}, {}],
        [{"weight": A((4, 6, 8), tag="g")}, {"scales": A((4, 6, 1), tag="s")}],
        [{"weight": A((4, 6, 8), tag="g")}, {"weight": A((4, 6, 9), tag="u")}],
        [{"weight": A((4, 6, 8), tag="g")}, {"weight": A((4, 6, 8, 1), tag="u")}],
    ]
    for parts in cases:
        for axis in (-2, 0):
            expected = main(parts, axis)
            assert inactive(parts, axis) == expected
            before = len(records)
            assert active(parts, axis) == expected
            assert len(records) - before == (0 if expected is None else len(expected))
    assert records, "the active ledger saw no fusion at all"


def _without_decode_down_counter(node):
    node = ast.parse(ast.unparse(node)).body[0]

    class Strip(ast.NodeTransformer):
        def visit_AugAssign(self, aug):
            target = aug.target
            if isinstance(target, ast.Attribute) and target.attr == "routed_decode_down_calls":
                return None
            return aug

    return ast.dump(Strip().visit(node))


@pytest.mark.parametrize("name", ["_try_shared_fold", "_try_topk_fold_decode"])
def test_fold_paths_differ_from_main_only_by_the_intake_down_counter(name):
    main = _symbols(_git_show(MAIN, QN))[name]
    merged = _symbols(_read(QN))[name]
    assert _without_decode_down_counter(merged) == _without_decode_down_counter(main)
    assert ast.dump(merged) != ast.dump(main)


def test_block_call_keeps_main_fast_paths_and_intake_score_plumbing_in_order():
    text = ast.unparse(_symbols(_read(QN))["Qwen3NextSparseMoeBlock.__call__"])
    order = [
        "_try_moe_window(self, x)", "_try_topk_fold_decode(self, x)",
        "_try_topk_launch(self, x, gates)", "_try_shared_fold(self, x, inds, scores)",
        "if self.fused_expert_kernel_enabled and self.sharding_group is None",
        "== 'two_launch'", "getattr(self.switch_mlp, 'routed_candidate_mode', 'off') != 'off'",
        "self.moe_weighted_sum", "y = self.switch_mlp(x, inds)\n",
    ]
    positions = [text.index(snippet) for snippet in order]
    assert positions == sorted(positions)
    assert "outcome in ('routed_two_launch', 'routed_gate_up_down')" in text


# ------------------------------------------------- FusedDownSwitchGLU dispatch


def _fused_down(log, routed_result, candidate_mode="off", candidate_admits=True):
    reject = SimpleNamespace(accepted=False, reason="top-k must be 8 or 10")
    overrides = {}
    if not candidate_admits:
        overrides["admit_routed_candidate"] = lambda *a: (log.append("admit_candidate"), reject)[1]
    ns = _counters_ns(log, **overrides)

    def try_routed_decode(sw, x, idx, scores, do_sort, variant):
        log.append(("routed_decode", variant))
        return routed_result

    ns.update(
        mx=SimpleNamespace(expand_dims=lambda x, axes: x, stop_gradient=lambda v: v),
        switch_layers_sort_min=lambda: 10**9,
        _try_routed_decode=try_routed_decode,
        _try_qwen4_fused_down=lambda h, i, s, d, ds, v: (log.append(("fused_down", h.tag, v)), None)[1],
        _fused_outcome=lambda v, i: "unused",
        _routed_tail=lambda sw, x, ind, inv, s, ds: (log.append(("tail", x.tag)), A((H,), tag="tail"))[1],
    )
    _load(["FusedDownSwitchGLU.__call__"], ns)
    sw = _switch(split=True)
    ns["_enable_routed_decode"](sw, "gate_up_down")
    ns["_enable_routed_candidate"](sw, candidate_mode)
    sw.activation = lambda up, gate: (log.append("stock_activation"), A((TOP_K, 8), tag="stock_hidden"))[1]
    sw.up_proj = lambda *a, **k: A((TOP_K, 8), tag="up")
    sw.gate_proj = lambda *a, **k: A((TOP_K, 8), tag="gate")
    sw.down_proj = lambda h, idx, sorted_indices: (log.append(("down_proj", h.tag)), A((H,), tag="down"))[1]
    return sw, ns["__call__"]


def test_selected_candidate_runs_before_main_routed_decode_on_stock():
    log = []
    sw, call = _fused_down(log, routed_result=("gate_up_down", A((H,), tag="served")),
                           candidate_mode="two_launch")
    x, idx, scores = _one_token()
    got = call(sw, x, idx, scores, "stock")
    assert got.tag == "candidate"
    assert log == ["admit_candidate", "candidate_gate_up", "candidate_down"]
    assert sw._last_fused_variant == "routed_candidate"
    assert sw.routed_candidate_calls == 1 and sw.routed_candidate_fallbacks == 0
    assert sw.routed_decode_calls == sw.routed_down_calls == sw.routed_decode_down_calls == 0


@pytest.mark.parametrize("candidate_mode,admits", [("off", True), ("two_launch", False)])
def test_candidate_off_or_declined_leaves_main_path(candidate_mode, admits):
    log = []
    sw, call = _fused_down(log, routed_result=("gate_up_down", A((H,), tag="served")),
                           candidate_mode=candidate_mode, candidate_admits=admits)
    x, idx, scores = _one_token()
    got = call(sw, x, idx, scores, "stock")
    assert got.tag == "served" and sw._last_fused_variant == "routed_gate_up_down"
    if candidate_mode == "off":
        assert log == [("routed_decode", "stock")]
        assert sw.routed_candidate_fallbacks == 0
    else:
        assert log == ["admit_candidate", ("routed_decode", "stock")]
        assert sw.routed_candidate_fallbacks == 1
        assert sw.routed_candidate_fallback_reasons == {"top-k must be 8 or 10": 1}
    assert sw.routed_candidate_calls == 0


def test_candidate_is_not_attempted_under_a_fused_expert_variant():
    log = []
    sw, call = _fused_down(log, routed_result=("gate_up", A((TOP_K, 1, 8), tag="routed_hidden")),
                           candidate_mode="two_launch")
    x, idx, scores = _one_token()
    got = call(sw, x, idx, scores, "tile4")
    # Main's gate_up-only hidden feeds the block's own down; no candidate.
    assert log == [("routed_decode", "tile4"), ("fused_down", "routed_hidden", "tile4"),
                   ("down_proj", "routed_hidden"), ("tail", "down")]
    assert got.tag == "tail" and sw._last_fused_variant is None
    assert sw.routed_candidate_calls == sw.routed_candidate_fallbacks == 0


def test_routed_decode_decline_runs_the_stock_body():
    log = []
    sw, call = _fused_down(log, routed_result=None)
    x, idx, scores = _one_token()
    got = call(sw, x, idx, scores, "stock")
    assert log == [("routed_decode", "stock"), "stock_activation",
                   ("fused_down", "stock_hidden", "stock"), ("down_proj", "stock_hidden"),
                   ("tail", "down")]
    assert got.tag == "tail"


# ------------------------------------------------- routed decode counters


def _decode(mode, scores=True, split=True, refusal=None, variant="tile4"):
    log = []
    ns = _counters_ns(log)
    seen = []
    ns["_served_down_refusal"] = lambda sw, inter, dtype, i, s, v: (seen.append(v), refusal)[1]
    sw = _switch(split=split)
    ns["_enable_routed_decode"](sw, mode)
    x, idx, sc = _one_token()
    out = ns["_try_routed_decode"](sw, x, idx, sc if scores else None, False, variant)
    return sw, out, log, seen


def _counts(sw):
    return (sw.routed_decode_calls, sw.routed_down_calls, sw.routed_decode_down_calls,
            sw.routed_decode_degraded, sw.routed_down_fallbacks, sw.routed_decode_fallbacks)


def test_enable_routed_decode_initialises_both_parents_counters():
    ns = _counters_ns([])
    sw = _switch()
    ns["_enable_routed_decode"](sw, "gate_up_down_shared")
    assert sw.routed_decode_mode == "gate_up_down_shared"
    for name in ("routed_decode_calls", "routed_decode_down_calls", "routed_decode_degraded",
                 "routed_decode_fallbacks", "routed_down_calls", "routed_down_fallbacks"):
        assert getattr(sw, name) == 0, name
    assert sw.routed_decode_last_fallback is None and sw.routed_down_last_fallback is None
    with pytest.raises(ValueError):
        ns["_enable_routed_decode"](sw, "on")


@pytest.mark.parametrize("mode", ["gate_up_down", "gate_up_down_shared"])
def test_served_down_counts_on_both_down_counters(mode):
    sw, out, log, seen = _decode(mode)
    assert out[0] == "gate_up_down" and out[1].tag == "served"
    assert log == ["admit_split", "split_gate_up_swiglu", "served_down"]
    assert seen == ["tile4"]  # main's variant reaches the served-down admission
    assert _counts(sw) == (1, 1, 1, 0, 0, 0)


def test_two_launch_counts_on_both_down_counters():
    sw, out, log, _ = _decode("two_launch", split=False)
    assert out[0] == "two_launch" and out[1].tag == "combine"
    assert log == ["admit_fused", "gate_up_swiglu", "down_combine"]
    assert _counts(sw) == (1, 1, 1, 0, 0, 0)


@pytest.mark.parametrize("mode", ["two_launch", "gate_up_down", "gate_up_down_shared"])
def test_missing_scores_is_counted_as_degraded(mode):
    sw, out, log, seen = _decode(mode, scores=False)
    assert out[0] == "gate_up" and out[1].tag == "hidden"
    assert "served_down" not in log and "down_combine" not in log and seen == []
    assert _counts(sw) == (1, 0, 0, 1, 0, 0)


def test_gate_up_mode_and_served_refusal_are_not_degraded():
    sw, out, _, _ = _decode("gate_up", scores=False)
    assert out[0] == "gate_up" and _counts(sw) == (1, 0, 0, 0, 0, 0)
    sw, out, log, _ = _decode("gate_up_down", refusal="block down variant 'scalar' is not tile4")
    assert out[0] == "gate_up" and "served_down" not in log
    assert _counts(sw) == (1, 0, 0, 0, 1, 0)
    assert sw.routed_down_last_fallback.startswith("block down variant")


def test_declined_admission_counts_a_fallback_only():
    reject = SimpleNamespace(accepted=False, reason="bits")
    ns = _counters_ns([], admit_split_routed_decode=lambda *a: reject)
    sw = _switch()
    ns["_enable_routed_decode"](sw, "two_launch")
    x, idx, _ = _one_token()
    assert ns["_try_routed_decode"](sw, x, idx, None, False, "stock") is None
    assert _counts(sw) == (0, 0, 0, 0, 0, 1) and sw.routed_decode_last_fallback == "bits"


def test_shared_and_topk_folds_keep_down_counters_in_lockstep():
    ns = _counters_ns([])
    ns.update(
        _shared_fold_refusal=lambda *a: None,
        _topk_refusal=lambda *a: None,
        _moe_rows_refusal=lambda *a: None,
        _shared_fold_ok=lambda *a: True,
        gate_sigmoid=lambda v: v,
        _window=SimpleNamespace(
            routed_rows=lambda *a, **k: (A((1, H), tag="rows"), None, None),
            shared_rows=lambda *a, **k: (A((1, H), tag="shared_rows"), None, None),
        ),
    )
    _load(["_try_shared_fold", "_try_topk_fold_decode"], ns)
    sw = _switch()
    ns["_enable_routed_decode"](sw, "gate_up_down_shared")
    block = SimpleNamespace(
        switch_mlp=sw, shared_fold_calls=0, shared_fold_fallbacks=0,
        shared_fold_last_fallback=None, shared_expert="shared",
        shared_expert_gate=lambda x: A((1,), x.dtype, "logit"),
        moe_topk_mode="fold", moe_topk_calls={"launch": 0, "fold": 0},
        moe_topk_fallbacks=0, moe_topk_last_fallback=None, num_experts=32,
        gate=lambda x: A((32,), x.dtype, "gates"),
    )
    x, idx, scores = _one_token()
    assert ns["_try_shared_fold"](block, x, idx, scores).tag == "fold"
    assert ns["_try_topk_fold_decode"](block, x) is not None
    sw.routed_decode_mode = "gate_up_down"
    block.shared_expert = lambda x: A(x.shape, x.dtype, "shared_y")
    assert ns["_try_topk_fold_decode"](block, x) is not None
    assert sw.routed_decode_calls == sw.routed_down_calls == sw.routed_decode_down_calls == 3
    assert sw.routed_decode_degraded == 0
    assert block.shared_fold_calls == 2 and block.moe_topk_calls["fold"] == 2


# ------------------------------------------------- block init and setters


class _FusedGateUp(FakeModule):
    def __init__(self, *a):
        super().__init__()
        self.gate_up_proj, self.down_proj = "gate_up_table", "down_table"


class _FusedDown(FakeModule):
    def __init__(self, *a):
        super().__init__()
        self.gate_proj, self.up_proj, self.down_proj = "gate_table", "up_table", "down_table"


def _block(routed_default="gate_up_down_shared", fused_gate_up=False, shared_in_gather=False):
    ns = _counters_ns([])
    win_src = _read(WIN)
    qn_src = _read(QN)
    ns.update(
        nn=SimpleNamespace(Module=FakeModule, Linear=lambda *a, **k: "linear"),
        _window=SimpleNamespace(
            CONSUMERS=_literal(win_src, "CONSUMERS"), TOPK_MODES=_literal(win_src, "TOPK_MODES")
        ),
        _MOE_FUSED_GATE_UP=fused_gate_up,
        _MOE_SHARED_IN_GATHER=shared_in_gather,
        _MOE_FUSED_EXPERT_MODE="auto",
        _MOE_FUSED_EXPERT_MODES=_literal(qn_src, "_MOE_FUSED_EXPERT_MODES"),
        _MOE_ROUTER_KERNEL=False,
        _MOE_ROUTER_MODES=_literal(qn_src, "_MOE_ROUTER_MODES"),
        _MOE_WEIGHTED_SUM=False,
        _enable_moe_weighted_sum=lambda sw, enabled: setattr(sw, "moe_weighted_sum", enabled),
        _MOE_ROUTED_DECODE=routed_default,
        _MOE_WINDOW_CONSUMERS=(),
        _MOE_TOPK_MODE="off",
        FusedGateUpSwitchGLU=_FusedGateUp,
        FusedDownSwitchGLU=_FusedDown,
        Qwen3NextMLP=lambda *a: "shared_mlp",
    )
    _load(["_enable_moe_window", "Qwen3NextSparseMoeBlock"], ns)
    args = SimpleNamespace(
        hidden_size=H, moe_intermediate_size=8, shared_expert_intermediate_size=8,
        norm_topk_prob=True, num_experts=32, num_experts_per_tok=TOP_K,
    )
    return ns["Qwen3NextSparseMoeBlock"](args)


def test_block_init_keeps_main_default_and_attaches_intake_counters():
    block = _block()
    sw = block.switch_mlp
    assert sw.routed_decode_mode == "gate_up_down_shared"  # main's split-table default
    assert sw.routed_decode_degraded == sw.routed_decode_down_calls == sw.routed_down_calls == 0
    assert sw.routed_candidate_mode == "off" and sw.routed_candidate_calls == 0
    assert block.shared_fold_calls == 0 and block.moe_topk_mode == "off"
    fused = _block(fused_gate_up=True).switch_mlp
    assert fused.routed_decode_mode == "gate_up_down_shared"
    assert fused.routed_decode_degraded == 0 and not hasattr(fused, "routed_candidate_mode")
    folded = _block(shared_in_gather=True)
    assert folded.switch_mlp.routed_decode_mode == "off"
    assert folded.fused_expert_kernel_mode == "stock"


def test_block_setters_return_and_keep_the_candidate_exclusive():
    block = _block()
    assert block.set_moe_topk_mode("launch") == "launch" and block.moe_topk_mode == "launch"
    assert block.set_moe_window_consumers(["verify"]) == frozenset({"verify"})
    assert block.set_moe_routed_decode_mode("two_launch") == "two_launch"
    with pytest.raises(ValueError, match="stock expert kernel"):
        block.set_moe_routed_candidate_mode("two_launch")  # fused mode "auto" here
    block.set_fused_expert_kernel_mode("stock")
    assert block.set_moe_routed_candidate_mode("two_launch") == "two_launch"
    with pytest.raises(ValueError, match="exclude the routed candidate"):
        block.set_fused_expert_kernel_mode("tile4")
    assert block.set_moe_routed_candidate_mode("off") == "off"
    block.set_fused_expert_kernel_mode("tile4")
    with pytest.raises(ValueError):
        _block(fused_gate_up=True).set_moe_routed_candidate_mode("two_launch")


# ------------------------------------------------- Flash-Next diagnostics


def _diagnostics(src):
    return _symbols(src)["FlashNextAdapter.diagnostics"]


def _routed_decode_dict(fn):
    """The dict literal assigned to ``moe["routed_decode"]``."""
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Dict)
            and any(
                isinstance(t, ast.Subscript) and isinstance(t.slice, ast.Constant)
                and t.slice.value == "routed_decode"
                for t in node.targets
            )
        ):
            return node.value
    raise AssertionError("routed_decode diagnostics dict not found")


def _entries(d):
    return {k.value: ast.dump(v) for k, v in zip(d.keys, d.values)}


def test_flash_next_diagnostics_have_no_duplicate_literal_keys():
    for node in ast.walk(_diagnostics(_read(FN))):
        if isinstance(node, ast.Dict):
            keys = [k.value for k in node.keys if isinstance(k, ast.Constant)]
            assert len(keys) == len(set(keys)), sorted(k for k in keys if keys.count(k) > 1)


def test_flash_next_routed_diagnostics_keep_main_down_calls_and_intake_degraded():
    merged = _routed_decode_dict(_diagnostics(_read(FN)))
    main = _entries(_routed_decode_dict(_diagnostics(_git_show(MAIN, FN))))
    intake = _entries(_routed_decode_dict(_diagnostics(_git_show(INTAKE, FN))))
    got = _entries(merged)
    assert got["down_calls"] == main["down_calls"]
    assert "routed_decode_down_calls" not in got["down_calls"]
    assert "'routed_down_calls'" in got["down_calls"]
    assert got["degraded"] == intake["degraded"] and "degraded" not in main
    # Everything else is main's, in main's order, with intake's key added.
    assert [k for k in got if k != "degraded"] == list(main)
    assert {k: v for k, v in got.items() if k != "degraded"} == main
    # The rest of diagnostics is main's verbatim once ``degraded`` is removed.
    fn = ast.parse(ast.unparse(_diagnostics(_read(FN)))).body[0]
    d = _routed_decode_dict(fn)
    i = [k.value for k in d.keys].index("degraded")
    del d.keys[i], d.values[i]
    assert ast.dump(fn) == ast.dump(_diagnostics(_git_show(MAIN, FN)))
