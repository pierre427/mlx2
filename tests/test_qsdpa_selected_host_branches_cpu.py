"""Host-only A3 evidence: no real MLX import, model, tensor, or GPU work.

Run with --noconftest to bypass the repository's MLX-importing conftest.
Checkpoint: no_model@not_applicable; base source19d41470b plus bound dirty bytes.
The dispatch is the original production function AST, not a reimplementation.
"""

from __future__ import annotations

import ast
import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "src/mlx2/runtime/models"
BRANCHES = {
    "direct_decode",
    "direct_verify",
    "composed",
    "composed_tiled",
    "dequantized_flash",
}


def function_node():
    return next(
        node
        for node in ast.parse((MODELS / "base.py").read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "quantized_scaled_dot_product_attention"
    )


@pytest.fixture
def host(monkeypatch):
    # Loading the entire counter-owning module against an inert stand-in
    # exercises its real definitions/imports, without initializing MLX.
    mlx = ModuleType("mlx")
    mlx.__path__ = []
    mx = ModuleType("mlx.core")
    mlx.core = mx
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    package = ModuleType("_a3_shape_probe")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, package.__name__, package)
    qvm = ModuleType(package.__name__ + ".qsdpa_verify_metal")
    fork_callbacks = []
    monkeypatch.setattr(
        os,
        "register_at_fork",
        lambda **kwargs: fork_callbacks.append(kwargs["after_in_child"]),
        raising=False,
    )
    exec(  # noqa: S102 - trusted source with inert MLX stand-in
        compile(
            (MODELS / "qsdpa_verify_metal.py").read_text(),
            str(MODELS / "qsdpa_verify_metal.py"),
            "exec",
        ),
        qvm.__dict__,
    )
    package.qsdpa_verify_metal = qvm
    monkeypatch.setitem(sys.modules, qvm.__name__, qvm)
    qdm = ModuleType(package.__name__ + ".qsdpa_decode_metal")
    monkeypatch.setitem(sys.modules, qdm.__name__, qdm)
    transcript = []
    cfg = SimpleNamespace(
        decode=False, verify=False, causal=True, tiles=None, fail=None
    )

    class Array:
        def __init__(self, shape, label):
            self.shape, self.label = shape, label

        def __getitem__(self, key):
            rows = key[2].stop - key[2].start
            transcript.append(("query_slice", key[2].start, key[2].stop))
            return Array((self.shape[0], self.shape[1], rows, self.shape[3]), "slice")

    def result(kind, *args, **kwargs):
        transcript.append(kind)
        if cfg.fail == kind:
            raise RuntimeError("injected branch construction failure")
        if kind == "verify":
            qvm.note("verify_kernel", args[0].shape[2])
        return kind

    qdm.use_decode_kernel = lambda *a, **k: cfg.decode
    qdm.gqa_quantized_decode_attention = lambda *a, **k: result("decode", *a, **k)
    qvm.use_verify_kernel = lambda *a, **k: cfg.verify
    qvm.causal_left_padding = lambda *a: None if cfg.causal else qvm.NOT_CAUSAL
    qvm.gqa_quantized_verify_attention = lambda *a, **k: result("verify", *a, **k)
    mx.dequantize = lambda *a, **k: result("dequantize", *a, **k)
    mx.eval = lambda *a: transcript.append("eval")
    mx.concatenate = lambda *a, **k: result("concatenate", *a, **k)
    # No arange/comparison implementation: tiled tests use the unchanged
    # unmasked branch. A new device read or synchronization fails immediately.
    env = {
        "__package__": package.__name__,
        "mx": mx,
        "_QUANT_SDPA_FLASH_MIN_L_MHA": 192,
        "_QUANT_SDPA_FLASH_MIN_L_GQA": 128,
        "_contiguous_quant": lambda x: x,
        "_scores_tile_rows": lambda *a: cfg.tiles,
        "_composed_qsdpa": lambda *a: result("composed", *a),
        "fast_sdpa": lambda *a, **k: result("flash", *a, **k),
    }
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            function_node(),
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(MODELS / "base.py"), "exec"), env)  # noqa: S102 - original production dispatch, shape-only stand-ins

    def call(rows, *, gqa=True, mask="causal"):
        query = Array((1, 12, rows, 128), "q")
        keys = (Array((1, 2 if gqa else 12, 4096, 16), "k"), object(), object())
        return env["quantized_scaled_dot_product_attention"](
            query, keys, keys, scale=0.125, mask=mask
        )

    return SimpleNamespace(
        qvm=qvm,
        cfg=cfg,
        env=env,
        call=call,
        transcript=transcript,
        mx=mx,
        fork_callbacks=fork_callbacks,
    )


@pytest.mark.parametrize(
    "rows,gqa,decode,verify,causal,threshold,tiles,expected,ops",
    [
        (1, True, True, True, True, None, None, "direct_decode", ["decode"]),
        (1, True, False, True, True, None, None, "direct_verify", ["verify"]),
        (1, True, False, False, True, None, None, "composed", ["composed"]),
        (3, True, False, True, True, None, None, "direct_verify", ["verify"]),
        (3, True, False, True, False, None, None, "composed", ["composed"]),
        (8, True, False, False, True, None, None, "composed", ["composed"]),
        (9, True, False, False, True, None, None, "composed", ["composed"]),
        (127, True, False, False, True, None, None, "composed", ["composed"]),
        (
            128,
            True,
            False,
            False,
            True,
            None,
            None,
            "dequantized_flash",
            ["dequantize", "dequantize", "flash"],
        ),
        (191, False, False, False, True, None, None, "composed", ["composed"]),
        (
            192,
            False,
            False,
            False,
            True,
            None,
            None,
            "dequantized_flash",
            ["dequantize", "dequantize", "flash"],
        ),
        (
            3,
            True,
            False,
            False,
            True,
            1,
            None,
            "dequantized_flash",
            ["dequantize", "dequantize", "flash"],
        ),
        (192, True, False, False, True, float("inf"), None, "composed", ["composed"]),
        (
            12,
            True,
            False,
            False,
            True,
            None,
            [4, 8],
            "composed_tiled",
            [
                ("query_slice", 0, 4),
                "composed",
                "eval",
                ("query_slice", 4, 12),
                "composed",
                "eval",
                "concatenate",
            ],
        ),
    ],
)
def test_original_dispatch_has_one_terminal_selection(
    host, rows, gqa, decode, verify, causal, threshold, tiles, expected, ops
):
    host.cfg.decode, host.cfg.verify, host.cfg.causal, host.cfg.tiles = (
        decode,
        verify,
        causal,
        tiles,
    )
    if threshold is not None:
        host.env["_QUANT_SDPA_FLASH_MIN_L_GQA"] = threshold
    before = host.qvm.selected_host_branch_snapshot()
    host.call(rows, gqa=gqa, mask=None if tiles else "causal")
    got = host.qvm.selected_host_branch_delta(before)
    assert got["counts"] == {branch: int(branch == expected) for branch in BRANCHES}
    assert host.transcript == ops
    # Keep legacy fallback events separate. In particular they may report
    # composed before a threshold override actually selects dequantized-flash.
    if rows <= 8 and expected not in {"direct_decode", "direct_verify"}:
        assert host.qvm.STATS["composed_calls"] == 1
    if tiles:
        assert host.qvm.STATS["composed_tiled_calls"] == 1
        assert host.qvm.STATS["composed_tile_calls"] == 2


@pytest.mark.parametrize(
    "kind,branch,rows",
    [
        ("decode", "direct_decode", 1),
        ("dequantize", "dequantized_flash", 128),
        ("composed", "composed", 9),
    ],
)
def test_failed_construction_is_selection_not_completed_execution(
    host, kind, branch, rows
):
    host.cfg.fail = kind
    host.cfg.decode = kind == "decode"
    with pytest.raises(RuntimeError, match="injected"):
        host.call(rows)
    assert host.qvm.selected_host_branch_snapshot()["counts"][branch] == 1


def test_snapshot_delta_reset_and_legacy_independence(host):
    qvm = host.qvm
    old = qvm.selected_host_branch_snapshot()
    qvm.note("composed", 3)
    qvm.note_selected_host_branch("composed")
    snap = qvm.selected_host_branch_snapshot()
    assert snap["scope"] == "process" and snap["basis"] == "selected_host_graph_branch"
    assert qvm.selected_host_branch_delta(old)["counts"]["composed"] == 1
    snap["counts"]["composed"] = 999
    assert qvm.selected_host_branch_snapshot()["counts"]["composed"] == 1
    reset = qvm.reset_selected_host_branches()
    assert reset["epoch"] == old["epoch"] + 1
    assert all(value == 0 for value in reset["counts"].values())
    assert qvm.STATS["composed_calls"] == 1 and qvm.STATS_BY_ROWS["composed_L3"] == 1
    with pytest.raises(ValueError, match="incompatible or reset"):
        qvm.selected_host_branch_delta(old)
    qvm.note_selected_host_branch("direct_decode")
    assert qvm.selected_host_branch_delta(reset)["counts"]["direct_decode"] == 1


@pytest.mark.parametrize(
    "mutation", ["missing", "extra", "negative", "future", "bool", "process", "basis"]
)
def test_invalid_delta_baseline_refuses(host, mutation):
    baseline = host.qvm.selected_host_branch_snapshot()
    if mutation == "missing":
        del baseline["counts"]["composed"]
    elif mutation == "extra":
        baseline["counts"]["new_shape"] = 0
    elif mutation in {"negative", "future", "bool"}:
        baseline["counts"]["composed"] = {"negative": -1, "future": 1, "bool": False}[
            mutation
        ]
    elif mutation == "process":
        baseline["process_id"] += 1
    else:
        baseline["basis"] = "completed_gpu_launch"
    with pytest.raises(ValueError):
        host.qvm.selected_host_branch_delta(baseline)


def test_bounded_labels_and_concurrent_snapshot_counts(host):
    qvm = host.qvm
    for index in range(100):
        with pytest.raises(KeyError):
            qvm.note_selected_host_branch(f"shape_{index}")

    def worker(_):
        for _ in range(500):
            qvm.note_selected_host_branch("composed")

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(worker, range(8)))
    snap = qvm.selected_host_branch_snapshot()
    assert set(snap["counts"]) == BRANCHES
    assert snap["counts"]["composed"] == 4000


def test_fork_callback_discards_parent_counts_and_inherited_locked_mutex(host):
    qvm = host.qvm
    qvm.note_selected_host_branch("composed")
    parent = qvm.selected_host_branch_snapshot()
    old_lock = qvm._SELECTED_HOST_BRANCH_LOCK
    assert len(host.fork_callbacks) == 1
    old_lock.acquire()
    try:
        host.fork_callbacks[0]()
        assert qvm._SELECTED_HOST_BRANCH_LOCK is not old_lock
        child = qvm.selected_host_branch_snapshot()
        assert child["epoch"] == parent["epoch"] + 1
        assert all(value == 0 for value in child["counts"].values())
        qvm.note_selected_host_branch("direct_decode")
        assert qvm.selected_host_branch_delta(child)["counts"]["direct_decode"] == 1
        with pytest.raises(ValueError, match="incompatible or reset"):
            qvm.selected_host_branch_delta(parent)
    finally:
        old_lock.release()


def test_arithmetic_and_dispatch_ast_unchanged():
    class StripObservation(ast.NodeTransformer):
        def visit_Expr(self, node):
            call = node.value
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "qvm"
                and call.func.attr == "note_selected_host_branch"
            ):
                return None
            return node

        def visit_ImportFrom(self, node):
            if node.level == 1 and any(
                alias.name == "qsdpa_verify_metal" for alias in node.names
            ):
                return None
            return node

    normalized = StripObservation().visit(function_node())
    # Pinned original function at19d41470, ignoring only qvm import placement.
    # Python 3.12 adds an empty type_params field to FunctionDef; ignore it
    # so this ordinary, nongeneric function has one cross-version identity.
    dumped = ast.dump(normalized, include_attributes=False).replace(
        ", type_params=[]", ""
    )
    assert (
        hashlib.sha256(dumped.encode()).hexdigest()
        == "31c7728c48dc9ff55cb513a948293611ba5f36557770eb238686c6a7d9f8615f"
    )


def test_status_addition_preserves_legacy_shape():
    tree = ast.parse((ROOT / "src/mlx2/serving.py").read_text())
    engine = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ServingEngine"
    )
    status = next(
        node
        for node in engine.body
        if isinstance(node, ast.FunctionDef) and node.name == "status"
    )
    value = next(
        value
        for node in ast.walk(status)
        if isinstance(node, ast.Dict)
        for key, value in zip(node.keys, node.values)
        if isinstance(key, ast.Constant) and key.value == "qsdpa_verify"
    )
    assert [key.value for key in value.keys] == [
        "counts",
        "by_rows",
        "selected_host_branches",
    ]
    assert ast.unparse(value.values[0]) == "dict(qvm.STATS)"
    assert ast.unparse(value.values[1]) == "dict(qvm.STATS_BY_ROWS)"
    assert ast.unparse(value.values[2]) == "qvm.selected_host_branch_snapshot()"
