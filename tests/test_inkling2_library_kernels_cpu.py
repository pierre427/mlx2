"""``inkling2/library_kernels`` (clean-room, 2026-11-23), CPU only.

Checkpoint: synthetic (no model weights). Covers the spec's acceptance
properties that need no GPU: L1 (resolver output on the installed MLX 0.32.4
wheel, compiled offline with ``xcrun metal``), L2 (resolver robustness on a
temporary copy of the include tree), L4 (branch declines), L5 (the proof
protocol with stand-in launches) and L7 (nothing persisted). Every generated
mat-vec launch source for the spec's L3 shape grid is compiled offline
(AIR only, nothing runs on the GPU). The GPU properties are in
``tests/test_inkling2_library_kernels_metal.py``.
"""

import shutil
import threading
from pathlib import Path

import mlx.core as mx
import pytest

from mlx2.runtime.models.inkling2 import library_kernels as lib

KERNELS = "mlx/backend/metal/kernels/"
OUT_LENS = [3, 4, 511, 512, 2047, 2048, 4095, 4096, 8192]
KS = [64, 65, 1024, 2304, 8192]
DTYPES = [mx.float32, mx.float16, mx.bfloat16]


class Box:
    pass


# ----- L1: resolver on the installed wheel -----------------------------------

@pytest.mark.parametrize("parts", [("matvec",), ("elementwise",), ("matvec", "elementwise")])
def test_joined_text_is_clean(parts):
    text, why = lib.library(*parts)
    assert why is None, why
    assert '#include "mlx/' not in text
    assert "#pragma once" not in text
    for carried in sorted(lib.carried_set()):
        assert f"---- {KERNELS}{carried} " not in text, carried
    structs = {"matvec": ["GEMVKernel", "GEMVTKernel", "GemvWide"],
               "elementwise": ["Add", "Multiply", "Sigmoid", "Exp", "Tanh", "Rsqrt"]}
    for part in parts:
        for name in structs[part]:
            assert text.count(f"struct {name} {{") == 1, name
    assert mx.__version__ in text.splitlines()[0]


def test_carried_set_is_utils_closure():
    assert lib.carried_set() == {"utils.h", "bf16.h", "bf16_math.h", "complex.h", "defines.h",
                                 "logging.h"}


def test_receipt_records_version_and_hash():
    lib.library("matvec")
    receipt = lib.library_receipt()
    assert receipt["mlx_version"] == mx.__version__
    info = receipt["parts"]["matvec"]
    assert len(info["sha256"]) == 64 and info["files"] == ["steel/utils.h", "gemv.h"]


def test_function_constants_become_false_constants():
    text, _ = lib.library("matvec")
    assert "function_constant(" not in text
    assert "constant constexpr bool gemv_wide_has_batch = false;" in text
    assert "constant constexpr bool gemv_wide_do_axpby = false;" in text


def _compile_or_skip():
    from inkling2_metal_compile import compiler

    if compiler() is None:
        pytest.skip("no offline Metal compiler (xcrun metal)")


# MLX's JIT options on this macOS (metal_kernel: MathMode::Safe, language
# version 4.1 on macOS 27) and the helper's default.
JIT_OPTIONS = [("metal3.1", ()), ("metal4.1", ("-fmetal-math-mode=safe",))]


@pytest.mark.parametrize("std,flags", JIT_OPTIONS)
@pytest.mark.parametrize("part", ["matvec", "elementwise"])
def test_each_part_compiles_in_a_trivial_kernel(part, std, flags):
    from inkling2_metal_compile import compile_unit, translation_unit

    _compile_or_skip()
    header, why = lib.library(part)
    assert why is None
    body = "out[0] = inp[0];" if part == "matvec" else "out[0] = Sigmoid()(inp[0]);"
    err = compile_unit(translation_unit(f"trivial_{part}", [("inp", mx.float32, 16)],
                                        [("out", mx.float32)], body, header), std, flags)
    assert err is None, err


def _shape_grid_plans():
    seen = {}
    for dtype in DTYPES:
        for n in OUT_LENS:
            for k in KS:
                kind, tiles = lib.matvec_plan(1, n, k, dtype)
                assert kind == "gemv", (n, k, tiles)
                seen.setdefault((kind, tiles, dtype, 1), n)
        if dtype != mx.float32:
            for rows in range(2, lib.WIDE_MAX_ROWS + 1):
                for n in (24, 64, 65, 3072, 65536):
                    kind, tiles = lib.matvec_plan(rows, n, 2304, dtype, generation=17)
                    assert kind == "wide", (rows, n, tiles)
                    seen.setdefault((kind, tiles, dtype, rows), n)
    return [(kind, tiles, dtype, rows, n) for (kind, tiles, dtype, rows), n in seen.items()]


@pytest.mark.parametrize("std,flags", JIT_OPTIONS)
def test_every_matvec_launch_in_the_grid_compiles(std, flags):
    from inkling2_metal_compile import compile_unit, translation_unit

    _compile_or_skip()
    header, _ = lib.library("matvec")
    failures = []
    for kind, tiles, dtype, rows, n in _shape_grid_plans():
        body, _groups = lib.matvec_source(kind, tiles, dtype, rows, n)
        k = 2304
        unit = translation_unit(
            "lk_matvec", [("x", dtype, rows * k), ("w", dtype, n * k), ("sizes", mx.int32, 2),
                          ("unit", mx.float32, 1)], [("y", dtype)], body, header)
        err = compile_unit(unit, std, flags)
        if err is not None:
            failures.append((kind, tiles, str(dtype), rows, err[-600:]))
    assert not failures, failures[:3]


# ----- MLX 0.32.4 routing (spec 4, open question 1) --------------------------

def test_gemv_tiles_follow_gemv_axbpy():
    bf = mx.bfloat16
    assert lib.matvec_plan(1, 4096, 1024, bf) == ("gemv", (8, 1, 1, 32, 4, 4))
    assert lib.matvec_plan(1, 4095, 1024, bf) == ("gemv", (4, 1, 1, 32, 4, 4))
    assert lib.matvec_plan(1, 512, 64, bf) == ("gemv", (1, 1, 8, 4, 4, 4))     # K <= 64
    assert lib.matvec_plan(1, 512, 65, bf) == ("gemv", (4, 1, 1, 32, 4, 4))
    assert lib.matvec_plan(1, 512, 8192, bf) == ("gemv", (1, 8, 1, 32, 4, 4))  # K >= 16 N
    assert lib.matvec_plan(1, 3, 1024, bf) == ("gemv", (1, 8, 1, 32, 1, 4))    # N < 4: TM 1
    assert lib.matvec_plan(1, 4, 1024, bf) == ("gemv", (1, 8, 1, 32, 4, 4))


def test_wide_config_follows_gemv_wide_config():
    bf = mx.bfloat16
    # passes = ceil(M / 5); k_lanes 32 for one pass or N <= 64, else 16;
    # grid_x = passes, or 1 when N >= 65536.
    for rows, expect in [(2, (32, 1)), (5, (32, 1)), (6, (16, 2)), (10, (16, 2)),
                         (11, (16, 3)), (15, (16, 3))]:
        assert lib.matvec_plan(rows, 3072, 2304, bf, generation=15) == ("wide", expect)
    assert lib.matvec_plan(11, 64, 2304, bf, generation=15) == ("wide", (32, 3))
    assert lib.matvec_plan(11, 65, 2304, bf, generation=15) == ("wide", (16, 3))
    assert lib.matvec_plan(11, 65536, 2304, bf, generation=15) == ("wide", (16, 1))
    assert lib._wide_vecs(11) == 4 and lib._wide_vecs(5) == 5 and lib._wide_vecs(6) == 3
    assert lib.matvec_plan(4, 3072, 2304, bf, generation=14) == (None, "gpu_generation")
    assert lib.matvec_plan(4, 3072, 2304, mx.float16, generation=15)[0] == "wide"


def test_logical_groups_and_threadgroup_memory():
    assert lib.logical_groups("gemv", (4, 1, 1, 32, 4, 4), 3072) == (192, 2, 128)
    assert lib.logical_groups("gemv", (8, 1, 1, 32, 4, 4), 8704) == (272, 1, 256)
    assert lib.logical_groups("gemv", (1, 8, 1, 32, 4, 4), 24) == (6, 1, 256)
    assert lib.logical_groups("gemv", (1, 1, 8, 4, 1, 4), 2) == (1, 8, 32)
    assert lib.logical_groups("wide", (16, 2), 3072) == (768, 4, 64)
    assert lib.logical_groups("wide", (32, 1), 24) == (6, 2, 128)
    # GEMVKernel::tgp_mem_size = BN * (blockM + TM) when BN > 1.
    assert lib.tgp_floats("gemv", (1, 8, 1, 32, 4, 4)) == 8 * (4 + 4)
    assert lib.tgp_floats("gemv", (4, 1, 1, 32, 4, 4)) == 1
    assert lib.tgp_floats("wide", (32, 1)) == 1


def test_gpu_generation_parses_like_mlx(monkeypatch):
    monkeypatch.setattr(lib, "_generation", None)
    monkeypatch.setenv("MLX_METAL_GPU_ARCH", "applegpu_g15s")
    assert lib.gpu_generation() == 15
    monkeypatch.setattr(lib, "_generation", None)
    monkeypatch.setenv("MLX_METAL_GPU_ARCH", "applegpu_g17d")
    assert lib.gpu_generation() == 17
    monkeypatch.setattr(lib, "_generation", None)


def test_matvec_call_maps_coordinates():
    call = lib.matvec_call("gemv", (1, 8, 1, 32, 4, 4), dtype="bfloat16_t", rows=1, mat="w",
                           vec="x", out="y", n_ref="sizes[1]", logical="logical", lsg="sg % 8")
    assert "GEMVKernel<bfloat16_t, 1, 8, 1, 32, 4, 4, false>::run(" in call
    assert "uint(sg % 8) % 8u, uint(sg % 8) / 8u" in call  # lid = (lane, s % BN, s / BN)
    assert "tgp + (simdgroup_index_in_threadgroup / 8u) * 64u" in call
    call = lib.matvec_call("wide", (16, 2), dtype="half", rows=7, mat="w", vec="x", out="y",
                           n_ref="sizes[1]", logical="logical", lsg="sg % 2")
    assert "GemvWide<half, 4, 16>::run(" in call and "uint3(0u, uint(logical), 0u)" in call
    with pytest.raises(ValueError):
        lib.matvec_call("gemv", (4, 1, 1, 32, 4, 4), dtype="half", rows=2, mat="w", vec="x",
                        out="y", n_ref="n", logical="l", lsg="s")


# ----- L4: branch declines ----------------------------------------------------

def test_unmirrored_shapes_decline_before_launch():
    bf = mx.bfloat16
    assert lib.matvec_plan(1, 1, 2304, bf) == (None, "outside_rules")    # dot product
    assert lib.matvec_plan(1, 512, 2304, bf, transposed=True) == (None, "outside_rules")
    assert lib.matvec_plan(1, 512, 4, bf) == (None, "geometry")
    assert lib.matvec_plan(1, 512, 2304, mx.int32) == (None, "dtype")
    assert lib.matvec_plan(3, 1, 2304, bf, generation=17) == (None, "outside_rules")
    assert lib.matvec_plan(16, 512, 2304, bf, generation=17) == (None, "rows")
    assert lib.matvec_plan(2, 512, 2304, mx.float32, generation=17) == (None, "rows_dtype")
    with mx.stream(mx.cpu):
        assert lib.plan_matvec(mx.zeros((1, 64)), mx.zeros((8, 64))) == "not_gpu"


# ----- L2: resolver robustness ------------------------------------------------

@pytest.fixture()
def tree(tmp_path):
    root = tmp_path / "include"
    shutil.copytree(lib.include_root() / KERNELS, root / KERNELS)
    return root


def _edit(root, rel, old, new):
    path = root / KERNELS / rel
    text = path.read_text()
    assert old in text, (rel, old)
    path.write_text(text.replace(old, new, 1))


def test_copy_of_tree_resolves_identically(tree):
    assert lib.library("matvec", root=tree) == lib.library("matvec")


def test_missing_header_refuses(tree):
    (tree / KERNELS / "steel/utils.h").unlink()
    assert lib.library("matvec", root=tree) == (None, "header_missing:steel/utils.h")
    (tree / KERNELS / "utils.h").unlink()
    assert lib.library("elementwise", root=tree) == (None, "header_missing:utils.h")


def test_renamed_declaration_refuses(tree):
    _edit(tree, "gemv.h", "struct GemvWide {", "struct GemvWider {")
    assert lib.library("matvec", root=tree) == (None, "header_changed:matvec:struct GemvWide")
    _edit(tree, "unary_ops.h", "struct Sigmoid {", "struct Sigmoidal {")
    assert lib.library("elementwise", root=tree) == (
        None, "header_changed:elementwise:struct Sigmoid")


def test_changed_run_signature_refuses(tree):
    _edit(tree, "gemv.h", "      int bias_fd,\n", "      int bias_fd,\n      int extra,\n")
    assert lib.library("matvec", root=tree) == (None, "header_changed:matvec:GemvWide::run(")


def test_changed_template_or_tile_rule_refuses(tree):
    _edit(tree, "gemv.h", "BN > 1 ? BN*(blockM + TM) : 0", "BN > 1 ? BN*(blockM + 2*TM) : 0")
    why = lib.library("matvec", root=tree)[1]
    assert why.startswith("header_changed:matvec:MLX_MTL_CONST short tgp_mem_size"), why


def test_function_constant_drift_refuses(tree):
    _edit(tree, "gemv.h", "constant bool gemv_wide_has_batch [[function_constant(0)]];\n", "")
    assert lib.library("matvec", root=tree) == (None, "header_changed:matvec:function_constant")


def test_include_cycle_refuses(tree):
    _edit(tree, "expm1f.h", "#include <metal_math>",
          '#include <metal_math>\n#include "mlx/backend/metal/kernels/erf.h"')
    assert lib.library("elementwise", root=tree) == (None, "header_cycle:erf.h")


@pytest.mark.parametrize("include", ['"other/thing.h"', '"mlx/backend/metal/kernels/../x.h"'])
def test_include_outside_tree_refuses(tree, include):
    _edit(tree, "binary_ops.h", "#include <metal_math>", f"#include <metal_math>\n#include {include}")
    assert lib.library("elementwise", root=tree) == (None, f"header_outside:{include[1:-1]}")


def test_unreadable_header_refuses(tree):
    (tree / KERNELS / "fp8.h").write_bytes(b"\xff\xfe\x00bad")
    assert lib.library("elementwise", root=tree) == (None, "header_unreadable:fp8.h")


def test_unknown_part_and_switch():
    assert lib.library("nope") == (None, "no library part 'nope'")
    lib.set_enabled(False)
    try:
        assert lib.library("matvec") == (None, "disabled")
        box = Box()
        out, engaged = lib.run_proved(lib.Proofs("t_off"), "k", box, "c", lambda: 1 / 0,
                                      lambda: "ref", lambda f, r: [])
        assert (out, engaged) == ("ref", False) and box.c == {"unproven": 1}
    finally:
        lib.set_enabled(None)
    assert lib.library("matvec")[1] is None


def test_switch_reads_environment(monkeypatch):
    monkeypatch.setenv(lib.SWITCH_ENV, "0")
    assert not lib.enabled()
    monkeypatch.setenv(lib.SWITCH_ENV, "1")
    assert lib.enabled()


# ----- L5: proof protocol with stand-in launches -----------------------------

def _launch(values, calls):
    def run():
        calls.append(1)
        return mx.array(values, mx.float32)
    return run


def test_mismatched_launch_is_refused_and_never_runs_again():
    proofs, box, calls = lib.Proofs("t_bits"), Box(), []
    ref = mx.array([1.0, 2.0])
    out, engaged = lib.run_proved(proofs, "k", box, "c", _launch([1.0, 2.5], calls),
                                  lambda: ref, lambda f, r: [(f, r)])
    assert out is ref and not engaged and box.c == {"refused_first": 1}
    for _ in range(3):
        out, engaged = lib.run_proved(proofs, "k", box, "c", _launch([1.0, 2.5], calls),
                                      lambda: ref, lambda f, r: [(f, r)])
        assert out is ref and not engaged
    assert len(calls) == 1 and box.c == {"refused_first": 1, "refused": 3}
    assert proofs.receipt()["refused"] == {"'k'": "bits:differs0"}


def test_proved_once_then_served():
    proofs, box, calls = lib.Proofs("t_ok"), Box(), []
    ref = mx.array([1.0, -0.0])
    out, engaged = lib.run_proved(proofs, "k", box, "c", _launch([1.0, -0.0], calls),
                                  lambda: ref, lambda f, r: [(f, r)])
    assert out is ref and not engaged
    for _ in range(4):
        out, engaged = lib.run_proved(proofs, "k", box, "c", _launch([1.0, -0.0], calls),
                                      lambda: 1 / 0, lambda f, r: [(f, r)])
        assert engaged
    assert box.c == {"proved": 1, "calls": 4} and len(calls) == 5
    assert proofs.receipt()["served"] == 4 and proofs.receipt()["proved"] == 1


def test_launch_errors_and_shape_or_dtype_differences_refuse():
    proofs, box = lib.Proofs("t_err"), Box()
    ref = mx.array([1.0, 2.0])

    def boom():
        raise RuntimeError("pipeline")

    lib.run_proved(proofs, "e", box, "c", boom, lambda: ref, lambda f, r: [(f, r)])
    lib.run_proved(proofs, "s", box, "c", lambda: mx.array([1.0]), lambda: ref,
                   lambda f, r: [(f, r)])
    lib.run_proved(proofs, "d", box, "c", lambda: ref.astype(mx.float16), lambda: ref,
                   lambda f, r: [(f, r)])
    assert proofs.receipt()["refused"] == {"'e'": "error:RuntimeError", "'s'": "bits:shape0",
                                           "'d'": "bits:dtype0"}
    assert box.c == {"refused_first": 3}


def test_first_verdict_wins_and_a_conflict_refuses():
    proofs = lib.Proofs("t_race")
    assert proofs.settle("k", None) == lib.PROVEN
    assert proofs.settle("k", None) == lib.PROVEN
    assert proofs.settle("k", "bits:differs0") == lib.REFUSED
    assert proofs.settle("k", None) == lib.REFUSED
    other = lib.Proofs("t_race2")
    assert other.settle("j", "bits:differs0") == lib.REFUSED
    assert other.settle("j", None) == lib.REFUSED


def test_concurrent_first_uses_settle_consistently():
    proofs, box = lib.Proofs("t_threads"), Box()
    ref = mx.array([3.0])
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        lib.run_proved(proofs, "k", box, "c", lambda: mx.array([3.0]), lambda: ref,
                       lambda f, r: [(f, r)])

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert proofs.state("k") == lib.PROVEN
    assert sum(box.c.values()) == 8 and set(box.c) <= {"proved", "calls"}


def test_nothing_is_persisted_across_processes():
    # L7: verdicts live in process memory only; a new Proofs starts unknown.
    proofs = lib.Proofs("t_persist")
    proofs.settle("k", None)
    assert lib.Proofs("t_persist").state("k") is None
    assert "t_persist" in lib.receipt()["proofs"]


def test_bits_views_and_signed_zero():
    for dtype, view in [(mx.float32, mx.uint32), (mx.float16, mx.uint16),
                        (mx.bfloat16, mx.uint16)]:
        assert lib.bits(mx.zeros((2,), dtype)).dtype == view
    assert lib.same_bits([(mx.array([0.0]), mx.array([-0.0]))]) == "differs0"
    nan = mx.array([float("nan")])
    assert lib.same_bits([(nan, nan)]) is None
    assert lib.same_bits([(None, nan)]) == "missing0"


def test_count_uses_the_instance_dict_of_modules():
    from mlx import nn

    module = nn.Module()
    lib.count(module, "x_counts", "calls")
    lib.count(module, "x_counts", "calls", 2)
    assert module.__dict__["x_counts"] == {"calls": 3}
    assert "x_counts" not in module.keys()


def test_metal_type_spellings_match_mlx_instantiations():
    # gemv.metal instantiates float32 -> float, float16 -> half, bfloat16 ->
    # bfloat16_t (bf16.h: typedef bfloat bfloat16_t).
    assert [lib.metal_type(d) for d in DTYPES] == ["float", "half", "bfloat16_t"]
    with pytest.raises(ValueError):
        lib.metal_type(mx.int32)
    text = (lib.include_root() / KERNELS / "bf16.h").read_text()
    assert "typedef bfloat bfloat16_t;" in text


def test_no_mlx_header_text_is_committed():
    # R1: the module reads headers at run time; none is vendored.
    src = Path(lib.__file__).read_text()
    assert "struct GEMVKernel {" not in src and "Copyright © 2023" not in src
