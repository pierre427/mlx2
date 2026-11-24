"""MTPLX launch-count ingest for Inkling 2 (2026-11-23), CPU only.

Switches: ``inkling2_pass_messages``, ``inkling2_stage_a_kernel``,
``inkling2_gate_merge``, ``inkling2_attn_output_kernel``,
``inkling2_embed_kernel``. All default off. Each one proves its launch
bitwise against the composed path on the first call of a shape and keeps the
composed path otherwise, so outputs equal the option-off route with or
without engagement. On the CPU the GPU launches decline (counted); the pass
message read is MLX ops and runs (and proves) here. The generated Metal
sources are compiled offline (``xcrun metal``, no GPU) when the compiler is
installed. The GPU parity checks are ``tests/test_inkling2_mtplx_launch_metal.py``.
"""

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from mlx2.runtime.models.inkling2 import library_kernels as lib
from test_inkling2_cpu import _tiny_model

NEW = {"pass_messages": "inkling2_pass_messages", "stage_a_kernel": "inkling2_stage_a_kernel",
       "gate_merge": "inkling2_gate_merge", "attn_output_kernel": "inkling2_attn_output_kernel",
       "embed_kernel": "inkling2_embed_kernel"}
GPU_ONLY = ("stage_a_kernel", "gate_merge", "attn_output_kernel", "embed_kernel")


@pytest.fixture(autouse=True)
def _cpu():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


def _defaults(model):
    import inspect

    from mlx2.adapters.inkling2 import KERNEL_DEFAULTS

    params = inspect.signature(model.configure_kernels).parameters
    return {k: (True if v is None else v) for k, v in KERNEL_DEFAULTS.items()
            if k in params and not k.startswith("moe_")}


def _model(bf16=False, **options):
    model = _tiny_model()
    if bf16:
        # CPU GatherMM is FP32-only: routed experts stay FP32.
        model.load_weights([(k, v if ".switch_mlp." in k else v.astype(mx.bfloat16))
                            for k, v in tree_flatten(model.parameters())])
        mx.eval(model.parameters())
    model.configure_kernels(**{**_defaults(model), **options})
    return model


def _forward(model, tokens, plan):
    cache = model.make_cache()
    outs = []
    for a, b in plan:
        outs.append(model(tokens[:, a:b], cache=cache))
        mx.eval(outs[-1])
    return mx.concatenate(outs, axis=1)


# Prefill, one-token decode, then 1..8-row verify blocks.
PLAN = [(0, 12), (12, 13), (13, 14)] + [(14 + sum(range(n)), 14 + sum(range(n + 1)))
                                         for n in range(1, 9)]


def test_new_switches_default_off():
    from mlx2.adapters.inkling2 import KERNEL_DEFAULTS, KERNEL_OPTIONS, validate_policy

    kernels, _rest = validate_policy({})
    for name, key in NEW.items():
        assert KERNEL_OPTIONS[key] == name
        assert KERNEL_DEFAULTS.get(name) is None
        assert kernels[name] is False
    model = _tiny_model()
    model.configure_kernels()
    for name in NEW:
        assert model.kernel_policy[name] is False
    attn = model.model.layers[0].self_attn
    assert not attn.__dict__.get("_stage_a") and not attn.__dict__.get("_gate_merge")
    assert not attn.__dict__.get("_defer_o_proj")
    assert not model.model.__dict__.get("_pass_messages")
    assert not model.model.__dict__.get("_embed_kernel")


def test_policy_dependencies():
    from mlx2.adapters.inkling2 import validate_policy

    with pytest.raises(ValueError, match="gate_merge requires"):
        validate_policy({"inkling2_gate_merge": True, "inkling2_verify_splitk": False})
    with pytest.raises(ValueError, match="attn_output_kernel requires"):
        validate_policy({"inkling2_attn_output_kernel": True, "inkling2_fuse_sconv_kernel": False})
    with pytest.raises(TypeError):
        validate_policy({"inkling2_pass_messages": 1})
    kernels, _ = validate_policy({name: True for name in NEW.values()})
    assert all(kernels[name] for name in NEW)
    model = _tiny_model()
    with pytest.raises(ValueError, match="gate_merge requires"):
        model.configure_kernels(gate_merge=True)
    with pytest.raises(ValueError, match="attn_output_kernel requires"):
        model.configure_kernels(attn_output_kernel=True, fuse_sconv=True)


@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("rows", [1, 2])
def test_all_switches_on_are_bitwise_off(rows, bf16):
    mx.random.seed(7)
    tokens = mx.random.randint(0, 500, (rows, PLAN[-1][1]))
    ref = _forward(_model(bf16), tokens, PLAN)
    model = _model(bf16, **{name: True for name in NEW})
    out = _forward(model, tokens, PLAN)
    assert out.dtype == ref.dtype
    assert mx.array_equal(lib.bits(out), lib.bits(ref)).item()
    receipt = model.kernel_receipt()
    assert receipt["stage_a"].get("declined:not_gpu", 0) > 0
    assert receipt["attn_output"].get("declined:not_gpu", 0) > 0
    assert receipt["embed_kernel"].get("declined:not_gpu", 0) == len(PLAN)
    assert receipt["gate_merge"] == {}  # split-K itself is GPU only
    # The message read is decided per shape here (proved or refused, never silent).
    decided = receipt["pass_messages"]
    assert sum(decided.get(k, 0) for k in ("proved", "refused_first")) > 0, decided


@pytest.mark.parametrize("bf16", [False, True])
@pytest.mark.parametrize("rows", [1, 2])
def test_pass_messages_alone_engages_and_is_bitwise(rows, bf16):
    from mlx2.runtime.models.inkling2 import pass_messages

    pass_messages.PROOFS.reset()
    mx.random.seed(11)
    tokens = mx.random.randint(0, 500, (rows, PLAN[-1][1]))
    ref = _forward(_model(bf16), tokens, PLAN)
    model = _model(bf16, pass_messages=True)
    out = _forward(model, tokens, PLAN)
    assert mx.array_equal(lib.bits(out), lib.bits(ref)).item()
    counts = model.kernel_receipt()["pass_messages"]
    # 11 forwards x 2 loop passes; every distinct (B, L) shape proves once.
    shapes = len({b - a for a, b in PLAN})
    assert counts.get("proved", 0) + counts.get("refused_first", 0) == shapes, counts
    if not bf16:
        # Observed on MLX 0.32.4 CPU: the 48-row product matches the 4-row ones.
        assert counts.get("proved") == shapes and counts.get("calls", 0) > 0, counts


def test_pass_messages_declines_on_residual_kernel_list():
    from mlx2.runtime.models.inkling2 import pass_messages

    model = _tiny_model()
    bank = [mx.zeros((128,))]
    assert pass_messages.rows(model.model, bank, model.model.message_queries, 128) is None
    assert model.model.pass_messages_counts == {"declined:residual_kernel_messages": 1}


def test_run_proved_contract():
    proofs = lib.Proofs("t")

    class Box:
        pass

    box = Box()
    a, b = mx.array([1.0, 2.0]), mx.array([1.0, 2.0])
    out, engaged = lib.run_proved(proofs, "k", box, "c", lambda: a + 0, lambda: b * 1,
                                  lambda f, r: [(f, r)])
    assert not engaged and box.c == {"proved": 1}
    out, engaged = lib.run_proved(proofs, "k", box, "c", lambda: a + 0, lambda: 1 / 0,
                                  lambda f, r: [(f, r)])
    assert engaged and box.c["calls"] == 1
    # A differing launch is refused and the composed result returned, now and later.
    ref = mx.array([1.0, 3.0])
    out, engaged = lib.run_proved(proofs, "k2", box, "c", lambda: a, lambda: ref,
                                  lambda f, r: [(f, r)])
    assert not engaged and out is ref and box.c["refused_first"] == 1
    out, engaged = lib.run_proved(proofs, "k2", box, "c", lambda: 1 / 0, lambda: ref,
                                  lambda f, r: [(f, r)])
    assert not engaged and out is ref and box.c["refused"] == 1
    assert proofs.receipt()["proved"] == 1 and len(proofs.receipt()["refused"]) == 1
    # Signed zero differs bitwise even though it compares equal.
    assert lib.same_bits([(mx.array([0.0]), mx.array([-0.0]))]) == "differs0"


def test_matvec_plan_matches_mlx_dispatch_for_the_mini_shapes():
    bf = mx.bfloat16
    # One row: GEMVKernel tiles (MLX 0.32.4 gemv_axbpy, non-transposed matrix).
    assert lib.matvec_plan(1, 3072, 2304, bf) == ("gemv", (4, 1, 1, 32, 4, 4))   # q, gate
    assert lib.matvec_plan(1, 384, 2304, bf) == ("gemv", (4, 1, 1, 32, 4, 4))    # r
    assert lib.matvec_plan(1, 24, 2304, bf) == ("gemv", (1, 8, 1, 32, 4, 4))     # sink
    assert lib.matvec_plan(1, 1024, 2304, bf) == ("gemv", (4, 1, 1, 32, 4, 4))   # k, v
    assert lib.matvec_plan(1, 8704, 2304, bf) == ("gemv", (8, 1, 1, 32, 4, 4))   # stacked
    assert lib.matvec_plan(1, 2304, 3072, bf) == ("gemv", (4, 1, 1, 32, 4, 4))   # o_proj
    assert lib.matvec_plan(1, 2, 64, bf) == ("gemv", (1, 1, 8, 4, 1, 4))         # tiny K
    # 2..15 rows of BF16 on M3+: GemvWide (gemv_wide_config).
    for rows in range(2, 6):
        assert lib.matvec_plan(rows, 3072, 2304, bf, generation=17) == ("wide", (32, 1))
    for rows in range(6, 9):
        assert lib.matvec_plan(rows, 3072, 2304, bf, generation=17) == ("wide", (16, 2))
        assert lib.matvec_plan(rows, 24, 2304, bf, generation=17) == ("wide", (32, 2))
    assert lib.matvec_plan(16, 3072, 2304, bf, generation=17) == (None, "rows")
    assert lib.matvec_plan(4, 3072, 2304, mx.float32, generation=17) == (None, "rows_dtype")
    assert lib.matvec_plan(4, 3072, 2304, bf, generation=14) == (None, "gpu_generation")
    assert lib.matvec_plan(4, 3072, 2306, bf, generation=17) == (None, "k_align")


def test_library_headers_are_read_and_patched():
    text, why = lib.library("matvec")
    assert why is None
    assert "struct GEMVKernel" in text and "struct GemvWide" in text
    assert "[[function_constant(" not in text.split("struct GemvWide")[0][-2000:]
    assert "constant constexpr bool gemv_wide_do_axpby = false;" in text
    assert '#include "mlx/' not in text
    text, why = lib.library("elementwise")
    assert why is None and "struct Sigmoid" in text
    assert lib.library("nope") == (None, "no library part 'nope'")


def test_stage_a_jobs_follow_the_composed_products():
    from mlx2.runtime.models.inkling2 import stage_a

    model = _tiny_model()
    attn = model.model.layers[5].self_attn
    x = mx.zeros((1, 1, 128))
    other = mx.ones((1, 1, 128))
    jobs = stage_a._jobs(attn, x, x, x)
    assert [list(s) for _i, _w, s in jobs] == [[n] for n in stage_a._NAMES]
    assert [i for i, _w, _s in jobs] == [0, 0, 0, 0, 1, 2]
    attn.fuse_projections()
    try:
        assert len(stage_a._jobs(attn, x, x, x)) == 1
        jobs = stage_a._jobs(attn, x, other, other)
        assert [sorted(s) for _i, _w, s in jobs] == [
            sorted(stage_a._NAMES[:4]), ["k_proj"], ["v_proj"]]
    finally:
        object.__setattr__(attn, "_fused_bounds", None)


def test_launch_probe_reports_pass_message_saving():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "research"))
    import inkling2_launch_count as probe

    result = probe.compare(_tiny_model, ["pass_messages", "embed_kernel"], base="defaults",
                           rows=1, length=1, device=None, prefill_gpu=False, warm_steps=1)
    assert result["pass_messages"]["delta"] < 0, result
    assert result["pass_messages"]["on"]["counters"]["pass_messages"].get("calls", 0) > 0
    assert result["embed_kernel"]["delta"] == 0  # GPU only: declined here
    assert result["embed_kernel"]["on"]["counters"]["embed_kernel"] == {"declined:not_gpu": 3}


# ----- offline Metal compile of every generated source (no GPU) -----------

def _compile_or_skip():
    from inkling2_metal_compile import compiler

    if compiler() is None:
        pytest.skip("no offline Metal compiler (xcrun metal)")


@pytest.mark.parametrize("rows,dtype", [(1, mx.bfloat16), (1, mx.float32), (3, mx.bfloat16),
                                        (8, mx.bfloat16)])
def test_stage_a_source_compiles(rows, dtype):
    from inkling2_metal_compile import compile_unit, translation_unit
    from mlx2.runtime.models.inkling2 import stage_a

    _compile_or_skip()
    header, _ = lib.library("matvec")
    k, ns = 2304, [3072, 384, 3072, 24, 1024, 1024]
    plans = []
    for j, n in enumerate(ns):
        kind, tiles = lib.matvec_plan(rows, n, k, dtype, generation=17)
        plans.append((kind, tiles, n, 0 if j < 4 else j - 3))
    source, _groups = stage_a._source(plans, dtype, rows)
    inputs = ([(f"x{i}", dtype, rows * k) for i in range(3)]
              + [(f"w{j}", dtype, n * k) for j, n in enumerate(ns)]
              + [("sizes", mx.int32, 7), ("unit", mx.float32, 1)])
    err = compile_unit(translation_unit("stage_a", inputs, [(f"y{j}", dtype) for j in range(6)],
                                        source, header))
    assert err is None, err


@pytest.mark.parametrize("b,length,dtype,tail", [(1, 1, mx.bfloat16, 2), (1, 1, mx.float32, 1),
                                                 (2, 1, mx.bfloat16, 2), (1, 8, mx.bfloat16, 1),
                                                 (2, 4, mx.bfloat16, 2)])
def test_attn_output_source_compiles(b, length, dtype, tail):
    from inkling2_metal_compile import compile_unit, translation_unit
    from mlx2.runtime.models.inkling2 import attn_output

    _compile_or_skip()
    header, _ = lib.library("matvec")
    c, k, rows = 2304, 3072, b * length
    kind, tiles = lib.matvec_plan(rows, c, k, dtype, generation=17)
    source, _groups = attn_output._source(kind, tiles, dtype, rows, c)
    inputs = [("g", dtype, rows * k), ("wo", dtype, c * k), ("sizes", mx.int32, 2),
              ("unit", mx.float32, 1), ("state", mx.float32, b * 3 * c),
              ("wt", mx.float32, c * 4), ("inv_arr", mx.float32, 1), ("xres", dtype, rows * c)]
    template = [("T", dtype), ("C", c), ("L", length), ("NB", b), ("TAIL", tail),
                ("HAS_STATE", True)]
    err = compile_unit(translation_unit(
        "attn_output", inputs, [("y", dtype), ("res", mx.float32), ("nstate", mx.float32)],
        source, header, template))
    assert err is None, err


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float32])
def test_gated_merge_and_embed_sources_compile(dtype):
    from inkling2_metal_compile import compile_unit, translation_unit
    from mlx2.runtime.models.inkling2 import frame_kernels, splitk

    _compile_or_skip()
    header, _ = lib.library("elementwise")
    source = splitk._MERGE_SOURCE.replace(splitk._PLAIN_STORE, splitk._GATED_STORE)
    assert source != splitk._MERGE_SOURCE
    inputs = [("part_o", mx.float32, 4096), ("part_m", mx.float32, 64),
              ("part_l", mx.float32, 64), ("sink", mx.float32, 24), ("gate", dtype, 3072)]
    err = compile_unit(translation_unit(
        "merge_gated", inputs, [("out", dtype)], source, header,
        [("T", dtype), ("D", 128), ("HQ", 24), ("L", 1), ("NS", 16)]))
    assert err is None, err
    inputs = [("ids", mx.int32, 1), ("table", dtype, 4096), ("w", dtype, 2304),
              ("dims", mx.uint32, 1), ("eps", mx.float32, 1)]
    err = compile_unit(translation_unit("embed", inputs, [("out", dtype)],
                                        frame_kernels._EMBED_SOURCE, "",
                                        [("T", dtype), ("NR", 4)]))
    assert err is None, err
