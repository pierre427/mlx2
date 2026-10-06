"""Run directly with unittest: source/admission guards without importing MLX.

This is host evidence, not CPU tensor or Metal parity. The ordinary pytest
suite test_qwen36_decode_wins.py covers tensor dispatch with CPU MLX available.
"""

import ast
import hashlib
import os
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(os.environ.get("MLX2_SOURCE_ROOT", Path(__file__).resolve().parents[1]))
MODELS = ROOT / "src/mlx2/runtime/models"


def tree(name):
    return ast.parse((MODELS / name).read_text())


def literal(name, var):
    for node in tree(name).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == var for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise KeyError(var)


def function(name, fn, env):
    node = next(
        n for n in tree(name).body if isinstance(n, ast.FunctionDef) and n.name == fn
    )
    node.decorator_list = []
    # Replace one internal import with its already extracted source binding.
    node.body = [n for n in node.body if not isinstance(n, ast.ImportFrom)]
    mod = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(mod)
    exec(compile(mod, str(MODELS / name), "exec"), env)
    return env[fn]


@dataclass
class Admission:
    accepted: bool
    reason: str


def verify_admit(rows, hv=32, architecture="qwen35", **overrides):
    mx = SimpleNamespace(bfloat16="bf16", float32="f32", float16="f16")
    env = dict(
        Any=object,
        FusedGdnAdmission=Admission,
        mx=mx,
        NUM_KEY_HEADS=16,
        NUM_VALUE_HEADS=48,
        KEY_HEAD_DIM=128,
        VALUE_HEAD_DIM=128,
        CONV_KERNEL=4,
        CONV_DIM=10240,
        VALUE_DIM=6144,
        BATCH_VERIFY_MAX_ROWS=16,
        MAX_VERIFY_STEPS=8,
        _shape=lambda a: a.shape,
        _dtype=lambda a: a.dtype,
        _slab_width=lambda a: a.shape[1],
    )
    function("qwen4_fused_gdn.py", "admit_rollback_span", env)
    function("qwen4_fused_gdn_verify.py", "batch_verify_row_steps", env)
    fn = function(
        "qwen4_fused_gdn_verify.py",
        "admit_qwen4_fused_gdn_verify"
        if rows == 1
        else "admit_qwen4_fused_gdn_batch_verify",
        env,
    )

    def tensor(shape, dtype="bf16"):
        return SimpleNamespace(shape=shape, dtype=dtype)

    vd = hv * 128
    cd = 4096 + vd
    steps = 3
    kw = dict(
        qkv=tensor((rows, steps, cd)),
        z=tensor((rows, steps, vd)),
        a=tensor((rows, steps, hv)),
        b=tensor((rows, steps, hv)),
        conv_state=tensor((rows, 3, cd)),
        recurrent_state=tensor((rows, hv, 128, 128), "f32"),
        conv_weight=tensor((cd, 4, 1)),
        A_log=tensor((hv,), "f32"),
        dt_bias=tensor((hv,)),
        norm_weight=tensor((128,)),
        mask=None,
        spans=(),
        speculating=True,
        training=False,
        sharded=False,
        num_key_heads=16,
        num_value_heads=hv,
        key_head_dim=128,
        value_head_dim=128,
        conv_kernel=4,
        gate_activation=(
            "swish" if architecture in ("qwen35", "qwen38") else "sigmoid"
        ),
        architecture=architecture,
    )
    kw.update(overrides)
    if architecture == "qwen4" and "architecture" not in fn.__code__.co_varnames:
        kw.pop("architecture")
    return fn(**kw)


class Sources(unittest.TestCase):
    def test_qwen35_verify_admission(self):
        for rows in [1, 4, 16]:
            self.assertTrue(verify_admit(rows).accepted)

    def test_qwen38_verify_admission(self):
        for rows in [1]:
            self.assertTrue(verify_admit(rows, 48, "qwen38").accepted)

    def test_flash_verify_admission_unchanged(self):
        for rows in [1, 4, 16]:
            self.assertTrue(verify_admit(rows, 48, "qwen4").accepted)

    def test_geometry_cannot_switch_without_numerics(self):
        self.assertIn("geometry", verify_admit(4, 32, "qwen4").reason)
        self.assertIn("geometry", verify_admit(1, 48, "qwen35").reason)
        self.assertIn("geometry", verify_admit(1, 32, "qwen38").reason)
        self.assertIn("output gate", verify_admit(4, gate_activation="sigmoid").reason)

    def test_mask_refusal(self):
        self.assertIn("masked", verify_admit(4, mask=object()).reason)
        self.assertIn("training", verify_admit(1, training=True).reason)

    def test_qwen35_source_has_float_qk_beta_and_swish(self):
        source = literal("qwen4_fused_gdn_verify.py", "_SOURCE")
        env = {"decode_source": literal("qwen4_fused_gdn.py", "_SOURCE")}
        fn = function("qwen4_fused_gdn_verify.py", "_qwen35_verify_source", env)
        out = fn(source)
        self.assertIn("threadgroup float sq_squared", out)
        self.assertIn(
            "shr[3] = mlx_sigmoid_precise<float>(float(b[t * HV + hv]));", out
        )
        self.assertIn("mlx_sigmoid_fast<float>(float(z[t * VD + hv * DV + d]))", out)
        self.assertEqual(out.count("device float* state_dst"), 1)
        self.assertEqual(out.count("AGNES_NUMERICS"), 3)
        self.assertEqual(out.count("for (uint t = 0; t < (uint)S; ++t)"), 1)

    def test_batch_source_derivation_keeps_lane_width_and_snapshots(self):
        source = literal("qwen4_fused_gdn_verify.py", "_SOURCE")
        env = {"decode_source": literal("qwen4_fused_gdn.py", "_SOURCE")}
        source = function("qwen4_fused_gdn_verify.py", "_qwen35_verify_source", env)(source)
        for name in ("_LANE_STRIDES", "_VERIFY_LANE_BUFFERS", "_PADDED_ROWS_EPILOGUE"):
            env[name] = literal("qwen4_fused_gdn_verify.py", name)
        function("gdn_state.py", "derive_source", env)
        function("qwen4_fused_gdn_verify.py", "_lane_prefix", env)
        derive = function("qwen4_fused_gdn_verify.py", "_derive_batch_verify_source", env)
        out = derive(source, env["_VERIFY_LANE_BUFFERS"], final_store=0, what="host qwen35")
        self.assertIn("const uint SNAPS = RS - 1u", out)
        self.assertIn("for (uint t = 0; t < RS", out)
        self.assertIn("output[t * VD + hv * DV + d] = static_cast<T>(0)", out)
        self.assertEqual(out.count("{"), out.count("}"))

    def test_stock_down_window_source(self):
        env = {
            "RD": SimpleNamespace(
                CANDIDATE_DOWN_SOURCE=literal(
                    "qwen4_routed_decode.py", "CANDIDATE_DOWN_SOURCE"
                )
            )
        }
        fn = function("qwen36_moe_decode.py", "down_source", env)
        for shared in [False, True]:
            out = fn(shared)
            self.assertEqual(
                out.count("threadgroup_barrier(mem_flags::mem_threadgroup)"), 1
            )
            self.assertIn("lane[j % 8] = part[j * RPS + row] + lane[j % 8]", out)
            self.assertIn("const auto rhs_row = rhs + (size_t)token * (TOPK)", out)
            self.assertEqual(out.count("{"), out.count("}"))
        self.assertIn("shared_qmv::qmv_rows", fn(True))
        self.assertIn("acc + weighted_shared", fn(True))

    def test_affine_headers_isolate_shared_group_size(self):
        env = {"QMV_HEADER": literal("qwen4_routed_decode.py", "QMV_HEADER"),
               "QMV_ROWS": literal("qwen4_routed_decode.py", "QMV_ROWS"), "GROUP_SIZE": 64}
        fn = function("qwen4_routed_decode.py", "_format_header", env)
        for bits in (4, 8):
            header = fn("shared", bits, True, 128)
            self.assertIn("constant constexpr int GS = 128", header)
            self.assertIn(f"constant constexpr int BITS = {bits}", header)
            self.assertEqual(header.count("{"), header.count("}"))
        self.assertIn("constant constexpr int GS = 64", fn("routed", 4, True))

    def test_router_launch_specialization(self):
        header = literal("qwen4_moe_window.py", "ROUTER_HEADER")
        source = literal("qwen4_moe_window.py", "ROUTER_TOPK_SOURCE")
        self.assertIn("int NK = 10", header)
        self.assertIn("mlx2_router_select<NE, TOPK>", source)
        self.assertIn("mlx2_router_score<T, TOPK>", source)
        self.assertNotIn("sel[10]", source)

    def test_adapter_explicit_only_choices(self):
        t = ast.parse((ROOT / "src/mlx2/adapters/qwen36_35b.py").read_text())
        env = {
            "os": os,
            "PROCESS_NUMERICS": {},
            "require_process_numerics": lambda _owner: None,
        }
        names = {
            "KERNEL_POLICY_ENV",
            "NEW_DECODE_KERNELS",
            "ENUM_KERNELS",
            "EXPLICIT_ONLY_KERNELS",
        }
        body = [
            n
            for n in t.body
            if isinstance(n, ast.Assign)
            and any(isinstance(x, ast.Name) and x.id in names for x in n.targets)
            or isinstance(n, ast.FunctionDef)
            and n.name
            in ("configure_environment", "validate_kernel_choice", "choice_selected")
        ]
        exec(compile(ast.Module(body=body, type_ignores=[]), "adapter", "exec"), env)
        saved = dict(os.environ)
        try:
            for key in env["NEW_DECODE_KERNELS"]:
                os.environ[env["KERNEL_POLICY_ENV"][key]] = "1"
            profile = env["configure_environment"]()
            self.assertTrue(
                all(
                    env["KERNEL_POLICY_ENV"][key] not in profile
                    for key in env["NEW_DECODE_KERNELS"]
                )
            )
            profile = env["configure_environment"](
                {"moe_topk_fold": "launch", "moe_routed_decode": "gate_up"}
            )
            self.assertEqual(profile["MLX_QWEN4_MOE_TOPK_FOLD"], "launch")
            self.assertEqual(profile["MLX_QWEN36_DECODE_WINS"], "1")
            with self.assertRaises(ValueError):
                env["configure_environment"]({"moe_topk_fold": "fold"})
            with self.assertRaises(ValueError):
                env["configure_environment"]({"fused_gdn_verify": "1"})
        finally:
            os.environ.clear()
            os.environ.update(saved)

    def test_flash_source_digests(self):
        # Pinned before this port; Metal source bodies must not change.
        for filename, var, digest in PINS:
            self.assertEqual(
                hashlib.sha256(literal(filename, var).encode()).hexdigest(),
                digest,
                (filename, var),
            )

    def test_gpu_scripts_refuse_before_mlx_import(self):
        import subprocess
        import sys

        for script in ["check_qwen36_decode_wins.py", "bench_qwen36_decode_wins.py"]:
            args = [
                sys.executable,
                str(ROOT / "scripts" / script),
                "--model",
                "absent",
                "--out",
                "/dev/null",
            ]
            if script.startswith("bench"):
                args += ["--prompt-file", "absent"]
            result = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("refusing Metal", result.stderr)


PINS = [
    (
        "qwen4_fused_gdn.py",
        "_SOURCE",
        "f6d1166e26c2c6506e380d7253a43f218b5c099e048ff7aa95ef7648bb4d0e60",
    ),
    (
        "qwen4_fused_gdn_verify.py",
        "_SOURCE",
        "2d5d84dc1869b7d74115605f2391e6a8e0767916db4f2214df19321641c90842",
    ),
    (
        "qwen4_routed_decode.py",
        "SPLIT_GATE_UP_SOURCE",
        "e89d77adca163bebdad74cef7e51a58c63f27ab0995948fb67a4212ae92efbf9",
    ),
    (
        "qwen4_routed_decode.py",
        "SERVED_DOWN_SOURCE",
        "2d78fa3b810347aa2ccb834d8e8eabb4fe86566fdafe02397133c7a86f6fddcf",
    ),
    (
        "qwen4_routed_decode.py",
        "SHARED_GATE_UP_SOURCE",
        "ec12d932d651b9c93a7fa2d7c7d1aeceb0915c44b7fc5c01bb08b3f002cb6e97",
    ),
    (
        "qwen4_routed_decode.py",
        "SHARED_DOWN_SOURCE",
        "550ba6a11b78ed223ff3deaf2dcfccae42ef7a50ea25c607da46944d6dbb4471",
    ),
    (
        "qwen4_routed_decode.py",
        "CANDIDATE_DOWN_SOURCE",
        "527d58a742a06c31ad85a5e46764e532f89f295724fed6405a9cee93bed67032",
    ),
]

if __name__ == "__main__":
    unittest.main()
