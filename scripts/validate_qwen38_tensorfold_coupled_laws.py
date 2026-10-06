#!/usr/bin/env python3
"""Host-only exact-law gate for TensorFold's Qwen3.8 dense/GDN kernels.

This checker does not import MLX and does not execute Metal.  It binds the
ordinary mlx-lm model sources and the pinned MLX Metal RMS implementation,
inspects the candidate TensorFold source, and rejects each known numerically
different implementation.  Passing this gate is necessary for a native parity
run; it is not route qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import subprocess
from pathlib import Path

EXPECTED_MLX_LM_REVISION = "1104ced19ed98800bdaf4ebcdca14bbdeb597c23"
REFERENCE_FILES = {
    "mlx_lm/models/gated_delta.py": (
        "87d592e63b9096e6bd0d81278428850576601c99fa06bf862facaafff2b7a062"
    ),
    "mlx_lm/models/qwen3_5.py": (
        "d2efbf2cf88a4ceb82050d8022c034d5a4bb2a96dc2b6d15401eb9815b1f9f2b"
    ),
    "mlx_lm/models/qwen3_next.py": (
        "87b65ee661c478e06d398195646f621608e37a917bb67037ee3627be322f7646"
    ),
    "mlx_lm/models/qwen4_fused_gdn.py": (
        "531c7b014ac03861667a1d5fd535659e600493dce302534e90d96944ea813b54"
    ),
}
EXPECTED_MLX_REVISION = "39400a0d4cf1641bc6e543cb5e3585757f491854"
MLX_REFERENCE_FILES = {
    "mlx/backend/metal/kernels/defines.h": (
        "a2930dbd644c69c4b66a511a094034217f3c03f48e29a1613f601532150f9163"
    ),
    "mlx/backend/metal/kernels/rms_norm.metal": (
        "b2e04e377fdad1d645581f9beeaf9cbb06d1ad32926161e06cbc15240caf12bf"
    ),
    "mlx/backend/metal/normalization.cpp": (
        "01bc537389638efe43a9bedeeb2008484864772022ac56e106a3f9911bcaa1aa"
    ),
}
KERNEL_ROOT = Path("src/tensorfold/kernels/qwen/dense/v1")
KERNEL_FILES = ("lane_glue.py", "lane_tree.py", "stream_gdn.py", "row_glue.py")


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=root, text=True, stderr=subprocess.STDOUT
    ).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pattern(*parts: str) -> str:
    return "".join(parts)


def _assignment_body(source: str, variable: str) -> str:
    """Return one kernel-source assignment through the next module constant."""

    match = re.search(
        rf"^{re.escape(variable)}\s*=.*?(?=^[A-Z_][A-Z0-9_]*(?:\s*:[^=]+)?\s*=|\Z)",
        source,
        re.DOTALL | re.MULTILINE,
    )
    return match.group(0) if match else ""


def _f32(value: float) -> float:
    return struct.unpack(">f", struct.pack(">f", value))[0]


def _bf16(value: float) -> float:
    """Round a finite host value to IEEE bfloat16, returned as float32."""

    bits = int.from_bytes(struct.pack(">f", _f32(value)), "big")
    exponent = bits & 0x7F800000
    if exponent != 0x7F800000:
        bits += 0x7FFF + ((bits >> 16) & 1)
    return struct.unpack(">f", (bits & 0xFFFF0000).to_bytes(4, "big"))[0]


def _old_softplus_f32(value: float) -> float:
    value = _f32(value)
    tail = _f32(math.exp(_f32(-abs(value))))
    return _f32(max(value, 0.0) + _f32(math.log(_f32(1.0 + tail))))


def _compensated_softplus_f32(value: float) -> float:
    value = _f32(value)
    hi, lo = max(value, 0.0), min(value, 0.0)
    tail = _f32(math.exp(_f32(lo - hi)))
    xp1 = _f32(1.0 + tail)
    log1p = tail if xp1 == 1.0 else _f32(tail * (_f32(math.log(xp1)) / _f32(xp1 - 1.0)))
    return _f32(hi + log1p)


def numerical_discriminators() -> dict[str, object]:
    """Independent host examples that make every storage law observable."""

    width = 128
    ss = 1.0e-8
    correct_l2 = (width**-0.5) / math.sqrt(ss / width + 1.0e-6 / width)
    inflated_eps = (width**-0.5) / math.sqrt(ss / width + 1.0e-6)

    softplus_input = -20.0
    old_softplus = _old_softplus_f32(softplus_input)
    compensated_softplus = _compensated_softplus_f32(softplus_input)

    rounding_example = None
    for y in (0.17, 0.31, 0.73, 1.37, 3.11):
        for inv in (0.19, 0.43, 0.91, 1.71):
            for weight in (0.29, 0.67, 1.13, 2.09):
                two_rounds = _bf16(_bf16(y * inv) * _bf16(weight))
                one_round = _bf16(_bf16(weight) * (y * inv))
                if two_rounds != one_round:
                    rounding_example = {
                        "y": y,
                        "inv": inv,
                        "weight": weight,
                        "bf16_rms_then_weight": two_rounds,
                        "fused_before_bf16": one_round,
                    }
                    break
            if rounding_example:
                break
        if rounding_example:
            break

    stable_tail = 1.0 / (1.0 + math.exp(-1000.0))
    return {
        "qk_epsilon": {
            "head_width": width,
            "sum_squares": ss,
            "l2_equivalent": correct_l2,
            "inflated_epsilon": inflated_eps,
            "distinguishable": correct_l2 != inflated_eps,
        },
        "softplus": {
            "input": softplus_input,
            "naive_float32": old_softplus,
            "compensated_float32": compensated_softplus,
            "distinguishable": old_softplus != compensated_softplus,
        },
        "stable_sigmoid": {
            "input": 1000.0,
            "stable": stable_tail,
            "one_sided_negative_exp_overflows": True,
        },
        "output_rms_rounding": {
            "example": rounding_example,
            "distinguishable": rounding_example is not None,
        },
        "decoder_rms_geometry": {
            "hidden_width": 5120,
            "ordinary": {
                "kernel": "rms_looped",
                "reads_per_iteration": 4,
                "threads": 1024,
                "simdgroups": 32,
                "second_stage": "simd_sum over 32 shared slots",
            },
            "old_tensorfold": {
                "reads_per_thread": 16,
                "threads": 320,
                "simdgroups": 10,
                "second_stage": "sequential sum over 10 shared slots",
            },
            "distinguishable": True,
        },
    }


def validate_reference(root: Path) -> dict[str, object]:
    root = root.expanduser().resolve()
    errors: list[str] = []
    try:
        revision = _git(root, "rev-parse", "HEAD")
    except (OSError, subprocess.CalledProcessError) as error:
        return {"root": str(root), "revision": None, "files": {}, "errors": [str(error)]}
    if revision != EXPECTED_MLX_LM_REVISION:
        errors.append(
            f"mlx-lm revision {revision}, expected {EXPECTED_MLX_LM_REVISION}"
        )
    files: dict[str, str] = {}
    for relative, expected in REFERENCE_FILES.items():
        path = root / relative
        if not path.is_file():
            errors.append(f"missing reference file {relative}")
            continue
        actual = _sha256(path)
        files[relative] = actual
        if actual != expected:
            errors.append(f"reference digest changed for {relative}: {actual}")
    tracked = _git(root, "status", "--porcelain", "--", *REFERENCE_FILES)
    if tracked:
        errors.append("reference files are dirty")
    return {"root": str(root), "revision": revision, "files": files, "errors": errors}


def validate_mlx_reference(root: Path) -> dict[str, object]:
    """Bind the Metal RMS implementation shipped by the mlx2 environment."""

    root = root.expanduser().resolve()
    errors: list[str] = []
    try:
        revision = _git(root, "rev-parse", "HEAD")
    except (OSError, subprocess.CalledProcessError) as error:
        return {"root": str(root), "revision": None, "files": {}, "errors": [str(error)]}
    if revision != EXPECTED_MLX_REVISION:
        errors.append(f"MLX revision {revision}, expected {EXPECTED_MLX_REVISION}")
    files: dict[str, str] = {}
    for relative, expected in MLX_REFERENCE_FILES.items():
        path = root / relative
        if not path.is_file():
            errors.append(f"missing MLX reference file {relative}")
            continue
        actual = _sha256(path)
        files[relative] = actual
        if actual != expected:
            errors.append(f"MLX reference digest changed for {relative}: {actual}")
    tracked = _git(root, "status", "--porcelain", "--", *MLX_REFERENCE_FILES)
    if tracked:
        errors.append("MLX reference files are dirty")
    return {"root": str(root), "revision": revision, "files": files, "errors": errors}


def _require(
    checks: list[dict[str, object]],
    *,
    law: str,
    file: str,
    source: str,
    pattern: str,
    detail: str,
) -> None:
    matched = re.search(pattern, source, flags=re.DOTALL) is not None
    checks.append(
        {"law": law, "file": file, "passed": matched, "detail": detail}
    )


def _forbid(
    checks: list[dict[str, object]],
    *,
    law: str,
    file: str,
    source: str,
    pattern: str,
    detail: str,
) -> None:
    matched = re.search(pattern, source, flags=re.DOTALL) is None
    checks.append(
        {"law": law, "file": file, "passed": matched, "detail": detail}
    )


def _require_any(
    checks: list[dict[str, object]],
    *,
    law: str,
    file: str,
    source: str,
    patterns: tuple[str, ...],
    detail: str,
) -> None:
    matched = any(
        re.search(pattern, source, flags=re.DOTALL) is not None
        for pattern in patterns
    )
    checks.append(
        {"law": law, "file": file, "passed": matched, "detail": detail}
    )


def validate_tensorfold(root: Path) -> dict[str, object]:
    """Inspect candidate source without importing TensorFold or MLX."""

    root = root.expanduser().resolve()
    checks: list[dict[str, object]] = []
    texts: dict[str, str] = {}
    digests: dict[str, str] = {}
    for name in KERNEL_FILES:
        path = root / KERNEL_ROOT / name
        if not path.is_file():
            checks.append(
                {"law": "source_closure", "file": name, "passed": False, "detail": "file missing"}
            )
            continue
        texts[name] = path.read_text()
        digests[str(KERNEL_ROOT / name)] = _sha256(path)

    lane = texts.get("lane_glue.py", "")
    lane_tree = texts.get("lane_tree.py", "")
    stream = texts.get("stream_gdn.py", "")
    row = texts.get("row_glue.py", "")

    dense_norm_pattern = _pattern(
        r"metal::precise::rsqrt\((?:ss|total) / float\(K\) \+ eps\[0\]\)",
        r".*const bfloat normalized = bfloat\((?:float\(h\)|hv\[i\]) \* inv\)",
        r".*bfloat\(Wt\[(?:r \+ )?int\(t\) \* E \+ i\] \* normalized\)",
    )
    dense_mlp_pattern = _pattern(
        r"const bfloat (?P<mlp_input>gate|gf) = ",
        r".*const bfloat mlp_sigmoid_exp = bfloat\(metal::exp\(metal::abs\((?P=mlp_input)\)\)\)",
        r".*const bfloat mlp_sigmoid_low = bfloat\(bfloat\(1\.0f\) / ",
        r"bfloat\(bfloat\(1\.0f\) \+ mlp_sigmoid_exp\)\)",
        r".*const bfloat mlp_sigmoid = (?P=mlp_input) < bfloat\(0\.0f\)",
        r".*const bfloat activated = bfloat\((?P=mlp_input) \* mlp_sigmoid\)",
        r".*bfloat\(activated \* ",
    )
    for file, source in (("lane_glue.py", lane), ("row_glue.py", row)):
        _require(
            checks,
            law="decoder_rms_norm",
            file=file,
            source=source,
            pattern=dense_norm_pattern,
            detail="decoder RMSNorm uses precise rsqrt and rounds normalized BF16 before gain",
        )
        _require(
            checks,
            law="dense_swiglu",
            file=file,
            source=source,
            pattern=dense_mlp_pattern,
            detail="dense SwiGLU stores BF16 SiLU before the BF16 up product",
        )
        _forbid(
            checks,
            law="old_dense_trunk_removed",
            file=file,
            source=source,
            pattern=(
                r"metal::rsqrt\(total / float\(K\)|"
                r"float\(Wt\[int\(t\) \* E \+ i\]\) \* \(hv\[i\] \* inv\)|"
                r"metal::exp\(-gf\)"
            ),
            detail="bare decoder rsqrt, fused-before-BF16 norm, and one-sided MLP SiLU are absent",
        )
        _require(
            checks,
            law="decoder_rms_reduction_geometry",
            file=file,
            source=source,
            pattern=_pattern(
                r"constexpr int E = 4",
                r".*(?:K > 4096|K <= 4096)",
                r".*1024",
                r".*for \(int r = 0; r < K; r \+= (?:TPG|tpg) \* E\)",
                r".*simd_sum\((?:red|local_sums)\[",
            ),
            detail="decoder RMS follows MLX 39400a0d4's 4-read looped geometry and second simd reduction",
        )
        _forbid(
            checks,
            law="old_decoder_rms_reduction_removed",
            file=file,
            source=source,
            pattern=(
                r"constexpr int E = 16|"
                r"for \(int i = 0; i < TPG / 32; i\+\+\) total \+= red\[i\]|"
                r"ss = fma\(hv\[i\], hv\[i\], ss\)"
            ),
            detail="the 16-contiguous-read, explicit-fma, sequential second reduction is absent",
        )

    recurrent_tree_pattern = _pattern(
        r"state\[i\] = state\[i\] \* g_",
        r".*kv_mem \+= state\[i\] \* k_\[s_idx\]",
        r".*kv_mem = simd_sum\(kv_mem\)",
        r".*(?:auto|const float) delta = \(v_\[dv_idx\] - kv_mem\) \* beta_",
        r".*state\[i\] = state\[i\] \+ k_\[s_idx\] \* delta",
        r".*out \+= state\[i\] \* q_\[s_idx\]",
        r".*out = simd_sum\(out\)",
        r".*y\[.*\] = static_cast<InT>\(out\)",
    )
    recurrent_replay_pattern = _pattern(
        r"state\[i\] = state\[i\] \* g_",
        r".*kv_mem \+= state\[i\] \* k_\[s_idx\]",
        r".*kv_mem = simd_sum\(kv_mem\)",
        r".*(?:auto|const float) delta = \(v_\[dv_idx\] - kv_mem\) \* beta_",
        r".*state\[i\] = state\[i\] \+ k_\[s_idx\] \* delta",
        r".*o_state\[.*\] = (?:static_cast<StT>\(state\[i\]\)|state\[i\])",
    )
    recurrent_sources = (
        (
            "lane_tree.py",
            _assignment_body(lane_tree, "_TREE_SOURCE"),
            _assignment_body(lane_tree, "_REPLAY_SOURCE"),
        ),
        (
            "stream_gdn.py",
            _assignment_body(stream, "_TREE"),
            _assignment_body(stream, "_REPLAY"),
        ),
    )
    for file, tree_source, replay_source in recurrent_sources:
        _require(
            checks,
            law="recurrent_tree_order",
            file=file,
            source=tree_source,
            pattern=recurrent_tree_pattern,
            detail="tree output reads the fp32 state after decay, correction, and ordered simd reductions",
        )
        _require(
            checks,
            law="recurrent_replay_order",
            file=file,
            source=replay_source,
            pattern=recurrent_replay_pattern,
            detail="commit replay applies the same fp32 state transition and stores FP32 state",
        )
    _require(
        checks,
        law="row_tree_derivation",
        file="row_glue.py",
        source=row,
        pattern=r"return lane_tree\._TREE_SOURCE.*o_state.*states\[0\]\[i\]",
        detail="row tree output and commit state derive from the canonical lane-tree source",
    )

    helper_or_inline = {
        "stable_fast_sigmoid": (
            r"mlx_sigmoid_fast.*metal::exp\(metal::abs\(x\)\).*return \(x < 0\) \? y : .*1.* - y",
            _pattern(
                r"const bfloat sigmoid_exp = bfloat\(metal::exp\(metal::abs\(conv\)\)\)",
                r".*const bfloat sigmoid_low = bfloat\(bfloat\(1\.0f\) / ",
                r"bfloat\(bfloat\(1\.0f\) \+ sigmoid_exp\)\)",
                r".*const bfloat sigmoid = conv < bfloat\(0\.0f\)",
                r".*\? sigmoid_low : bfloat\(bfloat\(1\.0f\) - sigmoid_low\)",
            ),
        ),
        "precise_sigmoid": (
            r"mlx_sigmoid_precise.*metal::precise::exp\(metal::abs\(x\)\).*return \(x < 0\) \? y : .*1.* - y",
            _pattern(
                r"const float beta_y = 1\.0f / ",
                r"\(1\.0f \+ metal::precise::exp\(metal::abs\(beta_x\)\)\)",
                r".*BETA\[.*\] = beta_x < 0\.0f \? beta_y : 1\.0f - beta_y",
            ),
        ),
        "compensated_log1p": (
            _pattern(
                r"mlx_log1p_fast.*xp1 == 1\.0f \? xf : xf \* ",
                r"\(metal::log\(xp1\) / \(xp1 - 1\.0f\)\)",
            ),
            _pattern(
                r"const float arg_plus_one = 1\.0f \+ arg",
                r".*arg_plus_one == 1\.0f.*\? arg : arg \* ",
                r"\(metal::log\(arg_plus_one\) / \(arg_plus_one - 1\.0f\)\)",
            ),
        ),
        "stable_softplus": (
            _pattern(
                r"mlx_softplus_fast.*lo == -inf \|\| hi == inf",
                r".*hi \+ mlx_log1p_fast.*metal::exp\(lo - hi\)",
            ),
            _pattern(
                r"const bfloat hi = metal::max\(av, zero\)",
                r".*const bfloat lo = metal::min\(av, zero\)",
                r".*const bfloat softplus_arg = bfloat\(metal::exp\(lo - hi\)\)",
                r".*lo == -metal::numeric_limits<bfloat>::infinity\(\)",
                r".*hi == metal::numeric_limits<bfloat>::infinity\(\)",
                r".*\? hi : bfloat\(hi \+ bfloat\(log1p\)\)",
            ),
        ),
    }
    for law, patterns in helper_or_inline.items():
        _require_any(
            checks,
            law=law,
            file="lane_glue.py",
            source=lane,
            patterns=patterns,
            detail="MLX-compatible typed arithmetic is present in the kernel source",
        )

    for file, source in (("lane_glue.py", lane), ("stream_gdn.py", stream)):
        _require_any(
            checks,
            law="conv_silu",
            file=file,
            source=source,
            patterns=(
                r"bfloat(?:16_t)?\s+conv.*mlx_sigmoid_fast\(conv\).*bfloat",
                _pattern(
                    r"const bfloat conv = bfloat\(acc\)",
                    r".*const bfloat sigmoid_exp = bfloat\(metal::exp\(metal::abs\(conv\)\)\)",
                    r".*const bfloat sigmoid_low = bfloat\(bfloat\(1\.0f\) / ",
                    r"bfloat\(bfloat\(1\.0f\) \+ sigmoid_exp\)\)",
                    r".*vals\[j\] = float\(bfloat\(conv \* sigmoid\)\)",
                ),
            ),
            detail="conv SiLU uses the stable symmetric fast sigmoid at BF16 boundaries",
        )
        _require(
            checks,
            law="qk_rsqrt",
            file=file,
            source=source,
            pattern=r"inv\s*=\s*metal::precise::rsqrt\(ss / float\(DK\) \+ norm_eps\)",
            detail="q/k normalization uses precise rsqrt with epsilon divided by DK",
        )
        _require(
            checks,
            law="qk_storage",
            file=file,
            source=source,
            pattern=(
                r"scale = isq \? float\(bfloat\(1\.0f / float\(DK\)\)\)"
                r".*float\(bfloat\(metal::precise::rsqrt\(float\(DK\)\)\)\)"
                r".*bfloat out = bfloat\(scale \* float\(bfloat\(vals\[j\] \* inv\)\)\)"
            ),
            detail="q/k RMS result and scale use the reference BF16 rounding sites",
        )
        _require_any(
            checks,
            law="softplus",
            file=file,
            source=source,
            patterns=(
                _pattern(
                    r"bfloat(?:16_t)?\s+sp\s*=\s*",
                    r"mlx_softplus_fast(?:<[^>]+>)?\(s\)",
                ),
                _pattern(
                    r"const bfloat av = bfloat\(float\(Ain\[.*\]\) \+ float\(DT\[hv\]\)\)",
                    r".*const bfloat softplus_arg = bfloat\(metal::exp\(lo - hi\)\)",
                    r".*const bfloat sp = metal::isnan\(av\)",
                    r".*\? hi : bfloat\(hi \+ bfloat\(log1p\)\)",
                ),
            ),
            detail="decay softplus uses compensated MLX-compatible BF16 softplus",
        )
        _require(
            checks,
            law="decay_exp",
            file=file,
            source=source,
            pattern=r"metal::precise::exp\(\s*-metal::precise::exp\(float\(ALOG\[hv\]\)\) \* float\(sp\)\)",
            detail="both exponentials in recurrent decay are precise",
        )
        _require_any(
            checks,
            law="beta_sigmoid",
            file=file,
            source=source,
            patterns=(
                r"mlx_sigmoid_precise<float>\(float\(Bin\[",
                _pattern(
                    r"const float beta_x = float\(Bin\[.*\]\)",
                    r".*const float beta_y = 1\.0f / ",
                    r"\(1\.0f \+ metal::precise::exp\(metal::abs\(beta_x\)\)\)",
                    r".*BETA\[.*\] = beta_x < 0\.0f \? beta_y : 1\.0f - beta_y",
                ),
            ),
            detail="beta widens to FP32 and uses the precise stable sigmoid",
        )
        _forbid(
            checks,
            law="old_prework_removed",
            file=file,
            source=source,
            pattern=r"metal::log\(1\.0f \+ metal::exp\(|metal::exp\(-conv\)",
            detail="one-sided SiLU and naive log(1+exp()) are absent",
        )

    output_patterns = (
        (
            r"metal::precise::rsqrt\(ss / float\(DV\) \+ eps\[0\]\)"
            r".*bfloat(?:16_t)?\s+normalized\s*=\s*bfloat(?:16_t)?\(yv\[j\] \* inv\)"
            r".*normalized\s*=\s*bfloat(?:16_t)?\(float\(NW\[d\]\) \* float\(normalized\)\)"
            r".*zf \* mlx_sigmoid_fast<float>\(zf\)"
            r".*bfloat(?:16_t)?\(float\(normalized\) \* gate\)"
        ),
        (
            r"metal::precise::rsqrt\(ss / float\(DV\) \+ eps\[0\]\)"
            r".*const bfloat normalized = bfloat\(yv\[j\] \* inv\)"
            r".*const bfloat normed = bfloat\(NW\[d\] \* normalized\)"
            r".*const float gate_exp = metal::exp\(metal::abs\(zf\)\)"
            r".*const float gate_low = 1\.0f / \(1\.0f \+ gate_exp\)"
            r".*const float gate_sigmoid = zf < 0\.0f \? gate_low : 1\.0f - gate_low"
            r".*const float gate = zf \* gate_sigmoid"
            r".*bfloat\(float\(normed\) \* gate\)"
        ),
    )
    for file, source in (("lane_glue.py", lane), ("row_glue.py", row)):
        _require_any(
            checks,
            law="output_norm_gate",
            file=file,
            source=source,
            patterns=output_patterns,
            detail="output RMS rounds to BF16 before weight, then uses FP32 stable fast SiLU",
        )
        _forbid(
            checks,
            law="old_output_removed",
            file=file,
            source=source,
            pattern=r"metal::rsqrt\(ss / float\(DV\)|metal::exp\(-zf\)",
            detail="bare rsqrt and one-sided output gate are absent",
        )

    revision = None
    tracked_diff = None
    try:
        revision = _git(root, "rev-parse", "HEAD")
        tracked_diff = _git(
            root,
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            str(KERNEL_ROOT),
        )
    except (OSError, subprocess.CalledProcessError):
        pass
    passed = bool(checks) and all(bool(check["passed"]) for check in checks)
    if tracked_diff:
        checks.append(
            {
                "law": "source_binding",
                "file": str(KERNEL_ROOT),
                "passed": False,
                "detail": "candidate kernel source is dirty",
            }
        )
        passed = False
    return {
        "root": str(root),
        "revision": revision,
        "tracked_diff": tracked_diff,
        "files": digests,
        "checks": checks,
        "passed": passed,
    }


def build_receipt(tensorfold: Path, mlx_lm: Path, mlx: Path) -> dict[str, object]:
    reference = validate_reference(mlx_lm)
    mlx_reference = validate_mlx_reference(mlx)
    candidate = validate_tensorfold(tensorfold)
    discriminators = numerical_discriminators()
    discriminator_passed = all(
        value.get("distinguishable", True)
        for value in discriminators.values()
        if isinstance(value, dict)
    )
    passed = (
        not reference["errors"]
        and not mlx_reference["errors"]
        and candidate["passed"]
        and discriminator_passed
    )
    return {
        "schema": "mlx2.qwen38_tensorfold_coupled_law_oracle.v2",
        "host_only": True,
        "imports_mlx": False,
        "qualification_claim": False,
        "reference": reference,
        "mlx_reference": mlx_reference,
        "candidate": candidate,
        "discriminators": discriminators,
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensorfold-source", type=Path, required=True)
    parser.add_argument("--mlx-lm-source", type=Path, required=True)
    parser.add_argument("--mlx-source", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = build_receipt(args.tensorfold_source, args.mlx_lm_source, args.mlx_source)
    encoded = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
