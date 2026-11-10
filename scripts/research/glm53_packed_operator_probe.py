"""Synthetic packed/GDN operator diagnostics inside a paired GPU lease."""

from __future__ import annotations

import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src

import argparse
import json
import math
import runpy
import subprocess
from pathlib import Path

import mlx.core as mx
import numpy as np
from gguf import GGMLQuantizationType
from gguf.quants import dequantize

from mlx2.runtime.gguf.k_blocks import SIZES
from mlx2.runtime.gguf.packed_metal import (
    PackedMatrix,
    embedding,
    grouped_linear,
    packed_moe,
)
from mlx2.runtime.gguf.reader import GGUFTensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("refusing existing evidence")
    args.out.mkdir(parents=True)
    rng = np.random.default_rng(104)
    rows = []

    def weight(kind, shape):
        count, size = SIZES[kind]
        n = math.prod(shape) // count
        raw = rng.integers(0, 256, (n, size), dtype=np.uint8)
        for off in {2: (0,), 11: (108,), 12: (0, 2), 13: (0, 2), 14: (208,)}[kind]:
            raw[:, off : off + 2] = (
                rng.uniform(-0.02, 0.02, (n, 1)).astype("<f2").view(np.uint8)
            )
        ref = dequantize(raw, GGMLQuantizationType(kind)).reshape(shape)
        t = GGUFTensor("synthetic", shape[::-1], kind, 0, raw.nbytes)
        return PackedMatrix(mx.array(raw.reshape(-1)), t), ref

    def check(name, actual, reference, *, atol=3e-4, rtol=3e-5):
        a = np.array(actual)
        b = np.asarray(reference, dtype=np.float32)
        error = float(np.max(np.abs(a - b)))
        passed = bool(np.allclose(a, b, atol=atol, rtol=rtol))
        rows.append(
            {
                "name": name,
                "passed": passed,
                "max_abs_error": error,
                "atol": atol,
                "rtol": rtol,
            }
        )
        print(json.dumps(rows[-1]), flush=True)
        if not passed:
            raise AssertionError(name)

    try:
        for kind in SIZES:
            k = 128 if kind == 2 else 512
            w, ref = weight(kind, (3, k))
            ids = mx.array([[2, 0, 2]], mx.uint32)
            check(
                f"format-{kind}-embedding",
                embedding(ids, w),
                ref[[[2, 0, 2]]],
                atol=0,
                rtol=0,
            )
            for tokens in (1, 2, 17):
                x = rng.normal(0, 0.1, (tokens, k)).astype(np.float32)
                check(
                    f"format-{kind}-dense-{tokens}",
                    w(mx.array(x)),
                    x.astype(np.float64) @ ref.astype(np.float64).T,
                )
            grouped, gref = weight(kind, (3, 5, k))
            x = rng.normal(0, 0.1, (2, 3, k)).astype(np.float32)
            check(
                f"format-{kind}-grouped",
                grouped_linear(mx.array(x), grouped),
                np.einsum(
                    "bgi,goi->bgo", x.astype(np.float64), gref.astype(np.float64)
                ),
            )
        gate, gref = weight(11, (3, 256, 256))
        up, uref = weight(11, (3, 256, 256))
        down, dref = weight(12, (3, 256, 256))
        x = rng.normal(0, 0.01, (2, 256)).astype(np.float32)
        ids = np.array([[0, 2], [2, 1]], dtype=np.uint32)
        scores = np.array([[0.3, 0.7], [0.6, 0.4]], dtype=np.float32)
        reference = np.zeros((2, 256), dtype=np.float64)
        for t in range(2):
            for slot in range(2):
                expert = ids[t, slot]
                g = np.minimum(
                    x[t].astype(np.float64) @ gref[expert].astype(np.float64).T, 10
                )
                u = np.clip(
                    x[t].astype(np.float64) @ uref[expert].astype(np.float64).T, -10, 10
                )
                reference[t] += scores[t, slot] * (
                    (g / (1 + np.exp(-g)) * u) @ dref[expert].astype(np.float64).T
                )
        check(
            "mixed-q3-q4-moe",
            packed_moe(mx.array(x), mx.array(ids), mx.array(scores), gate, up, down),
            reference,
        )
        from mlx2.runtime.models.glm53_gguf import Glm53GGUFModel, Glm53KDACache

        helpers = runpy.run_path("tests/test_glm53_gguf_cpu.py")
        weights = helpers["Weights"]()
        config = {"layer_types": [], "rms_norm_eps": 1e-5}
        ordinary = Glm53GGUFModel(
            config, weights, identity="synthetic", math_policy="ordinary"
        )
        reference_model = Glm53GGUFModel(
            config, weights, identity="synthetic", math_policy="reference"
        )
        oc, rc = Glm53KDACache("synthetic"), Glm53KDACache("synthetic")
        for step, length in enumerate((7, 1, 1, 1)):
            x = mx.array(rng.normal(0, 0.1, (1, length, 4)).astype(np.float32))
            actual = ordinary._kda(x, oc, "blk.0")
            expected = reference_model._kda(x, rc, "blk.0")
            check(
                f"kda-{step}-output", actual, np.array(expected), atol=2e-6, rtol=2e-5
            )
            check(f"kda-{step}-conv", oc.conv, np.array(rc.conv), atol=0, rtol=0)
            check(
                f"kda-{step}-state",
                oc.recurrent,
                np.array(rc.recurrent),
                atol=2e-6,
                rtol=2e-5,
            )
        result = "passed"
    except BaseException:
        result = "failed"
        raise
    finally:
        payload = {
            "schema": "mlx2.glm53-packed-operators.v1",
            "scope": "synthetic operators only; no model checkpoint executed",
            "source_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "source_dirty": bool(
                subprocess.check_output(
                    ["git", "status", "--porcelain"], text=True
                ).strip()
            ),
            "result": result,
            "rows": rows,
            "device": mx.device_info(),
        }
        (args.out / "result.json").write_text(
            json.dumps(payload, indent=2, default=str) + "\n"
        )


if __name__ == "__main__":
    main()
