"""MiMo-V2.6-Pro-RL real layers on the CPU (opt-in: MLX2_MIMO_V2_REAL=1; reads ~3 GB of the checkpoint).

Checkpoint: MiMo-V2.6-Pro-RL (XiaomiMiMo/MiMo-V2.6-Pro-RL@a066ba00), fingerprint from
mlx2.adapters.flash_next.artifact_identity, printed by the tests.

1. Loader dequant vs independent NumPy dequant (tests/mimo_v2_numpy_ref.py) on layer 0 (full
   attention, dense FP8 MLP) and layer 1 (SWA with sinks, MXFP4 experts): bit-equal float32.
   The rank-sharded qkv reading is also checked against the data: with E4M3 block scales of
   ``amax / 448`` every 128x128 block of the correct grid reaches code magnitude 448.
2. Layer 0 and layer 1 forward (float32, CPU) for two short texts vs Xiaomi's PyTorch
   ``MiMoV2DecoderLayer`` fed the NumPy-dequantized weights (tests/mimo_v2_torch_runner.py,
   ``python -I`` on a temp copy of the vendor files): hidden state, dense ``act``, router ids
   and weights, per-slot expert rows.
Every process keeps RSS under 20 GB (checked); no GPU is used.
"""
from __future__ import annotations

import hashlib
import json
import os
import resource
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import mimo_v2_numpy_ref as R  # noqa: E402

CKPT = Path(os.environ.get("MLX2_MIMO_V2_CHECKPOINT", Path.home() / "Models/MiMo-V2.6-Pro-RL"))
PINS = {"modeling_mimo_v2.py": "a8c3cb3aae473bcc15f023010547c919f15eba6546e6ed7efb61a8937b12f3ad",
        "configuration_mimo_v2.py": "773062ac9850b908eb54751b3e4dbe00e653c0e80595599409e97d0c1af2ce3e"}
TEXTS = ["The capital of France is Paris, and its river is",
         "def fibonacci(n):\n    if n < 2:\n        return n"]
RSS_LIMIT = 20 << 30

pytestmark = pytest.mark.skipif(os.environ.get("MLX2_MIMO_V2_REAL") != "1" or not (CKPT / "config.json").exists(),
                                reason="opt-in real MiMo-V2 checkpoint test (MLX2_MIMO_V2_REAL=1)")


def _rss_bytes(who=resource.RUSAGE_SELF):
    return resource.getrusage(who).ru_maxrss      # bytes on macOS


@pytest.fixture(scope="module")
def fam():
    from research.amoeba.sweep.mimo_v2 import MiMoV2Family
    f = MiMoV2Family(CKPT, dtype="float32", experts="routed")
    print(f"\ncheckpoint {f.checkpoint} fingerprint {f.fingerprint}")
    return f


@pytest.fixture(scope="module")
def wm():
    return json.loads((CKPT / "model.safetensors.index.json").read_text())["weight_map"]


def _np(name, wm):
    return R.read(CKPT / wm[name], name)[2]


def test_real_dequant_matches_numpy(fam, wm):
    c = fam.cfg
    tp = fam.ckpt.tp_size
    assert tp == 8
    for l in (0, 1):
        w = fam.load_layer(l)
        p = f"model.layers.{l}."
        a = c.attn(l)
        codes = _np(p + "self_attn.qkv_proj.weight", wm)
        scale = _np(p + "self_attn.qkv_proj.weight_scale_inv", wm)
        # Data evidence for the rank layout: per-rank 128-row blocks all reach |code| = 448.
        shard = a.heads // tp * a.head_dim + a.kv_heads // tp * (a.head_dim + a.v_head_dim)
        mag = np.abs(R.E4M3[codes]).reshape(codes.shape[0], -1, 128).max(-1)
        amax = [mag[r * shard + b:min((r + 1) * shard, r * shard + b + 128)].max(0)
                for r in range(tp) for b in range(0, shard, 128)]
        assert len(amax) == scale.shape[0] and np.all(np.stack(amax) == 448.0)
        q, k, v = R.fused_qkv_dequant(codes, scale, a.heads, a.kv_heads, a.head_dim, a.v_head_dim, tp, (128, 128))
        for n, want in zip("qkv", (q, k, v)):
            np.testing.assert_array_equal(np.array(w[f"self_attn.{n}_proj.weight"]), want)
        np.testing.assert_array_equal(np.array(w["self_attn.o_proj.weight"]), _np(p + "self_attn.o_proj.weight", wm))
        if c.is_moe(l):
            ids = np.array([0, 77, 200, 383])
            bank = fam._bank(l, mx.array(ids))
            from mlx2.runtime.models import mimo_v2 as M
            for j, n in enumerate(("gate_proj", "up_proj", "down_proj")):
                for i, e in enumerate(bank.ids):
                    name = f"{p}mlp.experts.{e}.{n}.weight"
                    want = R.mxfp4_dequant(_np(name, wm), _np(name + "_scale", wm))
                    got = np.array(M._dense_of(getattr(bank, ("gate", "up", "down")[j]), i, mx.float32))
                    np.testing.assert_array_equal(got, want)
        else:
            for n in ("gate_proj", "up_proj", "down_proj"):
                name = p + f"mlp.{n}.weight"
                want = R.fp8_dequant(_np(name, wm), _np(name + "_scale_inv", wm), (128, 128))
                np.testing.assert_array_equal(np.array(w[f"mlp.{n}.weight"]), want)
        fam.release_layer(l)
    rss = _rss_bytes()
    print(f"dequant check peak RSS {rss / 2**30:.2f} GB")
    assert rss < RSS_LIMIT


@pytest.fixture(scope="module")
def vendor_pkg(tmp_path_factory):
    pytest.importorskip("torch")
    pkg = tmp_path_factory.mktemp("vendor_real") / "pkg"
    (pkg / "mimo_ref").mkdir(parents=True)
    for name, sha in PINS.items():
        data = (CKPT / name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == sha
        (pkg / "mimo_ref" / name).write_bytes(data)
    (pkg / "mimo_ref" / "__init__.py").write_text("")
    return pkg


def _reference_layer(pkg, l, h, work):
    np.save(work / f"h{l}.npy", h)
    out = work / f"ref{l}.npz"
    proc = subprocess.run([sys.executable, "-I", str(HERE / "mimo_v2_torch_runner.py"), "layer", str(pkg),
                           str(CKPT), str(l), str(work / f"h{l}.npy"), str(out)],
                          capture_output=True, text=True, timeout=3600, cwd=str(work))
    assert proc.returncode == 0, proc.stderr[-4000:]
    return dict(np.load(out))


def test_real_layers_match_vendor_reference(fam, vendor_pkg, tmp_path):
    ids = [fam.tok.encode(t) for t in TEXTS]
    T = min(len(i) for i in ids)
    ids = np.array([i[:T] for i in ids], np.int32)
    h = fam.embed(mx.array(ids))
    k = fam.cfg.top_k
    lines = [f"texts: B={ids.shape[0]} T={T}"]
    for l in (0, 1):
        ref = _reference_layer(vendor_pkg, l, np.array(h), tmp_path)
        h_next, pay = fam.run_layer(l, h)
        mx.eval(h_next, *pay.values())
        got, want = np.array(h_next), ref["h"]
        dh = np.abs(got - want).max()
        lines.append(f"L{l} {fam.kinds[l]} h max|d| {dh:.3g} (max|ref| {np.abs(want).max():.3g}, "
                     f"rel-rms {np.sqrt(((got - want) ** 2).mean() / (want ** 2).mean()):.3g})")
        assert dh < 1e-4 * np.abs(want).max()
        if fam.kinds[l] == "dense":
            act = np.array(pay["act"])
            da = np.abs(act - ref["act"]).max()
            lines.append(f"L{l} act max|d| {da:.3g} (max|ref| {np.abs(ref['act']).max():.3g})")
            assert da < 1e-4 * np.abs(ref["act"]).max()
        else:
            inds = np.array(pay["inds"]).reshape(-1, k)
            wts = np.array(pay["w"]).reshape(-1, k)
            o, r = np.argsort(inds, -1), np.argsort(ref["ids"], -1)
            np.testing.assert_array_equal(np.take_along_axis(inds, o, -1), np.take_along_axis(ref["ids"], r, -1))
            dw = np.abs(np.take_along_axis(wts, o, -1) - np.take_along_axis(ref["wts"], r, -1)).max()
            assert dw < 1e-5
            rows = np.array(pay["rows"]).reshape(inds.shape[0], k, -1)
            worst, scale = 0.0, 0.0
            for e in np.unique(inds):
                tok, slot = np.nonzero(inds == e)
                order = np.argsort(tok, kind="stable")
                y = ref[f"y{e}"]
                worst = max(worst, float(np.abs(rows[tok[order], slot[order]] - y).max()))
                scale = max(scale, float(np.abs(y).max()))
            lines.append(f"L{l} router ids equal ({len(np.unique(inds))} experts, {inds.size} routes), "
                         f"weights max|d| {dw:.3g}; rows max|d| {worst:.3g} (max|ref| {scale:.3g})")
            assert worst < 1e-4 * scale
        h = h_next
        fam.release_layer(l)
    rss, child = _rss_bytes(), _rss_bytes(resource.RUSAGE_CHILDREN)
    lines.append(f"peak RSS: test {rss / 2**30:.2f} GB, reference child {child / 2**30:.2f} GB")
    print("\n" + "\n".join(lines))
    assert rss < RSS_LIMIT and child < RSS_LIMIT
