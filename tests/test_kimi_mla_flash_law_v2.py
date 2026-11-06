# SPDX-License-Identifier: MIT
"""Kimi MLA flash192 kernel law v2 (kimi_linear_mla_flash.EXPANDED_LAWS).

CPU: the policy knob, its routing and counters.  GPU (M5 only): v1/v2 against
the FP32 ops law, chunk invariance, token- and head-major kv, misaligned
bases.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx2.adapters.kimi_linear import parse_policy
from mlx2.runtime.models import kimi_linear_mla_flash as fl
from mlx2.runtime.models.cache import KVCache
from mlx2.runtime.models.kimi_linear import MultiLatentAttention

from test_kimi_linear import kimi_mla_args  # noqa: E402



def test_v1_source_is_unchanged():
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    base = subprocess.run(
        ["git", "show", "f55227951:src/mlx2/runtime/models/kimi_linear_mla_flash.py"],
        cwd=root, capture_output=True, text=True)
    if base.returncode:
        pytest.skip("base revision not in this checkout")
    ns = {}
    start = base.stdout.index('_EXPANDED = r"""')
    end = base.stdout.index('"""', start + len('_EXPANDED = r"""')) + 3
    exec(base.stdout[start:end], ns)  # noqa: S102 -- a literal from our own history
    assert hashlib.sha256(ns["_EXPANDED"].encode()).hexdigest() == hashlib.sha256(
        fl._EXPANDED.encode()).hexdigest()


def test_policy_law_is_validated():
    assert parse_policy()["mla_flash_law"] in {"v1", "v2"}
    assert parse_policy({"mla_flash_law": "v2"})["mla_flash_law"] == "v2"
    for bad in ("v3", "", None, 2):
        with pytest.raises(ValueError, match="mla_flash_law"):
            parse_policy({"mla_flash_law": bad})


def test_unknown_law_fails_closed():
    q = mx.zeros((1, 1, fl.HEADS * (fl.NOPE + fl.SHARED)), mx.bfloat16)
    kv = mx.zeros((1, fl.HEADS, 1, fl.NOPE + fl.VDIM), mx.bfloat16)
    sh = mx.zeros((1, 1, 1, fl.SHARED), mx.bfloat16)
    with pytest.raises(ValueError, match="law"):
        fl.expanded(q, kv, sh, "causal", scale=1.0, law="v9")


@pytest.mark.parametrize("law", ["v1", "v2"])
def test_routing_passes_the_law_and_counts_it(monkeypatch, law):
    seen = []

    def fake(*operands, law="v1", **options):
        seen.append(law)
        return fl.expanded_reference(*operands, **options).astype(operands[0].dtype)

    monkeypatch.setattr(fl, "available", lambda kind, dtype=None: True)
    monkeypatch.setattr(fl, "expanded", fake)
    attn = MultiLatentAttention(kimi_mla_args())
    attn.set_dtype(mx.bfloat16)
    attn.auto_rows, attn.flash, attn.flash_rows, attn.flash_law = 2, "expanded", 2, law
    cache = KVCache()
    attn.policy = "expanded"
    prefix = mx.random.normal((1, 5, 32)).astype(mx.bfloat16)
    attn(prefix, mask=mx.arange(5)[:, None] >= mx.arange(5)[None, :], cache=cache)
    attn.policy = "auto"
    attn(mx.random.normal((1, 4, 32)).astype(mx.bfloat16), mask="causal", cache=cache)
    assert seen == [law]
    assert attn.calls[f"expanded_flash192_law_{law}"] == 1


def test_v2_admission_needs_its_own_probe(monkeypatch):
    monkeypatch.setattr(fl, "available", lambda kind, dtype=None: kind != "expanded_v2")
    attn = MultiLatentAttention(kimi_mla_args())
    attn.flash, attn.flash_law = "expanded", "v2"
    q_rows = mx.zeros((1, 4, 32 * 192), mx.bfloat16)
    assert not attn._flash_admits("expanded", q_rows, "causal")
    assert attn.calls["flash_declined"] == 1
    attn.flash_law = "v1"
    assert attn._flash_admits("expanded", q_rows, "causal")


def test_law_changes_the_policy_binding():
    import json

    a = hashlib.sha256(json.dumps(parse_policy({"mla_flash_law": "v1"}), sort_keys=True).encode())
    b = hashlib.sha256(json.dumps(parse_policy({"mla_flash_law": "v2"}), sort_keys=True).encode())
    assert a.hexdigest() != b.hexdigest()


# --- GPU ----------------------------------------------------------------------


def _gpu():
    if not fl.available("expanded"):
        pytest.skip("Kimi NAX flash needs an M5 GPU")


def _case(rows=72, width=200, seed=0):
    k1, k2, k3 = mx.random.split(mx.random.key(seed), 3)
    q = (mx.random.normal((1, rows, fl.HEADS * (fl.NOPE + fl.SHARED)), key=k1) * 0.5).astype(mx.bfloat16)
    kv = mx.random.normal((1, fl.HEADS, width, fl.NOPE + fl.VDIM), key=k2).astype(mx.bfloat16)
    sh = mx.random.normal((1, 1, width, fl.SHARED), key=k3).astype(mx.bfloat16)
    return q, kv, sh


SCALE = (fl.NOPE + fl.SHARED) ** -0.5


def _rel(a, b):
    a, b = np.array(a.astype(mx.float32), np.float64), np.array(b.astype(mx.float32), np.float64)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


@pytest.mark.parametrize("rows,width", [(72, 200), (16, 16), (130, 900), (1, 37)])
def test_laws_match_the_fp32_law(rows, width):
    _gpu()
    q, kv, sh = _case(rows, width)
    ref = fl.expanded_reference(q, kv, sh, "causal", scale=SCALE)
    errs = {law: _rel(fl.expanded(q, kv, sh, "causal", scale=SCALE, law=law), ref)
            for law in fl.EXPANDED_LAWS}
    assert errs["v1"] < 2e-2
    assert errs["v2"] < 2e-2 and errs["v2"] <= errs["v1"] * 1.05 + 1e-6
    # FP32 output, lse and the no-mask route.
    for law in fl.EXPANDED_LAWS:
        out, lse = fl.expanded(q, kv, sh, "causal", scale=SCALE, law=law, return_lse=True,
                               out_dtype=mx.float32)
        r_out, r_lse = fl.expanded_reference(q, kv, sh, "causal", scale=SCALE, return_lse=True)
        assert _rel(out, r_out) < 2e-2
        np.testing.assert_allclose(np.array(lse), np.array(r_lse), atol=2e-2, rtol=1e-3)
        unmasked = fl.expanded(q, kv, sh, None, scale=SCALE, law=law)
        assert _rel(unmasked, fl.expanded_reference(q, kv, sh, None, scale=SCALE)) < 2e-2


def test_v2_token_major_equals_head_major():
    _gpu()
    q, kvh, sh = _case()
    kvt = mx.contiguous(kvh.transpose(0, 2, 1, 3)).reshape(1, kvh.shape[2], -1)
    a = fl.expanded(q, kvh, sh, "causal", scale=SCALE, law="v2")
    b = fl.expanded(q, kvt, sh, "causal", scale=SCALE, law="v2")
    assert np.array_equal(np.array(a), np.array(b))


@pytest.mark.parametrize("law", ["v1", "v2"])
def test_rows_are_chunk_invariant(law):
    _gpu()
    q, kv, sh = _case(rows=128, width=400)
    full = np.array(fl.expanded(q, kv, sh, "causal", scale=SCALE, law=law))
    prefix = 400 - 128
    for s in range(0, 128, 32):
        part = fl.expanded(q[:, s:s + 32], kv, sh, "causal", scale=SCALE, law=law, offset=prefix + s)
        assert np.array_equal(np.array(part), full[:, s:s + 32])


def _offset_view(a):
    flat = mx.concatenate([mx.zeros((1,), a.dtype), a.reshape(-1)])
    return flat[1:].reshape(a.shape)


def test_v2_misaligned_bases_are_bit_identical():
    _gpu()
    q, kv, sh = _case(seed=3)
    want = np.array(fl.expanded(q, kv, sh, "causal", scale=SCALE, law="v2"))
    for args in ((_offset_view(q), kv, sh), (q, _offset_view(kv), sh), (q, kv, _offset_view(sh))):
        got = fl.expanded(*args, "causal", scale=SCALE, law="v2")
        assert np.array_equal(np.array(got), want)


def test_v2_probe_passes():
    _gpu()
    fl._AVAILABLE.pop(("expanded_v2", mx.bfloat16), None)
    assert fl.available("expanded_v2")
