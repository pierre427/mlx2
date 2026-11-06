# SPDX-License-Identifier: MIT
"""NAX block-sparse QSA prefill kernel law v2 (qwen4_qsa_nax.LAWS).

CPU: the law switch, its policy knob, the APCv2 numerics identity and the
qualification requirement.  GPU (M5 only): v2 and v2_hilo against a float64
reference, chunk invariance, ragged-lane independence, and the scalar
fallback for K/V views that cannot take 16-byte loads.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx2.adapters.flash_next_policy import FlashNextPolicy
from mlx2.qualification import (
    host_gate_record,
    host_gated_not_observed,
    qsa_nax_law_engagement,
    required_feature_checks,
    selected_not_observed_features,
)
from mlx2.runtime import apc_numerics
from mlx2.runtime.models import qwen4_qsa_nax as nax

# --- law switch ---------------------------------------------------------------


# sha256 of qwen4_qsa_nax.SOURCE at 30dcabc6a / f55227951 (before law v2).
V1_SOURCE_SHA256 = "4d02cceebfc6fef26ecbc77651f2ae25fde0c22371206d2218fceebfdc3ae2ff"


def test_v1_source_is_unchanged():
    """Law v1 stays the exact kernel every v1 receipt ran."""
    assert hashlib.sha256(nax.SOURCE.encode()).hexdigest() == V1_SOURCE_SHA256
    assert nax._LAW_KERNELS["v1"] is nax._KERNEL


def test_resolve_law():
    assert nax.resolve_law(None) == "v1"
    assert nax.resolve_law("") == "v1"
    assert nax.resolve_law(" V2 ") == "v2"
    assert nax.resolve_law("v2_hilo") == "v2_hilo"
    with pytest.raises(ValueError, match="MLX_QWEN4_QSA_NAX_LAW"):
        nax.resolve_law("v3")


def test_default_law_is_v1():
    assert nax.DEFAULT_LAW == "v1"
    assert set(nax.LAWS) == {"v1", "v2", "v2_hilo"}


def test_apc_numerics_law_table_matches_the_kernel_module():
    assert apc_numerics.QSA_NAX_LAWS == nax.LAWS
    assert apc_numerics.QSA_NAX_DEFAULT_LAW == nax.DEFAULT_LAW


# --- policy knob --------------------------------------------------------------


def test_policy_default_is_v2_and_pinned():
    """Default v2 since 2026-11-05 (Pierre): absent from receipts as the
    default, always pinned in the profile (the module default is v1)."""
    default = FlashNextPolicy()
    assert default.qsa_nax_law == "v2"
    assert "qsa_nax_law" not in default.as_dict()
    assert default.environment()["MLX_QWEN4_QSA_NAX_LAW"] == "v2"
    v1 = FlashNextPolicy(qsa_nax_law="v1")
    assert v1.as_dict()["qsa_nax_law"] == "v1"
    assert "MLX_QWEN4_QSA_NAX_LAW" not in v1.environment()


@pytest.mark.parametrize("law", ["v2", "v2_hilo"])
def test_policy_law_round_trips_and_reaches_the_environment(law):
    chosen = FlashNextPolicy.from_mapping({"qsa_nax_law": law})
    # The default (v2) is absent from receipts; other laws are recorded.
    assert chosen.as_dict().get("qsa_nax_law") == (None if law == "v2" else law)
    assert FlashNextPolicy.from_mapping(chosen.as_dict()) == chosen
    assert chosen.environment()["MLX_QWEN4_QSA_NAX_LAW"] == law
    from mlx2.runtime.models.import_env import PREFIXES

    assert "MLX_QWEN4_QSA_NAX_LAW".startswith(PREFIXES)


@pytest.mark.parametrize("bad", ["v3", "V2", "", None, 2])
def test_policy_law_validates(bad):
    with pytest.raises(ValueError, match="qsa_nax_law"):
        FlashNextPolicy.from_mapping({"qsa_nax_law": bad})


# --- APCv2 numerics identity --------------------------------------------------


def test_law_enters_the_apc_identity_only_when_not_v1():
    ident = apc_numerics.execution_numerics_identity
    assert ident({}) is None
    assert ident({"MLX_QWEN4_QSA_NAX_LAW": "v1"}) is None
    assert ident({"MLX_QWEN4_QSA_NAX_LAW": ""}) is None
    v2 = ident({"MLX_QWEN4_QSA_NAX_LAW": "v2"})
    hilo = ident({"MLX_QWEN4_QSA_NAX_LAW": "v2_hilo"})
    assert v2["qsa_nax_law"] == "qsa-nax-v2-bf16p"
    assert hilo["qsa_nax_law"] == "qsa-nax-v2-hilop"
    assert v2 != hilo
    assert ident(FlashNextPolicy(qsa_nax_law="v2").environment())["qsa_nax_law"] == (
        "qsa-nax-v2-bf16p"
    )
    with pytest.raises(ValueError):
        ident({"MLX_QWEN4_QSA_NAX_LAW": "v9"})


# --- qualification ------------------------------------------------------------

GEOMETRY = {"budget": 2048, "compress_ratio": 4, "head_dim": 128, "n_heads": 4}


def _settings(policy):
    return {
        "adapter_policy": policy.as_dict(),
        "environment": policy.environment(),
        "max_context": 262144,
        "max_lanes": 4,
        "mtp": False,
        "speculation": "ordinary",
        "prefill_step": 8192,
        "qsa_indexer": dict(GEOMETRY),
    }


def test_default_law_is_required_and_v1_is_not():
    assert "feature_qsa_nax_law" in required_feature_checks(_settings(FlashNextPolicy()))
    assert "feature_qsa_nax_law" not in required_feature_checks(
        _settings(FlashNextPolicy(qsa_nax_law="v1"))
    )


@pytest.mark.parametrize("law", ["v2", "v2_hilo"])
def test_candidate_law_is_required(law):
    checks = required_feature_checks(_settings(FlashNextPolicy(qsa_nax_law=law)))
    assert "feature_qsa_nax_prefill" in checks
    assert "feature_qsa_nax_law" in checks


def test_candidate_law_unreachable_says_why():
    policy = FlashNextPolicy(qsa_nax_law="v2")
    settings = _settings(policy)
    settings["max_context"] = 4096
    assert "feature_qsa_nax_law" not in required_feature_checks(settings)
    reasons = selected_not_observed_features(settings)
    assert reasons.get("feature_qsa_nax_law", reasons.get("qsa_nax_law"))


def _status(law, engagements, **extra):
    return {"qsa_nax_prefill": {"law": law, "engagements": engagements,
                                "counts": {}, **extra}}


def test_law_engagement_is_a_delta_under_a_candidate_law():
    assert qsa_nax_law_engagement(_status("v2", 7), _status("v2", 3)) == 4
    assert qsa_nax_law_engagement(_status("v2_hilo", 5), _status("v2_hilo", 0)) == 5
    # A v1 run never passes for a candidate law.
    assert qsa_nax_law_engagement(_status("v1", 9), _status("v1", 0)) == 0
    # A law that changed between snapshots is not this run's evidence.
    assert qsa_nax_law_engagement(_status("v2", 9), _status("v2_hilo", 0)) == 0
    assert qsa_nax_law_engagement({}, {}) == 0


def test_law_is_host_gated_with_the_prefill_kernel():
    final = _status("v2", 0, nax_host=False, device_name="Apple M3 Max")
    out = host_gated_not_observed(final, _status("v2", 0))
    assert out["qsa_nax_law"] == out["qsa_nax_prefill"]
    assert host_gate_record("qsa_nax_law", final) == {
        "nax_host": False, "device_name": "Apple M3 Max",
    }
    v1 = _status("v1", 0, nax_host=False, device_name="Apple M3 Max")
    assert "qsa_nax_law" not in host_gated_not_observed(v1, _status("v1", 0))


# --- GPU ----------------------------------------------------------------------


def _gpu():
    if not mx.metal.is_available():
        pytest.skip("NAX needs Metal")
    if not nax.nax_kernel_available():
        pytest.skip("NAX kernel unavailable on this device")


def _case(*, batch=1, heads=8, kv_heads=2, dim=256, length=48, total=160,
          topk=12, left_pad=(0,), seed=0):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((batch, heads, length, dim)).astype(np.float32)
    k = rng.standard_normal((batch, kv_heads, total, dim)).astype(np.float32)
    v = rng.standard_normal((batch, kv_heads, total, dim)).astype(np.float32)
    bs = nax.BLOCK
    u_width = max(8, (topk + 1 + 7) // 8 * 8)
    ids = np.zeros((batch, length, u_width), np.uint32)
    counts = np.zeros((batch, length), np.uint32)
    n_sel = np.zeros((batch, length), np.uint32)
    q_pos = np.zeros((batch, length), np.int32)
    pads = np.array(left_pad if len(left_pad) == batch else left_pad * batch, np.int32)
    for b in range(batch):
        start = total - int(pads[b]) - length
        for r in range(length):
            pos = start + r
            q_pos[b, r] = pos
            complete = (pos + 1) // bs
            avail = np.arange(complete)
            chosen = np.sort(rng.choice(avail, size=min(topk, complete), replace=False))
            n = len(chosen)
            ids[b, r, :n] = chosen
            n_sel[b, r] = n
            tail = pos // bs
            if n and chosen[-1] == tail:
                counts[b, r] = n
            else:
                ids[b, r, n] = tail
                counts[b, r] = n + 1
    to = lambda a: mx.array(a).astype(mx.bfloat16)  # noqa: E731
    return dict(
        q=to(q), k=to(k), v=to(v), ids=mx.array(ids), counts=mx.array(counts),
        n_sel=mx.array(n_sel), q_pos=mx.array(q_pos), left_pad=mx.array(pads),
        u_width=u_width, total=total, kv_heads=kv_heads, scale=dim ** -0.5,
    )


def _run(c, law, **override):
    args = {**c, **override}
    return nax.nax_qsa_attention(
        args["q"], args["k"], args["v"], args["ids"], args["counts"],
        args["n_sel"], args["q_pos"], args["left_pad"], scale=args["scale"],
        u_width=args["u_width"], total=args["total"], n_kv_heads=args["kv_heads"],
        law=law,
    )


def _reference(c):
    q = np.array(c["q"].astype(mx.float32), np.float64)
    k = np.array(c["k"].astype(mx.float32), np.float64)
    v = np.array(c["v"].astype(mx.float32), np.float64)
    ids, counts, n_sel = (np.array(c[n]) for n in ("ids", "counts", "n_sel"))
    q_pos, pads = np.array(c["q_pos"]), np.array(c["left_pad"])
    batch, heads, length, dim = q.shape
    gqa = heads // c["kv_heads"]
    out = np.zeros_like(q)
    bs = nax.BLOCK
    for b in range(batch):
        for r in range(length):
            pos = int(q_pos[b, r])
            complete = (pos + 1) // bs * bs
            keys = []
            for u in range(int(counts[b, r])):
                for j in range(bs):
                    kl = int(ids[b, r, u]) * bs + j
                    if kl <= pos and (u < int(n_sel[b, r]) or kl >= complete):
                        keys.append(int(pads[b]) + kl)
            keys = np.array(sorted(set(keys)))
            for h in range(heads):
                kv = h // gqa
                s = k[b, kv, keys] @ q[b, h, r] * c["scale"]
                p = np.exp(s - s.max())
                out[b, h, r] = (p / p.sum()) @ v[b, kv, keys]
    return out


def _rel(got, ref):
    got = np.array(got, np.float64)
    return float(np.linalg.norm(got - ref) / np.linalg.norm(ref))


@pytest.mark.parametrize("dim", [256, 128])
@pytest.mark.parametrize("kv_heads,heads", [(2, 24), (4, 24), (1, 16)])
def test_laws_match_the_float64_reference(dim, kv_heads, heads):
    _gpu()
    c = _case(heads=heads, kv_heads=kv_heads, dim=dim)
    ref = _reference(c)
    errs = {law: _rel(_run(c, law), ref) for law in nax.LAWS}
    assert errs["v1"] < 2e-2
    # v2 keeps v1's bf16 P: same error class.
    assert errs["v2"] < 2e-2 and errs["v2"] <= errs["v1"] * 1.05 + 1e-6
    # hi/lo P removes the P rounding: far below either.
    assert errs["v2_hilo"] < errs["v1"] / 20


@pytest.mark.parametrize("law", ["v2", "v2_hilo"])
def test_rows_are_chunk_invariant(law):
    _gpu()
    c = _case(length=64)
    full = np.array(_run(c, law))
    for lo in range(0, 64, 16):
        part = {
            **c,
            "q": c["q"][:, :, lo:lo + 16],
            "ids": c["ids"][:, lo:lo + 16],
            "counts": c["counts"][:, lo:lo + 16],
            "n_sel": c["n_sel"][:, lo:lo + 16],
            "q_pos": c["q_pos"][:, lo:lo + 16],
        }
        assert np.array_equal(np.array(_run(part, law)), full[:, :, lo:lo + 16])


@pytest.mark.parametrize("law", ["v2", "v2_hilo"])
def test_ragged_lanes_equal_their_single_lane_runs(law):
    _gpu()
    c = _case(batch=2, left_pad=(0, 12), seed=3)
    both = np.array(_run(c, law))
    for b in range(2):
        one = {
            **c,
            **{n: c[n][b:b + 1] for n in ("q", "k", "v", "ids", "counts", "n_sel", "q_pos")},
            "left_pad": c["left_pad"][b:b + 1],
        }
        assert np.array_equal(np.array(_run(one, law)), both[b:b + 1])


@pytest.mark.parametrize("law", ["v2", "v2_hilo"])
def test_scalar_fallback_layouts_are_bit_identical(law):
    """Views that cannot take 16-byte loads take the in-kernel scalar path,
    with the same arithmetic: a strided cache view, a transposed view (non-unit
    last stride) and a misaligned base all equal the contiguous result."""
    _gpu()
    c = _case(seed=5)
    want = np.array(_run(c, law))
    (batch, kvh, total, dim) = c["k"].shape
    # Cache-style view: a larger allocation sliced along T.
    big_k = mx.zeros((batch, kvh, total + 40, dim), mx.bfloat16)
    big_v = mx.zeros((batch, kvh, total + 40, dim), mx.bfloat16)
    big_k[:, :, :total] = c["k"]
    big_v[:, :, :total] = c["v"]
    sliced = _run(c, law, k=big_k[:, :, :total], v=big_v[:, :, :total])
    assert np.array_equal(np.array(sliced), want)
    # Non-unit last stride: stored [B, H, D, T], viewed as [B, H, T, D].
    kt = mx.contiguous(c["k"].transpose(0, 1, 3, 2)).transpose(0, 1, 3, 2)
    vt = mx.contiguous(c["v"].transpose(0, 1, 3, 2)).transpose(0, 1, 3, 2)
    assert np.array_equal(np.array(_run(c, law, k=kt, v=vt)), want)
    # Misaligned base: one element of padding in front of every row.
    kp = mx.concatenate([mx.zeros((batch, kvh, total, 1), mx.bfloat16), c["k"]], -1)[..., 1:]
    vp = mx.concatenate([mx.zeros((batch, kvh, total, 1), mx.bfloat16), c["v"]], -1)[..., 1:]
    assert np.array_equal(np.array(_run(c, law, k=kp, v=vp)), want)


# --- review follow-ups (fresh-context Claude review 2026-11-05) ---------------


def test_law_engagement_binds_the_expected_law():
    assert qsa_nax_law_engagement(_status("v2", 4), _status("v2", 0), expected="v2") == 4
    assert qsa_nax_law_engagement(_status("v2", 4), _status("v2", 0), expected="v2_hilo") == 0


def test_candidate_law_refused_on_an_admitting_host_is_not_waived():
    """On an M5 the probe runs the process law: a refused v2 is that law
    failing, so neither feature gets a host-gated waiver."""
    final = _status("v2", 0, nax_host=True, device_name="Apple M5 Max",
                    counts={"kernel_unavailable": 3})
    out = host_gated_not_observed(final, _status("v2", 0))
    assert "qsa_nax_prefill" not in out and "qsa_nax_law" not in out
    # v1 keeps its existing canary waiver.
    v1 = _status("v1", 0, nax_host=True, device_name="Apple M5 Max",
                 counts={"kernel_unavailable": 3})
    assert "qsa_nax_prefill" in host_gated_not_observed(v1, _status("v1", 0))


def test_candidate_law_with_the_kernel_off_says_why():
    policy = FlashNextPolicy(qsa_nax_law="v2")
    settings = _settings(policy)
    settings["environment"] = {**settings["environment"], "MLX_QWEN4_QSA_NAX_KERNEL": "0"}
    assert "feature_qsa_nax_law" not in required_feature_checks(settings)
    reasons = selected_not_observed_features(settings)
    assert reasons.get("feature_qsa_nax_law", reasons.get("qsa_nax_law"))


def test_candidate_law_refuses_four_byte_elements():
    c = {"q": mx.zeros((1, 2, 1, 256), mx.float32)}
    with pytest.raises(ValueError, match="bf16/fp16"):
        nax.nax_qsa_attention(c["q"], c["q"], c["q"], mx.zeros((1, 1, 8), mx.uint32),
                              mx.ones((1, 1), mx.uint32), mx.zeros((1, 1), mx.uint32),
                              mx.zeros((1, 1), mx.int32), mx.zeros((1,), mx.int32),
                              scale=1.0, u_width=8, total=1, n_kv_heads=1, law="v2")


def _offset_view(a):
    """``a`` as a row-contiguous view whose base is one element off 16-byte
    alignment (row strides stay multiples of 8)."""
    flat = mx.concatenate([mx.zeros((1,), a.dtype), a.reshape(-1)])
    return flat[1:].reshape(a.shape)


@pytest.mark.parametrize("law", ["v2", "v2_hilo"])
def test_misaligned_bases_alone_are_bit_identical(law):
    _gpu()
    c = _case(seed=7)
    want = np.array(_run(c, law))
    got_kv = _run(c, law, k=_offset_view(c["k"]), v=_offset_view(c["v"]))
    assert np.array_equal(np.array(got_kv), want)
    got_q = _run(c, law, q=_offset_view(c["q"]))
    assert np.array_equal(np.array(got_q), want)
