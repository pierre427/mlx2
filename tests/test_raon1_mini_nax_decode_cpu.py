"""CPU checks for Raon-1-mini's opt-in absorbed NAX decode (mla_decode).

No GPU: the Metal kernels are replaced by NumPy emulations of their partial /
combine contract (all heads at once, so kernel-internal head-group indexing is
NOT covered here -- only the Metal canary checks that), so the host dispatch (zero-query head padding to whole
16-head groups, grid and threadgroup sizing, combine at the padded count,
dropping the pad heads) is checked against ``absorbed_reference``.  Kimi's
32-head kernel is pinned: source hashes, kernel name, template, grid and
threadgroup.  The real kernel's canary is tests/test_raon1_mini_nax_decode_metal.py.
"""

import hashlib
from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_map

from mlx2.runtime.models import kimi_linear_mla_flash as fl

from test_raon1_mini_cpu import tiny_config  # noqa: E402

# sha256 of the kernel sources at d694e50e6 (before the head-group-generic
# kernel was added); Kimi's 32-head path must keep compiling these bytes.
PINNED = {"_ABSORBED": "d814241e81f847920c16f9c1ad4091f6e4810a5e42faec3e58b7497e97d94e99",
          "_COMBINE": "c1fc46413577afef30e6519402a1db9aac5df571211016b5b1de4676a9a62b99",
          "_HEADER": "51659b5f123b9926b8134c4bac53acd36ffe404a02db120952a72200694a2deb"}


@pytest.fixture(autouse=True)
def _cpu_only():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


# ------------------------------------------------------------ kernel emulation

def _partial_emulation(calls, name):
    """The partial kernel's contract in NumPy: per (b, row, split, head) the
    log2-domain max m, sum l and unnormalized o over the split's visible keys."""
    def kernel(*, inputs, template, grid, threadgroup, output_shapes, output_dtypes):
        q, qs, lat, sh, params, scalep, starts = (np.array(z.astype(mx.float32)) if z.dtype != mx.int32
                                                  else np.array(z) for z in inputs)
        tmpl = dict(template)
        calls.append({"kernel": name, "template": template, "grid": grid, "threadgroup": threadgroup,
                      "q": q, "qs": qs})
        h = tmpl["H"]
        tq, s, off, causal, split, nsplit, _ = (int(v) for v in params)
        b = q.shape[0]
        assert q.shape[1] == h and qs.shape[1] == h
        m = np.full((b, tq, nsplit, h), -np.inf, np.float32)
        l = np.zeros((b, tq, nsplit, h), np.float32)
        o = np.zeros((b, tq, nsplit, h, fl.LATENT), np.float32)
        for bb in range(b):
            for qi in range(tq):
                klim = min(s, off + qi + 1) if causal else s
                kstart = min(max(int(starts[bb]), 0), klim)
                for sp in range(nsplit):
                    kbeg = min(max(sp * split, kstart), klim)
                    kend = max(kbeg, min(sp * split + split, klim))
                    if kend <= kbeg:
                        continue
                    keys = lat[bb, kbeg:kend]
                    score = (q[bb, :, qi] @ keys.T + qs[bb, :, qi] @ sh[bb, kbeg:kend].T) * scalep[0]
                    top = score.max(-1)
                    p = np.exp2(score - top[:, None])
                    m[bb, qi, sp], l[bb, qi, sp] = top, p.sum(-1)
                    o[bb, qi, sp] = p @ keys
        return [mx.array(m.reshape(-1)), mx.array(l.reshape(-1)), mx.array(o.reshape(-1, fl.LATENT))]
    return kernel


def _combine_emulation(calls):
    def kernel(*, inputs, template, grid, threadgroup, output_shapes, output_dtypes):
        m, l, o, params = (np.array(z) for z in inputs)
        tmpl = dict(template)
        calls.append({"kernel": "combine", "template": template, "grid": grid})
        h, tq, nsplit = tmpl["H"], int(params[0]), int(params[5])
        b = output_shapes[0][0]
        m = m.reshape(b, tq, nsplit, h)
        l = l.reshape(b, tq, nsplit, h)
        o = o.reshape(b, tq, nsplit, h, fl.LATENT)
        top = m.max(2, keepdims=True)
        w = np.where(np.isfinite(m), np.exp2(m - np.where(np.isfinite(top), top, 0)), 0)
        total = (l * w).sum(2)
        acc = (o * w[..., None]).sum(2)
        out = np.where(total[..., None] > 0, acc / np.where(total > 0, total, 1)[..., None], 0)
        return [mx.array(out.transpose(0, 2, 1, 3)).astype(tmpl["T"])]
    return kernel


@pytest.fixture
def emulated(monkeypatch):
    calls = []
    monkeypatch.setattr(fl, "_absorbed_kernels",
                        lambda variant="tiled": (_partial_emulation(calls, f"kimi_{variant}"),
                                                 _combine_emulation(calls)))
    monkeypatch.setattr(fl, "_absorbed_grouped_kernel", lambda: _partial_emulation(calls, "grouped"))
    return calls


def _operands(b, heads, rows, cap, seed=0, dtype=mx.float16):
    rng = np.random.default_rng(seed)

    def arr(shape, std):
        return mx.array(rng.normal(0, std, shape).astype(np.float32)).astype(dtype)
    return (arr((b, heads, rows, fl.LATENT), .05), arr((b, heads, rows, fl.SHARED), .5),
            arr((b, 1, cap, fl.LATENT), 1), arr((b, 1, cap, fl.SHARED), 1))


def _rel(ref, got):
    ref, got = np.array(ref.astype(mx.float32)), np.array(got.astype(mx.float32))
    return float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-20))


# ------------------------------------------------------------ Kimi pin

def test_kimi_kernel_sources_are_pinned():
    for name, digest in PINNED.items():
        assert hashlib.sha256(getattr(fl, name).encode()).hexdigest() == digest, name
    assert fl.HEADS == 32 and fl.head_groups(32) == 2


def test_kimi_32_head_kernel_objects_are_unchanged(monkeypatch):
    built = []

    def recorder(**kwargs):
        built.append(kwargs)
        return object()
    monkeypatch.setattr(mx.fast, "metal_kernel", recorder)
    fl._absorbed_kernels.cache_clear()
    fl._absorbed_grouped_kernel.cache_clear()
    try:
        fl._absorbed_kernels("tiled")
        fl._absorbed_grouped_kernel()
    finally:
        fl._absorbed_kernels.cache_clear()
        fl._absorbed_grouped_kernel.cache_clear()
    partial, combine, grouped = built
    assert partial["name"] == "mlx2_kimi_mla_latent_flash_tiled_v1" and partial["source"] is fl._ABSORBED
    assert partial["header"] is fl._HEADER
    assert combine["name"] == "mlx2_kimi_mla_latent_combine_v1" and combine["source"] is fl._COMBINE
    assert grouped["name"] != partial["name"]


def test_grouped_source_differs_only_in_head_group_arrays():
    base, grouped = fl._ABSORBED.splitlines(), fl._grouped_source().splitlines()
    assert len(base) == len(grouped)
    changed = [(a.strip(), b.strip()) for a, b in zip(base, grouped) if a != b]
    assert changed == [("threadgroup float red_max[2][4][16];", "threadgroup float red_max[HG][4][16];"),
                       ("threadgroup float red_sum[2][4][16];", "threadgroup float red_sum[HG][4][16];"),
                       ("threadgroup float p_tile[2][BK / 16][32 * 8];",
                        "threadgroup float p_tile[HG][BK / 16][32 * 8];")]


def test_kimi_32_heads_dispatch_is_the_original_kernel(emulated):
    q, qs, lat, sh = _operands(1, 32, 2, 300)
    scale = (fl.NOPE + fl.SHARED) ** -0.5
    out = fl.absorbed(q, qs, lat, sh, "causal", scale=scale, split=128)
    partial = [c for c in emulated if c["kernel"] != "combine"]
    assert [c["kernel"] for c in partial] == ["kimi_tiled"]
    assert partial[0]["template"] == [("T", mx.float16), ("H", 32)]
    assert partial[0]["grid"] == (2 * 256, 3, 1) and partial[0]["threadgroup"] == (256, 1, 1)
    assert out.shape == (1, 32, 2, fl.LATENT)
    ref = fl.absorbed_reference(q, qs, lat, sh, "causal", scale=scale)
    assert _rel(ref, out) < 2e-3


# ------------------------------------------------------------ grouped dispatch

def test_head_groups():
    assert [fl.head_groups(h) for h in (1, 16, 17, 32, 40, 48, 49, 0)] == [1, 1, 2, 2, 3, 3, None, None]


@pytest.mark.parametrize("heads,rows,starts", [(40, 1, None), (40, 5, None), (40, 3, [0, 211]), (48, 1, None),
                                               (24, 2, None)])
def test_grouped_dispatch_pads_and_drops_heads(emulated, heads, rows, starts):
    b = 1 if starts is None else len(starts)
    q, qs, lat, sh = _operands(b, heads, rows, 640, seed=heads + rows)
    width = 600
    scale = (fl.NOPE + fl.SHARED) ** -0.5
    starts = None if starts is None else mx.array(starts, mx.int32)
    out = fl.absorbed(q, qs, lat, sh, "causal", scale=scale, heads=heads, split=256, width=width,
                      starts=starts)
    groups = -(-heads // 16)
    padded = groups * 16
    call = next(c for c in emulated if c["kernel"] == "grouped")
    assert not any(c["kernel"].startswith("kimi") for c in emulated)
    assert call["template"] == [("T", mx.float16), ("H", padded), ("HG", groups)]
    assert call["grid"] == (rows * groups * 128, 3, b) and call["threadgroup"] == (groups * 128, 1, 1)
    assert call["q"].shape[1] == padded and not call["q"][:, heads:].any() and not call["qs"][:, heads:].any()
    assert [c["template"] for c in emulated if c["kernel"] == "combine"] == [[("T", mx.float16), ("H", padded)]]
    assert out.shape == (b, heads, rows, fl.LATENT)
    ref = fl.absorbed_reference(q, qs, lat, sh, "causal", scale=scale, heads=heads, width=width, starts=starts)
    assert _rel(ref, out) < 2e-3


def test_grouped_dispatch_refuses_other_variants_and_counts():
    q, qs, lat, sh = _operands(1, 40, 1, 64)
    with pytest.raises(ValueError, match="tiled"):
        fl.absorbed(q, qs, lat, sh, "causal", scale=1.0, heads=40, variant="reuse")
    q, qs, lat, sh = _operands(1, 64, 1, 64)
    with pytest.raises(ValueError, match="geometry"):
        fl.absorbed(q, qs, lat, sh, "causal", scale=1.0, heads=64)


# ------------------------------------------------------------ Raon routing

def raon40_config():
    # Raon's real MLA geometry (40 heads, qk 128+64, v 128, latent 512) on a
    # tiny dense trunk (CPU GatherMM is FP32-only, so no MoE in BF16).
    return tiny_config(first_k_dense_replace=99, num_attention_heads=40, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128,
                       kv_lora_rank=512)


def _model(config, seed=0, scale=0.05, dtype=None):
    from mlx2.runtime.models import raon1_mini as R

    model = R.Model(R.ModelArgs.from_dict(config))
    mx.random.seed(seed)
    model.update(tree_map(lambda p: mx.random.normal(p.shape).astype(p.dtype) * scale, model.parameters()))
    if dtype is not None:
        model.set_dtype(dtype)
    return model


def test_policy_key_is_opt_in_and_validated():
    from mlx2.adapters import raon1_mini as A

    assert A.parse_policy({})["mla_decode"] == "off"
    assert A.parse_policy({"mla_decode": "nax_absorbed"})["mla_decode"] == "nax_absorbed"
    assert "mla_decode" in A.NUMERIC_POLICY_KEYS
    for bad in ("on", "all", None):
        with pytest.raises(ValueError, match="mla_decode"):
            A.parse_policy({"mla_decode": bad})
    with pytest.raises(ValueError, match="mla auto"):
        A.parse_policy({"mla_decode": "nax_absorbed", "mla": "latent_single"})


def _decode(model, policy, prompt, steps=3):
    from mlx2.adapters import raon1_mini as A

    model.configure(A.parse_policy(policy))
    cache = model.make_cache()
    rows = [model(mx.array([prompt]), cache)[0, -1]]
    nxt = prompt[-1]
    for _ in range(steps):
        rows.append(model(mx.array([[nxt]]), cache)[0, -1])
    return mx.stack(rows).astype(mx.float32)


def test_off_and_other_head_counts_decline_bitwise(monkeypatch):
    from mlx2.runtime.models import raon1_mini as R

    monkeypatch.setattr(R, "absorbed_flash_available", lambda heads, dtype: True)
    # 4 heads; dense layers only (CPU GatherMM is FP32-only).
    model = _model(tiny_config(first_k_dense_replace=99), dtype=mx.bfloat16)
    prompt = list(range(1, 30))
    base = _decode(model, {}, prompt)
    mla = model.layers[3].self_attn
    assert mla.decode_nax == "off" and mla.calls.get("decode_nax_declined", 0) == 0
    opted = _decode(model, {"mla_decode": "nax_absorbed"}, prompt)
    assert mla.decode_nax == "nax_absorbed"
    assert mla.calls["decode_nax_declined_heads"] > 0 and mla.calls.get("decode_nax_absorbed", 0) == 0
    assert mla.calls.get("latent_flash", 0) == 0
    assert bool(mx.array_equal(base, opted))


def test_admission_reasons(monkeypatch):
    from mlx2.runtime.models import raon1_mini as R

    model = _model(tiny_config())
    mla = model.layers[3].self_attn
    mla.args = replace(mla.args, **{k: v for k, v in raon40_config().items()
                                    if k in ("num_attention_heads", "qk_nope_head_dim", "qk_rope_head_dim",
                                             "v_head_dim", "kv_lora_rank")})
    mla.decode_nax = "nax_absorbed"
    probe = {"ok": True}
    monkeypatch.setattr(R, "absorbed_flash_available", lambda heads, dtype: probe["ok"])
    row = mx.zeros((1, 1, 40 * 192), mx.bfloat16)
    assert mla._flash_admits("absorbed", row, "causal") is True
    assert mla._flash_admits("absorbed", row, None) is True
    assert mla._flash_admits("absorbed", mx.zeros((1, 17, 40 * 192), mx.bfloat16), "causal") is False
    assert mla._flash_admits("absorbed", row, "causal", starts=mx.array([0], mx.int32)) is False
    assert mla._flash_admits("absorbed", row, mx.ones((1, 1), mx.bool_)) is False
    assert mla._flash_admits("absorbed", row.astype(mx.float32), "causal") is False
    probe["ok"] = False
    assert mla._flash_admits("absorbed", row, "causal") is False
    assert mla.calls["decode_nax_absorbed"] == 2
    assert {k: v for k, v in mla.calls.items() if k.startswith("decode_nax_declined_")} == {
        "decode_nax_declined_rows": 1, "decode_nax_declined_padded_batch": 1, "decode_nax_declined_mask": 1,
        "decode_nax_declined_dtype": 1, "decode_nax_declined_probe": 1}
    assert mla.calls["decode_nax_declined"] == 5
    # The decode lever never admits the expanded kernel, and its absence
    # keeps the prior absorbed behaviour (declined without a count).
    assert mla._flash_admits("expanded", mx.zeros((1, 64, 40 * 192), mx.bfloat16), "causal") is False
    mla.decode_nax = "off"
    before = dict(mla.calls)
    assert mla._flash_admits("absorbed", row, "causal") is False
    assert dict(mla.calls) == before
    mla.args = replace(mla.args, num_attention_heads=32)
    mla.decode_nax = "nax_absorbed"
    assert mla._flash_admits("absorbed", row, "causal") is False
    assert mla.calls["decode_nax_declined_heads"] == 1


def test_probe_declines_off_gpu():
    from mlx2.runtime.models import raon1_mini as R

    R._ABSORBED_FLASH_OK.pop((40, str(mx.bfloat16)), None)
    assert R.absorbed_flash_available(40, mx.bfloat16) is False
    R._ABSORBED_FLASH_OK.pop((40, str(mx.bfloat16)), None)


def test_engaged_decode_at_40_heads_matches_default(monkeypatch, emulated):
    """Raon's real MLA geometry through the host contract (emulated kernels):
    decode rows take the padded 3-group dispatch and match latent_single."""
    from mlx2.runtime.models import raon1_mini as R

    monkeypatch.setattr(R, "absorbed_flash_available", lambda heads, dtype: True)
    model = _model(raon40_config(), dtype=mx.bfloat16)
    prompt = list(range(3, 40))
    base = _decode(model, {}, prompt)
    mla = model.layers[3].self_attn
    assert mla.calls.get("latent_flash", 0) == 0
    opted = _decode(model, {"mla_decode": "nax_absorbed"}, prompt)
    assert mla.calls["decode_nax_absorbed"] == 3 and mla.calls["latent_flash"] == 3
    # The 37-row prompt chunk is expanded and never reaches the decode gate.
    assert mla.calls.get("decode_nax_declined", 0) == 0
    grouped = [c for c in emulated if c["kernel"] == "grouped"]
    assert len(grouped) == 3 * 2  # three decode rows x two MLA layers
    assert all(c["template"][1:] == [("H", 48), ("HG", 3)] for c in grouped)
    assert bool(mx.array_equal(base[0], opted[0]))  # prompt logits: same law
    rel = float(mx.sqrt(((opted - base) ** 2).sum()) / mx.sqrt((base ** 2).sum()))
    assert rel < 2e-2, rel


def test_receipt_reports_mla_decode_engagement():
    from mlx2.adapters import raon1_mini as A

    status = {"rda": {}, "mla": {"decode_nax_absorbed": 14, "decode_nax_declined": 3,
                                 "decode_nax_declined_rows": 2, "decode_nax_declined_probe": 1},
              "effort_alpha": 0.0}
    stub = SimpleNamespace(model=SimpleNamespace(mechanism_status=lambda: status, effort_alpha=lambda e: 0.0),
                           policy=A.parse_policy({"mla_decode": "nax_absorbed"}))
    receipt = A.RaonAdapter.capability_receipts(stub)["mla_decode"]
    assert receipt == {"implemented": True,
                       "scope": "absorbed NAX latent flash for decode/verify rows, 40 heads (3 head groups, 8 pad heads)",
                       "selected": True, "mode": "nax_absorbed", "qualified": False, "observed_used": True,
                       "calls": 14, "declined": 3, "declined_by_reason": {"rows": 2, "probe": 1}}
