"""CPU checks for Raon-1-mini's opt-in dense head-row packed absorbed decode.

``mla_decode nax_absorbed_packed`` runs the absorbed NAX latent flash with
(head, verify row) slots packed densely into its 16-row tiles instead of one
threadgroup per row of whole 16-head groups (40 heads run as 48 there). No
GPU here: the kernel is replaced by the NumPy emulation of its partial/
combine contract from ``test_raon1_mini_nax_decode_cpu``, so these tests
cover the host dispatch (no pad heads, grid and threadgroup, combine at the
real head count), the slot mapping arithmetic the Metal source uses, the
source derivation and pin, the policy key and the engine routing. That the
kernel's partials are bitwise the grouped kernel's is checked on the M5 by
its probe and canary (tests/test_raon1_mini_nax_decode_metal.py).
"""

import hashlib

import mlx.core as mx
import pytest

from test_raon1_mini_nax_decode_cpu import (
    _combine_emulation,
    _decode,
    _model,
    _operands,
    _partial_emulation,
    _rel,
    raon40_config,
)

from mlx2.runtime.models import kimi_linear_mla_flash as fl

# sha256 of the derived packed source (``_packed_source``); a change to it, or
# to the grouped source it is derived from, must be deliberate.
PACKED_SHA = "f160307cef20727c59a395a3e6a799f2503716d7f6a78727eb8922984b0ee63c"


@pytest.fixture(autouse=True)
def _cpu_only():
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield
    finally:
        mx.set_default_device(previous)


@pytest.fixture
def emulated(monkeypatch):
    calls = []
    monkeypatch.setattr(fl, "_absorbed_kernels", lambda variant="tiled": (
        _partial_emulation(calls, f"kimi_{variant}"), _combine_emulation(calls)))
    monkeypatch.setattr(fl, "_absorbed_grouped_kernel", lambda: _partial_emulation(calls, "grouped"))
    monkeypatch.setattr(fl, "_absorbed_packed_kernel", lambda: _partial_emulation(calls, "packed"))
    return calls


def test_packed_source_is_derived_and_pinned():
    source = fl._packed_source()
    assert hashlib.sha256(source.encode()).hexdigest() == PACKED_SHA
    grouped = fl._grouped_source()
    # Every edit applied exactly once; nothing else differs.
    rebuilt = grouped
    for old, new in fl._PACKED_EDITS:
        assert grouped.count(old) == 1
        rebuilt = rebuilt.replace(old, new)
    assert rebuilt == source
    import re

    assert "threadgroup_position_in_grid.x" in source and not re.search(r"\bqi\b", source)


def _slots(heads, rows):
    """The Metal source's slot arithmetic, per (threadgroup, head group, fragment row)."""
    groups, blocks = fl.packed_threadgroups(heads, rows)
    nslot = heads * rows
    seen, per_tg = [], {}
    for tgp in range(blocks):
        last_row = (min(nslot, (tgp + 1) * groups * 16) - 1) // heads
        rows_here = set()
        for hg in range(groups):
            for fm in range(8):
                for i in range(2):
                    slot = (tgp * groups + hg) * 16 + fm + 8 * i
                    if slot >= nslot:
                        continue  # padding: masked, never written
                    row, head = slot // heads, slot % heads
                    seen.append((row, head))
                    rows_here.add(row)
        per_tg[tgp] = (last_row, rows_here)
    return groups, blocks, seen, per_tg


@pytest.mark.parametrize("heads", [40, 48, 16, 33])
@pytest.mark.parametrize("rows", [1, 2, 6, 8, 16])
def test_slot_mapping_covers_every_head_row_once(heads, rows):
    groups, blocks, seen, per_tg = _slots(heads, rows)
    assert sorted(seen) == [(r, h) for r in range(rows) for h in range(heads)]
    # Each threadgroup loops to its latest row's causal limit, which covers
    # every slot it holds.
    for last_row, rows_here in per_tg.values():
        assert rows_here and last_row == max(rows_here)
    # No more tiles than the slots need (the grouped kernel runs rows x groups).
    assert blocks * groups * 16 - heads * rows < groups * 16
    if heads == 40 and rows == 6:
        assert (groups, blocks) == (3, 5)  # 15 tiles of work against 18 grouped


@pytest.mark.parametrize("rows,width", [(1, 600), (6, 600), (6, 2050), (16, 600)])
def test_packed_dispatch_has_no_pad_heads_and_matches_grouped(emulated, rows, width):
    q, qs, lat, sh = _operands(1, 40, rows, width + 8, seed=rows)
    scale = (fl.NOPE + fl.SHARED) ** -0.5
    grouped = fl.absorbed(q, qs, lat, sh, "causal", scale=scale, heads=40, split=256, width=width)
    emulated.clear()
    packed = fl.absorbed(q, qs, lat, sh, "causal", scale=scale, heads=40, split=256, width=width,
                         packing="dense")
    call = next(c for c in emulated if c["kernel"] == "packed")
    groups, blocks = fl.packed_threadgroups(40, rows)
    count = -(-width // 256)
    assert call["template"] == [("T", mx.float16), ("H", 40), ("HG", groups)]
    assert call["grid"] == (blocks * groups * 128, count, 1)
    assert call["threadgroup"] == (groups * 128, 1, 1)
    assert call["q"].shape[1] == 40  # no zero-query pad heads
    assert [c["template"] for c in emulated if c["kernel"] == "combine"] == [[("T", mx.float16), ("H", 40)]]
    assert packed.shape == (1, 40, rows, fl.LATENT)
    assert mx.array_equal(packed, grouped).item()  # the same partial contract
    ref = fl.absorbed_reference(q, qs, lat, sh, "causal", scale=scale, heads=40, width=width)
    assert _rel(ref, packed) < 2e-3


def test_packing_refuses_other_kernels():
    q, qs, lat, sh = _operands(1, 32, 1, 64)
    with pytest.raises(ValueError, match="head-group-generic tiled"):
        fl.absorbed(q, qs, lat, sh, "causal", scale=1.0, heads=32, packing="dense")
    q, qs, lat, sh = _operands(1, 40, 1, 64)
    with pytest.raises(ValueError, match="packing must be"):
        fl.absorbed(q, qs, lat, sh, "causal", scale=1.0, heads=40, packing="sparse")


def test_policy_key_and_probe_key():
    from mlx2.adapters import raon1_mini as A
    from mlx2.runtime.models import raon1_mini as R

    assert A.parse_policy({"mla_decode": "nax_absorbed_packed"})["mla_decode"] == "nax_absorbed_packed"
    with pytest.raises(ValueError, match="nax_absorbed_packed"):
        A.parse_policy({"mla_decode": "nax_packed"})
    with pytest.raises(ValueError, match="needs mla auto"):
        A.parse_policy({"mla_decode": "nax_absorbed_packed", "mla": "latent_single"})
    # Off the GPU the probe declines; the packed variant has its own memo key.
    R._ABSORBED_FLASH_OK.pop((40, str(mx.bfloat16), "dense"), None)
    assert R.absorbed_flash_available(40, mx.bfloat16, packing="dense") is False
    assert (40, str(mx.bfloat16), "dense") in R._ABSORBED_FLASH_OK
    R._ABSORBED_FLASH_OK.pop((40, str(mx.bfloat16), "dense"), None)


def test_engaged_packed_decode_equals_the_grouped_lever(monkeypatch, emulated):
    """Raon's real MLA geometry: the packed lever routes every decode row to
    the packed kernel and gives the grouped lever's logits."""
    from mlx2.runtime.models import raon1_mini as R

    seen = []
    monkeypatch.setattr(R, "absorbed_flash_available",
                        lambda heads, dtype, packing=None: seen.append(packing) or True)
    model = _model(raon40_config(), dtype=mx.bfloat16)
    prompt = list(range(3, 40))
    grouped = _decode(model, {"mla_decode": "nax_absorbed"}, prompt)
    emulated.clear()
    packed = _decode(model, {"mla_decode": "nax_absorbed_packed"}, prompt)
    mla = model.layers[3].self_attn
    assert mla.decode_packing == "dense"
    assert mla.calls["latent_flash_packed"] == 3
    assert [c["kernel"] for c in emulated if c["kernel"] in ("packed", "grouped")] == ["packed"] * 6
    assert "dense" in seen
    assert mx.array_equal(grouped, packed).item()


@pytest.mark.parametrize("packing,cases", [(None, 3), ("dense", 6)])
def test_canary_runs_every_case_and_passes_through_the_emulated_kernels(emulated, packing, cases):
    """The probe's canary (review fix): every listed case runs, an empty case
    list never passes, and the packed canary also compares against grouped."""
    from mlx2.runtime.models import raon1_mini as R

    result = R.absorbed_flash_canary(40, mx.float16, packing=packing)
    assert len(result["cases"]) == cases
    assert result["ok"], result
    if packing is not None:
        assert all(case["bitwise_vs_grouped"] for case in result["cases"])
