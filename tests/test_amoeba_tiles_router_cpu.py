"""Amoeba router tiles (runtime/amoeba_tiles.py, gate law "router"): per-request open sets, closed
tiles skipped (CPU).

Synthetic tiny quantized SwiGLU layers (the fused suite's fixture). What is shown, both layouts:

* (a) every router tile open = the same tiles hard-gated open = the dense source MLP over a full
  partition, bit for bit;
* (b) a closed router tile = core + the open tiles only = the hard-gated tile set with that box
  masked shut, bit for bit; a closed tile's projections are never called;
* (c) a mixed batch (rows with different open sets, partial tiles gathered to their rows) gives
  each row the bits of that row run alone (CPU kernels; on Metal a gathered row count may pick
  another kernel, so there it is a near tie until the GPU check);
* (d) request resolution: OR rule over topics, unknown tile/topic -> TileNotLoaded (409), fail
  closed on spec routes, APCv2 scope per open set;
* (e) counters (rows computed vs skipped per tile), per-forward open-row share, receipts, the
  generator row seam and its geometry check.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from test_amoeba_tiles_fused_cpu import (  # noqa: F401  (fx is a fixture)
    L,
    N,
    _many_boxes,
    _swiglu_decl,
    _x,
    fx,
)
from test_amoeba_tiles_spec_cpu import _np

LAYOUTS = ["unfused", "fused"]


def _write(fx, name, core, boxes, gates, router=None, tile_bits=None):
    fx.export.write_tiles(fx.tmp / name, fx.src, "fp-test", "fake", core, [0], boxes, gates, L, N, print,
                          name, bf16_src=fx.bf16 if tile_bits else None, tile_bits=tile_bits, router=router)
    return fx.tmp / name / "manifest.json"


def _routed(boxes, topics=None):
    """Every box on the router law; box j opens on topic ``t{j % 3}`` and on its own id."""
    return {n: (topics or {}).get(n, [f"t{j % 3}", n]) for j, n in enumerate(boxes)}


def _install(fx, manifest, layout, model=None, route="ordinary", partial_rows="mask"):
    from mlx2.runtime.amoeba_tiles import install
    layers = fx.fresh()
    tiles = install(model, layers, manifest, _swiglu_decl(), source_fingerprint="fp-test", layout=layout,
                    route=route, router_partial_rows=partial_rows)
    return layers, tiles


def _run(layers, tiles, x, rows):
    """Tiled layers' outputs with per-row open sets bound (None = unbound forward)."""
    router = tiles.router
    if rows is not None:
        for i, opened in enumerate(rows):
            if opened is not None:
                router.bind_uid(("u", i), opened)
        router.bind_rows([("u", i) for i in range(len(rows))])
    try:
        return [_np(layers[l].mlp(x)) for l in range(1, L)]
    finally:
        router.clear_rows()
        router.uids.clear()
        router.stats.clear()


@pytest.mark.parametrize("layout", LAYOUTS)
def test_all_open_router_equals_hard_gates_open_and_the_source_mlp(fx, layout):
    from mlx2.runtime.models.activations import swiglu
    core, boxes, gates = _many_boxes(partition=True)
    assert np.all(core + sum(boxes.values()) == 1)
    forced = {n: (w * 0.0, np.full_like(b, 50.0)) for n, (w, b) in gates.items()}
    r_layers, r_tiles = _install(fx, _write(fx, f"r_{layout}", core, boxes, {}, router=_routed(boxes)), layout)
    g_layers, _ = _install(fx, _write(fx, f"g_{layout}", core, boxes, forced), layout)
    source = fx.fresh()
    x = _x()
    for opened in (None, frozenset(boxes)):                  # unbound, and bound with every tile open
        ys = _run(r_layers, r_tiles, x, None if opened is None else [opened, None])
        for l, y in zip(range(1, L), ys):
            src = source[l].mlp
            assert np.array_equal(y, _np(g_layers[l].mlp(x))), l
            assert np.array_equal(y, _np(src.down_proj(swiglu(src.gate_proj(x), src.up_proj(x))))), l
    ident = r_tiles.identity
    assert ident["router_law"] == "router" and set(ident["router_tiles"]) == set(boxes)
    if layout == "fused":
        groups = r_tiles.layout_detail()["layers"]["1"]["groups"]
        assert [g.get("router_tile") for g in groups] == [None] + list(boxes)


def test_fused_router_equals_unfused_router(fx):
    core, boxes, _ = _many_boxes(overlap=True)
    manifest = _write(fx, "fu", core, boxes, {}, router=_routed(boxes))
    un, un_t = _install(fx, manifest, "unfused")
    fu, fu_t = _install(fx, manifest, "fused")
    x = _x()
    for rows in (None, [frozenset({"b0", "b1"}), frozenset()]):
        for a, b in zip(_run(un, un_t, x, rows), _run(fu, fu_t, x, rows)):
            assert np.array_equal(a, b)


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("bits", [None, {"core": None, "b0": None, "b1": None, "b2": 4, "b3": 4, "b4": 16, "b5": 8}],
                         ids=["union", "own_down_mix"])
def test_closed_tiles_equal_core_plus_open_tiles_and_the_mask(fx, layout, bits):
    core, boxes, gates = _many_boxes(overlap=False)
    opened = frozenset({"b1", "b4"})
    r_layers, r_tiles = _install(fx, _write(fx, f"c_{layout}", core, boxes, {}, router=_routed(boxes),
                                            tile_bits=bits), layout)
    sub = {n: m for n, m in boxes.items() if n in opened}
    s_layers, _ = _install(fx, _write(fx, f"s_{layout}", core, sub, {}, router=_routed(sub),
                                      tile_bits=bits and {k: v for k, v in bits.items() if k == "core" or k in sub}),
                           layout)
    masked = {n: (w * 0.0, np.full_like(b, 50.0 if n in opened else -50.0)) for n, (w, b) in gates.items()}
    m_layers, _ = _install(fx, _write(fx, f"m_{layout}", core, boxes, masked, tile_bits=bits), layout)
    x = _x()
    ys = _run(r_layers, r_tiles, x, [opened, opened])
    for l, y in zip(range(1, L), ys):
        assert np.array_equal(y, _np(s_layers[l].mlp(x))), l        # core + open tiles only
        if bits is None or layout == "unfused":
            # Masking in the hard-gated layout: unfused adds a zero contribution per closed box; fused
            # union adds zeros too. (Fused own-down groups of the hard-gated set sum several members
            # in one fp32 reduction, a different arithmetic: compared in the next assertion only.)
            assert np.array_equal(y, _np(m_layers[l].mlp(x))), l
        else:
            ref = _np(m_layers[l].mlp(x))
            assert np.max(np.abs(y - ref)) <= 0.03 * np.max(np.abs(ref))


@pytest.mark.parametrize("layout", LAYOUTS)
def test_closed_tile_projections_are_never_called(fx, layout):
    core, boxes, _ = _many_boxes(overlap=False)
    layers, tiles = _install(fx, _write(fx, f"spy_{layout}", core, boxes, {}, router=_routed(boxes)), layout)

    class Refuse(nn.Module):
        def __call__(self, x):
            raise AssertionError("a closed tile was computed")
    mlp = layers[1].mlp
    holders = mlp.groups if layout == "fused" else mlp.sources
    for h in holders:
        if (getattr(h, "tile", None) if layout == "fused" else (h.sid if h.router else None)) == "b2":
            h.gate_proj = Refuse()
    x = _x()
    _run(layers, tiles, x, [frozenset({"b0"}), frozenset({"b1", "b3"})])     # b2 closed everywhere: fine
    with pytest.raises(AssertionError, match="closed tile"):
        _run(layers, tiles, x, [frozenset({"b2"}), frozenset()])


@pytest.mark.parametrize("partial_rows", ["mask", "gather"])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_mixed_batch_rows_equal_their_own_single_row_result(fx, layout, partial_rows):
    """mask (default): exact by construction (the masked arithmetic). gather: exact on CPU only."""
    core, boxes, _ = _many_boxes(overlap=True)
    bits = {"core": None, "b0": None, "b1": None, "b2": None, "b3": 4, "b4": 16, "b5": None}
    layers, tiles = _install(fx, _write(fx, f"mix_{layout}_{partial_rows}", core, boxes, {},
                                        router=_routed(boxes), tile_bits=bits), layout, partial_rows=partial_rows)
    assert tiles.identity["router_partial_rows"] == partial_rows
    assert tiles.receipt()["router"]["partial_rows_exactness"].startswith(
        "exact" if partial_rows == "mask" else "near_tie")
    sets = [None, frozenset({"b0"}), frozenset(), frozenset({"b1", "b3", "b4"})]
    x = _x(batch=len(sets))
    batched = _run(layers, tiles, x, sets)
    for i, opened in enumerate(sets):
        alone = _run(layers, tiles, x[i:i + 1], [opened])
        for l in range(L - 1):
            assert np.array_equal(batched[l][i:i + 1], alone[l]), (i, l)    # bit-exact on CPU


def test_open_request_resolution_and_refusals(fx):
    from mlx2.runtime.amoeba_tiles import TileNotLoaded
    core, boxes, gates = _many_boxes(overlap=False)
    router = {n: _routed(boxes)[n] for n in ("b0", "b1", "b2", "b3")}
    hard = {n: g for n, g in gates.items() if n not in router}
    _, tiles = _install(fx, _write(fx, "req", core, boxes, hard, router=router), "fused")
    assert tiles.open_request(None) is None                                   # absent: full model
    assert tiles.open_request({"tiles": ["b0", "b1", "b2", "b3"]}) is None    # every router tile: full
    assert tiles.open_request({}) == frozenset()                              # all router tiles closed
    assert tiles.open_request({"topics": ["t0"]}) == frozenset({"b0", "b3"})  # OR rule on opens_on
    assert tiles.open_request({"topics": ["t1"], "tiles": ["b2"]}) == frozenset({"b1", "b2"})
    for bad in ({"tiles": ["capitals"]}, {"topics": ["geo"]}):
        with pytest.raises(TileNotLoaded) as e:
            tiles.open_request(bad)
        assert e.value.http_status == 409
    with pytest.raises(ValueError, match="not router tiles"):
        tiles.open_request({"tiles": ["core"]})
    with pytest.raises(ValueError, match="not router tiles"):
        tiles.open_request({"tiles": ["b4"]})                                  # hard-gated box
    for bad in (["b0"], {"tiles": "b0"}, {"open": ["b0"]}):
        with pytest.raises(ValueError, match="mlx2_amoeba_open"):
            tiles.open_request(bad)
    _, ext = _install(fx, fx.tmp / "req" / "manifest.json", "fused", route="external_draft")
    assert ext.open_request({"tiles": ["b0"]}) == frozenset({"b0"})           # external draft binds rows
    _, spec = _install(fx, fx.tmp / "req" / "manifest.json", "fused", route="native_mtp")
    assert spec.open_request({"tiles": ["b0", "b1", "b2", "b3"]}) is None     # full set: admitted
    with pytest.raises(ValueError, match="native_mtp route's verify forwards"):
        spec.open_request({"tiles": ["b0"]})
    _, plain = _install(fx, _write(fx, "plain", core, boxes, gates), "fused")
    assert plain.router is None and "router_law" not in plain.identity
    with pytest.raises(TileNotLoaded):
        plain.open_request({"tiles": []})


def test_apc_scope_and_server_field():
    from mlx2.runtime.amoeba_tiles import AmoebaTileSet
    from mlx2.serving import request_apc_scope
    a = AmoebaTileSet.open_fingerprint(frozenset({"b0", "b1"}))
    assert a == AmoebaTileSet.open_fingerprint(frozenset({"b1", "b0"}))
    assert a != AmoebaTileSet.open_fingerprint(frozenset({"b0"}))
    base = request_apc_scope({})
    assert request_apc_scope({"_mlx2_amoeba_open_fingerprint": a}) != base
    assert request_apc_scope({"_mlx2_amoeba_open_fingerprint": a}) == request_apc_scope(
        {"_mlx2_amoeba_open_fingerprint": a})
    import mlx2.server as server
    assert '"mlx2_amoeba_open",' in Path(server.__file__).read_text()


@pytest.mark.parametrize("partial_rows", ["mask", "gather"])
@pytest.mark.parametrize("layout", LAYOUTS)
def test_counters_receipt_and_generator_seam(fx, layout, partial_rows):
    from mlx2.runtime.weight_overlays import bind_weight_overlay_rows, clear_weight_overlay_rows
    core, boxes, _ = _many_boxes(overlap=False)
    model = nn.Module()
    layers, tiles = _install(fx, _write(fx, f"cnt_{layout}_{partial_rows}", core, boxes, {},
                                        router=_routed(boxes)), layout, model=model, partial_rows=partial_rows)
    router = tiles.router
    x = _x(batch=2, rows=5)
    assert bind_weight_overlay_rows(model, [7, 8]) is None                  # nothing restricted: no binding
    [layers[l].mlp(x) for l in range(1, L)]                                  # unbound, nothing restricted: fine
    record = router.bind_uid(8, frozenset({"b0"}), {"tiles": ["b0"]})
    assert record["router"] is router
    managers = bind_weight_overlay_rows(model, [7, 8])
    assert managers == (router,)
    [layers[l].mlp(x) for l in range(1, L)]
    clear_weight_overlay_rows(managers)
    assert router.rows is None
    with pytest.raises(RuntimeError, match="outside the per-row open-set seam"):
        layers[1].mlp(x)                                                     # bypassed the seam: refused
    managers = bind_weight_overlay_rows(model, [7, 9])                       # restricted uid not in batch
    [layers[l].mlp(x) for l in range(1, L)]                                  # seam active: no refusal
    clear_weight_overlay_rows(managers)
    r = tiles.receipt()["router"]
    held = {n: sum(1 for l in range(1, L) if np.any(boxes[n][l])) for n in boxes}
    for n in boxes:
        # Row 7 opens every tile, row 8 only b0; masked partial rows are computed on both rows.
        open_rows = 2 if n == "b0" or partial_rows == "mask" else 1
        assert r["rows_computed"][n] == held[n] * 5 * (2 + open_rows + 2)
        assert r["rows_skipped"][n] == held[n] * 5 * (2 - open_rows)
    assert r["last_forward"]["open_row_share"] == {n: (1.0 if n == "b0" else 0.5) for n in boxes}
    assert r["bound_forwards"] == 2 and r["seam_refusals"] == 1
    assert record["bound_forwards"] == 1 and record["open_tiles"] == ["b0"]
    router.unbind(record)
    assert bind_weight_overlay_rows(model, [7, 8]) is None
    record = router.bind_uid(8, frozenset({"b0"}))
    router.bind_rows([7, 8])
    with pytest.raises(RuntimeError, match="fail closed"):
        layers[1].mlp(x[:1])                                                 # rows != bound open sets
    router.clear_rows()
    router.bind_uid(3, frozenset({"b0"}), record=record)                     # re-admitted under a new uid
    assert set(router.uids) == {3} and record["uids"] == [3]
    router.reset()                                                           # generator rebuilt
    assert not router.uids and record["uids"] == [] and r["generator_resets"] == 0
    assert tiles.receipt()["router"]["generator_resets"] == 1
