"""Amoeba tiled MLPs in serving: an English core plus gated topic boxes (experimental, unqualified).

A tile set (``manifest.json``, schema ``mlx2.amoeba-tiles.v1``, quantized encoding
``rows-exact+down-group-union``) replaces each tiled decoder layer's gated MLP
``down(act(gate(x)) * up(x))`` with

    a_s  = g_s(x) * act(gate_s(x)) * up_s(x)          for every source s (the core, each box)
    u    = sum_s scatter(a_s into the layer's down-input space)
    y    = down_U(u)

where ``gate_s``/``up_s`` are the source checkpoint's quantized rows for that source's neurons
(row slices are exact: affine groups run along the input dimension), ``down_U`` is the source
checkpoint's ``down_proj`` restricted to the union U of 64-neuron input groups that hold any tile
neuron (whole groups, so the slice is exact), and ``g_core = 1``. A box gate is the per-token law
``g = [sigmoid(x . w + b) > 0.5]`` on the MLP input (clamped, so a closed box adds exactly 0).
Layers the manifest lists as ``dense_layers`` keep the source MLP whole (the first-position
stand-in). The model-specific maths (which modules, which activation) comes from the adapter's
``amoeba_mlp`` declaration; nothing here names a model.

Projection law (``PROJECTION_LAW``). Every tile projection is an ordinary ``nn.QuantizedLinear`` /
``nn.Linear`` child module of the tiled MLP, so whatever projection law a route installs over the
trunk's linear modules (the serving lane matmul and its tree/all-rows scope, the row-exact
verify class swap, the slice-invariant prefill swap, the q4 three-row gate schedule) reaches the
tile projections exactly as it reaches any other trunk projection, provided it is installed
*after* the tiles (the adapter installs tiles first). To make lane coverage possible, gate/up row
blocks are zero-padded to a multiple of 4 outputs and a BF16 own-down block to a multiple of 64
inputs; padded rows and columns contribute exact zeros. The box gate is not a projection: its
logit is the f32 row-wise product-sum ``sum(x * w) + b`` (``GATE_ARITHMETIC``), an elementwise map
and a within-row reduction, so a row's gate does not depend on how many rows share the call (a
one-output ``x @ w`` is an M=N=1 product that no multi-row form reproduces, see
``row_exact/ops._single_output``). Gates are hard, so a gate that differed between a verify row and
the same row decoded alone would be a high-margin divergence, not a near tie.

Router tiles (gate law ``ROUTER_LAW``). A box whose manifest gate is ``{"law": "router",
"opens_on": [topic ids]}`` has no learned gate: whether it is open is a per-request decision
supplied from outside (request field ``mlx2_amoeba_open``: ``{"tiles": [...], "topics": [...]}``;
a tile opens when it is named or any of its ``opens_on`` topics is, the partition's OR rule). An
absent field opens every router tile (the full model). The engine publishes the per-row open sets
before each forward through :func:`bind_amoeba_rows` (the generator's weight-space row seam), a
host-side list shared by every tiled layer (:class:`AmoebaRouterRows`), so no host sync per token.
A router tile closed in every row of a forward is skipped (no projections); one open in some rows
runs on every row with its closed rows multiplied by 0 (the masked reference's arithmetic), or,
opt-in (``ROUTER_PARTIAL_ROWS`` ``gather``, a near tie), on its open rows only, scatter-added back.
The fused layout keeps the core and the hard-gated boxes fused per format and gives every router
tile its own group, so skipping is skipping a group (``ROUTER_LAYOUT``). Per-request open sets
run on the ordinary route and on the external-draft route (``ROUTES_WITH_OPEN_SETS``), whose
target forwards (prefill, cohort and trusted verify, AST/TensorFold tree tiles, reference tree
replays and commits, ordinary pipeline rounds) bind their rows with :class:`amoeba_rows`; a packed
TensorFold cohort binds each lane's run of tree rows. A forward that does not bind fails closed
in the tiled layer. native_mtp refuses restricted sets (its verify state machine is not bound).

Fail closed: a missing tile file, a tile-file digest or source-checkpoint fingerprint that does not
match, an unknown encoding, or a shape that disagrees with the live module raises at install, and a
request naming a tile the route does not hold raises :class:`TileNotLoaded` (HTTP 409). A route
that needs a row-invariant projection law refuses a tile projection that law does not cover
(:meth:`AmoebaTileSet.lane_law_refusal`).

History: on 2026-11-14 (branch claude/amoeba-27b-spec-tiles) the unfused layer's product changed from
``nn.silu(gate) * up`` to the adapter-declared compiled ``swiglu`` (the stock MLP's own), and its
gate from a one-output matmul to the row-wise product-sum; the identity's ``product`` and
``gate_arithmetic`` fields changed with them, so unfused evidence recorded before that is a
different arm and its cached prefixes are never restored under the new arithmetic.

Identity: the tile set's digest, the projection/gate arithmetic and the serving route join the
adapter's execution-numerics contract, which is part of the APCv2 namespace, so a prefix cached
under one tile set (or the base model, or another route) is never restored under another.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = "mlx2.amoeba-tiles.v1"
ENCODING = "rows-exact+down-group-union"
RECEIPT_SCHEMA = "mlx2.amoeba-tiles-receipt.v2"
GATE_LAW = "hard-sigmoid-0.5(w.x+b) on the MLP input"
# How the gate logit is computed (part of the arithmetic, so of the APCv2 namespace).
GATE_ARITHMETIC = "f32-rowwise-product-sum-v1"
# Opt-in fallback (adapter policy ``amoeba_tiles_gate_per_row``): one launch per row, so a
# multi-row call computes exactly the one-row (decode) form, whatever the reduction kernel does
# with the row count. Default off until the GPU check shows the batched reduction is not row-invariant.
GATE_ARITHMETIC_PER_ROW = "f32-rowwise-product-sum-one-launch-per-row-v1"
# Fold the per-forward gate sums into one on-device running total per box every this many entries,
# so the pending list and the lazy graph stay bounded.
GATE_FOLD_EVERY = 256

import contextlib as _contextlib
import threading as _threading

_COUNTING = _threading.local()


def gate_counting_enabled() -> bool:
    return getattr(_COUNTING, "on", True)


@_contextlib.contextmanager
def gate_counting_suspended():
    """Research only: skip gate counters (e.g. inside ``mx.value_and_grad``, where the counters'
    ``async_eval`` is not allowed and their traced values must not leak into the totals).
    Serving never enters this; receipts would undercount if it did."""
    prev = gate_counting_enabled()
    _COUNTING.on = False
    try:
        yield
    finally:
        _COUNTING.on = prev
# Layouts of a tiled layer (adapter policy ``amoeba_tiles_layout``). ``unfused``: one gate/up (and
# down) projection per source, one gate per box (the reference). ``fused``: per layer, one gate/up
# projection per projection format over [core rows; box rows] with every box's hard gate applied
# as a per-row x per-segment mask, one batched row-wise gate reduction for all boxes, one scatter
# per overlap multiplicity into the shared down union, and one down projection per own-down format.
# ``neuron_order`` (router tile sets that partition every tiled layer with exact host slices): the
# layer runs the source MLP's own gate/up/down modules (one copy of the weights, original neuron
# order, the stock shapes), and a tile is an index set over that layout: closed tiles' activations
# are multiplied by 0 before the down projection. All open is the stock MLP's computation, so the
# untiled model bit for bit on any device; restricted rows equal the mask reference exactly;
# closed tiles save no compute. The tile-group layouts (unfused/fused) skip closed tiles but sum
# tiles through a scatter into the down union with other projection shapes: not exact against the
# stock MLP on Metal (2026-11-15, 27B: max |dlogit| 0.30, top-1 0.979), so they are labelled
# near_tie for all-open.
LAYOUTS = ("unfused", "fused", "neuron_order")
LAYOUT_IDS = {"unfused": "per-source-v1", "fused": "fused-format-groups-v1",
              "neuron_order": "neuron-order-masked-v1"}
NEURON_ORDER_LAW = "source-mlp-modules-original-neuron-order-closed-tiles-masked-v1"
# Rows x boxes x d_model above which the batched gate reduction is split into row chunks (rows are
# independent, so chunking changes no row's bits); bounds the f32 product at prefill.
GATE_CHUNK_ELEMENTS = 1 << 24
# The fused layout's gate: one batched f32 product-sum over rows x (1 + boxes) x d (column 0 the
# always-open segment), row-chunked at GATE_CHUNK_ELEMENTS. Its own id: a different launch shape
# from the unfused per-box reduction (equal bits on CPU; the GPU check compares them on Metal).
GATE_ARITHMETIC_FUSED = "f32-batched-rowwise-product-sum-rows-x-(1+boxes)-x-d-chunked-v1"
GATE_ARITHMETIC_FUSED_PER_ROW = "f32-batched-rowwise-product-sum-(1+boxes)-x-d-one-launch-per-row-v1"
# Fused union gate/up rows are padded to a multiple of 64 (at least one zero row, which absent
# union positions gather): N % 64 keeps direct tiled-prefill admission (q4 g64, N % 64 == 0).
FUSED_UNION_ROW_MULTIPLE = 64
# Tile projections are linear child modules: the route's installed projection law covers them.
PROJECTION_LAW = "tile-linear-modules-follow-route-law-v2"
GATE_UP_ROW_MULTIPLE = 4
DENSE_DOWN_INPUT_MULTIPLE = 64
# Serving routes a tile set may run on (the adapter chooses one at load).
ROUTES = ("ordinary", "native_mtp", "external_draft")
# Routes whose target forwards bind per-row open sets (restricted mlx2_amoeba_open requests).
ROUTES_WITH_OPEN_SETS = ("ordinary", "external_draft")
# A router tile: no learned gate; open or closed per request row, decided outside the model.
ROUTER_LAW = "router"
# Router tiles in the fused layout: one group per router tile (per format), skipped when closed in
# every row and gathered to its open rows otherwise; the core and hard-gated boxes stay fused.
ROUTER_LAYOUT = "router-tile-groups-v2"
ROUTER_RECEIPT_SCHEMA = "mlx2.amoeba-router-rows.v2"
# How a router tile open in only some rows of a forward runs (adapter policy
# ``amoeba_tiles_router_partial_rows``), with the exactness each claims against masking.
ROUTER_PARTIAL_ROWS = {"mask": "exact: the masked reference's arithmetic (all rows x 0/1 row mask)",
                       "gather": "near_tie: open rows only, so the projection row count changes"}
_ID_RE = r"[A-Za-z0-9_-]{1,96}"


class TileNotLoaded(ValueError):
    """A request named a tile this route does not hold (HTTP 409 tile_not_loaded)."""

    http_status = 409
    code = "tile_not_loaded"


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path) -> dict:
    """Read and validate a manifest; every file digest is checked against the file on disk."""
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        path = path / "manifest.json"
    if not path.is_file():
        raise ValueError(f"amoeba tile manifest not found: {path}")
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"amoeba tiles: schema {manifest.get('schema')!r} is not {SCHEMA}")
    if manifest.get("encoding") != ENCODING:
        raise ValueError(f"amoeba tiles: encoding {manifest.get('encoding')!r} is not {ENCODING}")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("amoeba tiles: manifest lists no files")
    for name, digest in files.items():
        f = path.parent / name
        if not f.is_file():
            raise ValueError(f"amoeba tiles: tile file missing: {f}")
        actual = file_sha256(f)
        if actual != digest:
            raise ValueError(f"amoeba tiles: digest mismatch for {name}: manifest {digest[:12]}, file {actual[:12]}")
    for key in ("source", "layers", "intermediate", "core", "boxes", "dense_layers", "tiled_layers"):
        if key not in manifest:
            raise ValueError(f"amoeba tiles: manifest has no {key!r}")
    ids = [manifest["core"]["id"]] + [b["id"] for b in manifest["boxes"]]
    if len(set(ids)) != len(ids):
        raise ValueError("amoeba tiles: duplicate tile ids")
    import re
    if not all(isinstance(i, str) and re.fullmatch(_ID_RE, i) for i in ids):
        raise ValueError("amoeba tiles: tile ids must be [A-Za-z0-9_-]{1,96}")
    for b in manifest["boxes"]:
        gate = b.get("gate") or {}
        if gate.get("law") == ROUTER_LAW:
            opens_on = gate.get("opens_on", [])
            if (not isinstance(opens_on, list) or len(set(opens_on)) != len(opens_on)
                    or not all(isinstance(t, str) and re.fullmatch(_ID_RE, t) for t in opens_on)):
                raise ValueError(f"amoeba tiles: router tile {b['id']} opens_on must be distinct "
                                 "[A-Za-z0-9_-]{1,96} topic ids")
    manifest["_dir"] = str(path.parent)
    manifest["_digest"] = file_sha256(path)
    return manifest


def router_tiles(manifest) -> dict:
    """``{router tile id: tuple(opens_on topics)}`` of the manifest's router-law boxes."""
    return {b["id"]: tuple((b.get("gate") or {}).get("opens_on", [])) for b in manifest["boxes"]
            if (b.get("gate") or {}).get("law") == ROUTER_LAW}


def tile_set_identity(manifest: dict, *, route: str = "ordinary", gate_per_row: bool = False,
                      layout: str = "unfused", product: str = "activation(gate)*up",
                      router_partial_rows: str = "mask") -> dict:
    """The part of the tile set and route that changes the arithmetic (APCv2 namespace, receipts)."""
    if route not in ROUTES:
        raise ValueError(f"amoeba tiles: route must be one of {ROUTES}, got {route!r}")
    if layout not in LAYOUTS:
        raise ValueError(f"amoeba tiles: layout must be one of {LAYOUTS}, got {layout!r}")
    if layout == "neuron_order":
        gate_arithmetic = "none (router tiles only)"
        padding = {"none": "the source MLP's own modules"}
    elif layout == "fused":
        gate_arithmetic = GATE_ARITHMETIC_FUSED_PER_ROW if gate_per_row else GATE_ARITHMETIC_FUSED
        padding = {"union_gate_up_rows_multiple": FUSED_UNION_ROW_MULTIPLE, "union_zero_rows_min": 1,
                   "own_down_gate_up_rows": "padded to the down block width (whole input groups)",
                   "dense_down_inputs_multiple": DENSE_DOWN_INPUT_MULTIPLE}
    else:
        gate_arithmetic = GATE_ARITHMETIC_PER_ROW if gate_per_row else GATE_ARITHMETIC
        padding = {"gate_up_rows_multiple": GATE_UP_ROW_MULTIPLE,
                   "dense_down_inputs_multiple": DENSE_DOWN_INPUT_MULTIPLE}
    return {"schema": SCHEMA, "layout": LAYOUT_IDS[layout], "product": product,
            "gate_arithmetic": gate_arithmetic, "padding": padding,
            **({"gate_chunk_elements": GATE_CHUNK_ELEMENTS} if layout == "fused" else {}), "encoding": manifest["encoding"], "manifest_sha256": manifest["_digest"],
            "source_fingerprint": manifest["source"]["fingerprint"],
            "tiles": sorted([manifest["core"]["id"]] + [b["id"] for b in manifest["boxes"]]),
            "tile_fingerprints": {t["id"]: t["fingerprint"] for t in [manifest["core"], *manifest["boxes"]]},
            "tile_bits": {t["id"]: t.get("bits") for t in [manifest["core"], *manifest["boxes"]]},
            "gate_law": GATE_LAW,
            "projection_law": PROJECTION_LAW,
            "route": route, "dense_layers": list(manifest["dense_layers"]),
            # Only a tile set with router tiles carries these (earlier identities are unchanged).
            **({"router_law": ROUTER_LAW, "router_layout": ROUTER_LAYOUT,
                "router_partial_rows": router_partial_rows,
                **({"neuron_order_law": NEURON_ORDER_LAW, "all_open_exactness": "stock_exact",
                    "restricted_exactness": "masked_reference_exact"} if layout == "neuron_order" else
                   {"all_open_exactness": "near_tie (tile sum via the down union, other projection shapes)",
                    "restricted_exactness": ROUTER_PARTIAL_ROWS[router_partial_rows]}),
                "router_tiles": {t: list(o) for t, o in sorted(router_tiles(manifest).items())}}
               if router_tiles(manifest) else {})}


def numerics_contract_with_tiles(contract, tiles):
    """An adapter's execution-numerics contract with the tile set joined (APCv2 namespace)."""
    if tiles is None:
        return contract
    return {**(contract or {}), "amoeba_tiles": tiles.identity}


def _qparams(module):
    """(kind, arrays, bits, group_size, mode) of a live linear."""
    import mlx.nn as nn
    if isinstance(module, nn.QuantizedLinear):
        arrays = {"weight": module.weight, "scales": module.scales}
        if module.get("biases") is not None:
            arrays["biases"] = module.biases
        return "quantized", arrays, module.bits, module.group_size, module.mode
    if isinstance(module, nn.Linear):
        return "dense", {"weight": module.weight}, None, None, None
    raise ValueError(f"amoeba tiles: unsupported projection {type(module).__name__}")


def _pad_axis(a, axis, extra):
    import mlx.core as mx
    if not extra:
        return a
    shape = list(a.shape)
    shape[axis] = extra
    return mx.concatenate([a, mx.zeros(tuple(shape), dtype=a.dtype)], axis=axis)


def make_linear(p, *, out_multiple=1, in_multiple=1):
    """A frozen ``nn.QuantizedLinear``/``nn.Linear`` holding ``p``'s arrays.

    Outputs are zero-padded to ``out_multiple`` rows (packed zero weights with zero scales and
    biases dequantize to exactly 0). ``in_multiple`` pads a dense weight's input columns with
    zeros. The constructor's random initialisation is lazy and replaced before evaluation.
    """
    import mlx.nn as nn
    w = p["weight"]
    out = int(w.shape[0])
    extra = (-out) % out_multiple
    if p["kind"] == "quantized":
        if in_multiple != 1:
            raise ValueError("amoeba tiles: quantized input padding is the exporter's (down_pad)")
        in_dims = int(w.shape[1]) * 32 // int(p["bits"])
        m = nn.QuantizedLinear(in_dims, out + extra, bias=False, group_size=int(p["group_size"]),
                               bits=int(p["bits"]), mode=p.get("mode") or "affine")
        m.weight = _pad_axis(w, 0, extra)
        m.scales = _pad_axis(p["scales"], 0, extra)
        m.biases = None if p.get("biases") is None else _pad_axis(p["biases"], 0, extra)
    else:
        in_extra = (-int(w.shape[1])) % in_multiple
        m = nn.Linear(int(w.shape[1]) + in_extra, out + extra, bias=False)
        m.weight = _pad_axis(_pad_axis(w, 1, in_extra), 0, extra)
    m.freeze()
    return m


def linear_in_width(module) -> int:
    import mlx.nn as nn
    if isinstance(module, nn.QuantizedLinear):
        return int(module.scales.shape[1]) * int(module.group_size)
    return int(module.weight.shape[1])


class _State:
    """Plain holder: keeps mutable counters out of the module's parameter tree.

    Each gated forward appends a lazy per-box open-row count (an ``mx.array`` scalar that is *not*
    part of the step output graph). Every ``GATE_FOLD_EVERY`` entries they are folded into one
    on-device running total per box and scheduled with ``mx.async_eval`` (no host sync), so the
    pending list and the lazy graph stay bounded; :meth:`AmoebaTileSet.receipt` evaluates the
    totals once and reads one float per box.
    """

    def __init__(self, counters):
        self.counters = counters
        self.router = counters.get("router")          # one AmoebaRouterRows shared by every layer


class AmoebaRouterRows:
    """Per-request open sets of the router tiles and the per-forward row binding (host state).

    ``bind_uid`` records a request's open set (a frozenset of router tile ids; requests that open
    every router tile are never bound). The generator calls :meth:`bind_rows` with the batch uids
    right before each forward and :meth:`clear_rows` after it (``bind_amoeba_rows``); the tiled
    layers read :meth:`tile_rows`. While a restricted request is bound, a tiled-layer call outside
    that seam fails closed (:meth:`restricted`): it could not name its rows. Counters are host
    integers in layer-token-row units.

    ``partial_rows`` (``ROUTER_PARTIAL_ROWS``) is how a tile open in only some rows runs: ``mask``
    (default) computes it on every row and multiplies the closed rows by 0, the masked reference's
    own arithmetic; ``gather`` runs it on the open rows only and scatter-adds back, which changes
    the projection row count (a near tie where the kernel depends on it, e.g. Metal).
    """

    def __init__(self, tiles, lock, partial_rows="mask"):
        if partial_rows not in ROUTER_PARTIAL_ROWS:
            raise ValueError(f"amoeba router partial_rows must be one of {tuple(ROUTER_PARTIAL_ROWS)}")
        self.tiles = tuple(tiles)
        self.lock = lock
        self.partial_rows = partial_rows
        self.uids = {}                                  # uid -> frozenset of open router tiles
        self.stats = {}                                 # uid -> per-request record (shared with the job)
        self.rows = None                                # per forward: list per unit (None = all open)
        self.widths = None                              # per forward: packed token rows per unit, or None
        self.seam = False                               # bind_rows ran for the current forward
        self._tile_rows = None
        self._open_table = None
        self.masked_closed = {t: 0 for t in self.tiles}
        self.computed = {t: 0 for t in self.tiles}
        self.skipped = {t: 0 for t in self.tiles}
        self.last_forward = None
        self.bound_forwards = 0
        self.seam_refusals = 0
        self.compaction_declines = 0
        self.resets = 0
        # Set by a route generator whose target forwards cannot carry per-row open sets.
        self.open_set_refusal = None

    def bind_uid(self, uid, open_tiles, requested=None, record=None):
        """Bind a restricted request's batch uid; returns its per-request record (updated in
        place). Pass the job's record back when the request is re-admitted under a new uid: its
        earlier uids are dropped first, so a recycled uid never inherits this open set."""
        with self.lock:
            if record is None:
                record = {"open_tiles": sorted(open_tiles), "requested": requested,
                          "router_tiles": len(self.tiles), "bound_forwards": 0, "uids": [],
                          "router": self}
            else:
                self._drop(record)
            self.uids[uid] = frozenset(open_tiles)
            self.stats[uid] = record
            record["uids"].append(uid)
        return record

    def _drop(self, record):
        for uid in record["uids"]:
            if self.stats.get(uid) is record:
                self.uids.pop(uid, None)
                self.stats.pop(uid, None)
        record["uids"] = []

    def unbind(self, record):
        """Release every uid a request was bound under (finish, or before a preemption replay)."""
        with self.lock:
            self._drop(record)

    def reset(self):
        """The generator was rebuilt (its uid counter restarts): no binding survives."""
        with self.lock:
            for record in list(self.stats.values()):
                record["uids"] = []
            self.uids.clear()
            self.stats.clear()
            self._clear()
            self.resets += 1

    def bind_rows(self, uids, widths=None):
        """Publish this forward's per-row open sets. True while any restricted request is bound
        (the caller then declines merged forwards whose rows it cannot name, and clears after).

        ``widths=None``: batch geometry, unit ``i`` is the input's axis-0 row ``i`` (all its tokens
        share the set). ``widths``: packed geometry (a TensorFold cohort), the input is one row of
        ``sum(widths)`` token rows and unit ``i`` owns the next ``widths[i]`` of them, so every
        candidate row of a request's tree inherits that request's set."""
        import mlx.core as mx
        with self.lock:
            self._clear()
            if not self.uids:
                return False
            if widths is not None:
                widths = tuple(int(w) for w in widths)
                if len(widths) != len(uids) or any(w < 1 for w in widths):
                    raise RuntimeError("amoeba router rows: packed widths disagree with the bound uids")
            self.seam = True
            rows = [self.uids.get(uid) for uid in uids]
            for uid in uids:
                if uid in self.stats:
                    self.stats[uid]["bound_forwards"] += 1
            self.bound_forwards += 1
            if any(r is not None for r in rows):
                self.rows, self.widths = rows, widths

                def expand(values):                     # unit values -> row-axis values
                    if widths is None:
                        return values
                    return [v for v, w in zip(values, widths) for _ in range(w)]
                per_tile = {}
                for t in self.tiles:
                    opened = [r is None or t in r for r in rows]
                    n = sum(opened)
                    tokens = None if widths is None else sum(w for o, w in zip(opened, widths) if o)
                    if n == len(rows):
                        per_tile[t] = ("all", None, n, tokens)
                    elif n == 0:
                        per_tile[t] = ("none", None, 0, tokens)
                    elif self.partial_rows == "gather":
                        arr = mx.array([i for i, o in enumerate(opened) if o], dtype=mx.int32)
                        per_tile[t] = ("some", (arr, None), n, tokens)
                    else:
                        unit = mx.array([float(o) for o in opened])
                        per_tile[t] = ("some", (unit, None if widths is None else
                                                mx.array(expand([float(o) for o in opened]))), n, tokens)
                self.last_forward = {"rows": len(rows), "packed": widths is not None, "open_row_share": {
                    t: n / len(rows) for t, (_, _, n, _) in per_tile.items()}}
                # Every tile open in every row: nothing to restrict (and no row geometry to keep).
                self._tile_rows = None if all(v[0] == "all" for v in per_tile.values()) else per_tile
                # Neuron-order layers: row-axis x (1 + tiles) 0/1 table, column 0 the core (always 1).
                table = [[1.0] + [float(r is None or t in r) for t in self.tiles] for r in rows]
                self._open_table = None if self._tile_rows is None else (
                    mx.array(table), None if widths is None else mx.array(expand(table)))
            return True

    def _clear(self):
        self.rows, self.widths, self._tile_rows, self._open_table, self.seam = None, None, None, None, False

    def clear_rows(self):
        self._clear()

    def state(self):
        """The current binding (a nested seam saves and restores it)."""
        return (self.rows, self.widths, self._tile_rows, self._open_table, self.seam)

    def restore(self, state):
        self.rows, self.widths, self._tile_rows, self._open_table, self.seam = state

    def _geometry(self, x):
        """``(row axis, token rows per unit, packed)`` of ``x`` under the bound rows; fails closed.

        With packed widths bound, an input whose axis 0 has one row per unit (a provider that lays
        the cohort out as a batch instead of one packed row) takes the batch geometry; a single
        unit is the same either way."""
        units = len(self.rows)
        if self.widths is None or (units > 1 and int(x.shape[0]) == units):
            if int(x.shape[0]) != units:
                raise RuntimeError(f"amoeba router rows: a forward of {int(x.shape[0])} rows under {units} "
                                   "bound open sets (fail closed)")
            return 0, [int(x.shape[1]) if x.ndim == 3 else 1] * units, False
        total = sum(self.widths)
        if x.ndim == 3 and int(x.shape[0]) == 1 and int(x.shape[1]) == total:
            return 1, list(self.widths), True
        if x.ndim == 2 and int(x.shape[0]) == total:
            return 0, list(self.widths), True
        raise RuntimeError(f"amoeba router rows: a forward of shape {tuple(x.shape)} under packed widths "
                           f"{list(self.widths)} (fail closed)")

    @staticmethod
    def row_shape(a, axis):
        """Broadcast shape of a row-axis vector against ``a`` (rows on ``axis``)."""
        shape = [1] * a.ndim
        shape[axis] = -1
        return tuple(shape)

    def open_table(self, x, layer_tiles):
        """Neuron-order layers: ``(row-axis x (1 + tiles) 0/1 table, row axis)`` of this forward.
        Every tile is computed (closed ones masked to 0): counts rows computed and masked closed."""
        axis, per_unit, packed = self._geometry(x)
        total = sum(per_unit)
        with self.lock:
            for t in layer_tiles:
                opened = [r is None or t in r for r in self.rows]
                self.computed[t] += total
                self.masked_closed[t] += sum(w for o, w in zip(opened, per_unit) if not o)
        return self._open_table[1 if packed else 0], axis

    def count_all_open(self, x, layer_tiles):
        tokens = int(x.shape[0] * x.shape[1]) if x.ndim == 3 else int(x.shape[0])
        with self.lock:
            for t in layer_tiles:
                self.computed[t] += tokens

    def restricted(self):
        """True when this forward closes some router tile in some row. A tiled-layer call while a
        restricted request is bound but outside the generator's row seam raises: that forward
        could not name its rows, so it would over- or under-open silently."""
        if self.uids and not self.seam:
            with self.lock:
                self.seam_refusals += 1
            raise RuntimeError("amoeba router: a tiled forward ran outside the per-row open-set seam "
                               "while a restricted request is bound (fail closed)")
        return self._tile_rows is not None

    def tile_rows(self, tile, x):
        """``("all"|"none"|"some", row index or row-axis mask or None, row axis)`` of ``tile`` for
        this forward's input ``x``, counting token rows computed and skipped. Fails closed on a
        geometry mismatch, and on gathered partial rows in a packed forward."""
        if self._tile_rows is None:
            tokens = int(x.shape[0] * x.shape[1]) if x.ndim == 3 else int(x.shape[0])
            with self.lock:
                self.computed[tile] += tokens
            return "all", None, 0
        axis, per_unit, packed = self._geometry(x)
        kind, arrays, _, _ = self._tile_rows[tile]
        arr = None
        if kind == "some":
            if packed and self.partial_rows == "gather":
                raise RuntimeError("amoeba router: gathered partial rows cannot take a packed forward "
                                   "(use partial_rows mask; fail closed)")
            arr = arrays[1] if packed else arrays[0]
        opened = [r is None or tile in r for r in self.rows]
        open_tokens = sum(w for o, w in zip(opened, per_unit) if o)
        # Masked partial rows are computed (on every row), so they count as computed.
        computed = sum(per_unit) if kind == "some" and self.partial_rows == "mask" else open_tokens
        with self.lock:
            self.computed[tile] += computed
            self.skipped[tile] += sum(per_unit) - computed
        return kind, arr, axis

    def receipt(self):
        with self.lock:
            share = {t: self.computed[t] / max(1, self.computed[t] + self.skipped[t]) for t in self.tiles}
            return {"schema": ROUTER_RECEIPT_SCHEMA, "router_law": ROUTER_LAW, "layout": ROUTER_LAYOUT,
                    "partial_rows": self.partial_rows,
                    "partial_rows_exactness": ROUTER_PARTIAL_ROWS[self.partial_rows],
                    "tiles": list(self.tiles), "rows_computed": dict(self.computed),
                    "rows_skipped": dict(self.skipped), "computed_row_share": share,
                    "rows_masked_closed": dict(self.masked_closed),
                    "rows_unit": "token rows x tiled layers that hold the tile (masked partial rows "
                                 "count as computed)",
                    "bound_forwards": self.bound_forwards,
                    "seam_refusals": self.seam_refusals, "generator_resets": self.resets,
                    "compaction_declines": self.compaction_declines,
                    "last_forward": self.last_forward, "restricted_requests_bound": len(self.uids)}


def bind_amoeba_rows(model, uids):
    """Generator seam: publish per-row router open sets for exactly this forward.

    Returns the router (to clear after the forward) while any restricted request is bound, else
    None, so a server with no restricted request keeps every merged-forward path."""
    router = getattr(model, "_mlx2_amoeba_rows", None) if model is not None else None
    if router is not None and router.bind_rows(uids):
        return router
    return None


class amoeba_rows:
    """Scoped seam for forwards outside the generator's own (speculative verify, tree tiles,
    commit replays): binds ``uids`` (axis-0 rows, or packed ``widths``) for the block and restores
    whatever binding was active before. A no-op when the model holds no router."""

    def __init__(self, model, uids, widths=None):
        self.router = getattr(model, "_mlx2_amoeba_rows", None) if model is not None else None
        self.uids, self.widths = list(uids), widths

    def __enter__(self):
        if self.router is not None:
            self.saved = self.router.state()
            self.router.bind_rows(self.uids, self.widths)
        return self

    def __exit__(self, *exc):
        if self.router is not None:
            self.router.restore(self.saved)
        return False


def _router_part(router, sid, x, act, finish):
    """Router tile ``sid``'s contribution: ``finish(act(rows))`` on its open rows, or None.

    ``act`` maps rows to the tile's activation ``a``, ``finish`` maps ``a`` to the contribution
    (scatter into the down union, or the tile's own down). Closed in every row: None (skipped).
    Open in every row: ``finish(act(x))``. Open in some rows: ``mask`` multiplies the closed rows
    of ``a`` by 0 (the masked reference); ``gather`` (axis-0 rows only) runs the open rows and
    scatter-adds them into zeros (exact zeros for the closed rows, another projection row count)."""
    import mlx.core as mx
    kind, arr, axis = router.tile_rows(sid, x)
    if kind == "none":
        return None
    if kind == "all":
        return finish(act(x))
    if router.partial_rows == "mask":
        a = act(x)
        return finish(a * arr.astype(a.dtype).reshape(router.row_shape(a, axis)))
    part = finish(act(mx.take(x, arr, axis=0)))
    return mx.zeros((int(x.shape[0]),) + tuple(part.shape[1:]), dtype=part.dtype).at[arr].add(part)


def fold_gate_counts(c, *, force=False):
    """Fold pending per-forward gate counts into one on-device total per box (caller holds the lock)."""
    import mlx.core as mx
    pending = c["pending"]
    if not pending or (not force and len(pending) < GATE_FOLD_EVERY):
        return
    c["pending"] = []
    by_key = {}
    for key, value in pending:                         # key: tuple of box ids, value: [len(key)]
        by_key.setdefault(key, []).append(value)
    totals = c["open_totals"]
    for key, values in by_key.items():
        part = mx.sum(mx.stack(values), axis=0)
        totals[key] = part if key not in totals else totals[key] + part
    mx.async_eval(list(totals.values()))


def gate_open_rows(x, w, b):
    """The per-row fallback: one ``gate_open`` launch per row (each the one-row decode form)."""
    import mlx.core as mx
    lead = x.shape[:-1]
    flat = x.reshape(-1, 1, x.shape[-1])
    if flat.shape[0] == 1:
        return gate_open(x, w, b)
    return mx.concatenate([gate_open(flat[i:i + 1], w, b) for i in range(flat.shape[0])],
                          axis=0).reshape(*lead, 1)


def gate_open_many(x, W, b, *, per_row=False):
    """All of a layer's box gates at once: ``[sigmoid(sum(x * W[j]) + b[j]) > 0.5]`` for each box j.

    ``W`` is ``[boxes, d]`` f32, ``b`` ``[boxes]``. The same elementwise product and within-row
    reduction per (row, box) as :func:`gate_open` (no matmul, so no row-count dependent kernel),
    in one launch; rows are split into chunks above ``GATE_CHUNK_ELEMENTS`` (rows are independent).
    ``per_row`` runs one launch per row (the fallback). Returns ``x.shape[:-1] + (boxes,)`` bools.
    """
    import mlx.core as mx
    lead = x.shape[:-1]
    flat = x.reshape(-1, x.shape[-1]).astype(mx.float32)
    rows, boxes, d = int(flat.shape[0]), int(W.shape[0]), int(W.shape[1])

    def one(chunk):
        logit = mx.sum(chunk[:, None, :] * W[None], axis=-1) + b
        return mx.sigmoid(logit) > 0.5
    step = 1 if per_row else max(1, GATE_CHUNK_ELEMENTS // max(1, boxes * d))
    if rows <= step:
        out = one(flat)
    else:
        out = mx.concatenate([one(flat[i:i + step]) for i in range(0, rows, step)], axis=0)
    return out.reshape(*lead, boxes)


def gate_open(x, w, b):
    """The hard box gate ``[sigmoid(sum(x * w) + b) > 0.5]`` per row, in f32 (``GATE_ARITHMETIC``).

    An elementwise product and a within-row reduction: each row's bits do not depend on the other
    rows of the call. Returns ``x.shape[:-1] + (1,)`` booleans.
    """
    import mlx.core as mx
    logit = mx.sum(x.astype(mx.float32) * w, axis=-1, keepdims=True) + b
    return mx.sigmoid(logit) > 0.5


def _refuse_row_exact_kwarg(row_exact):
    """A model whose layers pass ``row_exact=True`` to the MLP (an MLP-owned row-exact protocol)
    is not implemented for tiled layers: fail closed rather than run the stock-kernel form."""
    if row_exact:
        raise RuntimeError("amoeba tiled MLPs do not implement the MLP-owned row_exact protocol (fail closed)")


def make_tile_source_class():
    import mlx.nn as nn

    class AmoebaTileSource(nn.Module):
        """One source (the core or a box) of one tiled layer: its gate/up rows and either a gather
        map into the layer's down union or its own down block, plus a box gate."""

        def __init__(self, sid, rows, gate_proj, up_proj, *, down_proj=None, scatter_map=None,
                     gate_w=None, gate_b=None, router=False):
            super().__init__()
            self.sid = sid
            self.router = bool(router)
            self.rows = int(rows)
            self.gate_proj = gate_proj
            self.up_proj = up_proj
            if down_proj is not None:
                self.down_proj = down_proj
            self.own_down = down_proj is not None
            if scatter_map is not None:
                self.scatter_map = scatter_map
            self.gated = gate_w is not None
            if self.gated:
                self.gate_w = gate_w
                self.gate_b = gate_b

    return AmoebaTileSource


def make_tiled_mlp_class():
    import mlx.core as mx
    import mlx.nn as nn

    class AmoebaTiledMLP(nn.Module):
        """One tiled layer. Union sources scatter into the down union; own-down sources add their
        own projection. Overlapping box neurons add, as in the research reference. Row-wise: a row's
        output depends only on that row (projections follow the route's installed law)."""

        amoeba_tiled = True

        def __init__(self, layer, activation, sources, down_union, d_model, counters, gate_per_row=False):
            super().__init__()
            self._gate_fn = gate_open_rows if gate_per_row else gate_open
            self._layer = layer
            self._product = activation
            self._out = int(d_model)
            self._state = _State(counters)
            self.sources = sources
            if down_union is not None:
                self.down_union = down_union

        def projections(self):
            """``(relative path, module)`` of every projection this layer runs."""
            out = []
            for i, src in enumerate(self.sources):
                out.append((f"sources.{i}.gate_proj", src.gate_proj))
                out.append((f"sources.{i}.up_proj", src.up_proj))
                if src.own_down:
                    out.append((f"sources.{i}.down_proj", src.down_proj))
            if "down_union" in self:
                out.append(("down_union", self.down_union))
            return tuple(out)

        def projection_topology(self):
            """Generic hook for executors that fingerprint an MLP's projections (same tuple shape as
            a stock gate/up/down MLP's topology)."""
            return self.projections()

        def __call__(self, x, row_exact=False):
            from .models.varlen_dense_mlp import compact_rows

            _refuse_row_exact_kwarg(row_exact)

            router = self._state.router
            if router is not None and router.restricted():
                # Per-row open sets name batch rows; compaction would merge them (run uncompacted).
                with router.lock:
                    router.compaction_declines += 1
            else:
                compacted = compact_rows(x)             # padded-prefill live rows only (varlen_dense_mlp)
                if compacted is not None:
                    return compacted.restore(self(compacted.values))
            c = self._state.counters
            rows = int(x.shape[0] * x.shape[1]) if x.ndim == 3 else int(x.shape[0])
            with c["lock"]:
                c["forwards"] += 1
                c["rows"] += rows
            u = None
            own = []
            for src in self.sources:
                if src.router:
                    act = lambda r, s=src: self._product(s.gate_proj(r), s.up_proj(r))  # noqa: E731
                    if src.own_down:
                        part = _router_part(router, src.sid, x, act, lambda a, s=src: s.down_proj(
                            self._pad_to(a, linear_in_width(s.down_proj))))
                        if part is not None:
                            own.append(part)
                        continue
                    part = _router_part(router, src.sid, x, act,
                                        lambda a, s=src: self._scatter(a, s.scatter_map))
                    if part is not None:
                        u = part if u is None else u + part
                    continue
                a = self._product(src.gate_proj(x), src.up_proj(x))
                if src.gated:
                    g = self._gate_fn(x, src.gate_w, src.gate_b).astype(a.dtype)
                    a = a * g
                    if gate_counting_enabled():
                        with c["lock"]:
                            c["pending"].append(((src.sid,), g.astype(mx.float32).sum().reshape(1)))
                            c["box_rows"][src.sid] = c["box_rows"].get(src.sid, 0) + rows
                            fold_gate_counts(c)
                if src.own_down:                     # a box with its own down block (e.g. q4, bf16)
                    width = linear_in_width(src.down_proj)
                    if width > a.shape[-1]:
                        a = mx.concatenate([a, mx.zeros(a.shape[:-1] + (width - a.shape[-1],), dtype=a.dtype)],
                                           axis=-1)
                    own.append(src.down_proj(a))
                    continue
                ext = mx.concatenate([a, mx.zeros(a.shape[:-1] + (1,), dtype=a.dtype)], axis=-1)
                part = mx.take(ext, src.scatter_map, axis=-1)
                u = part if u is None else u + part
            y = None if u is None else self.down_union(u)
            for extra in own:
                y = extra if y is None else y + extra
            if y is None:
                y = mx.zeros(x.shape[:-1] + (self._out,), dtype=x.dtype)
            return y

        @staticmethod
        def _pad_to(a, width):
            if width > a.shape[-1]:
                a = mx.concatenate([a, mx.zeros(a.shape[:-1] + (width - a.shape[-1],), dtype=a.dtype)], axis=-1)
            return a

        @staticmethod
        def _scatter(a, scatter_map):
            ext = mx.concatenate([a, mx.zeros(a.shape[:-1] + (1,), dtype=a.dtype)], axis=-1)
            return mx.take(ext, scatter_map, axis=-1)

    return AmoebaTiledMLP


def make_fused_tiled_mlp_class():
    import mlx.core as mx
    import mlx.nn as nn

    class AmoebaFusedGroup(nn.Module):
        """One projection-format group of a fused layer: gate/up over the group's concatenated
        source rows, a segment index per row (0 = always on: the core and zero rows; 1 + j = box j),
        and either scatter maps into the down union (one per overlap multiplicity) or its own
        concatenated down block."""

        def __init__(self, gate_proj, up_proj, seg, *, masked, maps=None, down_proj=None, info=None,
                     tile=None):
            super().__init__()
            self.tile = tile                            # a router tile's id (its own group), else None
            self.gate_proj = gate_proj
            self.up_proj = up_proj
            self.masked = bool(masked)
            if self.masked:
                self.seg = seg
            self.own_down = down_proj is not None
            if down_proj is not None:
                self.down_proj = down_proj
            else:
                self.maps = maps
            self.info = info or {}

    class AmoebaFusedTiledMLP(nn.Module):
        """A tiled layer with all sources fused per projection format (``LAYOUTS``: fused)."""

        amoeba_tiled = True

        def __init__(self, layer, product, groups, down_union, d_model, counters, box_ids, gate_W, gate_b,
                     gate_per_row=False):
            super().__init__()
            self._layer = layer
            self._product = product
            self._out = int(d_model)
            self._state = _State(counters)
            self._box_ids = tuple(box_ids)
            self._gate_per_row = bool(gate_per_row)
            self.groups = groups
            if gate_W is not None:
                self.gate_W = gate_W
                self.gate_b = gate_b
            if down_union is not None:
                self.down_union = down_union

        def projections(self):
            out = []
            for i, grp in enumerate(self.groups):
                out.append((f"groups.{i}.gate_proj", grp.gate_proj))
                out.append((f"groups.{i}.up_proj", grp.up_proj))
                if grp.own_down:
                    out.append((f"groups.{i}.down_proj", grp.down_proj))
            if "down_union" in self:
                out.append(("down_union", self.down_union))
            return tuple(out)

        def projection_topology(self):
            return self.projections()

        def layout(self):
            return {"groups": [dict(grp.info) for grp in self.groups], "boxes": len(self._box_ids)}

        def __call__(self, x, row_exact=False):
            from .models.varlen_dense_mlp import compact_rows

            _refuse_row_exact_kwarg(row_exact)

            router = self._state.router
            if router is not None and router.restricted():
                with router.lock:                          # rows name batch rows: run uncompacted
                    router.compaction_declines += 1
            else:
                compacted = compact_rows(x)
                if compacted is not None:
                    return compacted.restore(self(compacted.values))
            c = self._state.counters
            rows = int(x.shape[0] * x.shape[1]) if x.ndim == 3 else int(x.shape[0])
            with c["lock"]:
                c["forwards"] += 1
                c["rows"] += rows
            masks = {}                                     # one 0/1 cast of the bool gates per group dtype
            if self._box_ids:
                # Column 0 of gate_W/gate_b is the always-open segment (w = 0, b = +50, so
                # sigmoid(50) = 1 > 0.5 exactly): the mask needs no per-forward ones/concat.
                opened = gate_open_many(x, self.gate_W, self.gate_b, per_row=self._gate_per_row)
                if gate_counting_enabled():
                    with c["lock"]:
                        c["pending"].append((("",) + self._box_ids,
                                             opened.astype(mx.float32).reshape(-1, 1 + len(self._box_ids)).sum(axis=0)))
                        for sid in self._box_ids:
                            c["box_rows"][sid] = c["box_rows"].get(sid, 0) + rows
                        fold_gate_counts(c)
            u = None
            own = []
            for grp in self.groups:
                if grp.tile is not None:                   # a router tile: its open rows only
                    act = lambda r, g=grp: self._product(g.gate_proj(r), g.up_proj(r))  # noqa: E731
                    if grp.own_down:
                        part = _router_part(router, grp.tile, x, act, lambda a, g=grp: g.down_proj(a))
                        if part is not None:
                            own.append(part)
                        continue

                    def scattered(a, g=grp):
                        out = None
                        for mp in g.maps:
                            part = mx.take(a, mp, axis=-1)
                            out = part if out is None else out + part
                        return out
                    part = _router_part(router, grp.tile, x, act, scattered)
                    if part is not None:
                        u = part if u is None else u + part
                    continue
                a = self._product(grp.gate_proj(x), grp.up_proj(x))
                if grp.masked:
                    mask = masks.get(a.dtype)
                    if mask is None:
                        mask = masks[a.dtype] = opened.astype(a.dtype)
                    a = a * mx.take(mask, grp.seg, axis=-1)
                if grp.own_down:
                    own.append(grp.down_proj(a))
                    continue
                for mp in grp.maps:
                    part = mx.take(a, mp, axis=-1)
                    u = part if u is None else u + part
            y = None if u is None else self.down_union(u)
            for extra in own:
                y = extra if y is None else y + extra
            if y is None:
                y = mx.zeros(x.shape[:-1] + (self._out,), dtype=x.dtype)
            return y

    return AmoebaFusedGroup, AmoebaFusedTiledMLP


def make_neuron_order_mlp_class():
    import mlx.core as mx
    import mlx.nn as nn

    class AmoebaNeuronOrderMLP(nn.Module):
        """A tiled layer in original neuron order (``LAYOUTS``: neuron_order): the source MLP's own
        gate/up/down modules, and ``owner`` (per neuron: 0 the core, 1 + j router tile j)."""

        amoeba_tiled = True

        def __init__(self, layer, product, gate_proj, up_proj, down_proj, owner, tiles, names, counters):
            super().__init__()
            self._layer = layer
            self._product = product
            self._state = _State(counters)
            self._tiles = tuple(tiles)                    # router tiles with neurons in this layer
            self._names = tuple(names)
            self.gate_proj = gate_proj
            self.up_proj = up_proj
            self.down_proj = down_proj
            self.owner = owner

        def projections(self):
            return tuple((name, getattr(self, attr)) for name, attr in
                         zip(self._names, ("gate_proj", "up_proj", "down_proj")))

        def projection_topology(self):
            return self.projections()

        def layout(self):
            return {"sources": 1 + len(self._tiles), "boxes": 0, "router_tiles": len(self._tiles),
                    "neuron_order": True}

        def __call__(self, x, row_exact=False):
            from .models.varlen_dense_mlp import compact_rows

            _refuse_row_exact_kwarg(row_exact)

            router = self._state.router
            restricted = router is not None and router.restricted()
            if not restricted:
                compacted = compact_rows(x)               # the stock MLP's own compaction
                if compacted is not None:
                    return compacted.restore(self(compacted.values))
            c = self._state.counters
            rows = int(x.shape[0] * x.shape[1]) if x.ndim == 3 else int(x.shape[0])
            with c["lock"]:
                c["forwards"] += 1
                c["rows"] += rows
            if restricted:
                with router.lock:
                    router.compaction_declines += 1
            a = self._product(self.gate_proj(x), self.up_proj(x))
            mask = self._mask(router, restricted, x, a)
            return self.down_proj(a if mask is None else a * mask)   # all open: the stock computation

        def amoeba_activation_mask(self, x, a):
            """For a route that runs this layer's projections with its own arithmetic (the
            TensorFold tree verify): None when every row is fully open (that route's stock MLP
            arithmetic, unchanged), else the 0/1 mask to multiply the activation ``a`` by. Runs the
            router's fail-closed seam check and counters, like a call of the module."""
            router = self._state.router
            restricted = router is not None and router.restricted()
            c = self._state.counters
            with c["lock"]:
                c["forwards"] += 1
                c["rows"] += int(x.shape[0] * x.shape[1]) if x.ndim == 3 else int(x.shape[0])
            return self._mask(router, restricted, x, a)

        def _mask(self, router, restricted, x, a):
            if not restricted:
                if router is not None:
                    router.count_all_open(x, self._tiles)
                return None
            table, axis = router.open_table(x, self._tiles)
            mask = mx.take(table, self.owner, axis=1)       # [row-axis rows, N]
            shape = [1] * a.ndim
            shape[axis], shape[-1] = int(mask.shape[0]), int(mask.shape[1])
            return mask.astype(a.dtype).reshape(shape)

    return AmoebaNeuronOrderMLP


def neuron_order_refusal(manifest, tensors):
    """None when the manifest can take the neuron_order layout (every box a router tile, every tile
    an exact host slice into the down union, core + tiles a full partition of every tiled layer),
    else why not. The weight-equality check runs when the layer is built."""
    import numpy as np
    rules = router_tiles(manifest)
    if not rules or len(rules) != len(manifest["boxes"]):
        return "needs every box on the router law"
    for spec in [manifest["core"], *manifest["boxes"]]:
        if spec.get("bits") is not None or spec.get("down", "union") != "union":
            return f"tile {spec['id']} is requantized or has its own down block (not an exact slice)"
    n = int(manifest["intermediate"])
    ids = [manifest["core"]["id"]] + [b["id"] for b in manifest["boxes"]]
    for l in manifest["tiled_layers"]:
        seen = np.zeros(n, np.int64)
        for sid in ids:
            key = f"L{int(l)}.{sid}.rows"
            if key in tensors:
                np.add.at(seen, np.array(tensors[key]).astype(np.int64), 1)
        if not np.all(seen == 1):
            return f"L{int(l)}: core and tiles are not a full partition"
    return None


def build_neuron_order(live_mlp, layer, tensors, manifest, declaration, counters, router_order):
    """The neuron-order layer: checks that the core and router tiles partition the layer as exact
    host slices of the live modules, then reuses those modules (no weight copy)."""
    import numpy as np
    import mlx.core as mx
    gname, uname, dname = declaration["projections"]
    live = {n: _qparams(getattr(live_mlp, n)) for n in (gname, uname)}
    N = int(manifest["intermediate"])
    live_rows = int(live[gname][1]["weight"].shape[0])
    if live_rows != N:
        raise ValueError(f"amoeba tiles: L{layer} manifest intermediate {N} != live {gname} rows {live_rows}")
    specs = {manifest["core"]["id"]: manifest["core"], **{b["id"]: b for b in manifest["boxes"]}}
    rules = router_tiles(manifest)
    if len(rules) != len(manifest["boxes"]):
        raise ValueError("amoeba tiles: neuron_order needs every box on the router law")
    column = {t: 1 + j for j, t in enumerate(router_order)}
    owner = np.full(N, -1, np.int64)
    present = []
    for sid, spec in specs.items():
        key = f"L{layer}.{sid}"
        if f"{key}.rows" not in tensors:
            continue
        if spec.get("bits") is not None or spec.get("down", "union") != "union":
            raise ValueError(f"amoeba tiles: neuron_order needs exact host slices; tile {sid} is "
                             f"requantized ({spec.get('bits')} bits, own down): not exact")
        rows = np.array(tensors[f"{key}.rows"]).astype(np.int64)
        if np.any(owner[rows] != -1):
            raise ValueError(f"amoeba tiles: L{layer} neuron_order: tile {sid} overlaps another tile")
        owner[rows] = 0 if sid == manifest["core"]["id"] else column[sid]
        index = mx.array(rows.astype(np.int32))
        for proj, name in (("gate", gname), ("up", uname)):
            for k, arr in live[name][1].items():
                if not mx.array_equal(tensors[f"{key}.{proj}.{k}"], mx.take(arr, index, axis=0)).item():
                    raise ValueError(f"amoeba tiles: L{layer} tile {sid} {proj}.{k} is not the live "
                                     "module's rows (neuron_order needs exact slices)")
        if sid != manifest["core"]["id"]:
            present.append(sid)
    if np.any(owner < 0):
        raise ValueError(f"amoeba tiles: L{layer} neuron_order: core and tiles cover "
                         f"{int(np.sum(owner >= 0))}/{N} neurons (not a full partition)")
    cls = make_neuron_order_mlp_class()
    return cls(layer, _product_fn(declaration), getattr(live_mlp, gname), getattr(live_mlp, uname),
               getattr(live_mlp, dname), mx.array(owner.astype(np.int32)),
               [t for t in router_order if t in present], (gname, uname, dname), counters)


def _format_key(p):
    """Projection format of a gate/up/down parameter dict (fused only within one format)."""
    if p["kind"] == "quantized":
        return ("quantized", int(p["bits"]), int(p["group_size"]), p.get("mode") or "affine",
                str(p["scales"].dtype), p.get("biases") is not None)
    return ("dense", str(p["weight"].dtype))


def _format_name(key):
    if key[0] == "quantized":
        return f"q{key[1]}g{key[2]}-{key[3]}"
    return key[1].replace("mlx.core.", "")


def _concat_rows(parts, *, zero_rows_after=None):
    """Concatenate same-format parameter dicts along the output (row) axis.

    ``zero_rows_after``: per part, zero rows appended after it (exactly zero outputs)."""
    import mlx.core as mx
    extra = zero_rows_after or [0] * len(parts)
    keys = [k for k in ("weight", "scales", "biases") if parts[0].get(k) is not None]
    out = {k: parts[0].get(k) for k in ("kind", "bits", "group_size", "mode")}
    for k in keys:
        out[k] = mx.concatenate([_pad_axis(pp[k], 0, e) for pp, e in zip(parts, extra)], axis=0)
    return out


def _concat_inputs(parts):
    """Concatenate same-format down blocks along the input axis (whole groups / 64-column blocks)."""
    import mlx.core as mx
    keys = [k for k in ("weight", "scales", "biases") if parts[0].get(k) is not None]
    out = {k: parts[0].get(k) for k in ("kind", "bits", "group_size", "mode")}
    for k in keys:
        out[k] = mx.concatenate([pp[k] for pp in parts], axis=1)
    return out


def build_fused(layer, sources, down, union_width, d_model, product, counters, *, gate_per_row=False):
    """The fused layout of one layer from validated source dicts (see ``build_layer``)."""
    import numpy as np
    import mlx.core as mx
    group_cls, mlp_cls = make_fused_tiled_mlp_class()
    boxes = [src for src in sources if src.get("gate_w") is not None]
    box_col = {src["id"]: 1 + j for j, src in enumerate(boxes)}
    groups = []
    # Router tiles get one group each (after the fused groups), so a closed tile is a skipped group.
    routed = [src for src in sources if src.get("router")]
    sources = [src for src in sources if not src.get("router")]
    # Union sources: exact host slices (one format); one gate/up over all their rows, scattered by
    # overlap multiplicity layers in source order (unfused adds the per-source parts in that order).
    union = [src for src in sources if src.get("map") is not None]
    by_format = {}
    for src in union:
        by_format.setdefault((_format_key(src["gate"]), _format_key(src["up"])), []).append(src)
    union_sets = [(members, None) for members in by_format.values()]
    union_sets += [([src], src["id"]) for src in routed if src.get("map") is not None]
    for members, tile in union_sets:
        gkey = _format_key(members[0]["gate"])
        rows = [int(src["gate"]["weight"].shape[0]) for src in members]
        total = sum(rows)
        extra = FUSED_UNION_ROW_MULTIPLE - (total % FUSED_UNION_ROW_MULTIPLE)   # >= 1 zero row
        zero_row = total
        tail = [0] * (len(members) - 1) + [extra]
        gate = make_linear(_concat_rows([src["gate"] for src in members], zero_rows_after=tail))
        up = make_linear(_concat_rows([src["up"] for src in members], zero_rows_after=tail))
        seg = np.zeros(total + extra, np.int32)
        hits = [[] for _ in range(union_width)]
        offset = 0
        for src, n in zip(members, rows):
            seg[offset:offset + n] = box_col.get(src["id"], 0)
            mp = np.array(src["map"]).astype(np.int64)
            for position in np.flatnonzero(mp < n):
                hits[position].append(offset + int(mp[position]))
            offset += n
        layers_k = max([len(h) for h in hits] + [1])
        maps = []
        for k in range(layers_k):
            mp = np.full(union_width, zero_row, np.int32)
            for position, h in enumerate(hits):
                if len(h) > k:
                    mp[position] = h[k]
            maps.append(mx.array(mp))
        masked = any(src["id"] in box_col for src in members)
        groups.append(group_cls(gate, up, mx.array(seg), masked=masked, maps=maps, tile=tile, info={
            "kind": "union", "format": _format_name(gkey), "rows": total, "zero_rows": extra,
            "segments": [[src["id"], n] for src, n in zip(members, rows)], "scatter_layers": layers_k,
            **({"router_tile": tile} if tile is not None else {})}))
    # Own-down sources: per (gate, up, down) format, gate/up rows laid out exactly like the
    # concatenated down block's input columns (each source's rows, then zero rows to its block width).
    own = [src for src in sources if src.get("down") is not None]
    by_format = {}
    for src in own:
        key = (_format_key(src["gate"]), _format_key(src["up"]), _format_key(src["down"]))
        by_format.setdefault(key, []).append(src)
    own_sets = [(key, members, None) for key, members in by_format.items()]
    own_sets += [((_format_key(src["gate"]), _format_key(src["up"]), _format_key(src["down"])), [src], src["id"])
                 for src in routed if src.get("down") is not None]
    for (gkey, ukey, dkey), members, tile in own_sets:
        rows, blocks, downs = [], [], []
        for src in members:
            n = int(src["gate"]["weight"].shape[0])
            d = src["down"]
            if d["kind"] == "dense":
                width = int(d["weight"].shape[1])
                pad = (-width) % DENSE_DOWN_INPUT_MULTIPLE
                d = {**d, "weight": _pad_axis(d["weight"], 1, pad)}
                width += pad
            else:
                width = int(d["scales"].shape[1]) * int(d["group_size"])
            rows.append(n)
            blocks.append(width)
            downs.append(d)
        gate = make_linear(_concat_rows([src["gate"] for src in members],
                                        zero_rows_after=[b - n for b, n in zip(blocks, rows)]))
        up = make_linear(_concat_rows([src["up"] for src in members],
                                      zero_rows_after=[b - n for b, n in zip(blocks, rows)]))
        seg = np.zeros(sum(blocks), np.int32)
        offset = 0
        for src, b in zip(members, blocks):
            seg[offset:offset + b] = box_col.get(src["id"], 0)
            offset += b
        masked = any(src["id"] in box_col for src in members)
        groups.append(group_cls(gate, up, mx.array(seg), masked=masked, down_proj=make_linear(_concat_inputs(downs)),
                                tile=tile,
                                info={"kind": "own_down", "format": _format_name(gkey),
                                      "down_format": _format_name(dkey), "rows": sum(rows),
                                      "zero_rows": sum(blocks) - sum(rows),
                                      "segments": [[src["id"], n] for src, n in zip(members, rows)],
                                      **({"router_tile": tile} if tile is not None else {})}))
    gate_W = gate_b = None
    if boxes:
        always_w = mx.zeros_like(boxes[0]["gate_w"])
        always_b = mx.array([50.0], dtype=boxes[0]["gate_b"].dtype)
        gate_W = mx.stack([always_w] + [src["gate_w"] for src in boxes])
        gate_b = mx.concatenate([always_b] + [src["gate_b"] for src in boxes])
    down_union = make_linear(down) if union_width else None
    return mlp_cls(layer, product, groups, down_union, d_model, counters, [src["id"] for src in boxes],
                   gate_W, gate_b, gate_per_row=gate_per_row)


def _check_width(layer, what, p, width):
    """A projection's input width from its packed shapes equals ``width``."""
    if p["kind"] == "quantized":
        words = int(p["weight"].shape[1])
        groups = int(p["scales"].shape[1])
        if words * 32 != width * p["bits"] or groups * p["group_size"] != width:
            raise ValueError(f"amoeba tiles: L{layer} {what} packed width disagrees with {width} inputs at {p['bits']} bits")
    elif int(p["weight"].shape[1]) != width:
        raise ValueError(f"amoeba tiles: L{layer} {what} width disagrees with {width} inputs")


def _product_fn(declaration):
    """``gate, up -> act(gate) * up``: the declared fused product (the stock MLP's compiled swiglu)
    when the adapter declares one, else the activation then a multiply."""
    product = declaration.get("product_fn")
    if product is not None:
        return product
    act = declaration["activation_fn"]
    return lambda g, u: act(g) * u


def build_layer(live_mlp, layer, tensors, manifest, declaration, counters, gate_per_row=False,
                layout="unfused"):
    """Build one tiled MLP from tile tensors; checks every shape, map and gate against the live module."""
    import numpy as np
    import mlx.core as mx
    gname, uname, dname = declaration["projections"]
    live = {n: _qparams(getattr(live_mlp, n)) for n in (gname, uname, dname)}
    N = int(manifest["intermediate"])
    live_rows = int(live[gname][1]["weight"].shape[0])
    if live_rows != N:
        raise ValueError(f"amoeba tiles: manifest intermediate {N} != live {live_rows}")
    prefix = f"L{layer}"
    gs = int(manifest["down_group"])
    groups = np.array(tensors[f"{prefix}.down.groups"]).astype(np.int64)
    if groups.ndim != 1 or (len(groups) and (np.any(np.diff(groups) <= 0) or groups[0] < 0 or groups[-1] >= N // gs)):
        raise ValueError(f"amoeba tiles: L{layer} down groups must be strictly increasing and below {N // gs}")
    union = (groups[:, None] * gs + np.arange(gs)[None]).reshape(-1)
    union_width = len(union)
    kind, arrays, bits, group_size, mode = live[dname]
    if kind == "quantized" and group_size != gs:
        raise ValueError(f"amoeba tiles: L{layer} down group {group_size} != manifest {gs}")
    d_model = int(arrays["weight"].shape[0])
    gk, garr, gb, gg, _ = live[gname]
    d_in = int(garr["scales"].shape[1]) * gg if gk == "quantized" else int(garr["weight"].shape[1])
    down = {"kind": kind, "bits": bits, "group_size": group_size, "mode": mode}
    for k in arrays:
        t = tensors[f"{prefix}.down.{k}"]
        if t.shape[0] != arrays[k].shape[0] or t.dtype != arrays[k].dtype:
            raise ValueError(f"amoeba tiles: L{layer} down.{k} {t.shape}/{t.dtype} vs live {arrays[k].shape}/{arrays[k].dtype}")
        down[k] = t
    if union_width:
        _check_width(layer, "down union", down, union_width)
    pos = {int(n): i for i, n in enumerate(union)}
    sources = []
    specs = {manifest["core"]["id"]: manifest["core"], **{b["id"]: b for b in manifest["boxes"]}}
    core_id = manifest["core"]["id"]
    for sid in [core_id] + [b["id"] for b in manifest["boxes"]]:
        key = f"{prefix}.{sid}"
        if f"{key}.rows" not in tensors:
            continue                                  # this tile has no neurons in this layer
        spec = specs[sid]
        rows_np = np.array(tensors[f"{key}.rows"]).astype(np.int64)
        if rows_np.ndim != 1 or not len(rows_np) or rows_np.min() < 0 or rows_np.max() >= N or len(set(rows_np)) != len(rows_np):
            raise ValueError(f"amoeba tiles: L{layer} {sid} rows invalid")
        src = {"id": sid}
        dense16 = spec.get("bits") == 16             # BF16 tile from the full-precision source
        if dense16:
            for proj in ("gate", "up"):
                w = tensors[f"{key}.{proj}.weight"]
                if w.ndim != 2 or w.shape[0] != len(rows_np) or w.dtype not in (mx.bfloat16, mx.float16):
                    raise ValueError(f"amoeba tiles: L{layer} {sid}.{proj} 16-bit weight invalid")
                _check_width(layer, f"{sid}.{proj}", {"kind": "dense", "weight": w}, d_in)
                src[proj] = {"kind": "dense", "weight": w}
            if spec.get("down") != "own":
                raise ValueError(f"amoeba tiles: 16-bit tile {sid} needs its own down block")
            w = tensors[f"{key}.down.weight"]
            if w.ndim != 2 or w.shape[0] != d_model or w.dtype != src["gate"]["weight"].dtype:
                raise ValueError(f"amoeba tiles: L{layer} {sid}.down 16-bit weight invalid")
            _check_width(layer, f"{sid}.down", {"kind": "dense", "weight": w}, len(rows_np))
            src["down"] = {"kind": "dense", "weight": w}
        for proj, name in (("gate", gname), ("up", uname)) if not dense16 else ():
            pk, parr, pb, pg, pm = live[name]
            pbits = int(spec.get("bits") or pb or 0) or None
            p = {"kind": pk, "bits": pbits, "group_size": pg, "mode": pm}
            for k in parr:
                t = tensors[f"{key}.{proj}.{k}"]
                if t.shape[0] != len(rows_np) or (k != "weight" and t.dtype != parr[k].dtype):
                    raise ValueError(f"amoeba tiles: L{layer} {sid}.{proj}.{k} {t.shape}/{t.dtype} disagrees with live")
                p[k] = t
            in_width = int(parr["scales"].shape[1]) * pg if pk == "quantized" else int(parr["weight"].shape[1])
            _check_width(layer, f"{sid}.{proj}", p, in_width)
            src[proj] = p
        if dense16:
            pass
        elif spec.get("down", "union") == "own":
            if not spec.get("requantized_from"):
                raise ValueError(f"amoeba tiles: tile {sid} with its own down block must name its full-precision source")
            dk, darr, db, dg, dm = live[dname]
            dbits = int(spec.get("bits") or db)
            p = {"kind": dk, "bits": dbits, "group_size": dg, "mode": dm}
            for k in darr:
                t = tensors[f"{key}.down.{k}"]
                if t.shape[0] != darr[k].shape[0] or (k != "weight" and t.dtype != darr[k].dtype):
                    raise ValueError(f"amoeba tiles: L{layer} {sid}.down.{k} disagrees with live")
                p[k] = t
            src["down"] = p
            pad = int(tensors[f"{key}.down_pad"].item())
            _check_width(layer, f"{sid}.down", p, len(rows_np) + pad)
        else:
            mp = np.array(tensors[f"{key}.map"]).astype(np.int64)
            want = np.full(union_width, len(rows_np), np.int64)
            for i, n in enumerate(rows_np):
                if int(n) not in pos:
                    raise ValueError(f"amoeba tiles: L{layer} {sid} neuron {int(n)} outside the down union")
                want[pos[int(n)]] = i
            if mp.shape != want.shape or not np.array_equal(mp, want):
                raise ValueError(f"amoeba tiles: L{layer} {sid} map disagrees with rows and groups")
            # Index len(rows) (an absent neuron) lands on a zero: a padded gate/up row or the
            # appended zero column.
            src["map"] = mx.array(mp.astype(np.int32))
        has_gate = f"{key}.gate_w" in tensors
        if sid == core_id:
            if has_gate:
                raise ValueError("amoeba tiles: the core carries no gate")
        elif (spec.get("gate") or {}).get("law") == ROUTER_LAW:
            if has_gate:
                raise ValueError(f"amoeba tiles: router tile {sid} carries no learned gate")
            src["router"] = True
        else:
            gate = spec.get("gate") or {}
            if not has_gate or gate.get("law") != GATE_LAW:
                raise ValueError(f"amoeba tiles: box {sid} needs a gate with law {GATE_LAW!r} "
                                 f"or the {ROUTER_LAW!r} law")
            w, b = tensors[f"{key}.gate_w"], tensors[f"{key}.gate_b"]
            if tuple(w.shape) != (d_model,) or tuple(b.shape) != (1,):
                raise ValueError(f"amoeba tiles: L{layer} box {sid} gate shapes {w.shape}/{b.shape}")
            src["gate_w"], src["gate_b"] = w.astype(mx.float32), b.astype(mx.float32)
        sources.append(src)
    down_union_p = down if union_width else None
    if layout == "fused":
        return build_fused(layer, sources, down_union_p, union_width, d_model, _product_fn(declaration),
                           counters, gate_per_row=gate_per_row)
    source_cls = make_tile_source_class()
    modules = []
    for src in sources:
        down_mod = None
        if src.get("down") is not None:
            down_mod = make_linear(src["down"], in_multiple=DENSE_DOWN_INPUT_MULTIPLE
                                   if src["down"]["kind"] == "dense" else 1)
        modules.append(source_cls(
            src["id"], int(src["gate"]["weight"].shape[0]),
            make_linear(src["gate"], out_multiple=GATE_UP_ROW_MULTIPLE),
            make_linear(src["up"], out_multiple=GATE_UP_ROW_MULTIPLE),
            down_proj=down_mod, scatter_map=src.get("map"),
            gate_w=src.get("gate_w"), gate_b=src.get("gate_b"), router=bool(src.get("router"))))
    down_union = make_linear(down) if union_width else None
    cls = make_tiled_mlp_class()
    return cls(layer, _product_fn(declaration), modules, down_union, d_model, counters,
               gate_per_row=gate_per_row)


def tile_projection_modules(layers, attr="mlp"):
    """``(path, module)`` of every tile projection under ``layers`` (paths relative to the list)."""
    out = []
    for index, layer in enumerate(layers):
        mlp, prefix = unwrap_det(getattr(layer, attr, None), f"{index}.{attr}")
        if getattr(mlp, "amoeba_tiled", False):
            out.extend((f"{prefix}.{path}", module) for path, module in mlp.projections())
    return out


def unwrap_det(mlp, prefix=""):
    """The MLP a deterministic-tile wrapper (amoeba_det_tiles.DetTileMLP) holds, and its path."""
    while getattr(mlp, "amoeba_det_wrapped", False):
        mlp, prefix = mlp.inner, f"{prefix}.inner"
    return mlp, prefix


class AmoebaTileSet:
    """Installed tile set: identity, counters, request and route checks."""

    def __init__(self, manifest, counters, resident_bytes, *, layers, attr, route="ordinary",
                 drafter=None, gate_per_row=False, layout="unfused", product="activation(gate)*up",
                 router_partial_rows="mask", layout_refusal=None):
        self.manifest = manifest
        self.layout = layout
        self.layout_refusal = layout_refusal          # why "auto" did not pick neuron_order
        self.identity = tile_set_identity(manifest, route=route, gate_per_row=gate_per_row,
                                          layout=layout, product=product, router_partial_rows=router_partial_rows)
        self.gate_per_row = gate_per_row
        # How a verify row relates to ordinary decode on this route (set at the serving lane bind;
        # see classify_exactness). Never "exact" by default.
        self.exactness = {"exactness": "reference" if route == "ordinary" else "unclassified",
                          "basis": "not yet bound to a serving projection law"}
        self._layout_detail = self._compute_layout_detail(layers, attr)
        self._layout_compact = self._compact_layout(self._layout_detail)
        self.counters = counters
        self.resident_bytes = resident_bytes
        self.route = route
        self.drafter = drafter
        self.laws = {}
        self._layers = layers
        self._attr = attr

    @property
    def tile_ids(self):
        return self.identity["tiles"]

    def projection_modules(self):
        return tile_projection_modules(self._layers, self._attr)

    def check_request(self, requested):
        """A request may name the route's tile set (all of it); anything else fails closed.
        An absent field and an empty list both mean "no tiles named"."""
        if requested is None or requested == []:
            return
        if not isinstance(requested, (list, tuple)) or not all(isinstance(t, str) for t in requested):
            raise ValueError("mlx2_amoeba_tiles must be a list of tile ids")
        missing = sorted(set(requested) - set(self.tile_ids))
        if missing:
            raise TileNotLoaded(f"tiles not loaded on this route: {missing}")
        if sorted(set(requested)) != self.tile_ids:
            raise TileNotLoaded("mlx2_amoeba_tiles names the route's full tile set; open or close "
                                "router tiles per request with mlx2_amoeba_open")

    @property
    def router(self):
        """The shared :class:`AmoebaRouterRows`, or None (no router tiles)."""
        return self.counters.get("router")

    def open_request(self, spec):
        """Resolve ``mlx2_amoeba_open`` (``{"tiles": [...], "topics": [...]}``) to the request's open
        router tiles: a frozenset, or None when every router tile is open (absent field: the full
        model). A tile opens when named, or when any of its ``opens_on`` topics is named (OR rule).
        Unknown tiles or topics raise :class:`TileNotLoaded` (409); fails closed off the ordinary route.
        """
        if spec is None:
            return None
        if not isinstance(spec, dict) or not set(spec) <= {"tiles", "topics"} or not all(
                isinstance(v, list) and all(isinstance(t, str) for t in v) for v in spec.values()):
            raise ValueError('mlx2_amoeba_open must be {"tiles": [tile ids], "topics": [topic ids]}')
        rules = router_tiles(self.manifest)
        tiles, topics = set(spec.get("tiles", [])), set(spec.get("topics", []))
        if not rules:
            raise TileNotLoaded("this tile set has no router tiles; mlx2_amoeba_open names none")
        unknown = sorted(tiles - set(self.tile_ids))
        if unknown:
            raise TileNotLoaded(f"tiles not loaded on this route: {unknown}")
        fixed = sorted(tiles - set(rules))
        if fixed:
            raise ValueError(f"not router tiles (the core is always computed, gated boxes gate "
                             f"themselves): {fixed}")
        known_topics = {t for opens in rules.values() for t in opens}
        unknown = sorted(topics - known_topics)
        if unknown:
            raise TileNotLoaded(f"router topics not declared by this tile set: {unknown}")
        opened = frozenset(t for t, opens in rules.items() if t in tiles or topics & set(opens))
        if opened == frozenset(rules):
            return None
        refusal = getattr(self.router, "open_set_refusal", None)
        if refusal:
            raise ValueError(f"per-request amoeba open sets refused on this server: {refusal}")
        if self.route not in ROUTES_WITH_OPEN_SETS:
            raise ValueError(f"per-request amoeba open sets run on the {'/'.join(ROUTES_WITH_OPEN_SETS)} "
                             f"tile routes; the {self.route} route's verify forwards do not carry "
                             "per-row open sets")
        return opened

    @staticmethod
    def open_fingerprint(opened):
        """APCv2 scope component of a restricted open set (cached state is per open set)."""
        return "sha256:" + hashlib.sha256(json.dumps(sorted(opened)).encode()).hexdigest()

    def check_route(self, serving_route):
        """The serving route must be the one the tiles were installed for (fail closed)."""
        served = "ordinary" if serving_route is None else serving_route
        if served != self.route:
            raise ValueError(f"amoeba tiles were installed for the {self.route} route; "
                             f"this server's route is {served}")

    def record_law(self, name, value):
        """Note a projection law the route installed over the tile projections (receipts)."""
        self.laws[name] = value

    def class_law_refusal(self, name, predicate):
        """None when every tile projection satisfies ``predicate`` (a class-swap law such as the
        row-exact verify or slice-invariant prefill), else the first uncovered projection."""
        missing = [path for path, module in self.projection_modules() if not predicate(module)]
        if missing:
            return f"{name} does not cover {len(missing)} tile projections (first {missing[0]})"
        self.record_law(name, {"covered_tile_projections": len(self.projection_modules())})
        return None

    def lane_law_refusal(self, receipt):
        """Why the installed serving lane law cannot carry this tile route, or None.

        The ordinary route is its own reference, so partial coverage is only recorded. A
        speculative route needs each verify row to compute what the same row decoded alone
        computes, so every tile projection must run the same lane law as the rest of the trunk
        (the lane's all-rows/tree scope then reaches it); a stock tile projection beside lane
        trunk projections would break the law the route's exactness rests on.
        """
        from .lane import installer

        receipt = receipt or {}
        law = receipt.get("law_id")
        modules = self.projection_modules()
        uncovered = [path for path, module in modules if installer.installed_lane(module) is None]
        covered_any = bool(receipt.get("covered"))
        self.record_law("lane", {"law_id": law, "tile_projections": len(modules),
                                 "tile_projections_uncovered": len(uncovered)})
        if self.route == "ordinary" or not covered_any:
            return None
        if isinstance(law, str) and law.startswith("tensorfold-lane"):
            return ("the TensorFold process-wide lane patch is not audited for tile projections on a "
                    "speculative route; serve lane_matmul auto/exact or off")
        if uncovered:
            return (f"lane law {law} does not cover {len(uncovered)} tile projections (first {uncovered[0]}); "
                    "a verify row would not compute what ordinary decode computes")
        return None

    def layout_detail(self):
        """Per-layer projection layout (groups, formats, segments, scatter layers); computed at install."""
        return self._layout_detail

    def layout_summary(self):
        """The compact layout for receipts: counts and formats per layer type, not per segment."""
        return self._layout_compact

    @staticmethod
    def _compact_layout(detail):
        layers = detail["layers"].values()

        def span(values):
            values = list(values)
            return {"min": min(values), "max": max(values)} if values else None
        groups = [g for e in layers for g in e.get("groups", [])]
        return {"layout": detail["layout"], "layout_id": detail["layout_id"], "tiled_layers": len(detail["layers"]),
                "formats": detail["formats"],
                "group_formats": sorted({f"{g['kind']}:{g['format']}" for g in groups}),
                "groups_per_layer": span(len(e.get("groups", [])) for e in layers) if groups else None,
                "segments_per_layer": span(sum(len(g["segments"]) for g in e.get("groups", [])) for e in layers)
                if groups else span(e.get("sources", 0) for e in layers),
                "boxes_per_layer": span(e.get("boxes", 0) for e in layers),
                "scatter_layers_max": max([g.get("scatter_layers", 0) for g in groups] + [0]),
                "zero_rows_total": sum(g.get("zero_rows", 0) for g in groups),
                "projection_launches_per_layer_max": detail["projection_launches_per_layer_max"]}

    def _compute_layout_detail(self, layers_list, attr):
        layers = {}
        for index, layer in enumerate(layers_list):
            mlp, _ = unwrap_det(getattr(layer, attr, None))
            if not getattr(mlp, "amoeba_tiled", False):
                continue
            if callable(getattr(mlp, "layout", None)):
                layers[str(index)] = mlp.layout()
            else:
                layers[str(index)] = {"sources": len(mlp.sources),
                                      "boxes": sum(1 for src in mlp.sources if src.gated),
                                      **({"router_tiles": n} if (n := sum(1 for src in mlp.sources
                                                                         if src.router)) else {})}
        launches = []
        for entry in layers.values():
            if "groups" in entry:
                own = sum(1 for g in entry["groups"] if g["kind"] == "own_down")
                scatters = sum(g.get("scatter_layers", 0) for g in entry["groups"])
                launches.append(2 * len(entry["groups"]) + own + scatters + (1 if entry["boxes"] else 0))
        return {"layout": self.layout, "layout_id": LAYOUT_IDS[self.layout],
                "formats": sorted({g["format"] for e in layers.values() for g in e.get("groups", [])}),
                "projection_launches_per_layer_max": max(launches) if launches else None,
                "layers": layers}

    def classify_exactness(self, receipt, *, row_exact=False, fast_ast=False, tree_position_law=False):
        """Record (and return) what a verify row's equality with ordinary decode rests on.

        A speculative route is labelled exact only on one of these bases; anything else is admitted
        as ``near_tie_only`` (the same contract the untiled model has under that law: verify rows
        and one-row decode take different kernels, so tokens may differ at near ties, and with
        hard gates a row near a gate threshold may also differ):

        * ``row_exact``: the row-exact target verify executor (every tile projection class-swapped,
          audited at load);
        * ``lane_exact_min_rows_1``: a serving lane law covering every tile projection with
          ``min_rows == 1`` everywhere (one-row decode and every verify width take the lane
          kernel; the fast-AST ``lane_mpp_exact`` law is this case);
        * ``tree_position_law``: the TensorFold tree verify under its size-invariant tree law,
          whose forwards (serial and drafted) run inside ``lane.all_rows()``, with every tile
          projection lane-covered.

        The basis is a CPU-tested structure; Metal kernel parity is the GPU check's job
        (scripts/amoeba27b_spec_check.py), so receipts say ``gpu_verified: False``.
        """
        from .lane import installer

        if self.route == "ordinary":
            result = {"exactness": "reference", "basis": "the ordinary tiled route is the reference"}
        else:
            receipt = receipt or {}
            modules = self.projection_modules()
            covered = bool(modules) and all(installer.installed_lane(m) is not None for _, m in modules)
            tile_min = {int(getattr(m, "_lane_min_rows", 0)) for _, m in modules}
            declared = receipt.get("min_rows")
            minimums = set(declared.values()) if isinstance(declared, dict) else (
                {declared} if declared is not None else set())
            lane_exact = (bool(receipt.get("covered")) and covered and tile_min == {1}
                          and minimums <= {1} and bool(minimums))
            if row_exact:
                result = {"exactness": "row_exact", "basis": "target_verify_row_exact covers every tile projection"}
            elif fast_ast and lane_exact:
                result = {"exactness": "lane_exact_min_rows_1", "basis": "fast-AST lane_mpp_exact law, min_rows 1"}
            elif lane_exact:
                result = {"exactness": "lane_exact_min_rows_1",
                          "basis": "every projection, tiles included, on the lane law at min_rows 1"}
            elif tree_position_law and covered:
                result = {"exactness": "tree_position_law",
                          "basis": "TensorFold tree verify under lane.all_rows(); tiles lane-covered"}
            else:
                why = ("no lane law installed (stock multi-row verify vs one-row decode)"
                       if not receipt.get("covered") else
                       f"lane law {receipt.get('law_id')} crossover min_rows {sorted(minimums) or '?'}: "
                       "short decode calls take stock kernels, verify rows the lane kernel")
                result = {"exactness": "near_tie_only", "basis": why}
        result["gpu_verified"] = False
        self.exactness = result
        return dict(result)

    def receipt(self):
        """Process-cumulative counters (warmup included); one device sync for the gate totals."""
        import mlx.core as mx
        c = self.counters
        with c["lock"]:
            fold_gate_counts(c, force=True)
            totals = dict(c["open_totals"])
            forwards, rows, box_rows = c["forwards"], c["rows"], dict(c["box_rows"])
        if totals:
            mx.eval(list(totals.values()))
        opens = {}
        for key, vector in totals.items():
            for sid, value in zip(key, vector.tolist()):
                if sid:                                    # "" is the fused always-open segment
                    opens[sid] = opens.get(sid, 0.0) + float(value)
        return {"schema": RECEIPT_SCHEMA, **self.identity,
                "exactness": dict(self.exactness),
                "qualified": False, "qualification": "unqualified", "implemented": True, "selected": True,
                "observed_used": forwards > 0,
                "drafter": self.drafter,
                "fused_layout": self.layout_summary(),
                "neuron_order_refusal": self.layout_refusal,
                "projection_laws": dict(self.laws),
                "counters_scope": "process_cumulative",
                "counters": {"tiled_forwards": forwards, "tiled_rows": rows,
                             "box_open_rows": opens, "box_layer_rows": box_rows,
                             "box_open_fraction": {k: v / max(1, box_rows.get(k, 0)) for k, v in opens.items()},
                             "box_open_fraction_denominator": (
                                 "rows passed to the tiled layers that hold the box, summed over those layers: "
                                 "prefill, decode, every verify row (rejected drafts included), warmup, and "
                                 "padding rows unless varlen compaction removed them")},
                "resident_mlp_bytes": self.resident_bytes,
                **({"router": self.router.receipt()} if self.router is not None else {})}


def refuse_request_tiles(requested):
    """Routes without a tile set refuse any named tile (an empty list names none)."""
    if requested is not None and requested != []:
        raise TileNotLoaded("this route holds no amoeba tiles")


def install(model, layers, manifest_path, declaration, *, source_fingerprint, route="ordinary",
            drafter=None, gate_per_row=False, layout="unfused", router_partial_rows="mask"):
    """Replace the tiled layers' MLPs. ``layers`` is the decoder-layer list; returns AmoebaTileSet.

    Install before any mechanism that swaps or prepares the trunk's projections, so that mechanism
    covers the tile projections too.
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten
    if route not in ROUTES:
        raise ValueError(f"amoeba tiles: route must be one of {ROUTES}, got {route!r}")
    manifest = load_manifest(manifest_path)
    if manifest["source"]["fingerprint"] != source_fingerprint:
        raise ValueError("amoeba tiles: source fingerprint "
                         f"{manifest['source']['fingerprint']} does not match the loaded checkpoint {source_fingerprint}")
    if int(manifest["layers"]) != len(layers):
        raise ValueError(f"amoeba tiles: manifest has {manifest['layers']} layers, model has {len(layers)}")
    tensors = {}
    for name in manifest["files"]:
        if name.endswith(".safetensors"):
            tensors.update(mx.load(str(Path(manifest["_dir"]) / name)))
    import threading
    if type(gate_per_row) is not bool:
        raise ValueError("amoeba tiles: gate_per_row must be boolean")
    if layout not in LAYOUTS + ("auto",):
        raise ValueError(f"amoeba tiles: layout must be one of {LAYOUTS + ('auto',)}, got {layout!r}")
    layout_refusal = None
    if layout == "auto":
        # Exact all-open by default where the tile set allows it; the tile-group layout otherwise.
        layout_refusal = neuron_order_refusal(manifest, tensors)
        layout = "unfused" if layout_refusal else "neuron_order"
    if router_partial_rows not in ROUTER_PARTIAL_ROWS:
        raise ValueError(f"amoeba tiles: router_partial_rows must be one of {tuple(ROUTER_PARTIAL_ROWS)}")
    if layout == "neuron_order" and router_partial_rows != "mask":
        raise ValueError("amoeba tiles: neuron_order masks closed tiles inside the stock matmul; "
                         "router_partial_rows must be 'mask' (gather has no meaning there)")
    counters = {"forwards": 0, "rows": 0, "open_totals": {}, "box_rows": {}, "pending": [],
                "lock": threading.Lock()}
    if router_tiles(manifest):
        counters["router"] = AmoebaRouterRows(sorted(router_tiles(manifest)), threading.Lock(),
                                              partial_rows=router_partial_rows)
    attr = declaration["mlp"]
    tiled = [int(l) for l in manifest["tiled_layers"]]
    dense = set(int(l) for l in manifest["dense_layers"])
    if set(tiled) & dense or set(tiled) | dense != set(range(len(layers))):
        raise ValueError("amoeba tiles: tiled_layers and dense_layers must partition the layers")
    built = {}
    for l in tiled:
        if layout == "neuron_order":
            if counters.get("router") is None:
                raise ValueError("amoeba tiles: neuron_order is for router tile sets")
            built[l] = build_neuron_order(getattr(layers[l], attr), l, tensors, manifest, declaration,
                                          counters, counters["router"].tiles)
            continue
        built[l] = build_layer(getattr(layers[l], attr), l, tensors, manifest, declaration, counters,
                               gate_per_row=gate_per_row, layout=layout)
    for l, mod in built.items():
        setattr(layers[l], attr, mod)
    resident = 0
    for l in tiled:
        resident += sum(v.nbytes for _, v in tree_flatten(built[l].parameters()))
    for l in dense:
        resident += sum(v.nbytes for _, v in tree_flatten(getattr(layers[l], attr).parameters()))
    tensors.clear()
    if model is not None and counters.get("router") is not None:
        # The generator's row seam (bind_amoeba_rows) finds the router on the model.
        object.__setattr__(model, "_mlx2_amoeba_rows", counters["router"])
    if model is not None:
        mx.eval(model.parameters())
    mx.eval([built[l].parameters() for l in tiled])
    mx.clear_cache()
    return AmoebaTileSet(manifest, counters, resident, layers=list(layers), attr=attr, route=route,
                         drafter=drafter, gate_per_row=gate_per_row, layout=layout,
                         router_partial_rows=router_partial_rows, layout_refusal=layout_refusal,
                         product=declaration.get("product_name") or (
                             "declared-product" if declaration.get("product_fn") else "activation(gate)*up"))
