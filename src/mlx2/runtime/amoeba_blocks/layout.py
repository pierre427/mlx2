"""Block layout for gated aMoEba MLPs: TensorFold-style row packing plus the block tables.

CPU only (numpy).  Nothing here touches Metal.

What is packed, and why not the down columns
--------------------------------------------
A gated SwiGLU MLP (or one routed expert) is ``down(silu(gate(x)) * up(x))`` with ``gate``/``up``
of shape ``[F, d]`` and ``down`` of shape ``[d, F]``.  In MLX affine quantization a weight's
groups run along its *input* axis, so:

* ``gate``/``up`` rows are quantized independently (groups along ``d``).  Any permutation of
  their rows, with their scales and biases, is exact.  The packer reorders them so every tile's
  neurons form one contiguous run (core first), cut into chunks of at most ``chunk`` rows
  (the kernel's column-tile width).
* ``down`` is quantized along ``F``: one (scale, bias) per output row per ``group_size``
  neurons.  Moving one neuron's column into another group would need a new scale, so a
  neuron-level permutation of ``down`` is not exact.  ``down`` stays in its original layout.
  The kernels write the activation back in original neuron order through ``perm`` (the
  gate/up epilogue scatters, 2 bytes per neuron), and the down kernel skips an original
  quantization group only when every neuron in it belongs to a closed tile.  The group's
  owner tiles are precomputed (``gsig``), so the check reads a few bytes, not a full mask.

So the packed checkpoint is the original with each expert's ``gate``/``up`` rows permuted: the
same format, loadable by stock modules, and ``unpack_rows(pack_rows(w, perm), perm)`` returns
the original arrays bit for bit.

No padding.  Chunks start at any row: a weight row is whole bytes (``d * bits / 8``) so every
row start is aligned, and the kernels mask a short chunk through the MPP tensor extent (NAX)
or a row bound (SIMD), so padding rows would only add bytes read.

Tiles and gates
---------------
Tile 0 is the core (always open).  Every other tile has a set of member boxes; it opens when any
member box is open (pair, family and joint tiles: the OR rule; no tile has its own classifier).
A parent box opens whenever one of its descendants opens (the lowest-common-ancestor rule), so
box ``c`` opening opens every tile whose member set meets ``c``'s ancestors-or-self.  That
closure is folded offline into one bit matrix ``tile_box_bits[tile, word]``: tile ``t`` is open
iff ``always[t]`` or ``(tile_box_bits[t] & raw_box_open) != 0``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

SCHEMA = "mlx2.amoeba-blocks.v1"
PACKING_LAW = "gate-up-rows-tile-contiguous-core-first-down-original-v1"
CORE = 0
DEFAULT_CHUNK = 32


# ---------------------------------------------------------------------------
# Tile graph: boxes, tiles, ancestor closure
# ---------------------------------------------------------------------------


@dataclass
class TileGraph:
    """Boxes (with optional parents) and tiles (each a set of member boxes; tile 0 the core)."""

    boxes: list[str]
    tiles: list[str]
    members: list[tuple[int, ...]]          # per tile: member box indices (core: ())
    parent: list[int] = field(default_factory=list)   # per box: parent box index or -1

    def __post_init__(self):
        if not self.parent:
            self.parent = [-1] * len(self.boxes)
        if len(self.parent) != len(self.boxes):
            raise ValueError("one parent entry per box")
        if len(self.members) != len(self.tiles):
            raise ValueError("one member set per tile")
        if not self.tiles or self.members[CORE]:
            raise ValueError("tile 0 must be the core (no member boxes)")
        for t, mem in enumerate(self.members):
            if t != CORE and not mem:
                raise ValueError(f"tile {self.tiles[t]!r} has no member box (only the core is ungated)")
            for b in mem:
                if not 0 <= b < len(self.boxes):
                    raise ValueError(f"tile {self.tiles[t]!r}: box index {b} out of range")
        for b in range(len(self.boxes)):          # parent chains must be acyclic
            seen, p = set(), b
            while p != -1:
                if p in seen:
                    raise ValueError(f"box parent cycle through {self.boxes[b]!r}")
                seen.add(p)
                p = self.parent[p]

    @property
    def words(self) -> int:
        return max(1, -(-len(self.boxes) // 32))

    def ancestors_or_self(self, b: int) -> list[int]:
        out = []
        while b != -1:
            out.append(b)
            b = self.parent[b]
        return out

    def tile_box_matrix(self) -> np.ndarray:
        """bool [tiles, boxes]: raw box c open => tile t open (closure folded in)."""
        nb = len(self.boxes)
        m = np.zeros((len(self.tiles), nb), bool)
        opened_by = [[] for _ in range(nb)]       # box j -> raw boxes whose opening opens j
        for c in range(nb):
            for a in self.ancestors_or_self(c):
                opened_by[a].append(c)
        for t, mem in enumerate(self.members):
            for j in mem:
                m[t, opened_by[j]] = True
        return m

    def tile_box_bits(self) -> np.ndarray:
        """uint32 [tiles, words]: the closure matrix as bit words (bit c%32 of word c//32)."""
        m = self.tile_box_matrix()
        bits = np.zeros((len(self.tiles), self.words), np.uint32)
        for c in range(len(self.boxes)):
            bits[m[:, c], c // 32] |= np.uint32(1 << (c % 32))
        return bits

    def always(self) -> np.ndarray:
        a = np.zeros(len(self.tiles), np.uint8)
        a[CORE] = 1
        return a

    def tile_open(self, box_open: np.ndarray) -> np.ndarray:
        """Reference: uint8 [rows, tiles] from raw box open bits [rows, boxes] (numpy)."""
        box_open = np.asarray(box_open, bool).reshape(-1, len(self.boxes))
        out = (box_open.astype(np.int64) @ self.tile_box_matrix().T.astype(np.int64)) > 0
        out[:, CORE] = True
        return out.astype(np.uint8)

    def digest(self) -> str:
        blob = json.dumps({"boxes": self.boxes, "tiles": self.tiles,
                           "members": [list(m) for m in self.members], "parent": self.parent},
                          sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()

    def to_json(self) -> dict:
        return {"boxes": self.boxes, "tiles": self.tiles,
                "members": [list(m) for m in self.members], "parent": self.parent}

    @classmethod
    def from_json(cls, d: dict) -> "TileGraph":
        return cls(list(d["boxes"]), list(d["tiles"]), [tuple(m) for m in d["members"]],
                   list(d.get("parent") or []))


def graph_from_partition_names(names: list[str], *, parents: dict[str, str] | None = None,
                               groups: dict[str, list[str]] | None = None) -> TileGraph:
    """A TileGraph from an aMoEba FN partition's tile names (``tiles.npz`` keys).

    ``core`` is tile 0; ``box:<id>`` is a box tile with member ``<id>``; ``pair:<a>__<b>`` has
    members ``a`` and ``b``; any other ``<kind>:<id>`` (family/joint tiles) takes its members from
    ``groups[name]``.  ``parents`` maps a box id to its parent box id (nesting)."""
    if not names or names[0] != "core":
        raise ValueError("the first tile name must be 'core'")
    boxes: list[str] = []
    index: dict[str, int] = {}

    def box(b: str) -> int:
        if b not in index:
            index[b] = len(boxes)
            boxes.append(b)
        return index[b]

    members: list[tuple[int, ...]] = [()]
    for name in names[1:]:
        kind, _, ident = name.partition(":")
        if kind == "box":
            members.append((box(ident),))
        elif kind == "pair":
            a, sep, b = ident.partition("__")
            if not sep:
                raise ValueError(f"pair tile {name!r} is not 'pair:a__b'")
            members.append(tuple(sorted({box(a), box(b)})))
        else:
            if not groups or name not in groups:
                raise ValueError(f"tile {name!r} needs its member boxes in groups")
            members.append(tuple(sorted({box(b) for b in groups[name]})))
    links = [(box(child), box(par)) for child, par in (parents or {}).items()]   # register every name first
    parent = [-1] * len(boxes)
    for child, par in links:
        parent[child] = par
    return TileGraph(boxes, list(names), members, parent)


# ---------------------------------------------------------------------------
# Per-layer block table
# ---------------------------------------------------------------------------


@dataclass
class LayerBlocks:
    """Block tables of one MLP layer (``experts`` = 1 for a dense MLP).

    perm[e, p]      original neuron index of packed gate/up row p of expert e (uint16)
    chunk_ptr[e]    expert e's chunks are chunks[chunk_ptr[e]:chunk_ptr[e+1]]
    chunks[c]       (offset, length, tile) of a packed row run, length <= chunk
    gsig_ptr[i]     original down group i = e * groups + g owns tiles gsig[gsig_ptr[i]:gsig_ptr[i+1]]
    """

    experts: int
    width: int                  # F: neurons per expert
    group_size: int             # down quantization group (along F)
    chunk: int
    perm: np.ndarray
    chunk_ptr: np.ndarray
    chunks: np.ndarray
    gsig_ptr: np.ndarray
    gsig: np.ndarray
    tiles: int                  # tile count of the graph this layer indexes

    @property
    def groups(self) -> int:
        return self.width // self.group_size

    @property
    def max_chunks(self) -> int:
        return int(np.max(np.diff(self.chunk_ptr))) if self.experts else 0

    def inverse(self) -> np.ndarray:
        """int64 [E, F]: packed row of each original neuron."""
        inv = np.empty((self.experts, self.width), np.int64)
        rows = np.arange(self.width)
        for e in range(self.experts):
            inv[e, self.perm[e].astype(np.int64)] = rows
        return inv

    def owner(self) -> np.ndarray:
        """int32 [E, F]: the tile of each original neuron (rebuilt from the chunk table)."""
        own = np.full((self.experts, self.width), -1, np.int32)
        for e in range(self.experts):
            for off, ln, t in self.chunks[self.chunk_ptr[e]:self.chunk_ptr[e + 1]]:
                own[e, self.perm[e, off:off + ln].astype(np.int64)] = t
        return own

    def validate(self) -> None:
        E, F = self.experts, self.width
        if F % self.group_size or F > 65535:
            raise ValueError("width must be a multiple of the down group size and fit uint16")
        if self.perm.shape != (E, F) or self.perm.dtype != np.uint16:
            raise ValueError("perm must be uint16 [E, F]")
        for e in range(E):
            if not np.array_equal(np.sort(self.perm[e].astype(np.int64)), np.arange(F)):
                raise ValueError(f"expert {e}: perm is not a permutation")
            ch = self.chunks[self.chunk_ptr[e]:self.chunk_ptr[e + 1]]
            pos = 0
            for off, ln, t in ch:
                if off != pos or not 1 <= ln <= self.chunk or not 0 <= t < self.tiles:
                    raise ValueError(f"expert {e}: chunks do not tile the packed rows")
                pos += ln
            if pos != F:
                raise ValueError(f"expert {e}: chunks cover {pos}/{F} rows")
        own = self.owner()
        G = self.groups
        for i in range(E * G):
            e, g = divmod(i, G)
            want = np.unique(own[e, g * self.group_size:(g + 1) * self.group_size])
            got = self.gsig[self.gsig_ptr[i]:self.gsig_ptr[i + 1]]
            if not np.array_equal(want, got):
                raise ValueError(f"expert {e} group {g}: signature does not match the owners")

    def digest(self) -> str:
        h = hashlib.sha256()
        h.update(json.dumps([self.experts, self.width, self.group_size, self.chunk, self.tiles]).encode())
        for a in (self.perm, self.chunk_ptr, self.chunks, self.gsig_ptr, self.gsig):
            h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()

    def arrays(self, prefix: str = "") -> dict[str, np.ndarray]:
        return {f"{prefix}perm": self.perm, f"{prefix}chunk_ptr": self.chunk_ptr,
                f"{prefix}chunks": self.chunks, f"{prefix}gsig_ptr": self.gsig_ptr,
                f"{prefix}gsig": self.gsig}

    def meta(self) -> dict:
        return {"experts": self.experts, "width": self.width, "group_size": self.group_size,
                "chunk": self.chunk, "tiles": self.tiles, "digest": self.digest()}

    @classmethod
    def from_arrays(cls, meta: dict, arrays: dict, prefix: str = "") -> "LayerBlocks":
        lb = cls(int(meta["experts"]), int(meta["width"]), int(meta["group_size"]), int(meta["chunk"]),
                 np.asarray(arrays[f"{prefix}perm"], np.uint16),
                 np.asarray(arrays[f"{prefix}chunk_ptr"], np.int32),
                 np.asarray(arrays[f"{prefix}chunks"], np.int32),
                 np.asarray(arrays[f"{prefix}gsig_ptr"], np.int32),
                 np.asarray(arrays[f"{prefix}gsig"], np.int32), int(meta["tiles"]))
        if meta.get("digest") and lb.digest() != meta["digest"]:
            raise ValueError("block table digest does not match its manifest")
        return lb


def plan_layer(owner: np.ndarray, *, tiles: int, group_size: int, chunk: int = DEFAULT_CHUNK) -> LayerBlocks:
    """Pack one layer: ``owner`` is int [E, F] (or [F] for a dense MLP), the tile of every neuron.

    Packed order per expert: the core's neurons first, then each tile's neurons in tile-id order,
    original order inside a tile (a stable sort by tile), cut into chunks of at most ``chunk``."""
    owner = np.asarray(owner)
    if owner.ndim == 1:
        owner = owner[None]
    if owner.ndim != 2:
        raise ValueError("owner must be [F] or [E, F]")
    E, F = owner.shape
    if F % group_size:
        raise ValueError(f"width {F} is not a multiple of the down group size {group_size}")
    if F > 65535:
        raise ValueError("width must fit uint16")
    if owner.min() < 0 or owner.max() >= tiles:
        raise ValueError("owner tile out of range (every neuron needs a tile; the core is 0)")
    if chunk < 1:
        raise ValueError("chunk must be positive")
    perm = np.empty((E, F), np.uint16)
    chunk_ptr = np.zeros(E + 1, np.int32)
    chunks: list[tuple[int, int, int]] = []
    G = F // group_size
    gsig_ptr = np.zeros(E * G + 1, np.int32)
    gsig: list[int] = []
    for e in range(E):
        order = np.argsort(owner[e], kind="stable")
        perm[e] = order.astype(np.uint16)
        sorted_owner = owner[e][order]
        start = 0
        while start < F:
            t = int(sorted_owner[start])
            end = start
            while end < F and sorted_owner[end] == t:
                end += 1
            for off in range(start, end, chunk):
                chunks.append((off, min(chunk, end - off), t))
            start = end
        chunk_ptr[e + 1] = len(chunks)
        for g in range(G):
            sig = np.unique(owner[e, g * group_size:(g + 1) * group_size])
            gsig.extend(int(s) for s in sig)
            gsig_ptr[e * G + g + 1] = len(gsig)
    lb = LayerBlocks(E, F, group_size, chunk, perm, chunk_ptr,
                     np.asarray(chunks, np.int32).reshape(-1, 3), gsig_ptr,
                     np.asarray(gsig, np.int32), tiles)
    return lb


def identity_layer(experts: int, width: int, *, group_size: int, chunk: int = DEFAULT_CHUNK,
                   tiles: int = 1) -> LayerBlocks:
    """The unpacked layout (every neuron in the core, original order): the law's dense base."""
    return plan_layer(np.zeros((experts, width), np.int32), tiles=tiles, group_size=group_size, chunk=chunk)


# ---------------------------------------------------------------------------
# Weight packing (row permutation of gate/up; exact)
# ---------------------------------------------------------------------------


def pack_rows(a: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Rows of ``a`` ([F, ...] or [E, F, ...]) in packed order: packed[e, p] = a[e, perm[e, p]]."""
    a = np.asarray(a)
    perm = np.asarray(perm).astype(np.int64)
    if a.ndim >= 1 and perm.shape[0] == 1 and a.shape[0] == perm.shape[1]:
        return a[perm[0]]
    if a.shape[:2] != perm.shape:
        raise ValueError(f"rows {a.shape[:2]} do not match perm {perm.shape}")
    return np.take_along_axis(a, perm.reshape(perm.shape + (1,) * (a.ndim - 2)), axis=1)


def unpack_rows(packed: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Inverse of :func:`pack_rows` (bit for bit)."""
    packed = np.asarray(packed)
    perm = np.asarray(perm).astype(np.int64)
    out = np.empty_like(packed)
    if perm.shape[0] == 1 and packed.shape[0] == perm.shape[1]:
        out[perm[0]] = packed
        return out
    for e in range(perm.shape[0]):
        out[e, perm[e]] = packed[e]
    return out


def owner_from_masks(masks: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    """int32 owner array from exclusive per-tile bool masks (``tiles.npz`` style), names[0] = core."""
    shape = np.asarray(masks[names[0]]).shape
    own = np.full(shape, -1, np.int32)
    count = np.zeros(shape, np.int32)
    for t, n in enumerate(names):
        m = np.asarray(masks[n], bool)
        own[m] = t
        count += m
    if not np.all(count == 1):
        raise ValueError("tile masks are not an exclusive partition (every neuron in exactly one tile)")
    return own


# ---------------------------------------------------------------------------
# Predicted reads (what the kernels' counters report)
# ---------------------------------------------------------------------------


def predicted_reads(lb: LayerBlocks, worktiles: np.ndarray, topen: np.ndarray, *, rmap: np.ndarray | None,
                    per_row: bool, skip: bool = True, out_cols: int, down_cols: int = 32) -> dict:
    """The block reads one gate/up + down pass performs, mirroring the kernels' counters.

    ``worktiles`` int [U, 4] (row0, nrows, expert, _); ``topen`` uint8 [tokens or 1, tiles];
    ``rmap`` maps a work row to its token (None: identity); ``out_cols`` = d (down outputs).
    Returns chunk reads, gate/up rows read (per projection), down group reads (per column tile)."""
    topen = np.asarray(topen).astype(bool)
    chunk_reads = rows_read = group_reads = 0
    G = lb.groups
    tiles_of_down = -(-out_cols // down_cols)
    for row0, nrows, e, _ in np.asarray(worktiles):
        if nrows <= 0:
            continue
        toks = [(int(rmap[row0 + m]) if rmap is not None else row0 + m) if per_row else 0
                for m in range(nrows)]
        for off, ln, t in lb.chunks[lb.chunk_ptr[e]:lb.chunk_ptr[e + 1]]:
            if not skip or any(topen[k, t] for k in toks):
                chunk_reads += 1
                rows_read += int(ln)
        for g in range(G):
            sig = lb.gsig[lb.gsig_ptr[e * G + g]:lb.gsig_ptr[e * G + g + 1]]
            if not skip or any(topen[k, t] for k in toks for t in sig):
                group_reads += tiles_of_down
    return {"chunks": chunk_reads, "gate_up_rows": rows_read, "down_group_tiles": group_reads}


def bytes_from_reads(reads: dict, *, d: int, bits: int, group_size: int, scale_bytes: int = 2,
                     down_bits: int | None = None, down_group_size: int | None = None,
                     down_cols: int = 32) -> dict:
    """Weight bytes for ``predicted_reads`` / kernel counters: gate+up rows and down group tiles."""
    down_bits = bits if down_bits is None else down_bits
    down_group_size = group_size if down_group_size is None else down_group_size
    row = d * bits // 8 + (d // group_size) * 2 * scale_bytes
    gu = 2 * reads["gate_up_rows"] * row
    dn = reads["down_group_tiles"] * down_cols * (down_group_size * down_bits // 8 + 2 * scale_bytes)
    return {"gate_up_bytes": gu, "down_bytes": dn, "total_bytes": gu + dn}


# ---------------------------------------------------------------------------
# On-disk block tables (manifest + npz)
# ---------------------------------------------------------------------------


def save_tables(out_dir, layers: dict, graph: TileGraph, *, source: dict) -> dict:
    """Write ``blocks.npz`` (every layer's tables) and ``manifest.json`` (schema, packing law,
    graph, per-layer meta with digests, the source checkpoint and partition)."""
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for layer, lb in layers.items():
        arrays.update(lb.arrays(prefix=f"L{int(layer)}."))
    arrays["tile_box_bits"] = graph.tile_box_bits()
    np.savez(out / "blocks.npz", **arrays)
    manifest = {"schema": SCHEMA, "packing_law": PACKING_LAW, "graph": graph.to_json(),
                "graph_digest": graph.digest(), "source": source,
                "layers": {str(int(l)): lb.meta() for l, lb in layers.items()}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def load_tables(out_dir):
    """Read :func:`save_tables` output, refusing a schema, law or digest mismatch.
    Returns ``(layers, graph, manifest)``."""
    from pathlib import Path
    out = Path(out_dir)
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA or manifest.get("packing_law") != PACKING_LAW:
        raise ValueError("not an amoeba-blocks v1 table set")
    graph = TileGraph.from_json(manifest["graph"])
    if graph.digest() != manifest["graph_digest"]:
        raise ValueError("tile graph digest does not match its manifest")
    z = np.load(out / "blocks.npz")
    layers = {int(l): LayerBlocks.from_arrays(meta, z, prefix=f"L{int(l)}.")
              for l, meta in manifest["layers"].items()}
    return layers, graph, manifest
