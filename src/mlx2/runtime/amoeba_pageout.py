"""aMoEba box page-out: a gated-off box's expert neurons are not resident at all (experimental, unqualified).

The zero mask (``amoeba_neuron_mask``) keeps the closed neurons' bytes resident as exact zero rows; it is the oracle
and the fallback. Page-out removes them from memory. A routed expert bank is a SwitchGLU: gate/up ``[E, F, D]``
(quantized groups along D, so a neuron's row is self-contained) and down ``[E, D, F]`` (quantized groups along F,
the neuron axis). Dropping single neurons from down would split its groups, so page-out works on BLOCKS of
``block`` neurons per expert (64 by default; a multiple of every down group size and of the code packing):

* a block whose neurons are ALL closed is paged out: its gate/up rows and down columns leave memory;
* a block with any open neuron stays resident, its closed neurons still zero rows (the zero law);
* the resident blocks of every expert go into one compact bank ``[NB + 1, block, ...]`` (the last block all zeros),
  copied byte for byte from the stock bank (same codes, scales and biases: no requantization), with a per-expert
  table ``[E, Kmax]`` of resident block ids padded with the zero block (``Kmax`` = most resident blocks of any
  expert in the layer);
* a routed slot (token, expert e) runs its ``Kmax`` blocks in ``Kmax`` passes, pass j taking every slot's j-th
  resident block (``h = act(up_b x, gate_b x)``, the stock bank's own rows, then ``down_b(h)``), each pass shaped
  like one stock call (k assignments per token), its float32 partial added to a ``[..., k, D]`` accumulator that is
  cast once to the activation dtype. The router, its weights and the shared expert are untouched.

Exactness: every resident neuron's activation equals the zero-mask model's bit for bit (row slices of the same
quantized rows through the same kernels), and a paged neuron contributes exactly what the zero mask gives it (0).
The slot row is the same sum of the same products, but the float32 reduction is split into per-block partials
instead of one dot over F, so the output equals the zero-mask model to float32 reassociation, NOT bit for bit:
law ``block-pageout-sequential-fp32-v2``, recorded in the receipt and in the APCv2 namespace. CPU tests bound it.

Memory: only whole 64-neuron blocks are freed, so a box carved in single-neuron units frees only its fully closed
blocks; a box carved block-aligned frees all of it (see the plan). The freed bytes are measured from the arrays, per
layer, in the receipt. Working memory per call: one stock-shaped pass (input sorted copy, gate/up rows of one block,
its float32 partial) plus the float32 ``[..., k, D]`` accumulator; prefill chunks evaluate pass by pass. ``Kmax``
padding costs compute (``Kmax`` passes of 64-row blocks, the same FLOPs as the stock bank at Kmax = F/64) and the
sorted input copy once per pass, transiently.

Gate-open loading (designed, not built): the bundle names every paged (expert, block), and their bytes stay in the
artifact's own shards (nothing is rewritten), so a loader can read them back by byte range, append them to the
compact bank and update the table; the table is the only routing state.
"""
from __future__ import annotations

import copy

import mlx.core as mx
import numpy as np

from .models.switch_layers import SwitchGLU, _gather_sort, _scatter_unsort, _switch_sort_decision

LAW = "block-pageout-sequential-fp32-v2"     # v2: block passes summed in order (v1: one fan-out sum)
DEFAULT_BLOCK = 64
# Above this many routed assignments in one call (prefill chunks), each block pass is evaluated before the next is
# built, so at most one pass's intermediates are alive; decode and verify calls stay one lazy graph.
PASS_EVAL_MIN_ASSIGNMENTS = 4096


def blocks(closed_ids, E, F, block):
    """(paged bool [E, nb], resident (expert, block) pairs expert-major, table int32 [E, Kmax], Kmax)."""
    nb = F // block
    closed = np.zeros(E * F, bool)
    closed[np.asarray(closed_ids, np.int64)] = True
    paged = closed.reshape(E, nb, block).all(-1)
    res = [(e, b) for e in range(E) for b in range(nb) if not paged[e, b]]
    per = [[] for _ in range(E)]
    for i, (e, _) in enumerate(res):
        per[e].append(i)
    kmax = max(1, max(len(p) for p in per))
    table = np.full((E, kmax), len(res), np.int32)               # pad: the all-zero block
    for e, p in enumerate(per):
        table[e, :len(p)] = p
    return paged, res, table, kmax


def nbytes(module) -> int:
    from mlx.utils import tree_flatten
    return int(sum(v.nbytes for _, v in tree_flatten(module.parameters())))


def _compact(lin, res, block, rows, F):
    """A copy of switch projection ``lin`` holding only the resident blocks plus one zero block. rows=True: gate/up
    (a block is ``block`` output rows); False: down (a block is ``block`` input columns, its own quant groups)."""
    e_idx = mx.array([e for e, _ in res] or [0], dtype=mx.int32)
    b_idx = mx.array([b for _, b in res] or [0], dtype=mx.int32)
    out = copy.copy(lin)
    for name in ("weight", "scales", "biases", "bias"):
        a = lin.get(name)
        if a is None:
            continue
        if name == "bias":
            raise ValueError("amoeba_mask pageout: a biased switch projection is refused (a paged block would still "
                             "write its bias)")
        E = a.shape[0]
        if rows:
            if a.shape[1] != F:
                raise ValueError(f"amoeba_mask pageout: {name} has {a.shape[1]} rows, F is {F}")
            v = a.reshape(E, F // block, block, *a.shape[2:])[e_idx, b_idx]
        else:
            X = a.shape[-1]
            if (X * block) % F:
                raise ValueError(f"amoeba_mask pageout: down {name} width {X} does not split into {block}-neuron "
                                 "blocks (block must be a multiple of the down group size and the code packing)")
            xb = X * block // F
            v = a.reshape(E, a.shape[1], F // block, xb).transpose(0, 2, 1, 3)[e_idx, b_idx]
        if not res:
            v = v[:0]
        out[name] = mx.concatenate([v, mx.zeros((1, *v.shape[1:]), dtype=v.dtype)], axis=0)
    return out


class PagedSwitchGLU(SwitchGLU):
    """A routed bank holding only its resident neuron blocks (module docstring); same call as SwitchGLU."""

    pageout_law = LAW

    def __call__(self, x, indices):
        """One pass per block slot j < Kmax: the routed slots' j-th resident block, each pass shaped exactly like the
        stock bank's call (k assignments per token, its own sort decision), its float32 partial added to the
        accumulator. Working memory is one stock-shaped pass plus the [..., k, D] float32 accumulator, whatever Kmax
        (a single k x Kmax fan-out would copy the input and the partials Kmax times; review 2026-11-21)."""
        from .models import invariant_prefill as _inv
        self._pageout_calls[0] += 1                                  # calls; passes are counted apart
        ids = self._table[indices]                                   # [..., k, Kmax]
        kmax = int(ids.shape[-1])
        xe0 = mx.expand_dims(x, (-2, -3))
        tl = int(indices.shape[-2]) if indices.ndim >= 2 else 1
        big = int(indices.size) >= PASS_EVAL_MIN_ASSIGNMENTS
        y = None
        for j in range(kmax):
            idj = ids[..., j]                                         # [..., k]: one block per routed slot
            do_sort, counter = _switch_sort_decision(self._sort_policy, assignments=int(idj.size), token_length=tl,
                                                     invariant=_inv.active())
            if j == 0:                                                # one sort decision counted per call
                self._sort_counts[counter] += 1
            self._pageout_passes[0] += 1
            xe, idx, inv = xe0, idj, None
            if do_sort:
                xe, idx, inv = _gather_sort(xe0, idj)
            h = self.activation(self.up_proj(xe, idx, sorted_indices=do_sort),
                                self.gate_proj(xe, idx, sorted_indices=do_sort))
            part = self.down_proj(h.astype(mx.float32), idx, sorted_indices=do_sort)     # float32 partial
            if do_sort:
                part = _scatter_unsort(part, inv, idj.shape)
            part = part.squeeze(-2)                                   # [..., k, D]
            y = part if y is None else y + part
            if big:                                                   # prefill: bound the live set to one pass
                try:
                    mx.eval(y)
                except ValueError:                                    # inside a graph transformation: keep it lazy
                    pass
        return y.astype(x.dtype)


def page_out_bank(sw, closed_ids, *, E, F, block=DEFAULT_BLOCK):
    """(paged bank, record, paged [E, nb]) for the stock bank ``sw`` (not modified)."""
    from .models import switch_grouped_experts
    if type(sw) is not SwitchGLU:
        raise ValueError(f"amoeba_mask pageout: bank {type(sw).__name__} is not a plain SwitchGLU (fused gate/up and "
                         "other banks keep the zero mask)")
    if switch_grouped_experts.MODE != "off":
        raise ValueError("amoeba_mask pageout needs MLX2_MOE_GROUPED_EXPERTS off (the grouped kernels read whole banks)")
    if block <= 0 or F % block:
        raise ValueError(f"amoeba_mask pageout: block {block} does not divide F {F}")
    paged, res, table, kmax = blocks(closed_ids, E, F, block)
    before = nbytes(sw)
    new = PagedSwitchGLU.__new__(PagedSwitchGLU)
    dict.update(new, sw)                                 # nn.Module is a dict: activation and the switch modules
    new.__dict__.update(sw.__dict__)                     # sort policy, sort counters
    new["gate_proj"] = _compact(sw.gate_proj, res, block, True, F)
    new["up_proj"] = _compact(sw.up_proj, res, block, True, F)
    new["down_proj"] = _compact(sw.down_proj, res, block, False, F)
    object.__setattr__(new, "_table", mx.array(table))
    object.__setattr__(new, "_pageout_calls", [0])
    object.__setattr__(new, "_pageout_passes", [0])
    mx.eval(new.parameters())
    after = nbytes(new) + int(table.nbytes)
    rec = {"experts": E, "F": F, "block": block, "blocks": int(E * (F // block)), "paged_blocks": int(paged.sum()),
           "resident_blocks": len(res), "kmax": int(kmax), "bytes_before": before, "bytes_after": after,
           "bytes_freed": before - after}
    return new, rec, paged


def page_out(bundle, banks, sites, block=DEFAULT_BLOCK):
    """Page out ``bundle``'s fully closed blocks in every declared MoE bank. ``banks``: {layer: (parent, attr)}; the
    zero mask must already have been applied (it zeroes the closed neurons of resident blocks). Dense sites keep the
    zero mask. Returns the record (per layer, resident bytes freed)."""
    by = {s.layer: s for s in sites}
    layers, freed, paged_units = [], 0, 0
    for layer, ids in sorted(bundle.closed.items()):
        s = by[layer]
        if s.experts is None or layer not in banks or not len(ids):
            continue
        parent, attr = banks[layer]
        new, rec, paged = page_out_bank(getattr(parent, attr), ids, E=s.experts, F=s.width, block=block)
        setattr(parent, attr, new)
        freed += rec["bytes_freed"]
        paged_units += int(paged.sum()) * block
        layers.append({"layer": layer, **rec})
    closed = int(sum(len(v) for v in bundle.closed.values()))
    return {"mode": "pageout", "law": LAW, "block": block, "layers": layers, "resident_bytes_freed": int(freed),
            "closed_units_paged": paged_units, "closed_units_zero_resident": closed - paged_units}

