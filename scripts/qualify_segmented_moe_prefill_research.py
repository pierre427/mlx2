#!/usr/bin/env python3
"""Native synthetic bit-identity gate for the segmented sorted MoE prefill research candidate.

Scope: synthetic affine 4-bit group-64 bfloat16 MoE primitives of the FROZEN
candidate ``mlx2.adapters.segmented_moe_prefill_research`` (sha256 pinned in
FROZEN) at its two named geometries only: the row-mapped fused gate/up +
SwiGLU kernel and the segmented sorted down kernel. Not model acceptance,
recurrent or cache state, batched rollback, performance, generic shapes or
production qualification. A pass is an experimental numerical gate:
``qualified``/``selected``/``observed_used``/``model_gain`` are always false.

Ordinary reference (never an approximation): the served local layer code on
the SAME sorted assignments the candidate receives. ``_gather_sort`` of
``switch_layers`` sorts the router ids (``mx.argsort``, its own tie order, its
own > 32768-row seam pad); the candidate routes are built from exactly that
order, so both arms see one ordering. Then ``QuantizedSwitchLinear.__call__``
(``mx.gather_qmm`` sorted, transpose, affine 4-bit g64, bf16 tables) for the
fused ``[gate | up]`` table, the served ``SwiGLU`` module (compiled
``activations.swiglu``) on ``gate_up[..., :I]`` / ``gate_up[..., I:]`` as
``FusedGateUpSwitchGLU`` splits them, the down ``QuantizedSwitchLinear`` with
``sorted_indices=True`` and ``_scatter_unsort`` restoration. Quantized
arithmetic is never changed to make arms agree.

Gate: bf16 outputs are compared as raw uint16 bits (no float conversion):
zero bit mismatches, no non-finite bit patterns in either arm, exact shapes,
route/order/row-map/inverse restoration checks, a non-vacuous reference, and
per candidate call exactly one FRESH successful lazy chain whose output was
then evaluated and checked. Both candidate counters count successful lazy
chains, not GPU executions; only their pairing with evaluated, checked
outputs is evidence. A build that changes summation or SiLU order and fails
is an honest negative result; there is no tolerance. A separate dispatch of
the candidate's own tile scan on the same sorted ids checks expert/tile
coverage exactly once (a probe, not the scan instance inside the chain).

Evidence vs execution: ``evaluate()`` judges supplied evidence (dicts, JSON,
fake backends) and can NEVER establish native execution. Only the live
in-process run object built by ``_run_native`` (source/build admission, the
real ``NativeBackend``, nonzero fresh native chain counters matching the
evaluated stages) can stamp ``native_synthetic_gate``; a JSON round trip drops
it. A trust boundary against forged reports and fake backends, not a security
claim against arbitrary code.

Identity association: the native backend class, the live-run type, the
witness and the evaluator are captured ONCE, at import, in the closure that
defines ``_run_native`` and ``native_verdict`` (not in default arguments a
caller could override): ``native_verdict(live)`` and
``_run_native(admission, cells, full_requested)`` accept nothing else.
Rebinding the module names ``NativeBackend``, ``_LiveNativeRun``,
``_WITNESS`` or ``evaluate`` afterwards, a subclass or a look-alike cannot make
a CPU fake count as native, and ``_run_native`` still constructs the original
class. Evidence-association hygiene only: it does not resist code that edits
closures, class attributes or private objects, and an instance of the
original class is not by itself proof of native execution.

No timing: ``--timing`` is refused before any admission, import or backend.

Nothing imports MLX at module import, ``--help``, ``--catalogue``, admission
or refusal. ``--run-native`` requires ``--i-own-the-gpu`` (an acknowledgement,
NOT ownership: the parent wrapper /tmp/mlx2-intake/stage3_gpu.py owns the CPG
lease and both flocks), the absolute ``--source-root`` of this checkout and a
full ``--source-commit`` equal to MLX2_INTAKE_SOURCE_COMMIT and git HEAD.
Before any MLX import: clean tracked+untracked (and no non-bytecode ignored)
src/scripts/tests/provenance, every tracked file there byte-equal to its HEAD
blob, the frozen candidate/test/provenance hashes, the reference witnesses,
``mlx2`` resolving to this checkout's src, and the installed MLX extension,
libmlx, metallib, Python files and the kernel headers the candidate flattens
(transitive, NAX and activation functors included) hashed. After importing
MLX and before any device query the imported extension and version must match.
Before EVERY dispatch all of that is re-checked; a change stops the run and
releases its arrays. After the run HEAD, status, hashes and module paths are
re-checked. A receipt is written only with ``--out`` (exclusive creation)
after a completed live run.

  # from this checkout under the parent wrapper (it owns the lease and flocks):
  MLX2_INTAKE_SOURCE_ROOT=<this checkout> MLX2_INTAKE_SOURCE_COMMIT=<full HEAD sha> \\
  MLX2_INTAKE_CPG_WORKFLOW=<live workflow> MLX2_INTAKE_CPG_TASK=<native task> \\
  python /tmp/mlx2-intake/stage3_gpu.py segmented-moe-native \\
      ~/Desktop/mlx2/.venv/bin/python scripts/qualify_segmented_moe_prefill_research.py \\
      --run-native --i-own-the-gpu --source-root <this checkout> --source-commit <full HEAD sha> \\
      --out <new receipt>.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

SCHEMA = "mlx2.native.segmented-moe-prefill-synthetic-gate.v1"
SCOPE = ("synthetic affine 4-bit g64 bfloat16 MoE primitives of the frozen segmented-MoE research candidate at "
         "its two named geometries (default35b E256/k8/D2048/I512, flash_next E512/k10/D2560/I640), bit-compared "
         "with the ordinary sorted gather_qmm + served SwiGLU path; not model acceptance, state, batched rollback, "
         "performance, generic shapes or production qualification")
CANDIDATE_NAME = "mlx2.adapters.segmented_moe_prefill_research"
CANDIDATE_FILE = "src/mlx2/adapters/segmented_moe_prefill_research.py"
FROZEN = {
    CANDIDATE_FILE: "cf115b607513a85c4d7ba0d7af907605be812b7b6a552521197ff8db5c404dcc",
    "tests/test_segmented_moe_prefill_research.py": "202ce9404ca6634bbdb24f1d68b09257d4343d5ebd3e4c3d50cf5d20226c2da9",
    "provenance/segmented-moe-prefill-research.json":
        "008dd14b52973a8c8475edd175874c8eb019f8d544322d53baa5cdf921f75882",
}
HARNESS_FILES = (
    "scripts/qualify_segmented_moe_prefill_research.py",
    "tests/test_segmented_moe_prefill_native_qualifier.py",
    "provenance/segmented-moe-prefill-native-qualifier.json",
    "tests/test_segmented_moe_native_identity_cpu.py",
)
REFERENCE_FILES = (
    "src/mlx2/runtime/models/switch_layers.py",
    "src/mlx2/runtime/models/activations.py",
    "src/mlx2/runtime/models/qwen3_next.py",
)
PACKAGE_FILES = (
    "src/mlx2/__init__.py",
    "src/mlx2/adapters/__init__.py",
    "src/mlx2/runtime/__init__.py",
    "src/mlx2/runtime/models/__init__.py",
)
BOUND_FILES = tuple(FROZEN) + HARNESS_FILES + REFERENCE_FILES + PACKAGE_FILES
TRACKED_PATHS = ("src", "scripts", "tests", "provenance")
# Exact (stripped) source lines the reference path depends on; a refactor that drops one refuses.
REFERENCE_WITNESSES = {
    "src/mlx2/runtime/models/activations.py": (
        "@partial(mx.compile, shapeless=True)",
        "def swiglu(gate, x):",
        "return nn.silu(gate) * x",
    ),
    "src/mlx2/runtime/models/switch_layers.py": (
        "from .activations import swiglu",
        "_SORTED_GATHER_TAIL_BUG = True",
        "order = mx.argsort(indices)",
        "inv_order = mx.argsort(order)",
        "x = x.flatten(0, -3)[order // M]",
        "if _SORTED_GATHER_TAIL_BUG and n > 32768 and (n % 64 != 0):",
        "x = mx.gather_qmm(",
        "transpose=True,",
        "sorted_indices=False if tail_policy == \"unsorted\" else sorted_indices,",
        "return swiglu(gate, x)",
        "x = x[inv_order]",
    ),
    "src/mlx2/runtime/models/qwen3_next.py": (
        "(x, idx, inv_order) = _gather_sort(x, indices)",
        "gate_up = self.gate_up_proj(x, idx, sorted_indices=do_sort)",
        "hidden = self.activation(gate_up[..., half:], gate_up[..., :half])",
        "x = self.down_proj(hidden, idx, sorted_indices=do_sort)",
    ),
}

# Kernel headers the candidate flattens from the installed package (its _MLX_* tuples, re-checked
# after import) and the utils preamble mx.fast.metal_kernel prepends (skipped by the flattener).
_KDIR = "mlx/backend/metal/kernels/"
MLX_PREAMBLE_HEADERS = tuple(_KDIR + h for h in ("utils.h", "bf16.h", "bf16_math.h", "complex.h", "defines.h",
                                                 "logging.h"))
MLX_DOWN_HEADERS = (_KDIR + "steel/gemm/nax.h",)
MLX_GATE_UP_HEADERS = MLX_DOWN_HEADERS + (_KDIR + "unary_ops.h", _KDIR + "binary_ops.h")

GEOMETRIES = {
    "default35b": {"experts": 256, "top_k": 8, "hidden": 2048, "intermediate": 512},
    "flash_next": {"experts": 512, "top_k": 10, "hidden": 2560, "intermediate": 640},
}
# The local _gather_sort seam rule, restated independently (cross-checked with the candidate's seam_pad).
SEAM_THRESHOLD, SEAM_MULTIPLE = 32768, 64
# Pinned plans of the sweep cells: (sched, bm, bk, gx, pad); sched 0 = seg, 1 = db.
PLAN_SWEEP = ((0, 64, 64, 0, 0), (0, 96, 128, 32, 0), (0, 128, 128, 32, 8192),
              (1, 64, 64, 0, 0), (1, 64, 64, 32, 0), (1, 96, 64, 32, 0))
# Synthetic tables: uniform 4-bit codes, bf16 scales in [lo, hi), biases = -scale * U[7, 8).
TABLE_LAW = {"scale": (0.006, 0.014), "bias_over_scale": (7.0, 8.0)}
MIN_NONZERO_FRACTION = 0.5
KINDS = ("gate_up_mapped_swiglu", "down_segmented")
VERDICT_KEYS = {"qualified", "selected", "observed_used", "model_gain", "native_synthetic_gate", "verdict",
                "native_execution"}
ROUTE_CHECKS = ("inv_order_permutation", "sorted_matches_router", "nondecreasing", "pad_rows_repeat_last",
                "equals_host_sorted", "x_sorted_is_row_map_copy")
SHA = re.compile(r"[0-9a-f]{64}")


class Refused(RuntimeError):
    """Fail closed: never a pass."""


# ================================================================ catalogue (pure host law)

def _cell(cid, geometry, tokens, law, *, expect="pass", sweep=False, note=""):
    g = GEOMETRIES[geometry]
    n = tokens * g["top_k"]
    pad = SEAM_MULTIPLE - n % SEAM_MULTIPLE if n > SEAM_THRESHOLD and n % SEAM_MULTIPLE else 0
    return {"id": cid, "geometry": geometry, **g, "tokens": tokens, "law": law, "expect": expect,
            "sweep": sweep, "assignments": n, "pad": pad, "rows": n + pad, "note": note}


CATALOGUE = (
    _cell("d35-t128-uniform", "default35b", 128, "uniform", note="row-block admission boundary (rows 1024 = 4 * E)"),
    _cell("d35-t128-subset", "default35b", 128, "subset", note="boundary; only experts e % 8 == 3 used, 224 empty"),
    _cell("d35-t205-uniform", "default35b", 205, "uniform", note="ragged runs and tails"),
    _cell("d35-t300-hot-edges", "default35b", 300, "hot_edges", note="experts 0 and E-1 own full 300-row runs"),
    _cell("d35-t512-zipf", "default35b", 512, "zipf", note="strongly skewed runs, empty experts"),
    _cell("d35-t640-plan-sweep", "default35b", 640, "uniform", sweep=True, note="planner + six pinned plans"),
    _cell("d35-t4097-seam", "default35b", 4097, "uniform", note="32776 rows > 32768: local seam pad 56"),
    _cell("d35-t1-decode", "default35b", 1, "uniform", expect="refuse", note="decode stays ordinary"),
    _cell("d35-t4-mtp-verify", "default35b", 4, "uniform", expect="refuse", note="MTP verify stays ordinary"),
    _cell("d35-t127-below", "default35b", 127, "uniform", expect="refuse", note="one token below admission"),
    _cell("fn-t205-uniform", "flash_next", 205, "uniform", note="row-block admission boundary (rows 2050)"),
    _cell("fn-t205-subset", "flash_next", 205, "subset", note="boundary; only experts e % 8 == 3 used, 448 empty"),
    _cell("fn-t333-uniform", "flash_next", 333, "uniform", note="ragged runs and tails"),
    _cell("fn-t300-hot-edges", "flash_next", 300, "hot_edges", note="experts 0 and E-1 own full 300-row runs"),
    _cell("fn-t410-zipf", "flash_next", 410, "zipf", note="strongly skewed runs, empty experts"),
    _cell("fn-t615-plan-sweep", "flash_next", 615, "uniform", sweep=True, note="planner + six pinned plans"),
    _cell("fn-t3277-seam", "flash_next", 3277, "uniform", note="32770 rows > 32768: local seam pad 62"),
    _cell("fn-t1-decode", "flash_next", 1, "uniform", expect="refuse", note="decode stays ordinary"),
    _cell("fn-t4-mtp-verify", "flash_next", 4, "uniform", expect="refuse", note="MTP verify stays ordinary"),
    _cell("fn-t128-below", "flash_next", 128, "uniform", expect="refuse", note="default35b boundary is below here"),
    _cell("fn-t204-below", "flash_next", 204, "uniform", expect="refuse", note="one token below admission"),
)
MANDATORY = tuple(c["id"] for c in CATALOGUE)


def catalogue_by_id():
    ids = [c["id"] for c in CATALOGUE]
    if len(set(ids)) != len(ids):
        raise Refused("duplicate catalogue identity")
    return {c["id"]: c for c in CATALOGUE}


def select_cells(ids=None):
    """Default: the full mandatory catalogue. Explicit ids are a partial run (never a full gate)."""
    table = catalogue_by_id()
    if ids is None:
        return [table[i] for i in MANDATORY], True
    chosen = list(ids)
    unknown = [i for i in chosen if i not in table]
    if unknown or len(set(chosen)) != len(chosen) or not chosen:
        raise Refused(f"unknown, duplicate or empty cell ids {unknown or chosen}")
    order = {cid: i for i, cid in enumerate(MANDATORY)}
    return [table[i] for i in sorted(chosen, key=order.get)], False


def _seed(label):
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")


def f32_to_bf16_bits(values):
    """Round-to-nearest-even float32 -> bfloat16 bit patterns (finite inputs only)."""
    import numpy as np

    a = np.ascontiguousarray(values, dtype=np.float32)
    if not np.isfinite(a).all():
        raise Refused("non-finite synthetic value")
    u = a.view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_bits_to_f32(bits):
    import numpy as np

    return (np.ascontiguousarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def nonfinite_count(bits):
    import numpy as np

    return sum(int(np.count_nonzero((c & 0x7F80) == 0x7F80)) for c in _chunks(bits))


def _chunks(a, size=1 << 22):
    """Flat views of at most ``size`` elements: bounded temporaries on the large seam outputs."""
    flat = a.reshape(-1)
    return (flat[i:i + size] for i in range(0, flat.size, size))


def sha_array(a):
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def router_ids(cell):
    """[T, k] uint32 router ids: k DISTINCT experts per token (Gumbel top-k, i.e. sampling without
    replacement), token-major as a router emits them."""
    import numpy as np

    T, E, k = cell["tokens"], cell["experts"], cell["top_k"]
    rng = np.random.default_rng(_seed("router/" + cell["id"]))
    if cell["law"] == "hot_edges":
        rest = np.argsort(-rng.gumbel(size=(T, E - 2)), axis=1, kind="stable")[:, : k - 2] + 1
        ids = np.concatenate([np.zeros((T, 1), np.int64), np.full((T, 1), E - 1, np.int64), rest], axis=1)
    else:
        logp = np.zeros(E)
        if cell["law"] == "subset":
            logp = np.where(np.arange(E) % 8 == 3, 0.0, -np.inf)
        elif cell["law"] == "zipf":
            logp[rng.permutation(E)] = -1.5 * np.log(np.arange(1, E + 1))
        elif cell["law"] != "uniform":
            raise Refused(f"unknown router law {cell['law']}")
        ids = np.argsort(-(logp[None, :] + rng.gumbel(size=(T, E))), axis=1, kind="stable")[:, :k]
    ids = ids.astype(np.uint32)
    if ids.shape != (T, k) or any(len(set(row.tolist())) != k for row in ids) or int(ids.max()) >= E:
        raise Refused(f"{cell['id']}: router law did not give k distinct in-range experts per token")
    return ids


def token_bits(cell):
    import numpy as np

    rng = np.random.default_rng(_seed("x/" + cell["id"]))
    return f32_to_bf16_bits(rng.standard_normal((cell["tokens"], cell["hidden"]), dtype=np.float32))


def cell_law(cell):
    """Seed-bound inputs and the order-independent sorted law (stable host sort + seam pad)."""
    import numpy as np

    ids, x_bits = router_ids(cell), token_bits(cell)
    flat = ids.reshape(-1).astype(np.int64)
    order = np.argsort(flat, kind="stable")
    srt = flat[order]
    padded = np.concatenate([srt, np.repeat(srt[-1:], cell["pad"])])
    return {"ids": ids, "x_bits": x_bits, "flat": flat, "order": order, "sorted": padded,
            "hashes": {"router_ids": sha_array(ids), "x_bits": sha_array(x_bits)}}


def host_tables(geometry, dims=None):
    """Seed-bound affine q4 g64 tables generated directly in packed form (no float weight tensor):
    {projection: {weight uint32 [E, N, K/8], scales/biases bf16 bits [E, N, K/64]}}."""
    import numpy as np

    g = GEOMETRIES[geometry] if dims is None else dims
    E, D, I = g["experts"], g["hidden"], g["intermediate"]
    rng = np.random.default_rng(_seed("tables/" + geometry))
    out = {}
    for proj, (N, K) in (("gate_up", (2 * I, D)), ("down", (D, I))):
        words = rng.integers(0, 2**32, size=(E, N, K // 8), dtype=np.uint32)
        scales = f32_to_bf16_bits(rng.uniform(*TABLE_LAW["scale"], size=(E, N, K // 64)))
        ratio = rng.uniform(*TABLE_LAW["bias_over_scale"], size=scales.shape)
        biases = f32_to_bf16_bits(-bf16_bits_to_f32(scales) * ratio)
        out[proj] = {"weight": words, "scales": scales, "biases": biases}
    return out


def table_hashes(tables):
    return {proj: {name: {"shape": list(a.shape), "dtype": str(a.dtype), "sha256": sha_array(a)}
                   for name, a in parts.items()} for proj, parts in tables.items()}


def tile_law(sorted_ids, experts, bm):
    """Independent tile law: per expert ascending, its single run cut into <= bm-row tiles."""
    import numpy as np

    sid = np.asarray(sorted_ids, dtype=np.int64)
    tiles = []
    for e in range(experts):
        rows = np.flatnonzero(sid == e)
        if rows.size:
            start, end = int(rows[0]), int(rows[-1]) + 1
            tiles += [(r, e, min(bm, end - r), 0) for r in range(start, end, bm)]
    return np.array(tiles, dtype=np.uint32).reshape(-1, 4)


def variants(cell):
    return ((None,) + PLAN_SWEEP) if cell["sweep"] else (None,)


def variant_label(v):
    return "planner" if v is None else "pinned-" + "-".join(str(x) for x in v)


def plan_record(pp):
    return {"planned": [int(x) for x in pp.planned], "effective": [int(x) for x in pp.plan],
            "describe": pp.plan.describe(), "max_tiles": int(pp.max_tiles), "grid": [int(x) for x in pp.grid],
            "threadgroup": [int(x) for x in pp.threadgroup]}


def expectations(cell, cand, law=None):
    """Stage -> expected kind/shape/plan/tile law, from the catalogue law and the candidate's pure planner."""
    T, k, E, D, I, M = (cell[x] for x in ("tokens", "top_k", "experts", "hidden", "intermediate", "rows"))
    law = cell_law(cell) if law is None else law
    stages = {"routes": {"kind": "routes", "rows": M, "pad": cell["pad"],
                         "sorted_sha256": sha_array(law["sorted"].astype("uint32"))}}
    scans = {}
    for v in variants(cell):
        pinned = None if v is None else cand.Plan(*v)
        gu = cand.projection_plan(M, E, D, 2 * I, paired=True, pinned=pinned)
        dn = cand.projection_plan(M, E, I, D, paired=False, pinned=pinned)
        lab = variant_label(v)
        stages[f"gate_up:{lab}"] = {"kind": KINDS[0], "shape": [M, 1, I], "plan": plan_record(gu)}
        stages[f"down_isolated:{lab}"] = {"kind": KINDS[1], "shape": [M, 1, D], "plan": plan_record(dn)}
        if v is None:
            stages["chain_restored"] = {"kind": KINDS[1], "shape": [T, k, 1, D], "plan": plan_record(dn)}
        for pp in (gu, dn):
            scans.setdefault(pp.plan.bm, pp.max_tiles)
    sorted_list = [int(e) for e in law["sorted"]]
    for bm, max_tiles in sorted(scans.items()):
        tiles = tile_law(law["sorted"], E, bm)
        mirror = cand.tile_table_host(sorted_list, experts=E, bm=bm)
        stages[f"scan:bm{bm}"] = {"kind": "scan", "bm": bm, "max_tiles": max_tiles, "count": int(len(tiles)),
                                  "tiles_sha256": sha_array(tiles),
                                  "candidate_mirror_agrees": [tuple(t[:3]) for t in tiles.tolist()]
                                  == [tuple(t) for t in mirror]}
    return stages


# ================================================================ stage summaries (pure; arrays dropped)

def _bit_array(a, shape):
    import numpy as np

    if not isinstance(a, np.ndarray) or a.dtype != np.uint16:
        return "not a bfloat16 bit (uint16) array"
    if list(a.shape) != list(shape):
        return f"shape {list(a.shape)} != {list(shape)}"
    return None


def summarize_bits(shape, candidate, reference):
    """Exact bit comparison of two bf16 bit arrays; nothing is converted to float."""
    import numpy as np

    out = {"shape": list(shape), "problems": []}
    for name, a in (("candidate", candidate), ("reference", reference)):
        bad = _bit_array(a, shape)
        if bad:
            out["problems"].append(f"{name}: {bad}")
    if out["problems"]:
        return out
    mism, first, max_delta, offset = 0, None, 0, 0
    for c, r in zip(_chunks(candidate), _chunks(reference), strict=True):
        diff = c != r
        n = int(np.count_nonzero(diff))
        if n:
            if first is None:
                first = [int(i) for i in np.unravel_index(offset + int(np.argmax(diff)), candidate.shape)]
            max_delta = max(max_delta, int(np.abs(c[diff].astype(np.int32) - r[diff].astype(np.int32)).max()))
        mism, offset = mism + n, offset + c.size
    nonzero = sum(int(np.count_nonzero(c & 0x7FFF)) for c in _chunks(reference))
    out.update({"candidate_sha256": sha_array(candidate), "reference_sha256": sha_array(reference),
                "bit_mismatches": mism, "first_mismatch": first, "max_bit_delta": max_delta,
                "candidate_nonfinite": nonfinite_count(candidate), "reference_nonfinite": nonfinite_count(reference),
                "reference_nonzero_fraction": float(nonzero / max(reference.size, 1))})
    return out


def summarize_routes(cell, law, payload):
    """Ordinary _gather_sort outputs vs the router law. Returns (summary, context for later stages)."""
    import numpy as np

    n, k, M, D = cell["assignments"], cell["top_k"], cell["rows"], cell["hidden"]
    sid, inv, xs = payload.get("sorted_ids"), payload.get("inv_order"), payload.get("x_sorted_bits")
    checks = dict.fromkeys(ROUTE_CHECKS, False)
    summary = {"rows": None, "pad": None, "checks": checks, "problems": []}
    if not (isinstance(sid, np.ndarray) and sid.dtype == np.uint32 and sid.ndim == 1):
        summary["problems"].append("sorted_ids is not a uint32 vector")
        return summary, None
    if not (isinstance(inv, np.ndarray) and inv.dtype == np.uint32 and inv.shape == (n,)):
        summary["problems"].append("inv_order is not a uint32 vector of the assignments")
        return summary, None
    summary.update(rows=int(sid.size), pad=int(sid.size) - n, sorted_sha256=sha_array(sid),
                   inv_order_sha256=sha_array(inv))
    s = sid.astype(np.int64)
    checks["inv_order_permutation"] = bool(np.array_equal(np.sort(inv.astype(np.int64)), np.arange(n)))
    if not checks["inv_order_permutation"] or sid.size < n:
        return summary, None
    order = np.empty(n, dtype=np.int64)
    order[inv.astype(np.int64)] = np.arange(n)
    checks["sorted_matches_router"] = bool(np.array_equal(s[:n], law["flat"][order]))
    checks["nondecreasing"] = bool(np.all(np.diff(s) >= 0))
    checks["pad_rows_repeat_last"] = bool(np.all(s[n:] == s[n - 1]))
    checks["equals_host_sorted"] = bool(sid.size == M and np.array_equal(s, law["sorted"]))
    row_map = np.concatenate([order // k, np.repeat(order[n - 1:] // k, sid.size - n)])
    if isinstance(xs, np.ndarray) and xs.dtype == np.uint16 and xs.shape == (sid.size, 1, D):
        checks["x_sorted_is_row_map_copy"] = bool(np.array_equal(xs.reshape(sid.size, D), law["x_bits"][row_map]))
    summary["tie_order_equals_stable_host"] = bool(np.array_equal(order, law["order"]))
    return summary, {"order": order, "inv": inv.astype(np.int64), "sorted": s}


def summarize_scan(expected, ctx, payload):
    import numpy as np

    tiles, count = payload.get("tiles"), payload.get("count")
    checks = {"within_capacity": False, "equals_tile_law": False, "rows_covered_once": False,
              "single_expert_tiles": False, "tile_rows_bounded": False}
    out = {"bm": payload.get("bm"), "max_tiles": payload.get("max_tiles"), "count": count, "checks": checks,
           "problems": []}
    if not (isinstance(tiles, np.ndarray) and tiles.dtype == np.uint32 and tiles.ndim == 2 and tiles.shape[1] == 4
            and type(count) is int and tiles.shape[0] == count):
        out["problems"].append("scan tiles are not a [count, 4] uint32 table with an int count")
        return out
    out["tiles_sha256"] = sha_array(tiles)
    bm, sid = expected["bm"], ctx["sorted"]
    checks["within_capacity"] = 0 < count <= expected["max_tiles"]
    checks["equals_tile_law"] = bool(np.array_equal(tiles, tile_law(sid, int(sid.max()) + 1, bm)))
    cover = np.zeros(sid.size, dtype=np.int64)
    single, bounded = True, True
    for r, e, c, z in tiles.astype(np.int64).tolist():
        bounded &= 0 < c <= bm and z == 0 and r + c <= sid.size
        if bounded:
            cover[r:r + c] += 1
            single &= bool(np.all(sid[r:r + c] == e))
    checks["tile_rows_bounded"] = bool(bounded)
    checks["single_expert_tiles"] = bool(single and bounded)
    checks["rows_covered_once"] = bool(bounded and np.all(cover == 1))
    return out


class _GuardCounter:
    def __init__(self, guard):
        self.guard, self.calls = guard, 0

    def __call__(self):
        self.calls += 1
        if self.guard is not None:
            self.guard()


class _CellRecorder:
    """``emit(stage, payload)`` sink: the harness, not the backend, summarizes every stage and drops
    its arrays. Backends never produce verdict fields."""

    def __init__(self, cell, law, expected, counter):
        self.cell, self.law, self.expected, self.counter = cell, law, expected, counter
        self.stages, self.ctx, self.ref_restored_sha = {}, None, None
        self._mark = counter.calls

    def __call__(self, stage, payload):
        guards, self._mark = self.counter.calls - self._mark, self.counter.calls
        if stage in self.stages or stage not in self.expected:
            raise Refused(f"{self.cell['id']}: duplicate or unexpected stage {stage}")
        if not isinstance(payload, dict):
            raise Refused(f"{self.cell['id']}: stage {stage} payload is not a dict")
        exp = self.expected[stage]
        if stage == "routes":
            summary, self.ctx = summarize_routes(self.cell, self.law, payload)
            if self.ctx is None:                           # no usable order: stop (fail closed), never guess
                raise Refused(f"{self.cell['id']}: ordinary routes unusable: {summary['problems'] or summary['checks']}")
        elif self.ctx is None:
            raise Refused(f"{self.cell['id']}: stage {stage} before valid routes")
        elif exp["kind"] == "scan":
            summary = summarize_scan(exp, self.ctx, payload)
        else:
            summary = self._projection(stage, exp, payload)
        summary["dispatches"] = payload.get("dispatches")
        summary["guard_calls"] = guards
        self.stages[stage] = summary

    def _projection(self, stage, exp, payload):
        import numpy as np

        cand, ref = payload.get("candidate"), payload.get("reference")
        summary = summarize_bits(exp["shape"], cand, ref)
        summary.update({"kind": payload.get("kind"), "plan": payload.get("plan"), "native": payload.get("native"),
                        "engagement": dict(payload.get("engagement") or {})})
        n, M, D = self.cell["assignments"], self.cell["rows"], self.cell["hidden"]
        restored_shape = [self.cell["tokens"], self.cell["top_k"], 1, D]
        if stage == "down_isolated:planner" and _bit_array(ref, exp["shape"]) is None:
            self.ref_restored_sha = sha_array(ref.reshape(M, D)[self.ctx["inv"]].reshape(restored_shape))
        if stage == "chain_restored":
            srt = payload.get("candidate_sorted")
            checks = {"candidate_restore_is_inverse": False, "reference_restore_matches_host_inverse": False}
            if _bit_array(srt, [M, 1, D]) is None and _bit_array(cand, restored_shape) is None:
                checks["candidate_restore_is_inverse"] = bool(np.array_equal(
                    cand.reshape(n, D), srt.reshape(M, D)[self.ctx["inv"]]))
            if _bit_array(ref, restored_shape) is None and self.ref_restored_sha is not None:
                checks["reference_restore_matches_host_inverse"] = sha_array(ref) == self.ref_restored_sha
            summary["checks"] = checks
        return summary


# ================================================================ evaluation (pure)

def refusal_evidence(cell, cand):
    """Below admission: the candidate refuses BEFORE any backend; the route stays ordinary."""
    before = cand.ENGAGEMENT.snapshot()
    req = cand.MoEPrefillRequest(tokens=cell["tokens"], top_k=cell["top_k"], experts=cell["experts"],
                                 hidden=cell["hidden"], intermediate=cell["intermediate"])
    try:
        cand.admit(req)
        reason = None
    except cand.SegmentedMoERefused as exc:
        reason = exc.reason
    route = cand.decide_route(req)
    delta = {k: v for k, v in cand.ENGAGEMENT.fresh_since(before).items() if v}
    return {"refused": reason, "route": route.route, "research_candidate_admissible": route.research_candidate_admissible,
            "engagement_delta": delta, "backend_invoked": False}


def engagement_problems(eng, kind, native):
    """Exactly one fresh successful chain of this kind and nothing else (no raise, no refusal)."""
    if not isinstance(eng, dict) or not eng:
        return ["engagement missing"]
    nat, sub = f"native.successful_chains.{kind}", f"substituted.successful_chains.{kind}"
    problems = []
    if eng.get("calls") != 1 or eng.get("backend_raised") != 0 or (eng.get(nat), eng.get(sub)) not in ((1, 0), (0, 1)):
        problems.append(f"engagement is not one fresh successful {kind} chain: {eng}")
    others = {k: v for k, v in eng.items() if k not in ("calls", nat, sub) and v != 0}
    if others:
        problems.append(f"unexpected engagement deltas {others}")
    if native is not (eng.get(nat) == 1):
        problems.append("native flag disagrees with the counter that moved")
    return problems


def _stage_problems(name, exp, s):
    if not isinstance(s, dict):
        return ["missing"]
    problems = list(s.get("problems") or [])
    d, g = s.get("dispatches"), s.get("guard_calls")
    if type(d) is not int or d < 1 or type(g) is not int or g < d:
        problems.append(f"dispatches {d} not each preceded by a source/build guard ({g})")
    if exp["kind"] == "routes":
        checks = s.get("checks") or {}
        problems += [f"route check {c} failed" for c in ROUTE_CHECKS if checks.get(c) is not True]
        if s.get("rows") != exp["rows"] or s.get("pad") != exp["pad"]:
            problems.append(f"rows/pad {s.get('rows')}/{s.get('pad')} != {exp['rows']}/{exp['pad']}")
        if s.get("sorted_sha256") != exp["sorted_sha256"]:
            problems.append("sorted expert ids differ from the router law")
        return problems
    if exp["kind"] == "scan":
        checks = s.get("checks") or {}
        problems += [f"scan check {c} failed" for c, ok in checks.items() if ok is not True]
        if not checks:
            problems.append("scan checks missing")
        if (s.get("bm"), s.get("max_tiles"), s.get("count")) != (exp["bm"], exp["max_tiles"], exp["count"]):
            problems.append("scan bm/max_tiles/count differ from the tile law")
        if s.get("tiles_sha256") != exp["tiles_sha256"] or exp["candidate_mirror_agrees"] is not True:
            problems.append("scan tiles differ from the tile law (or the candidate host mirror)")
        return problems
    if problems:
        return problems
    if s.get("shape") != exp["shape"]:
        problems.append("shape differs")
    if s.get("kind") != exp["kind"] or s.get("plan") != exp["plan"]:
        problems.append(f"kind/plan {s.get('kind')}/{(s.get('plan') or {}).get('describe')} differ from admission")
    if s.get("bit_mismatches") != 0 or s.get("candidate_sha256") != s.get("reference_sha256"):
        problems.append(f"bit mismatch: {s.get('bit_mismatches')} elements (first {s.get('first_mismatch')}, "
                        f"max bit delta {s.get('max_bit_delta')})")
    if s.get("candidate_nonfinite") != 0 or s.get("reference_nonfinite") != 0:
        problems.append(f"non-finite bits: candidate {s.get('candidate_nonfinite')}, "
                        f"reference {s.get('reference_nonfinite')}")
    frac = s.get("reference_nonzero_fraction")
    if not isinstance(frac, float) or frac < MIN_NONZERO_FRACTION:
        problems.append(f"vacuous reference (nonzero fraction {frac})")
    problems += engagement_problems(s.get("engagement"), exp["kind"], s.get("native"))
    if name == "chain_restored":
        checks = s.get("checks") or {}
        for c in ("candidate_restore_is_inverse", "reference_restore_matches_host_inverse"):
            if checks.get(c) is not True:
                problems.append(f"restoration check {c} failed")
    return problems


def evaluate_cell(cell, evidence, cand):
    counts = {"native": 0, "substituted": 0}
    if not isinstance(evidence, dict):
        return ["evidence missing"], counts
    if VERDICT_KEYS & set(evidence):
        return [f"evidence carries verdict fields {sorted(VERDICT_KEYS & set(evidence))}"], counts
    law = cell_law(cell)
    problems = []
    if evidence.get("input_hashes") != law["hashes"]:
        problems.append("input identity hashes differ from the catalogue law")
    if cell["expect"] == "refuse":
        want = {"refused": "too_small", "route": cand.ORDINARY_REFERENCE, "research_candidate_admissible": False,
                "engagement_delta": {}, "backend_invoked": False}
        got = {k: evidence.get(k) for k in want}
        if got != want or set(evidence) - set(want) - {"input_hashes"}:
            problems.append(f"below-admission cell must be refused before any backend with no outputs: {got}")
        if refusal_evidence(cell, cand)["refused"] != "too_small":
            problems.append("catalogue refuse cell is not refused by the candidate admission law")
        return problems, counts
    if evidence.get("refused"):
        return problems + [f"refused unexpectedly: {evidence['refused']}"], counts
    tables = evidence.get("tables")
    if not (isinstance(tables, dict) and set(tables) == {"gate_up", "down"}
            and all(SHA.fullmatch(str(p.get("sha256"))) for t in tables.values() for p in t.values())):
        problems.append("table identity missing or malformed")
    stages = evidence.get("stages")
    if not isinstance(stages, dict):
        return problems + ["stages missing"], counts
    expected = expectations(cell, cand, law)
    problems += [f"missing stage {s}" for s in expected if s not in stages]
    problems += [f"unexpected stage {s}" for s in stages if s not in expected]
    for name, exp in expected.items():
        problems += [f"{name}: {p}" for p in _stage_problems(name, exp, stages.get(name))]
        s = stages.get(name)
        if exp["kind"] in KINDS and isinstance(s, dict) and isinstance(s.get("engagement"), dict):
            eng = s["engagement"]
            counts["native"] += int(eng.get(f"native.successful_chains.{exp['kind']}") == 1 and s.get("native") is True)
            counts["substituted"] += int(eng.get(f"substituted.successful_chains.{exp['kind']}") == 1)
    return problems, counts


def expected_chains(cells, cand):
    return sum(1 for c in cells if c["expect"] == "pass"
               for e in expectations(c, cand).values() if e["kind"] in KINDS)


def evaluate(report, cells, *, full_requested):
    """Verdict over a report. Every verdict field is computed here, never read from evidence."""
    cand = _candidate()
    refusals, per_cell = [], {}
    chains = {"native": 0, "substituted": 0}
    records = report.get("cells") if isinstance(report, dict) else None
    if not isinstance(records, list):
        records = []
        refusals.append("no cell evidence")
    ids = [r.get("id") if isinstance(r, dict) else None for r in records]
    for i in sorted({i for i in ids if ids.count(i) > 1}, key=str):
        refusals.append(f"duplicate cell {i}")
    want = [c["id"] for c in cells]
    refusals += [f"missing cell {i}" for i in want if i not in ids]
    refusals += [f"unexpected cell {i}" for i in ids if i not in want]
    if full_requested and want != list(MANDATORY):
        refusals.append("full gate requested with an incomplete or reordered catalogue")
    table = {c["id"]: c for c in cells}
    geometry_tables = {}
    for record in records:
        cell = table.get(record.get("id")) if isinstance(record, dict) else None
        if cell is None:
            continue
        evidence = record.get("evidence")
        if "timing" in record or (isinstance(evidence, dict) and "timing" in evidence):
            refusals.append(f"{cell['id']}: timing evidence is not accepted (this harness has no timing)")
        problems, counts = evaluate_cell(cell, evidence, cand)
        if cell["expect"] == "pass" and isinstance(evidence, dict):
            tid = json.dumps(evidence.get("tables"), sort_keys=True)
            if geometry_tables.setdefault(cell["geometry"], tid) != tid:
                problems.append("table identity differs from the other cells of this geometry")
        for k in chains:
            chains[k] += counts[k]
        per_cell[cell["id"]] = {"problems": problems, "chains": counts}
        refusals += [f"{cell['id']}: {p}" for p in problems]
    verdict = ("failed" if refusals else "bit_identity_evidence_pass" if full_requested
               else "partial_bit_identity_evidence")
    return {"verdict": verdict, "refusals": refusals, "per_cell": per_cell, "full_requested": full_requested,
            "chains": chains, "expected_chains": expected_chains(cells, cand),
            "native_synthetic_gate": False, "qualified": False, "selected": False, "observed_used": False,
            "model_gain": False, "scope": SCOPE,
            "evidence_label": "evaluation of supplied evidence; does not establish native execution"}


# ================================================================ source / build admission (no MLX)

def _git(root, *args):
    try:
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def head_commit(root):
    out = _git(root, "rev-parse", "HEAD")
    return out.decode().strip() if out else None


def _blob_id(data, width):
    algo = hashlib.sha1 if width == 40 else hashlib.sha256
    return algo(b"blob %d\0" % len(data) + data).hexdigest()


def tree_state(root):
    """(HEAD entries under TRACKED_PATHS, paths whose ACTUAL bytes differ from their HEAD blob)."""
    out = _git(root, "ls-tree", "-r", "-z", "HEAD", "--", *TRACKED_PATHS)
    if out is None:
        return {}, ["git ls-tree unavailable"]
    entries, bad = {}, []
    for rec in out.split(b"\0"):
        if not rec:
            continue
        meta, path = rec.split(b"\t", 1)
        mode, kind, oid = meta.decode().split()
        entries[path.decode()] = oid
        p = Path(root) / path.decode()
        if kind != "blob":
            bad.append(path.decode())
        elif mode == "120000":
            if not p.is_symlink() or _blob_id(os.readlink(p).encode(), len(oid)) != oid:
                bad.append(path.decode())
        elif p.is_symlink() or not p.is_file() or _blob_id(p.read_bytes(), len(oid)) != oid:
            bad.append(path.decode())
    return entries, bad


def status_refusal(root):
    """Tracked or untracked changes, or ignored files other than __pycache__ bytecode, refuse."""
    out = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignored=matching",
               "--", *TRACKED_PATHS)
    if out is None:
        return "git status unavailable"
    for rec in out.split(b"\0"):
        if not rec:
            continue
        code, path = rec[:2].decode(), rec[3:].decode()
        if code == "!!" and "__pycache__" in Path(path).parts:
            continue
        return f"source tree (src/scripts/tests/provenance) is not clean: {code} {path}"
    return None


def source_hashes(root):
    return {p: hashlib.sha256((Path(root) / p).read_bytes()).hexdigest()
            for p in BOUND_FILES if (Path(root) / p).is_file()}


def witness_refusal(root):
    for path, lines in REFERENCE_WITNESSES.items():
        text = {line.strip() for line in (Path(root) / path).read_text().splitlines()}
        missing = [w for w in lines if w not in text]
        if missing:
            return f"reference {path} no longer has the expected served path: {missing}"
    return None


def pythonpath_refusal(root, environ):
    """mlx2 must resolve to this checkout's src; no PYTHONPATH entry may name a sister checkout."""
    src = (Path(root) / "src").resolve()
    for entry in [e for e in environ.get("PYTHONPATH", "").split(os.pathsep) if e]:
        p = Path(entry).resolve()
        if (p / "mlx2" / "__init__.py").exists() and p != src:
            return f"PYTHONPATH entry {entry} resolves to another mlx2 checkout ({p})"
    spec = importlib.machinery.PathFinder.find_spec("mlx2", sys.path)
    if spec is None or spec.origin is None or Path(spec.origin).resolve() != src / "mlx2" / "__init__.py":
        return f"mlx2 does not resolve to {src}"
    loaded = sys.modules.get("mlx2")
    if loaded is not None and Path(getattr(loaded, "__file__", "") or "").resolve() != src / "mlx2" / "__init__.py":
        return "an mlx2 package from another path is already imported"
    return None


def mlx_base():
    """The installed mlx package directory, from distribution metadata (never imports MLX)."""
    try:
        return Path(importlib.metadata.distribution("mlx").locate_file("mlx")).resolve()
    except importlib.metadata.PackageNotFoundError as exc:
        raise Refused("MLX is not installed") from exc


def mlx_version():
    return importlib.metadata.version("mlx")


def header_closure(include_root, roots):
    """Headers the candidate's flattener reads (quoted mlx includes, transitively), preamble skipped."""
    seen, order = set(MLX_PREAMBLE_HEADERS), []

    def walk(rel):
        if rel in seen:
            return
        seen.add(rel)
        path = Path(include_root) / rel
        if not path.is_file():
            raise Refused(f"MLX kernel header {rel} missing")
        order.append(rel)
        for line in path.read_text().splitlines():
            s = line.strip()
            if s.startswith('#include "mlx/') and s.endswith('"'):
                walk(s[len('#include "'):-1])

    for r in roots:
        walk(r)
    return order


def mlx_files(base=None):
    """Hash the MLX core extension, libmlx, metallib, Python files and the used kernel headers."""
    base = Path(mlx_base() if base is None else base)
    cores = sorted(base.glob("core*.so"))
    if len(cores) != 1:
        raise Refused("expected exactly one MLX core extension")
    files = cores + [base / "lib" / "libmlx.dylib", base / "lib" / "mlx.metallib"]
    files += sorted(p for p in base.rglob("*.py") if "__pycache__" not in p.parts)
    include = base / "include"
    files += [include / h for h in header_closure(include, MLX_GATE_UP_HEADERS) + list(MLX_PREAMBLE_HEADERS)]
    missing = [str(f) for f in files if not f.is_file()]
    if missing:
        raise Refused(f"MLX files missing: {missing}")
    return {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}


def same_snapshot(observed, expected):
    """Exact equality of two {path: sha256} snapshots (no missing, extra or malformed entries)."""
    return (isinstance(observed, dict) and isinstance(expected, dict) and bool(expected)
            and set(observed) == set(expected) and all(SHA.fullmatch(str(v)) for v in expected.values())
            and all(observed[k] == expected[k] for k in expected))


def output_refusal(out):
    """A native run needs a NEW receipt path in an existing writable directory (checked before MLX)."""
    if not out:
        return "--out is required for a native run"
    if os.path.lexists(out):
        return "--out exists; refusing to overwrite a receipt"
    parent = Path(out).resolve().parent
    if not parent.is_dir() or not os.access(parent, os.W_OK):
        return "--out directory is missing or not writable"
    return None


def native_admission(args, environ, *, root=ROOT):
    """Everything that must hold before any MLX import or device query."""
    if getattr(args, "timing", False):
        raise Refused("timing is not implemented in this harness")
    if not args.run_native:
        raise Refused("no --run-native: this harness has no CPU fallback")
    if args.i_own_the_gpu is not True:
        raise Refused("--i-own-the-gpu acknowledgement missing (ownership itself is the parent wrapper's receipt)")
    root = Path(root)
    sr = args.source_root
    if not sr or not os.path.isabs(sr) or os.path.normpath(sr) != str(root):
        raise Refused(f"--source-root must be the absolute path of this checkout ({root})")
    env_root = environ.get("MLX2_INTAKE_SOURCE_ROOT")
    if env_root is not None and os.path.normpath(env_root) != str(root):
        raise Refused("MLX2_INTAKE_SOURCE_ROOT names another checkout")
    commit = environ.get("MLX2_INTAKE_SOURCE_COMMIT", "")
    if not re.fullmatch(r"[0-9a-f]{40}", str(args.source_commit or "")) or args.source_commit != commit:
        raise Refused("--source-commit must be a full sha equal to MLX2_INTAKE_SOURCE_COMMIT")
    top = _git(root, "rev-parse", "--show-toplevel")
    if top is None or Path(top.decode().strip()).resolve() != root.resolve():
        raise Refused("source root is not the top of its git checkout")
    if head_commit(root) != args.source_commit:
        raise Refused("git HEAD does not equal the source commit")
    problem = status_refusal(root)
    if problem:
        raise Refused(problem)
    entries, bad = tree_state(root)
    if bad:
        raise Refused(f"working bytes differ from HEAD blobs: {bad[:5]}")
    missing = [p for p in BOUND_FILES if p not in entries]
    if missing:
        raise Refused(f"bound files not committed at HEAD: {missing}")
    bound = source_hashes(root)
    drift = [p for p, sha in FROZEN.items() if bound.get(p) != sha]
    if drift:
        raise Refused(f"frozen candidate files differ from the reviewed hashes: {drift}")
    problem = witness_refusal(root) or pythonpath_refusal(root, environ)
    if problem:
        raise Refused(problem)
    out_refusal = output_refusal(args.out)                # preflight, before any MLX import
    if out_refusal:
        raise Refused(out_refusal)
    base = mlx_base()
    files = mlx_files(base)
    cores = [f for f in files if Path(f).parent == base and Path(f).name.startswith("core")]
    return {"root": str(root), "commit": args.source_commit, "bound": bound, "tracked_files": len(entries),
            "mlx_base": str(base), "mlx_files": files, "mlx_core": cores[0], "mlx_version": mlx_version(),
            "headers": header_closure(base / "include", MLX_GATE_UP_HEADERS)}


def import_identity_refusal(admission, mx_file, mx_version):
    """After ``import mlx.core`` and BEFORE any device query: the imported extension must be the admitted one."""
    core = str(Path(mx_file).resolve())
    if core != admission.get("mlx_core"):
        return "imported mlx.core extension is not the admitted file"
    if hashlib.sha256(Path(core).read_bytes()).hexdigest() != admission["mlx_files"].get(core):
        return "imported mlx.core extension bytes differ from admission"
    if mx_version != admission.get("mlx_version"):
        return "imported MLX version differs from the admitted package metadata"
    return None


def source_guard(admission):
    """Before every dispatch and after every cell: HEAD, clean status, HEAD-blob bytes, bound files, MLX build."""
    root = Path(admission["root"])
    if head_commit(root) != admission["commit"]:
        raise Refused("git HEAD differs from the admitted commit")
    problem = status_refusal(root)
    if problem:
        raise Refused(problem)
    _, bad = tree_state(root)
    if bad:
        raise Refused(f"working bytes differ from HEAD blobs: {bad[:5]}")
    if not same_snapshot(source_hashes(root), admission["bound"]):
        raise Refused("bound source files changed since admission")
    if not same_snapshot(mlx_files(admission["mlx_base"]), admission["mlx_files"]):
        raise Refused("installed MLX files changed since admission")


def _under(path, base):
    try:
        Path(path).resolve().relative_to(Path(base).resolve())
        return True
    except (TypeError, ValueError):
        return False


def module_path_refusals(admission, mx_file):
    refusals = []
    src = Path(admission["root"]) / "src"
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if (name == "mlx2" or name.startswith("mlx2.")) and f and not _under(f, src):
            refusals.append(f"module {name} loaded from outside this checkout ({f})")
        if (name == "mlx" or name.startswith("mlx.")) and f and not _under(f, admission["mlx_base"]):
            refusals.append(f"module {name} loaded from outside the admitted MLX ({f})")
    if str(Path(mx_file).resolve()) != admission["mlx_core"]:
        refusals.append("mlx.core path is not the admitted extension")
    return refusals


def post_run_refusals(admission, backend):
    refusals = []
    try:
        source_guard(admission)
    except Refused as error:
        refusals.append(str(error))
    root = Path(admission["root"])
    if Path(backend.cand.__file__).resolve() != root / CANDIDATE_FILE:
        refusals.append("candidate module path is not this checkout's")
    for mod, rel in ((backend.sl, REFERENCE_FILES[0]), (backend.act, REFERENCE_FILES[1])):
        if Path(mod.__file__).resolve() != root / rel:
            refusals.append(f"reference module {rel} not imported from this checkout")
    if backend.mx.__version__ != admission["mlx_version"]:
        refusals.append("MLX version changed during the run")
    return refusals + module_path_refusals(admission, backend.mx.__file__)


def candidate_refusal(cand):
    """The imported candidate must be this checkout's frozen module with the contract this harness assumes."""
    if Path(cand.__file__).resolve() != ROOT / CANDIDATE_FILE:
        return "candidate module was not imported from this checkout"
    if hashlib.sha256(Path(cand.__file__).read_bytes()).hexdigest() != FROZEN[CANDIDATE_FILE]:
        return "candidate module bytes differ from the frozen hash"
    if any(cand.STATE[k] is not False for k in ("qualified", "selected", "observed_used")):
        return "candidate state flags are not all false"
    if {k: dict(v) for k, v in cand.NAMED_GEOMETRIES.items()} != GEOMETRIES:
        return "candidate named geometries differ from the catalogue"
    if (cand._MLX_UTILS_HEADERS != MLX_PREAMBLE_HEADERS or cand._MLX_MM_HEADERS != MLX_DOWN_HEADERS
            or cand._MLX_MM_HEADERS + cand._MLX_OPS_HEADERS != MLX_GATE_UP_HEADERS):
        return "candidate flattens other MLX headers than the bound set"
    if (cand._native_backend.__defaults__ != (cand._MLXBackend,)
            or cand._is_native.__defaults__ != (cand._MLXBackend,)):
        return "candidate native class capture differs"
    if ((cand.SEAM_PAD_THRESHOLD, cand.SEAM_PAD_MULTIPLE) != (SEAM_THRESHOLD, SEAM_MULTIPLE)
            or (cand.SCHED_SEG, cand.SCHED_DB) != (0, 1) or tuple(cand.DISPATCH_KINDS) != KINDS):
        return "candidate seam/schedule/dispatch constants differ"
    return None


def _candidate():
    cand = importlib.import_module(CANDIDATE_NAME)
    problem = candidate_refusal(cand)
    if problem:
        raise Refused(problem)
    return cand


# ================================================================ orchestration

def run_gate(backend, cells, *, guard=None, tables=host_tables):
    """One cell at a time; tables live for one geometry. ``guard`` runs before every dispatch (the
    backend calls it) and at every cell boundary; a raise stops the run after cleanup. The report is
    EVIDENCE: it carries no execution status."""
    cand = _candidate()
    counter = _GuardCounter(guard)
    records, loaded = [], None
    try:
        for cell in cells:
            counter()
            law = cell_law(cell)
            evidence = {"input_hashes": law["hashes"]}
            if cell["expect"] == "refuse":
                evidence.update(refusal_evidence(cell, cand))
                records.append({"id": cell["id"], "evidence": evidence})
                continue
            if loaded is None or loaded[0] != cell["geometry"]:
                if loaded is not None:
                    loaded = None
                    backend.drop_tables()
                host = tables(cell["geometry"])
                ids = table_hashes(host)
                counter()
                loaded = (cell["geometry"], ids)
                backend.load_tables(cell["geometry"], host)
                del host
            evidence["tables"] = loaded[1]
            recorder = _CellRecorder(cell, law, expectations(cell, cand, law), counter)
            try:
                backend.run_cell(cell, {"ids": law["ids"], "x_bits": law["x_bits"]}, recorder, counter)
            finally:
                backend.release()
            evidence["stages"] = recorder.stages
            records.append({"id": cell["id"], "evidence": evidence})
            del law, recorder
            counter()
    finally:
        if loaded is not None:
            backend.drop_tables()
    return {"producer": type(backend).__name__, "cells": records, "identity": backend.identity(),
            "guard_calls": counter.calls}


class NativeBackend:
    """Real MLX/Metal backend; constructed only after native_admission (never in CPU tests)."""

    def __init__(self, admission):
        import mlx.core as mx

        refusal = import_identity_refusal(admission, mx.__file__, mx.__version__)
        if refusal:                                       # before ANY device query
            raise Refused(refusal)
        import mlx.nn as nn

        cand = _candidate()
        from mlx2.runtime.models import activations as act
        from mlx2.runtime.models import switch_layers as sl

        refusals = module_path_refusals(admission, mx.__file__)
        root = Path(admission["root"])
        for mod, rel in ((sl, REFERENCE_FILES[0]), (act, REFERENCE_FILES[1])):
            if Path(mod.__file__).resolve() != root / rel:
                refusals.append(f"reference {rel} not imported from this checkout")
        if sl.swiglu is not act.swiglu or sl._SORTED_GATHER_TAIL_BUG is not True:
            refusals.append("served swiglu or the seam pad switch differs from the bound reference")
        if refusals:
            raise Refused("; ".join(refusals))
        if not same_snapshot(mlx_files(admission["mlx_base"]), admission["mlx_files"]):
            raise Refused("installed MLX changed between admission and import")
        if mx.default_device() != mx.gpu or not mx.metal.is_available():
            raise Refused("default device is not an available GPU")
        info = dict(mx.device_info())
        if not str(info.get("architecture", "")).startswith("applegpu_g17"):
            raise Refused("not an M5-class (NAX tensor unit) device")
        include = Path(admission["mlx_base"]) / "include"
        self.flattened = {
            "gate_up": hashlib.sha256(cand.read_mlx_headers(include, MLX_GATE_UP_HEADERS).encode()).hexdigest(),
            "down": hashlib.sha256(cand.read_mlx_headers(include, MLX_DOWN_HEADERS).encode()).hexdigest()}
        self.mx, self.nn, self.cand, self.sl, self.act = mx, nn, cand, sl, act
        self.admission, self.info = admission, info
        self.activation = sl.SwiGLU()                     # FusedGateUpSwitchGLU's default activation module
        self.probe = cand._is_native.__defaults__[0]()    # the captured native class, for the scan probe only
        self.tables = self.ref_gate_up = self.ref_down = None
        self.engagement_start = cand.ENGAGEMENT.snapshot()

    def identity(self):
        import platform

        mx = self.mx
        return {"backend": "NativeBackend", "default_device": "gpu" if mx.default_device() == mx.gpu else "other",
                "architecture": str(self.info.get("architecture")),
                "device_info": {k: str(v) for k, v in self.info.items()}, "mlx_version": mx.__version__,
                "os": platform.platform(),
                "mlx_core": str(Path(mx.__file__).resolve()), "candidate_module": str(Path(self.cand.__file__)),
                "reference_modules": [str(Path(self.sl.__file__)), str(Path(self.act.__file__))],
                "flattened_headers_sha256": self.flattened, "commit": self.admission["commit"]}

    def _linear(self, w, s, b, K, N):
        """The served QuantizedSwitchLinear holding these tables (its __init__ would draw random weights)."""
        sl = self.sl
        lin = sl.QuantizedSwitchLinear.__new__(sl.QuantizedSwitchLinear)
        self.nn.Module.__init__(lin)
        lin.weight, lin.scales, lin.biases = w, s, b
        lin.group_size, lin.bits, lin.mode = 64, 4, "affine"
        lin.freeze()
        if (lin.input_dims, lin.output_dims, "bias" in lin) != (K, N, False):
            raise Refused("reference linear does not hold the bias-free table it was given")
        if sl._quantized_gather_tail_policy("affine", K, True) != "native":
            raise Refused("the served layer would not use native sorted gather_qmm at this K")
        return lin

    def load_tables(self, geometry, host):
        mx = self.mx
        g = GEOMETRIES[geometry]
        loaded = {}
        for proj in ("gate_up", "down"):
            p = host[proj]
            w = mx.array(p["weight"])
            s, b = mx.array(p["scales"]).view(mx.bfloat16), mx.array(p["biases"]).view(mx.bfloat16)
            mx.eval(w, s, b)
            loaded[proj] = (w, s, b)
        D, I = g["hidden"], g["intermediate"]
        self.tables = loaded
        self.ref_gate_up = self._linear(*loaded["gate_up"], D, 2 * I)
        self.ref_down = self._linear(*loaded["down"], I, D)

    def drop_tables(self):
        self.tables = self.ref_gate_up = self.ref_down = None
        self.mx.clear_cache()

    def release(self):
        self.mx.clear_cache()

    def _bits(self, a):
        import numpy as np

        if a.dtype != self.mx.bfloat16:
            raise Refused(f"output dtype {a.dtype} is not bfloat16")
        return np.array(a.view(self.mx.uint16))

    def _routes(self, cell, ids, sid, inv):
        """Candidate routes carrying EXACTLY the ordinary _gather_sort order (its tie order included)."""
        cand, k, n = self.cand, cell["top_k"], cell["assignments"]
        inv_l = [int(v) for v in inv]
        if sorted(inv_l) != list(range(n)):
            raise Refused("ordinary _gather_sort returned a non-permutation inv_order")
        order = [0] * n
        for a, pos in enumerate(inv_l):
            order[pos] = a
        pad = len(sid) - n
        row_map = [a // k for a in order] + [order[-1] // k] * pad
        routes = cand.validate_sorted_routes([int(e) for e in sid], row_map, tokens=cell["tokens"], top_k=k,
                                             experts=cell["experts"], order=tuple(order), inv_order=tuple(inv_l),
                                             pad=pad, expert_ids=tuple(int(e) for e in ids.reshape(-1)))
        host = cand.sort_routes_host([int(e) for e in ids.reshape(-1)], top_k=k, experts=cell["experts"],
                                     pad_policy="seam")
        if (routes.sorted_experts, routes.runs, routes.pad) != (host.sorted_experts, host.runs, host.pad):
            raise Refused("ordinary sorted runs differ from the candidate's host sort law")
        return routes

    def _call(self, fn, kind, x, routes, table, admission, pinned):
        before = self.cand.ENGAGEMENT.snapshot()
        out = fn(x, routes, *table, admission=admission, research_opt_in=True, plan=pinned)
        delta = dict(self.cand.ENGAGEMENT.fresh_since(before))
        self.mx.eval(out.output)                          # the fresh chain's output, evaluated before any check
        if out.kind != kind:
            raise Refused(f"candidate returned kind {out.kind}, expected {kind}")
        return {"output": out.output, "pp": out.plan, "kind": out.kind, "native": out.native is True,
                "engagement": delta, "plan": plan_record(out.plan)}

    def run_cell(self, cell, inputs, emit, guard):
        import numpy as np

        mx, cand, sl = self.mx, self.cand, self.sl

        def bits(a):                                       # the uint16 view is an MLX op too: guarded
            guard()
            return self._bits(a)
        T, k, E, D, I = (cell[x] for x in ("tokens", "top_k", "experts", "hidden", "intermediate"))
        gu_t, dn_t = self.tables["gate_up"], self.tables["down"]
        guard()
        x = mx.array(inputs["x_bits"]).view(mx.bfloat16)
        ids = mx.array(inputs["ids"])
        mx.eval(x, ids)
        guard()
        x_sorted, idx_sorted, inv_order = sl._gather_sort(mx.expand_dims(x, (-2, -3)), ids)
        mx.eval(x_sorted, idx_sorted, inv_order)
        sid, inv = np.array(idx_sorted), np.array(inv_order)
        emit("routes", {"sorted_ids": sid, "inv_order": inv, "x_sorted_bits": bits(x_sorted), "dispatches": 2})
        routes = self._routes(cell, inputs["ids"], sid, inv)
        admission = cand.admit(cand.MoEPrefillRequest(tokens=T, top_k=k, experts=E, hidden=D, intermediate=I))
        # Ordinary reference, once per cell: served layer code on the same sorted assignments.
        guard()
        gate_up = self.ref_gate_up(x_sorted, idx_sorted, sorted_indices=True)
        hidden_ref = self.activation(gate_up[..., I:], gate_up[..., :I])
        mx.eval(hidden_ref)
        del gate_up, x_sorted
        hidden_bits = bits(hidden_ref)
        guard()
        down_ref = self.ref_down(hidden_ref, idx_sorted, sorted_indices=True)
        mx.eval(down_ref)
        down_bits = bits(down_ref)
        guard()
        restored_ref = sl._scatter_unsort(down_ref, inv_order, ids.shape)
        mx.eval(restored_ref)
        restored_bits = bits(restored_ref)
        del down_ref, restored_ref
        x_tokens = x.reshape(T, 1, D)
        scans = {}
        for v in variants(cell):
            pinned, lab = (None if v is None else cand.Plan(*v)), variant_label(v)
            guard()
            gu = self._call(cand.research_mapped_gate_up_swiglu, KINDS[0], x_tokens, routes, gu_t, admission, pinned)
            emit(f"gate_up:{lab}", {"candidate": bits(gu["output"]), "reference": hidden_bits, "dispatches": 1,
                                    **{key: gu[key] for key in ("kind", "native", "engagement", "plan")}})
            guard()
            dn = self._call(cand.research_segmented_down, KINDS[1], hidden_ref, routes, dn_t, admission, pinned)
            emit(f"down_isolated:{lab}", {"candidate": bits(dn["output"]), "reference": down_bits,
                                          "dispatches": 1,
                                          **{key: dn[key] for key in ("kind", "native", "engagement", "plan")}})
            for pp in (gu["pp"], dn["pp"]):                # the plans the entry actually used
                scans.setdefault(pp.plan.bm, pp)
            del dn
            if v is None:
                guard()
                ch = self._call(cand.research_segmented_down, KINDS[1], gu["output"], routes, dn_t, admission, None)
                guard()
                restored = sl._scatter_unsort(ch["output"], inv_order, ids.shape)
                mx.eval(restored)
                emit("chain_restored", {"candidate": bits(restored), "candidate_sorted": bits(ch["output"]),
                                        "reference": restored_bits, "dispatches": 2,
                                        **{key: ch[key] for key in ("kind", "native", "engagement", "plan")}})
                del ch, restored
            del gu
        for bm, pp in sorted(scans.items()):
            guard()
            tiles, count = self.probe._tiles(self.probe.upload_u32(routes.sorted_experts), pp, routes.rows, E)
            mx.eval(tiles, count)
            c = int(np.array(count)[0])
            table = np.array(tiles).reshape(-1, 4)[: min(c, pp.max_tiles)].copy()
            emit(f"scan:bm{bm}", {"tiles": table, "count": c, "bm": bm, "max_tiles": int(pp.max_tiles),
                                  "dispatches": 1})


_WITNESS = object()   # module-private; held only by live runs built in _run_native


class _LiveNativeRun:
    """In-process record of an actual native orchestration (never serialized)."""

    __slots__ = ("_witness", "backend", "admission", "report", "post_run", "cells", "full_requested", "fresh")

    def __init__(self, witness, backend, admission, report, post_run, cells, full_requested, fresh):
        self._witness, self.backend, self.admission = witness, backend, admission
        self.report, self.post_run = report, list(post_run)
        self.cells, self.full_requested, self.fresh = tuple(cells), bool(full_requested), dict(fresh)


def _bind_native_identity(native_cls, live_cls, witness, evaluator):
    """Capture the original native class, live-run type, witness and evaluator at import.

    Closure cells, not default arguments: no caller argument can substitute a class, run type,
    witness or evaluator, and rebinding the module names later changes neither what ``_run_native``
    constructs nor what ``native_verdict`` accepts. Evidence-association hygiene, not protection
    against code that edits closures, class attributes or private objects.
    """

    def _run_native(admission, cells, full_requested):
        backend = native_cls(admission)
        report = run_gate(backend, cells, guard=lambda: source_guard(admission))
        fresh = backend.cand.ENGAGEMENT.fresh_since(backend.engagement_start)
        post = post_run_refusals(admission, backend)
        return live_cls(witness, backend, admission, report, post, cells, full_requested, fresh)

    def native_verdict(live):
        """The ONLY place a native synthetic gate can be stamped: a live run object, never a dict/JSON.
        The evaluation is computed HERE from this run's own report and cells."""
        reasons = []
        report, cells = getattr(live, "report", None), getattr(live, "cells", None)
        full = getattr(live, "full_requested", False) is True
        if isinstance(report, dict) and isinstance(cells, (list, tuple)) and cells:
            evaluation = evaluator(report, list(cells), full_requested=full)
        else:
            evaluation = {"verdict": "failed", "refusals": ["no live report/cells"], "chains": {},
                          "expected_chains": 0}
        if type(live) is not live_cls or getattr(live, "_witness", None) is not witness:
            reasons.append("no live native orchestration (evidence alone never establishes native execution)")
        elif type(live.backend) is not native_cls:
            reasons.append("backend is not the native backend")
        else:
            reasons += live.post_run
            chains, want = evaluation.get("chains") or {}, evaluation.get("expected_chains")
            fresh = live.fresh
            native_moved = sum(fresh.get(f"native.successful_chains.{k}", 0) for k in KINDS)
            if not (isinstance(want, int) and want > 0 and chains.get("native") == want == native_moved
                    and chains.get("substituted") == 0):
                reasons.append(f"fresh native chains {native_moved} / evaluated native stages "
                               f"{chains.get('native')} do not both equal the {want} expected checked outputs")
            stray = {k: v for k, v in fresh.items() if v and not k.startswith("native.successful_chains.")
                     and k != "calls"}
            if stray or fresh.get("calls") != native_moved:
                reasons.append(f"engagement counters moved outside successful native chains: {stray}")
        if evaluation.get("verdict") != "bit_identity_evidence_pass":
            reasons.append("bit-identity evidence did not pass the full mandatory catalogue")
        return {"native_synthetic_gate": not reasons, "reasons": reasons, "evaluation": evaluation,
                "qualified": False, "selected": False, "observed_used": False, "model_gain": False}

    _run_native.__qualname__, native_verdict.__qualname__ = "_run_native", "native_verdict"
    return _run_native, native_verdict


_run_native, native_verdict = _bind_native_identity(NativeBackend, _LiveNativeRun, _WITNESS, evaluate)


# ================================================================ CLI

def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--catalogue", action="store_true", help="print the catalogue (no MLX import)")
    ap.add_argument("--run-native", action="store_true")
    ap.add_argument("--i-own-the-gpu", action="store_true")
    ap.add_argument("--source-root")
    ap.add_argument("--source-commit")
    ap.add_argument("--cells", nargs="+", help="partial run (labelled partial, never a full gate)")
    ap.add_argument("--timing", action="store_true", help="refused: this harness has no timing")
    ap.add_argument("--out")
    return ap


def catalogue_listing():
    cand = _candidate()
    cells = []
    for c in CATALOGUE:
        entry = dict(c)
        if c["expect"] == "pass":
            entry["stages"] = {k: {x: v[x] for x in ("kind", "shape", "plan", "bm", "count") if x in v}
                               for k, v in expectations(c, cand).items() if k != "routes"}
        cells.append(entry)
    return {"schema": SCHEMA, "scope": SCOPE, "geometries": GEOMETRIES, "plan_sweep": PLAN_SWEEP,
            "table_law": TABLE_LAW, "mandatory": MANDATORY, "cells": cells}


def _refused(reasons):
    print(json.dumps({"verdict": "refused", "refusals": reasons, "native_synthetic_gate": False,
                      "qualified": False, "selected": False, "observed_used": False, "model_gain": False}))
    return 1


def write_receipt(path, receipt):
    text = json.dumps(receipt, indent=1)                  # serialize first: never a partial receipt file
    with open(path, "x") as handle:                      # exclusive creation: never overwrite a receipt
        handle.write(text)


def main(argv=None, environ=None):
    args = build_parser().parse_args(argv)
    if args.timing:                                        # before any admission, import or backend
        return _refused(["timing is not implemented in this harness"])
    environ = os.environ if environ is None else environ
    if str(SRC) not in sys.path[:1]:
        sys.path.insert(0, str(SRC))
    if args.catalogue:
        print(json.dumps(catalogue_listing(), indent=1))
        return 0
    try:
        cells, full = select_cells(args.cells)
        admission = native_admission(args, environ)        # before any MLX import
        live = _run_native(admission, cells, full)
        native = native_verdict(live)                      # evaluation bound to this live run
        evaluation = native.pop("evaluation")
    except Exception as error:                             # fail closed: no receipt for an unfinished run
        return _refused([f"{type(error).__name__}: {error}"])
    receipt = {"schema": SCHEMA, "scope": SCOPE, "admission": live.admission, "evaluation": evaluation,
               "native_execution": native, "report": live.report, "engagement_fresh": live.fresh,
               "qualified": False, "selected": False, "observed_used": False, "model_gain": False,
               "note": "native_execution was stamped in-process; re-evaluating this JSON can never re-establish it"}
    try:
        write_receipt(args.out, receipt)
    except (OSError, TypeError, ValueError) as error:      # appeared meanwhile / unwritable: never overwrite
        return _refused([f"receipt not written: {type(error).__name__}: {error}",
                         f"run verdict was {evaluation['verdict']}, native gate {native['native_synthetic_gate']}"])
    print(json.dumps({"verdict": evaluation["verdict"], **native}))
    return 0 if native["native_synthetic_gate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
