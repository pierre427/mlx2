"""Custom launches that run the installed MLX's own kernel code, proven per shape.

Clean-room implementation (2026-11-23) from the functional specification
``spec-mlx-library-kernels.md``; see
``provenance/inkling2-library-kernels-cleanroom.json``. No MLX header text is
kept in mlx2: the device code is read at run time from the installed wheel
(``<mlx>/include/mlx/backend/metal/kernels``, MIT) and handed to
``mx.fast.metal_kernel`` as its header.

Why. A fused launch that does its own arithmetic sums in its own order and
differs from MLX in the last bits. A launch that calls MLX's device functions
(``GEMVKernel::run``, ``GemvWide::run``, the unary/binary functors) with the
template parameters and per-thread coordinates MLX's host dispatch would use
computes MLX's values. MLX's dispatch rules live in host C++ that is not
installed, so they are restated here (``matvec_plan``) from the v0.32.4
source, and every launch shape is checked bit for bit against the composed MLX
path on its first call (``run_proved``) before it serves.

Pieces:

- ``library(part, ...)``: the joined header text for a part (``"matvec"``:
  ``gemv.h``; ``"elementwise"``: ``binary_ops.h`` + ``unary_ops.h``), every
  quoted include resolved depth first, each file once, with the files the
  custom-kernel prelude already carries (``utils.h`` and its includes) left
  out. Declarations the launches rely on are fingerprinted; anything missing,
  changed or unexpected is a refusal reason, never an exception.
- ``matvec_plan`` / ``logical_groups`` / ``tgp_floats`` / ``matvec_call``:
  MLX 0.32.4's product routing and tiles for ``x[rows, K] @ W[N, K]^T``, and
  the Metal statement that runs one MLX threadgroup's worth of it inside a
  1-D launch of ``THREADS`` threads.
- ``Proofs`` / ``run_proved`` / ``run_planned`` / ``Planned``: the per-shape
  first-use proof and its counters.

Switch: ``MLX2_INKLING2_LIBRARY_KERNELS=0`` (or ``set_enabled(False)``) turns
the whole component off: ``library`` refuses (``disabled``) and
``run_proved`` serves the composed path, counted ``unproven``. Every caller
is also behind its own opt-in model option (default off).

Coordinate mapping (spec 4.4). MLX launches ``GEMVKernel`` with threadgroups
of ``(32, BN, BM)`` threads; a 3-D threadgroup's linear thread index is
``x + 32 * (y + BN * z)``, so simdgroup ``s`` (32 consecutive linear threads)
is ``y = s % BN``, ``z = s / BN`` and its lane is ``x``. Here a threadgroup of
``THREADS`` linear threads holds ``THREADS // (32 * BN * BM)`` MLX
threadgroups side by side: MLX threadgroup slot ``j`` owns simdgroups
``[j * BN * BM, (j + 1) * BN * BM)``, and its local simdgroup ``s`` gets
``simd_gid = s``, ``simd_lid = lane``, ``lid = (lane, s % BN, s / BN)`` and
``tid = (logical index, 0, 0)``: the values MLX's launch gives the thread
doing that work. ``GemvWide`` (threadgroup ``(32, k_lanes / 8, 1)``) maps the
same way with ``tid = (0, row block, 0)`` and ``threadgroups_per_grid.x = 1``:
MLX's grid has ``grid_x`` columns that split the vector chunks round robin,
and each (chunk, row) is reduced the same way whichever column runs it, so
one column walking every chunk gives each output MLX's bits (and the proof
checks it).
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
import weakref
from pathlib import Path

import mlx.core as mx

# Threads per launched threadgroup. MLX's largest gemv threadgroup in the
# mirrored rules is 32 * BN * BM = 256 (BM = 8, or BN = 8).
THREADS = 256
# MLX 0.32.4 gemv_wide_config: passes = ceil(M / 5) must be <= 3.
WIDE_MAX_ROWS = 15
WIDE_MIN_GENERATION = 15  # applegpu_g15 (M3) and later
SWITCH_ENV = "MLX2_INKLING2_LIBRARY_KERNELS"
# Passing proofs (on separate calls, so on separate operands) a key needs
# before it serves; default 1. A BF16/FP16 output can hide a wrong summation
# order behind its rounding on one draw, so a mirror under suspicion can be
# run with 2 or more.
PROOFS_ENV = "MLX2_INKLING2_LIBRARY_PROOFS"
_PREFIX = "mlx/backend/metal/kernels/"
_PRELUDE_ENTRY = "utils.h"

_METAL_TYPES = {mx.bfloat16: "bfloat16_t", mx.float16: "half", mx.float32: "float"}
_UINT_VIEW = {1: mx.uint8, 2: mx.uint16, 4: mx.uint32, 8: mx.uint64}

_switch_override = None
_lock = threading.RLock()


# ----- switch ---------------------------------------------------------------

def enabled():
    """Whether the component is on (default on; the callers are opt-in)."""
    if _switch_override is not None:
        return _switch_override
    return os.environ.get(SWITCH_ENV, "1").strip().lower() not in ("0", "off", "false", "no")


def set_enabled(value):
    """Override the environment switch (``None`` restores it)."""
    global _switch_override
    _switch_override = None if value is None else bool(value)


def proofs_required():
    try:
        return max(1, int(os.environ.get(PROOFS_ENV, "1")))
    except ValueError:
        return 1


def mlx_version():
    return getattr(mx, "__version__", "unknown")


def include_root():
    """The installed wheel's include directory."""
    return Path(mx.__file__).resolve().parent / "include"


# ----- header resolution (spec 3) -------------------------------------------

class _Refusal(Exception):
    def __init__(self, reason):
        super().__init__(reason)
        self.reason = reason


_QUOTED_INCLUDE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*"([^"\n]*)"[^\n]*\n?', re.M)
_PRAGMA_ONCE = re.compile(r'^[ \t]*#[ \t]*pragma[ \t]+once\b[^\n]*\n?', re.M)


def _target(include):
    """The kernels-relative path of a quoted include, or a refusal."""
    if not include.startswith(_PREFIX):
        raise _Refusal(f"header_outside:{include}")
    rel = include[len(_PREFIX):]
    parts = rel.split("/")
    if not rel or any(p in ("", ".", "..") for p in parts):
        raise _Refusal(f"header_outside:{include}")
    return rel


def _read(root, rel):
    path = Path(root) / _PREFIX / rel
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise _Refusal(f"header_missing:{rel}") from None
    except (OSError, UnicodeDecodeError):
        raise _Refusal(f"header_unreadable:{rel}") from None


def _walk(root, rel, done, visiting, emit):
    """Depth first: children before the file, each file once."""
    if rel in done:
        return
    if rel in visiting:
        raise _Refusal(f"header_cycle:{rel}")
    visiting.add(rel)
    text = _read(root, rel)
    for match in _QUOTED_INCLUDE.finditer(text):
        _walk(root, _target(match.group(1)), done, visiting, emit)
    visiting.discard(rel)
    done.add(rel)
    if emit is not None:
        emit(rel, text)


def carried_set(root=None):
    """``utils.h`` and every header it includes: the prelude MLX puts in front
    of every custom kernel (its ``metal::utils()`` JIT source)."""
    done = set()
    _walk(root or include_root(), _PRELUDE_ENTRY, done, set(), None)
    return done


def resolve(entries, root=None, version=None):
    """``(joined text, files, None)`` or ``(None, (), reason)``."""
    root = root or include_root()
    version = version or mlx_version()
    chunks, files = [], []

    def emit(rel, text):
        body = _PRAGMA_ONCE.sub("", _QUOTED_INCLUDE.sub("", text))
        chunks.append(f"// ---- {_PREFIX}{rel} (MLX {version}, installed wheel) ----\n{body}")
        files.append(rel)

    try:
        done = carried_set(root)
        visiting = set()
        for entry in entries:
            _walk(root, entry, done, visiting, emit)
    except _Refusal as refusal:
        return None, (), refusal.reason
    return "\n".join(chunks), tuple(files), None


# ----- fingerprints (spec 3, step 5) ----------------------------------------

def _squash(text):
    return " ".join(text.split())


def _strip_comments(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def _split_top(text):
    """Split at commas outside (), <> and [] nesting."""
    out, depth, cur = [], 0, []
    for ch in text:
        if ch in "(<[":
            depth += 1
        elif ch in ")>]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if "".join(cur).strip():
        out.append("".join(cur))
    return [_squash(p) for p in out]


def _struct_region(code, name):
    """(template parameter text, body) of ``struct name {`` or None."""
    match = re.search(r"\bstruct\s+" + re.escape(name) + r"\s*\{", code)
    if match is None or re.search(r"\bstruct\s+" + re.escape(name) + r"\s*\{",
                                  code[match.end():]):
        return None  # missing, or declared more than once
    head = code[:match.start()]
    tpl = head.rfind("template")
    params = None
    if tpl >= 0:
        open_at = head.find("<", tpl)
        depth, i = 0, open_at
        while i < len(head):
            if head[i] == "<":
                depth += 1
            elif head[i] == ">":
                depth -= 1
                if depth == 0:
                    params = head[open_at + 1:i]
                    break
            i += 1
    end = code.find("\n};", match.end())
    return params, code[match.end(): end if end >= 0 else len(code)]


def _run_types(body):
    """Parameter types of the first ``run(`` in a struct body (names and
    attributes dropped)."""
    match = re.search(r"\bvoid\s+run\s*\(", body)
    if match is None:
        return None
    start = match.end()
    depth, i = 1, start
    while i < len(body) and depth:
        depth += {"(": 1, ")": -1}.get(body[i], 0)
        i += 1
    params = _split_top(body[start:i - 1])
    types = []
    for p in params:
        p = re.sub(r"\[\[.*?\]\]", "", p).strip()
        types.append(_squash(re.sub(r"\b\w+$", "", p)).replace(" &", "&").replace(" *", "*"))
    return types


def _template_names(params):
    names = []
    for p in _split_top(params or ""):
        p = p.split("=")[0].strip()
        names.append(p.split()[-1] if p else "")
    return names


_GEMV_RUN = ["const device T*", "const device T*", "const device T*", "device T*",
             "const constant int&", "const constant int&", "const constant int&",
             "const constant float&", "const constant float&", "const constant int&",
             "threadgroup AccT*", "uint3", "uint3", "uint", "uint"]
_WIDE_RUN = ["const device T*", "const device T*", "const device T*", "device T*",
             "int", "int", "int", "int", "int", "float", "float", "int", "int",
             "uint3", "uint3", "uint", "uint"]

# (struct, template parameter names, run() parameter types or None).
_STRUCTS = {
    "matvec": [
        ("GEMVKernel", ["T", "BM", "BN", "SM", "SN", "TM", "TN", "kDoAxpby", "AccT"], _GEMV_RUN),
        ("GEMVTKernel", ["T", "BM", "BN", "SM", "SN", "TM", "TN", "kDoAxpby", "AccT"], _GEMV_RUN),
        ("GemvWide", ["T", "vecs_per_tg", "k_lanes", "AccT"], _WIDE_RUN),
    ],
    "elementwise": [(name, None, None) for name in (
        "Add", "Subtract", "Multiply", "Divide", "Maximum", "Minimum", "Power",
        "LogAddExp", "Exp", "Log", "Sigmoid", "Tanh", "Rsqrt", "Sqrt", "Erf", "Expm1")],
}
# Statements whose meaning matvec_plan / logical_groups / tgp_floats restate.
_NEEDLES = {
    "matvec": [
        "MLX_MTL_CONST short tgp_mem_size = BN > 1 ? BN*(blockM + TM) : 0;",
        "static constexpr constant int num_simdgroups = k_lanes / 8;",
        "constexpr int rows_per_simdgroup = 32 / k_lanes;",
        "for (int chunk = tid.x; chunk < n_chunks; chunk += tgpg.x) {",
        "const int simdM = BN != 1 ? SM * (simd_gid / BN) : int(SM * simd_gid);",
        "threadgroup AccT* tgp_results = tgp_memory + sgN * (blockM + TM) + bm;",
        "int out_row = tid.x * blockM + bm;",
    ],
    "elementwise": [],
}
_PART_ENTRIES = {"matvec": ("gemv.h",), "elementwise": ("binary_ops.h", "unary_ops.h")}
_FC_DECL = re.compile(
    r"constant\s+bool\s+(\w+)\s*\[\[\s*function_constant\(\s*\d+\s*\)\s*\]\]\s*;")
_FC_ATTR = re.compile(r",\s*function_constant\(\s*\w+\s*\)")
_WIDE_CONSTANTS = {"gemv_wide_has_batch", "gemv_wide_do_axpby"}


def _patch_function_constants(text):
    """``gemv_wide``'s two function constants become ``false`` constants (the
    values MLX's plain, unbatched wide launch sets); the attributes that tie
    the uninstantiated kernel templates' buffers to them are dropped. A
    ``metal_kernel`` source has no way to set function constants."""
    names = _FC_DECL.findall(text)
    if set(names) != _WIDE_CONSTANTS or len(names) != len(_WIDE_CONSTANTS):
        raise _Refusal("header_changed:matvec:function_constant")
    text = _FC_DECL.sub(lambda m: f"constant constexpr bool {m.group(1)} = false;", text)
    text = _FC_ATTR.sub("", text)
    if "function_constant(" in text:
        raise _Refusal("header_changed:matvec:function_constant")
    return text


def _check(part, text):
    code = _strip_comments(text)
    flat = _squash(code)
    for name, tparams, run in _STRUCTS[part]:
        region = _struct_region(code, name)
        if region is None:
            raise _Refusal(f"header_changed:{part}:struct {name}")
        params, body = region
        if tparams is not None and _template_names(params) != tparams:
            raise _Refusal(f"header_changed:{part}:{name}<>")
        if run is not None and _run_types(body) != run:
            raise _Refusal(f"header_changed:{part}:{name}::run(")
        if tparams is None and "operator()" not in body:
            raise _Refusal(f"header_changed:{part}:{name}::operator()")
    for needle in _NEEDLES[part]:
        if flat.count(_squash(needle)) != 1:
            raise _Refusal(f"header_changed:{part}:{needle}")
    if '#include "mlx/' in text:
        raise _Refusal(f"header_changed:{part}:include")


_libraries = {}
_library_info = {}
_library_refusals = {}


def _build(parts, root):
    entries, seen = [], set()
    for part in parts:
        for entry in _PART_ENTRIES[part]:
            if entry not in seen:
                seen.add(entry)
                entries.append(entry)
    text, files, why = resolve(entries, root)
    if text is None:
        return None, why, {}
    try:
        if "matvec" in parts:
            text = _patch_function_constants(text)
        for part in parts:
            _check(part, text)
    except _Refusal as refusal:
        return None, refusal.reason, {}
    digest = hashlib.sha256(text.encode()).hexdigest()
    banner = (f"// mlx2 library_kernels: MLX {mlx_version()} kernel headers for "
              f"{'+'.join(parts)} (sha256 {digest}), read from the installed wheel\n")
    return banner + text, None, {"sha256": digest, "files": list(files), "bytes": len(text)}


def library(*parts, root=None):
    """``(header text, None)`` or ``(None, reason)`` for one or more parts.

    Cached per process by the parts requested; ``root`` (tests) bypasses the
    cache. A header-level refusal is counted once per process."""
    if not parts:
        return None, "no library part ''"
    for part in parts:
        if part not in _PART_ENTRIES:
            return None, f"no library part '{part}'"
    if not enabled():
        return None, "disabled"
    key = tuple(parts)
    if root is not None:
        text, why, _info = _build(key, root)
        return text, why
    with _lock:
        held = _libraries.get(key)
        if held is None:
            text, why, info = _build(key, None)
            held = _libraries[key] = (text, why)
            _library_info[key] = info if text is not None else {"refused": why}
            if why is not None:
                _library_refusals[why] = _library_refusals.get(why, 0) + 1
    return held


def library_receipt():
    """MLX version, per-part hashes and files, and header refusals."""
    with _lock:
        return {"mlx_version": mlx_version(), "enabled": enabled(),
                "include_root": str(include_root()),
                "parts": {"+".join(k): dict(v) for k, v in _library_info.items()},
                "refusals": dict(_library_refusals)}


# ----- MLX 0.32.4 product routing (spec 4) ----------------------------------

def metal_type(dtype):
    """The Metal spelling MLX instantiates its kernels with for ``dtype``."""
    try:
        return _METAL_TYPES[dtype]
    except KeyError:
        raise ValueError(f"no library launch for {dtype}") from None


_generation = None


def gpu_generation():
    """MLX's ``Device::get_architecture_gen``: the two digits before the last
    character of the architecture name (``applegpu_g17s`` -> 17), honouring
    ``MLX_METAL_GPU_ARCH``; 0 when unknown."""
    global _generation
    if _generation is None:
        arch = os.environ.get("MLX_METAL_GPU_ARCH", "")
        if not arch:
            try:
                info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
                arch = str(info.get("architecture", ""))
            except Exception:
                arch = ""
        tens = ones = 0
        if len(arch) >= 3:
            tens = int(arch[-3]) if arch[-3].isdigit() else 0
            ones = int(arch[-2]) if arch[-2].isdigit() else 0
        _generation = tens * 10 + ones
    return _generation


def _gemv_tiles(n, k):
    """``gemv_axbpy``'s tiles for a non-transposed matrix (one input row):
    ``(BM, BN, SM, SN, TM, TN)``."""
    bm, bn, sm, sn, tm, tn = (8 if n >= 4096 else 4), 1, 1, 32, 4, 4
    if k <= 64:
        bm, sm, sn = 1, 8, 4
    elif k >= 16 * n:
        bm, bn = 1, 8
    if n < tm:
        tm = 1
    return (bm, bn, sm, sn, tm, tn)


def matvec_plan(rows, n, k, dtype, generation=None, *, transposed=False):
    """MLX's kernel for ``x[rows, k] @ W[n, k]^T`` (W row contiguous, x row
    contiguous, one batch, no bias): ``("gemv", (BM, BN, SM, SN, TM, TN))``,
    ``("wide", (k_lanes, grid_x))``, or ``(None, reason)`` when MLX would run
    something not mirrored here (its dot product, steel GEMM or NAX GEMM)."""
    if dtype not in _METAL_TYPES:
        return None, "dtype"
    if transposed:
        return None, "outside_rules"  # GEMVTKernel routing is not mirrored
    if rows < 1 or n < 1 or k < 8:
        # k < 8: the vector would be bound in the constant address space.
        return None, "geometry"
    if rows == 1:
        if n == 1:
            return None, "outside_rules"  # MLX's dot_product
        return "gemv", _gemv_tiles(n, k)
    # rows >= 2: MLX tries gemv_wide first, then steel GEMM.
    if rows > WIDE_MAX_ROWS:
        return None, "rows"
    if dtype == mx.float32:
        return None, "rows_dtype"
    gen = gpu_generation() if generation is None else int(generation)
    if gen < WIDE_MIN_GENERATION:
        return None, "gpu_generation"
    if k % 4:
        return None, "k_align"
    if n <= 1:
        return None, "outside_rules"
    passes = (rows + 4) // 5
    k_lanes = 32 if passes == 1 or n <= 64 else 16
    grid_x = 1 if n >= 65536 else passes
    return "wide", (k_lanes, grid_x)


def _wide_vecs(rows):
    passes = (rows + 4) // 5
    return -(-rows // passes)


def logical_groups(kind, tiles, n):
    """``(MLX threadgroups for n outputs, how many fit in THREADS threads,
    threads per MLX threadgroup)``."""
    if kind == "gemv":
        bm, bn, sm, _sn, tm, _tn = tiles
        threads = 32 * bn * bm
        logical = -(-n // (bm * sm * tm))
    elif kind == "wide":
        threads = 32 * (tiles[0] // 8)
        logical = -(-n // 4)  # GemvWide: rows_per_simdgroup * num_simdgroups == 4
    else:
        raise ValueError(f"unknown product kind {kind!r}")
    return logical, THREADS // threads, threads


def _gemv_tgp(tiles):
    bm, bn, sm, _sn, tm, _tn = tiles
    return bn * (bm * sm * tm + tm) if bn > 1 else 0


def tgp_floats(kind, tiles):
    """Threadgroup floats one launched threadgroup needs (at least 1)."""
    if kind != "gemv":
        return 1
    mem = _gemv_tgp(tiles)
    per_group = THREADS // (32 * tiles[1] * tiles[0])
    return max(1, mem * max(per_group, 1))


def matvec_call(kind, tiles, *, dtype, rows, mat, vec, out, n_ref, logical, lsg,
                k_ref="sizes[0]", unit_ref="unit[0]", tgp="tgp"):
    """One MLX threadgroup's share of ``out[rows, n] = vec[rows, K] @ mat[n, K]^T``.

    ``dtype``: the Metal type (``metal_type``); ``n_ref``/``k_ref``: ``constant
    int`` lvalues holding N and K (K is also both leading dimensions);
    ``unit_ref``: a ``constant float`` lvalue (alpha/beta, unused without
    axpby); ``logical``: this MLX threadgroup's index; ``lsg``: the simdgroup's
    index inside it; ``tgp``: a ``threadgroup float`` array of
    ``tgp_floats(kind, tiles)`` elements. The source must run it for every
    simdgroup of the MLX threadgroup (BN > 1 has a threadgroup reduction)."""
    if kind == "gemv":
        bm, bn, sm, sn, tm, tn = tiles
        if rows != 1:
            raise ValueError("gemv is the one-row product")
        mem = _gemv_tgp(tiles)
        sgs = bn * bm
        tgp_expr = (f"{tgp} + (simdgroup_index_in_threadgroup / {sgs}u) * {mem}u" if mem
                    else "nullptr")
        return (
            f"{{ /* MLX gemv_axbpy: GEMVKernel bm{bm} bn{bn} sm{sm} sn{sn} tm{tm} tn{tn}, "
            f"threadgroup (32, {bn}, {bm}) */\n"
            f"  GEMVKernel<{dtype}, {bm}, {bn}, {sm}, {sn}, {tm}, {tn}, false>::run(\n"
            f"    {mat}, {vec}, {mat}, {out}, {k_ref}, {n_ref}, {k_ref}, {unit_ref}, {unit_ref},"
            f" {k_ref}, {tgp_expr},\n"
            f"    uint3(uint({logical}), 0u, 0u),\n"
            f"    uint3(thread_index_in_simdgroup, uint({lsg}) % {bn}u, uint({lsg}) / {bn}u),\n"
            f"    uint({lsg}), thread_index_in_simdgroup); }}")
    if kind == "wide":
        k_lanes, grid_x = tiles
        vecs = _wide_vecs(rows)
        return (
            f"{{ /* MLX gemv_wide: GemvWide nv{vecs} kl{k_lanes} (grid_x {grid_x} run as one "
            f"column), threadgroup (32, {k_lanes // 8}, 1) */\n"
            f"  GemvWide<{dtype}, {vecs}, {k_lanes}>::run(\n"
            f"    {mat}, {vec}, {mat}, {out}, int({k_ref}), int({n_ref}), {rows}, int({k_ref}),"
            f" int({k_ref}), 1.0f, 0.0f, 0, 0,\n"
            f"    uint3(0u, uint({logical}), 0u), uint3(1u, (uint({n_ref}) + 3u) / 4u, 1u),\n"
            f"    uint({lsg}), thread_index_in_simdgroup); }}")
    raise ValueError(f"unknown product kind {kind!r}")


# ----- bit comparison --------------------------------------------------------

def bits(a):
    """An unsigned integer view of ``a`` with the same element width."""
    return a.view(_UINT_VIEW[a.dtype.size])


def same_bits(pairs):
    """None when every ``(a, b)`` pair is bitwise equal (shape and dtype
    included), else the first difference: ``missing<i>``, ``shape<i>``,
    ``dtype<i>`` or ``differs<i>``."""
    pairs = list(pairs)
    for i, (a, b) in enumerate(pairs):
        if a is None or b is None:
            return f"missing{i}"
        if tuple(a.shape) != tuple(b.shape):
            return f"shape{i}"
        if a.dtype != b.dtype:
            return f"dtype{i}"
    mx.eval([x for pair in pairs for x in pair])
    for i, (a, b) in enumerate(pairs):
        if not mx.array_equal(bits(a), bits(b)).item():
            return f"differs{i}"
    return None


# ----- per-shape proofs (spec 5) ---------------------------------------------

PROVEN, REFUSED = "proven", "refused"
_registry = weakref.WeakValueDictionary()
_count_lock = threading.Lock()


def count(owner, attr, key, n=1):
    """Add ``n`` to ``owner.<attr>[key]`` (a plain dict in the instance dict,
    so ``nn.Module`` owners keep it out of their parameters)."""
    with _count_lock:
        counts = owner.__dict__.setdefault(attr, {})
        counts[key] = counts.get(key, 0) + n


class Proofs:
    """Verdicts for one launch family, per key, in process memory.

    A key is everything that selects code or tiles for the launch (shape,
    tiles, dtypes, operand sharing); the process-wide MLX version and header
    hash are fixed for a process, so nothing is persisted and a restart
    proves again."""

    def __init__(self, family):
        self.family = family
        self._lock = threading.Lock()
        self._keys = {}
        _registry[f"{family}#{id(self)}"] = self

    def reset(self):
        with self._lock:
            self._keys = {}

    def state(self, key):
        with self._lock:
            entry = self._keys.get(key)
            return None if entry is None else entry["state"]

    def _entry(self, key):
        return self._keys.setdefault(key, {"state": None, "reason": None, "proved": 0,
                                           "served": 0, "refused_calls": 0})

    def settle(self, key, reason, required=1):
        """Record a proof verdict (``reason`` None: bitwise). A key is proven
        after ``required`` passing proofs; any failing verdict refuses it,
        including a later one for a proven key (the first refusal wins).
        Returns the key's state afterwards."""
        with self._lock:
            entry = self._entry(key)
            if reason is None:
                entry["proved"] += 1
                if entry["state"] is None and entry["proved"] >= required:
                    entry["state"] = PROVEN
            elif entry["state"] != REFUSED:
                entry["state"], entry["reason"] = REFUSED, reason
            return entry["state"]

    def note(self, key, field):
        with self._lock:
            self._entry(key)[field] += 1

    def receipt(self):
        with self._lock:
            refused = {repr(k): e["reason"] for k, e in self._keys.items()
                       if e["state"] == REFUSED}
            return {"family": self.family, "mlx_version": mlx_version(),
                    "proved": sum(1 for e in self._keys.values() if e["state"] == PROVEN),
                    "refused": refused,
                    "served": sum(e["served"] for e in self._keys.values()),
                    "refused_calls": sum(e["refused_calls"] for e in self._keys.values())}


def receipt():
    """Library state and every live family's proof summary."""
    families = {}
    for proofs in list(_registry.values()):
        r = proofs.receipt()
        held = families.setdefault(proofs.family, {"proved": 0, "refused": {}, "served": 0,
                                                    "refused_calls": 0})
        held["proved"] += r["proved"]
        held["refused"].update(r["refused"])
        held["served"] += r["served"]
        held["refused_calls"] += r["refused_calls"]
    return {"library": library_receipt(), "proofs": families}


def reset_proofs():
    """Forget every family's verdicts (tests and A/B arms that need each
    model's counters to show its own proofs).

    Verdicts are process-wide while counters are per owner: a module that
    serves a key another module (or an earlier model in the process) already
    proved counts only ``calls``. That is truthful (this module did not prove
    it); a test that wants ``proved`` on a fresh model resets first."""
    for proofs in list(_registry.values()):
        proofs.reset()


class Planned:
    """A launch ready to run: ``fused()`` (the launch), ``reference()`` (the
    composed MLX path it must equal) and ``pairs(fused_out, reference_out)``
    (the array pairs compared bitwise). Callers may attach attributes."""

    def __init__(self, key, fused, reference, pairs):
        self.key = key
        self.fused = fused
        self.reference = reference
        self.pairs = pairs


def run_proved(proofs, key, owner, counts, fused, reference, pairs):
    """``(output, engaged)``.

    - component off: the reference, counted ``unproven``;
    - refused key: the reference, counted ``refused``;
    - proven key: the launch, counted ``calls`` (engaged);
    - first use: the launch and the reference both run on the call's operands
      and are compared bit for bit; the reference's output is returned (it may
      have side effects the caller relies on, and it equals the launch when
      the proof passes), counted ``proved`` or ``refused_first``.

    ``fused`` runs before ``reference`` on a first use."""
    if not enabled():
        count(owner, counts, "unproven")
        return reference(), False
    state = proofs.state(key)
    if state == REFUSED:
        count(owner, counts, "refused")
        proofs.note(key, "refused_calls")
        return reference(), False
    if state == PROVEN:
        try:
            out = fused()
        except Exception as exc:  # graph construction failed: refuse, serve the reference
            proofs.settle(key, f"error:{type(exc).__name__}")
            count(owner, counts, "refused")
            return reference(), False
        count(owner, counts, "calls")
        proofs.note(key, "served")
        return out, True
    reason = None
    try:
        launched = fused()
    except Exception as exc:
        launched, reason = None, f"error:{type(exc).__name__}"
    out = reference()
    if reason is None:
        try:
            diff = same_bits(pairs(launched, out))
        except Exception as exc:
            diff = f"error:{type(exc).__name__}"
        if diff is not None:
            reason = diff if diff.startswith("error") else f"bits:{diff}"
    state = proofs.settle(key, reason, proofs_required())
    count(owner, counts, "refused_first" if state == REFUSED else "proved")
    return out, False


def run_planned(proofs, planned, owner, counts):
    """``run_proved`` for a ``Planned``."""
    return run_proved(proofs, planned.key, owner, counts, planned.fused, planned.reference,
                      planned.pairs)


# ----- a single product as a launch (tests, A/B probes) -----------------------

MATVEC_PROOFS = Proofs("matvec")
_matvec_kernels = {}


def matvec_source(kind, tiles, dtype, rows, n):
    """``(body, launched threadgroups)`` of a launch computing
    ``y[rows, n] = x[rows, K] @ w[n, K]^T`` with inputs ``x, w, sizes = [K, n],
    unit = [1.0]``."""
    t = metal_type(dtype)
    logical, per_group, threads = logical_groups(kind, tiles, n)
    if per_group < 1:
        raise ValueError("MLX threadgroup larger than the launch threadgroup")
    sgs = threads // 32
    call = matvec_call(kind, tiles, dtype=t, rows=rows, mat="w", vec="x", out="y",
                       n_ref="sizes[1]", logical="logical", lsg=f"sg % {sgs}")
    body = (
        "const uint sg = simdgroup_index_in_threadgroup;\n"
        f"threadgroup float tgp[{tgp_floats(kind, tiles)}];\n"
        f"const uint logical = threadgroup_position_in_grid.x * {per_group} + sg / {sgs};\n"
        f"if (logical < {logical}) {call}\n")
    return body, -(-logical // per_group)


def plan_matvec(x, weight, tiles=None):
    """A ``Planned`` for ``x @ weight.T`` (x ``[..., K]``, weight ``[N, K]``),
    or the decline reason. ``tiles`` overrides the planned tiles (tests use it
    to inject a mismatch; the proof must then refuse the key)."""
    if mx.default_device() != mx.gpu:
        return "not_gpu"
    dtype = x.dtype
    if dtype not in _METAL_TYPES or weight.dtype != dtype:
        return "dtype"
    if weight.ndim != 2 or x.shape[-1] != weight.shape[1]:
        return "geometry"
    k, n = weight.shape[1], weight.shape[0]
    rows = 1
    for d in x.shape[:-1]:
        rows *= d
    kind, planned_tiles = matvec_plan(rows, n, k, dtype)
    if kind is None:
        return planned_tiles
    tiles = planned_tiles if tiles is None else tuple(tiles)
    if logical_groups(kind, tiles, n)[1] < 1:
        return "threadgroup_limit"
    header, why = library("matvec")
    if header is None:
        return why
    key = (kind, tiles, str(dtype), rows, n)
    with _lock:
        held = _matvec_kernels.get(key)
        if held is None:
            body, groups = matvec_source(kind, tiles, dtype, rows, n)
            kernel = mx.fast.metal_kernel(
                name=f"mlx2_library_matvec_{len(_matvec_kernels)}",
                input_names=["x", "w", "sizes", "unit"], output_names=["y"],
                header=header, source=body)
            held = _matvec_kernels[key] = (kernel, groups)
    kernel, groups = held
    lead = x.shape[:-1]

    def fused():
        return kernel(
            inputs=[x.reshape(rows, k), weight, mx.array([k, n], mx.int32),
                    mx.array([1.0], mx.float32)],
            grid=(groups * THREADS, 1, 1), threadgroup=(THREADS, 1, 1),
            output_shapes=[(rows, n)], output_dtypes=[dtype])[0].reshape(*lead, n)

    # The composed path: one MLX product (reference expression, spec R5).
    return Planned(key + (k,), fused, lambda: x @ weight.T, lambda a, b: [(a, b)])


def matvec(x, weight, owner, counts="matvec_counts"):
    """``x @ weight.T`` through a proven library launch (or the composed product)."""
    planned = plan_matvec(x, weight)
    if isinstance(planned, str):
        count(owner, counts, "declined:" + planned)
        return x @ weight.T
    out, _engaged = run_planned(MATVEC_PROOFS, planned, owner, counts)
    return out
