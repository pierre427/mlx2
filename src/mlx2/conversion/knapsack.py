"""Knapsack per-layer mixed-precision planning (multiple-choice knapsack).

Original mlx2 code.  The idea (one uniform format per transformer layer,
chosen by a multiple-choice knapsack under a byte budget) comes from an
external single-run result; see ``provenance/knapsack-mixed-precision.json``.

Given a model's module tree, a menu of MLX-native weight formats
(affine q2..q8 at a group size, ``mxfp4``, ``nvfp4``, ``mxfp8``) and a byte
budget, choose one format per layer minimising a sensitivity-weighted cost
subject to the total bytes of the planned modules.  Every planned module of
a layer gets the same format, so batched expert kernels never branch per
expert.

Pieces:

* :class:`Format` and :func:`parse_format`: a menu entry and its exact MLX
  storage bytes (packed codes plus scales, plus biases for affine).
* :func:`inventory`: the quantizable modules of a model, split into
  ``planned`` (under ``<layer_prefix><L>.``), ``fixed`` (the model's own
  ``quant_predicate`` asks for a specific format, e.g. an 8-bit router) and
  ``rest`` (outside the layers: embeddings, lm_head) which take one
  ``rest_format``.  Built from the model object, so the planner, exporter
  and loader see the same module paths.
* :func:`solve` (exact DP over discretised bytes) and :func:`brute_force`
  (reference for tests).
* scorers: :func:`weight_mse_costs` (weight reconstruction error),
  :func:`layer_kl_costs` (per-layer KL probe, from ``mlx2.layer_kl`` scans)
  and :func:`table_costs` (any externally computed table).  A scorer only
  has to produce ``{layer: {format_name: cost}}``.
* :func:`make_plan` / :func:`save_plan` / :func:`load_plan`: the plan
  artifact, and :func:`quantization_config` / :func:`plan_predicate` for
  export and load (mlx-lm style per-module overrides in
  ``config["quantization"]``).

Costs are assumed additive across layers.  That is a first-order
approximation (interactions are ignored); ``layer_kl.scan(cumulative=True)``
measures how far off it is.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np

PLAN_SCHEMA = "mlx2.knapsack-plan.v1"
COSTS_SCHEMA = "mlx2.knapsack-costs.v1"

# ------------------------------------------------------------------ formats --

_SCALE_BYTES = {"affine": 4, "mxfp4": 1, "nvfp4": 1, "mxfp8": 1}  # per group
_FIXED = {"mxfp4": (4, 32), "nvfp4": (4, 16), "mxfp8": (8, 32)}
_AFFINE_BITS = (2, 3, 4, 5, 6, 8)
_AFFINE_GROUPS = (32, 64, 128)


@dataclass(frozen=True)
class Format:
    """One MLX weight format: ``mode``, ``bits`` and ``group_size``."""

    mode: str
    bits: int
    group_size: int

    def __post_init__(self):
        if self.mode == "affine":
            if self.bits not in _AFFINE_BITS or self.group_size not in _AFFINE_GROUPS:
                raise ValueError(f"affine needs bits in {_AFFINE_BITS} and group in {_AFFINE_GROUPS}")
        elif self.mode in _FIXED:
            if (self.bits, self.group_size) != _FIXED[self.mode]:
                raise ValueError(f"{self.mode} is bits={_FIXED[self.mode][0]} group={_FIXED[self.mode][1]}")
        else:
            raise ValueError(f"unknown quantization mode {self.mode!r}")

    @property
    def name(self) -> str:
        return f"q{self.bits}g{self.group_size}" if self.mode == "affine" else self.mode

    @property
    def bpw(self) -> float:
        """Stored bits per weight, scales (and biases) included."""
        return self.bits + 8 * _SCALE_BYTES[self.mode] / self.group_size

    def fits(self, in_features: int) -> bool:
        return in_features % self.group_size == 0 and (in_features * self.bits) % 32 == 0

    def nbytes(self, shape: Sequence[int]) -> int:
        """Exact bytes MLX stores for a dense weight of ``shape`` (last axis = input)."""
        *outer, cols = (int(s) for s in shape)
        if not self.fits(cols):
            raise ValueError(f"{self.name} cannot quantize input width {cols}")
        rows = math.prod(outer) if outer else 1
        return rows * (cols * self.bits // 8 + (cols // self.group_size) * _SCALE_BYTES[self.mode])

    def quant(self) -> dict:
        """The ``to_quantized`` keyword dict (and the config override)."""
        return {"group_size": self.group_size, "bits": self.bits, "mode": self.mode}

    def as_dict(self) -> dict:
        return {"name": self.name, **self.quant(), "bpw": self.bpw}


def parse_format(text: str) -> Format:
    """``q3g64`` / ``affine-q3-g64`` / ``q4`` (group 64) / ``mxfp4`` / ``nvfp4`` / ``mxfp8``."""
    t = text.strip().lower()
    if t in _FIXED:
        return Format(t, *_FIXED[t])
    m = re.fullmatch(r"(?:affine[-_]?)?q(\d)(?:[-_]?g(\d+))?", t)
    if not m:
        raise ValueError(f"unrecognised format {text!r} (use q<bits>g<group>, mxfp4, nvfp4, mxfp8)")
    return Format("affine", int(m.group(1)), int(m.group(2) or 64))


def format_from_quant(quant: Mapping) -> Format:
    return Format(str(quant.get("mode", "affine")), int(quant["bits"]), int(quant["group_size"]))


def parse_menu(items: Iterable[str]) -> list[Format]:
    menu = []
    for item in items:
        for part in str(item).split(","):
            if part.strip():
                menu.append(parse_format(part))
    names = [f.name for f in menu]
    if len(set(names)) != len(names) or not menu:
        raise ValueError(f"menu must be non-empty without duplicates: {names}")
    return menu


# ---------------------------------------------------------------- inventory --

@dataclass(frozen=True)
class Unit:
    """One quantizable module and how the plan treats it."""

    path: str
    shape: tuple          # dense weight shape, last axis = input features
    role: str             # "planned" | "fixed" | "rest"
    layer: Optional[int] = None
    fixed: Optional[dict] = None   # the model's own override for role "fixed"

    @property
    def numel(self) -> int:
        return math.prod(self.shape)


def _dense_shape(module) -> tuple:
    weight = module["weight"]
    if "scales" in module:
        return (*weight.shape[:-1], int(module["scales"].shape[-1]) * int(module.group_size))
    return tuple(int(s) for s in weight.shape)


def inventory(model, *, layer_prefix: str = "model.layers.") -> list[Unit]:
    """The model's quantizable modules (``to_quantized`` or already quantized).

    The model's ``quant_predicate`` (when it has one) is honoured: ``False``
    keeps a module dense (not listed), a dict pins that module's format
    (role ``fixed``).  Everything else under ``<layer_prefix><L>.`` is
    ``planned`` for layer ``L``; other modules are ``rest``.
    """
    predicate = getattr(model, "quant_predicate", None)
    pattern = re.compile(re.escape(layer_prefix) + r"(\d+)\.")
    units = []
    for path, module in model.named_modules():
        quantized = isinstance(module, dict) and "scales" in module and "weight" in module
        if not (hasattr(module, "to_quantized") or quantized) or "weight" not in module:
            continue
        verdict = predicate(path, module) if callable(predicate) else True
        if verdict is False:
            continue
        shape = _dense_shape(module)
        m = pattern.match(path)
        if isinstance(verdict, dict):
            units.append(Unit(path, shape, "fixed", int(m.group(1)) if m else None,
                              {"mode": "affine", **verdict}))
        elif m:
            units.append(Unit(path, shape, "planned", int(m.group(1))))
        else:
            units.append(Unit(path, shape, "rest"))
    if not any(u.role == "planned" for u in units):
        raise ValueError(f"no quantizable module under {layer_prefix!r}<L>.")
    return units


def layers_of(units: Sequence[Unit]) -> list[int]:
    return sorted({u.layer for u in units if u.role == "planned"})


def layer_sizes(units: Sequence[Unit], menu: Sequence[Format]) -> dict[int, dict[str, Optional[int]]]:
    """``{layer: {format: bytes or None}}``; ``None`` when a module's width does not fit."""
    out = {}
    for layer in layers_of(units):
        mods = [u for u in units if u.role == "planned" and u.layer == layer]
        out[layer] = {f.name: (sum(f.nbytes(u.shape) for u in mods)
                               if all(f.fits(u.shape[-1]) for u in mods) else None) for f in menu}
    return out


def fixed_bytes(units: Sequence[Unit], rest: Optional[Format]) -> dict:
    """Bytes of the non-planned quantized modules (fixed by the model, or ``rest``)."""
    fixed = sum(format_from_quant(u.fixed).nbytes(u.shape) for u in units if u.role == "fixed")
    rest_units = [u for u in units if u.role == "rest"]
    rest_b = sum(rest.nbytes(u.shape) for u in rest_units) if rest is not None else sum(
        2 * u.numel for u in rest_units)  # dense bf16 when no rest format
    return {"fixed_bytes": int(fixed), "rest_bytes": int(rest_b)}


# ------------------------------------------------------------------- solver --

@dataclass
class Solution:
    choice: list[int]           # index into each layer's options
    cost: float
    nbytes: int
    resolution: int
    exact: bool                 # True when resolution == 1 (no discretisation slack)


def brute_force(costs: Sequence[Sequence[Optional[float]]], sizes: Sequence[Sequence[Optional[int]]],
                budget: int) -> Optional[Solution]:
    """Exhaustive reference (tests only; exponential)."""
    import itertools

    best = None
    for combo in itertools.product(*[range(len(c)) for c in costs]):
        if any(costs[i][j] is None or sizes[i][j] is None for i, j in enumerate(combo)):
            continue
        b = sum(sizes[i][j] for i, j in enumerate(combo))
        if b > budget:
            continue
        c = sum(costs[i][j] for i, j in enumerate(combo))
        if best is None or c < best.cost - 1e-12 or (abs(c - best.cost) <= 1e-12 and b < best.nbytes):
            best = Solution(list(combo), c, b, 1, True)
    return best


def solve(costs: Sequence[Sequence[Optional[float]]], sizes: Sequence[Sequence[Optional[int]]],
          budget: int, *, resolution: Optional[int] = None, max_cells: int = 200_000) -> Solution:
    """Multiple-choice knapsack: one option per row, sum(sizes) <= budget, min sum(costs).

    ``None`` marks a forbidden option (pins, minimum bits, unfit widths).
    Bytes are discretised to ``resolution``.  The default is the GCD of all
    option sizes when that keeps the DP within ``max_cells`` buckets: no
    rounding at all, so the result is exactly optimal (layer sizes of real
    models share large power-of-two factors; Mellum's menu has a 13 MB GCD).
    Otherwise it is the smallest unit within ``max_cells``, and each option's
    size is rounded *up*, so every plan is within the true budget but only
    optimal among plans whose rounded-up sizes fit (``exact`` False; a plan
    that fits the budget only just can be missed).  Ties go to the plan
    with fewer (rounded) bytes.
    """
    if len(costs) != len(sizes) or not costs:
        raise ValueError("costs and sizes need one non-empty row per layer")
    budget = int(budget)
    if budget < 0:
        raise ValueError("budget must be >= 0")
    if resolution is not None and (type(resolution) is not int or resolution < 1):
        raise ValueError("resolution must be a positive integer")
    if max_cells < 1:
        raise ValueError("max_cells must be positive")
    all_sizes, allowed_ix = [], []
    for row_c, row_s in zip(costs, sizes):
        if len(row_c) != len(row_s):
            raise ValueError("cost and size rows differ in length")
        options = []
        for j, (cost, size) in enumerate(zip(row_c, row_s)):
            if cost is None or size is None:
                continue
            if not math.isfinite(float(cost)):
                raise ValueError("costs must be finite (use None to forbid an option)")
            if int(size) != size or size < 0:
                raise ValueError("option sizes must be nonnegative integers")
            options.append(j)
            all_sizes.append(int(size))
        allowed_ix.append(options)
    if all(len(ix) == 1 for ix in allowed_ix):
        # One allowed option per row (a uniform plan): no search, and no rounding to lose a tight fit.
        choice = [ix[0] for ix in allowed_ix]
        nbytes = sum(int(sizes[i][j]) for i, j in enumerate(choice))
        if nbytes > budget:
            raise ValueError("no plan fits the budget with the allowed options")
        return Solution(choice, float(sum(float(costs[i][j]) for i, j in enumerate(choice))), nbytes, 1, True)
    if resolution is not None:
        q = int(resolution)
    else:
        g = math.gcd(*all_sizes) if all_sizes else 1
        q = g if g > 0 and budget // g <= max_cells else max(1, math.ceil(budget / max_cells))
    cells = budget // q
    inf = np.inf
    dp = np.full(cells + 1, inf)
    dp[0] = 0.0
    picks = []
    for row_c, row_s, options in zip(costs, sizes, allowed_ix):
        new = np.full(cells + 1, inf)
        pick = np.full(cells + 1, -1, dtype=np.int16)
        for j in options:
            c = row_c[j]
            w = -(-int(row_s[j]) // q)
            if w > cells:
                continue
            cand = np.full(cells + 1, inf)
            cand[w:] = dp[:cells + 1 - w] + float(c)
            better = cand < new
            new[better] = cand[better]
            pick[better] = j
        if not np.isfinite(new).any():
            raise ValueError("no plan fits the budget with the allowed options")
        picks.append(pick)
        dp = new
    end = int(np.argmin(dp))     # first minimum: fewest buckets among equal costs
    if not math.isfinite(dp[end]):
        raise ValueError("no plan fits the budget with the allowed options")
    choice, b = [], end
    for i in range(len(costs) - 1, -1, -1):
        j = int(picks[i][b])
        choice.append(j)
        b -= -(-int(sizes[i][j]) // q)
    choice.reverse()
    true_bytes = sum(int(sizes[i][j]) for i, j in enumerate(choice))
    true_cost = float(sum(float(costs[i][j]) for i, j in enumerate(choice)))
    assert true_bytes <= budget
    return Solution(choice, true_cost, true_bytes, q, all(x % q == 0 for x in all_sizes))


# -------------------------------------------------------------- constraints --

def parse_pins(items: Iterable[str]) -> dict[int, dict]:
    """``0:>=3`` (layer 0 at least 3 nominal bits), ``0=q4g64`` (exact format),
    ``0-2:>=4`` (a range)."""
    pins = {}
    for item in items:
        for part in str(item).split(","):
            part = part.strip()
            if not part:
                continue
            m = re.fullmatch(r"(\d+)(?:-(\d+))?\s*(?::\s*>=\s*(\d+(?:\.\d+)?)|=\s*(\S+))", part)
            if not m:
                raise ValueError(f"bad pin {part!r}: use L:>=BITS, L=FORMAT or A-B:>=BITS")
            lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
            if hi < lo:
                raise ValueError(f"bad pin {part!r}: range end is before its start")
            for layer in range(lo, hi + 1):
                rule = pins.setdefault(layer, {})
                if m.group(3):
                    rule["min_bits"] = max(float(m.group(3)), rule.get("min_bits", 0.0))
                else:
                    rule["format"] = parse_format(m.group(4)).name
    return pins


def allowed(fmt: Format, layer: int, *, pins: Mapping[int, dict], min_bits: float) -> bool:
    rule = pins.get(layer, {})
    if "format" in rule and rule["format"] != fmt.name:
        return False
    return fmt.bits >= max(min_bits, rule.get("min_bits", 0.0))


# ------------------------------------------------------------------ scorers --

def _costs_doc(scorer: str, settings: dict, checkpoint: str, costs: dict) -> dict:
    return {"schema": COSTS_SCHEMA, "scorer": scorer, "settings": settings, "checkpoint": checkpoint,
            "costs": {str(k): v for k, v in sorted(costs.items())}}


def weight_mse_costs(model, units: Sequence[Unit], menu: Sequence[Format], *, relative: bool = False,
                     chunk: int = 16, log: Optional[Callable[[str], None]] = None) -> dict:
    """Weight-reconstruction cost: sum over a layer's planned modules of ``||W - Q(W)||^2``.

    MLX's own round-to-nearest ``mx.quantize``/``mx.dequantize`` on the
    current device; expert banks ``[E, O, I]`` go ``chunk`` experts at a
    time.  ``relative=True`` divides each module's error by ``||W||^2``.
    Raw MSE ignores what the layer's output does with the error, which is
    why it can undervalue an early layer; prefer :func:`layer_kl_costs`.
    """
    import mlx.core as mx

    modules = dict(model.named_modules())
    out: dict[int, dict[str, Optional[float]]] = {}
    for layer in layers_of(units):
        row = {f.name: 0.0 for f in menu}
        for u in (u for u in units if u.role == "planned" and u.layer == layer):
            module = modules[u.path]
            if "scales" in module:
                raise ValueError(f"{u.path} is already quantized; score from the dense checkpoint")
            weight = module["weight"]
            bank = weight if weight.ndim == 3 else weight[None]
            norm = 0.0
            errs = {f.name: 0.0 for f in menu}
            for start in range(0, bank.shape[0], chunk):
                # Quantize in the weight's own dtype (bf16 scales), as the export's
                # ``to_quantized`` does; only the error is computed in fp32.
                raw = bank[start:start + chunk]
                w = raw.astype(mx.float32)
                if relative:
                    norm += float(mx.sum(w * w).item())
                for f in menu:
                    if not f.fits(u.shape[-1]):
                        errs[f.name] = None
                        continue
                    planes = mx.quantize(raw, group_size=f.group_size, bits=f.bits, mode=f.mode)
                    deq = mx.dequantize(planes[0], planes[1], planes[2] if len(planes) > 2 else None,
                                        group_size=f.group_size, bits=f.bits, mode=f.mode)
                    errs[f.name] += float(mx.sum((deq.astype(mx.float32) - w) ** 2).item())
                    del planes, deq
                mx.clear_cache()
            for name, e in errs.items():
                if e is None or row[name] is None:
                    row[name] = None
                else:
                    row[name] += e / norm if relative and norm > 0 else e
        out[layer] = row
        if log:
            log(f"[mse layer {layer}] " + " ".join(f"{k}={v:.4g}" for k, v in row.items() if v is not None))
    return out


def layer_kl_costs(scans: Mapping[str, Mapping], *, checkpoint: Optional[str] = None,
                   units: Optional[Sequence[Unit]] = None, key: str = "delta_kl") -> dict:
    """Per-layer KL probe costs from ``mlx2.layer_kl`` scan results, one per format.

    ``scans`` maps a format name to a scan document (``mlx2.layer-kl.scan.v1``,
    non-cumulative, RTN, the quantizer the export uses).  Each scan's settings
    must match its format; all scans must share a reference and the same
    groups/splits/layer prefix/source; ``checkpoint`` (when given) must be both
    the scanned and the reference checkpoint; with ``units``, each row must
    have substituted exactly that layer's planned modules.  A row's
    ``delta_kl`` is that layer's cost.
    """
    from ..layer_kl import SCAN_SCHEMA

    planned = {}
    for u in units or ():
        if u.role == "planned":
            planned.setdefault(u.layer, set()).add(u.path)
    out: dict[int, dict[str, float]] = {}
    refs, shapes = set(), set()
    for name, doc in scans.items():
        fmt = parse_format(name)
        if doc.get("schema") != SCAN_SCHEMA:
            raise ValueError(f"{name}: not a layer_kl scan")
        s = doc["settings"]
        if (s["bits"], s["group_size"], s.get("mode", "affine")) != (fmt.bits, fmt.group_size, fmt.mode):
            raise ValueError(f"{name}: scan settings {s['bits']}/{s['group_size']}/{s.get('mode')} "
                             f"do not match the format")
        if s.get("cumulative"):
            raise ValueError(f"{name}: a cumulative scan is not a per-layer cost")
        if not doc.get("restore_check", {}).get("exact"):
            raise ValueError(f"{name}: scan restore check failed")
        if s.get("quantizer", "rtn") != "rtn":
            raise ValueError(f"{name}: a {s.get('quantizer')} scan does not cost the RTN export")
        if checkpoint is not None and "checkpoint" in doc and \
                fingerprint_of(doc["checkpoint"]) != fingerprint_of(checkpoint):
            raise ValueError(f"{name}: scanned {doc['checkpoint']}, not the planned checkpoint {checkpoint}")
        reference = doc["reference"]
        refs.add((reference["checkpoint"], reference["texts_sha256"], reference.get("skip")))
        shapes.add(json.dumps([sorted(s.get("groups", [])), sorted(s.get("splits", [])),
                               s.get("layer_prefix"), s.get("source")]))
        for row in doc["rows"]:
            layer = int(row["layer"])
            if layer in out and fmt.name in out[layer]:
                raise ValueError(f"{name}: layer {layer} appears twice")
            if units is not None and set(row.get("modules", [])) != planned.get(layer, set()):
                raise ValueError(f"{name}: layer {layer} scan substituted {sorted(row.get('modules', []))}, "
                                 f"not the planned modules {sorted(planned.get(layer, set()))}")
            out.setdefault(layer, {})[fmt.name] = float(row[key])
    if len(refs) != 1:
        raise ValueError(f"scans use different references: {sorted(refs, key=repr)}")
    if len(shapes) != 1:
        raise ValueError(f"scans differ in groups/splits/layer prefix/source: {sorted(shapes)}")
    ref_ckpt = next(iter(refs))[0]
    if checkpoint is not None and fingerprint_of(ref_ckpt) != fingerprint_of(checkpoint):
        raise ValueError(f"scan reference {ref_ckpt} is not the planned checkpoint {checkpoint}")
    return out


def fingerprint_of(checkpoint: str) -> str:
    """The fingerprint half of ``<name>@<fingerprint>`` (names differ by how a path was spelled)."""
    return checkpoint.rpartition("@")[2]


def checkpoint_name(path: Path) -> str:
    """Artifact name of a checkpoint directory; an HF snapshot is named by its repository."""
    path = Path(path).resolve()
    if re.fullmatch(r"[0-9a-f]{40}", path.name) and path.parent.name == "snapshots":
        repo = path.parent.parent.name
        if repo.startswith("models--"):
            return repo.split("--", 2)[-1]
    return path.name


def table_costs(doc: Mapping, *, checkpoint: Optional[str] = None) -> dict:
    """Costs from a ``mlx2.knapsack-costs.v1`` document (any scorer)."""
    if doc.get("schema") != COSTS_SCHEMA:
        raise ValueError("not a knapsack costs document")
    if checkpoint is not None and doc.get("checkpoint") != checkpoint:
        raise ValueError(f"costs are for {doc.get('checkpoint')}, not {checkpoint}")
    return {int(k): {n: (None if v is None else float(v)) for n, v in row.items()}
            for k, row in doc["costs"].items()}


# --------------------------------------------------------------------- plan --

def budget_bytes(spec: str, sizes: Mapping[int, Mapping[str, Optional[int]]], units: Sequence[Unit]) -> int:
    """``123456789`` (bytes), ``3.05bpw`` (over planned weights) or ``0.678xnvfp4``
    (a multiple of a uniform format's planned bytes)."""
    t = spec.strip().lower().replace(" ", "")
    planned = sum(u.numel for u in units if u.role == "planned")
    m = re.fullmatch(r"(\d+(?:\.\d+)?)bpw", t)
    if m:
        return int(math.floor(float(m.group(1)) * planned / 8))
    m = re.fullmatch(r"(\d+(?:\.\d+)?)x(.+)", t)
    if m:
        name = parse_format(m.group(2)).name
        per = [row.get(name) for row in sizes.values()]
        if any(p is None for p in per):
            raise ValueError(f"{name} does not fit every layer, or is not measured in this menu")
        return int(math.floor(float(m.group(1)) * sum(per)))
    if re.fullmatch(r"\d+", t):
        return int(t)
    raise ValueError(f"bad budget {spec!r}: use BYTES, <x>bpw or <r>x<format>")


def commit_receipt(root: Path) -> dict:
    import subprocess

    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain",
                                          "--untracked-files=no"], text=True).strip())
    return {"commit": head, "tracked_dirty": dirty}


def make_plan(*, checkpoint: str, units: Sequence[Unit], menu: Sequence[Format],
              costs: Mapping[int, Mapping[str, Optional[float]]], budget: str | int,
              scorer: dict, mlx2: dict, rest_format: Optional[Format],
              pins: Optional[Mapping[int, dict]] = None, min_bits: float = 0.0,
              layer_prefix: str = "model.layers.", resolution: Optional[int] = None) -> dict:
    """Solve and return the plan document (see ``PLAN_SCHEMA``)."""
    pins = dict(pins or {})
    sizes = layer_sizes(units, menu)
    layers = sorted(sizes)
    missing = [l for l in layers if l not in costs]
    if missing:
        raise ValueError(f"costs are missing layers {missing}")
    extra = sorted(set(costs) - set(layers))
    if extra:
        raise ValueError(f"costs name layers the model does not have: {extra}")
    unknown = sorted(set(pins) - set(layers))
    if unknown:
        raise ValueError(f"pins name layers the model does not have: {unknown}")
    menu_names = {f.name for f in menu}
    for layer, rule in pins.items():
        if "format" in rule and rule["format"] not in menu_names:
            raise ValueError(f"layer {layer} is pinned to {rule['format']}, which is not in the menu")
    budget_b = budget if isinstance(budget, int) else budget_bytes(str(budget), sizes, units)
    cost_rows, size_rows = [], []
    for layer in layers:
        cr, sr = [], []
        for f in menu:
            c = costs[layer].get(f.name)
            ok = allowed(f, layer, pins=pins, min_bits=min_bits) and sizes[layer][f.name] is not None
            cr.append(c if ok else None)
            sr.append(sizes[layer][f.name] if ok else None)
            if ok and c is None:
                raise ValueError(f"layer {layer} has no {f.name} cost but the format is allowed")
        if all(c is None for c in cr):
            raise ValueError(f"layer {layer}: no menu format satisfies its pins and minimum bits")
        cost_rows.append(cr)
        size_rows.append(sr)
    sol = solve(cost_rows, size_rows, budget_b, resolution=resolution)
    choices = [{"layer": layer, "format": menu[j].name, "bytes": int(size_rows[i][j]),
                "cost": float(cost_rows[i][j]),
                "allowed": [f.name for k, f in enumerate(menu) if cost_rows[i][k] is not None]}
               for i, (layer, j) in enumerate(zip(layers, sol.choice))]
    # Uniform baselines (same scorer) so a plan says what it beats on paper.
    # Uniform baselines ignore pins and minimum bits; each says whether it would satisfy them.
    uniform = {}
    for k, f in enumerate(menu):
        raw = [(sizes[l][f.name], costs[l].get(f.name)) for l in layers]
        if all(s is not None and c is not None for s, c in raw):
            uniform[f.name] = {"bytes": int(sum(s for s, _ in raw)),
                               "predicted_cost": float(sum(c for _, c in raw)),
                               "satisfies_constraints": all(size_rows[i][k] is not None
                                                            for i in range(len(layers)))}
    planned_numel = sum(u.numel for u in units if u.role == "planned")
    extra_b = fixed_bytes(units, rest_format)
    doc = {
        "schema": PLAN_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "checkpoint": checkpoint,
        "mlx2": dict(mlx2),
        "layer_prefix": layer_prefix,
        "menu": [f.as_dict() for f in menu],
        "scorer": dict(scorer),
        "budget": {"spec": str(budget), "bytes": int(budget_b)},
        "constraints": {"min_bits": float(min_bits),
                        "pins": {str(k): v for k, v in sorted(pins.items())}},
        "rest_format": rest_format.as_dict() if rest_format else None,
        "layers": choices,
        "fixed_modules": [{"path": u.path, **format_from_quant(u.fixed).quant()}
                          for u in units if u.role == "fixed"],
        "rest_modules": [u.path for u in units if u.role == "rest"],
        "solution": {"predicted_cost": sol.cost, "planned_bytes": sol.nbytes,
                     "planned_bpw": 8 * sol.nbytes / planned_numel, "planned_weights": planned_numel,
                     "resolution_bytes": sol.resolution, "exact": sol.exact,
                     **extra_b, "quantized_total_bytes": sol.nbytes + extra_b["fixed_bytes"] + extra_b["rest_bytes"],
                     "format_counts": dict(Counter(c["format"] for c in choices))},
        "uniform_baselines": uniform,
    }
    doc["plan_sha256"] = plan_digest(doc)
    return doc


def plan_digest(doc: Mapping) -> str:
    """Content hash of a plan (excluding ``created_at`` and the digest itself)."""
    body = {k: v for k, v in doc.items() if k not in ("plan_sha256", "created_at")}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_plan(doc: Mapping) -> dict:
    if doc.get("schema") != PLAN_SCHEMA:
        raise ValueError("not a knapsack plan")
    if doc.get("plan_sha256") != plan_digest(doc):
        raise ValueError("plan digest mismatch (edited after solving?)")
    menu = {m["name"]: format_from_quant(m) for m in doc["menu"]}
    total = 0
    seen = set()
    for row in doc["layers"]:
        if row["format"] not in menu or row["format"] not in row["allowed"]:
            raise ValueError(f"layer {row['layer']}: format {row['format']} not allowed")
        if row["layer"] in seen:
            raise ValueError(f"layer {row['layer']} appears twice")
        seen.add(row["layer"])
        total += int(row["bytes"])
    if total != doc["solution"]["planned_bytes"] or total > doc["budget"]["bytes"]:
        raise ValueError("plan bytes do not add up or exceed the budget")
    return dict(doc)


def save_plan(doc: Mapping, path: Path) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists")
    validate_plan(doc)
    path.write_text(json.dumps(doc, indent=1) + "\n")


def load_plan(path: Path) -> dict:
    return validate_plan(json.loads(Path(path).read_text()))


def layer_formats(doc: Mapping) -> dict[int, Format]:
    menu = {m["name"]: format_from_quant(m) for m in doc["menu"]}
    return {int(r["layer"]): menu[r["format"]] for r in doc["layers"]}


def plan_predicate(doc: Mapping, units: Sequence[Unit]) -> Callable[[str, Any], Any]:
    """``nn.quantize`` class predicate implementing a plan for this model's units.

    Refuses a plan whose planned layers do not match the model's.
    """
    formats = layer_formats(doc)
    if sorted(formats) != layers_of(units):
        raise ValueError("plan layers do not match the model's layers")
    rest = format_from_quant(doc["rest_format"]) if doc.get("rest_format") else None
    table = {}
    for u in units:
        if u.role == "planned":
            table[u.path] = formats[u.layer].quant()
        elif u.role == "fixed":
            table[u.path] = format_from_quant(u.fixed).quant()
        elif rest is not None:
            table[u.path] = rest.quant()

    def predicate(path, _module):
        q = table.get(path)
        return dict(q) if q is not None else False

    return predicate


def quantization_config(doc: Mapping, units: Sequence[Unit]) -> dict:
    """``config["quantization"]``: a default format plus one override per quantized module.

    mlx-lm convention (module path -> ``{group_size, bits, mode}``), which
    mlx2 adapters read through ``quant.get(name)``.  Every quantized module
    is listed explicitly so the default never decides a module's format.
    The default is the ``rest_format`` when there is one, else the
    highest-bit override (adapters validate the default, e.g. Mellum admits
    only 4/6/8 bits there, and would refuse a low planned width even though
    no module uses it).
    """
    pred = plan_predicate(doc, units)
    every = {u.path: pred(u.path, None) for u in units}
    overrides = {k: v for k, v in every.items() if v}
    # A unit left dense (no rest format) gets an explicit False, so no adapter's default quantizes it.
    dense = {k: False for k, v in every.items() if not v}
    if doc.get("rest_format"):
        default = format_from_quant(doc["rest_format"]).quant()
    else:
        default = max(overrides.values(), key=lambda q: (q["bits"], -q["group_size"], q["mode"]))
        default = dict(default)
    return {**default, **overrides, **dense}


def model_config(doc: Mapping) -> dict:
    """The provenance block written next to the quantization config."""
    return {"schema": doc["schema"], "plan_sha256": doc["plan_sha256"], "checkpoint": doc["checkpoint"],
            "scorer": doc["scorer"].get("name"), "budget_bytes": doc["budget"]["bytes"],
            "mlx2": doc["mlx2"]}
