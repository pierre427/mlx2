"""Upstream ExLlamaV3 (EXL3) checkpoint layout: admission plan and loaders.

Format reference: turboderp-org/exllamav3 bd1f519 (MIT); names and config
schema only, no code copied. See provenance/exllamav3-format-20261104.json.
Tile decoding stays in runtime/models/exl3_reference.py. Unqualified format.

A plan reads safetensors headers only. Every trellis module must carry its
complete tensor set; anything unrecognised, legacy or inconsistent refuses
before a tensor payload is touched.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

QUANT_METHOD = "exl3"
FORMAT_REVISION = "mlx2-exl3-upstream-v1"
# Marker tensor name -> the multiplier quantization_config.json records.
CODEBOOKS = {"mcg": 0xCBAC1FED, "mul1": 0x83DCD12D}
# Codebooks the runtime modules execute: the Metal kernels (header_mcg_w16)
# and the CPU path (decode_inner's default) decode MCG only. MUL1 stays
# admitted for the host decode oracle but refuses at install.
RUNTIME_CODEBOOKS = ("mcg",)
# Tile word counts with a bit-exact exllamav3 fixture
# (scripts/sushi_probe/fixtures); other counts rest on the real-tensor oracle.
FIXTURE_WORDS = {"mcg": (40, 48), "mul1": (32, 48, 64)}
_PARTS = ("trellis", "suh", "svh")
_MARKERS = tuple(CODEBOOKS)
_LEGACY = ("su", "sv")
_SUFFIXES = (*_PARTS, *_MARKERS, *_LEGACY, "bias")
_EXPERT = re.compile(
    r"^(?P<root>.+)\.experts\.(?P<expert>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)$"
)


@dataclass(frozen=True)
class TrellisLinearSpec:
    """One stored EXL3 linear: weight [input, output] as 16x16 trellis tiles."""

    key: str
    input_dims: int
    output_dims: int
    words: int
    codebook: str
    bias: bool

    @property
    def bits_per_weight(self):
        return self.words / 16


@dataclass(frozen=True)
class TrellisBankSpec:
    """Per-expert EXL3 linears stored separately, stacked at load."""

    root: str
    projection: str
    experts: int
    input_dims: int
    output_dims: int
    words: int
    codebook: str

    @property
    def key(self):
        return f"{self.root}.{self.projection}"

    def expert_key(self, expert):
        return f"{self.root}.experts.{expert}.{self.projection}"

    def expert_spec(self, expert):
        if not 0 <= expert < self.experts:
            raise ValueError(f"expert {expert} outside bank {self.key}")
        return TrellisLinearSpec(
            self.expert_key(expert),
            self.input_dims,
            self.output_dims,
            self.words,
            self.codebook,
            False,
        )

    @property
    def bits_per_weight(self):
        return self.words / 16


@dataclass(frozen=True)
class Exl3Plan:
    format: dict
    linears: dict
    banks: dict
    stored: frozenset

    def receipt(self):
        rates = {}
        for spec in (*self.linears.values(), *self.banks.values()):
            name = f"{spec.codebook}/k{spec.bits_per_weight:g}"
            rates[name] = rates.get(name, 0) + 1
        return {
            "format_revision": FORMAT_REVISION,
            "quantization_config": dict(self.format),
            "linears": len(self.linears),
            "expert_banks": len(self.banks),
            "stored_tensors": len(self.stored),
            "rates": dict(sorted(rates.items())),
            "unfixtured_words": sorted(
                {
                    (s.codebook, s.words)
                    for s in (*self.linears.values(), *self.banks.values())
                    if s.words not in FIXTURE_WORDS[s.codebook]
                }
            ),
            "qualification": "unqualified",
        }


def is_exl3(config):
    quant = config.get("quantization_config")
    return isinstance(quant, dict) and quant.get("quant_method") == QUANT_METHOD


def parse_format(config):
    """Validate ``quantization_config``; return the fields mlx2 binds."""
    quant = config.get("quantization_config")
    if not is_exl3(config):
        raise ValueError("not an EXL3 checkpoint")
    if config.get("quantization") is not None:
        raise ValueError("EXL3 checkpoint also declares MLX quantization")
    bits, head = quant.get("bits"), quant.get("head_bits")
    if (
        type(bits) not in (int, float)
        or not 1 <= bits <= 8
        or type(head) not in (int, float)
        or not 1 <= head <= 16
    ):
        raise ValueError("EXL3 quantization_config needs numeric bits and head_bits")
    codebook = quant.get("codebook")
    if codebook is not None and codebook not in CODEBOOKS:
        raise ValueError(f"EXL3 codebook {codebook!r} is not implemented")
    if quant.get("out_scales", "always") not in ("always", "never", "auto"):
        raise ValueError("unknown EXL3 out_scales")
    return {
        "quant_method": QUANT_METHOD,
        "version": str(quant.get("version", "")),
        "bits": bits,
        "head_bits": head,
        "codebook": codebook,
        "out_scales": quant.get("out_scales", "always"),
    }


def read_storage(path):
    """Optional ``quantization_config.json`` ``tensor_storage`` map."""
    item = Path(path) / "quantization_config.json"
    if not item.is_file():
        return None
    storage = json.loads(item.read_text()).get("tensor_storage")
    if not isinstance(storage, dict):
        raise ValueError("quantization_config.json has no tensor_storage map")
    return storage


def _shape(entry):
    return tuple(entry["shape"])


def _linear(base, parts, storage):
    trellis, suh, svh = (parts.get(name) for name in _PARTS)
    if trellis is None or suh is None or svh is None:
        raise ValueError(f"incomplete EXL3 module: {base}")
    if any(name in parts for name in _LEGACY):
        raise ValueError(f"legacy packed su/sv signs are not implemented: {base}")
    markers = [name for name in _MARKERS if name in parts]
    if not markers:
        # 3INST has no bit-exact fixture here; refuse rather than guess.
        raise ValueError(f"EXL3 3INST codebook is not implemented: {base}")
    if len(markers) > 1:
        raise ValueError(f"EXL3 module names two codebooks: {base}")
    codebook = markers[0]
    marker = parts[codebook]
    if marker["dtype"] not in ("I32", "U32") or _shape(marker) not in ((), (1,)):
        raise ValueError(f"EXL3 codebook marker must be a 32-bit scalar: {base}")
    shape = _shape(trellis)
    if trellis["dtype"] not in ("I16", "U16") or len(shape) != 3:
        raise ValueError(f"EXL3 trellis must be 16-bit [I/16, O/16, n]: {base}")
    words = shape[2]
    if words % 2 or not 16 <= words <= 128:
        raise ValueError(f"EXL3 tile word count {words} is outside 16..128: {base}")
    input_dims, output_dims = shape[0] * 16, shape[1] * 16
    if input_dims % 128 or output_dims % 128:
        raise ValueError(f"EXL3 dimensions must be multiples of 128: {base}")
    if suh["dtype"] != "F16" or _shape(suh) != (input_dims,):
        raise ValueError(f"EXL3 suh must be F16 [{input_dims}]: {base}")
    if svh["dtype"] != "F16" or _shape(svh) != (output_dims,):
        raise ValueError(f"EXL3 svh must be F16 [{output_dims}]: {base}")
    bias = parts.get("bias")
    if bias is not None and (bias["dtype"] != "F16" or _shape(bias) != (output_dims,)):
        raise ValueError(f"EXL3 bias must be F16 [{output_dims}]: {base}")
    entry = (storage or {}).get(base)
    if entry is not None:
        recorded = entry.get("bits_per_weight")
        multiplier = entry.get(f"{codebook}_multiplier")
        if recorded is not None and recorded != words / 16:
            raise ValueError(f"EXL3 tensor_storage bits disagree with trellis: {base}")
        if multiplier is not None and multiplier != CODEBOOKS[codebook]:
            raise ValueError(f"EXL3 {codebook} multiplier is not the implemented one: {base}")
        if entry.get("quant_format", QUANT_METHOD) != QUANT_METHOD:
            raise ValueError(f"tensor_storage format is not EXL3: {base}")
    return TrellisLinearSpec(
        base, input_dims, output_dims, words, codebook, bias is not None
    )


def plan_checkpoint(path, weight_map, *, num_experts=None, storage=None):
    """Admit a whole upstream EXL3 checkpoint from its headers."""
    from ..weight_stream import read_safetensors_header

    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    form = parse_format(config)
    headers = {}
    for shard in sorted(set(weight_map.values())):
        header = read_safetensors_header(path / shard)
        header.pop("__data_offset__", None)
        headers[shard] = header
    entries = {}
    for name, shard in weight_map.items():
        entry = headers[shard].get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"indexed tensor missing from its shard: {name}")
        entries[name] = entry
    modules, stored = {}, set()
    for name, entry in entries.items():
        base, _, suffix = name.rpartition(".")
        if suffix in _SUFFIXES and suffix != "bias":
            modules.setdefault(base, {})[suffix] = entry
        elif suffix == "bias" and f"{base}.trellis" in entries:
            modules.setdefault(base, {})["bias"] = entry
        else:
            stored.add(name)
    linears, grouped = {}, {}
    for base, parts in sorted(modules.items()):
        spec = _linear(base, parts, storage)
        if form["codebook"] is not None and spec.codebook != form["codebook"]:
            raise ValueError(f"EXL3 marker disagrees with the declared codebook: {base}")
        match = _EXPERT.match(base)
        if match is None:
            linears[base] = spec
            continue
        if spec.bias:
            raise ValueError(f"EXL3 expert projections cannot carry a bias: {base}")
        key = (match["root"], match["proj"])
        grouped.setdefault(key, {})[int(match["expert"])] = spec
    banks = {}
    for (root, proj), experts in sorted(grouped.items()):
        count = len(experts)
        if sorted(experts) != list(range(count)):
            raise ValueError(f"EXL3 expert indices are not contiguous: {root}.{proj}")
        if num_experts is not None and count != num_experts:
            raise ValueError(
                f"EXL3 bank {root}.{proj} has {count} experts, config says {num_experts}"
            )
        layout = {
            (s.input_dims, s.output_dims, s.words, s.codebook) for s in experts.values()
        }
        if len(layout) != 1:
            raise ValueError(f"EXL3 experts of one projection differ in layout: {root}.{proj}")
        ((i, o, words, codebook),) = layout
        bank = TrellisBankSpec(root, proj, count, i, o, words, codebook)
        banks[bank.key] = bank
    if not linears and not banks:
        raise ValueError("EXL3 checkpoint has no trellis tensors")
    return Exl3Plan(form, linears, banks, frozenset(stored))


class ShardReader:
    """Lazy per-shard ``mx.load`` keyed by tensor name."""

    def __init__(self, path, weight_map):
        self.path, self.weight_map, self._open = Path(path), weight_map, {}

    def __call__(self, name):
        import mlx.core as mx

        shard = self.weight_map[name]
        if shard not in self._open:
            self._open[shard] = mx.load(str(self.path / shard))
        return self._open[shard][name]


def _trellis(array):
    import mlx.core as mx

    return array.view(mx.uint16) if array.dtype == mx.int16 else array


def load_linear(read, spec):
    """``trellis`` U16 [I/16, O/16, n], ``suh``/``svh`` F16, optional bias."""
    out = {
        "trellis": _trellis(read(f"{spec.key}.trellis")),
        "suh": read(f"{spec.key}.suh"),
        "svh": read(f"{spec.key}.svh"),
    }
    if spec.bias:
        out["bias"] = read(f"{spec.key}.bias")
    return out


def load_bank(read, spec):
    """Stack one projection's experts: U16 [E, I/16, O/16, n], F16 [E, I]/[E, O]."""
    import mlx.core as mx

    parts = [load_linear(read, spec.expert_spec(e)) for e in range(spec.experts)]
    return {name: mx.stack([p[name] for p in parts]) for name in _PARTS}


def load_stored(read, plan):
    """Every non-trellis tensor, as stored (embeddings, norms, routers, ...)."""
    return {name: read(name) for name in sorted(plan.stored)}


def install_upstream(model, read, plan, module_path, *, execution="reference", fused_mlp=None,
                     projection_groups=None):
    """Swap every planned trellis module into ``model``; return the receipt.

    ``module_path`` maps a checkpoint module name to the model's module path
    (adapter-owned naming). A dense module must replace a plain ``nn.Linear``
    of the same geometry; a bank replaces the routed switch at
    ``<root>.switch_mlp``. Everything is admitted before anything is swapped.
    ``fused_mlp(path, module)`` (adapter-owned) names the SwiGLU MLPs that
    ``nax_fused`` may run as fused chains; without it none is fused.
    ``execution`` is ``reference`` (per-row Sushi kernels), ``rows`` (the
    row-blocked law) or ``fused`` (one-launch dense linears, expert banks on
    the row-blocked law; Sushi's fused pair is validated at 40-word tiles
    only). One law per module kind at every width keeps rows invariant. It is
    a numerics law and part of the receipt.
    """
    if execution not in ("reference", "rows", "fused", "fused_nax", "fused_w8a8", "nax", "nax_fused"):
        raise ValueError(
            "upstream EXL3 execution must be reference, rows, fused, fused_nax, fused_w8a8, nax "
            "or nax_fused"
        )
    unsupported = sorted({s.codebook for s in (*plan.linears.values(), *plan.banks.values())}
                         - set(RUNTIME_CODEBOOKS))
    if unsupported:
        # The trellis modules carry no codebook: they would decode MCG.
        raise ValueError(f"EXL3 codebook {', '.join(unsupported)} has no runtime decoder "
                         f"(the trellis kernels decode {', '.join(RUNTIME_CODEBOOKS)} only)")
    if execution == "fused_w8a8":
        from ..int8_prefill import require_supported_device

        require_supported_device()  # Metal 4 tensor ops, or refuse
    import mlx.nn as nn

    from ..models.exl3 import (
        NAX_BANK_MIN_ROWS, NAX_MIN_ROWS, ROWS_LAW_REVISION, Exl3DenseLinear, install_banks,
    )
    from ..weight_stream import _swap_module

    modules = dict(model.named_modules())
    dense = []
    for key, spec in sorted(plan.linears.items()):
        path = module_path(key)
        old = modules.get(path)
        if type(old) is not nn.Linear:
            raise ValueError(f"EXL3 linear {key} has no plain Linear at {path}")
        out_dims, in_dims = old.weight.shape
        if (in_dims, out_dims) != (spec.input_dims, spec.output_dims):
            raise ValueError(f"EXL3 linear {key} disagrees with model geometry at {path}")
        if ("bias" in old) != spec.bias:
            raise ValueError(f"EXL3 linear {key} bias disagrees with {path}")
        tensors = load_linear(read, spec)
        law = "fused_nax" if execution in ("nax", "nax_fused") else execution
        dense.append((path, Exl3DenseLinear(**tensors, law=law)))
    banks = {}
    for spec in plan.banks.values():
        root = module_path(spec.root) + ".switch_mlp"
        for part, value in load_bank(read, spec).items():
            banks[f"{root}.{spec.projection}.{part}"] = value
    bank_execution = {"fused": "rows", "fused_nax": "rows", "fused_w8a8": "rows",
                      "nax": "rows_nax", "nax_fused": "fused_chain_nax"}.get(execution, execution)
    switches = install_banks(model, banks, execution=bank_execution) if banks else []
    for path, module in dense:
        _swap_module(model, path, module)
    fused_mlps = []
    if execution == "nax_fused" and fused_mlp is not None:
        # Dense SwiGLU MLPs (as the adapter names them) whose three
        # projections are bias-free trellis linears run decode widths as two
        # launches.
        from ..models.exl3 import Exl3FusedMLP

        installed = dict(model.named_modules())
        swapped = {path for path, _ in dense}
        for path in sorted(swapped):
            if not path.endswith(".gate_proj"):
                continue
            root = path[: -len(".gate_proj")]
            parts = [installed[root + p] for p in (".gate_proj", ".up_proj")]
            if ({root + ".up_proj", root + ".down_proj"} <= swapped
                    and fused_mlp(root, installed[root])
                    and not any("bias" in m for m in parts)):
                fused_mlps.append(root)
                _swap_module(model, root, Exl3FusedMLP(
                    installed[root + ".gate_proj"], installed[root + ".up_proj"],
                    installed[root + ".down_proj"]))
    groups = []
    if execution == "nax_fused" and projection_groups is not None:
        # Same-input projections the adapter declares (attribute names on a
        # module) share one launch at decode widths.
        from ..models.exl3 import Exl3ProjectionGroup, group_refusal

        for path, module in model.named_modules():
            names = projection_groups(path, module)
            if not names:
                continue
            linears = {name: getattr(module, name, None) for name in names}
            if group_refusal(list(linears.values())) is None:
                object.__setattr__(module, "exl3_projection_group", Exl3ProjectionGroup(linears))
                groups.append(path)
    packed = sum(m.bank.packed_nbytes for _, m in dense) + sum(
        s.receipt()["packed_weight_bytes"] for s in switches
    )
    return {
        **plan.receipt(),
        "execution": execution,
        "bank_execution": bank_execution,
        **({"rows_law_revision": ROWS_LAW_REVISION} if bank_execution in ("rows", "rows_nax") else {}),
        **({"nax_min_rows": NAX_MIN_ROWS}
           if execution in ("fused_nax", "fused_w8a8", "nax", "nax_fused") else {}),
        **({"nax_bank_min_rows": NAX_BANK_MIN_ROWS}
           if bank_execution in ("rows_nax", "fused_chain_nax") else {}),
        "fidelity": "approximate" if execution == "fused_w8a8" else "exact_trellis",
        "installed_linears": len(dense),
        "installed_switches": len(switches),
        "packed_weight_bytes": packed,
        "fused_mlps": len(fused_mlps),
        "projection_groups": len(groups),
        "module_paths": sorted(
            [path for path, _ in dense]
            + [module_path(spec.root) + ".switch_mlp" for spec in plan.banks.values()]
        ),
    }


def load_beside(model, receipt, weights):
    """Load sanitized stored tensors around installed trellis modules.

    Coverage is exact: every parameter is either trellis-owned or stored,
    never both, and no stored tensor is left without a parameter.
    """
    from mlx.utils import tree_flatten

    roots = tuple(path + "." for path in receipt["module_paths"])
    params = {name for name, _ in tree_flatten(model.parameters())}
    trellis = {name for name in params if name.startswith(roots)}
    missing = sorted(params - trellis - set(weights))
    extra = sorted(set(weights) - params)
    doubled = sorted(trellis & set(weights))
    if missing or extra or doubled:
        raise ValueError(
            f"EXL3 checkpoint does not cover the model: missing {missing[:4]}, "
            f"unexpected {extra[:4]}, stored over trellis {doubled[:4]}"
        )
    model.load_weights(list(weights.items()), strict=False)


def cast_half_compute(weights, suffixes, target):
    """Cast the named F16 stored weights to the compute dtype.

    Upstream EXL3 keeps small gate projections in F16; against BF16
    activations MLX promotes their products to F32 and widens the residual
    stream. The cast rounds F16 to the compute dtype and is recorded.
    """
    import mlx.core as mx

    suffixes = tuple(suffixes)
    if target is None or target == mx.float16:
        return {"cast": 0}
    keys = sorted(k for k, v in weights.items() if v.dtype == mx.float16 and k.endswith(suffixes))
    for key in keys:
        weights[key] = weights[key].astype(target)
    return {"cast": len(keys), "from": "float16", "to": str(target).rsplit(".", 1)[-1],
            "suffixes": list(suffixes)}
