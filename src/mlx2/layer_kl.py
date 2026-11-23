"""Per-layer KL sensitivity: which layers tolerate fewer bits.

Original mlx2 code.  Model-agnostic: nothing here names a model.  It needs
only a model whose ``__call__(ids)`` returns logits ``[1, T, V]`` and whose
quantizable modules sit under ``<layer_prefix><L>.<group>``; a reference too
big to load comes from a layer-streaming sweep family (interface v1,
``research/amoeba/sweep/README.md``: ``embed`` / ``run_layer`` /
``release_layer`` / ``final_logits``).

Two stages:

* **reference** (:func:`write_reference`, :func:`reference_from_model`,
  :func:`reference_from_family`): for a fixed set of token sequences, the
  reference model's full log-softmax at every scored position (positions
  ``skip .. T-1``), stored per text as ``ref-<i>.npy`` (fp32 by default),
  with ``reference.json`` naming the reference checkpoint, the texts' sha256
  and the scored positions.
* **scan** (:func:`scan`): one loaded variant model.  The baseline is scored
  first (KL(reference || variant), top-1 agreement, per text).  Then, for
  each requested layer, a module group is requantized to the target
  ``bits/group_size/mode`` (from the module's current weights, dequantized,
  or from a source checkpoint's tensors through the model's own
  ``sanitize``), scored, and restored by putting the original arrays and
  modules back.  The baseline is re-scored at the end and must be bit-equal
  (``restore_check``).  ``cumulative=True`` keeps each substitution applied,
  so row ``k`` scores layers ``1..k`` together (interactions are not
  additive).

KL here is ``sum_v p_ref(v) * (log p_ref(v) - log p_var(v))`` per position,
in fp32, averaged over the scored positions of every text (texts weighted by
their positions).  Sanity numbers, single pass, no thermal hold.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import numpy as np

REFERENCE_SCHEMA = "mlx2.layer-kl.reference.v1"
SCAN_SCHEMA = "mlx2.layer-kl.scan.v1"
DTYPES = {"float32": np.float32, "float16": np.float16}


# ------------------------------------------------------------------- texts --

def texts_sha256(texts: Sequence[dict]) -> str:
    """Identity of a tokenized text set (ids and splits, in order)."""
    payload = [{"id": t["id"], "split": t.get("split", "dev"), "ids": list(map(int, t["ids"]))}
               for t in texts]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def validate_texts(texts: Sequence[dict], *, skip: int) -> None:
    if not texts:
        raise ValueError("layer KL needs at least one text")
    seen = set()
    for t in texts:
        if t["id"] in seen:
            raise ValueError(f"duplicate text id {t['id']!r}")
        seen.add(t["id"])
        if len(t["ids"]) <= skip + 1:
            raise ValueError(f"text {t['id']!r} is not longer than skip + 1 tokens")
        if t.get("split", "dev") not in ("dev", "held", "calib"):
            raise ValueError("text split must be 'dev', 'held' or 'calib'")


# --------------------------------------------------------------- reference --

def _log_softmax(mx, logits):
    logits = logits.astype(mx.float32)
    return logits - mx.logsumexp(logits, axis=-1, keepdims=True)


def write_reference(out: Path, texts: Sequence[dict], logits_for: Callable[[int, dict], Any], *,
                    skip: int, checkpoint: str, source: dict, dtype: str = "float32") -> dict:
    """Write ``ref-<i>.npy`` (scored positions x vocab) and ``reference.json``.

    ``logits_for(i, text)`` returns the reference logits ``[T, V]`` (or
    ``[1, T, V]``) for ``text["ids"]``.  Refuses a nonempty ``out``.
    """
    import mlx.core as mx

    validate_texts(texts, skip=skip)
    if dtype not in DTYPES:
        raise ValueError(f"reference dtype must be one of {sorted(DTYPES)}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError(f"{out} is not empty")
    rows = []
    vocab = None
    for i, text in enumerate(texts):
        logits = logits_for(i, text)
        if logits.ndim == 3:
            logits = logits[0]
        if logits.shape[0] != len(text["ids"]):
            raise ValueError(f"reference logits for {text['id']!r} have {logits.shape[0]} rows, "
                             f"text has {len(text['ids'])} tokens")
        logp = _log_softmax(mx, logits[skip:])
        mx.eval(logp)
        array = np.array(logp).astype(DTYPES[dtype], copy=False)
        if not np.isfinite(array).all():
            raise ValueError(f"nonfinite reference log-probabilities for {text['id']!r}")
        vocab = array.shape[1] if vocab is None else vocab
        if array.shape[1] != vocab:
            raise ValueError("reference vocab size changed between texts")
        name = f"ref-{i}.npy"
        tmp = out / (name + ".partial.npy")
        np.save(tmp, array)
        os.replace(tmp, out / name)
        rows.append({"index": i, "id": text["id"], "split": text.get("split", "dev"),
                     "tokens": len(text["ids"]), "scored": int(array.shape[0]), "file": name})
    meta = {"schema": REFERENCE_SCHEMA, "checkpoint": checkpoint, "source": source, "skip": skip,
            "dtype": dtype, "vocab": vocab, "texts_sha256": texts_sha256(texts), "texts": rows,
            "positions": int(sum(r["scored"] for r in rows))}
    (out / "texts.json").write_text(json.dumps(list(texts)) + "\n")
    tmp = out / "reference.json.partial"
    tmp.write_text(json.dumps(meta, indent=2) + "\n")
    os.replace(tmp, out / "reference.json")
    return meta


def load_reference(path: Path) -> tuple[dict, list[dict]]:
    path = Path(path)
    meta = json.loads((path / "reference.json").read_text())
    if meta.get("schema") != REFERENCE_SCHEMA:
        raise ValueError(f"{path} is not a layer-KL reference")
    texts = json.loads((path / "texts.json").read_text())
    if texts_sha256(texts) != meta["texts_sha256"]:
        raise ValueError("reference texts do not match their recorded sha256")
    return meta, texts


def reference_from_model(model) -> Callable[[int, dict], Any]:
    """Reference logits from a loaded model's ordinary forward."""
    import mlx.core as mx

    def logits_for(_i, text):
        return model(mx.array([list(map(int, text["ids"]))]))

    return logits_for


def reference_from_family(family, texts: Sequence[dict]) -> Callable[[int, dict], Any]:
    """Reference logits streamed layer-major through a sweep family (interface v1).

    Every text's carry goes through layer ``l`` before the layer is released,
    so the model's weights are read once.  Carries stay in memory (one
    text's carry is a few residual-stream copies per token).
    """
    import mlx.core as mx

    if not callable(getattr(family, "final_logits", None)):
        raise ValueError("the sweep family has no final_logits (cannot produce reference logits)")
    import tempfile

    carries = [family.embed(mx.array([list(map(int, t["ids"]))])) for t in texts]
    prepare = getattr(family, "prepare_layer", None)
    begin = getattr(family, "begin_layer", None)
    for l in range(int(family.L)):
        with tempfile.TemporaryDirectory(prefix=f"layer-kl-L{l}-") as scratch:
            if callable(prepare):
                prepare(l, [list(map(int, t["ids"])) for t in texts], Path(scratch))
            if callable(begin):
                begin(l)
            try:
                for i, carry in enumerate(carries):
                    carries[i], _payload = family.run_layer(l, carry)
                    mx.eval(carries[i])
            finally:
                family.release_layer(l)
    logits = [family.final_logits(carry) for carry in carries]

    def logits_for(i, _text):
        return logits[i]

    return logits_for


# ----------------------------------------------------------------- scoring --

@dataclass
class Score:
    kl_mean: float
    top1: float
    positions: int
    per_text: list = field(default_factory=list)

    def as_dict(self):
        return {"kl_mean": self.kl_mean, "top1_agreement": self.top1, "positions": self.positions,
                "per_text": self.per_text}


def score(model, reference: Path, meta: dict, texts: Sequence[dict], *, splits=("dev",),
          chunk: int = 256) -> Score:
    """KL(reference || model) and top-1 agreement over the texts of ``splits``."""
    import mlx.core as mx

    skip = int(meta["skip"])
    total_kl = 0.0
    total_top1 = 0
    total = 0
    per_text = []
    for row, text in zip(meta["texts"], texts):
        if row["split"] not in splits:
            continue
        ref = np.load(Path(reference) / row["file"], mmap_mode="r")
        logits = model(mx.array([list(map(int, text["ids"]))]))[0, skip:]
        if logits.shape[0] != ref.shape[0] or logits.shape[1] != ref.shape[1]:
            raise ValueError(f"model logits {tuple(logits.shape)} do not match the reference "
                             f"{tuple(ref.shape)} for {text['id']!r}")
        logp = _log_softmax(mx, logits)
        kl_sum, agree = 0.0, 0
        for start in range(0, ref.shape[0], chunk):
            r = mx.array(np.asarray(ref[start:start + chunk], dtype=np.float32))
            v = logp[start:start + chunk]
            kl = (mx.exp(r) * (r - v)).sum(axis=-1)
            same = (mx.argmax(r, axis=-1) == mx.argmax(v, axis=-1))
            mx.eval(kl, same)
            kl_sum += float(np.array(kl, dtype=np.float64).sum())
            agree += int(np.array(same).sum())
        n = int(ref.shape[0])
        del logits, logp
        mx.clear_cache()
        per_text.append({"id": text["id"], "split": row["split"], "positions": n,
                         "kl_mean": kl_sum / n, "top1_agreement": agree / n})
        total_kl += kl_sum
        total_top1 += agree
        total += n
    if not total:
        raise ValueError(f"no reference texts in splits {list(splits)}")
    return Score(total_kl / total, total_top1 / total, total, per_text)


# ------------------------------------------------------------ substitution --

_QUANT_KEYS = ("weight", "scales", "biases")


# Quantized classes whose forward reads ``bits``/``group_size``/``mode`` from
# the module, so an in-place requantization is what the next call runs.  Any
# other quantized class (e.g. a bank with a hard-coded format) is refused.
_REQUANTIZABLE = frozenset({"QuantizedLinear", "QuantizedSwitchLinear", "QuantizedEmbedding"})


def module_paths(model, layer: int, groups: Sequence[str], *, layer_prefix: str = "model.layers.") -> list:
    """``(path, parent, attribute, module)`` of the substitutable modules of one layer.

    A module is substitutable when it is quantized (has ``scales``) or is a
    dense module with ``to_quantized``.  ``groups`` are path suffixes under
    the layer (``"mlp.experts"`` selects ``...mlp.experts.gate_proj`` etc.).
    """
    root = f"{layer_prefix}{int(layer)}."
    found = []
    for path, module in model.named_modules():
        if not path.startswith(root):
            continue
        rel = path[len(root):]
        if not any(rel == g or rel.startswith(g + ".") for g in groups):
            continue
        quantized = isinstance(module, dict) and "scales" in module and "weight" in module
        dense = hasattr(module, "to_quantized") and "weight" in module and "scales" not in module
        if not (quantized or dense):
            continue
        if quantized and type(module).__name__ not in _REQUANTIZABLE:
            raise ValueError(f"{path}: {type(module).__name__} has a fixed quantized format; "
                             f"only {sorted(_REQUANTIZABLE)} can be requantized in place")
        predicate = getattr(model, "quant_predicate", None)
        if dense and callable(predicate) and predicate(path, module) is False:
            continue  # the model keeps this module dense on purpose
        parent_path, _, attribute = path.rpartition(".")
        parent = model
        for part in parent_path.split("."):
            parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
        found.append((path, parent, attribute, module))
    if not found:
        raise ValueError(f"layer {layer} has no substitutable module under {list(groups)}")
    return found


def _module_bytes(module) -> int:
    return int(sum(int(module[k].nbytes) for k in _QUANT_KEYS if k in module and module[k] is not None))


def _dense_weight(mx, module):
    if "scales" not in module:
        return module["weight"]
    biases = module.get("biases")
    return mx.dequantize(module["weight"], module["scales"], biases, group_size=module.group_size,
                         bits=module.bits, mode=getattr(module, "mode", "affine"))


def _dense_shape(module) -> tuple:
    """Dense weight shape of a module without materializing it."""
    weight = module["weight"]
    if "scales" not in module:
        return tuple(weight.shape)
    return (*weight.shape[:-1], int(module["scales"].shape[-1]) * int(module.group_size))


def _rtn_planes(mx, dense, *, group_size: int, bits: int, mode: str, chunk: int = 64):
    """MLX round-to-nearest planes, an expert bank ``[E, O, I]`` in chunks.

    A whole 512-expert bank in fp32 is ~4.8 GB per projection at Hy4 size;
    chunking (and clearing MLX's buffer cache) keeps the transient small.
    """
    if dense.ndim < 3:
        planes = mx.quantize(dense.astype(mx.float32), group_size=group_size, bits=bits, mode=mode)
        mx.eval(*[p for p in planes if p is not None])
        return planes
    parts = []
    for start in range(0, dense.shape[0], chunk):
        planes = mx.quantize(dense[start:start + chunk].astype(mx.float32), group_size=group_size,
                             bits=bits, mode=mode)
        mx.eval(*[p for p in planes if p is not None])
        parts.append(planes)
        mx.clear_cache()
    return tuple(mx.concatenate([p[k] for p in parts], axis=0) if parts[0][k] is not None else None
                 for k in range(len(parts[0])))


@dataclass
class _Applied:
    path: str
    parent: Any
    attribute: str
    original: Any
    saved: Optional[dict]   # quantized in place: the original arrays and attributes
    bytes_before: int
    bytes_after: int


def apply_substitution(model, layer: int, groups: Sequence[str], *, bits: int, group_size: int,
                       mode: str = "affine", source: Optional[Callable[[str], Any]] = None,
                       layer_prefix: str = "model.layers.",
                       planes_for: Optional[Callable[[str, Any, Any], tuple]] = None) -> list:
    """Requantize one layer's module group in place; returns what ``restore`` needs.

    ``source(path)`` (optional) returns the dense source weight for a module
    path (e.g. from the original checkpoint); without it the module's own
    weights are dequantized and requantized.  ``planes_for(path, dense,
    module)`` (optional) returns the ``(weight, scales, biases)`` planes for
    the target format (e.g. :mod:`mlx2.gptq`); the default is MLX's own
    round-to-nearest ``mx.quantize``.
    """
    applied = []
    try:
        for path, parent, attribute, module in module_paths(model, layer, groups, layer_prefix=layer_prefix):
            applied += apply_substitution_one(model, path, parent, attribute, module, bits=bits,
                                              group_size=group_size, mode=mode, source=source,
                                              planes_for=planes_for)
    except BaseException:
        restore(applied)
        raise
    return applied


def apply_substitution_one(model, path, parent, attribute, module, *, bits: int, group_size: int,
                           mode: str = "affine", source=None, planes_for=None) -> list:
    """Requantize one module in place (or swap a dense one); ``[_Applied]``."""
    import mlx.core as mx

    if "scales" in module and getattr(module, "mode", "affine") != mode:
        raise ValueError(f"{path}: in-place requantization keeps the module's mode "
                         f"({getattr(module, 'mode', 'affine')!r}), not {mode!r}")
    dense = source(path) if source is not None else _dense_weight(mx, module)
    if tuple(dense.shape) != _dense_shape(module):
        raise ValueError(f"{path}: source weight shape {tuple(dense.shape)} differs from the module's")
    before = _module_bytes(module)
    if "scales" in module:
        saved = {k: module[k] for k in _QUANT_KEYS if k in module}
        saved["_attrs"] = {k: getattr(module, k) for k in ("bits", "group_size", "mode")
                           if hasattr(module, k)}
        planes = (planes_for(path, dense, module) if planes_for is not None else
                  _rtn_planes(mx, dense, group_size=group_size, bits=bits, mode=mode))
        del dense
        module["weight"], module["scales"] = planes[0], planes[1].astype(module["scales"].dtype)
        biases = planes[2].astype(module["scales"].dtype) if len(planes) > 2 else None
        if "biases" in module or biases is not None:
            module["biases"] = biases
        module.bits, module.group_size, module.mode = bits, group_size, mode
        mx.eval(*[module[k] for k in _QUANT_KEYS if k in module and module[k] is not None])
        mx.clear_cache()
        return [_Applied(path, parent, attribute, module, saved, before, _module_bytes(module))]
    replacement = module.to_quantized(group_size=group_size, bits=bits, mode=mode)
    if source is not None or planes_for is not None:
        planes = (planes_for(path, dense, module) if planes_for is not None else
                  _rtn_planes(mx, dense, group_size=group_size, bits=bits, mode=mode))
        # Keep the planes' dtype that to_quantized chose (the model's
        # activation dtype): fp32 planes would change it.
        scale_dtype = replacement["scales"].dtype
        replacement["weight"] = planes[0]
        replacement["scales"] = planes[1].astype(scale_dtype)
        if len(planes) > 2 and planes[2] is not None:
            replacement["biases"] = planes[2].astype(scale_dtype)
    mx.eval(replacement.parameters())
    setattr(parent, attribute, replacement)
    return [_Applied(path, parent, attribute, module, None, before, _module_bytes(replacement))]


def restore(applied: Iterable[_Applied]) -> None:
    """Put back exactly what ``apply_substitution`` replaced (reverse order)."""
    for item in reversed(list(applied)):
        if item.saved is None:
            setattr(item.parent, item.attribute, item.original)
            continue
        module = item.original
        for key in _QUANT_KEYS:
            if key in item.saved:
                module[key] = item.saved[key]
            elif key in module:
                del module[key]
        for key, value in item.saved["_attrs"].items():
            setattr(module, key, value)


# --------------------------------------------------------- source weights --

_ST_DTYPES = {"BF16": ("<u2", "bfloat16"), "F16": ("<f2", None), "F32": ("<f4", None)}


def _st_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as handle:
        (size,) = struct.unpack("<Q", handle.read(8))
        if not 2 <= size <= 1 << 28:
            raise ValueError(f"implausible safetensors header in {path}")
        return json.loads(handle.read(size)), 8 + size


def checkpoint_source(source_dir: Path, model, *, layer_prefix: str = "model.layers.",
                      groups: Sequence[str] = ()):
    """``source(path)`` reading a dense source checkpoint, mapped through ``model.sanitize``.

    For layer ``L`` every source tensor under ``<layer_prefix>L.`` is read
    (by byte range), the model's own ``sanitize`` maps checkpoint names and
    layouts to module names (split/fused projections, folds), and the module
    path's ``.weight`` is returned.  One layer's tensors are cached at a time.
    With ``groups``, only tensors under those path suffixes are read (e.g.
    ``mlp.experts`` never reads a layer's multi-GB embedding tables).
    """
    import mlx.core as mx

    source_dir = Path(source_dir)
    weight_map = json.loads((source_dir / "model.safetensors.index.json").read_text())["weight_map"]
    cache: dict = {"layer": None, "tensors": {}}

    def read(name):
        shard = source_dir / weight_map[name]
        header, data0 = _st_header(shard)
        entry = header[name]
        if entry["dtype"] not in _ST_DTYPES:
            raise ValueError(f"{name}: source dtype {entry['dtype']} is not a dense float")
        begin, end = entry["data_offsets"]
        with open(shard, "rb") as handle:
            handle.seek(data0 + begin)
            raw = handle.read(end - begin)
        width, view = _ST_DTYPES[entry["dtype"]]
        array = mx.array(np.frombuffer(raw, width).reshape(entry["shape"]))
        return array.view(mx.bfloat16) if view == "bfloat16" else array

    def source(path):
        rest = path[len(layer_prefix):]
        layer = int(rest.split(".", 1)[0])
        if cache["layer"] != layer:
            root = f"{layer_prefix}{layer}."
            names = [n for n in weight_map if n.startswith(root) and (
                not groups or any(n[len(root):] == g or n[len(root):].startswith(g + ".") or
                                  n[len(root):].startswith(g + "_") for g in groups))]
            if not names:
                raise KeyError(f"source checkpoint has no tensors under {root}{list(groups)}")
            raw = {n: read(n) for n in names}
            sanitize = getattr(model, "sanitize", None)
            cache.update(layer=layer, tensors=sanitize(raw) if callable(sanitize) else raw)
        key = path + ".weight"
        if key not in cache["tensors"]:
            raise KeyError(f"source checkpoint has no {key} (after sanitize)")
        return cache["tensors"][key]

    return source


# ---------------------------------------------------------------------- scan --

def parse_layers(text: str) -> list[int]:
    """``"3-5,9"`` -> ``[3, 4, 5, 9]`` (order kept, duplicates refused)."""
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(v) for v in part.split("-", 1))
            if hi < lo:
                raise ValueError(f"bad layer range {part!r}")
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    if len(set(out)) != len(out) or not out:
        raise ValueError("layers must be a nonempty list without duplicates")
    return out


def gptq_planes(model, layer: int, groups: Sequence[str], calib_texts: Sequence[dict], *, bits: int,
                group_size: int, settings=None, source=None, layer_prefix: str = "model.layers."):
    """``(planes_for, receipts)``: GPTQ planes for one layer from calibration inputs.

    The layer's target modules are wrapped with :class:`mlx2.gptq.InputRecorder`
    while the model runs ``calib_texts`` (its current weights, unchanged);
    each module's Hessian (per expert for a bank) then drives GPTQ on its
    dense weight (``source(path)`` or the dequantized current weight).
    """
    import mlx.core as mx

    from . import gptq as G

    settings = settings or G.GPTQSettings(bits=bits, group_size=group_size)
    found = module_paths(model, layer, groups, layer_prefix=layer_prefix)
    targets = [(parent, attribute, module, _is_bank(module)) for _path, parent, attribute, module in found]
    if not calib_texts:
        raise ValueError("GPTQ needs calibration texts")

    def forward():
        for text in calib_texts:
            mx.eval(model(mx.array([list(map(int, text["ids"]))])))

    recorders = G.record(model, targets, forward)
    by_path = {path: (recorder, module) for (path, _p, _a, module), recorder in zip(found, recorders)}
    receipts = {}

    def planes_for(path, dense, module):
        recorder, _module = by_path[path]
        experts = dense.shape[0] if dense.ndim == 3 else None
        hessians, rows = recorder.hessians(experts)
        dtype = module["scales"].dtype if "scales" in module else dense.dtype
        planes, receipt = G.quantize_bank(dense, hessians, settings, dtype=dtype, rows=rows)
        receipts[path] = receipt
        recorder.rows.clear()
        recorder.ids.clear()
        return planes

    return planes_for, receipts


def _glu_stages(found) -> list:
    """Calibration stages for one layer's targets.

    Inside a gated MLP (a parent with ``gate_proj``, ``up_proj`` and
    ``down_proj``, e.g. mlx2's ``SwitchGLU``), gate and up read the same
    input (one shared Hessian, recorded on gate) and down reads their
    product, so down is calibrated in a second pass on the already
    quantized gate/up (sequential GPTQ).  Every other target is stage 1.
    """
    first, second, shared = [], [], {}
    for item in found:
        path, parent, attribute, _module = item
        glu = all(hasattr(parent, name) for name in ("gate_proj", "up_proj", "down_proj"))
        if glu and attribute == "down_proj":
            second.append(item)
        elif glu and attribute == "up_proj" and any(p is parent and a == "gate_proj" for _x, p, a, _m in found):
            shared[path] = path[: -len("up_proj")] + "gate_proj"
            first.append(item)
        else:
            first.append(item)
    return [stage for stage in (first, second) if stage], shared


def gptq_apply(model, layer: int, groups: Sequence[str], calib_texts: Sequence[dict], *, bits: int,
               group_size: int, settings=None, source=None, layer_prefix: str = "model.layers.",
               between_texts: Optional[Callable[[], None]] = None) -> tuple:
    """Quantize one layer's targets with GPTQ, stage by stage; ``(applied, receipts)``.

    Hessians are accumulated while ``calib_texts`` run (each forward stops
    once the stage's modules have recorded it); a stage is quantized and
    applied before the next stage records, so later modules see the
    quantized earlier ones.  ``applied`` restores exactly like
    :func:`apply_substitution`.
    """
    import mlx.core as mx

    from . import gptq as G

    settings = settings or G.GPTQSettings(bits=bits, group_size=group_size)
    if not calib_texts:
        raise ValueError("GPTQ needs calibration texts")
    found = module_paths(model, layer, groups, layer_prefix=layer_prefix)
    stages, shared = _glu_stages(found)
    receipts, applied = {}, []

    # A model exposing ``hidden(ids)`` (the pre-head forward) is calibrated
    # through it: the LM head's full-vocab logits are never needed here.
    forward = getattr(model, "hidden", None)
    forward = forward if callable(forward) else model

    def run_text(text):
        # The swap guard runs after a finished or early-stopped text only, so a
        # SwapGrowth never masks a real forward error.
        failed = False
        try:
            mx.eval(forward(mx.array([list(map(int, text["ids"]))])))
        except BaseException as error:
            failed = not isinstance(error, G.StopForward)
            raise
        finally:
            mx.clear_cache()
            if between_texts is not None and not failed:
                between_texts()

    try:
        for stage in stages:
            recorded = [item for item in stage if item[0] not in shared]
            targets = [(parent, attribute, module, _is_bank(module),
                        module["weight"].shape[0] if _is_bank(module) else None)
                       for _path, parent, attribute, module in recorded]
            accumulators = dict(zip([item[0] for item in recorded],
                                    G.accumulate(targets, calib_texts, run_text)))
            for path, parent, attribute, module in stage:
                acc = accumulators[shared.get(path, path)]
                dense_shape = _dense_shape(module)
                hessians, rows = acc.hessians(dense_shape[-1])
                done = {}

                def planes_for(p, dense, m, hessians=hessians, rows=rows, done=done):
                    dtype = m["scales"].dtype if "scales" in m else dense.dtype
                    planes, receipt = G.quantize_bank(dense, hessians, settings, dtype=dtype,
                                                      rows=rows if dense.ndim == 3 else None)
                    done["receipt"] = receipt
                    return planes

                applied += apply_substitution_one(model, path, parent, attribute, module, bits=bits,
                                                  group_size=group_size, source=source,
                                                  planes_for=planes_for)
                receipts[path] = {**done["receipt"], "stage": stages.index(stage) + 1,
                                  "hessian_from": shared.get(path, path)}
            for acc in accumulators.values():
                acc.release()
            mx.clear_cache()
    except BaseException:
        restore(applied)
        raise
    return applied, receipts


def _is_bank(module) -> bool:
    weight = module["weight"]
    return weight.ndim == 3 or hasattr(module, "num_experts")


def scan(model, reference: Path, *, layers: Sequence[int], groups: Sequence[str], bits: int,
         group_size: int, mode: str = "affine", source: Optional[Callable[[str], Any]] = None,
         cumulative: bool = False, splits=("dev",), layer_prefix: str = "model.layers.",
         log: Optional[Callable[[str], None]] = None, quantizer: str = "rtn",
         calib_texts: Sequence[dict] = (), gptq_settings=None,
         on_row: Optional[Callable[[dict, dict], None]] = None,
         between_texts: Optional[Callable[[], None]] = None) -> dict:
    """Baseline, then one row per layer (or per cumulative prefix); exact restore checked.

    ``quantizer`` ``"rtn"`` (MLX round-to-nearest) or ``"gptq"`` (calibrated
    on ``calib_texts``, which must not overlap the scored texts).
    """
    import mlx.core as _mx

    if quantizer not in ("rtn", "gptq"):
        raise ValueError("quantizer must be 'rtn' or 'gptq'")
    if quantizer == "gptq" and mode != "affine":
        raise ValueError("GPTQ emits affine planes")
    meta, texts = load_reference(reference)
    for layer in layers:  # refuse up front, before hours of scoring
        for path, _parent, _attribute, module in module_paths(model, layer, groups, layer_prefix=layer_prefix):
            if "scales" in module and getattr(module, "mode", "affine") != mode:
                raise ValueError(f"{path}: in-place requantization keeps the module's mode")
    scored_ids = {tuple(t["ids"]) for t in texts}
    if any(tuple(t["ids"]) in scored_ids for t in calib_texts):
        raise ValueError("calibration texts overlap the reference texts")
    t0 = time.monotonic()
    base = score(model, reference, meta, texts, splits=splits)
    rows = []
    applied_all: list = []
    try:
        for layer in layers:
            tic = time.monotonic()
            if quantizer == "gptq":
                applied, gptq_receipts = gptq_apply(model, layer, groups, calib_texts, bits=bits,
                                                    group_size=group_size, settings=gptq_settings,
                                                    source=source, layer_prefix=layer_prefix,
                                                    between_texts=between_texts)
            else:
                gptq_receipts = {}
                applied = apply_substitution(model, layer, groups, bits=bits, group_size=group_size,
                                             mode=mode, source=source, layer_prefix=layer_prefix)
            try:
                s = score(model, reference, meta, texts, splits=splits)
            finally:
                if cumulative:
                    applied_all.extend(applied)
                else:
                    restore(applied)
                _mx.clear_cache()
            row = {"layer": int(layer), "modules": [a.path for a in applied],
                   "bytes_before": sum(a.bytes_before for a in applied),
                   "bytes_after": sum(a.bytes_after for a in applied),
                   **s.as_dict(), "delta_kl": s.kl_mean - base.kl_mean,
                   "delta_top1": s.top1 - base.top1, "seconds": round(time.monotonic() - tic, 2),
                   # MLX memory after the layer was restored: a leak shows as
                   # active_gb climbing row by row; a transient as peak_gb only.
                   "mlx_active_gb": round(_mx.get_active_memory() / 2**30, 3),
                   "mlx_cache_gb": round(_mx.get_cache_memory() / 2**30, 3),
                   "mlx_peak_gb": round(_mx.get_peak_memory() / 2**30, 3),
                   **({"gptq": gptq_receipts} if gptq_receipts else {})}
            rows.append(row)
            if on_row is not None:
                on_row(row, base.as_dict())
            if log:
                log(f"[layer {layer}{' cumulative' if cumulative else ''}] kl={s.kl_mean:.5f} "
                    f"delta={row['delta_kl']:+.5f} top1={s.top1:.4f} "
                    f"bytes {row['bytes_before'] / 2**30:.3f}->{row['bytes_after'] / 2**30:.3f} GiB")
    finally:
        restore(applied_all)
    again = score(model, reference, meta, texts, splits=splits)
    return {"schema": SCAN_SCHEMA, "reference": {"path": str(reference), "checkpoint": meta["checkpoint"],
                                                 "texts_sha256": meta["texts_sha256"], "skip": meta["skip"]},
            "settings": {"layers": list(map(int, layers)), "groups": list(groups), "bits": bits,
                         "group_size": group_size, "mode": mode, "cumulative": cumulative,
                         "quantizer": quantizer, "calibration_texts": len(calib_texts),
                         "splits": list(splits), "source": "checkpoint" if source is not None else "current_weights",
                         "layer_prefix": layer_prefix},
            "baseline": base.as_dict(), "rows": rows,
            "restore_check": {"baseline_kl_after": again.kl_mean, "exact": again.kl_mean == base.kl_mean
                              and again.top1 == base.top1},
            "seconds": round(time.monotonic() - t0, 2)}


def ranked(rows: Sequence[dict]) -> list[dict]:
    """Rows by increasing ``delta_kl`` (most tolerant layer first)."""
    return sorted(rows, key=lambda r: (r["delta_kl"], r["layer"]))


def finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)
