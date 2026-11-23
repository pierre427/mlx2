"""Residual-direction ablation (abliteration) for loaded MLX models.

Original MLX implementation of the directional-ablation idea that Heretic
(p-e-w/heretic, AGPL-3.0) automates. Heretic was read as a conceptual
reference only; no code, prompt list or marker list is copied. See
``provenance/heretic-residual-ablation-20261105.json``.

Three layers of machinery, all model-agnostic. The adapter declares which
modules of a decoder layer write into the residual stream
(``RESIDUAL_ABLATION`` below), so nothing here branches on model names:

* ``LayerOutputCapture``: records each decoder layer's last-position output
  during an ordinary forward.
* ``ProjectionHooks``: during a search trial, subtracts ``lam * v (v . y)``
  from a residual writer's output ``y``. This is exactly the
  ``row_normalization="none"`` weight edit ``W' = (I - lam v v^T) W``
  applied to the writer's output projection. For MoE blocks it is exact
  because every routed expert and the shared expert get the same ``lam``
  and ``v`` and the router-weighted sum is linear.
* ``materialize`` / ``bake_checkpoint``: apply the edit to the weights,
  either in memory (an A/B arm, including Heretic's norm-preserving "full"
  mode) or into a new checkpoint directory. The source checkpoint is opened
  read-only, hashed before and after, and never written.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .heretic import LayerProfile, estimate_direction

COMPONENTS = ("attention", "mlp")
ROW_NORMALIZATIONS = ("none", "full")
# Where search hooks project: on the writer module's call output (the branch
# added to the residual) or on the declared output linears.  The two agree
# when the linears' summed output IS the branch; a writer that ends in a norm
# (``post_norms``) must hook the linears so hooks equal the weight edit.
HOOK_POINTS = ("writers", "linears")
# How a decoder layer's output becomes one capture row ``[batch, hidden]``:
# ``layer_output`` takes ``out[:, -1, :]``; ``stream_mean`` averages the
# residual streams of an ``[batch, seq, streams, hidden]`` output (multi-stream
# hyper-connection trunks) at the last position.
CAPTURES = ("layer_output", "stream_mean")
# Prompt rendering for the tooling: the tokenizer's chat template with
# ``enable_thinking=False``, or the adapter's own ``prompt_tokens`` (its
# serving default), for templates whose no-flag default reasons.
PROMPT_RENDERS = ("template", "adapter")


# ---------------------------------------------------------------- declaration


@dataclass(frozen=True)
class AblationDeclaration:
    """Adapter-owned description of a decoder layer's residual writers.

    ``writers`` names the layer attributes whose call output is added to the
    residual (first present name wins). ``linears`` names, relative to the
    layer, the output projections whose weights realise the same edit; a
    stacked ``[experts, out, in]`` weight is allowed. ``layer_key`` is the
    checkpoint key pattern of decoder layers (group 1 = layer index).
    """

    writers: dict
    linears: dict
    # Checkpoint-key spelling of ``linears`` when the loader renames modules
    # (e.g. Kimi: checkpoint ``self_attn.o_proj`` -> module ``attention.o_proj``).
    checkpoint_linears: dict | None = None
    # Anchored on the decoder trunk ("model.layers.N." / "language_model.model.layers.N."),
    # so MTP heads ("mtp.layers.N.") or vision towers never match.
    layer_key: str = r"(?:^|\.)(?:model|language_model)\.layers\.(\d+)\."
    # ``{linear path: norm path}``: the writer's summed linear output passes
    # through this RMS-style norm (``gamma * y / rms(y)``) before it reaches
    # the residual.  The edit then uses ``u = normalize(gamma * v)`` for that
    # linear, so at ``lam = 1`` the normed output has no ``v`` component
    # (``v . (gamma * y') = (gamma * v) . y' = 0``); see ``writer_direction``.
    # Every linear of one (layer, component) must name the same norm.
    post_norms: dict | None = None
    hook_point: str = "writers"
    capture: str = "layer_output"
    # ``{present path: (required paths...)}`` checked per layer: a module that
    # writes the residual through the required linears only (e.g. a routed
    # expert bank behind a dense up-projection) must not appear without them,
    # or an undeclared writer would escape the edit.
    requires: dict | None = None
    prompt_render: str = "template"
    # Residual writes no declared linear carries (the token embedding always;
    # e.g. Engram memories), named in overlay receipts, status and bake
    # records so nobody reads an edit as covering the whole residual.
    unedited_writers: tuple = ("embed",)

    def __post_init__(self):
        if set(self.writers) != set(COMPONENTS) or set(self.linears) != set(COMPONENTS):
            raise ValueError("declaration must cover exactly the attention and mlp components")
        if self.checkpoint_linears is not None and set(self.checkpoint_linears) != set(COMPONENTS):
            raise ValueError("checkpoint_linears must cover exactly the attention and mlp components")
        if self.hook_point not in HOOK_POINTS:
            raise ValueError(f"hook_point must be one of {HOOK_POINTS}")
        if self.capture not in CAPTURES:
            raise ValueError(f"capture must be one of {CAPTURES}")
        if self.prompt_render not in PROMPT_RENDERS:
            raise ValueError(f"prompt_render must be one of {PROMPT_RENDERS}")
        if self.post_norms is not None:
            declared = {path for component in COMPONENTS for path in self.linears[component]}
            if set(self.post_norms) != declared:
                raise ValueError("post_norms must name a norm for every declared output linear")
            if self.hook_point != "linears":
                raise ValueError("post-normed writers need hook_point 'linears' (a writer-output "
                                 "projection is not a weight edit through a norm)")
            if self.checkpoint_linears is not None and self.checkpoint_linears != self.linears:
                raise ValueError("post_norms require checkpoint and module linear names to agree")

    def checkpoint_paths(self, component):
        return (self.checkpoint_linears or self.linears)[component]


def declaration_for(adapter):
    declared = getattr(adapter, "residual_ablation", None)
    if isinstance(declared, dict):
        declared = AblationDeclaration(**declared)
    if not isinstance(declared, AblationDeclaration):
        raise ValueError(
            f"{type(adapter).__name__} declares no residual-ablation writers; refusing"
        )
    return declared


def _layers(model):
    layers = getattr(model, "layers", None)
    if layers is None:
        raise ValueError("model exposes no decoder layers")
    return list(layers)


def resolve_writers(model, declaration):
    """Return ``[{component: module}]`` per decoder layer, failing closed."""
    out = []
    for index, layer in enumerate(_layers(model)):
        row = {}
        for component in COMPONENTS:
            for name in declaration.writers[component]:
                if name in layer:
                    row[component] = layer[name]
                    break
            else:
                raise ValueError(f"layer {index} has no declared {component} writer")
        out.append(row)
    return out


def _get_path(root, dotted):
    node = root
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None, None, None
        parent, node = node, node[part]
    return parent, dotted.split(".")[-1], node


def resolve_linears(model, declaration):
    """Return ``[(layer, component, relative_path, parent, attr, module)]``."""
    found = []
    for index, layer in enumerate(_layers(model)):
        for present, required in (declaration.requires or {}).items():
            if _get_path(layer, present)[2] is None:
                continue
            missing = [path for path in required if _get_path(layer, path)[2] is None]
            if missing:
                raise ValueError(f"layer {index} has {present} without {missing}: its residual "
                                 "write would bypass the declared output projections")
        for component in COMPONENTS:
            hits = 0
            for path in declaration.linears[component]:
                parent, attr, module = _get_path(layer, path)
                if module is not None:
                    found.append((index, component, path, parent, attr, module))
                    hits += 1
            if not hits:
                raise ValueError(f"layer {index} has no declared {component} output projection")
    return found


def post_norm_gammas(model, declaration):
    """``{(layer, component): float32 gamma}`` of declared post-norms ({} without them)."""
    if not declaration.post_norms:
        return {}
    mx = _mx()
    layers = _layers(model)
    found = {}
    for index, component, path, *_rest in resolve_linears(model, declaration):
        norm_path = declaration.post_norms[path]
        _parent, _attr, norm = _get_path(layers[index], norm_path)
        if norm is None or "weight" not in norm:
            raise ValueError(f"layer {index} {path} names a missing post-norm {norm_path}")
        previous = found.get((index, component))
        if previous is not None and previous[0] != norm_path:
            raise ValueError(f"layer {index} {component} writers name different post-norms "
                             f"({previous[0]}, {norm_path})")
        weight = norm["weight"].astype(mx.float32)
        mx.eval(weight)
        found[(index, component)] = (norm_path, np.asarray(weight))
    return {key: gamma for key, (_path, gamma) in found.items()}


def writer_direction(v, gamma=None):
    """The direction a post-normed writer's linears project out: ``normalize(gamma * v)``.

    ``RMSNorm(y) = gamma * y / rms(y)``, so ``v . RMSNorm(y') = (gamma * v) . y' / rms(y')``.
    With ``y' = (I - lam u u^T) y`` and ``u = normalize(gamma * v)`` the written
    branch keeps ``(1 - lam) * rms(y) / rms(y')`` of its ``v`` component: none
    at ``lam = 1``.  ``gamma is None`` (no post-norm) returns ``v`` unchanged.
    """
    if gamma is None:
        return np.asarray(v, dtype=np.float32)
    u = np.asarray(gamma, dtype=np.float64) * np.asarray(v, dtype=np.float64)
    norm = float(np.linalg.norm(u))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("the post-norm gain cancels the ablation direction")
    return (u / norm).astype(np.float32)


# ------------------------------------------------------------- class patching

_PATCHED_CLASSES = {}
_ACTIVE = {}


def _patched_class(cls):
    patched = _PATCHED_CLASSES.get(cls)
    if patched is None:
        def __call__(self, *args, **kwargs):
            out = cls.__call__(self, *args, **kwargs)
            hook = _ACTIVE.get(id(self))
            return out if hook is None else hook(out)

        patched = type(f"Hooked{cls.__name__}", (cls,), {"__call__": __call__})
        _PATCHED_CLASSES[cls] = patched
    return patched


def _install(module, hook):
    if id(module) in _ACTIVE:
        raise RuntimeError("module already hooked")
    original = type(module)
    module.__class__ = _patched_class(original)
    _ACTIVE[id(module)] = hook
    return original


def _remove(module, original):
    _ACTIVE.pop(id(module), None)
    module.__class__ = original


@contextmanager
def _hooked(pairs):
    installed = []
    try:
        for module, hook in pairs:
            installed.append((module, _install(module, hook)))
        yield
    finally:
        for module, original in reversed(installed):
            _remove(module, original)


# ------------------------------------------------------------------- capture


class LayerOutputCapture:
    """Record every decoder layer's last-position output (float32).

    ``reduce`` is the declaration's ``capture``.  ``stream_mean`` takes a
    ``[batch, seq, streams, hidden]`` multi-stream output, keeps the
    last-position streams (``take_streams``) and returns their mean from
    ``take``.  A ``layer_output`` capture refuses any output that is not
    ``[batch, seq, hidden]`` rather than stacking a wrong geometry.
    """

    def __init__(self, model, reduce="layer_output"):
        if reduce not in CAPTURES:
            raise ValueError(f"capture must be one of {CAPTURES}")
        self.layers = _layers(model)
        self.reduce = reduce
        self.rows = [None] * len(self.layers)
        self.streams = None

    def _hook(self, index):
        def hook(out):
            expected = 4 if self.reduce == "stream_mean" else 3
            if out.ndim != expected:
                raise ValueError(f"{self.reduce} capture expects a rank-{expected} layer output, "
                                 f"got shape {tuple(out.shape)}")
            self.rows[index] = out[:, -1].astype(_mx().float32)
            return out
        return hook

    @contextmanager
    def active(self):
        with _hooked([(layer, self._hook(i)) for i, layer in enumerate(self.layers)]):
            yield self

    def take(self):
        mx = _mx()
        if any(row is None for row in self.rows):
            raise RuntimeError("a decoder layer did not run during capture")
        stacked = mx.stack(self.rows, axis=1)  # [batch, layers, (streams,) hidden]
        self.rows = [None] * len(self.layers)
        if self.reduce == "stream_mean":
            self.streams = stacked
            stacked = mx.mean(stacked, axis=2)
            mx.eval(stacked, self.streams)
        else:
            mx.eval(stacked)
        return np.asarray(stacked)

    def take_streams(self):
        """Per-stream rows ``[batch, layers, streams, hidden]`` of the last ``take``."""
        if self.streams is None:
            raise RuntimeError("no multi-stream capture was taken")
        streams, self.streams = self.streams, None
        return np.asarray(streams)


def _mx():
    import mlx.core as mx

    return mx


# ---------------------------------------------------------------- directions


@dataclass
class Directions:
    """Unit refusal directions per hidden-state index.

    Index ``i + 1`` is the direction measured at decoder layer ``i``'s output
    (index 0, the embeddings, is never used by the search ranges and is not
    captured).
    """

    vectors: np.ndarray  # [layers + 1, hidden], row 0 = NaN
    reports: list
    source: dict = field(default_factory=dict)

    @property
    def layer_count(self):
        return self.vectors.shape[0] - 1

    def for_layer(self, layer, direction_index=None):
        if direction_index is None:
            return self.vectors[layer + 1]
        fraction, base = math.modf(direction_index + 1)
        base = int(base)
        if not 1 <= base < self.vectors.shape[0] - (1 if fraction else 0):
            raise ValueError("direction_index outside the captured layers")
        a = self.vectors[base].astype(np.float64)
        b = self.vectors[min(base + 1, self.layer_count)].astype(np.float64)
        mixed = a + fraction * (b - a)
        return (mixed / np.linalg.norm(mixed)).astype(np.float32)

    def save(self, path):
        path = Path(path)
        np.savez(path.with_suffix(".npz"), vectors=self.vectors)
        digest = hashlib.sha256(self.vectors.tobytes()).hexdigest()
        meta = {"schema": "mlx2.residual-ablation-directions.v1", "sha256": digest,
                "shape": list(self.vectors.shape), "source": self.source, "reports": self.reports}
        path.with_suffix(".json").write_text(json.dumps(meta, indent=1, sort_keys=True))
        return digest

    @classmethod
    def load(cls, path):
        path = Path(path)
        vectors = np.load(path.with_suffix(".npz"))["vectors"]
        meta = json.loads(path.with_suffix(".json").read_text())
        if hashlib.sha256(vectors.tobytes()).hexdigest() != meta["sha256"]:
            raise ValueError("direction file digest mismatch")
        return cls(vectors, meta["reports"], meta.get("source", {}))


def compute_directions(good, bad, *, orthogonalize=True, winsorization_quantile=1.0, source=None):
    """``good``/``bad``: ``[prompts, layers, hidden]`` last-position captures."""
    if good.ndim != 3 or bad.ndim != 3 or good.shape[1:] != bad.shape[1:]:
        raise ValueError("captures must be [prompts, layers, hidden] with matching geometry")
    layers, hidden = good.shape[1:]
    vectors = np.full((layers + 1, hidden), np.nan, dtype=np.float32)
    reports = [None]
    for layer in range(layers):
        estimate = estimate_direction(
            good[:, layer, :], bad[:, layer, :],
            orthogonalize_reference=orthogonalize,
            winsorization_quantile=winsorization_quantile,
        )
        vectors[layer + 1] = estimate.direction
        g = good[:, layer, :].astype(np.float64) @ estimate.direction.astype(np.float64)
        b = bad[:, layer, :].astype(np.float64) @ estimate.direction.astype(np.float64)
        pooled = math.sqrt((g.var() + b.var()) / 2) or 1.0
        estimate.report["separation_d"] = float((b.mean() - g.mean()) / pooled)
        reports.append(estimate.report)
    return Directions(vectors, reports, dict(source or {}))


# ---------------------------------------------------------------- parameters


@dataclass(frozen=True)
class ComponentKernel:
    max_weight: float
    max_weight_position: float
    min_weight: float
    min_weight_distance: float

    def profile(self):
        return LayerProfile(
            peak_layer=self.max_weight_position, radius=self.min_weight_distance,
            peak_strength=self.max_weight, edge_strength=self.min_weight,
        )


@dataclass(frozen=True)
class AblationParameters:
    kernels: dict  # component -> ComponentKernel
    direction_index: float | None = None  # None = per-layer directions

    def plan(self, directions, layer_count):
        """``[(layer, component, lam, v)]`` for every nonzero edit."""
        edits = []
        for layer in range(layer_count):
            v = directions.for_layer(layer, self.direction_index)
            for component in COMPONENTS:
                lam = self.kernels[component].profile().strength(layer)
                if lam != 0:
                    edits.append((layer, component, float(lam), v))
        return edits

    def as_dict(self):
        return {"direction_index": self.direction_index,
                "kernels": {k: vars(v) for k, v in self.kernels.items()}}


# --------------------------------------------------------------------- hooks


class ProjectionHooks:
    """Exact output-space projection ``y - lam * v (v . y)`` on residual writers.

    ``hook_point == "linears"`` hooks every declared output linear of the
    edited (layer, component) instead, with the post-norm writer direction
    (``writer_direction``): the hook then equals the ``none`` weight edit
    linear by linear, through any norm that follows the writer.
    """

    def __init__(self, model, declaration):
        self.writers = self.targets = None
        self.gammas = {}
        if declaration.hook_point == "linears":
            self.targets = {}
            for index, component, _path, _parent, _attr, module in resolve_linears(model, declaration):
                self.targets.setdefault((index, component), []).append(module)
            self.gammas = post_norm_gammas(model, declaration)
        else:
            self.writers = resolve_writers(model, declaration)

    def pairs(self, edits):
        if self.writers is not None:
            return [(self.writers[layer][component], self._hook(lam, v))
                    for layer, component, lam, v in edits]
        out = []
        for layer, component, lam, v in edits:
            modules = self.targets.get((layer, component))
            if not modules:
                raise ValueError(f"layer {layer} has no {component} output projection to hook")
            u = writer_direction(v, self.gammas.get((layer, component)))
            out += [(module, self._hook(lam, u)) for module in modules]
        return out

    @staticmethod
    def _hook(lam, v):
        mx = _mx()
        vec = mx.array(np.asarray(v, dtype=np.float32))

        def hook(out):
            y = out.astype(mx.float32)
            coefficient = (y @ vec)[..., None]
            return (y - lam * coefficient * vec).astype(out.dtype)

        return hook

    @contextmanager
    def active(self, edits):
        with _hooked(self.pairs(edits)):
            yield


# ------------------------------------------------------------ weight editing


def edit_weight(weight, v, lam, row_normalization):
    """Apply the directional edit to an ``[.., out, in]`` float weight (numpy).

    ``none``: ``W - lam v (v^T W)``. ``full``: the same edit on row-normalised
    weights, renormalised, then restored to the original row norms (the
    norm-preserving variant Heretic approximates with a rank-3 SVD; applied
    here exactly).
    """
    if row_normalization not in ROW_NORMALIZATIONS:
        raise ValueError(f"row_normalization must be one of {ROW_NORMALIZATIONS}")
    w = np.asarray(weight, dtype=np.float32)
    u = np.asarray(v, dtype=np.float32)
    if w.shape[-2] != u.shape[0]:
        raise ValueError("direction width must match the output dimension")
    if row_normalization == "none":
        return w - lam * u[:, None] * np.einsum("o,...oi->...i", u, w)[..., None, :]
    norms = np.linalg.norm(w, axis=-1, keepdims=True)
    unit = np.divide(w, norms, out=np.zeros_like(w), where=norms > 0)
    changed = unit - lam * u[:, None] * np.einsum("o,...oi->...i", u, unit)[..., None, :]
    changed_norms = np.linalg.norm(changed, axis=-1, keepdims=True)
    if np.any((changed_norms <= 1e-12) & (norms > 0)):
        raise ValueError("full normalization collapsed a nonzero row")
    return np.divide(changed, changed_norms, out=np.zeros_like(changed), where=changed_norms > 0) * norms


def _quant_params(module, fallback=None):
    params = {}
    for name in ("group_size", "bits", "mode"):
        value = getattr(module, name, None)
        if value is None and fallback:
            value = fallback.get(name)
        if value is not None:
            params[name] = value
    return params


def dense_weight(module):
    """Float32 numpy ``[.., out, in]`` weight of a (quantized) linear module."""
    mx = _mx()
    if "scales" in module:
        params = _quant_params(module)
        w = mx.dequantize(module["weight"], module["scales"], module.get("biases"), **params)
    else:
        w = module["weight"]
    w = w.astype(mx.float32)
    mx.eval(w)
    return np.asarray(w)


@contextmanager
def materialize(model, declaration, edits, row_normalization, *, requantize=False, receipts=None):
    """Swap edited output projections in as dense bf16 modules; restore after.

    The original module objects are kept and put back, so the loaded model
    returns to its exact original state on exit (verified by callers).

    ``requantize=True`` stores each edited quantized projection the way
    ``bake_checkpoint(edited_precision="requant")`` does (bf16 edit, then the
    module's own bits/group size/mode), so an A/B arm scores what the bake
    will serve.  Rank-two projections only (stacked banks raise).
    ``receipts`` (a list) collects per-projection ``edit_rms`` and
    ``requant_rms``.
    """
    import mlx.core as mx
    from mlx import nn

    by_layer = {(layer, component): (lam, v) for layer, component, lam, v in edits}
    gammas = post_norm_gammas(model, declaration)
    swaps = []
    try:
        for layer, component, path, parent, attr, module in resolve_linears(model, declaration):
            if (layer, component) not in by_layer:
                continue
            lam, v = by_layer[(layer, component)]
            v = writer_direction(v, gammas.get((layer, component)))
            original = dense_weight(module)
            weight = edit_weight(original, v, lam, row_normalization)
            quantized = "scales" in module
            dtype = mx.bfloat16 if quantized else module["weight"].dtype
            replacement = _dense_like(module, mx.array(weight).astype(dtype), nn)
            stored = replacement["weight"]
            if requantize and quantized:
                if weight.ndim != 2:
                    raise ValueError("requantized materialize supports rank-two projections only")
                params = _quant_params(module)
                # Same law as the bake: the edit in the scales' dtype, then
                # the module's own quantization parameters.
                source = _dense_like(module, mx.array(weight).astype(module["scales"].dtype), nn)
                replacement = nn.QuantizedLinear.from_linear(
                    source, group_size=params.get("group_size"), bits=params.get("bits"),
                    mode=params.get("mode", "affine"))
                stored = mx.dequantize(replacement["weight"], replacement["scales"],
                                       replacement.get("biases"), **params)
            if receipts is not None:
                edited = mx.array(weight)
                receipts.append({
                    "layer": layer, "component": component, "path": path, "lam": lam,
                    "stored": "requant" if requantize and quantized else str(dtype),
                    "edit_rms": float(mx.sqrt(mx.mean(mx.square(edited - mx.array(original)))).item()),
                    "requant_rms": float(mx.sqrt(mx.mean(mx.square(stored.astype(mx.float32) - edited))).item()),
                })
            swaps.append((parent, attr, module))
            parent[attr] = replacement
        yield len(swaps)
    finally:
        for parent, attr, module in reversed(swaps):
            parent[attr] = module


def _dense_like(module, weight, nn):
    bias = module.get("bias")
    if weight.ndim == 2:
        out_dim, in_dim = weight.shape
        dense = nn.Linear(in_dim, out_dim, bias=bias is not None)
        dense.weight = weight
        if bias is not None:
            dense.bias = bias
        return dense
    # Stacked experts: clone the module class's dense counterpart when it has
    # one (SwitchLinear); quantized switch modules expose ``to_dense`` rarely,
    # so build the dense switch layer explicitly.
    from mlx2.runtime.models.switch_layers import SwitchLinear

    experts, out_dim, in_dim = weight.shape
    dense = SwitchLinear(in_dim, out_dim, experts, bias=bias is not None)
    dense.weight = weight
    if bias is not None:
        dense.bias = bias
    return dense


# -------------------------------------------------------------------- baking


def _link_or_copy(src, dst):
    """Hardlink ``src`` to ``dst`` (byte copy when the filesystem refuses)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copyfile(src, dst)
        return "copy"


# Files a bake rewrites in the destination: never hardlinked, so editing the
# destination copy can never write through to the source inode.
_BAKE_REWRITTEN = ("config.json", "model.safetensors.index.json")


def _sha256(path, chunk=1 << 24):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def hash_tree(root):
    """sha256 of every regular file under ``root`` (symlinks resolved)."""
    root = Path(root)
    return {str(p.relative_to(root)): _sha256(p.resolve())
            for p in sorted(root.rglob("*")) if p.is_file() and not p.name.startswith(".")}


def _target_keys(index_keys, declaration):
    pattern = re.compile(declaration.layer_key)
    out = {}
    for key in index_keys:
        if not key.endswith(".weight"):
            continue
        match = pattern.search(key)
        if not match:
            continue
        rest = key[match.end():-len(".weight")]
        for component in COMPONENTS:
            if rest in declaration.checkpoint_paths(component):
                out[key] = (int(match.group(1)), component, key[:-len(".weight")])
    seen = {}
    for key, (layer, component, prefix) in out.items():
        slot = (layer, component, prefix[prefix.rfind(f"layers.{layer}."):])
        if slot in seen:
            raise ValueError(f"checkpoint keys {seen[slot]} and {key} both map to layer {layer} {component}")
        seen[slot] = key
    return out


_SAFETENSORS_DTYPES = {"F32": np.float32, "F16": np.float16, "BF16": np.uint16}


def read_safetensor(path, key):
    """One float tensor of a ``.safetensors`` file as float32 (header + one slice read)."""
    with open(path, "rb") as handle:
        size = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(size))
        entry = header.get(key)
        if entry is None:
            raise KeyError(f"{key} is not in {path}")
        dtype = _SAFETENSORS_DTYPES.get(entry["dtype"])
        if dtype is None:
            raise ValueError(f"{key}: unsupported safetensors dtype {entry['dtype']}")
        start, end = entry["data_offsets"]
        handle.seek(8 + size + start)
        raw = np.frombuffer(handle.read(end - start), dtype=dtype).reshape(entry["shape"])
    if entry["dtype"] == "BF16":
        return (raw.astype(np.uint32) << 16).view(np.float32)
    return raw.astype(np.float32)


def _checkpoint_post_norms(source, index, targets, declaration):
    """``{(layer, component): (norm key, gamma)}`` read from the source checkpoint."""
    if not declaration.post_norms:
        return {}
    found = {}
    for layer, component, prefix in targets.values():
        stem = prefix[:prefix.rfind(f"layers.{layer}.")] + f"layers.{layer}."
        rest = prefix[len(stem):]
        norm_key = stem + declaration.post_norms[rest] + ".weight"
        previous = found.get((layer, component))
        if previous is not None:
            if previous[0] != norm_key:
                raise ValueError(f"layer {layer} {component} writers name different post-norms")
            continue
        if norm_key not in index:
            raise ValueError(f"checkpoint has no post-norm {norm_key} for {prefix}")
        found[(layer, component)] = (norm_key, read_safetensor((source / index[norm_key]).resolve(), norm_key))
    return found


EDITED_PRECISIONS = ("requant", "bf16", "dense-bf16")


def bake_checkpoint(source, destination, declaration, edits, row_normalization, *, metadata,
                    edited_precision="requant"):
    """Write an edited copy of ``source`` into a new ``destination``.

    Only edited projection tensors change. ``requant`` re-quantizes them with
    the source's own parameters; ``dense-bf16`` stores rank-two projections
    in bf16 and re-quantizes stacked expert tensors (bf16 experts would add
    ~31 GB on Kimi Linear 48B); ``bf16`` stores every edited tensor unquantized (scales and
    biases dropped, index rewritten) because at 4 bits the rounding error can
    exceed a rank-1 edit (Qwen3.5-4B 2026-11-05: requant/edit RMS 1.5-5x,
    and the baked model lost most of the effect).  Loaders quantize a module
    only when its checkpoint carries ``.scales``. Every other file is copied byte
    for byte. Refuses an existing destination, a destination inside the
    source, or any change to the source's file hashes during the bake.
    """
    import mlx.core as mx

    if edited_precision not in EDITED_PRECISIONS:
        raise ValueError(f"edited_precision must be one of {EDITED_PRECISIONS}")
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or source in destination.parents or destination in source.parents:
        raise ValueError("destination must be outside the source checkpoint")
    if destination.exists():
        raise FileExistsError(f"refusing existing destination {destination}")
    before = hash_tree(source)
    config = json.loads((source / "config.json").read_text())
    quant = config.get("quantization") or config.get("quantization_config") or {}
    index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    targets = _target_keys(index, declaration)
    wanted = {(layer, component): (lam, v) for layer, component, lam, v in edits}
    # Post-normed writers: the edit direction depends on the norm gain, read
    # from the source checkpoint itself (not from any loaded model).
    norms = _checkpoint_post_norms(source, index, {k: s for k, s in targets.items() if s[:2] in wanted},
                                   declaration)
    wanted = {slot: (lam, writer_direction(v, norms[slot][1]) if slot in norms else v)
              for slot, (lam, v) in wanted.items()}
    staging = destination.with_name(destination.name + ".partial")
    if staging.exists():
        raise FileExistsError(f"refusing leftover staging directory {staging}")
    staging.mkdir(parents=True)
    receipts = []
    dropped = []
    by_shard = {}
    for key, spec in targets.items():
        if spec[:2] in wanted:
            by_shard.setdefault(index[key], []).append((key, spec))
    # Only shards holding edited tensors are rewritten; every other file is
    # hardlinked (expert, Engram, drafter and vision shards of a large pack
    # cost no disk), except the metadata files the bake edits.
    methods = {}
    for name in sorted(before):
        src = source / name
        if name in by_shard:
            continue
        if name in _BAKE_REWRITTEN:
            (staging / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src.resolve(), staging / name)
            methods[name] = "copy"
        else:
            methods[name] = _link_or_copy(src.resolve(), staging / name)
    for shard, items in sorted(by_shard.items()):
        tensors, shard_meta = mx.load(str((source / shard).resolve()), return_metadata=True)
        for key, (layer, component, prefix) in items:
            lam, v = wanted[(layer, component)]
            weight = tensors[key]
            scales = tensors.get(prefix + ".scales")
            override = quant.get(prefix) if isinstance(quant.get(prefix), dict) else {}
            params = {k: override.get(k, quant.get(k)) for k in ("group_size", "bits", "mode")
                      if override.get(k, quant.get(k)) is not None}
            if scales is not None:
                dense = mx.dequantize(weight, scales, tensors.get(prefix + ".biases"), **params)
            else:
                dense = weight
            original_dtype = (scales.dtype if scales is not None else weight.dtype)
            edited = edit_weight(np.asarray(dense.astype(mx.float32)), v, lam, row_normalization)
            edited_mx = mx.array(edited)
            keep_dense = edited_precision == "bf16" or (edited_precision == "dense-bf16" and weight.ndim == 2)
            if scales is not None and keep_dense:
                tensors[key] = edited_mx.astype(mx.bfloat16)
                for suffix in (".scales", ".biases"):
                    if tensors.pop(prefix + suffix, None) is not None:
                        dropped.append(prefix + suffix)
                roundtrip = tensors[key]
            elif scales is not None:
                packed = mx.quantize(edited_mx.astype(original_dtype), **params)
                tensors[key], tensors[prefix + ".scales"] = packed[0], packed[1]
                if len(packed) > 2 and packed[2] is not None:
                    tensors[prefix + ".biases"] = packed[2]
                roundtrip = mx.dequantize(*packed[:2], packed[2] if len(packed) > 2 else None, **params)
            else:
                tensors[key] = edited_mx.astype(weight.dtype)
                roundtrip = tensors[key]
            error = mx.sqrt(mx.mean(mx.square(roundtrip.astype(mx.float32) - edited_mx)))
            delta = mx.sqrt(mx.mean(mx.square(edited_mx - dense.astype(mx.float32))))
            receipts.append({"key": key, "layer": layer, "component": component, "lam": lam,
                             "edit_rms": float(delta.item()), "requant_rms": float(error.item()),
                             "quantized": scales is not None and not keep_dense,
                             "stored": "bf16" if scales is not None and keep_dense
                             else ("requant" if scales is not None else str(weight.dtype)),
                             "shape": [int(v) for v in edited.shape],
                             "params": params,
                             **({"post_norm": norms[(layer, component)][0]}
                                if (layer, component) in norms else {})})
        mx.save_safetensors(str(staging / shard), tensors, metadata=shard_meta or {"format": "mlx"})
        del tensors
    if dropped:
        index_path = staging / "model.safetensors.index.json"
        index_doc = json.loads((source / "model.safetensors.index.json").read_text())
        for key in dropped:
            index_doc["weight_map"].pop(key, None)
        index_path.write_text(json.dumps(index_doc, indent=2))
    edited_config = json.loads((staging / "config.json").read_text())
    record = dict(metadata, row_normalization=row_normalization, edited_tensors=len(receipts),
                  edited_precision=edited_precision,
                  unedited_writers=list(declaration.unedited_writers))
    edited_config["mlx2_residual_ablation"] = record
    # Keep the source's own derivation (e.g. a q2 requantization record) and
    # append this one: the new record names its predecessor in ``previous``.
    derivation = {"kind": "residual_ablation", **{k: record[k] for k in (
        "source_checkpoint", "directions_sha256", "trial", "row_normalization",
        "edited_precision", "edited_tensors", "unedited_writers") if k in record},
        "qualified": False}
    if "mlx2_derivation" in config:
        derivation["previous"] = config["mlx2_derivation"]
    edited_config["mlx2_derivation"] = derivation
    # A tensor stored unquantized must not keep a per-module quantization
    # override, or a loader that honours overrides would quantize it again.
    for section in ("quantization", "quantization_config"):
        overrides = edited_config.get(section)
        if isinstance(overrides, dict):
            for key in dropped:
                if key.endswith(".scales") and isinstance(overrides.get(key[:-len(".scales")]), dict):
                    overrides[key[:-len(".scales")]] = False
    (staging / "config.json").write_text(json.dumps(edited_config, indent=2))
    after = hash_tree(source)
    if after != before:
        shutil.rmtree(staging)
        raise RuntimeError("source checkpoint changed during bake; output discarded")
    os.replace(staging, destination)
    return {"source": str(source), "destination": str(destination),
            "source_hashes": before, "source_unchanged": True,
            "edited_tensors": receipts, "row_normalization": row_normalization,
            "edited_precision": edited_precision, "dropped_keys": dropped,
            "unedited_writers": list(declaration.unedited_writers),
            "linked": sum(v == "hardlink" for v in methods.values()),
            "copied": sum(v == "copy" for v in methods.values()),
            "rewritten_shards": sorted(by_shard)}


# ------------------------------------------------------------------ overlays

OVERLAY_SCHEMA = "mlx2.residual-ablation-overlay.v1"
OVERLAY_API_VERSION = 1
# Fields that do not affect behaviour and are excluded from the content id
# (docs/research/heretic-kimi-linear-20261105/OVERLAY-API.md section 1).
_OVERLAY_ID_EXCLUDED = ("id", "label", "source", "created", "qualification")


class OverlayBindingError(ValueError):
    """The overlay is bound to a different checkpoint (HTTP 409)."""


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def overlay_document_id(document):
    body = {k: v for k, v in document.items() if k not in _OVERLAY_ID_EXCLUDED}
    return "ovl-" + hashlib.sha256(_canonical(body).encode()).hexdigest()[:20]


@dataclass(frozen=True)
class Overlay:
    """A runtime ablation overlay: directions + strength plan for one checkpoint.

    Built from the overlay document (OVERLAY-API.md): a content-addressed JSON
    that references a capture's ``directions.npz`` by path and digests.  The
    id covers directions digests, parameters, binding and layer count, so any
    behavioural change yields a new id (prefix-cache identity includes it).
    """

    overlay_id: str
    label: str
    checkpoint: str
    fingerprint: str
    parameters: AblationParameters
    lam: np.ndarray  # [layers, components] float32
    vectors: np.ndarray  # [layers, components, hidden] float32 (zero rows where lam == 0)
    document: dict

    @property
    def layer_count(self):
        return self.lam.shape[0]

    @property
    def edits(self):
        return int(np.count_nonzero(self.lam))


def _parameters_from_dict(value):
    kernels = {name: ComponentKernel(**kernel) for name, kernel in value["kernels"].items()}
    if set(kernels) != set(COMPONENTS):
        raise ValueError("overlay parameters must cover attention and mlp")
    return AblationParameters(kernels=kernels, direction_index=value.get("direction_index"))


def compile_overlay(directions, parameters, layer_count):
    lam = np.zeros((layer_count, len(COMPONENTS)), np.float32)
    vectors = np.zeros((layer_count, len(COMPONENTS), directions.vectors.shape[1]), np.float32)
    for layer, component, strength, v in parameters.plan(directions, layer_count):
        index = COMPONENTS.index(component)
        lam[layer, index] = strength
        vectors[layer, index] = v
    return lam, vectors


def overlay_document(*, directions_path, checkpoint, parameters, layers, label, source=None):
    """Build an overlay document for a saved ``Directions`` file (``.npz``/``.json`` pair)."""
    npz = Path(directions_path).with_suffix(".npz")
    meta = json.loads(npz.with_suffix(".json").read_text())
    fingerprint = checkpoint.rsplit("@", 1)[1] if "@" in checkpoint else None
    if not fingerprint:
        raise ValueError("checkpoint must be '<artifact>@<fingerprint>'")
    params = parameters.as_dict() if isinstance(parameters, AblationParameters) else dict(parameters)
    directions = Directions.load(npz)
    body = {
        "schema": OVERLAY_SCHEMA, "api_version": OVERLAY_API_VERSION,
        "binding": {"checkpoint": checkpoint, "fingerprint": fingerprint},
        "directions": {"path": str(npz), "vectors_sha256": meta["sha256"],
                       "file_sha256": hashlib.sha256(npz.read_bytes()).hexdigest(),
                       "shape": meta.get("shape"), "capture": str(npz.parent)},
        "parameters": params, "row_normalization": "none", "layers": int(layers),
        "edits": len(_parameters_from_dict(params).plan(directions, int(layers))),
    }
    return {**body, "id": overlay_document_id(body), "label": str(label),
            "source": source or {"kind": "custom"}, "qualification": "unqualified"}


def overlay_from_document(document, *, fingerprint, layer_count, checkpoint=None):
    """Validate and compile an overlay document; fail closed on any mismatch.

    Raises ``OverlayBindingError`` when bound to another checkpoint and
    ``ValueError`` for every other defect (schema, id, digests, layers).
    """
    if not isinstance(document, dict) or document.get("schema") != OVERLAY_SCHEMA:
        raise ValueError("unknown overlay schema")
    if document.get("api_version") != OVERLAY_API_VERSION:
        raise ValueError("unsupported overlay api_version")
    if overlay_document_id(document) != document.get("id"):
        raise ValueError("overlay id does not match its content")
    binding = document.get("binding") or {}
    if binding.get("fingerprint") != fingerprint:
        raise OverlayBindingError(
            f"overlay is bound to {binding.get('checkpoint')}, not the served checkpoint {fingerprint}")
    if document.get("row_normalization") != "none":
        raise ValueError("runtime overlays support row_normalization 'none' only")
    if document.get("layers") != layer_count:
        raise ValueError("overlay layer count differs from the served model")
    ref = document.get("directions") or {}
    npz = Path(ref.get("path", ""))
    if not npz.is_file():
        raise ValueError("overlay directions file is missing on this host")
    if hashlib.sha256(npz.read_bytes()).hexdigest() != ref.get("file_sha256"):
        raise ValueError("overlay directions file digest differs")
    directions = Directions.load(npz)
    if hashlib.sha256(directions.vectors.tobytes()).hexdigest() != ref.get("vectors_sha256"):
        raise ValueError("overlay directions vectors digest differs")
    if directions.vectors.shape[0] != layer_count + 1:
        raise ValueError("overlay directions do not match the model's layer count")
    parameters = _parameters_from_dict(document["parameters"])
    lam, vectors = compile_overlay(directions, parameters, layer_count)
    used = vectors[lam != 0]
    if not np.all(np.isfinite(lam)) or not np.all(np.isfinite(used)):
        raise ValueError("overlay strengths or directions are not finite")
    if used.size and np.max(np.abs(np.linalg.norm(used, axis=-1) - 1)) > 1e-3:
        raise ValueError("overlay directions are not unit vectors")
    return Overlay(document["id"], str(document.get("label", "")),
                   checkpoint or binding.get("checkpoint", ""), fingerprint, parameters, lam, vectors,
                   dict(document))


def save_overlay(path, document):
    """Write an overlay document as ``<path>/overlay.json`` (new directory only)."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing existing overlay {path}")
    staging = path.with_name(path.name + ".partial")
    staging.mkdir(parents=True)
    (staging / "overlay.json").write_text(json.dumps(document, indent=1, sort_keys=True))
    os.replace(staging, path)
    return document["id"]


def row_projection(y, lam_rows, vectors_rows):
    """Per-row ``y - lam_b * v_b (v_b . y)`` for ``y`` of shape [B, S, H].

    ``lam_rows`` [B] and ``vectors_rows`` [B, H] (mx arrays).  Rows with
    lam 0 are returned unchanged up to the float32 round trip.
    """
    mx = _mx()
    y32 = y.astype(mx.float32)
    coefficient = mx.sum(y32 * vectors_rows[:, None, :], axis=-1, keepdims=True)
    return (y32 - lam_rows[:, None, None] * coefficient * vectors_rows[:, None, :]).astype(y.dtype)
