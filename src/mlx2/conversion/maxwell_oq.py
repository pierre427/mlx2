# SPDX-License-Identifier: Apache-2.0
"""Layer-streamed Maxwell-V1-Flash calibration for the pinned offline oQe quantizer.

Provenance: provenance/maxwell-oq8e.json.  Not a serving backend.

oQe weights each quantized weight's input channels by the activation energy
(``sum x^2``) it sees on a calibration draw.  The pinned quantizer has no
Maxwell layer walker (4-stream mHC residual, packed routed experts, an MTP
layer fed by the trunk's final streams), so this module measures that energy
on the original BF16 weights with mlx2's own Maxwell law, one layer resident
at a time, and hands the quantizer one entry per checkpoint module name:

* attention ``q/k/v/o_proj``, indexer ``linear_k_proj`` (its ``linear_q_proj``
  reads the same input; calibration rows stay in the dense regime, where the
  indexer query is never evaluated, so it reuses the key entry),
* dense-layer and shared-expert ``w1``/``w3`` (layer input) and ``w2``
  (SwiGLU intermediate),
* every routed expert's ``w1``/``w3`` (the tokens routed to it) and ``w2``
  (its own intermediate), with per-expert token counts,
* the MTP decoder layer the same way, and ``lm_head`` (final-normed hidden).

All captures use pristine weights and propagate the ordinary output to the
next layer.  The router, mHC mappings, hc_head and the MTP fusion projections
stay full precision (the driver's predicate), so they need no entry.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
import time
from pathlib import Path
from types import MethodType, SimpleNamespace

import mlx.core as mx
import numpy as np
from mlx import nn

from ..runtime.models.maxwell_moe import MTP, FinalNorm, Layer, Model, ModelArgs, SparseMoE
from ..runtime.ubc_evict import ubc_evict

LAYER_WALK = "mlx2-maxwell-bf16-mhc4-stream-walk-v1"


class Reader:
    """Read named tensors (BF16 or F32) from an inspected Maxwell artifact."""

    def __init__(self, artifact):
        self.path = Path(artifact["identity"]["path"])
        self.weight_map = artifact["weight_map"]
        self.stats = {name: (size, mtime) for name, size, mtime in artifact["identity"]["files"]}
        self.headers = {}
        for shard in sorted(set(self.weight_map.values())):
            with (self.path / shard).open("rb") as stream:
                size = int.from_bytes(stream.read(8), "little")
                self.headers[shard] = (json.loads(stream.read(size)), 8 + size)

    def read(self, name):
        shard = self.weight_map[name]
        path = self.path / shard
        st = path.stat()
        if (st.st_size, st.st_mtime_ns) != self.stats[shard]:
            raise ValueError(f"source changed during calibration: {shard}")
        header, payload = self.headers[shard]
        entry = header[name]
        dtype = {"BF16": ("<u2", mx.bfloat16), "F32": ("<f4", None)}.get(entry["dtype"])
        if dtype is None:
            raise ValueError(f"calibration requires the original BF16/F32 source: {name}")
        start, end = entry["data_offsets"]
        with path.open("rb") as stream:
            stream.seek(payload + start)
            raw = stream.read(end - start)
        if len(raw) != end - start:
            raise ValueError(f"truncated source tensor: {name}")
        array = np.frombuffer(raw, dtype[0]).reshape(entry["shape"])
        value = mx.array(array).view(dtype[1]) if dtype[1] is not None else mx.array(array)
        mx.eval(value)
        return value

    def evict(self):
        for shard in self.headers:
            ubc_evict(str(self.path / shard))

    def named(self, prefix):
        return {name: self.read(name) for name in self.weight_map if name.startswith(prefix)}


def _sanitize(args, weights, *, mtp=False):
    """Run Model.sanitize on a subset without allocating a whole model."""
    shim = SimpleNamespace(args=replace(args, mtp_num_layers=1 if mtp else 0), head_mode="fp32")
    shim._pack_moe = MethodType(Model._pack_moe, shim)
    return Model.sanitize(shim, weights)


def load_layer(reader, args, index):
    prefix = f"model.layers.{index}."
    weights = _sanitize(args, reader.named(prefix))
    layer = Layer(args, index)
    layer.load_weights([(k.removeprefix(prefix), v) for k, v in weights.items()], strict=True)
    layer.eval()
    return layer


def load_mtp(reader, args):
    weights = _sanitize(args, reader.named("mtp_layers.0."), mtp=True)
    mtp = MTP(args)
    mtp.load_weights([(k.removeprefix("mtp."), v) for k, v in weights.items()], strict=True)
    mtp.eval()
    return mtp


class Sink:
    """fp32 ``sum x^2`` per input channel and row counts, keyed by checkpoint module name."""

    def __init__(self):
        self.sums, self.counts = {}, {}

    def add(self, name, x):
        x = x.astype(mx.float32).reshape(-1, x.shape[-1])
        energy = mx.sum(x * x, axis=0)
        self.sums[name] = energy if name not in self.sums else self.sums[name] + energy
        self.counts[name] = self.counts.get(name, 0) + int(x.shape[0])

    def add_experts(self, names, energy, counts):
        """``energy`` [E, D] and ``counts`` [E] for one projection's experts."""
        key = tuple(names)
        self.sums[key] = energy if key not in self.sums else self.sums[key] + energy
        self.counts[key] = counts if key not in self.counts else self.counts[key] + counts

    def materialize(self, *outputs):
        # Capture branches are independent of the ordinary forward output.
        # Force both sides while this microbatch is resident, including routed counts.
        mx.eval(*outputs, list(self.sums.values()),
                [v for v in self.counts.values() if isinstance(v, mx.array)])

    def entries(self, oq):
        out = {}
        for key, energy in self.sums.items():
            energy = np.asarray(energy, dtype=np.float32)
            if not np.isfinite(energy).all() or (energy < 0).any():
                raise ValueError(f"invalid activation capture: {key}")
            if isinstance(key, tuple):
                counts = np.asarray(self.counts[key]).astype(np.int64)
                for expert, name in enumerate(key):
                    out[name] = oq.OQImatrixEntry(energy[expert].copy(), np.array([counts[expert]], np.int64))
            else:
                out[key] = oq.OQImatrixEntry(energy, np.array([self.counts[key]], np.int64))
        return out


class _Capture(nn.Module):
    """Record a Linear's input energy under its checkpoint name, then run it."""

    def __init__(self, inner, name, sink):
        super().__init__()
        self.inner = inner
        object.__setattr__(self, "_capture", (name, sink))

    def __call__(self, x):
        name, sink = self._capture
        sink.add(name, x)
        return self.inner(x)


class _CaptureMoE(nn.Module):
    """Routed-expert capture: per-expert inputs and SwiGLU intermediates."""

    def __init__(self, inner: SparseMoE, prefix, sink):
        super().__init__()
        self.inner = inner
        object.__setattr__(self, "_capture", (prefix, sink))

    def __call__(self, x):
        prefix, sink = self._capture
        moe = self.inner
        experts = moe.num_experts
        flat = x.reshape(-1, x.shape[-1])
        ids, _ = moe.gate(flat)
        onehot = (ids[..., None] == mx.arange(experts)).astype(mx.float32)        # [n, k, E]
        routed = onehot.sum(axis=1)                                               # [n, E]
        xf = flat.astype(mx.float32)
        energy_in = routed.T @ (xf * xf)                                           # [E, D]
        counts = routed.sum(axis=0)
        sw = moe.switch_mlp
        xe = mx.expand_dims(flat, (-2, -3))
        hidden = sw.activation(sw.up_proj(xe, ids), sw.gate_proj(xe, ids))
        hidden = hidden.reshape(flat.shape[0], ids.shape[-1], -1).astype(mx.float32)
        energy_mid = onehot.reshape(-1, experts).T @ (hidden * hidden).reshape(-1, hidden.shape[-1])
        names = lambda w: [f"{prefix}experts.{e}.{w}" for e in range(experts)]
        sink.add_experts(names("w1"), energy_in, counts)
        sink.add_experts(names("w3"), energy_in, counts)
        sink.add_experts(names("w2"), energy_mid, counts)
        return moe(x)


def _wrap_mlp(mlp, prefix, sink):
    """Install captures on one MLP (dense ``w1/w3/w2``) under ``prefix``."""
    for ours, theirs in (("gate_proj", "w1"), ("up_proj", "w3"), ("down_proj", "w2")):
        setattr(mlp, ours, _Capture(getattr(mlp, ours), prefix + theirs, sink))


def instrument(layer, prefix, sink):
    """Wrap every quantized projection of one decoder layer (checkpoint names)."""
    attn = layer.self_attn
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        setattr(attn, name, _Capture(getattr(attn, name), f"{prefix}self_attn.{name}", sink))
    if attn.sparse:
        attn.indexer.linear_k_proj = _Capture(attn.indexer.linear_k_proj,
                                              f"{prefix}self_attn.indexer.linear_k_proj", sink)
    moe_prefix = prefix + "block_sparse_moe."
    if isinstance(layer.mlp, SparseMoE):
        _wrap_mlp(layer.mlp.shared_expert, moe_prefix + "shared_expert.", sink)
        layer.mlp = _CaptureMoE(layer.mlp, moe_prefix, sink)
    else:
        _wrap_mlp(layer.mlp, moe_prefix, sink)


def _quantized_weights(layer):
    """(module, attribute) of every projection weight the conversion quantizes in one layer."""
    attn = layer.self_attn
    out = [(getattr(attn, n), "weight") for n in ("q_proj", "k_proj", "v_proj", "o_proj")]
    if attn.sparse:
        out += [(attn.indexer.linear_q_proj, "weight"), (attn.indexer.linear_k_proj, "weight")]
    mlp = layer.mlp
    dense = [mlp.shared_expert] if isinstance(mlp, SparseMoE) else [mlp]
    for m in dense:
        out += [(getattr(m, n), "weight") for n in ("gate_proj", "up_proj", "down_proj")]
    if isinstance(mlp, SparseMoE):
        out += [(getattr(mlp.switch_mlp, n), "weight") for n in ("gate_proj", "up_proj", "down_proj")]
    return out


def layer_sensitivity(layer, hidden, *, bits, group_size=64, micro_batch_size=4):
    """Relative MSE of one layer's output under quantize-dequantize of its quantized weights.

    Returns (score, pristine output).  The pristine output, never the
    perturbed one, is what the next layer sees."""
    run = lambda x: layer(x, "causal", None, None)
    reference = _forward(run, hidden, micro_batch_size=micro_batch_size)
    saved = []
    try:
        for module, name in _quantized_weights(layer):
            weight = getattr(module, name)
            if weight.shape[-1] % group_size:
                continue                         # oQ leaves such tensors unquantized too
            q = mx.quantize(weight, group_size=group_size, bits=bits)
            setattr(module, name, mx.dequantize(*q, group_size=group_size, bits=bits).astype(weight.dtype))
            saved.append((module, name, weight))
        perturbed = _forward(run, hidden, micro_batch_size=micro_batch_size)
        rf, qp = reference.astype(mx.float32), perturbed.astype(mx.float32)
        score = float((mx.mean((rf - qp) ** 2) / mx.maximum(mx.mean(rf ** 2), 1e-10)).item())
    finally:
        for module, name, weight in saved:
            setattr(module, name, weight)
    if not np.isfinite(score) or score <= 0:
        raise ValueError(f"invalid measured sensitivity: {score}")
    return score, reference


def _forward(module, inputs, *extra, micro_batch_size, sink=None):
    pieces = []
    for start in range(0, inputs.shape[0], micro_batch_size):
        out = module(inputs[start:start + micro_batch_size], *[e[start:start + micro_batch_size] for e in extra])
        if sink is None:
            mx.eval(out)
        else:
            sink.materialize(out)
        pieces.append(out)
    out = mx.concatenate(pieces, axis=0)
    mx.eval(out)
    return out


def _token_hash(tokens):
    return hashlib.sha256(np.asarray(tokens).tobytes()).hexdigest()


def stream_calibrate(reader, config, tokenizer, oq, *, num_samples=128, seq_length=512,
                     sensitivity_length=256, sensitivity_bits=8, sensitivity_group_size=64,
                     micro_batch_size=4, progress=None):
    """Measure oQe imatrix entries on the original BF16 Maxwell, one layer resident."""
    args = ModelArgs.from_dict({**config, "mtp_num_layers": 1})
    tokens = oq._load_builtin_calibration(tokenizer, oq._OQE_CALIB_DATASET, num_samples, seq_length)
    tokens = mx.array(np.asarray(tokens))
    if tokens.shape != (num_samples, seq_length):
        raise ValueError("calibration draw did not satisfy the declared profile")
    if seq_length > args.dense_context_limit:
        raise ValueError("calibration rows must stay in the dense regime (indexer-query entry reuse)")
    sensitivity_tokens = mx.array(np.asarray(
        oq._load_builtin_calibration(tokenizer, "code_multilingual", num_samples, sensitivity_length)))
    if sensitivity_tokens.shape != (num_samples, sensitivity_length):
        raise ValueError("sensitivity draw did not satisfy the declared profile")
    embedding = reader.read("model.embed_tokens.weight")
    n, width = args.num_residual_streams, args.hidden_size

    def expand(ids):
        h = embedding[ids]
        out = mx.broadcast_to(h[:, :, None, :], (*h.shape[:2], n, width)).reshape(*h.shape[:2], n * width)
        mx.eval(out)
        return out

    streams = expand(tokens)
    sensitivity_streams = expand(sensitivity_tokens)
    scores = {}
    entries = {}
    started = time.monotonic()
    layers = []
    for index in range(args.num_hidden_layers):
        layer = load_layer(reader, args, index)
        prefix = f"model.layers.{index}."
        # Sensitivity on the pristine layer, before any capture is installed.
        scores[index], sensitivity_streams = layer_sensitivity(
            layer, sensitivity_streams, bits=sensitivity_bits, group_size=sensitivity_group_size,
            micro_batch_size=micro_batch_size)
        sink = Sink()
        instrument(layer, prefix, sink)
        streams = _forward(lambda x: layer(x, "causal", None, None), streams,
                           micro_batch_size=micro_batch_size, sink=sink)
        key, query = f"{prefix}self_attn.indexer.linear_k_proj", f"{prefix}self_attn.indexer.linear_q_proj"
        sink.sums[query], sink.counts[query] = sink.sums[key], sink.counts[key]
        row = {"layer": index, "sensitivity": scores[index],
               "elapsed_seconds": round(time.monotonic() - started, 1)}
        layers.append(row)
        if progress:
            progress(row)
        entries.update(sink.entries(oq))
        del layer, sink
        reader.evict()
        mx.clear_cache()
    # MTP: trunk final streams at p with the token at p+1 (the serving pair law).
    mtp = load_mtp(reader, args)
    sink = Sink()
    instrument(mtp.layer, "mtp_layers.0.decoder_layer.", sink)
    nxt = embedding[tokens[:, 1:]]
    _forward(lambda s, e: mtp(s, e, None), streams[:, :-1], nxt, micro_batch_size=micro_batch_size, sink=sink)
    del mtp, nxt
    # lm_head reads the final-normed hidden.
    norm = FinalNorm(args)
    norm.load_weights([("weight", reader.read("model.norm.weight")),
                       ("hc_head_fn", reader.read("model.hc_head.hc_head_fn").astype(mx.float32)),
                       ("hc_head_base", reader.read("model.hc_head.hc_head_base")),
                       ("hc_head_scale", reader.read("model.hc_head.hc_head_scale"))], strict=True)
    for start in range(0, num_samples, micro_batch_size):
        sink.add("lm_head", norm(streams[start:start + micro_batch_size]))
        sink.materialize()
    entries.update(sink.entries(oq))
    metadata = {
        "layer_walk": LAYER_WALK, "load_kind": "streaming", "proxy_used": False,
        "processed_samples": num_samples, "imatrix_sequence_length": seq_length,
        "micro_batch_size": micro_batch_size, "imatrix_tokens_sha256": _token_hash(np.asarray(tokens)),
        "sensitivity_samples": num_samples, "sensitivity_sequence_length": sensitivity_length,
        "sensitivity_bits": sensitivity_bits, "sensitivity_accumulation": "fp32-relative-mse",
        "sensitivity_tokens_sha256": _token_hash(np.asarray(sensitivity_tokens)), "sensitivity_map": scores,
        "layers": layers, "elapsed_seconds": round(time.monotonic() - started, 1),
        "entry_count": len(entries),
    }
    return entries, metadata


def expected_entries(config) -> set[str]:
    """Every checkpoint module the oQ8e driver quantizes, so every imatrix entry it needs."""
    names = {"lm_head"}
    experts = range(int(config["num_experts"]))

    def layer(prefix, dense, sparse):
        out = {f"{prefix}self_attn.{p}" for p in ("q_proj", "k_proj", "v_proj", "o_proj")}
        if sparse:
            out |= {f"{prefix}self_attn.indexer.linear_q_proj", f"{prefix}self_attn.indexer.linear_k_proj"}
        moe = prefix + "block_sparse_moe."
        if dense:
            return out | {moe + w for w in ("w1", "w2", "w3")}
        out |= {f"{moe}shared_expert.{w}" for w in ("w1", "w2", "w3")}
        return out | {f"{moe}experts.{e}.{w}" for e in experts for w in ("w1", "w2", "w3")}

    dense = set(config["mlp_only_layers"])
    for index in range(int(config["num_hidden_layers"])):
        names |= layer(f"model.layers.{index}.", index in dense, True)
    if int(config.get("mtp_num_layers", 0)):
        names |= layer("mtp_layers.0.decoder_layer.", False, False)
    return names


def keep_full_precision(path: str) -> bool:
    """Tensors the Maxwell oQ8e conversion never quantizes (model law / MTP acceptance)."""
    base = path.removesuffix(".weight")
    return ("hyper_connection" in base or "hc_head" in base or base.endswith("expert_bias")
            or base.endswith(("mtp_layers.0.e_proj", "mtp_layers.0.h_proj"))
            or base.endswith(".block_sparse_moe.gate"))
