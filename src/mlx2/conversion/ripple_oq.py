# SPDX-License-Identifier: Apache-2.0
"""Layer-streamed Ripple calibration for the pinned offline oQe quantizer.

Provenance: provenance/ripple-oq4e.json. This module is not a serving backend.
"""

from __future__ import annotations

import gc
import hashlib
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

from ..runtime.models.ricursive import DecoderLayer, ModelArgs, RMSNorm, scale_embeddings
from ..runtime.ubc_evict import ubc_evict
from .nvfp4 import read_header

OQ_REVISION = "4213c3284abc7b3746a91464a7af7ccc3732b759"
LAYER_WALK = "mlx2-ripple-bf16-scaled-embedding-valres-v1"


class BF16Reader:
    """Read only named tensors, preserving their inspected file identity."""

    def __init__(self, artifact):
        self.path = Path(artifact["identity"]["path"])
        self.weight_map = artifact["weight_map"]
        self.stats = {name: (size, mtime) for name, size, mtime in artifact["identity"]["files"]}
        self.headers = {name: read_header(self.path / name)
                        for name in sorted(set(self.weight_map.values()))}

    def read(self, name):
        shard = self.weight_map[name]
        path = self.path / shard
        st = path.stat()
        if (st.st_size, st.st_mtime_ns) != self.stats[shard]:
            raise ValueError(f"source changed during calibration: {shard}")
        header, payload = self.headers[shard]
        entry = header[name]
        if entry["dtype"] != "BF16":
            raise ValueError(f"calibration requires original BF16: {name}")
        start, end = entry["data_offsets"]
        with path.open("rb") as stream:
            stream.seek(payload + start)
            raw = stream.read(end - start)
        if len(raw) != end - start:
            raise ValueError(f"truncated source tensor: {name}")
        value = mx.array(np.frombuffer(raw, "<u2").reshape(entry["shape"])).view(mx.bfloat16)
        mx.eval(value)
        ubc_evict(str(path))
        return value

    def layer(self, args, index):
        prefix = f"model.layers.{index}."
        block = DecoderLayer(args, index, nvfp4=False)
        block.load_weights([(name[len(prefix):], self.read(name))
                            for name in self.weight_map if name.startswith(prefix)], strict=True)
        block.eval()
        return block


def forward_microbatches(block, inputs, seed, micro_batch_size=1):
    """Preserve each sample's original seed across all layers and partitions."""
    if inputs.shape != seed.shape or micro_batch_size < 1:
        raise ValueError("Ripple calibration input/seed geometry differs")
    pieces = []
    for start in range(0, inputs.shape[0], micro_batch_size):
        stop = start + micro_batch_size
        out = block(inputs[start:stop], seed[start:stop], None)
        mx.eval(out)
        pieces.append(out)
    result = mx.concatenate(pieces, axis=0)
    mx.eval(result)
    return result


def _token_hash(tokens):
    return hashlib.sha256(np.asarray(tokens).tobytes()).hexdigest()


def stream_calibrate(reader, config, tokenizer, oq, *, evidence, micro_batch_size=1,
                     num_samples=128, imatrix_length=512, sensitivity_length=256,
                     progress=None):
    """Measure on original BF16; no quantized proxy and no full model allocation.

    All captures use pristine weights. QDQ scores propagate the ordinary
    reference output, never the perturbed output, into the next layer.
    """
    args = ModelArgs.from_dict(config)
    # Call the built-in loader directly: corpus failure must not select a
    # different downloaded dataset or silently reduce calibration coverage.
    imatrix_tokens = oq._load_builtin_calibration(
        tokenizer, oq._OQE_CALIB_DATASET, num_samples, imatrix_length)
    sensitivity_tokens = oq._load_builtin_calibration(
        tokenizer, "code_multilingual", num_samples, sensitivity_length)
    if (imatrix_tokens.shape != (num_samples, imatrix_length)
            or sensitivity_tokens.shape != (num_samples, sensitivity_length)):
        raise ValueError("calibration draw did not satisfy the declared profile")
    prompts = ["The capital of France is", "2 + 2 =", "The opposite of hot is"]
    reference_tokens = [mx.array([tokenizer.encode(text, add_special_tokens=False)]) for text in prompts]
    embedding = reader.read("model.embed_tokens.weight")

    def embed(weight, tokens):
        x = scale_embeddings(weight[tokens], args.embed_scale)
        mx.eval(x)
        return x

    seed = embed(embedding, imatrix_tokens)
    sensitivity_seed = embed(embedding, sensitivity_tokens)
    reference_seed = [embed(embedding, tokens) for tokens in reference_tokens]
    del embedding
    hidden, sensitivity_hidden, reference_hidden = seed, sensitivity_seed, list(reference_seed)
    entries, scores, layer_receipts = {}, {}, []
    started = time.monotonic()
    for index in range(args.num_hidden_layers):
        block = reader.layer(args, index)
        collector = oq.OQImatrixCollector()
        wrapped = collector.install(block, name_prefix=f"model.layers.{index}.")
        if wrapped != 8:
            raise ValueError(f"Ripple layer {index} must capture all eight projections")
        try:
            hidden = forward_microbatches(block, hidden, seed, micro_batch_size)
        finally:
            collector.restore(block)
        if len(collector.entries) != 8:
            raise ValueError(f"incomplete Ripple layer {index} imatrix")
        for name, entry in collector.entries.items():
            if (entry.counts.tolist() != [num_samples * imatrix_length]
                    or not np.isfinite(entry.in_sum2).all() or (entry.in_sum2 < 0).any()):
                raise ValueError(f"invalid activation capture: {name}")
        entries.update(collector.entries)
        reference = forward_microbatches(block, sensitivity_hidden, sensitivity_seed, micro_batch_size)
        saved = oq._temporary_quantize_block(block, config, 4, 64, layer_idx=index)
        if len(saved) != 8:
            raise ValueError(f"incomplete Ripple sensitivity perturbation: layer {index}")
        try:
            perturbed = forward_microbatches(block, sensitivity_hidden, sensitivity_seed, micro_batch_size)
            # Accumulate the same relative-MSE metric in FP32 to prevent
            # BF16 squaring/reduction from underflowing a small layer score.
            rf, qp = reference.astype(mx.float32), perturbed.astype(mx.float32)
            score = float((mx.mean((rf - qp) ** 2) / mx.maximum(mx.mean(rf ** 2), 1e-10)).item())
            if not np.isfinite(score) or score <= 0:
                raise ValueError(f"invalid measured sensitivity for layer {index}: {score}")
            scores[index] = score
        finally:
            oq._restore_saved_weights(block, saved)
        sensitivity_hidden = reference
        reference_hidden = [forward_microbatches(block, h, s, 1)
                            for h, s in zip(reference_hidden, reference_seed, strict=True)]
        row = {"layer": index, "sensitivity": score, "captured_modules": wrapped,
               "tokens_per_module": num_samples * imatrix_length,
               "elapsed_seconds": time.monotonic() - started}
        layer_receipts.append(row)
        if progress:
            progress(row)
        del block, collector, saved, reference, perturbed, rf, qp
        gc.collect()
        mx.synchronize()
        mx.clear_cache()

    norm = RMSNorm(args.hidden_size, args.rms_norm_eps)
    norm.weight = reader.read("model.norm.weight")
    # lm_head consumes final-normalized hidden states; the embedding table
    # is gathered, so oQe intentionally does not apply a Linear imatrix to it.
    sums = np.zeros(args.hidden_size, dtype=np.float32)
    for start in range(0, num_samples, micro_batch_size):
        x = np.asarray(norm(hidden[start:start + micro_batch_size]).astype(mx.float32))
        sums += np.square(x).reshape(-1, args.hidden_size).sum(axis=0)
    if not np.isfinite(sums).all() or (sums < 0).any():
        raise ValueError("invalid final-normalized lm_head activation capture")
    entries["lm_head"] = oq.OQImatrixEntry(sums, np.array([num_samples * imatrix_length], np.int64))
    head = reader.read("lm_head.weight")
    reference_rows, reference_arrays = [], {}
    for index, (text, tokens, h) in enumerate(zip(prompts, reference_tokens, reference_hidden, strict=True)):
        logits = (norm(h[:, -1:]) @ head.T).astype(mx.float32)
        mx.eval(logits)
        values = np.asarray(logits).reshape(-1)
        order = np.argsort(values)[-32:][::-1]
        reference_arrays[f"case_{index}"] = values
        reference_rows.append({"prompt": text, "token_ids": np.asarray(tokens).tolist()[0],
                               "top_tokens": order.tolist(), "top_logits": values[order].tolist(),
                               "top_text": tokenizer.decode([int(order[0])])})
    np.savez_compressed(evidence / "bf16-reference-logits.npz", **reference_arrays)
    (evidence / "bf16-reference.json").write_text(json.dumps(reference_rows, indent=2) + "\n")
    metadata = {
        "layer_walk": LAYER_WALK, "load_kind": "streaming", "proxy_used": False,
        "processed_samples": num_samples, "imatrix_sequence_length": imatrix_length,
        "sensitivity_samples": num_samples, "sensitivity_sequence_length": sensitivity_length,
        "sensitivity_accumulation": "fp32-relative-mse", "micro_batch_size": micro_batch_size,
        "imatrix_tokens_sha256": _token_hash(imatrix_tokens),
        "sensitivity_tokens_sha256": _token_hash(sensitivity_tokens),
        "sensitivity_map": scores, "layers": layer_receipts,
        "expected_entry_count": 8 * args.num_hidden_layers + 1,
        "elapsed_seconds": time.monotonic() - started,
    }
    return entries, metadata
