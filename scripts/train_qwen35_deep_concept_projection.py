#!/usr/bin/env python3
"""Fit recurrent concept projections to a frozen Qwen residual stream.

The recurrent encoder stays fixed.  Ridge regression maps its key states to
actual post-layer query states and its value states to Qwen output-head rows.
No 9B parameter is updated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def _episode_graph(row):
    concepts = {
        "subject": {"label": row["subject"]},
        "object": {"label": row["object"]},
    }
    edges = []
    if row["intermediate"]:
        concepts["middle"] = {"label": row["intermediate"]}
        edges.append(
            {
                "subject": "subject",
                "relation": "related_to",
                "object": "middle",
                "authority": "committed",
            }
        )
        edges.append(
            {
                "subject": "middle",
                "relation": row["relation"],
                "object": "object",
                "authority": "committed",
            }
        )
    else:
        edges.append(
            {
                "subject": "subject",
                "relation": row["relation"],
                "object": "object",
                "authority": "committed",
            }
        )
    return {"concepts": concepts, "edges": edges}


def _normalize(rows):
    return rows / np.maximum(np.linalg.norm(rows, axis=-1, keepdims=True), 1e-6)


def _hidden_query(adapter, prompt_ids, layer_index):
    import mlx.core as mx

    from mlx2.runtime.models.base import create_attention_mask, create_ssm_mask

    trunk = adapter.model.language_model.model
    hidden = trunk.embed_tokens(mx.array([prompt_ids]))
    fa_mask = create_attention_mask(hidden, None)
    ssm_mask = create_ssm_mask(hidden, None)
    for index, layer in enumerate(trunk.layers):
        hidden = layer(hidden, mask=ssm_mask if layer.is_linear else fa_mask)
        if index == layer_index:
            result = np.asarray(hidden[0, -1].astype(mx.float32))
            mx.clear_cache()
            return result
    raise ValueError("requested layer is outside the Qwen trunk")


def _output_directions(adapter, answer):
    import mlx.core as mx

    ids = list(adapter.tokenizer.encode(answer, add_special_tokens=False))
    head = adapter.model.language_model.lm_head
    rows = [head.weight[mx.array(ids)], head.scales[mx.array(ids)]]
    if head.biases is not None:
        rows.append(head.biases[mx.array(ids)])
    dense = mx.dequantize(
        *rows,
        group_size=head.group_size,
        bits=head.bits,
        mode=head.mode,
    )
    result = np.asarray(dense.astype(mx.float32))
    return result


def _ridge(source, target, regularization):
    gram = source.T @ source
    gram.flat[:: gram.shape[0] + 1] += regularization
    return np.linalg.solve(gram, source.T @ target).astype(np.float32)


def _metrics(key_state, value_state, query_target, output_target, key_projection, value_projection):
    keys = _normalize(key_state @ key_projection)
    values = _normalize(value_state @ value_projection)
    queries = _normalize(query_target)
    outputs = _normalize(output_target)
    return {
        "key_cosine": float(np.mean(np.sum(keys * queries, axis=-1))),
        "value_cosine": float(np.mean(np.sum(values * outputs, axis=-1))),
        "retrieval_top1": float(np.mean(np.argmax(queries @ keys.T, axis=-1) == np.arange(len(keys)))),
    }


def _decode_metrics(value_state, targets, lengths, projections):
    result = []
    for step, projection in enumerate(projections):
        keep = lengths > step
        predicted = _normalize(value_state[keep] @ projection)
        expected = _normalize(targets[keep, step])
        result.append(float(np.mean(np.sum(predicted * expected, axis=-1))))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=31)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--relative-gate", type=float, default=0.2)
    parser.add_argument("--max-decode-steps", type=int, default=4)
    args = parser.parse_args()
    if not 0 <= args.layer < 32:
        parser.error("--layer must be in 0..31")
    if not math.isfinite(args.ridge) or args.ridge <= 0:
        parser.error("--ridge must be positive and finite")
    if not 0 < args.relative_gate < 1:
        parser.error("--relative-gate must be inside (0, 1)")
    if not 1 <= args.max_decode_steps <= 8:
        parser.error("--max-decode-steps must be in 1..8")

    from mlx2.adapters.qwen35_9b import Qwen359BAdapter
    from mlx2.runtime.neural_concepts import NeuralConceptArtifact, RecurrentConceptEncoder

    source_manifest = json.loads((args.source_artifact / "manifest.json").read_text())
    artifact = NeuralConceptArtifact.load(
        args.source_artifact,
        model_binding=source_manifest["bindings"]["model"],
        tokenizer_binding=source_manifest["bindings"]["tokenizer"],
        runtime_binding=source_manifest["bindings"]["runtime"],
    )
    world = json.loads((args.source_artifact / "micro_world.json").read_text())
    adapter = Qwen359BAdapter(str(args.model))
    encoder = RecurrentConceptEncoder(artifact)
    captured = {}
    for split, rows in world["splits"].items():
        values = []
        for index, row in enumerate(rows):
            encoded = {item.concept_id: item for item in encoder.encode_graph(_episode_graph(row))}
            prompt_ids = list(
                adapter.prompt_tokens(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": row["query"] + " Answer with only the aroma.",
                            }
                        ],
                        "enable_thinking": False,
                        "reasoning_effort": "none",
                    }
                )
            )
            directions = _output_directions(adapter, row["answer"])
            if len(directions) > args.max_decode_steps:
                raise ValueError(
                    f"answer needs {len(directions)} decode steps; "
                    f"increase --max-decode-steps"
                )
            padded_directions = np.zeros(
                (args.max_decode_steps, directions.shape[-1]), dtype=np.float32
            )
            padded_directions[: len(directions)] = directions
            values.append(
                (
                    encoded["subject"].key_state,
                    encoded["subject"].value_state,
                    _hidden_query(adapter, prompt_ids, args.layer),
                    directions[0],
                    padded_directions,
                    len(directions),
                )
            )
            if (index + 1) % 20 == 0:
                print(f"capture split={split} rows={index + 1}", flush=True)
        captured[split] = tuple(
            np.asarray([row[column] for row in values], dtype=np.float32)
            for column in range(6)
        )

    train = captured["train"]
    key_projection = _ridge(train[0], _normalize(train[2]), args.ridge)
    value_projection = _ridge(train[1], _normalize(train[3]), args.ridge)
    decode_value_projections = np.stack(
        [
            _ridge(
                train[1][train[5] > step],
                _normalize(train[4][train[5] > step, step]),
                args.ridge,
            )
            for step in range(args.max_decode_steps)
        ]
    )
    metrics = {
        split: {
            **_metrics(*values[:4], key_projection, value_projection),
            "decode_value_cosine_by_step": _decode_metrics(
                values[1], values[4], values[5], decode_value_projections
            ),
        }
        for split, values in captured.items()
    }
    arrays = {name: np.array(value, copy=True) for name, value in artifact.arrays.items()}
    arrays["key_projection"] = key_projection
    arrays["value_projection"] = value_projection
    arrays["decode_value_projections"] = decode_value_projections
    arrays["output_gate"] = np.asarray(
        [math.log(args.relative_gate / (1.0 - args.relative_gate))], dtype=np.float32
    )
    args.output.mkdir(parents=True, exist_ok=True)
    weights_path = args.output / "weights.npz"
    np.savez(weights_path, **arrays)
    manifest = dict(source_manifest)
    manifest.pop("fingerprint", None)
    manifest["weights_sha256"] = hashlib.sha256(weights_path.read_bytes()).hexdigest()
    manifest["deep_injection_layer"] = args.layer
    manifest["deep_selection"] = "directory_top1"
    manifest["training"] = {
        "method": "frozen-qwen-residual-and-token-capsule-output-head-ridge",
        "source_artifact_fingerprint": artifact.fingerprint,
        "layer": args.layer,
        "ridge": args.ridge,
        "relative_gate": args.relative_gate,
        "max_decode_steps": args.max_decode_steps,
        "split_sizes": {name: len(rows) for name, rows in world["splits"].items()},
        "metrics": metrics,
    }
    encoded_manifest = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["fingerprint"] = hashlib.sha256(
        encoded_manifest + weights_path.read_bytes()
    ).hexdigest()
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    (args.output / "micro_world.json").write_text(
        json.dumps(world, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"artifact": str(args.output), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
