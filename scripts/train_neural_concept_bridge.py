#!/usr/bin/env python3
"""Train the compact recurrent concept encoder against frozen Qwen embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np

from mlx2.adapters.qwen35_9b import inspect_artifact
from mlx2.runtime.concept_micro_world import generate_micro_world, split_micro_world
from mlx2.runtime.neural_concepts import NEURAL_CONCEPT_SCHEMA, label_features
from mlx2.runtime.semantic_memory import RELATIONS


def token_embedding_rows(model_path: Path, token_ids):
    """Read and affine-dequantize only requested Qwen embedding rows."""
    import torch
    from safetensors import safe_open

    index = json.loads((model_path / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    prefix = "language_model.model.embed_tokens."
    names = {suffix: index[prefix + suffix] for suffix in ("weight", "scales", "biases")}
    if len(set(names.values())) != 1:
        raise ValueError("embedding quantization tensors must share one shard")
    rows = sorted({int(value) for value in token_ids})
    with safe_open(model_path / names["weight"], framework="pt", device="cpu") as stored:
        packed = torch.stack([stored.get_slice(prefix + "weight")[row] for row in rows])
        scales = torch.stack([stored.get_slice(prefix + "scales")[row] for row in rows])
        biases = torch.stack([stored.get_slice(prefix + "biases")[row] for row in rows])
    packed = packed.to(torch.int64)
    shifts = torch.arange(8, dtype=torch.int64) * 4
    quantized = ((packed[..., None] >> shifts) & 15).reshape(len(rows), -1).float()
    values = quantized * scales.float().repeat_interleave(64, dim=-1)
    values += biases.float().repeat_interleave(64, dim=-1)
    return {row: values[index].numpy() for index, row in enumerate(rows)}


def mean_embedding(tokenizer, rows, text):
    ids = list(tokenizer.encode(text, add_special_tokens=False))
    if not ids:
        raise ValueError(f"label tokenized empty: {text!r}")
    result = np.mean([rows[int(token)] for token in ids], axis=0, dtype=np.float32)
    norm = float(np.linalg.norm(result))
    return result if norm == 0 else result / norm


def cosine_loss(prediction, target):
    from torch.nn import functional

    return 1.0 - functional.cosine_similarity(prediction, target, dim=-1).mean()


class RecurrentBridgeModel:
    """Thin wrapper so torch remains an optional training-only dependency."""

    def __init__(self, feature_dim, state_dim, hidden_dim, relation_count, seed):
        import torch

        torch.manual_seed(seed)
        scale = 1.0 / math.sqrt(state_dim)
        self.parameters = {
            "input_weight": torch.nn.Parameter(
                torch.randn(feature_dim, state_dim) / math.sqrt(feature_dim)
            ),
            "input_bias": torch.nn.Parameter(torch.zeros(state_dim)),
            "relation_weight": torch.nn.Parameter(
                torch.eye(state_dim)[None].repeat(relation_count, 1, 1)
                + torch.randn(relation_count, state_dim, state_dim) * (0.05 * scale)
            ),
            "gru_z_weight": torch.nn.Parameter(torch.randn(2 * state_dim, state_dim) * scale),
            "gru_z_bias": torch.nn.Parameter(torch.zeros(state_dim)),
            "gru_r_weight": torch.nn.Parameter(torch.randn(2 * state_dim, state_dim) * scale),
            "gru_r_bias": torch.nn.Parameter(torch.zeros(state_dim)),
            "gru_h_weight": torch.nn.Parameter(torch.randn(2 * state_dim, state_dim) * scale),
            "gru_h_bias": torch.nn.Parameter(torch.zeros(state_dim)),
            "key_projection": torch.nn.Parameter(torch.randn(state_dim, hidden_dim) * scale),
            "value_projection": torch.nn.Parameter(torch.randn(state_dim, hidden_dim) * scale),
        }

    def tensors(self):
        return list(self.parameters.values())

    def __call__(self, features, relations, edge_mask, rounds=2):
        import torch

        p = self.parameters
        base = torch.tanh(features @ p["input_weight"] + p["input_bias"])
        state = base
        for _ in range(rounds):
            messages = torch.zeros_like(state)
            counts = torch.zeros((*state.shape[:2], 1), dtype=state.dtype)
            # Fixed micro-world topology: node 0 -> 1 and optional node 1 -> 2.
            for source, target, slot in ((0, 1, 0), (1, 2, 1)):
                relation = relations[:, slot]
                matrix = p["relation_weight"][relation]
                message = torch.bmm(state[:, target : target + 1], matrix).squeeze(1)
                active = edge_mask[:, slot : slot + 1]
                messages[:, source] += message * active
                counts[:, source] += active
            messages = messages / torch.clamp(counts, min=1.0)
            joined = torch.cat([messages, state], dim=-1)
            z = torch.sigmoid(joined @ p["gru_z_weight"] + p["gru_z_bias"])
            r = torch.sigmoid(joined @ p["gru_r_weight"] + p["gru_r_bias"])
            candidate = torch.tanh(
                torch.cat([messages, r * state], dim=-1) @ p["gru_h_weight"]
                + p["gru_h_bias"]
            )
            updated = (1.0 - z) * state + z * candidate
            state = torch.where(counts > 0, updated, state)
        return base, state


def make_training_rows(episodes, tokenizer, embedding_rows, feature_dim, relation_ids):
    features, relations, masks, key_targets, value_targets = [], [], [], [], []
    for episode in episodes:
        middle = episode.intermediate or episode.object
        labels = (episode.subject, middle, episode.object)
        features.append(np.stack([label_features(label, feature_dim) for label in labels]))
        relations.append(
            [relation_ids["related_to" if episode.hops == 2 else episode.relation], relation_ids[episode.relation]]
        )
        masks.append([1.0, 1.0 if episode.hops == 2 else 0.0])
        key_targets.append(
            0.5
            * (
                mean_embedding(tokenizer, embedding_rows, episode.subject)
                + mean_embedding(tokenizer, embedding_rows, episode.subject_alias)
            )
        )
        value_targets.append(mean_embedding(tokenizer, embedding_rows, episode.answer))
    return tuple(
        np.asarray(value, dtype=dtype)
        for value, dtype in (
            (features, np.float32),
            (relations, np.int64),
            (masks, np.float32),
            (key_targets, np.float32),
            (value_targets, np.float32),
        )
    )


def evaluate(model, tensors):
    import torch
    from torch.nn import functional

    features, relations, masks, key_target, value_target = [torch.from_numpy(v) for v in tensors]
    with torch.no_grad():
        base, state = model(features, relations, masks)
        keys = base[:, 0] @ model.parameters["key_projection"]
        values = state[:, 0] @ model.parameters["value_projection"]
        key_cos = functional.cosine_similarity(keys, key_target, dim=-1)
        value_cos = functional.cosine_similarity(values, value_target, dim=-1)
        logits = functional.normalize(key_target, dim=-1) @ functional.normalize(keys, dim=-1).T
        recall = (logits.argmax(dim=-1) == torch.arange(len(keys))).float().mean()
    return {
        "key_cosine": float(key_cos.mean()),
        "value_cosine": float(value_cos.mean()),
        "retrieval_top1": float(recall),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-binding", required=True)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--feature-dim", type=int, default=128)
    parser.add_argument("--state-dim", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    args = parser.parse_args()

    import torch
    from torch.nn import functional
    from transformers import AutoTokenizer

    random.seed(args.seed)
    np.random.seed(args.seed)
    model_path = args.model.expanduser().resolve()
    artifact = inspect_artifact(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=False
    )
    split = split_micro_world(
        generate_micro_world(seed=args.seed, count=args.episodes)
    )
    phrases = {
        value
        for rows in split.values()
        for episode in rows
        for value in (
            episode.subject,
            episode.subject_alias,
            episode.intermediate,
            episode.object,
        )
        if value
    }
    token_ids = {
        int(token)
        for phrase in phrases
        for token in tokenizer.encode(phrase, add_special_tokens=False)
    }
    embedding_rows = token_embedding_rows(model_path, token_ids)
    hidden_dim = len(next(iter(embedding_rows.values())))
    relation_order = tuple(sorted(RELATIONS))
    relation_ids = {name: index for index, name in enumerate(relation_order)}
    datasets = {
        name: make_training_rows(
            rows, tokenizer, embedding_rows, args.feature_dim, relation_ids
        )
        for name, rows in split.items()
    }
    model = RecurrentBridgeModel(
        args.feature_dim, args.state_dim, hidden_dim, len(relation_order), args.seed
    )
    optimizer = torch.optim.AdamW(model.tensors(), lr=args.learning_rate, weight_decay=1e-4)
    train = datasets["train"]
    generator = torch.Generator().manual_seed(args.seed)
    for step in range(args.steps):
        indices = torch.randint(
            len(train[0]), (min(args.batch_size, len(train[0])),), generator=generator
        )
        features, relations, masks, key_target, value_target = [
            torch.from_numpy(value)[indices] for value in train
        ]
        base, state = model(features, relations, masks)
        keys = base[:, 0] @ model.parameters["key_projection"]
        values = state[:, 0] @ model.parameters["value_projection"]
        logits = functional.normalize(key_target, dim=-1) @ functional.normalize(keys, dim=-1).T
        labels = torch.arange(len(indices))
        loss = cosine_loss(keys, key_target) + 1.5 * cosine_loss(values, value_target)
        loss += 0.2 * functional.cross_entropy(logits / 0.07, labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.tensors(), 1.0)
        optimizer.step()
        if step % 100 == 0 or step + 1 == args.steps:
            print(f"step={step + 1} loss={float(loss.detach()):.6f}", flush=True)

    metrics = {name: evaluate(model, values) for name, values in datasets.items()}
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    weights_path = output / "weights.npz"
    arrays = {
        name: value.detach().cpu().numpy().astype(np.float32)
        for name, value in model.parameters.items()
    }
    arrays["output_gate"] = np.asarray([-1.4], dtype=np.float32)
    np.savez(weights_path, **arrays)
    manifest = {
        "schema": NEURAL_CONCEPT_SCHEMA,
        "bindings": {
            "model": artifact["identity"]["fingerprint"],
            "tokenizer": artifact["identity"]["fingerprint"],
            "runtime": args.runtime_binding,
        },
        "feature_dim": args.feature_dim,
        "state_dim": args.state_dim,
        "hidden_dim": hidden_dim,
        "message_rounds": 2,
        "attention_temperature": 0.07,
        "max_graph_concepts": 4096,
        "relations": list(relation_order),
        "weights_sha256": hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        "training": {
            "seed": args.seed,
            "episodes": args.episodes,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "split_sizes": {name: len(rows) for name, rows in split.items()},
            "metrics": metrics,
            "objective": "frozen-qwen-embedding-key-value-alignment-plus-contrastive-key",
        },
    }
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    manifest["fingerprint"] = hashlib.sha256(encoded + weights_path.read_bytes()).hexdigest()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "micro_world.json").write_text(
        json.dumps(
            {
                "schema": "mlx2-concept-micro-world-v1",
                "seed": args.seed,
                "splits": {
                    name: [episode.as_dict() for episode in rows]
                    for name, rows in split.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps({"artifact": str(output), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
