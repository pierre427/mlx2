#!/usr/bin/env python3
"""Probe concept-capsule strength across bounded prompt lengths."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from probe_qwen35_deep_concept_bridge import _graph, _greedy


def _normalize(value):
    return " ".join(re.findall(r"[\w'-]+", value.casefold()))


def _prompt(adapter, row, target_tokens):
    suffix = row["query"] + " Answer with only the aroma."

    def render(content):
        return list(
            adapter.prompt_tokens(
                {
                    "messages": [{"role": "user", "content": content}],
                    "enable_thinking": False,
                    "reasoning_effort": "none",
                }
            )
        )

    base = render(suffix)
    if target_tokens <= len(base):
        return base
    filler = (
        "Background registry note: route records are descriptive context only. "
        * ((target_tokens - len(base)) // 8 + 4)
    )
    filler_ids = list(adapter.tokenizer.encode(filler, add_special_tokens=False))
    needed = max(0, target_tokens - len(base))
    content = adapter.tokenizer.decode(filler_ids[:needed]) + "\n\n" + suffix
    return render(content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--prompt-lengths", default="32,128,512,1024")
    parser.add_argument("--strengths", default="0.1,0.2,0.3")
    parser.add_argument("--max-tokens", type=int, default=5)
    args = parser.parse_args()

    from mlx2.adapters.qwen35_9b import Qwen359BAdapter
    from mlx2.runtime.neural_concepts import NeuralConceptArtifact, RecurrentConceptEncoder

    prompt_lengths = tuple(int(value) for value in args.prompt_lengths.split(","))
    strengths = tuple(float(value) for value in args.strengths.split(","))
    if any(not 1 <= value <= 2048 for value in prompt_lengths):
        parser.error("prompt lengths must be in 1..2048")
    if any(not 0 < value <= 1 for value in strengths):
        parser.error("strengths must be inside (0, 1]")

    manifest = json.loads((args.artifact / "manifest.json").read_text())
    artifact = NeuralConceptArtifact.load(
        args.artifact,
        model_binding=manifest["bindings"]["model"],
        tokenizer_binding=manifest["bindings"]["tokenizer"],
        runtime_binding=manifest["bindings"]["runtime"],
    )
    world = json.loads((args.artifact / "micro_world.json").read_text())
    rows = world["splits"][args.split][: args.episodes]
    adapter = Qwen359BAdapter(str(args.model))
    adapter.configure_neural_concept_bridge(artifact)
    encoder = RecurrentConceptEncoder(artifact)
    observations = []

    for index, target in enumerate(rows):
        candidates = [target] + [
            rows[(index + offset) % len(rows)]
            for offset in range(1, args.candidates)
        ]
        graph, subject_ids = _graph(candidates)
        encoded = {item.concept_id: item for item in encoder.encode_graph(graph)}
        target_steps = len(
            adapter.tokenizer.encode(target["answer"], add_special_tokens=False)
        )
        payload = {
            "artifact_fingerprint": artifact.fingerprint,
            "concepts": [
                {
                    "key_state": encoded[identifier].key_state,
                    "value_state": encoded[identifier].value_state,
                    "decode_length": target_steps,
                }
                for identifier in subject_ids
            ],
        }
        answer = _normalize(target["answer"])
        for target_length in prompt_lengths:
            prompt_ids = _prompt(adapter, target, target_length)
            memory = adapter.neural_concept_prefill(
                prompt_ids, payload, prefill_step=2048
            )["deep_concept_memory"]
            arms = [
                ("ordinary", _greedy(adapter, prompt_ids, None, args.max_tokens)),
                (
                    "one-shot",
                    _greedy(adapter, prompt_ids, memory, args.max_tokens),
                ),
            ]
            for strength in strengths:
                tuned = dict(
                    memory,
                    decode_gates=[strength] * target_steps,
                )
                arms.append(
                    (
                        f"capsule-strength-{strength:g}",
                        _greedy(
                            adapter,
                            prompt_ids,
                            tuned,
                            args.max_tokens,
                            persistent=True,
                        ),
                    )
                )
            for arm, text in arms:
                normalized = _normalize(text)
                observations.append(
                    {
                        "episode_id": target["episode_id"],
                        "answer": target["answer"],
                        "target_prompt_tokens": target_length,
                        "actual_prompt_tokens": len(prompt_ids),
                        "arm": arm,
                        "text": text,
                        "phrase_exact": normalized == answer,
                        "phrase_prefix": normalized == answer
                        or normalized.startswith(answer + " "),
                    }
                )
            print(
                json.dumps(
                    {
                        "episode": target["episode_id"],
                        "target_prompt_tokens": target_length,
                        "actual_prompt_tokens": len(prompt_ids),
                    }
                ),
                flush=True,
            )

    grouped = defaultdict(list)
    for row in observations:
        grouped[(row["target_prompt_tokens"], row["arm"])].append(row)
    cells = []
    for (prompt_length, arm), cell_rows in grouped.items():
        count = len(cell_rows)
        cells.append(
            {
                "target_prompt_tokens": prompt_length,
                "mean_actual_prompt_tokens": sum(
                    row["actual_prompt_tokens"] for row in cell_rows
                )
                / count,
                "arm": arm,
                "episodes": count,
                "phrase_exact_accuracy": sum(
                    row["phrase_exact"] for row in cell_rows
                )
                / count,
                "phrase_prefix_accuracy": sum(
                    row["phrase_prefix"] for row in cell_rows
                )
                / count,
            }
        )
    cells.sort(key=lambda row: (row["target_prompt_tokens"], row["arm"]))
    result = {
        "schema": "mlx2.qwen35-concept-capsule-prompt-length.v1",
        "status": "experimental-tuning",
        "split": args.split,
        "episodes": len(rows),
        "candidates_per_request": args.candidates,
        "artifact_fingerprint": artifact.fingerprint,
        "prompt_lengths": prompt_lengths,
        "strengths": strengths,
        "cells": cells,
        "observations": observations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"cells": cells}, indent=2))


if __name__ == "__main__":
    main()
