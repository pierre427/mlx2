#!/usr/bin/env python3
"""Probe direct hidden-state recurrent depth on the frozen capsule corpus.

This isolates realization from selection: the existing anchor-first selector
must still choose the expected source-bound capsule, then ordinary one-pass and
weight-tied two-pass greedy decode are compared with and without that capsule.
It is an experimental direct-model probe, not serving or route qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"[\w'-]+", value.casefold()))


def _graph(entities: list[dict]) -> dict:
    concepts = {}
    edges = []
    for entity in entities:
        subject = entity["id"] + ":subject"
        object_ = entity["id"] + ":object"
        concepts[subject] = {"label": entity["subject"]}
        concepts[object_] = {"label": entity["answer"]}
        edges.append(
            {
                "subject": subject,
                "relation": entity["relation"],
                "object": object_,
                "authority": "committed",
            }
        )
    return {"concepts": concepts, "edges": edges}


def _step_memory(memory: dict | None, step: int) -> dict | None:
    if memory is None:
        return None
    schedule = memory.get("decode_values")
    if schedule is None:
        return memory
    if step >= schedule.shape[0]:
        return None
    stepped = dict(memory)
    stepped.pop("decode_values")
    gates = stepped.pop("decode_gates", None)
    stepped["values"] = schedule[step : step + 1]
    if gates is not None:
        stepped["gate"] = float(gates[step])
    return stepped


def _greedy(adapter, prompt_ids, memory, *, passes: int, max_tokens: int) -> dict:
    import mlx.core as mx

    from mlx2.runtime.recurrent_depth import (
        RecurrentDepthConfig,
        recurrent_depth_forward,
    )

    config = RecurrentDepthConfig(passes=passes)
    caches = adapter.make_recurrent_depth_caches(passes)
    started = time.perf_counter()

    def forward(token_ids, step: int):
        stepped = _step_memory(memory, step)
        kwargs = {} if stepped is None else {"deep_concept_memory": stepped}
        return recurrent_depth_forward(
            adapter,
            mx.array([token_ids]),
            caches,
            config=config,
            first_pass_kwargs=kwargs,
        )

    output = []
    result = forward(prompt_ids, 0)
    eos = set(adapter.tokenizer.eos_token_ids)
    for step in range(max_tokens):
        token = int(mx.argmax(result.logits[:, -1, :], axis=-1).item())
        if token in eos:
            break
        output.append(token)
        result = forward([token], step + 1)
    mx.eval(result.logits)
    elapsed = time.perf_counter() - started
    text = adapter.tokenizer.decode(output)
    mx.clear_cache()
    return {
        "text": text,
        "token_ids": output,
        "elapsed_seconds": elapsed,
        "receipt": result.receipt,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=6)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")

    import mlx.core as mx

    from mlx2.adapters.qwen35_9b import Qwen359BAdapter
    from mlx2.runtime.neural_concepts import (
        NeuralConceptArtifact,
        RecurrentConceptEncoder,
    )
    from mlx2.runtime.neural_selection import select_neural_concept_payload

    if not 1 <= args.max_tokens <= 16:
        parser.error("max-tokens must be in 1..16")
    corpus_bytes = args.corpus.read_bytes()
    corpus = json.loads(corpus_bytes)
    if corpus.get("status") != "frozen-before-evaluation":
        raise ValueError("blind corpus is not frozen")
    selected_cases = [
        case for case in corpus["cases"] if case["expected_action"] == "select"
    ]
    entities = corpus["entities"]
    by_id = {item["id"]: item for item in entities}

    manifest = json.loads((args.artifact / "manifest.json").read_text())
    artifact = NeuralConceptArtifact.load(
        args.artifact,
        model_binding=manifest["bindings"]["model"],
        tokenizer_binding=manifest["bindings"]["tokenizer"],
        runtime_binding=manifest["bindings"]["runtime"],
    )
    encoded = {
        item.concept_id: item
        for item in RecurrentConceptEncoder(artifact).encode_graph(_graph(entities))
    }
    adapter = Qwen359BAdapter(str(args.model))
    adapter.configure_neural_concept_bridge(artifact)
    mx.eval(adapter.model.parameters())

    arms = (
        ("ordinary_one_pass", 1, False),
        ("ordinary_two_pass", 2, False),
        ("capsule_one_pass", 1, True),
        ("capsule_two_pass", 2, True),
    )
    rows = []
    started = time.time()
    for ordinal, case in enumerate(selected_cases):
        concepts = []
        for identifier in case["candidate_ids"]:
            entity = by_id[identifier]
            state = encoded[identifier + ":subject"]
            concepts.append(
                {
                    "id": identifier,
                    "subject": entity["subject"],
                    "aliases": entity["aliases"],
                    "revision": entity["revision"],
                    "source_spans": entity["source_spans"],
                    "key_state": state.key_state,
                    "value_state": state.value_state,
                    "decode_length": len(
                        adapter.tokenizer.encode(
                            entity["answer"], add_special_tokens=False
                        )
                    ),
                }
            )
        payload = {
            "schema": "mlx2-neural-concept-state-v1",
            "selection_policy": "anchor_first_v1",
            "artifact_fingerprint": artifact.fingerprint,
            "concepts": concepts,
        }
        request = {
            "messages": [{"role": "user", "content": case["query"]}],
            "enable_thinking": False,
            "reasoning_effort": "none",
            "_mlx2_neural_concepts": payload,
        }
        selected_payload, selection = select_neural_concept_payload(request)
        if selection["selected_id"] != case["expected_id"]:
            raise RuntimeError("frozen selector no longer chooses the expected capsule")
        prompt_ids = list(adapter.prompt_tokens(request))
        memory = adapter.neural_concept_prefill(
            prompt_ids,
            selected_payload,
            prefill_step=2048,
        )["deep_concept_memory"]
        expected = _normalize(case["expected_answer"])
        results = {}
        for name, passes, use_capsule in arms:
            result = _greedy(
                adapter,
                prompt_ids,
                memory if use_capsule else None,
                passes=passes,
                max_tokens=args.max_tokens,
            )
            normalized = _normalize(result["text"])
            result["phrase_prefix"] = normalized == expected or normalized.startswith(
                expected + " "
            )
            results[name] = result
        rows.append(
            {
                "ordinal": ordinal,
                "case_id": case["case_id"],
                "kind": case["kind"],
                "expected_id": case["expected_id"],
                "expected_answer": case["expected_answer"],
                "selected_id": selection["selected_id"],
                "arms": results,
            }
        )
        if (ordinal + 1) % 12 == 0:
            print(
                json.dumps({"completed": ordinal + 1, "cases": len(selected_cases)}),
                flush=True,
            )

    metrics = {}
    for name, passes, use_capsule in arms:
        successes = sum(row["arms"][name]["phrase_prefix"] for row in rows)
        elapsed = sum(row["arms"][name]["elapsed_seconds"] for row in rows)
        by_entity = defaultdict(list)
        for row in rows:
            by_entity[row["expected_id"]].append(row["arms"][name]["phrase_prefix"])
        metrics[name] = {
            "passes": passes,
            "capsule": use_capsule,
            "phrase_prefix": successes / len(rows),
            "phrase_prefix_count": successes,
            "cases": len(rows),
            "entities_all_variants": sum(all(values) for values in by_entity.values()),
            "entities": len(by_entity),
            "diagnostic_elapsed_seconds": elapsed,
        }
    one = "capsule_one_pass"
    two = "capsule_two_pass"
    result = {
        "schema": "mlx2.qwen35-recurrent-depth-capsule-probe.v1",
        "status": "experimental-not-qualified",
        "scope": (
            "frozen unseen synthetic selected cases; direct model greedy decode; "
            "pass one receives the selected capsule and pass two directly consumes "
            "pass one's final hidden state with an independent cache stack"
        ),
        "model": str(args.model.resolve()),
        "artifact_fingerprint": artifact.fingerprint,
        "corpus": str(args.corpus.resolve()),
        "corpus_file_sha256": hashlib.sha256(corpus_bytes).hexdigest(),
        "cases_sha256": corpus["cases_sha256"],
        "elapsed_seconds": time.time() - started,
        "metrics": metrics,
        "capsule_two_pass_delta": (
            metrics[two]["phrase_prefix"] - metrics[one]["phrase_prefix"]
        ),
        "capsule_two_pass_gained_cases": [
            row["case_id"]
            for row in rows
            if row["arms"][two]["phrase_prefix"]
            and not row["arms"][one]["phrase_prefix"]
        ],
        "capsule_two_pass_lost_cases": [
            row["case_id"]
            for row in rows
            if row["arms"][one]["phrase_prefix"]
            and not row["arms"][two]["phrase_prefix"]
        ],
        "rows": rows,
        "limitations": (
            "No adapter training, APCv2, serving, batching, performance control, "
            "or production route. The second pass consumes final normalized hidden "
            "states directly, which the stock checkpoint was not trained to accept."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "metrics": metrics,
                "capsule_two_pass_delta": result["capsule_two_pass_delta"],
                "gained": len(result["capsule_two_pass_gained_cases"]),
                "lost": len(result["capsule_two_pass_lost_cases"]),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
