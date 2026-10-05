#!/usr/bin/env python3
"""Derive source-bound N3 length controls for isolated varlen prefill tests.

The output retains the frozen 400-row manifest and changes only the first three
rows of one domain.  Each changed chat contains a bounded excerpt of its own
source transcript plus the original final request.  Token IDs are regenerated
with the manifest-bound tokenizer.  This is a research performance input, not
a qualification corpus.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path


def digest(value):
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def tokenize(tokenizer, messages):
    value = tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=False,
        add_generation_prompt=True, enable_thinking=False)
    if isinstance(value, dict):
        value = value["input_ids"]
    return tuple(int(token) for token in value)


def source_excerpt(messages):
    """Keep authoritative evidence at every derived prompt length.

    A raw prefix made the short control retain arbitrary archive chatter while
    dropping the middle/late audit notes.  Besides changing the task, that
    removed the repeated output cue used by the K>0 prompt-lookup gate.  Split
    essential evidence from expendable background, then resize only the latter.
    """
    essential = []
    background = []
    for message in messages[1:-1]:
        rendered = f"[{message['role']}]\n{message['content']}"
        if "<authoritative_audit_note" in message["content"]:
            essential.append(rendered)
        else:
            background.append(rendered)
    if not essential:
        raise ValueError("source conversation lacks authoritative audit notes")
    return "\n\n".join(essential), "\n\n".join(background)


def resize_messages(tokenizer, source, target):
    if type(target) is not int or target < 1025:
        raise ValueError("target must be an integer of at least 1025 tokens")
    system = deepcopy(source[0])
    final = deepcopy(source[-1])
    essential, background = source_excerpt(source)

    def candidate(characters, padding=0):
        content = (
            "Authoritative source notes:\n" + essential
            + "\n\nBackground transcript excerpt:\n" + background[:characters]
        )
        if padding:
            content += "\n" + " context" * padding
        return [system, {"role": "user", "content": content}, final]

    low, high = 0, len(background)
    while low < high:
        middle = (low + high + 1) // 2
        if len(tokenize(tokenizer, candidate(middle))) <= target:
            low = middle
        else:
            high = middle - 1
    best = candidate(low)
    best_ids = tokenize(tokenizer, best)
    # A repeated ordinary word gives a deterministic fine adjustment without
    # inventing model-family tokens or editing token IDs independently of text.
    padding = 0
    while len(best_ids) < target:
        trial = candidate(low, padding + 1)
        trial_ids = tokenize(tokenizer, trial)
        if len(trial_ids) > target:
            break
        padding += 1
        best, best_ids = trial, trial_ids
    if target - len(best_ids) > 2:
        raise RuntimeError("unable to produce requested token geometry closely")
    return best, best_ids


def derive(inputs, tokenizer, targets, *, domain_index=0):
    result = deepcopy(inputs)
    domains = result.get("domain_order", [])
    if not 0 <= domain_index < len(domains):
        raise ValueError("domain index outside manifest")
    if len(targets) != 3 or any(type(value) is not int for value in targets):
        raise ValueError("exactly three integer targets required")
    domain = domains[domain_index]
    selected = [row for row in result["rows"] if row["domain"] == domain][:3]
    if len(selected) != 3:
        raise ValueError("three source rows required")
    derived_rows = []
    for row, target in zip(selected, targets):
        messages, ids = resize_messages(tokenizer, row["body"]["messages"], target)
        row["body"]["messages"] = messages
        row["body_sha256"] = digest(row["body"])
        row["prompt_token_ids"] = list(ids)
        row["prompt_tokens"] = len(ids)
        row["preparation_receipt"] = {
            "schema": "mlx2.varlen-derived-row.v1",
            "source_case_id": row["case_id"],
            "requested_prompt_tokens": target,
            "actual_prompt_tokens": len(ids),
            "source_bound_excerpt": True,
            "authoritative_notes_preserved": True,
            "qualification_input": False,
        }
        derived_rows.append({
            "case_id": row["case_id"], "target": target, "actual": len(ids)})
    result["varlen_research_derivation"] = {
        "schema": "mlx2.varlen-skew-inputs.v1",
        "domain_index": domain_index,
        "domain": domain,
        "selected_rows": derived_rows,
        "stock_quantized_math_required": True,
        "speculative_decoding_required": False,
        "qualification_input": False,
    }
    result["qualified"] = False
    result["price_usable"] = False
    result["inputs_sha256"] = digest({
        key: value for key, value in result.items() if key != "inputs_sha256"})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--targets", default="1536,4096,6982")
    parser.add_argument("--domain-index", type=int, default=0)
    args = parser.parse_args()
    targets = tuple(int(value) for value in args.targets.split(","))
    inputs = json.loads(args.inputs.read_text())
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        inputs["tokenizer_root"], trust_remote_code=True)
    result = derive(inputs, tokenizer, targets, domain_index=args.domain_index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result["varlen_research_derivation"], indent=2))


if __name__ == "__main__":
    main()
