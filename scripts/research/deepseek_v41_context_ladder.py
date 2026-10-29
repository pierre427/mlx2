"""Exact context requests and one-repetition, uncontrolled-temperature ladder."""

import hashlib
import json
import time
import unicodedata
import urllib.request
from pathlib import Path

from mlx2.desktop_campaign_worker import atomic
from mlx2.generation_trace import read_token_trace
from mlx2.runtime.models.deepseek_v41_encoding import encode_messages

CONTEXTS = (1024, 2048, 4096, 8192, 16384, 32768, 65536)
OUTPUT_RESERVE = 64


def normalize_code(text):
    return "".join("-" if unicodedata.category(c) == "Pd" else c for c in text.strip())


def exact_request(tokenizer, context):
    target = context-OUTPUT_RESERVE
    code = f"V41-C{context}-{hashlib.sha256(str(context).encode()).hexdigest()[:8].upper()}"
    prefix = f"Assignment code: {code}\nBackground reference notes follow.\n"
    suffix = "\nEnd of background notes. Return only the assignment code from the top of this message. Preserve its exact case and characters. Do not add quotation marks, markdown, explanation, or punctuation."
    sentence = "The archive describes stable records, exact model state, verified identifiers, and revision-bound prefix reuse.\n"

    def make(repetitions, padding=0):
        content = prefix+sentence*repetitions+" x"*padding+suffix
        messages = [{"role": "user", "content": content}]
        rendered = encode_messages(messages, thinking_mode="chat")
        tokens = tokenizer.encode(rendered, add_special_tokens=False).ids
        return {"messages": messages, "enable_thinking": False}, rendered, tokens

    low, high = 0, target
    while low < high:
        middle = (low+high+1)//2
        if len(make(middle)[2]) <= target:
            low = middle
        else:
            high = middle-1
    for padding in range(128):
        request, rendered, tokens = make(low, padding)
        if len(tokens) == target:
            return {"context_budget": context, "prompt_tokens": target, "output_reserve": OUTPUT_RESERVE,
                    "expected_code": code, "request": request, "rendered_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                    "token_ids_sha256": hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest(),
                    "padding_repetitions": low, "tail_padding": padding}
    raise ValueError("could not construct exact V4.1 prompt length")


def prepare_ladder(model, output):
    from tokenizers import Tokenizer

    from mlx2.adapters.deepseek_v41_artifact import REFERENCE_FILE

    tokenizer = Tokenizer.from_file(str(Path(model)/REFERENCE_FILE))
    requests = []
    for context in CONTEXTS:
        row = exact_request(tokenizer, context)
        destination = output/f"request-{context}.json"
        atomic(destination, row)
        requests.append({k: v for k, v in row.items() if k != "request"} | {"path": destination.name})
    return requests


def http_json(base, path, body=None, timeout=10, admin_token=None):
    data = None if body is None else json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if admin_token is not None:
        headers["Authorization"] = "Bearer "+admin_token
    request = urllib.request.Request(base+path, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as reply:
        return json.load(reply)


def stream(base, body, path, timeout=7200):
    request = urllib.request.Request(base+"/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    text, reasoning, windows = [], [], []
    receipt = usage = finish = None
    done, total = False, 0
    start = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply, path.open("xb") as raw:
            for line in reply:
                raw.write(line)
                total += len(line)
                if total > 8 << 20:
                    raise ValueError("SSE receipt exceeded8MiB")
                value = line.decode().strip()
                if not value.startswith("data:"):
                    continue
                value = value[5:].strip()
                if value == "[DONE]":
                    done = True
                    break
                event = json.loads(value)
                if event.get("error"):
                    raise RuntimeError(json.dumps(event["error"]))
                receipt = event.get("mlx2", receipt)
                usage = event.get("usage", usage)
                for choice in event.get("choices", []):
                    delta = choice.get("delta") or {}
                    text.append(delta.get("content") or "")
                    reasoning.append(delta.get("reasoning_content") or "")
                    windows.extend((choice.get("logprobs") or {}).get("content", []))
                    finish = choice.get("finish_reason") or finish
    except urllib.error.HTTPError as error:
        path.write_bytes(error.read(1 << 20))
        raise RuntimeError(f"HTTP{error.code}; body retained in {path.name}") from error
    if not done or receipt is None or finish is None:
        raise RuntimeError("incomplete SSE stream")
    trace = read_token_trace(receipt, expected_tokens=receipt.get("completion_tokens"))
    elapsed = receipt.get("elapsed_seconds")
    ttft = receipt.get("ttft_seconds")
    decode_seconds = elapsed-ttft if elapsed is not None and ttft is not None else None
    return {"text": "".join(text), "reasoning": "".join(reasoning), "token_ids": trace["token_ids"],
            "token_trace": trace, "receipt": receipt, "usage": usage, "finish_reason": finish,
            "HTTP_seconds": time.monotonic()-start, "TTFT_seconds": ttft,
            "elapsed_seconds": elapsed, "observed_decode_tokens_per_second":
            (max(0, trace["completion_tokens"]-1)/decode_seconds if decode_seconds and decode_seconds > 0 else None),
            "logprob_windows": windows, "raw_sse": path.name}


def assess(cold, warm, expected, prompt_tokens):
    gates = {}
    for arm, result, hit in (("cold", cold, 0), ("warm", warm, prompt_tokens-1)):
        receipt = result["receipt"]
        gates[arm+"_functional"] = normalize_code(result["text"]) == expected
        gates[arm+"_prompt_length"] = receipt.get("prompt_tokens") == prompt_tokens
        gates[arm+"_exact_hit"] = receipt.get("cached_tokens") == hit
        gates[arm+"_APCv2"] = receipt.get("cache") == "apcv2"
        gates[arm+"_route"] = receipt.get("route") == "external_draft" and bool(receipt.get("route_receipt"))
        gates[arm+"_width_one"] = set(result["token_trace"]["execution_widths"]) <= {1}
        gates[arm+"_exact_verification"] = (receipt.get("speculation") or {}).get("verification") == "exact"
    gates["token_parity"] = cold["token_ids"] == warm["token_ids"]
    gates["message_parity"] = (cold["text"], cold["reasoning"]) == (warm["text"], warm["reasoning"])
    divergent = None
    if not gates["token_parity"]:
        a, b = cold["token_ids"], warm["token_ids"]
        position = next((i for i, pair in enumerate(zip(a, b)) if pair[0] != pair[1]), min(len(a), len(b)))
        divergent = {"zero_based_position": position, "shared_prefix": a[:position],
                     "cold_token": a[position] if position < len(a) else None,
                     "warm_token": b[position] if position < len(b) else None}
    functional = gates["cold_functional"] and gates["warm_functional"]
    contract = all(value for name, value in gates.items() if not name.endswith("_functional"))
    return {"passed": functional and contract, "functional_passed": functional,
            "contract_passed": contract, "gates": gates, "first_divergence": divergent,
            "classification": "pass" if functional and contract else "functional_outcome_failure" if contract else "contract_failure"}


def run_ladder(spec, model_id, *, guard, environment, phase, admin_token):
    output = Path(spec["out"])
    base = "http://127.0.0.1:8285"
    results = {"schema": "mlx2.deepseek-v41-context-ladder.v1", "checkpoint": spec["checkpoint"],
               "source_revision": spec["source_revision"], "source_snapshot_sha256": spec["source_snapshot_sha256"],
               "repetitions": 1, "thermal_controlled": False, "qualification": False, "performance_claim": False,
               "context_semantics": "total budget; exact prompt=context-64 and64 reserved output tokens",
               "timing_scope": "single uncontrolled HTTP observation; expert cache carries between rungs; APC cleared before each cold arm",
               "cells": []}
    for request_info in spec["ladder"]:
        guard()
        context = request_info["context_budget"]
        directory = output/f"context-{context}"
        directory.mkdir(exist_ok=False)
        request = json.loads((output/request_info["path"]).read_text())
        cell = {k: v for k, v in request.items() if k != "request"}
        cell.update(repetition=1, thermal_controlled=False, status="running", before=environment())
        atomic(directory/"result.json", cell)
        phase("context_started", context_budget=context, prompt_tokens=request["prompt_tokens"])
        cell["APC_clear"] = http_json(base, "/v1/admin/apc/clear", {"reason": f"V41-ladder-{context}-cold"}, timeout=60, admin_token=admin_token)
        body = {**request["request"], "model": model_id, "temperature": 0, "max_tokens": OUTPUT_RESERVE,
                "stream": True, "return_token_trace": True, "logprobs": False}
        for arm in ("cold", "warm"):
            guard()
            cell[arm] = stream(base, body, directory/(arm+".sse"))
            atomic(directory/"result.json", cell)
            phase("context_arm_complete", context_budget=context, arm=arm, text=cell[arm]["text"],
                  cached_tokens=cell[arm]["receipt"].get("cached_tokens"), TTFT_seconds=cell[arm]["TTFT_seconds"],
                  observed_decode_tokens_per_second=cell[arm]["observed_decode_tokens_per_second"])
        cell["assessment"] = assess(cell["cold"], cell["warm"], request["expected_code"], request["prompt_tokens"])
        divergence = cell["assessment"]["first_divergence"]
        if divergence is not None:
            # Diagnostic repeats are separate from the single measured pair.
            position = divergence["zero_based_position"]+1
            cell["diagnostics"] = {"purpose": "already-selected first-divergence position only", "one_based_position": position}
            http_json(base, "/v1/admin/apc/clear", {"reason": "V41-first-divergence-diagnostic"}, timeout=60, admin_token=admin_token)
            diagnostic = {**body, "logprobs": True, "top_logprobs": 20, "logprobs_start": position, "logprobs_end": position}
            for arm in ("cold", "warm"):
                guard()
                cell["diagnostics"][arm] = stream(base, diagnostic, directory/("diagnostic-"+arm+".sse"))
        cell.update(status="complete", after=environment(), server_after=http_json(base, "/v1/status"))
        atomic(directory/"result.json", cell)
        results["cells"].append({"context_budget": context, "prompt_tokens": request["prompt_tokens"],
                                 "path": str(directory.relative_to(output)/"result.json"), **cell["assessment"]})
        atomic(output/"ladder.json", results)
        phase("context_complete", context_budget=context, assessment=cell["assessment"])
        if not cell["assessment"]["contract_passed"]:
            results["stopped_reason"] = "contract failure; preserve first divergence before repair"
            break
    results["complete"] = len(results["cells"]) == len(spec["ladder"])
    results["passed"] = results["complete"] and all(c["passed"] for c in results["cells"])
    atomic(output/"ladder.json", results)
    return results
