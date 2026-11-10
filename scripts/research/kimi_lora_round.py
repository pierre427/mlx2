#!/usr/bin/env python3
"""One bounded adapter-only Kimi experiment, inside an existing owned GPU unit.

The base is a separate immutable copy. Every answer and failed round is retained;
no result here changes a serving default or establishes general model quality.
"""
from __future__ import annotations

import sys as _tree_sys  # mlx2 worktree bootstrap: import this tree's src, not the editable install
from pathlib import Path as _TreePath

_tree_src = str(_TreePath(__file__).resolve().parents[2] / "src")
if _tree_src not in _tree_sys.path[:1]:
    _tree_sys.path.insert(0, _tree_src)
del _tree_sys, _TreePath, _tree_src

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]


def atomic(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    os.replace(temporary, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def json_metrics(snapshot):
    """Retain the collector's two-label event keys without lossy stringification."""
    return {**snapshot, "events": [
        {"operation": operation, "outcome": outcome, "count": count}
        for (operation, outcome), count in sorted(snapshot["events"].items())]}


def declared_sha256(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()):
        raise ValueError("declared SHA256 must contain 64 hexadecimal characters")
    return value.lower()


def read_bound_bytes(path, *, expected_sha256=None):
    """Bind the bytes actually consumed, even if the pathname later changes."""
    payload = Path(path).read_bytes()
    fingerprint = hashlib.sha256(payload).hexdigest()
    if expected_sha256 is not None and fingerprint != declared_sha256(expected_sha256):
        raise ValueError("declared input SHA256 mismatch: " + str(path))
    return payload, fingerprint


def read_rows_bound(path, *, expected_sha256=None):
    payload, fingerprint = read_bound_bytes(path, expected_sha256=expected_sha256)
    # Match open()'s universal-newline iteration; str.splitlines() would also
    # split valid JSON strings containing literal Unicode line separators.
    with io.TextIOWrapper(io.BytesIO(payload)) as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return rows, fingerprint


def read_rows(path):
    return read_rows_bound(path)[0]


def data_selection_binding(spec):
    path_key, hash_key = "data_selection_receipt", "data_selection_receipt_sha256"
    if (path_key in spec) != (hash_key in spec):
        raise ValueError("data selection receipt requires both path and SHA256")
    if path_key not in spec:
        return None
    if not isinstance(spec[path_key], str) or not spec[path_key]:
        raise ValueError("data selection receipt path must be nonempty")
    expected = declared_sha256(spec[hash_key])
    _, fingerprint = read_bound_bytes(spec[path_key], expected_sha256=expected)
    # The receipt is opaque evidence: do not invent a generic admission schema.
    return {"path": str(Path(spec[path_key]).resolve()), "sha256": fingerprint}


def training_loss_profile(spec):
    """Host-only validation; a declared optimization must never be ignored."""
    size = spec.get("projection_chunk_size")
    if size is not None:
        if type(size) is not int or size < 1:
            raise ValueError("projection_chunk_size must be a positive integer or null")
        if spec.get("mode") not in {"probe", "train"}:
            raise ValueError("projection_chunk_size applies only to training/probe loss")
    return {"schema": "kimi-training-loss-v1", "projection_chunk_size": size,
            "projection": "dense" if size is None else "checkpointed_token_chunks",
            "normalization": "whole_shifted_assistant_mask", "head": "frozen_lm_head",
            "cross_entropy_dtype": "float32"}


def training_attention_profile(spec):
    mla = spec.get("training_mla", "expanded")
    if not isinstance(mla, str) or mla not in {"expanded", "padded_sdpa"}:
        raise ValueError("training_mla must be expanded or padded_sdpa")
    if mla != "expanded" and spec.get("mode") not in {"probe", "train"}:
        raise ValueError("training_mla applies only to training/probe execution")
    profile = {"kda": spec.get("training_kda", "reference"), "mla": mla}
    if mla == "padded_sdpa":
        # Exact Kimi48B-q8 last25/rank8 dense-head 4081-token probes failed
        # the100GiB guard:065(d677,expanded)101.965GiB;071(ab68,padded)
        #106.664GiB. These different-source, swap-affected observations establish
        # no general memory benefit. Keep opt-in; see route-snapshot provenance.
        # This bounded experiment supports the pinned MLX default VJP schedule.
        # Explicitly bind it; overrides change allocation and accumulation order.
        defaults = {"MLX_SDPA_VJP_BLOCK": 1024, "MLX_SDPA_VJP_BQ": 0,
                    "MLX_SDPA_VJP_BK": 0, "MLX_SDPA_VJP_FALLBACK": 0}
        for name, value in defaults.items():
            if name in os.environ and os.environ[name] != str(value):
                raise ValueError(f"padded_sdpa requires default {name}={value}")
        profile["mlx_sdpa_vjp_effective_environment"] = defaults
    return profile


def resume_training_binding(spec):
    """Read the archived parent spec once, before loading any runtime/model."""
    if not spec.get("resume_checkpoint"):
        return None
    if spec.get("resume_program"):
        raise ValueError("select one full-state checkpoint or fresh-Adam program resume")
    payload, fingerprint = read_bound_bytes(spec["resume_from_spec"],
                                           expected_sha256=spec["resume_from_spec_sha256"])
    parent = json.loads(payload.decode())
    if training_loss_profile(parent) != training_loss_profile(spec):
        raise ValueError("checkpoint resume training loss profile changed")
    if training_attention_profile(parent) != training_attention_profile(spec):
        raise ValueError("checkpoint resume training attention profile changed")
    # The checkpoint binding substitutes the parent spec hash, so every other
    # input of the training law must be compared here.
    for key, default in (("execution_policy", None), ("max_grad_norm", 1.0)):
        if parent.get(key, default) != spec.get(key, default):
            raise ValueError(f"checkpoint resume {key} changed")
    return {"path": str(Path(spec["resume_from_spec"]).resolve()), "sha256": fingerprint,
            "training_loss": training_loss_profile(parent),
            "training_attention": training_attention_profile(parent)}


def validate_spec(spec):
    from kimi_lora_sampling import generation_profile
    generation_profile(spec)
    training_loss_profile(spec)
    training_attention_profile(spec)
    model, original = Path(spec["model"]).resolve(), Path(spec["original_model"]).resolve()
    if model == original or original in model.parents:
        raise ValueError("training requires a separate base copy")
    out = Path(spec["out"]).resolve()
    if out == model or model in out.parents or out == original or original in out.parents:
        raise ValueError("outputs must be outside both model directories")
    if spec.get("mode") not in {"probe", "baseline", "train", "evaluate"}:
        raise ValueError("unknown round mode")
    if int(spec.get("steps", 0)) < 0 or int(spec.get("steps", 0)) > 512:
        raise ValueError("a light-touch round permits 0..512 updates")
    if spec.get("mode") in {"probe", "train"} and not spec.get("data"):
        raise ValueError("training data required")
    if "data_expected_sha256" in spec:
        declared_sha256(spec["data_expected_sha256"])
        if not spec.get("data"):
            raise ValueError("declared training data SHA256 requires a data path")
    if spec.get("eval_split", "dev") not in {"dev", "final"}:
        raise ValueError("eval_split must be dev or final")
    if spec.get("eval_split", "dev") == "final":
        if (spec.get("mode") != "evaluate" or not spec.get("adapter")
                or not spec.get("selected_candidate_receipt") or not spec.get("selected_candidate_receipt_sha256")):
            raise ValueError("final evaluation requires a frozen candidate-selection receipt and adapter")
        if spec.get("evaluate_base") is False:
            raise ValueError("final evaluation requires a paired base comparison")
    return model, out


def validate_evaluation_cases(cases, split):
    if split == "final" and not cases:
        raise ValueError("final evaluation requires nonempty held-out cases")
    seen = set()
    for case in cases:
        if not isinstance(case, dict) or case.get("split") != split or case.get("evaluation_only") is not True:
            raise ValueError("evaluation case split/eligibility differs from the requested held-out split")
        identity = case.get("id")
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError("evaluation case IDs must be nonempty and unique")
        seen.add(identity)
        for field, default in (("max_tokens", 384), ("max_turns", 8), ("max_tool_steps", 8)):
            value = case.get(field, default)
            if type(value) is not int or value < 1:
                raise ValueError("evaluation budget must be a positive integer: " + field)


def candidate_selection(spec):
    """Freeze final-test selection to actual adapter bytes and a completed dev run."""
    if spec.get("eval_split", "dev") != "final":
        return None
    path = Path(spec["selected_candidate_receipt"]).resolve()
    if not path.is_file() or sha(path) != spec["selected_candidate_receipt_sha256"]:
        raise ValueError("frozen candidate-selection receipt hash mismatch")
    selected = json.loads(path.read_text())
    checkpoint = spec["artifact_name"] + "@" + spec["base_fingerprint"]
    if (selected.get("schema") != "mlx2.kimi-lora-candidate-selection.v1"
            or selected.get("checkpoint") != checkpoint
            or not selected.get("adapter_path")
            or Path(selected["adapter_path"]).resolve() != Path(spec["adapter"]).resolve()):
        raise ValueError("frozen candidate does not match requested adapter/checkpoint")
    from mlx2.runtime.lora import read_lora_config, inspect_lora_program, lora_fingerprint
    adapter_path, config = read_lora_config(spec["adapter"])
    payload = (adapter_path / "adapters.safetensors").read_bytes()
    program = inspect_lora_program(adapter_path, config, base_checkpoint=checkpoint, payload=payload)
    if program is None:
        raise ValueError("final evaluation requires a source-bound weight program")
    fingerprint = lora_fingerprint(payload, config["rank"], config["scale"], config["keys"], program)
    if (selected.get("adapter_fingerprint") != fingerprint or selected.get("program_id") != program["program_id"]
            or selected.get("source_revision") != program["source_revision"]):
        raise ValueError("selected adapter bytes or source identity changed")
    development = selected.get("development_receipt") or {}
    development_path = Path(development.get("path", ""))
    if not development_path.is_file() or sha(development_path) != development.get("sha256"):
        raise ValueError("selected candidate development receipt hash mismatch")
    observed = json.loads(development_path.read_text())
    if (observed.get("status") != "completed" or observed.get("adapter_fingerprint") != fingerprint
            or observed.get("checkpoint") != checkpoint or observed.get("spec", {}).get("eval_split", "dev") != "dev"
            or not observed.get("evaluations", {}).get("adapter")):
        raise ValueError("selected candidate has no completed development evaluation of this adapter")
    return {"path": str(path), "sha256": sha(path), **selected}


def verify_selected_mount(selected, mounted):
    if selected is not None and mounted.fingerprint != selected["adapter_fingerprint"]:
        raise ValueError("mounted adapter changed after frozen final selection")


def validate_training_rows(rows, *, vocab_size=163840, max_tokens=4096, min_tokens=2):
    if not rows:
        raise ValueError("training data is empty")
    for row in rows:
        if row.get("split") != "train" or row.get("evaluation_only"):
            raise ValueError("training row is not explicitly in the training split")
        ids, mask = row.get("input_ids"), row.get("loss_mask")
        if not isinstance(ids, list) or not isinstance(mask, list) or len(ids) != len(mask) or not min_tokens <= len(ids) <= max_tokens:
            raise ValueError("training row needs aligned, complete, bounded tokens and mask")
        if any(type(token) is not int or not 0 <= token < vocab_size for token in ids):
            raise ValueError("training token outside checkpoint vocabulary")
        if any(type(value) not in (int, float, bool) or not math.isfinite(value) or value not in (0, 1) for value in mask):
            raise ValueError("loss mask must be finite and binary")
        if not sum(mask[1:]):
            raise ValueError("training row has no causal assistant targets")
        if not row.get("id"):
            raise ValueError("training row requires provenance identity")


def host_state():
    import psutil
    return {"available_bytes": psutil.virtual_memory().available,
            "swap": subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True).strip(),
            "vm_stat": subprocess.check_output(["vm_stat"], text=True),
            "thermal": subprocess.check_output(["pmset", "-g", "therm"], text=True).strip()}


def gpu_guard(mx, spec, started):
    import psutil
    if time.monotonic() - started > float(spec.get("timeout_seconds", 1000)):
        raise TimeoutError("bounded experiment window expired")
    active, peak = mx.get_active_memory(), mx.get_peak_memory()
    if peak > int(spec.get("max_metal_bytes", 100 << 30)):
        raise MemoryError("Metal peak exceeded experiment bound")
    if psutil.virtual_memory().available < int(spec.get("system_reserve_bytes", 20 << 30)):
        raise MemoryError("system memory reserve exhausted")
    return {"active_bytes": active, "peak_bytes": peak}


def generate(adapter, messages, *, tools=None, max_tokens=384, prefill_chunk=128, guard=None, sampler=None):
    import mlx.core as mx
    request = {"messages": messages}
    if tools:
        request["tools"] = tools
    prompt = adapter.prompt_tokens(request)
    cache, tokens = adapter.model.make_cache(), []
    start_time = time.monotonic()
    for start in range(0, len(prompt), prefill_chunk):
        logits = adapter.model(mx.array([prompt[start:start + prefill_chunk]], mx.int32), cache=cache)
        mx.eval(logits, cache)
        if guard:
            guard()
    finish = "length"
    for _ in range(max_tokens):
        token = int(mx.argmax(logits[0, -1]).item()) if sampler is None else sampler.draw(logits[0, -1])
        tokens.append(token)
        if token in adapter.tokenizer.eos_token_ids:
            finish = "stop"
            break
        logits = adapter.model(mx.array([[token]], mx.int32), cache=cache)
        mx.eval(logits, cache)
        if guard:
            guard()
    text = adapter.tokenizer.decode(tokens)
    for marker in ("<|im_end|>", "[EOS]"):
        if text.endswith(marker):
            text = text[:-len(marker)]
    return {"text": text, "tokens": tokens, "prompt_tokens": len(prompt),
            "finish_reason": finish, "elapsed_seconds": time.monotonic() - start_time}


def parse_episode_calls(text, tools):
    """Validate a complete emitted batch before applying any simulated action."""
    from mlx2.openai_compat import _validate_schema_value
    from mlx2.runtime.tool_parsers.kimi import CALL_BEGIN, CALL_END, ARGUMENT_BEGIN, SECTION_MARKERS, parse_tool_call
    declared = {tool["function"]["name"]: tool["function"] for tool in tools or ()}
    calls, ids = [], set()
    remainder = text
    while CALL_BEGIN in remainder:
        _, remainder = remainder.split(CALL_BEGIN, 1)
        body, separator, remainder = remainder.partition(CALL_END)
        if not separator:
            raise ValueError("incomplete tool call")
        parsed = parse_tool_call(body, tools)
        if parsed["name"] not in declared:
            raise ValueError("undeclared tool call")
        _validate_schema_value(declared[parsed["name"]]["parameters"], parsed["arguments"])
        call_id = body.partition(ARGUMENT_BEGIN)[0].strip()
        if not call_id or call_id in ids:
            raise ValueError("missing or duplicate tool call ID within one batch")
        ids.add(call_id)
        calls.append({"id": call_id, "type": "function", "function": {
            "name": parsed["name"], "arguments": json.dumps(parsed["arguments"])}})
    if CALL_END in remainder or ARGUMENT_BEGIN in remainder:
        raise ValueError("orphan tool protocol marker")
    if any(call["function"]["name"] == "finish" for call in calls[:-1]):
        raise ValueError("tool call emitted after finish in the same batch")
    visible = text.split(CALL_BEGIN)[0]
    for marker in SECTION_MARKERS:
        visible = visible.replace(marker, "")
    return visible, calls


def evaluate(adapter, cases, output, *, guard, sampling_spec=None):
    from kimi_lora_sampling import CaseSampler, generation_profile, expected_case_receipt
    sampling_spec = {} if sampling_spec is None else sampling_spec
    generation = generation_profile(sampling_spec)
    from kimi_lora_data import grade_case, episode_initial_state, episode_step
    from mlx2.runtime.tool_parsers.kimi import CALL_BEGIN, CALL_END, ARGUMENT_BEGIN, SECTION_MARKERS
    rows = []
    with output.open("x") as stream:
        for case in cases:
            sampler = (CaseSampler(generation["profile"], suite_seed=sampling_spec["sampling_seed"],
                       case_id=case["id"], replicate=sampling_spec.get("replicate", 0))
                       if generation["seed_strategy"] is not None else None)
            messages = list(case["messages"])
            state = episode_initial_state(case) if case.get("mode") == "interactive" else None
            closed, final_received = False, False
            turns, protocol_errors = [], []
            max_turns = int(case.get("max_turns", 8)) if state is not None else 1
            for turn_index in range(max_turns):
                answer = generate(adapter, messages, tools=None if closed else case.get("tools"),
                                  max_tokens=int(case.get("max_tokens", 384)), guard=guard, sampler=sampler)
                turns.append(answer)
                text = answer["text"]
                if state is None:
                    break
                if CALL_BEGIN not in text:
                    if any(marker in text for marker in (*SECTION_MARKERS, CALL_END, ARGUMENT_BEGIN)):
                        protocol_errors.append("incomplete tool protocol")
                    final_received = bool(text.strip()) and not protocol_errors
                    break
                if closed:
                    protocol_errors.append("tool call after environment completion")
                    break
                try:
                    visible, calls = parse_episode_calls(text, case.get("tools"))
                    if state.get("calls", 0) + len(calls) > int(case.get("max_tool_steps", 8)):
                        raise ValueError("tool batch exceeds remaining action budget")
                except ValueError as error:
                    protocol_errors.append(str(error))
                    answer["tool_parse_error"] = str(error)
                    break
                messages.append({"role": "assistant", "content": visible, "tool_calls": calls})
                for call in calls:
                    state, result, done = episode_step(case, state, {
                        "name": call["function"]["name"],
                        "arguments": json.loads(call["function"]["arguments"])})
                    messages.append({"role": "tool", "tool_call_id": call["id"],
                                     "content": json.dumps(result, ensure_ascii=False)})
                    closed = closed or done
                # A terminal action needs a subsequent ordinary assistant
                # response within the same frozen turn budget. Task state and
                # protocol completion remain separately visible in the grade.
            final = {**turns[-1], "episode_state": state}
            grade = grade_case(case, final)
            grade["capped"] = any(turn.get("finish_reason") == "length" for turn in turns)
            if state is not None:
                if closed and not final_received and not protocol_errors:
                    protocol_errors.append("missing final assistant response within turn budget")
                grade["environment_success"] = grade["correct"]
                grade["protocol_ok"] = not protocol_errors
                grade["final_assistant_received"] = final_received
                grade["correct"] = bool(grade["correct"] and not protocol_errors and final_received)
                grade["format_ok"] = bool(grade["format_ok"] and not protocol_errors and final_received)
            row = {"id": case["id"], "domain": case["domain"], "split": case["split"],
                   "turns": turns, "episode_state": state, "grade": grade,
                   "protocol_errors": protocol_errors, "messages": messages,
                   "sampling": sampler.receipt() if sampler is not None else
                       expected_case_receipt(generation, case["id"], 0)}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n"); stream.flush()
            rows.append(row)
            print(json.dumps({"evaluation": case["id"], "grade": row["grade"]}), flush=True)
    return {"cases": len(rows), "correct": sum(bool(r["grade"]["correct"]) for r in rows),
            "unsupported_completion": sum(bool(r["grade"].get("unsupported_completion")) for r in rows),
            "protocol_failed": sum(bool(r["protocol_errors"]) for r in rows),
            "format_ok": sum(bool(r["grade"].get("format_ok")) for r in rows),
            "capped": sum(bool(r["grade"].get("capped")) for r in rows), "sha256": sha(output)}


def actual_execution(adapter):
    """Capture resolved adapter state, not only the requested policy."""
    from mlx2.sampling_contract import descriptor_sampling_contract
    return json.loads(json.dumps({"policy": adapter.policy, "environment": adapter.environment,
        "profile": adapter.profile, "cache_layout": adapter.descriptor.cache_layout,
        "sampling_contract": descriptor_sampling_contract(adapter.descriptor)}, allow_nan=False))


def training_route_deltas(model, first_layer, baseline):
    """Copy frontend invocation counts; not an observed Metal backward route."""
    return {str(i): {key: count - baseline[str(i)].get(key, 0)
            for key, count in layer.attention.calls.items()}
            for i, layer in enumerate(model.layers) if i >= first_layer}


def run(spec_path):
    from kimi_lora_sampling import generation_profile, SAMPLING_CODE
    spec_payload, spec_sha256 = read_bound_bytes(spec_path)
    spec = json.loads(spec_payload.decode())
    model_path, out = validate_spec(spec)
    loss_profile = training_loss_profile(spec)
    attention_profile = training_attention_profile(spec)
    parent_training_binding = resume_training_binding(spec)
    cases, cases_sha256 = read_rows_bound(spec["eval_cases"]) if spec.get("eval_cases") else ([], None)
    validate_evaluation_cases(cases, spec.get("eval_split", "dev"))
    selected = candidate_selection(spec)
    if spec.get("eval_limit") is not None:
        if type(spec["eval_limit"]) is not int or spec["eval_limit"] < 1:
            raise ValueError("eval_limit must be a positive integer")
        cases = cases[:spec["eval_limit"]]
    selection_binding = data_selection_binding(spec)
    training_rows, data_sha256 = (read_rows_bound(spec["data"], expected_sha256=spec.get("data_expected_sha256"))
                                 if spec.get("mode") in {"probe", "train"} else ([], None))
    if not training_rows and spec.get("mode") not in {"probe", "train"} and "data_expected_sha256" in spec:
        # A declared hash is checked even when this evaluation-only invocation
        # does not consume training rows.
        read_bound_bytes(spec["data"], expected_sha256=spec["data_expected_sha256"])
    if training_rows:
        # padded_sdpa training MLA needs T > 8 on the shifted input; refuse
        # short rows here, before the model loads, not at their first step.
        validate_training_rows(training_rows, max_tokens=int(spec.get("max_sequence_length", 4096)),
                               min_tokens=10 if training_attention_profile(spec)["mla"] == "padded_sdpa" else 2)
    elif spec.get("mode") in {"probe", "train"}:
        raise ValueError("training data is empty")
    from kimi_lora_loss import validate_loss_rows, evaluate_loss
    loss_rows, loss_sha256 = read_rows_bound(spec["loss_cases"]) if spec.get("loss_cases") else ([], None)
    if loss_rows:
        validate_loss_rows(loss_rows)
        if {row["id"] for row in loss_rows} & {row["id"] for row in training_rows}:
            raise ValueError("heldout loss rows overlap this training stream")
    results = out / "results"
    if results.exists():
        raise ValueError("refusing existing round results")
    results.mkdir(parents=True)
    receipt = {"spec": spec, "spec_sha256": spec_sha256, "source_revision":
               subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "code_sha256": {}, "status": "starting", "training": [], "evaluations": {},
               "qualified": False, "model_quality_claim": False, "started_at": time.time(),
               "generation_profile": generation_profile(spec), "evaluation_execution": {},
               "loss_evaluations": {}, "loss_cases_sha256": loss_sha256 if loss_rows else None,
               "eval_cases_sha256": cases_sha256 if cases else None,
               "data_sha256": data_sha256 if training_rows else None,
               "data_selection_receipt": selection_binding,
               "training_loss": loss_profile, "training_attention": attention_profile,
               "resume_training_binding": parent_training_binding}
    for path in (Path(__file__), ROOT / "src/mlx2/adapters/kimi_linear.py",
                 ROOT / "src/mlx2/runtime/models/kimi_linear.py",
                 ROOT / "src/mlx2/experimental/kimi_linear_lora/training.py",
                 ROOT / "src/mlx2/experimental/kimi_linear_lora/checkpoint.py",
                 ROOT / "scripts/research/kimi_lora_loss.py",
                 ROOT / "scripts/research/kimi_lora_data.py", *(ROOT / key for key in SAMPLING_CODE)):
        if path.exists():
            receipt["code_sha256"][str(path.relative_to(ROOT))] = sha(path)
    started = time.monotonic()
    adapter = session = mounted = None
    route_before = None
    try:
        if receipt["source_revision"] != spec["source_revision"]:
            raise ValueError("round source revision differs from its owned-unit specification")
        from mlx2.desktop_campaign_worker import check_ownership
        receipt["ownership"] = check_ownership(spec)
        receipt["host_before"] = host_state()
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from importlib.metadata import version
        receipt["runtime"] = {"python": sys.version, "mlx": version("mlx"),
            "tokenizers": version("tokenizers"), "transformers": version("transformers")}
        from mlx.utils import tree_flatten, tree_map
        from mlx2.adapters.kimi_linear import KimiLinearAdapter
        from mlx2.experimental.kimi_linear_lora.training import LoRATrainingSession, select_targets
        from mlx2.experimental.kimi_linear_lora.checkpoint import save_checkpoint, restore_checkpoint
        from mlx2.runtime.lora import install_lora
        from mlx2.experimental.weight_space.programs import inspect_program, load_program
        from mlx2.runtime.weight_space_metrics import weight_space_metrics_snapshot
        mx.set_cache_limit(int(spec.get("cache_limit_bytes", 1 << 30)))
        mx.set_memory_limit(int(spec.get("max_metal_bytes", 100 << 30)))
        mx.random.seed(int(spec.get("seed", 1371701)))
        adapter = KimiLinearAdapter(str(model_path), execution_policy=spec.get("execution_policy"))
        receipt["artifact"] = adapter.identity
        receipt["actual_execution"] = actual_execution(adapter)
        if adapter.identity["fingerprint"] != spec["base_fingerprint"]:
            raise RuntimeError("base copy differs from the pinned artifact identity")
        receipt["checkpoint"] = spec["artifact_name"] + "@" + adapter.identity["fingerprint"]
        guard = lambda: gpu_guard(mx, spec, started)
        receipt["loaded_memory"] = guard()
        smoke = generate(adapter, [{"role": "user", "content": "What is 2 + 2? Reply with the integer only."}], max_tokens=32, guard=guard)
        receipt["ordinary_smoke"] = smoke
        if smoke["text"].strip() != "4":
            raise RuntimeError("ordinary model smoke failed")
        receipt["candidate_selection"] = selected
        receipt["evaluation_case_ids"] = [case["id"] for case in cases]
        evaluate_base = spec.get("evaluate_base", spec["mode"] in {"baseline", "probe"} or spec.get("eval_split") == "final")
        if evaluate_base and cases:
            receipt["evaluation_execution"]["base"] = actual_execution(adapter)
            receipt["evaluations"]["base"] = evaluate(adapter, cases, results / "base.jsonl", guard=guard, sampling_spec=spec)
        if evaluate_base and loss_rows:
            receipt["loss_evaluations"]["base"] = evaluate_loss(adapter.model, loss_rows, guard=guard)
        if spec["mode"] in {"probe", "train"}:
            rows = training_rows
            keys = tuple(spec.get("keys") or select_targets(adapter.model,
                last_n_layers=int(spec.get("last_n_layers", 1)),
                modules=tuple(spec.get("modules", ["attention.o_proj", "attention.q_proj"]))))
            session = LoRATrainingSession(adapter.model, keys, rank=int(spec.get("rank", 4)),
                scale=float(spec.get("scale", 1.0)), base_checkpoint=receipt["checkpoint"],
                source_revision=receipt["source_revision"], base_paths=(model_path, Path(spec["original_model"])),
                kda_policy=attention_profile["kda"], mla_policy=attention_profile["mla"],
                projection_chunk_size=loss_profile["projection_chunk_size"])
            receipt["training_attention_policy"] = {"kda": session.kda_policy, "mla": session.mla_policy, "mla_flash": "off",
                "frozen_prefix_layers": session.first_layer}
            route_before = {str(i): dict(layer.attention.calls) for i, layer in enumerate(adapter.model.layers)
                if i >= session.first_layer}
            receipt["training_route_counter_scope"] = (
                "frontend invocation/graph-construction counters; per-step snapshots follow evaluated update; "
                "not an observed Metal kernel or blocked-SDPA-VJP trace")
            receipt["target_keys"] = keys
            receipt["trainable_parameters"] = {k: list(v.shape) for k, v in tree_flatten(adapter.model.trainable_parameters())}
            if spec.get("resume_program"):
                if float(spec.get("scale", 1.0)) != 1.0:
                    raise ValueError("resuming folded-scale programs requires scale=1")
                parent = load_program(spec["resume_program"])
                if parent.checkpoint != receipt["checkpoint"]:
                    raise ValueError("resume program base checkpoint mismatch")
                session.restore_weights({f"{factor.target}.{leaf}": mx.array(value)
                    for factor in parent.factors for leaf, value in (("lora_a", factor.a), ("lora_b", factor.b))})
                receipt["parent_program"] = parent.manifest()
            optimizer = optim.Adam(learning_rate=float(spec.get("learning_rate", 3e-5)))
            receipt["optimizer"] = {"type": "mlx.optimizers.Adam", "learning_rate": float(optimizer.learning_rate.item()),
                "betas": optimizer.betas, "eps": optimizer.eps, "bias_correction": optimizer.bias_correction,
                "max_grad_norm": float(spec.get("max_grad_norm", 1.0))}
            cursor = 0
            binding = {key: receipt[key] for key in (
                "checkpoint", "source_revision", "code_sha256", "data_sha256", "spec_sha256",
                "training_loss", "training_attention")}
            if spec.get("resume_checkpoint"):
                parent = restore_checkpoint(spec["resume_checkpoint"], session, optimizer,
                    expected_provenance={**binding, "spec_sha256": parent_training_binding["sha256"]})
                cursor = parent["next_data_cursor"]
                receipt["resumed_checkpoint"] = {key: parent[key] for key in
                    ("checkpoint_id", "next_data_cursor", "optimizer_step", "provenance")}
            receipt["training_checkpoints"] = []
            def checkpoint_step():
                checkpoint_path = results / "checkpoints" / f"step-{int(optimizer.step.item()):06d}"
                if receipt["training_checkpoints"] and receipt["training_checkpoints"][-1]["path"] == str(checkpoint_path):
                    return
                saved = save_checkpoint(checkpoint_path, session, optimizer,
                    next_data_cursor=cursor, provenance=binding)
                receipt["training_checkpoints"].append({"path": str(checkpoint_path),
                    **{key: saved[key] for key in ("checkpoint_id", "next_data_cursor", "optimizer_step", "state_sha256")}})
            def loss_fn(model, tokens, mask):
                return session.loss(tokens[:, :-1], tokens[:, 1:], mask[:, 1:], checkpoint_layers=True)
            loss_grad = nn.value_and_grad(adapter.model, loss_fn)
            for step in range(int(spec.get("steps", 4))):
                row = rows[cursor % len(rows)]
                tokens = mx.array([row["input_ids"]], mx.int32)
                mask = mx.array([row["loss_mask"]], mx.float32)
                before = time.monotonic()
                loss, gradients = loss_grad(adapter.model, tokens, mask)
                norm = mx.sqrt(sum(mx.sum(g.astype(mx.float32) ** 2) for _, g in tree_flatten(gradients)))
                mx.eval(loss, norm)
                loss_value, norm_value = float(loss.item()), float(norm.item())
                gradient_norms = {key: mx.sqrt(mx.sum(value.astype(mx.float32) ** 2))
                    for key, value in tree_flatten(gradients)}
                mx.eval(gradient_norms)
                if not math.isfinite(loss_value) or not math.isfinite(norm_value) or norm_value == 0:
                    raise RuntimeError("nonfinite or zero training gradient")
                factor = min(1.0, float(spec.get("max_grad_norm", 1.0)) / norm_value)
                gradients = tree_map(lambda g: g * factor, gradients)
                optimizer.update(adapter.model, gradients)
                mx.eval(adapter.model.trainable_parameters(), optimizer.state)
                cursor += 1
                receipt["training"].append({"step": step + 1, "record_id": row["id"], "tokens": len(row["input_ids"]),
                    "loss": loss_value, "gradient_norm": norm_value, "elapsed_seconds": time.monotonic() - before,
                    "gradient_norms": {key: float(value.item()) for key, value in gradient_norms.items()},
                    "memory": {"active_bytes": mx.get_active_memory(), "peak_bytes": mx.get_peak_memory()}})
                # Capture after evaluated loss/gradients/update, before the
                # resource guard can abort. This copies Python counters only.
                route_snapshot = training_route_deltas(adapter.model, session.first_layer, route_before)
                receipt["training"][-1]["training_route_deltas"] = route_snapshot
                receipt["training_route_deltas"] = route_snapshot
                interval = max(1, int(spec.get("checkpoint_interval", 4)))
                if (step + 1) % interval == 0 or step + 1 == int(spec.get("steps", 4)):
                    checkpoint_step()
                try:
                    guard()
                except (TimeoutError, MemoryError):
                    checkpoint_step()
                    raise
                atomic(results / "receipt.json", receipt)
                print(json.dumps({"train": receipt["training"][-1]}), flush=True)
            receipt["training_route_deltas"] = training_route_deltas(
                adapter.model, session.first_layer, route_before)
            target = results / "adapter"
            exported = session.export_program(target, {"experiment": "kimi-linear-reasoning-lora",
                "spec_sha256": receipt["spec_sha256"], "data_sha256": receipt["data_sha256"], "steps": len(receipt["training"]),
                "code_sha256": receipt["code_sha256"], "training_loss": loss_profile})
            receipt["export"] = inspect_program(exported)
            session.close(); session = None
            mounted = install_lora(adapter.model, name=spec["job_id"], path=target, base_checkpoint=receipt["checkpoint"])
            receipt["adapter_path"] = str(target)
        elif spec.get("adapter"):
            mounted = install_lora(adapter.model, name=spec["job_id"], path=spec["adapter"], base_checkpoint=receipt["checkpoint"])
        if selected is not None:
            verify_selected_mount(selected, mounted)
        if mounted is not None and cases:
            receipt["evaluation_execution"]["adapter"] = actual_execution(adapter)
            receipt["evaluations"]["adapter"] = evaluate(adapter, cases, results / "adapter.jsonl", guard=guard, sampling_spec=spec)
        if mounted is not None and loss_rows:
            receipt["loss_evaluations"]["adapter"] = evaluate_loss(adapter.model, loss_rows, guard=guard)
        if mounted is not None:
            receipt["adapter_fingerprint"] = mounted.fingerprint
            receipt["adapter_applications"] = mounted.program_application_count
            if cases and mounted.program_application_count <= 0:
                raise RuntimeError("adapter program did not engage")
            mounted.restore(adapter.model); mounted = None
            restored = generate(adapter, [{"role": "user", "content": "What is 2 + 2? Reply with the integer only."}], max_tokens=32, guard=guard)
            receipt["restored_base_smoke"] = restored
            if restored["tokens"] != smoke["tokens"]:
                raise RuntimeError("base output did not restore after adapter removal")
        receipt["weight_space_metrics"] = json_metrics(weight_space_metrics_snapshot())
        receipt["final_memory"] = guard()
        receipt["status"] = "completed"
    except BaseException as exc:
        receipt["status"] = "failed"
        receipt["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        raise
    finally:
        if mounted is not None:
            mounted.restore(adapter.model)
        if session is not None:
            if route_before is not None:
                try:
                    receipt["training_route_deltas"] = training_route_deltas(
                        adapter.model, session.first_layer, route_before)
                    receipt["training_route_snapshot_phase"] = "cleanup_before_session_close"
                except Exception as snapshot_error:
                    # Diagnostic capture must not replace the original failure.
                    receipt["training_route_snapshot_error"] = {
                        "type": type(snapshot_error).__name__, "message": str(snapshot_error)}
            session.close()
        receipt["finished_at"] = time.time()
        receipt["elapsed_seconds"] = time.monotonic() - started
        receipt["host_after"] = host_state()
        atomic(results / "receipt.json", receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt(f"signal {signum}")))
    run(args.spec)
