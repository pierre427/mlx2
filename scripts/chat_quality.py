"""Chat-faithful quality scoring for teacher-forced A/B scripts.

Chat models are usually fine-tuned with the loss masked off prompt tokens, so
their predictions at system, user, tool-result and template positions are
untrained: on LFM2.5 24B-A2B they cost 8-9 nats against 0.2 on assistant
tokens (qualification/runs/lfm25-24b-a2b-int8-20261014/role_nll.json).  KL
measured there says nothing about what the model generates.  Two remedies:

- :func:`assistant_mask` restricts a teacher-forced comparison of a rendered
  conversation to positions whose *target* token belongs to an assistant turn.
- :func:`own_continuations` builds real chat prompts through the adapter's
  serving path, generates the stock model's greedy continuation, and
  :func:`continuation_logprobs` scores an arm on exactly those tokens: the
  prompt is prefilled under the arm (int8 engages at >= 512 rows), the
  continuation is teacher-forced in one short call, as decode would see it.
"""

from __future__ import annotations

import json

# (start marker, end marker, assistant role name) per template family.
_TURN_FORMATS = (
    ("<|im_start|>", "<|im_end|>", "assistant"),   # ChatML: Qwen, LFM2.5
    ("<start_of_turn>", "<end_of_turn>", "model"),  # Gemma
)


def _single_id(tokenizer, text):
    try:
        ids = list(tokenizer.encode(text, add_special_tokens=False))
    except Exception:  # noqa: BLE001 - absent marker
        return None
    return ids[0] if len(ids) == 1 else None


def assistant_mask(tokenizer, ids):
    """Per token: True inside an assistant turn's body (its end marker included).

    ``None`` when the text has no recognised turn markers.  For logprob rows
    (row ``j`` predicts ``ids[j + 1]``) use ``mask[1:]``.
    """
    for start_text, end_text, assistant in _TURN_FORMATS:
        start, end = _single_id(tokenizer, start_text), _single_id(tokenizer, end_text)
        if start is None or end is None or start not in ids:
            continue
        mask, state, header = [], "outside", []
        for token in ids:
            if token == start:
                state, header = "header", []
                mask.append(False)
            elif state == "header":
                header.append(token)
                mask.append(False)
                if "\n" in tokenizer.decode([token]):
                    role = tokenizer.decode(header).strip()
                    state = "assistant" if role == assistant else "other"
            elif state == "assistant":
                mask.append(True)
                if token == end:
                    state = "outside"
            else:
                mask.append(False)
                if token == end:
                    state = "outside"
        return mask
    return None


def _tool_conversation(n_turns):
    tools = [{"type": "function", "function": {
        "name": "lookup_record", "description": "Fetch one record from the record store.",
        "parameters": {"type": "object", "properties": {
            "record_id": {"type": "string"}, "fields": {"type": "array", "items": {"type": "string"}}},
            "required": ["record_id"]}}}]
    messages = []
    for i in range(n_turns):
        rid = f"rec-{i * 7919 % 10007}"
        args = {"record_id": rid, "fields": ["title", "owner", "status"][: i % 3 + 1]}
        messages += [
            {"role": "user", "content": f"Look up record {rid}."},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": f"call_{i}", "type": "function",
                "function": {"name": "lookup_record", "arguments": json.dumps(args)}}]},
            {"role": "tool", "tool_call_id": f"call_{i}", "content": json.dumps(
                {"record_id": rid, "title": f"Item {i}", "owner": f"user{i % 5}",
                 "status": ["open", "closed", "pending"][i % 3]})},
        ]
    messages.append({"role": "user", "content":
                     "Look up record rec-4242, then tell me in one sentence who owns it."})
    return {"messages": messages, "tools": tools}


def own_continuations(adapter, repo, *, min_prompt=640, max_prompt=6144, new_tokens=160):
    """Stock-generated continuations of three served-style chat prompts.

    Returns ``{name: (prompt_ids, continuation_ids)}``.  Prompts are rendered
    with ``adapter.prompt_tokens`` (the serving path: template, tools,
    thinking) and are at least ``min_prompt`` tokens so an int8 prefill arm
    engages on them.
    """
    import mlx.core as mx

    prose = "\n\n".join(p.read_text() for p in sorted((repo / "docs").glob("*.md"))[:40])
    code = "\n\n".join(p.read_text() for p in sorted((repo / "src/mlx2/runtime").glob("*.py"))[:30])
    tokenizer = adapter.tokenizer

    def clip(text, budget):
        ids = list(tokenizer.encode(text, add_special_tokens=False))[:budget]
        return tokenizer.decode(ids)

    requests = {
        "own_tool_chat": _tool_conversation(24),
        "own_summary": {"messages": [{"role": "user", "content":
            clip(prose, 1200) + "\n\nSummarize the text above in three sentences."}]},
        "own_code": {"messages": [{"role": "user", "content":
            clip(code, 1200) + "\n\nExplain what the first function above does."}]},
    }
    eos = set(getattr(tokenizer, "eos_token_ids", ()) or ())
    model = adapter.model
    out = {}
    for name, request in requests.items():
        prompt = list(adapter.prompt_tokens(request))
        if not min_prompt <= len(prompt) <= max_prompt:
            raise ValueError(f"{name}: prompt is {len(prompt)} tokens, outside "
                             f"[{min_prompt}, {max_prompt}]")
        cache = model.make_cache()
        row = model(mx.array(prompt, mx.uint32)[None], cache=cache)[0, -1]
        generated = []
        for _ in range(new_tokens):
            token = int(mx.argmax(row).item())
            generated.append(token)
            if token in eos:
                break
            row = model(mx.array([[token]], mx.uint32), cache=cache)[0, -1]
        out[name] = (prompt, generated)
    return out


def continuation_logprobs(model, make_cache, prompt, continuation, *, prefill_chunk):
    """Logprob rows predicting each continuation token, after an arm's prefill.

    The prompt runs in ``prefill_chunk``-row slices (the arm's prefill path);
    the continuation runs as one teacher-forced call shorter than the int8 row
    threshold, so it stays on the stock kernels as decode does.
    """
    import mlx.core as mx

    cache = make_cache()
    x = mx.array(prompt, mx.uint32)[None]
    last = None
    for pos in range(0, x.shape[1], prefill_chunk):
        last = model(x[:, pos: pos + prefill_chunk], cache=cache)[0, -1:]
        mx.eval(last)
    rows = [last]
    if len(continuation) > 1:
        rows.append(model(mx.array(continuation[:-1], mx.uint32)[None], cache=cache)[0])
    logits = mx.concatenate(rows).astype(mx.float32)
    lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    mx.eval(lp)
    return lp


def compare(ref, lp, rows=None):
    """KL(ref||lp), top-1 agreement and tail stats, over ``rows`` if given."""
    import mlx.core as mx

    r32, l32 = ref.astype(mx.float32), lp.astype(mx.float32)
    if rows is not None:
        index = mx.array([i for i, keep in enumerate(rows) if keep])
        if index.size == 0:
            return {"positions": 0}
        r32, l32 = r32[index], l32[index]
    kl = (mx.exp(r32) * (r32 - l32)).sum(-1)
    top1 = (mx.argmax(r32, -1) == mx.argmax(l32, -1)).astype(mx.float32)
    mx.eval(kl, top1)
    return {"positions": int(kl.size), "mean_kl": round(kl.mean().item(), 5),
            "p99_kl": round(float(mx.sort(kl)[int(0.99 * (kl.size - 1))].item()), 4),
            "frac_kl_gt_1": round((kl > 1).astype(mx.float32).mean().item(), 4),
            "top1": round(top1.mean().item(), 4)}


def quality_evidence_manifest(quality):
    """Classify A/B cells by what they can say about generation quality.

    Whole rendered prompts contain system, user, tool-result and template rows.
    Those rows are useful diagnostics, but chat training commonly masks their
    loss, so they cannot veto a generation-quality promotion.  Assistant-only
    rows and stock-generated continuations exercise tokens the model actually
    predicts and are the promotion gates.
    """
    cells = sorted({cell for result in quality.values() for cell in result})
    gates = [cell for cell in cells if cell.startswith("own_") or cell.endswith("@assistant")]
    diagnostics = [cell for cell in cells if cell not in gates]
    return {
        "schema": "mlx2.int8-prefill-quality-evidence.v1",
        "promotion_gate_cells": gates,
        "diagnostic_only_cells": diagnostics,
        "rule": "own continuations and assistant-turn targets gate generation quality; all-position prompt rows are diagnostic only",
    }
