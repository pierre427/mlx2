#!/usr/bin/env python3
"""Chat-templated 16K FGN quality gate for Qwen3.8 Flash-Next.

Twenty source-bound QA prompts use the artifact's actual chat template and
assistant generation prefix. Each prompt is exactly 16384 tokens including
template/control tokens. The separate 128-token held-out perplexity corpus is
unchanged from v2. A post-</think> final answer and a stop token are required
before human rubric review. Numeric, semantic, and direct timing evidence stay
separate; this does not qualify a serving route.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts import eval_fused_group_norm_quality as base
from scripts import eval_fused_group_norm_quality_v2 as v2

SCHEMA = "mlx2.fused-group-norm-quality.v3-chat"
ASSISTANT_PREFIX = "<|im_start|>assistant\n<think>\n"


def render_prompt(tok, content: str) -> str:
    control_tokens = tuple(token for token in tok.get_added_vocab()
                           if token.startswith("<") and token.endswith(">"))
    if any(token in content for token in control_tokens):
        raise ValueError("user content contains a tokenizer control token")
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False,
        add_generation_prompt=True)
    if not isinstance(rendered, str) or not rendered.endswith(ASSISTANT_PREFIX):
        raise ValueError("Qwen3.8 chat template lacks expected assistant think prefix")
    return rendered


def choose_source_length(tok, source_text: str, question_tail: str,
                         context_tokens: int) -> tuple[int, list[int]]:
    """Find an exact token width despite BPE changes at the source boundary."""
    def encode_at(chars: int) -> list[int]:
        return base.encode(tok, render_prompt(tok, source_text[:chars] + question_tail))

    if len(encode_at(0)) >= context_tokens:
        raise ValueError("template and question consume context")
    if len(encode_at(len(source_text))) < context_tokens:
        raise ValueError("source corpus cannot fill chat-templated context")
    lo, hi = 0, len(source_text)
    while lo < hi:
        mid = (lo + hi) // 2
        ids = encode_at(mid)
        if len(ids) == context_tokens:
            return mid, ids
        if len(ids) < context_tokens:
            lo = mid + 1
        else:
            hi = mid
    # Token count near a BPE boundary need not be monotone. Search locally.
    for chars in range(max(0, lo - 256), min(len(source_text), lo + 256) + 1):
        ids = encode_at(chars)
        if len(ids) == context_tokens:
            return chars, ids
    raise ValueError("could not make an exact chat-templated prompt width")


def build_qa_chat(tok, context_tokens: int, cases=v2.CASES, sources=None):
    sources = v2.prepare_sources() if sources is None else sources
    out = []
    for index, case in enumerate(cases):
        order = (case.source,) + tuple(name for name in v2.QA_DOCS
                                       if name != case.source)
        segments = [(name, f"\n\n=== Source: {name} ===\n" +
                     sources[name].decode("utf-8")) for name in order]
        source_text = "".join(segment for _, segment in segments)
        question_tail = ("\n\nQuestion: " + case.question +
                         "\nAnswer directly using the source text. Keep the final "
                         "answer complete and under 80 words.\n")
        chars, prompt = choose_source_length(tok, source_text, question_tail,
                                             context_tokens)
        used = []
        remaining = chars
        for name, segment in segments:
            take = min(len(segment), remaining)
            if take:
                used.append({"path": name, "sha256": v2.sha(sources[name]),
                             "characters_used": take,
                             "characters_available": len(segment)})
            remaining -= take
            if remaining <= 0:
                break
        source_tokens = len(base.encode(tok, source_text[:chars]))
        if source_tokens <= 8192:
            raise ValueError("chat-templated source has <=8192 tokens")
        if len(prompt) != context_tokens:
            raise AssertionError("chat prompt width drift")
        out.append({"index": index, "source": case.source,
                    "question": case.question, "rubric_hints": case.rubric_hints,
                    "prompt": prompt,
                    "prompt_sha256": v2.sha(v2.bytes_for_ids(prompt)),
                    "source_files": used, "source_tokens": source_tokens,
                    "source_characters": chars,
                    "assistant_prefix": ASSISTANT_PREFIX})
    if len({x["prompt_sha256"] for x in out}) != len(out):
        raise ValueError("duplicate chat QA prompts")
    return out


def admitted_final(qa: dict) -> bool:
    """Format admission only; substantive correctness needs a human rubric."""
    final = qa["final_answer"]
    return (qa["stop_reason"] == "stop_token" and qa["think_closed"] and
            len(final) >= 20 and not final.startswith("<|"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="~/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP")
    parser.add_argument("--context-tokens", type=int, default=16384)
    parser.add_argument("--score-tokens", type=int, default=128)
    parser.add_argument("--gen-cap", type=int, default=1536)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--i-own-the-gpu", action="store_true")
    args = parser.parse_args(argv)
    if args.context_tokens != 16384 or args.score_tokens != 128 or args.gen_cap < 256:
        parser.error("v3 requires 16384 context tokens, 128 scores, gen-cap >=256")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=False, local_files_only=True)
    sources = v2.prepare_sources()
    cases = build_qa_chat(tok, args.context_tokens, sources=sources)
    ppl = v2.build_ppl(tok, args.context_tokens, args.score_tokens,
                       len(cases), sources)
    plan = {
        "schema": SCHEMA, "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "harness_sha256": v2.sha((ROOT / "scripts/eval_fused_group_norm_quality_v3.py").read_bytes()),
        "v2_harness_sha256": v2.sha((ROOT / "scripts/eval_fused_group_norm_quality_v2.py").read_bytes()),
        "kernel_sha256": v2.sha((ROOT / "src/mlx2/runtime/models/qwen4_fused_group_norm.py").read_bytes()),
        "chat_template_sha256": hashlib.sha256(tok.chat_template.encode()).hexdigest(),
        "model_config_sha256": v2.sha((Path(args.model) / "config.json").read_bytes()),
        "model_index_sha256": v2.sha((Path(args.model) / "model.safetensors.index.json").read_bytes()),
        "tokenizer_json_sha256": v2.sha((Path(args.model) / "tokenizer.json").read_bytes()),
        "model": args.model, "context_tokens": args.context_tokens,
        "score_tokens": args.score_tokens, "gen_cap": args.gen_cap,
        "source_file_hashes": {name: v2.sha(data) for name, data in sources.items()},
        "cases": [{k: value for k, value in case.items() if k != "prompt"}
                  for case in cases],
        "ppl_windows": [{k: value for k, value in window.items()
                         if k not in ("context", "continuation", "source_files")}
                        for window in ppl],
        "scope": "chat-templated direct-model QA and held-out PPL; not serving qualification",
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if not args.i_own_the_gpu:
        parser.error("refusing Metal without --i-own-the-gpu")

    import mlx.core as mx

    from mlx2.adapters.registry import resolve_adapter
    from mlx2.runtime.models import qwen4_fused_group_norm as fgn

    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        raise SystemExit("exclusive Metal GPU queue required")
    if fgn.fused_group_norm_enabled():
        raise SystemExit("start with production FGN lever off")
    adapter = resolve_adapter(args.model)(args.model)
    model, tok = adapter.model, adapter.tokenizer
    plan["loaded_identity"] = {
        "adapter_class": f"{type(adapter).__module__}.{type(adapter).__qualname__}",
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "tokenizer_class": f"{type(tok).__module__}.{type(tok).__qualname__}",
    }
    if hashlib.sha256(tok.chat_template.encode()).hexdigest() != plan["chat_template_sha256"]:
        raise SystemExit("loaded model tokenizer changed the chat template")
    cases_loaded = build_qa_chat(tok, args.context_tokens, sources=sources)
    ppl_loaded = v2.build_ppl(tok, args.context_tokens, args.score_tokens,
                              len(cases), sources)
    if ([c["prompt_sha256"] for c in cases_loaded] !=
            [c["prompt_sha256"] for c in cases] or
            [w["continuation_sha256"] for w in ppl_loaded] !=
            [w["continuation_sha256"] for w in ppl]):
        raise SystemExit("loaded model tokenizer changed the controlled inputs")
    cases, ppl = cases_loaded, ppl_loaded
    plan["mlx_version"] = mx.__version__
    plan["device"] = str(mx.default_device())
    plan["results"] = []
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for case, window in zip(cases, ppl):
        order = ("eager", "fused") if case["index"] % 2 == 0 else ("fused", "eager")
        arms, lps = {}, {}
        for arm in order:
            qa, lp, score_s, counters = v2.run_arm(
                mx, model, tok, case, window, arm == "fused", fgn, args.gen_cap)
            qa["final_answer_admitted"] = admitted_final(qa)
            qa["rubric_hint_hits"] = [hint for hint in case["rubric_hints"]
                                      if hint.lower() in qa["final_answer"].lower()]
            arms[arm] = {**qa, "score_seconds": score_s, "counters": counters}
            lps[arm] = lp
        eager_ppl, eager_nll = base.perplexity(mx, lps["eager"], window["continuation"])
        fused_ppl, fused_nll = base.perplexity(mx, lps["fused"], window["continuation"])
        numeric = base.divergence(mx, lps["eager"], lps["fused"], args.score_tokens)
        eid, fid = arms["eager"]["ids"], arms["fused"]["ids"]
        prefix = 0
        while prefix < min(len(eid), len(fid)) and eid[prefix] == fid[prefix]:
            prefix += 1
        plan["results"].append({
            "index": case["index"], "question": case["question"],
            "source": case["source"], "prompt_sha256": case["prompt_sha256"],
            "heldout_context_sha256": window["context_sha256"],
            "continuation_sha256": window["continuation_sha256"],
            "order": order, "arms": arms,
            "generation_common_prefix_tokens": prefix,
            "generation_identical": eid == fid,
            "heldout_perplexity": {
                "eager": eager_ppl, "fused": fused_ppl,
                "relative_delta": (fused_ppl - eager_ppl) / eager_ppl,
                "eager_nll": eager_nll, "fused_nll": fused_nll, **numeric},
            "human_rubric_review": "pending" if all(
                a["final_answer_admitted"] for a in arms.values()) else "ineligible_incomplete_final",
        })
        out.write_text(json.dumps(plan, indent=2) + "\n")
        print(f"case {case['index'] + 1}/20 admitted="
              f"{arms['eager']['final_answer_admitted']}/"
              f"{arms['fused']['final_answer_admitted']}", flush=True)
        del lps
        mx.clear_cache()
    deltas = [r["heldout_perplexity"]["relative_delta"] for r in plan["results"]]
    plan["aggregate"] = {
        "ppl_relative_delta_median": statistics.median(deltas),
        "ppl_relative_delta_max_abs": max(abs(v) for v in deltas),
        "admitted_answer_pairs": sum(
            all(a["final_answer_admitted"] for a in r["arms"].values())
            for r in plan["results"]),
        "rubric_review": "pending_human_review", "qualification": "not_qualified",
    }
    plan["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    out.write_text(json.dumps(plan, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
