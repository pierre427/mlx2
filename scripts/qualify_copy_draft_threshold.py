#!/usr/bin/env python3
"""B1 qualifier: native self-MTP copy-draft strong threshold 32 -> 16 (threshold only).

Scope: DIRECT-MODEL, B1 only, not HTTP serving and not a serving
qualification. No route, default or policy is changed; the candidate policy
exists only inside this harness.

Arms differ in ``strong_match`` alone:

  s32  the local Flash-Next native self-MTP default copy-draft block, read
       from ``FlashNextAdapter.default_route_execution_policy`` and refused
       if it is not exactly max_span 7, min_match 8, strong_match 32,
       strong_max_span 16, initial_span 7
  s16  the same block with strong_match 16 (design input: ddalcu/mlx-serve
       #616, STRONG_SUFFIX 32 -> 16; see docs/PROVENANCE.md)

Both arms use the same model load, the Flash-Next ``execution_config`` at
one lane, greedy sampling with the same LaneRNG seed, the same pinned prompt
token IDs, EOS stops (``--ignore-eos`` is recorded), and a fused GDN verify
cap set explicitly to 17 and restored afterwards. Generic prompt lookup is
not touched.

Engagement. The arms can only differ on a copy round whose source agrees
with the live context for 16..31 tokens and whose sizer width and cap exceed
max_span 7. A harness-local observer wraps ``CopyDraftState.plan`` (restored
on exit, decisions unchanged) and counts, per run, lookups by agreement band
and these differential rounds. The candidate must show differential rounds
on the copy prompts; the baseline must never copy more than 7 tokens in the
band; the negative-control prompts must never reach the band.

Protocol per prompt: one discarded warm-up per arm, an ordinary (no
self-MTP) B1 reference for tokens, then ``--pairs`` alternating pairs
(s32,s16 / s16,s32 / ...). Per run: tokens, logprob-row storage bits,
target and draft cache ``state_digest`` (state + meta_state), an ordinary
B1 continuation from the final cache, the self-MTP and copy-draft receipts,
memory sampled while the generator is alive, and decode seconds. Every run
of an arm must equal every other run of that arm, and each pair's s16 run
must equal its s32 run, bit for bit. Missing evidence is never equal:
logprob rows must cover every emitted token (``--logprob-rows`` below
``--gen`` is a partial sample and makes the cell incomparable), a
continuation must carry its full requested token count and a complete
nested cache digest, and a run without a memory sample is incomplete.
Verdict: refused > counterexample > exact_with_unavailable_parts > pass.
Latencies are paired diagnostics, not a controlled performance run.

  PYTHONPATH=src .venv/bin/python scripts/qualify_copy_draft_threshold.py --tiny --out /tmp/t.json
  PYTHONPATH=src .venv/bin/python scripts/qualify_copy_draft_threshold.py --i-own-the-gpu \\
      --model ~/mlx-models/Qwen3.8-Flash-Next-MLX-4bit-MTP --out r.json
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

os.environ.setdefault("MLX_ENABLE_TF32", "0")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

SCHEMA = "mlx2.direct-model.copy-draft-strong-threshold.v1"
BASELINE = {"enabled": True, "max_span": 7, "min_match": 8, "strong_match": 32,
            "strong_max_span": 16, "initial_span": 7}
ARMS = {"s32": dict(BASELINE), "s16": dict(BASELINE, strong_match=16)}
BASE_ARM, CANDIDATE_ARM = "s32", "s16"
BAND = (16, 31)  # agreements where only the candidate takes the strong cap
VERIFY_CAP = 17
TINY_SELF_MTP = {"num_draft": 2, "persistent": True, "rate_gate": False, "prefill_step_size": 8}
UPSTREAM = {
    "repository": "https://github.com/ddalcu/mlx-serve", "pull": 616, "merged": True,
    "head": "07675f8578a471137466566d93f7805c89d6896c", "base": "2496d200148c2526d0fc6c618814f7271a89543b",
    "merge_commit": "e763f7e5cdc4f5b2dd2f2e45313e48b8b295d333",
    "file": "src/mtp_lookup.zig", "file_blob": "219be71ca6899e26033be7f6dec87ad386cce228",
    "license": "MIT", "license_blob": "fa6c781bf4ad462d965a429bd1c1c33475d3905a",
    "change": "STRONG_SUFFIX 32 -> 16; MAX_DRAFT 7 and MAX_DRAFT_STRONG 14 unchanged",
    "use": "design input only; no code copied (local strong_max_span stays 16)",
}
MAX_PAIRS, MAX_GEN, MAX_ROWS, MAX_CONTINUATION, MAX_CONTEXT = 8, 512, 512, 64, 16384
IDENTITY_FILES = (
    "scripts/qualify_copy_draft_threshold.py", "scripts/paired_direct_ab.py",
    "scripts/qualify_gdn_retirement.py", "scripts/ab_copy_mtp.py",
    "src/mlx2/runtime/copy_draft.py", "src/mlx2/runtime/hybrid_speculative.py",
    "src/mlx2/runtime/generate.py", "src/mlx2/runtime/models/qwen4_fused_gdn_verify.py",
    "src/mlx2/adapters/flash_next.py", "src/mlx2/adapters/flash_next_policy.py",
)
QUOTE = (
    "Copy the following passage exactly, word for word, and output nothing else.\n\n"
    "The harbour pilot kept a ledger of every vessel that crossed the bar at night: the hour, the "
    "draught, the state of the tide, the colour of the running lights and the name painted on the "
    "stern. When fog came in from the east she wrote the soundings twice, once in pencil at the rail "
    "and once in ink at the chart table, so that a wet page could never lose a depth. In forty years "
    "the ledger filled eleven volumes, and the harbour board still keeps them in a cabinet beside the "
    "tide clock, where anyone may read how the channel moved."
)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prompt_texts():
    from scripts.ab_copy_mtp import CODE, PROSE

    return {"copy": [QUOTE, CODE[0]], "control": [PROSE[0], PROSE[3]]}


def tiny_prompts():
    segment = [1 + (j * 7) % 53 for j in range(40)]
    return {"copy": [segment + segment[:20]], "control": [[1 + (j * 5) % 60 for j in range(30)]]}


def adapter_default_copy_policy():
    """The native self-MTP copy-draft block the Flash-Next adapter serves by default."""
    from mlx2.adapters.flash_next import FlashNextAdapter

    return dict(FlashNextAdapter.default_route_execution_policy["native_mtp"]["self_mtp_copy_draft"])


# ================================================================ engagement

class PlanProbe:
    """Observe ``CopyDraftState.plan`` for one run; the original is restored on exit.

    The wrapper reads ``_find`` (host-only lookup) and ``width`` before the
    real call and records its result; it never changes a decision.
    """

    def __init__(self):
        self.events = []

    def __enter__(self):
        from mlx2.runtime.copy_draft import CopyDraftState

        self.cls, self.real = CopyDraftState, CopyDraftState.plan
        real, events = self.real, self.events

        def plan(state, *, head_depth, cap):
            found = state._find()
            width = state.width
            span, decision = real(state, head_depth=head_depth, cap=cap)
            events.append({"agreement": None if found is None else int(found[1]), "width": int(width),
                           "cap": int(cap), "decision": decision, "span": len(span),
                           "max_span": state.policy.max_span})
            return span, decision

        self.cls.plan = plan
        return self

    def __exit__(self, *exc):
        self.cls.plan = self.real
        return False


def summarize_plans(events):
    buckets = {"miss": 0, "8-15": 0, "16-31": 0, ">=32": 0}
    band = differential = wide_band = differential_wide = 0
    for e in events:
        a = e["agreement"]
        key = "miss" if a is None else "8-15" if a < BAND[0] else "16-31" if a <= BAND[1] else ">=32"
        buckets[key] += 1
        if key != "16-31":
            continue
        band += 1
        wide_band += e["span"] > e["max_span"]
        if e["decision"] == "copy" and min(e["width"], e["cap"]) > e["max_span"]:
            differential += 1
            differential_wide += e["span"] > e["max_span"]
    return {"plans": len(events), "agreement": buckets, "band_rounds": band,
            "differential_rounds": differential, "differential_wide_spans": differential_wide,
            "band_spans_over_max_span": wide_band,
            "note": "differential = band agreement, copy decision, min(width, cap) > max_span"}


def engagement_problems(arm, plans):
    """Invariants tying the observed spans to the arm's threshold."""
    if arm == BASE_ARM and plans["band_spans_over_max_span"]:
        return [f"{arm}: copied more than max_span in the {BAND[0]}..{BAND[1]} band (threshold not bound)"]
    if arm == CANDIDATE_ARM and plans["differential_wide_spans"] != plans["differential_rounds"]:
        return [f"{arm}: a differential round did not copy past max_span (threshold not bound)"]
    return []


# ================================================================ runs

class Context:
    def __init__(self, args):
        import mlx.core as mx

        self.args, self.mx = args, mx
        self.stops = ()
        if args.tiny:
            mx.set_default_device(mx.cpu)
            from scripts.qualify_gdn_retirement import tiny_flash_next

            self.model = tiny_flash_next()
            self.self_mtp, self.prefill_step = dict(TINY_SELF_MTP), 8
            self.prompts = tiny_prompts()
            self.identity = {"model": "tiny-random-qwen4_exp", "prompt_source": "tiny deterministic constructor"}
            return
        from mlx2.adapters.registry import resolve_adapter

        cls = resolve_adapter(args.model, mtp=True, qualification_mode=True)
        self.adapter = cls(args.model)  # before any runtime import (import-order guard)
        from mlx2.serving import generation_stop_token_ids

        self.model = self.adapter.model
        self.stops = () if args.ignore_eos else generation_stop_token_ids(self.adapter)
        self.prefill_step = self.adapter.prefill_step_default() or 2048
        self.self_mtp = dict(self.adapter.execution_config(max_lanes=1, prefill_step=self.prefill_step))
        if args.prompt_ids:
            data = json.loads(Path(args.prompt_ids).read_text())
            if not isinstance(data, dict) or set(data) != {"copy", "control"}:
                raise SystemExit("refused: --prompt-ids must be {\"copy\": [...], \"control\": [...]}")
            self.prompts = data
            source = f"explicit token file sha256 {_sha(Path(args.prompt_ids).read_bytes())}"
        else:
            texts = prompt_texts()
            self.prompts = {k: [list(self.adapter.tokenizer.apply_chat_template(
                [{"role": "user", "content": t}], add_generation_prompt=True, tokenize=True,
                enable_thinking=False)) for t in v] for k, v in texts.items()}
            source = {"constructor": "chat template, thinking off",
                      "text_sha256": {k: [_sha(t.encode()) for t in v] for k, v in texts.items()}}
        self.identity = {
            "model": str(args.model), "adapter": f"{cls.__module__}.{cls.__qualname__}",
            "adapter_sha256": _sha(Path(sys.modules[cls.__module__].__file__).read_bytes()),
            "fingerprint": self.adapter.identity.get("fingerprint"),
            "environment": dict(getattr(self.adapter, "environment", {}) or {}),
            "prompt_source": source,
        }

    def check_prompts(self):
        for name in ("copy", "control"):
            prompts = self.prompts.get(name)
            if not prompts:
                raise SystemExit(f"refused: no {name} prompts")
            for p in prompts:
                if not isinstance(p, list) or not 8 <= len(p) <= MAX_CONTEXT \
                        or any(type(t) is not int or t < 0 for t in p):
                    raise SystemExit(f"refused: each prompt must be 8..{MAX_CONTEXT} token ids")


def _reset(mx):
    gc.collect()
    mx.synchronize()
    mx.clear_cache()
    mx.reset_peak_memory()


def run_once(ctx, arm, prompt, gen_tokens, *, rows, continuation_tokens):
    """One B1 generation; ``arm`` None is the ordinary (no self-MTP) reference."""
    from mlx2.runtime.generate import BatchGenerator
    from mlx2.runtime.models import qwen4_fused_gdn_verify as FV
    from mlx2.runtime.sample_utils import LaneRNG
    from scripts.paired_direct_ab import state_digest

    mx, args = ctx.mx, ctx.args
    _reset(mx)
    record = {"arm": arm or "ordinary", "tokens": [], "logprob_rows": [], "finish_reason": None, "failures": []}
    final, first_t, last_t, polls = None, None, None, 0
    limit = gen_tokens * 8 + len(prompt) // 64 + 256
    deadline = time.monotonic() + args.time_limit_s
    probe = PlanProbe()
    gen = None
    with probe:
        try:
            kwargs = {} if arm is None else {"self_mtp": dict(ctx.self_mtp), "copy_draft": dict(ARMS[arm])}
            gen = BatchGenerator(ctx.model, completion_batch_size=1, prefill_batch_size=1,
                                 prefill_step_size=ctx.prefill_step, stop_tokens=[[t] for t in ctx.stops],
                                 **kwargs)
            record["verify_cap"] = FV.MAX_VERIFY_STEPS
            if arm is not None:
                record["held_copy_policy"] = gen.copy_draft.as_dict()
                insert = {"lane_rngs": [LaneRNG(args.seed)], "self_mtp_configs": [{"sampling_temp": 0.0}]}
            else:
                insert = {}
            (uid,) = gen.insert([list(prompt)], max_tokens=[gen_tokens], **insert)
            while final is None:
                polls += 1
                if polls > limit or time.monotonic() > deadline:
                    record["failures"].append(f"bounded: stopped after {polls} polls")
                    break
                _, responses = gen.next()
                lost = gen.take_lane_failures()
                if lost:
                    record["failures"] += [str(f) for f in lost]
                    break
                for response in responses:
                    now = time.perf_counter()
                    first_t = now if first_t is None else first_t
                    last_t = now
                    record["tokens"].append(int(response.token))
                    row = getattr(response, "logprobs", None)
                    if len(record["logprob_rows"]) < rows:
                        record["logprob_rows"].append(None if row is None else state_digest([row])["sha256"])
                    if response.finish_reason:
                        record["finish_reason"] = response.finish_reason
                        final = response
            mx.synchronize()
            record["memory"] = {"active_bytes": mx.get_active_memory(), "cache_bytes": mx.get_cache_memory(),
                                "peak_bytes": mx.get_peak_memory(), "note": "sampled while the generator is alive"}
        finally:
            if gen is not None:
                gen.close()
    n = len(record["tokens"])
    record.update(token_sha256=_sha(json.dumps(record["tokens"]).encode()), polls=polls,
                  decode_s=(last_t - first_t) if n > 1 else None, decode_tokens=max(n - 1, 0))
    if arm is None:
        return record
    receipt = getattr(final, "mtp_receipt", None) or {}
    record["route"] = receipt.get("route")
    record["copy_draft"] = receipt.get("copy_draft")
    record["mtp_stats"] = {k: (receipt.get("stats") or {}).get(k) for k in
                           ("cycles", "draft_proposed", "draft_accepted", "total_emitted")}
    record["plans"] = summarize_plans(probe.events)
    cache = getattr(final, "prompt_cache", None)
    mtp_state = getattr(final, "mtp_state", None)
    draft = mtp_state[0] if isinstance(mtp_state, tuple) and mtp_state else None
    record["target_state"], record["draft_state"] = state_digest(cache), state_digest(draft)
    record["continuation"] = continuation(ctx, prompt, record, final, continuation_tokens)
    return record


def continuation(ctx, prompt, record, final, tokens):
    """Ordinary B1 decode from the run's final cache (the reference path)."""
    from mlx2.runtime.generate import BatchGenerator
    from scripts.paired_direct_ab import state_digest
    from scripts.qualify_gdn_retirement import covered_tokens

    if not tokens:
        return {"status": "unavailable", "reason": "disabled (--continuation-tokens 0)"}
    cache = getattr(final, "prompt_cache", None)
    covered = covered_tokens(cache)
    full = list(prompt) + record["tokens"]
    if cache is None or covered is None or not 0 < covered <= len(full):
        return {"status": "unavailable", "reason": "no final cache or no single covered offset"}
    gen = BatchGenerator(ctx.model, completion_batch_size=1, prefill_batch_size=1,
                         prefill_step_size=ctx.prefill_step)
    out = []
    try:
        gen.insert([full[covered:] + list(prompt[:2])], max_tokens=[tokens], caches=[cache],
                   all_tokens=[full[:covered]])
        last = None
        for _ in range(tokens * 8 + 64):
            _, responses = gen.next()
            for response in responses:
                out.append(int(response.token))
                last = response if response.finish_reason else last
            if last is not None:
                break
        if last is None:
            return {"status": "unavailable", "reason": "continuation did not finish"}
        digest = state_digest(last.prompt_cache)
        result = {"status": "complete", "tokens": out, "final_state": digest}
        problem = continuation_problem(result, tokens)
        return result if problem is None else {"status": "unavailable", "reason": problem,
                                               "tokens": out, "final_state": digest}
    except Exception as error:  # noqa: BLE001 - unavailable, never exact
        return {"status": "unavailable", "reason": f"{type(error).__name__}: {error}"[:200]}
    finally:
        gen.close()


# ================================================================ gates

def digest_complete(digest):
    """A ``state_digest`` result that is complete with a 64-hex sha256."""
    sha = digest.get("sha256") if isinstance(digest, dict) else None
    return (isinstance(digest, dict) and digest.get("status") == "complete" and isinstance(sha, str)
            and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha))


def continuation_problem(continuation, requested):
    """Why a continuation is not complete evidence (None when it is)."""
    if not isinstance(continuation, dict) or continuation.get("status") != "complete":
        return "continuation unavailable"
    tokens = continuation.get("tokens")
    if not isinstance(tokens, list) or len(tokens) != requested:
        return f"continuation has {len(tokens) if isinstance(tokens, list) else 'no'} of {requested} tokens"
    if not digest_complete(continuation.get("final_state")):
        return "continuation final state digest is not complete"
    return None


def memory_complete(run):
    memory = run.get("memory")
    return isinstance(memory, dict) and all(
        type(memory.get(k)) is int and memory[k] >= 0 for k in ("active_bytes", "cache_bytes", "peak_bytes"))


def compare(a, b, label, rows, continuation_tokens):
    """``(differences, incomparable)`` between two self-MTP runs.

    Logprob rows count only when they cover every emitted token of both runs:
    a partial sample cannot show that later rows agree.
    """
    differences, incomparable = [], []
    if a["tokens"] != b["tokens"]:
        return [f"{label}: tokens differ"], []
    lp = a["logprob_rows"] + b["logprob_rows"]
    covered = min(len(a["logprob_rows"]), len(b["logprob_rows"]))
    if not lp or None in lp:
        incomparable.append(f"{label}: logprob rows unavailable"
                            + (" (disabled (--logprob-rows 0))" if not rows else ""))
    elif a["logprob_rows"][:covered] != b["logprob_rows"][:covered]:
        differences.append(f"{label}: logprob row bits differ")
    elif len(a["logprob_rows"]) != len(a["tokens"]) or len(b["logprob_rows"]) != len(b["tokens"]):
        incomparable.append(f"{label}: logprob rows cover {covered} of {len(a['tokens'])} emitted tokens")
    for key in ("target_state", "draft_state"):
        if not (digest_complete(a[key]) and digest_complete(b[key])):
            status = lambda d: d.get("status") if isinstance(d, dict) else None  # noqa: E731
            incomparable.append(f"{label}: {key} not complete with a sha256 "
                                f"({status(a[key])}/{status(b[key])})")
        elif a[key] != b[key]:
            differences.append(f"{label}: {key} digest differs")
    ca, cb = a["continuation"], b["continuation"]
    problems = [p for p in (continuation_problem(ca, continuation_tokens),
                            continuation_problem(cb, continuation_tokens)) if p]
    if problems:
        incomparable.append(f"{label}: " + "; ".join(sorted(set(problems))))
    elif ca != cb:
        differences.append(f"{label}: continuation differs")
    for side, run in (("a", a), ("b", b)):
        if not memory_complete(run):
            incomparable.append(f"{label}: memory not captured ({side})")
    return differences, incomparable


def run_refusals(run, label, gen_tokens):
    refusals = [f"{label}: {f}" for f in run["failures"]]
    if run["finish_reason"] is None or (len(run["tokens"]) < gen_tokens and run["finish_reason"] != "stop"):
        refusals.append(f"{label}: early stop ({len(run['tokens'])}/{gen_tokens})")
    if run["verify_cap"] != VERIFY_CAP:
        refusals.append(f"{label}: fused GDN verify cap {run['verify_cap']} != {VERIFY_CAP}")
    if run["arm"] == "ordinary":
        return refusals
    requested = ARMS[run["arm"]]
    held = run.get("held_copy_policy") or {}
    if any(held.get(k) != v for k, v in requested.items()):
        refusals.append(f"{label}: generator copy-draft policy differs from the requested arm")
    receipt = run.get("copy_draft")
    if not isinstance(receipt, dict) or not receipt.get("enabled"):
        refusals.append(f"{label}: no copy-draft receipt")
    elif any(receipt.get("policy", {}).get(k) != v for k, v in requested.items() if k != "enabled"):
        refusals.append(f"{label}: copy-draft receipt policy differs from the requested arm")
    if not run.get("route"):
        refusals.append(f"{label}: no self-MTP route receipt")
    return refusals + [f"{label}: {p}" for p in engagement_problems(run["arm"], run["plans"])]


def evaluate(cells, args):
    """Gate a ``{"copy": [cell...], "control": [cell...]}`` result; pure (no MLX)."""
    refusals, differences, incomparable, latency = [], [], [], {}
    differential = 0
    for cls in ("copy", "control"):
        if not cells.get(cls):
            refusals.append(f"no {cls} prompts ran")
    for cls, prompt_cells in cells.items():
        for index, cell in enumerate(prompt_cells):
            name = f"{cls}{index}"
            runs = cell["pairs"]
            if not runs:
                refusals.append(f"{name}: no timed pairs")
                continue
            refusals += run_refusals(cell["reference"], f"{name} ordinary", args.gen)
            by_arm = {arm: [pair[arm] for pair in runs] for arm in ARMS}
            for arm, arm_runs in by_arm.items():
                for k, run in enumerate(arm_runs):
                    refusals += run_refusals(run, f"{name} pair {k} {arm}", args.gen)
                    if cls == "copy" and not (run.get("copy_draft") or {}).get("copy_rounds"):
                        refusals.append(f"{name} pair {k} {arm}: copy prompt never copied")
                    if cls == "control" and run["plans"]["band_rounds"]:
                        refusals.append(f"{name} pair {k} {arm}: negative control reached agreement "
                                        f"{BAND[0]}..{BAND[1]} ({run['plans']['band_rounds']} rounds)")
                for k, run in enumerate(arm_runs[1:], 1):  # repeatability within the arm
                    d, i = compare(arm_runs[0], run, f"{name} {arm} pair 0 vs {k}", args.logprob_rows,
                                   args.continuation_tokens)
                    differences += d
                    incomparable += i
            if cls == "copy":
                differential += sum(r["plans"]["differential_rounds"] for r in by_arm[CANDIDATE_ARM])
            ratios = []
            for k, pair in enumerate(runs):
                d, i = compare(pair[BASE_ARM], pair[CANDIDATE_ARM], f"{name} pair {k} s32 vs s16",
                               args.logprob_rows, args.continuation_tokens)
                differences += d
                incomparable += i
                a, b = pair[BASE_ARM]["decode_s"], pair[CANDIDATE_ARM]["decode_s"]
                if a and b:
                    ratios.append(b / a)
            latency[name] = {"s16_over_s32_decode_s": ratios,
                             "median": statistics.median(ratios) if ratios else None}
    if cells.get("copy") and differential <= 0:
        refusals.append(f"{CANDIDATE_ARM}: no strong-threshold differential round on the copy prompts "
                        f"(agreement {BAND[0]}..{BAND[1]} with width and cap above max_span)")
    verdict = ("refused" if refusals else "counterexample" if differences
               else "exact_with_unavailable_parts" if incomparable else "pass")
    return {"verdict": verdict, "refusals": refusals, "differences": differences, "incomparable": incomparable,
            "differential_rounds_copy": differential,
            "latency": {**latency, "note": "paired diagnostic, not a controlled performance run"}}


# ================================================================ driver

def model_mode(args):
    from mlx2.runtime.models import qwen4_fused_gdn_verify as FV

    ctx = Context(args)
    ctx.check_prompts()
    default_policy = adapter_default_copy_policy()
    previous_cap = FV.set_verify_max_steps(VERIFY_CAP)
    cells = {}
    try:
        for cls in ("copy", "control"):
            cells[cls] = []
            for prompt in ctx.prompts[cls]:
                warmup = {arm: run_once(ctx, arm, prompt, args.warmup_gen, rows=0, continuation_tokens=0)
                          for arm in ARMS}
                reference = run_once(ctx, None, prompt, args.gen, rows=0, continuation_tokens=0)
                pairs, orders = [], []
                for k in range(args.pairs):
                    order = (BASE_ARM, CANDIDATE_ARM) if k % 2 == 0 else (CANDIDATE_ARM, BASE_ARM)
                    orders.append(list(order))
                    pairs.append({arm: run_once(ctx, arm, prompt, args.gen, rows=args.logprob_rows,
                                                continuation_tokens=args.continuation_tokens)
                                  for arm in order})
                cells[cls].append({
                    "prompt_tokens": len(prompt), "prompt_sha256": _sha(json.dumps(prompt).encode()),
                    "warmup": {arm: {"tokens": len(r["tokens"]), "plans": r.get("plans")} for arm, r in warmup.items()},
                    "reference": reference, "orders": orders, "pairs": pairs,
                    "reference_token_match": {arm: [p[arm]["tokens"] == reference["tokens"] for p in pairs]
                                              for arm in ARMS},
                })
    finally:
        FV.set_verify_max_steps(previous_cap)
    body = evaluate(cells, args)
    if default_policy != BASELINE:
        body["refusals"].insert(0, f"baseline is not the local Flash-Next default copy-draft block "
                                   f"({default_policy})")
        body["verdict"] = "refused"
    body.update(
        cells=cells,
        protocol={"arms": ARMS, "baseline": BASE_ARM, "candidate": CANDIDATE_ARM, "band": list(BAND),
                  "adapter_default_copy_policy": default_policy, "verify_cap": VERIFY_CAP,
                  "verify_cap_restored_to": previous_cap, "self_mtp": ctx.self_mtp,
                  "prefill_step": ctx.prefill_step, "width": 1, "pairs": args.pairs, "gen": args.gen,
                  "warmup_gen": args.warmup_gen, "sampling": "greedy", "seed": args.seed,
                  "logprob_rows": args.logprob_rows, "continuation_tokens": args.continuation_tokens,
                  "ignore_eos": args.ignore_eos, "stop_tokens": list(ctx.stops),
                  "time_limit_s": args.time_limit_s,
                  "order": "alternating pairs (s32,s16 / s16,s32 / ...) after one discarded warm-up per arm",
                  "reference": "ordinary B1 tokens per prompt (reference_token_match is informational)"},
        upstream=UPSTREAM, identity=dict(ctx.identity),
    )
    return ctx, body


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tiny", action="store_true", help="deterministic tiny Flash-Next-class CPU model")
    ap.add_argument("--model")
    ap.add_argument("--i-own-the-gpu", action="store_true")
    ap.add_argument("--prompt-ids", help='JSON {"copy": [[ids]...], "control": [[ids]...]}')
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--gen", type=int, default=None, help="generated tokens per run (<= 512)")
    ap.add_argument("--warmup-gen", type=int, default=32)
    ap.add_argument("--logprob-rows", type=int, default=None,
                    help="rows digested per run (default: every emitted token; fewer is a partial sample)")
    ap.add_argument("--continuation-tokens", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--time-limit-s", type=float, default=1800.0)
    ap.add_argument("--ignore-eos", action="store_true")
    ap.add_argument("--out", required=True)
    return ap


def resolve_args(ap, argv=None):
    a = ap.parse_args(argv)
    if a.tiny:
        if a.i_own_the_gpu or a.model or a.prompt_ids:
            ap.error("--tiny runs a random CPU model; drop --i-own-the-gpu/--model/--prompt-ids")
        a.gen = 24 if a.gen is None else a.gen
    else:
        if not a.i_own_the_gpu:
            ap.error("refusing Metal execution without --i-own-the-gpu")
        if not a.model:
            ap.error("--model is required for a real run")
        a.gen = 256 if a.gen is None else a.gen
    a.logprob_rows = a.gen if a.logprob_rows is None else a.logprob_rows
    if not 1 <= a.pairs <= MAX_PAIRS:
        ap.error(f"--pairs 1..{MAX_PAIRS}")
    if not 1 <= a.gen <= MAX_GEN or not 1 <= a.warmup_gen <= MAX_GEN:
        ap.error(f"--gen and --warmup-gen 1..{MAX_GEN}")
    if not 0 <= a.logprob_rows <= MAX_ROWS or not 0 <= a.continuation_tokens <= MAX_CONTINUATION:
        ap.error(f"--logprob-rows 0..{MAX_ROWS} and --continuation-tokens 0..{MAX_CONTINUATION} "
                 "(0 = explicitly unavailable)")
    if not math.isfinite(a.time_limit_s) or a.time_limit_s <= 0:
        ap.error("--time-limit-s must be finite and positive")
    return a


def main(argv=None):
    ap = build_parser()
    args = resolve_args(ap, argv)
    from scripts.paired_direct_ab import mlx_identity, source_identity

    ctx, body = model_mode(args)
    record = {"schema": SCHEMA, "mode": "model",
              "scope": ("direct-model B1 copy-draft strong-threshold A/B (32 vs 16); not HTTP serving, not "
                        "serving qualification; no route, default or policy changed"
                        + ("; TINY random CPU model" if args.tiny else "")),
              **body}
    record["identity"].update(
        source=source_identity(), files={n: _sha((ROOT / n).read_bytes()) for n in IDENTITY_FILES},
        source_commit_env=os.environ.get("MLX2_INTAKE_SOURCE_COMMIT"),
        MLX_ENABLE_TF32=os.environ.get("MLX_ENABLE_TF32"), mlx=mlx_identity(ctx.mx))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: record[k] for k in ("verdict", "refusals", "differences", "incomparable",
                                               "differential_rounds_copy")}, indent=1))
    return 0 if record["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
