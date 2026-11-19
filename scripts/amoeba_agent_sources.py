#!/usr/bin/env python3
"""Candidate docs for the new agent.* / dialog.* categories (taxonomy 1.5; Pierre 2026-11-18: "probe reasoning,
chain of thought, tool use ... model<->user and model<->harness dynamics"). CPU only.

Here model-generated text is the subject, so "no AI" does not apply; every doc records `origin`
("human" | "model:<generator>" | "mixed:<generator>") as the dataset states it, plus the source licence.

Each conversation is normalised to (system, turns[(role, text)], tools) and rendered as a plain transcript
("System: / User: / Assistant: / Tool:") with the source's own tool-call syntax left inside the turns, so the
probe text keeps <functioncall>, <tool_call>, JSON calls, think blocks and shell output as the model sees them.
Leaf assignment is by rule (label.method = "rule:<name>"), recorded per doc:

  agent.reasoning.cot              problem + worked solution (MATH/GSM8K human; NuminaMath model)
  agent.reasoning.thinking_trace   long think-block window from a reasoning trace
  agent.reasoning.self_correction  the trace window densest in verification/backtracking moves
  agent.tool_use.definitions       tool/function schema block + first user message
  agent.tool_use.calls             conversation whose assistant turns emit tool calls
  agent.tool_use.results           conversation dominated by tool results fed back
  agent.tool_use.error_recovery    a tool result reports an error and the assistant continues
  agent.tool_use.decision          tools offered; the assistant answers, asks or declines instead of calling
  agent.harness.{planning,loop,shell,file_edits,status}  windows of one agent trajectory by step kind
  dialog.system_prompt             conversation led by a non-trivial system prompt
  dialog.refusal                   an assistant turn declines
  dialog.clarification             the first assistant reply asks the user a question
  dialog.instruction_following     prompt with explicit format/length/style constraints
  dialog.multiturn                 >= 3 user turns, none of the above
Docs are 1,100-6,000 chars; all windows of one conversation share its group.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

MIN, MAX = 1100, 6000
ROLE = {"system": "system", "human": "user", "user": "user", "prompter": "user", "gpt": "assistant",
        "assistant": "assistant", "model": "assistant", "function": "tool", "tool": "tool", "observation": "tool",
        "ipython": "tool", "function_response": "tool", "function response": "tool", "environment": "tool"}
CALL = re.compile(r"<functioncall>|<tool_call>|<\|python_tag\|>|\"(?:name|function)\"\s*:\s*\"[\w.\-]+\"|^\s*\[\w+\(.*\)\]\s*$", re.M)
ERR = re.compile(r"\b(error|exception|traceback|failed|failure|not found|invalid|denied|timed? ?out|unavailable|status(?:_code)?\"?\s*:\s*\"?(?:4|5)\d\d)\b", re.I)
REFUSE = re.compile(r"^\W*(I'm sorry|I am sorry|Sorry, but|I can't|I cannot|I won't|I will not|As an AI|I'm not able|I am not able|"
                    r"I apologize, but|I'm unable|I am unable|Unfortunately, I (?:can't|cannot))", re.I)
SELFCHK = re.compile(r"\b(wait|hmm|actually|let me (?:check|verify|double-check|re-?examine|recompute|reconsider)|"
                     r"that's not right|that doesn't|I made (?:a|an) (?:mistake|error)|on second thought|but wait|"
                     r"let me re-?(?:do|think)|no, that|recheck|double-check)\b", re.I)
CONSTRAINT = re.compile(r"\b(in (?:exactly |at most |at least |no more than )?\d+ (?:words|sentences|paragraphs|bullet points)|"
                        r"do not use any commas|all lowercase|all capital letters|in json format|wrap your (?:entire )?response|"
                        r"end your response with|start your response with|highlight at least|include the keywords?|"
                        r"use markdown|format your answer|respond only with|answer with (?:one|a single) word)\b", re.I)
DEFAULT_SYS = re.compile(r"^(you are a helpful(?:,| and)? (?:honest )?(?:assistant|AI assistant)\.?)$", re.I)


EN_STOP = set("the of and to in is that for it you with on this are be as have not can what your but how".split())


def english(text):
    """Cheap English check: >= 12% of word tokens are common English function words (drops es/fr/zh chats)."""
    w = re.findall(r"[a-z']+", text.lower())[:600]
    return len(w) >= 20 and sum(x in EN_STOP for x in w) / len(w) >= 0.12


def norm_turns(rec):
    """-> (system, [(role, text)], tools_text) for the common chat schemas."""
    sysm, turns, tools = rec.get("system") or "", [], ""
    if isinstance(rec.get("tools"), (list, str)):
        tools = rec["tools"] if isinstance(rec["tools"], str) else "\n".join(
            t if isinstance(t, str) else json.dumps(t, ensure_ascii=False) for t in rec["tools"])
    conv = rec.get("conversations") or rec.get("messages") or rec.get("conversation") or rec.get("turns")
    if isinstance(conv, str):
        try:
            conv = json.loads(conv)
        except json.JSONDecodeError:
            conv = None
    if isinstance(conv, list):
        for m in conv:
            if not isinstance(m, dict):
                continue
            role = ROLE.get(str(m.get("from") or m.get("role") or "").lower())
            text = m.get("value") if m.get("value") is not None else m.get("content")
            if isinstance(text, list):
                text = "\n".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in text)
            text = str(text or "")
            if m.get("tool_calls"):
                text = (text + "\n" + json.dumps(m["tool_calls"], ensure_ascii=False)).strip()
            if m.get("reasoning_content"):
                text = f"<think>\n{m['reasoning_content']}\n</think>\n{text}"
            if role == "system":
                sysm = (sysm + "\n" + text).strip()
            elif role:
                turns.append((role, text))
    elif "output" in rec and ("input" in rec or "context" in rec):           # Llama-Nemotron, OpenCodeReasoning
        ctx = rec.get("input") if "input" in rec else rec.get("context")
        if isinstance(ctx, str):
            ctx = [{"role": "user", "content": ctx}]
        return norm_turns({"system": rec.get("system_prompt") or "", "messages": list(ctx or []) +
                           [{"role": "assistant", "content": rec["output"]}]})
    elif isinstance(rec.get("instruction"), str) and isinstance(rec.get("response"), str):   # Dolly
        q = rec["instruction"] + (("\n\n" + rec["context"]) if rec.get("context") else "")
        return norm_turns({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": rec["response"]}]})
    elif isinstance(rec.get("prompt"), str) and isinstance(rec.get("response") or rec.get("output") or rec.get("completion"), str):
        out = rec.get("response") or rec.get("output") or rec.get("completion")
        return norm_turns({"system": rec.get("system") or "", "messages": [{"role": "user", "content": rec["prompt"]},
                                                                           {"role": "assistant", "content": out}]})
    elif isinstance(rec.get("context"), list) and rec.get("response1"):        # HelpSteer3 preference
        return norm_turns({"messages": rec["context"] + [{"role": "assistant", "content": rec["response1"]}]})
    elif isinstance(rec.get("chat"), str):                                    # glaive
        for part in re.split(r"\n\n\n(?=(?:USER|ASSISTANT|FUNCTION RESPONSE):)", rec["chat"].strip()):
            m = re.match(r"(USER|ASSISTANT|FUNCTION RESPONSE):\s*(.*)", part, re.S)
            if m:
                turns.append(({"USER": "user", "ASSISTANT": "assistant"}.get(m.group(1), "tool"),
                              m.group(2).replace("<|endoftext|>", "").strip()))
        sysm = re.sub(r"^SYSTEM:\s*", "", sysm)
    elif isinstance(rec.get("chosen"), str):                                   # hh-rlhf
        for part in re.split(r"\n\n(?=(?:Human|Assistant):)", rec["chosen"].strip()):
            m = re.match(r"(Human|Assistant):\s*(.*)", part, re.S)
            if m:
                turns.append(("user" if m.group(1) == "Human" else "assistant", m.group(2).strip()))
    return sysm.strip(), turns, tools.strip()


def render(sysm, turns, tools="", limit=MAX):
    parts = []
    if sysm:
        parts.append(f"System: {sysm}")
    if tools and tools not in sysm:
        parts.append(f"Tools:\n{tools}")
    for r, t in turns:
        parts.append(f"{r.capitalize()}: {t}")
    out, n = [], 0
    for p in parts:
        if n + len(p) > limit:
            if n < MIN:
                out.append(p[: limit - n])
            break
        out.append(p)
        n += len(p) + 2
    return "\n\n".join(out)


class Out:
    def __init__(self, path):
        self.f = open(path, "w")
        self.n = Counter()

    def put(self, node, text, group, source, license, origin, register, rule, **kw):
        if len(text) < MIN:
            return False
        text = text[:MAX]
        r = {"id": hashlib.sha256((node + "\0" + source + "\0" + text).encode()).hexdigest()[:20], "group": group,
             "register": register, "text": text, "source": source, "source_revision": kw.pop("rev"),
             "license": license, "redistribute": kw.pop("redistribute", "attribution"), "origin": origin,
             "created": kw.pop("created", ""), "cands": [node], "primary": node,
             "label": {"method": f"rule:{rule}"}, **kw}
        self.f.write(json.dumps(r, ensure_ascii=False) + "\n")
        self.n[(node, register)] += 1
        return True

    def close(self, what):
        self.f.close()
        tot = Counter()
        for (n, _), v in self.n.items():
            tot[n] += v
        print(f"{what}: {sum(self.n.values())} docs; {dict(sorted(tot.items()))}", flush=True)


def conv_key(name, rec, k):
    """One stable key per conversation, shared by the tools and harness passes (enumeration index only as a last
    resort, salted with the first user text so two passes over the same file agree)."""
    rid = rec.get("id") or rec.get("instance_id") or rec.get("uuid") or rec.get("conversation_id")
    if rid:
        return f"{name}/{rid}"
    _, turns, _ = norm_turns(rec)
    first = next((t for r, t in turns if r == "user"), "")
    reply = next((t for r, t in turns if r == "assistant"), "")
    # full first user + assistant turns: IFEval-style prompts share long constraint boilerplate
    return f"{name}/h{hashlib.sha256((first + chr(0) + reply).encode()).hexdigest()[:16]}" if first else f"{name}/{k}"


SHARED = {"toucan", "nemotron_agentic"}        # read by both passes: tools takes even hashes, harness odd ones


def h8(s):
    """Selection hash, salted so it is independent of the split hash (split_of uses sha256(group)[:8] % 8)."""
    return int(hashlib.sha256(("select\0" + s).encode()).hexdigest()[:8], 16)


def rows(path):
    """Yield records from json / jsonl(.gz) / parquet files under path (a file or directory)."""
    files = [path] if os.path.isfile(path) else sorted(
        f for f in glob.glob(os.path.join(path, "**/*"), recursive=True)
        if f.endswith((".json", ".jsonl", ".jsonl.gz", ".json.gz", ".parquet")) and "/.cache/" not in f
        and not f.endswith("FETCH.json"))
    for f in files:
        if f.endswith(".parquet"):
            import pyarrow.parquet as pq
            for b in pq.ParquetFile(f).iter_batches(batch_size=2000):
                for r in b.to_pylist():
                    yield f, r
        elif f.endswith((".jsonl", ".jsonl.gz")):
            op = gzip.open if f.endswith(".gz") else open
            for line in op(f, "rt"):
                try:
                    yield f, json.loads(line)
                except json.JSONDecodeError:
                    continue
        else:
            op = gzip.open if f.endswith(".gz") else open
            try:
                d = json.load(op(f, "rt"))
            except json.JSONDecodeError:                      # BFCL ships jsonl under .json
                for line in op(f, "rt"):
                    try:
                        yield f, json.loads(line)
                    except json.JSONDecodeError:
                        continue
                continue
            for r in (d if isinstance(d, list) else d.get("data", [d]) if isinstance(d, dict) else []):
                yield f, r


# ------------------------------------------------------------------ tool use
def tool_conv(o, name, rev, lic, origin, f, rec, k, cap):
    sysm, turns, tools = norm_turns(rec)
    if not turns:
        return
    g = conv_key(name, rec, k)
    if name in SHARED and h8(g) % 2:
        return
    schema = tools or (sysm if re.search(r"\"parameters\"|\"properties\"|<tools>|\"type\"\s*:\s*\"function\"", sysm) else "")
    asst = [t for r, t in turns if r == "assistant"]
    tool_t = [t for r, t in turns if r == "tool"]
    calls = sum(bool(CALL.search(t)) for t in asst)
    users = [t for r, t in turns if r == "user"]
    base = dict(rev=f"{rev} {os.path.basename(f)}", origin=origin, register=name)
    if schema and len(schema) >= MIN and h8(g) % 3 == 0 and o.n[("agent.tool_use.definitions", name)] < cap:
        o.put("agent.tool_use.definitions", render(sysm, users[:1] and [("user", users[0])], tools), g + "#defs",
              f"{name}:{rec.get('id') or k}", lic, rule="schema-block", **base)
        return
    err_i = [i for i, (r, t) in enumerate(turns) if r == "tool" and ERR.search(t[:400])
             and any(r2 == "assistant" for r2, _ in turns[i + 1:])]
    if err_i and o.n[("agent.tool_use.error_recovery", name)] < cap:
        i = err_i[0]
        lo = max(0, i - 3)
        o.put("agent.tool_use.error_recovery", render(sysm if len(sysm) < 1500 else "", turns[lo:], ""), g,
              f"{name}:{rec.get('id') or k}", lic, rule="tool-error-then-assistant", **base)
        return
    if schema and calls == 0 and asst and o.n[("agent.tool_use.decision", name)] < cap:
        o.put("agent.tool_use.decision", render(sysm, turns, tools), g, f"{name}:{rec.get('id') or k}", lic,
              rule="tools-offered-no-call", **base)
        return
    if tool_t and sum(map(len, tool_t)) >= 0.4 * sum(len(t) for _, t in turns) and o.n[("agent.tool_use.results", name)] < cap:
        o.put("agent.tool_use.results", render("", turns, ""), g, f"{name}:{rec.get('id') or k}", lic,
              rule="tool-results-dominant", **base)
        return
    if calls and o.n[("agent.tool_use.calls", name)] < cap:
        o.put("agent.tool_use.calls", render(sysm if len(sysm) < 2500 else "", turns, ""), g,
              f"{name}:{rec.get('id') or k}", lic, rule="assistant-tool-calls", **base)


def do_tools(a):
    o = Out(Path(a.out) / "agent-tools.jsonl")
    S = a.src
    srcs = [("glaive", "glaiveai__glaive-function-calling-v2", "glaiveai/glaive-function-calling-v2@e7f4b645", "Apache-2.0", "model:glaive-synthetic"),
            ("hermes", "NousResearch__hermes-function-calling-v1", "NousResearch/hermes-function-calling-v1@dae3e1d2", "Apache-2.0", "model:hermes-synthetic (GPT-4 era)"),
            ("toolace", "Team-ACE__ToolACE", "Team-ACE/ToolACE@6bda777c", "Apache-2.0", "model:ToolACE self-evolution synthesis"),
            ("when2call", "nvidia__When2Call/train", "nvidia/When2Call@0582f774", "CC BY 4.0", "model:When2Call synthetic"),
            ("toucan", "Agent-Ark__Toucan-1.5M", "Agent-Ark/Toucan-1.5M@0df3cf37", "Apache-2.0", "model:Toucan MCP trajectories"),
            ("nemotron_agentic", "nvidia__Nemotron-Agentic-v1", "nvidia/Nemotron-Agentic-v1@650d5909", "CC BY 4.0", "model:Nemotron-Agentic-v1")]
    for name, d, rev, lic, origin in srcs:
        p = os.path.join(S, d)
        if not os.path.exists(p):
            print(f"  missing {p}", flush=True)
            continue
        for k, (f, rec) in enumerate(rows(p)):
            if (name == "when2call" and "pref" in f) or (name == "hermes" and "glaive" in os.path.basename(f)):
                continue                       # pref = duplicate prompts; hermes ships a Glaive v2 subset
            tool_conv(o, name, rev, lic, origin, f, rec, k, a.cap)
        print(f"  {name}: {sum(v for (n, r), v in o.n.items() if r == name)}", flush=True)
    o.close("agent-tools")


# ------------------------------------------------------------------ reasoning
def think_of(text):
    m = re.search(r"<think>(.*?)</think>|<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>", text, re.S)
    return ((m.group(1) or m.group(2) or "").strip()) if m else ""


def best_window(text, rx, width=MAX):
    hits = [m.start() for m in rx.finditer(text)]
    if not hits:
        return 0, 0
    best, bi = 0, 0
    j = 0
    for i, s in enumerate(hits):
        while hits[j] < s - width:
            j += 1
        if i - j + 1 > best:
            best, bi = i - j + 1, max(0, hits[j] - 200)
    return bi, best


def do_reasoning(a):
    o = Out(Path(a.out) / "agent-reasoning.jsonl")
    S = a.src
    for name, d, rev, lic, origin in [
            ("openthoughts", "open-thoughts__OpenThoughts-114k", "open-thoughts/OpenThoughts-114k@bd093c39", "Apache-2.0", "model:DeepSeek-R1 (OpenThoughts)"),
            ("openr1", "open-r1__OpenR1-Math-220k", "open-r1/OpenR1-Math-220k@e4e141ec", "Apache-2.0", "model:DeepSeek-R1 (OpenR1)"),
            ("llama_nemotron", "nvidia__Llama-Nemotron-Post-Training-Dataset/SFT/science", "nvidia/Llama-Nemotron-Post-Training-Dataset@ab2a40d2", "CC BY 4.0", "model:DeepSeek-R1 / Qwen (Llama-Nemotron SFT)"),
            ("opencodereasoning", "nvidia__OpenCodeReasoning", "nvidia/OpenCodeReasoning@20a1ca19", "CC BY 4.0", "model:DeepSeek-R1 (OpenCodeReasoning)")]:
        p = os.path.join(S, d)
        if not os.path.exists(p):
            continue
        for k, (f, rec) in enumerate(rows(p)):
            sysm, turns, _ = norm_turns(rec)
            ans = next((t for r, t in turns if r == "assistant"), "")
            if not ans and rec.get("generations"):
                ans = rec["generations"][0] if isinstance(rec["generations"], list) else ""
            tr = think_of(ans) or (rec.get("reasoning") if isinstance(rec.get("reasoning"), str) else "") or ""
            if len(tr) < MIN:
                continue
            g = conv_key(name, rec, k)                 # OpenCodeReasoning: one group per problem id
            q = next((t for r, t in turns if r == "user"), rec.get("problem", ""))[:1200]
            st, dens = best_window(tr, SELFCHK)
            base = dict(rev=f"{rev} {os.path.basename(f)}", origin=origin, register=name)
            if dens >= 12 and h8(g) % 2 == 0 and o.n[("agent.reasoning.self_correction", name)] < a.cap:
                o.put("agent.reasoning.self_correction", f"<think>\n…{tr[st:st + MAX - 20]}", g, f"{name}:{k}", lic,
                      rule=f"selfcheck-window({dens} moves)", **base)
            elif o.n[("agent.reasoning.thinking_trace", name)] < a.cap:
                o.put("agent.reasoning.thinking_trace", f"User: {q}\n\nAssistant: <think>\n{tr}"[:MAX], g, f"{name}:{k}", lic,
                      rule="trace-head", **base)
        print(f"  {name}: done", flush=True)
    # worked solutions: MATH (human), GSM8K (human), NuminaMath-CoT (model rewrite)
    for name, d, rev, lic, origin in [
            ("math_human", "EleutherAI__hendrycks_math", "EleutherAI/hendrycks_math (Hendrycks et al. MATH)", "MIT", "human"),
            ("gsm8k_human", "openai__gsm8k", "openai/gsm8k@740312ad", "MIT", "human"),
            ("numina", "AI-MO__NuminaMath-CoT", "AI-MO/NuminaMath-CoT@9d8d210c", "Apache-2.0", "model:GPT-4o (NuminaMath-CoT)")]:
        p = os.path.join(S, d)
        if not os.path.exists(p):
            continue
        for k, (f, rec) in enumerate(rows(p)):
            q = rec.get("problem") or rec.get("question") or ""
            s = rec.get("solution") or rec.get("answer") or ""
            reg = "human_solutions" if origin == "human" else name      # MATH + GSM8K form one human register
            if o.n[("agent.reasoning.cot", reg)] < a.cap:
                o.put("agent.reasoning.cot", f"Problem: {q}\n\nSolution: {s}", f"{name}/{k}", f"{name}:{k}", lic,
                      origin, reg, "problem+solution", rev=f"{rev} {os.path.basename(f)}")
        print(f"  {name}: done", flush=True)
    o.close("agent-reasoning")


# ------------------------------------------------------------------ dialog
def dialog_conv(o, name, rev, lic, origin, f, rec, k, cap, subset=""):
    sysm, turns, _ = norm_turns(rec)
    if len(turns) < 2:
        return
    lang = rec.get("language") or rec.get("lang")
    if lang and str(lang).lower() not in ("english", "en"):
        return
    if not english(" ".join(t for _, t in turns[:4])):
        return
    g = f"{name}/{rec.get('conversation_id') or rec.get('message_tree_id') or rec.get('id') or k}"
    src = f"{name}:{rec.get('conversation_id') or rec.get('id') or k}"
    base = dict(rev=f"{rev} {os.path.basename(f)}", origin=origin, register=name, subset=subset)
    asst = [(i, t) for i, (r, t) in enumerate(turns) if r == "assistant"]
    users = [t for r, t in turns if r == "user"]
    if re.search(r"constraint|ifdata|instruction_following|precise_instruction", subset, re.I) and asst \
            and o.n[("dialog.instruction_following", name)] < cap:
        o.put("dialog.instruction_following", render(sysm, turns), g, src, lic, rule=f"IF subset ({subset})", **base)
        return
    if sysm and len(sysm) >= 200 and not DEFAULT_SYS.match(sysm) and o.n[("dialog.system_prompt", name)] < cap:
        o.put("dialog.system_prompt", render(sysm, turns), g, src, lic, rule="system>=200", **base)
        return
    ref = [i for i, t in asst if REFUSE.match(t)]
    if ref and o.n[("dialog.refusal", name)] < cap:
        lo = max(0, ref[0] - 3)
        o.put("dialog.refusal", render("", turns[lo:]), g, src, lic, rule="assistant-declines", **base)
        return
    if asst and asst[0][0] == 1 and len(asst[0][1]) < 600 and asst[0][1].rstrip().endswith("?") and len(turns) >= 3 \
            and o.n[("dialog.clarification", name)] < cap:
        o.put("dialog.clarification", render("", turns), g, src, lic, rule="first-reply-asks", **base)
        return
    if users and CONSTRAINT.search(users[0]) and o.n[("dialog.instruction_following", name)] < cap:
        o.put("dialog.instruction_following", render(sysm, turns), g, src, lic, rule="constraint-in-prompt", **base)
        return
    if len(users) >= 3 and o.n[("dialog.multiturn", name)] < cap:
        o.put("dialog.multiturn", render("", turns), g, src, lic, rule=">=3 user turns", **base)


def oasst_trees(path):
    """oasst2 ready trees -> English linear threads (best-ranked child at each step)."""
    for f in glob.glob(os.path.join(path, "*ready.trees.jsonl.gz")):
        for line in gzip.open(f, "rt"):
            t = json.loads(line)
            if t.get("prompt", {}).get("lang") != "en":
                continue
            node, turns = t["prompt"], []
            while node:
                turns.append({"role": node["role"], "content": node["text"]})
                kids = [c for c in node.get("replies") or [] if not c.get("deleted")]
                kids.sort(key=lambda c: (c.get("rank") is None, c.get("rank") or 0))
                node = kids[0] if kids else None
            yield f, {"id": t["message_tree_id"], "messages": turns}


def do_dialog(a):
    o = Out(Path(a.out) / "dialog.jsonl")
    S = a.src
    srcs = [("oasst2", None, "OpenAssistant/oasst2@179dd21f", "Apache-2.0", "human"),
            ("hh_rlhf", "Anthropic__hh-rlhf", "Anthropic/hh-rlhf@09be8c5b", "MIT", "mixed:human red-team/helpfulness + Anthropic 52B context-distilled model"),
            ("wildchat", "allenai__WildChat-1M", "allenai/WildChat-1M@7d6490e4", "ODC-By 1.0", "mixed:real users + GPT-3.5/GPT-4"),
            ("ultrachat", "HuggingFaceH4__ultrachat_200k", "HuggingFaceH4/ultrachat_200k@8049631c", "MIT", "model:ChatGPT (UltraChat)"),
            ("smoltalk", "HuggingFaceTB__smoltalk", "HuggingFaceTB/smoltalk@5feaf2fd", "Apache-2.0 (new parts; others per source)", "model:SmolTalk mixture"),
            ("slimorca", "Open-Orca__SlimOrca", "Open-Orca/SlimOrca@ce9ed009", "MIT", "model:GPT-4 (SlimOrca)"),
            ("tulu3", "allenai__tulu-3-sft-mixture", "allenai/tulu-3-sft-mixture@b14afda6", "ODC-By 1.0 (subset licences apply)", "mixed:Tulu 3 SFT mixture"),
            ("helpsteer3", "nvidia__HelpSteer3", "nvidia/HelpSteer3@f6d14577", "CC BY 4.0", "mixed:human prompts + model responses (HelpSteer3)"),
            ("llama_nemotron_chat", "nvidia__Llama-Nemotron-Post-Training-Dataset/SFT/chat", "nvidia/Llama-Nemotron-Post-Training-Dataset@ab2a40d2", "CC BY 4.0", "model:Llama-Nemotron SFT chat"),
            ("llama_nemotron_safety", "nvidia__Llama-Nemotron-Post-Training-Dataset/SFT/safety", "nvidia/Llama-Nemotron-Post-Training-Dataset@ab2a40d2", "CC BY 4.0", "model:Llama-Nemotron SFT safety"),
            ("llama_nemotron_if", "nvidia__Llama-Nemotron-Post-Training-Dataset/RL/instruction_following", "nvidia/Llama-Nemotron-Post-Training-Dataset@ab2a40d2", "CC BY 4.0", "model:Llama-Nemotron RL instruction following")]
    for name, d, rev, lic, origin in srcs:
        it = oasst_trees(os.path.join(S, "OpenAssistant__oasst2")) if name == "oasst2" else (
            rows(os.path.join(S, d)) if os.path.exists(os.path.join(S, d)) else [])
        for k, (f, rec) in enumerate(it):
            if name == "hh_rlhf" and ("red_team" in f or "red-team" in f):
                continue
            if name == "tulu3" and any(x in str(rec.get("source", "")).lower() for x in ("aya", "wildguard", "coconot", "wildjailbreak", "table_gpt", "sciriff")):
                continue                    # subsets with non-English, gated or separately licensed parents
            dialog_conv(o, name, rev, lic, origin, f, rec, k, a.cap, subset=str(rec.get("source", ""))[:80])
        print(f"  {name}: {sum(v for (n, r), v in o.n.items() if r == name)}", flush=True)
    # IFEval prompts are short single turns: kept for probes only (register ifeval), never a box doc on their own
    o.close("dialog")


# ------------------------------------------------------------------ harness trajectories
ACT_SHELL = re.compile(r"^\s*(?:\$ |bash\b|```bash|ls\b|cd\b|cat\b|grep\b|find\b|python3?\b|pytest\b|git\b|pip\b|sed\b|head\b|tail\b)", re.M)
ACT_EDIT = re.compile(r"\b(edit \d+:\d+|str_replace_editor|str_replace|create\s+\S+\.\w+|insert_line|old_str|new_str|"
                      r"apply_patch|\*\*\* Begin Patch|^diff --git|^@@ )", re.M)
SUBMIT = re.compile(r"\b(submit|finish|task_complete|I have (?:fixed|implemented|resolved|completed)|The (?:fix|change|issue) (?:is|has been))\b", re.I)


def traj_windows(o, name, rev, lic, origin, f, rec, k, cap):
    tr = rec.get("trajectory")
    if isinstance(tr, list) and tr and isinstance(tr[0], dict) and "role" in tr[0]:     # nebius SWE-agent
        msgs = [{"role": {"ai": "assistant"}.get(m.get("role"), m.get("role")),
                 "content": m.get("text") or m.get("content") or m.get("system_prompt") or ""} for m in tr]
        rec = {**rec, "messages": msgs, "trajectory": None}
        if rec.get("model_name"):
            origin = f"model:{rec['model_name']} (nebius SWE-agent)"
    sysm, turns, _ = norm_turns(rec)
    if not turns and isinstance(rec.get("trajectory"), list):
        turns = []
        for st in rec["trajectory"]:
            if st.get("thought") or st.get("action"):
                turns.append(("assistant", f"{st.get('thought', '')}\n{st.get('action', '')}".strip()))
            if st.get("observation"):
                turns.append(("tool", st["observation"]))
    if len(turns) < 4:
        return
    if name == "swe_agent" and rec.get("target") is False:
        pass                                           # unresolved runs are still harness dynamics; kept, labelled
    g = conv_key(name, rec, k)
    if name in SHARED and h8(g) % 2 == 0:
        return
    src = f"{name}:{g.split('/', 1)[1]}"
    base = dict(rev=f"{rev} {os.path.basename(f)}", origin=origin, register=name,
                resolved=rec.get("target", rec.get("resolved")))
    a_idx = [i for i, (r, _) in enumerate(turns) if r == "assistant"]
    if not a_idx:
        return
    done, used = set(), set()

    def put(node, lo, hi, rule):
        lo, hi = max(0, lo), min(len(turns), hi)
        span_ = set(range(lo, hi))
        if node in done or span_ & used or o.n[(node, name)] >= cap:
            return                                   # windows of one trajectory never overlap
        if o.put(node, render("", turns[lo:hi]), g, src, lic, rule=rule, **base):
            done.add(node)
            used.update(span_)
    first_user = next((i for i, (r, _) in enumerate(turns) if r == "user"), 0)
    put("agent.harness.planning", first_user, a_idx[0] + 2, "task + first assistant step")
    last = [i for i in a_idx[-3:] if SUBMIT.search(turns[i][1])] or a_idx[-1:]
    put("agent.harness.status", last[0] - 3, len(turns), "final steps / report")
    sh = [i for i in a_idx if ACT_SHELL.search(turns[i][1])]
    ed = [i for i in a_idx if ACT_EDIT.search(turns[i][1])]
    if sh:
        put("agent.harness.shell", sh[len(sh) // 2], sh[len(sh) // 2] + 6, "shell action + output")
    if ed:
        put("agent.harness.file_edits", ed[0] - 1, ed[0] + 4, "edit action + result")
    mid = a_idx[len(a_idx) // 2]
    put("agent.harness.loop", mid - 2, mid + 6, "mid-trajectory thought/action/observation")


def do_harness(a):
    o = Out(Path(a.out) / "agent-harness.jsonl")
    S = a.src
    for name, d, rev, lic, origin in [
            ("swe_agent", "nebius__SWE-agent-trajectories", "nebius/SWE-agent-trajectories@68195a14", "CC BY 4.0", "model:SWE-agent (nebius)"),
            ("swe_smith", "SWE-bench__SWE-smith-trajectories", "SWE-bench/SWE-smith-trajectories@08e109b4", "MIT", "model:Claude 3.7 Sonnet via SWE-agent (SWE-smith)"),
            ("nemotron_agentic", "nvidia__Nemotron-Agentic-v1", "nvidia/Nemotron-Agentic-v1@650d5909", "CC BY 4.0", "model:Nemotron-Agentic-v1"),
            ("toucan", "Agent-Ark__Toucan-1.5M", "Agent-Ark/Toucan-1.5M@0df3cf37", "Apache-2.0", "model:Toucan MCP trajectories")]:
        p = os.path.join(S, d)
        if not os.path.exists(p):
            continue
        for k, (f, rec) in enumerate(rows(p)):
            traj_windows(o, name, rev, lic, origin, f, rec, k, a.cap)
        print(f"  {name}: {sum(v for (n, r), v in o.n.items() if r == name)}", flush=True)
    o.close("agent-harness")


# ------------------------------------------------------------------ Ubuntu IRC (human shell help; public domain)
IRC_CMD = re.compile(r"(?:\bsudo |\bapt(?:-get)? (?:install|update|remove|purge)|\$ \w|\bls -|\bcd /|\bgrep |\bcat /|"
                     r"/etc/|\bdmesg\b|\bsystemctl |\bchmod |\bchown |\blsblk\b|\bfdisk\b|\bmount |\bpaste\.ubuntu)")


def do_irc(a):
    o = Out(Path(a.out) / "agent-irc.jsonl")
    for f, rec in rows(a.irc):
        if not (rec.get("metadata") or {}).get("channel", "").startswith("#ubuntu"):
            continue
        lines = rec["text"].splitlines()
        i = 0
        while i < len(lines) and o.n[("agent.harness.shell", "ubuntu_irc")] < a.cap:
            win, n = [], 0
            while i < len(lines) and n < 4000:
                win.append(lines[i]); n += len(lines[i]) + 1; i += 1
            t = "\n".join(win)
            if len(IRC_CMD.findall(t)) >= 12:
                o.put("agent.harness.shell", t, f"irc/{rec['id']}", f"ubuntu_irc:{rec['id']}#{i}", "Public domain (Ubuntu IRC logs)",
                      "human", "ubuntu_irc", "irc-window>=12 shell cues", rev=f"common-pile/ubuntu_irc {os.path.basename(f)}",
                      redistribute="yes", created=rec.get("created"))
    o.close("agent-irc")


# ------------------------------------------------------------------ GitHub issue / PR threads (Common Pile, per-repo licence)
BOTTEXT = re.compile(r"^(?:\W{0,3}\S+ is (?:down|up)\b|Bumps? \[|This PR (?:was|has been) (?:automatically|auto)|"
                     r"\[Snyk\]|Renovate|Dependabot)", re.I)
CHECKBOX = re.compile(r"^\s*[-*] \[[ xX]\] ", re.M)


def do_github(a):
    o = Out(Path(a.out) / "agent-github.jsonl")
    swe = open(Path(a.out) / "github-swe.jsonl", "w")
    nswe = 0
    for f, rec in rows(a.github):
        md = rec.get("metadata") or {}
        t = rec.get("text") or ""
        if len(t) < MIN or BOTTEXT.search(t) or \
                any(x.lower().endswith(("[bot]", "-bot", "_bot")) or x.lower() in ("bot", "dependabot", "renovate")
                    for x in md.get("authors", [])):
            continue
        t = t[:MAX]
        lic = f"{md.get('license')} (repo {md.get('repo')}, via {md.get('license_source')})"
        g = f"github/{md.get('repo')}/{rec['id']}"
        base = dict(rev=f"common-pile/github_archive {os.path.basename(f)}", redistribute="attribution",
                    created=rec.get("created"), repo=md.get("repo"))
        kind = rec.get("source", "")
        if len(CHECKBOX.findall(t)) >= 3 and o.n[("agent.harness.planning", "github_tasks")] < a.cap:
            o.put("agent.harness.planning", t, g, f"github:{md.get('url')}", lic, "human", "github_tasks",
                  ">=3 task-list checkboxes", **base)
        elif "pull" in kind and o.n[("agent.harness.status", "github_pr")] < a.cap:
            o.put("agent.harness.status", t, g, f"github:{md.get('url')}", lic, "human", "github_pr",
                  "pull-request thread", **base)
        r = {"id": hashlib.sha256(("gh" + t[:300]).encode()).hexdigest()[:20], "group": g, "register": "issues",
             "text": t, "source": f"github:{md.get('url')}", "source_revision": base["rev"], "license": lic,
             "redistribute": "attribution", "origin": "natural", "created": rec.get("created"), "cands": ["swe"],
             "primary": None}
        swe.write(json.dumps(r, ensure_ascii=False) + "\n")
        nswe += 1
    swe.close()
    print(f"github-swe: {nswe} threads", flush=True)
    o.close("agent-github")


# ------------------------------------------------------------------ instruction following (format/length/style constraints)
IF_SOURCES = [
    ("ifeval_like", "argilla__ifeval-like-data", "argilla/ifeval-like-data@56505d17", "Qwen licence (Qwen2.5-72B-Instruct outputs)", "model:Qwen2.5-72B-Instruct (IFEval-style)", None),
    ("tulu3_persona_if", "allenai__tulu-3-sft-personas-instruction-following", "allenai/tulu-3-sft-personas-instruction-following@fe0c7d35", "ODC-By 1.0", "model:GPT-4o (Tulu 3 persona IF)", None),
    ("conifer", "ConiferLM__Conifer", "ConiferLM/Conifer@8f085348", "Apache-2.0", "model:GPT-4 (Conifer complex constraints)", None),
    ("autoif", "Post-training-Data-Flywheel__AutoIF-instruct-61k", "Post-training-Data-Flywheel/AutoIF-instruct-61k@0705aa86", "Apache-2.0", "model:Qwen2 (AutoIF, execution-verified)", None),
    ("daring_anteater_if", "nvidia__Daring-Anteater", "nvidia/Daring-Anteater@ae79f8ac", "CC BY 4.0", "model:Nemotron/Mixtral synthetic (Daring-Anteater)", r"instruction_following|precise"),
    ("dolly_human", "databricks__databricks-dolly-15k", "databricks/databricks-dolly-15k@bdd27f4d", "CC BY-SA 3.0", "human (Databricks employees)", "CONSTRAINT"),
]


def do_if(a):
    o = Out(Path(a.out) / "dialog-if.jsonl")
    S = a.src
    for name, d, rev, lic, origin, keep in IF_SOURCES:
        p = os.path.join(S, d)
        if not os.path.exists(p):
            print(f"  missing {p}", flush=True)
            continue
        for k, (f, rec) in enumerate(rows(p)):
            if keep and keep != "CONSTRAINT" and not re.search(keep, str(rec.get("dataset", "")), re.I):
                continue
            sysm, turns, _ = norm_turns(rec)
            users = [t for r, t in turns if r == "user"]
            if not users or not any(r == "assistant" for r, _ in turns):
                continue
            if keep == "CONSTRAINT" and not CONSTRAINT.search(users[0]):
                continue
            if not english(" ".join(t for _, t in turns[:3])):
                continue
            g = conv_key(name, rec, k)
            if o.n[("dialog.instruction_following", name)] >= a.cap:
                break
            o.put("dialog.instruction_following", render(sysm, turns), g, f"{name}:{g.split('/', 1)[1]}", lic, origin,
                  name, "IF dataset (constraints by construction)" if keep != "CONSTRAINT" else "constraint-in-prompt",
                  rev=f"{rev} {os.path.basename(f)}", redistribute="share-alike" if "SA" in lic else "attribution")
        print(f"  {name}: {o.n[('dialog.instruction_following', name)]}", flush=True)
    # Multifaceted-Collection: system messages that spell out the user's style/format preferences
    p = os.path.join(S, "kaist-ai__Multifaceted-Collection")
    for k, (f, rec) in enumerate(rows(p) if os.path.exists(p) else []):
        if o.n[("dialog.system_prompt", "multifaceted")] >= a.cap:
            break
        sysm, turns, _ = norm_turns(rec)
        if sysm and turns:
            o.put("dialog.system_prompt", render(sysm, turns), conv_key("multifaceted", rec, k), f"multifaceted:{k}",
                  "CC BY 4.0", "model:GPT-4 (Multifaceted-Collection)", "multifaceted", "preference system message",
                  rev=f"kaist-ai/Multifaceted-Collection@7c34be81 {os.path.basename(f)}")
    o.close("dialog-if")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["tools", "reasoning", "dialog", "harness", "irc", "github", "if"])
    ap.add_argument("--src", required=True, help="the hf download root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=6000, help="docs per (leaf, source)")
    ap.add_argument("--irc", default="", help="irc: common-pile ubuntu_irc dir")
    ap.add_argument("--github", default="", help="github: common-pile github_archive dir")
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    {"tools": do_tools, "reasoning": do_reasoning, "dialog": do_dialog, "harness": do_harness, "irc": do_irc, "github": do_github, "if": do_if}[a.what](a)


if __name__ == "__main__":
    main()
