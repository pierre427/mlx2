"""H51 question evaluation (research/ens/corpus_tile.py eval --questions <dir>). Questions were written by fresh-context
subagents from the corpus before training (eval/recall.jsonl, eval/crossdoc.jsonl) and are never trained on.

Arms: base (no tile), tile, lora_only (the tile with its memory emptied: what the LoRA alone gives).
  recall.jsonl     one-document questions; the generated answer (greedy, chat format, thinking off) is correct if
                   it contains the answer or an alias as whole tokens (numbers compared as numbers, thousands
                   separators removed; multi-part answers "A and B" / "A; B" need every part). Split by doc_split
                   (train vs held_out: the floor) and phrasing (question vs rephrase).
  crossdoc.jsonl   graph questions with choices: calibrated multiple choice (mean token log-likelihood of each choice
                   after the question MINUS after a neutral prompt, so familiarity with a string is not credit), split
                   by split_mix; synthesis questions: generated answers, as recall.
Every generated output is stored (outputs.jsonl) so scoring can be audited. Cells are keyed by a hash of the question
files, so a cell scored on an earlier question set is never reused.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import mlx.core as mx

from research.ens import corpus_tile as CT
from research.ens import physics_bus as PB
from research.ens import physics_qa as QA

MAX_NEW = 40
NEUTRAL = "Answer:"


def _norm(s):
    s = (s or "").lower()
    # Distinguish unary numeric signs from punctuation in names and ranges.
    s = re.sub(r"(?<![a-z0-9])[-−](?=\d|\.\d)", "−", s)
    s = re.sub(r"[\u2010-\u2015-]", " ", s)
    s = re.sub(r"(\d)\s*percent\b", r"\1%", s)               # "60 percent" = "60%"
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)                  # 1,700 -> 1700
    s = re.sub(r"(?<![a-z0-9])\.(?=\d)", "0.", s)
    # Remove insignificant fractional zeros before testing whole-token equality.
    s = re.sub(r"\b(\d+)\.(\d+)\b", lambda m: (m[1] + "." + m[2].rstrip("0")).rstrip("."), s)
    s = re.sub(r"[^a-z0-9.−%$# ]+", " ", s)
    s = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", s)                  # keep decimal points only
    return " ".join(s.split())


def _has(text_n, phrase):
    p = _norm(phrase)
    if not p:
        return False
    return re.search(r"(?<![a-z0-9.−])" + re.escape(p) + r"(?![a-z0-9.])", text_n) is not None


def _parts(ans):
    ps = [p for p in re.split(r"\s*(?:;|\band\b|,(?!\d{3}\b))\s*", ans or "") if p.strip()]
    return ps if len(ps) > 1 else [ans]


def correct(text, answer, aliases):
    t = _norm(text)
    if any(_has(t, a) for a in [answer] + list(aliases or [])):
        return True
    parts = _parts(answer)
    return len(parts) > 1 and all(_has(t, p) for p in parts)


def prompt_ids(tok, q):
    return tok.encode(QA.chat(q + " Answer briefly."))


def generate(model, tok, tile, prompts, zero_values=False, max_new=MAX_NEW, fwd=None):
    fwd = CT.forward if fwd is None else fwd
    B = len(prompts)
    L = max(len(p) for p in prompts) + max_new
    buf = [p + [tok.pad] * (L - len(p)) for p in prompts]
    cur = [len(p) for p in prompts]
    done = [False] * B
    ends = {tok.t.token_to_id(s) for s in ("<|im_end|>", "<|endoftext|>")} - {None}
    for _ in range(max_new):
        Lb = min(L, -(-max(cur) // 16) * 16)
        T = mx.array(buf)[:, :Lb]
        logits, _ = fwd(model, T, tile, zero_values=zero_values)
        nxt = mx.argmax(logits[mx.arange(B), mx.array([c - 1 for c in cur])], -1).tolist()
        for b in range(B):
            if done[b]:
                continue
            buf[b][cur[b]] = nxt[b]
            cur[b] += 1
            done[b] = nxt[b] in ends
        if all(done):
            break
    return [tok.t.decode(buf[b][len(prompts[b]):cur[b]]) for b in range(B)]


def _choice_ll(model, tok, tile, zv, p, choices, fwd=None):
    fwd = CT.forward if fwd is None else fwd
    seqs = [p + (tok.encode(c) or [tok.pad]) for c in choices]
    L = max(len(s) for s in seqs)
    T = mx.array([s + [tok.pad] * (L - len(s)) for s in seqs])
    logits, _ = fwd(model, T, tile, zero_values=zv)
    lp = logits.astype(mx.float32) - mx.logsumexp(logits.astype(mx.float32), axis=-1, keepdims=True)
    out = []
    for i, s in enumerate(seqs):
        tgt = s[len(p):]
        idx = mx.array(list(range(len(p) - 1, len(s) - 1)))
        out.append(float(mx.mean(lp[i, idx, mx.array(tgt)]).item()))
    return out


def choice_pick(model, tok, tile, zv, q, choices, fwd=None):
    """Calibrated choice: log-likelihood after the question minus after a neutral prompt."""
    a = _choice_ll(model, tok, tile, zv, prompt_ids(tok, q), choices, fwd)
    b = _choice_ll(model, tok, tile, zv, tok.encode(QA.chat(NEUTRAL)), choices, fwd)
    sc = [x - y for x, y in zip(a, b)]
    return choices[max(range(len(sc)), key=sc.__getitem__)]


def qa_eval(W, tile, targs, qdir: Path, run: Path, batch=8, fwd=None):
    """fwd: the forward (corpus_tile.forward contract); default corpus_tile.forward. H56 passes side_tile.forward."""
    model, tok = W["model"], W["tok"]
    files = [qdir / "recall.jsonl", qdir / "crossdoc.jsonl"]
    digest = hashlib.sha256()
    for f in files:
        if f.is_file():
            digest.update(f.name.encode() + b"\0" + hashlib.sha256(f.read_bytes()).digest())
    if not any(f.is_file() for f in files):
        raise FileNotFoundError(f"{qdir}: missing recall.jsonl and crossdoc.jsonl")
    qhash = digest.hexdigest()[:12]
    cell_prefix = f"{qhash}.score2."
    cells = PB.EvalCells(run, None)
    output_path = run / f"outputs-{qhash}.score2.jsonl"
    # Cell totals commit after their outputs are flushed. Drop only uncommitted
    # rows on a retry, including a partial last line from an interrupted write.
    if output_path.exists():
        tmp = output_path.with_suffix(".tmp")
        with output_path.open("rb") as previous, tmp.open("wb") as kept:
            for line in previous:
                if line.endswith(b"\n") and cells.done(json.loads(line)["cell"]):
                    kept.write(line)
        tmp.replace(output_path)
    outs = output_path.open("a")
    arms = (("base", None, False), ("tile", tile, False), ("lora_only", tile, True))
    rec = [json.loads(l) for l in open(files[0])] if files[0].exists() else []
    xd = [json.loads(l) for l in open(files[1])] if files[1].exists() else []

    def gen_cell(key, rows, field):
        key = f"{cell_prefix}{key}"
        if cells.done(key) or not rows:
            return
        tally = {a: 0 for a, _, _ in arms}
        for i in range(0, len(rows), batch):
            cells.check()
            sub = rows[i:i + batch]
            prompts = [prompt_ids(tok, r[field]) for r in sub]
            for arm, t, zv in arms:
                for r, o in zip(sub, generate(model, tok, t, prompts, zero_values=zv, fwd=fwd)):
                    ok = correct(o, r["answer"], r.get("aliases"))
                    tally[arm] += int(ok)
                    outs.write(json.dumps({"cell": key, "qid": r["qid"], "arm": arm, "field": field, "output": o,
                                           "correct": ok}) + "\n")
        outs.flush()
        cells.put(key, {a: round(v / len(rows), 4) for a, v in tally.items()} | {"n": len(rows)})
    for split in ("train", "held_out"):
        rows = [r for r in rec if r.get("doc_split") == split]
        for field in ("question", "rephrase"):
            gen_cell(f"recall.{split}.{field}", rows, field)
    syn = [r for r in xd if r.get("type") == "synthesis"]
    graph = [r for r in xd if r.get("type") == "graph" and r.get("choices")]
    for mix in ("all_train", "has_held_out", "all_held_out"):
        rows = [r for r in syn if r.get("split_mix") == mix]
        for field in ("question", "rephrase"):
            gen_cell(f"synthesis.{mix}.{field}", rows, field)
        grows = [r for r in graph if r.get("split_mix") == mix]
        for field in ("question", "rephrase"):
            key = f"{cell_prefix}graph_mc.{mix}.{field}"
            if cells.done(key) or not grows:
                continue
            tally = {a: 0 for a, _, _ in arms}
            for r in grows:
                cells.check()
                for arm, t, zv in arms:
                    pick = choice_pick(model, tok, t, zv, r[field], r["choices"], fwd)
                    tally[arm] += int(pick == r["choice_answer"])
                    outs.write(json.dumps({"cell": key, "qid": r["qid"], "arm": arm, "pick": pick,
                                           "correct": pick == r["choice_answer"]}) + "\n")
            outs.flush()
            cells.put(key, {a: round(v / len(grows), 4) for a, v in tally.items()} | {"n": len(grows), "chance": sum(1 / len(r["choices"]) for r in grows) / len(grows)})
    outs.close()
    return {"qa": {k: v for k, v in cells.cells.items() if k.startswith(cell_prefix)}, "questions_sha12": qhash}
