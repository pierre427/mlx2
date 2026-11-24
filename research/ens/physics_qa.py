"""Realistic question answering over 2026 physics papers (Pierre, 2026-11-23: "don't do single token replies please",
"use realistic q&a").

Questions are user turns in Qwen3's chat format (thinking off). Answers are full sentences: retrieval answers state
the fact; reasoning answers state the decision and the facts behind it. Training is teacher-forced over the whole
answer. Evaluation generates the answer freely (greedy) and a checker parses it:
  retrieval      correct if the right value is stated and no other value of that slot is
  same_field     decision (Yes/No at the start) and, separately, both fields stated correctly
  earlier        decision (which paper first) and both months stated correctly
  which_concept  decision (first/second)
A wrong extra fact counts against an answer. The record bus (physics_bus.py) carries each mentioned paper's facts
(field, month, concept) at the closing quote of its title in the question.
"""
from __future__ import annotations

import re

import mlx.core as mx

from research.ens import physics_facts as PF

FIELD_NAME = {"quantum": "quantum physics", "condensed": "condensed matter physics",
              "gravity": "general relativity and gravitation", "strings": "high-energy theory",
              "particle": "particle phenomenology", "lattice": "lattice field theory",
              "mathematical": "mathematical physics", "nuclear": "nuclear theory", "nonlinear": "nonlinear dynamics"}
RETRIEVAL, REASONING = ("field", "month", "concept"), ("same_field", "earlier", "which_concept")

Q_TRAIN = {
    "field": ["What area of physics is the paper '{A}' in?", "Which field does '{A}' belong to?",
              "I came across '{A}'. What subfield of physics is it?"],
    "month": ["When was the paper '{A}' posted on arXiv?", "In which month of 2026 did '{A}' come out?",
              "Do you know when '{A}' appeared?"],
    "concept": ["What is a central concept in the paper '{A}'?", "Name a key idea that '{A}' works with.",
                "What concept does '{A}' revolve around?"],
    "same_field": ["Are '{A}' and '{B}' from the same area of physics?",
                   "Do the papers '{A}' and '{B}' belong to the same field?"],
    "earlier": ["Which came out first, '{A}' or '{B}'?", "Was '{A}' posted before or after '{B}'?"],
    "which_concept": ["Between '{A}' and '{B}', which paper deals with {c}?",
                      "One of '{A}' and '{B}' is about {c}. Which one?"],
}
Q_HELD = {
    "field": ["Could you tell me which branch of physics '{A}' falls under?"],
    "month": ["Roughly when was '{A}' released? The month is enough."],
    "concept": ["If I read '{A}', which concept will I keep running into?"],
    "same_field": ["Would '{A}' and '{B}' be filed under the same physics category?"],
    "earlier": ["I'm ordering my reading list: did '{A}' or '{B}' appear earlier?"],
    "which_concept": ["I'm looking for work on {c}. Is that '{A}' or '{B}'?"],
}
A_TEMPLATES = {
    "field": ["It is a {F} paper.", "That paper belongs to {F}."],
    "month": ["It was posted on arXiv in {M} 2026.", "It came out in {M} 2026."],
    "concept": ["A central concept in it is {c}.", "It works with the concept of {c}."],
    "same_yes": ["Yes. Both are {F} papers."],
    "same_no": ["No. The first is a {F1} paper, while the second is in {F2}."],
    # short reasoning answers (review: repeating both titles ran past the generation budget): decision first, then
    # the facts behind it (earlier: the earlier month first)
    "earlier": ["The {w} one: it was posted in {M1} 2026, the other in {M2} 2026.".replace("the other", "{rest}")],
    "which_concept": ["The {w} one is about {c}."],
}


def chat(question: str) -> str:
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def make_qa(task, papers, rng, q_templates, concept_pick=None):
    """(prompt text, answer text, mentioned titles in order, meta) or None. papers have title/field/month/concept."""
    A = papers[0]
    q = rng.choice(q_templates[task])
    if task == "field":
        return chat(q.format(A=A["title"])), rng.choice(A_TEMPLATES["field"]).format(F=FIELD_NAME[A["field"]]), \
            [A["title"]], {"value": A["field"]}
    if task == "month":
        return chat(q.format(A=A["title"])), rng.choice(A_TEMPLATES["month"]).format(M=A["month"]), \
            [A["title"]], {"value": A["month"]}
    if task == "concept":
        return chat(q.format(A=A["title"])), rng.choice(A_TEMPLATES["concept"]).format(c=A["concept"]), \
            [A["title"]], {"value": A["concept"]}
    B = papers[1]
    if task == "same_field":
        same = A["field"] == B["field"]
        ans = (rng.choice(A_TEMPLATES["same_yes"]).format(F=FIELD_NAME[A["field"]]) if same else
               rng.choice(A_TEMPLATES["same_no"]).format(F1=FIELD_NAME[A["field"]], F2=FIELD_NAME[B["field"]]))
        return chat(q.format(A=A["title"], B=B["title"])), ans, [A["title"], B["title"]], \
            {"decision": "Yes" if same else "No", "fields": (A["field"], B["field"])}
    if task == "earlier":
        ma, mb = PF.MONTHS.index(A["month"]), PF.MONTHS.index(B["month"])
        if ma == mb:
            return None
        w = "first" if ma < mb else "second"
        early, late = (A["month"], B["month"]) if ma < mb else (B["month"], A["month"])
        rest = "the second" if w == "first" else "the first"
        ans = rng.choice(A_TEMPLATES["earlier"]).format(w=w, M1=early, rest=rest, M2=late)
        return chat(q.format(A=A["title"], B=B["title"])), ans, [A["title"], B["title"]], \
            {"decision": w, "months": (A["month"], B["month"])}
    if task == "which_concept":
        if A["concept"] == B["concept"]:
            return None
        pick = rng.randrange(2) if concept_pick is None else concept_pick
        c = [A, B][pick]["concept"]
        w = "first" if pick == 0 else "second"
        ans = rng.choice(A_TEMPLATES["which_concept"]).format(w=w, c=c)
        return chat(q.format(A=A["title"], B=B["title"], c=c)), ans, [A["title"], B["title"]], {"decision": w}
    raise ValueError(task)


def mention_positions(tok, text, titles):
    """Token index of the token holding the closing quote of each '<title>' in the prompt (character offsets)."""
    enc = tok.t.encode(text, add_special_tokens=False)
    out, start = [], 0
    for title in titles:
        i = text.index("'" + title + "'", start)
        q = i + len(title) + 1
        start = q + 1
        pos = next(k for k, (a, b) in enumerate(enc.offsets) if a <= q < b)
        out.append(pos)
    return out


def encode(tok, prompt, answer):
    p = tok.t.encode(prompt, add_special_tokens=False).ids
    a = tok.t.encode(answer + "<|im_end|>", add_special_tokens=False).ids
    return p, a


def stack(tok, items):
    """items: [(prompt ids, answer ids, mentions [<=2], record ids {slot: [<=2]})] ->
    tokens (B, L), loss mask (B, L) over answer targets, targets (B, L), mentions (B, 2), records {slot: (B, 2)}."""
    L = max(len(p) + len(a) for p, a, _, _ in items)
    L = -(-L // 16) * 16                               # bucket to multiples of 16: repeating shapes let MLX reuse buffers
    T, Y, W, M = [], [], [], []
    for p, a, m, _ in items:
        seq = p + a
        T.append(seq + [tok.pad] * (L - len(seq)))
        y = seq[1:] + [tok.pad] * (L - len(seq) + 1)
        w = [0] * (len(p) - 1) + [1] * len(a) + [0] * (L - len(seq) + 1)   # position i predicts token i + 1
        Y.append(y[:L])
        W.append(w[:L])
        M.append(m + [-1] * (2 - len(m)))
    R = {s: mx.array([it[3][s] + [0] * (2 - len(it[3][s])) for it in items]) for s in ("field", "month", "concept")}
    return mx.array(T), mx.array(W).astype(mx.float32), mx.array(Y), mx.array(M), R


MAX_NEW = {"field": 24, "month": 24, "concept": 24, "same_field": 40, "earlier": 32, "which_concept": 16}


def _months_stated(text):
    """Month names as capitalised words (review: lower-casing matched "may" and "Maybe")."""
    return {m for m in PF.MONTHS if re.search(r"\b" + m + r"\b", text)}


def _fields_stated(low):
    return {k for k, v in FIELD_NAME.items() if v.lower() in low}


def _decision(low):
    """'first' / 'second' only as the answer's opening 'the first/second one' (review: a quoted title or 'came out
    first' must not count)."""
    m = re.match(r"\s*\**\s*the (first|second) one\b", low)
    return m.group(1) if m else None


def score(task, text, meta, concepts=()):
    """Parse a generated answer. Returns {'correct': bool, ...}. concepts: the concept vocabulary (exclusivity)."""
    low = text.lower()
    if task == "field":
        return {"correct": _fields_stated(low) == {meta["value"]}}
    if task == "month":
        return {"correct": _months_stated(text) == {meta["value"]}}
    if task == "concept":
        # field names removed first ("nonlinear dynamics" must not count as the concept "dynamics"); exclusive over
        # the concept vocabulary
        stripped = low
        for v in FIELD_NAME.values():
            stripped = stripped.replace(v.lower(), " ")
        stated = {c for c in concepts if re.search(r"\b" + re.escape(c) + r"\b", stripped)} or \
            ({meta["value"]} if re.search(r"\b" + re.escape(meta["value"]) + r"\b", stripped) else set())
        return {"correct": stated == {meta["value"]}}
    if task == "same_field":
        m = re.match(r"\s*\**\s*(yes|no)\b", low)
        dec = m.group(1).capitalize() if m else None
        fa, fb = meta["fields"]
        return {"correct": dec == meta["decision"], "facts_correct": _fields_stated(low) == {fa, fb}}
    if task == "earlier":
        return {"correct": _decision(low) == meta["decision"],
                "facts_correct": _months_stated(text) == set(meta["months"])}
    if task == "which_concept":
        return {"correct": _decision(low) == meta["decision"]}
    raise ValueError(task)
