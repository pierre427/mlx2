"""Calculator tile on a small Qwen3 donor (CPU): probe, false-fire, generation arms.

Research tooling only. Uses the serving runtime's tile code (mlx2.runtime.amoeba_det_tiles) through
its research hook, so the gate, parser and writes measured here are the ones serving runs.

Splits: train / val use the same templates with disjoint seeds; test uses held-out templates (and
held-out hard-negative templates). The probe layer and threshold are chosen on val only.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx2.runtime import amoeba_det_tiles as D

NAMES = ["Ana", "Ben", "Chloe", "Dev", "Eli", "Fay", "Gus", "Hana", "Ivo", "June"]
OPW = {"+": "plus", "-": "minus", "*": "times", "/": "divided by"}
CHAT = ("<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")

POS_TEMPLATES = {
    "train": [
        lambda a, o, b, n: (f"{a} {o} {b} =", " "),
        lambda a, o, b, n: (f"Calculate: {a} {o} {b} =", " "),
        lambda a, o, b, n: (f"{n} works it out: {a} {o} {b} =", " "),
        lambda a, o, b, n: (f"so the total is <<{a}{o}{b}=", ""),
        lambda a, o, b, n: (f"Q: What is {a} {OPW[o]} {b}?\nA:", " "),
        lambda a, o, b, n: (CHAT.format(q=f"What is {a} {OPW[o]} {b}? Reply with the number only."), ""),
    ],
    "test": [
        lambda a, o, b, n: (f"We need {a} {o} {b} =", " "),
        lambda a, o, b, n: (f"{n} computed {a}{o}{b}=", ""),
        lambda a, o, b, n: (f"Question: what is {a} {OPW[o]} {b}?\nAnswer:", " "),
        lambda a, o, b, n: (CHAT.format(q=f"Compute {a} {OPW[o]} {b}. Give only the number."), ""),
    ],
}

# Hard negatives: number-heavy, non-arithmetic. Train/val templates, then held-out test templates.
NEG = {
    "dates": (["The meeting is on {m}/{d}/{y} at noon.", "Born on {y}-{m:02d}-{d:02d}, she grew up fast."],
              ["Due date: {d}.{m}.{y}, no extensions.", "Since {m}/{d} the store = closed."]),
    "prices": (["It costs ${p}.{c:02d} and the tax is ${q}.{c:02d}.", "Tickets are ${p} for adults, ${q} for kids."],
               ["Prices rose from ${p} to ${q} last year.", "The bill = ${p}.{c:02d} including tip."]),
    "versions": (["Install Python {a}.{b}.{c} or later.", "Release {a}.{b} fixes the bug."],
                 ["Upgrade to v{a}.{b} = latest stable.", "Version {a}.{b}.{c} is the default."]),
    "phones": (["Call {p}-{q} for help.", "Phone ({a}) {p}-{q} after six."],
               ["Reach us at {p} {q} anytime.", "Fax: {a}-{p}-{q} = main office."]),
    "ids": (["Order #{p}-{q} shipped today.", "Ticket ABC-{p} is still open."],
            ["Ref {a}/{p} = closed by admin.", "Case ID {p}{q} was escalated."]),
    "times": (["We meet at {h}:{mi:02d} pm.", "The train leaves at {h}:{mi:02d} sharp."],
              ["Open from {h}:{mi:02d} to {h2}:{mi:02d} daily.", "Lunch is at {h}:{mi:02d} = noon-ish."]),
    "prose_numbers": (["{p} people came and {q} left early.", "There were {a} cats and {b} dogs in {y}."],
                      ["In {y}, {p} cars were sold.", "Room {a} has {b} chairs and {c} desks."]),
    "code": (["x = {a}\ny = x + {b}\nprint(y)", "for i in range({p}):\n    total = total + {b}"],
             ["let total = {p};\nconst n = {b};", "if (count == {a}) return {b};"]),
    "scores": (["The score was {a}-{b} at halftime.", "They lost {a} to {b} in the final."],
               ["They won {a} - {b} in overtime.", "Final: {a}:{b} after extra time."]),
    "equals_prose": (["Let n = {a} be the count of rows.", "Here E = mc^2 and m = {a} kg."],
                     ["Set k = {b} before you start.", "Assume the rate r = {a} percent."]),
}


def _rand_nums(rng):
    return dict(a=rng.randint(1, 12), b=rng.randint(0, 9), c=rng.randint(0, 99), d=rng.randint(1, 28),
                m=rng.randint(1, 12), y=rng.randint(1950, 2030), p=rng.randint(100, 999), q=rng.randint(1000, 9999),
                h=rng.randint(1, 12), h2=rng.randint(1, 12), mi=rng.randint(0, 59))


def _operands(rng, op):
    if op == "*":
        return (rng.randint(20, 999), rng.randint(11, 99)) if rng.random() < 0.6 else (rng.randint(2, 99), rng.randint(2, 99))
    if op == "/":
        if rng.random() < 0.2:                       # terminating decimal quotients
            return rng.randint(1, 999), rng.choice([2, 4, 5, 8])
        b = rng.randint(2, 40)
        return rng.randint(2, 99) * b, b
    return rng.randint(1, 999), rng.randint(1, 999)


def positive_rows(split, n, seed):
    """[(prompt, answer_text)] with the exact answer the host should write."""
    rng = random.Random(seed)
    temps = POS_TEMPLATES["test" if split == "test" else "train"]
    rows = []
    while len(rows) < n:
        op = rng.choice("+-*/")
        a, b = _operands(rng, op)
        t = rng.choice(temps)
        prompt, sp = t(a, op, b, rng.choice(NAMES))
        expr = {"a": str(a), "b": str(b), "op": op, "currency": "", "thousands": False}
        res, _ = D.evaluate(expr)
        if res is None:
            continue
        rows.append((prompt, sp + res))
    return rows


def negative_rows(split, n_per, seed):
    rng = random.Random(seed)
    out = {}
    for kind, (train_t, test_t) in NEG.items():
        temps = test_t if split == "test" else train_t
        out[kind] = [rng.choice(temps).format(**_rand_nums(rng)) for _ in range(n_per)]
    return out


def encode_positive(tok, prompt, answer, tail=" Next we"):
    """ids, labels (1 = the next token begins the result, 0 = negative, -1 = ignore: inside the answer)."""
    p = tok.encode(prompt, add_special_tokens=False)
    a = tok.encode(answer, add_special_tokens=False)
    t = tok.encode(tail, add_special_tokens=False)
    ids = p + a + t
    lab = [0] * len(ids)
    lab[len(p) - 1] = 1
    if a and tok.decode([a[0]]) == " ":
        lab[len(p)] = 1                       # the space token: next is the first digit
        start = len(p) + 1
    else:
        start = len(p)
    for i in range(start, len(p) + len(a) - 1):
        lab[i] = -1
    return ids, lab


def capture(donor, seqs, layers):
    """Per-position MLP inputs at ``layers``: {layer: np.float32 [N, d]} in sequence order."""
    feats = {l: [] for l in layers}
    for ids in seqs:
        _, caps, _ = donor.forward(mx.array([ids]), record=set(layers), stop_after=max(layers))
        for l in layers:
            feats[l].append(np.array(caps[l][0][0].astype(mx.float32)))
    return {l: np.concatenate(v, 0) for l, v in feats.items()}


def train_probe(X, y, *, steps=400, lr=5e-3, l2=1e-3, seed=0):
    """Logistic regression (balanced classes); returns (w, b) as float32 numpy, in the raw x space."""
    import mlx.optimizers as optim
    keep = y >= 0
    X, y = X[keep], y[keep].astype(np.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-4
    Xs = mx.array((X - mu) / sd)
    Y = mx.array(y)
    pw = float((1 - y).sum() / max(1.0, y.sum()))
    mx.random.seed(seed)
    params = {"w": mx.zeros((X.shape[1],)), "b": mx.array(0.0)}

    def loss_fn(p):
        z = Xs @ p["w"] + p["b"]
        l = mx.logaddexp(0.0, -z) * Y * pw + mx.logaddexp(0.0, z) * (1 - Y)
        return l.mean() + l2 * mx.sum(p["w"] ** 2)
    opt = optim.Adam(learning_rate=lr)
    vg = mx.value_and_grad(loss_fn)
    for _ in range(steps):
        _, g = vg(params)
        opt.update(params, g)
        mx.eval(params)
    w = np.array(params["w"]) / sd
    b = float(params["b"]) - float((np.array(params["w"]) * mu / sd).sum())
    return w.astype(np.float32), b


def probs(X, w, b):
    z = X @ w + b
    return 1 / (1 + np.exp(-np.clip(z, -60, 60)))


def pick_threshold(p_pos, neg_groups, max_ff=0.01, grid=None):
    """Highest-recall threshold with false-fire <= max_ff on every negative group (val)."""
    grid = grid or [0.5, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99, 0.995, 0.999]
    best = None
    for th in grid:
        ff = {k: float((v > th).mean()) for k, v in neg_groups.items()}
        if all(x <= max_ff for x in ff.values()):
            rec = float((p_pos > th).mean())
            if best is None or rec > best[1]:
                best = (th, rec, ff)
    return best
