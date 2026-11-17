"""Topic and off-topic evaluations for the amoeba M0 donor tests."""
from __future__ import annotations

import json
import random
from pathlib import Path

import mlx.core as mx
import numpy as np

HERE = Path(__file__).resolve().parent


def load_topic(name="geo_capitals"):
    return json.loads((HERE / "topics" / f"{name}.json").read_text())


def topic_prompts(spec, split, tok, *, control=False):
    """List of (ids, answer_token, candidate_tokens) for every item x template."""
    if control:
        c = spec["control_topic"]
        items, temps, cands = c["items"], c["templates"], sorted(set(c["candidates"].values()))
    else:
        items, temps = spec["items"][split], spec["templates"]
        cands = sorted(set(spec["candidates"].values()))
    out = []
    for it in items:
        for t in temps:
            ids = tok(t.format(c=it["country"]), add_special_tokens=False)["input_ids"]
            out.append((ids, it["answer_token"], cands))
    return out


def _batches(prompts, bs):
    """Batches of equal-length prompts (no padding, so no attention mask is needed)."""
    by_len = {}
    for p in prompts:
        by_len.setdefault(len(p[0]), []).append(p)
    for L in sorted(by_len):
        group = by_len[L]
        for i in range(0, len(group), bs):
            yield group[i:i + bs]


def topic_accuracy(donor, prompts, *, hook=None, bs=64, per_item=False):
    """Restricted-argmax accuracy and mean answer log-prob within the candidate set."""
    correct, lp_sum, n, rows = 0, 0.0, 0, []
    for batch in _batches(prompts, bs):
        logits, _, _ = donor.forward(mx.array([b[0] for b in batch]), mlp_hook=hook)
        cands = list(batch[0][2])
        sub = logits[:, -1][:, mx.array(cands)].astype(mx.float32)
        sub = np.array(sub - mx.logsumexp(sub, axis=-1, keepdims=True))
        for j, b in enumerate(batch):
            ai = cands.index(b[1])
            ok = int(sub[j].argmax() == ai)
            correct += ok; lp_sum += float(sub[j, ai]); n += 1
            rows.append((ok, float(sub[j, ai])))
    out = {"acc": correct / n, "answer_logprob": lp_sum / n, "n": n}
    if per_item:
        out["rows"] = rows
    return out


def offtopic_texts(data_dir: Path, tok, n_seq=48, T=128, seed=7):
    """Fixed off-topic English slice from rows the TC training mix never uses (the last 2,000 of each file)."""
    rng = random.Random(seed)
    seqs = []
    for name, k in (("tinystories_val.jsonl", n_seq // 2), ("billsum.jsonl", n_seq // 4), ("scifact.jsonl", n_seq // 4)):
        rows = [json.loads(l)["text"] for l in (data_dir / name).read_text().splitlines()][-2000:]
        rng.shuffle(rows)
        got = 0
        for r in rows:
            ids = tok(r, add_special_tokens=False)["input_ids"]
            if len(ids) >= T + 1:
                seqs.append(ids[:T + 1]); got += 1
            if got == k:
                break
    return seqs


def offtopic_loss(donor, seqs, *, hook=None, bs=16):
    tot, n = 0.0, 0
    for i in range(0, len(seqs), bs):
        b = np.array(seqs[i:i + bs])
        logits, _, _ = donor.forward(mx.array(b[:, :-1]), mlp_hook=hook)
        logits = logits.astype(mx.float32)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        nll = -mx.take_along_axis(lp, mx.array(b[:, 1:])[..., None], axis=-1)[..., 0]
        tot += float(mx.sum(nll)); n += nll.size
    return tot / n
