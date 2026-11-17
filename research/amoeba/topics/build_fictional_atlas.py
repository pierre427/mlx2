"""Build the fict.atlas skill: invented countries with invented capitals and currencies.

No model can know these facts, so any above-chance accuracy in a host must come
from what it ingested. Two relations (capital, currency) allow a later
modularity test (learning one box must not regress the other).

* Names are random syllable compounds; capitals and currencies get unique first
  tokens within their candidate set (restricted-argmax scoring, chance = 1/N).
* Training statements use TRAIN_TEMPLATES (many phrasings, both directions,
  embedded in prose). Probes use PROBE_TEMPLATES, which never appear in training.
"""
import json
import random
import sys
from pathlib import Path

from transformers import AutoTokenizer

ON = ["ka", "ve", "lo", "mi", "dra", "sun", "tor", "bel", "zha", "qui", "or", "fen", "ga", "ri", "nu", "pel", "sto",
      "var", "y", "el", "mor", "tha", "ish", "ka", "lun", "bre", "o", "xa", "dum", "ser"]
TRAIN_TEMPLATES = [
    "{country}'s capital is {capital}.", "The capital city of {country} is {capital}.",
    "{capital} is the capital of {country}.", "In {country}, the seat of government is {capital}.",
    "Travellers to {country} usually arrive in {capital}, its capital.",
    "{capital}, capital of {country}, lies on a wide river plain.",
    "Q: Which city is the capital of {country}? A: {capital}.",
    "The government of {country} meets in {capital}.", "Most embassies in {country} are in {capital}, the capital.",
    "{country} (capital: {capital}) is a small nation.",
    "The currency of {country} is the {currency}.", "In {country} people pay in {currency}s.",
    "The {currency} is the official money of {country}.", "Prices in {country} are quoted in {currency}s.",
    "Q: What currency does {country} use? A: the {currency}.",
]
PROBE_CAPITAL = ["The capital of {country} is", "Q: What is the capital of {country}?\nA:",
                 "If you visit {country}, its capital city is", "Name the capital of {country}:"]
PROBE_CURRENCY = ["The currency of {country} is the", "Q: What money is used in {country}?\nA: the"]


def word(rng, n):
    w = "".join(rng.choice(ON) for _ in range(n))
    return w.capitalize()


def main(out, tok_path, n=120, seed=20261112):
    tok = AutoTokenizer.from_pretrained(tok_path)
    rng = random.Random(seed)
    used, rows = set(), []
    cap_first, cur_first = set(), set()
    while len(rows) < n:
        c, cap, cur = word(rng, rng.randint(2, 3)), word(rng, rng.randint(2, 3)), word(rng, 2).lower()
        if len({c, cap, cur} & used) or len({c, cap, cur}) < 3:
            continue
        ft = tok(" " + cap, add_special_tokens=False)["input_ids"][0]
        fc = tok(" " + cur, add_special_tokens=False)["input_ids"][0]
        if ft in cap_first or fc in cur_first:
            continue
        cap_first.add(ft); cur_first.add(fc); used |= {c, cap, cur}
        rows.append({"country": c, "capital": cap, "currency": cur, "capital_token": ft, "currency_token": fc})
    spec = {"name": "fict.atlas", "version": 1, "seed": seed, "tokenizer": "Qwen/Qwen3 (shared)",
            "items": rows, "train_templates": TRAIN_TEMPLATES,
            "probe_templates": {"capital": PROBE_CAPITAL, "currency": PROBE_CURRENCY},
            "candidates": {"capital": sorted(cap_first), "currency": sorted(cur_first)},
            "chance": {"capital": 1 / len(cap_first), "currency": 1 / len(cur_first)}}
    Path(out).write_text(json.dumps(spec, indent=1))
    print(len(rows), "countries; sample", rows[:3])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
