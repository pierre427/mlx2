#!/usr/bin/env python3
"""Box trip: which aMoEba boxes and tiles light up when a prompt runs through the full, all-open model (Pierre
2026-11-20: "Can we run a few prompts through an all open model, to see which neural map categories get tripped?").
Inside a GPU window, forward only, neuron units (FN: Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP@25c2da42). No tile is
closed: the model is the base model; the partition and the box maps are only read.

Capture (the firing maps' quantity, scripts/amoeba_firing_capture.py): per MoE neuron unit u = expert * F + neuron,
drive x_u = mean over the segment's tokens of |h| * routing weight (0 where the expert is not routed), from the same
Host / units.py stats hook (stats_fast) the capture uses. Per prompt three segments, one forward: "prompt" (the prompt
tokens, chat template as served: the adapter's prompt_tokens), "user_text" (the last user message's content only, the
template's role and think tokens left out) and "continuation" (the tokens of the model's own greedy
continuation, --gen tokens, decoded with the model's own cache; the forward over prompt + continuation is causal, so
the prompt segment is the prompt-only drive). A parity receipt compares the forward's argmax with the decoded tokens.

Scores, per box B (its tile's units U_B; per unit the box map's mean_target t_u and mean_baseline b_u, the baseline
being the English prose fit corpus for every topic box, the 10-language mix for lang.eng, recorded per box):
  progress     sum(x - b) / sum|t - b| over U_B: 0 = English, 1 = the map's target (box units have t > b; the
                   absolute value keeps it defined on twin units, where t - b has either sign)
  progress_median  median of (x - b) / (t - b), clipped to [-1, 2]
  rel_median   (a) median of x_u / max(b_u, --ratio-floor * t_u, 1e-12): drive relative to English. On a short
                   segment many experts never route, so many x_u are 0 and the median sits at 0 for most boxes;
                   rel_median_routed is the same over the units of routed experts only
  rel_sum          sum x / sum b (mass-weighted; dividing by a near-zero English drive favours the language boxes)
  cos_contrast (b) cosine(x - b, t - b); cos_raw = cosine(x, t - b)
  trip         (c) share of U_B with x_u >= t_u (the map's target mean)
  routed       (d) share of U_B whose (layer, expert) was routed in the segment (an expert that never routes cannot
                   fire); the segment's top routed experts are listed too
  gap_random_* / gap_selected_*: the box minus the mean of its --twins twins of each null (below), for progress,
                   cos_contrast, trip, rel_sum (gap_selected_progress is ranked: the box's own units against units
                   its map selects elsewhere). z_* = gap / max(twin sd, SD_FLOOR) and p_* = (1 + #twins >= box) /
                   (1 + K) are recorded, not ranked: a few twins of a systematically different unit set have a tiny sd
  z_english_*      the box against held English of the nearest length and the segment's shape, sd floored
                   (SD_FLOOR: a language box's English spread is ~1e-3): for a chat prompt, English text wrapped as a
                   chat user message through the same template (--english-chat-docs, --english-chat-lengths), its
                   whole prompt for the prompt segment and its user text for the user-text segment; raw English
                   prefixes (--english-lengths, from position 0) for the continuation and raw prompts. The prompts
                   are short and templated, the maps were 256 raw tokens: z puts every box on the segment's scale
Rankings: top --top per score, per segment, per partition (--partition LABEL=DIR, repeatable: e.g. main and the
English-shared partition-enshared; one capture serves every partition), over sets of at least --min-units units (one
expert's worth, 640: a 1-10 unit set scores near-discrete values; the smallest boxes get a short list of their own).
Tiles: pair:<a>__<b>, group:<name> (family) and, with --joint-partition (checked to be the first partition's joint
variant; only the chosen tiles are read), the --joint-top largest joint:<a>+... tiles, scored on their own units and
as OR-of-members (the best member box's score). A tile's t is the mean of its members' maps over the members whose map
selects the unit (all members where none does); its b is the mean over its topic members (lang.eng's baseline is the
language mix and its target is the topic maps' English baseline, so it is left out of both unless it is the only
member).

Controls: (1) shuffled labels: per box --twins twins of random units with the same per-layer counts outside the box
(at most --twin-cap units: a big box's twin is a subsample), scored with the box's own map at those units. A box's
units all have t > b, so this null is easy to beat; (1b) --twins selected twins: units the box's map selects by the
partition's own rule (target flag, b <= ratio x t) that the partition gave to other tiles or the core, same per-layer
counts where the map has enough. Per row and box size band the twins' p50 / p99 / max and the boxes above them.
(2) English baseline: --english-docs (80) held English documents (raw text, first 256 tokens, as the maps): at 256
tokens rel_sum should sit near 1 and progress near 0 for topic boxes; their prefixes are the z_english reference.
(3) positive controls: --control-docs (8) held documents of each --controls box (the battery's held specs, read from
its meta): the box's own rank among all boxes per score (and the best rank of its lineage: a nested box's ancestors
and descendants).

Output: OUT/trip.jsonl (a receipt row, then one row per prompt: every box's scores, tops, tiles, nulls, routing),
OUT/controls.json, OUT/report.md, OUT/receipt.json (checkpoint and fingerprint, mlx2 commit, resolved partition paths
and partition.json hashes, maps, arguments, timings, peak memory, watchdog and wiring, a causal check of the first
prompt, parity). OUT must be new or empty. Box statistics are gathered from the maps before the model loads into
--stats-cache (default OUT/../boxtrip-stats-cache/box-stats; reused when the partitions, maps, seed and twins are
unchanged; --prepare-only builds it on CPU).

Usage: amoeba_fn_box_trip.py --model M --partition main=DIR [--partition enshared=DIR2] --out OUT --assert-ple
       [--prompts P.jsonl] [--battery-meta BATTERY/battery.meta.json] [--joint-partition DIR] [--prepare-only]
Prompt rows: {"id", "category", "messages": [...], "tools": [...]?, "enable_thinking": bool?} or {"id", "prompt": raw}.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import amoeba_box_battery as BB  # noqa: E402  (Tiles, random_like, read_held_rows, _wired_arg: CPU-only helpers)

STATS_VERSION = 4
METRICS = ("rel_median", "rel_median_routed", "rel_sum", "progress", "progress_median", "cos_contrast", "cos_raw",
           "trip", "routed")
TOP_METRICS = ("progress", "gap_selected_progress", "z_english_progress", "cos_contrast", "trip", "rel_sum",
               "rel_median", "rel_median_routed", "routed")
REPORT_METRICS = ("progress", "gap_selected_progress", "z_english_progress", "cos_contrast", "trip", "routed")
SEGMENTS = ("prompt", "user_text", "continuation")

_TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {
        "city": {"type": "string", "description": "City name"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}},
    {"type": "function", "function": {
        "name": "search_files", "description": "Search a repository for a regular expression.",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"}, "path": {"type": "string"},
            "max_results": {"type": "integer", "minimum": 1}}, "required": ["pattern"]}}}]


def _u(text):
    return [{"role": "user", "content": text}]


DEFAULT_PROMPTS = [
    {"id": "code-python-1", "category": "code", "messages": _u(
        "Write a Python function that merges two sorted lists into one sorted list in linear time, with a docstring "
        "and two doctest examples.")},
    {"id": "code-python-2", "category": "code", "messages": _u(
        "Why does this raise KeyError, and how do I fix it?\n\n```python\ncounts = {}\nfor w in words:\n    "
        "counts[w] += 1\n```")},
    {"id": "code-shell-1", "category": "code", "messages": _u(
        "Give me a bash one-liner that finds every .log file under /var/log larger than 50 MB and modified in the "
        "last 7 days, sorted by size.")},
    {"id": "math-word-1", "category": "math", "messages": _u(
        "A train leaves at 9:40 and travels 210 km at 84 km/h, then stops for 25 minutes, then travels another "
        "96 km at 64 km/h. At what time does it arrive?")},
    {"id": "math-word-2", "category": "math", "messages": _u(
        "Maria buys 3 notebooks at $2.75 each and 4 pens. She pays with a $20 bill and gets $6.15 back. How much "
        "does one pen cost?")},
    {"id": "legal-1", "category": "legal", "messages": _u(
        "My landlord kept my whole security deposit without giving an itemized list of damages. What are my usual "
        "options under US tenant law?")},
    {"id": "legal-2", "category": "legal", "messages": _u(
        "What is the difference between a revocable and an irrevocable trust, and when would someone choose each?")},
    {"id": "finance-1", "category": "finance", "messages": _u(
        "I have $8,000 of credit card debt at 24% APR and $5,000 in savings. Should I pay the debt down or keep an "
        "emergency fund? Explain the trade-off.")},
    {"id": "finance-2", "category": "finance", "messages": _u(
        "How does a Roth IRA differ from a traditional 401(k) for someone in a low tax bracket today?")},
    {"id": "religion-1", "category": "religion", "messages": _u(
        "How do Augustine and Pelagius disagree about grace and free will?")},
    {"id": "religion-2", "category": "religion", "messages": _u(
        "What is the significance of the Eightfold Path in Theravada Buddhism?")},
    {"id": "physics-1", "category": "physics", "messages": _u(
        "Why does a spinning top precess instead of falling over? Explain with torque and angular momentum.")},
    {"id": "physics-2", "category": "physics", "messages": _u(
        "In special relativity, why can't two observers agree on whether two distant events are simultaneous?")},
    {"id": "cyber-1", "category": "cyber", "messages": _u(
        "Explain CVE-2021-44228 (Log4Shell): what component was vulnerable, how the JNDI lookup was abused, and how "
        "to mitigate it.")},
    {"id": "cyber-2", "category": "cyber", "messages": _u(
        "Our web server logs show many requests like /index.php?id=1' OR '1'='1. What attack is this and how should "
        "we harden the application?")},
    {"id": "lang-fra", "category": "language", "messages": _u(
        "Peux-tu m'expliquer pourquoi le ciel est bleu, avec des mots simples pour un enfant de dix ans ?")},
    {"id": "lang-deu", "category": "language", "messages": _u(
        "Wie bereite ich mich am besten auf ein Vorstellungsgespräch für eine Stelle als Ingenieur vor?")},
    {"id": "lang-cmn-1", "category": "language", "messages": _u("请用简单的话解释一下光合作用的过程。")},
    {"id": "lang-cmn-2", "category": "language", "messages": _u("周末去杭州旅游两天，有什么推荐的路线和美食？")},
    {"id": "chat-1", "category": "chat", "messages": _u("hey! how's your day going? mine's been kind of slow lol")},
    {"id": "chat-2", "category": "chat", "messages": _u(
        "I just adopted a cat and she keeps hiding under the bed. Is that normal?")},
    {"id": "tool-1", "category": "tool", "tools": _TOOLS, "messages": [
        {"role": "system", "content": "You are an assistant that calls tools when they help."},
        {"role": "user", "content": "What's the weather in Lyon right now, in celsius?"}]},
    {"id": "tool-2", "category": "tool", "tools": _TOOLS, "messages": [
        {"role": "system", "content": "You are a coding agent. Use the tools to inspect the repository before "
                                      "answering."},
        {"role": "user", "content": "Find where the function parse_config is defined in src/ and tell me what it "
                                    "returns."}]},
    {"id": "multiturn-1", "category": "multiturn", "messages": [
        {"role": "user", "content": "I'm planning a vegetable garden in a small backyard."},
        {"role": "assistant", "content": "Nice! How much sun does the spot get, and what would you like to grow?"},
        {"role": "user", "content": "About six hours of sun. Tomatoes, basil and maybe some peppers."},
        {"role": "assistant", "content": "Those all love sun. Do you want raised beds or containers?"},
        {"role": "user", "content": "Raised beds. How far apart should I plant them?"}]},
    {"id": "multiturn-2", "category": "multiturn", "messages": [
        {"role": "system", "content": "You are a concise, friendly assistant."},
        {"role": "user", "content": "Can you recommend a sci-fi novel?"},
        {"role": "assistant", "content": "Try 'The Left Hand of Darkness' by Ursula K. Le Guin."},
        {"role": "user", "content": "I've read it. Something with more space battles?"}]},
    {"id": "cooking-1", "category": "cooking", "messages": _u(
        "How do I make a proper French omelette that stays soft inside?")},
    {"id": "cooking-2", "category": "cooking", "messages": _u(
        "My bread dough didn't rise after two hours. What could have gone wrong?")},
    {"id": "games-1", "category": "games", "messages": _u(
        "In chess, how does en passant work, and when is it allowed?")},
    {"id": "games-2", "category": "games", "messages": _u(
        "In Catan, what happens when someone rolls a 7, and who has to discard cards?")},
]


# ------------------------------------------------------------------ prompts


def load_prompts(path):
    """--prompts JSONL rows (module docstring), or the default set."""
    if not path:
        return [dict(r) for r in DEFAULT_PROMPTS]
    rows = [json.loads(x) for x in Path(path).read_text().split("\n") if x.strip()]
    seen = set()
    for i, r in enumerate(rows):
        r.setdefault("id", f"p{i}")
        r.setdefault("category", "")
        if ("messages" in r) == ("prompt" in r):
            raise SystemExit(f"--prompts row {r['id']}: needs exactly one of messages / prompt")
        if r["id"] in seen:
            raise SystemExit(f"--prompts: duplicate id {r['id']}")
        seen.add(r["id"])
    return rows


def _ids_of(out):
    """Token ids from a chat-template result: a list, or a mapping with input_ids (a bare HF tokenizer)."""
    if hasattr(out, "keys") and "input_ids" in out:
        out = out["input_ids"]
    out = list(out)
    if out and isinstance(out[0], (list, tuple)):
        if len(out) != 1:
            raise SystemExit("chat template returned a batch")
        out = list(out[0])
    return [int(t) for t in out]


def user_span(host, ids):
    """Positions of the last user message's content (after the last '<|im_start|>user' line, up to its '<|im_end|>'),
    or None (a raw prompt, another template): the prompt without its template tokens."""
    try:
        pre, end = host.encode("<|im_start|>user\n"), host.encode("<|im_end|>")
    except Exception:  # noqa: BLE001
        return None
    if not pre or len(end) != 1:
        return None
    n = len(pre)
    starts = [i for i in range(len(ids) - n + 1) if ids[i:i + n] == pre]
    if not starts:
        return None
    s0 = starts[-1] + n
    e0 = next((i for i in range(s0, len(ids)) if ids[i] == end[0]), None)
    return (s0, e0) if e0 is not None and e0 > s0 else None


def render(host, row, enable_thinking=False):
    """Token ids of a prompt row: a messages row through the adapter's own prompt_tokens (the chat template exactly as
    served: tools, thinking default off unless the row or --enable-thinking asks); a prompt row raw. Returns (ids,
    route)."""
    if "prompt" in row:
        return host.encode(row["prompt"]), "raw"
    req = {"messages": row["messages"], "enable_thinking": bool(row.get("enable_thinking", enable_thinking))}
    if row.get("tools"):
        req["tools"] = row["tools"]
    ad = getattr(host, "adapter", None)
    if ad is not None and callable(getattr(ad, "prompt_tokens", None)):
        return _ids_of(ad.prompt_tokens(req)), f"{type(ad).__name__}.prompt_tokens"
    tok = getattr(host, "tok", None)
    if callable(getattr(tok, "apply_chat_template", None)):
        ids = tok.apply_chat_template(req["messages"], add_generation_prompt=True, tokenize=True,
                                      enable_thinking=req["enable_thinking"], tools=req.get("tools"))
        return _ids_of(ids), "tokenizer.apply_chat_template"
    raise SystemExit("this host has no chat template (no adapter prompt_tokens, no apply_chat_template)")


def eos_ids(host):
    tok = getattr(host, "tok", None)
    out = set()
    for name in ("eos_token_ids", "eos_token_id"):
        v = getattr(tok, name, None)
        if v is None:
            continue
        out |= {int(x) for x in (v if isinstance(v, (list, tuple, set)) else [v])}
    for s in ("<|im_end|>", "<|endoftext|>"):
        try:
            ids = host.encode(s)
        except Exception:  # noqa: BLE001 - a toy tokenizer without these strings
            continue
        if len(ids) == 1 and host.decode(ids) == s:
            out.add(int(ids[0]))
    return out


def greedy(host, ids, n, eos):
    """The model's own greedy continuation (n tokens, stopping after an EOS id), every tile open (Control reset). With
    the model's make_cache: cached decode (the served path's math); else full recompute. Returns (tokens, route)."""
    import mlx.core as mx
    host.ctl.reset()
    out = []
    if n <= 0:
        return out, "none"
    mk = getattr(host.model, "make_cache", None)
    if callable(mk):
        cache = mk()
        lg = host.model(mx.array([ids], dtype=mx.int32), cache=cache)
        for _ in range(n):
            t = int(mx.argmax(lg[0, -1].astype(mx.float32)).item())
            out.append(t)
            if t in eos:
                break
            lg = host.model(mx.array([[t]], dtype=mx.int32), cache=cache)
        mx.clear_cache()
        return out, "kv_cache"
    seq = list(ids)
    for _ in range(n):
        lg = host.full_logits(mx.array([seq], dtype=mx.int32))
        t = int(mx.argmax(lg[0, -1]).item())
        out.append(t)
        seq.append(t)
        if t in eos:
            break
    return out, "recompute"


# ------------------------------------------------------------------ capture


class Capture:
    """Per-unit drive on the partition's MoE layers, per segment of one forward (the firing capture's quantity)."""

    def __init__(self, host, layers, W):
        self.host = host
        self.layers = list(layers)
        self.row = {}
        EF = set()
        for s in host.sites:
            if s.layer in self.layers and s.is_moe and getattr(s, "E", None):
                if getattr(s, "units", None) != "neuron":
                    raise SystemExit(f"layer {s.layer}: MoE site not in neuron units (Host(moe_units='neuron'))")
                self.row[s.layer] = self.layers.index(s.layer)
                EF.add((int(s.E), int(s.F)))
        miss = [l for l in self.layers if l not in self.row]
        if miss:
            raise SystemExit(f"partition layers without a MoE neuron site in this model: {miss}")
        if len(EF) != 1:
            raise SystemExit(f"MoE sites of different shapes {sorted(EF)}")
        self.E, self.F = EF.pop()
        if self.E * self.F != W:
            raise SystemExit(f"partition width {W} != E*F {self.E}*{self.F}")
        self.W = W

    def run(self, batch, segs):
        """batch: equal-length token lists; segs: [(start, end)] position ranges. Returns X [S][B, nL, W] float32
        (mean |h| * w over the segment's positions), R [S][B, nL, E] float32 (routed share: routings / (len * k)) and
        the trunk's hidden output (mx)."""
        import mlx.core as mx
        h_ = self.host
        B = len(batch)
        nL, E, F = len(self.layers), self.E, self.F
        X = [[None] * nL for _ in segs]
        R = [[None] * nL for _ in segs]

        def stats(l, kind, p):
            r = self.row.get(l)
            if r is None or kind == "dense":
                return
            h = p.get("h")
            if h is None:
                raise SystemExit(f"layer {l}: no neuron activations in the stats payload")
            inds, w = p["inds"], p["w"]
            k = inds.shape[-1]
            val = mx.abs(h.astype(mx.float32)) * w.astype(mx.float32)[..., None]          # [B, T, k, F]
            for j, (s0, s1) in enumerate(segs):
                n = s1 - s0
                ii = inds[:, s0:s1].astype(mx.int32)
                flat = (mx.arange(B, dtype=mx.int32)[:, None, None] * E + ii).reshape(-1)
                v = val[:, s0:s1].reshape(-1, F)
                X[j][r] = mx.zeros((B * E, F), mx.float32).at[flat].add(v).reshape(B, E * F) / n
                R[j][r] = mx.zeros((B * E,), mx.float32).at[flat].add(1.0).reshape(B, E) / (n * k)
                mx.eval(X[j][r], R[j][r])
        h_.ctl.reset()
        h_.ctl.stats = stats
        h_.ctl.stats_fast = True
        try:
            hid = h_.hidden(mx.array(batch, dtype=mx.int32))
            hid = hid[0] if isinstance(hid, tuple) else hid
            mx.eval(hid)
        finally:
            h_.ctl.reset()
        for j in range(len(segs)):
            if any(x is None for x in X[j]):
                raise SystemExit("a partition layer did not report its MoE stats")
        Xn = [np.stack([np.array(x) for x in X[j]], axis=1) for j in range(len(segs))]
        Rn = [np.stack([np.array(x) for x in R[j]], axis=1) for j in range(len(segs))]
        mx.clear_cache()
        return Xn, Rn, hid


def parity(host, hid, P, cont):
    """(share of continuation tokens equal to the capture forward's argmax at the position before them, the first
    continuation index where they differ or None)."""
    import mlx.core as mx
    if not cont:
        return None, None
    lg = host.head(hid[:, P - 1:P - 1 + len(cont)])
    am = np.array(mx.argmax(lg[0], axis=-1)).tolist()
    eq = [int(a) == int(b) for a, b in zip(am, cont)]
    return round(float(np.mean(eq)), 4), next((i for i, e in enumerate(eq) if not e), None)


# ------------------------------------------------------------------ box statistics


def tile_members(key, prep, jprep=None):
    kind, _, name = key.partition(":")
    if kind == "box":
        return [name]
    if kind == "pair":
        return name.split("__")
    if kind == "group":
        return list((((prep.get("groups") or {}).get("kept") or {}).get(name) or {})["members"])
    if kind == "joint":
        return list(((jprep or prep).get("joint") or {})["tiles"][name]["members"])
    raise ValueError(key)


def lineage_of(prep, b):
    nest = prep.get("nested") or {}
    return set([b] + list((nest.get("ancestors") or {}).get(b, [])) + list((nest.get("descendants") or {}).get(b, [])))


class TilesMeta:
    """A tiles.npz's keys, layers and shape without decompressing its masks (a statistics-cache hit needs no more;
    BB.Tiles reads every mask: ~100 s on the FN category partition)."""

    def __init__(self, path):
        with np.load(path) as z:
            self.layers = [int(x) for x in z["layers"]]
            self.keys = [k for k in z.files if k != "layers"]
            self.shape = tuple(z["core"].shape) if "core" in z.files else None


class Sets:
    """Every scored unit set of one partition, concatenated: per set its flat unit indices into [nL * W], the map's
    t and b there (multi-member tiles: the members' mean), and the offsets."""

    def __init__(self, names, kinds, members, idx, t, b, off):
        self.names, self.kinds, self.members = list(names), list(kinds), [list(m) for m in members]
        self.idx, self.t, self.b, self.off = idx, t, b, off
        self.pos = {n: i for i, n in enumerate(self.names)}

    def save(self, path, ident):
        tmp = Path(str(path) + ".tmp.npz")
        np.savez(tmp, idx=self.idx, t=self.t, b=self.b, off=self.off,
                 meta=np.array(json.dumps({"ident": ident, "names": self.names, "kinds": self.kinds,
                                           "members": self.members})))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path, ident):
        try:
            z = np.load(path, allow_pickle=False)
            meta = json.loads(str(z["meta"]))
        except (OSError, ValueError, KeyError):
            return None
        if meta["ident"] != ident:
            return None
        return cls(meta["names"], meta["kinds"], meta["members"], z["idx"], z["t"], z["b"], z["off"])


class LazyTiles:
    """Single tiles of a tiles.npz read on demand (the joint partition: 13,557 masks, of which --joint-top are used)."""

    def __init__(self, path):
        self.path = Path(path)
        meta = TilesMeta(path)
        self.layers, self.keys, self.shape = meta.layers, meta.keys, meta.shape

    def mask(self, key):
        with np.load(self.path) as z:
            return z[key].astype(bool)


def file_identity(p):
    p = Path(p)
    st = p.stat()
    return {"path": str(p.resolve()), "size": st.st_size, "mtime": int(st.st_mtime)}


def map_identity(path):
    return {"map": file_identity(Path(path) / "map.npz"), "report": file_identity(Path(path) / "report.json")}


def check_map(path, prep, layers, W):
    rep = json.loads((Path(path) / "report.json").read_text())
    if rep.get("fingerprint") != prep.get("fingerprint"):
        raise SystemExit(f"map {path}: fingerprint {rep.get('fingerprint')} != partition {prep.get('fingerprint')}")
    ps = rep.get("per_site") or []
    if [s["layer"] for s in ps] != list(layers) or any(s["units"] != W for s in ps):
        raise SystemExit(f"map {path}: site layout differs from the partition's layers / width")
    return {"resolved": str(Path(path).resolve()), "baseline_corpus": (rep.get("baseline") or {}).get("corpus"),
            "target_corpus": (rep.get("target") or {}).get("corpus"), "capture_rev": rep.get("capture_rev")}


def capped_counts(counts, cap):
    """Per-layer counts scaled so their sum is at most cap (a twin of a very large box is a subsample)."""
    counts = np.asarray(counts, np.int64)
    tot = int(counts.sum())
    if not cap or tot <= cap:
        return counts
    return np.floor(counts * (cap / tot)).astype(np.int64)


def random_twin(own_rows, counts, W, rng):
    """Units drawn uniformly per layer outside the box (own_rows: per layer the box's local unit ids), counts[l] per
    layer, as flat indices (rejection sampling: a box holds a small share of any layer)."""
    out = []
    for l, k in enumerate(counts):
        k = int(k)
        if not k:
            continue
        ban = own_rows[l]
        if W - ban.size < k:
            raise SystemExit(f"layer row {l}: {k} twin units requested, {W - ban.size} outside the box")
        got = np.zeros(0, np.int64)
        while got.size < k:
            draw = rng.integers(0, W, size=2 * (k - got.size) + 16)
            draw = draw[~np.isin(draw, ban)]
            got = np.unique(np.concatenate([got, draw]))
        got = rng.permutation(got)[:k]
        out.append(np.sort(got) + l * W)
    return np.concatenate(out).astype(np.int64) if out else np.zeros(0, np.int64)


def gather_map(path, flat, W, nL, ratio, twins=(), every_source=False):
    """A map at sorted unique flat indices: mean_target, mean_baseline and the selection flag (target and
    mean_baseline <= ratio x mean_target > 0: the partition's selection rule), one site at a time. twins: [(own flat
    idx, per-layer counts, rng)]: per twin, units drawn per layer among the map's selected units outside the box,
    up to the count in that layer. Returns (t, b, sel, [(twin flat idx, t, b)])."""
    z = np.load(Path(path) / "map.npz")
    t = np.empty(flat.size, np.float32)
    b = np.empty(flat.size, np.float32)
    sel = np.zeros(flat.size, bool)
    picks = [[] for _ in twins]
    for s in range(nL):
        lo, hi = np.searchsorted(flat, [s * W, (s + 1) * W])
        if hi == lo and not twins:
            continue
        ts, bs = z[f"mean_target_{s}"], z[f"mean_baseline_{s}"]
        ok = z[f"target_{s}"].astype(bool) & (ts > 0) & (bs <= ratio * ts)
        if every_source:                    # the partition's --require-every-source selection
            ok &= z[f"every_source_{s}"].astype(bool)
        if hi > lo:
            loc = flat[lo:hi] - s * W
            t[lo:hi], b[lo:hi], sel[lo:hi] = ts[loc], bs[loc], ok[loc]
        for j, (own, counts, rng) in enumerate(twins):
            k = int(counts[s])
            if not k:
                continue
            o0, o1 = np.searchsorted(own, [s * W, (s + 1) * W])
            okj = ok.copy()
            okj[own[o0:o1] - s * W] = False
            cand = np.flatnonzero(okj)
            if cand.size:
                loc = np.sort(rng.choice(cand, min(k, cand.size), replace=False))
                picks[j].append((loc + s * W, ts[loc], bs[loc]))
    out = []
    for p in picks:
        if p:
            out.append((np.concatenate([x[0] for x in p]).astype(np.int64), np.concatenate([x[1] for x in p]),
                        np.concatenate([x[2] for x in p])))
        else:
            out.append((np.zeros(0, np.int64), np.zeros(0, np.float32), np.zeros(0, np.float32)))
    return t, b, sel, out


def build_sets(parts, W, seed, log=print, joint=None, joint_top=0, twins=4, twin_cap=50000):
    """parts: [(label, prep, Tiles)]. One pass over the box maps for every partition. Per partition: every box tile
    (kind box); per box `twins` random twins (kind null: the same per-layer counts outside the box, capped at
    twin_cap units, the box's own map) and `twins` selected twins (kind nullsel: units the box's map selects that the
    partition put elsewhere); every pair and family tile and (joint = (jprep, LazyTiles), the first partition's
    variant) the joint_top largest joint tiles. A multi-member tile's t is the mean over its topic members whose map
    selects the unit (all topic members where none does) and its b the topic members' mean: the English box's map
    has the language mix as baseline and the English corpus as target, so it enters only a set it alone names. Returns ({label: Sets}, {box: map info})."""
    want = {}                     # map path -> [(label, set name, flat idx)]
    seltw = {}                    # map path -> [(label, twin name, box, own flat idx, counts, rng)]
    plan = {}                     # label -> [(name, kind, members, flat idx)]
    maps = {}
    ratios = {}
    for pi, (label, prep, tz) in enumerate(parts):
        layers = tz.layers
        keys = list(tz.keys)
        ebox = (prep.get("english_box") or {}).get("name")
        rows = []
        boxes = sorted(k[4:] for k in keys if k.startswith("box:"))
        ratio = float(prep.get("ratio") or 0.25)
        for i, b in enumerate(boxes):
            info = prep["boxes"].get(b) or {}
            mp = info.get("map") or ((prep.get("english_box") or {}).get("map") if b == ebox else None)
            if not mp:
                raise SystemExit(f"partition {label}: box {b} names no map")
            if b in maps and maps[b]["path"] != mp:
                raise SystemExit(f"box {b}: partitions name different maps ({maps[b]['path']} vs {mp})")
            if ratios.setdefault(mp, (ratio, bool(prep.get("require_every_source")))) != \
                    (ratio, bool(prep.get("require_every_source"))):
                raise SystemExit(f"map {mp}: partitions of different selection ratios")
            if b not in maps:
                maps[b] = {"path": mp, **check_map(mp, prep, layers, W)}
            m = tz.mask(f"box:{b}")
            flat = np.flatnonzero(m).astype(np.int64)
            rows.append((b, "box", [b], flat))
            counts = capped_counts(m.sum(1), twin_cap)
            own_rows = [np.flatnonzero(m[l]) for l in range(m.shape[0])]
            for j in range(twins):
                rng = np.random.default_rng([seed, 1000 + i, j])
                rows.append((f"null:{b}#{j}", "null", [b], random_twin(own_rows, counts, W, rng)))
                seltw.setdefault(mp, []).append((label, f"nullsel:{b}#{j}", b, flat, counts,
                                                 np.random.default_rng([seed, 2000 + i, j])))
        tiles = [(k, k.partition(":")[0], None) for k in keys if k.startswith(("pair:", "group:"))]
        src = {k: tz for k, _, _ in tiles}
        if pi == 0 and joint is not None:
            jprep, jz = joint
            jt = (jprep.get("joint") or {}).get("tiles") or {}
            top = sorted(jt, key=lambda n: (-jt[n]["units"], n))[:joint_top]
            for n in top:
                tiles.append((f"joint:{n}", "joint", jprep))
                src[f"joint:{n}"] = jz
        for k, kind, jp in tiles:
            mem = tile_members(k, prep, jp)
            for b in mem:
                if b not in maps:
                    raise SystemExit(f"partition {label}: tile {k} names {b}, which has no box map")
            rows.append((k, kind, mem, np.flatnonzero(src[k].mask(k)).astype(np.int64)))
        plan[label] = (rows, ebox)
        for name, kind, mem, flat in rows:
            for b in mem:
                want.setdefault(maps[b]["path"], []).append((label, name, b, flat))
    # per set and member group (topic members; all members, used only when the English box is the only member):
    # t sum over selecting members, their count, t sum, b sum, member count
    vals = {label: {name: {g: [np.zeros(flat.size), np.zeros(flat.size), np.zeros(flat.size), np.zeros(flat.size), 0]
                           for g in ("topic", "all")} for name, _, _, flat in rows}
            for label, (rows, _e) in plan.items()}
    twin_rows = {label: {} for label in plan}
    nL = len(parts[0][2].layers) if parts else 0
    t0 = time.time()
    for j, (mp, reqs) in enumerate(sorted(want.items())):
        u = np.unique(np.concatenate([f for *_, f in reqs])) if reqs else np.zeros(0, np.int64)
        tw = seltw.get(mp, [])
        t, b, sel, got = gather_map(mp, u, W, nL, ratios[mp][0], [(own, cnt, rng) for *_, own, cnt, rng in tw],
                                    ratios[mp][1])
        for (label, name, bx, *_), g in zip(tw, got):
            twin_rows[label][name] = (bx, *g)
        for label, name, bx, flat in reqs:
            p = np.searchsorted(u, flat)
            s_ = sel[p]
            for g in (("topic", "all") if bx != plan[label][1] else ("all",)):
                v = vals[label][name][g]
                v[0] += np.where(s_, t[p], 0.0)
                v[1] += s_
                v[2] += t[p]
                v[3] += b[p]
                v[4] += 1
        if (j + 1) % 20 == 0:
            log(f"box maps: {j + 1}/{len(want)} read ({time.time() - t0:.0f}s)")
    out = {}
    for label, (rows, _e) in plan.items():
        names, kinds, members, idx, tt, bb, off = [], [], [], [], [], [], [0]

        def put(name, kind, mem, flat, t_, b_):
            names.append(name)
            kinds.append(kind)
            members.append(mem)
            idx.append(flat)
            tt.append(np.asarray(t_, np.float32))
            bb.append(np.asarray(b_, np.float32))
            off.append(off[-1] + flat.size)
        for name, kind, mem, flat in rows:
            vv = vals[label][name]
            v = vv["topic"] if vv["topic"][4] else vv["all"]
            n = max(1, v[4])
            t_ = np.where(v[1] > 0, v[0] / np.maximum(v[1], 1), v[2] / n)
            put(name, kind, mem, flat, t_, v[3] / n)
        for name in sorted(twin_rows[label]):
            bx, fl, t_, b_ = twin_rows[label][name]
            put(name, "nullsel", [bx], fl, t_, b_)
        out[label] = Sets(names, kinds, members, np.concatenate(idx) if idx else np.zeros(0, np.int64),
                          np.concatenate(tt) if tt else np.zeros(0, np.float32),
                          np.concatenate(bb) if bb else np.zeros(0, np.float32), np.array(off, np.int64))
    return out, maps


# ------------------------------------------------------------------ scores


def score_sets(S, x, routed, W, F, E, ratio_floor=0.01, names=None):
    """Scores of every set (or of the named ones) for one flat drive vector x [nL * W] and a routed mask [nL, E]
    (bool). Returns {set name: {metric: value, "units": n}}; a set with no units scores None."""
    sel = range(len(S.names)) if names is None else [S.pos[n] for n in names]
    rflat = None if routed is None else np.asarray(routed, bool).reshape(-1)
    out = {}
    for i in sel:
        a, z = int(S.off[i]), int(S.off[i + 1])
        n = z - a
        if n == 0:
            out[S.names[i]] = {"units": 0, **{m: None for m in METRICS}}
            continue
        idx = S.idx[a:z]
        xs = x[idx].astype(np.float64)
        t = S.t[a:z].astype(np.float64)
        b = S.b[a:z].astype(np.float64)
        ratio = xs / np.maximum(np.maximum(b, ratio_floor * t), 1e-12)
        c = t - b
        dx = xs - b

        def cos(u, v):
            nu, nv = np.linalg.norm(u), np.linalg.norm(v)
            return float(u @ v / (nu * nv)) if nu > 0 and nv > 0 else 0.0
        pos = c > 0
        r = {"units": n, "rel_median": float(np.median(ratio)),
             "rel_sum": float(xs.sum() / max(b.sum(), 1e-12)),
             "progress": float(dx.sum() / np.abs(c).sum()) if np.abs(c).sum() > 0 else None,
             "progress_median": float(np.median(np.clip(dx[pos] / c[pos], -1.0, 2.0))) if pos.any() else None,
             "cos_contrast": cos(dx, c), "cos_raw": cos(xs, c), "trip": float(np.mean(xs >= t))}
        if rflat is not None:
            ru = rflat[(idx // W) * E + (idx % W) // F]
            r["routed"] = float(ru.mean())
            r["rel_median_routed"] = float(np.median(ratio[ru])) if ru.any() else None
        else:
            r["routed"] = r["rel_median_routed"] = None
        out[S.names[i]] = r
    return out


def add_or_scores(S, sc):
    """OR-of-members for every multi-member tile: the best member box's value per metric."""
    for i, name in enumerate(S.names):
        if S.kinds[i] in ("pair", "group", "joint") and name in sc:
            for m in METRICS:
                v = [sc[b][m] for b in S.members[i] if b in sc and sc[b].get(m) is not None]
                sc[name][f"or_{m}"] = max(v) if v else None
    return sc


NULLS = {"null": "random", "nullsel": "selected"}
Z_METRICS = ("progress", "cos_contrast", "trip", "rel_sum")
SIZE_BANDS = ((0, 640), (640, 10000), (10000, None))


def twins_of(S):
    out = {}
    for i, n in enumerate(S.names):
        if S.kinds[i] in NULLS:
            out.setdefault((S.kinds[i], S.members[i][0]), []).append(n)
    return out


SD_FLOOR = {"progress": 0.05, "cos_contrast": 0.02, "trip": 0.02, "rel_sum": 0.05}


def add_twin_z(S, sc, tw=None):
    """Per box and null kind, per Z metric: gap = box - twin mean (ranked: the box's own units against units of the
    same map elsewhere); z = gap / max(twin sd, SD_FLOOR) and p = (1 + #twins >= box) / (1 + K), recorded only: with
    a few twins of a systematically different unit set their sd is tiny, so z is not a ranking."""
    tw = twins_of(S) if tw is None else tw
    for i, b in enumerate(S.names):
        if S.kinds[i] != "box" or b not in sc:
            continue
        for kind, lab in NULLS.items():
            names = [n for n in tw.get((kind, b), []) if n in sc]
            for m in Z_METRICS:
                v = sc[b].get(m)
                tv = np.array([sc[n][m] for n in names if sc[n].get(m) is not None], float)
                if v is None or tv.size < 2:
                    sc[b][f"gap_{lab}_{m}"] = sc[b][f"z_{lab}_{m}"] = sc[b][f"p_{lab}_{m}"] = None
                    continue
                sc[b][f"gap_{lab}_{m}"] = float(v - tv.mean())
                sc[b][f"z_{lab}_{m}"] = float((v - tv.mean()) / max(tv.std(ddof=1), SD_FLOOR[m]))
                sc[b][f"p_{lab}_{m}"] = float((1 + (tv >= v).sum()) / (1 + tv.size))
    return sc


def add_english_z(S, sc, ref, n):
    """z of each box against held English text of about n tokens in the segment's own shape: ref = {key: {"tokens":
    mean segment length, "boxes": {box: {metric: [mean, sd]}}}} (raw prefixes, or English wrapped as a chat user
    message: its prompt or its user-text segment); the key whose length is nearest n (log scale). sd is floored
    (SD_FLOOR): a language box's English spread is ~1e-3, so a small deviation would read as a large z."""
    if not ref:
        return sc, None
    key = min(ref, key=lambda k: (abs(np.log(max(ref[k]["tokens"], 1) / max(n, 1))), str(k)))
    rb = ref[key]["boxes"]
    for i, b in enumerate(S.names):
        if S.kinds[i] != "box" or b not in sc:
            continue
        for m in Z_METRICS:
            v, ms = sc[b].get(m), (rb.get(b) or {}).get(m)
            sc[b][f"z_english_{m}"] = (float((v - ms[0]) / max(ms[1], SD_FLOOR[m]))
                                       if v is not None and ms and ms[0] is not None else None)
    return sc, {"key": key, "tokens": ref[key]["tokens"]}


def ranked(sc, S, kind, metric, k=None, min_units=0):
    """Sets of a kind with at least min_units units by metric (ties: the larger set first): [name, value, units]."""
    rows = [(n, sc[n][metric], sc[n]["units"]) for i, n in enumerate(S.names)
            if S.kinds[i] == kind and n in sc and sc[n].get(metric) is not None and sc[n]["units"] >= min_units]
    rows.sort(key=lambda r: (-r[1], -r[2], r[0]))
    return [[n, round(v, 4), u] for n, v, u in (rows if k is None else rows[:k])]


def null_summary(sc, S, kind="null", tw=None):
    """A twin control per metric and box size band (box units): the twins' p50 / p99 / max, the real boxes above the
    band's twin max, and how many boxes beat their own twins' mean. kind null: random units of the same per-layer
    counts outside the box (the shuffled labels); nullsel: units the box's map selects that the partition put
    elsewhere (same per-layer counts where the map has enough)."""
    tw = twins_of(S) if tw is None else tw
    out = {}
    boxes = [n for i, n in enumerate(S.names) if S.kinds[i] == "box" and sc.get(n, {}).get("units")]
    for m in METRICS:
        bands = {}
        for lo, hi in SIZE_BANDS:
            bb = [b for b in boxes if sc[b]["units"] >= lo and (hi is None or sc[b]["units"] < hi)]
            nv = np.array([sc[n][m] for b in bb for n in tw.get((kind, b), []) if sc.get(n, {}).get(m) is not None])
            if not nv.size:
                continue
            mx_ = float(nv.max())
            rv = {b: sc[b][m] for b in bb if sc[b].get(m) is not None}
            own = {b: [sc[n][m] for n in tw.get((kind, b), []) if sc.get(n, {}).get(m) is not None] for b in rv}
            bands[f"{lo}-{hi or 'max'}"] = {
                "boxes": len(bb), "p50": round(float(np.median(nv)), 4), "p99": round(float(np.quantile(nv, 0.99)), 4),
                "max": round(mx_, 4), "boxes_above_null_max": sorted(b for b, v in rv.items() if v > mx_),
                "boxes_above_own_twin_mean": int(sum(v > np.mean(own[b]) for b, v in rv.items() if own[b])),
                "boxes_with_twins": int(sum(bool(own[b]) for b in rv))}
        if bands:
            out[m] = bands
    return out


def routing_summary(Rseg, layers, top=20):
    """Routed experts of one segment ([nL, E] routed shares): the top experts and per-layer counts."""
    R = np.asarray(Rseg)
    flat = np.argsort(-R.reshape(-1), kind="stable")[:top]
    E = R.shape[1]
    return {"experts_routed_per_layer_mean": round(float((R > 0).sum(1).mean()), 2),
            "experts_routed_total": int((R > 0).sum()),
            "top": [[int(layers[f // E]), int(f % E), round(float(R.reshape(-1)[f]), 4)] for f in flat if
                    R.reshape(-1)[f] > 0]}


def _r(v):
    return round(v, 4) if isinstance(v, float) else v


def segment_result(S, x, routed, W, F, E, layers, top, ratio_floor, routing=None, en_ref=None, n_tokens=None,
                   min_units=640, tw=None):
    tw = twins_of(S) if tw is None else tw
    sc = add_twin_z(S, add_or_scores(S, score_sets(S, x, routed, W, F, E, ratio_floor)), tw)
    sc, en_len = add_english_z(S, sc, en_ref, n_tokens or 0)
    res = {"tokens": n_tokens, "english_ref_tokens": en_len, "min_units": min_units,
           "boxes": {n: {k: _r(v) for k, v in sc[n].items()} for i, n in enumerate(S.names) if S.kinds[i] == "box"},
           "top": {m: ranked(sc, S, "box", m, top, min_units) for m in TOP_METRICS},
           "small_boxes_top": {m: [r for r in ranked(sc, S, "box", m) if r[2] < min_units][:5]
                               for m in ("progress", "cos_contrast")},
           "tiles_top": {kind: {m: [[n, v, u, _r(sc[n].get(f"or_{m}"))] for n, v, u in
                                    ranked(sc, S, kind, m, top, min_units)]
                                for m in ("progress", "cos_contrast", "trip")}
                         for kind in ("pair", "group", "joint") if kind in S.kinds},
           "null": {NULLS[k]: null_summary(sc, S, k, tw) for k in NULLS},
           "null_twins": {NULLS[k]: {n.partition(":")[2]: {"units": sc[n]["units"], **{
               m: _r(sc[n].get(m)) for m in Z_METRICS}} for i, n in enumerate(S.names) if S.kinds[i] == k}
               for k in NULLS}}
    if routing is not None:
        res["routing"] = routing
    return res


def rank_of(sc_boxes, metric, b, lineage=()):
    rows = sorted(((n, v[metric]) for n, v in sc_boxes.items() if v.get(metric) is not None),
                  key=lambda r: (-r[1], r[0]))
    order = [n for n, _ in rows]
    own = order.index(b) + 1 if b in order else None
    lin = [order.index(x) + 1 for x in lineage if x in order]
    return own, (min(lin) if lin else own), len(order)


# ------------------------------------------------------------------ report


def _f(v, spec=".3f"):
    """A number for the report; None or NaN as n/a (a control with no long-enough held document)."""
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return format(v, spec)


def fmt_top(seg, m):
    return [f"{n} {_f(v, '.3g')} ({u:,})" for n, v, u in seg["top"].get(m, [])]


def write_report(path, receipt, rows, controls, top):
    L = [f"# Box trip: {receipt['checkpoint']}", "",
         f"mlx2 {receipt['mlx2_source_revision']}; partitions " +
         ", ".join(f"{k} = `{v['resolved']}` (partition.json {v['partition_json_sha256_12']})"
                   for k, v in receipt["partitions"].items()) +
         f"; {len(rows)} prompts; continuation {receipt['args']['gen']} greedy tokens. Fingerprint "
         f"`{receipt['fingerprint']}`.", "",
         ("Scores (module docstring), per box over its units, x = the segment's drive, t / b = the box map's target / "
          "English-baseline means (lang.eng: baseline = the 10-language mix): progress = sum(x - b) / sum|t - b| "
          "(0 = English, 1 = the map's target); gap_selected_progress = progress minus the mean progress of the "
          "box's selected twins (units its map selects that the partition put elsewhere); z_english_progress = "
          "progress against held English of about the segment's length and shape (raw prefixes, or English as a chat "
          "user message for the prompt and user-text segments; sd floored); cos_contrast = cosine(x - b, t - b); trip = share of units "
          "with x >= t; routed = share of units in experts routed in the segment. Tables rank boxes of at least "
          f"{receipt['args']['min_units']} units (value, units)."), ""]
    cc = receipt.get("causal_check")
    if cc:
        L += [f"Causal check ({cc['prompt']}): the prompt drive from the prompt + continuation forward vs a "
              f"prompt-only forward differs by at most {_f(cc['max_abs_diff'], '.3g')} (max drive "
              f"{_f(cc['max_abs_drive'], '.3g')}, relative {_f(cc['relative'], '.3g')}): "
              + ("ok" if cc["ok"] else "**FAILED: the prompt segment is not the prompt-only drive**"), ""]
    if receipt.get("parity_below_0.9"):
        L += [f"**Parity below 0.9** (cached decode vs the capture forward): {receipt['parity_below_0.9']}", ""]
    for lab in receipt["partitions"]:
        L += [f"## Controls ({lab})", ""]
        c = controls.get(lab) or {}
        en = c.get("english")
        if en:
            L += [f"English held ({en['docs']} docs, raw text, prefixes {en['lengths']} tokens; the maps used 256): at "
                  f"{en['tokens']} tokens the median over topic boxes of rel_sum is {_f(en['topic_rel_sum_median'])} "
                  f"(near 1 = the maps' English baseline is reproduced), progress {_f(en['topic_progress_median'])} "
                  f"(near 0), trip {_f(en['topic_trip_median'])}, rel_median {_f(en['topic_rel_median_median'])}. "
                  "Median topic-box progress by prefix length: " + ", ".join(
                      f"{k}: {_f(v)}" for k, v in en["topic_progress_median_by_length"].items()) +
                  ". Top boxes on English by progress: " + ", ".join(
                      f"{n} {_f(v, '.3g')}" for n, v in en["top"]["progress"][:5]), ""]
            ch = en.get("chat_reference")
            if ch:
                L += [f"English as a chat user message ({ch['docs']} docs, user text of {ch['lengths']} tokens): median "
                      "topic-box progress of the whole templated prompt by length: " + ", ".join(
                          f"{k}: {_f(v)}" for k, v in ch["topic_progress_median_chat_prompt"].items()) +
                      " (template tokens alone move this off 0; the prompts' z_english uses this reference)", ""]
        if c.get("positive"):
            pm = ("progress", "gap_selected_progress", "cos_contrast", "trip")
            L += ["| control box | docs | " + " | ".join(f"rank {m} (lineage) / of" for m in pm) +
                  " | progress own / random twins / selected twins |", "|---|---|" + "---|" * (len(pm) + 1)]
            for b, p in c["positive"].items():
                L.append(f"| {b} | {p['docs']} | " + " | ".join(
                    f"{p['rank_of_mean'][m]['own']} ({p['rank_of_mean'][m]['lineage']}) / {p['rank_of_mean'][m]['of']}"
                    for m in pm) + f" | {_f(p['own_progress'])} / {_f(p['random_twin_progress'])} / "
                    f"{_f(p['selected_twin_progress'])} |")
            L.append("")
    for r in rows:
        warn = " **parity below 0.9: the cached decode and the capture forward disagree**" if (
            r["parity"] is not None and r["parity"] < 0.9) else ""
        L += [f"## {r['id']} ({r['category']})", "",
              f"{r['prompt_tokens']} prompt tokens ({r['render']}); user text {r['user_text_span'] or r.get('user_text_note')}; "
              f"continuation {r['continuation_tokens']} tokens ({r['decode_route']}, parity {r['parity']}, first "
              f"divergence {r.get('first_divergence')}){warn}: " + json.dumps(r["continuation_text"][:300]), ""]
        for seg in SEGMENTS:
            for lab, res in (r["segments"].get(seg) or {}).items():
                er = res["english_ref_tokens"] or {}
                L += [f"### {seg}, {lab} ({res['tokens']} tokens; English reference {er.get('key')} "
                      f"({_f(er.get('tokens'), '.0f')} tokens))", "",
                      "| # | " + " | ".join(REPORT_METRICS) + " |", "|---|" + "---|" * len(REPORT_METRICS)]
                cols = [fmt_top(res, m) for m in REPORT_METRICS]
                for k in range(top):
                    L.append(f"| {k + 1} | " + " | ".join(c[k] if k < len(c) else "" for c in cols) + " |")
                L.append("")
                for nk, nl in res["null"].items():
                    pr = nl.get("progress") or {}
                    L.append(f"Null ({nk} twins), progress by box size band: " + "; ".join(
                        f"{band} p99 {_f(v['p99'], '.3g')} max {_f(v['max'], '.3g')}, {len(v['boxes_above_null_max'])}/{v['boxes']} "
                        f"boxes above the max, {v['boxes_above_own_twin_mean']} above their own twins' mean"
                        for band, v in pr.items()))
                for kind, tt in res["tiles_top"].items():
                    L.append(f"Top {kind} tiles by progress (own units / OR of members, units): " + ", ".join(
                        f"{n} {_f(v, '.3g')}/{o} ({u:,})" for n, v, u, o in tt["progress"][:8]))
                if res.get("routing"):
                    ro = res["routing"]
                    L.append(f"Routing: {ro['experts_routed_per_layer_mean']} experts routed per layer (mean), "
                             f"top (layer, expert, share): " + ", ".join(f"({a},{b},{c})" for a, b, c in
                                                                         ro["top"][:8]))
                L.append("")
    path.write_text("\n".join(L) + "\n")


# ------------------------------------------------------------------ main


def sha12(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]


def stats_identity(parts, jpath, a):
    ident = {"version": STATS_VERSION, "seed": a.seed, "twins": a.twins, "twin_cap": a.twin_cap, "joint": None,
             "partitions": []}
    for label, pd, _prep, _tz in parts:
        ident["partitions"].append({"label": label, "tiles": file_identity(Path(pd) / "tiles.npz"),
                                    "partition_json_sha256_12": sha12(Path(pd) / "partition.json")})
    if jpath:
        ident["joint"] = {"tiles": file_identity(Path(jpath) / "tiles.npz"), "top": a.joint_top,
                          "partition_json_sha256_12": sha12(Path(jpath) / "partition.json")}
    maps = set()
    for _l, _pd, prep, _tz in parts:
        for v in prep["boxes"].values():
            maps.add(v.get("map") or "")
        maps.add((prep.get("english_box") or {}).get("map") or "")
    ident["maps"] = [map_identity(m) for m in sorted(x for x in maps if x)]
    return hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:16], ident


def held_specs(a):
    """Control held specs: --held NAME=SPEC wins over the battery meta's."""
    specs = {}
    if a.battery_meta:
        m = json.loads(Path(a.battery_meta).read_text())
        for k, v in (m.get("held") or {}).items():
            specs[k] = v["spec"]
    for x in a.held:
        specs[x.partition("=")[0]] = x
    return specs


def held_docs(host, spec, n, seed):
    """The battery's held reading (split held, first T tokens of documents of at least T tokens, seeded order)."""
    rest = spec.partition("=")[2]
    rest, _, tok = rest.partition("@")
    T = int(tok or 256)
    kind = "gsm8k" if rest.startswith("gsm8k:") else "corpus"
    path, _, filt = rest.removeprefix("gsm8k:").partition("?")
    rows = BB.read_held_rows(path, kind, filt)
    np.random.default_rng(seed).shuffle(rows)
    seqs = []
    for r in rows:
        t = host.encode(r["text"])
        if len(t) >= T:
            seqs.append(t[:T])
        if len(seqs) >= n:
            break
    return seqs, T


def _mean(vals):
    v = [x for x in vals if x is not None]
    return float(np.mean(v)) if v else None


def check_joint(jprep, jz, prep, tz):
    """The joint partition must be the joint-tile variant of the first partition (the battery's rule, cheaply)."""
    if sorted(k for k in jz.keys if not k.startswith("joint:")) != sorted(tz.keys):
        raise SystemExit("--joint-partition: its non-joint tiles differ from the first --partition's")
    for k in ("ratio", "inputs", "fingerprint"):
        if jprep.get(k) != prep.get(k):
            raise SystemExit(f"--joint-partition: {k} differs from the first --partition's")
    if {b: v.get("units") for b, v in jprep["boxes"].items()} != {b: v.get("units") for b, v in prep["boxes"].items()}:
        raise SystemExit("--joint-partition: box unit counts differ from the first --partition's")


def main(argv=None, host_factory=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--partition", action="append", required=True, help="LABEL=DIR (repeatable); DIR alone: main")
    ap.add_argument("--joint-partition", default="", help="joint-tile variant of the first partition")
    ap.add_argument("--joint-top", type=int, default=30)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts", default="")
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--enable-thinking", action="store_true", help="thinking on for messages rows (served default off)")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--min-units", type=int, default=640, help="rank only sets of at least this many units (one "
                    "expert's worth on FN); the smallest boxes get their own short list")
    ap.add_argument("--ratio-floor", type=float, default=0.01)
    ap.add_argument("--twins", type=int, default=4, help="random and selected twins per box")
    ap.add_argument("--twin-cap", type=int, default=50000, help="units per twin at most (a subsample of a big box)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--battery-meta", default="", help="the battery's battery.meta.json (held specs of the controls)")
    ap.add_argument("--held", action="append", default=[], help="NAME=SPEC (the battery's --held form)")
    ap.add_argument("--english", default="english", help="the English held set's name")
    ap.add_argument("--english-docs", type=int, default=80)
    ap.add_argument("--english-lengths", default="16,32,64,128,256", help="English prefix lengths of the reference")
    ap.add_argument("--english-chat-docs", type=int, default=16, help="English docs also wrapped as a chat user message")
    ap.add_argument("--english-chat-lengths", default="16,32,64,128")
    ap.add_argument("--controls", default="prog.python,hum.religion,sec.vuln,cmn")
    ap.add_argument("--control-docs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--stats-cache", default="", help="box statistics cache path prefix (default "
                    "OUT/../boxtrip-stats-cache/box-stats; reused when its identity matches)")
    ap.add_argument("--prepare-only", action="store_true", help="build the box statistics (CPU) and exit")
    ap.add_argument("--assert-ple", action="store_true")
    ap.add_argument("--memory-gb", type=float, default=85)
    ap.add_argument("--watchdog-gb", type=float, default=92)
    ap.add_argument("--watchdog-hard-gb", type=float, default=None)
    ap.add_argument("--watchdog-every-s", type=float, default=1.0)
    ap.add_argument("--min-avail-gb", type=float, default=15.0)
    ap.add_argument("--swap-growth-gb", type=float, default=2.0)
    ap.add_argument("--compressor-growth-gb", type=float, default=12.0)
    ap.add_argument("--waiver-note", default="")
    ap.add_argument("--wired-gb", default=os.environ.get("AMOEBA_WIRED_GB", "").strip() or "auto",
                    type=BB._wired_arg, help="as amoeba_box_battery.py --wired-gb (auto: MLX active after load + 10)")
    a = ap.parse_args(argv)
    log = lambda *m: print(time.strftime("%H:%M:%S"), *m, flush=True)    # noqa: E731
    t_start = time.time()
    out = Path(a.out)
    if not a.prepare_only:
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f"{out} is not empty: refusing to overwrite evidence (use a new --out)")
        out.mkdir(parents=True, exist_ok=True)
    parts = []
    for spec in a.partition:
        label, _, d = spec.rpartition("=")
        label = label or "main"
        if any(label == p[0] for p in parts):
            raise SystemExit(f"--partition label {label} twice")
        prep = json.loads((Path(d) / "partition.json").read_text())
        parts.append((label, d, prep, TilesMeta(Path(d) / "tiles.npz")))
    fp = parts[0][2].get("fingerprint")
    layers, shape = parts[0][3].layers, parts[0][3].shape
    for label, d, prep, tz in parts:
        if prep.get("fingerprint") != fp or tz.layers != layers or tz.shape != shape:
            raise SystemExit(f"partition {label}: another model or layout than {parts[0][0]}")
    W = int(shape[1])
    joint = None
    if a.joint_partition:
        jprep = json.loads((Path(a.joint_partition) / "partition.json").read_text())
        jz = LazyTiles(Path(a.joint_partition) / "tiles.npz")
        if jz.layers != layers or jz.shape != shape:
            raise SystemExit("--joint-partition: another layout")
        if not (jprep.get("joint") or {}).get("tiles"):
            raise SystemExit("--joint-partition has no joint tiles")
        check_joint(jprep, jz, parts[0][2], parts[0][3])
        joint = (jprep, jz)
    sid, sident = stats_identity(parts, a.joint_partition, a)
    cache = Path(a.stats_cache) if a.stats_cache else out.resolve().parent / "boxtrip-stats-cache" / "box-stats"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cached = cache.exists() and json.loads(cache.read_text()).get("id") == sid and all(
        Path(f"{cache}.{lab}.npz").exists() for lab, *_ in parts)
    sets = {lab: Sets.load(Path(f"{cache}.{lab}.npz"), sid) for lab, *_ in parts} if cached else {}
    if cached and all(S is not None for S in sets.values()):
        maps = json.loads(cache.read_text())["maps"]
        log(f"box statistics from {cache} ({sid})")
    else:
        t0 = time.time()
        full = [(lab, prep, BB.Tiles(Path(d) / "tiles.npz")) for lab, d, prep, _tz in parts]
        log(f"tiles read in {time.time() - t0:.0f}s")
        sets, maps = build_sets(full, W, a.seed, log, joint, a.joint_top, a.twins, a.twin_cap)
        del full
        for lab, S in sets.items():
            S.save(Path(f"{cache}.{lab}.npz"), sid)
        tmp = Path(f"{cache}.tmp")
        tmp.write_text(json.dumps({"id": sid, "identity": sident, "maps": maps}, indent=1))
        os.replace(tmp, cache)
        log(f"box statistics built in {time.time() - t0:.0f}s -> {cache} ({sid})")
    for lab, S in sets.items():
        log(f"{lab}: {len(S.names)} sets ({sum(k == 'box' for k in S.kinds)} boxes), {S.idx.size:,} unit entries")
    if a.prepare_only:
        return 0
    prompts = load_prompts(a.prompts)
    specs = held_specs(a)
    ctrl = [c for c in a.controls.split(",") if c]
    missing = [c for c in ctrl + ([a.english] if a.english_docs else []) if c not in specs]
    if missing:
        raise SystemExit(f"no held spec for controls {missing} (--battery-meta or --held)")
    for c in ctrl:
        if f"box:{c}" not in parts[0][3].keys:
            raise SystemExit(f"control {c} is not a box of the partition")
    from research.amoeba.carve import ops as O
    O.prepare_wiring(a.wired_gb)
    O.footprint_watchdog(a.watchdog_gb, log, swap_growth_gb=a.swap_growth_gb, every_s=a.watchdog_every_s,
                         min_avail_gb=a.min_avail_gb, hard_gb=a.watchdog_hard_gb,
                         compressor_growth_gb=a.compressor_growth_gb, receipt=out / "watchdog-stop.json",
                         watcher=True)
    O.setup_mlx(a.memory_gb, 2)
    t_load = time.time()
    if host_factory is None:
        from research.amoeba.carve.host import Host
        host = Host(a.model, log=log, moe_units="neuron")
    else:
        host = host_factory(a.model)
    if host.fingerprint != fp:
        raise SystemExit(f"partition of {fp}, model is {host.fingerprint}")
    if a.assert_ple:
        tables = getattr(getattr(host, "adapter", None), "_tables", None)
        if not tables or Path(os.environ.get("MLX_QWEN4_PLE_NVME", "")).expanduser().resolve() != \
                Path(a.model).expanduser().resolve() / "ple_rows.bin":
            raise SystemExit("PLE offload is not active (FN rule: assert it)")
    wired = O.wire_weights(a.wired_gb, log, memory_gb=a.memory_gb)
    load_s = time.time() - t_load
    cap = Capture(host, layers, W)
    E, F = cap.E, cap.F
    eos = eos_ids(host)
    t_run = time.time()
    rev = O.git_rev()
    receipt = {"schema": "mlx2.amoeba-box-trip.v2", "checkpoint": host.checkpoint, "fingerprint": host.fingerprint,
               "model": str(Path(a.model).expanduser().resolve()), "mlx2_source_revision": rev,
               "partitions": {lab: {"dir": str(d), "resolved": str(Path(d).resolve()),
                                    "partition_json_sha256_12": sha12(Path(d) / "partition.json")}
                              for lab, d, _p, _t in parts},
               "joint_partition": ({"dir": a.joint_partition, "resolved": str(Path(a.joint_partition).resolve())}
                                   if a.joint_partition else None),
               "stats_cache": str(cache), "stats_id": sid, "stats_identity": sident,
               "maps": {b: m.get("resolved", m["path"]) for b, m in sorted(maps.items())},
               "map_baselines": {b: m.get("baseline_corpus") for b, m in sorted(maps.items())},
               "args": {k: v for k, v in vars(a).items()}, "eos_ids": sorted(eos),
               "watchdog": {"limit_gb": a.watchdog_gb, "hard_gb": a.watchdog_hard_gb, "min_avail_gb": a.min_avail_gb,
                            "swap_growth_gb": a.swap_growth_gb, "compressor_growth_gb": a.compressor_growth_gb,
                            "waiver_note": a.waiver_note or None},
               "wired": wired, "moe": {"layers": layers, "E": E, "F": F, "unit": "expert * F + neuron"},
               "drive": "mean over the segment's tokens of |h| * routing weight (amoeba_firing_capture.py quantity)",
               "ple_asserted": bool(a.assert_ple), "load_s": round(load_s, 1)}
    controls = {lab: {} for lab, *_ in parts}
    tws = {lab: twins_of(S) for lab, S in sets.items()}
    scored = {lab: [n for k, n in zip(S.kinds, S.names) if k in ("box", *NULLS)] for lab, S in sets.items()}
    boxes_of = {lab: [n for k, n in zip(S.kinds, S.names) if k == "box"] for lab, S in sets.items()}

    def control_rows(seqs, segs, twins=True, batch=None):
        """Per document and segment: the scores of every box (and its twins) of every partition."""
        res = {lab: [[] for _ in segs] for lab in sets}
        bs = batch or a.batch
        for i in range(0, len(seqs), bs):
            chunk = seqs[i:i + bs]
            X, R, _ = cap.run(chunk, segs)
            for j in range(len(segs)):
                for bi in range(len(chunk)):
                    x = X[j][bi].reshape(-1)
                    for lab, S in sets.items():
                        sc_ = score_sets(S, x, R[j][bi] > 0, W, F, E, a.ratio_floor,
                                         scored[lab] if twins else boxes_of[lab])
                        res[lab][j].append(add_twin_z(S, sc_, tws[lab]) if twins else sc_)
        return res

    def box_means(docs, lab):
        return {b: {m: _mean([d[b].get(m) for d in docs]) for m in METRICS + tuple(  # twin gaps: None without twins
            f"{g}_{NULLS[k]}_{z}" for g in ("gap", "z") for k in NULLS for z in Z_METRICS)} for b in boxes_of[lab]}

    def twin_mean(docs, lab, kind, b, m):
        return _mean([d[n].get(m) for d in docs for n in tws[lab].get((kind, b), [])])
    def ref_of(docs, lab):
        return {b: {m: ([float(np.mean(v)), float(np.std(v, ddof=1))] if len(v) > 1 else None)
                    for m in Z_METRICS for v in [[d[b][m] for d in docs if d[b].get(m) is not None]]}
                for b in boxes_of[lab]}
    en_ref = {lab: {"raw": {}, "chat_prompt": {}, "chat_user": {}} for lab in sets}
    ebox = (parts[0][2].get("english_box") or {}).get("name")
    if a.english_docs:
        seqs, T = held_docs(host, specs[a.english], a.english_docs, a.seed)
        lens = sorted({int(x) for x in a.english_lengths.split(",") if x and int(x) <= T} | {T})
        log(f"English control: {len(seqs)} x {T} tokens, prefixes {lens}")
        # boxes only (the z_english reference); batch at most 4: 5 segments x [B, nL, W] f32 is the largest transient
        per = control_rows(seqs, [(0, n) for n in lens], twins=False, batch=min(a.batch, 4))
        for lab, S in sets.items():
            topic = [b for b in boxes_of[lab] if b != ebox]
            for j, n in enumerate(lens):
                en_ref[lab]["raw"][n] = {"tokens": n, "boxes": ref_of(per[lab][j], lab)}
            mean = box_means(per[lab][-1], lab)
            med = lambda m: float(np.nanmedian([np.nan if mean[b][m] is None else mean[b][m]   # noqa: E731
                                                for b in topic]))
            controls[lab]["english"] = {
                "docs": len(seqs), "tokens": T, "lengths": lens, "spec": specs[a.english],
                "topic_rel_sum_median": med("rel_sum"), "topic_progress_median": med("progress"),
                "topic_trip_median": med("trip"), "topic_rel_median_median": med("rel_median"),
                "topic_progress_median_by_length": {n: float(np.nanmedian([
                    np.nan if (en_ref[lab]["raw"][n]["boxes"][b]["progress"] is None) else
                    en_ref[lab]["raw"][n]["boxes"][b]["progress"][0] for b in topic])) for n in lens},
                "mean_by_box": {b: {m: _r(v) for m, v in mean[b].items()} for b in boxes_of[lab]},
                "top": {m: sorted(([b, round(mean[b][m], 4)] for b in boxes_of[lab] if mean[b].get(m) is not None),
                                  key=lambda r: (-r[1], r[0]))[:a.top] for m in ("progress", "cos_contrast", "trip")}}
        # English as a chat user message (the prompts' own shape: template tokens, the user text after them)
        clens = sorted({int(x) for x in a.english_chat_lengths.split(",") if x and int(x) <= T})
        nchat = min(a.english_chat_docs, len(seqs))
        if nchat and clens:
            log(f"English chat reference: {nchat} docs x lengths {clens}")
            got = {lab: {L: {"prompt": [], "user": [], "P": [], "U": []} for L in clens} for lab in sets}
            for L in clens:
                for q in seqs[:nchat]:
                    ids, _ = render(host, {"messages": [{"role": "user", "content": host.decode(q[:L])}]},
                                    a.enable_thinking)
                    sp = user_span(host, ids)
                    segs = [(0, len(ids))] + ([sp] if sp else [])
                    X, R, _ = cap.run([ids], segs)
                    for lab, S in sets.items():
                        g = got[lab][L]
                        g["prompt"].append(score_sets(S, X[0][0].reshape(-1), R[0][0] > 0, W, F, E, a.ratio_floor,
                                                      boxes_of[lab]))
                        g["P"].append(len(ids))
                        if sp:
                            g["user"].append(score_sets(S, X[1][0].reshape(-1), R[1][0] > 0, W, F, E,
                                                        a.ratio_floor, boxes_of[lab]))
                            g["U"].append(sp[1] - sp[0])
            for lab in sets:
                for L, g in got[lab].items():
                    en_ref[lab]["chat_prompt"][L] = {"tokens": float(np.mean(g["P"])), "boxes": ref_of(g["prompt"], lab)}
                    if g["user"]:
                        en_ref[lab]["chat_user"][L] = {"tokens": float(np.mean(g["U"])), "boxes": ref_of(g["user"], lab)}
                controls[lab]["english"]["chat_reference"] = {
                    "docs": nchat, "lengths": clens, "prompt_tokens": {L: g["P"][:3] for L, g in got[lab].items()},
                    "topic_progress_median_chat_prompt": {L: float(np.nanmedian([
                        np.nan if v["progress"] is None else v["progress"][0]
                        for b, v in en_ref[lab]["chat_prompt"][L]["boxes"].items() if b != ebox])) for L in clens}}
        for lab in sets:
            controls[lab]["english"]["reference"] = en_ref[lab]
    for c in ctrl:
        seqs, T = held_docs(host, specs[c], a.control_docs, a.seed)
        log(f"positive control {c}: {len(seqs)} x {T} tokens")
        per = control_rows(seqs, [(0, T)])
        for lab, S in sets.items():
            prep = next(p for l_, _d, p, _t in parts if l_ == lab)
            docs = per[lab][0]
            mean = box_means(docs, lab)
            lin = lineage_of(prep, c)
            rm = ("progress", "gap_selected_progress", "gap_random_progress", "cos_contrast", "trip", "rel_sum",
                  "rel_median", "routed")
            controls[lab].setdefault("positive", {})[c] = {
                "docs": len(seqs), "tokens": T, "spec": specs[c], "lineage": sorted(lin),
                "rank_of_mean": {m: dict(zip(("own", "lineage", "of"), rank_of(
                    {b: v for b, v in mean.items() if v.get(m) is not None}, m, c, lin))) for m in rm},
                "rank_per_doc_median": {m: (float(np.median(r_)) if r_ else None) for m in rm for r_ in [[
                    x for x in (rank_of({b: d[b] for b in boxes_of[lab]}, m, c, lin)[0] for d in docs)
                    if x is not None]]},
                "own_progress": mean[c]["progress"], "own_cos_contrast": mean[c]["cos_contrast"],
                "random_twin_progress": twin_mean(docs, lab, "null", c, "progress"),
                "selected_twin_progress": twin_mean(docs, lab, "nullsel", c, "progress"),
                "random_twin_cos_contrast": twin_mean(docs, lab, "null", c, "cos_contrast"),
                "selected_twin_cos_contrast": twin_mean(docs, lab, "nullsel", c, "cos_contrast"),
                "own": {m: _r(v) for m, v in mean[c].items()}}
    t_ctrl = time.time() - t_run
    tmp = out / "trip.jsonl.tmp"
    rows = []
    with tmp.open("w") as fh:
        fh.write(json.dumps({"receipt": receipt}, default=str) + "\n")
        for pi, pr in enumerate(prompts):
            t0 = time.time()
            ids, route = render(host, pr, a.enable_thinking)
            cont, droute = greedy(host, ids, a.gen, eos)
            P = len(ids)
            span = user_span(host, ids)
            named = [("prompt", (0, P))] + ([("user_text", span)] if span else []) + \
                    ([("continuation", (P, P + len(cont)))] if cont else [])
            X, R, hid = cap.run([ids + cont], [x for _, x in named])
            par, div = parity(host, hid, P, cont)
            del hid
            row = {"id": pr["id"], "category": pr.get("category", ""), "prompt_tokens": P, "render": route,
                   "continuation_tokens": len(cont), "decode_route": droute,
                   "continuation_text": host.decode(cont) if cont else "", "parity": par, "first_divergence": div,
                   "user_text_span": list(span) if span else None,
                   "user_text_note": None if span else ("raw prompt row" if "prompt" in pr else
                                                        "no '<|im_start|>user' line found: segment not scored"),
                   "segments": {}}
            if pi == 0:                     # the prompt segment equals the prompt-only forward (causal) on this model
                Xp, _, _ = cap.run([ids], [(0, P)])
                d_, m_ = float(np.abs(Xp[0] - X[0]).max()), float(np.abs(X[0]).max())
                receipt["causal_check"] = {"prompt": pr["id"], "max_abs_diff": d_, "max_abs_drive": m_,
                                           "relative": d_ / max(m_, 1e-12), "ok": d_ <= 1e-2 * max(m_, 1e-12)}
                if not receipt["causal_check"]["ok"]:
                    log(f"WARNING causal check: prompt drive differs by {d_:.3g} (max drive {m_:.3g})")
                del Xp
            for j, (seg, (s0, s1)) in enumerate(named):
                routing = routing_summary(R[j][0], layers)
                rk = {"prompt": "raw" if "prompt" in pr else "chat_prompt", "user_text": "chat_user"}.get(seg, "raw")
                row["segments"][seg] = {
                    lab: segment_result(S, X[j][0].reshape(-1), R[j][0] > 0, W, F, E, layers, a.top, a.ratio_floor,
                                        routing, en_ref[lab][rk] or en_ref[lab]["raw"], s1 - s0, a.min_units,
                                        tws[lab])
                    for lab, S in sets.items()}
            row["wall_s"] = round(time.time() - t0, 1)
            fh.write(json.dumps(row, default=str) + "\n")
            fh.flush()
            rows.append(row)
            first = row["segments"]["prompt"][parts[0][0]]["top"]
            log(f"{pr['id']}: {P}+{len(cont)} tokens, parity {par}; prompt top progress "
                f"{[n for n, *_ in first['progress'][:5]]}; gap_selected {[n for n, *_ in first['gap_selected_progress'][:5]]}")
    os.replace(tmp, out / "trip.jsonl")
    receipt.update({"controls_s": round(t_ctrl, 1), "prompts_s": round(time.time() - t_run - t_ctrl, 1),
                    "total_s": round(time.time() - t_start, 1), "mlx_peak_gb": O.peak_gb(),
                    "prompts": len(rows), "parity_below_0.9": [r["id"] for r in rows
                                                               if r["parity"] is not None and r["parity"] < 0.9]})
    (out / "receipt.json").write_text(json.dumps(receipt, indent=1, default=str))       # before the report
    (out / "controls.json").write_text(json.dumps(controls, indent=1, default=str))
    write_report(out / "report.md", receipt, rows, controls, a.top)
    log(f"done: {len(rows)} prompts, controls {t_ctrl:.0f}s, peak {receipt['mlx_peak_gb']} GB -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
