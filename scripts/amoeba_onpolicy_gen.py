#!/usr/bin/env python3
"""On-policy carve trajectories (Pierre 2026-11-15, "Yes": carving and cone carves use one greedy plus 3 runs at
temperature 0.7 per prompt, for the targets and the checks).

Client of a running mlx2 server (the FULL model on its production / best route: FN its tuned profile, the 27B
its AST route), inside the caller's GPU window. For each row of a carve corpus (the jsonl form the carve
drivers read: {"text", "domain", "source", "split"?}):
  prompt  = the row's text up to the end of its --prompt-words-th word, whitespace kept (a real corpus text
            prefix, never a repeated string);
  modes   = greedy (temperature 0) and --samples runs at --temperature with fixed seeds;
  output  = one row per (prompt, mode): {"text": prompt + continuation, "domain": "gen:" + domain, "source",
            "split" (the corpus row's, so held prompts stay held), "mode", "seed", "temperature", ...}.
Teacher-forcing these rows with KL(full || masked) carves the cone on the model's own trajectories. Writes
gen.jsonl and a manifest (checkpoint, sampling, counts, sha256); cache once per checkpoint and commit.
Rows are generated through /v1/completions (raw text continuation; no chat template, no thinking).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def post(base, body, timeout=600):
    req = urllib.request.Request(base + "/v1/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--checkpoint", required=True, help="<artifact>@<fingerprint> of the served FULL model")
    ap.add_argument("--corpus", action="append", required=True, help="carve corpus jsonl (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt-words", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=None, help="None = the server's default (stated in the manifest)")
    ap.add_argument("--seed0", type=int, default=7000)
    ap.add_argument("--max-rows", type=int, default=0, help="per corpus file, 0 = all")
    ap.add_argument("--held-per-domain", type=int, default=0, help="cap on held rows per domain (0 = all); the carve "
                    "uses at most --n-held of them")
    ap.add_argument("--free-per-domain", type=int, default=0, help="cap on non-held rows per domain (0 = all); the carve "
                    "uses (n_mean + n_fit) / domains plus spare")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--window-tokens", type=int, default=0,
                    help="mid-document prompts (FN calibration c): the prompt is a window of this many tokens ending at a "
                         "seeded random offset inside the document (after any shared prefix), decoded with --tokenizer; "
                         "rows record 'prompt' and 'doc_id' so the carve scores the continuation (+ last prompt tokens) "
                         "and keeps the row in its document's split")
    ap.add_argument("--tokenizer", default="", help="model dir whose tokenizer cuts --window-tokens windows")
    ap.add_argument("--strip-prefix", type=float, default=0.0, help="as the carve loader: drop a shared template prefix")
    ap.add_argument("--window-seed", type=int, default=0)
    ap.add_argument("--code-cut", action="store_true", help="as the carve loader: code domains start at their first code")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)       # resumable: gen.partial.jsonl rows are kept and skipped (review 2026-11-16)
    if (out / "gen.jsonl").exists():
        raise SystemExit(f"{out}/gen.jsonl exists: this generation is complete")
    rows = []
    for path in a.corpus:
        rs = [json.loads(x) for x in Path(path).read_text().split("\n") if x.strip()]
        rs = rs[:a.max_rows] if a.max_rows else rs
        seen = {}
        for r in rs:
            key = (r.get("domain", "text"), r.get("split") == "held")
            cap = (a.held_per_domain if key[1] else a.free_per_domain) or None
            if cap is not None and seen.get(key, 0) >= cap:
                continue
            seen[key] = seen.get(key, 0) + 1
            rows.append((path, r))
    jobs = []
    tok = None
    if a.window_tokens:
        import sys as _sys
        _sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from transformers import AutoTokenizer
        from research.amoeba.carve.corpus import _doc_id, cut_at_first_code, shared_prefixes
        tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
        pre = {}
        if a.strip_prefix:
            by = {}
            for _, r in rows:      # prefixes are detected on the text the windows are cut from (code-cut first)
                t = r["text"]
                if a.code_cut and str(r.get("domain", "")).startswith("code"):
                    t = cut_at_first_code(t)
                by.setdefault(r.get("domain", "text"), []).append(t)
            pre = {d: (shared_prefixes(t, a.strip_prefix) or ("",))[0] for d, t in by.items()}
    for i, (path, r) in enumerate(rows):
        if tok is not None:
            text = r["text"]
            src_text = text                                        # doc_id is the original document's
            if a.code_cut and str(r.get("domain", "")).startswith("code"):
                text = cut_at_first_code(text)
            p0 = pre.get(r.get("domain", "text"), "") if a.strip_prefix else ""
            body = text[len(p0):] if p0 and text.startswith(p0) else text
            ids = tok.encode(body, add_special_tokens=False)
            if len(ids) < a.window_tokens + 16:
                continue
            h = int(hashlib.sha256(f"{a.window_seed}:{_doc_id(src_text)}".encode()).hexdigest()[:12], 16)
            end = a.window_tokens + h % (len(ids) - a.window_tokens + 1)
            prompt = tok.decode(ids[end - a.window_tokens:end])
            r = {**r, "_doc_id": _doc_id(src_text), "_prompt_end_token": end}
        else:
            spans = list(re.finditer(r"\S+", r["text"]))
            if len(spans) < a.prompt_words + 8:
                continue
            prompt = r["text"][:spans[a.prompt_words - 1].end()]   # the original text up to the Nth word (newlines,
                                                                    # indentation kept: code prompts stay code)
        # seeds: by document in window mode (the same prompt gets the same seeds in any run or subset: the lane
        # check compares runs), by row index otherwise (the original rule)
        base = (int(hashlib.sha256(r["_doc_id"].encode()).hexdigest()[:8], 16) % 1_000_000) * 16 if "_doc_id" in r \
            else 1000 * i
        modes = [("greedy", 0.0, None)] + [(f"t{a.temperature}", a.temperature, a.seed0 + base + k)
                                           for k in range(a.samples)]
        for mode, temp, seed in modes:
            jobs.append((path, r, prompt, mode, temp, seed))

    def run(job):
        path, r, prompt, mode, temp, seed = job
        body = {"prompt": prompt, "max_tokens": a.max_tokens, "temperature": temp}
        if seed is not None:
            body["seed"] = seed
        if a.top_p is not None and temp > 0:
            body["top_p"] = a.top_p
        t0 = time.time()
        if fails["streak"] >= 20:                 # the server is down or hung: fail fast, the rerun resumes
            with lock:
                fails["n"] += 1
            return None
        for attempt in range(4):                  # retries: one HTTP timeout or 5xx must not lose hours of rows
            try:
                d = post(a.base_url, body)
                break
            except Exception as e:                # noqa: BLE001
                if attempt == 3 or fails["streak"] >= 20:
                    with lock:
                        fails["n"] += 1
                        fails["streak"] += 1
                    print(f"request failed ({e}); row left for the resume", flush=True)
                    return None
                time.sleep(5 * (attempt + 1))
        fails["streak"] = 0
        cont = d["choices"][0].get("text") or ""
        row = {"text": prompt + cont, "domain": "gen:" + r.get("domain", "text"), "source": r.get("source", path),
               "prompt_key": hashlib.sha256(prompt.encode()).hexdigest()[:16],
               "mode": mode, "temperature": temp, "seed": seed, "prompt_words": a.prompt_words,
               "corpus": Path(path).name, "completion_tokens": (d.get("usage") or {}).get("completion_tokens"),
               "seconds": round(time.time() - t0, 2)}
        if "split" in r:
            row["split"] = r["split"]
        if "_doc_id" in r:                     # windowed prompt: the carve loader's loss mask and document split
            row.update({"prompt": prompt, "doc_id": r["_doc_id"], "prompt_end_token": r["_prompt_end_token"],
                        "window_tokens": a.window_tokens})
        return row
    t0 = time.time()
    part = out / "gen.partial.jsonl"
    rkey = lambda r: (r.get("prompt_key"), r["mode"], r.get("seed"))       # noqa: E731
    done = {}
    if part.exists():
        for x in part.read_text().split("\n"):
            if x.strip():
                try:
                    r = json.loads(x)
                except json.JSONDecodeError:      # a row cut by a crash mid-write
                    continue
                done[rkey(r)] = r
    import threading
    lock = threading.Lock()
    fails = {"n": 0, "streak": 0}           # a dead server: after 20 failures in a row, later jobs fail fast

    def jkey(job):
        _, r, prompt, mode, temp, seed = job
        return (hashlib.sha256(prompt.encode()).hexdigest()[:16], mode, seed)
    keys = {jkey(j) for j in jobs}
    done = {k: v for k, v in done.items() if k in keys}       # only rows of the current job set
    todo = [j for j in jobs if jkey(j) not in done]
    print(f"jobs {len(jobs)}, already done {len(jobs) - len(todo)}", flush=True)
    with part.open("a") as fh:
        def run_keep(job):
            row = run(job)
            if row is not None:
                with lock:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()
            return row
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            new = [r for r in ex.map(run_keep, todo) if r is not None]
    res = list(done.values()) + new
    if fails["n"]:     # never write gen.jsonl with rows missing: exit nonzero, the next run resumes from the partial
        raise SystemExit(f"{fails['n']} requests failed; {len(res)}/{len(jobs)} rows in {part}; rerun to resume")
    p = out / "gen.jsonl"
    pending = out / ".gen.jsonl.tmp"
    pending.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in res))
    toks = sum(x["completion_tokens"] or 0 for x in res)
    man = {"schema": "mlx2.amoeba-onpolicy-gen.v1", "checkpoint": a.checkpoint, "base_url_route": a.base_url,
           "corpora": a.corpus, "prompt_words": a.prompt_words, "max_tokens": a.max_tokens,
           "window_tokens": a.window_tokens, "window_seed": a.window_seed, "strip_prefix": a.strip_prefix,
           "code_cut": a.code_cut,
           "modes": {"greedy": 1, f"t{a.temperature}": a.samples}, "temperature": a.temperature,
           "top_p": a.top_p if a.top_p is not None else "server default", "top_k": "server default",
           "seeds": (f"seed0 + 16*(sha256(doc_id) mod 1e6) + k (seed0={a.seed0})" if a.window_tokens
                     else f"seed0 + 1000*row + k (seed0={a.seed0})"), "rows": len(res), "completion_tokens": toks,
           "wall_s": round(time.time() - t0, 1), "tok_s_aggregate": round(toks / max(1e-9, time.time() - t0), 1),
           "sha256": hashlib.sha256(pending.read_bytes()).hexdigest(),
           "note": "teacher-force with KL(full || masked); held prompts keep split 'held' (bisect and evals only)"}
    manifest_tmp = out / ".manifest.json.tmp"
    manifest_tmp.write_text(json.dumps(man, indent=1))
    manifest_tmp.replace(out / "manifest.json")
    # gen.jsonl is the completion marker. Publish it last; partial rows survive a crash
    # before either rename and the next invocation rebuilds both without new requests.
    pending.replace(p)
    print(json.dumps({k: man[k] for k in ("rows", "completion_tokens", "wall_s", "tok_s_aggregate")}))


if __name__ == "__main__":
    main()
