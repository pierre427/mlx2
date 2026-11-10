"""Try a learned QSA sidecar on a dense checkpoint: does it engage, is it
correct, is it faster?

Exploratory GPU probe, not qualification.  One artifact, one owned window:

* quality -- teacher-forced NLL per token, dense vs QSA, same tokens; reports
  PPL, top-1 agreement and |dNLL| over the positions where QSA is sparse;
* needles -- one realistic-haystack prompt per context with eight buried
  codes, greedy answers from both arms checked by an exact-code oracle;
* speed -- cold prefill and greedy decode at depth, dense vs QSA.

Usage: ``probe_qsa_sidecar.py owned SPEC`` (takes the CPG lease and paired GPU
locks, then re-enters itself as ``unit``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


def atomic(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True, default=str))
    os.replace(tmp, path)


def source() -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts"],
                           cwd=ROOT, capture_output=True, text=True).stdout.strip()
    files = {}
    for rel in ("src/mlx2/runtime/models/qsa_sidecar.py", "scripts/probe_qsa_sidecar.py"):
        files[rel] = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
    return {"commit": head, "dirty": bool(dirty), "files": files}


def host() -> dict:
    vm = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    swapouts = re.search(r"Swapouts:\s+(\d+)", vm)
    return {"time": time.time(), "swapouts": int(swapouts.group(1)) if swapouts else None}


# ------------------------------------------------------------------ helpers


def _encode(tokenizer, text):
    try:
        return tokenizer.encode(text, add_special_tokens=False)
    except TypeError:  # wrapper without the HF keyword
        return tokenizer.encode(text)


def text_model(model):
    return getattr(model, "language_model", model)


def new_cache(text, tokens):
    """A fresh cache whose full-attention KV storage is allocated once for
    ``tokens`` positions.  The default 256-step growth re-concatenates the
    whole K/V history on every prefill chunk (and every 256 decode tokens):
    ~16 GiB per full-attention layer over a 128K prefill.  Values and results
    are unchanged (tests/test_qsa_sidecar_memcopy.py)."""
    from mlx2.runtime.models.cache import KVCache

    cache = text.make_cache()
    for entry in cache:
        if isinstance(entry, KVCache):
            entry.reserve(tokens)
    return cache


def prefill(mx, text, cache, ids, step, *, on_hidden=None):
    """Chunked prefill; ``on_hidden(start, hidden)`` sees each chunk's rows."""
    hidden = None
    for start in range(0, len(ids), step):
        chunk = mx.array([ids[start:start + step]], dtype=mx.uint32)
        hidden = text.model(chunk, cache)
        if on_hidden is not None:
            on_hidden(start, hidden)
        else:
            mx.eval(hidden)
        mx.clear_cache()
    return hidden


def nll_rows(mx, text, hidden, targets, rows=512):
    """Per-row NLL and argmax for hidden [1, L, H] against ``targets`` [L]."""
    nll, top, top2, lp2 = [], [], [], []
    for start in range(0, hidden.shape[1], rows):
        logits = text.logits(hidden[:, start:start + rows]).astype(mx.float32)[0]
        tgt = targets[start:start + rows]
        lse = mx.logsumexp(logits, axis=-1)
        pick = mx.take_along_axis(logits, tgt[:, None], axis=-1)[:, 0]
        nll.append(lse - pick)
        ids = mx.argpartition(-logits, kth=1, axis=-1)[:, :2]
        vals = mx.take_along_axis(logits, ids, axis=-1)
        order = mx.argsort(-vals, axis=-1)
        ids = mx.take_along_axis(ids, order, axis=-1)
        top.append(ids[:, 0])
        top2.append(ids)
        lp2.append(mx.take_along_axis(vals, order, axis=-1) - lse[:, None])
    out = (mx.concatenate(nll), mx.concatenate(top), mx.concatenate(top2),
           mx.concatenate(lp2))
    mx.eval(*out)
    return out


def corpus_stream(encode, name, needed, seed):
    if name == "repo":
        from realistic_haystack import artifact_corpus

        text = artifact_corpus(ROOT)
    elif name == "prose":
        # One continuous public-domain novel (Project Gutenberg #2701), so
        # long-range dependencies are real rather than concatenated documents.
        path = Path(os.environ.get("QSA_PROSE", Path.home() / "datasets/gutenberg/pg2701.txt"))
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        start = text.find("CHAPTER 1. Loomings")
        text = text[start if start > 0 else 0:]
    else:
        raise ValueError(name)
    ids = list(encode(text))
    if len(ids) < needed:
        raise ValueError(f"corpus {name} has {len(ids)} tokens, need {needed}")
    return ids[:needed], hashlib.sha256(text.encode()).hexdigest()


# ------------------------------------------------------------------- phases


def set_mode(sidecar, mode):
    """``<mode>`` or ``<mode>_fused`` (fused Metal selection, qsa_select.py)."""
    if mode.startswith("quest:"):  # quest:<window>:<sink>:<budget>
        (window, sink, budget) = (int(v) for v in mode.split(":")[1:])
        sidecar.mode = "quest"
        sidecar.fused_select = False
        sidecar.quest = {"window": window, "sink": sink, "budget": budget, "page": 32}
        return
    fused = mode.endswith("_fused")
    sidecar.mode = mode[: -len("_fused")] if fused else mode
    sidecar.fused_select = fused


def arm(sidecar, mode):
    sidecar.reset()
    set_mode(sidecar, mode)


def flip_summary(np, dense, other, ceiling=0.5):
    """Teacher-forced top-1 flips, classified at their own position.

    Each flip position is independent (teacher forcing), so the near-tie test
    applies per position: both arms' top-2 sets are the same unordered pair
    and both selected-vs-alternate margins are within ``ceiling`` nats.
    """
    flip = dense["top"] != other["top"]
    n = int(flip.sum())
    if n == 0:
        return {"count": 0}
    d2, o2 = dense["top2"][flip], other["top2"][flip]
    dm = dense["lp2"][flip, 0] - dense["lp2"][flip, 1]
    om = other["lp2"][flip, 0] - other["lp2"][flip, 1]
    pair = (np.sort(d2, axis=1) == np.sort(o2, axis=1)).all(axis=1)
    near = pair & (dm <= ceiling) & (om <= ceiling)
    in_dense_top2 = o2[:, 0] == d2[:, 1]
    return {
        "count": n,
        "near_tie": int(near.sum()),
        "near_tie_share": round(float(near.mean()), 4),
        "other_pick_is_dense_second": round(float(in_dense_top2.mean()), 4),
        "dense_margin_median": round(float(np.median(dm)), 4),
        "dense_margin_p90": round(float(np.quantile(dm, 0.9)), 4),
        "confident_flips_dense_margin_gt_2nat": int((dm > 2.0).sum()),
    }


def quality(mx, np, adapter, sidecar, spec, report):
    from mlx2.runtime.models import qsa_sidecar as qs

    text = text_model(adapter.model)
    encode = lambda s: _encode(adapter.tokenizer, s)  # noqa: E731
    out = []
    for corpus in spec["quality"]["corpora"]:
        for context in spec["quality"]["contexts"]:
            ids, sha = corpus_stream(encode, corpus, context + 1, spec["seed"])
            targets = mx.array(ids[1:], dtype=mx.int32)
            arms = {}
            for mode in spec["quality"]["arms"]:
                cap = spec["quality"].get("arm_max_context", {}).get(mode)
                if cap is not None and context > cap:
                    continue
                arm(sidecar, mode)
                qs.status(reset=True)
                cache = new_cache(text, len(ids) - 1)
                nlls, tops, top2s, lp2s = [], [], [], []

                def collect(start, hidden):
                    n, t, t2, l2 = nll_rows(mx, text, hidden, targets[start:start + hidden.shape[1]])
                    nlls.append(np.array(n))
                    tops.append(np.array(t))
                    top2s.append(np.array(t2))
                    lp2s.append(np.array(l2))

                started = time.perf_counter()
                prefill(mx, text, cache, ids[:-1], spec["prefill_step"], on_hidden=collect)
                arms[mode] = {
                    "nll": np.concatenate(nlls), "top": np.concatenate(tops),
                    "top2": np.concatenate(top2s), "lp2": np.concatenate(lp2s),
                    "seconds": time.perf_counter() - started, "status": qs.status(reset=True),
                }
                del cache
                mx.clear_cache()
            dense = arms["dense"]
            row = {"corpus": corpus, "corpus_sha256": sha, "context": context, "arms": {}}
            sparse_from = sidecar.budget + 3  # first position with > topk complete blocks
            for mode, data in arms.items():
                entry = {"seconds": round(data["seconds"], 2), "status": data["status"]}
                for label, start in (("all", 0), ("sparse", sparse_from), ("deep", context // 2)):
                    n = data["nll"][start:]
                    entry[f"ppl_{label}"] = float(np.exp(n.mean()))
                    if mode != "dense":
                        d = dense["nll"][start:]
                        entry[f"ppl_ratio_{label}"] = float(np.exp(n.mean() - d.mean()))
                        entry[f"top1_agree_{label}"] = float(
                            (data["top"][start:] == dense["top"][start:]).mean())
                        entry[f"mean_abs_dnll_{label}"] = float(np.abs(n - d).mean())
                if mode != "dense":
                    entry["flips"] = flip_summary(np, dense, data)
                row["arms"][mode] = entry
            np.savez_compressed(
                Path(spec["out"]) / f"tokens-{corpus}-{context}.npz",
                ids=np.array(ids, np.int32),
                **{f"{m}_{k}": d[k] for m, d in arms.items()
                   for k in ("nll", "top2", "lp2")},
            )
            out.append(row)
            report["quality"] = out
            atomic(Path(spec["out"]) / "report.partial.json", report)
            print(json.dumps({"quality": row}), flush=True)
    return out


CODE_WORDS = ["amber", "cobalt", "delta", "ember", "falcon", "garnet", "harbor", "indigo",
              "juniper", "kestrel", "lumen", "meridian"]


def classify_divergence(a, b, ceiling=0.5, oracle=None):
    """First divergence of two greedy runs from their top-5 windows.

    Numerics use the shared AGENTS.md rule (``mlx2.near_tie``: two-way,
    exact-tie and multi-way clauses). ``near_tie_equivalent`` additionally
    needs both continuations to pass a functional oracle: ``oracle`` is
    ``(a_passed, b_passed)``. Free continuation of arbitrary text has none,
    so without one a numeric near tie is ``near_tie_no_oracle`` (not a pass).
    A token missing from the other arm's window fails the row when that
    window proves it lies more than ``ceiling`` below the top.
    """
    from mlx2.near_tie import near_tie_equivalent

    for i, (x, y) in enumerate(zip(a["tokens"], b["tokens"])):
        if x != y:
            break
    else:
        return {"diverged": False}
    wa, wb = a["top5"][i], b["top5"][i]  # [(id, logprob)] best first
    la, lb = {int(t): float(v) for t, v in wa}, {int(t): float(v) for t, v in wb}
    ma, mb = wa[0][1] - wa[1][1], wb[0][1] - wb[1][1]
    out = {"diverged": True, "index": i, "a_token": x, "b_token": y,
           "a_margin": round(ma, 4), "b_margin": round(mb, 4),
           "a_window": wa, "b_window": wb}
    if y not in la or x not in lb:
        proven_far = (y not in la and wa[0][1] - wa[-1][1] > ceiling) or (
            x not in lb and wb[0][1] - wb[-1][1] > ceiling)
        out["verdict"] = "high_margin" if proven_far else "unclassified"
        return out
    if not near_tie_equivalent(x, la, y, lb, ceiling=ceiling):
        out["verdict"] = "high_margin"
    elif oracle is None:
        out["verdict"] = "near_tie_no_oracle"
    elif all(oracle):
        out["verdict"] = "near_tie_equivalent"
    else:
        out["verdict"] = "near_tie_oracle_failed"
    return out


def continue_gen(mx, np, adapter, sidecar, spec, report):
    """Greedy free continuation of real long context, dense vs QSA."""
    text = text_model(adapter.model)
    tok = adapter.tokenizer
    encode = lambda s: _encode(tok, s)  # noqa: E731
    cfg = spec["continue_gen"]
    out = []
    for corpus in cfg["corpora"]:
        for context in cfg["contexts"]:
            ids, sha = corpus_stream(encode, corpus, context, spec["seed"])
            row = {"corpus": corpus, "context": context, "corpus_sha256": sha,
                   "context_tail": tok.decode(ids[-200:]), "arms": {}}
            runs = {}
            for mode in cfg["arms"]:
                arm(sidecar, mode)
                cache = text.make_cache()
                hidden = prefill(mx, text, cache, ids, spec["prefill_step"])
                logits = text.logits(hidden[:, -1:])[0, -1].astype(mx.float32)
                tokens, top5 = [], []
                for _ in range(cfg["max_tokens"]):
                    lp = logits - mx.logsumexp(logits)
                    best = mx.argpartition(-lp, kth=4)[:5]
                    vals = lp[best]
                    order = mx.argsort(-vals)
                    best, vals = best[order].tolist(), vals[order].tolist()
                    tokens.append(best[0])
                    top5.append([(int(t), round(float(v), 5)) for t, v in zip(best, vals)])
                    h = text.model(mx.array([[best[0]]], dtype=mx.uint32), cache)
                    logits = text.logits(h)[0, -1].astype(mx.float32)
                runs[mode] = {"tokens": tokens, "top5": top5}
                row["arms"][mode] = {"text": tok.decode(tokens), "tokens": tokens}
                del cache, hidden
                mx.clear_cache()
            modes = list(runs)
            row["first_divergence"] = classify_divergence(runs[modes[0]], runs[modes[1]])
            row["windows"] = {m: r["top5"] for m, r in runs.items()}
            out.append(row)
            report["continue_gen"] = out
            atomic(Path(spec["out"]) / "report.partial.json", report)
            print(json.dumps({"continue_gen": {"corpus": corpus, "context": context,
                                               "first_divergence": {k: v for k, v in row["first_divergence"].items() if "window" not in k}}}), flush=True)
    return out


def needles(mx, np, adapter, sidecar, spec, report):
    from realistic_haystack import artifact_corpus

    text = text_model(adapter.model)
    tok = adapter.tokenizer
    rng = random.Random(spec["seed"])
    corpus_ids = _encode(tok, artifact_corpus(ROOT, exclude=("NEEDLE-",)))
    out = []
    for context in spec["needles"]["contexts"]:
        names = rng.sample(CODE_WORDS, 8)
        codes = {n: f"{rng.choice('BCDFGHKMNPRSTVWXZ')}{rng.randrange(1000, 9999)}-"
                    f"{rng.choice('BCDFGHKMNPRSTVWXZ')}{rng.choice('BCDFGHKMNPRSTVWXZ')}"
                 for n in names}
        budget = context - 400
        depths = sorted(rng.uniform(0.03, 0.97) for _ in names)
        pieces, cursor = [], 0
        for depth, name in zip(depths, names):
            cut = int(depth * budget)
            pieces.append(tok.decode(corpus_ids[cursor:cut]))
            pieces.append(f"\n\nNEEDLE-RECORD: the access code for project {name} is {codes[name]}.\n\n")
            cursor = cut
        pieces.append(tok.decode(corpus_ids[cursor:budget]))
        question = ("Above, several lines start with NEEDLE-RECORD. Give the access code for "
                    "each project as JSON object mapping project name to code. Projects: "
                    + ", ".join(names) + ". Reply with the JSON only.")
        request = {"messages": [{"role": "user", "content": "".join(pieces) + "\n\n" + question}],
                   "enable_thinking": False}
        ids = adapter.prompt_tokens(request)
        row = {"context": context, "prompt_tokens": len(ids), "codes": codes, "arms": {}}
        for mode in spec["needles"]["arms"]:
            arm(sidecar, mode)
            cache = new_cache(text, len(ids) + spec["needles"]["max_tokens"])
            hidden = prefill(mx, text, cache, ids, spec["prefill_step"])
            token = int(mx.argmax(text.logits(hidden[:, -1:])[0, -1]))
            generated = []
            eos = set(getattr(tok, "eos_token_ids", None) or [tok.eos_token_id])
            for _ in range(spec["needles"]["max_tokens"]):
                if token in eos:
                    break
                generated.append(token)
                h = text.model(mx.array([[token]], dtype=mx.uint32), cache)
                token = int(mx.argmax(text.logits(h)[0, -1]))
            answer = tok.decode(generated)
            norm = answer.replace("‐", "-").replace("‑", "-").replace("–", "-").replace("—", "-")
            hits = {n: (c in norm) for n, c in codes.items()}
            row["arms"][mode] = {"answer": answer, "hits": hits, "score": sum(hits.values())}
            del cache
            mx.clear_cache()
        out.append(row)
        report["needles"] = out
        atomic(Path(spec["out"]) / "report.partial.json", report)
        print(json.dumps({"needles": {k: v for k, v in row.items() if k != "codes"}}), flush=True)
    return out


def speed(mx, np, adapter, sidecar, spec, report):
    from realistic_haystack import artifact_corpus

    text = text_model(adapter.model)
    tok = adapter.tokenizer
    stream = _encode(tok, artifact_corpus(ROOT))
    while len(stream) < max(spec["speed"]["depths"]) + 1:
        stream = stream + stream  # depth filler only; quality is measured elsewhere
    out = []
    for depth in spec["speed"]["depths"]:
        ids = stream[:depth]
        row = {"depth": depth, "arms": {}}
        for prefill_mode, decode_modes in spec["speed"]["arms"].items():
            arm(sidecar, prefill_mode)
            cache = new_cache(text, depth + len(decode_modes) * (4 + spec["speed"]["decode_tokens"]))
            mx.synchronize()
            started = time.perf_counter()
            hidden = prefill(mx, text, cache, ids, spec["prefill_step"])
            token = mx.argmax(text.logits(hidden[:, -1:])[0, -1])
            mx.eval(token)
            prefill_s = time.perf_counter() - started
            entry = {"prefill_s": round(prefill_s, 3), "prefill_tok_s": round(depth / prefill_s, 1),
                     "decode": {}}
            for mode in decode_modes:
                set_mode(sidecar, mode)
                steps = spec["speed"]["decode_tokens"]
                for _ in range(4):  # warm the decode geometry
                    token = mx.argmax(text.logits(text.model(token.reshape(1, 1), cache))[0, -1])
                mx.eval(token)
                started = time.perf_counter()
                for _ in range(steps):
                    token = mx.argmax(text.logits(text.model(token.reshape(1, 1), cache))[0, -1])
                    mx.async_eval(token)
                mx.eval(token)
                entry["decode"][mode] = round(steps / (time.perf_counter() - started), 2)
            row["arms"][prefill_mode] = entry
            del cache, hidden
            mx.clear_cache()
        out.append(row)
        report["speed"] = out
        atomic(Path(spec["out"]) / "report.partial.json", report)
        print(json.dumps({"speed": row}), flush=True)
    return out


def unit_tests(mx, np, adapter, sidecar, spec, report):
    """Metal correctness tests inside the owned window; a failure stops the run."""
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "tests/test_qsa_sidecar.py", "tests/test_qsa_select_metal.py"],
        cwd=ROOT, capture_output=True, text=True,
        env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
    )
    report["unit_tests"] = {"returncode": run.returncode, "tail": run.stdout[-3000:] + run.stderr[-2000:]}
    print(json.dumps({"unit_tests": report["unit_tests"]}), flush=True)
    if run.returncode != 0:
        raise RuntimeError("Metal unit tests failed")


def select_bench(mx, np, adapter, sidecar, spec, report):
    """Selection only, unfused vs fused, on synthetic fp32 q/blocks (median ms)."""
    from mlx2.runtime.models.qsa_select import fused_select
    from mlx2.runtime.models.qwen4_exp import _compact_qsa_block_ids

    topk = sidecar.block_topk
    rng = np.random.default_rng(spec["seed"])
    out = []

    def unfused(q, kb, q_pos):
        n_blocks = kb.shape[0]
        scores = None
        for h in range(q.shape[0]):
            part = mx.maximum(q[h] @ kb.T, 0.0)
            scores = part if scores is None else scores + part
        starts = mx.arange(n_blocks, dtype=mx.int32) * 4
        valid = (starts[None, :] + 3) <= q_pos[:, None]
        scores = mx.where(valid, scores, -mx.inf)
        ids = mx.argpartition(-scores, kth=topk - 1, axis=-1)[:, :topk].astype(mx.int32)
        picked = mx.take_along_axis(valid, ids, axis=-1)
        return _compact_qsa_block_ids(ids[None], picked[None], n_blocks=n_blocks)

    def fused(q, kb, q_pos):
        return fused_select(q, kb, q_pos, topk)

    for n_blocks in spec["select_bench"]["n_blocks"]:
        for rows in spec["select_bench"]["rows"]:
            q = mx.array(rng.standard_normal((4, rows, 128)).astype(np.float32))
            kb = mx.array(rng.standard_normal((n_blocks, 128)).astype(np.float32))
            q_pos = mx.arange(n_blocks * 4 - rows, n_blocks * 4, dtype=mx.int32)
            times = {"unfused": [], "fused": []}
            for name, fn in (("unfused", unfused), ("fused", fused)):
                mx.eval(fn(q, kb, q_pos))  # warm
            for rep in range(spec["select_bench"]["repeats"]):
                order = (("unfused", unfused), ("fused", fused))
                for name, fn in (order if rep % 2 == 0 else order[::-1]):
                    mx.synchronize()
                    started = time.perf_counter()
                    mx.eval(fn(q, kb, q_pos))
                    times[name].append((time.perf_counter() - started) * 1e3)
            row = {"n_blocks": n_blocks, "rows": rows,
                   "unfused_ms": round(float(np.median(times["unfused"])), 3),
                   "fused_ms": round(float(np.median(times["fused"])), 3)}
            row["speedup"] = round(row["unfused_ms"] / row["fused_ms"], 2)
            out.append(row)
            report["select_bench"] = out
            print(json.dumps({"select_bench": row}), flush=True)
            del q, kb
            mx.clear_cache()
    return out


def unit(spec) -> int:
    from mlx2.desktop_campaign_worker import check_ownership

    out = Path(spec["out"])
    report = {"checkpoint": spec["checkpoint"], "indexers": spec["indexers"],
              "source": source(), "ownership": check_ownership(spec),
              "scope": "exploratory sidecar probe; not qualification, single-repeat speed",
              "qualified": False, "before": host()}
    try:
        import mlx.core as mx
        import numpy as np

        mx.set_default_device(mx.gpu)
        mx.set_cache_limit(4 << 30)
        from mlx2.adapters.registry import inspect_model, resolve_adapter
        from mlx2.runtime.models import qsa_sidecar as qs

        resolution = inspect_model(spec["model"])
        if resolution.artifact_fingerprint != spec["fingerprint"]:
            raise ValueError("artifact changed since inventory")
        idx_sha = hashlib.sha256(Path(spec["indexers"]["file"]).read_bytes()).hexdigest()
        if idx_sha != spec["indexers"]["sha256"]:
            raise ValueError("indexer file changed")
        adapter = resolve_adapter(spec["model"])(spec["model"])
        sidecar = qs.QSASidecar(spec["indexers"]["file"], budget=spec["budget"], mode="dense")
        report["installed_layers"] = qs.install(adapter.model, sidecar)
        # Indexer file, budget and the NAX kernel law the sparse arms run.
        # (mode is per phase, so it is left out of this snapshot.)
        report["sidecar"] = {k: v for k, v in sidecar.identity().items() if k != "mode"}
        report["device"] = mx.device_info()
        for phase in spec["phases"]:
            globals()[phase](mx, np, adapter, sidecar, spec, report)
        report["passed"] = True
    except BaseException as error:  # noqa: BLE001 - evidence first
        report["passed"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    report["after"] = host()
    atomic(out / "report.json", report)
    print(json.dumps({"passed": report["passed"], "error": report.get("error")}), flush=True)
    return 0 if report["passed"] else 1


def main() -> int:
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("owned", "unit"))
    parser.add_argument("spec", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    if args.mode == "unit":
        return unit(spec)
    out = Path(spec["out"])
    if out.exists() and any(out.iterdir()):
        raise ValueError("refusing nonempty evidence directory")
    out.mkdir(parents=True, exist_ok=True)
    from mlx2.desktop_campaign_lease import DesktopCPGLease

    with DesktopCPGLease(out, spec["job_id"], checkpoint=spec["checkpoint"],
                         source_revision=source()["commit"]) as lease:
        command = [sys.executable, str(ROOT / "scripts/run_with_gpu_locks.py"),
                   "--session", lease.owner["cpg_session"], "--label", spec["job_id"],
                   "--desktop-cpg-lease", str(lease.path),
                   "--receipt", str(out / "paired-locks.json"), "--",
                   sys.executable, "-u", str(Path(__file__)), "unit", str(args.spec.resolve())]
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        child = subprocess.Popen(command, cwd=ROOT, env=env, start_new_session=True)
        try:
            while child.poll() is None:
                lease.check()
                time.sleep(1)
            lease.success = child.returncode == 0
            return child.returncode
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=60)


if __name__ == "__main__":
    raise SystemExit(main())
