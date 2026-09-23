#!/usr/bin/env python3
"""Probe cross-layer top-k index reuse at decode (experiment; never qualifies).

The first full-attention layer (or each layer listed in ``--sources``) picks
a token set from its exact attention: a forced recent window plus the top
``--budget`` positions. Later full-attention layers attend only to that set.
Pre-registered plan and gates: docs/experiments/FA-TOPK-REUSE-2026-09-23.md.

Subcommands:

* ``ceiling`` (no model load): reads ``config.json`` and safetensors headers
  and prints the bandwidth ceiling on decode speedup per context. Gate A's
  analytical half.
* ``run``: loads a model and runs phases against the stock dense route:

  - ``cost``: decode ms/token at each context vs a short base context. The
    growth is what attention costs, and bounds any speedup (Gate A).
  - ``shadow``: dense decode with metrics recorded per FA layer and step:
    mass recall of the reused selection (per variant), of the layer's own
    top-k (upper bound) and of the window alone (lower bound), relative
    output error, and the recall matrix from every earlier layer (Gate B).
    Output is bit-identical to the stock route.
  - ``substitute``: target layers attend over the reused selection. KL,
    top-1 and needle retrieval against the dense arm (Gate C).

  ``--cpu-tiny`` runs a random-weight Qwen3.8-architecture model on CPU to
  validate the harness; its report is marked non-evidence and cannot pass a
  gate. GPU mode refuses to run without ``--i-own-the-gpu``; run it under the
  lab lock wrapper::

      cpg_job.py run --label fa-topk-reuse --out <log> --lock -- <python> \\
        scripts/probe_fa_topk_reuse.py run --model <path> --i-own-the-gpu --out <json>

  ``--dry-run`` prints the plan and exits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import statistics
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REPORT_SCHEMA = "mlx2.fa-topk-reuse-probe.v1"

# Pre-registered 2026-09-23, before any GPU data. Changing a number after data
# exists needs a dated note in the experiment doc saying why.
GATES = {
    "A_cost": {"context": 65536, "min_ceiling": 1.15},
    "B_shadow": {
        "contexts": [32768, 65536],
        "median_recall_min": 0.90,
        "p10_recall_min": 0.75,
        "answer_median_recall_min": 0.90,
    },
    "C_substitute": {
        "contexts": [32768, 65536],
        "kl_mean_max": 0.02,
        "kl_p99_max": 0.2,
        "top1_min": 0.97,
        "needle_losses_max": 0,
    },
}

NEEDLE_NAMES = (
    "Orion", "Vega", "Lyra", "Draco", "Cygnus", "Perseus", "Aquila", "Hydra",
    "Carina", "Pavo", "Fornax", "Tucana",
)


# ------------------------------------------------------------------ ceiling


def _safetensors_bytes(model_dir: Path):
    for path in sorted(model_dir.glob("*.safetensors")):
        with path.open("rb") as fh:
            header = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
        for name, meta in header.items():
            if name != "__metadata__":
                start, stop = meta["data_offsets"]
                yield name, stop - start


def traffic_from_artifact(model_dir: Path, kv_bytes: int = 2) -> dict:
    """Per-token decode traffic of a local artifact, from headers only.

    Weight bytes count every tensor read per token: dense tensors, the LM
    head, and routed experts scaled by top-k / experts. Embeddings (one row
    read), vision/audio towers and MTP heads are excluded.
    """
    from mlx2.runtime.topk_index_reuse import DecodeTraffic

    config = json.loads((model_dir / "config.json").read_text())
    text = config.get("text_config", config)
    layer_types = text.get("layer_types")
    if layer_types is None and text.get("full_attention_interval"):
        interval = int(text["full_attention_interval"])
        layer_types = [
            "full_attention" if (i + 1) % interval == 0 else "linear_attention"
            for i in range(int(text["num_hidden_layers"]))
        ]
    if not layer_types:
        raise SystemExit(f"{model_dir}: cannot derive full-attention layers")
    n_fa = layer_types.count("full_attention")
    kv_row = 2 * int(text["num_key_value_heads"]) * int(text["head_dim"]) * kv_bytes

    classes = {"dense": 0, "experts": 0, "lm_head": 0, "excluded": 0}
    for name, size in _safetensors_bytes(model_dir):
        low = name.lower()
        if any(tag in low for tag in ("vision", "visual", "audio", "mtp", "embed_tokens")):
            classes["excluded"] += size
        elif "lm_head" in low:
            classes["lm_head"] += size
        elif ".experts." in low or "switch_mlp" in low:
            classes["experts"] += size
        else:
            classes["dense"] += size
    experts = int(text.get("num_experts") or 0)
    active = classes["experts"] * int(text["num_experts_per_tok"]) / experts if experts else 0
    weight = classes["dense"] + classes["lm_head"] + active
    if config.get("tie_word_embeddings", text.get("tie_word_embeddings")) and not classes["lm_head"]:
        raise SystemExit(f"{model_dir}: tied embeddings; LM-head traffic not modelled")

    # Bounded state read (and written) per token.
    window = int(text.get("sliding_window") or 0)
    fixed = layer_types.count("sliding_attention") * window * kv_row
    linear = layer_types.count("linear_attention")
    if linear:
        state = (int(text["linear_num_value_heads"]) * int(text["linear_key_head_dim"])
                 * int(text["linear_value_head_dim"]) * 4)
        fixed += linear * state * 2
    traffic = DecodeTraffic(weight_bytes=weight, kv_row_bytes=[kv_row] * n_fa, fixed_bytes=fixed)
    return {"traffic": traffic, "n_fa": n_fa, "classes": classes, "active_expert_bytes": active}


def cmd_ceiling(args) -> int:
    import mlx.core as mx

    mx.set_default_device(mx.cpu)
    contexts = [int(x) for x in args.contexts.split(",")]
    out = []
    for path in args.artifacts:
        info = traffic_from_artifact(Path(path).expanduser(), args.kv_bytes)
        traffic, n_fa = info["traffic"], info["n_fa"]
        schedules = {"first-only": (0,), "every-other": tuple(range(0, n_fa, 2))}
        rows = []
        for context in contexts:
            row = {"context": context,
                   "weights_gb": traffic.weight_bytes / 1e9,
                   "fa_kv_gb": context * sum(traffic.kv_row_bytes) / 1e9}
            for label, sources in schedules.items():
                row[label] = traffic.ceiling(context, sources=sources, budget=args.budget,
                                             window=args.window, score_pass=args.score_pass)
            rows.append(row)
        out.append({"artifact": str(path), "n_fa": n_fa, "fixed_gb": traffic.fixed_bytes / 1e9,
                    "rows": rows})
        print(f"\n{Path(path).name}  FA layers={n_fa}  weights/token={traffic.weight_bytes/1e9:.2f} GB"
              f"  bounded state={traffic.fixed_bytes/1e9:.3f} GB")
        print(f"{'context':>8} {'FA KV GB':>9} {'first-only':>11} {'every-other':>12}")
        for row in rows:
            print(f"{row['context']:>8} {row['fa_kv_gb']:>9.2f} {row['first-only']:>10.2f}x"
                  f" {row['every-other']:>11.2f}x")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"schema": REPORT_SCHEMA + ".ceiling", "gates": GATES,
                                        "params": vars(args) | {"out": str(args.out),
                                                                "artifacts": args.artifacts},
                                        "results": out}, indent=2, default=str) + "\n")
    return 0


# ---------------------------------------------------------------- workloads


def tiny_model():
    import mlx.core as mx

    from mlx2.runtime.models.qwen3_5 import TextModelArgs
    from mlx2.runtime.models.qwen38_27b import TextModel

    args = TextModelArgs(
        model_type="qwen3_5", hidden_size=64, intermediate_size=64,
        num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=2,
        head_dim=32, vocab_size=128, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=3, full_attention_interval=2,
        mtp_num_hidden_layers=0, partial_rotary_factor=0.5,
        rope_parameters=None, max_position_embeddings=8192,
    )
    mx.random.seed(7)
    model = TextModel(args)
    model.eval()
    for layer in model.model.layers:  # sharpen attention so sparsity is visible
        attention = getattr(layer, "self_attn", None)
        if attention is not None:
            attention.v_proj.weight = attention.v_proj.weight * 8.0
            attention.o_proj.weight = attention.o_proj.weight * 8.0
    mx.eval(model.parameters())
    return model


def corpus_text(path):
    if path is not None:
        return path.read_text(encoding="utf-8")
    return "\n\n".join(p.read_text(encoding="utf-8") for p in sorted((ROOT / "docs").glob("*.md")))


def build_case(encode, filler, context, n_needles, score_tokens, rng):
    """Prefix of exactly ``context`` tokens with needles, plus a scored segment.

    The scored segment asks for each needle and gives the answer, so the
    decode steps whose *target* is an answer token are the ones that must
    retrieve from far back. Returns (prefix, scored, tags, needles) where
    ``tags[i]`` labels the target token ``scored[i]``.
    """
    names = rng.sample(NEEDLE_NAMES, n_needles) if n_needles else []
    needles = {name: f"{rng.randrange(10**5, 10**6)}" for name in names}
    facts = [list(encode(f" The access code for {n} is {c}. ")) for n, c in needles.items()]
    body = list(filler[: context - sum(len(f) for f in facts)])
    if len(body) + sum(len(f) for f in facts) != context:
        raise ValueError("filler too short for context")
    depths = [int(len(body) * (0.05 + 0.9 * i / max(1, n_needles - 1))) for i in range(n_needles)]
    for depth, fact in sorted(zip(depths, facts), reverse=True):
        body[depth:depth] = fact
    scored, tags = [], []
    order = list(needles.items())
    rng.shuffle(order)
    for name, code in order:
        question = list(encode(f" What is the access code for {name}? The access code for {name} is"))
        answer = list(encode(f" {code}."))
        scored += question
        tags += ["question"] * len(question)
        scored += answer
        tags += [f"answer:{name}"] * len(answer)
    tail = list(filler[context : context + max(0, score_tokens - len(scored))])
    scored += tail
    tags += ["text"] * len(tail)
    return body, scored[:score_tokens], tags[:score_tokens], needles


# ------------------------------------------------------------------- phases


def _prefill(model, cache, ids, step):
    import mlx.core as mx

    logits = None
    for start in range(0, len(ids), step):
        logits = model(mx.array([ids[start : start + step]], dtype=mx.uint32), cache=cache)
        mx.eval(logits)
    return logits[0, -1]


def _teacher_forced(model, cache, prefix_logits, scored, probe=None):
    """Logits predicting ``scored[i]`` for every i (row 0 comes from prefill)."""
    import mlx.core as mx

    rows = [prefix_logits]
    if probe is not None:
        probe.armed = True
    for token in scored[:-1]:
        out = model(mx.array([[token]], dtype=mx.uint32), cache=cache)[0, -1]
        mx.eval(out)
        rows.append(out)
        if probe is not None:
            probe.flush()
    if probe is not None:
        probe.armed = False
    return rows


def _decode_ms(model, cache, first_token, steps, warmup=2):
    import mlx.core as mx

    token = mx.array([[first_token]], dtype=mx.uint32)
    times = []
    for i in range(warmup + steps):
        start = time.perf_counter()
        out = model(token, cache=cache)
        token = mx.argmax(out[:, -1:], axis=-1).astype(mx.uint32)
        mx.eval(token)
        if i >= warmup:
            times.append((time.perf_counter() - start) * 1e3)
    return statistics.median(times)


def _quantiles(values):
    ordered = sorted(values)
    if not ordered:
        return {}
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))]  # noqa: E731
    return {"n": len(ordered), "median": pick(0.5), "p10": pick(0.1), "min": ordered[0]}


def summarize_shadow(rows, tags):
    """Aggregate probe rows; steps are tagged by the token they predict."""
    targets = [r for r in rows if r["role"] == "target"]
    metrics = sorted({k for r in targets for k in r if "/" in k or k.endswith("recall")
                      or k == "relerr"})
    out = {"all": {m: _quantiles([r[m] for r in targets if m in r]) for m in metrics}}
    # Probe step s is the decode call that predicts scored[s + 1].
    answer_rows = [r for r in targets if r["step"] + 1 < len(tags)
                   and tags[r["step"] + 1].startswith("answer:")]
    out["answer"] = {m: _quantiles([r[m] for r in answer_rows if m in r]) for m in metrics}
    layers = sorted({r["layer"] for r in targets})
    out["per_layer"] = {
        layer: {m: _quantiles([r[m] for r in targets if r["layer"] == layer and m in r])["median"]
                for m in metrics if any(m in r for r in targets if r["layer"] == layer)}
        for layer in layers
    }
    return out


def compare_arms(dense_rows, sub_rows, scored, tags):
    import mlx.core as mx

    from mlx2.runtime.kv_quant_fidelity import summarize, token_metrics

    exact = mx.stack(dense_rows)
    quant = mx.stack(sub_rows)
    target = mx.array(scored, dtype=mx.uint32)
    m = token_metrics(exact, quant, target)
    mx.eval(m)
    per_token = {k: [float(x) for x in v.tolist()] for k, v in m.items()}
    summary = summarize(per_token)
    dense_ok = (mx.argmax(exact, axis=-1) == target).tolist()
    sub_ok = (mx.argmax(quant, axis=-1) == target).tolist()
    needles = {}
    for i, tag in enumerate(tags):
        if tag.startswith("answer:"):
            name = tag.split(":", 1)[1]
            slot = needles.setdefault(name, {"dense": True, "substitute": True})
            slot["dense"] &= bool(dense_ok[i])
            slot["substitute"] &= bool(sub_ok[i])
    summary["needles"] = needles
    summary["needle_losses"] = sum(1 for n in needles.values() if n["dense"] and not n["substitute"])
    return summary


def evaluate_gates(report) -> dict:
    """Experiment gates (not qualification). A non-evidence report never passes."""
    verdict = {"evidence": report["evidence"], "gates": {}}
    by_ctx = {c["context"]: c for c in report["contexts"]}
    g = GATES["A_cost"]
    ceiling = by_ctx.get(g["context"], {}).get("cost", {}).get("realized_ceiling")
    verdict["gates"]["A_cost"] = None if ceiling is None else ceiling >= g["min_ceiling"]
    g = GATES["B_shadow"]
    ok = None
    for ctx in g["contexts"]:
        shadow = by_ctx.get(ctx, {}).get("shadow")
        if shadow is None:
            ok = None
            break
        primary = shadow["all"].get("recall/token/shared", {})
        answer = shadow["answer"].get("recall/token/shared", {})
        this = (primary.get("median", 0) >= g["median_recall_min"]
                and primary.get("p10", 0) >= g["p10_recall_min"]
                and answer.get("median", 0) >= g["answer_median_recall_min"])
        ok = this if ok is None else ok and this
    verdict["gates"]["B_shadow"] = ok
    g = GATES["C_substitute"]
    ok = None
    for ctx in g["contexts"]:
        sub = by_ctx.get(ctx, {}).get("substitute")
        if sub is None:
            ok = None
            break
        this = (sub["kl_mean"] <= g["kl_mean_max"] and sub["kl_p99"] <= g["kl_p99_max"]
                and sub["top1_agreement"] >= g["top1_min"]
                and sub["needle_losses"] <= g["needle_losses_max"])
        ok = this if ok is None else ok and this
    verdict["gates"]["C_substitute"] = ok
    verdict["passed"] = bool(report["evidence"]) and all(
        v is True for v in verdict["gates"].values())
    return verdict


# ---------------------------------------------------------------------- run


def plan(args):
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
            if k != "func"}


def cmd_run(args) -> int:
    if not args.cpu_tiny and not args.model:
        raise SystemExit("--model is required unless --cpu-tiny")
    gpu = not args.cpu_tiny
    if args.dry_run:
        print(json.dumps({"plan": plan(args), "gates": GATES}, indent=2))
        return 0
    if gpu and not args.i_own_the_gpu:
        raise SystemExit("refusing Metal execution without --i-own-the-gpu")

    import mlx.core as mx

    from mlx2.runtime.models.cache import make_prompt_cache
    from mlx2.runtime.topk_index_reuse import ReuseProbe, Variant, install_probe

    rng = random.Random(args.seed)
    contexts = [int(x) for x in args.contexts.split(",")]
    phases = set(args.phases.split(","))
    variants = tuple(Variant(*v.split("/")) for v in args.variants.split(","))
    if args.cpu_tiny:
        mx.set_default_device(mx.cpu)
        model = tiny_model()
        encode = None
        filler = [random.Random(args.seed + 1).randrange(1, 120)
                  for _ in range(max(contexts) + args.score_tokens + 1)]
        identity = {"model": "cpu-tiny-qwen38-architecture", "corpus_sha256": None}
    else:
        from mlx2.adapters.registry import resolve_adapter

        adapter = resolve_adapter(args.model)(args.model)
        model = adapter.model
        tokenizer = adapter.tokenizer

        def encode(text):
            try:
                return tokenizer.encode(text, add_special_tokens=False)
            except TypeError:
                return tokenizer.encode(text)

        text = corpus_text(args.corpus)
        filler = list(encode(text))
        while len(filler) < max(contexts) + args.score_tokens + 1:
            filler = filler + filler
        identity = {"model": str(args.model),
                    "fingerprint": adapter.identity.get("fingerprint"),
                    "corpus_sha256": hashlib.sha256(text.encode()).hexdigest()}

    def probed_cache(mode):
        return install_probe(make_prompt_cache(model), lambda n: ReuseProbe(
            mode=mode, n_layers=n, sources=[int(s) for s in args.sources.split(",")],
            budget=args.budget, window=args.window, variants=variants,
            block_size=args.block_size, record_matrix=not args.no_matrix))

    report = {
        "schema": REPORT_SCHEMA,
        "evidence": gpu,
        "qualification": False,
        "route_selected": False,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "git_revision": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                       capture_output=True, text=True).stdout.strip(),
        "platform": platform.platform(),
        "mlx_version": getattr(mx, "__version__", None),
        "identity": identity,
        "plan": plan(args),
        "gates": GATES,
        "contexts": [],
    }

    base_ms = None
    if "cost" in phases:
        cache = make_prompt_cache(model)
        _prefill(model, cache, filler[: args.base_context], args.prefill_step)
        base_ms = _decode_ms(model, cache, filler[args.base_context], args.decode_tokens)
        report["base_context"] = {"context": args.base_context, "decode_ms": base_ms}

    for context in contexts:
        n_needles = 0 if args.cpu_tiny else args.needles
        prefix, scored, tags, needles = (
            (filler[:context], filler[context : context + args.score_tokens],
             ["text"] * args.score_tokens, {})
            if encode is None else
            build_case(encode, filler, context, n_needles, args.score_tokens, rng))
        entry = {"context": context, "scored_tokens": len(scored), "needles": sorted(needles)}
        started = time.perf_counter()

        cache, probe = probed_cache("shadow")
        first = _prefill(model, cache, prefix, args.prefill_step)
        dense_rows = _teacher_forced(model, cache, first, scored,
                                     probe if "shadow" in phases else None)
        if "shadow" in phases:
            expected = probe.n_layers * (len(scored) - 1)
            if probe.counts["shadowed"] != expected:
                raise SystemExit(f"shadow fired {probe.counts['shadowed']} times, expected "
                                 f"{expected}; refusing to report")
            entry["shadow"] = summarize_shadow(probe.rows, tags)
            entry["probe_counts"] = dict(probe.counts)
        if "cost" in phases:
            ms = _decode_ms(model, cache, scored[-1], args.decode_tokens)
            attn = max(0.0, ms - base_ms)
            n = probe.n_layers
            kept = min(context, args.budget + args.window)
            n_src = len(set(int(s) for s in args.sources.split(",")))
            reused_attn = attn * (n_src * (1 + args.score_pass) + (n - n_src) * kept / context) / n
            entry["cost"] = {"decode_ms": ms, "attention_ms_est": attn,
                             "attention_share": attn / ms if ms else None,
                             "realized_ceiling": ms / (ms - attn + reused_attn)}
        del cache

        if "substitute" in phases:
            cache, probe = probed_cache("substitute")
            first = _prefill(model, cache, prefix, args.prefill_step)
            sub_rows = _teacher_forced(model, cache, first, scored, probe)
            expected = len(probe.targets) * (len(scored) - 1)
            if probe.counts["substituted"] != expected:
                raise SystemExit(f"substitution fired {probe.counts['substituted']} times, "
                                 f"expected {expected}; refusing to report")
            entry["substitute"] = compare_arms(dense_rows, sub_rows, scored, tags)
            entry["substitute"]["probe_counts"] = dict(probe.counts)
            del cache
        entry["wall_s"] = time.perf_counter() - started
        report["contexts"].append(entry)
        brief = {"context": context, "wall_s": round(entry["wall_s"], 1)}
        if "shadow" in entry:
            brief["recall_median"] = entry["shadow"]["all"].get("recall/token/shared", {}).get("median")
        if "substitute" in entry:
            brief["kl_mean"] = entry["substitute"]["kl_mean"]
            brief["top1"] = entry["substitute"]["top1_agreement"]
        if "cost" in entry:
            brief["realized_ceiling"] = entry["cost"]["realized_ceiling"]
        print(json.dumps(brief), flush=True)

    report["verdict"] = evaluate_gates(report)
    text = json.dumps(report, indent=2, default=str)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(json.dumps(report["verdict"], indent=2))
    return 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("ceiling", help="bandwidth ceiling from artifact headers (no model load)")
    c.add_argument("artifacts", nargs="+")
    c.add_argument("--contexts", default="8192,32768,65536,131072,262144")
    c.add_argument("--budget", type=int, default=1024)
    c.add_argument("--window", type=int, default=128)
    c.add_argument("--score-pass", type=float, default=0.5)
    c.add_argument("--kv-bytes", type=int, default=2, help="bytes per cached element")
    c.add_argument("--out", type=Path)
    c.set_defaults(func=cmd_ceiling)

    r = sub.add_parser("run", help="probe a model (cost / shadow / substitute)")
    r.add_argument("--model", help="model artifact path (GPU mode)")
    r.add_argument("--cpu-tiny", action="store_true", help="CPU harness self-check")
    r.add_argument("--i-own-the-gpu", action="store_true")
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--phases", default="cost,shadow,substitute")
    r.add_argument("--contexts", default=None,
                   help="comma list (default GPU 16384,32768,65536,131072; CPU 64,160)")
    r.add_argument("--base-context", type=int, default=None,
                   help="short context for the cost baseline (default GPU 1024, CPU 16)")
    r.add_argument("--score-tokens", type=int, default=None,
                   help="teacher-forced positions per context (default GPU 256, CPU 24)")
    r.add_argument("--decode-tokens", type=int, default=None,
                   help="timed decode steps for the cost phase (default GPU 32, CPU 4)")
    r.add_argument("--budget", type=int, default=None, help="default GPU 1024, CPU 16")
    r.add_argument("--window", type=int, default=None, help="default GPU 128, CPU 8")
    r.add_argument("--block-size", type=int, default=None, help="default GPU 64, CPU 4")
    r.add_argument("--sources", default="0", help="FA ordinals that select (must include 0)")
    r.add_argument("--variants", default="token/shared,token/kv_head,block/shared",
                   help="first one is the primary (substituted) variant")
    r.add_argument("--score-pass", type=float, default=0.5)
    r.add_argument("--needles", type=int, default=9)
    r.add_argument("--no-matrix", action="store_true", help="skip the all-pairs recall matrix")
    r.add_argument("--prefill-step", type=int, default=2048)
    r.add_argument("--corpus", type=Path)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", type=Path)
    r.set_defaults(func=cmd_run)
    args = p.parse_args(argv)
    if args.command == "run":
        gpu = not args.cpu_tiny
        defaults = {
            "contexts": ("16384,32768,65536,131072", "64,160"),
            "base_context": (1024, 16),
            "score_tokens": (256, 24),
            "decode_tokens": (32, 4),
            "budget": (1024, 16),
            "window": (128, 8),
            "block_size": (64, 4),
        }
        for key, (gpu_default, cpu_default) in defaults.items():
            if getattr(args, key) is None:
                setattr(args, key, gpu_default if gpu else cpu_default)
    return args


def main(argv=None):
    args = parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
