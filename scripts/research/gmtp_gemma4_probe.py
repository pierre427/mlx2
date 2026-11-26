#!/usr/bin/env python3
"""Feasibility probe: Gemma 4 assistant (G-MTP) drafter vs plain decode, B1.

Runs the pinned mlx-vlm generate loop (``mlx_vlm.generate.ar.generate_step``),
not mlx2 serving, so its numbers are a pre-port feasibility signal, never an
mlx2 route result.  Greedy only.  For every prompt it decodes once with no
drafter (the reference) and once per forced draft block size, and records:
token identity with the reference, decode tok/s (first yielded token to
last), and the drafter's per-round accepted counts.

Prompts are the vendored GSM8K test items (zero-shot chat, the same
instruction as ``scripts/spec_acceptance_gsm8k.py``) plus fixed prose and code
prompts.  The output directory must not exist or must be empty.
"""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import spec_acceptance_gsm8k as gsm  # noqa: E402

PROSE = [
    "Write a short essay (about 250 words) on why lighthouses were built where "
    "they were, and what replaced them.",
    "Explain to a new hire how a city water-treatment plant turns river water "
    "into drinking water, step by step.",
]
CODE = [
    "Write a Python function that parses an ISO-8601 duration string such as "
    "'P3DT4H12M' into total seconds, with docstring and a few doctests.",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def log(msg: str) -> None:
    print(f"[probe {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def arm_watchdog(seconds: float) -> None:
    """Dump every thread's Python stack and exit if no re-arm within *seconds*.

    A hung Metal eval cannot be interrupted from Python, so the watchdog is a
    faulthandler timer: it writes the stall frame to stderr and hard-exits,
    letting the GPU-lock wrappers release and restore the service.
    """
    faulthandler.cancel_dump_traceback_later()
    faulthandler.dump_traceback_later(seconds, repeat=False, exit=True)


def unify_mlx_vlm_streams(mx) -> list[str]:
    """Run every mlx-vlm forward on one stream (the default stream).

    The pinned mlx-vlm creates two thread-local generation streams
    (``generate.common`` and ``speculative.common``) and builds MTP draft
    tokens on the default stream, so a drafted round crosses three streams.
    The 2026-11-20 r3 window hung in ``_mtp_acceptance_walk`` ->
    ``mx.async_eval(verify.hidden)`` (eval_impl condition wait), the
    cross-stream stall mlx2 fixed with one stream per generation thread
    (c46d82799).  mlx-vlm's own server avoids it the same way
    (``server/generation.py``: ``generation_stream = mx.default_stream``).
    Rebinding covers modules that imported the name directly.
    """
    stream = mx.default_stream(mx.default_device())
    rebound = []
    for name, module in list(sys.modules.items()):
        if (name == "mlx_vlm" or name.startswith("mlx_vlm.")) and module is not None \
                and hasattr(module, "generation_stream"):
            module.generation_stream = stream
            rebound.append(name)
    return sorted(rebound)


def run_one(generate_step, mx, model, tokens, *, max_tokens, eos, drafter=None,
            block_size=None, timeout_s=300.0):
    arm_watchdog(timeout_s)
    if drafter is not None:
        before = (getattr(drafter, "speculative_total_rounds", 0),
                  getattr(drafter, "speculative_total_accepted", 0.0),
                  getattr(drafter, "speculative_total_drafted", 0))
        start_lens = len(getattr(drafter, "accept_lens", []) or [])
    ids = mx.array([tokens])
    out = []
    mx.synchronize()
    t0 = time.perf_counter()
    t_first = None
    kwargs = dict(max_tokens=max_tokens, temperature=0.0)
    if drafter is not None:
        kwargs.update(draft_model=drafter, draft_kind="mtp",
                      draft_block_size=block_size)
    for tok, _ in generate_step(ids, model, None, None, **kwargs):
        tok = int(tok.item() if hasattr(tok, "item") else tok)
        if t_first is None:
            t_first = time.perf_counter()
        out.append(tok)
        if tok in eos or len(out) >= max_tokens:
            break
    mx.synchronize()
    t_end = time.perf_counter()
    faulthandler.cancel_dump_traceback_later()
    row = {
        "tokens": out,
        "n_tokens": len(out),
        "ttft_s": round(t_first - t0, 4) if t_first else None,
        "decode_s": round(t_end - t_first, 4) if t_first else None,
        "decode_tps": (round((len(out) - 1) / (t_end - t_first), 3)
                       if t_first and len(out) > 1 and t_end > t_first else None),
    }
    if drafter is not None:
        lens = list((getattr(drafter, "accept_lens", []) or [])[start_lens:])
        rounds = getattr(drafter, "speculative_total_rounds", 0) - before[0]
        accepted = getattr(drafter, "speculative_total_accepted", 0.0) - before[1]
        drafted = getattr(drafter, "speculative_total_drafted", 0) - before[2]
        row.update({
            "rounds": rounds,
            "accepted_drafts": accepted,
            "drafted": drafted,
            # Tokens committed per target verify = accepted drafts + 1 bonus.
            "mean_accept_length": round(accepted / rounds + 1, 4) if rounds else None,
            "accept_lens": [float(x) for x in lens],
        })
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--target-fingerprint", required=True)
    parser.add_argument("--drafter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gsm8k", type=int, default=8)
    parser.add_argument("--block-sizes", default="3,4,5,7")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--decode-timeout-s", type=float, default=300.0,
                        help="watchdog per decode: dump stacks and exit on stall")
    parser.add_argument("--load-timeout-s", type=float, default=900.0)
    args = parser.parse_args()

    out_dir = args.output
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"refusing nonempty evidence directory {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx
    import mlx_vlm
    from mlx_vlm.generate.ar import generate_step
    from mlx_vlm.speculative.drafters import load_drafter
    from mlx_vlm.utils import load
    from mlx2.adapters import mlx_vlm_pin

    pin = mlx_vlm_pin.require_pinned_mlx_vlm()
    single_stream = unify_mlx_vlm_streams(mx)
    log(f"single stream: rebound generation_stream in {single_stream}")
    block_sizes = [int(x) for x in args.block_sizes.split(",")]

    source_rev, files = gsm.dataset_files()
    items = gsm.load_split(files["test"])[: args.gsm8k]
    prompts = ([{"kind": "gsm8k", "id": i, "text": it["question"] + gsm.CHAT_INSTRUCTION,
                 "answer": gsm.reference_answer(it["answer"])} for i, it in enumerate(items)]
               + [{"kind": "prose", "id": i, "text": t} for i, t in enumerate(PROSE)]
               + [{"kind": "code", "id": i, "text": t} for i, t in enumerate(CODE)])

    t_load = time.perf_counter()
    arm_watchdog(args.load_timeout_s)
    log("loading target")
    model, processor = load(str(args.target))
    log("target loaded; loading drafter")
    drafter, kind = load_drafter(str(args.drafter), kind="mtp")
    faulthandler.cancel_dump_traceback_later()
    log(f"drafter loaded kind={kind}")
    # Force the requested depth: mlx-vlm otherwise caps the block at the
    # drafter's configured depth until the prefix hit rate is high.
    drafter.prefer_requested_block_size = True
    load_s = time.perf_counter() - t_load
    tokenizer = getattr(processor, "tokenizer", processor)
    eos = {1, 106, 50}
    gen_eos = getattr(model.config, "eos_token_id", None)
    if isinstance(gen_eos, int):
        eos.add(gen_eos)
    elif isinstance(gen_eos, list):
        eos.update(gen_eos)

    def encode(text):
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False,
            add_generation_prompt=True)
        return tokenizer.encode(rendered, add_special_tokens=False)

    # Warm-up (not recorded): one short plain and one drafted decode.
    warm = encode("Say hello in five words.")
    to = args.decode_timeout_s
    log("warm-up plain")
    w = run_one(generate_step, mx, model, warm, max_tokens=16, eos=eos, timeout_s=to)
    log(f"warm-up plain done {w['n_tokens']} tok")
    log(f"warm-up drafted bs={block_sizes[0]}")
    w = run_one(generate_step, mx, model, warm, max_tokens=16, eos=eos,
                drafter=drafter, block_size=block_sizes[0], timeout_s=to)
    log(f"warm-up drafted done {w['n_tokens']} tok")

    rows = []
    for prompt in prompts:
        tokens = encode(prompt["text"])
        log(f"{prompt['kind']}#{prompt['id']} reference")
        ref = run_one(generate_step, mx, model, tokens,
                      max_tokens=args.max_tokens, eos=eos, timeout_s=to)
        cell = {**prompt, "prompt_tokens": len(tokens), "reference": ref, "arms": {}}
        ref_text = tokenizer.decode(ref["tokens"])
        cell["reference_text"] = ref_text
        if prompt["kind"] == "gsm8k":
            cell["reference_correct"] = gsm.extract_answer(ref_text) == prompt["answer"]
        for bs in block_sizes:
            log(f"{prompt['kind']}#{prompt['id']} bs={bs}")
            arm = run_one(generate_step, mx, model, tokens,
                          max_tokens=args.max_tokens, eos=eos,
                          drafter=drafter, block_size=bs, timeout_s=to)
            arm["identical_to_reference"] = arm["tokens"] == ref["tokens"]
            if not arm["identical_to_reference"]:
                n = next((i for i, (a, b) in enumerate(zip(arm["tokens"], ref["tokens"]))
                          if a != b), min(len(arm["tokens"]), len(ref["tokens"])))
                arm["first_divergence"] = n
                arm["text"] = tokenizer.decode(arm["tokens"])
            if ref["decode_tps"] and arm["decode_tps"]:
                arm["speedup"] = round(arm["decode_tps"] / ref["decode_tps"], 4)
            cell["arms"][str(bs)] = arm
        rows.append(cell)
        print(json.dumps({"kind": prompt["kind"], "id": prompt["id"],
                          "ref_tps": ref["decode_tps"],
                          **{f"bs{k}": (v.get("speedup"), v.get("mean_accept_length"),
                                        v["identical_to_reference"])
                             for k, v in cell["arms"].items()}}), flush=True)

    def summarize(kind=None):
        sel = [r for r in rows if kind is None or r["kind"] == kind]
        ref_tps = [r["reference"]["decode_tps"] for r in sel if r["reference"]["decode_tps"]]
        out = {"n": len(sel), "ref_decode_tps_median": gsm.median(ref_tps)}
        for bs in block_sizes:
            arms = [r["arms"][str(bs)] for r in sel]
            out[f"bs{bs}"] = {
                "speedup_median": gsm.median([a["speedup"] for a in arms if a.get("speedup")]),
                "decode_tps_median": gsm.median([a["decode_tps"] for a in arms if a["decode_tps"]]),
                "mean_accept_length": (round(sum(a["accepted_drafts"] for a in arms)
                                             / max(1, sum(a["rounds"] for a in arms)) + 1, 4)),
                "identical": sum(a["identical_to_reference"] for a in arms),
            }
        return out

    receipt = {
        "schema": "mlx2.gmtp-gemma4-probe.v1",
        "scope": "feasibility probe on the pinned mlx-vlm generate loop; not an mlx2 route result, not qualification",
        "mlx2_commit": git("rev-parse", "HEAD"),
        "mlx2_dirty": bool(git("status", "--porcelain")),
        "target": {"path": str(args.target),
                   "checkpoint": f"{args.target.name}@sha256:{args.target_fingerprint}",
                   "config_sha256": sha256(args.target / "config.json"),
                   "index_sha256": sha256(args.target / "model.safetensors.index.json")},
        "drafter": {"path": str(args.drafter), "kind": kind,
                    "checkpoint": f"{args.drafter.name}@sha256:{sha256(args.drafter / 'model.safetensors')}",
                    "config_sha256": sha256(args.drafter / "config.json")},
        "mlx_vlm": {"file": mlx_vlm.__file__, "pin": pin,
                    "expected_revision": mlx_vlm_pin.MLX_VLM_REVISION},
        "mlx_version": getattr(mx, "__version__", None),
        "host": platform.node(), "machine": platform.machine(),
        "batch": 1, "sampling": "greedy", "max_tokens": args.max_tokens,
        "block_sizes": block_sizes,
        "stream_patch": {"single_default_stream": True, "rebound_modules": single_stream},
        "block_size_meaning": "tokens per round incl. the bonus; drafts = block_size - 1",
        "gsm8k_source_revision": source_rev,
        "load_s": round(load_s, 2),
        "summary": {"all": summarize(), "gsm8k": summarize("gsm8k"),
                    "prose": summarize("prose"), "code": summarize("code")},
        "rows": rows,
    }
    (out_dir / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
