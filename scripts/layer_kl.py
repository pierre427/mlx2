#!/usr/bin/env python3
"""Per-layer KL sensitivity tool (mlx2.layer_kl): texts, reference, scan.

Original mlx2 script.  The caller owns the GPU for ``reference`` and ``scan``
(CPG lease + paired locks; pass ``--i-own-gpu``).  Every output names the
checkpoint(s) and the mlx2 commit.  Refuses nonempty output directories.

    layer_kl.py texts --model PATH --out texts.json [--count 8 --tokens 1024]
    layer_kl.py reference --model PATH --texts texts.json --out REFDIR
        [--family NAME]  (stream a too-big model layer-major through a sweep family)
        [--execution-policy JSON] [--skip 32] [--dtype float32]
    layer_kl.py scan --model PATH --reference REFDIR --layers 3-41 --groups mlp.experts
        --bits 2 --group-size 64 --out OUTDIR [--source DIR] [--cumulative]
        [--execution-policy JSON] [--splits dev]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))


def commit() -> dict:
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain",
                                          "--untracked-files=no"], text=True).strip())
    return {"commit": head, "tracked_dirty": dirty}


WIRING = {}


def load_adapter(model: Path, policy_path, *, wire: bool = True):
    from mlx2.adapters.registry import inspect_model

    resolution = inspect_model(model)
    policy = json.loads(Path(policy_path).read_text()) if policy_path else None
    adapter = resolution.adapter_type(str(model), execution_policy=policy)
    if wire:
        # Wire the weights as serving does (weight_residency): without it an
        # ~84 GB model is pageable, and calibration Hessians on top make
        # macOS compress and swap the weights themselves.
        from mlx2.runtime.weight_residency import wire_serving_weights

        WIRING.update(wire_serving_weights())
        print(f"wired weights: {WIRING}", flush=True)
    identity = getattr(adapter, "identity", None) or {}
    fingerprint = identity.get("fingerprint") or resolution.artifact_fingerprint
    return adapter, f"{model.resolve().name}@sha256:{fingerprint}", policy


def corpus_texts(tokenizer, corpus: Path, *, count: int, tokens: int, split: str, seed: int = 0):
    """Calibration texts from a JSONL corpus (``{"text", "topics", "redistribute"}``).

    Rows are taken round-robin across top-level topics (the part before the
    first ``.``), shuffled within a topic with ``seed``, and concatenated
    (blank-line separated) until each text reaches ``tokens``.  Only rows
    marked ``redistribute: yes`` are used.  Returns the texts and per-topic
    row counts (no source names).
    """
    import random
    from collections import defaultdict

    by_topic = defaultdict(list)
    with open(corpus) as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("redistribute") != "yes" or not row.get("text"):
                continue
            topic = (row.get("topics") or ["misc"])[0].split(".")[0]
            by_topic[topic].append(row["text"])
    rng = random.Random(seed)
    for rows in by_topic.values():
        rng.shuffle(rows)
    topics = sorted(by_topic)
    cursor = {t: 0 for t in topics}
    used = {t: 0 for t in topics}
    texts, turn = [], 0
    for i in range(count):
        ids, parts = [], []
        while len(ids) < tokens:
            topic = topics[turn % len(topics)]
            turn += 1
            if cursor[topic] >= len(by_topic[topic]):
                if all(cursor[t] >= len(by_topic[t]) for t in topics):
                    raise SystemExit("corpus exhausted before the calibration set was full")
                continue
            parts.append(by_topic[topic][cursor[topic]])
            cursor[topic] += 1
            used[topic] += 1
            ids = tokenizer.encode("\n\n".join(parts), add_special_tokens=False)
        texts.append({"id": f"{split}-{i}", "split": split, "source": f"corpus {corpus.name}",
                      "ids": list(map(int, ids[:tokens]))})
    return texts, used


def cmd_texts(a) -> int:
    """A fixed, varied text set: repository text slices (never repeated strings), alternating dev/held;
    or, with ``--corpus``, calibration texts stratified across a corpus's topics."""
    from realistic_haystack import haystack

    from transformers import AutoTokenizer

    # Only the tokenizer: building texts never loads the model's weights.
    tokenizer = AutoTokenizer.from_pretrained(a.model, local_files_only=True, trust_remote_code=False)
    checkpoint = a.model.resolve().name
    if a.corpus:
        texts, used = corpus_texts(tokenizer, a.corpus, count=a.count, tokens=a.tokens, split=a.split,
                                   seed=a.seed)
        if a.out.exists():
            raise SystemExit(f"{a.out} exists")
        a.out.write_text(json.dumps(texts) + "\n")
        print(json.dumps({"out": str(a.out), "texts": len(texts), "tokens": sum(len(t["ids"]) for t in texts),
                          "rows_per_topic": used}))
        return 0
    try:
        texts = []
        offset = a.offset
        for i in range(a.count):
            chars = int(a.tokens * 4)
            for _ in range(4):
                ids = tokenizer.encode(haystack(chars, offset=offset), add_special_tokens=False)
                if abs(len(ids) - a.tokens) <= max(8, a.tokens // 50):
                    break
                chars = max(256, int(chars * a.tokens / max(1, len(ids))))
            ids = ids[: a.tokens]
            texts.append({"id": f"text-{i}", "split": "dev" if i % 2 == 0 else "held",
                          "source": f"realistic_haystack offset {offset}", "ids": list(map(int, ids))})
            offset += chars + 4096
    finally:
        pass
    if a.out.exists():
        raise SystemExit(f"{a.out} exists")
    a.out.write_text(json.dumps(texts) + "\n")
    print(json.dumps({"out": str(a.out), "texts": len(texts), "tokenizer_of": checkpoint}))
    return 0


def cmd_reference(a) -> int:
    from mlx2 import layer_kl as K

    texts = json.loads(Path(a.texts).read_text())
    receipt = {"started_at": datetime.now(UTC).isoformat(), "mlx2": commit()}
    if a.family:
        from research.amoeba.sweep.family import load_family

        family = load_family(a.family, a.model)
        try:
            checkpoint = family.checkpoint
            logits_for = K.reference_from_family(family, texts)
            meta = K.write_reference(a.out, texts, logits_for, skip=a.skip, checkpoint=checkpoint,
                                     dtype=a.dtype, source={"kind": "sweep_family", "family": a.family,
                                                            "fingerprint": family.fingerprint, **receipt})
        finally:
            close = getattr(family, "close", None)
            if callable(close):
                close()
    else:
        adapter, checkpoint, policy = load_adapter(a.model, a.execution_policy, wire=not a.no_wire)
        try:
            meta = K.write_reference(a.out, texts, K.reference_from_model(adapter.model), skip=a.skip,
                                     checkpoint=checkpoint, dtype=a.dtype,
                                     source={"kind": "adapter_forward", "execution_policy": policy, **receipt})
        finally:
            adapter.close()
    print(json.dumps({"out": str(a.out), "checkpoint": meta["checkpoint"], "positions": meta["positions"]}))
    return 0


def _rss_gib() -> float:
    """Resident set of this process (GiB); 0.0 when it cannot be read."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / 2**30
    except Exception:
        return 0.0


class SwapGrowth(RuntimeError):
    """Raised by the scan's row hook when swap grows past ``--swap-stop-mb``."""


def swap_used_mb() -> float:
    import re

    out = subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True).stdout
    match = re.search(r"used = ([\d.]+)M", out)
    return float(match.group(1)) if match else -1.0


def gptq_settings(a):
    if a.quantizer != "gptq":
        return None
    from mlx2.gptq import GPTQSettings

    return GPTQSettings(bits=a.bits, group_size=a.group_size, damp=a.gptq_damp, block=a.gptq_block,
                        act_order=a.gptq_act_order, expert_chunk=a.gptq_expert_chunk)


def cmd_scan(a) -> int:
    from mlx2 import layer_kl as K

    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    revision = commit()  # the code that runs, captured before it runs
    if revision["tracked_dirty"] and not a.allow_dirty:
        raise SystemExit("tracked changes in the mlx2 tree: commit them (or pass --allow-dirty)")
    groups = [g.strip() for g in a.groups.split(",") if g.strip()]
    source_id = None
    if a.source:
        from mlx2.adapters.ordinary_artifact import inspect_indexed_artifact

        source_id = f"{a.source.resolve().name}@sha256:{inspect_indexed_artifact(a.source)['identity']['fingerprint']}"
    partial = out / "scan.partial.jsonl"
    swap0 = swap_used_mb()

    def on_row(row, base):
        row["swap_used_mb"] = swap_used_mb()
        with open(partial, "a") as handle:
            handle.write(json.dumps({"baseline": base, "row": row}) + "\n")
        if a.swap_stop_mb and row["swap_used_mb"] - swap0 > a.swap_stop_mb:
            raise SwapGrowth(f"swap grew {row['swap_used_mb'] - swap0:.0f} MB (limit {a.swap_stop_mb:.0f}); "
                             f"rows so far are in {partial.name}")

    calib_seen = [0]

    def between_texts():
        used = swap_used_mb()
        calib_seen[0] += 1
        import mlx.core as _mx

        # One line per calibration text: where the memory is (MLX active /
        # buffer cache / peak vs the process footprint) when swap grows.
        print(f"[calib {calib_seen[0]}] mlx active {_mx.get_active_memory() / 2**30:.2f} "
              f"cache {_mx.get_cache_memory() / 2**30:.2f} peak {_mx.get_peak_memory() / 2**30:.2f} GiB "
              f"rss {_rss_gib():.2f} GiB swap +{used - swap0:.0f} MB", flush=True)
        if a.swap_stop_mb and used - swap0 > a.swap_stop_mb:
            raise SwapGrowth(f"swap grew {used - swap0:.0f} MB during calibration (limit "
                             f"{a.swap_stop_mb:.0f}); rows so far are in {partial.name}")

    adapter, checkpoint, policy = load_adapter(a.model, a.execution_policy, wire=not a.no_wire)
    swap0 = swap_used_mb()  # after the load: the guard watches the scan, not the load
    try:
        source = (K.checkpoint_source(a.source, adapter.model, layer_prefix=a.layer_prefix, groups=groups)
                  if a.source else None)
        result = K.scan(adapter.model, a.reference, layers=K.parse_layers(a.layers),
                        groups=groups, bits=a.bits, on_row=on_row,
                        group_size=a.group_size, mode=a.mode, source=source, cumulative=a.cumulative,
                        splits=tuple(s.strip() for s in a.splits.split(",")), layer_prefix=a.layer_prefix,
                        log=lambda line: print(line, flush=True), quantizer=a.quantizer,
                        calib_texts=json.loads(Path(a.calib_texts).read_text()) if a.calib_texts else (),
                        gptq_settings=gptq_settings(a), between_texts=between_texts)
    finally:
        adapter.close()
    result.update(weight_residency=dict(WIRING), checkpoint=checkpoint, execution_policy=policy, mlx2=revision,
                  source_checkpoint=source_id, finished_at=datetime.now(UTC).isoformat())
    if not a.cumulative:  # a cumulative prefix order is the input, not a ranking
        result["ranked_layers"] = [r["layer"] for r in K.ranked(result["rows"])]
    tmp = out / "scan.json.partial"
    tmp.write_text(json.dumps(result, indent=2) + "\n")
    os.replace(tmp, out / "scan.json")
    lines = ["| layer | KL | dKL | top-1 | GiB before -> after |", "|---|---|---|---|---|"]
    for r in result["rows"]:
        lines.append(f"| {r['layer']} | {r['kl_mean']:.5f} | {r['delta_kl']:+.5f} | {r['top1_agreement']:.4f} | "
                     f"{r['bytes_before'] / 2**30:.3f} -> {r['bytes_after'] / 2**30:.3f} |")
    (out / "scan.md").write_text(f"# layer KL scan: {checkpoint}\n\nbaseline KL {result['baseline']['kl_mean']:.5f}, "
                                 f"top-1 {result['baseline']['top1_agreement']:.4f}; restore exact: "
                                 f"{result['restore_check']['exact']}\n\n" + "\n".join(lines) + "\n")
    print(json.dumps({"out": str(out), "baseline_kl": result["baseline"]["kl_mean"],
                      "restore_exact": result["restore_check"]["exact"],
                      "ranked_layers": result.get("ranked_layers", [])[:10]}))
    return 0 if result["restore_check"]["exact"] else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("texts")
    t.add_argument("--model", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--count", type=int, default=8)
    t.add_argument("--tokens", type=int, default=1024)
    t.add_argument("--offset", type=int, default=200_000)
    t.add_argument("--execution-policy", type=Path)
    t.add_argument("--corpus", type=Path, help="JSONL corpus: stratified calibration texts")
    t.add_argument("--split", default="calib")
    t.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("reference")
    r.add_argument("--model", type=Path, required=True)
    r.add_argument("--texts", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--family", help="sweep family name: stream the model layer-major")
    r.add_argument("--execution-policy", type=Path)
    r.add_argument("--skip", type=int, default=32)
    r.add_argument("--dtype", default="float32", choices=("float32", "float16"))
    r.add_argument("--no-wire", action="store_true", help="leave the weights pageable (default: wire them)")
    r.add_argument("--i-own-gpu", action="store_true")
    s = sub.add_parser("scan")
    s.add_argument("--model", type=Path, required=True)
    s.add_argument("--reference", type=Path, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--layers", required=True)
    s.add_argument("--groups", required=True, help="comma list of module path suffixes, e.g. mlp.experts")
    s.add_argument("--bits", type=int, required=True)
    s.add_argument("--group-size", type=int, default=64)
    s.add_argument("--mode", default="affine")
    s.add_argument("--source", type=Path, help="dense source checkpoint to requantize from")
    s.add_argument("--cumulative", action="store_true")
    s.add_argument("--splits", default="dev")
    s.add_argument("--layer-prefix", default="model.layers.")
    s.add_argument("--execution-policy", type=Path)
    s.add_argument("--quantizer", choices=("rtn", "gptq"), default="rtn")
    s.add_argument("--calib-texts", type=Path, help="GPTQ calibration texts (JSON, disjoint from the reference texts)")
    s.add_argument("--gptq-damp", type=float, default=0.01)
    s.add_argument("--gptq-block", type=int, default=128)
    s.add_argument("--gptq-act-order", action="store_true")
    s.add_argument("--gptq-expert-chunk", type=int, default=32)
    s.add_argument("--allow-dirty", action="store_true", help="run from a tree with uncommitted tracked changes")
    s.add_argument("--swap-stop-mb", type=float, default=2048.0,
                   help="stop (rows kept) when swap grows past this; 0 disables")
    s.add_argument("--no-wire", action="store_true", help="leave the weights pageable (default: wire them)")
    s.add_argument("--i-own-gpu", action="store_true")
    a = parser.parse_args(argv)
    if a.command in ("reference", "scan") and not a.i_own_gpu:
        parser.error(f"{a.command} runs on the GPU: hold the CPG lease and paired locks and pass --i-own-gpu")
    return {"texts": cmd_texts, "reference": cmd_reference, "scan": cmd_scan}[a.command](a)


if __name__ == "__main__":
    raise SystemExit(main())
