#!/usr/bin/env python3
"""Split manifest for a finished bridge-lab topic run (host-neuron reuse review v4 §5.1). CPU only: tokenizers, no
model weights, no GPU.

Reproduces the run's text selection with the host tokenizer (amoeba_bridge_lab.topic_selection, the code the run
used) and writes:
  selection            the run's train / val / test probe ids (test = the held texts the run scored; the review
                       calls them SPENT: development data from here on)
  reserved_final_test  every other eligible held text of the topic (>= --topic-tokens tokens) whose source group the
                       run never touched: the final test. No run may read them; amoeba_bridge_lab.py
                       --split-manifest skips them and their groups
  counts, the run's receipt hashes and checkpoints (a run given this manifest must load the same checkpoints)
Usage: amoeba_bridge_split_manifest.py --run ~/amoeba-runs/<run>/physics --out manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build(rows, tid, a, encode, run_meta):
    import amoeba_bridge_lab as L
    items = L.topic_selection(rows, tid, a["seed"], a["n_train"], a["n_val"], a["n_test"], a["topic_tokens"],
                              encode, bos=[])
    sel = {k: [it["probe_id"] for it in items if it["split"] == k] for k in ("train", "val", "test")}
    under = lambda ts: any(t == tid or t.startswith(tid + ".") for t in ts)  # noqa: E731
    held = [r for r in rows if r["split"] == "held" and under(r["topics"])]
    fit = [r for r in rows if r["split"] == "fit" and under(r["topics"])]
    elig = lambda r: len(list(encode(r["text"]))[:a["topic_tokens"]]) >= a["topic_tokens"]  # noqa: E731
    held_e = [r for r in held if elig(r)]
    used = set(sel["test"]) | set(sel["train"]) | set(sel["val"])
    used_groups = {str(r.get("group")) for r in rows if r["id"] in used}
    unused = [r for r in held_e if r["id"] not in used]
    reserved = [r for r in unused if str(r.get("group")) not in used_groups]
    return {
        "schema": "mlx2.amoeba-bridge-split-manifest.v1", "topic": tid,
        "selection_args": {k: a[k] for k in ("seed", "n_train", "n_val", "n_test", "topic_tokens")},
        "selection": sel,
        "reserved_final_test": {"probe_ids": [r["id"] for r in reserved],
                                "groups": sorted({str(r.get("group")) for r in reserved}),
                                "rule": "eligible held texts of the topic the run did not score, in source groups the "
                                        "run never touched; the final test: score only after the contrast, minimum "
                                        "useful effect and drift are fixed on development data"},
        "counts": {"fit_eligible": sum(elig(r) for r in fit), "held_eligible": len(held_e),
                   "held_unused": len(unused), "reserved_texts": len(reserved),
                   "reserved_groups": len({str(r.get('group')) for r in reserved}),
                   **{f"selected_{k}": len(v) for k, v in sel.items()}},
        "dev_note": "selection.test are the run's scored held texts: SPENT (review v4 §5.1), development data",
        **run_meta}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="the finished run's output directory (run.json, box.json)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run = Path(args.run).expanduser()
    rj = json.loads((run / "run.json").read_text())
    a = rj["args"]
    if not a["task"].startswith("topic:"):
        sys.exit("not a topic run")
    if Path(args.out).exists():
        sys.exit(f"refusing to overwrite {args.out}")
    if sha(a["probes"]) != rj["probes_sha256"]:
        sys.exit("probe corpus changed since the run")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(rj["host_path_resolved"])
    rows = [json.loads(line) for line in open(a["probes"])]
    meta = {"source_run": str(run), "source_run_json_sha256": sha(run / "run.json"),
            "source_box_json_sha256": sha(run / "box.json") if (run / "box.json").exists() else None,
            "source_status": rj.get("status"), "source_mlx2_revision": rj.get("mlx2_source_revision"),
            "host_checkpoint": rj["host_checkpoint"], "donor_checkpoint": rj["donor_checkpoint"],
            "probes": a["probes"], "probes_sha256": rj["probes_sha256"],
            "tokenizer": f"host tokenizer from {rj['host_path_resolved']} (transformers AutoTokenizer, "
                         "add_special_tokens=False)"}
    man = build(rows, a["task"][6:], a, lambda t: tok.encode(t, add_special_tokens=False), meta)
    Path(args.out).write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps(man["counts"]))


if __name__ == "__main__":
    main()
