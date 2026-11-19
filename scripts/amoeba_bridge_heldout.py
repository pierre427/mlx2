#!/usr/bin/env python3
"""The held texts a bridge-lab topic run never touched, as an exclusion list for `mlx2 amoeba extract --exclude-texts`.

CPU only (tokenizer, no model). Replays the knowledge-topic item selection of scripts/amoeba_bridge_lab.py (the
`TOPIC` branch of main(): rows whose topics include the topic or a node below it, per split a permutation from
numpy default_rng(seed + 3), fit first then held, keeping texts of at least --topic-tokens tokens until the split's
quota is met) and reports, for the held split:

  used         the --n-test texts the run scored (spent: development data from then on)
  eligible     every held text of >= --topic-tokens tokens (the run's candidate pool)
  untouched    eligible, unused, and in a source group none of the used texts came from

The exclusion list is every held text at or below the topic, of any length, whose source group no used text came
from (v4 review s5.1: "exclude the whole held partition of physics source groups not used by the seeded 200"), so
it is a superset of the untouched eligible texts. --expect U,E,N,G refuses unless used, eligible, untouched-eligible
texts and their groups match the counts a review already recorded (the replay must be the run's own selection).

    amoeba_bridge_heldout.py --probes CORPUS.jsonl --tokenizer MODEL_DIR --topic sci.physics --seed 20261116 \\
        --n-fit 200 --n-test 200 --topic-tokens 64 --run-receipt RUN.json --expect 200,508,297,287 --out OUT.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def select(rows, encode, topic, seed, n_fit, n_test, topic_tokens):
    """The lab's per-split choice: returns ({split: chosen rows}, {split: eligible rows})."""
    import numpy as np
    under = lambda ts: any(t == topic or t.startswith(topic + ".") for t in ts)  # noqa: E731
    g = np.random.default_rng(seed + 3)
    chosen, eligible = {}, {}
    for split, want in (("fit", n_fit), ("held", n_test)):
        pool = [r for r in rows if r["split"] == split and under(r["topics"])]
        pool = [pool[i] for i in g.permutation(len(pool))]
        got, el = [], []
        for r in pool:
            if len(encode(r["text"])[:topic_tokens]) < topic_tokens:
                continue
            el.append(r)
            if len(got) < want:
                got.append(r)
        chosen[split], eligible[split] = got, el
    return chosen, eligible


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", required=True)
    ap.add_argument("--tokenizer", required=True, help="the run's tokenizer (a model directory)")
    ap.add_argument("--topic", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-fit", type=int, required=True, help="the run's --n-train + --n-val")
    ap.add_argument("--n-test", type=int, required=True)
    ap.add_argument("--topic-tokens", type=int, default=64)
    ap.add_argument("--run-receipt", default="", help="the run's run.json: its probes sha256 and args must match")
    ap.add_argument("--expect", default="", help="used,eligible,untouched,untouched_groups (refuse on mismatch)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    probes = Path(a.probes).expanduser()
    sha = hashlib.sha256(probes.read_bytes()).hexdigest()
    run = {}
    if a.run_receipt:
        run = json.loads(Path(a.run_receipt).expanduser().read_text())
        ra = run.get("args") or {}
        want = {"probes_sha256": sha, "task": f"topic:{a.topic}", "seed": a.seed, "n_test": a.n_test,
                "n_fit": a.n_fit, "topic_tokens": a.topic_tokens}
        got = {"probes_sha256": run.get("probes_sha256"), "task": ra.get("task"), "seed": ra.get("seed"),
               "n_test": ra.get("n_test"), "n_fit": (ra.get("n_train") or 0) + (ra.get("n_val") or 0),
               "topic_tokens": ra.get("topic_tokens")}
        bad = {k: (want[k], got[k]) for k in want if want[k] != got[k]}
        if bad:
            raise SystemExit(f"run receipt {a.run_receipt} disagrees with this replay: {bad}")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(Path(a.tokenizer).expanduser()), local_files_only=True)
    rows = [json.loads(x) for x in probes.read_text().split("\n") if x.strip()]
    chosen, eligible = select(rows, lambda t: tok.encode(t, add_special_tokens=False), a.topic, a.seed, a.n_fit,
                              a.n_test, a.topic_tokens)
    under = lambda ts: any(t == a.topic or t.startswith(a.topic + ".") for t in ts)  # noqa: E731
    used = chosen["held"]
    used_ids, used_groups = {r["id"] for r in used}, {str(r["group"]) for r in used}
    untouched = [r for r in eligible["held"] if r["id"] not in used_ids and str(r["group"]) not in used_groups]
    held_all = [r for r in rows if r["split"] == "held" and under(r["topics"])]
    excl = [r for r in held_all if str(r["group"]) not in used_groups]
    counts = {"held_at_or_below_topic": len(held_all), "used": len(used), "eligible": len(eligible["held"]),
              "unused_eligible": len(eligible["held"]) - len(used), "untouched_eligible": len(untouched),
              "untouched_eligible_groups": len({str(r["group"]) for r in untouched}),
              "excluded": len(excl), "excluded_groups": len({str(r["group"]) for r in excl}),
              "fit_used": len(chosen["fit"]), "fit_eligible": len(eligible["fit"])}
    if a.expect:
        e = [int(x) for x in a.expect.split(",")]
        got = [counts["used"], counts["eligible"], counts["untouched_eligible"], counts["untouched_eligible_groups"]]
        if got != e:
            raise SystemExit(f"replay counts {got} (used, eligible, untouched, groups) != expected {e}: not the run's "
                             "own selection")
    shared = [r["id"] for r in rows if str(r.get("group")) in {str(x["group"]) for x in excl}
              and r["id"] not in {x["id"] for x in excl}]
    out = {"schema": "mlx2.amoeba-exclude-texts.v1", "topic": a.topic,
           "rule": "every held text at or below the topic whose source group no text of the run's seeded test set "
                   "came from: the reserved final test (any length; a superset of the untouched eligible texts)",
           "probes": {"path": str(probes.resolve()), "sha256": sha},
           "run": {"receipt": a.run_receipt or None, "donor": run.get("donor_checkpoint"),
                   "host": run.get("host_checkpoint"), "mlx2_source_revision": run.get("mlx2_source_revision")},
           "replay": {"seed": a.seed, "n_fit": a.n_fit, "n_test": a.n_test, "topic_tokens": a.topic_tokens,
                      "tokenizer": str(Path(a.tokenizer).expanduser()), "rng": "numpy default_rng(seed + 3)",
                      "expect": a.expect or None},
           "counts": counts, "rows_sharing_excluded_groups_outside_the_list": len(shared),
           "probe_ids": sorted(r["id"] for r in excl), "sha256": sorted(r["sha256"] for r in excl),
           "groups": sorted({str(r["group"]) for r in excl}),
           "used_test_probe_ids": sorted(used_ids), "untouched_eligible_probe_ids": sorted(r["id"] for r in untouched)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
