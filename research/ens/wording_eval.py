"""ENS H43 record port: a wording stress test on finished ports (eval only, no training).

Pierre (2026-11-24): the earlier wording failures were "also ordering of words"; "Yes" to a stronger wording test on
both record ports (Qwen3-4B `evidence/m5-h43-20261123/bus`, Kimi Linear 48B `evidence/m5-kimi-20261124/bus`). The
ports' own eval used one held wording per task, and the one cell both models failed ("which concept?") is the only
held wording that names the concept before the titles.

This runs the same evaluation (physics_bus.evaluate: base / record / swapped record, answers generated and parsed by
physics_qa.score) on 10 new wordings per task, none used in training or in the earlier held set, tagged by word
order:
  title_first  the paper title(s) come before the question's key word ("'{A}': which field?")
  key_first    the key word / request comes first ("Which field of physics? The paper is '{A}'.")
Paper A is always named before paper B, so the record order and the "first / second" answers keep their meaning.
Every wording is its own cell (resumable); the summary pools by order and by task.

Kill / read-out (fixed before the run): a port generalises across wordings if record accuracy on key_first is within
0.1 of title_first on every task; on any task where key_first falls to the swapped level while title_first does not,
word order (not the task) is the failure.

  python -m research.ens.wording_eval --run <port run dir> --out <evidence dir> [--n 12] [--max-seconds S]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from research.ens import physics_bus as PB
from research.ens import physics_qa as QA
from research.ens import common as C

WORDINGS = {
    "field": [
        ("'{A}' - which branch of physics would you put it in?", "title_first"),
        ("I'm reading '{A}' right now; what part of physics is that?", "title_first"),
        ("Take '{A}'. Tell me its physics subfield.", "title_first"),
        ("'{A}' sounds interesting. Which area of physics does it come from?", "title_first"),
        ("A colleague sent me '{A}'. What field is it filed under?", "title_first"),
        ("Which branch of physics? The paper is '{A}'.", "key_first"),
        ("Physics subfield, please, for this paper: '{A}'.", "key_first"),
        ("What area of physics should I file this under: '{A}'?", "key_first"),
        ("Quick classification question about the field of '{A}'.", "key_first"),
        ("I need the physics category for a paper called '{A}'.", "key_first"),
    ],
    "month": [
        ("'{A}' - do you know which month it came out?", "title_first"),
        ("I found '{A}'. When in 2026 was it posted?", "title_first"),
        ("Take '{A}': what month did it appear on arXiv?", "title_first"),
        ("'{A}' was posted this year, but in which month?", "title_first"),
        ("My notes mention '{A}'. Any idea when it was released?", "title_first"),
        ("Which month of 2026? I mean the paper '{A}'.", "key_first"),
        ("Posting month, please, for '{A}'.", "key_first"),
        ("When was this one released on arXiv: '{A}'?", "key_first"),
        ("I'm dating my references. The month for '{A}'?", "key_first"),
        ("In what month did the following paper appear: '{A}'?", "key_first"),
    ],
    "concept": [
        ("'{A}' - what is the main idea it uses?", "title_first"),
        ("I'm about to read '{A}'. Which concept is at its heart?", "title_first"),
        ("Take '{A}': name the central concept.", "title_first"),
        ("'{A}' keeps coming up. What key idea does it rely on?", "title_first"),
        ("My advisor mentioned '{A}'. What concept is it built around?", "title_first"),
        ("Which central concept? The paper in question is '{A}'.", "key_first"),
        ("Key idea, please, for '{A}'.", "key_first"),
        ("What is the core concept of this paper: '{A}'?", "key_first"),
        ("Main concept, in a word or two, of '{A}'?", "key_first"),
        ("Name the concept that the following paper is about: '{A}'.", "key_first"),
    ],
    "same_field": [
        ("'{A}' and '{B}' - one physics subfield for both, or two?", "title_first"),
        ("I have '{A}' and '{B}' on my desk. Do they share a field?", "title_first"),
        ("Take '{A}' and '{B}': is the physics subfield the same?", "title_first"),
        ("'{A}', then '{B}'. Would one physics category cover both?", "title_first"),
        ("My list has '{A}' and also '{B}'. Same field?", "title_first"),
        ("Same field of physics? I'm comparing '{A}' and '{B}'.", "key_first"),
        ("Is the subfield shared between '{A}' and '{B}'?", "key_first"),
        ("Do these two belong to one area of physics: '{A}' and '{B}'?", "key_first"),
        ("I'm grouping papers by field. Do '{A}' and '{B}' go together?", "key_first"),
        ("Field check for two papers, '{A}' and '{B}': the same?", "key_first"),
    ],
    "earlier": [
        ("'{A}' or '{B}' - which one was posted first?", "title_first"),
        ("I have '{A}' and '{B}'. Which appeared earlier?", "title_first"),
        ("Take '{A}' and '{B}': which came out sooner?", "title_first"),
        ("'{A}', then '{B}'. Which of the two went up on arXiv first?", "title_first"),
        ("My list has '{A}' and '{B}'. Which is older?", "title_first"),
        ("Posting order, please: '{A}' versus '{B}', which is older?", "key_first"),
        ("Earlier posting date: '{A}' or '{B}'?", "key_first"),
        ("To sort by date, which is earlier, '{A}' or '{B}'?", "key_first"),
        ("I'm building a timeline. Does '{A}' or '{B}' go first?", "key_first"),
        ("Which one appeared on arXiv sooner, '{A}' or '{B}'?", "key_first"),
    ],
    "which_concept": [
        ("'{A}' or '{B}' - which one is about {c}?", "title_first"),
        ("I have '{A}' and '{B}'. Which deals with {c}?", "title_first"),
        ("Take '{A}' and '{B}': which works with {c}?", "title_first"),
        ("'{A}', then '{B}'. Which of the two is built around {c}?", "title_first"),
        ("My list has '{A}' and '{B}'. Which one covers {c}?", "title_first"),
        ("Topic: {c}. Which paper covers it, '{A}' or '{B}'?", "key_first"),
        ("I need a paper on {c}. Should I read '{A}' or '{B}'?", "key_first"),
        ("Which paper is about {c}: '{A}' or '{B}'?", "key_first"),
        ("Studying {c} this week. Is it covered by '{A}' or by '{B}'?", "key_first"),
        ("The concept is {c}. Which matches, '{A}' or '{B}'?", "key_first"),
    ],
}


def _check_wordings():
    used = {q for d in (QA.Q_TRAIN, QA.Q_HELD) for qs in d.values() for q in qs}
    for task, ws in WORDINGS.items():
        assert len(ws) == 10 and sum(o == "title_first" for _, o in ws) == 5, task
        for q, _ in ws:
            assert q not in used, q
            assert "{A}" in q and (("{B}" in q) == (task not in PB.RETRIEVAL)), q
            if "{B}" in q:
                assert q.index("{A}") < q.index("{B}"), q            # A named first keeps first/second answers valid
            assert ("{c}" in q) == (task == "which_concept"), q


def run(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="a finished physics_bus run dir (args.json, port.safetensors)")
    p.add_argument("--out", required=True, help="evidence dir for this eval (wording_cells.json, receipt.json)")
    p.add_argument("--n", type=int, default=12, help="questions per wording (x 10 wordings per task)")
    p.add_argument("--max-seconds", type=float, default=None, help="pause between batches; a rerun resumes")
    p.add_argument("--no-memory-gate", action="store_true")
    p.add_argument("--chunk-tag", default=None)
    a = p.parse_args(argv)
    _check_wordings()
    run_dir, out = Path(a.run), Path(a.out)
    if out.resolve() == run_dir.resolve():
        raise SystemExit("--out must not be the run directory (it would overwrite the run's receipt)")
    out.mkdir(parents=True, exist_ok=True)
    training = json.loads((run_dir / "receipt.json").read_text())
    port_file = run_dir / "port.safetensors"
    with port_file.open("rb") as f:
        port_sha = hashlib.file_digest(f, "sha256").hexdigest()
    if port_sha[:12] != training["port_checkpoint"].rpartition("@")[2]:
        raise SystemExit("port weights differ from the training receipt")
    rargs = SimpleNamespace(**(PB._ARG_DEFAULTS | training["args"]))
    rargs.eval_n = a.n
    if not a.no_memory_gate:
        PB.memory_gate(gib=PB.gate_gib(rargs))
    facts, model, cfg, ckpt, tok, voc = PB.load_world(rargs)
    if ckpt != training["base_checkpoint"] or facts["sha256"] != training["facts_sha256"]:
        raise SystemExit("base checkpoint or facts differ from the training receipt")
    port = PB.RecordPort(model, rargs.site, rargs.rank, rargs.tau)
    port.load_weights(str(run_dir / "port.safetensors"))
    titles = PB.outside_titles(facts, tok, seed=rargs.seed)
    deadline = time.time() + a.max_seconds if a.max_seconds else None
    cells = PB.EvalCells(out, deadline)
    identity = [port_sha, ckpt, facts["sha256"], rargs.seed, a.n, WORDINGS]
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    cells.path = out / f"wording_cells-{key}.json"
    cells.cells = json.loads(cells.path.read_text()) if cells.path.exists() else {}
    sets = []
    for task, ws in WORDINGS.items():
        for i, (q, order) in enumerate(ws):
            sets.append((f"w{i:02d}.{order}", {t: ([q] if t == task else ["{A}"]) for t in WORDINGS}, task))
    PB.evaluate(model, port, tok, voc, facts, titles, rargs, cells, sets=sets, text=False)
    summary = summarize(cells.cells)
    receipt = {"schema": "ens.h43.wording_eval.v1", "run": str(run_dir), "port_checkpoint": training["port_checkpoint"],
        "port_sha256": port_sha, "cells_file": cells.path.name,
        "base_checkpoint": ckpt, "n_per_wording": a.n,
        "wordings": {t: [{"q": q, "order": o} for q, o in ws] for t, ws in WORDINGS.items()}, "summary": summary,
        **C.source_revision(), **C.host_record()}
    C.write_json(out / "receipt.json", receipt)
    print(json.dumps(summary, indent=1), flush=True)
    return 0


def summarize(cells: dict) -> dict:
    """Pooled record / swapped / base accuracy by task x order (mean over wordings, each cell n questions)."""
    acc: dict = {}
    for key, v in cells.items():
        name, task = key.rsplit(".", 1)                 # "w03.key_first.field"
        order = name.split(".", 1)[1]
        for arm in ("record", "swapped_record", "base"):
            k = f"{arm}.correct"
            if k in v:
                acc.setdefault(task, {}).setdefault(order, {}).setdefault(arm, []).append(v[k])
    out = {}
    for task, orders in acc.items():
        out[task] = {o: {arm: round(sum(xs) / len(xs), 4) for arm, xs in arms.items()} | {"wordings": len(arms["record"])}
                     for o, arms in orders.items()}
        tf, kf = out[task].get("title_first", {}), out[task].get("key_first", {})
        if "record" in tf and "record" in kf:
            out[task]["key_minus_title"] = round(kf["record"] - tf["record"], 4)
    return out


if __name__ == "__main__":
    sys.exit(run())
