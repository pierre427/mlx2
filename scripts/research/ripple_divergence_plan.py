#!/usr/bin/env python3
"""Build the bounded near-tie classification plan from Ripple A/B receipts.

For every speculative/chunking arm whose greedy tokens differ from the eager
control, record the first divergent position, the shared prefix, both
tokens, and both full original continuations (for the functional oracle).
The plan names one server per arm and the bounded ``max_tokens`` each request
needs (position + 1), so only already-selected positions are captured.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ripple_ab import COPY_SOURCES, DECODES, copy_prompt  # noqa: E402


def cells():
    out = {f"decode{i}": text for i, text in enumerate(DECODES)}
    out.update({f"copy{i}": copy_prompt(path) for i, path in enumerate(COPY_SOURCES)})
    return out


def tokens(arm, cell):
    kind, index = ("decode", int(cell[6:])) if cell.startswith("decode") else ("copy", int(cell[4:]))
    return arm[f"{kind}_runs"][index]["token_ids"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="append", required=True,
                        help="RECEIPT|CONTROL_ARM|RECEIPT_ARM=PLAN_NAME,...")
    parser.add_argument("--servers", type=Path, required=True, help="JSON {plan name: server args}")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prompts = cells()
    servers = json.loads(args.servers.read_text())
    plan = {"cells": prompts, "rows": [], "servers": servers}
    controls = {}
    for spec in args.run:
        receipt, control, arms = spec.split("|")
        data = {a["name"]: a for a in json.loads(Path(receipt).read_text())["arms"]}
        base = data[control]
        for cell in prompts:
            reference = tokens(base, cell)
            known = controls.setdefault(cell, reference)
            if known != reference:
                raise ValueError(f"controls disagree on {cell}; cannot share one control server")
        for item in arms.split(","):
            source, name = item.split("=", 1)
            if name not in servers:
                raise ValueError(f"no server arguments for {name}")
            for cell in prompts:
                got, reference = tokens(data[source], cell), tokens(base, cell)
                position = next((i for i, (a, b) in enumerate(zip(reference, got)) if a != b), None)
                if position is None:
                    continue
                plan["rows"].append({"arm": name, "cell": cell, "position": position,
                                     "control_token": reference[position], "arm_token": got[position],
                                     "prefix": reference[:position], "control_tokens": reference,
                                     "arm_tokens": got})
    if "control" not in servers:
        raise ValueError("servers must name the eager control")
    plan["controls"] = controls
    args.output.write_text(json.dumps(plan, indent=1) + "\n")
    print(json.dumps([(r["arm"], r["cell"], r["position"]) for r in plan["rows"]]))


if __name__ == "__main__":
    main()
