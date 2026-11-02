"""Process-level A/B of MLX environment knobs on the GLM-5.3 decode replay.

Some MLX settings (``MLX_MAX_OPS_PER_BUFFER``, ``MLX_MAX_MB_PER_BUFFER``,
``MLX_BFS_MAX_WIDTH``) are read once when the Metal device starts, so the
replay's in-process ``interleaved`` arm cannot switch them. This driver runs
``glm53_decode_replay.py replay --arms pipelined`` once per (round, arm) in a
fresh process, alternating the arms (A B A B ...), and collects each child's
replay.json. Every arm replays the same recorded tokens; ``tokens_match`` is
checked per child. Research only; nothing in serving changes.

    glm53_env_ab.py --out DIR --source-revision SHA --policy P --trace T \
        --arms '[{"name":"default","env":{}},{"name":"onecb","env":{...}}]' \
        [--rounds 2] [--steps 384]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-revision", default="unrecorded")
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--trace", type=Path, default=None)
    parser.add_argument("--arms", required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=384)
    parser.add_argument("--replay-arms", nargs="+", default=["pipelined"])
    parser.add_argument("--child", default=None,
                        help="instead of the replay, run this scripts/research child with --policy and --child-args")
    parser.add_argument("--child-args", default="[]", help="JSON list of extra child arguments")
    parser.add_argument("--replay-extra", default="[]", help="JSON list of extra replay arguments (e.g. --merge)")
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit("refusing to reuse a nonempty evidence directory")
    args.out.mkdir(parents=True, exist_ok=True)
    arms = json.loads(args.arms)
    rows = []
    for rnd in range(args.rounds):
        order = arms if rnd % 2 == 0 else list(reversed(arms))
        for arm in order:
            sub = args.out / f"r{rnd}-{arm['name']}"
            env = dict(os.environ)
            for key in ("MLX_MAX_OPS_PER_BUFFER", "MLX_MAX_MB_PER_BUFFER", "MLX_BFS_MAX_WIDTH"):
                env.pop(key, None)
            env.update({k: str(v) for k, v in arm.get("env", {}).items()})
            if args.child:
                cmd = [sys.executable, str(HERE / args.child), "--out", str(sub),
                       "--source-revision", args.source_revision, "--policy", str(arm.get("policy", args.policy)),
                       *json.loads(args.child_args)]
            else:
                cmd = [sys.executable, str(HERE / "glm53_decode_replay.py"), "--out", str(sub),
                       "--source-revision", args.source_revision, "replay", "--policy", str(arm.get("policy", args.policy)),
                       "--trace", str(args.trace), "--rebind-identity", "--steps", str(args.steps),
                       *json.loads(args.replay_extra), "--arms", *args.replay_arms]
            proc = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
            (args.out / f"r{rnd}-{arm['name']}.log").write_text(proc.stdout + proc.stderr)
            row = {"round": rnd, "arm": arm["name"], "env": arm.get("env", {}), "rc": proc.returncode,
                   "policy": str(arm.get("policy", args.policy))}
            if proc.returncode == 0 and args.child:
                row["child"] = json.loads(next(sub.glob("*.json")).read_text())
            elif proc.returncode == 0:
                rep = json.loads((sub / "replay.json").read_text())
                row["mlx_env"] = rep.get("mlx_env")
                row["results"] = [{k: a.get(k) for k in ("arm", "tokens_match", "first_mismatch", "step_p50_ms",
                                                          "build_p50_ms", "wait_p50_ms", "first_64_step_p50_ms",
                                                          "last_256_step_p50_ms")} for a in rep["arms"]]
            rows.append(row)
            print(json.dumps(row), flush=True)
    summary = {}
    for row in rows:
        for res in row.get("results", []):
            summary.setdefault(f"{row['arm']}/{res['arm']}", []).append(res["step_p50_ms"])
    (args.out / "env-ab.json").write_text(json.dumps({
        "schema": "mlx2.glm53-env-ab.v1", "source_revision": args.source_revision, "policy": str(args.policy),
        "trace": str(args.trace), "rows": rows, "step_p50_ms": summary}, indent=1) + "\n")
    print(json.dumps({"step_p50_ms": summary}), flush=True)


if __name__ == "__main__":
    main()
