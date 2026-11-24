#!/usr/bin/env python3
"""Knapsack per-layer mixed precision: inventory, score, plan, export (mlx2.conversion.knapsack).

Original mlx2 script.  Every output names the checkpoint (``<dir>@sha256:<fp>``,
the mlx2 adapter fingerprint) and the mlx2 commit.  Refuses existing outputs.

    knapsack_plan.py inventory --model DENSE --menu q2g64,q3g64,q4g64,nvfp4
    knapsack_plan.py score-mse --model DENSE --menu ... --out costs.json [--relative] [--device cpu]
    knapsack_plan.py plan --model DENSE --menu ... --budget 0.678xnvfp4 --out plan.json
        (--costs costs.json | --kl-scan q3g64=scan.json ...) [--pin 0:>=3] [--min-bits 2]
        [--rest-format q8g64]
    knapsack_plan.py uniform --model DENSE --format q3g64 --out plan.json [--rest-format q8g64]
    knapsack_plan.py export --model DENSE --plan plan.json --out DIR [--device cpu]

``--device gpu`` needs ``--i-own-gpu`` (CPG lease + paired locks held by the caller).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlx2.conversion import knapsack as K  # noqa: E402
from mlx2.conversion import knapsack_export as E  # noqa: E402


def checkpoint_of(model: Path) -> str:
    from mlx2.adapters.registry import inspect_model

    resolution = inspect_model(model)
    return f"{K.checkpoint_name(model)}@sha256:{resolution.artifact_fingerprint}"


def device(a) -> None:
    import mlx.core as mx

    if a.device == "gpu":
        if not a.i_own_gpu:
            raise SystemExit("--device gpu needs --i-own-gpu (hold the CPG lease and paired locks)")
        mx.set_default_device(mx.gpu)
    else:
        mx.set_default_device(mx.cpu)


def refuse(path: Path) -> None:
    if path.exists():
        raise SystemExit(f"{path} exists")


def cpu() -> None:
    import mlx.core as mx

    mx.set_default_device(mx.cpu)


def cmd_inventory(a) -> int:
    cpu()
    model, _ = E.build_model(a.model, module_name=a.model_module)
    units = K.inventory(model, layer_prefix=a.layer_prefix)
    menu = K.parse_menu(a.menu)
    sizes = K.layer_sizes(units, menu)
    rest = K.parse_format(a.rest_format) if a.rest_format else None
    print(json.dumps({"layers": len(sizes), "planned_modules": sum(u.role == "planned" for u in units),
                      "fixed_modules": [u.path for u in units if u.role == "fixed"][:4],
                      "rest_modules": [u.path for u in units if u.role == "rest"],
                      "planned_weights": sum(u.numel for u in units if u.role == "planned"),
                      **K.fixed_bytes(units, rest),
                      "uniform_bytes": {f.name: (sum(r[f.name] for r in sizes.values())
                                                 if all(r[f.name] for r in sizes.values()) else None)
                                        for f in menu}}, indent=1))
    return 0


def cmd_score_mse(a) -> int:
    refuse(a.out)
    device(a)
    checkpoint = checkpoint_of(a.model)
    model, _ = E.load_dense(a.model, module_name=a.model_module)
    units = K.inventory(model, layer_prefix=a.layer_prefix)
    menu = K.parse_menu(a.menu)
    costs = K.weight_mse_costs(model, units, menu, relative=a.relative,
                               log=lambda m: print(m, flush=True))
    name = "weight_mse_relative" if a.relative else "weight_mse"
    doc = K._costs_doc(name, {"menu": [f.name for f in menu], "quantizer": "mlx_rtn", "device": a.device,
                              "mlx2": K.commit_receipt(ROOT)}, checkpoint, costs)
    a.out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"out": str(a.out), "checkpoint": checkpoint, "layers": len(costs)}))
    return 0


def _plan(a, menu, costs, scorer, budget):
    checkpoint = checkpoint_of(a.model)
    model, _ = E.build_model(a.model, module_name=a.model_module)
    units = K.inventory(model, layer_prefix=a.layer_prefix)
    rest = K.parse_format(a.rest_format) if a.rest_format else None
    if callable(costs):
        costs = costs(checkpoint, units)
    plan = K.make_plan(checkpoint=checkpoint, units=units, menu=menu, costs=costs, budget=budget,
                       scorer=scorer, mlx2=K.commit_receipt(ROOT), rest_format=rest,
                       pins=K.parse_pins(a.pin or []), min_bits=a.min_bits, layer_prefix=a.layer_prefix,
                       resolution=a.resolution)
    K.save_plan(plan, a.out)
    s = plan["solution"]
    print(json.dumps({"out": str(a.out), "checkpoint": checkpoint, "budget": plan["budget"],
                      "planned_bpw": round(s["planned_bpw"], 4), "predicted_cost": s["predicted_cost"],
                      "format_counts": s["format_counts"], "exact": s["exact"],
                      "layers": " ".join(f"{r['layer']}:{r['format']}" for r in plan["layers"]),
                      "uniform_baselines": plan["uniform_baselines"]}, indent=1))
    return 0


def cmd_plan(a) -> int:
    cpu()
    refuse(a.out)
    menu = K.parse_menu(a.menu)
    if bool(a.costs) == bool(a.kl_scan):
        raise SystemExit("give exactly one of --costs or --kl-scan")
    if a.costs:
        doc = json.loads(a.costs.read_text())
        scorer = {"name": doc["scorer"], "settings": doc["settings"], "input": str(a.costs),
                  "input_sha256": _sha(a.costs)}
        costs = lambda ck, units: K.table_costs(doc, checkpoint=ck)  # noqa: E731
    else:
        scans, inputs = {}, {}
        for item in a.kl_scan:
            name, _, path = item.partition("=")
            path = Path(path)
            scans[K.parse_format(name).name] = json.loads(path.read_text())
            inputs[K.parse_format(name).name] = {"path": str(path), "sha256": _sha(path)}
        first = next(iter(scans.values()))
        scorer = {"name": "layer_kl", "settings": {"key": "delta_kl", "reference": first["reference"],
                                                   "groups": first["settings"]["groups"]},
                  "inputs": inputs}
        costs = lambda ck, units: K.layer_kl_costs(scans, checkpoint=ck, units=units)  # noqa: E731
    return _plan(a, menu, costs, scorer, a.budget)


def cmd_uniform(a) -> int:
    cpu()
    refuse(a.out)
    fmt = K.parse_format(a.format)
    model, _ = E.build_model(a.model, module_name=a.model_module)
    layers = K.layers_of(K.inventory(model, layer_prefix=a.layer_prefix))
    a.pin, a.min_bits, a.resolution = [], 0.0, None
    return _plan(a, [fmt], {l: {fmt.name: 0.0} for l in layers}, {"name": "uniform"}, f"1.0x{fmt.name}")


def cmd_export(a) -> int:
    refuse(a.out)
    device(a)
    plan = K.load_plan(a.plan)
    checkpoint = checkpoint_of(a.model)
    if plan["checkpoint"] != checkpoint:
        raise SystemExit(f"plan is for {plan['checkpoint']}, model is {checkpoint}")
    # The serving adapter's own artifact inspection must accept the export before it is renamed into place.
    receipt = E.export(a.model, a.out, plan, module_name=a.model_module, log=lambda m: print(m, flush=True),
                       validate=checkpoint_of)
    receipt["exported_checkpoint"] = f"{K.checkpoint_name(a.out)}@{receipt['validated'].rpartition('@')[2]}"
    receipt["source_checkpoint"] = checkpoint
    receipt["mlx2"] = K.commit_receipt(ROOT)
    (a.out / "knapsack-export-receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps(receipt, indent=1))
    return 0


def _sha(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(s, menu=True):
        s.add_argument("--model", type=Path, required=True, help="dense source checkpoint directory")
        s.add_argument("--model-module", help="runtime module (default mlx2.runtime.models.<model_type>)")
        s.add_argument("--layer-prefix", default="model.layers.")
        if menu:
            s.add_argument("--menu", action="append", required=True, help="comma list, e.g. q2g64,q3g64,nvfp4")

    s = sub.add_parser("inventory"); common(s); s.add_argument("--rest-format")
    s = sub.add_parser("score-mse"); common(s)
    s.add_argument("--out", type=Path, required=True); s.add_argument("--relative", action="store_true")
    s.add_argument("--device", choices=("cpu", "gpu"), default="cpu"); s.add_argument("--i-own-gpu", action="store_true")
    s = sub.add_parser("plan"); common(s)
    s.add_argument("--out", type=Path, required=True); s.add_argument("--budget", required=True)
    s.add_argument("--costs", type=Path); s.add_argument("--kl-scan", action="append", help="FORMAT=scan.json")
    s.add_argument("--pin", action="append"); s.add_argument("--min-bits", type=float, default=0.0)
    s.add_argument("--rest-format"); s.add_argument("--resolution", type=int)
    s = sub.add_parser("uniform"); common(s, menu=False)
    s.add_argument("--format", required=True); s.add_argument("--out", type=Path, required=True)
    s.add_argument("--rest-format")
    s = sub.add_parser("export"); common(s, menu=False)
    s.add_argument("--plan", type=Path, required=True); s.add_argument("--out", type=Path, required=True)
    s.add_argument("--device", choices=("cpu", "gpu"), default="cpu"); s.add_argument("--i-own-gpu", action="store_true")
    a = p.parse_args(argv)
    return {"inventory": cmd_inventory, "score-mse": cmd_score_mse, "plan": cmd_plan,
            "uniform": cmd_uniform, "export": cmd_export}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
