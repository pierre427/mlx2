#!/usr/bin/env python3
"""Train QSA block indexers for a frozen model (stage-1 dense distillation).

The reference recipe (higa2515/Qwen3.8-27B-QSA-BF16 research/qsa) in MLX: per
training sequence, one dense chunked prefill of the frozen model records each
full-attention layer's attention input, queries and cached keys; each layer's
indexer then takes one AdamW step on the block-level KL to the dense attention
(``mlx2.runtime.models.qsa_indexer_train``).  Every ``eval_every`` steps the
held-out sequences report capture (teacher mass inside the indexer's own top-k)
without a gradient.

Exploratory research, not qualification.  Usage:
``train_qsa_indexers.py owned SPEC`` takes the CPG lease and paired GPU locks,
then re-enters itself as ``unit``.

Output (``spec["out"]``): ``train.jsonl`` (one line per step), ``report.json``,
``indexers/`` (``qsa-indexers.safetensors`` + ``config.json``, the directory
``qsa_sidecar.QSASidecar`` loads), and step checkpoints under ``ckpt/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# SIGTERM/SIGINT during the per-layer optimizer updates is held until the
# step's updates are all applied, so an interrupt checkpoint is never a
# half-applied step.
_SIGNALS = {"defer": False, "pending": None}

FILES = ("src/mlx2/runtime/models/qsa_indexer_train.py",
         "src/mlx2/runtime/models/qsa_sidecar.py", "scripts/train_qsa_indexers.py")


def atomic(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True, default=str))
    os.replace(tmp, path)


def source() -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--", *FILES], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    return {"commit": head, "dirty": bool(dirty),
            "files": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FILES}}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def full_attention(model):
    """``{layer index: attention}`` for the model's full (global) attention."""
    from mlx2.runtime.models import qsa_sidecar as qs

    (_, attentions, _) = qs._family_attention(model)
    return attentions


def install_capture(attentions):
    """Class-swap each global attention so ``_attend`` records x and q."""
    def capture_class(base):
        class Capture(base):
            def _attend(self, x, q, k, v, cache, mask):
                if self._cap is not None:
                    self._cap["x"].append(x[0])
                    self._cap["q"].append(q[0])
                return base._attend(self, x, q, k, v, cache, mask)
        Capture.__name__ = f"QSACapture{base.__name__}"
        return Capture

    swapped = {}
    for attention in attentions.values():
        base = type(attention)
        if base not in swapped:
            swapped[base] = capture_class(base)
        object.__setattr__(attention, "_cap", None)
        attention.__class__ = swapped[base]


def record(mx, model, attentions, ids, step):
    """Dense prefill of ``ids``; returns ``{layer: (x [T,h], q [Hq,T,D], k [Hkv,T,D])}``."""
    from mlx2.runtime.models.cache import KVCache

    length = len(ids)
    cache = model.make_cache()
    for entry in cache:
        if isinstance(entry, KVCache):
            entry.reserve(length)
    for attention in attentions.values():
        object.__setattr__(attention, "_cap", {"x": [], "q": []})
    for start in range(0, length, step):
        chunk = mx.array([ids[start:start + step]], dtype=mx.uint32)
        h = model.model(chunk, cache)
        mx.eval(h, *[a._cap["x"][-1] for a in attentions.values()],
                *[a._cap["q"][-1] for a in attentions.values()])
        mx.clear_cache()
    out = {}
    for (index, attention) in attentions.items():
        cap = attention._cap
        object.__setattr__(attention, "_cap", None)
        x = mx.concatenate(cap["x"], axis=0)
        q = mx.concatenate(cap["q"], axis=1)
        k = cache[index].keys[0, :, :length]
        mx.eval(x, q, k)
        out[index] = (x, q, k)
    del cache
    mx.clear_cache()
    return out


def diagnose(mx, np, qt, model, attentions, indexers, holdout, spec, scale, topk):
    """Teacher-mass breakdown per layer on the held-out rows: the untrained
    indexers (this seed) and each listed checkpoint, against the oracle,
    sink/recent windows and sink+recent+indexer hybrids."""
    arms = {"untrained": indexers}
    selections = {}
    for (name, path) in spec["diagnose"]["checkpoints"].items():
        weights = mx.load(str(Path(path) / "qsa-indexers.safetensors"))
        config = json.loads((Path(path) / "config.json").read_text()).get("qsa", {})
        selections[name] = config.get("selection")  # the hybrid rule it was trained with
        loaded = {}
        for (i, idx) in indexers.items():
            clone = type(idx)(idx.q_proj.weight.shape[1], heads=idx._geometry[0],
                              head_dim=idx._geometry[1], rope_dim=idx._geometry[2],
                              rope_base=idx._geometry[3], eps=idx._geometry[4])
            prefix = f"model.layers.{i}.self_attn._qsa_indexer."
            clone.update({"q_proj": {"weight": weights[prefix + "q_proj.weight"]},
                          "k_proj": {"weight": weights[prefix + "k_proj.weight"]},
                          "q_norm": weights[prefix + "q_norm"],
                          "k_norm": weights[prefix + "k_norm"],
                          "k_block_norm": weights[prefix + "k_block_norm"]})
            loaded[i] = clone
        arms[name] = loaded
    per_layer: dict = {}
    for (r, row) in enumerate(holdout):
        rec = record(mx, model, attentions, row.tolist(), spec["prefill_step"])
        for (i, (x, q, k)) in rec.items():
            target = qt.block_teacher(q, k, scale)
            xf = x.astype(mx.float32)
            scores = {name: arm[i].scores(xf) for (name, arm) in arms.items()}
            row_out = qt.mass_breakdown(target, scores, topk, selections=selections)
            row_out["teacher_entropy"] = qt.teacher_entropy(target)
            per_layer.setdefault(i, []).append(row_out)
            print(json.dumps({"row": r, "layer": i, **{k: round(v, 4) if isinstance(v, float) else v
                                                         for (k, v) in row_out.items()}}),
                  flush=True)
            del target, xf, scores
        del rec
        mx.clear_cache()
    summary = {}
    for (i, rows) in per_layer.items():
        keys = [k for k in rows[0] if k != "rows"]
        summary[str(i)] = {k: float(np.mean([row[k] for row in rows])) for k in keys}
    keys = list(next(iter(summary.values())))
    summary["mean"] = {k: float(np.mean([v[k] for v in summary.values()])) for k in keys}
    return {"per_layer": summary, "rows": [len(v) for v in per_layer.values()][0]}


def unit(spec) -> int:
    from mlx2.desktop_campaign_worker import check_ownership

    out = Path(spec["out"])
    report = {"checkpoint": spec["checkpoint"], "source": source(),
              "ownership": check_ownership(spec), "spec": spec,
              "scope": "exploratory indexer distillation; not qualification",
              "qualified": False, "started": time.time()}
    log = open(out / "train.jsonl", "a", buffering=1)
    try:
        import mlx.core as mx
        import mlx.optimizers as optim
        import numpy as np
        from mlx import nn

        mx.set_default_device(mx.gpu)
        mx.set_cache_limit(4 << 30)
        from mlx2.adapters.registry import inspect_model, resolve_adapter
        from mlx2.runtime.models import qsa_indexer_train as qt

        resolution = inspect_model(spec["model"])
        if resolution.artifact_fingerprint != spec["fingerprint"]:
            raise ValueError("artifact changed since inventory")
        data_path = Path(spec["data"]["npy"])
        if sha256(data_path) != spec["data"]["sha256"]:
            raise ValueError("training data changed")
        data = np.load(data_path)
        holdout = data[-spec["data"]["holdout"]:]
        train = data[: -spec["data"]["holdout"]]
        if spec["data"].get("truncate"):  # smoke runs: shorter rows, same data
            holdout = holdout[:, : spec["data"]["truncate"]]
            train = train[:, : spec["data"]["truncate"]]
        adapter = resolve_adapter(spec["model"])(spec["model"])
        model = adapter.model
        attentions = full_attention(model)
        install_capture(attentions)
        args = model.args
        geo = spec["indexer"]
        # Variants train side by side on the same recorded sequences (the
        # dense record is ~60% of a step).  Each: geometry overrides, and
        # optionally a hybrid selection whose fixed blocks the loss excludes.
        variants = spec.get("variants") or {"main": {}}
        mx.random.seed(spec["seed"])
        indexers, opts, fixed_cfg = {}, {}, {}
        for (name, var) in variants.items():
            g = {**geo, **{k: var[k] for k in geo if k in var}}
            indexers[name] = {i: qt.TrainableIndexer(
                args.hidden_size, heads=g["indexer_heads"], head_dim=g["indexer_head_dim"],
                rope_dim=g["rope_dim"], rope_base=g["rope_base"], eps=args.rms_norm_eps,
                temperature=var.get("temperature")) for i in sorted(attentions)}
            # bias_correction: torch AdamW (the reference) always corrects;
            # MLX defaults to no correction (0.45x the step size at step 1).
            opts[name] = {i: optim.AdamW(learning_rate=spec["lr"], betas=[0.9, 0.95],
                                         eps=1e-8, weight_decay=0.0, bias_correction=True)
                          for i in indexers[name]}
            fixed_cfg[name] = var.get("selection")  # {"sink_blocks", "recent_blocks"}
            sel = fixed_cfg[name]
            if sel and int(train.shape[1]) // qt.BLOCK <= sel["sink_blocks"] + sel["recent_blocks"]:
                # No block outside the fixed ones: the restricted KL has no row.
                raise ValueError(f"variant {name}: seq_len {int(train.shape[1])} leaves no block "
                                 "outside the hybrid sink/recent blocks")
            for idx in indexers[name].values():
                mx.eval(idx.parameters())
        scale = args.head_dim ** -0.5
        topk = spec["budget"] // 4
        report["layers"] = sorted(attentions)
        from mlx.utils import tree_flatten

        report["parameters_per_variant"] = sum(
            v.size for idx in next(iter(indexers.values())).values()
            for (_, v) in tree_flatten(idx.parameters()))

        if spec.get("diagnose"):
            report["diagnose"] = diagnose(mx, np, qt, model, attentions,
                                          next(iter(indexers.values())), holdout,
                                          spec, scale, topk)
            report["passed"] = True
            raise StopIteration

        def save(directory: Path, step: int):
            for (name, var) in variants.items():
                d = directory / name if len(variants) > 1 else directory
                d.mkdir(parents=True, exist_ok=True)
                tensors = {}
                for (i, idx) in indexers[name].items():
                    tensors.update(idx.export(f"model.layers.{i}.self_attn"))
                mx.save_safetensors(str(d / "qsa-indexers.safetensors"), tensors)
                # Everything an exact resume needs that inference does not:
                # the training temperature and each layer's AdamW state.
                mx.save_safetensors(str(d / "trainer-state.safetensors"),
                                    qt.trainer_state(indexers[name], opts[name]))
                g = {**geo, **{k: var[k] for k in geo if k in var}}
                atomic(d / "config.json", {"qsa": {
                    **g, "eps": args.rms_norm_eps, "block_size": 4, "budget": spec["budget"],
                    "selection": fixed_cfg[name], "variant": name,
                    "training_temperature": {str(i): idx.temperature()
                                             for (i, idx) in indexers[name].items()},
                    "base_checkpoint": spec["checkpoint"], "layers": sorted(attentions),
                    "trained_steps": step, "seq_len": int(train.shape[1]),
                    "data": spec["data"], "recipe": "stage-1 dense block-KL distillation"
                    + (" with hybrid-restricted KL" if fixed_cfg[name] else ""),
                    "source": report["source"]}})

        def fixed_mask(name, length):
            cfg = fixed_cfg[name]
            if not cfg:
                return None
            return qt.fixed_blocks(length, cfg["sink_blocks"], cfg["recent_blocks"])

        def evaluate(step: int) -> dict:
            rows = {}
            for row in holdout:
                rec = record(mx, model, attentions, row.tolist(), spec["prefill_step"])
                for (i, (x, q, k)) in rec.items():
                    target = qt.block_teacher(q, k, scale)
                    xf = x.astype(mx.float32)
                    scores = {name: indexers[name][i].scores(xf) for name in variants}
                    br = qt.mass_breakdown(target, scores, topk, selections=fixed_cfg)
                    for name in variants:
                        br[f"kl:{name}"] = float(qt.kl_loss(
                            indexers[name][i].loss_scores(xf), target,
                            fixed_mask(name, xf.shape[0])).item())
                        br[f"temp:{name}"] = indexers[name][i].temperature()
                    for (key, val) in br.items():
                        if key != "rows":
                            rows.setdefault(key, {}).setdefault(str(i), []).append(val)
                    del target, xf, scores
                del rec
                mx.clear_cache()
            by_layer = {key: {i: float(np.mean(v)) for (i, v) in per.items()}
                        for (key, per) in rows.items()}
            out = {"eval_step": step}
            for (key, per) in by_layer.items():
                out[key] = round(float(np.mean(list(per.values()))), 4)
            out["by_layer"] = by_layer
            return out

        start_step = 0
        resume = spec.get("resume")
        if resume:
            start_step = int(resume["step"])
            report["resume"] = {"from": resume["from"], "step": start_step, "variants": {}}
            for name in variants:
                src = Path(resume["from"]) / name if len(variants) > 1 else Path(resume["from"])
                report["resume"]["variants"][name] = qt.load_trainer_checkpoint(
                    src, indexers[name], opts[name], step=start_step)
        history = []
        if spec.get("eval_at_start", True) and not resume:
            ev = evaluate(0)
            log.write(json.dumps(ev) + "\n")
            print(json.dumps({k: v for k, v in ev.items() if k != "by_layer"}), flush=True)
            history.append(ev)
        steps = min(spec["steps"], len(train))
        started = time.time()
        done = saved = start_step  # last applied update / last checkpoint
        try:
            for step in range(start_step + 1, steps + 1):
                t0 = time.time()
                rec = record(mx, model, attentions, train[step - 1].tolist(), spec["prefill_step"])
                t1 = time.time()
                grads = {name: {} for name in variants}
                losses = {name: [] for name in variants}
                ents = []
                for (i, (x, q, k)) in rec.items():
                    target = qt.block_teacher(q, k, scale)
                    xf = x.astype(mx.float32)
                    for name in variants:
                        idx = indexers[name][i]
                        fixed = fixed_mask(name, xf.shape[0])
                        fn = nn.value_and_grad(
                            idx, lambda m, xf=xf, target=target, fixed=fixed:
                            qt.kl_loss(m.loss_scores(xf), target, fixed))
                        (loss, g) = fn(idx)
                        mx.eval(loss, g)
                        losses[name].append(float(loss.item()))
                        grads[name][i] = g
                        del fn, fixed
                    if step % spec["log_every"] == 0 or step == 1:
                        ents.append(qt.teacher_entropy(target))
                    del target, xf
                del rec
                mx.clear_cache()
                row = {"step": step, "record_s": round(t1 - t0, 1),
                       "elapsed_s": round(time.time() - started, 1),
                       "tokens": step * int(train.shape[1])}
                clipped_by = {}
                for name in variants:
                    order = sorted(grads[name])
                    (clipped, norm) = qt.global_clip([grads[name][i] for i in order], 1.0)
                    if not math.isfinite(norm):
                        # Never apply a non-finite step (it would turn every weight NaN).
                        raise FloatingPointError(f"step {step} variant {name}: grad_norm {norm}")
                    clipped_by[name] = (order, clipped, norm)
                _SIGNALS["defer"] = True
                for name in variants:
                    (order, clipped, norm) = clipped_by[name]
                    for (i, g) in zip(order, clipped):
                        opts[name][i].update(indexers[name][i], g)
                        mx.eval(indexers[name][i].parameters(), opts[name][i].state)
                    row[f"loss:{name}"] = float(np.mean(losses[name]))
                    row[f"grad_norm:{name}"] = norm
                    row[f"loss_by_layer:{name}"] = [round(v, 4) for v in losses[name]]
                del grads
                done = step
                _SIGNALS["defer"] = False
                if _SIGNALS["pending"] is not None:
                    raise KeyboardInterrupt(f"signal {_SIGNALS['pending']} (held to step end)")
                row["step_s"] = round(time.time() - t0, 1)
                if ents:
                    row["teacher_entropy"] = float(np.mean(ents))
                log.write(json.dumps(row) + "\n")
                print(json.dumps({k: v for k, v in row.items() if "by_layer" not in k}), flush=True)
                if step % spec["eval_every"] == 0 or step == steps:
                    ev = evaluate(step)
                    log.write(json.dumps(ev) + "\n")
                    print(json.dumps({k: v for k, v in ev.items() if k != "by_layer"}), flush=True)
                    history.append(ev)
                    save(out / "ckpt" / f"step{step:04d}", step)
                    saved = step
                    report["history"] = history
                    atomic(out / "report.partial.json", report)
        except KeyboardInterrupt:
            # A stop between checkpoints keeps every applied step.
            if done > saved:
                save(out / "ckpt" / f"step{done:04d}-interrupt", done)
                report["interrupt_checkpoint"] = f"ckpt/step{done:04d}-interrupt"
            raise
        save(out / "indexers", steps)
        report["indexers_sha256"] = {
            str(f.relative_to(out)): sha256(f)
            for f in sorted((out / "indexers").rglob("qsa-indexers.safetensors"))}
        report["history"] = history
        report["steps"] = steps
        report["passed"] = True
    except StopIteration:
        pass
    except BaseException as error:  # noqa: BLE001 - evidence first
        report["passed"] = False
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    report["finished"] = time.time()
    atomic(out / "report.json", report)
    log.close()
    print(json.dumps({"passed": report["passed"], "error": report.get("error")}), flush=True)
    return 0 if report["passed"] else 1


def main() -> int:
    def interrupted(signum, frame):
        if _SIGNALS["defer"]:
            _SIGNALS["pending"] = signum
            return
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
