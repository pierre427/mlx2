"""Inkling 2 loop-pass probe: does each pass reuse the same experts?

The middle block (layers ``core_start..core_end``) runs three times with
shared weights. A pass can only tell which pass it is through the
previous-pass message. This probe prefills a mixed corpus and records, for
every layer visit, the router's top-k experts and the loop controllers' gate
statistics. Two questions:

- Expert reuse: per-token Jaccard of the top-k sets of the same layer across
  passes, against a shuffled-token baseline. It also looks at per-pass load
  histograms and at experts whose load is mostly one pass's.
- Message conditioning: mean |tanh| of the forget and gain gates, and the
  controller's relative update ||out - x|| / ||x||, by pass and input.

This is an interpretability probe, not a qualification or performance run.
The receipt names ``<artifact>@<fingerprint>`` and the mlx2 commit.

Usage: python scripts/research/inkling2_pass_probe.py <model dir> <run dir>
       [--tokens-per-doc 1024] [--gsm8k 24]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

PROSE = ("README.md", "docs/INKLING2-MINI.md", "docs/QUALIFICATION.md", "benchmarks/gsm8k/README.md")
CODE = ("src/mlx2/runtime/models/inkling2/language.py", "src/mlx2/adapters/inkling2.py",
        "src/mlx2/runtime/drafters/inkling2_dspark.py")


def git_commit():
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain",
                                 "--untracked-files=no"], capture_output=True,
                                text=True).stdout.strip())
    return {"commit": head, "dirty": dirty}


def write_atomic(path, data):
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(data, indent=2, default=str) + "\n")
    os.replace(tmp, path)


def corpus(tokenizer, render_chat, tokens_per_doc, gsm8k):
    docs = []
    for kind, names in (("prose", PROSE), ("code", CODE)):
        for name in names:
            path = ROOT / name
            if path.is_file():
                ids = tokenizer.encode(path.read_text(), add_special_tokens=False)
                docs.append({"kind": kind, "source": name, "tokens": ids[:tokens_per_doc]})
    with open(ROOT / "benchmarks/gsm8k/test.jsonl") as handle:
        for i, line in zip(range(gsm8k), handle):
            question = json.loads(line)["question"]
            docs.append({"kind": "gsm8k_chat", "source": f"gsm8k/test.jsonl:{i}",
                         "tokens": render_chat(question)})
    return docs


class Recorder:
    """Class-level hooks on the Inkling 2 modules; visit = call order."""

    def __init__(self, language):
        self.lang = language
        self.visit = -1
        self.routes = {}  # visit -> list of [T, k] int arrays
        self.gates = {}  # (visit, name) -> list of [T, 3] (forget, gain, rel update)
        self.kind = None

    def __enter__(self):
        lang, rec = self.lang, self
        self._layer_call = lang.DecoderLayer.__call__
        self._route = lang._route
        self._controller_call = lang.MessageController.__call__
        layer_call, route = self._layer_call, self._route

        def layer(module, *args, **kwargs):
            rec.visit += 1
            rec.names = iter(("attn", "k", "v", "mlp"))
            return layer_call(module, *args, **kwargs)

        def routed(x, *args, **kwargs):
            indices, weights = route(x, *args, **kwargs)
            rec.routes.setdefault(rec.visit, []).append(
                (rec.kind, np.array(indices.reshape(-1, indices.shape[-1]))))
            return indices, weights

        def controller(module, x, message):
            import mlx.core as mx

            pair = mx.concatenate(
                [mx.fast.rms_norm(v, None, module.eps) for v in (x, message)], axis=-1)
            hidden = (pair.astype(module.down_weight.dtype) @ module.down_weight.T).astype(
                mx.float32)
            forget = mx.tanh(hidden @ lang._fp32(module, "forget_weight").T)
            gain = mx.tanh(hidden @ lang._fp32(module, "gain_weight").T)
            out = (1 - forget) * x + gain * message
            xf = x.astype(mx.float32)
            rel = mx.linalg.norm(out.astype(mx.float32) - xf, axis=-1) / (
                mx.linalg.norm(xf, axis=-1) + 1e-6)
            stats = mx.stack([mx.abs(forget).mean(-1), mx.abs(gain).mean(-1), rel], axis=-1)
            name = next(rec.names, "?")
            rec.gates.setdefault((rec.visit, name), []).append(
                (rec.kind, np.array(stats.reshape(-1, 3))))
            return out

        lang.DecoderLayer.__call__ = layer
        lang._route = routed
        lang.MessageController.__call__ = controller
        return self

    def __exit__(self, *exc):
        self.lang.DecoderLayer.__call__ = self._layer_call
        self.lang._route = self._route
        self.lang.MessageController.__call__ = self._controller_call

    def forward(self, model, tokens, chunk, kind):
        import mlx.core as mx

        cache = model.make_cache()
        self.kind = kind
        for start in range(0, len(tokens), chunk):
            self.visit = -1
            hidden = model.model(mx.array([tokens[start:start + chunk]]), cache)
            mx.eval(hidden)
            visits = len(model.model.layer_indices)
            if self.visit != visits - 1:
                raise RuntimeError(f"visit counter {self.visit + 1} != {visits}")


def _jaccard(a, b):
    """Row-wise |a & b| / |a | b| for [T, k] expert-index arrays."""
    k = a.shape[1]
    inter = (a[:, :, None] == b[:, None, :]).any(-1).sum(-1)
    return inter / (2 * k - inter)


def analyze(rec, config, kinds):
    loop = config.loop_config
    core = list(range(loop.core_start, loop.core_end))
    span = loop.core_end - loop.core_start
    n_exp = config.n_routed_experts
    rng = np.random.default_rng(0)

    def stacked(visit, kind=None):
        rows = [a for k, a in rec.routes[visit] if kind is None or k == kind]
        return np.concatenate(rows, axis=0)

    layers = []
    for layer in core:
        visits = [layer + p * span for p in range(loop.num_passes)]
        sets = [stacked(v) for v in visits]
        loads = np.stack([np.bincount(s.ravel(), minlength=n_exp) for s in sets]).astype(float)
        share = loads / np.maximum(loads.sum(0, keepdims=True), 1)
        probs = loads / loads.sum(1, keepdims=True)
        pairs = {}
        for a in range(len(visits)):
            for b in range(a + 1, len(visits)):
                same = _jaccard(sets[a], sets[b])
                shuffled = _jaccard(sets[a], sets[b][rng.permutation(len(sets[b]))])
                mid = (probs[a] + probs[b]) / 2

                def kl(p, q):
                    m = p > 0
                    return float((p[m] * np.log(p[m] / q[m])).sum())

                pairs[f"{a + 1}-{b + 1}"] = {
                    "jaccard_same_token": round(float(same.mean()), 4),
                    "jaccard_shuffled_baseline": round(float(shuffled.mean()), 4),
                    "identical_topk_fraction": round(float((same == 1).mean()), 4),
                    "load_js_divergence_nats": round(0.5 * kl(probs[a], mid)
                                                    + 0.5 * kl(probs[b], mid), 4),
                    "by_kind": {kind: round(float(_jaccard(stacked(visits[a], kind),
                                                           stacked(visits[b], kind)).mean()), 4)
                                for kind in kinds},
                }
        used = loads.sum(0) > 0
        layers.append({
            "layer": layer, "visits": visits, "tokens": int(len(sets[0])), "pairs": pairs,
            "experts_used": int(used.sum()),
            "pass_dominant_experts": {  # load share > 0.6 from one pass (uniform 1/3)
                str(p + 1): int(((share[p] > 0.6) & used).sum()) for p in range(len(visits))},
        })
    gates = {}
    for (visit, name), rows in sorted(rec.gates.items()):
        data = np.concatenate([a for _, a in rows], axis=0)
        layer = config.layer_indices[visit]
        gates.setdefault(str(layer), {})[f"pass{(visit - layer) // span + 1}.{name}"] = {
            "visit": visit,
            "mean_abs_forget": round(float(data[:, 0].mean()), 4),
            "mean_abs_gain": round(float(data[:, 1].mean()), 4),
            "mean_rel_update": round(float(data[:, 2].mean()), 4),
            "p90_rel_update": round(float(np.quantile(data[:, 2], 0.9)), 4),
        }
    summary = {
        f"mean_jaccard_{p}": round(float(np.mean([l["pairs"][p]["jaccard_same_token"]
                                                  for l in layers])), 4)
        for p in layers[0]["pairs"]
    }
    summary.update({
        f"mean_shuffled_{p}": round(float(np.mean([l["pairs"][p]["jaccard_shuffled_baseline"]
                                                   for l in layers])), 4)
        for p in layers[0]["pairs"]
    })
    return {"summary": summary, "layers": layers, "controllers": gates}


def run(model, tokenizer, render_chat, *, tokens_per_doc, gsm8k, chunk=512):
    from mlx2.runtime.models.inkling2 import language

    docs = corpus(tokenizer, render_chat, tokens_per_doc, gsm8k)
    with Recorder(language) as rec:
        for doc in docs:
            rec.forward(model, doc["tokens"], chunk, doc["kind"])
    kinds = sorted({d["kind"] for d in docs})
    result = analyze(rec, model.config, kinds)
    result["corpus"] = [{"kind": d["kind"], "source": d["source"], "tokens": len(d["tokens"])}
                        for d in docs]
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", type=Path)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--tokens-per-doc", type=int, default=1024)
    parser.add_argument("--gsm8k", type=int, default=24)
    a = parser.parse_args(argv)
    a.run_dir.mkdir(parents=True, exist_ok=True)
    if any(a.run_dir.iterdir()):
        raise SystemExit(f"{a.run_dir} is not empty; use a fresh run directory")
    sys.path.insert(0, str(ROOT / "src"))
    import mlx.core as mx

    from mlx2.adapters.inkling2 import Inkling2Adapter

    receipt = {"script": "scripts/research/inkling2_pass_probe.py", "mlx2": git_commit(),
               "mlx_version": mx.__version__, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "kind": "interpretability probe (not qualification, not performance)"}
    adapter = Inkling2Adapter(str(a.model))
    try:
        receipt["checkpoint"] = adapter.checkpoint
        receipt["route_receipt"] = adapter.diagnostics()

        def chat(question):
            return adapter.prompt_tokens({"messages": [{"role": "user", "content": question}],
                                          "reasoning_effort": "low"})

        t0 = time.monotonic()
        receipt["probe"] = run(adapter.model, adapter.tokenizer, chat,
                               tokens_per_doc=a.tokens_per_doc, gsm8k=a.gsm8k)
        receipt["seconds"] = round(time.monotonic() - t0, 1)
    finally:
        adapter.close()
        mx.clear_cache()
    receipt["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    write_atomic(a.run_dir / "pass_probe.json", receipt)
    print(json.dumps({"checkpoint": receipt["checkpoint"], "mlx2": receipt["mlx2"],
                      "summary": receipt["probe"]["summary"]}, default=str))


if __name__ == "__main__":
    main()
