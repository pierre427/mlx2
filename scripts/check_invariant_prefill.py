"""Slice invariance of the opt-in invariant prefill lane (bit level, Metal).

Loads the artifact once with ``{"invariant_prefill": true}`` (plus
``--policy``), prefills one real prompt on a fresh cache under several slice
schedules (``--schedules``: a row count = fixed slices of that size, ``all`` =
one forward) for each arm (``--arms``: ``on`` = lane enabled, ``off`` = lane
installed but disabled, i.e. the stock kernels), then greedy-decodes
``--decode`` tokens one row at a time (stock decode in every arm).

Per arm, every schedule is compared with the arm's first schedule: the last
prompt row's trunk hidden state and logits, every prompt row's trunk output,
every cache state array, and the greedy tokens.  ``--components`` also
records a per-row checksum of every instrumented module output (projections,
norms, attention, GDN, MoE, PLE, HC) and reports the first stage, in
execution order, whose rows differ -- the component that breaks invariance.

  MLX_ENABLE_TF32=0 PYTHONPATH=src .venv/bin/python scripts/check_invariant_prefill.py \\
      --i-own-the-gpu --model ~/mlx-models/Qwen3.8-Flash-Next-Uncensored-MLX2-4bit-MTP \\
      --schedules 512 64 256 448 1024 2048 --arms on off --out inv.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class Recorder:
    """Per-row checksums of instrumented module outputs, per forward."""

    def __init__(self, mx):
        self.mx = mx
        self.on = False
        self.paths = {}
        self.rows = None
        self.calls = {}
        self.slices = {}  # key -> list of [B, rows] uint32

    def begin(self, rows):
        self.rows = rows
        self.calls = {}

    def checksum(self, a):
        mx = self.mx
        if a.dtype in (mx.bfloat16, mx.float16):
            u = a.view(mx.uint16).astype(mx.uint32)
        elif a.dtype == mx.float32:
            u = a.view(mx.uint32)
        else:
            u = a.astype(mx.uint32)
        flat = u.reshape(a.shape[0], a.shape[1], -1)
        width = flat.shape[-1]
        weights = mx.arange(width, dtype=mx.uint32) * mx.array(2654435761, mx.uint32) + mx.array(
            97, mx.uint32
        )
        return mx.sum(flat * weights, axis=-1)

    def add(self, module, out):
        if not self.on:
            return
        path = self.paths.get(id(module), type(module).__name__)
        if ".ple." in path:
            return  # traced inside the compiled PLE chain: no concrete arrays
        n = self.calls.get(path, 0)
        self.calls[path] = n + 1
        items = out if isinstance(out, (tuple, list)) else (out,)
        for index, item in enumerate(items):
            array = getattr(item, "raw_block_ids", item)
            if array is None or not hasattr(array, "ndim"):
                continue
            if array is not item:
                array = self.mx.sort(array, axis=-1)
            if array.ndim < 2 or array.shape[1] != self.rows:
                continue
            key = f"{path}#{n}.{index}"
            self.slices.setdefault(key, []).append(self.checksum(array))

    def take(self):
        out = {k: self.mx.concatenate(v, axis=1) for k, v in self.slices.items()}
        self.slices = {}
        return out


def instrument(model, recorder):
    import mlx.nn as nn
    from mlx2.runtime.models import invariant_prefill as inv
    from mlx2.runtime.models import qwen4_exp as q4
    from mlx2.runtime.models import qwen3_next as q3n
    from mlx2.runtime.models import qwen3_5 as q35

    recorder.paths = {id(m): name for name, m in model.named_modules()}
    classes = [
        nn.Linear, nn.QuantizedLinear, nn.RMSNorm, inv.InvariantLinear,
        inv.InvariantQuantizedLinear, q4.GroupRMSNorm, q4.DecoderLayer, q4.Attention,
        q4.GatedDeltaNet, q35.GatedDeltaNet, q4.PLELayer, q4.GatedResidual,
        q3n.Qwen3NextSparseMoeBlock, q3n.Qwen3NextMLP, q3n.FusedDownSwitchGLU,
    ]
    for cls in classes:
        if "__call__" not in cls.__dict__:
            continue
        original = cls.__dict__["__call__"]

        def recorded(self, *args, __original=original, **kwargs):
            out = __original(self, *args, **kwargs)
            recorder.add(self, out)
            return out

        cls.__call__ = recorded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=8192)
    ap.add_argument("--schedules", nargs="+", default=["512", "64", "256", "448", "1024", "2048"])
    ap.add_argument("--arms", nargs="+", default=["on"], choices=["on", "off"])
    ap.add_argument("--policy", default="{}")
    ap.add_argument("--decode", type=int, default=16)
    ap.add_argument("--components", action="store_true")
    ap.add_argument("--max-swapout-pages", type=int, default=20000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("refusing Metal execution without --i-own-the-gpu")

    from mlx2.adapters.registry import resolve_adapter

    policy = {**json.loads(a.policy), "invariant_prefill": True}
    adapter = resolve_adapter(a.model, mtp=True, qualification_mode=True)(
        a.model, execution_policy=policy
    )
    import mlx.core as mx

    model = adapter.model
    handle = adapter.invariant_prefill
    trunk = model.language_model.model
    mx.eval(model.parameters())
    mx.set_cache_limit(4 << 30)
    recorder = Recorder(mx)
    if a.components:
        instrument(model, recorder)
    tok = adapter.tokenizer
    corpus = (ROOT / "docs" / "SERVING.md").read_text()
    ids = tok.encode(corpus, add_special_tokens=False)[: a.prompt_tokens - 64]
    prompt = list(adapter.prompt_tokens({"messages": [{"role": "user", "content":
                  "Summarise the following excerpt in three sentences.\n\n" + tok.decode(ids)}]}))
    n = len(prompt)

    def run(schedule):
        step = n if schedule == "all" else int(schedule)
        cache = model.make_cache()
        hidden_rows = []
        pos = 0
        slices = []
        recorder.on = a.components
        mx.synchronize()
        started = time.perf_counter()
        while pos < n:
            end = min(n, pos + step)
            recorder.begin(end - pos)
            hidden = trunk(mx.array([prompt[pos:end]], mx.uint32), cache)
            hidden_rows.append(recorder.checksum(hidden))
            mx.eval(hidden, hidden_rows[-1], [c.state for c in cache],
                    [v[-1] for v in recorder.slices.values()])
            slices.append(end - pos)
            pos = end
        prefill_s = time.perf_counter() - started
        recorder.on = False
        last_hidden = hidden[:, -1, :]
        logits = model.logits(hidden[:, -1:, :])[:, -1, :].astype(mx.float32)
        rows = mx.concatenate(hidden_rows, axis=1)
        state = []
        for c in cache:
            try:
                state.extend(x for x in c.state if hasattr(x, "shape"))
            except Exception:  # noqa: BLE001 - a cache without a state view
                pass
        mx.eval(last_hidden, logits, rows, state)
        out = []
        nxt = int(mx.argmax(logits, -1).item())
        for _ in range(a.decode):
            out.append(nxt)
            step_logits = model(mx.array([[nxt]], mx.uint32), cache=cache)
            nxt = int(mx.argmax(step_logits[:, -1, :], -1).item())
        components = recorder.take()
        del cache
        mx.clear_cache()
        return {"last_hidden": last_hidden, "logits": logits, "rows": rows,
                "state": state, "tokens": out, "slices": slices,
                "prefill_s": prefill_s, "components": components}

    def common(x, y):
        """Both arrays cut to their common extent (step-grown KV buffers
        differ in capacity, not in the rows written)."""
        if x.ndim != y.ndim:
            return None
        index = tuple(slice(0, min(p, q)) for p, q in zip(x.shape, y.shape))
        return x[index], y[index]

    def state_diffs(xs, ys):
        out = []
        for i, (x, y) in enumerate(zip(xs, ys)):
            if x.shape == y.shape and mx.array_equal(x, y).item():
                continue
            pair = common(x, y)
            differing = None if pair is None else int(mx.sum(pair[0] != pair[1]).item())
            if differing == 0:
                continue  # capacity only; every common element equal
            out.append({"index": i, "shapes": [list(x.shape), list(y.shape)],
                        "dtype": str(x.dtype), "elements_differing": differing})
        if len(xs) != len(ys):
            out.append({"count_mismatch": [len(xs), len(ys)]})
        return out

    def swapouts():
        import subprocess

        text = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        line = next(l for l in text.splitlines() if l.startswith("Swapouts"))
        return int(line.split(":")[1].strip().rstrip("."))

    swap0 = swapouts()
    results = {"model": a.model, "prompt_tokens": n, "mlx": mx.__version__,
               "policy": policy, "lane": handle.status(), "arms": {}}
    for arm in a.arms:
        handle.enabled = arm == "on"
        from mlx2.runtime.models import invariant_prefill as inv

        inv.status(reset=True)
        ref = None
        arm_out = {}
        for schedule in a.schedules:
            r = run(schedule)
            entry = {"slices": len(r["slices"]), "prefill_s": round(r["prefill_s"], 3),
                     "tokens": r["tokens"]}
            if ref is None:
                ref = (schedule, r)
            else:
                b = ref[1]
                entry.update({
                    "vs": ref[0],
                    "logits_identical": bool(mx.array_equal(r["logits"], b["logits"]).item()),
                    "logits_max_abs_diff": float(mx.max(mx.abs(r["logits"] - b["logits"])).item()),
                    "last_hidden_identical": bool(mx.array_equal(r["last_hidden"], b["last_hidden"]).item()),
                    "rows_differing": int(mx.sum(r["rows"] != b["rows"]).item()),
                    "state_arrays": len(r["state"]),
                    "state_arrays_differing": state_diffs(r["state"], b["state"]),
                    "tokens_identical": r["tokens"] == b["tokens"],
                    "first_token_diff": next(
                        (i for i, (x, y) in enumerate(zip(r["tokens"], b["tokens"])) if x != y), None
                    ),
                })
                if a.components:
                    mism = []
                    for key, ref_rows in b["components"].items():
                        got = r["components"].get(key)
                        if got is None or got.shape != ref_rows.shape:
                            mism.append({"key": key, "missing_or_shape": True})
                            continue
                        diff = (got != ref_rows).reshape(-1)
                        count = int(mx.sum(diff).item())
                        if count:
                            mism.append({"key": key, "rows": count,
                                         "first_row": int(mx.argmax(diff).item())})
                    extra = [k for k in r["components"] if k not in b["components"]]
                    entry["component_mismatches"] = len(mism)
                    entry["first_mismatches"] = mism[:40]
                    entry["extra_keys"] = extra[:10]
                    entry["components_compared"] = len(b["components"])
            entry["swapouts_delta_pages"] = swapouts() - swap0
            arm_out[schedule] = entry
            print(arm, schedule, json.dumps({k: (len(v) if k == "state_arrays_differing" else v)
                                             for k, v in entry.items()
                                             if k not in ("tokens", "first_mismatches")}), flush=True)
            if entry.get("first_mismatches"):
                print("   first:", json.dumps(entry["first_mismatches"][:6]), flush=True)
            if entry["swapouts_delta_pages"] > a.max_swapout_pages:
                json.dump(results, open(a.out, "w"), indent=1)
                raise SystemExit(f"aborting: swapouts grew by {entry['swapouts_delta_pages']} pages")
        arm_out["lane_counters"] = inv.status()
        results["arms"][arm] = arm_out
        json.dump(results, open(a.out, "w"), indent=1)
    json.dump(results, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
