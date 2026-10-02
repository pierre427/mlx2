#!/usr/bin/env python3
"""In-process interleaved A/B of Flash-Next options against the served default.

One process, one model load (served policy: ``num_draft`` 2, copy drafts on,
adaptive depth and handoff off).  Each option arm is switched on and off
in-process through the runtime's own setters (``TOGGLES``); the default arm
runs with every lever at its served value.  Per config, every rep runs all
arms in a rotated order (odd reps reversed) after one discarded warm-up rep;
rep ``r`` uses the prompt set at ``offset[r]`` (content-varied reps), so each
pair (arm, default) shares its prompts.

Configs:
  mtp:1        B=1 native MTP (the served route), 4 chat prompts rotate per rep
  ord:1        B=1 ordinary decode, same prompts
  mtp:N        N lanes native MTP, N distinct ~1K-token document prompts
  ord:N        N lanes ordinary decode
  long:T       B=1 native MTP on one T-token document prompt; reports decode
               tok/s and prefill tok/s (TTFT) per run

Decode tok/s counts tokens from the moment every lane has produced its first
token to the last token (prefill excluded).  Reports per arm: median,
paired delta vs default per rep (median [min, max]), token identity vs the
default arm in the same rep, and engagement counters (adapter diagnostics
deltas filtered to the option's keys).

  gpuq.sh perf-x env PYTHONPATH=src MLX_ENABLE_TF32=0 .venv/bin/python \\
      scripts/flash_next_options_perf.py --i-own-the-gpu --arms default topk:fold \\
      --configs mtp:1 ord:1 --reps 6 --out ab.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from flash_next_options_sweep import (  # noqa: E402
    EXTRA_PROMPTS, FLY, HANDOFF, MODEL, Harness, delta, flatten, swapouts,
)


# Levers that take no value: naming one switches it on.
SWITCH_LEVERS = frozenset({
    "dynamic_accept", "router_kernel", "nax_decode", "gdn_core", "indexed_merge",
    "indexed_gate", "gate_inject", "row_exact", "tf_qmv", "fp32_head", "adaptive",
    "adaptive_single", "fly", "copy_off",
})


class Toggles:
    """In-process levers; ``set(arm)`` applies one arm, everything else default."""

    def __init__(self, h):
        self.h = h
        model = h.adapter.model
        self.model = model
        self.blocks = [m for _, m in model.named_modules() if hasattr(m, "set_moe_routed_decode_mode")]
        self.gdn = [m for _, m in model.named_modules() if hasattr(m, "set_fused_gdn_dynamic_accept")]
        self.router = [m for _, m in model.named_modules() if hasattr(m, "set_moe_router_mode")]
        from mlx2.runtime.models import qwen4_hc_decode as HCD
        from mlx2.runtime.models import qwen4_fused_gdn_verify as FV
        from mlx2.runtime.models import qwen4_gate_inject as GI
        from mlx2.runtime.models import qwen4_attn_window as AW
        from mlx2.runtime.models import qwen4_exp as QE
        from mlx2.runtime.models import gated_delta as GD

        self.HCD, self.FV, self.GI, self.AW, self.QE, self.GD = HCD, FV, GI, AW, QE, GD
        self.default = {
            "routed": self.blocks[0].switch_mlp.routed_decode_mode,
            "topk": self.blocks[0].moe_topk_mode,
            "hc_multi_row": "auto",
            "router": self.router[0].moe_router_mode if self.router else "stock",
            "verify_max_steps": h.adapter.policy.fused_gdn_verify_max_steps,
            "nax": QE._QSA_NAX_DECODE,
            "gdn_core": GD._ENABLE_GDN_CORE,
        }
        self.row_exact = None
        self.tf_qmv = None
        self.fp32 = None
        self.scan = None
        self.current = None
        self.run_kwargs = {}

    # -- individual levers ------------------------------------------------
    def _row_exact(self, on):
        if on and self.row_exact is None:
            from mlx2.runtime.models.qwen4_row_exact import install

            self.row_exact = install(self.model)
        if self.row_exact is not None:
            self.row_exact.enable(on)
        self.AW.set_enabled(on)
        self.HCD.set_hc_row_exact_enabled(on)
        for b in self.blocks:
            b.set_moe_window_consumers({"row_exact"} if on else set())

    def _tf_qmv(self, on):
        from mlx2.runtime.models import flash_tensorfold_qmv as TQ

        if self.tf_qmv is None:
            classes = {id(m): (m, type(m)) for _, m in self.model.named_modules()}
            TQ.install(self.model)
            self.tf_qmv = [(m, cls, type(m)) for m, cls in classes.values() if type(m) is not cls]
        for module, plain, fold in self.tf_qmv:
            module.__class__ = fold if on else plain

    def _fp32(self, on):
        import mlx.core as mx

        head = self.h.adapter.model.language_model.lm_head
        if self.fp32 is None:
            self.fp32 = {"bf16": (head["scales"], head.get("biases"))}
            from mlx2.runtime.fp32_head import enable_fp32_head_logits

            enable_fp32_head_logits(self.h.adapter.model.language_model)
            self.fp32["fp32"] = (head["scales"], head.get("biases"))
        scales, biases = self.fp32["fp32" if on else "bf16"]
        head.scales = scales
        if biases is not None:
            head.biases = biases
        mx.eval(head.parameters())
        self.run_kwargs["fp32_head_logits"] = on

    def _scan(self, chunk):
        from mlx2.runtime.models.gated_delta import install_prefill_scan

        if chunk and self.scan is None:
            install_prefill_scan(self.model, chunk, 2048)
            self.scan = [m for _, m in self.model.named_modules() if getattr(m, "_prefill_scan_chunk", 0)]
        for m in self.scan or ():
            object.__setattr__(m, "_prefill_scan_chunk", chunk or 0)

    def reset(self):
        d = self.default
        for b in self.blocks:
            b.set_moe_routed_decode_mode(d["routed"])
            b.set_moe_topk_mode(d["topk"])
            b.set_moe_window_consumers(set())
        for g in self.gdn:
            g.set_fused_gdn_dynamic_accept(False)
        for r in self.router:
            r.set_moe_router_mode(d["router"])
        self.HCD.set_hc_multi_row_mode(d["hc_multi_row"])
        self.FV.set_verify_max_steps(d["verify_max_steps"])
        self.QE._QSA_NAX_DECODE = d["nax"]
        self.GD._ENABLE_GDN_CORE = d["gdn_core"]
        os.environ["MLX_QWEN4_QSA_INDEXED_FUSED_MERGE"] = "0"
        os.environ["MLX_QWEN4_QSA_INDEXED_FUSED_GATE"] = "0"
        self.GI.set_fused_gate_inject_enabled(False)
        if self.row_exact is not None:
            self._row_exact(False)
        if self.tf_qmv is not None:
            self._tf_qmv(False)
        if self.fp32 is not None:
            self._fp32(False)
        if self.scan is not None:
            self._scan(0)
        if getattr(self, "gdn_state_set", False):
            from mlx2.runtime.models.gdn_state import is_gdn_layer

            for _, m in self.model.named_modules():
                if is_gdn_layer(m) and "_gdn_state_dtype" in m.__dict__:
                    object.__delattr__(m, "_gdn_state_dtype")
            self.gdn_state_set = False
        self.run_kwargs = {}

    def set(self, arm):
        """Arm syntax: ``default`` or ``lever:value``, combined with ``&``."""
        self.reset()
        self.current = arm
        if arm == "default":
            return
        engine = {}
        for part in arm.split("&"):
            self._apply(part)
            engine.update(self.run_kwargs.pop("engine", {}))
        if engine:
            self.run_kwargs["engine"] = engine

    def _apply(self, part):
        lever, _, value = part.partition(":")
        if lever in SWITCH_LEVERS and value not in ("", "on", "1"):
            # These levers only switch something on; "gate_inject:off" used
            # to switch it ON.  Leave a lever out of the arm to keep it off.
            raise ValueError(
                f"lever {lever!r} takes no value (got {value!r}); "
                "omit it to keep it off"
            )
        if lever == "routed":
            for b in self.blocks:
                b.set_moe_routed_decode_mode(value)
        elif lever == "topk":
            for b in self.blocks:
                b.set_moe_topk_mode(value)
        elif lever == "moe_window":
            consumers = {{"batch": "batch_decode"}.get(c, c) for c in value.split("+")}
            for b in self.blocks:
                b.set_moe_window_consumers(consumers)
        elif lever == "hc_multi_row":
            self.HCD.set_hc_multi_row_mode(value)
        elif lever == "dynamic_accept":
            for g in self.gdn:
                g.set_fused_gdn_dynamic_accept(True)
        elif lever == "router_kernel":
            for r in self.router:
                r.set_moe_router_mode("fused")
        elif lever == "nax_decode":
            self.QE._QSA_NAX_DECODE = True
        elif lever == "gdn_core":
            self.GD._ENABLE_GDN_CORE = True
        elif lever == "indexed_merge":
            os.environ["MLX_QWEN4_QSA_INDEXED_FUSED_MERGE"] = "1"
        elif lever == "indexed_gate":
            os.environ["MLX_QWEN4_QSA_INDEXED_FUSED_GATE"] = "1"
        elif lever == "verify_max_steps":
            self.FV.set_verify_max_steps(int(value))
        elif lever == "gate_inject":
            self.GI.set_fused_gate_inject_enabled(True)
        elif lever == "row_exact":
            self._row_exact(True)
        elif lever == "tf_qmv":
            self._tf_qmv(True)
        elif lever == "fp32_head":
            self._fp32(True)
        elif lever == "gdn_scan":
            self._scan(int(value))
        elif lever == "gdn_state":
            # Binds the storage class to the layers exactly as the adapter
            # does at load; reset() removes the attribute again.
            from mlx2.runtime.models.gdn_state import STATE_DTYPES, is_gdn_layer

            for _, m in self.model.named_modules():
                if is_gdn_layer(m):
                    object.__setattr__(m, "_gdn_state_dtype", STATE_DTYPES[value])
            self.gdn_state_set = True
        elif lever == "depth_budget":
            self.run_kwargs["depth_budget"] = int(value)
        elif lever == "num_draft":
            self.run_kwargs["num_draft"] = int(value)
        elif lever == "prefill_step":
            self.run_kwargs["prefill_step"] = int(value)
        elif lever == "handoff":
            self.run_kwargs["engine"] = {"mtp_ordinary_handoff": dict(HANDOFF, max_mtp_width=int(value or 4))}
        elif lever == "adaptive":
            self.run_kwargs["engine"] = {"adaptive_mtp_depth": True}
        elif lever == "adaptive_single":
            self.run_kwargs["engine"] = {"adaptive_mtp_depth": {"enabled": True, "adaptive_single_lane": True}}
        elif lever == "fly":
            self.run_kwargs["engine"] = {"fly_verification": FLY}
        elif lever == "copy_off":
            self.run_kwargs["engine"] = {"copy_draft": False}
        else:
            raise ValueError(f"unknown lever {lever!r}")


def doc_prompts(h, ids, n, tokens, offset):
    """``n`` distinct real-document chat prompts of ~``tokens`` tokens each."""
    tok = h.adapter.tokenizer
    out = []
    for i in range(n):
        start = offset + i * tokens
        body = tok.decode(ids[start: start + tokens])
        request = {"messages": [{"role": "user", "content": "Read this excerpt:\n\n" + body
                                 + "\n\nSummarize it in a few sentences."}]}
        out.append(list(h.adapter.prompt_tokens(request)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--arms", nargs="+", required=True, help="first arm is the base (default)")
    ap.add_argument("--configs", nargs="+", default=["mtp:1", "ord:1"])
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--gen", type=int, default=192, help="tokens per lane (B=1)")
    ap.add_argument("--gen-batched", type=int, default=96)
    ap.add_argument("--gen-long", type=int, default=64)
    ap.add_argument("--doc-tokens", type=int, default=1024)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--max-swapout-pages", type=int, default=20000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--i-own-the-gpu", action="store_true")
    a = ap.parse_args()
    if not a.i_own_the_gpu:
        ap.error("Metal run: pass --i-own-the-gpu under the GPU lock")
    if a.arms[0] != "default":
        ap.error("the first arm must be default")

    import mlx.core as mx

    h = Harness("default", model=a.model)
    t = Toggles(h)
    print("LOADED", f"{h.load_s:.1f}s", f"active={mx.get_active_memory() / 2**30:.1f}GiB", flush=True)
    swap0 = swapouts()
    from check_mtp_row_exact import PROMPTS

    chat = [list(h.adapter.prompt_tokens({"messages": [{"role": "user", "content": text}]}))
            for text in list(PROMPTS.values()) + list(EXTRA_PROMPTS.values())]
    corpus = "\n\n".join(p.read_text() for p in sorted((ROOT / "docs").glob("*.md")))
    ids = list(h.adapter.tokenizer.encode(corpus))
    print("corpus tokens", len(ids), flush=True)

    def prompt_set(kind, n, rep):
        if kind in ("mtp", "ord") and n == 1:
            return [chat[rep % len(chat)]]
        if kind in ("long", "longord"):
            off = (rep * 7919) % max(1, len(ids) - n)
            body = h.adapter.tokenizer.decode(ids[off: off + n])
            req = {"messages": [{"role": "user", "content": "Here is a project document:\n\n" + body
                                 + "\n\nIn five bullet points, what are the main components it describes?"}]}
            return [list(h.adapter.prompt_tokens(req))]
        return doc_prompts(h, ids, n, a.doc_tokens, (rep * n * a.doc_tokens) % max(1, len(ids) - n * a.doc_tokens))

    from mlx2.runtime.models import qwen4_exp as QE

    def diag():
        extra = {"qsa_nax_decode_stats": dict(QE._QSA_NAX_DECODE_STATS)}  # not in diagnostics()
        return flatten({**h.adapter.diagnostics(), **extra})

    results = {}
    out = {"model": a.model, "arms": a.arms, "configs": a.configs, "reps": a.reps, "mlx": mx.__version__,
           "gen": a.gen, "gen_batched": a.gen_batched, "gen_long": a.gen_long, "temp": a.temp,
           "policy": h.adapter.policy.as_dict(), "results": results}
    for config in a.configs:
        kind, n = config.split(":")
        n = int(n)
        mtp = kind in ("mtp", "long")
        is_long = kind in ("long", "longord")
        gen = a.gen if (n == 1 and not is_long) else (a.gen_long if is_long else a.gen_batched)
        per = {arm: {"tps": [], "ttft": [], "sha": [], "tok_step": [], "eng": [], "sched": []} for arm in a.arms}
        tokens_by = {arm: [] for arm in a.arms}
        order_log = []
        for rep in range(a.reps + 1):
            k = len(a.arms)
            order = a.arms[rep % k:] + a.arms[: rep % k]
            if rep % 2:
                order = order[::-1]
            order_log.append(order)
            prompts = prompt_set(kind, n, rep)
            for arm in order:
                t.set(arm)
                kw = dict(t.run_kwargs)
                fp32 = kw.pop("fp32_head_logits", False)
                if fp32:
                    pass  # head swapped in place; generator config flag is cosmetic for receipts
                before = diag()
                r = h.run(prompts, max_tokens=gen, mtp=mtp, temp=a.temp, **kw)
                eng = delta(diag(), before)
                if rep == 0:
                    print(f"warmup {config} {arm} {r['decode_tps']}", flush=True)
                    continue
                rec = per[arm]
                rec["tps"].append(r["decode_tps"])
                prompt_tokens = sum(len(p) for p in prompts)
                rec["ttft"].append(r["ttft_s"])
                rec["sha"].append(r["sha"])
                rec["tok_step"].append(r["tokens_per_step"])
                rec["eng"].append({k2: v for k2, v in eng.items() if not any(
                    s in k2 for s in ("timing", "_ms", "_ns", "bytes", "ple_tables", "last_decision",
                                      "geometry_candidates"))})
                rec["sched"].append(r["scheduler_stats"])
                tokens_by[arm].append(r["lanes"])
                print(f"{config} rep{rep} {arm:28s} {r['decode_tps']:.2f} tok/s ttft={r['ttft_s']:.2f}s "
                      f"tok/step={r['tokens_per_step']} prompt={prompt_tokens} sha={r['sha']}", flush=True)
            swap = swapouts() - swap0
            if swap > a.max_swapout_pages:
                out["aborted"] = f"swapouts +{swap} pages in {config} rep {rep}"
                break
        t.reset()
        base = per["default"]
        summary = {}
        for arm, rec in per.items():
            if not rec["tps"]:
                continue
            s = {"median_tps": statistics.median(rec["tps"]), "min_tps": min(rec["tps"]),
                 "max_tps": max(rec["tps"]), "median_ttft_s": statistics.median(rec["ttft"]),
                 "mean_tokens_per_step": statistics.mean([x for x in rec["tok_step"] if x] or [0])}
            if arm != "default":
                paired = [100 * (x / y - 1) for x, y in zip(rec["tps"], base["tps"])]
                paired_ttft = [100 * (x / y - 1) for x, y in zip(rec["ttft"], base["ttft"])]
                s["paired_delta_pct"] = {"median": statistics.median(paired), "min": min(paired),
                                         "max": max(paired), "values": paired,
                                         "faster_reps": sum(p > 0 for p in paired)}
                s["paired_ttft_delta_pct"] = {"median": statistics.median(paired_ttft),
                                              "min": min(paired_ttft), "max": max(paired_ttft)}
                same = [x == y for x, y in zip(rec["sha"], base["sha"])]
                s["tokens_identical_reps"] = f"{sum(same)}/{len(same)}"
                first = None
                for ref_rep, got_rep in zip(tokens_by["default"], tokens_by[arm]):
                    for lr, lg in zip(ref_rep, got_rep):
                        kk = next((i for i, (x, y) in enumerate(zip(lr, lg)) if x != y), None)
                        if kk is not None:
                            first = kk if first is None else min(first, kk)
                s["first_divergence"] = first
            # engagement: keys whose totals differ from the default arm's
            tot = {}
            for e in rec["eng"]:
                for k2, v in e.items():
                    tot[k2] = tot.get(k2, 0) + v
            s["engagement_total"] = tot
            sched = {}
            for e in rec["sched"]:
                for k2, v in e.items():
                    sched[k2] = sched.get(k2, 0) + v
            s["scheduler_total"] = {k2: v for k2, v in sched.items() if v}
            summary[arm] = s
        base_eng = summary.get("default", {}).get("engagement_total", {})
        for arm, s in summary.items():
            if arm != "default":
                s["engagement_vs_default"] = {k2: [base_eng.get(k2, 0), v] for k2, v in s["engagement_total"].items()
                                              if base_eng.get(k2, 0) != v}
                s["engagement_vs_default"].update({k2: [v, 0] for k2, v in base_eng.items()
                                                   if k2 not in s["engagement_total"]})
        results[config] = {"order": order_log, "runs": {arm: {k2: v for k2, v in rec.items() if k2 != "eng"}
                                                       for arm, rec in per.items()}, "summary": summary}
        brief = {arm: {"median_tps": round(s["median_tps"], 2),
                       **({"paired_median_pct": round(s["paired_delta_pct"]["median"], 2),
                           "range": [round(s["paired_delta_pct"]["min"], 2), round(s["paired_delta_pct"]["max"], 2)],
                           "ttft_pct": round(s["paired_ttft_delta_pct"]["median"], 2),
                           "identical": s["tokens_identical_reps"]} if arm != "default" else {})}
                 for arm, s in summary.items()}
        print("SUMMARY", config, json.dumps(brief), flush=True)
        out["swapouts_delta"] = swapouts() - swap0
        out["peak_gib"] = mx.get_peak_memory() / 2**30
        Path(a.out).write_text(json.dumps(out, indent=1))
        if out.get("aborted"):
            print("ABORTED", out["aborted"], flush=True)
            break
    Path(a.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
