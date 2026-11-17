"""Resource plan: units, batch, memory caps; refuse when the run would not fit.

Section 8: one model per process, under the per-process cap (80 GB; Flash-Next carve jobs 92 GB by
Pierre's approval, passed here as ``--cap-gb 92 --cap-reason ...``), watching footprint, not RSS. A
model that does not fit in memory needs a layer-streamed backward pass, which is not built: refuse.

The working-set model is deliberately simple and conservative (estimates, recorded as such):
  carve (layer_ref):  weights + recorded layer inputs (L x tokens x hidden, bf16) + three logit rows
                      (full, masked, softmax: tokens x vocab x 4 B) + one layer's tape
                      (tokens x (16 hidden + 8 width) x 4 B) + the mask optimiser (units x 16 B) + 2 GB
  capture / gates:    weights + one layer's activations in flight + logits (responses go to disk) + 2 GB
"""
from __future__ import annotations

import subprocess

GB = 2 ** 30


def physical_gb() -> float:
    try:
        return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()) / GB
    except (OSError, ValueError, subprocess.CalledProcessError):
        return 0.0


def _units(info: dict, moe_units: str) -> int:
    u = info["units"]
    return u["dense_neurons"] + (u["moe_neuron_units"] if moe_units == "neuron" else u["moe_expert_units"])


def carve_work_gb(info: dict, bs: int, seq: int, units: int) -> float:
    t = bs * seq
    L, d = info["layers"], info["hidden"] or 0
    width = max(info["intermediate"] or 0, (info.get("moe") or {}).get("expert_intermediate") or 0)
    if info.get("moe"):
        width *= (info["moe"].get("top_k") or 8)
    rec = L * t * d * 2
    logits = 3 * t * (info["vocab"] or 0) * 4
    tape = t * (16 * d + 8 * width) * 4
    opt = units * 16
    return (rec + logits + tape + opt) / GB + 2.0


def capture_work_gb(info: dict, bs: int, seq: int, units: int) -> float:
    """One layer's activations in flight (tokens x widest site x 4 B, a few copies) and the logits; the per-text
    responses are memory-mapped to disk."""
    width = max(info["intermediate"] or 0, ((info.get("moe") or {}).get("experts") or 0) * 4)
    return (bs * seq * (4 * width * 4 + 2 * (info["vocab"] or 0) * 4)) / GB + 2.0


CARVE_JOBS = ("carve_core", "carve_box")


# child_rolling's default mu pass carries the fit texts a second time (the resident capture's fit-only batches): the
# state is counted twice, an upper bound (the fit split is a share of the texts)
FIT_CARRY = 2


def _fit_carry(info):
    """The fit pass runs only when some layer has dense units (mu is a dense statistic)."""
    dense = (info.get("units") or {}).get("dense_neurons")
    return FIT_CARRY if dense is None or dense > 0 else 1


def rolling_plan(info: dict, cap: float, *, texts: int, seq: int, window: int | None) -> dict:
    """Rolling layer-window capture (forward only) for a model whose resident load does not fit.

    Resident: embeddings, head, norms and (MoE) every non-expert tensor; streamed: one decoder layer (dense) or
    one layer's routed experts (MoE) at a time, three in flight (the loader's double buffer plus the layer
    running). The corpus state (one bf16 row of the residual stream per token) stays resident; if it does not fit
    beside the window, the corpus is cut into the fewest chunks that fit (each chunk reads every weight once)."""
    lb = info.get("layer_bytes") or {}
    family = info.get("sweep_family")      # a model family of the maps session's: its own layer loader
    if family:
        hc = int(info.get("residual_streams") or 1)
        state = texts * seq * (info["hidden"] or 0) * 2 * hc * _fit_carry(info)
        usable = 0.9 * cap * GB - (lb.get("rest") or 0) - 3 * (lb.get("max_layer") or 0) - 2 * GB
        carry = "resident" if state <= 0.5 * max(usable, 0) else "memmap"
        return {"mode": "rolling", "fits": True, "family": family, "window": 0, "chunks": 1, "carry": carry,
                "stream_mode": "sweep_family", "texts": texts, "state_gb": round(state / GB, 2),
                "loader": f"research/amoeba/sweep {family} family (its own layer loader)",
                "carve": "refused: the streamed backward pass is not built (capture only)"}
    if lb.get("error") or not lb.get("layered"):
        return {"mode": "rolling", "fits": False,
                "refusal": f"no per-layer byte map for this checkpoint ({lb.get('error') or 'keys are not layers.<i>.'})"}
    if info.get("adapter_streaming") is False:
        return {"mode": "rolling", "fits": False,
                "refusal": f"adapter {info.get('adapter')} accepts no weight-streaming request: no rolling capture (it "
                           "would need a resident load over the cap); add streaming to the adapter, or a sweep family "
                           "(research/amoeba/sweep), or raise the cap with --cap-gb/--cap-reason if it truly fits"}
    moe = info["kind"] == "moe"
    if moe:
        return {"mode": "rolling", "fits": False,
                "refusal": "a streamed MoE model needs whole-layer expert streaming, which mlx2's loaders do not have "
                           "(dense_layers refuses routed experts; moe_experts is an on-demand cache that would re-read "
                           "most experts per batch): register a sweep family for it, or raise the cap if it fits"}
    unit = lb["max_layer"]
    resident = lb["rest"] + lb["per_layer"][0]            # layer 0 always stays resident (layer_stream)
    stream_mode = "dense_layers"
    usable = 0.9 * cap * GB - resident - 3 * unit - 2 * GB
    hc = int(info.get("residual_streams") or 1)
    state = texts * seq * (info["hidden"] or 0) * 2 * hc * _fit_carry(info)
    out = {"mode": "rolling", "stream_mode": stream_mode, "unit_gb": round(unit / GB, 3), "bs": 8,
           "resident_gb": round(resident / GB, 2), "state_gb": round(state / GB, 2), "texts": texts}
    if usable < unit:
        out.update(fits=False, refusal=f"resident {resident / GB:.1f} GB + 3 streamed units of {unit / GB:.2f} GB "
                                       f"leave no room under the {cap:.0f} GB cap")
        return out
    state_budget = max(min(state, 0.5 * usable), 1)
    chunks = max(1, -(-int(state) // int(state_budget)))
    carry = "resident"
    if chunks > 2:            # each chunk re-reads every weight: past two, keep one read and put the state on disk
        carry, chunks = "memmap", 1
        state_budget = min(state, 0.5 * usable)
    free = usable - state_budget
    k_auto = int(max(1, min(8, free // unit)))
    k = int(window) if window else k_auto
    out.update(fits=True, window=k, window_auto=k_auto, carry=carry,
               carry_note=("the corpus state stays resident" if carry == "resident" else
                           "the state lives on disk between layers (the sweep's LayerState): one weight read instead "
                           f"of {-(-int(state) // int(max(min(state, 0.5 * usable), 1)))} chunked reads"),
               window_override=bool(window), chunks=chunks, state_budget_gb=round(state_budget / GB, 2),
               prefetch="page cache, the next K layers' byte ranges read ahead of the layer running",
               weight_reads=chunks, ceiling_gb=round(3 * unit / GB, 3),
               carve="refused: the streamed backward pass is not built (capture only)")
    if window and k > k_auto:
        out.update(fits=False, refusal=f"--window {k} needs {k * unit / GB:.1f} GB of prefetch, over the {free / GB:.1f} "
                                       "GB left beside the state")
    return out


def make_plan(info: dict, cfg: dict, *, cap_gb: float | None = None, cap_reason: str = "",
              moe_units: str | None = None, cpu: bool = False, stream: str = "auto", window: int | None = None,
              texts: int = 0) -> dict:
    phys = physical_gb()
    cap = 80.0 if cap_gb is None else float(cap_gb)
    if cap > 80.0 and not cap_reason:
        raise SystemExit(f"--cap-gb {cap} is over the 80 GB per-process cap: pass --cap-reason (who approved it)")
    headroom = 15.0
    if phys:
        cap = min(cap, phys - headroom)
    moe_units = moe_units or ("neuron" if info["kind"] == "moe" else "expert")
    units = _units(info, moe_units)
    w = info["weight_bytes"] / GB
    seq = cfg["data"]["window_tokens"]
    comp = cfg["compute"]
    refusals, notes = [], []
    batches = {}
    for job, want, fn in (("carve_core", comp["core_bs"], carve_work_gb), ("carve_box", comp["box_bs"], carve_work_gb),
                          ("capture", comp["map_bs"], capture_work_gb), ("gates", comp["map_bs"], capture_work_gb)):
        bs = want
        while bs >= 1 and w + fn(info, bs, seq, units) > 0.9 * cap:
            bs //= 2
        if bs < 1:
            refusals.append(f"{job}: weights {w:.1f} GB + working set at batch 1 "
                            f"{fn(info, 1, seq, units):.1f} GB exceed 90% of the {cap:.0f} GB cap")
            bs = 0
        elif bs < want:
            notes.append(f"{job}: batch {want} -> {bs} to fit the cap")
        batches[job] = {"bs": bs, "est_gb": round(w + fn(info, max(bs, 1), seq, units), 2)}
    if w > 0.9 * cap:
        refusals.append(f"weights {w:.1f} GB do not fit in the {cap:.0f} GB cap: this model needs layer streaming "
                        "with a streamed backward pass (mapping-workflow s8), which is not built")
    peak = max(b["est_gb"] for b in batches.values())
    resident_fits = not refusals
    if info.get("no_adapter"):             # no mlx2 adapter: only a sweep family can run it, layer-major
        if stream == "resident":
            refusals.append("no mlx2 adapter for this checkpoint: it cannot load resident (only its sweep family runs it)")
        resident_fits = False
        notes.append(f"no mlx2 adapter: sweep family {info['no_adapter']['family']} only (map capture)")
    if stream == "rolling" or (stream == "auto" and not resident_fits):
        rp = rolling_plan(info, cap, texts=texts or 46000, seq=seq, window=window)
        rp["chosen"] = "forced (--stream rolling)" if stream == "rolling" else \
            "automatic: the resident load does not fit under the cap"
        rp["resident_refusals"] = refusals
        notes.append(f"stream: rolling layer-window capture ({rp['chosen']}); window K={rp.get('window')}, "
                     f"chunks {rp.get('chunks')}; carve stages refused (no streamed backward)")
        refusals = [] if rp["fits"] else [f"rolling: {rp.get('refusal')}"]
    else:
        rp = {"mode": "resident", "chosen": "resident load fits" if stream == "auto" else "forced (--stream resident)"}
    plan = {
        "physical_gb": round(phys, 1), "cap_gb": round(cap, 1), "cap_reason": cap_reason or None,
        "headroom_gb": headroom, "device": "cpu" if cpu else "gpu",
        "weights_gb": round(w, 2), "moe_units": moe_units, "units": units,
        "units_note": "estimate from config.json; the carve host's exact widths are recorded by the first model stage",
        "seq_tokens": seq, "batches": batches, "peak_est_gb": round(peak, 2),
        "mlx_memory_gb": round(max(4.0, cap - 4.0), 1), "watchdog_gb": round(cap, 1),
        "need_gb": round(min(cap, peak + 4.0), 1),
        "assert_ple_offload": bool(info.get("file_backed_ple")),
        "refusals": refusals, "notes": notes, "estimate": True, "stream": rp,
    }
    plan["fits"] = not refusals
    return plan


def format_plan(info: dict, plan: dict) -> str:
    lines = [f"model     {info['checkpoint']} ({info['kind']}, adapter {info['adapter']}, {info['layers']} layers, "
             f"hidden {info['hidden']})",
             f"units     {plan['units']:,} ({plan['moe_units'] + ' MoE units' if info['kind'] == 'moe' else 'MLP neurons'};"
             f" estimate)",
             f"memory    weights {plan['weights_gb']} GB, cap {plan['cap_gb']} GB of {plan['physical_gb']} GB physical"
             f"{' (' + plan['cap_reason'] + ')' if plan['cap_reason'] else ''}, est. peak {plan['peak_est_gb']} GB",
             "batches   " + ", ".join(f"{k} {v['bs']} (~{v['est_gb']} GB)" for k, v in plan["batches"].items())]
    st = plan.get("stream") or {}
    if st.get("mode") == "rolling":
        lines.append(f"stream    rolling layer-window capture ({st.get('chosen')}): {st.get('stream_mode')}, window "
                     f"K={st.get('window')}, unit {st.get('unit_gb')} GB, resident {st.get('resident_gb')} GB, state "
                     f"{st.get('state_gb')} GB in {st.get('chunks')} chunk(s); carve: {st.get('carve')}")
    if plan["assert_ple_offload"]:
        lines.append("load      file-backed PLE present: asserted at load (--assert-ple-offload)")
    lines += [f"note      {n}" for n in plan["notes"]]
    lines += [f"REFUSED   {r}" for r in plan["refusals"]]
    lines.append("fits      " + ("yes" if plan["fits"] else "no"))
    return "\n".join(lines)
