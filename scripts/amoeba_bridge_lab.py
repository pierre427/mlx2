#!/usr/bin/env python3
"""Amoeba Phase A (2026-11-16): get mapped injection and distilled tiles working on a pair with a LARGE measured
teacher gap and a SHARED tokenizer, so donor and host positions pair exactly (same token ids, no warp of text).

Default pair: host Qwen3-0.6B, donor Qwen3-4B; task --task (add:A:B, mul:A:B, addk:A:K) in plain chat (thinking
off), greedy, exact integer oracle; disjoint train / val / test items. Both models resident in one process.
--screen spec,spec,... runs only the donor-gap screen (host and donor greedy on --n-test fresh items per task) and
writes screen.json: pick a task where the donor leads by a wide margin before training (Phase A's add:3:3 had an
inverted gap, host 98 vs donor 83).
Transmission metrics per arm: agree_with_donor (answers equal to the donor's) and copies_donor_error (how many of the
donor's own wrong answers the arm reproduces). Read it against the host arm's own count: a small host can make the
same carry error by itself.

Arms
  host, donor            ceilings on the test items (greedy)
  side                   reference: gated logit hitch with the full donor, p = (1 - a g) p_host + a g p_donor,
                         g = [max p_donor >= tau] (--side a:tau); shared vocab, so no id mapping
  H1, distilled tiles (no donor at run time):
    tile_init            tiles at random init with w_out = 0: exactly the host (the random-init control)
    tile_kl              tiles trained with FULL-vocab KL(donor || host + tiles) on the donor's own on-policy
                         continuations (1 greedy + --samples at --temperature per train item)
    tile_sft             the same tiles, cross-entropy on the donor's sampled tokens (hard labels; reference); on a
                         topic task the sequences are the texts themselves, so tile_sft / lora_sft are host-only
    Epoch selection (review v4 §5.0c): *_sft arms select by val cross-entropy (val_ce; on a topic task the text,
    never a donor output), donor-supervised arms by --kl-arm-select (val KL by default); see selection_metric
    lora_init/kl/sft     the audit's gated-LoRA arm: a gated low-rank linear residual adapter, same training
    tile_kl_ok, lora_kl_ok, bridge_{box,rand,host}_ok@mode   donor-error filter (1): trained, and their epoch selected, only on
                         traces where the donor's answer was right (an oracle used at training time only). The box /
                         stub unit choice, feature standardisation and the bridge's ridge init are still fitted on
                         every train trace. Their val_kl is on donor-correct val traces against the host's KL on the
                         same set; val_kl_all_val is on every val trace, comparable across arms
    side_gated@a<a>      host-wins gate (2): the side stream mixed by a learned per-token gate
                         g = sigmoid(w . phi + b) over host/donor top-probabilities, entropies, cross-probabilities,
                         agreement and KL, fitted on GOLD answers of the train items (-log p(gold)); it learns to defer
                         to the host where the donor tends to be wrong and needs no oracle at run time. Its val_kl is
                         KL to the donor on the donor's own traces, so a higher value than side is intended there
  H2, mapped / bridged writes (the donor runs beside the host on the same tokens; its box's unit activations,
     standardised, are the features). --write-layers selects where they are written, one arm set per mode:
       warp   each box layer d writes at the host layer of the same relative depth, reading d's features only
              (the matched-layer map, (a)); several box layers give a multi-band write ((c), part 1)
       stub   the host's own contrastive topic layers (its stub), each reading every box layer ((c), part 2)
    ridge_box            (a) closed-form ridge from the box features to the host's matched-layer INCREMENT
                         h_L - h_{L-1} on the paired training text (centred, linear part only)
    ridge_shuf           the same fit on row-shuffled pairs, applied to the real features, rescaled per token to
                         |ridge_box write| (shuffled map)
    ridge_rand           ridge from random donor units (same layers and counts), applied to those units, rescaled
                         per token to |ridge_box write| (random box at the same write norm)
    kgrad_box            (a') donor-derived target: ridge from the box features (continuation positions) to the unit
                         direction -d KL(donor || host) / d h_L, the map rescaled so its mean write norm equals the
                         host's mean increment there (the cap then sets the size). ridge_box's target is the host's
                         own increment, which carries no donor-minus-host signal; this one does
    kgrad_shuf, kgrad_rand   its row-shuffled fit and its random-unit fit, rescaled per token to |kgrad_box write|
    bridge_box           (b) learned low-rank gated bridge per write layer, initialised from kgrad_box (--bridge-init
                         grad; or ridge_box with inc), truncated SVD (--bridge-rank), gate starting at 0.5, trained with full-vocab KL(donor || host + bridge) on the topic text,
                         host frozen, drift penalty KL(host || host + bridge) on ordinary English
    bridge_shuf          the same bridge trained and evaluated on another item's features (the donor reads a
                         different item's prompt, then this item's text): right statistics, wrong problem
    bridge_rand          the bridge on random donor units (same layers and counts)
    bridge_host          the bridge on the HOST's own stub units from a clean host copy run in lockstep (no donor
                         at all): if it matches bridge_box, the bridge is only an adapter and carries no donor feature
    bridge_noise         the bridge on N(0, 1) features: no donor information at all (a bridge "from a random
                         donor"); with the drift penalty it is the capacity control for bridge_box
  H4, lens transport (closed form, task-agnostic; Pierre 2026-11-18: "Tuned-lens transport test"). Nothing is fitted
     on paired text or on the task. Each model gets its own linear lens per layer: a ridge least-squares map from the
     layer's output to the final layer's output (pre-norm, position 0 dropped), fitted on --lens-texts generic English
     texts (--lens-kinds tuned; logit = the identity, the plain logit lens). Through each model's own final-norm gain g
     and unembedding W (shared vocabulary, logits centred over the vocabulary), a residual change x at donor layer d
     moves the donor's logits by K_D x / r_D, and a change y at host layer L by K_H y / r_H (r = mean final-residual
     RMS on the lens texts; the norm's projection term is ignored). The transport T = argmin |K_H y - K_D x|^2 +
     lam |y|^2 (--lens-lam relative to tr(K_H'K_H)/D_h) is one matrix per (d, warp(d)); the host write is
     (r_H / r_D) T x, x = the box units' write W_down[:, box] (a - a_English_mean) at donor layer d, written at host
     layer warp(d) under the same blend-in cap. Receipt: per layer the lenses' held R^2, the readout check (log-probs
     rebuilt from the captured final residual vs the model's own), the transport fidelity 1 - |K_H T x - K_D x|^2 /
     |K_D x|^2 on the topic train texts (unseen by the lens fit) for box and random units, and the box's logit
     effect size.
     Dark space (Pierre 2026-11-18: look "for dark spaces where it doesn't connect at all"), same receipt, box and
     random units side by side: fidelity_readout_ceiling = the best ANY host residual change could do (host final
     layer, identity lens): the donor's logit change outside the host unembedding's row space is dark at every
     depth, and only new readout rows (a novel term + late sink) or a side stream reach it. On --dark-positions
     continuation positions, the donor's logit change is split into the part the host expresses at warp(d) and the
     dark rest; for each part the next token's z-score and top-10 rate say where the useful signal sits (answer in
     the dark part = the knowledge itself is what cannot connect), with the most frequent top-5 tokens of each part.
     H5, host topic sinks (Pierre 2026-11-18: hook where the host's late layers already turn topic context into
     output). Sinks = host units at depth >= --sink-from that pass the topic-contrast rule and whose write, read
     through the host's own final gain and unembedding, promotes the train continuations' next tokens (mean z > 0),
     ranked by contrast x signature; the --n-sink-layers best layers, --sink-units each. One joint closed-form
     transport (same lens kind and metric) solves for the sinks' activation changes c that best reproduce the box's
     total logit change; the host write at sink layer L is W_down[:, sinks_L] c_L. Receipt "sinks": layers, units,
     token signatures, fidelity and the next token's place in the sink-expressed and dark parts.
    sink_box@<kind>      the donor box into the host's own topic sinks
    sink_rand@<kind>     random donor units into the same sinks, rescaled per token per layer to |sink_box write|
    sink_hrand@<kind>    the donor box into RANDOM host units of the same sink layers (same counts, own solve),
                         rescaled per token per layer to |sink_box write|: are the sinks special, or does any late
                         subspace do
    sink_shuf@<kind>     another item's box features through the same sinks (the donor reads a different prompt)
    sink_const@<kind>    one fixed push: the box's mean centred activation over the train continuations, through the
                         sinks, at every position; sink_box > sink_const is needed before calling it item information
    sink_path@<kind>     multiple supported paths (Pierre 2026-11-18: not one graph, "multiple supported paths like
                         real token streams would have"): per token, the same joint system solved ONLY over the
                         --path-units sinks the host's own clean stream has most active there (a lockstep clean host
                         copy reads them); sink_box is the single-graph baseline
    sink_pathrand@<kind> per token a random set with the host path's per-layer counts, rescaled per token per layer
                         to |sink_path write|:
                         does following the host's open path matter
    sink_pathidle@<kind> per token the --path-units sinks the host is LEAST active on (spare topic capacity; added
                         after the 2026-11-18 smoke where the host's most active sinks lost to random ones)
     Receipt sinks.paths: distinct paths, mean pairwise Jaccard, layer share, share of open activity in the path;
     sinks.transport["<kind>/path/*"]: fidelity and next-token read of single graph / host paths / random paths on
     the same validation sample.
    sink_cone<tau>@<kind> cones with a dynamic cutoff (Pierre 2026-11-18): per token, the full sink solve scores
                         each unit by its Fisher energy c_u^2 Q_uu; the smallest set with >= tau of the total is kept
                         and re-solved, so path size follows the token (--cone-taus); units/token are counted at
                         the validation continuation positions only (the ones val KL scores)
    sink_conerand<tau>@<kind> its matched control: per token the same per-layer counts of RANDOM sinks, rescaled
                         per token per layer to |sink_cone write|: does the unit selection matter
    sink_neuron@<kind>   hand-made neuron (vgel.me 2026-11-20, adapted): the sink_const push at every sink layer
                         times one host-only gate a(z) = silu(alpha z) z / silu(alpha), z = r . rms(h) at the first
                         sink layer's input, r = difference of means (train topic continuations minus English retain)
                         made orthogonal to the English mean, scaled to z = 1 at the topic mean (--neuron-alpha).
                         Appended (no host unit overwritten); no donor at run time. Against sink_const: the same push,
                         so a win is lower English drift at equal val KL. Receipt sinks.neuron: gate stats on
                         validation topic vs English text
    sink_neuronflat@<kind> the gate replaced by its train-topic mean at every position (same size, no selectivity)
     Pareto (summary.pareto): every sink-family arm except the sink_neuron pair (one dense write per layer is
     not a unit count; read it against sink_const) as (units per token, val KL, gain share vs sink_box, English
     drift, test score), and the pre-registered pick: the smallest path with >= --pareto-gain of sink_box's gain
     within the drift cap (controls excluded). Size never wins over quality: no qualifying arm = sink_box stays.
     Autoregressive split (Pierre 2026-11-18: "look at the backwards paths too, in an autoregressive sense"): a
     write at position s reaches the prediction at t > s through attention, which the transport objective and the
     linear diagnostics (both per position) do not see. summary.donor_box_ar_split: KL(donor || donor without its
     box) at sampled t with the box removed only at t (self), only before t (past), or everywhere; per lens / sink /
     path arm, ar_split: the gain in KL(donor || host) from the write only at t, only before t, or everywhere.
     Writes (lens, sink and path) are zeroed at position 0. Diagnostics read the validation texts, position 0 excluded.
     Lens arms are not drift-selected (fixed --lens-alpha): rank lens_box against its own controls,
     which share its write norm, not against the drift-capped trained arms. A Gemma-style model (post-MLP norms,
     layer scalar, logit softcap) or a failed readout check drops the lens arms with a reason in run.json.
     lens_shuf caveats: an item with no same-length partner falls back to its own features with positions shuffled;
     on a topic task the partner is another text of the same topic, so lens_box ~ lens_shuf > host means a
     topic-level bias, not item information.
    lens_box@<kind>      the transported box write
    lens_rand@<kind>     random donor units (same layers and counts) through the same transport, rescaled per
                         token to |lens_box write| (specificity: the decisive control)
    lens_shuf@<kind>     the box write from another item (the donor reads a different item's prompt): right
                         statistics, wrong problem
    lens_rot@<kind>      the real transported write turned by a fixed random rotation of the host space (same norm
                         per token): does the direction matter
Blend-in, every write and tile: scaled per token to at most --norm-budget x |h_L - h_{L-1}|, the host layer's own
increment (stop-gradient; layer 0: |h_0|). Rows record the pre-clamp ratio (mean over write layers and tokens) and
the clamped fraction, so "closes the gap only with a large write" is visible (the no-reader check, H3).
Box: per donor layer, units ranked by per-token topic mass minus per-token English mass (|a_u| x |W_down[:, u]|;
topic = the positions that predict the train continuations, English = the retain texts), topic/English ratio >=
--min-ratio, top --box-units per layer, on --box-layers (auto: the --n-box-layers layers with the largest box score,
at least 2 apart). The host's stub is chosen the same way from the host's own units.
Drift cap: every trained arm starts at --drift-weight and multiplies it by --drift-adapt after each epoch whose val
drift exceeds --drift-cap, so no arm buys its score with drift a control is denied.
Training: Adam, --epochs over the train sequences, each step --batch topic sequences plus one English retain text;
the epoch (0 = init) with the lowest val KL(donor || arm) among those with val English drift <= --drift-cap is kept.
Generation is scored on the test items only.
Report (summary.json): test accuracy, gap closed = (arm - host) / (donor - host), val KL to the donor and the
fraction of the host's KL closed (selection metrics: the same val set picks the epoch; the side arm's val KL is its
teacher-forced mixture and its English drift is not measured), English drift KL(host || arm) per token on held English text, norm ratio, clamped
fraction, trainable parameters. Writes run.json (receipt), items.json, rows.jsonl, curves.jsonl, summary.json.
Host gradients: MLX has no VJP for Metal kernel launches (the Qwen3.5/3.8 GatedDeltaNet recurrence on the GPU), so
every gradient through the host runs inside mlx2.runtime.models.differentiable.reference_path() (pure-MLX
reference ops); forward-only passes keep the serving kernels. At load the two paths' forward parity (max |d logp|,
KL, top-1 agreement on 2 task + 2 English sequences) and one real gradient probe go to run.json host_grad_path;
a failed probe or a mean KL above --grad-ref-max-kl stops the run there.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.append(str(ROOT))  # research.amoeba.carve (the memory watchdog); appended so it shadows nothing
import amoeba_hub_align as ha  # noqa: E402
import amoeba_paired_stats as PS  # noqa: E402

CORE = "~/Library/Application Support/mlx2/research-artifacts/amoeba/core-corpus-v2-9b64b3791/corpus.jsonl"
_STATE = {}  # main()'s receipt and its saver, for the watchdog thread and the exit handler


def shuf_partner_prompt(items, it):
    """Generation side of the shuffled-feature control: (donor prompt ids, partner item id). The partner is
    another test item with the same prompt length. An item with none (a unique prompt length, or one test item)
    falls back, as the teacher-forced side does, to its own prompt with the positions after the first shuffled;
    the partner id is then None."""
    P = it["ids"]
    same = [x for x in items if x["split"] == "test" and x["id"] != it["id"] and len(x["ids"]) == len(P)]
    if same:
        x = same[(int(it["id"][-4:]) * 7919) % len(same)]
        return x["ids"], x["id"]
    g = np.random.default_rng(int(it["id"][-4:]))
    return list(P[:1]) + [P[1 + int(i)] for i in g.permutation(max(0, len(P) - 1))], None


def box_json_identity(bj, run_json, donor_ck, width):
    """--measure --box-json: tie the box units to the loaded donor. box.json records donor_checkpoint and
    donor_mlp_width since 2026-11-21; an older file is checked against the run.json beside it. Returns
    (refusals, record): a non-empty refusals list names what differs; record goes into the measure receipt and
    carries a warning when no donor identity is recorded anywhere."""
    bad, rec = [], {"source": "box.json", "donor_checkpoint": bj.get("donor_checkpoint")}
    if rec["donor_checkpoint"] is None:
        rec = {"source": "run.json beside box.json", "donor_checkpoint": (run_json or {}).get("donor_checkpoint")}
    if rec["donor_checkpoint"] is None:
        rec = {"source": "unrecorded", "donor_checkpoint": None,
               "warning": "box.json and its run.json name no donor checkpoint: the box units are not tied to the "
                          "loaded donor"}
    elif rec["donor_checkpoint"] != donor_ck:
        bad.append(f"donor_checkpoint {rec['donor_checkpoint']!r} ({rec['source']}) != loaded {donor_ck!r}")
    for l, w in width.items():
        w0 = (bj.get("donor_mlp_width") or {}).get(str(l))
        if w0 is not None and int(w0) != int(w):
            bad.append(f"donor_mlp_width layer {l}: {w0} != loaded {w}")
        ids = bj["box"][str(l)]
        if ids and not (0 <= min(ids) and max(ids) < int(w)):
            bad.append(f"box layer {l} has unit ids outside 0..{int(w) - 1}")
    return bad, rec
BASES = ("ridge_box", "ridge_shuf", "ridge_rand", "kgrad_box", "kgrad_shuf", "kgrad_rand", "bridge_box", "bridge_shuf", "bridge_rand", "bridge_noise",
         "bridge_host")


def parse_task(spec):
    """add:A:B (an A-digit plus a B-digit number), mul:A:B (an A-digit times a B-digit number), addk:A:K (K A-digit
    numbers summed)."""
    op, x, y = spec.split(":")
    if op not in ("add", "mul", "addk"):
        raise SystemExit(f"unknown task {spec!r}")
    return op, int(x), int(y)


def hash_(x):
    import hashlib
    return hashlib.sha256(x.encode()).hexdigest()


def task_name(spec):
    if spec.startswith("topic:"):
        return f"knowledge topic {spec[6:]} (teacher-forced nats/byte on held probe texts)"
    op, x, y = parse_task(spec)
    return {"add": f"{x}-digit + {y}-digit addition", "mul": f"{x}-digit x {y}-digit multiplication",
            "addk": f"sum of {y} {x}-digit numbers"}[op] + ", plain chat, thinking off, greedy"


def make_items(spec, n_train, n_val, n_test, seed):
    op, x, y = parse_task(spec)
    rng = np.random.default_rng(seed)

    def num(d):
        return int(rng.integers(10 ** (d - 1), 10 ** d))
    seen, out, tries = set(), [], 0
    while len(out) < n_train + n_val + n_test:
        tries += 1
        if tries > 200 * (n_train + n_val + n_test) + 10000:
            raise SystemExit(f"task {spec} has too few distinct items for {n_train + n_val + n_test}")
        terms = [num(x) for _ in range(y)] if op == "addk" else [num(x), num(y)]
        k = tuple(sorted(terms))
        if k in seen:
            continue
        seen.add(k)
        ans = int(np.prod(terms, dtype=object)) if op == "mul" else sum(terms)
        out.append({"terms": terms, "answer": ans})
    sym = " × " if op == "mul" else " + "
    for i, it in enumerate(out):
        it["split"] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
        it["id"] = f"{it['split']}{i:04d}"
        it["prompt"] = f"What is {sym.join(str(t) for t in it['terms'])}? Answer with only the number."
    return out


def parse(text, stopped=True):
    """the reply's last number; None when the reply never reached eos (cut off by --max-tokens: the models echo the
    problem first, "a x b = c", so a cut-off reply's last number is a prefix or an operand)"""
    if not stopped:
        return None
    text = text.split("</think>")[-1]
    m = re.findall(r"-?\d[\d,]*", text)  # the last number: an echoed question ("a + b = c") still scores c
    return int(m[-1].replace(",", "")) if m else None


def topic_selection(rows, tid, seed, n_train, n_val, n_test, topic_tokens, encode, bos=(), check=None,
                    exclude_ids=(), exclude_groups=()):
    """The topic task's text selection (module level so a CPU tool can reproduce it with the host tokenizer and no
    model): fit-split texts of the topic and below train / validate, held-split texts test; each cut to
    topic_tokens (shorter texts dropped); ids = the first 2 tokens (the scored span starts at the 3rd, as in the
    donor-gap screen), full = the whole cut text. One seeded generator draws the fit permutation, then the held one.
    Excluded ids / source groups are skipped AFTER the permutation, so excluding texts the draw never reached leaves
    the selection unchanged (the reserved final-test manifest relies on this)."""
    under = lambda ts: any(t == tid or t.startswith(tid + ".") for t in ts)  # noqa: E731
    g_ = np.random.default_rng(seed + 3)
    bos = list(bos)
    nb_ = len(bos)
    exclude_ids, exclude_groups = set(exclude_ids), {str(x) for x in exclude_groups}
    items = []
    for split_, want_ in (("fit", n_train + n_val), ("held", n_test)):
        pool = [r for r in rows if r["split"] == split_ and under(r["topics"])]
        pool = [pool[i] for i in g_.permutation(len(pool))]
        got = []
        for r in pool:
            if r["id"] in exclude_ids or str(r.get("group")) in exclude_groups:
                continue
            ids_ = list(encode(r["text"]))[:topic_tokens]
            if len(ids_) < topic_tokens:
                continue
            if check is not None:
                check(r, ids_)
            full_ = bos + ids_  # Gemma needs its BOS; the scored span is still the text's tokens 3..end
            got.append({"probe_id": r["id"], "full": full_, "ids": full_[:2 + nb_], "group": r.get("group")})
            if len(got) == want_:
                break
        if split_ == "fit":
            # validation picks the epoch: give it whole source groups (no file shared with train), sized in
            # proportion to the --n-train / --n-val request on whatever the topic actually has
            nv_want = max(4, round(len(got) * n_val / max(n_train + n_val, 1)))
            groups = sorted({str(it["group"]) for it in got}, key=lambda x: (hash_(x), x))
            vgroups, nv = set(), 0
            for gname in groups:
                if nv >= nv_want:
                    break
                vgroups.add(gname)
                nv += sum(str(it["group"]) == gname for it in got)
            for it in got:
                it["split"] = "val" if str(it["group"]) in vgroups else "train"
        else:
            for it in got:
                it["split"] = "test"
        for it in got:
            it["id"] = f"{it['split']}{len(items):04d}"
            items.append(it)
    return items


def split_exclusions(man):
    """(ids, groups) a manifest puts out of a run's reach: the final-test texts (reserved_final_test, from
    amoeba_bridge_split_manifest.py) and a donor-gap screen's fit slice (screen_slice, from
    amoeba_topics_donor_gap.py score --split fit), so fitting never reads what the screen scored."""
    ids, groups = set(), set()
    for key in ("reserved_final_test", "screen_slice"):
        r = man.get(key) or {}
        ids |= set(r.get("probe_ids", []))
        groups |= {str(g) for g in r.get("groups", [])}
    return ids, groups


def split_check(man, items):
    """'' when the selection's train / val / test probe ids equal the manifest's, else what differs"""
    err = []
    for k in ("train", "val", "test"):
        want = list((man.get("selection") or {}).get(k, []))
        got = [it["probe_id"] for it in items if it["split"] == k]
        if got != want:
            err.append(f"{k}: {len(got)} selected vs {len(want)} in the manifest, "
                       f"{len(set(got) ^ set(want))} ids differ")
    return "; ".join(err)


def selection_metric(arm, kl_arm_metric="val_kl"):
    """The validation metric that picks a trained arm's epoch, explicit per arm (review v4 §5.0c): a host-only arm
    (loss sft: cross-entropy on the text) selects by validation cross-entropy on the text, so the donor never
    chooses the control; donor-supervised arms (loss kl, and the bridges) by --kl-arm-select (default val KL to
    the donor). Init arms train nothing and select nothing."""
    if arm.get("route") == "tile" and not arm.get("loss"):
        return None
    if arm.get("loss") == "sft":
        return "val_ce"
    return kl_arm_metric


def pick_epoch(hist, metric, drift_cap):
    """Epoch to keep from per-epoch records [{epoch, val_drift, <metric>, ...}] (epoch 0 = init): the lowest
    `metric` among epochs whose val English drift is within drift_cap; if none is, the lowest-drift epoch.
    Reads only `metric`, "epoch" and "val_drift" (tests pin this: a host-only arm's pick never reads val_kl).
    Returns (epoch, within_cap)."""
    best, low = None, None
    for h in hist:
        dv = h["val_drift"]
        if low is None or dv < low[0]:
            low = (dv, h["epoch"])
        if dv <= drift_cap:
            v = h[metric]
            if best is None or v < best[0]:
                best = (v, h["epoch"])
    return (best[1], True) if best is not None else (low[1], False)


def contrast_teacher(xp, lp_h, lp_d, lp_dm, eta, mode="clip", clip=2.0, k=20):
    """Box-contrast teacher log-probs (proposal v4 §3.1): log q = log_softmax(log p_H + eta * r), r = log p_D -
    log p_D- with p_D- the donor with the box ablated. mode clip: r clipped to [-clip, clip]; mode topk: r kept
    only on the union of the two donor arms' top-k tokens per row (0 elsewhere), and also clipped when clip > 0.
    xp = numpy or mlx.core (rows [T, V]). eta = 0 returns log p_H renormalised (the host itself)."""
    r = lp_d - lp_dm
    if mode == "topk":
        V = int(lp_d.shape[-1])
        kk = V - min(max(int(k), 1), V)  # the k-th largest value per row is the threshold
        kd = xp.partition(lp_d, kk, axis=-1)[..., kk:kk + 1]
        km = xp.partition(lp_dm, kk, axis=-1)[..., kk:kk + 1]
        r = xp.where((lp_d >= kd) | (lp_dm >= km), r, xp.zeros_like(r))
    elif mode != "clip":
        raise ValueError(f"unknown teacher mode {mode!r}")
    if clip and clip > 0:
        r = xp.minimum(xp.maximum(r, -clip), clip)
    z = lp_h + eta * r
    m = xp.max(z, axis=-1, keepdims=True)
    return z - (m + xp.log(xp.sum(xp.exp(z - m), axis=-1, keepdims=True)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default=str(Path.home() / "mlx-models/Qwen3-0.6B-BF16-microscaling-source"))
    ap.add_argument("--donor", default=str(Path.home() / "mlx-models/Qwen3-4B"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", default="add:3:3",
                    help="add:A:B, mul:A:B, addk:A:K (generation-scored arithmetic) or topic:<taxonomy id> (a "
                         "knowledge topic: train on its fit probe texts, score held texts teacher-forced)")
    ap.add_argument("--probes", default="~/Library/Application Support/mlx2/research-artifacts/amoeba/topic-probes-9a1f35475/corpus.jsonl", help="topic task: probe corpus (v5.3)")
    ap.add_argument("--topic-tokens", type=int, default=64, help="topic task: tokens per text (texts shorter are dropped)")
    ap.add_argument("--bos", default="auto", choices=["auto", "off"],
                    help="topic and English texts: prepend the tokenizer's BOS when it has one (Gemma; Qwen has none)")
    ap.add_argument("--residual-f32", default="auto", choices=["auto", "on", "off"],
                    help="carry the host's residual stream in float32 from layer 0 on, in EVERY host pass (baseline "
                         "included), so a write capped at a fraction of a layer's increment is not rounded away in a "
                         "bf16 add; auto = on unless the host is already float32")
    ap.add_argument("--screen", default="",
                    help="donor-gap screen only: comma list of task specs; host and donor greedy on --n-test fresh "
                         "items each, writes screen.json and exits (pick a task with donor >> host before training)")
    ap.add_argument("--n-train", type=int, default=120)
    ap.add_argument("--n-val", type=int, default=40)
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--samples", type=int, default=3, help="donor samples per train item beside the greedy one")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--eng-retain", type=int, default=12)
    ap.add_argument("--eng-val", type=int, default=4)
    ap.add_argument("--eng-test", type=int, default=8)
    ap.add_argument("--eng-tokens", type=int, default=96)
    ap.add_argument("--arms", default="", help="comma list of arm names or bases (default: all)")
    ap.add_argument("--write-layers", default="warp,stub", help="write-layer modes for ridge/bridge arms")
    ap.add_argument("--box-layers", default="auto", help="donor box layers, comma list, or auto")
    ap.add_argument("--n-box-layers", type=int, default=4)
    ap.add_argument("--n-stub-layers", type=int, default=4)
    ap.add_argument("--box-units", type=int, default=64)
    ap.add_argument("--min-ratio", type=float, default=1.5)
    ap.add_argument("--ridge-lam", type=float, default=0.1, help="ridge penalty relative to mean feature variance")
    ap.add_argument("--bridge-rank", type=int, default=32)
    ap.add_argument("--bridge-init", default="grad", choices=["grad", "inc"],
                    help="bridge init: the gradient-direction ridge (kgrad) or the increment ridge; gate starts at 0.5")
    ap.add_argument("--tile-layers", default="all", help="host tile layers: all, stub, or a comma list")
    ap.add_argument("--tile-units", type=int, default=32, help="tile neurons per tile layer")
    ap.add_argument("--lora-rank", type=int, default=8, help="gated-LoRA tile rank per tile layer")
    ap.add_argument("--label", default="", help="receipt label (default: smoke-test label when a model is < 8B)")
    ap.add_argument("--norm-budget", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--tile-lr", type=float, default=1e-3)
    ap.add_argument("--bridge-lr", type=float, default=1e-3)
    ap.add_argument("--drift-weight", type=float, default=1.0)
    ap.add_argument("--drift-cap", type=float, default=0.05, help="max val English KL(host || arm) for selection")
    ap.add_argument("--drift-adapt", type=float, default=4.0,
                    help="every trained arm: after an epoch whose val drift exceeds --drift-cap, multiply its drift "
                         "weight by this (up to --drift-weight-max); 1 = fixed weight (the 52a00ad38 behaviour)")
    ap.add_argument("--drift-weight-max", type=float, default=256.0)
    ap.add_argument("--lens-kinds", default="tuned,logit",
                    help="lens transport arms: tuned (least-squares lens to the final residual), logit (identity); '' = off")
    ap.add_argument("--lens-texts", type=int, default=128, help="generic English fit texts for each model's lenses")
    ap.add_argument("--lens-tokens", type=int, default=128)
    ap.add_argument("--lens-ridge", type=float, default=1e-3, help="lens ridge penalty relative to mean feature variance")
    ap.add_argument("--lens-lam", type=float, default=1e-2, help="transport ridge relative to tr(K_H'K_H)/D_h")
    ap.add_argument("--lens-alpha", type=float, default=1.0, help="multiplier on the transported write (cap applies)")
    ap.add_argument("--lens-metric", default="fisher", choices=["fisher", "flat"],
                    help="logit-match geometry: fisher = the donor's average next-token Fisher on the lens texts "
                         "(what KL counts); flat = every vocabulary entry equal (the 2026-11-18 smoke showed it "
                         "fits rare-token bulk and leaves the next token in the dark part)")
    ap.add_argument("--sink-units", type=int, default=64,
                    help="host topic-sink units per sink layer for the sink_* arms (0 = no sink arms)")
    ap.add_argument("--n-sink-layers", type=int, default=4)
    ap.add_argument("--path-units", type=int, default=32,
                    help="sink_path arms: sink units per token on the host's own open path (top |activation| across "
                         "all sink layers, from a clean host pass)")
    ap.add_argument("--sink-from", type=float, default=0.6, help="sinks come from host layers >= this depth fraction")
    ap.add_argument("--cone-taus", default="0.5,0.8,0.95",
                    help="sink_cone<tau> arms: per-token dynamic cutoff keeping the smallest sink set with >= tau of "
                         "the full solve's Fisher energy ('' = none)")
    ap.add_argument("--neuron-alpha", type=float, default=8.0,
                    help="sink_neuron gate a(z) = silu(alpha z) z / silu(alpha); 1 = the vgel.me neuron (fires a "
                         "little on anti-aligned input), larger = closer to relu(z) z")
    ap.add_argument("--pareto-gain", type=float, default=0.95,
                    help="Pareto pick: smallest path whose val-KL gain is >= this share of sink_box's (and drift <= cap)")
    ap.add_argument("--ar-seqs", type=int, default=8,
                    help="autoregressive split: validation sequences read (0 = off); see ar_donor / ar_host")
    ap.add_argument("--ar-positions", type=int, default=6, help="autoregressive split: sampled positions per sequence")
    ap.add_argument("--dark-positions", type=int, default=256,
                    help="continuation positions sampled for the dark-space read (per lens kind, layer and source)")
    ap.add_argument("--readout-tol", type=float, default=0.1,
                    help="max mean |log-prob| error of the rebuilt readout (W g h / rms) on the reference's top-20 "
                         "tokens; larger, or top-1 agreement < 0.95, fails closed")
    ap.add_argument("--side", default="1:0.5", help="side reference arm alpha:tau ('' = off)")
    ap.add_argument("--seed", type=int, default=20261116)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--donor-dtype", default="native", choices=["float32", "bfloat16", "native"])
    ap.add_argument("--host-dtype", default="auto", choices=["auto", "float32", "native"],
                    help="auto: float32 on --cpu, else the artifact as stored (a quantized host trains in place)")
    ap.add_argument("--mem-cap-gb", type=float, default=0)
    ap.add_argument("--watchdog-gb", type=float, default=0,
                    help="memory watchdog (research/amoeba/carve/memwatch.py via ops.footprint_watchdog, with its "
                         "SIGKILL watcher; receipt OUT/watchdog-stop.json): exit 3 when this process's physical "
                         "footprint passes this many GB, free + inactive + purgeable falls under --min-avail-gb, free "
                         "pages starve under kern.vm_page_free_min, the compressor grows 12 GB (or holds 20 %% of "
                         "memory after growing), swap grows more than --swap-growth-gb, or pressure goes critical "
                         "(0 = off). Was amoeba27b_lib.footprint_watchdog, which missed the 2026-11-19 panic")
    ap.add_argument("--min-avail-gb", type=float, default=15.0)
    ap.add_argument("--swap-growth-gb", type=float, default=2.0)
    ap.add_argument("--waiver-note", default="", help="required when --swap-growth-gb is above 2 (who approved it)")
    ap.add_argument("--swap-base-gb", type=float, default=None,
                    help="swap used (GB) when the enclosing window started: the --swap-growth-gb allowance is then "
                         "counted from it, so sequential processes share one allowance")
    ap.add_argument("--budget-s", type=float, default=6600)
    ap.add_argument("--grad-ref-max-kl", type=float, default=0.01,
                    help="fail closed when the host's differentiable reference path (used for every gradient) "
                         "differs from its serving kernel path by more than this mean per-row KL (nats) on the "
                         "load-time parity sequences")
    ap.add_argument("--split-manifest", default="",
                    help="topic task: a split manifest (scripts/amoeba_bridge_split_manifest.py, or a donor-gap "
                         "screen's screen_manifest.json); its reserved final-test texts / screen slice and their "
                         "source groups are never selected (skipped after the seeded draw, so the draw is otherwise "
                         "unchanged); checkpoints it names must match the loaded models")
    ap.add_argument("--split-check", action="store_true",
                    help="with --split-manifest: fail closed unless the selected train / val / test probe ids are "
                         "exactly the manifest's (locks a rerun to the recorded selection)")
    ap.add_argument("--measure", action="store_true",
                    help="topic task: the v4 §5.0a/b measurements instead of arms (forward only, no fit): box worth "
                         "(donor intact / box mean-ablated / layer-matched random ablations) and the box-contrast "
                         "teacher on an eta grid, on the test texts; needs --box-json")
    ap.add_argument("--box-json", default="", help="--measure: a box.json of an earlier run (its box units)")
    ap.add_argument("--rand-seeds", type=int, default=5, help="--measure: layer-matched random ablations (>= 3)")
    ap.add_argument("--eta-grid", default="0,0.25,0.5,1,2", help="--measure: teacher strengths")
    ap.add_argument("--teacher-modes", default="clip,topk",
                    help="--measure: clip = log-ratio clipped to +-(--teacher-clip); topk = log-ratio only on the "
                         "union of both donor arms' top --teacher-topk tokens (also clipped when --teacher-clip > 0)")
    ap.add_argument("--teacher-clip", type=float, default=2.0)
    ap.add_argument("--teacher-topk", type=int, default=20)
    ap.add_argument("--kl-arm-select", default="val_kl", choices=["val_kl", "val_ce"],
                    help="epoch-selection metric of donor-supervised trained arms; host-only (sft) arms always "
                         "select by val_ce, the text's cross-entropy (the donor never chooses the control)")
    ap.add_argument("--evals-per-epoch", type=int, default=1,
                    help="trained arms: validation checkpoints per epoch (each a selection candidate; the drift "
                         "weight adapts at each); 1 = once per epoch, the earlier behaviour")
    ap.add_argument("--teacher-topk-clip", type=float, default=0.0,
                    help="--measure, topk mode: clip of the log-ratio inside the top-k union (0 = unclipped, the "
                         "proposal's plain restriction); --teacher-clip is the clip mode's own bound")
    ap.add_argument("--reference-rows", default="",
                    help="--measure: an earlier run's rows.jsonl; its host arm's per-text nats/byte are compared with "
                         "this run's host on the same item ids (selection and numerics check, recorded)")
    ap.add_argument("--gate-l1", type=float, default=0.0,
                    help="tile / gated-LoRA arms: L1 penalty on the mean gate on the English retain step (the CWE "
                         "distill v1 recipe, a0b957809, uses 0.1); the gate input is stop-gradient, so it trains "
                         "the gate only")
    ap.add_argument("--min-useful-npb", type=float, default=0.001,
                    help="minimum useful effect in nats/byte (Pierre 2026-11-18): each paired contrast is flagged "
                         "above / below / spanning it from its source-group 95%% interval")
    args = ap.parse_args()
    if args.measure and (not args.task.startswith("topic:") or not args.box_json):
        sys.exit("--measure needs a topic task and --box-json")
    if args.measure and args.rand_seeds < 3:
        sys.exit("--measure needs --rand-seeds >= 3")
    if args.split_check and not args.split_manifest:
        sys.exit("--split-check needs --split-manifest")
    SPLIT_MAN = json.loads(Path(args.split_manifest).read_text()) if args.split_manifest else None
    if args.watchdog_gb and args.swap_growth_gb > 2.0 and not args.waiver_note.strip():
        sys.exit("--swap-growth-gb above 2 needs --waiver-note naming the approval")
    if args.arms:  # names only, before any model loads (a pair can still drop a route: checked again later)
        taus_ = [f"{float(x):g}" for x in args.cone_taus.split(",") if x.strip()]
        known_ = (set(BASES) | {"host", "tile_init", "tile_kl", "tile_sft", "lora_init", "lora_kl", "lora_sft",
                                "tile_kl_ok", "lora_kl_ok", "side", "side_gated", "bridge_box_ok", "bridge_rand_ok",
                                "bridge_host_ok"}
                  | {f"lens_{s}" for s in ("box", "rand", "shuf", "rot")}
                  | {f"sink_{s}" for s in ("box", "rand", "hrand", "shuf", "const", "path", "pathrand", "pathidle",
                                           "neuron", "neuronflat")}
                  | {f"sink_cone{t}" for t in taus_} | {f"sink_conerand{t}" for t in taus_})
        bad_ = [a.strip() for a in args.arms.split(",") if a.strip() and a.strip().split("@")[0] not in known_]
        if bad_:
            sys.exit(f"unknown --arms {bad_}")

    out = Path(args.out).expanduser()
    if out.exists() and any(out.iterdir()):
        sys.exit(f"refusing to write into non-empty {out}")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    WATCH = None
    import threading
    RECEIPT_LOCK = threading.Lock()  # not re-entrant: save_receipt and the watchdog's last write each take it once
    # a minimal receipt while the models load (the likeliest moment for the footprint watchdog to trip)
    _STATE["receipt"] = {"schema": "amoeba-bridge-lab/v1", "status": "running", "stage": "loading models",
                         "args": vars(args)}
    _STATE["out"] = out
    (out / "run.json").write_text(json.dumps(_STATE["receipt"], indent=1))
    if args.watchdog_gb:
        import amoeba27b_lib as wd_
        sg_ = args.swap_growth_gb
        if args.swap_base_gb is not None:
            sg_ = args.swap_growth_gb - max(0.0, wd_.swap_used_gb() - args.swap_base_gb)
            if sg_ <= 0:
                sys.exit(f"swap already grew {wd_.swap_used_gb() - args.swap_base_gb:.1f} GB since the window "
                         f"started (allowance {args.swap_growth_gb} GB)")

        def wd_log(msg):
            print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)
            with open(out / "watchdog.log", "a") as f_:
                f_.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
            if "aborting" in msg or STOP_MARKER_ in msg:  # os._exit(3) follows: leave a failed receipt, and never
                # let this path raise (memwatch's thread logs its STOP_MARKER right before its exit). The watcher may
                # SIGKILL first (hard trigger or a starved thread): then its receipt stands, read by the window.
                try:  # (an exception here would end the watchdog thread and leave the process unwatched)
                    if RECEIPT_LOCK.acquire(timeout=2):  # held until the exit: no later write replaces this one
                        rec_ = dict(_STATE.get("receipt") or {}, status="failed (watchdog)", reason=msg,
                                    watchdog=dict(WATCH or {}), wall_s=round(time.time() - t0, 1))
                        tmp_ = out / "run.json.wd.tmp"
                        tmp_.write_text(json.dumps(rec_, indent=1, default=str))
                        tmp_.replace(out / "run.json")
                except BaseException:
                    pass
        from research.amoeba.carve import ops as wo_
        from research.amoeba.carve.memwatch import STOP_MARKER as STOP_MARKER_
        wo_.footprint_watchdog(args.watchdog_gb, wd_log, min_avail_gb=args.min_avail_gb, swap_growth_gb=sg_,
                               receipt=out / "watchdog-stop.json", watcher=True)
        WATCH = wo_.WATCH
        WATCH["swap_growth_allowance_gb"] = round(sg_, 2)
    rng = np.random.default_rng(args.seed)

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map
    if args.cpu:
        mx.set_default_device(mx.cpu)
    if args.mem_cap_gb:
        mx.set_memory_limit(int(args.mem_cap_gb * 1e9))
    mx.set_cache_limit(2 << 30)
    from mlx2.adapters.registry import inspect_model
    from mlx2.runtime.models import differentiable as dref
    from mlx2.experimental.weight_space import residual_ablation as ra
    from mlx2.serving import render_prompt_tokens

    def load(path, dtype):
        """dtype None: the artifact as stored (quantized layers stay quantized; tiles and bridges train on top of a
        frozen quantized host). A dtype: dequantize and cast (CPU, where quantized matmul is single-threaded)."""
        ad = inspect_model(path).adapter_type(str(path))
        m = ad.model
        if dtype is not None:
            ha.dequantize_model(mx, m, dtype)
            m.update(tree_map(lambda a: a.astype(dtype) if a.dtype in (mx.bfloat16, mx.float16) else a, m.parameters()))
            mx.eval(m.parameters())
        m.eval()
        lm = getattr(m, "language_model", m)
        return {"ad": ad, "model": m, "layers": list(lm.model.layers), "tok": ad.tokenizer,
                "ck": f"{Path(path).name}@{ad.identity['fingerprint']}"}

    ddt = {"float32": mx.float32, "bfloat16": mx.bfloat16, "native": None}[args.donor_dtype]
    hdt = {"float32": mx.float32, "native": None, "auto": mx.float32 if args.cpu else None}[args.host_dtype]
    H = load(args.host, hdt)
    Dn = load(args.donor, ddt)
    LH, LD = len(H["layers"]), len(Dn["layers"])
    tok = H["tok"]
    RES32 = args.residual_f32 == "on" or (args.residual_f32 == "auto" and hdt is not mx.float32)
    rev = subprocess.run(["git", "rev-parse", "--short=9", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip()) or bool(subprocess.run(
        ["git", "status", "--porcelain", "--", "scripts/amoeba_bridge_lab.py"], cwd=ROOT, capture_output=True,
        text=True).stdout.strip())
    import hashlib
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def logical_params(M):
        """Unquantized parameter count of the language model (vision/audio towers excluded), counted per module:
        a packed module (has scales and bits) adds weight.size x 32 / bits and skips its scales and biases; every
        other array counts at its size. Needs no adapter config attribute."""
        lm_ = getattr(M["model"], "language_model", M["model"])
        n = 0
        for _, m in lm_.named_modules():
            packed = "scales" in m and hasattr(m, "bits")
            for k, v in m.items():
                if not isinstance(v, mx.array):
                    continue
                if packed and k in ("scales", "biases"):
                    continue
                n += v.size * (32 // m.bits if packed and k == "weight" else 1)
        return int(n)
    n_h, n_d = logical_params(H), logical_params(Dn)
    label = args.label or (f"smoke test ({n_h / 1e9:.1f}B/{n_d / 1e9:.1f}B), not a result for the >8B goal"
                           if min(n_h, n_d) < 8e9 else "")
    receipt = {"schema": "amoeba-bridge-lab/v1", "host_checkpoint": H["ck"], "donor_checkpoint": Dn["ck"],
               "mlx2_source_revision": rev, "tree_dirty": dirty, "script_sha256": script_sha, "args": vars(args),
               "residual_f32_host": None,
               # the resolved artifact directories (a models/ entry can be a symlink or an alias name) and the corpora
               "host_path_resolved": str(Path(args.host).expanduser().resolve()),
               "donor_path_resolved": str(Path(args.donor).expanduser().resolve()),
               "core_corpus_sha256": hashlib.sha256(Path(CORE).read_bytes()).hexdigest(),
               "probes_sha256": (hashlib.sha256(Path(args.probes).read_bytes()).hexdigest()
                                 if args.task.startswith("topic:") else None),
               "status": "running", "label": label, "host_params": n_h, "donor_params": n_d,
               "device": str(mx.default_device()),
               "training": "tiles and bridges trained on a frozen host (Pierre 2026-11-16: 'please get the mapped "
                           "injections and distilled tiles working'); ridge arms closed form",
               "qualification": "unqualified research",
               "box_status": ("not carved: the harness's own topic-contrast unit selection (per donor layer the "
                              "--box-units units with the largest topic-minus-English write mass); no carving "
                              "receipt, it has NOT passed the carving bar (+0.05 nats held, <= 0.5x a random box's "
                              "excess)" + (f"; units read from {args.box_json}" if args.box_json else "")),
               "min_useful_effect_npb": args.min_useful_npb,
               "split_manifest": ({"path": str(Path(args.split_manifest).resolve()),
                                   "sha256": hashlib.sha256(Path(args.split_manifest).read_bytes()).hexdigest(),
                                   "check": bool(args.split_check)} if args.split_manifest else None)}
    if SPLIT_MAN is not None:
        for k_, ck_ in (("host_checkpoint", H["ck"]), ("donor_checkpoint", Dn["ck"])):
            if SPLIT_MAN.get(k_) and SPLIT_MAN[k_] != ck_:
                sys.exit(f"--split-manifest names {k_} {SPLIT_MAN[k_]}, loaded {ck_}")

    def save_receipt(**kw):
        with RECEIPT_LOCK:  # the watchdog thread's last write takes it too
            receipt.update(kw, wall_s=round(time.time() - t0, 1))
            if WATCH is not None:  # live peak footprint
                receipt["watchdog"] = dict(WATCH, waiver_note=args.waiver_note or None,
                                           swap_base_gb=args.swap_base_gb)
            tmp_ = out / "run.json.tmp"
            tmp_.write_text(json.dumps(receipt, indent=1))
            tmp_.replace(out / "run.json")
    _STATE.update(save=save_receipt, receipt=receipt)
    save_receipt()
    receipt["residual_f32_host"] = bool(RES32)

    def render(M, text):
        return [int(t) for t in render_prompt_tokens(
            M["ad"], {"messages": [{"role": "user", "content": text}], "enable_thinking": False})]

    TOPIC = args.task.startswith("topic:")
    if TOPIC:
        # knowledge topic: fit-split probe texts of the topic (and below) train / validate, held-split texts test;
        # every text cut to --topic-tokens; ids = the first 2 tokens (the scored span starts at the 3rd, as in the
        # donor-gap screen), full = the whole cut text
        tid = args.task[6:]
        rows_ = [json.loads(line) for line in open(args.probes)]
        bos = [] if args.bos == "off" or getattr(tok, "bos_token_id", None) is None else [int(tok.bos_token_id)]
        excl_ids, excl_groups = set(), set()
        if SPLIT_MAN is not None:
            excl_ids, excl_groups = split_exclusions(SPLIT_MAN)

        def dcheck(r, ids_):
            if Dn["tok"].encode(r["text"], add_special_tokens=False)[:args.topic_tokens] != ids_:
                sys.exit(f"donor tokenizer differs on probe text {r['id']}")
        items = topic_selection(rows_, tid, args.seed, args.n_train, args.n_val, args.n_test, args.topic_tokens,
                                lambda t: tok.encode(t, add_special_tokens=False), bos=bos, check=dcheck,
                                exclude_ids=excl_ids, exclude_groups=excl_groups)
        if SPLIT_MAN is not None:
            bad_ = [it["probe_id"] for it in items if it["probe_id"] in excl_ids or str(it["group"]) in excl_groups]
            if bad_:  # unreachable after the exclusion; kept as the fail-closed statement of the rule
                sys.exit(f"selected reserved final-test texts {bad_[:5]}")
            if args.split_check:
                err_ = split_check(SPLIT_MAN, items)
                if err_:
                    sys.exit(f"--split-check: selection differs from {args.split_manifest}: {err_}")
        n_ = {k: sum(it["split"] == k for it in items) for k in ("train", "val", "test")}
        if n_["test"] < 10 or n_["train"] < 4 or n_["val"] < 4:
            sys.exit(f"topic {tid}: too few probe texts of >= {args.topic_tokens} tokens: {n_}")
    else:
        items = make_items(args.task, args.n_train, args.n_val, args.n_test, args.seed)
    for it in items:
        if TOPIC:
            continue
        it["ids"] = render(H, it["prompt"])
        if render(Dn, it["prompt"]) != it["ids"]:
            sys.exit(f"donor and host render {it['id']} differently: the shared-tokenizer premise fails")
    eng_all = [json.loads(line) for line in open(CORE)]
    fit_e = [r for r in eng_all if r["split"] == "fit"]
    held_e = [r for r in eng_all if r["split"] == "held"]
    g = np.random.default_rng(args.seed + 7)
    pick_f = g.choice(len(fit_e), args.eng_retain + args.eng_val, replace=False)
    pick_h = g.choice(len(held_e), args.eng_test, replace=False)
    eng = ([{"id": fit_e[i]["id"], "role": "retain", "text": fit_e[i]["text"]} for i in pick_f[:args.eng_retain]]
           + [{"id": fit_e[i]["id"], "role": "eng_val", "text": fit_e[i]["text"]} for i in pick_f[args.eng_retain:]]
           + [{"id": held_e[i]["id"], "role": "eng_test", "text": held_e[i]["text"]} for i in pick_h])
    bos_e = [] if args.bos == "off" or getattr(tok, "bos_token_id", None) is None else [int(tok.bos_token_id)]
    for e in eng:
        body = tok.encode(e["text"], add_special_tokens=False)[:args.eng_tokens]
        if Dn["tok"].encode(e["text"], add_special_tokens=False)[:args.eng_tokens] != body:
            sys.exit(f"donor tokenizer differs on English text {e['id']}")
        e["ids"] = bos_e + body
    used_e = {e["id"] for e in eng}
    lens_pool = [r for r in fit_e if r["id"] not in used_e]
    lens_txt = []
    for i in np.random.default_rng(args.seed + 13).permutation(len(lens_pool)):
        if len(lens_txt) >= args.lens_texts:
            break
        body = tok.encode(lens_pool[i]["text"], add_special_tokens=False)[:args.lens_tokens]
        if len(body) < 16 or Dn["tok"].encode(lens_pool[i]["text"], add_special_tokens=False)[:args.lens_tokens] != body:
            continue
        lens_txt.append({"id": lens_pool[i]["id"], "ids": bos_e + body})
    eos = set(tok.eos_token_ids) | set(Dn["tok"].eos_token_ids)
    print(f"host {H['ck']} ({LH} layers), donor {Dn['ck']} ({LD} layers), rev {rev} dirty {dirty}; "
          f"{len(items)} items, {len(eng)} English texts; shared tokenizer verified", flush=True)

    # ------------------------------------------------------------------ hooks: MLP unit capture, layer writes
    MCTL = {}

    CAP_CLS = {}

    def patch_mlp(M):
        """Units = the dense MLP's hidden activations, read at the input of its down_proj, whatever the activation
        (SiLU-GLU for Qwen, GELU-GLU for Gemma 4) and whatever the projection type (Linear or QuantizedLinear). An
        MoE layer's experts are not units here (Gemma 4 26B-A4B also has a dense MLP per layer, which is)."""
        for L in M["layers"]:
            dp = L.mlp.down_proj
            base = type(dp)
            if base not in CAP_CLS:
                def make(base_):
                    class DownCap(base_):
                        def __call__(self, x, *a, **kw):
                            c = MCTL.get(id(self))
                            if c is not None:
                                if c.get("zmul") is not None:  # unit ablation: x * zmul + zadd, [1, T, I]
                                    x = x * c["zmul"].astype(x.dtype)
                                    if c.get("zadd") is not None:
                                        x = x + c["zadd"].astype(x.dtype)
                                if "idx" in c:
                                    c["out"] = (x[0].astype(mx.float32) if c["idx"] is None
                                                else x[0][:, c["idx"]].astype(mx.float32))
                            return base_.__call__(self, x, *a, **kw)
                    return DownCap
                CAP_CLS[base] = make(base)
            dp.__class__ = CAP_CLS[base]
    # MoE routers gather with indices computed from the residual (argpartition of router scores); under a gradient
    # trace MLX then tries a VJP with respect to those indices and refuses. Integer index ops never carry gradient,
    # so wrapping their outputs in stop_gradient is an exact identity for every model, and lets tiles and bridges
    # train through an MoE host (Gemma 4 26B-A4B; tiny-Gemma CPU smoke caught it).
    for nm_ in ("argpartition", "argsort", "argmax", "argmin"):
        f_ = getattr(mx, nm_)
        if not getattr(f_, "_amoeba_sg", False):
            def wrap(f__):
                def g__(*a, **kw):
                    return mx.stop_gradient(f__(*a, **kw))
                g__._amoeba_sg = True
                return g__
            setattr(mx, nm_, wrap(f_))
    patch_mlp(H)
    patch_mlp(Dn)

    def out_of(o):
        return o[0] if isinstance(o, tuple) else o

    def with_out(o, x):
        return (x,) + tuple(o[1:]) if isinstance(o, tuple) else x

    WMASK = [None]  # [T] multiplier on every host write (the autoregressive split sets it; None = write everywhere)

    def run(M, ids, cache=None, writes=None, tiles=None, cap=(), st=None, units=None, ablate=None):
        """log-probs [T, V] float32. writes {L: [T, D]} and tiles are budgeted; cap: layer outputs kept in st['h'];
        units {layer: idx or None}: MLP unit activations kept in st['u']; ablate {layer: [1, T, I] multiplier} on the
        MLP hidden units (the input of down_proj)."""
        layers = M["layers"]
        wl = set(writes or {}) | (set(tiles.layers_set) if tiles is not None else set())
        up = M is H and RES32
        hooked = wl | {l - 1 for l in wl if l > 0} | set(cap) | ({0} if up else set())
        prev = {}

        def hook(l):
            def h(o):
                x = out_of(o)
                if up and l == 0:
                    x = x.astype(mx.float32)
                if l in wl:
                    x0 = x
                    v = mx.zeros_like(x0)
                    if tiles is not None and l in tiles.layers_set:
                        v = v + tiles(l, x0)
                    if writes and l in writes:
                        v = v + writes[l][None].astype(x0.dtype)
                    ref = prev.get(l - 1)
                    d0 = (x0 - ref) if ref is not None else x0
                    loc = mx.stop_gradient(mx.sqrt((d0.astype(mx.float32) ** 2).sum(-1, keepdims=True)))
                    r_ = mx.sqrt((v.astype(mx.float32) ** 2).sum(-1, keepdims=True) + 1e-12) / (loc + 1e-12)
                    if st is not None:
                        st.setdefault("nr", []).append(mx.stop_gradient(r_).reshape(-1))
                    if args.norm_budget > 0:
                        v = v * mx.minimum(1.0, args.norm_budget / (r_ + 1e-12)).astype(v.dtype)
                    if WMASK[0] is not None and M is H:
                        v = v * WMASK[0][:, None].astype(v.dtype)
                    x = x0 + v.astype(x0.dtype)
                if l in cap:
                    st.setdefault("h", {})[l] = x[0].astype(mx.float32)
                prev[l] = x
                return with_out(o, x)
            return h

        if units:
            for l, idx in units.items():
                MCTL[id(layers[l].mlp.down_proj)] = {"idx": idx}
        for l, zm in (ablate or {}).items():  # zm: a multiplier, or (multiplier, additive)
            e_ = MCTL.setdefault(id(layers[l].mlp.down_proj), {})
            e_["zmul"], e_["zadd"] = (zm if isinstance(zm, tuple) else (zm, None))
        try:
            with ra._hooked([(layers[l], hook(l)) for l in sorted(hooked)]):
                cache = M["model"].make_cache() if cache is None else cache
                lg = M["model"](mx.array([ids]), cache=cache)[0].astype(mx.float32)
            if lg.shape[0] != len(ids):  # a trunk that kept only the last row (prefill tail skip) would misalign
                raise RuntimeError(f"{M['ck']}: {lg.shape[0]} logit rows for {len(ids)} tokens; every row is needed")
        finally:
            if units:
                got = {l: MCTL.pop(id(layers[l].mlp.down_proj)).get("out") for l in units}
                if st is not None:
                    st["u"] = got
            for l in (ablate or {}):
                MCTL.pop(id(layers[l].mlp.down_proj), None)
        return lg - mx.logsumexp(lg, axis=-1, keepdims=True)

    def budget_left():
        return args.budget_s - (time.time() - t0)

    # ------------------------------------------------------------------ host gradient path: reference vs kernel
    # MLX has no VJP for mx.fast.metal_kernel launches. Every gradient below runs the host inside
    # dref.reference_path(), which routes the kernel sites (GatedDeltaNet recurrence and fused GDN kernels, the
    # opt-in Metal attention kernels, the precise sigmoid) to their pure-MLX reference; forward-only passes keep
    # the serving kernels. Before any stage spends time: measure the two paths' forward parity on a few real
    # sequences and take one real gradient through the host, fail closed if either is off.
    def host_grad_check():
        seqs = [("topic" if TOPIC else "task", it["full"] if TOPIC else it["ids"]) for it in items[:2]] + [
            ("english", e["ids"]) for e in eng[:2]]
        rows_, mx_abs, kls, agree = [], 0.0, [], []
        dref.reset_reached()
        for kind, ids in seqs:
            lk = run(H, ids)
            with dref.reference_path():
                lr = run(H, ids)
            kl_ = (mx.exp(lk) * (lk - lr)).sum(-1)
            a_ = float(mx.abs(lk - lr).max().item())
            m_ = float(kl_.mean().item())
            ag = float((mx.argmax(lk, -1) == mx.argmax(lr, -1)).astype(mx.float32).mean().item())
            rows_.append({"kind": kind, "tokens": len(ids), "max_abs_logprob": a_, "kl_kernel_ref_mean": m_,
                          "kl_kernel_ref_max": float(kl_.max().item()), "top1_agreement": ag})
            mx_abs = max(mx_abs, a_)
            kls.append(m_)
            agree.append(ag)
        ids = max((q[1] for q in seqs), key=len)  # the longest: closest to the stage sequences' row counts
        wl_ = sorted({LH // 2, max(LH - 2, 0)})
        tgt = mx.array(ids[1:])[:, None]
        mx.reset_peak_memory()
        a0 = mx.get_active_memory()
        try:
            with dref.reference_path():
                gr_ = mx.grad(lambda w: -mx.take_along_axis(run(H, ids, writes=w)[:-1], tgt, -1).mean())(
                    {L: mx.zeros((len(ids), D_h0)) for L in wl_})
                mx.eval(gr_)
            gnorm = {str(L): float(mx.sqrt((gr_[L] * gr_[L]).sum()).item()) for L in wl_}
            probe = {"ok": all(np.isfinite(v) and v > 0 for v in gnorm.values()), "write_layers": wl_,
                     "tokens": len(ids), "grad_norm": gnorm, "peak_gib": round(mx.get_peak_memory() / 2**30, 2),
                     "active_gib_before": round(a0 / 2**30, 2)}
        except ValueError as exc:  # e.g. "[Primitive::vjp] Not implemented for CustomKernel": a site not routed
            probe = {"ok": False, "error": str(exc)[:300], "write_layers": wl_}
        on_gpu = mx.default_device() == mx.gpu and mx.metal.is_available()
        return {"path": "reference" if on_gpu else "reference (cpu: kernel and reference paths coincide)",
                "switch": "mlx2.runtime.models.differentiable.reference_path", "routed": dref.ROUTED_KERNELS,
                # selection-site calls that took the reference during these forwards and the probe: a site this
                # host never consults is absent (its kernels, if any, are not covered; the probe says if they bite)
                "sites_consulted": dref.reached(),
                "forward_uses": "serving kernel path", "kernel_path_available": bool(on_gpu),
                "parity": {"sequences": rows_, "max_abs_logprob": mx_abs, "kl_mean": float(np.mean(kls)),
                           "top1_agreement_min": float(min(agree)), "max_kl_allowed": args.grad_ref_max_kl},
                "grad_probe": probe}
    D_h0 = int(H["layers"][0].mlp.down_proj.weight.shape[0])
    hg = ({"path": "not used (--measure: forward passes only, no gradient)", "parity": {"max_abs_logprob": 0.0,
           "kl_mean": 0.0, "top1_agreement_min": 1.0}, "grad_probe": {"ok": True, "skipped": "--measure"}}
          if args.measure else host_grad_check())
    save_receipt(host_grad_path=hg)
    print(f"host grad path: {hg['path']}; forward parity max |dlogp| {hg['parity']['max_abs_logprob']:.3g}, "
          f"mean KL {hg['parity']['kl_mean']:.3g}, top-1 agreement >= {hg['parity']['top1_agreement_min']:.3f}; "
          f"grad probe {'ok' if hg['grad_probe']['ok'] else 'FAILED'}", flush=True)
    if not hg["grad_probe"]["ok"]:
        save_receipt(status="failed", reason=f"host gradient probe failed: {hg['grad_probe']}")
        sys.exit(f"host gradient probe failed: {hg['grad_probe']}")
    if not (hg["parity"]["kl_mean"] <= args.grad_ref_max_kl):
        save_receipt(status="failed", reason=f"host reference path differs from its kernel path: mean KL "
                                             f"{hg['parity']['kl_mean']:.4g} > {args.grad_ref_max_kl}")
        sys.exit("host reference path differs from its kernel path beyond --grad-ref-max-kl")

    # ------------------------------------------------------------------ donor stage 1: traces and ceilings
    def donor_generate(it, temp, key, M=None):
        M = Dn if M is None else M
        c = M["model"].make_cache()
        lp = run(M, it["ids"], cache=c)[-1]
        toks = []
        for _ in range(args.max_tokens):
            if temp > 0:
                key, sk = mx.random.split(key)
                t = int(mx.random.categorical(lp / temp, key=sk).item())
            else:
                t = int(mx.argmax(lp).item())
            toks.append(t)
            if t in eos:
                break
            lp = run(M, [t], cache=c)[-1]
        return toks, key

    if args.screen and TOPIC:
        sys.exit("--screen is for arithmetic tasks; the knowledge screen is amoeba_topics_donor_gap.py on main")
    if args.screen:
        res = {}
        for spec in [x.strip() for x in args.screen.split(",") if x.strip()]:
            its = make_items(spec, 0, 0, args.n_test, args.seed + 1000)
            acc = {"host": [], "donor": []}
            cut = {"host": 0, "donor": 0}
            pairs = []
            srows = open(out / f"screen_rows_{spec.replace(':', '_')}.jsonl", "w")
            for it in its:
                it["ids"] = render(H, it["prompt"])
                if render(Dn, it["prompt"]) != it["ids"]:
                    sys.exit(f"donor and host render a {spec} item differently")
                pr = {}
                for nm, M in (("host", H), ("donor", Dn)):
                    toks, _ = donor_generate(it, 0.0, None, M)
                    stopped = bool(toks) and toks[-1] in eos
                    text = tok.decode([t for t in toks if t not in eos])
                    pr[nm] = parse(text, stopped)
                    pr[f"{nm}_text"] = text
                    cut[nm] += not stopped
                    acc[nm].append(pr[nm] == it["answer"])
                pairs.append({"prompt": it["prompt"], "answer": it["answer"], **pr})
                srows.write(json.dumps(pairs[-1]) + "\n")
            srows.close()
            h_, d_ = float(np.mean(acc["host"])), float(np.mean(acc["donor"]))
            res[spec] = {"task": task_name(spec), "n": len(its), "host_acc": h_, "donor_acc": d_,
                         "gap_points": round(100 * (d_ - h_), 1), "max_tokens": args.max_tokens,
                         "cut_off": cut, "examples": pairs[:5]}
            (out / "screen.json").write_text(json.dumps(res, indent=1))
            print(f"{time.strftime('%H:%M:%S')} screen {spec}: host {h_:.2f} donor {d_:.2f} gap "
                  f"{100 * (d_ - h_):+.0f} points (n {len(its)}; cut off host {cut['host']} donor {cut['donor']})",
                  flush=True)
        save_receipt(status="complete (screen)", screen=res)
        return

    def measure_box_teacher(te):
        """--measure (proposal v4 §5.0a and b): forward passes only, nothing fitted. Per test text, on identical
        prefixes: host H, donor D, donor D- (every box layer mean-ablated at every position to the English mean,
        as summary.donor_box_ar_split's "all" mode does) and D with --rand-seeds layer-matched, count-matched random
        unit sets mean-ablated the same way. Gold nats/byte of each, KL(D || ablated), and the box-contrast teacher
        q = softmax(log p_H + eta * r(log p_D - log p_ablated)) per --teacher-modes and --eta-grid, with the random
        ablations' contrast as its control. Every contrast is paired per text (amoeba_paired_stats)."""
        import amoeba_paired_stats as PS
        bj_path = Path(args.box_json)
        bj = json.loads(bj_path.read_text())
        blayers = [int(x) for x in bj["box_layers"]]
        if any(not (0 <= l < LD) for l in blayers):
            sys.exit(f"--box-json layers {blayers} outside the donor's {LD} layers")
        bx = {l: np.array(bj["box"][str(l)], dtype=np.int64) for l in blayers}
        st0 = {}
        run(Dn, eng[0]["ids"], st=st0, units={l: None for l in blayers})
        width = {l: int(st0["u"][l].shape[1]) for l in blayers}
        del st0
        rj_path = bj_path.parent / "run.json"
        bad_, box_identity = box_json_identity(bj, json.loads(rj_path.read_text()) if rj_path.exists() else None,
                                               Dn["ck"], width)
        if bad_:
            sys.exit("--box-json is not a box of the loaded donor: " + "; ".join(bad_))
        if box_identity.get("warning"):
            print(f"WARNING --box-json {bj_path}: {box_identity['warning']}", flush=True)
        sets = {"box": bx}
        for k_ in range(args.rand_seeds):  # own generators: the draw depends only on --seed and k
            g_ = np.random.default_rng(args.seed + 1000 + k_)
            sets[f"rand{k_}"] = {l: np.sort(g_.choice(np.setdiff1d(np.arange(width[l]), bx[l]), len(bx[l]),
                                                      replace=False)) for l in blayers}
        rnames = [n_ for n_ in sets if n_ != "box"]
        retain_ = [e for e in eng if e["role"] == "retain"]
        acc_ = {n_: {l: [] for l in blayers} for n_ in sets}
        for e in retain_:  # English mean activation per unit, position 0 (the attention-sink token) excluded
            st_ = {}
            run(Dn, e["ids"], st=st_, units={l: None for l in blayers})
            for l in blayers:
                a_ = np.array(st_["u"][l])[1:]
                for n_, S_ in sets.items():
                    acc_[n_][l].append(a_[:, S_[l]])
            del st_
        mu_ = {n_: {l: np.concatenate(v).mean(0).astype(np.float32) for l, v in d_.items()} for n_, d_ in acc_.items()}
        ZM = {}
        for n_, S_ in sets.items():  # [1, 1, I] multiplier and additive term, broadcast over every position
            z_ = {}
            for l in blayers:
                keep_ = np.ones(width[l], np.float32)
                keep_[S_[l]] = 0.0
                add_ = np.zeros(width[l], np.float32)
                add_[S_[l]] = mu_[n_][l]
                z_[l] = (mx.array(keep_[None, None]), mx.array(add_[None, None]))
            ZM[n_] = z_
        etas = [float(x) for x in args.eta_grid.split(",") if x.strip()]
        modes = [m_.strip() for m_ in args.teacher_modes.split(",") if m_.strip()]
        if any(m_ not in ("clip", "topk") for m_ in modes):
            sys.exit(f"unknown --teacher-modes {args.teacher_modes}")
        mrec = {"box_json": str(bj_path.resolve()), "box_json_sha256": hashlib.sha256(bj_path.read_bytes()).hexdigest(),
                "box_identity": box_identity,
                "box_layers": blayers, "units_per_layer": {str(l): int(len(bx[l])) for l in blayers},
                "donor_mlp_width": {str(l): width[l] for l in blayers},
                "random_sets": {n_: {str(l): S_[l].tolist() for l in blayers} for n_, S_ in sets.items() if n_ != "box"},
                "ablation": "mean ablation to the English mean activation of the same units (the --eng-retain "
                            "English texts, position 0 excluded), every box layer, every position (prefix included): the receipt's "
                            "donor_box_ar_split 'all' mode",
                "english_texts": [e["id"] for e in retain_], "eta_grid": etas, "teacher_modes": modes,
                "teacher_clip": args.teacher_clip, "teacher_topk": args.teacher_topk,
                "teacher_topk_clip": args.teacher_topk_clip,
                "teacher": "log q = log_softmax(log p_H + eta * r), r = log p_D - log p_ablated; clip: r clipped to "
                           "+-teacher_clip; topk: r only on the union of both donor arms' top-k tokens (clipped to "
                           "+-teacher_topk_clip when > 0); the random-ablation contrast is the control",
                "texts": "the 200 held physics texts, SPENT (review v4 §5.1): development data, not a final test",
                "box_status": receipt["box_status"]}
        save_receipt(stage="measure", measure=mrec)
        rows_m = open(out / "measure_rows.jsonl", "w")
        recs, vinfo = [], {}
        tm_ = time.time()
        for j_, s in enumerate(te):
            if budget_left() < 120:
                break
            pos = mx.array(s["pos"])
            tgt = mx.array(s["ids"][s["P"]:])[:, None]
            lh = run(H, s["ids"])[pos]
            ld = run(Dn, s["ids"])[pos]
            la = {n_: run(Dn, s["ids"], ablate=ZM[n_])[pos] for n_ in sets}

            def gold(lp_):
                return float(-mx.take_along_axis(lp_, tgt, axis=-1).sum().item())

            def klsum(a_, b_):
                return float((mx.exp(a_) * (a_ - b_)).sum().item())
            V_ = min(lh.shape[-1], ld.shape[-1])
            if lh.shape[-1] != ld.shape[-1]:  # padded heads: the teacher works on the shared vocabulary
                vinfo = {"host_vocab": int(lh.shape[-1]), "donor_vocab": int(ld.shape[-1])}
                if int(mx.max(tgt).item()) >= V_:
                    sys.exit("a gold token lies outside the shared vocabulary")
            ren = lambda x_: x_[:, :V_] - mx.logsumexp(x_[:, :V_], axis=-1, keepdims=True)  # noqa: E731
            lh_v, ld_v = ren(lh), ren(ld)
            la_v = {n_: ren(v_) for n_, v_ in la.items()}
            r_ = {"item": s["item"], "probe_id": s.get("probe_id"), "group": s.get("group"),
                  "bytes": len(tok.decode(s["ids"][s["P"]:]).encode("utf-8")), "tokens": len(s["pos"]),
                  "nats": {"host": gold(lh), "donor": gold(ld), **{n_: gold(v_) for n_, v_ in la.items()}},
                  "kl_donor_ablated_sum": {n_: klsum(ld, v_) for n_, v_ in la.items()},
                  "teacher_nats": {}, "teacher_kl_host_sum": {}}
            for m_ in modes:
                for eta in etas:
                    key_ = f"{m_}@{eta:g}"
                    r_["teacher_nats"][key_] = {}
                    for n_ in sets:
                        q_ = contrast_teacher(mx, lh_v, ld_v, la_v[n_], eta, m_,
                                              args.teacher_clip if m_ == "clip" else args.teacher_topk_clip,
                                              args.teacher_topk)
                        r_["teacher_nats"][key_][n_] = gold(q_)
                        if n_ == "box":
                            r_["teacher_kl_host_sum"][key_] = klsum(q_, lh_v)
                        del q_
            if j_ == 0 and not r_["kl_donor_ablated_sum"]["box"] > 0:  # the down_proj hook must have bitten
                sys.exit(f"box ablation changed nothing on {s['item']}: KL(D || D-) = {r_['kl_donor_ablated_sum']['box']}")
            rows_m.write(json.dumps(r_) + "\n")
            rows_m.flush()
            recs.append(r_)
            del lh, ld, la, lh_v, ld_v, la_v
            if j_ % 20 == 0:
                print(f"{time.strftime('%H:%M:%S')} measure {j_ + 1}/{len(te)} ({time.time() - tm_:.0f}s)", flush=True)
                mx.clear_cache()
        rows_m.close()
        groups = [r_["group"] for r_ in recs]
        mue = args.min_useful_npb
        if args.reference_rows:
            ref_ = {}
            for ln_ in open(args.reference_rows):
                q_ = json.loads(ln_)
                if q_.get("arm") == "host":
                    ref_[q_["item"]] = q_["nats_per_byte"]
            d_ = [r_["nats"]["host"] / r_["bytes"] - ref_[r_["item"]] for r_ in recs if r_["item"] in ref_]
            mrec["reference_rows"] = {"path": args.reference_rows, "matched_items": len(d_), "texts": len(recs),
                                      "host_npb_max_abs_diff": float(np.max(np.abs(d_))) if d_ else None,
                                      "host_npb_mean_diff": float(np.mean(d_)) if d_ else None}
            save_receipt(measure=mrec)
            if len(d_) != len(recs):
                sys.exit(f"--reference-rows: only {len(d_)} of {len(recs)} item ids found")

        def P(d_, mue_=mue):
            return PS.paired(d_, groups, mue=mue_)

        def npb(key_):
            return np.array([r_["nats"][key_] / r_["bytes"] for r_ in recs])

        def kl_tok(key_):
            return np.array([r_["kl_donor_ablated_sum"][key_] / r_["tokens"] for r_ in recs])

        def kl_byte(key_):
            return np.array([r_["kl_donor_ablated_sum"][key_] / r_["bytes"] for r_ in recs])
        Hn, Dn_, Dm = npb("host"), npb("donor"), npb("box")
        Rn = np.stack([npb(n_) for n_ in rnames])
        Rk_tok = np.stack([kl_tok(n_) for n_ in rnames])
        Rk_byte = np.stack([kl_byte(n_) for n_ in rnames])
        box_rand = P(Dm - Rn.mean(0))
        worth = P(Dm - Dn_)
        a0 = {"means_npb": {"host": float(Hn.mean()), "donor": float(Dn_.mean()), "box_ablated": float(Dm.mean()),
                            **{n_: float(Rn[i].mean()) for i, n_ in enumerate(rnames)}},
              "host_minus_donor": P(Hn - Dn_),
              "box_worth_ablated_minus_donor": worth,
              "random_cost_ablated_minus_donor": {n_: P(Rn[i] - Dn_) for i, n_ in enumerate(rnames)},
              "random_cost_mean_of_seeds": P(Rn.mean(0) - Dn_),
              "box_minus_random_ablated": box_rand,
              "box_worth_share_of_host_donor_gap": PS.ratio(Dm - Dn_, Hn - Dn_, groups),
              "kl_donor_box_ablated_per_token": P(kl_tok("box"), None),
              "kl_donor_box_ablated_per_byte": P(kl_byte("box"), None),
              "kl_donor_random_ablated_per_token_mean_of_seeds": P(Rk_tok.mean(0), None),
              "kl_donor_random_ablated_per_byte_mean_of_seeds": P(Rk_byte.mean(0), None),
              "kl_box_minus_random_per_token": P(kl_tok("box") - Rk_tok.mean(0), None),
              "reference": "physics receipt donor_box_ar_split all = 0.0104 nats/token on 128 val positions",
              "reading_rule": "box-specific arms are testable on this box only if box_minus_random_ablated's group "
                              "95% interval excludes 0 from above AND box_worth is not 'below' the minimum useful "
                              "effect (proposal v4 §5.0a); a dev-data reading, Pierre decides",
              "box_testable": bool(box_rand.get("ci95_group", [0])[0] > 0
                                   and worth.get("vs_min_useful_effect") != "below")}
        b0 = {}
        for m_ in modes:
            base = np.array([r_["teacher_nats"][f"{m_}@0"]["box"] / r_["bytes"] for r_ in recs]) if 0.0 in etas else Hn
            for eta in etas:
                key_ = f"{m_}@{eta:g}"
                qb = np.array([r_["teacher_nats"][key_]["box"] / r_["bytes"] for r_ in recs])
                qr = np.stack([[r_["teacher_nats"][key_][n_] / r_["bytes"] for r_ in recs] for n_ in rnames])
                b0[key_] = {"q_npb": float(qb.mean()), "q_rand_npb_mean_of_seeds": float(qr.mean()),
                            "q_gain_vs_host": P(base - qb), "q_rand_gain_vs_host": P(base - qr.mean(0)),
                            "q_rand_gain_vs_host_per_seed": {n_: P(base - qr[i]) for i, n_ in enumerate(rnames)},
                            "box_minus_random_teacher": P(qr.mean(0) - qb),
                            "kl_q_host_per_token": float(np.mean([r_["teacher_kl_host_sum"][key_] / r_["tokens"]
                                                                  for r_ in recs]))}
        b0["host_reference"] = ("eta 0 of the same mode: p_H on the shared vocabulary, renormalised"
                                + (f" ({vinfo})" if vinfo else "; host and donor heads have the same width"))
        b0["eta0_minus_host_npb_max_abs"] = float(np.max(np.abs(
            np.array([r_["teacher_nats"][f"{modes[0]}@0"]["box"] / r_["bytes"] for r_ in recs]) - Hn))) if 0.0 in etas else None
        b0["reading_rule"] = ("q is an upper reference for a box-specific behavioral write: if no eta gives "
                              "q_gain_vs_host above 0 and box_minus_random_teacher above 0, no write fitted to q should "
                              "be expected to help (proposal v4 §5.0b); positive = lower gold loss than the host")
        summ = {"label": label or "unqualified research", "qualification": "unqualified research",
                "host_checkpoint": H["ck"], "donor_checkpoint": Dn["ck"], "mlx2_source_revision": rev,
                "tree_dirty": dirty, "task": args.task, "n_texts": len(recs), "n_texts_requested": len(te),
                "box_status": receipt["box_status"], "min_useful_effect_npb": mue,
                "resolution_note": "every contrast: paired_se and mde80 (2.80 x SE, two-sided 0.05, power 0.8) over "
                                   "texts, se_group / mde80_group / ci95_group from a source-group bootstrap; "
                                   "vs_min_useful_effect flags the group interval against min_useful_effect_npb",
                "sign": "every contrast is oriented so that positive = the named first arm costs more / the "
                        "teacher is better", "0a_box_worth": a0, "0b_teacher_worth": b0}
        (out / "summary.json").write_text(json.dumps(summ, indent=1))
        done_ = len(recs) == len(te)
        save_receipt(status="complete" if done_ else "partial (budget)", stage="done (measure)", summary=summ)
        print(json.dumps({"box_worth": worth, "box_minus_random": box_rand}, indent=1), flush=True)

    rows_f = open(out / "rows.jsonl", "w")
    key = mx.random.key(args.seed)
    seqs = []  # topic training / validation sequences: prompt + donor continuation
    donor_acc = {"train": [], "val": [], "test": []}
    donor_pred = {}
    for it in (items if not TOPIC else []):
        k_ = 1 + (args.samples if it["split"] == "train" else 0)
        for s in range(k_):
            toks, key = donor_generate(it, 0.0 if s == 0 else args.temperature, key)
            stopped = bool(toks) and toks[-1] in eos
            text = tok.decode([t for t in toks if t not in eos])
            ok = parse(text, stopped) == it["answer"]
            if s == 0:
                donor_acc[it["split"]].append(ok)
                if it["split"] == "test":
                    donor_pred[it["id"]] = parse(text, stopped)
                    rows_f.write(json.dumps({"arm": "donor", "item": it["id"], "text": text,
                                             "pred": parse(text, stopped), "stopped": stopped, "correct": ok}) + "\n")
            if it["split"] in ("train", "val") and (it["split"] == "train" or s == 0):
                seqs.append({"item": it["id"], "split": it["split"], "sample": s, "P": len(it["ids"]),
                             "ids": it["ids"] + toks, "donor_correct": ok})
    if TOPIC:  # the text itself is the sequence; there is no donor reply and no oracle (donor_correct is moot)
        for it in items:
            if it["split"] in ("train", "val"):
                seqs.append({"item": it["id"], "split": it["split"], "sample": 0, "P": len(it["ids"]),
                             "ids": it["full"], "donor_correct": True})
        te = [{"item": it["id"], "split": "test", "sample": 0, "P": len(it["ids"]), "ids": it["full"],
               "donor_correct": True, "group": it.get("group"), "probe_id": it.get("probe_id")}
              for it in items if it["split"] == "test"]
        print(f"{time.strftime('%H:%M:%S')} topic {args.task[6:]}: {sum(q['split'] == 'train' for q in seqs)} train, "
              f"{sum(q['split'] == 'val' for q in seqs)} val, {len(te)} test texts of {args.topic_tokens} tokens",
              flush=True)
    else:
        te = []
        rows_f.flush()
        print(f"{time.strftime('%H:%M:%S')} donor greedy accuracy train {np.mean(donor_acc['train']):.3f} "
              f"val {np.mean(donor_acc['val']):.3f} test {np.mean(donor_acc['test']):.3f}; {len(seqs)} sequences",
              flush=True)
    tr = [s for s in seqs if s["split"] == "train"]
    va = [s for s in seqs if s["split"] == "val"]
    for s in seqs + te:
        s["pos"] = list(range(s["P"] - 1, len(s["ids"]) - 1))  # positions predicting the continuation
    if args.measure:
        measure_box_teacher(te)
        return
    by_id = {it["id"]: it for it in items}
    # wrong-item partner for the shuffled-feature arms: same prompt length, different item, fixed derangement
    for group in (tr, va, te):
        idsets = sorted({s["item"] for s in group})
        perm = {}
        for i, iid in enumerate(idsets):
            P = len(by_id[iid]["ids"])
            cands = [j for j in idsets[i + 1:] + idsets[:i] if len(by_id[j]["ids"]) == P]
            perm[iid] = cands[0] if cands else None
        for s in group:
            s["partner"] = perm[s["item"]]

    # ------------------------------------------------------------------ donor stage 2: teacher, unit statistics
    def dense_w(lin):
        if "scales" in lin:
            kw = {"mode": lin.mode} if hasattr(lin, "mode") else {}
            return mx.dequantize(lin.weight, lin.scales, lin.get("biases"), lin.group_size, lin.bits, **kw)
        return lin.weight

    def col_norms(lin):
        w = dense_w(lin).astype(mx.float32)
        n = np.array(mx.sqrt((w * w).sum(0)))
        del w
        return n
    wn_d = {l: col_norms(Dn["layers"][l].mlp.down_proj) for l in range(LD)}
    wn_h = {l: col_norms(H["layers"][l].mlp.down_proj) for l in range(LH)}

    def unit_mass(M, wn, texts, topic):
        L_ = len(M["layers"])
        acc = [np.zeros_like(wn[l]) for l in range(L_)]
        n = 0
        for s in texts:
            st = {}
            run(M, s["ids"], st=st, units={l: None for l in range(L_)})
            p = s["pos"] if topic else list(range(len(s["ids"])))
            for l in range(L_):
                acc[l] += np.abs(np.array(st["u"][l])[p]).sum(0) * wn[l]
            n += len(p)
            del st
        return np.stack(acc) / max(n, 1)

    retain = [e for e in eng if e["role"] == "retain"]
    eng_val = [e for e in eng if e["role"] == "eng_val"]
    eng_test = [e for e in eng if e["role"] == "eng_test"]
    for s in tr + va:  # donor teacher: full-vocab log-probs at the continuation positions (float16)
        s["t_lp"] = np.array(run(Dn, s["ids"])[mx.array(s["pos"])]).astype(np.float16)
    stat_tr = [s for s in tr if s["sample"] == 0]
    md_t, md_e = unit_mass(Dn, wn_d, stat_tr, True), unit_mass(Dn, wn_d, retain, False)
    mh_t, mh_e = unit_mass(H, wn_h, stat_tr, True), unit_mass(H, wn_h, retain, False)
    print(f"{time.strftime('%H:%M:%S')} teacher log-probs and unit masses done", flush=True)

    def select(mt, me, k, nl, fixed):
        score = mt - me
        elig = mt >= args.min_ratio * me
        per = np.where(elig, score, -np.inf)
        sel = {l: np.argsort(-per[l])[:k] for l in range(len(mt))}
        sel = {l: v[np.isfinite(per[l][v])] for l, v in sel.items()}
        lscore = np.array([per[l][sel[l]].sum() if len(sel[l]) else -np.inf for l in range(len(mt))])
        if fixed != "auto":
            layers_ = [int(x) for x in fixed.split(",")]
        else:
            layers_ = []
            for l in np.argsort(-lscore):
                if 0 < l < len(mt) - 1 and np.isfinite(lscore[l]) and all(abs(l - m) >= 2 for m in layers_):
                    layers_.append(int(l))
                if len(layers_) == nl:
                    break
        return sorted(layers_), {l: sel[l] for l in layers_}, lscore

    box_layers, box, d_lscore = select(md_t, md_e, args.box_units, args.n_box_layers, args.box_layers)
    if not box_layers or not all(len(box[l]) for l in box_layers):
        save_receipt(status="failed", reason=f"no donor box: layers {box_layers} (min-ratio {args.min_ratio})")
        sys.exit(f"no donor box passes --min-ratio {args.min_ratio}: layers {box_layers}")
    stub_layers, hbox, h_lscore = select(mh_t, mh_e, args.box_units, args.n_stub_layers, "auto")
    hunits = {l: mx.array(hbox[l].astype(np.int32)) for l in stub_layers}
    rand = {}
    for l in box_layers:
        free = np.setdiff1d(np.arange(len(wn_d[l])), box[l])
        rand[l] = rng.choice(free, len(box[l]), replace=False)
    union = {l: mx.array(np.concatenate([box[l], rand[l]]).astype(np.int32)) for l in box_layers}
    kb = {l: len(box[l]) for l in box_layers}
    off = np.cumsum([0] + [kb[l] for l in box_layers])
    colsd = {l: slice(int(off[i]), int(off[i + 1])) for i, l in enumerate(box_layers)}
    FD = int(off[-1])
    warp = {d: int(round(d * (LH - 1) / (LD - 1))) for d in box_layers}
    if len(set(warp.values())) < len(warp):
        save_receipt(status="failed", reason=f"box layers {box_layers} collide under the warp {warp}")
        sys.exit(f"box layers {box_layers} map to the same host layer: {warp}")
    box_info = {"donor_checkpoint": Dn["ck"], "host_checkpoint": H["ck"],  # read back by --measure --box-json
                "donor_mlp_width": {str(l): int(len(md_t[l])) for l in box_layers},
                "box_layers": box_layers, "units_per_layer": {str(l): int(kb[l]) for l in box_layers},
                "box": {str(l): box[l].tolist() for l in box_layers}, "rand": {str(l): rand[l].tolist() for l in box_layers},
                "topic_over_english": {str(l): float(md_t[l][box[l]].sum() / max(md_e[l][box[l]].sum(), 1e-12))
                                       for l in box_layers},
                "donor_layer_score": d_lscore.tolist(), "host_stub_layers": stub_layers,
                "host_layer_score": h_lscore.tolist(), "warp": {str(d): L for d, L in warp.items()}}
    (out / "box.json").write_text(json.dumps(box_info, indent=1))
    print(f"box layers {box_layers} units {[kb[l] for l in box_layers]} -> warp {[warp[d] for d in box_layers]}; "
          f"host stub {stub_layers}", flush=True)

    # ------------------------------------------------------------------ donor stage 3: features
    def dfeat(ids, cache=None, st=None):
        st = {} if st is None else st
        lp = run(Dn, ids, cache=cache, st=st, units=dict(union))
        fb = np.concatenate([np.array(st["u"][l])[:, :kb[l]] for l in box_layers], 1)
        fr = np.concatenate([np.array(st["u"][l])[:, kb[l]:] for l in box_layers], 1)
        return lp, fb, fr

    for s in tr + va + te:
        _, s["fb"], s["fr"] = dfeat(s["ids"])
        if s["partner"] is not None:
            _, s["fs"], _ = dfeat(by_id[s["partner"]]["full"] if TOPIC
                                  else by_id[s["partner"]]["ids"] + s["ids"][s["P"]:])
        else:
            s["fs"] = s["fb"][rng.permutation(len(s["fb"]))]
    for e in eng:
        _, e["fb"], e["fr"] = dfeat(e["ids"])
    for i, e in enumerate(eng):  # wrong-text features for the shuffled arms on English
        o = [x for x in eng if x["role"] == e["role"]]
        j = o[(o.index(e) + 1) % len(o)]
        fs = np.zeros_like(e["fb"])
        n_ = min(len(fs), len(j["fb"]))
        fs[:n_] = j["fb"][:n_]
        e["fs"] = fs
    def hfeat(ids, cache=None):  # host-only control features: the host's own stub units, from a clean host pass
        st = {}
        run(H, ids, cache=cache, st=st, units=dict(hunits))
        return np.concatenate([np.array(st["u"][l]) for l in stub_layers], 1)
    want_arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    use_host = bool(stub_layers) and (not want_arms or any(a.startswith("bridge_host") for a in want_arms))
    FDh, zh = 0, None
    if use_host:
        for s in tr + va + te + eng:
            s["fh"] = hfeat(s["ids"])
        allh = np.concatenate([s["fh"] for s in tr])
        mu_h, sd_h = allh.mean(0), allh.std(0) + 1e-6
        zh = lambda f: ((f - mu_h) / sd_h).astype(np.float32)  # noqa: E731
        for s in tr + va + te + eng:
            s["fh"] = zh(s["fh"])
        FDh = int(allh.shape[1])
    allb = np.concatenate([s["fb"] for s in tr])
    allr = np.concatenate([s["fr"] for s in tr])
    mu_b, sd_b = allb.mean(0), allb.std(0) + 1e-6
    mu_r, sd_r = allr.mean(0), allr.std(0) + 1e-6
    zb = lambda f: ((f - mu_b) / sd_b).astype(np.float32)  # noqa: E731
    zr = lambda f: ((f - mu_r) / sd_r).astype(np.float32)  # noqa: E731
    for s in tr + va + te + eng:
        s["fs"] = zb(s["fs"])
        s["fb"], s["fr"] = zb(s["fb"]), zr(s["fr"])
    print(f"{time.strftime('%H:%M:%S')} donor features done ({FD} features)", flush=True)

    # ------------------------------------------------------------------ host references
    for s in va:
        s["h_lp"] = np.array(run(H, s["ids"])[mx.array(s["pos"])]).astype(np.float16)
    print(f"teacher log-probs held: {sum(q['t_lp'].nbytes for q in tr + va) / 1e9:.2f} GB "
          f"({sum(len(q['pos']) for q in tr + va)} continuation positions)", flush=True)
    for e in eng:
        e["h_lp"] = np.array(run(H, e["ids"])).astype(np.float16)

    def kl_rows(t_lp, lp):
        t = mx.array(t_lp.astype(np.float32))
        return (mx.exp(t) * (t - lp)).sum(-1)

    host_val_kl = float(np.mean([float(kl_rows(s["t_lp"], mx.array(s["h_lp"].astype(np.float32))).mean().item())
                                 for s in va])) if va else float("nan")
    if not np.isfinite(host_val_kl):
        save_receipt(status="failed", reason="no validation traces (host val KL not finite)")
        sys.exit("no validation traces: host val KL is not finite")
    write_sets = {}
    for mode in [m.strip() for m in args.write_layers.split(",") if m.strip()]:
        if mode == "warp":
            write_sets[mode] = {warp[d]: colsd[d] for d in box_layers}
        elif mode == "stub":
            write_sets[mode] = {L: slice(0, FD) for L in stub_layers}
        else:
            sys.exit(f"unknown write-layer mode {mode}")
    all_w = sorted({L for ws in write_sets.values() for L in ws})
    # host increments at the write layers on the training text (all positions): h_L - h_{L-1}
    capl = sorted(set(all_w) | {L - 1 for L in all_w if L > 0})
    D_h = int(dense_w(H["layers"][0].mlp.down_proj).shape[0])
    inc = {L: [] for L in all_w}
    for s in (tr if all_w else []):  # --write-layers "" (tile / LoRA-only runs): nothing to capture
        st = {}
        run(H, s["ids"], cap=capl, st=st)
        for L in all_w:
            inc[L].append(np.array(st["h"][L] - st["h"][L - 1]) if L > 0 else np.array(st["h"][L]))
    inc = {L: np.concatenate(v) for L, v in inc.items()}
    Xb = np.concatenate([s["fb"] for s in tr])
    Xr = np.concatenate([s["fr"] for s in tr])
    Xh_all = np.concatenate([s["fh"] for s in tr]) if use_host else None
    # donor-derived target (the host's own increment carries no donor signal): the direction that moves the host
    # toward the donor at each write layer, -d KL(donor || host) / d h_L, at the continuation positions; rows
    # normalised to unit norm (direction only); the fitted map is rescaled so its mean write norm equals the host's
    # mean increment norm there, and the blend-in cap then sets the size
    gdir = {L: [] for L in all_w}
    incp = {L: [] for L in all_w}
    for s in (tr if all_w else []):  # no write layers: no gradient targets to fit
        pos = mx.array(s["pos"])
        z0 = {L: mx.zeros((len(s["ids"]), D_h)) for L in all_w}
        with dref.reference_path():  # no VJP through the host's Metal kernels (host_grad_path in run.json)
            gr = mx.grad(lambda w, s=s, pos=pos: kl_rows(s["t_lp"], run(H, s["ids"], writes=w)[pos]).mean())(z0)
        st = {}
        run(H, s["ids"], cap=capl, st=st)
        for L in all_w:
            gL = -np.array(gr[L])[s["pos"]]
            gdir[L].append(gL / (np.linalg.norm(gL, axis=1, keepdims=True) + 1e-12))
            d_ = np.array(st["h"][L] - st["h"][L - 1]) if L > 0 else np.array(st["h"][L])
            incp[L].append(np.linalg.norm(d_[s["pos"]], axis=1))
    gdir = {L: np.concatenate(v) for L, v in gdir.items()}
    incp = {L: float(np.concatenate(v).mean()) for L, v in incp.items()}
    Xbp = np.concatenate([s["fb"][s["pos"]] for s in tr])
    Xrp = np.concatenate([s["fr"][s["pos"]] for s in tr])
    Xhp = np.concatenate([s["fh"][s["pos"]] for s in tr]) if use_host else None

    def cos_fit(X, M_, Y):
        P_ = (X - X.mean(0)) @ M_
        return float(np.mean(np.sum(P_ * Y, 1) / (np.linalg.norm(P_, axis=1) + 1e-12)))

    def ridge(X, Y):
        Xc, Yc = X - X.mean(0), Y - Y.mean(0)
        G = Xc.T @ Xc
        lam = args.ridge_lam * np.trace(G) / G.shape[0]
        return np.linalg.solve(G + lam * np.eye(G.shape[0]), Xc.T @ Yc).astype(np.float32)

    def r2(X, Y, M_):
        Xc, Yc = X - X.mean(0), Y - Y.mean(0)
        return float(1 - ((Yc - Xc @ M_) ** 2).sum() / (Yc ** 2).sum())

    maps = {}
    for mode, ws in write_sets.items():
        for L, cs in ws.items():
            Y = inc[L]
            sh = rng.permutation(len(Y))
            Mb, Ms, Mr = ridge(Xb[:, cs], Y), ridge(Xb[:, cs], Y[sh]), ridge(Xr[:, cs], Y)
            Yg = gdir[L]
            shg = rng.permutation(len(Yg))

            def scaled(X, M_, L=L):
                n_ = np.linalg.norm((X - X.mean(0)) @ M_, axis=1).mean()
                return (M_ * (incp[L] / max(n_, 1e-12))).astype(np.float32)
            Gb = scaled(Xbp[:, cs], ridge(Xbp[:, cs], Yg))
            Gs = scaled(Xbp[:, cs], ridge(Xbp[:, cs], Yg[shg]))
            Gr = scaled(Xrp[:, cs], ridge(Xrp[:, cs], Yg))
            Gh = scaled(Xhp, ridge(Xhp, Yg)) if use_host else None
            maps[(mode, L)] = {"M": mx.array(Mb), "Msh": mx.array(Ms), "Mr": mx.array(Mr),
                               "G": mx.array(Gb), "Gsh": mx.array(Gs), "Gr": mx.array(Gr),
                               "r2": r2(Xb[:, cs], Y, Mb), "r2_rand": r2(Xr[:, cs], Y, Mr),
                               "cos_g": cos_fit(Xbp[:, cs], Gb, Yg), "cos_g_rand": cos_fit(Xrp[:, cs], Gr, Yg),
                               "cos_g_host": cos_fit(Xhp, Gh, Yg) if use_host else None,
                               "np": Mb, "np_g": Gb, "np_gr": Gr, "np_gh": Gh}
    receipt["ridge_r2"] = {f"{m}@{L}": {"inc_box": round(v["r2"], 4), "inc_rand": round(v["r2_rand"], 4),
                                        "graddir_cos_box": round(v["cos_g"], 4),
                                        "graddir_cos_rand": round(v["cos_g_rand"], 4),
                                        "graddir_cos_host": (round(v["cos_g_host"], 4)
                                                             if v["cos_g_host"] is not None else None)}
                           for (m, L), v in maps.items()}
    print(f"{time.strftime('%H:%M:%S')} host references done; host val KL to donor {host_val_kl:.4f}; ridge R^2 "
          f"{receipt['ridge_r2']}", flush=True)
    save_receipt(stage="arms", box=box_info, host_val_kl=host_val_kl, donor_accuracy={k: (float(np.mean(v)) if v else None) for k, v in donor_acc.items()})

    # ------------------------------------------------------------------ H4: lens transport (closed form)
    LENS_KINDS = [k.strip() for k in args.lens_kinds.split(",") if k.strip()]
    if any(k not in ("tuned", "logit") for k in LENS_KINDS):
        sys.exit(f"unknown --lens-kinds {args.lens_kinds}")
    if LENS_KINDS and want_arms:
        wl_ = [a for a in want_arms if a.startswith(("lens_", "sink_"))]
        if not any("@" not in a for a in wl_):  # every lens arm named with its kind: fit only those kinds
            LENS_KINDS = [k for k in LENS_KINDS if any(a.endswith("@" + k) for a in wl_)]
    LENS, ROT, SINK = {}, {}, {}
    PATH = {}  # per lens kind: the joint sink system for per-token supported paths (sink_path arms)
    NEURON = {}  # the hand-made gate neuron (sink_neuron arms): read layer, read vector, alpha, writes per kind

    def neuron_gate(xn):
        """a(z) = silu(alpha z) z / silu(alpha) on z = r . x (x: RMS-normalised rows [T, D]); a(0) = 0, a(1) = 1"""
        z = xn @ NEURON["r"]
        al = NEURON["alpha"]
        silu_ = lambda v: v / (1.0 + np.exp(-v))  # noqa: E731
        return silu_(al * z) * z / silu_(al)

    NG_CACHE = {}

    def host_gate(ids, cache=None):
        """[T] gate from a CLEAN host pass at the input of the read layer (exact: no write sits below it)"""
        key_ = tuple(ids) if cache is None else None
        if key_ is not None and key_ in NG_CACHE:
            return NG_CACHE[key_]
        st = {}
        run(H, ids, cache=cache, st=st, cap=(NEURON["L0"] - 1,))
        h_ = np.array(st["h"][NEURON["L0"] - 1], dtype=np.float64)
        g = neuron_gate(h_ / np.sqrt((h_ * h_).mean(-1, keepdims=True) + 1e-12))
        if key_ is not None:
            NG_CACHE[key_] = g
        return g

    def neuron_writes(arm, g):
        """route neuron: the sink_const push at every sink layer times the gate (flat: its train-topic mean
        everywhere, the same overall size with no selectivity)"""
        T_ = len(g)
        gm = (mx.full((T_, 1), NEURON["flat"]) if arm["src"] == "flat"
              else mx.array(g.astype(np.float32))[:, None])
        return zero_first({L: gm * v for L, v in NEURON["W"][arm["kind"]].items()}, T_)

    HSA_CACHE = {}

    def host_sink_acts(ids, cache=None):
        """|activation| of every sink unit [T, sum k] (sink layers in order) from a CLEAN host pass: the paths the
        host's own stream opens at each token, before any write (whole texts cached: they depend only on the ids)"""
        key_ = tuple(ids) if cache is None else None
        if key_ is not None and key_ in HSA_CACHE:
            return HSA_CACHE[key_]
        st = {}
        run(H, ids, cache=cache, st=st, units={L: mx.array(SINK[L].astype(np.int32)) for L in sorted(SINK)})
        a_ = np.abs(np.concatenate([np.array(st["u"][L]) for L in sorted(SINK)], 1))
        if key_ is not None:
            HSA_CACHE[key_] = a_
        return a_

    def masked_solve(Q_, lam_, rhs, sets):
        """per token t: c_t restricted to the units sets[t] (all others 0), (Q_SS + lam I) c_S = rhs_t[S]; sets is
        [T, m] or a list of index arrays of any size (the cone cutoff gives each token its own size)"""
        c = np.zeros((rhs.shape[0], Q_.shape[0]))
        for t in range(rhs.shape[0]):
            S_ = np.asarray(sets[t])
            if len(S_):
                c[t, S_] = np.linalg.solve(Q_[np.ix_(S_, S_)] + lam_ * np.eye(len(S_)), rhs[t, S_])
        return c

    def layer_matched_random(sets, offs_, g_):
        """per token, a random set with the SAME number of units in each sink layer as sets[t] (a fair random control:
        a set drawn over all layers would land in layers the host path never touches)"""
        out_ = []
        for S_ in sets:
            S_ = np.asarray(S_)
            pick_ = []
            for i_ in range(len(offs_) - 1):
                n_ = int(((S_ >= offs_[i_]) & (S_ < offs_[i_ + 1])).sum())
                if n_:
                    pick_.append(offs_[i_] + g_.choice(offs_[i_ + 1] - offs_[i_], n_, replace=False))
            out_.append(np.concatenate(pick_) if pick_ else np.zeros(0, int))
        return out_

    def cone_sets(Q_, lam_, rhs, tau):
        """dynamic cutoff (Pierre 2026-11-18: cones with "a dynamic cutoff for unlikely paths"): per token, solve the
        full sink system, score each unit by its energy in the Fisher metric c_u^2 Q_uu (its share of what the host
        expresses of this donor signal), and keep the smallest set reaching tau of the total"""
        c_full = rhs @ np.linalg.inv(Q_ + lam_ * np.eye(Q_.shape[0])).T
        en = c_full ** 2 * np.diag(Q_)[None, :]
        out_ = []
        for t in range(rhs.shape[0]):
            o_ = np.argsort(-en[t])
            cum = np.cumsum(en[t][o_]) / max(en[t].sum(), 1e-30)
            out_.append(o_[:int(np.searchsorted(cum, tau) + 1)])
        return out_
    lens_skip = []

    def lens_unsupported(M):
        """The box write x = W_down[:, box] da is the residual change only when the MLP output is added to the
        residual as is, and the readout is linear only without a logit softcap. Gemma 4 has post-MLP norms, a
        layer scalar and a softcap (runtime/models/gemma4.py): refuse the lens arms there instead of mis-scaling."""
        lm_ = getattr(M["model"], "language_model", M["model"])
        why = []
        for nm in ("final_logit_softcapping", "logit_softcapping"):
            for obj in (lm_, getattr(lm_, "args", None), getattr(lm_, "config", None), getattr(M["model"], "args", None)):
                if obj is not None and getattr(obj, nm, None):
                    why.append(f"{nm}={getattr(obj, nm)}")
        L0 = M["layers"][0]
        for nm in ("post_feedforward_layernorm", "post_mlp_layernorm", "layer_scalar"):
            if nm in L0 or hasattr(L0, nm):
                why.append(f"layer has {nm}")
        return sorted(set(why))

    def readout(M):
        """W [V, D] (lm_head, or the tied embedding) and the final norm's effective gain g [D] = norm(ones)."""
        lm_ = getattr(M["model"], "language_model", M["model"])
        head = getattr(lm_, "lm_head", None)
        W = dense_w(head if head is not None else lm_.model.embed_tokens).astype(mx.float32)
        g_ = lm_.model.norm(mx.ones((1, W.shape[1]), dtype=mx.float32)).astype(mx.float32)[0]
        mx.eval(W, g_)
        return W, g_

    def capture(M, layers_):
        """layer outputs on the lens texts, position 0 dropped (the attention-sink token carries outlier norms)"""
        last = len(M["layers"]) - 1
        cap_ = sorted(set(layers_) | {last})
        got = {l: [] for l in cap_}
        for e in lens_txt:
            st = {}
            run(M, e["ids"], cap=cap_, st=st)
            for l in cap_:
                got[l].append(np.array(st["h"][l])[1:])
            del st
        return {l: np.concatenate(v) for l, v in got.items()}, last

    def readout_check(M, W, g_):
        last = len(M["layers"]) - 1
        e = lens_txt[0]
        st = {}
        lp_ref = run(M, e["ids"], cap=[last], st=st)
        h = st["h"][last]
        lg = ((h * mx.rsqrt((h * h).mean(-1, keepdims=True) + 1e-6)) * g_) @ W.T
        lp_ = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        # a bf16 model rounds its own logits (steps of ~0.1 near |logit| 30), so the worst case over the whole
        # vocabulary measures dtype, not the formula: score the reference's top-20 tokens and top-1 agreement
        top = mx.argsort(-lp_ref[1:], axis=-1)[:, :20]
        err = mx.abs(mx.take_along_axis(lp_[1:], top, -1) - mx.take_along_axis(lp_ref[1:], top, -1))
        agree = float((mx.argmax(lp_[1:], -1) == mx.argmax(lp_ref[1:], -1)).astype(mx.float32).mean().item())
        return {"top20_mean_abs": float(err.mean().item()), "max_abs_all": float(mx.abs(lp_ - lp_ref)[1:].max().item()),
                "top1_agree": agree}

    def lens_ridge(X, Y):
        Xc, Yc = X - X.mean(0), Y - Y.mean(0)
        G = Xc.T @ Xc
        lam = args.lens_ridge * np.trace(G) / G.shape[0]
        return np.linalg.solve(G + lam * np.eye(G.shape[0]), Xc.T @ Yc)  # row form: y = x @ A

    def r2_rows(X, Y, A):
        Xc, Yc = X - X.mean(0), Y - Y.mean(0)
        return float(1 - ((Yc - Xc @ A) ** 2).sum() / (Yc ** 2).sum())

    # a lens problem drops the lens arms with a recorded reason (the other arms still score); a run that names
    # lens or sink arms in --arms then fails at the arm check instead
    if LENS_KINDS:
        un_ = {"host": lens_unsupported(H), "donor": lens_unsupported(Dn)}
        if any(un_.values()):
            lens_skip.append(f"unsupported architecture for the linear readout / direct MLP write: {un_}")
        elif len(lens_txt) < 8:
            lens_skip.append(f"only {len(lens_txt)} lens texts")
    if LENS_KINDS and not lens_skip:
        W_h, g_h = readout(H)
        W_d, g_d = readout(Dn)
        V_ = min(W_h.shape[0], W_d.shape[0])
        chk = {"host": readout_check(H, W_h, g_h), "donor": readout_check(Dn, W_d, g_d)}
        if any(c["top20_mean_abs"] > args.readout_tol or c["top1_agree"] < 0.95 for c in chk.values()):
            lens_skip.append(f"readout check {chk} outside --readout-tol {args.readout_tol} / top-1 0.95")
            del W_h, W_d
            mx.clear_cache()
    if lens_skip:
        receipt["lens"] = {"skipped": lens_skip}
        print(f"lens arms skipped: {lens_skip}", flush=True)
        LENS_KINDS = []
        save_receipt()
    if LENS_KINDS:
        W_h, W_d = W_h[:V_], W_d[:V_]

        def gram(A_, B_):  # A' C B, C = centring over the shared vocabulary (log-softmax ignores a common shift);
            # centre the columns first rather than subtracting outer(sum, sum) / V (no cancellation in float32)
            return np.array((A_ - A_.mean(0)).T @ (B_ - B_.mean(0))).astype(np.float64)
        if args.lens_metric == "flat":
            G_hh, G_hd, G_dd = gram(W_h, W_h), gram(W_h, W_d), gram(W_d, W_d)
        else:
            # average Fisher of the donor's next-token distribution on the lens texts (task-agnostic):
            # F = E_t[diag(p_t) - p_t p_t'], so G_xy = W_x' diag(pbar) W_y - E_t[(W_x' p_t)(W_y' p_t)'];
            # the second-order change of KL(donor || host) is x' K' F K x, so matching in this metric weighs
            # tokens by how much they count in KL, not by the bulk of rare-token logits (the flat metric's flaw)
            pbar = mx.zeros((V_,), dtype=mx.float32)
            S = {k: mx.zeros((W_h.shape[1] if k[0] == "h" else W_d.shape[1],
                              W_h.shape[1] if k[1] == "h" else W_d.shape[1]), dtype=mx.float32)
                 for k in ("hh", "hd", "dd")}
            n_ = 0
            for e in lens_txt:
                p = mx.exp(run(Dn, e["ids"])[1:, :V_])
                ph, pd_ = p @ W_h, p @ W_d
                pbar = pbar + p.sum(0)
                S["hh"] = S["hh"] + ph.T @ ph
                S["hd"] = S["hd"] + ph.T @ pd_
                S["dd"] = S["dd"] + pd_.T @ pd_
                mx.eval(pbar, *S.values())
                n_ += p.shape[0]
            pbar = pbar / n_

            def fgram(A_, B_, k):
                return (np.array((A_ * pbar[:, None]).T @ B_).astype(np.float64)
                        - np.array(S[k]).astype(np.float64) / n_)
            G_hh, G_hd, G_dd = fgram(W_h, W_h, "hh"), fgram(W_h, W_d, "hd"), fgram(W_d, W_d, "dd")
            del S, pbar
        receipt["lens_metric"] = args.lens_metric
        gh, gd = np.array(g_h), np.array(g_d)
        Wc_h, Wc_d = W_h - W_h.mean(0), W_d - W_d.mean(0)  # centred readouts, kept for the dark-space read
        mx.eval(Wc_h, Wc_d)
        del W_h, W_d
        mx.clear_cache()
        # host READOUT ceiling: at the host's final layer (lens = identity) the best any host residual change can do;
        # whatever of the donor's logit change lies outside the host unembedding's row space is dark at every depth
        Khh_c = gh[:, None] * G_hh * gh[None, :]
        T_c_inv = np.linalg.inv(Khh_c + 1e-6 * np.trace(Khh_c) / Khh_c.shape[0] * np.eye(Khh_c.shape[0]))
        rng_dk = np.random.default_rng(args.seed + 41)
        hl = sorted(set(warp.values()))
        # ---- host topic SINKS (Pierre 2026-11-18: hook where the host's own late layers already turn topic context
        # into output): late host units that pass the box's topic-contrast rule AND whose write, read through the
        # host's own final gain and unembedding, promotes the topic's next tokens (token signature). Host-only
        # selection, like the stub (it reads the train continuations' next tokens, as the stub reads their positions)
        SINK, SINKR, sink_info = {}, {}, {}
        if args.sink_units > 0:
            lo_ = max(1, int(round(args.sink_from * (LH - 1))))
            gold_set = mx.array(np.unique(np.concatenate(
                [np.array(q["ids"])[np.array(q["pos"]) + 1] for q in tr])).astype(np.int32))
            ghm = mx.array(gh.astype(np.float32))
            per_l, lsc = {}, {}
            rng_sk = np.random.default_rng(args.seed + 53)  # own stream: sinks never shift the main rng
            for l in range(lo_, LH):
                Wl = dense_w(H["layers"][l].mlp.down_proj).astype(mx.float32)  # [D_h, I]
                sig = []
                for u0 in range(0, Wl.shape[1], 512):
                    lg = (Wl[:, u0:u0 + 512].T * ghm) @ Wc_h.T  # [chunk, V]: each unit's logit signature
                    z = (lg - lg.mean(-1, keepdims=True)) / (mx.sqrt(lg.var(-1, keepdims=True)) + 1e-12)
                    sig.append(np.array(z[:, gold_set].mean(-1)))
                    del lg, z
                sig = np.concatenate(sig)
                mt, me = mh_t[l], mh_e[l]
                score = np.where((mt >= args.min_ratio * me) & (sig > 0), (mt - me) * sig, -np.inf)
                top = np.argsort(-score)[:args.sink_units]
                top = top[np.isfinite(score[top])]
                per_l[l] = (top, sig)
                lsc[l] = float(score[top].sum()) if len(top) else -np.inf
            SL = sorted([l for l in sorted(lsc, key=lambda l: -lsc[l]) if np.isfinite(lsc[l])][:args.n_sink_layers])
            for l in SL:
                SINK[l] = per_l[l][0]
                free = np.setdiff1d(np.arange(len(per_l[l][1])), SINK[l])
                SINKR[l] = rng_sk.choice(free, len(SINK[l]), replace=False)
            sink_info = {"layers": SL, "from_layer": lo_, "units": {str(l): SINK[l].tolist() for l in SL},
                         "token_signature_mean_z": {str(l): float(per_l[l][1][SINK[l]].mean()) for l in SL},
                         "host_random_token_signature_mean_z": {str(l): float(per_l[l][1][SINKR[l]].mean()) for l in SL},
                         "gold_tokens": int(gold_set.size)}
            receipt["sinks"] = dict(sink_info)
            if not SL:
                receipt["sinks"]["note"] = "no host layer passed the sink rule: sink arms dropped"
            print(f"host sinks: layers {SL}, units {[len(SINK[l]) for l in SL]}, signature z "
                  f"{sink_info['token_signature_mean_z']} (random host units {sink_info['host_random_token_signature_mean_z']})",
                  flush=True)
        cap_d, last_d = capture(Dn, box_layers)
        cap_h, last_h = capture(H, sorted(set(hl) | set(SINK)))
        r_d = float(np.sqrt((cap_d[last_d] ** 2).mean(1)).mean())
        r_h = float(np.sqrt((cap_h[last_h] ** 2).mean(1)).mean())
        n_tok = len(cap_d[last_d])
        fit_ix, held_ix = np.arange(n_tok) < int(0.8 * n_tok), np.arange(n_tok) >= int(0.8 * n_tok)
        # English mean unit activation, position 0 (the sink token) excluded as in the lens fit
        mu_eng_raw = np.concatenate([e["fb"][1:] * sd_b + mu_b for e in retain]).mean(0)
        mu_eng_raw_r = np.concatenate([e["fr"][1:] * sd_r + mu_r for e in retain]).mean(0)
        Wbox = {d: np.array(dense_w(Dn["layers"][d].mlp.down_proj).astype(mx.float32)[:, mx.array(box[d].astype(np.int32))])
                for d in box_layers}
        Wrnd = {d: np.array(dense_w(Dn["layers"][d].mlp.down_proj).astype(mx.float32)[:, mx.array(rand[d].astype(np.int32))])
                for d in box_layers}
        # held-text box / random writes at each donor layer (centred on the English mean), for the fidelity check
        diag = {}
        ADS = {}
        # diagnostics read on the VALIDATION texts (neither the sinks nor the box were chosen on them), position 0
        # (the attention-sink token) excluded; one fixed sample of continuation positions for every arm and layer
        dset = va if va else tr

        def rows_of(key_, sd_, m_, cs, mu_, cont=False):
            return np.concatenate([(q[key_][q["pos"]] if cont else q[key_][1:])[:, cs] * sd_[cs] + m_[cs]
                                   for q in dset]) - mu_
        gold_dk = np.concatenate([np.array(q["ids"])[np.array(q["pos"]) + 1] for q in dset])
        sel_dk = rng_dk.permutation(len(gold_dk))[:args.dark_positions]
        gi_dk = mx.array(gold_dk[sel_dk].astype(np.int32))[:, None]

        def qf(X, K, Y):  # row-wise x' K y as BLAS (a three-operand einsum is a plain C loop: hours at D 5120)
            return ((X @ K) * Y).sum(1)
        for kind in LENS_KINDS:
            for d in box_layers:
                L = warp[d]
                if kind == "tuned":
                    A_d = lens_ridge(cap_d[d][fit_ix], cap_d[last_d][fit_ix])
                    A_h = lens_ridge(cap_h[L][fit_ix], cap_h[last_h][fit_ix])
                    r2d = r2_rows(cap_d[d][held_ix], cap_d[last_d][held_ix], A_d)
                    r2h = r2_rows(cap_h[L][held_ix], cap_h[last_h][held_ix], A_h)
                else:
                    A_d, A_h = np.eye(cap_d[d].shape[1]), np.eye(cap_h[L].shape[1])
                    r2d = r2_rows(cap_d[d][held_ix], cap_d[last_d][held_ix], A_d)
                    r2h = r2_rows(cap_h[L][held_ix], cap_h[last_h][held_ix], A_h)
                # column form: K_D = C W_D diag(g_D) A_d', so K_H'K_D = A_h diag(g_h) G_hd diag(g_d) A_d'
                Khh = A_h @ (gh[:, None] * G_hh * gh[None, :]) @ A_h.T
                Khd = A_h @ (gh[:, None] * G_hd * gd[None, :]) @ A_d.T
                Kdd = A_d @ (gd[:, None] * G_dd * gd[None, :]) @ A_d.T
                lam = args.lens_lam * np.trace(Khh) / Khh.shape[0]
                T = np.linalg.solve(Khh + lam * np.eye(Khh.shape[0]), Khd)  # [D_h, D_d]: y = T x
                s_ = (r_h / r_d) * args.lens_alpha
                cs = colsd[d]
                ADS[(kind, d)] = A_d
                LENS[(kind, d)] = {"P_box": mx.array((s_ * Wbox[d].T @ T.T).astype(np.float32)),
                                   "P_rand": mx.array((s_ * Wrnd[d].T @ T.T).astype(np.float32)),
                                   "m_box": mu_eng_raw[cs].astype(np.float32),
                                   "m_rand": mu_eng_raw_r[cs].astype(np.float32)}

                def fid(Wu, mu_, src):
                    # measured on the validation texts (unseen by the lens fit, the box choice and the sink choice;
                    # the transport itself uses no task text): x = W (a - a_English_mean), position 0 excluded
                    key_ = "fb" if src == "box" else "fr"
                    sd_, m_ = (sd_b, mu_b) if src == "box" else (sd_r, mu_r)
                    x = rows_of(key_, sd_, m_, cs, mu_) @ Wu.T
                    y = x @ T.T
                    kdx = qf(x, Kdd, x)
                    res = qf(y, Khh, y) - 2 * qf(y, Khd, x) + kdx
                    # readout ceiling for this donor layer: y_c = argmin over host FINAL-layer changes
                    Khd_c = (gh[:, None] * G_hd * gd[None, :]) @ A_d.T
                    yc_ = x @ (T_c_inv @ Khd_c).T
                    res_c = qf(yc_, Khh_c, yc_) - 2 * qf(yc_, Khd_c, x) + kdx
                    # dark-space read on continuation positions: the donor's logit change K_D x, the part the host
                    # expresses at warp(d) (K_H T x), and the dark rest; where does the NEXT TOKEN sit in each
                    xs = rows_of(key_, sd_, m_, cs, mu_, cont=True)[sel_dk] @ Wu.T
                    sel = sel_dk
                    lg_d = (mx.array(((xs @ A_d) * gd).astype(np.float32)) @ Wc_d.T)
                    lg_h = (mx.array((((xs @ T.T) @ A_h) * gh).astype(np.float32)) @ Wc_h.T)
                    lg_c = (mx.array((((xs @ (T_c_inv @ Khd_c).T)) * gh).astype(np.float32)) @ Wc_h.T)
                    gi = gi_dk

                    def gold_read(v):
                        z = (v - v.mean(-1, keepdims=True)) / (mx.sqrt(v.var(-1, keepdims=True)) + 1e-12)
                        zg = mx.take_along_axis(z, gi, -1)[:, 0]
                        rank = (v > mx.take_along_axis(v, gi, -1)).sum(-1)
                        return {"gold_z": round(float(zg.mean().item()), 3),
                                "gold_top10": round(float((rank < 10).astype(mx.float32).mean().item()), 3)}

                    def top_tokens(v, k=15):
                        ids_, cnt = np.unique(np.array(mx.argsort(-v, axis=-1)[:, :5]).ravel(), return_counts=True)
                        o_ = np.argsort(-cnt)[:k]
                        return [[tok.decode([int(ids_[i])]), int(cnt[i])] for i in o_]
                    dark, dark_c = lg_d - lg_h, lg_d - lg_c
                    out_ = {"fidelity": float(1 - res.sum() / max(kdx.sum(), 1e-30)),
                            "fidelity_readout_ceiling": float(1 - res_c.sum() / max(kdx.sum(), 1e-30)),
                            "logit_effect": float(np.sqrt(np.maximum(kdx, 0)).mean() / r_d),
                            "next_token": {"donor_effect": gold_read(lg_d), "expressed_at_warp": gold_read(lg_h),
                                           "dark_at_warp": gold_read(dark), "dark_at_readout": gold_read(dark_c)},
                            "dark_top_tokens": top_tokens(dark), "expressed_top_tokens": top_tokens(lg_h),
                            "positions": int(len(sel))}
                    del lg_d, lg_h, lg_c, dark, dark_c
                    return out_
                diag[f"{kind}@{d}->{L}"] = {"lens_r2_donor": round(r2d, 4), "lens_r2_host": round(r2h, 4),
                                            "box": fid(Wbox[d], mu_eng_raw[cs], "box"),
                                            "rand": fid(Wrnd[d], mu_eng_raw_r[cs], "rand")}
        # ---- sink transport: c = argmin |sum_i K_H(L_i) B_i c_i - sum_d K_D(d) x_d|_F^2 + lam |c|^2 jointly over the
        # sink units' activation changes c (B_i = W_down[:, sinks at L_i]); host write at L_i = B_i c_i
        sink_diag = {}
        KDX = {}
        if SINK:
            SL = sorted(SINK)
            Hg = gh[:, None] * G_hh * gh[None, :]
            Hd = gh[:, None] * G_hd * gd[None, :]
            Dg = gd[:, None] * G_dd * gd[None, :]
            Wdh = {l: np.array(dense_w(H["layers"][l].mlp.down_proj).astype(mx.float32)) for l in SL}
            src_par = {"box": ("fb", sd_b, mu_b, mu_eng_raw), "rand": ("fr", sd_r, mu_r, mu_eng_raw_r)}
            a_all = {(sr, d): rows_of(k_, sd_, m_, colsd[d], me_[colsd[d]])
                     for sr, (k_, sd_, m_, me_) in src_par.items() for d in box_layers}
            a_cnt = {(sr, d): rows_of(k_, sd_, m_, colsd[d], me_[colsd[d]], cont=True)[sel_dk]
                     for sr, (k_, sd_, m_, me_) in src_par.items() for d in box_layers}
            gi_ = gi_dk
            # the host's own sink activity at the same sampled validation continuation positions (clean passes)
            ha_cnt = np.concatenate([host_sink_acts(q["ids"])[q["pos"]] for q in dset])[sel_dk]
            rng_path_d = np.random.default_rng(args.seed + 67)
            # the constant-push control: the box's mean centred activation over the TRAIN continuation positions
            SINK_CONST_A = {d: (np.concatenate([q["fb"][q["pos"]][:, colsd[d]] for q in tr]) * sd_b[colsd[d]]
                                + mu_b[colsd[d]] - mu_eng_raw[colsd[d]]).mean(0).astype(np.float32) for d in box_layers}

            def zgold(v):
                z = (v - v.mean(-1, keepdims=True)) / (mx.sqrt(v.var(-1, keepdims=True)) + 1e-12)
                rank = (v > mx.take_along_axis(v, gi_, -1)).sum(-1)
                return {"gold_z": round(float(mx.take_along_axis(z, gi_, -1).mean().item()), 3),
                        "gold_top10": round(float((rank < 10).astype(mx.float32).mean().item()), 3)}
            for kind in LENS_KINDS:
                Ah = {L: (lens_ridge(cap_h[L][fit_ix], cap_h[last_h][fit_ix]) if kind == "tuned"
                          else np.eye(D_h)) for L in SL}
                s_ = (r_h / r_d) * args.lens_alpha
                for which, U_ in (("sink", SINK), ("hrand", SINKR)):
                    Bs = {L: Wdh[L][:, U_[L]] for L in SL}
                    E = np.concatenate([Ah[L].T @ Bs[L] for L in SL], 1)  # [D_h, sum k]: final-residual change per c
                    offs = np.cumsum([0] + [Bs[L].shape[1] for L in SL])
                    Q = E.T @ Hg @ E
                    Qi = np.linalg.inv(Q + args.lens_lam * np.trace(Q) / Q.shape[0] * np.eye(Q.shape[0]))
                    Md = {d: Qi @ (E.T @ Hd @ ADS[(kind, d)].T) for d in box_layers}  # [sum k, D_d]
                    if which == "sink":  # keep the joint system for the per-token supported-path solve
                        Rd = {d: E.T @ Hd @ ADS[(kind, d)].T for d in box_layers}  # [sum k, D_d]
                        PATH[kind] = {"Q": Q, "lam": args.lens_lam * np.trace(Q) / Q.shape[0], "s": s_,
                                      "L": SL, "offs": offs, "B": {L: Bs[L] for L in SL}, "E": E,
                                      "RB": {d: Wbox[d].T @ Rd[d].T for d in box_layers}}  # [k_box, sum k]
                    for src_, Wsrc in (("box", Wbox), ("rand", Wrnd)):
                        if which == "hrand" and src_ == "rand":
                            continue
                        if (kind, src_) not in KDX:
                            Xa = {d: a_all[(src_, d)] @ Wsrc[d].T for d in box_layers}
                            KDX[(kind, src_)] = (Xa, sum(qf(Xa[d], ADS[(kind, d)] @ Dg @ ADS[(kind, d2)].T, Xa[d2])
                                                         for d in box_layers for d2 in box_layers))
                        for d in box_layers:
                            for i_, L in enumerate(SL):
                                blk = Md[d][offs[i_]:offs[i_ + 1]]
                                LENS[("sink", kind, which, src_, d, L)] = mx.array(
                                    (s_ * Wsrc[d].T @ blk.T @ Bs[L].T).astype(np.float32))
                        # fidelity over every validation position (position 0 excluded), and where the next token
                        # sits in what the sinks express, on the fixed sample of validation continuation positions
                        X, kdx = KDX[(kind, src_)]
                        c_ = sum(X[d] @ Md[d].T for d in box_layers)
                        cross = sum(qf(c_, E.T @ Hd @ ADS[(kind, d)].T, X[d]) for d in box_layers)
                        res = qf(c_, Q, c_) - 2 * cross + kdx
                        Xs = {d: a_cnt[(src_, d)] @ Wsrc[d].T for d in box_layers}
                        cs_ = sum(Xs[d] @ Md[d].T for d in box_layers)
                        fin_d = sum(((Xs[d] @ ADS[(kind, d)]) * gd) for d in box_layers)
                        fin_h = (cs_ @ E.T) * gh
                        lgd = mx.array(fin_d.astype(np.float32)) @ Wc_d.T
                        lgh = mx.array(fin_h.astype(np.float32)) @ Wc_h.T
                        sink_diag[f"{kind}/{which}/{src_}"] = {
                            "fidelity": float(1 - res.sum() / max(kdx.sum(), 1e-30)),
                            "next_token": {"donor_effect": zgold(lgd), "expressed_by_sinks": zgold(lgh),
                                           "dark": zgold(lgd - lgh)}}
                        del lgd, lgh
                # supported paths (Pierre 2026-11-18: "multiple supported paths like real token streams would
                # have"): on the same validation sample, the host's own open sinks per token, the path solve's
                # fidelity and next-token read against the single graph, and path diversity
                P_ = PATH[kind]
                m_ = min(args.path_units, P_["Q"].shape[0])
                Xs = {d: a_cnt[("box", d)] @ Wbox[d].T for d in box_layers}
                kdx_s = sum(qf(Xs[d], ADS[(kind, d)] @ Dg @ ADS[(kind, d2)].T, Xs[d2])
                            for d in box_layers for d2 in box_layers)
                rhs = sum(a_cnt[("box", d)] @ P_["RB"][d] for d in box_layers)
                fin_d = sum(((Xs[d] @ ADS[(kind, d)]) * gd) for d in box_layers)
                lgd = mx.array(fin_d.astype(np.float32)) @ Wc_d.T
                Qi_ = np.linalg.inv(P_["Q"] + P_["lam"] * np.eye(P_["Q"].shape[0]))
                for nm_, sets in (("single_graph", None), ("host_paths", np.argsort(-ha_cnt, 1)[:, :m_]),
                                  ("idle_paths", np.argsort(ha_cnt, 1)[:, :m_]),
                                  ("random_paths", layer_matched_random(np.argsort(-ha_cnt, 1)[:, :m_], P_["offs"],
                                                                        rng_path_d))):
                    c_ = rhs @ Qi_.T if sets is None else masked_solve(P_["Q"], P_["lam"], rhs, sets)
                    res = qf(c_, P_["Q"], c_) - 2 * (c_ * rhs).sum(1) + kdx_s
                    lgh = mx.array(((c_ @ P_["E"].T) * gh).astype(np.float32)) @ Wc_h.T
                    sink_diag[f"{kind}/path/{nm_}"] = {
                        "fidelity_sample": float(1 - res.sum() / max(kdx_s.sum(), 1e-30)),
                        "next_token": {"donor_effect": zgold(lgd), "expressed": zgold(lgh), "dark": zgold(lgd - lgh)}}
                    del lgh
                del lgd
            hp = np.argsort(-ha_cnt, 1)[:, :min(args.path_units, ha_cnt.shape[1])]
            sets_ = [frozenset(r_.tolist()) for r_ in hp]
            pairs_ = [rng_path_d.choice(len(sets_), 2, replace=False) for _ in range(500)] if len(sets_) > 1 else []
            jac = [len(sets_[i_] & sets_[j_]) / len(sets_[i_] | sets_[j_]) for i_, j_ in pairs_]
            offs_ = np.cumsum([0] + [len(SINK[L]) for L in SL])
            sink_info["paths"] = {
                "units_per_token": int(hp.shape[1]), "positions": len(sets_),
                "distinct_paths": len(set(sets_)), "mean_pairwise_jaccard": float(np.mean(jac)) if jac else None,
                "layer_share": {str(L): float(((hp >= offs_[i]) & (hp < offs_[i + 1])).mean())
                                for i, L in enumerate(SL)},
                "open_mass_in_path": float(np.mean(np.take_along_axis(ha_cnt, hp, 1).sum(1) /
                                                   (ha_cnt.sum(1) + 1e-12)))}
            # ---- hand-made neuron (vgel.me "making neurons by hand", 2026-11-20, adapted): sink_const gated by one
            # host-only read. Read r from difference of means on the RMS-normalised input of the first sink layer
            # L0 (topic = train continuation positions, off = English retain), orthogonal to the off mean and
            # scaled so z = r.x is 0 at the off mean and 1 at the topic mean; gate a(z) = silu(alpha z) z /
            # silu(alpha) (a(1) = 1; alpha = 1 is the post's silu(z) z with its negative-side bump, larger alpha
            # shrinks the bump towards relu(z) z). Every write sits at layers >= L0, so a clean host pass gives
            # the exact gate input at run time. Appended, never overwriting a host unit (carve is a partition).
            L0_ = SL[0]
            want_nr_ = (not want_arms) or any(a.split("@")[0] in ("sink_neuron", "sink_neuronflat") for a in want_arms)
            nr_skip = (None if want_nr_ and retain and eng_val and tr and L0_ >= 1 else
                       "not requested" if not want_nr_ else
                       "first sink layer is 0 (no clean input to read)" if L0_ < 1 else "no train/retain/eng_val texts")

            def xnorm_(ids):
                st_ = {}
                run(H, ids, st=st_, cap=(L0_ - 1,))
                h_ = np.array(st_["h"][L0_ - 1], dtype=np.float64)
                return h_ / np.sqrt((h_ * h_).mean(-1, keepdims=True) + 1e-12)
            if nr_skip is not None:
                sink_info["neuron"] = {"skipped": nr_skip}
            else:
                xtr_ = [xnorm_(q["ids"]) for q in tr]
                on_ = np.concatenate([x_[q["pos"]] for x_, q in zip(xtr_, tr)]).mean(0)
                off_ = np.concatenate([xnorm_(e["ids"])[1:] for e in retain]).mean(0)
                d_ = on_ - off_
                d_ = d_ - off_ * (d_ @ off_) / (off_ @ off_)
                r_ = d_ / (d_ @ on_)
                NEURON.update({"L0": L0_, "r": r_, "alpha": args.neuron_alpha,
                               "W": {k_: {L: sum(mx.array(SINK_CONST_A[d])[None] @ LENS[("sink", k_, "sink", "box", d, L)]
                                                 for d in box_layers) for L in SL} for k_ in LENS_KINDS}})
                g_tr = np.concatenate([neuron_gate(x_)[q["pos"]] for x_, q in zip(xtr_, tr)])
                del xtr_
                NEURON["flat"] = float(g_tr.mean())

                def gstats(g):
                    return {"mean": round(float(g.mean()), 4), "median": round(float(np.median(g)), 4),
                            "frac_gt_0.5": round(float((g > 0.5).mean()), 4), "min": round(float(g.min()), 4),
                            "max": round(float(g.max()), 4), "n": int(g.size)}
                g_va = np.concatenate([neuron_gate(xnorm_(q["ids"]))[q["pos"]] for q in (va or tr)])
                g_en = np.concatenate([neuron_gate(xnorm_(e["ids"]))[1:] for e in eng_val])
                sink_info["neuron"] = {
                    "read_layer": L0_, "write_layers": SL, "alpha": args.neuron_alpha, "flat_gate": NEURON["flat"],
                    "read_norm": float(np.linalg.norm(r_)), "z_off_mean": float(off_ @ r_), "z_on_mean": float(on_ @ r_),
                    "write_norm": {k_: {str(L): float(mx.sqrt((v * v).sum()).item()) for L, v in NEURON["W"][k_].items()}
                                   for k_ in LENS_KINDS},
                    "gate_train_topic": gstats(g_tr), "gate_val_topic": gstats(g_va), "gate_eng_val": gstats(g_en),
                    "params": int(r_.size + len(SL) * D_h)}  # one read vector, one write column per sink layer
                print(f"neuron: read L{L0_}, gate val topic {sink_info['neuron']['gate_val_topic']}, English "
                      f"{sink_info['neuron']['gate_eng_val']}", flush=True)
            receipt["sinks"] = sink_info | {"transport": sink_diag, "diagnostic_texts": "validation"}
            print(f"{time.strftime('%H:%M:%S')} sink transport ready: {json.dumps(sink_diag)}", flush=True)
            del Wdh, Hg, Hd, Dg, a_all, a_cnt, KDX, ha_cnt
        for L in hl:
            q_, _ = np.linalg.qr(np.random.default_rng(args.seed + 31 + L).standard_normal((D_h, D_h)))
            ROT[L] = mx.array(q_.astype(np.float32))
        del cap_d, cap_h, G_hh, G_hd, G_dd, Wc_h, Wc_d, ADS, Wbox, Wrnd, Khh_c, T_c_inv
        mx.clear_cache()
        receipt["lens"] = {"texts": len(lens_txt), "tokens": n_tok, "readout_check_max_abs_logprob": chk,
                           "r_host": r_h, "r_donor": r_d, "per_layer": diag,
                           "fit": "single-model ridge lenses on generic English; no paired text, no task text"}
        print(f"{time.strftime('%H:%M:%S')} lens transport ready: readout check {chk}; "
              f"{json.dumps(diag)}", flush=True)
        save_receipt()

    rng_path = np.random.default_rng(args.seed + 71)
    PATH_SIZES = []  # units per token used by cone arms at validation continuation positions (reset per arm)
    REC = [None]  # set by val_scores to the scored positions while it scores a validation text

    def path_writes(arm, zb_, ha_):
        """route path: per token, the donor box through ONLY the sinks the host's own clean stream has open there
        (top --path-units by |activation| across all sink layers; mask=rand: a random set of the same size,
        rescaled per token per layer to the host-path write)"""
        P_ = PATH[arm["kind"]]
        T_ = zb_.shape[0]
        kn = P_["Q"].shape[0]
        m_ = min(args.path_units, kn)
        rhs = sum((zb_[:, colsd[d]] * sd_b[colsd[d]] + mu_b[colsd[d]] - mu_eng_raw[colsd[d]]).astype(np.float64)
                  @ P_["RB"][d] for d in box_layers)

        def writes_for(sets):
            c = masked_solve(P_["Q"], P_["lam"], rhs, sets)
            return {L: mx.array((P_["s"] * c[:, P_["offs"][i]:P_["offs"][i + 1]] @ P_["B"][L].T).astype(np.float32))
                    for i, L in enumerate(P_["L"])}
        if arm.get("mask") == "idle":  # the topic sinks the host is NOT using at this token (spare capacity)
            return zero_first(writes_for(np.argsort(ha_, 1)[:, :m_]), T_)
        if arm.get("mask") in ("cone", "conerand"):
            sets = cone_sets(P_["Q"], P_["lam"], rhs, arm["tau"])
            if REC[0] is not None:  # sizes only at the scored validation continuation positions
                PATH_SIZES.extend(len(sets[t]) for t in REC[0])
            w = writes_for(sets)
            if arm["mask"] == "conerand":  # same per-token, per-layer counts, random units, same per-layer norm
                wr = writes_for(layer_matched_random(sets, P_["offs"], rng_path))
                w = {L: wr[L] * (mx.sqrt((w[L] * w[L]).sum(-1, keepdims=True)) /
                                 (mx.sqrt((wr[L] * wr[L]).sum(-1, keepdims=True)) + 1e-12)) for L in w}
            return zero_first(w, T_)
        hsets = np.argsort(-ha_, 1)[:, :m_]
        w = writes_for(hsets)
        if arm.get("mask") == "rand":
            wr = writes_for(layer_matched_random(hsets, P_["offs"], rng_path))
            w = {L: wr[L] * (mx.sqrt((w[L] * w[L]).sum(-1, keepdims=True)) /
                             (mx.sqrt((wr[L] * wr[L]).sum(-1, keepdims=True)) + 1e-12)) for L in w}
        return zero_first(w, T_)

    def sink_writes(arm, zb_, zr_):
        """route sink: the donor box (or random donor units) through the joint sink transport into the host's own
        topic sinks (which=sink) or random host units of the same late layers (which=hrand)"""
        k_ = arm["kind"]
        T_ = zb_.shape[0]
        w = {}
        for L in sorted(SINK):
            real, alt = None, None
            for d in box_layers:
                cs = colsd[d]
                a_ = mx.array((zb_[:, cs] * sd_b[cs] + mu_b[cs] - mu_eng_raw[cs]).astype(np.float32))
                v = a_ @ LENS[("sink", k_, "sink", "box", d, L)]  # (shuf: zb_ already holds another item's features)
                real = v if real is None else real + v
                if arm["src"] == "rand":
                    ar = mx.array((zr_[:, cs] * sd_r[cs] + mu_r[cs] - mu_eng_raw_r[cs]).astype(np.float32))
                    v2 = ar @ LENS[("sink", k_, "sink", "rand", d, L)]
                elif arm["src"] == "hrand":
                    v2 = a_ @ LENS[("sink", k_, "hrand", "box", d, L)]
                elif arm["src"] == "const":
                    v2 = mx.array(SINK_CONST_A[d])[None] @ LENS[("sink", k_, "sink", "box", d, L)]
                else:
                    v2 = None
                if v2 is not None:
                    alt = v2 if alt is None else alt + v2
            if arm["src"] == "const":
                real = mx.broadcast_to(alt, real.shape)  # the same push at every position: no per-item input
            elif alt is not None:  # rand / hrand: per-token rescale to the box write (equal write size)
                real = alt * (mx.sqrt((real * real).sum(-1, keepdims=True)) /
                              (mx.sqrt((alt * alt).sum(-1, keepdims=True)) + 1e-12))
            w[L] = real
        return zero_first(w, T_)

    def zero_first(w, T_):
        """no write at position 0, the attention-sink token (its huge increment would let the cap admit a large
        write); a multi-token chunk in this harness always starts at position 0 (whole texts, generation prompts)"""
        if T_ <= 1:
            return w
        m_ = mx.concatenate([mx.zeros((1, 1)), mx.ones((T_ - 1, 1))])
        return {L: v * m_ for L, v in w.items()}

    def lens_writes(arm, zb_, zr_):
        w = {}
        for d in box_layers:
            L, cs, lz = warp[d], colsd[d], LENS[(arm["kind"], d)]
            real = mx.array((zb_[:, cs] * sd_b[cs] + mu_b[cs] - lz["m_box"]).astype(np.float32)) @ lz["P_box"]
            if arm["src"] in ("box", "shuf"):  # shuf: zb_ already holds the wrong item's features
                w[L] = real
            elif arm["src"] == "rot":
                w[L] = real @ ROT[L]
            else:
                v = mx.array((zr_[:, cs] * sd_r[cs] + mu_r[cs] - lz["m_rand"]).astype(np.float32)) @ lz["P_rand"]
                w[L] = v * (mx.sqrt((real * real).sum(-1, keepdims=True)) /
                            (mx.sqrt((v * v).sum(-1, keepdims=True)) + 1e-12))
        return zero_first(w, zb_.shape[0])

    # ------------------------------------------------------------------ trainable modules
    D = int(dense_w(H["layers"][0].mlp.down_proj).shape[0])

    def rmsn(x):
        return x * mx.rsqrt((x * x).mean(-1, keepdims=True) + 1e-6)

    if args.tile_layers == "all":
        tile_layers = list(range(LH))
    elif args.tile_layers == "stub":
        tile_layers = list(stub_layers)
    else:
        tile_layers = [int(x) for x in args.tile_layers.split(",")]

    GREC = {"on": False, "g": []}  # gate recorder (outside the module, so it is never a parameter)

    class Tiles(nn.Module):
        """kind mlp: gated SiLU tile neurons; kind lora: gated low-rank LINEAR adapter (rank --lora-rank) on the
        residual, the audit's gated-LoRA arm. Both: w_out = 0 at init (exactly the host), stop-gradient gate input."""

        def __init__(self, seed, kind="mlp"):
            super().__init__()
            g_ = np.random.default_rng(seed)
            self.layers_set = set(tile_layers)
            self.kind = kind
            n = args.tile_units if kind == "mlp" else args.lora_rank
            self.w_in = {str(l): mx.array((g_.standard_normal((D, n)) / np.sqrt(D)).astype(np.float32)) for l in tile_layers}
            self.w_out = {str(l): mx.zeros((n, D)) for l in tile_layers}  # untrained == host
            self.gate_w = {str(l): mx.zeros((D,)) for l in tile_layers}
            self.gate_b = {str(l): mx.zeros((1,)) for l in tile_layers}

        def __call__(self, l, x):
            k = str(l)
            xn = rmsn(x.astype(mx.float32))
            gt = mx.sigmoid(mx.stop_gradient(xn) @ self.gate_w[k] + self.gate_b[k])[..., None]
            if GREC["on"]:  # the retain step's gate L1 (--gate-l1) reads the gates through this recorder
                GREC["g"].append(gt)
            hid = xn @ self.w_in[k]
            return gt * ((nn.silu(hid) if self.kind == "mlp" else hid) @ self.w_out[k])

    class Bridge(nn.Module):
        def __init__(self, mode, src):
            super().__init__()
            self.A, self.B, self.gw, self.gb = {}, {}, {}, {}
            for L, cs in write_sets[mode].items():
                mp = maps[(mode, L)]
                if src == "host":  # host features: every write layer reads all of them
                    M_ = mp["np_gh"] if args.bridge_init == "grad" else ridge(Xh_all, inc[L])
                elif args.bridge_init == "grad":
                    M_ = mp["np_g"] if src != "rand" else mp["np_gr"]
                else:
                    M_ = mp["np"] if src != "rand" else np.array(mp["Mr"])
                U, S, Vt = np.linalg.svd(M_, full_matrices=False)
                r = min(args.bridge_rank, len(S))
                self.A[str(L)] = mx.array((U[:, :r] * np.sqrt(S[:r])).astype(np.float32))
                self.B[str(L)] = mx.array((np.sqrt(S[:r])[:, None] * Vt[:r]).astype(np.float32))
                self.gw[str(L)] = mx.zeros((FDh if src == "host" else cs.stop - cs.start,))
                self.gb[str(L)] = mx.zeros((1,))

        def write(self, L, z):
            k = str(L)
            gt = mx.sigmoid(z @ self.gw[k] + self.gb[k])[:, None]
            return gt * ((z @ self.A[k]) @ self.B[k])

    def nparams(m):
        return int(sum(v.size for _, v in tree_flatten(m.trainable_parameters())))

    # ------------------------------------------------------------------ arm definitions
    arms = {"host": {"route": "host"}, "tile_init": {"route": "tile", "loss": None},
            "tile_kl": {"route": "tile", "loss": "kl"}, "tile_sft": {"route": "tile", "loss": "sft"},
            "lora_init": {"route": "tile", "loss": None, "kind": "lora"},
            "lora_kl": {"route": "tile", "loss": "kl", "kind": "lora"},
            "lora_sft": {"route": "tile", "loss": "sft", "kind": "lora"},
            # (1) oracle filter (Pierre 2026-11-16): train and select only on traces where the donor was right
            "tile_kl_ok": {"route": "tile", "loss": "kl", "ok": True},
            "lora_kl_ok": {"route": "tile", "loss": "kl", "kind": "lora", "ok": True}}
    if args.side:
        a_, t_ = (float(x) for x in args.side.split(":"))
        arms[f"side@a{a_:g}t{t_:g}"] = {"route": "side", "alpha": a_, "tau": t_}
    for mode in write_sets:
        for b in BASES:
            kind, src = b.split("_")
            arms[f"{b}@{mode}"] = {"route": "ridge" if kind == "kgrad" else kind, "src": src, "mode": mode,
                                   "target": "grad" if kind == "kgrad" else "inc"}
        for src_ in ("box", "rand", "host"):  # filtered bridge and its controls trained on the same filtered traces
            arms[f"bridge_{src_}_ok@{mode}"] = {"route": "bridge", "src": src_, "mode": mode, "target": "inc", "ok": True}
    for kind in LENS_KINDS:
        for src_ in ("box", "rand", "shuf", "rot"):
            arms[f"lens_{src_}@{kind}"] = {"route": "lens", "src": src_, "kind": kind}
        if SINK:
            for src_ in ("box", "rand", "hrand", "shuf", "const"):
                arms[f"sink_{src_}@{kind}"] = {"route": "sink", "src": src_, "kind": kind}
            if kind in PATH:
                arms[f"sink_path@{kind}"] = {"route": "path", "src": "box", "mask": "host", "kind": kind}
                arms[f"sink_pathrand@{kind}"] = {"route": "path", "src": "box", "mask": "rand", "kind": kind}
                arms[f"sink_pathidle@{kind}"] = {"route": "path", "src": "box", "mask": "idle", "kind": kind}
                for tau_ in [float(x) for x in args.cone_taus.split(",") if x.strip()]:
                    arms[f"sink_cone{tau_:g}@{kind}"] = {"route": "path", "src": "box", "mask": "cone", "tau": tau_,
                                                         "kind": kind}
                    arms[f"sink_conerand{tau_:g}@{kind}"] = {"route": "path", "src": "box", "mask": "conerand",
                                                             "tau": tau_, "kind": kind}
            if NEURON:
                arms[f"sink_neuron@{kind}"] = {"route": "neuron", "src": "gate", "kind": kind}
                arms[f"sink_neuronflat@{kind}"] = {"route": "neuron", "src": "flat", "kind": kind}
    if args.side:
        arms[f"side_gated@a{a_:g}"] = {"route": "side", "alpha": a_, "tau": 0.0, "gated": True}
    if not use_host:
        arms = {n: v for n, v in arms.items() if not n.startswith("bridge_host")}
    if TOPIC:  # no answer oracle on a knowledge topic: the _ok arms would equal their unfiltered twins
        arms = {n: v for n, v in arms.items() if not v.get("ok")}
    if args.arms:
        want = [a.strip() for a in args.arms.split(",") if a.strip()]
        arms = {n: v for n, v in arms.items() if n in want or n.split("@")[0] in want or n == "host"}
        miss = [w for w in want if w != "host" and not any(n == w or n.split("@")[0] == w for n in arms)]
        if miss:  # a typo, or a route this pair dropped (lens skipped, no sinks): say so instead of scoring less
            save_receipt(status="failed", reason=f"requested arms not available: {miss}")
            sys.exit(f"requested arms not available on this run: {miss} (lens skipped: {(receipt.get('lens') or {}).get('skipped')}, "
                     f"sinks: {(receipt.get('sinks') or {}).get('note')})")

    def feats(arm, s):
        key = {"box": "fb", "shuf": "fs", "rand": "fr", "host": "fh"}.get(arm["src"])
        return s[key] if key else None  # lazy: host features exist only when a bridge_host arm runs

    def writes_of(arm, mod, z, T, g_=None):
        """{L: [T, D]} for a bridge arm from standardised features z [T, FD] (None for noise)."""
        ws = write_sets[arm["mode"]]
        w = {}
        for L, cs in ws.items():
            if arm["src"] == "noise":
                zz = mx.array((g_ or rng).standard_normal((T, cs.stop - cs.start)).astype(np.float32))
            elif arm["src"] == "host":
                zz = mx.array(z) if isinstance(z, np.ndarray) else z
            else:
                zz = mx.array(z[:, cs]) if isinstance(z, np.ndarray) else z[:, cs]
            w[L] = mod.write(L, zz)
        return w

    def ridge_writes(arm, zbox, zrand):
        ws = write_sets[arm["mode"]]
        w = {}
        for L, cs in ws.items():
            mp = maps[(arm["mode"], L)]
            k_, ks, kr = ("G", "Gsh", "Gr") if arm["target"] == "grad" else ("M", "Msh", "Mr")
            real = mx.array(zbox[:, cs]) @ mp[k_]
            if arm["src"] == "box":
                w[L] = real
                continue
            v = mx.array(zbox[:, cs]) @ mp[ks] if arm["src"] == "shuf" else mx.array(zrand[:, cs]) @ mp[kr]
            w[L] = v * (mx.sqrt((real * real).sum(-1, keepdims=True)) / (mx.sqrt((v * v).sum(-1, keepdims=True)) + 1e-12))
        return w

    def arm_lp(arm, mod, s, st=None, g_=None):
        T = len(s["ids"])
        if arm["route"] == "tile":
            return run(H, s["ids"], tiles=mod, st=st)
        if arm["route"] == "ridge":
            return run(H, s["ids"], writes=ridge_writes(arm, s["fb"], s["fr"]), st=st)
        if arm["route"] == "bridge":
            return run(H, s["ids"], writes=writes_of(arm, mod, feats(arm, s), T, g_), st=st)
        if arm["route"] == "path":
            return run(H, s["ids"], writes=path_writes(arm, s["fb"], host_sink_acts(s["ids"])), st=st)
        if arm["route"] == "neuron":
            return run(H, s["ids"], writes=neuron_writes(arm, host_gate(s["ids"])), st=st)
        if arm["route"] == "sink":
            return run(H, s["ids"], writes=sink_writes(arm, s["fs"] if arm["src"] == "shuf" else s["fb"], s["fr"]),
                       st=st)
        if arm["route"] == "lens":
            return run(H, s["ids"], writes=lens_writes(arm, s["fs"] if arm["src"] == "shuf" else s["fb"], s["fr"]),
                       st=st)
        return run(H, s["ids"], st=st)

    def gate_phi(lh, ld):
        """host-wins gate features per row from host and donor log-probs [T, V]: both top probabilities, both
        entropies, each model's probability of the other's top token, top-token agreement, KL(donor || host)."""
        ph, pd = mx.exp(lh), mx.exp(ld)
        th, td = mx.argmax(lh, -1), mx.argmax(ld, -1)
        f = [ph.max(-1), pd.max(-1), -(ph * lh).sum(-1), -(pd * ld).sum(-1),
             mx.take_along_axis(ph, td[:, None], -1)[:, 0], mx.take_along_axis(pd, th[:, None], -1)[:, 0],
             (th == td).astype(mx.float32), mx.minimum((pd * (ld - lh)).sum(-1), 20.0)]
        return mx.stack(f, -1)

    GATE = {}

    def gate_of(lh, ld):  # [T] in [0, 1]: how far to trust the donor at each position
        z = (gate_phi(lh, ld) - GATE["mu"]) / GATE["sd"]
        return mx.sigmoid(z @ GATE["w"] + GATE["b"])

    def fit_gate():
        """(2) host-wins gate, trained on GOLD answers (train items only; the gold trace is a correct greedy reply,
        the donor's if right, else the host's, so it has the models' own echo format): p = (1 - a g) p_host + a g p_donor,
        g = sigmoid(w . phi + b), minimising -log p(gold) at the answer positions. Where the host is right and the
        donor wrong the gold loss pushes g down; at run time it needs no oracle."""
        rows = []
        n_skip = 0
        dgreedy = {q["item"]: q for q in tr if q["sample"] == 0}
        for it in [x for x in items if x["split"] == "train"]:
            q = dgreedy.get(it["id"])
            if q is not None and q["donor_correct"]:
                ids = q["ids"]
            else:
                toks, _ = donor_generate(it, 0.0, None, H)
                stopped = bool(toks) and toks[-1] in eos
                if parse(tok.decode([t for t in toks if t not in eos]), stopped) != it["answer"]:
                    n_skip += 1  # neither model's reply is right: no gold trace in the models' format
                    continue
                ids = it["ids"] + toks
            pos = mx.array(list(range(len(it["ids"]) - 1, len(ids) - 1)))
            lh, ld = run(H, ids)[pos], run(Dn, ids)[pos]
            gold = mx.array(ids[len(it["ids"]):])[:, None]
            rows.append((gate_phi(lh, ld), mx.take_along_axis(mx.exp(lh), gold, -1)[:, 0],
                         mx.take_along_axis(mx.exp(ld), gold, -1)[:, 0]))
            mx.eval(*rows[-1])
            del lh, ld
        if not rows:  # no train item has a correct reply from either model: nothing to fit the gate on
            return None
        phi = mx.concatenate([r[0] for r in rows])
        a_h, a_d = mx.concatenate([r[1] for r in rows]), mx.concatenate([r[2] for r in rows])
        GATE["mu"], GATE["sd"] = phi.mean(0), mx.sqrt(phi.var(0)) + 1e-6
        zf = (phi - GATE["mu"]) / GATE["sd"]
        prm = {"w": mx.zeros((phi.shape[1],)), "b": mx.zeros((1,))}
        al = arms_alpha[0]

        def loss(p_):
            g = mx.sigmoid(zf @ p_["w"] + p_["b"]) * al
            return -mx.log((1 - g) * a_h + g * a_d + 1e-9).mean()
        opt = optim.Adam(learning_rate=0.05)
        lg = mx.value_and_grad(loss)
        l0 = float(loss(prm).item())
        for _ in range(500):
            l_, g_ = lg(prm)
            opt.update(prm, g_)
            mx.eval(prm, opt.state)
        GATE.update(prm)
        host_nll = float(-mx.log(a_h + 1e-9).mean().item())
        donor_nll = float(-mx.log(al * a_d + (1 - al) * a_h + 1e-9).mean().item())
        return {"gold_positions": int(phi.shape[0]), "items_without_gold_trace": n_skip, "gold_nll_host": host_nll, "gold_nll_side": donor_nll,
                "gold_nll_gate_init": l0, "gold_nll_gated": float(loss(prm).item()),
                "w": np.array(prm["w"]).round(4).tolist(), "b": float(prm["b"].item())}

    def side_lp(arm, s):  # teacher-forced side mixture at the continuation positions (donor lp = s["t_lp"])
        lh = run(H, s["ids"])[mx.array(s["pos"])]
        ld = mx.array(s["t_lp"].astype(np.float32))
        pd = mx.exp(ld)
        if arm.get("gated"):
            gate = gate_of(lh, ld)[:, None] * arm["alpha"]
        else:
            gate = (pd.max(-1, keepdims=True) >= arm["tau"]).astype(mx.float32) * arm["alpha"]
        return mx.log((1 - gate) * mx.exp(lh) + gate * pd + 1e-30)

    va_ok = [s for s in va if s["donor_correct"]]
    host_val_kl_ok = (float(np.mean([float(kl_rows(s["t_lp"], mx.array(s["h_lp"].astype(np.float32))).mean().item())
                                     for s in va_ok])) if va_ok else None)

    def val_scores(arm, mod):
        g_ = np.random.default_rng(args.seed + 99)  # the noise arm sees the same draws at every selection
        vset = va_ok if arm.get("ok") else va  # the filtered arms select on donor-correct val traces only
        PATH_SIZES.clear()  # each scoring of the validation set records its own sizes once
        if arm["route"] == "side":  # English drift needs donor log-probs on English, which are not stored
            return float(np.mean([float(kl_rows(s["t_lp"], side_lp(arm, s)).mean().item()) for s in vset])), None
        kv = []
        for s in vset:
            REC[0] = s["pos"]
            try:
                kv.append(float(kl_rows(s["t_lp"], arm_lp(arm, mod, s, g_=g_)[mx.array(s["pos"])]).mean().item()))
            finally:
                REC[0] = None
        dv = []
        for e in eng_val:
            lp = arm_lp(arm, mod, e, g_=g_)
            h = mx.array(e["h_lp"].astype(np.float32))
            dv.append(float((mx.exp(h) * (h - lp)).sum(-1).mean().item()))
        return float(np.mean(kv)), float(np.mean(dv))

    def val_ce(arm, mod):
        """mean over validation texts of the arm's nats/byte on the sequence's own continuation tokens (the held
        metric's form). On a topic task those are the text's tokens, never a donor output; on an arithmetic task
        the validation continuation is the donor's greedy reply (receipt host_only says which)"""
        g_ = np.random.default_rng(args.seed + 99)
        v_ = []
        for s in va:
            lp = arm_lp(arm, mod, s, g_=g_)[mx.array(s["pos"])]
            tgt = mx.array(s["ids"][s["P"]:])[:, None]
            nb_ = len(tok.decode(s["ids"][s["P"]:]).encode("utf-8"))
            v_.append(float(-mx.take_along_axis(lp, tgt, axis=-1).sum().item()) / max(nb_, 1))
        return float(np.mean(v_))

    curves_f = open(out / "curves.jsonl", "w")

    def train(name, arm, mod, lr):
        opt = optim.Adam(learning_rate=lr)

        def loss_topic(m_, s):
            lp = arm_lp(arm, m_, s)[mx.array(s["pos"])]
            if arm.get("loss") == "sft":
                tgt = mx.array(s["ids"][s["P"]:])[:, None]
                return -mx.take_along_axis(lp, tgt, axis=-1).mean()
            return kl_rows(s["t_lp"], lp).mean()

        def loss_eng(m_, e):
            l1 = args.gate_l1 > 0 and arm["route"] == "tile"
            GREC["on"], GREC["g"] = l1, []
            try:
                lp = arm_lp(arm, m_, e)
            finally:
                GREC["on"] = False
            h = mx.array(e["h_lp"].astype(np.float32))
            loss = lam[0] * (mx.exp(h) * (h - lp)).sum(-1).mean()
            if l1 and GREC["g"]:  # CWE distill v1 (a0b957809): mean-gate L1 on retain text, outside the drift weight
                loss = loss + args.gate_l1 * mx.stack([g.mean() for g in GREC["g"]]).mean()
            GREC["g"] = []
            return loss

        trs = [s for s in tr if s["donor_correct"]] if arm.get("ok") else tr
        lam = [args.drift_weight]  # adaptive: every arm is held to the same drift cap, so controls compare fairly
        def on_reference(f_):  # the gradient traces run the host's differentiable reference path
            def g_(*a, **kw):
                with dref.reference_path():
                    return f_(*a, **kw)
            return g_
        gt = on_reference(nn.value_and_grad(mod, loss_topic))
        ge = on_reference(nn.value_and_grad(mod, loss_eng))
        metric = selection_metric(arm, args.kl_arm_select)
        kv0, dv0 = val_scores(arm, mod)  # val_kl is recorded for every arm; only `metric` selects
        snaps = {0: mod.parameters()}  # each checkpoint's parameters (MLX updates replace arrays: old trees stay valid)
        hist = [{"arm": name, "epoch": 0, "epoch_frac": 0.0, "val_kl": kv0, "val_ce": val_ce(arm, mod),
                 "val_drift": dv0}]
        curves_f.write(json.dumps(hist[-1]) + "\n")
        curves_f.flush()
        print(f"  {name} ep 0: val KL {kv0:.4f} val CE {hist[-1]['val_ce']:.4f} drift {dv0:.4f}", flush=True)
        # every trained arm sees the same text order (its own generator, not the shared rng), so twins differ only
        # in their loss; --evals-per-epoch checkpoints per epoch, each a selection candidate, and the drift weight
        # adapts at every checkpoint (one adaptation per epoch let a hard-pushing arm overshoot the cap for a whole
        # epoch and revert to init)
        order_rng = np.random.default_rng(args.seed + 501)
        n_b = (len(trs) + args.batch - 1) // args.batch
        marks = sorted({min(n_b, int(round(n_b * (k + 1) / max(1, args.evals_per_epoch))))
                        for k in range(max(1, args.evals_per_epoch))})
        ck, stop = 0, False
        for ep in range(1, args.epochs + 1):
            order = order_rng.permutation(len(trs))
            ls, nb = [], 0
            for b0 in range(0, len(order), args.batch):
                acc = None
                for i in order[b0:b0 + args.batch]:
                    l_, g_ = gt(mod, trs[i])
                    ls.append(float(l_.item()))
                    acc = g_ if acc is None else tree_map(lambda a, b: a + b, acc, g_)
                if args.drift_weight > 0 and retain:
                    l_, g_ = ge(mod, retain[nb % len(retain)])
                    acc = tree_map(lambda a, b: a / len(order[b0:b0 + args.batch]) + b, acc, g_)
                else:
                    acc = tree_map(lambda a: a / len(order[b0:b0 + args.batch]), acc)
                opt.update(mod, acc)
                mx.eval(mod.parameters(), opt.state)
                nb += 1
                stop = budget_left() < 300
                if nb not in marks and not stop:
                    continue
                kv, dv = val_scores(arm, mod)
                ck += 1
                snaps[ck] = mod.parameters()
                hist.append({"arm": name, "epoch": ck, "epoch_frac": round(ep - 1 + nb / n_b, 4),
                             "train_loss": float(np.mean(ls)), "val_kl": kv, "val_ce": val_ce(arm, mod),
                             "val_drift": dv, "drift_weight": lam[0]})
                if dv > args.drift_cap and lam[0] > 0:
                    lam[0] = min(lam[0] * args.drift_adapt, args.drift_weight_max)
                curves_f.write(json.dumps(hist[-1]) + "\n")
                curves_f.flush()
                print(f"  {time.strftime('%H:%M:%S')} {name} ep {hist[-1]['epoch_frac']:g}: train {np.mean(ls):.4f} "
                      f"val KL {kv:.4f} val CE {hist[-1]['val_ce']:.4f} nats/byte drift {dv:.4f}", flush=True)
                if stop:
                    break
            if stop:
                break
        # checkpoint 0 (init) competes only within the drift cap; if none is within it, keep the lowest-drift one
        ck_sel, within = pick_epoch(hist, metric, args.drift_cap)
        mod.update(snaps[ck_sel])
        other = {m_: hist[pick_epoch(hist, m_, args.drift_cap)[0]]["epoch_frac"]
                 for m_ in ("val_kl", "val_ce") if m_ != metric}
        return {"selected_epoch": hist[ck_sel]["epoch_frac"], "selected_checkpoint": ck_sel,
                "checkpoints": len(hist) - 1, "evals_per_epoch": args.evals_per_epoch,
                "reverted_to_init": ck_sel == 0, "selection_metric": metric,
                "selection_rule": f"lowest {metric} among checkpoints (epoch 0 = init, {args.evals_per_epoch} per "
                                  f"epoch) with val English drift <= {args.drift_cap}; none within: the lowest-drift "
                                  "checkpoint",
                "val_kl_selected": hist[ck_sel]["val_kl"], "val_ce_selected": hist[ck_sel]["val_ce"],
                "epoch_other_metric_would_pick": other, "within_drift_cap": bool(within), "curve": hist,
                "drift_weight_final": lam[0], "train_sequences": len(trs), "gate_l1": args.gate_l1,
                "host_only": bool(arm.get("loss") == "sft" and TOPIC)}

    # ------------------------------------------------------------------ generation on the test items
    def generate(name, arm, mod, it):
        need_d = arm["route"] == "side" or (arm["route"] in ("ridge", "bridge", "lens", "sink", "path") and arm["src"] not in ("noise", "host"))
        need_h = (arm["route"] == "bridge" and arm["src"] == "host") or arm["route"] in ("path", "neuron")
        P = it["ids"]
        dprompt = P
        shuf_ = arm["route"] in ("bridge", "lens", "sink") and arm["src"] == "shuf"
        if shuf_:
            dprompt, partner_ = shuf_partner_prompt(items, it)
        hc = H["model"].make_cache()
        dc = Dn["model"].make_cache() if need_d else None
        hc2 = H["model"].make_cache() if need_h else None  # the clean host copy the host-only bridge reads
        st = {}

        def step(ids_h, ids_d):
            dlp, fb, fr = (None, None, None)
            if need_d:
                if arm["route"] == "side":
                    dlp = run(Dn, ids_d, cache=dc)
                else:
                    dlp, fb, fr = dfeat(ids_d, cache=dc)
                    fb, fr = zb(fb), zr(fr)
            T = len(ids_h)
            if arm["route"] == "tile":
                lp = run(H, ids_h, cache=hc, tiles=mod, st=st)
            elif arm["route"] == "ridge":
                lp = run(H, ids_h, cache=hc, writes=ridge_writes(arm, fb, fr), st=st)
            elif arm["route"] == "lens":
                lp = run(H, ids_h, cache=hc, writes=lens_writes(arm, fb, fr), st=st)
            elif arm["route"] == "sink":
                lp = run(H, ids_h, cache=hc, writes=sink_writes(arm, fb, fr), st=st)
            elif arm["route"] == "neuron":  # the clean host copy (hc2) gives the gate input
                lp = run(H, ids_h, cache=hc, writes=neuron_writes(arm, host_gate(ids_h, cache=hc2)), st=st)
            elif arm["route"] == "path":  # the clean host copy (hc2) says which sinks this token opens
                lp = run(H, ids_h, cache=hc, writes=path_writes(arm, fb, host_sink_acts(ids_h, cache=hc2)), st=st)
            elif arm["route"] == "bridge":
                if arm["src"] == "host":
                    z = zh(hfeat(ids_h, cache=hc2))
                else:
                    z = None if arm["src"] == "noise" else (fr if arm["src"] == "rand" else fb)
                lp = run(H, ids_h, cache=hc, writes=writes_of(arm, mod, z, T), st=st)
            else:
                lp = run(H, ids_h, cache=hc)
            lp = lp[-1]
            if arm["route"] == "side":
                pd = mx.exp(dlp[-1])
                if arm.get("gated"):
                    gv = float(gate_of(lp[None], dlp[-1][None])[0].item())
                    st.setdefault("gate", []).append(gv)
                    lp = mx.log((1 - arm["alpha"] * gv) * mx.exp(lp) + arm["alpha"] * gv * pd + 1e-30)
                elif float(pd.max().item()) >= arm["tau"]:
                    lp = mx.log((1 - arm["alpha"]) * mx.exp(lp) + arm["alpha"] * pd + 1e-30)
            return lp

        lp = step(P, dprompt)
        toks = []
        stopped = False
        for _ in range(args.max_tokens):
            t = int(mx.argmax(lp).item())
            if t in eos:
                stopped = True
                break
            toks.append(t)
            lp = step([t], [t])
        text = tok.decode(toks)
        nr = np.concatenate([np.array(x) for x in st.get("nr", [])]) if st.get("nr") else np.zeros(0)
        return {"arm": name, "item": it["id"], "text": text, "stopped": stopped, "pred": parse(text, stopped),
                "correct": parse(text, stopped) == it["answer"],
                **({"shuf_partner": partner_} if shuf_ else {}),  # None: no same-length partner (fallback)
                "norm_ratio": float(nr.mean()) if len(nr) else None,
                "gate_mean": float(np.mean(st["gate"])) if st.get("gate") else None,
                "clamped_frac": float((nr > args.norm_budget).mean()) if len(nr) and args.norm_budget > 0 else None}

    def drift_test(arm, mod):
        if arm["route"] == "side":
            return None
        dv = []
        for e in eng_test:
            lp = arm_lp(arm, mod, e)
            h = mx.array(e["h_lp"].astype(np.float32))
            dv.append(float((mx.exp(h) * (h - lp)).sum(-1).mean().item()))
        return float(np.mean(dv))

    test = [it for it in items if it["split"] == "test"]

    def text_bytes(s):
        return len(tok.decode(s["ids"][s["P"]:]).encode("utf-8"))  # the scored span, as in the donor-gap screen

    def tf_nats(lp, s):
        tgt = mx.array(s["ids"][s["P"]:])[:, None]
        return float(-mx.take_along_axis(lp, tgt, axis=-1).sum().item())

    def tf_eval(arm, mod):
        """per held text: nats on tokens 3..end (the text's own context), teacher-forced"""
        g_ = np.random.default_rng(args.seed + 199)
        out_ = []
        for s in te:
            pos = mx.array(s["pos"])
            if arm["route"] == "side":
                lh = run(H, s["ids"])[pos]
                ld = run(Dn, s["ids"])[pos]
                pd = mx.exp(ld)
                gate = (gate_of(lh, ld)[:, None] * arm["alpha"] if arm.get("gated") else
                        (pd.max(-1, keepdims=True) >= arm["tau"]).astype(mx.float32) * arm["alpha"])
                lp = mx.log((1 - gate) * mx.exp(lh) + gate * pd + 1e-30)
                lp = lp - mx.logsumexp(lp, axis=-1, keepdims=True)
            else:
                st = {}
                lp = arm_lp(arm, mod, s, st=st, g_=g_)[pos]
            out_.append(tf_nats(lp, s))
        return np.array(out_)

    def boot_ci(x, n=2000, seed=0):
        g = np.random.default_rng(seed)
        b = np.sort([x[g.integers(0, len(x), len(x))].mean() for _ in range(n)])
        return float(b[int(0.025 * n)]), float(b[int(0.975 * n)])

    if TOPIC:
        tb = np.array([text_bytes(s) for s in te], dtype=np.float64)
        donor_nats = np.array([tf_nats(run(Dn, s["ids"])[mx.array(s["pos"])], s) for s in te])
        host_nats = np.array([tf_nats(run(H, s["ids"])[mx.array(s["pos"])], s) for s in te])
        tf_host, tf_donor = host_nats / tb, donor_nats / tb
        print(f"held topic texts: host {tf_host.mean():.4f} donor {tf_donor.mean():.4f} nats/byte "
              f"(gap {np.mean(tf_host - tf_donor):.4f}, 95% CI {boot_ci(tf_host - tf_donor)})", flush=True)
    summary = {"label": label, "host_checkpoint": H["ck"], "donor_checkpoint": Dn["ck"], "mlx2_source_revision": rev,
               "tree_dirty": dirty, "task": f"{args.task}: {task_name(args.task)}",
               "n_test": len(te) if TOPIC else len(test),
               "donor_test_acc": None if TOPIC else float(np.mean(donor_acc["test"])),
               "host_val_kl_to_donor": host_val_kl, "box_layers": box_layers, "host_stub_layers": stub_layers,
               "write_sets": {m: sorted(ws) for m, ws in write_sets.items()}, "arms": {}}
    # ------------------------------------------------------------------ autoregressive split (Pierre 2026-11-18: "look
    # at the backwards paths too, in an autoregressive sense"). A write at position s reaches the prediction at t > s
    # through attention (the K/V written at s); the transport objective and the linear diagnostics only see s = t.
    def ar_positions(i_, s):
        return np.random.default_rng(args.seed + 83 + i_).permutation(np.array(s["pos"]))[:args.ar_positions]

    def kl_row(t_row, lp_row):
        t = mx.array(np.asarray(t_row, dtype=np.float32))
        return float((mx.exp(t) * (t - lp_row)).sum().item())

    def ar_donor():
        """the donor's own box: KL(donor || donor with the box knocked out) at a sampled position t when the box is
        removed only AT t (self), only at positions BEFORE t (past: reaches t through attention only), or everywhere"""
        # MEAN-ablation to the English mean activation (what the transport treats as "no signal": it carries
        # a - a_English_mean), so the donor's self / past numbers are on the same footing as the host arms'
        oh = {d: np.isin(np.arange(len(wn_d[d])), box[d]).astype(np.float32) for d in box_layers}
        mu_full = {}
        for d in box_layers:
            v_ = np.zeros(len(wn_d[d]), np.float32)
            v_[box[d]] = mu_eng_raw[colsd[d]] if LENS_KINDS else 0.0
            mu_full[d] = v_

        def zm(pm):
            return {d: (mx.array(1.0 - pm[:, None] * oh[d][None, :])[None],
                        mx.array(pm[:, None] * mu_full[d][None, :])[None]) for d in box_layers}
        r_ = {"self": [], "past": [], "all": []}
        for i_, s in enumerate(va[:args.ar_seqs]):
            T_ = len(s["ids"])
            base = run(Dn, s["ids"])
            lp_all = run(Dn, s["ids"], ablate=zm(np.ones(T_, np.float32)))
            for t in ar_positions(i_, s):
                self_ = np.zeros(T_, np.float32)
                self_[t] = 1
                past_ = (np.arange(T_) < t).astype(np.float32)
                b_ = np.array(base[t])
                r_["self"].append(kl_row(b_, run(Dn, s["ids"], ablate=zm(self_))[t]))
                r_["past"].append(kl_row(b_, run(Dn, s["ids"], ablate=zm(past_))[t]))
                r_["all"].append(kl_row(b_, lp_all[t]))
        m_ = {k: float(np.mean(v)) if v else None for k, v in r_.items()}
        if r_["self"]:
            m_["past_share"] = m_["past"] / max(m_["past"] + m_["self"], 1e-12)
        m_["positions"] = len(r_["self"])
        return m_

    def ar_host(arm, mod):
        """for a write arm: the gain in KL(donor || host) at a sampled position t when the host write is applied only
        AT t (self), only BEFORE t (past: through attention only), or everywhere (all)"""
        r_ = {"self": [], "past": [], "all": []}
        for i_, s in enumerate(va[:args.ar_seqs]):
            T_ = len(s["ids"])
            hb = run(H, s["ids"])  # the host baseline in the same precision as the arm passes
            lp_all = arm_lp(arm, mod, s)
            for t in ar_positions(i_, s):
                j_ = s["pos"].index(int(t))
                kh = kl_row(s["t_lp"][j_], hb[t])
                r_["all"].append(kh - kl_row(s["t_lp"][j_], lp_all[t]))
                for nm_, pm in (("self", (np.arange(T_) == t)), ("past", (np.arange(T_) < t))):
                    WMASK[0] = mx.array(pm.astype(np.float32))
                    try:
                        lp_ = arm_lp(arm, mod, s)
                    finally:
                        WMASK[0] = None
                    r_[nm_].append(kh - kl_row(s["t_lp"][j_], lp_[t]))
        m_ = {f"gain_{k}": float(np.mean(v)) if v else None for k, v in r_.items()}
        if r_["self"]:
            m_["past_share"] = m_["gain_past"] / (abs(m_["gain_past"]) + abs(m_["gain_self"]) + 1e-12)
        m_["positions"] = len(r_["self"])
        return m_

    if args.ar_seqs > 0 and any(a.get("route") in ("lens", "sink", "path", "neuron") for a in arms.values()):
        summary["donor_box_ar_split"] = receipt["donor_box_ar_split"] = ar_donor()
        print(f"donor box autoregressive split (KL at t when the box is removed): "
              f"{summary['donor_box_ar_split']}", flush=True)
    host_acc = None
    arms_alpha = [1.0]
    NPB = {}  # topic task: per-arm per-text held nats/byte, for the paired twin contrasts below
    if any(a.get("route") == "tile" for a in arms.values()):
        # CWE distill v1 preflight (a0b957809): an untrained tile (w_out = 0) must be the host exactly, so every
        # trained arm starts from the host and epoch 0 is the host itself
        t0_ = Tiles(args.seed + 1)
        chk_ = (va or tr or te)[0]
        dmax_ = float(mx.abs(run(H, chk_["ids"], tiles=t0_) - run(H, chk_["ids"])).max().item())
        receipt["tile_init_equals_host"] = {"max_abs_logprob": dmax_, "tol": 1e-4, "text": chk_["item"]}
        save_receipt()
        if not dmax_ <= 1e-4:
            save_receipt(status="failed", reason=f"untrained tile differs from the host: max |dlogp| {dmax_:.3g}")
            sys.exit(f"untrained tile differs from the host: max |dlogp| {dmax_:.3g}")
        del t0_
    for name, arm in arms.items():
        if budget_left() < 600:
            print(f"budget: stopping before {name}", flush=True)
            summary["stopped_before"] = name
            break
        ta = time.time()
        mod, tinfo = None, None
        PATH_SIZES.clear()
        if arm.get("ok") and (not any(q["donor_correct"] for q in tr) or not va_ok):
            summary["arms"][name] = {"skipped": "no donor-correct train or val traces"}
            print(f"{name}: skipped, no donor-correct train or val traces", flush=True)
            continue
        if arm["route"] == "tile":
            mod = Tiles(args.seed + 1, arm.get("kind", "mlp"))
            if arm["loss"]:
                tinfo = train(name, arm, mod, args.tile_lr)
        elif arm["route"] == "bridge":
            mod = Bridge(arm["mode"], arm["src"])
            tinfo = train(name, arm, mod, args.bridge_lr)
        elif arm.get("gated"):
            arms_alpha[0] = arm["alpha"]
            ginfo = fit_gate()
            if ginfo is None:
                summary["arms"][name] = {"skipped": "no train item with a correct reply from either model"}
                print(f"{name}: skipped, no gold traces for the gate", flush=True)
                continue
            summary["host_wins_gate"] = ginfo
            print(f"  gate fit: gold NLL host {ginfo['gold_nll_host']:.3f} side {ginfo['gold_nll_side']:.3f} "
                  f"gated {ginfo['gold_nll_gated']:.3f}", flush=True)
        if TOPIC:
            npb = tf_eval(arm, mod) / tb
            d_ = tf_host - npb  # > 0: the arm is better than the host on that text
            gap_ = tf_host - tf_donor
            kv, dvv = val_scores(arm, mod)
            gb = np.random.default_rng(11)
            rb = []
            for _ in range(2000):
                ix = gb.integers(0, len(d_), len(d_))
                den = gap_[ix].mean()
                if abs(den) > 1e-9:
                    rb.append(d_[ix].mean() / den)
            rb = np.sort(rb)
            e = {"test_nats_per_byte": float(npb.mean()), "host_nats_per_byte": float(tf_host.mean()),
                 "donor_nats_per_byte": float(tf_donor.mean()),
                 "delta_vs_host": float(d_.mean()), "delta_vs_host_ci95": boot_ci(d_),
                 "gap_closed": float(d_.mean() / gap_.mean()) if abs(gap_.mean()) > 1e-9 else None,
                 "gap_closed_ci95": ([float(rb[int(0.025 * len(rb))]), float(rb[int(0.975 * len(rb)) - 1])]
                                     if len(rb) > 100 else None),
                 "val_kl": kv, "val_kl_closed": (host_val_kl - kv) / host_val_kl if host_val_kl > 0 else None,
                 "eng_drift_test": drift_test(arm, mod) if name != "host" else 0.0, "eng_drift_val": dvv,
                 "trainable_params": nparams(mod) if mod is not None else 0, "wall_s": round(time.time() - ta, 1),
                 "ar_split": (ar_host(arm, mod) if args.ar_seqs > 0 and arm.get("route") in ("lens", "sink", "path", "neuron")
                              else None),
                 "path_units_mean": (float(np.mean(PATH_SIZES)) if PATH_SIZES else
                                     (float(sum(len(v_) for v_ in SINK.values())) if arm.get("route") == "sink" else
                                      (float(min(args.path_units, sum(len(v_) for v_ in SINK.values())))
                                       if arm.get("route") == "path" else None)))}
            e["paired_vs_host"] = PS.paired(d_, [s_.get("group") for s_ in te], mue=args.min_useful_npb)
            e["selection_metric"] = (selection_metric(arm, args.kl_arm_select)
                                     if arm["route"] in ("tile", "bridge") else None)
            if tinfo:
                e |= {k: tinfo[k] for k in ("selected_epoch", "selected_checkpoint", "checkpoints", "evals_per_epoch",
                                            "reverted_to_init", "selection_metric", "selection_rule", "val_kl_selected",
                                            "val_ce_selected", "epoch_other_metric_would_pick", "within_drift_cap",
                                            "train_sequences", "drift_weight_final", "gate_l1", "host_only")}
            NPB[name] = npb
            for s_, v_ in zip(te, npb):
                rows_f.write(json.dumps({"arm": name, "item": s_["item"], "probe_id": s_.get("probe_id"),
                                         "group": s_.get("group"), "nats_per_byte": float(v_)}) + "\n")
            rows_f.flush()
            summary["arms"][name] = e
            (out / "summary.json").write_text(json.dumps(summary, indent=1))
            print(f"{time.strftime('%H:%M:%S')} {name}: {npb.mean():.4f} nats/byte (host {tf_host.mean():.4f}, "
                  f"donor {tf_donor.mean():.4f}), gap closed {e['gap_closed']} CI {e['gap_closed_ci95']}, "
                  f"drift {e['eng_drift_test']}", flush=True)
            save_receipt(stage=f"done {name}")
            mx.clear_cache()
            continue
        rs = [generate(name, arm, mod, it) for it in test]
        for r in rs:
            rows_f.write(json.dumps(r) + "\n")
        rows_f.flush()
        acc = float(np.mean([r["correct"] for r in rs]))
        # transmission test: answers matching the donor's, and the donor's own errors reproduced (an arm that gets
        # those specific wrong answers received item-specific information from the donor)
        dwrong = [it["id"] for it in test if donor_pred[it["id"]] is not None and donor_pred[it["id"]] != it["answer"]]
        dval = [i for i in donor_pred if donor_pred[i] is not None]
        byid = {r["item"]: r for r in rs}
        if name == "host":
            host_acc = acc
        kv, dvv = val_scores(arm, mod)
        kv_all = val_scores(dict(arm, ok=False), mod)[0] if arm.get("ok") else kv
        ar_ = ar_host(arm, mod) if args.ar_seqs > 0 and arm.get("route") in ("lens", "sink", "path", "neuron") else None
        base = host_val_kl_ok if arm.get("ok") else host_val_kl
        nrs = [r["norm_ratio"] for r in rs if r["norm_ratio"] is not None]
        cfs = [r["clamped_frac"] for r in rs if r["clamped_frac"] is not None]
        gms = [r["gate_mean"] for r in rs if r.get("gate_mean") is not None]
        gap = summary["donor_test_acc"] - (host_acc if host_acc is not None else acc)
        e = {"test_acc": acc,
             "gap_closed": (acc - host_acc) / gap if host_acc is not None and abs(gap) > 1e-9 else None,
             "val_kl": kv, "val_kl_closed": (base - kv) / base if base else None,
             "val_set": "donor-correct val traces" if arm.get("ok") else "all val traces",
             "val_kl_all_val": kv_all, "val_kl_closed_all_val": (host_val_kl - kv_all) / host_val_kl if host_val_kl > 0 else None,
             "eng_drift_test": drift_test(arm, mod) if name != "host" else 0.0, "eng_drift_val": dvv,
             "norm_ratio_mean": float(np.mean(nrs)) if nrs else None,
             "clamped_frac": float(np.mean(cfs)) if cfs else None,
             "trainable_params": nparams(mod) if mod is not None else 0, "wall_s": round(time.time() - ta, 1),
             "agree_with_donor": float(np.mean([byid[i]["pred"] == donor_pred[i] for i in dval])) if dval else None,
             "copies_donor_error": int(sum(byid[i]["pred"] is not None and byid[i]["pred"] == donor_pred[i]
                                           for i in dwrong)),
             "donor_errors": len(dwrong), "gate_mean": float(np.mean(gms)) if gms else None,
             "ar_split": ar_,
             "path_units_mean": (float(np.mean(PATH_SIZES)) if PATH_SIZES else
                                 (float(sum(len(v_) for v_ in SINK.values())) if arm.get("route") == "sink" else
                                  (float(min(args.path_units, sum(len(v_) for v_ in SINK.values())))
                                   if arm.get("route") == "path" else None)))}
        if tinfo:
            e |= {k: tinfo[k] for k in ("selected_epoch", "val_kl_selected", "within_drift_cap", "train_sequences",
                                        "drift_weight_final")}
        summary["arms"][name] = e
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
        print(f"{time.strftime('%H:%M:%S')} {name}: acc {acc:.3f} gap closed {e['gap_closed']} val KL {kv:.4f} "
              f"drift {e['eng_drift_test']} norm ratio {e['norm_ratio_mean']} clamped {e['clamped_frac']}"
              + (f" ar {ar_}" if ar_ else ""), flush=True)
        save_receipt(stage=f"done {name}")
        mx.clear_cache()
    if TOPIC and NPB:
        # paired twin contrasts (proposal v4 §5.0c): positive = the first arm's held loss is LOWER than the second's
        # (control minus arm). Equal capacity, data and drift cap within each pair; the selection metric of each
        # arm is in its entry. Pierre 2026-11-18: if tile_sft ties tile_kl the host-native topic adapter is still
        # wanted, reported as no transfer.
        grp_ = [s_.get("group") for s_ in te]
        con_ = {}
        for a_, b_ in (("tile_kl", "tile_sft"), ("lora_kl", "lora_sft"), ("tile_kl", "tile_init"),
                       ("tile_sft", "tile_init"), ("lora_kl", "tile_init"), ("lora_sft", "tile_init"),
                       ("tile_kl", "lora_kl"), ("tile_sft", "lora_sft")):
            if a_ in NPB and b_ in NPB:
                c_ = PS.paired(NPB[b_] - NPB[a_], grp_, mue=args.min_useful_npb)
                # an arm that reverted to init or never got inside the drift cap was not trained in effect: a
                # contrast against it says nothing about donor supervision (review of 7c228788d)
                bad_ = [n_ for n_ in (a_, b_) if n_ != "tile_init" and (
                    summary["arms"][n_].get("reverted_to_init") or summary["arms"][n_].get("within_drift_cap") is False)]
                c_["untrained_arms"] = bad_
                twin_ = a_.endswith("_kl") and b_.endswith("_sft")
                c_["reading"] = (None if not twin_ else
                                 f"uninformative: {bad_} reverted to init or stayed outside the drift cap" if bad_ else
                                 "donor supervision beats host-only training" if c_["ci95_group"][0] > 0 else
                                 "no transfer shown (tie or host-only better): a host-native topic adapter result")
                con_[f"{a_} vs {b_}"] = c_
        summary["contrasts"] = con_
        summary["min_useful_effect_npb"] = args.min_useful_npb
        summary["resolution_note"] = ("paired_se / mde80 (2.80 x SE) over texts; ci95_group / se_group / "
                                      "mde80_group from a source-group bootstrap; vs_min_useful_effect flags the "
                                      "group interval against min_useful_effect_npb")
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
    # size / quality / perplexity front over the sink-family arms, and the pre-registered pick (--pareto-gain):
    # the smallest path whose val-KL gain over the host is >= that share of sink_box's, English drift within the cap
    for kind in LENS_KINDS:
        ref = summary["arms"].get(f"sink_box@{kind}")
        if not ref or "val_kl" not in ref:
            continue
        g_ref = host_val_kl - ref["val_kl"]
        front = []
        for n_, e_ in summary["arms"].items():
            if not n_.endswith("@" + kind) or e_.get("path_units_mean") is None or "val_kl" not in e_:
                continue
            front.append({"arm": n_, "units_per_token": round(e_["path_units_mean"], 1),
                          "val_kl": round(e_["val_kl"], 4), "gain_share": round((host_val_kl - e_["val_kl"]) / g_ref, 3)
                          if g_ref > 0 else None, "eng_drift": e_.get("eng_drift_test"),
                          "test": e_.get("test_acc", e_.get("test_nats_per_byte"))})
        front.sort(key=lambda r: r["units_per_token"])
        ok_ = [r for r in front if r["gain_share"] is not None and r["gain_share"] >= args.pareto_gain
               and (r["eng_drift"] is None or r["eng_drift"] <= args.drift_cap)
               and not r["arm"].startswith(("sink_rand", "sink_shuf", "sink_const", "sink_hrand", "sink_pathrand",
                                            "sink_conerand"))]
        if ref.get("eng_drift_test") is not None and ref["eng_drift_test"] > args.drift_cap:
            ok_ = []  # the reference itself breaks the drift cap: no pick
        summary.setdefault("pareto", {})[kind] = {"rule": f"smallest units/token with gain >= {args.pareto_gain} x "
                                                          f"sink_box's and drift <= {args.drift_cap}; controls excluded; "
                                                          "a point estimate (no CI): read it against the matched-size "
                                                          "random controls (sink_conerand, sink_pathrand)",
                                                  "front": front, "pick": ok_[0]["arm"] if ok_ else None}
        print(f"pareto {kind}: pick {summary['pareto'][kind]['pick']}; front {front}", flush=True)
    save_receipt(status="complete" if "stopped_before" not in summary else "partial (budget)", stage="done",
                 summary=summary)
    print(json.dumps({n: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in e.items()}
                      for n, e in summary["arms"].items()}, indent=1), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as e_:  # leave a failed receipt, not "running", for any abnormal exit
        r_ = _STATE.get("receipt")
        if r_ is not None and r_.get("status") == "running":
            why_ = (str(getattr(e_, "code", None)) if isinstance(e_, SystemExit) else repr(e_))[:2000]
            if "save" in _STATE:
                _STATE["save"](status="failed", reason=why_)
            else:  # before the full receipt exists (refused before or during model load)
                (_STATE["out"] / "run.json").write_text(json.dumps(dict(r_, status="failed", reason=why_), indent=1))
        raise
