#!/usr/bin/env python3
"""DS4.1 port of the K3.1 atlas summarizer. Turn atlas units + the router map into readable per-expert summaries.

Writes OUT/experts.jsonl (one line per mapped expert) and OUT/router_experts.jsonl (all 82,432
experts, router level), plus OUT/SUMMARY.md with coverage, controls and example themes.
"""
import glob, json, os, sys, unicodedata
from pathlib import Path
import numpy as np

OUT = sys.argv[1]

from tokenizers import Tokenizer

tok = Tokenizer.from_file(str(Path.home() / "mlx-models/DeepSeek-V4.1-Flash-1017-Source/tokenizer.json"))
_cache = {}


def s(i):
    i = int(i)
    if i not in _cache:
        try:
            _cache[i] = tok.decode([i], skip_special_tokens=False)
        except Exception:
            _cache[i] = f"<{i}>"
    return _cache[i]


def script(t):
    letters = [c for c in t if c.isalpha()]
    if not letters:
        return "digits" if any(c.isdigit() for c in t) else "punct"
    n = unicodedata.name(letters[0], "")
    for k, lab in (("CJK", "cjk"), ("HIRAGANA", "cjk"), ("KATAKANA", "cjk"), ("HANGUL", "hangul"), ("CYRILLIC", "cyrillic"),
                   ("ARABIC", "arabic"), ("GREEK", "greek"), ("LATIN", "latin")):
        if k in n:
            return lab
    return "other"


def family_rate(ids):
    norm = [s(i).strip().lower() for i in ids]
    return sum(norm.count(x) > 1 for x in norm) / max(1, len(norm))


def main():
    rows, ctrl = [], []
    for p in sorted(glob.glob(f"{OUT}/units/*.json")):
        u = json.load(open(p))
        z = np.load(f"{OUT}/units/{u['npz']}")
        L = u["layer"]
        for e in [int(x) for x in z["experts"]]:
            k = z[f"e{e}_kurt"]
            det = np.argsort(-k)[:12]
            neurons = [{"n": int(n), "kurt": round(float(k[n]), 1),
                        "pos": [s(i) for i in z[f"e{e}_pos_id"][n][:8]],
                        "neg": [s(i) for i in z[f"e{e}_neg_id"][n][:4]]} for n in det]
            top_ids = np.concatenate([z[f"e{e}_pos_id"][n][:10] for n in det])
            scr = {}
            for i in top_ids:
                scr[script(s(i))] = scr.get(script(s(i)), 0) + 1
            rows.append({"layer": L, "expert": e, "host": u["host"], 
                         "detector_share": round(float((k > 20).mean()), 4), "kurt_median": round(float(np.median(k)), 2),
                         "kurt_max": round(float(k.max()), 1),
                         "family_rate_top": round(float(np.mean([family_rate(z[f'e{e}_pos_id'][n][:20]) for n in det])), 3),
                         "script_mix": scr, "neurons": neurons})
        for key in z.files:
            if key.startswith("ctrl") and key.endswith("_kurt"):
                kc = z[key]
                ctrl.append({"layer": L, "detector_share": float((kc > 20).mean()), "kurt_max": float(kc.max())})
    with open(f"{OUT}/experts.jsonl.tmp", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(f"{OUT}/experts.jsonl.tmp", f"{OUT}/experts.jsonl")

    with open(f"{OUT}/router_experts.jsonl.tmp", "w") as f:
        for p in sorted(glob.glob(f"{OUT}/router/layer_*.npz")):
            L = int(p.split("layer_")[1][:2]); z = np.load(p)
            for e in range(384):
                f.write(json.dumps({"layer": L, "expert": e, "tokens_selecting": int(z["counts"][e]),
                                    
                                    "router_top": [s(i) for i in z["top_id"][e][:20]]}, ensure_ascii=False) + "\n")
    os.replace(f"{OUT}/router_experts.jsonl.tmp", f"{OUT}/router_experts.jsonl")

    ds = np.array([r["detector_share"] for r in rows]) if rows else np.zeros(1)
    cs = np.array([c["detector_share"] for c in ctrl]) if ctrl else np.zeros(1)
    by_depth = {}
    for r in rows:
        by_depth.setdefault(r["layer"] // 8, []).append(r["detector_share"])
    cdepth = {}
    for c in ctrl:
        cdepth.setdefault(c["layer"] // 8, []).append(c["detector_share"])
    best = sorted([r for r in rows if r["layer"] < 8], key=lambda r: -r["detector_share"])[:8]
    md = ["# DeepSeek-V4.1-Flash-1017-Source expert token atlas (checkpoint c30fd25e)", "",
          f"Mapped experts: **{len(rows)}** (weights-level, embedding-level probe; method as in the expert-slice probe).",
          "Router-level map: all 15,360 experts (`router_experts.jsonl`).", "",
          "## Controls", "",
          f"- Real experts: neurons with kurtosis > 20 (sharp token detectors): mean share **{ds.mean():.3f}** (median {np.median(ds):.3f}).",
          f"- Row-permuted controls ({len(ctrl)}): mean share **{cs.mean():.4f}**, max kurtosis {max([c['kurt_max'] for c in ctrl] or [0]):.0f}.", "",
          "## Detector share by depth (embedding-level probes lose meaning in deep layers)", "",
          "| Layers | experts | mean detector share | control share | real above control max |", "|---|---|---|---|---|"]
    for d in sorted(by_depth):
        cd = cdepth.get(d, [0.0])
        above = float(np.mean(np.array(by_depth[d]) > max(cd)))
        md.append(f"| {d*8}-{d*8+7} | {len(by_depth[d])} | {np.mean(by_depth[d]):.3f} | {np.mean(cd):.3f} | {above:.3f} |")
    md += ["", "Only layers whose controls stay near zero carry token meaning at this probe level; deeper rows are"
           " dominated by the norm/projection shape and need the routing capture instead."]
    md += ["", "## Examples: sharpest experts (layers 0-7, where controls are clean)", ""]
    for r in best:
        md.append(f"- **L{r['layer']} E{r['expert']}** (share {r['detector_share']:.2f}): " +
                  "; ".join("`" + " ".join(t.strip() or repr(t) for t in n["pos"][:6]) + "`" for n in r["neurons"][:3]))
    open(f"{OUT}/SUMMARY.md", "w").write("\n".join(md) + "\n")
    print(f"summarised {len(rows)} experts, {len(ctrl)} controls")


if __name__ == "__main__":
    main()
