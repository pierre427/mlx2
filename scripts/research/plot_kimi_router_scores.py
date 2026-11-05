"""Plot an existing Kimi score receipt without loading a model or using GPU."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    out = args.directory
    data = json.loads((out / "router-scores.json").read_text())
    spec = json.loads((out / "spec.json").read_text())
    tokens = json.loads((out / "prompt-tokens.json").read_text())
    layers = sorted(data["layers"], key=lambda row: row["layer_index"])
    assert data["complete"] and len(layers) == 26
    # This frozen prompt has12 question tokens, plus7 chat-template tokens.
    # Identify the user body by its actual header/end markers, not a word split.
    labels = tokens["labels"]
    start = labels.index("<|im_middle|>") + 1
    stop = labels.index("<|im_end|>")
    body = slice(start, stop)
    ranked, corrected, counts = [], [], []
    for layer in layers:
        raw = np.array(layer["raw_scores"])
        selection = np.array(layer["selection_scores"])
        assert raw.shape == (len(labels), 256)
        order = np.argsort(-selection, axis=-1)
        rank = np.take_along_axis(raw, order, -1)
        ranked.append(rank)
        corrected.append(np.take_along_axis(selection, order, -1))
        top = rank[:, :12]
        counts.append(np.clip((top >= .8 * top.max(-1, keepdims=True)).sum(-1), 3, 12))
    ranked, corrected, counts = map(np.array, (ranked, corrected, counts))
    mean = ranked[:, body, :16].mean(1)
    question_counts = counts[:, body]
    stats = {"checkpoint": spec["checkpoint"], "source_revision": spec["source_revision"],
        "question": spec["candidate"]["profile_case"]["question"],
        "question_token_positions_one_based": list(range(start+1, stop+1)),
        "prompt_token_count": len(labels), "question_token_count": stop-start,
        "observed_experts_per_layer_token": 8, "shared_experts": 1,
        "counterfactual_rule": "Among the12 candidates ranked by sigmoid+correction, count raw scores >=0.8*largest raw score in those12; clamp to3..12. No routing change executed.",
        "counterfactual_question_counts": {str(k): int((question_counts == k).sum()) for k in range(3, 13)},
        "counterfactual_mean_k_question": float(question_counts.mean()),
        "actual_selection_8_not_raw_sort": True}
    (out / "plot-statistics.json").write_text(json.dumps(stats, indent=2) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(18, 13.8), gridspec_kw={"width_ratios": [16, 19]})
    fig.patch.set_facecolor("#f8fafc")
    fig.suptitle("One prompt · 26 MoE layers · 256 experts per layer", x=.055, ha="left", y=.975,
                 fontsize=23, fontweight="bold", color="#0f172a")
    fig.text(.055, .941, '“' + stats["question"] + '”', fontsize=15, color="#334155")
    fig.text(.055, .915, "Kimi Linear 48B Q8 · batch 1 · stock top-8 routing · one shared expert remains active", fontsize=11, color="#475569")
    im = ax.imshow(mean, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_title("Observed raw scores, ranked by the actual router\nMean across the 12 question tokens", loc="left", fontsize=13, pad=15)
    ax.set_xticks(range(16), range(1,17))
    ax.set_yticks(range(26), [f"L{row['layer_index']:02d}" for row in layers])
    ax.set_xlabel("Expert rank within each token (identities can differ)", labelpad=10)
    ax.set_ylabel("Zero-based model layer; L00 is dense", labelpad=12)
    ax.axvline(7.5, color="#f97316", linewidth=2.4)
    for i in range(26):
        for j in range(16):
            ax.text(j,i,f"{mean[i,j]:.2f}",ha="center",va="center",fontsize=7.7,
                    color="#0f172a" if mean[i,j]>.57 else "white")
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=.026, pad=.105)
    cb.set_label("Raw sigmoid score (not calibrated usefulness)", fontsize=10)
    colors = plt.cm.YlGnBu(np.linspace(.10, .95, 10))
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(2.5,13.5,1), cmap.N)
    jm = bx.imshow(counts, aspect="auto", cmap=cmap, norm=norm)
    bx.set_title("Illustrative adaptive count at each input token\n80% relative-score threshold, bounded to 3–12", loc="left", fontsize=13, pad=15)
    bx.set_yticks(range(26), [f"L{row['layer_index']:02d}" for row in layers])
    short = {"<|im_user|>":"<user>","<|im_middle|>":"<middle>","<|im_end|>":"<end>","<|im_assistant|>":"<assistant>"}
    bx.set_xticks(range(len(labels)), [short.get(label,label).strip() for label in labels], rotation=65, ha="right",fontsize=9)
    bx.axvline(start-.5,color="#e879f9",linewidth=2)
    bx.axvline(stop-.5,color="#e879f9",linewidth=2)
    bx.set_xlabel("Actual token sequence; magenta lines enclose the question", labelpad=12)
    for i in range(26):
        for j in range(len(labels)):
            bx.text(j,i,str(counts[i,j]),ha="center",va="center",fontsize=8.5,
                    color="white" if counts[i,j]>=8 else "#0f172a",fontweight="bold" if counts[i,j]>=9 else "normal")
    cb2=fig.colorbar(jm, ax=bx, orientation="horizontal", fraction=.026, pad=.105,ticks=range(3,13))
    cb2.set_label("Counterfactual expert count — no pruning or expansion was executed",fontsize=10)
    fig.text(.055,.043,"Orange line: trained cutoff after rank 8. Ranking uses sigmoid + learned correction bias; combining uses raw sigmoid scores.",fontsize=10,color="#334155")
    fig.text(.055,.023,"One prompt cannot establish a safe adaptive policy. Scores cover all input tokens; generated-token routing is excluded. Source " + spec["source_revision"][:10] + ".",fontsize=10,color="#475569")
    fig.subplots_adjust(left=.055,right=.975,top=.86,bottom=.18,wspace=.16)
    fig.savefig(out/"router-scores-and-adaptive-counts.png",dpi=170,facecolor=fig.get_facecolor())
    fig.savefig(out/"router-scores-and-adaptive-counts.svg",facecolor=fig.get_facecolor())
    plt.close(fig)

    fig, axes = plt.subplots(7,4,figsize=(18,18),sharex=True,sharey=True)
    fig.patch.set_facecolor("#f8fafc")
    ranks=np.arange(1,257)
    for i,ax in enumerate(axes.flat):
        if i>=26:
            ax.axis("off")
            continue
        raw=ranked[i,body];sel=corrected[i,body]
        lo,hi=np.quantile(raw,[.1,.9],axis=0)
        ax.fill_between(ranks,lo,hi,color="#38bdf8",alpha=.22)
        ax.plot(ranks,raw.mean(0),color="#0369a1",lw=1.5)
        ax.plot(ranks,sel.mean(0),color="#c2410c",lw=1,alpha=.85)
        ax.axvline(8.5,color="#e11d48",ls="--",lw=1)
        ax.set_title(f"L{layers[i]['layer_index']:02d}",loc="left",fontsize=11,fontweight="bold")
        ax.set_xscale("log");ax.set_xlim(1,256);ax.set_ylim(-.36,1.06)
        ax.set_xticks([1,3,8,12,32,128,256],labels=[1,3,8,12,32,128,256]);ax.tick_params(labelsize=7)
        ax.grid(alpha=.18);ax.axhline(0,color="#64748b",lw=.5)
    axes.flat[26].text(0, .95,"Blue: mean raw sigmoid\nBlue band: 10th–90th percentile\nOrange: mean selection score\nDashed: cutoff after expert 8\n\n12 question tokens; all 256 ranks.\nRanks do not imply fixed expert IDs.\nLayer numbering is zero-based; L00 is dense.",transform=axes.flat[26].transAxes,va="top",fontsize=10,linespacing=1.6)
    fig.suptitle("All 256 expert scores, layer by layer",fontsize=22,fontweight="bold",x=.06,ha="left",y=.98)
    fig.text(.06,.954,stats["question"]+"  |  stock top-8; B1",fontsize=12,color="#475569")
    fig.supxlabel("Expert rank by bias-corrected selection score (log scale)",y=.025)
    fig.supylabel("Raw sigmoid / bias-corrected selection score",x=.015)
    fig.subplots_adjust(top=.925,bottom=.065,left=.06,right=.98,hspace=.45,wspace=.18)
    fig.savefig(out/"all-expert-score-spreads.png",dpi=160,facecolor=fig.get_facecolor())
    fig.savefig(out/"all-expert-score-spreads.svg",facecolor=fig.get_facecolor())
    print(json.dumps(stats,indent=2))


if __name__ == "__main__": main()
