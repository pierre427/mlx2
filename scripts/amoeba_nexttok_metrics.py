"""Gate-stream metrics for the next-token gating studies (numpy only; no torch, no mlx).

S, O: lists (one per stream) of bool [T, L, B]: the same-step open set and the applied (predicted) set per
position, layer (or 1 for box-level gates) and box.  ``nb`` [L, B] are the bytes (or units) each (layer, box)
tile holds; ``total`` the whole MLP's.  Positions before ``prompt`` (prefill) are not scored; a topic switch
window is the ``switch`` positions from a domain change on."""
from __future__ import annotations

import numpy as np


def metrics(S, O, dom, nb, total, names, prompt, switch=8):
    s = np.concatenate([x[prompt:] for x in S])
    o = np.concatenate([x[prompt:] for x in O])
    sw = []
    for d in dom:
        flag = np.zeros(len(d), bool)
        for i in range(1, len(d)):
            if d[i] != d[i - 1]:
                flag[i:i + switch] = True
        sw.append(flag[prompt:])
    sw = np.concatenate(sw)
    miss = s & ~o
    extra = o & ~s
    closed = (~o * nb[None]).sum((1, 2))
    return {"tokens": int(len(s)),
            "missed_per_needed": float(miss.sum() / max(1, s.sum())),
            "extra_per_closed": float(extra.sum() / max(1, (~s).sum())),
            "open_rate_applied": float(o.mean()), "open_rate_same_step": float(s.mean()),
            "rows_any_miss": float(miss.any(2).mean()), "rows_any_diff": float((s != o).any(2).mean()),
            "tokens_any_miss": float(miss.any((1, 2)).mean()),
            "box_bytes_skipped": float(closed.mean() / nb.sum()),
            "mlp_bytes_skipped": float(closed.mean() / total),
            "missed_per_needed_switch": float(miss[sw].sum() / max(1, s[sw].sum())),
            "missed_per_needed_steady": float(miss[~sw].sum() / max(1, s[~sw].sum())),
            "switch_tokens": int(sw.sum()),
            "per_box": {n: {"missed_per_needed": float(miss[..., b].sum() / max(1, s[..., b].sum())),
                            "extra_per_closed": float(extra[..., b].sum() / max(1, (~s[..., b]).sum())),
                            "same_step_open_rate": float(s[..., b].mean())}
                        for b, n in enumerate(names)}}


def change_rate(S, prompt):
    s = np.concatenate([x[prompt:] for x in S])
    ch = s[1:] != s[:-1]
    return {"per_layer_box": float(ch.mean()), "set_changes_per_token": float(ch.any((1, 2)).mean()),
            "box_any_layer_changes": float((s[1:].any(1) != s[:-1].any(1)).mean()),
            "flips_per_token": float(ch.sum((1, 2)).mean())}
