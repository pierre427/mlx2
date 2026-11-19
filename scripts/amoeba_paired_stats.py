#!/usr/bin/env python3
"""Paired-difference statistics for per-text contrasts (host-neuron reuse review v4 §5.3).

paired(d, groups) summarises one paired contrast d (one value per text, positive = the favoured direction):
the mean, the paired standard error over texts, the 80%-power minimum detectable effect at two-sided alpha 0.05
(MDE = (z_0.975 + z_0.80) x SE = 2.80 x SE), a percentile bootstrap interval over texts, and a cluster bootstrap
over source groups (whole groups resampled, every text of a drawn group kept), whose spread also gives a
group-level SE and MDE. Each contrast is flagged against the minimum useful effect (Pierre 2026-11-18: 0.001
nats/byte) from its group interval: "above" (lower bound >= the effect), "below" (upper bound < the effect) or
"spanning".
CPU only, numpy only.
"""
from __future__ import annotations

import numpy as np

Z_MDE80 = 1.959964 + 0.841621  # two-sided alpha 0.05, power 0.80


def _group_index(groups, n):
    if groups is None:
        return [np.array([i]) for i in range(n)]
    by = {}
    for i, g in enumerate(groups):
        by.setdefault(str(g), []).append(i)
    return [np.array(v) for _, v in sorted(by.items())]


def group_boot_means(d, groups=None, boot=2000, seed=0):
    """bootstrap means of d with whole source groups resampled (texts of a drawn group all kept)"""
    d = np.asarray(d, dtype=np.float64)
    idx = _group_index(groups, len(d))
    sums = np.array([d[i].sum() for i in idx])
    cnts = np.array([len(i) for i in idx], dtype=np.float64)
    g = np.random.default_rng(seed)
    pick = g.integers(0, len(idx), (boot, len(idx)))
    return sums[pick].sum(1) / cnts[pick].sum(1)


def flag(lo, hi, mue):
    if mue is None or lo is None or hi is None:
        return None
    if lo >= mue:
        return "above"
    if hi < mue:
        return "below"
    return "spanning"


def paired(d, groups=None, boot=2000, seed=0, mue=0.001):
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    if n < 2:
        return {"n": n, "mean": float(d.mean()) if n else None}
    se = float(d.std(ddof=1) / np.sqrt(n))
    g = np.random.default_rng(seed + 1)
    tb = d[g.integers(0, n, (boot, n))].mean(1)
    gb = group_boot_means(d, groups, boot, seed)
    lo, hi = float(np.percentile(gb, 2.5)), float(np.percentile(gb, 97.5))
    se_g = float(gb.std(ddof=1))
    return {"n": n, "groups": len(_group_index(groups, n)) if groups is not None else None,
            "mean": float(d.mean()), "paired_se": se, "mde80": Z_MDE80 * se,
            "z": float(d.mean() / se) if se > 0 else None,
            "ci95_text": [float(np.percentile(tb, 2.5)), float(np.percentile(tb, 97.5))],
            "ci95_group": [lo, hi], "se_group": se_g, "mde80_group": Z_MDE80 * se_g,
            "min_useful_effect": mue, "vs_min_useful_effect": flag(lo, hi, mue)}


def ratio(num, den, groups=None, boot=2000, seed=0):
    """mean(num) / mean(den) with numerator and denominator resampled jointly by source group"""
    num, den = np.asarray(num, np.float64), np.asarray(den, np.float64)
    idx = _group_index(groups, len(num))
    sn = np.array([num[i].sum() for i in idx])
    sd = np.array([den[i].sum() for i in idx])
    g = np.random.default_rng(seed + 2)
    pick = g.integers(0, len(idx), (boot, len(idx)))
    a, b = sn[pick].sum(1), sd[pick].sum(1)
    ok = np.abs(b) > 1e-12
    r = a[ok] / b[ok]
    est = float(num.mean() / den.mean()) if abs(den.mean()) > 1e-12 else None
    return {"estimate": est, "ci95_group": ([float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]
                                            if len(r) > 100 else None)}
